"""
Run sliding-window fall prediction over selected videos and visualize P(fall) vs time.

Generates professional multi-panel plots for reporting: each panel shows the model's
predicted fall probability curve overlaid with the ground-truth fall annotation region.

Usage:
    python3 -m spwm.visualize_predictions --device cuda

Output:
    outputs/fall_prediction_curves.png   (multi-panel visualization)
    outputs/fall_prediction_curves.pdf   (vector version for reports)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# --- Config ------------------------------------------------------------------

FRAME_SIZE = 224
CONTEXT_FRAMES = 16
AUDIO_SR = 48000
STRIDE = 2          # predict every N frames (2 = smooth, 4 = fast)

# Selected videos: diverse scenes, fall at different positions
SELECTED_VIDEOS = [
    # scene, video_id, description
    ("Coffee_room_01", "video (47)", "Coffee Room — Long lead time (25s before fall)"),
    ("Coffee_room_01", "video (19)", "Coffee Room — Early fall"),
    ("Home_01", "video (1)", "Home — Living room fall"),
    ("Home_01", "video (11)", "Home — Long video, walking then fall"),
    ("Home_02", "video (37)", "Home 2 — Different background"),
]


# --- Helpers -----------------------------------------------------------------

def load_model(checkpoint_path, device, encoder_device):
    from spwm.classifier_model import JEPAClassifier

    model = JEPAClassifier(
        video_dim=1024,
        audio_dim=768,
        fusion_hidden=512,
        fusion_out=256,
        dropout=0.3,
        use_audio=True,
        encoder_device=encoder_device,
    )
    model._place_encoders(encoder_device)
    model = model.to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt

    # Strip "module." prefix if present
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    print(f"[Model] Loaded from {checkpoint_path}")
    return model


def load_video(video_path):
    from av import open as av_open

    with av_open(video_path) as container:
        vs = container.streams.video[0]
        fps = float(vs.average_rate) if vs.average_rate else 25.0
        total_frames = vs.frames or 0

        frames = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format='rgb24')
            frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)

        # Audio
        audio = None
        audio_sr = 48000  # default
        try:
            container.seek(0)
            audio_stream = container.streams.audio[0]
            audio_sr = audio_stream.sample_rate or 48000
            audio_chunks = []
            for frame in container.decode(audio=0):
                samples = frame.to_ndarray()
                audio_chunks.append(torch.from_numpy(samples.astype(np.float32)))
            if audio_chunks:
                audio = torch.cat(audio_chunks, dim=-1).mean(dim=0)
        except Exception:
            pass

    video_frames = torch.stack(frames) if frames else None
    if video_frames is None or video_frames.shape[0] < CONTEXT_FRAMES:
        return None, None, None, None, None

    return video_frames, audio, fps, total_frames, audio_sr


def load_annotation(video_path):
    """Parse Le2i annotation: returns (fall_start, fall_end) or None."""
    ann_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), 'Annotation_files')
    base = os.path.splitext(os.path.basename(video_path))[0]
    for ext in ['.txt', '_gt.txt']:
        ann_path = os.path.join(ann_dir, base + ext)
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
            if len(lines) >= 2:
                try:
                    fs = int(lines[0].split(',')[0])
                    fe = int(lines[1].split(',')[0])
                    if fs > 0 and fe > fs:
                        return fs, fe
                except ValueError:
                    pass
    return None


def resize_frames(frames):
    """Resize (T, C, H, W) → (T, C, 224, 224)."""
    T, C, H, W = frames.shape
    if H == FRAME_SIZE and W == FRAME_SIZE:
        return frames
    frames_flat = frames.reshape(T * C, H, W).unsqueeze(0)
    resized = F.interpolate(frames_flat, size=(FRAME_SIZE, FRAME_SIZE),
                            mode='bilinear', align_corners=False)
    return resized.squeeze(0).reshape(T, C, FRAME_SIZE, FRAME_SIZE)


def extract_audio_window(audio, start_frame, n_frames, fps, orig_sr):
    """Extract audio segment aligned with frame window.

    Matches PreprocessedLe2iDataset._resample_audio exactly:
    1. Extract segment aligned with context frames
    2. Resample to 16kHz via linear interpolation
    3. Pad/trim to fixed 48000 samples (= 3s @ 16kHz, WavJEPA expected input)
    """
    target_sr = 16000
    target_len = 48000

    if audio is None:
        return torch.zeros(target_len)

    samples_per_frame = orig_sr / fps
    a_start = int(start_frame * samples_per_frame)
    a_end = int((start_frame + n_frames) * samples_per_frame)
    a_start = max(0, a_start)
    a_end = min(len(audio), a_end)

    if a_end <= a_start:
        return torch.zeros(target_len)

    seg = audio[a_start:a_end].clone()

    # Resample to 16kHz
    if orig_sr != target_sr:
        new_len = int(len(seg) * target_sr / orig_sr)
        if new_len > 0:
            seg = F.interpolate(
                seg.unsqueeze(0).unsqueeze(0),
                size=new_len, mode='linear', align_corners=False,
            ).squeeze()
        else:
            return torch.zeros(target_len)

    # Pad or trim to exactly target_len samples
    if seg.shape[0] < target_len:
        seg = torch.cat([seg, torch.zeros(target_len - seg.shape[0])])
    elif seg.shape[0] > target_len:
        seg = seg[:target_len]

    return seg


def apply_sliding_window(model, frames, audio, fps, audio_sr, stride=2, device='cuda', encoder_device='cuda'):
    """Run sliding-window inference over entire video.

    For each position i, take frames[i:i+16] + corresponding audio → P(fall).
    Returns array of (frame_index, probability) pairs.
    """
    T = frames.shape[0]
    max_start = T - CONTEXT_FRAMES
    if max_start < 0:
        return np.array([]), np.array([])

    positions = list(range(0, max_start, stride))

    all_probs = []
    batch_frames = []
    batch_audio = []

    model_device = next(model.parameters()).device

    for start_frame in positions:
        clip = frames[start_frame:start_frame + CONTEXT_FRAMES]
        clip = resize_frames(clip)
        audio_seg = extract_audio_window(audio, start_frame, CONTEXT_FRAMES, fps, audio_sr)

        batch_frames.append(clip)
        batch_audio.append(audio_seg)

    # Batch inference
    BATCH = 16
    for i in range(0, len(batch_frames), BATCH):
        bf = torch.stack(batch_frames[i:i+BATCH]).to(model_device)
        ba = torch.stack(batch_audio[i:i+BATCH]).to(model_device)
        with torch.no_grad():
            probs = model.predict(bf, ba)
        all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    positions = np.array(positions, dtype=np.float32)

    # Convert to time axis (offset to the MIDDLE of each context window)
    times = (positions + CONTEXT_FRAMES / 2) / fps

    return times, all_probs


# --- Main visualization ------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/classifier_best.pt')
    parser.add_argument('--data_root', type=str, default='Le2i')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--encoder_device', type=str, default='cuda')
    parser.add_argument('--stride', type=int, default=STRIDE)
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model once
    model = load_model(args.checkpoint, args.device, args.encoder_device)
    print(f"[Setup] Model loaded. Stride={args.stride}")

    # Process each video
    results = []
    for scene, video_id, desc in SELECTED_VIDEOS:
        video_path = os.path.join(args.data_root, scene, scene, 'Videos', f'{video_id}.avi')
        if not os.path.exists(video_path):
            print(f"  [SKIP] {video_path} not found")
            continue

        print(f"\n[Processing] {desc}")
        print(f"  {video_path}")

        frames, audio, fps, total, audio_sr = load_video(video_path)
        if frames is None:
            print(f"  [FAIL] Cannot load video")
            continue

        ann = load_annotation(video_path)
        print(f"  Frames: {frames.shape[0]}, FPS: {fps:.1f}, Audio SR: {audio_sr}, Annotation: {ann}")

        times, probs = apply_sliding_window(model, frames, audio, fps, audio_sr,
                                            stride=args.stride,
                                            device=args.device,
                                            encoder_device=args.encoder_device)
        results.append({
            'scene': scene,
            'video_id': video_id,
            'description': desc,
            'times': times,
            'probs': probs,
            'fps': fps,
            'fall_annotation': ann,
            'total_frames': frames.shape[0],
        })
        print(f"  Predictions: {len(probs)} windows, P(fall) range=[{probs.min():.3f}, {probs.max():.3f}]")

    if not results:
        print("No videos processed!")
        return

    # --- Plotting ------------------------------------------------------------
    n_videos = len(results)
    fig, axes = plt.subplots(n_videos, 1, figsize=(14, 3.5 * n_videos), sharex=False)

    if n_videos == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, (ax, res) in enumerate(zip(axes, results)):
        times = res['times']
        probs = res['probs']
        ann = res['fall_annotation']
        fps = res['fps']
        total_frames = res['total_frames']

        # Plot probability curve
        ax.plot(times, probs, color=colors[0], linewidth=1.5, alpha=0.9, label='P(fall)')
        ax.fill_between(times, 0, probs, color=colors[0], alpha=0.10)

        # Decision threshold
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label='Threshold (0.5)')

        # Ground-truth fall region
        if ann is not None:
            fall_start_t = ann[0] / fps
            fall_end_t = ann[1] / fps
            ax.axvspan(fall_start_t, fall_end_t, color='red', alpha=0.15,
                       label=f'Ground Truth Fall\n(frame {ann[0]}–{ann[1]})')
            # Mark fall onset
            ax.axvline(x=fall_start_t, color='darkred', linestyle=':', linewidth=1.2, alpha=0.8)

        # Context window warning zone (first 0.64s have no prediction)
        ctx_dur = CONTEXT_FRAMES / fps
        ax.axvspan(0, ctx_dur, color='gray', alpha=0.05)
        ax.text(ctx_dur / 2, 0.95, 'warmup', ha='center', va='top',
                fontsize=8, color='gray', transform=ax.get_xaxis_transform())

        # Styling
        ax.set_xlim(0, total_frames / fps)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel('P(fall)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_title(res['description'], fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add FPS and stride annotation
        ax.text(0.99, 0.01,
                f'fps={fps:.0f} | stride={args.stride} | ctx={CONTEXT_FRAMES} frames',
                transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                color='gray', style='italic')

    fig.suptitle('JEPA Classifier — Fall Detection Probability Over Time',
                 fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()

    png_path = os.path.join(args.output_dir, 'fall_prediction_curves.png')
    pdf_path = os.path.join(args.output_dir, 'fall_prediction_curves.pdf')
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[Done] Saved to:\n  {png_path}\n  {pdf_path}")

    # --- Summary table -------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"{'Video':<45} {'P_max':>8} {'P_mean':>8} {'P@fall':>8} {'Detected?':>10}")
    print("-" * 80)
    for res in results:
        probs = res['probs']
        ann = res['fall_annotation']
        p_max = probs.max()
        p_mean = probs.mean()

        # Probability during the fall window
        p_fall_str = "N/A"
        detected = "N/A"
        if ann is not None:
            fps = res['fps']
            fall_start_t = ann[0] / fps
            fall_end_t = ann[1] / fps
            fall_mask = (res['times'] >= fall_start_t - 0.3) & (res['times'] <= fall_end_t + 0.3)
            if fall_mask.any():
                p_fall = probs[fall_mask].max()
                p_fall_str = f"{p_fall:.3f}"
                detected = "YES" if p_fall > 0.5 else "NO"

        desc_short = res['description'][:42]
        print(f"{desc_short:<45} {p_max:>8.3f} {p_mean:>8.3f} {p_fall_str:>8} {detected:>10}")
    print("=" * 80)


if __name__ == '__main__':
    main()
