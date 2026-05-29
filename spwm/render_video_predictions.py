"""
Render selected fall-detection videos with real-time P(fall) overlay.

For each video:
  1. Run sliding-window JEPA Classifier inference
  2. Render each frame with prediction score overlaid in top-right corner
  3. Color-coded danger bar, fall warning text, ground-truth annotation border
  4. Output H.264 MP4 videos

Usage:
    python3 -m spwm.render_video_predictions --device cuda

Output:
    outputs/videos/<scene>_<video_id>_prediction.mp4
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

# --- Config ------------------------------------------------------------------

FRAME_SIZE = 224
CONTEXT_FRAMES = 16
STRIDE = 2

SELECTED_VIDEOS = [
    ("Coffee_room_01", "video (47)", "Coffee Room - Long lead time"),
    ("Coffee_room_01", "video (19)", "Coffee Room - Early fall"),
    ("Home_01", "video (1)", "Home - Living room fall"),
    ("Home_01", "video (11)", "Home - Walking then fall"),
    ("Home_02", "video (37)", "Home 2 - Different background"),
]

# Color scheme
COLOR_SAFE = (34, 139, 34)         # forest green
COLOR_WARN = (255, 165, 0)         # orange
COLOR_DANGER = (220, 20, 60)       # crimson red
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED_BORDER = (220, 20, 60)
COLOR_BG = (15, 15, 25)            # dark panel background


# --- Helpers -----------------------------------------------------------------

def load_model(checkpoint_path, device, encoder_device):
    from spwm.classifier_model import JEPAClassifier

    model = JEPAClassifier(
        video_dim=1024, audio_dim=768,
        fusion_hidden=512, fusion_out=256,
        dropout=0.3, use_audio=True,
        encoder_device=encoder_device,
    )
    model._place_encoders(encoder_device)
    model = model.to(device).eval()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    print(f"[Model] Loaded from {checkpoint_path}")
    return model


def load_video(video_path):
    from av import open as av_open

    with av_open(video_path) as container:
        vs = container.streams.video[0]
        fps = float(vs.average_rate) if vs.average_rate else 25.0
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format='rgb24')
            frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)

        audio, audio_sr = None, 48000
        try:
            container.seek(0)
            astream = container.streams.audio[0]
            audio_sr = astream.sample_rate or 48000
            chunks = []
            for frame in container.decode(audio=0):
                samples = frame.to_ndarray()
                chunks.append(torch.from_numpy(samples.astype(np.float32)))
            if chunks:
                audio = torch.cat(chunks, dim=-1).mean(dim=0)
        except Exception:
            pass

    video_frames = torch.stack(frames) if frames else None
    return video_frames, audio, fps, audio_sr


def load_annotation(video_path):
    ann_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), 'Annotation_files')
    base = os.path.splitext(os.path.basename(video_path))[0]
    for ext in ['.txt', '_gt.txt']:
        ann_path = os.path.join(ann_dir, base + ext)
        if os.path.exists(ann_path):
            with open(ann_path) as f:
                lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
            if len(lines) >= 2:
                try:
                    fs, fe = int(lines[0].split(',')[0]), int(lines[1].split(',')[0])
                    if fs > 0 and fe > fs:
                        return fs, fe
                except ValueError:
                    pass
    return None


def resize_frames(frames):
    T, C, H, W = frames.shape
    if H == FRAME_SIZE and W == FRAME_SIZE:
        return frames
    frames_flat = frames.reshape(T * C, H, W).unsqueeze(0)
    resized = F.interpolate(frames_flat, size=(FRAME_SIZE, FRAME_SIZE),
                            mode='bilinear', align_corners=False)
    return resized.squeeze(0).reshape(T, C, FRAME_SIZE, FRAME_SIZE)


def extract_audio_window(audio, start_frame, n_frames, fps, orig_sr):
    target_sr, target_len = 16000, 48000
    if audio is None:
        return torch.zeros(target_len)

    samples_per_frame = orig_sr / fps
    a_start = max(0, int(start_frame * samples_per_frame))
    a_end = min(len(audio), int((start_frame + n_frames) * samples_per_frame))
    if a_end <= a_start:
        return torch.zeros(target_len)

    seg = audio[a_start:a_end].clone()
    if orig_sr != target_sr:
        new_len = max(1, int(len(seg) * target_sr / orig_sr))
        seg = F.interpolate(seg.unsqueeze(0).unsqueeze(0),
                            size=new_len, mode='linear', align_corners=False).squeeze()
    if seg.shape[0] < target_len:
        seg = torch.cat([seg, torch.zeros(target_len - seg.shape[0])])
    else:
        seg = seg[:target_len]
    return seg


def run_inference(model, frames, audio, fps, audio_sr, device='cuda'):
    """Sliding-window inference. Returns dict: frame_index → probability."""
    T = frames.shape[0]
    if T < CONTEXT_FRAMES:
        return {}

    batch_frames, batch_audio, positions = [], [], []
    model_device = next(model.parameters()).device

    for start in range(0, T - CONTEXT_FRAMES, STRIDE):
        clip = resize_frames(frames[start:start + CONTEXT_FRAMES])
        aseg = extract_audio_window(audio, start, CONTEXT_FRAMES, fps, audio_sr)
        batch_frames.append(clip)
        batch_audio.append(aseg)
        positions.append(start)

    all_probs = []
    BATCH = 16
    for i in range(0, len(batch_frames), BATCH):
        bf = torch.stack(batch_frames[i:i+BATCH]).to(model_device)
        ba = torch.stack(batch_audio[i:i+BATCH]).to(model_device)
        with torch.no_grad():
            probs = model.predict(bf, ba)
        all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs)

    # Interpolate: for each frame, use the nearest prediction window
    frame_probs = {}
    for pos, prob in zip(positions, all_probs):
        # This prediction corresponds to frames [pos, pos+CONTEXT_FRAMES)
        mid = pos + CONTEXT_FRAMES // 2
        frame_probs[mid] = float(prob)

    return frame_probs


def get_frame_prob(frame_idx, frame_probs):
    """Get P(fall) for a frame by interpolating from nearest predictions."""
    if not frame_probs:
        return 0.0
    positions = sorted(frame_probs.keys())
    probs = [frame_probs[p] for p in positions]

    # Nearest-neighbor lookup
    idx = np.searchsorted(positions, frame_idx)
    if idx == 0:
        return probs[0]
    if idx >= len(positions):
        return probs[-1]

    left_pos, right_pos = positions[idx - 1], positions[idx]
    left_prob, right_prob = probs[idx - 1], probs[idx]
    if right_pos == left_pos:
        return left_prob
    t = (frame_idx - left_pos) / (right_pos - left_pos)
    return left_prob + t * (right_prob - left_prob)


def draw_overlay(frame_rgb, prob, is_fall_gt, frame_idx, fps, font, font_bold, font_small):
    """Draw prediction overlay on a single RGB frame (numpy array HxWx3 uint8).

    Returns the annotated frame.
    """
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # --- Top danger bar (height proportional to screen) ---
    bar_h = max(20, int(H * 0.07))
    bar_w = int(W * 0.45)
    bar_x = W - bar_w - 15
    bar_y = 12

    # Background
    draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                           radius=6, fill=COLOR_BG, outline=(60, 60, 80), width=2)

    # Filled portion based on probability
    fill_w = max(1, int(bar_w * prob)) if prob > 0.001 else 0
    if fill_w >= 4:  # need at least 4px for rounded_rectangle
        # Color gradient: green → yellow → orange → red
        if prob < 0.3:
            r = int(COLOR_WARN[0] * prob / 0.3 + COLOR_SAFE[0] * (1 - prob / 0.3))
            g = int(COLOR_WARN[1] * prob / 0.3 + COLOR_SAFE[1] * (1 - prob / 0.3))
            b = int(COLOR_WARN[2] * prob / 0.3 + COLOR_SAFE[2] * (1 - prob / 0.3))
        elif prob < 0.5:
            r = int(COLOR_DANGER[0] * (prob - 0.3) / 0.2 + COLOR_WARN[0] * (0.5 - prob) / 0.2)
            g = int(COLOR_DANGER[1] * (prob - 0.3) / 0.2 + COLOR_WARN[1] * (0.5 - prob) / 0.2)
            b = int(COLOR_DANGER[2] * (prob - 0.3) / 0.2 + COLOR_WARN[2] * (0.5 - prob) / 0.2)
        else:
            r, g, b = COLOR_DANGER
        fill_color = (r, g, b)
        draw.rounded_rectangle(
            [bar_x + 2, bar_y + 2, bar_x + fill_w - 2, bar_y + bar_h - 2],
            radius=4, fill=fill_color,
        )

    # Score text inside bar
    score_text = f"P(FALL) = {prob:.3f}"
    t_bbox = draw.textbbox((0, 0), score_text, font=font_bold)
    t_w = t_bbox[2] - t_bbox[0]
    draw.text((bar_x + bar_w // 2 - t_w // 2, bar_y + bar_h // 2 - 1),
              score_text, fill=COLOR_WHITE, font=font_bold, anchor='lm')

    # Threshold marker at 0.5
    thresh_x = bar_x + int(bar_w * 0.5)
    draw.line([(thresh_x, bar_y - 3), (thresh_x, bar_y + bar_h + 3)],
              fill=(255, 255, 255, 180), width=1)

    # --- FALL DETECTED warning ---
    if prob > 0.5:
        warn_text = "⚠ FALL DETECTED"
        warn_y = bar_y + bar_h + 14
        # Red background pill
        padding = 14
        t_bbox = draw.textbbox((0, 0), warn_text, font=font_bold)
        tw = t_bbox[2] - t_bbox[0]
        th = t_bbox[3] - t_bbox[1]
        wx = bar_x + bar_w // 2 - tw // 2 - padding
        draw.rounded_rectangle(
            [wx, warn_y - 4, wx + tw + padding * 2, warn_y + th + 6],
            radius=8, fill=COLOR_DANGER,
        )
        draw.text((bar_x + bar_w // 2, warn_y + th // 2),
                  warn_text, fill=COLOR_WHITE, font=font_bold, anchor='mm',
                  stroke_width=1, stroke_fill=(180, 10, 40))

    # --- Ground truth fall border ---
    if is_fall_gt:
        border_w = max(4, int(min(W, H) * 0.015))
        for i in range(border_w):
            draw.rectangle(
                [i, i, W - 1 - i, H - 1 - i],
                outline=COLOR_RED_BORDER, width=2,
            )
        # "Ground Truth" label
        gt_text = "Ground Truth: FALL"
        gt_y = H - 40
        t_bbox = draw.textbbox((0, 0), gt_text, font=font_small)
        tw = t_bbox[2] - t_bbox[0]
        draw.rounded_rectangle(
            [W // 2 - tw // 2 - 12, gt_y - 4, W // 2 + tw // 2 + 12, gt_y + t_bbox[3] - t_bbox[1] + 6],
            radius=5, fill=COLOR_DANGER,
        )
        draw.text((W // 2, gt_y + (t_bbox[3] - t_bbox[1]) // 2),
                  gt_text, fill=COLOR_WHITE, font=font_small, anchor='mm')

    # --- Bottom info bar ---
    time_sec = frame_idx / fps
    info = f"Frame {frame_idx}  |  {time_sec:.1f}s  |  {fps:.0f} fps"
    info_y = H - 14
    t_bbox = draw.textbbox((0, 0), info, font=font_small)
    tw = t_bbox[2] - t_bbox[0]
    draw.text((W - tw - 14, info_y), info, fill=(180, 180, 200), font=font_small, anchor='lm')

    # --- Legend dot ---
    dot_r = 5
    dot_x, dot_y = 14, H - 18
    dot_color = COLOR_DANGER if prob > 0.5 else (COLOR_WARN if prob > 0.3 else COLOR_SAFE)
    draw.ellipse([dot_x, dot_y, dot_x + dot_r * 2, dot_y + dot_r * 2], fill=dot_color)
    status = "DANGER" if prob > 0.5 else ("WARNING" if prob > 0.3 else "NORMAL")
    draw.text((dot_x + dot_r * 2 + 6, dot_y + dot_r - 1), status,
              fill=dot_color, font=font_small, anchor='lm')

    return np.array(img)


def render_video(video_path, output_path, model, ann, fps, audio_sr,
                 trim_range=None, device='cuda'):
    """Render a video with prediction overlay."""
    from av import open as av_open

    frames_raw, audio, actual_fps, actual_sr = load_video(video_path)
    if frames_raw is None:
        print("  Cannot load frames")
        return False

    T = frames_raw.shape[0]
    if actual_fps is not None and actual_fps > 0:
        fps = actual_fps
    if actual_sr is not None and actual_sr > 0:
        audio_sr = actual_sr

    # Trim if requested
    start_frame, end_frame = 0, T
    if trim_range:
        start_frame = max(0, trim_range[0])
        end_frame = min(T, trim_range[1])

    print(f"  Rendering frames {start_frame}-{end_frame} ({end_frame - start_frame} frames)")

    # Run inference on the trimmed range (needs context before start)
    infer_start = max(0, start_frame - CONTEXT_FRAMES)
    infer_end = min(T, end_frame + CONTEXT_FRAMES)
    infer_frames = frames_raw[infer_start:infer_end]
    infer_audio = audio[..., :] if audio is not None else None
    # Adjust audio to match inference range
    if infer_audio is not None:
        spf = len(infer_audio) / max(T, 1)
        infer_audio = infer_audio[int(infer_start * spf):int(infer_end * spf)]

    frame_probs = run_inference(model, infer_frames, infer_audio, fps, audio_sr, device)
    # Shift predictions back to original frame indices
    frame_probs = {k + infer_start: v for k, v in frame_probs.items()}

    # Setup fonts
    try:
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=16)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=13)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=11)
    except Exception:
        font_bold = ImageFont.load_default()
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # H.264 encoding via PyAV
    output_scale = 2  # upscale factor for presentation quality
    output = av_open(output_path, 'w')
    ostream = output.add_stream('libx264', rate=int(fps))
    ostream.width = frames_raw.shape[-1] * output_scale
    ostream.height = frames_raw.shape[-2] * output_scale
    ostream.pix_fmt = 'yuv420p'
    ostream.options = {
        'crf': '18',
        'preset': 'slow',
        'profile': 'high',
    }

    total = end_frame - start_frame
    for i, frame_idx in enumerate(range(start_frame, end_frame)):
        # Original frame as numpy RGB
        raw = frames_raw[frame_idx]
        rgb = (raw.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        prob = get_frame_prob(frame_idx, frame_probs)

        is_fall = False
        if ann is not None:
            fs, fe = ann
            is_fall = (fs <= frame_idx <= fe)

        annotated = draw_overlay(rgb, prob, is_fall, frame_idx, fps,
                                 font, font_bold, font_small)

        # Upscale for presentation quality
        if output_scale != 1:
            pil_img = Image.fromarray(annotated)
            new_w = annotated.shape[1] * output_scale
            new_h = annotated.shape[0] * output_scale
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            annotated = np.array(pil_img)

        # Encode
        from av import VideoFrame
        vf = VideoFrame.from_ndarray(annotated, format='rgb24')
        for packet in ostream.encode(vf):
            output.mux(packet)

        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"    [{i+1}/{total}] rendered", end='\r')

    # Flush
    for packet in ostream.encode():
        output.mux(packet)
    output.close()
    print(f"\n  Saved to {output_path}")
    return True


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/classifier_best.pt')
    parser.add_argument('--data_root', type=str, default='Le2i')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--encoder_device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='outputs/videos')
    parser.add_argument('--trim_seconds', type=float, default=0,
                        help='Only render N seconds around fall (0=full video)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.checkpoint, args.device, args.encoder_device)
    print(f"[Setup] Model loaded.\n")

    for scene, video_id, desc in SELECTED_VIDEOS:
        video_path = os.path.join(args.data_root, scene, scene, 'Videos', f'{video_id}.avi')
        if not os.path.exists(video_path):
            print(f"[SKIP] {video_path}")
            continue

        ann = load_annotation(video_path)

        # Determine trim range
        trim_range = None
        if args.trim_seconds > 0 and ann is not None:
            fs, fe = ann
            fps = 25.0
            try:
                from av import open as av_open
                with av_open(video_path) as c:
                    fps = float(c.streams.video[0].average_rate)
            except Exception:
                pass
            margin = int(args.trim_seconds * fps)
            trim_range = (max(0, fs - margin), fe + margin)

        # Load video to get actual FPS
        try:
            from av import open as av_open
            with av_open(video_path) as c:
                fps = float(c.streams.video[0].average_rate)
                audio_sr = c.streams.audio[0].sample_rate if c.streams.audio else 48000
        except Exception:
            fps, audio_sr = 25.0, 48000

        out_name = f"{scene}_{video_id.replace(' ', '_')}_prediction.mp4"
        out_path = os.path.join(args.output_dir, out_name)

        print(f"[Rendering] {desc}")
        print(f"  {video_path}")

        ok = render_video(video_path, out_path, model, ann, fps, audio_sr,
                          trim_range=trim_range, device=args.device)
        if ok:
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f"  Output: {out_path} ({size_mb:.1f} MB)")
        print()

    print("[Done] All videos rendered.")


if __name__ == '__main__':
    main()
