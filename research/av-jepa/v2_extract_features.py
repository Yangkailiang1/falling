"""
AV-JEPA v2 Feature Extraction
------------------------------
Pre-extract video (V-JEPA 2) and audio (WavJEPA) features from Le2i dataset.

This avoids loading 500M encoders during training:
  1. Run V-JEPA 2 on all video clips → save to disk
  2. Run WavJEPA on all audio clips → save to disk
  3. Training loads pre-extracted features + trains lightweight projector/predictor

Usage:
    python v2_extract_features.py --data_root /path/to/Le2i --output_dir features_v2/
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Extract V-JEPA 2 + WavJEPA features")
    p.add_argument("--data_root", type=str,
                   default="/home/yangkailiang/.cache/kagglehub/datasets/tuyenldvn/falldataset-imvia/versions/2")
    p.add_argument("--output_dir", type=str, default="features_v2")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--vjepa2_ckpt", type=str, default="",
                   help="Path to V-JEPA 2 checkpoint, or empty for torch.hub")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Process one video at a time (GPU memory)")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--clip_duration", type=float, default=2.0)
    p.add_argument("--target_gap", type=float, default=1.0)
    p.add_argument("--target_fps", type=float, default=12.0)
    p.add_argument("--audio_sr", type=int, default=16000)
    return p.parse_args()


def build_encoders(args, device):
    """Build and return (video_encoder, audio_encoder). Returns None for unavailable."""
    from v2_encoders import VJEPA2VideoEncoder, WavJEPAAudioEncoder, resample_audio

    # Video encoder
    print("Loading V-JEPA 2 video encoder...")
    video_enc = VJEPA2VideoEncoder(
        model_name="vit_large",
        checkpoint_path=args.vjepa2_ckpt,
        img_size=args.frame_size,
        num_frames=args.num_frames,
        embed_dim=1024,
    ).to(device).eval()

    # Audio encoder
    print("Loading WavJEPA audio encoder...")
    audio_enc = WavJEPAAudioEncoder(
        model_name="labhamlet/wavjepa-base",
        sample_rate=args.audio_sr,
        embed_dim=768,
    ).to(device).eval()

    return video_enc, audio_enc


def extract_from_video(video_enc, audio_enc, frames, audio, audio_sr_orig, args, device):
    """
    Extract context and target features from a single video's frames+audio.

    Args:
        frames: (T_total, C, H, W) all decoded frames
        audio:  (total_samples,) or None
        audio_sr_orig: original audio sample rate

    Returns:
        list of (v_ctx, a_ctx, v_tgt, a_tgt, label) tuples
    """
    from v2_encoders import resample_audio
    from le2i_dataset import extract_frame_range, extract_audio_segment

    T = frames.shape[0]
    ctx_len = int(args.clip_duration * args.target_fps)  # frame count
    tgt_len = ctx_len
    gap = int(args.target_gap * args.target_fps)
    stride = max(1, int(ctx_len * 0.5))

    # Resample audio to target sr
    if audio is not None and audio.numel() > 100:
        if audio_sr_orig and audio_sr_orig != args.audio_sr:
            audio = resample_audio(audio, audio_sr_orig, args.audio_sr)
        effective_sr = args.audio_sr
    else:
        effective_sr = args.audio_sr

    results = []
    start = 0
    while start + ctx_len + gap + tgt_len <= T:
        # Extract frames
        ctx_frames = extract_frame_range(frames, start, start + ctx_len, args.num_frames)
        tgt_frames = extract_frame_range(
            frames, start + ctx_len + gap,
            start + ctx_len + gap + tgt_len, args.num_frames
        )

        # Extract audio
        if audio is not None and audio.numel() > 100:
            ctx_audio = extract_audio_segment(
                audio, start / args.target_fps,
                (start + ctx_len) / args.target_fps,
                effective_sr
            )
            tgt_audio = extract_audio_segment(
                audio,
                (start + ctx_len + gap) / args.target_fps,
                (start + ctx_len + gap + tgt_len) / args.target_fps,
                effective_sr
            )
        else:
            ctx_audio = torch.zeros(args.audio_sr * args.clip_duration)
            tgt_audio = torch.zeros(args.audio_sr * args.clip_duration)

        # Run encoders
        with torch.no_grad():
            v_ctx = video_enc(ctx_frames.unsqueeze(0).to(device))  # (1, Nv, Dv)
            a_ctx = audio_enc(ctx_audio.unsqueeze(0).to(device))   # (1, Na, Da)
            v_tgt = video_enc(tgt_frames.unsqueeze(0).to(device))
            a_tgt = audio_enc(tgt_audio.unsqueeze(0).to(device))

        results.append((
            v_ctx.squeeze(0).cpu(),   # (Nv, Dv)
            a_ctx.squeeze(0).cpu(),   # (Na, Da)
            v_tgt.squeeze(0).cpu(),
            a_tgt.squeeze(0).cpu(),
        ))

        start += stride

    return results


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build encoders
    video_enc, audio_enc = build_encoders(args, device)

    # Scan dataset
    from le2i_dataset import Le2iDataset, parse_annotation, has_fall
    dataset = Le2iDataset(
        root_dir=args.data_root,
        split="eval",
        clip_duration=args.clip_duration,
        target_gap=args.target_gap,
        target_fps=args.target_fps,
        frame_size=(args.frame_size, args.frame_size),
        num_frames=args.num_frames,
        audio_sample_rate=args.audio_sr,
    )

    print(f"\nProcessing {len(dataset)} clips from {args.data_root}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Group by video file to avoid redundant decoding
    from le2i_dataset import decode_video_audio
    video_groups = defaultdict(list)
    for idx in range(len(dataset)):
        video_path, _, label, ctx_s, ctx_e, tgt_s, tgt_e = dataset.samples[idx]
        video_groups[video_path].append((idx, label, ctx_s, ctx_e, tgt_s, tgt_e))

    all_features = []
    all_labels = []

    for video_path, samples in tqdm(video_groups.items(), desc="Extracting"):
        try:
            frames, audio, total_f, fps, audio_sr = decode_video_audio(
                video_path,
                target_fps=args.target_fps,
                target_size=(args.frame_size, args.frame_size),
            )
        except Exception as e:
            print(f"  Skip {os.path.basename(video_path)}: {e}")
            continue

        # Extract features for all clips in this video
        features = extract_from_video(
            video_enc, audio_enc, frames, audio, audio_sr, args, device
        )

        # Match features to dataset samples
        for fi, (v_ctx, a_ctx, v_tgt, a_tgt) in enumerate(features):
            if fi < len(samples):
                idx, label, _, _, _, _ = samples[fi]
                all_features.append({
                    "v_ctx": v_ctx, "a_ctx": a_ctx,
                    "v_tgt": v_tgt, "a_tgt": a_tgt,
                    "label": label,
                    "idx": idx,
                    "video": os.path.basename(video_path),
                })
                all_labels.append(label)

    # Split: train = label==0, eval = all
    train_features = [f for f in all_features if f["label"] == 0]
    eval_features = all_features

    # Save
    train_path = os.path.join(args.output_dir, "train_features.pt")
    eval_path = os.path.join(args.output_dir, "eval_features.pt")

    torch.save(train_features, train_path)
    torch.save(eval_features, eval_path)

    print(f"\n{'='*50}")
    print(f"Feature extraction complete")
    print(f"{'='*50}")
    print(f"  Train (normal only): {len(train_features)} clips")
    print(f"  Eval (mixed):        {len(eval_features)} clips")
    print(f"  Eval falls:          {sum(1 for f in eval_features if f['label']==1)}")
    print(f"  Eval normal:         {sum(1 for f in eval_features if f['label']==0)}")
    print(f"  Saved to:            {args.output_dir}/")
    print(f"\n  Feature shapes:")
    if train_features:
        f0 = train_features[0]
        print(f"    Video ctx:  {f0['v_ctx'].shape}  ({f0['v_ctx'].shape[0]} tokens)")
        print(f"    Audio ctx:  {f0['a_ctx'].shape}  ({f0['a_ctx'].shape[0]} tokens)")
        print(f"    Video tgt:  {f0['v_tgt'].shape}")
        print(f"    Audio tgt:  {f0['a_tgt'].shape}")


if __name__ == "__main__":
    main()
