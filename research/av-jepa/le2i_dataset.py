"""
Le2i Dataset Loader for AV-JEPA
--------------------------------
Loads Le2i fall detection dataset with synchronized video+audio.

Dataset structure:
    Le2i/
    └── {scene}/{scene}/
        ├── Videos/video (N).avi      # 320×240, 24fps, rawvideo + PCM audio
        └── Annotation_files/video (N).txt  # frame-level annotations

Annotation format:
    <fall_start_frame>
    <fall_end_frame>
    <frame>,<label>,<x>,<y>,<w>,<h>  (per-frame bounding boxes)

Strategy:
    - Train JEPA ONLY on normal clips (unsupervised world model)
    - Evaluate on held-out normal + all fall clips
    - Fall clip = clip that contains the fall event
"""

import os
import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset

import av  # PyAV for video/audio decoding
from torchvision import transforms


# ─── Annotation Parsing ────────────────────────────────────────────────


def parse_annotation(filepath: str) -> Tuple[int, int, List[Dict]]:
    """
    Parse Le2i annotation file.

    Two formats exist:
    1. With fall: first 2 lines = fall_start, fall_end (single ints),
       then frame-level CSV data.
    2. Without fall: directly frame-level CSV data.

    Returns:
        fall_start: frame number where fall begins (0 if no fall)
        fall_end: frame number where fall ends (0 if no fall)
        frames: list of per-frame bounding box dicts
    """
    with open(filepath) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    fall_start = 0
    fall_end = 0
    data_start = 0

    # Try to parse first two lines as fall boundaries (single ints)
    if len(lines) >= 2:
        try:
            val1 = int(lines[0])
            val2 = int(lines[1])
            # If they don't contain commas, they're fall boundaries
            if "," not in lines[0] and "," not in lines[1]:
                fall_start = val1
                fall_end = val2
                data_start = 2
        except ValueError:
            pass  # No fall boundaries, data starts at line 0

    frames = []
    for line in lines[data_start:]:
        parts = line.split(",")
        if len(parts) >= 6:
            try:
                frames.append({
                    "frame": int(parts[0]),
                    "label": int(parts[1]),
                    "x": int(parts[2]),
                    "y": int(parts[3]),
                    "w": int(parts[4]),
                    "h": int(parts[5]),
                })
            except (ValueError, IndexError):
                continue

    return fall_start, fall_end, frames


def has_fall(filepath: str) -> bool:
    """Check if annotation file indicates a fall event."""
    start, end, _ = parse_annotation(filepath)
    return start > 0 and end > start


# ─── Video/Audio Decoding ──────────────────────────────────────────────


def decode_video_audio(
    video_path: str,
    target_fps: float = 24.0,
    target_size: Tuple[int, int] = (224, 224),
    max_frames: int = -1,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, float]:
    """
    Decode video frames and audio from an AVI file.

    Args:
        video_path: path to .avi file
        target_fps: target frame rate for sampling
        target_size: (H, W) for frame resize
        max_frames: max frames to extract (-1 = all)

    Returns:
        frames: (T, C, H, W) tensor, float32 in [0, 1]
        audio: (samples,) tensor or None if no audio track
        total_frames: number of frames in video
        fps: actual video fps
    """
    container = av.open(video_path)

    # ── Decode video ──
    video_stream = container.streams.video[0]
    actual_fps = float(video_stream.average_rate)

    # Calculate frame sampling
    if target_fps and actual_fps > 0:
        stride = max(1, int(actual_fps / target_fps))
    else:
        stride = 1

    frames_list = []
    frame_idx = 0
    for frame in container.decode(video_stream):
        if max_frames > 0 and frame_idx >= max_frames:
            break
        if frame_idx % stride == 0:
            img = frame.to_ndarray(format="rgb24")  # (H, W, 3)
            img = torch.from_numpy(img).float() / 255.0
            img = img.permute(2, 0, 1)  # (C, H, W)
            # Resize
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            frames_list.append(img)
        frame_idx += 1

    frames = torch.stack(frames_list) if frames_list else torch.zeros(0, 3, *target_size)
    total_frames = frame_idx

    # ── Decode audio ──
    audio = None
    audio_sample_rate = None
    try:
        audio_stream = container.streams.audio[0]
        audio_sample_rate = audio_stream.sample_rate
        audio_chunks = []
        # Use decode() which handles interleaving better
        for frame in container.decode(audio_stream):
            try:
                samples = frame.to_ndarray()
                audio_chunks.append(samples.flatten())
            except Exception:
                continue
        if audio_chunks:
            audio = torch.from_numpy(np.concatenate(audio_chunks)).float()
    except (IndexError, Exception) as e:
        pass  # no audio or corrupt stream — video-only fallback

    container.close()
    return frames, audio, total_frames, actual_fps, audio_sample_rate


def extract_frame_range(
    frames: torch.Tensor,
    start_frame: int,
    end_frame: int,
    num_sample_frames: int = 8,
) -> torch.Tensor:
    """
    Extract and uniformly sample frames from a range.

    Args:
        frames: (T, C, H, W)
        start_frame: start index (inclusive)
        end_frame: end index (exclusive)
        num_sample_frames: number of frames to sample

    Returns:
        sampled: (num_sample_frames, C, H, W)
    """
    segment = frames[start_frame:end_frame]
    T = segment.shape[0]
    if T <= num_sample_frames:
        # Pad with last frame
        pad = num_sample_frames - T
        if pad > 0:
            segment = torch.cat([segment, segment[-1:].expand(pad, -1, -1, -1)], dim=0)
        indices = torch.arange(num_sample_frames)
    else:
        indices = torch.linspace(0, T - 1, num_sample_frames).long()
    return segment[indices]


def extract_audio_segment(
    audio: torch.Tensor,
    start_time: float,
    end_time: float,
    sample_rate: int = 11025,
) -> torch.Tensor:
    """
    Extract audio segment by time.

    Args:
        audio: (samples,) tensor
        start_time: start time in seconds
        end_time: end time in seconds
        sample_rate: audio sample rate

    Returns:
        segment: (samples_in_range,) tensor
    """
    if audio is None:
        return torch.zeros(1)

    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    segment = audio[start_sample:end_sample]

    if len(segment) == 0:
        return torch.zeros(1)

    return segment


# ─── Le2i Dataset Class ─────────────────────────────────────────────────


class Le2iDataset(Dataset):
    """
    Le2i dataset for AV-JEPA training/evaluation.

    Each sample = (context_frames, context_audio, target_frames, target_audio, label)
    where label = 0 (normal) or 1 (fall).

    For JEPA training: only normal clips are used.
    For evaluation: both normal and fall clips.
    """

    def __init__(
        self,
        root_dir: str = "/home/yangkailiang/.cache/kagglehub/datasets/tuyenldvn/falldataset-imvia/versions/2",
        split: str = "train",
        clip_duration: float = 2.0,       # seconds per clip
        context_duration: float = 2.0,     # context clip duration
        target_gap: float = 1.0,           # gap between context end and target start
        target_fps: float = 12.0,          # frames per second for sampling
        frame_size: Tuple[int, int] = (224, 224),
        num_frames: int = 8,               # frames per clip
        audio_sample_rate: int = 11025,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.clip_duration = clip_duration
        self.context_duration = context_duration
        self.target_gap = target_gap
        self.target_fps = target_fps
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.audio_sample_rate = audio_sample_rate

        # Frame conversion
        self.context_frames_clip = int(context_duration * target_fps)
        self.target_frames_clip = int(clip_duration * target_fps)
        self.gap_frames = int(target_gap * target_fps)

        # Collect all video clips
        self.samples = []  # list of (video_path, annot_path, is_fall, start_frame)
        self._scan_dataset()

    def _scan_dataset(self):
        """Scan dataset and build sample list."""
        scenes = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

        for scene in scenes:
            scene_inner = os.path.join(self.root_dir, scene, scene)
            video_dir = os.path.join(scene_inner, "Videos")
            # Handle dataset inconsistency: "Annotation_files" vs "Annotations_files"
            annot_dir = os.path.join(scene_inner, "Annotation_files")
            if not os.path.exists(annot_dir):
                annot_dir = os.path.join(scene_inner, "Annotations_files")

            if not os.path.exists(video_dir) or not os.path.exists(annot_dir):
                continue

            for video_name in sorted(os.listdir(video_dir)):
                video_path = os.path.join(video_dir, video_name)
                video_num = re.search(r"\((\d+)\)", video_name)
                if not video_num:
                    continue

                annot_path = os.path.join(
                    annot_dir, f"video ({video_num.group(1)}).txt"
                )
                if not os.path.exists(annot_path):
                    continue

                # Skip videos with broken audio
                if not self._has_valid_audio(video_path):
                    continue

                is_fall = has_fall(annot_path)
                fall_start, fall_end, _ = parse_annotation(annot_path)
                n_frames = self._count_frames(video_path)

                # Generate clips
                stride_frames = max(1, int(self.clip_duration * self.target_fps * 0.5))
                clip_start = 0

                while clip_start + self.context_frames_clip + self.gap_frames + self.target_frames_clip <= n_frames:
                    # Determine if this clip covers the fall
                    ctx_start = clip_start
                    ctx_end = clip_start + self.context_frames_clip
                    tgt_start = ctx_end + self.gap_frames
                    tgt_end = tgt_start + self.target_frames_clip

                    # Check if context or target overlaps with fall
                    clip_has_fall = False
                    if is_fall:
                        clip_has_fall = (
                            (ctx_start < fall_end and ctx_end > fall_start) or
                            (tgt_start < fall_end and tgt_end > fall_start)
                        )

                    # Training: only normal clips
                    if self.split == "train" and not clip_has_fall:
                        self.samples.append((
                            video_path, annot_path, 0,
                            ctx_start, ctx_end, tgt_start, tgt_end,
                        ))
                    # Eval: all clips
                    elif self.split == "eval":
                        self.samples.append((
                            video_path, annot_path,
                            1 if clip_has_fall else 0,
                            ctx_start, ctx_end, tgt_start, tgt_end,
                        ))

                    clip_start += stride_frames

    def _has_valid_audio(self, video_path: str) -> bool:
        """Check if video has a decodable audio track."""
        try:
            import av
            c = av.open(video_path)
            if len(c.streams.audio) == 0:
                c.close()
                return False
            for frame in c.decode(c.streams.audio[0]):
                break  # at least one frame decodes successfully
            c.close()
            return True
        except Exception:
            return False

    def _count_frames(self, video_path: str) -> int:
        """Count actual frames in video using PyAV."""
        try:
            import av
            c = av.open(video_path)
            n = c.streams.video[0].frames
            c.close()
            return max(n, 1) if n > 0 else 500
        except Exception:
            return 500

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, annot_path, label, ctx_s, ctx_e, tgt_s, tgt_e = self.samples[idx]

        # Decode video+audio
        try:
            frames, audio, total_frames, fps, audio_sr = decode_video_audio(
                video_path,
                target_fps=self.target_fps,
                target_size=self.frame_size,
            )
        except Exception:
            return self._get_dummy()

        # Use actual audio sample rate if available
        effective_audio_sr = audio_sr if audio_sr else self.audio_sample_rate

        n_frames = frames.shape[0]
        if n_frames < self.num_frames:
            return self._get_dummy()

        # Clamp frame indices to valid range
        ctx_s = min(ctx_s, n_frames)
        ctx_e = min(ctx_e, n_frames)
        tgt_s = min(tgt_s, n_frames)
        tgt_e = min(tgt_e, n_frames)

        # Ensure minimum clip length
        if ctx_e - ctx_s < 4 or tgt_e - tgt_s < 4:
            return self._get_dummy()

        # Extract context frames
        ctx_frames = extract_frame_range(frames, ctx_s, ctx_e, self.num_frames)
        # Extract target frames
        tgt_frames = extract_frame_range(frames, tgt_s, tgt_e, self.num_frames)

        # Extract audio segments
        ctx_audio_s = ctx_s / self.target_fps
        ctx_audio_e = ctx_e / self.target_fps
        tgt_audio_s = tgt_s / self.target_fps
        tgt_audio_e = tgt_e / self.target_fps

        if audio is not None and audio.numel() > 1:
            ctx_audio = extract_audio_segment(
                audio, ctx_audio_s, ctx_audio_e, effective_audio_sr
            )
            tgt_audio = extract_audio_segment(
                audio, tgt_audio_s, tgt_audio_e, effective_audio_sr
            )
        else:
            ctx_audio = torch.zeros(1)
            tgt_audio = torch.zeros(1)

        return ctx_frames, ctx_audio, tgt_frames, tgt_audio, torch.tensor(label, dtype=torch.long)

    def _get_dummy(self):
        """Return a dummy sample for videos that fail to decode."""
        dummy_frames = torch.randn(self.num_frames, 3, *self.frame_size)
        dummy_audio = torch.zeros(1)
        return dummy_frames, dummy_audio, dummy_frames, dummy_audio, torch.tensor(-1, dtype=torch.long)


def create_le2i_dataloaders(
    root_dir: str = "/home/yangkailiang/.cache/kagglehub/datasets/tuyenldvn/falldataset-imvia/versions/2",
    batch_size: int = 8,
    num_workers: int = 0,
    **dataset_kwargs,
):
    """
    Create train (normal only) and eval (mixed) dataloaders.

    Returns:
        train_loader, eval_loader
    """
    train_dataset = Le2iDataset(
        root_dir=root_dir,
        split="train",
        **dataset_kwargs,
    )
    eval_dataset = Le2iDataset(
        root_dir=root_dir,
        split="eval",
        **dataset_kwargs,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"Le2i Dataset:")
    print(f"  Train (normal only): {len(train_dataset)} clips")
    print(f"  Eval (mixed):        {len(eval_dataset)} clips")
    n_fall = sum(1 for s in eval_dataset.samples if s[2] == 1)
    n_normal = sum(1 for s in eval_dataset.samples if s[2] == 0)
    print(f"    Eval falls:         {n_fall}")
    print(f"    Eval normal:        {n_normal}")

    return train_loader, eval_loader


# ─── Quick test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    print("Testing Le2i data loading...")
    dataset = Le2iDataset(split="train")
    print(f"Total train clips: {len(dataset)}")

    if len(dataset) > 0:
        ctx_f, ctx_a, tgt_f, tgt_a, label = dataset[0]
        print(f"  Context frames: {ctx_f.shape}")  # (8, 3, 224, 224)
        print(f"  Context audio:  {ctx_a.shape}")
        print(f"  Target frames:  {tgt_f.shape}")
        print(f"  Target audio:   {tgt_a.shape}")
        print(f"  Label:          {label.item()}")
    else:
        print("  ⚠️  No clips generated! Check clip parameters.")

    # Eval
    eval_dataset = Le2iDataset(split="eval")
    print(f"\nTotal eval clips: {len(eval_dataset)}")
    falls = sum(1 for s in eval_dataset.samples if s[2] == 1)
    print(f"  Fall clips: {falls}")
    print(f"  Normal clips: {len(eval_dataset) - falls}")
