"""
Balanced Le2i Dataset: video-level train/test split with multi-window negatives.

Reads le2i_split.json and produces fixed context windows with controlled
leak levels (0-16) for fall videos.

Each fall video contributes exactly one sample at its assigned leak level.
Each non-fall video contributes multiple negative samples via sliding window
(16 context + 8 future frames, all outside any fall range).
"""

import os
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional


class BalancedLe2iDataset(Dataset):
    """Video-level split dataset with balanced leak level assignment.

    Reads from pre-processed .pt files under data_root. The split JSON
    specifies which videos belong to train/test and their context window.
    """

    def __init__(
        self,
        split_json: str,
        split: str = 'train',
        data_root: str = 'Le2i_processed',
        num_context_frames: int = 16,
        num_future_frames: int = 8,
        neg_stride: int = 32,
    ):
        self.data_root = data_root
        self.split = split
        self.num_context_frames = num_context_frames
        self.num_future_frames = num_future_frames
        self.neg_stride = neg_stride
        self.window_size = num_context_frames + num_future_frames

        with open(split_json, 'r') as f:
            self.split_data = json.load(f)

        self.metadata = self.split_data['metadata']

        # Filter videos for this split
        self.videos = {}
        for name, info in self.split_data['videos'].items():
            if info['split'] == split:
                self.videos[name] = info

        self.video_names = sorted(self.videos.keys())

        # Build sample index: (video_name, ctx_start)
        self.samples = []
        for name in self.video_names:
            info = self.videos[name]
            if info['is_fall']:
                # Fall video: 1 sample at the assigned leak-level window
                self.samples.append((name, info['ctx_start']))
            else:
                # Non-fall video: enumerate multiple valid windows
                n_frames = info['n_frames']
                max_start = n_frames - self.window_size
                if max_start >= 0:
                    for start in range(0, max_start + 1, neg_stride):
                        self.samples.append((name, start))
                elif n_frames >= num_context_frames:
                    # Short video: at least get context frames
                    self.samples.append((name, 0))
                # else: skip videos too short even for context

        n_fall = sum(1 for v in self.videos.values() if v['is_fall'])
        n_nonfall = len(self.videos) - n_fall
        n_neg_samples = len(self.samples) - n_fall

        # Pre-compute sample labels for WeightedRandomSampler
        self.labels = torch.zeros(len(self.samples), dtype=torch.float32)
        for i, (name, _) in enumerate(self.samples):
            if self.videos[name]['is_fall']:
                self.labels[i] = 1.0

        print(f"[BalancedLe2iDataset] {split}: {len(self.videos)} videos "
              f"({n_fall} fall, {n_nonfall} non-fall) → {len(self.samples)} samples "
              f"({n_fall} pos, {n_neg_samples} neg, stride={neg_stride})")

    def __len__(self) -> int:
        return len(self.samples)

    def _resample_audio(self, audio: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample to 16kHz, then time-stretch to 48000 samples via linear interpolation.

        Video context is 16 frames = 0.64s. WavJEPA expects 3.0s (48000@16kHz).
        We stretch the 0.64s audio segment by ~4.69x to fill the 3s window,
        keeping audio and video synchronized in content.
        """
        target_sr = 16000
        target_len = 48000
        if audio.numel() <= 1:
            return torch.zeros(target_len)

        # Step 1: Resample to 16kHz (preserves 0.64s duration, changes sample count)
        if orig_sr != target_sr:
            from torch.nn.functional import interpolate
            a = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
            new_len = int(len(audio) * target_sr / orig_sr)
            a = interpolate(a, size=new_len, mode='linear', align_corners=False)
            audio = a.squeeze()

        # Step 2: Time-stretch to exactly 48000 samples (0.64s → 3.0s)
        if audio.numel() > 1:
            from torch.nn.functional import interpolate
            a = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
            a = interpolate(a, size=target_len, mode='linear', align_corners=False)
            audio = a.squeeze()

        # Safety: ensure exactly target_len
        if audio.shape[0] < target_len:
            audio = torch.cat([audio, torch.zeros(target_len - audio.shape[0])])
        elif audio.shape[0] > target_len:
            audio = audio[:target_len]

        return audio

    def __getitem__(self, idx: int) -> Dict:
        name, ctx_start = self.samples[idx]
        info = self.videos[name]
        vd = os.path.join(self.data_root, name)

        ctx_end = ctx_start + self.num_context_frames

        # Load frames and slice context window
        frames_path = os.path.join(vd, 'frames.pt')
        frames = torch.load(frames_path, map_location='cpu', weights_only=True)
        n_total = frames.shape[0]

        # Clamp if needed
        if ctx_end > n_total:
            ctx_start = max(0, n_total - self.num_context_frames)
            ctx_end = ctx_start + self.num_context_frames
        ctx_frames = frames[ctx_start:ctx_end].clone()

        # Handle edge case: if we have fewer frames, pad
        if ctx_frames.shape[0] < self.num_context_frames:
            pad = ctx_frames[-1:].repeat(self.num_context_frames - ctx_frames.shape[0], 1, 1, 1)
            ctx_frames = torch.cat([ctx_frames, pad], dim=0)

        # Load real audio, align with context window timing
        audio_path = os.path.join(vd, 'audio.pt')
        meta_path = os.path.join(vd, 'meta.pt')
        audio = torch.load(audio_path, map_location='cpu', weights_only=True)
        meta = torch.load(meta_path, map_location='cpu', weights_only=True)
        audio_sr = meta.get('audio_sr', 44100)

        if audio.numel() > 1:
            audio_per_frame = len(audio) / max(n_total, 1)
            a_start = int(ctx_start * audio_per_frame)
            a_end = int(ctx_end * audio_per_frame)
            ctx_audio = audio[a_start:a_end]
        else:
            ctx_audio = torch.zeros(1)

        audio = self._resample_audio(ctx_audio, audio_sr)

        # Label and leak
        is_fall = info['is_fall']
        label = 1.0 if is_fall else 0.0
        n_fall_frames = info.get('leak', 0) if is_fall else 0

        return {
            'frames': ctx_frames,
            'audio': audio,
            'label': torch.tensor(label, dtype=torch.float32),
            'n_fall_frames': n_fall_frames,
            'video_path': vd,
        }
