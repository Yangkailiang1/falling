"""
NTU RGB+D 120 Skeleton Dataset for JEPA Pretraining.

Format: ASCII .skeleton files, 25 joints x 3D coordinates (x, y, z).
114,480 sequences from 106 subjects, 120 action classes, 3 camera views.

Parsing:
  Line 0: frame_count (int)
  Per frame:
    num_bodies (int, 1 or 2)
    For each body:
      1 line: tracking_metadata (10 space-separated values)
      1 line: joint_count (always 25)
      25 lines: x y z depthX depthY colorX colorY orientW orientX orientY orientZ state

We take only the FIRST body's (x,y,z) per frame.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple


class NTUSkeletonDataset(Dataset):
    """NTU120 skeleton dataset for JEPA self-supervised pretraining.

    Sliding window through each sequence: context (T_ctx frames) -> gap -> target (T_tgt frames).
    Returns normalized 3D keypoints: (T, 25, 3).
    """

    def __init__(
        self,
        data_root: str = "NTU120_skeleton/nturgbd_skeletons",
        num_context_frames: int = 16,
        num_target_frames: int = 16,
        gap_frames: int = 16,
        normalize: bool = True,
        max_files: Optional[int] = None,
        min_frames: int = 48,
        seed: int = 42,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames
        self.gap_frames = gap_frames
        self.normalize = normalize
        self.window_size = num_context_frames + gap_frames + num_target_frames

        self.files = self._collect_files()
        if max_files is not None:
            random.Random(seed).shuffle(self.files)
            self.files = self.files[:max_files]

        self.files = [f for f in self.files if self._get_frame_count(f) >= min_frames]
        self._windows = self._build_windows()

    def _collect_files(self) -> List[Path]:
        files = []
        for root, _dirs, filenames in os.walk(self.data_root):
            for fn in filenames:
                if fn.endswith(".skeleton"):
                    files.append(Path(root) / fn)
        return sorted(files)

    def _get_frame_count(self, path: Path) -> int:
        with open(path, "r") as f:
            return int(f.readline().strip())

    def _build_windows(self) -> List[Tuple[Path, int]]:
        windows = []
        stride = max(1, self.num_target_frames // 2)
        for fp in self.files:
            T = self._get_frame_count(fp)
            for start in range(0, T - self.window_size + 1, stride):
                windows.append((fp, start))
        return windows

    def __len__(self) -> int:
        return len(self._windows)

    def _parse_skeleton(self, path: Path, start: int, length: int) -> np.ndarray:
        """Parse a window of skeleton data.

        NTU format:
          Line 0: frame_count
          Per frame: num_bodies, then per body: tracking(10vals) + joint_count + 25 joints
        We take only the first body's (x,y,z) per frame.
        """
        with open(path, "r") as f:
            lines = f.readlines()

        fc = int(lines[0].strip())
        end = min(start + length, fc)
        data = []

        line_idx = 1
        for frame_num in range(fc):
            num_bodies = int(lines[line_idx].strip()); line_idx += 1

            frame_joints = None
            for body_idx in range(num_bodies):
                tracking = lines[line_idx].strip(); line_idx += 1  # skip
                n_joints = int(lines[line_idx].strip()); line_idx += 1

                if frame_num < start or frame_num >= end:
                    line_idx += n_joints  # skip joints
                    continue

                if body_idx == 0:
                    joints = []
                    for _ in range(n_joints):
                        parts = lines[line_idx].strip().split(); line_idx += 1
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        joints.append([x, y, z])
                    frame_joints = joints
                else:
                    line_idx += n_joints  # skip extra bodies

            if frame_joints is not None:
                data.append(frame_joints)

        if len(data) == 0:
            arr = np.zeros((length, 25, 3), dtype=np.float32)
        else:
            arr = np.array(data, dtype=np.float32)
            actual_len = arr.shape[0]
            if actual_len < length:
                pad = length - actual_len
                arr = np.pad(arr, ((0, pad), (0, 0), (0, 0)), mode="edge")

        if self.normalize:
            arr = self._normalize(arr)
        return arr

    def _normalize(self, kp: np.ndarray) -> np.ndarray:
        """Normalize: center at spine midpoint, scale by shoulder-hip distance.

        NTU joint indices:
          0=base_spine, 1=mid_spine, 4=left_shoulder, 8=right_shoulder,
          12=left_hip, 16=right_hip
        """
        spine = kp[:, 1, :]  # mid_spine
        shoulder_mid = (kp[:, 4, :] + kp[:, 8, :]) / 2
        hip_mid = (kp[:, 12, :] + kp[:, 16, :]) / 2

        torso_dist = np.linalg.norm(shoulder_mid - hip_mid, axis=-1)
        torso_dist = torso_dist[torso_dist > 0.01]
        scale = np.median(torso_dist) if len(torso_dist) > 0 else 1.0
        scale = max(scale, 1e-6)

        kp_norm = kp - spine[:, None, :]
        kp_norm = kp_norm / scale
        return kp_norm.astype(np.float32)

    def __getitem__(self, idx: int):
        fp, start = self._windows[idx]
        ctx = self._parse_skeleton(fp, start, self.num_context_frames)
        tgt_start = start + self.num_context_frames + self.gap_frames
        tgt = self._parse_skeleton(fp, tgt_start, self.num_target_frames)
        return torch.from_numpy(ctx), torch.from_numpy(tgt)


def ntu_jepa_collate(batch):
    ctx = torch.stack([item[0] for item in batch])
    tgt = torch.stack([item[1] for item in batch])
    return ctx, tgt


def create_ntu_jepa_loaders(
    data_root: str = "NTU120_skeleton/nturgbd_skeletons",
    num_context_frames: int = 16,
    num_target_frames: int = 16,
    gap_frames: int = 16,
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.02,
    max_files: Optional[int] = None,
    seed: int = 42,
):
    """Create train/val dataloaders for NTU JEPA pretraining."""
    full_ds = NTUSkeletonDataset(
        data_root=data_root,
        num_context_frames=num_context_frames,
        num_target_frames=num_target_frames,
        gap_frames=gap_frames,
        max_files=max_files,
        seed=seed,
    )

    n_total = len(full_ds)
    n_val = max(500, int(n_total * val_split))
    indices = list(range(n_total))
    random.Random(seed).shuffle(indices)

    train_ds = torch.utils.data.Subset(full_ds, indices[n_val:])
    val_ds = torch.utils.data.Subset(full_ds, indices[:n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=ntu_jepa_collate,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=ntu_jepa_collate,
        pin_memory=True,
    )
    return train_loader, val_loader
