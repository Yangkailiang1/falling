"""
Skeleton keypoints dataset for JEPA pretraining and fall classification.

Data format:
  - Keypoints: (T, 17, 2) float32 - COCO 17-keypoint xy coordinates
  - Confidence: (T, 17) float32 - per-keypoint confidence scores
  - le2i_split.json: video-level train/test split with fall annotations

Modes:
  - JEPA mode: returns (context_kp, target_kp) for self-supervised future prediction
  - Classification mode: returns (context_kp, label) for supervised fall detection
"""

import json
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class SkeletonDataset(Dataset):
    """Skeleton keypoints dataset supporting JEPA and classification modes.

    JEPA mode: context window → encoder → predict target window representation.
      Context: [t, t+ctx_frames)
      Target:  [t+ctx_frames+gap, t+ctx_frames+gap+tgt_frames)
      Only from normal (non-fall) videos.

    Classification mode: context window → label = fall in next future_frames?
      Context: [t, t+ctx_frames)
      Label: 1 if any fall frame in [t+ctx_frames, t+ctx_frames+future_frames)
      From both fall and non-fall videos.
    """

    def __init__(
        self,
        data_root: str = "le2i_keypoints",
        split_json: str = "le2i_split.json",
        split: str = "train",
        mode: str = "classification",
        num_context_frames: int = 16,
        num_target_frames: int = 16,
        num_future_frames: int = 32,
        gap_frames: int = 16,
        neg_stride: int = 16,
        normalize: bool = True,
        augment: bool = False,
        seed: int = 42,
        jepa_normal_only: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.mode = mode
        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames
        self.num_future_frames = num_future_frames
        self.gap_frames = gap_frames
        self.neg_stride = neg_stride
        self.normalize = normalize
        self.augment = augment
        self.jepa_normal_only = jepa_normal_only

        # Load split
        with open(split_json) as f:
            split_data = json.load(f)
        self.videos = split_data["videos"]

        # Scene names that have keypoint data
        self._kp_scenes = {"Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"}

        # Build sample index
        self.samples = self._build_index()
        self._rng = np.random.RandomState(seed)

    def _split_name_to_kp_name(self, vname: str) -> Optional[str]:
        """Map split video name to keypoint file basename."""
        for scene in self._kp_scenes:
            if vname.startswith(scene):
                m = re.search(r"\((\d+)\)", vname)
                if m:
                    return f"{scene}_video_{m.group(1)}"
        return None

    def _find_kp_file(self, kp_name: str) -> Optional[Tuple[Path, str]]:
        """Find keypoint file path and label (fall/normal)."""
        for label in ["fall", "normal"]:
            kp_path = self.data_root / label / f"{kp_name}_keypoints.npy"
            if kp_path.exists():
                return kp_path, label
        return None

    def _build_index(self) -> List[Dict]:
        samples = []
        for vname, info in self.videos.items():
            if info["split"] != self.split:
                continue
            is_fall = info.get("is_fall", False)

            kp_name = self._split_name_to_kp_name(vname)
            if kp_name is None:
                continue
            result = self._find_kp_file(kp_name)
            if result is None:
                continue
            kp_path, kp_label = result

            kp = np.load(kp_path)  # (T, 17, 2)
            conf_path = str(kp_path).replace("_keypoints.npy", "_confs.npy")
            conf = np.load(conf_path) if os.path.exists(conf_path) else np.ones((len(kp), 17), dtype=np.float32)

            T = len(kp)

            if self.mode == "jepa":
                ctx_tgt_total = self.num_context_frames + self.gap_frames + self.num_target_frames
                if T >= ctx_tgt_total:
                    if self.jepa_normal_only and is_fall:
                        continue
                    stride = max(1, self.num_target_frames // 2)
                    for start in range(0, T - ctx_tgt_total + 1, stride):
                        samples.append({
                            "kp_path": str(kp_path),
                            "conf_path": conf_path,
                            "ctx_start": start,
                            "tgt_start": start + self.num_context_frames + self.gap_frames,
                            "is_fall": is_fall,
                            "n_frames": T,
                        })
            else:  # classification
                ctx_plus_future = self.num_context_frames + self.num_future_frames
                fall_start = info.get("fall_start", -1)
                fall_end = info.get("fall_end", -1)
                leak_frames = info.get("leak", 0)

                if is_fall and fall_start >= 0:
                    # Positive sample: context ends just before pre-fall, label=1 means fall
                    # Use leak level from split: ctx_start is calculated from annotation
                    ctx_start = info.get("ctx_start", 0)
                    if ctx_start + ctx_plus_future <= T:
                        future_start = ctx_start + self.num_context_frames
                        future_end = ctx_start + ctx_plus_future
                        has_fall = future_start < fall_end and fall_start < future_end
                        label = 1.0 if has_fall else 0.0
                        n_fall_in_future = max(0, min(future_end, fall_end) - max(future_start, fall_start))
                        samples.append({
                            "kp_path": str(kp_path),
                            "conf_path": conf_path,
                            "ctx_start": ctx_start,
                            "future_start": future_start,
                            "future_end": future_end,
                            "label": label,
                            "is_fall": is_fall,
                            "leak": leak_frames,
                            "n_fall_frames": n_fall_in_future,
                            "n_frames": T,
                        })
                else:
                    # Negative samples: stride through non-fall video
                    if T >= ctx_plus_future:
                        for start in range(0, T - ctx_plus_future + 1, self.neg_stride):
                            samples.append({
                                "kp_path": str(kp_path),
                                "conf_path": conf_path,
                                "ctx_start": start,
                                "future_start": start + self.num_context_frames,
                                "future_end": start + ctx_plus_future,
                                "label": 0.0,
                                "is_fall": False,
                                "leak": -1,
                                "n_fall_frames": 0,
                                "n_frames": T,
                            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_keypoints(self, kp_path: str, conf_path: str, start: int, length: int) -> np.ndarray:
        """Load and preprocess a keypoint window.

        Returns:
            (T, 17, 3) array with (x, y, confidence) stacked.
        """
        kp = np.load(kp_path)  # (total_T, 17, 2)
        conf = np.load(conf_path) if os.path.exists(conf_path) else np.ones((len(kp), 17), dtype=np.float32)

        end = min(start + length, len(kp))
        kp_win = kp[start:end].copy()  # (window_T, 17, 2)
        conf_win = conf[start:end].copy()  # (window_T, 17)

        actual_len = kp_win.shape[0]
        if actual_len < length:
            pad = length - actual_len
            kp_win = np.pad(kp_win, ((0, pad), (0, 0), (0, 0)), mode="edge")
            conf_win = np.pad(conf_win, ((0, pad), (0, 0)), mode="edge")

        if self.normalize:
            kp_win, conf_win = self._normalize_keypoints(kp_win, conf_win)

        # Stack: (T, 17, 3)
        result = np.concatenate([kp_win, conf_win[..., None]], axis=-1).astype(np.float32)
        return result

    def _normalize_keypoints(self, kp: np.ndarray, conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize keypoints: center at hip midpoint, scale by shoulder-hip distance.

        Uses COCO keypoint indices:
          5=left shoulder, 6=right shoulder, 11=left hip, 12=right hip

        Returns normalized (x, y) and thresholded confidence.
        """
        # Hip center = midpoint of left and right hips
        hip_left = kp[:, 11, :]   # (T, 2)
        hip_right = kp[:, 12, :]  # (T, 2)

        # Handle zero/occluded keypoints by using available ones
        hip_center = np.zeros_like(hip_left)
        for t in range(len(kp)):
            valid_hips = []
            for idx in [11, 12]:
                if conf[t, idx] > 0.3:
                    valid_hips.append(kp[t, idx])
            if len(valid_hips) >= 2:
                hip_center[t] = (valid_hips[0] + valid_hips[1]) / 2
            elif len(valid_hips) == 1:
                hip_center[t] = valid_hips[0]
            elif conf[t, 5] > 0.3 and conf[t, 6] > 0.3:
                # Fallback to shoulder midpoint
                hip_center[t] = (kp[t, 5] + kp[t, 6]) / 2
            # else keep as zeros

        # Shoulder midpoint
        shoulder_center = np.zeros_like(hip_left)
        for t in range(len(kp)):
            if conf[t, 5] > 0.3 and conf[t, 6] > 0.3:
                shoulder_center[t] = (kp[t, 5] + kp[t, 6]) / 2
            elif conf[t, 5] > 0.3:
                shoulder_center[t] = kp[t, 5]
            elif conf[t, 6] > 0.3:
                shoulder_center[t] = kp[t, 6]

        # Scale factor: mean shoulder-hip distance across frames
        torso_dist = np.linalg.norm(shoulder_center - hip_center, axis=-1)  # (T,)
        torso_dist = torso_dist[torso_dist > 0.01]
        scale = np.median(torso_dist) if len(torso_dist) > 0 else 1.0
        scale = max(scale, 1e-6)

        # Center at hip and scale
        kp_norm = kp - hip_center[:, None, :]  # center at hips
        kp_norm = kp_norm / scale

        # Zero out low-confidence keypoints
        low_conf = conf < 0.1
        kp_norm[low_conf] = 0.0

        return kp_norm.astype(np.float32), conf.astype(np.float32)

    def _augment_keypoints(self, kp_3d: np.ndarray) -> np.ndarray:
        """Apply random augmentations to normalized keypoints (T, 17, 3).

        Safe operations for fall detection:
        - Small rotation around origin (hip-centered already)
        - Slight scaling
        - Gaussian jitter
        - Keypoint occlusion (dropout)
        """
        T = kp_3d.shape[0]
        kp = kp_3d[..., :2].copy()      # (T, 17, 2)
        conf = kp_3d[..., 2].copy()     # (T, 17)

        # 1. Random rotation ±10° around origin (hip-centered coords)
        angle = np.random.uniform(-10, 10) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        kp = kp @ rot.T

        # 2. Random scaling 0.9-1.1
        scale = np.random.uniform(0.9, 1.1)
        kp = kp * scale

        # 3. Gaussian jitter (σ = 0.02 in normalized coords ≈ 2% of body scale)
        noise = np.random.randn(*kp.shape).astype(np.float32) * 0.02
        kp = kp + noise

        # 4. Random keypoint dropout (10% chance per keypoint per frame)
        dropout_mask = np.random.rand(T, 17) < 0.1
        kp[dropout_mask] = 0.0
        conf[dropout_mask] = 0.0

        return np.concatenate([kp, conf[..., None]], axis=-1).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        if self.mode == "jepa":
            ctx_kp = self._load_keypoints(
                sample["kp_path"], sample["conf_path"],
                sample["ctx_start"], self.num_context_frames,
            )
            tgt_kp = self._load_keypoints(
                sample["kp_path"], sample["conf_path"],
                sample["tgt_start"], self.num_target_frames,
            )
            if self.augment:
                ctx_kp = self._augment_keypoints(ctx_kp)
            return {
                "context": torch.from_numpy(ctx_kp),
                "target": torch.from_numpy(tgt_kp),
                "is_fall": sample["is_fall"],
            }
        else:
            ctx_kp = self._load_keypoints(
                sample["kp_path"], sample["conf_path"],
                sample["ctx_start"], self.num_context_frames,
            )
            if self.augment:
                ctx_kp = self._augment_keypoints(ctx_kp)
            return {
                "context": torch.from_numpy(ctx_kp),
                "label": torch.tensor(sample["label"], dtype=torch.float32),
                "leak": sample.get("leak", -1),
                "n_fall_frames": sample.get("n_fall_frames", 0),
                "is_fall": sample["is_fall"],
            }


def jepa_collate(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate for JEPA mode: context and target keypoint sequences."""
    ctx = torch.stack([item["context"] for item in batch])  # (B, ctx_T, 17, 3)
    tgt = torch.stack([item["target"] for item in batch])   # (B, tgt_T, 17, 3)
    return ctx, tgt


def classification_collate(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate for classification mode."""
    ctx = torch.stack([item["context"] for item in batch])      # (B, ctx_T, 17, 3)
    labels = torch.stack([item["label"] for item in batch])     # (B,)
    leaks = torch.tensor([item.get("leak", -1) for item in batch])
    return ctx, labels, leaks
