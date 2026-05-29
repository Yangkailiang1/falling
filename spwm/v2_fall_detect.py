#!/usr/bin/env python3
"""
V-JEPA 2 + WavJEPA Fall Detection on Le2i Dataset

Architecture:
  Context frames (4) → [V-JEPA 2 ViT-L, frozen] → v_ctx (1024) ↘
  Context audio      → [WavJEPA, frozen]        → a_ctx (768)  ↗
                                                       ↓
                                               [JointProjector] → z_ctx (256)
                                                       ↓
                                               [JEPA Predictor] (3-layer Transformer)
                                                       ↓
  Target frames (4)  → [V-JEPA 2 ViT-L, frozen] → v_tgt (1024) ↘  z_pred (256)
  Target audio       → [WavJEPA, frozen]        → a_tgt (768)  ↗  ↓
                                                       ↓         MSE + SIGReg
                                               [EMA Projector] → z_target (256)

Training: unsupervised, normal activity only. Falls = high prediction error.

Usage:
  # Train + evaluate (CPU, recommended for 4GB VRAM):
  python spwm/v2_fall_detect.py --mode all --epochs 30 --device cpu

  # Train on GPU with encoder CPU offloading:
  python spwm/v2_fall_detect.py --mode all --epochs 30 --device cuda --encoder_device cpu

  # Evaluate only:
  python spwm/v2_fall_detect.py --mode eval --checkpoint checkpoints/v2_fall_detect.pt
"""

import os
import sys
import re
import copy
import math
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add project paths
_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT / "research" / "av-jepa"))

warnings.filterwarnings("ignore")
import av
av.logging.set_level(av.logging.PANIC)  # suppress FFmpeg log spam
# Nuclear option: also set via libavutil to AV_LOG_QUIET (-8)
try:
    import ctypes
    libav = ctypes.CDLL("libavutil.so")
    libav.av_log_set_level(-8)
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

class Config:
    # Video
    video_num_frames: int = 4
    video_frame_size: int = 224
    video_embed_dim: int = 1024  # V-JEPA 2 ViT-L

    # Audio
    audio_sample_rate: int = 16000
    audio_duration: float = 2.0  # seconds per clip
    audio_embed_dim: int = 768   # WavJEPA

    # Joint embedding
    joint_dim: int = 256

    # Predictor
    predictor_num_layers: int = 3
    predictor_num_heads: int = 4
    predictor_hidden_dim: int = 512
    predictor_dropout: float = 0.1

    # SIGReg
    sigreg_weight: float = 0.3
    sigreg_target_std: float = 1.0

    # EMA
    ema_decay_start: float = 0.996
    ema_decay_end: float = 1.0

    # Training
    batch_size: int = 4
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    context_sec: float = 0.5    # context window in seconds
    target_gap_sec: float = 0.3  # gap between context and target
    clip_stride_sec: float = 0.3

    # Data
    data_root: str = "Le2i"
    num_workers: int = 0  # PyAV not thread-safe, must be 0

    # Devices
    device: str = "cpu"
    encoder_device: str = "cpu"

    def update(self, args):
        for k, v in vars(args).items():
            if v is None:
                continue
            if k == "epochs":
                self.num_epochs = v
            elif hasattr(self, k):
                setattr(self, k, v)


cfg = Config()


# ═══════════════════════════════════════════════════════════════
# Encoders (frozen, from local weights)
# ═══════════════════════════════════════════════════════════════

class VJEPA2VideoEncoder(nn.Module):
    """V-JEPA 2 ViT-L video encoder. Loads from spwm/model_weights/."""

    def __init__(self):
        super().__init__()
        self.embed_dim = 1024
        self.num_frames = cfg.video_num_frames

        _weights = _PROJ_ROOT / "spwm" / "model_weights"
        _repo = str(_weights / "vjepa2")

        encoder, _predictor = torch.hub.load(
            _repo, "vjepa2_vit_large",
            source="local", pretrained=False, skip_validation=True,
        )

        ckpt_path = _weights / "vitl.pt"
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            state = ckpt.get("target_encoder", ckpt.get("encoder", ckpt))
            state = {k.replace("module.", "").replace("backbone.", ""): v
                     for k, v in state.items()}
            encoder.load_state_dict(state, strict=False)
            print(f"[VJEPA2] Loaded vitl.pt ({sum(p.numel() for p in encoder.parameters())/1e6:.0f}M)")
        else:
            print("[VJEPA2] WARNING: vitl.pt not found, using random init")

        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) e.g. (B, 4, 3, 224, 224)
        Returns:
            features: (B, embed_dim) pooled
        """
        B, T, C, H, W = frames.shape
        # V-JEPA 2 expects (B, C, T, H, W)
        x = frames.permute(0, 2, 1, 3, 4)
        x = (x - 0.5) / 0.5  # normalize to [-1, 1]
        feats = self.encoder(x)  # (B, num_patches, 1024)
        return feats.mean(dim=1)  # (B, 1024)


class WavJEPAAudioEncoder(nn.Module):
    """WavJEPA audio encoder. Loads from spwm/model_weights/wavjepa-base/."""

    def __init__(self):
        super().__init__()
        from transformers import AutoModel

        local_path = _PROJ_ROOT / "spwm" / "model_weights" / "wavjepa-base"
        model_name = str(local_path) if local_path.exists() else "labhamlet/wavjepa-base"

        print(f"[WavJEPA] Loading from: {model_name}")
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float32,
        )
        self.embed_dim = 768
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        print(f"[WavJEPA] Loaded ({sum(p.numel() for p in self.model.parameters())/1e6:.0f}M)")

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (B, samples) raw waveform at 16kHz
        Returns:
            features: (B, embed_dim) pooled
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # (B, L) → (B, 1, L)
        output = self.model(audio)
        if hasattr(output, "last_hidden_state"):
            feats = output.last_hidden_state
        elif isinstance(output, tuple):
            feats = output[0]
        else:
            feats = output
        if feats.dim() == 3:
            feats = feats.mean(dim=1)  # (B, N, D) → (B, D)
        return feats


# ═══════════════════════════════════════════════════════════════
# Projector + Predictor
# ═══════════════════════════════════════════════════════════════

class JointProjector(nn.Module):
    """MLP: concat(video, audio) → joint space."""

    def __init__(self, video_dim: int, audio_dim: int, joint_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(video_dim + audio_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, joint_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JEPAPredictor(nn.Module):
    """3-layer Transformer: z_ctx → z_pred."""

    def __init__(self, dim: int = 256, n_layers: int = 3,
                 n_heads: int = 4, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.ctx_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        B = z_ctx.shape[0]
        x = z_ctx.unsqueeze(1)  # (B, 1, D)
        ctx_token = self.ctx_token.expand(B, -1, -1)
        x = torch.cat([ctx_token, x], dim=1)
        x = self.transformer(x)
        return self.out(x[:, 0, :])


class SIGRegLoss(nn.Module):
    """Covariance regularization: push latent toward isotropic Gaussian."""

    def __init__(self, target_std: float = 1.0):
        super().__init__()
        self.target_std = target_std

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z.device)
        std = torch.std(z, dim=0).mean()
        return (std - self.target_std).pow(2) * 0.5


# ═══════════════════════════════════════════════════════════════
# Full V2 Fall Detection Model
# ═══════════════════════════════════════════════════════════════

class V2FallDetector(nn.Module):
    """V-JEPA 2 + WavJEPA for fall detection."""

    def __init__(self):
        super().__init__()
        self.video_encoder = VJEPA2VideoEncoder()
        self.audio_encoder = WavJEPAAudioEncoder()
        self.projector = JointProjector(cfg.video_embed_dim, cfg.audio_embed_dim, cfg.joint_dim)
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False
        self.predictor = JEPAPredictor(
            dim=cfg.joint_dim, n_layers=cfg.predictor_num_layers,
            n_heads=cfg.predictor_num_heads, hidden_dim=cfg.predictor_hidden_dim,
            dropout=cfg.predictor_dropout,
        )
        self.sigreg = SIGRegLoss(cfg.sigreg_target_std)

    def _encode_joint(self, frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Encode video + audio into joint embedding."""
        v = self.video_encoder(frames).to(frames.device)
        a = self.audio_encoder(audio).to(frames.device)
        return torch.cat([v, a], dim=-1)

    def encode_context(self, frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        joint = self._encode_joint(frames, audio)
        return self.projector(joint)

    @torch.no_grad()
    def encode_target(self, frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        joint = self._encode_joint(frames, audio)
        return self.target_projector(joint)

    def forward(self, ctx_frames, ctx_audio, tgt_frames, tgt_audio):
        z_ctx = self.encode_context(ctx_frames, ctx_audio)
        z_pred = self.predictor(z_ctx)
        z_target = self.encode_target(tgt_frames, tgt_audio)
        return z_pred, z_target, z_ctx

    @torch.no_grad()
    def compute_surprise(self, ctx_frames, ctx_audio, tgt_frames, tgt_audio):
        """Compute per-sample prediction error (surprise)."""
        z_pred, z_target, _ = self.forward(ctx_frames, ctx_audio, tgt_frames, tgt_audio)
        return F.mse_loss(z_pred, z_target, reduction="none").mean(dim=-1)

    def update_target_encoder(self, step: int, total_steps: int):
        tau = cfg.ema_decay_end - \
              (cfg.ema_decay_end - cfg.ema_decay_start) * \
              (math.cos(math.pi * step / total_steps) + 1) / 2
        for p_tgt, p_src in zip(self.target_projector.parameters(), self.projector.parameters()):
            p_tgt.data.copy_(tau * p_tgt.data + (1.0 - tau) * p_src.data)

    def place_encoders_on(self, device: torch.device):
        """Move frozen encoders to specified device."""
        self.video_encoder.encoder.to(device)
        self.audio_encoder.model.to(device)


# ═══════════════════════════════════════════════════════════════
# Le2i Dataset
# ═══════════════════════════════════════════════════════════════

def parse_annotation(filepath: str):
    """
    Parse Le2i annotation file.
    Returns (fall_start, fall_end, list_of_frame_dicts).
    Supports both header formats:
      - numeric_headers: first 2 lines are fall_start, fall_end
      - csv_only: all lines are "frame,label,x,y,w,h"
    """
    with open(filepath) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    fall_start, fall_end = 0, 0
    data_start = 0

    # Detect format: if first 2 lines are single numbers, they're headers
    if len(lines) >= 2:
        try:
            a, b = int(lines[0].split(",")[0].strip()), int(lines[1].split(",")[0].strip())
            # If these look like frame numbers with label data (csv format)
            # they would have commas. Check if they're pure numbers.
            if "," not in lines[0] and "," not in lines[1]:
                fall_start, fall_end = int(lines[0]), int(lines[1])
                data_start = 2
            # else: csv-only format, all lines are annotations
        except ValueError:
            pass

    frames = []
    for line in lines[data_start:]:
        parts = line.split(",")
        if len(parts) >= 6:
            try:
                frames.append({
                    "frame": int(parts[0]),
                    "label": int(parts[1]),
                    "x": int(parts[2]), "y": int(parts[3]),
                    "w": int(parts[4]), "h": int(parts[5]),
                })
            except (ValueError, IndexError):
                continue

    return fall_start, fall_end, frames


class Le2iJEPADataset(Dataset):
    """
    Le2i dataset for JEPA training.
    Each item: (ctx_frames, ctx_audio, tgt_frames, tgt_audio, has_fall)
    """

    def __init__(self, normal_only: bool = True):
        self.data_root = cfg.data_root
        self.normal_only = normal_only
        self.num_frames = cfg.video_num_frames
        self.frame_size = cfg.video_frame_size
        self.audio_sr = cfg.audio_sample_rate
        self.audio_samples = int(cfg.audio_duration * self.audio_sr)

        self.samples: List[Dict] = []
        self._scan()

    def _scan(self):
        """Scan Le2i directory for videos and annotations."""
        scenes = sorted(d for d in os.listdir(self.data_root)
                       if os.path.isdir(os.path.join(self.data_root, d)))

        for scene in scenes:
            inner = os.path.join(self.data_root, scene, scene)
            video_dir = os.path.join(inner, "Videos")
            annot_dir = os.path.join(inner, "Annotation_files")
            if not os.path.isdir(annot_dir):
                annot_dir = os.path.join(inner, "Annotations_files")

            if not os.path.isdir(video_dir) or not os.path.isdir(annot_dir):
                continue

            for fname in sorted(os.listdir(video_dir)):
                if not fname.lower().endswith(".avi"):
                    continue

                # Match video number: "video (N).avi"
                m = re.search(r"\((\d+)\)", fname)
                if not m:
                    continue
                vnum = m.group(1)
                video_path = os.path.join(video_dir, fname)
                annot_path = os.path.join(annot_dir, f"video ({vnum}).txt")
                if not os.path.exists(annot_path):
                    continue

                fall_start, fall_end, _ = parse_annotation(annot_path)
                has_fall = (fall_start > 0 and fall_end > fall_start)

                # Probe frame count via PyAV
                try:
                    container = av.open(video_path)
                    vs = container.streams.video[0]
                    n_frames = vs.frames or 0
                    fps = float(vs.average_rate) if vs.average_rate else 25.0
                    container.close()
                except Exception:
                    n_frames, fps = 300, 25.0

                if n_frames <= 0:
                    continue

                # Slide window over video
                ctx_len = int(cfg.context_sec * fps)
                gap_len = int(cfg.target_gap_sec * fps)
                stride = int(cfg.clip_stride_sec * fps)
                stride = max(1, stride)

                pos = 0
                while pos + ctx_len * 2 + gap_len <= n_frames:
                    ctx_s = pos
                    ctx_e = pos + ctx_len
                    tgt_s = ctx_e + gap_len
                    tgt_e = tgt_s + ctx_len

                    # Check fall overlap
                    clip_has_fall = False
                    if has_fall:
                        clip_has_fall = any(
                            (ctx_s <= f < ctx_e) or (tgt_s <= f < tgt_e)
                            for f in range(fall_start, fall_end + 1)
                        )

                    if self.normal_only and clip_has_fall:
                        pos += stride
                        continue

                    self.samples.append({
                        "video_path": video_path,
                        "ctx_s": ctx_s, "ctx_e": ctx_e,
                        "tgt_s": tgt_s, "tgt_e": tgt_e,
                        "has_fall": clip_has_fall,
                        "fps": fps,
                    })
                    pos += stride

        total_fall = sum(1 for s in self.samples if s["has_fall"])
        print(f"[Le2i] {'Normal-only' if self.normal_only else 'All'}: "
              f"{len(self.samples)} clips ({total_fall} with falls)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            container = av.open(s["video_path"])
            vs = container.streams.video[0]
            fps = s["fps"]

            # Decode all relevant frames
            all_frames = []
            min_f = min(s["ctx_s"], s["tgt_s"])
            max_f = max(s["ctx_e"], s["tgt_e"])
            frame_idx = 0
            for frame in container.decode(vs):
                if frame_idx > max_f:
                    break
                if min_f <= frame_idx < max_f:
                    img = frame.to_ndarray(format="rgb24")
                    img = torch.from_numpy(img.copy()).float() / 255.0
                    img = img.permute(2, 0, 1)  # (C, H, W)
                    if img.shape[1] != self.frame_size or img.shape[2] != self.frame_size:
                        img = F.interpolate(
                            img.unsqueeze(0), size=(self.frame_size, self.frame_size),
                            mode="bilinear", align_corners=False,
                        ).squeeze(0)
                    all_frames.append(img)
                frame_idx += 1

            # Extract context/target segments
            total = len(all_frames)
            ctx_offset = s["ctx_s"] - min_f
            tgt_offset = s["tgt_s"] - min_f
            ctx_len = s["ctx_e"] - s["ctx_s"]
            tgt_len = s["tgt_e"] - s["tgt_s"]

            def sample_segment(seg_list, offset, seg_len, target_n):
                end = min(offset + seg_len, len(seg_list))
                seg = seg_list[offset:end]
                if len(seg) == 0:
                    return torch.zeros(target_n, 3, self.frame_size, self.frame_size)
                T = len(seg)
                if T <= target_n:
                    idxs = torch.arange(target_n).clamp(0, T - 1)
                else:
                    idxs = torch.linspace(0, T - 1, target_n).long()
                return torch.stack([seg[i] for i in idxs])

            ctx_frames = sample_segment(all_frames, ctx_offset, ctx_len, self.num_frames)
            tgt_frames = sample_segment(all_frames, tgt_offset, tgt_len, self.num_frames)

            # Decode audio
            ctx_audio = torch.zeros(self.audio_samples)
            tgt_audio = torch.zeros(self.audio_samples)
            try:
                astream = container.streams.audio[0]
                asr = astream.sample_rate if astream.sample_rate else 48000
                audio_chunks = []
                for frame in container.decode(astream):
                    try:
                        audio_chunks.append(frame.to_ndarray().flatten())
                    except Exception:
                        continue
                if audio_chunks:
                    full_audio = torch.from_numpy(np.concatenate(audio_chunks)).float()
                    # Resample to target rate
                    if asr != self.audio_sr:
                        full_audio = F.interpolate(
                            full_audio.unsqueeze(0).unsqueeze(0),
                            size=int(len(full_audio) * self.audio_sr / asr),
                            mode="linear", align_corners=False,
                        ).squeeze()
                    # Map frame ranges to audio ranges
                    audio_per_frame = len(full_audio) / max(frame_idx, 1)
                    a_ctx_s = int(s["ctx_s"] * audio_per_frame)
                    a_ctx_e = a_ctx_s + self.audio_samples
                    a_tgt_s = int(s["tgt_s"] * audio_per_frame)
                    a_tgt_e = a_tgt_s + self.audio_samples

                    def slice_audio(audio, start, end, target_len):
                        if start >= len(audio):
                            return torch.zeros(target_len)
                        seg = audio[start:min(end, len(audio))]
                        if len(seg) < target_len:
                            seg = F.pad(seg, (0, target_len - len(seg)))
                        return seg[:target_len]

                    ctx_audio = slice_audio(full_audio, a_ctx_s, a_ctx_e, self.audio_samples)
                    tgt_audio = slice_audio(full_audio, a_tgt_s, a_tgt_e, self.audio_samples)
            except Exception:
                pass

            container.close()
            return ctx_frames, ctx_audio, tgt_frames, tgt_audio, torch.tensor(1.0 if s["has_fall"] else 0.0)

        except Exception as e:
            return (torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size),
                    torch.zeros(self.audio_samples),
                    torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size),
                    torch.zeros(self.audio_samples),
                    torch.tensor(-1.0))


def collate_jepa(batch):
    """Filter invalid samples, stack tensors."""
    valid = [(a, b, c, d, e) for a, b, c, d, e in batch if e.item() >= 0]
    if not valid:
        return None
    cf = torch.stack([x[0] for x in valid])
    ca = torch.stack([x[1] for x in valid])
    tf = torch.stack([x[2] for x in valid])
    ta = torch.stack([x[3] for x in valid])
    lb = torch.stack([x[4] for x in valid])
    return cf, ca, tf, ta, lb


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train():
    print("=" * 60)
    print("V-JEPA 2 + WavJEPA Fall Detection Training on Le2i")
    print("=" * 60)
    print(f"  Context frames: {cfg.video_num_frames}, Target frames: {cfg.video_num_frames}")
    print(f"  Context window: {cfg.context_sec}s, Gap: {cfg.target_gap_sec}s")
    print(f"  Device: {cfg.device}, Encoder device: {cfg.encoder_device}")
    print(f"  Epochs: {cfg.num_epochs}, Batch: {cfg.batch_size}, LR: {cfg.learning_rate}")

    train_dataset = Le2iJEPADataset(normal_only=True)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_jepa, drop_last=True,
    )

    eval_dataset = Le2iJEPADataset(normal_only=False)
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_jepa,
    )

    device = torch.device(cfg.device)
    enc_device = torch.device(cfg.encoder_device)

    model = V2FallDetector()
    model.place_encoders_on(enc_device)
    model.projector.to(device)
    model.target_projector.to(device)
    model.predictor.to(device)
    model.sigreg.to(device)

    trainable = list(model.projector.parameters()) + list(model.predictor.parameters())
    n_params = sum(p.numel() for p in trainable)
    print(f"  Trainable params: {n_params/1e3:.1f}K")

    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs * len(train_loader),
    )

    total_steps = cfg.num_epochs * len(train_loader)
    step = 0
    best_ratio = 0.0

    print(f"\n[Training] {cfg.num_epochs} epochs x {len(train_loader)} steps")
    print()

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_pred = 0.0
        epoch_sig = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            cf, ca, tf, ta, _ = batch

            # Move data to encoder device, encode, then bring to train device
            cf_enc = cf.to(enc_device)
            ca_enc = ca.to(enc_device)
            tf_enc = tf.to(enc_device)
            ta_enc = ta.to(enc_device)

            optimizer.zero_grad()
            z_pred, z_target, z_ctx = model(cf_enc, ca_enc, tf_enc, ta_enc)
            z_pred = z_pred.to(device)
            z_target = z_target.to(device)
            z_ctx = z_ctx.to(device)

            pred_loss = F.mse_loss(z_pred, z_target)
            sig_loss = model.sigreg(z_ctx) + model.sigreg(z_target)
            loss = pred_loss + cfg.sigreg_weight * sig_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.update_target_encoder(step, total_steps)
            step += 1

            epoch_loss += loss.item()
            epoch_pred += pred_loss.item()
            epoch_sig += sig_loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(f"  E{epoch+1:02d} [{batch_idx+1:4d}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} pred={pred_loss.item():.4f} sig={sig_loss.item():.4f}")

        n = max(len(train_loader), 1)
        print(f"--- Epoch {epoch+1} avg loss={epoch_loss/n:.4f} "
              f"pred={epoch_pred/n:.4f} sig={epoch_sig/n:.4f}")

        # Evaluation
        if (epoch + 1) % 5 == 0:
            ratio = evaluate_surprise(model, eval_loader, device, enc_device)
            print(f"--- Surprise ratio: {ratio:.2f}x (fall/normal)")
            if ratio > best_ratio:
                best_ratio = ratio
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": {
                        k: v for k, v in model.state_dict().items()
                        if not k.startswith(("video_encoder.encoder", "audio_encoder.model"))
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {k: v for k, v in vars(cfg).items()
                              if not k.startswith("__") and not callable(v)},
                }, "checkpoints/v2_fall_detect_best.pt")
                print(f"--- Saved best (ratio={ratio:.2f}x)")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "epoch": cfg.num_epochs - 1,
        "model_state_dict": {
            k: v for k, v in model.state_dict().items()
            if not k.startswith(("video_encoder.encoder", "audio_encoder.model"))
        },
        "config": {k: v for k, v in vars(cfg).items()
                   if not k.startswith("__") and not callable(v)},
    }, "checkpoints/v2_fall_detect.pt")
    print(f"\nTraining done. Best surprise ratio: {best_ratio:.2f}x")


def evaluate_surprise(model, loader, device, enc_device, max_batches=None):
    """Compute mean surprise for normal vs fall clips."""
    model.eval()
    normal_errs, fall_errs = [], []

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        if batch is None:
            continue
        cf, ca, tf, ta, lbs = batch
        cf_enc, ca_enc = cf.to(enc_device), ca.to(enc_device)
        tf_enc, ta_enc = tf.to(enc_device), ta.to(enc_device)

        surprise = model.compute_surprise(cf_enc, ca_enc, tf_enc, ta_enc).cpu()

        for j in range(len(lbs)):
            if lbs[j] > 0.5:
                fall_errs.append(surprise[j].item())
            else:
                normal_errs.append(surprise[j].item())

    if not normal_errs or not fall_errs:
        return 0.0
    return np.mean(fall_errs) / (np.mean(normal_errs) + 1e-8)


# ═══════════════════════════════════════════════════════════════
# Full Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate(checkpoint_path: str):
    print("=" * 60)
    print("V-JEPA 2 + WavJEPA Fall Detection Evaluation")
    print("=" * 60)

    device = torch.device(cfg.device)
    enc_device = torch.device(cfg.encoder_device)

    model = V2FallDetector()
    model.place_encoders_on(enc_device)
    model.projector.to(device)
    model.target_projector.to(device)
    model.predictor.to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    model.eval()

    eval_dataset = Le2iJEPADataset(normal_only=False)
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_jepa,
    )

    all_surprises = []
    all_labels = []
    all_paths = []

    print(f"Computing surprise on {len(eval_dataset)} clips...")
    with torch.no_grad():
        for batch in eval_loader:
            if batch is None:
                continue
            cf, ca, tf, ta, lbs = batch
            cf_enc, ca_enc = cf.to(enc_device), ca.to(enc_device)
            tf_enc, ta_enc = tf.to(enc_device), ta.to(enc_device)

            surprise = model.compute_surprise(cf_enc, ca_enc, tf_enc, ta_enc).cpu()
            all_surprises.extend(surprise.tolist())
            all_labels.extend(lbs.tolist())

    all_surprises = np.array(all_surprises)
    all_labels = np.array(all_labels)

    n_fall = (all_labels > 0.5).sum()
    n_normal = (all_labels <= 0.5).sum()

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"  Total clips:   {len(all_labels)}")
    print(f"  Fall clips:    {n_fall}")
    print(f"  Normal clips:  {n_normal}")

    if n_fall > 0 and n_normal > 0:
        fall_errs = all_surprises[all_labels > 0.5]
        normal_errs = all_surprises[all_labels <= 0.5]

        print(f"\n  Normal error:  {np.mean(normal_errs):.6f} +/- {np.std(normal_errs):.6f}")
        print(f"  Fall error:    {np.mean(fall_errs):.6f} +/- {np.std(fall_errs):.6f}")
        print(f"  Surprise ratio: {np.mean(fall_errs)/np.mean(normal_errs):.2f}x")

        sigma = (np.mean(fall_errs) - np.mean(normal_errs)) / (np.std(normal_errs) + 1e-8)
        print(f"  Separation:    {sigma:.2f} sigma")

        # AUROC
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
            auroc = roc_auc_score(all_labels, all_surprises)
            auprc = average_precision_score(all_labels, all_surprises)
            print(f"\n  AUROC:         {auroc:.4f}")
            print(f"  AUPRC:         {auprc:.4f}")
        except ImportError:
            pass

        # Best F1
        best_f1, best_th, best_p, best_r = 0, 0, 0, 0
        for pct in range(40, 100):
            th = np.percentile(all_surprises, pct)
            preds = (all_surprises > th).astype(float)
            tp = ((preds == 1) & (all_labels == 1)).sum()
            fp = ((preds == 1) & (all_labels == 0)).sum()
            fn = ((preds == 0) & (all_labels == 1)).sum()
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f1 = 2 * p * r / (p + r + 1e-8)
            if f1 > best_f1:
                best_f1, best_th, best_p, best_r = f1, pct, p, r

        print(f"\n  Best threshold: {best_th}th percentile")
        print(f"  Precision:      {best_p:.4f}")
        print(f"  Recall:         {best_r:.4f}")
        print(f"  F1:             {best_f1:.4f}")

        # Per-scene breakdown
        # (simplified - uses labels only)
        print(f"\n  Detection Rate: {best_r*100:.1f}% of falls detected")


# ═══════════════════════════════════════════════════════════════
# Quick Test: verify encoders load and dataset works
# ═══════════════════════════════════════════════════════════════

def quick_test():
    """Quick smoke test: load encoders, run one batch."""
    print("=" * 60)
    print("Quick Test: Encoder + Dataset Sanity Check")
    print("=" * 60)

    # Test dataset
    print("\n[1/3] Testing Le2i dataset...")
    ds = Le2iJEPADataset(normal_only=False)
    print(f"  Total clips: {len(ds)}")
    if len(ds) > 0:
        ctx_f, ctx_a, tgt_f, tgt_a, lbl = ds[0]
        print(f"  ctx_frames: {ctx_f.shape}, ctx_audio: {ctx_a.shape}")
        print(f"  tgt_frames: {tgt_f.shape}, tgt_audio: {tgt_a.shape}")
        print(f"  has_fall: {lbl.item()}")

    # Test encoder loading
    print("\n[2/3] Loading V-JEPA 2 + WavJEPA encoders...")
    try:
        video_enc = VJEPA2VideoEncoder()
        print("  V-JEPA 2: OK")
    except Exception as e:
        print(f"  V-JEPA 2 FAILED: {e}")
        return

    try:
        audio_enc = WavJEPAAudioEncoder()
        print("  WavJEPA: OK")
    except Exception as e:
        print(f"  WavJEPA FAILED: {e}")
        return

    # Test forward pass
    print("\n[3/3] Testing forward pass...")
    device = torch.device("cpu")
    video_enc.encoder.to(device)
    audio_enc.model.to(device)

    dummy_frames = torch.randn(2, cfg.video_num_frames, 3, 224, 224)
    dummy_audio = torch.randn(2, int(cfg.audio_duration * cfg.audio_sample_rate))

    with torch.no_grad():
        v_feat = video_enc(dummy_frames)
        print(f"  Video output: {v_feat.shape}")
        a_feat = audio_enc(dummy_audio)
        print(f"  Audio output: {a_feat.shape}")

    print("\nAll tests passed!")
    print("Ready for training: python spwm/v2_fall_detect.py --mode all --epochs 30")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V-JEPA 2 + WavJEPA Fall Detection on Le2i")
    parser.add_argument("--mode", default="all", choices=["train", "eval", "all", "test"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder_device", default="cpu")
    parser.add_argument("--checkpoint", default="checkpoints/v2_fall_detect.pt")
    parser.add_argument("--data_root", default="Le2i")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context_sec", type=float, default=0.5)
    parser.add_argument("--target_gap_sec", type=float, default=0.3)
    args = parser.parse_args()

    cfg.update(args)

    if args.mode == "test":
        quick_test()
    elif args.mode == "train":
        train()
    elif args.mode == "eval":
        evaluate(args.checkpoint)
    else:
        train()
        evaluate("checkpoints/v2_fall_detect.pt")


if __name__ == "__main__":
    main()
