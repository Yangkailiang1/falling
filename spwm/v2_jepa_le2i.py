#!/usr/bin/env python3
"""
V-JEPA 2 + Audio JEPA Style 2-Modality Fall Detection on Le2i

Architecture:
  Context frames (8) → [V-JEPA 2 ViT-L, frozen] → v_ctx (1024) ↘
                                                          → concat (1536) → [Projector] → z_ctx (256)
  Context audio       → [CLAP HTSAT, frozen]    → a_ctx (512)  ↗             ↓
                                                                   [JEPA Predictor] (3-layer Transformer)
                                                                           ↓
  Target frames (8)  → [V-JEPA 2 ViT-L, frozen] → v_tgt (1024) ↘         z_pred (256)
                                                          → concat → [EMA Projector] → z_target (256)
  Target audio       → [CLAP HTSAT, frozen]    → a_tgt (512)  ↗

Loss = MSE(z_pred, z_target) + λ·SIGReg(z, z_target)

Train on normal activity only. Detect falls by prediction error (surprise).

Usage:
  # Train on Le2i (CPU/MPS, small scale):
  python spwm/v2_jepa_le2i.py --mode train --epochs 30 --batch_size 2 --device cpu

  # Evaluate:
  python spwm/v2_jepa_le2i.py --mode eval --checkpoint checkpoints/v2_jepa_le2i.pt

  # Train + eval:
  python spwm/v2_jepa_le2i.py --mode all --epochs 30 --batch_size 2 --device cpu
"""

import os
import sys
import copy
import math
import argparse
import warnings
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add research/av-jepa to path for dataset loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research', 'av-jepa'))

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

class Config:
    """2-modality JEPA configuration."""
    # Video: V-JEPA 2 ViT-L
    vjepa2_model: str = "vjepa2_vit_large"
    video_embed_dim: int = 1024   # V-JEPA 2 ViT-L embed_dim
    video_num_frames: int = 8
    video_frame_size: int = 256

    # Audio: Audio-JEPA ViT (ltuncay/Audio-JEPA)
    audio_jepa_ckpt: str = "ltuncay/Audio-JEPA"  # HuggingFace repo
    audio_embed_dim: int = 768    # Audio-JEPA ViT dim
    audio_sample_rate: int = 16000
    audio_duration: float = 2.0
    mel_bins: int = 256  # mel spectrogram bins (256/16 * 128/16 = 16*8 = 128 patches)
    mel_time: int = 128  # time frames

    # Joint
    joint_dim: int = 256

    # Predictor
    predictor_num_layers: int = 3
    predictor_num_heads: int = 8
    predictor_hidden_dim: int = 512
    predictor_dropout: float = 0.1

    # SIGReg
    sigreg_weight: float = 0.3
    sigreg_target_std: float = 1.0

    # EMA
    ema_decay: float = 0.996
    ema_end_decay: float = 1.0

    # Training
    batch_size: int = 2
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # Data
    data_root: str = "datasets/Le2i"
    num_workers: int = 0

    # Device
    device: str = "cpu"

    def __post_init__(self, args):
        if args.device:
            self.device = args.device
        if args.batch_size:
            self.batch_size = args.batch_size
        if args.epochs:
            self.num_epochs = args.epochs
        if args.data_root:
            self.data_root = args.data_root


# ═══════════════════════════════════════════════════════════════
# V-JEPA 2 Video Encoder
# ═══════════════════════════════════════════════════════════════

class VJEPA2Encoder(nn.Module):
    """V-JEPA 2 ViT-L video encoder (frozen)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self._encoder = None
        self._initialized = False

    def _lazy_init(self, device):
        if self._initialized:
            return
        self._initialized = True

        print("[VJEPA2] Loading V-JEPA 2 ViT-L from torch.hub...")
        encoder, predictor = torch.hub.load(
            'facebookresearch/vjepa2',
            self.cfg.vjepa2_model,
            pretrained=True
        )
        self._encoder = encoder.to(device).eval()
        for p in self._encoder.parameters():
            p.requires_grad = False
        print(f"[VJEPA2] Loaded. Params: {sum(p.numel() for p in self._encoder.parameters())/1e6:.0f}M")

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) = (B, 8, 3, 224, 224)
        Returns:
            video_embed: (B, 1024)
        """
        self._lazy_init(frames.device)
        B, T, C, H, W = frames.shape

        # V-JEPA 2 expects (B, C, T, H, W) format
        frames_c_t = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        # V-JEPA 2 forward: outputs (B, num_patches, embed_dim)
        features = self._encoder(frames_c_t)  # encoder.__call__ invokes forward_features

        # Pool to single vector: mean over patch dimension
        video_embed = features.mean(dim=1)  # (B, 1024)
        return video_embed


# ═══════════════════════════════════════════════════════════════
# Audio-JEPA ViT Encoder
# ═══════════════════════════════════════════════════════════════

class AudioJEPAEncoder(nn.Module):
    """Audio-JEPA ViT encoder (frozen, ltuncay/Audio-JEPA checkpoint)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self._encoder = None
        self._mel_extractor = None
        self._initialized = False

    def _lazy_init(self, device):
        if self._initialized:
            return
        self._initialized = True

        print("[AudioJEPA] Loading Audio-JEPA ViT...")
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(cfg.audio_jepa_ckpt, 'JEPA.ckpt')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        from .v2_audio_jepa_vit import build_audio_jepa_vit
        self._encoder = build_audio_jepa_vit(ckpt['state_dict']).to(device).eval()
        for p in self._encoder.parameters():
            p.requires_grad = False
        print(f"[AudioJEPA] Loaded. Params: {sum(p.numel() for p in self._encoder.parameters())/1e6:.0f}M")

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        self._lazy_init(audio.device)
        mel = self._compute_mel(audio)  # (B, 1, 256, 128)
        features = self._encoder(mel)   # (B, 128, 768)
        return features.mean(dim=1)     # (B, 768)

    def _compute_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert raw waveform to mel spectrogram (B, 1, 256, 128)."""
        B = waveform.shape[0]
        try:
            import torchaudio
            mels = []
            for i in range(B):
                mel = torchaudio.compliance.kaldi.fbank(
                    waveform[i:i+1] * 32767,
                    htk_compat=True,
                    sample_frequency=cfg.audio_sample_rate,
                    use_energy=False, window_type='hanning',
                    num_mel_bins=cfg.mel_bins, dither=0.0, frame_shift=10,
                )
                T = mel.shape[0]
                if T < cfg.mel_time:
                    pad = torch.zeros(cfg.mel_time - T, cfg.mel_bins, device=mel.device)
                    mel = torch.cat([mel, pad], dim=0)
                elif T > cfg.mel_time:
                    mel = mel[:cfg.mel_time]
                mel = mel.T.unsqueeze(0)  # (1, mel_bins, T)
                mels.append(mel)
            return torch.cat(mels, dim=0)  # (B, 1, mel_bins, T)
        except ImportError:
            spec = torch.stft(waveform.reshape(-1, waveform.shape[-1]),
                n_fft=400, hop_length=160, return_complex=True).abs()
            return F.interpolate(spec.unsqueeze(1),
                size=(cfg.mel_bins, cfg.mel_time), mode='bilinear', align_corners=False)


# ═══════════════════════════════════════════════════════════════
# Projector: concat(video, audio) → joint space
# ═══════════════════════════════════════════════════════════════

class JointProjector(nn.Module):
    """MLP that projects concatenated [video | audio] into joint space."""

    def __init__(self, video_dim: int, audio_dim: int, joint_dim: int):
        super().__init__()
        input_dim = video_dim + audio_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, joint_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, video_dim + audio_dim) concatenated embeddings
        Returns:
            z: (B, joint_dim)
        """
        return self.proj(x)


# ═══════════════════════════════════════════════════════════════
# JEPA Predictor (Transformer)
# ═══════════════════════════════════════════════════════════════

class JEPAPredictor(nn.Module):
    """3-layer Transformer predictor: z_ctx → z_pred."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 3,
                 hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.context_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        B = z_ctx.shape[0]
        x = z_ctx.unsqueeze(1)  # (B, 1, D)
        ctx_token = self.context_token.expand(B, -1, -1)
        x = torch.cat([ctx_token, x], dim=1)  # (B, 2, D)
        x = self.transformer(x)  # (B, 2, D)
        return self.output_proj(x[:, 0, :])  # (B, D)


# ═══════════════════════════════════════════════════════════════
# SIGReg Regularization
# ═══════════════════════════════════════════════════════════════

class SIGRegLoss(nn.Module):
    """Covariance regularization: push latent space toward isotropic Gaussian."""

    def __init__(self, target_std: float = 1.0):
        super().__init__()
        self.target_std = target_std

    def forward(self, z: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z.device)
        std_z = torch.std(z, dim=0).mean()
        std_zt = torch.std(z_target, dim=0).mean()
        return (std_z - self.target_std).pow(2) * 0.5 + (std_zt - self.target_std).pow(2) * 0.5


# ═══════════════════════════════════════════════════════════════
# Full V2JEPA Model
# ═══════════════════════════════════════════════════════════════

class V2JEPA(nn.Module):
    """V-JEPA 2 + Audio-JEPA model for fall detection."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Frozen encoders
        self.video_encoder = VJEPA2Encoder(cfg)
        self.audio_encoder = AudioJEPAEncoder(cfg)

        # Trainable projector
        self.projector = JointProjector(cfg.video_embed_dim, cfg.audio_embed_dim, cfg.joint_dim)

        # EMA target projector
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False

        # Predictor
        self.predictor = JEPAPredictor(
            embed_dim=cfg.joint_dim,
            num_heads=cfg.predictor_num_heads,
            num_layers=cfg.predictor_num_layers,
            hidden_dim=cfg.predictor_hidden_dim,
            dropout=cfg.predictor_dropout,
        )

        # SIGReg
        self.sigreg = SIGRegLoss(target_std=cfg.sigreg_target_std)

    def _encode(self, frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Encode video+audio into joint space using (frozen encoder + projector)."""
        v = self.video_encoder(frames)  # (B, 1024)
        a = self.audio_encoder(audio)   # (B, 512)
        joint = torch.cat([v, a], dim=-1)  # (B, 1536)
        return joint

    def forward_context(self, frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        joint = self._encode(frames, audio)
        return self.projector(joint)

    @torch.no_grad()
    def forward_target(self, frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        joint = self._encode(frames, audio)
        return self.target_projector(joint)

    def forward(self, ctx_frames, ctx_audio, tgt_frames, tgt_audio):
        z_ctx = self.forward_context(ctx_frames, ctx_audio)
        z_pred = self.predictor(z_ctx)
        z_target = self.forward_target(tgt_frames, tgt_audio)
        return z_pred, z_target, z_ctx

    def compute_loss(self, z_pred, z_target, z_ctx):
        pred_loss = F.mse_loss(z_pred, z_target)
        sigreg_loss = self.sigreg(z_ctx, z_target)
        total = pred_loss + self.cfg.sigreg_weight * sigreg_loss
        return total, {"pred": pred_loss.item(), "sigreg": sigreg_loss.item()}

    @torch.no_grad()
    def compute_surprise(self, ctx_frames, ctx_audio, tgt_frames, tgt_audio):
        z_pred, z_target, _ = self.forward(ctx_frames, ctx_audio, tgt_frames, tgt_audio)
        return F.mse_loss(z_pred, z_target, reduction="none").mean(dim=-1)  # (B,)

    @torch.no_grad()
    def update_target_encoder(self, step: int, total_steps: int):
        tau = self.cfg.ema_end_decay - \
              (self.cfg.ema_end_decay - self.cfg.ema_decay) * \
              (math.cos(math.pi * step / total_steps) + 1) / 2
        for p_tgt, p_src in zip(self.target_projector.parameters(), self.projector.parameters()):
            p_tgt.data.copy_(tau * p_tgt.data + (1.0 - tau) * p_src.data)


# ═══════════════════════════════════════════════════════════════
# Le2i Dataset (adapted from research/av-jepa)
# ═══════════════════════════════════════════════════════════════

def parse_le2i_annotation(filepath: str):
    """Parse Le2i annotation file. Returns (fall_start, fall_end, frames_list)."""
    with open(filepath) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    fall_start = 0
    fall_end = 0
    data_start = 0

    if len(lines) >= 2:
        try:
            v1, v2 = int(lines[0]), int(lines[1])
            if "," not in lines[0] and "," not in lines[1]:
                fall_start, fall_end = v1, v2
                data_start = 2
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


class Le2iAVDataset(torch.utils.data.Dataset):
    """
    Le2i dataset for 2-modality JEPA.
    Each sample: (ctx_frames, ctx_audio, tgt_frames, tgt_audio, has_fall).
    """

    def __init__(self, data_root: str, num_frames: int = 8, frame_size: int = 224,
                 audio_duration: float = 2.0, audio_sr: int = 48000, normal_only: bool = True,
                 context_duration: float = 1.0, target_gap: float = 0.5):
        super().__init__()
        self.data_root = data_root
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.audio_duration = audio_duration
        self.audio_sr = audio_sr
        self.normal_only = normal_only
        self.context_frames = num_frames
        self.target_frames = num_frames
        self.context_duration = context_duration
        self.target_gap = target_gap

        self.samples = []  # (video_path, ctx_start, ctx_end, tgt_start, tgt_end, has_fall)
        self._scan_dataset()
        print(f"[Le2i] {'Normal-only' if normal_only else 'Mixed'} dataset: {len(self.samples)} clips")

    def _scan_dataset(self):
        import glob, re

        scenes = sorted([d for d in os.listdir(self.data_root)
                        if os.path.isdir(os.path.join(self.data_root, d))])

        for scene in scenes:
            scene_inner = os.path.join(self.data_root, scene, scene)
            video_dir = os.path.join(scene_inner, "Videos")
            annot_dir = os.path.join(scene_inner, "Annotation_files")
            if not os.path.exists(annot_dir):
                annot_dir = os.path.join(scene_inner, "Annotations_files")

            if not os.path.exists(video_dir) or not os.path.exists(annot_dir):
                continue

            for video_name in sorted(os.listdir(video_dir)):
                if not video_name.lower().endswith('.avi'):
                    continue

                video_path = os.path.join(video_dir, video_name)
                video_num = re.search(r"\((\d+)\)", video_name)
                if not video_num:
                    continue

                annot_path = os.path.join(annot_dir, f"video ({video_num.group(1)}).txt")
                if not os.path.exists(annot_path):
                    continue

                # Get fall info
                fall_start, fall_end, _ = parse_le2i_annotation(annot_path)
                has_fall_event = fall_start > 0 and fall_end > fall_start

                # Count frames
                try:
                    import av
                    c = av.open(video_path)
                    n_frames = c.streams.video[0].frames or 300
                    fps = float(c.streams.video[0].average_rate) if c.streams.video[0].average_rate else 25.0
                    c.close()
                except Exception:
                    n_frames = 300
                    fps = 25.0

                # Generate clips
                ctx_frames_clip = int(self.context_duration * fps)
                gap_frames = int(self.target_gap * fps)
                stride = max(1, ctx_frames_clip // 2)
                clip_start = 0

                while clip_start + ctx_frames_clip + gap_frames + ctx_frames_clip < n_frames:
                    ctx_s = clip_start
                    ctx_e = clip_start + ctx_frames_clip
                    tgt_s = ctx_e + gap_frames
                    tgt_e = tgt_s + ctx_frames_clip

                    # Check overlap with fall
                    clip_has_fall = False
                    if has_fall_event:
                        clip_has_fall = (
                            (ctx_s < fall_end and ctx_e > fall_start) or
                            (tgt_s < fall_end and tgt_e > fall_start)
                        )

                    if self.normal_only and clip_has_fall:
                        clip_start += stride
                        continue

                    self.samples.append((video_path, ctx_s, ctx_e, tgt_s, tgt_e, clip_has_fall, fps))
                    clip_start += stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, ctx_s, ctx_e, tgt_s, tgt_e, has_fall, fps = self.samples[idx]

        try:
            # Load video + audio
            import av
            container = av.open(video_path)
            video_stream = container.streams.video[0]

            # Decode frames in range
            all_frames = []
            frame_idx = 0
            for frame in container.decode(video_stream):
                if frame_idx > max(ctx_e, tgt_e):
                    break
                if frame_idx >= min(ctx_s, tgt_s):
                    img = frame.to_ndarray(format="rgb24")
                    img = torch.from_numpy(img).float() / 255.0
                    img = img.permute(2, 0, 1)  # (C, H, W)
                    # Resize
                    if img.shape[1] != self.frame_size:
                        img = F.interpolate(
                            img.unsqueeze(0), size=(self.frame_size, self.frame_size),
                            mode='bilinear', align_corners=False
                        ).squeeze(0)
                    all_frames.append(img)
                frame_idx += 1

            # Audio
            all_audio = None
            try:
                audio_stream = container.streams.audio[0]
                audio_sr_actual = audio_stream.sample_rate
                audio_chunks = []
                for frame in container.decode(audio_stream):
                    try:
                        samples = frame.to_ndarray()
                        audio_chunks.append(samples.flatten())
                    except Exception:
                        continue
                if audio_chunks:
                    all_audio = torch.from_numpy(np.concatenate(audio_chunks)).float()
            except Exception:
                audio_sr_actual = self.audio_sr

            container.close()

            # Extract context/target frames
            total_decoded = len(all_frames)
            ctx_offset = ctx_s - min(ctx_s, tgt_s)
            tgt_offset = tgt_s - min(ctx_s, tgt_s)

            def sample_frames(frames_list, offset, count, target_n):
                seg = frames_list[offset:offset + count]
                if len(seg) == 0:
                    return torch.zeros(target_n, 3, self.frame_size, self.frame_size)
                T = len(seg)
                if T <= target_n:
                    idxs = torch.arange(target_n).clamp(0, T - 1)
                else:
                    idxs = torch.linspace(0, T - 1, target_n).long()
                return torch.stack([seg[i] for i in idxs])

            ctx_count = ctx_e - ctx_s
            tgt_count = tgt_e - tgt_s
            ctx_frames = sample_frames(all_frames, ctx_offset, ctx_count, self.num_frames)
            tgt_frames = sample_frames(all_frames, tgt_offset, tgt_count, self.num_frames)

            # Extract context/target audio
            target_audio_samples = int(self.audio_duration * self.audio_sr)
            if all_audio is not None:
                # Resample if needed
                if audio_sr_actual != self.audio_sr:
                    samples_in = all_audio.shape[0]
                    target_len = int(samples_in * self.audio_sr / audio_sr_actual)
                    all_audio = F.interpolate(
                        all_audio.unsqueeze(0).unsqueeze(0),
                        size=target_len, mode='linear', align_corners=False
                    ).squeeze()

                audio_ctx_start = int(ctx_s / fps * self.audio_sr)
                audio_ctx_end = audio_ctx_start + target_audio_samples
                audio_tgt_start = int(tgt_s / fps * self.audio_sr)
                audio_tgt_end = audio_tgt_start + target_audio_samples

                def extract_audio_segment(audio, start, end):
                    if start >= audio.shape[0]:
                        return torch.zeros(target_audio_samples)
                    end = min(end, audio.shape[0])
                    seg = audio[start:end]
                    if len(seg) < target_audio_samples:
                        seg = F.pad(seg, (0, target_audio_samples - len(seg)))
                    return seg[:target_audio_samples]

                ctx_audio = extract_audio_segment(all_audio, audio_ctx_start, audio_ctx_end)
                tgt_audio = extract_audio_segment(all_audio, audio_tgt_start, audio_tgt_end)
            else:
                ctx_audio = torch.zeros(target_audio_samples)
                tgt_audio = torch.zeros(target_audio_samples)

            return ctx_frames, ctx_audio, tgt_frames, tgt_audio, torch.tensor(1.0 if has_fall else 0.0)

        except Exception as e:
            # Return dummy on error
            dummy_frames = torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size)
            dummy_audio = torch.zeros(int(self.audio_duration * self.audio_sr))
            return dummy_frames, dummy_audio, dummy_frames, dummy_audio, torch.tensor(-1.0)


def collate_fn(batch):
    """Filter out failed loads (label == -1)."""
    valid = [(f1, a1, f2, a2, l) for f1, a1, f2, a2, l in batch if l.item() >= 0]
    if len(valid) == 0:
        return None
    ctx_f = torch.stack([x[0] for x in valid])
    ctx_a = torch.stack([x[1] for x in valid])
    tgt_f = torch.stack([x[2] for x in valid])
    tgt_a = torch.stack([x[3] for x in valid])
    labels = torch.stack([x[4] for x in valid])
    return ctx_f, ctx_a, tgt_f, tgt_a, labels


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train(cfg: Config):
    print("=" * 60)
    print("Training V2-JEPA on Le2i (Normal Only)")
    print("=" * 60)

    # Dataset
    train_dataset = Le2iAVDataset(
        data_root=cfg.data_root,
        num_frames=cfg.video_num_frames,
        frame_size=cfg.video_frame_size,
        audio_duration=cfg.audio_duration,
        audio_sr=cfg.audio_sample_rate,
        normal_only=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn, drop_last=True,
    )

    # Eval dataset (for validation during training)
    eval_dataset = Le2iAVDataset(
        data_root=cfg.data_root,
        num_frames=cfg.video_num_frames,
        frame_size=cfg.video_frame_size,
        audio_duration=cfg.audio_duration,
        audio_sr=cfg.audio_sample_rate,
        normal_only=False,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
    )

    # Model
    device = torch.device(cfg.device)
    model = V2JEPA(cfg).to(device)

    # Only train projector + predictor
    trainable_params = list(model.projector.parameters()) + list(model.predictor.parameters())
    n_params = sum(p.numel() for p in trainable_params)
    print(f"\n[Model] Trainable params: {n_params/1e3:.1f}K")

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs * len(train_loader))

    total_steps = cfg.num_epochs * len(train_loader)
    step = 0
    best_surprise_ratio = 0.0

    print(f"\n[Training] {cfg.num_epochs} epochs, {len(train_loader)} steps/epoch")
    print(f"[Device] {cfg.device}")
    print()

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_pred = 0.0
        epoch_sigreg = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            ctx_f, ctx_a, tgt_f, tgt_a, _ = [x.to(device) for x in batch]

            optimizer.zero_grad()
            z_pred, z_target, z_ctx = model(ctx_f, ctx_a, tgt_f, tgt_a)
            loss, loss_dict = model.compute_loss(z_pred, z_target, z_ctx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            model.update_target_encoder(step, total_steps)
            step += 1

            epoch_loss += loss.item()
            epoch_pred += loss_dict["pred"]
            epoch_sigreg += loss_dict["sigreg"]

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{cfg.num_epochs} | Step {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} (pred: {loss_dict['pred']:.4f}, sigreg: {loss_dict['sigreg']:.4f})")

        # Epoch summary
        n_batches = max(len(train_loader), 1)
        print(f"--- Epoch {epoch+1} Avg Loss: {epoch_loss/n_batches:.4f} "
              f"(pred: {epoch_pred/n_batches:.4f}, sigreg: {epoch_sigreg/n_batches:.4f})")

        # Quick eval every 5 epochs
        if (epoch + 1) % 5 == 0:
            ratio = evaluate_surprise(model, eval_loader, device)
            print(f"--- Eval Surprise Ratio: {ratio:.2f}x (higher = better separation)")

            if ratio > best_surprise_ratio:
                best_surprise_ratio = ratio
                save_checkpoint(model, optimizer, epoch, "checkpoints/v2_jepa_le2i_best.pt")
                print(f"--- Saved best checkpoint (surprise ratio: {ratio:.2f}x)")

    # Final save
    os.makedirs("checkpoints", exist_ok=True)
    save_checkpoint(model, optimizer, cfg.num_epochs - 1, "checkpoints/v2_jepa_le2i.pt")
    print(f"\nTraining complete. Best surprise ratio: {best_surprise_ratio:.2f}x")


def evaluate_surprise(model, eval_loader, device):
    """Quick evaluation: compute surprise ratio (fall / normal)."""
    model.eval()
    normal_errors = []
    fall_errors = []

    for batch in eval_loader:
        if batch is None:
            continue
        ctx_f, ctx_a, tgt_f, tgt_a, labels = [x.to(device) for x in batch]

        surprise = model.compute_surprise(ctx_f, ctx_a, tgt_f, tgt_a)

        for i in range(len(labels)):
            if labels[i] > 0.5:
                fall_errors.append(surprise[i].item())
            else:
                normal_errors.append(surprise[i].item())

    if not normal_errors or not fall_errors:
        return 0.0

    ratio = np.mean(fall_errors) / (np.mean(normal_errors) + 1e-8)
    return ratio


# ═══════════════════════════════════════════════════════════════
# Full Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate(cfg: Config, checkpoint_path: str):
    print("=" * 60)
    print("Evaluating V2-JEPA on Le2i")
    print("=" * 60)

    device = torch.device(cfg.device)
    model = V2JEPA(cfg).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Trained for {ckpt['epoch']+1} epochs")

    # Evaluation dataset (mixed)
    eval_dataset = Le2iAVDataset(
        data_root=cfg.data_root,
        num_frames=cfg.video_num_frames,
        frame_size=cfg.video_frame_size,
        audio_duration=cfg.audio_duration,
        audio_sr=cfg.audio_sample_rate,
        normal_only=False,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
    )

    # Collect predictions
    all_surprises = []
    all_labels = []

    print(f"\nComputing surprise scores on {len(eval_dataset)} clips...")
    with torch.no_grad():
        for batch in eval_loader:
            if batch is None:
                continue
            ctx_f, ctx_a, tgt_f, tgt_a, labels = [x.to(device) for x in batch]
            surprise = model.compute_surprise(ctx_f, ctx_a, tgt_f, tgt_a)
            all_surprises.extend(surprise.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    all_surprises = np.array(all_surprises)
    all_labels = np.array(all_labels)

    n_fall = (all_labels > 0.5).sum()
    n_normal = (all_labels <= 0.5).sum()

    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"  Total clips:    {len(all_labels)}")
    print(f"  Fall clips:     {n_fall}")
    print(f"  Normal clips:   {n_normal}")
    print()

    # Per-class statistics
    fall_errors = all_surprises[all_labels > 0.5]
    normal_errors = all_surprises[all_labels <= 0.5]

    if len(fall_errors) > 0 and len(normal_errors) > 0:
        print(f"  Normal mean±std:  {np.mean(normal_errors):.6f} ± {np.std(normal_errors):.6f}")
        print(f"  Fall mean±std:    {np.mean(fall_errors):.6f} ± {np.std(fall_errors):.6f}")
        ratio = np.mean(fall_errors) / np.mean(normal_errors)
        print(f"  Surprise ratio:   {ratio:.2f}x (fall/normal)")
        print()

        # Compute AUROC
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auroc = roc_auc_score(all_labels, all_surprises)
            auprc = average_precision_score(all_labels, all_surprises)
            print(f"  AUROC:           {auroc:.4f}")
            print(f"  AUPRC:           {auprc:.4f}")
        except ImportError:
            auroc = None
            print("  (sklearn not available for AUROC)")

        # Best threshold by F1
        best_f1 = 0
        best_thresh = 0
        best_prec = 0
        best_rec = 0
        for pct in range(50, 100):
            thresh = np.percentile(all_surprises, pct)
            preds = (all_surprises > thresh).astype(float)
            tp = ((preds == 1) & (all_labels == 1)).sum()
            fp = ((preds == 1) & (all_labels == 0)).sum()
            fn = ((preds == 0) & (all_labels == 1)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1, best_thresh, best_prec, best_rec = f1, pct, prec, rec

        print(f"\n  Best threshold:  {best_thresh}th percentile")
        print(f"  Precision:      {best_prec:.4f}")
        print(f"  Recall:         {best_rec:.4f}")
        print(f"  F1-score:       {best_f1:.4f}")

        # Sigma separation
        mean_normal = np.mean(normal_errors)
        std_normal = np.std(normal_errors)
        mean_fall = np.mean(fall_errors)
        if std_normal > 0:
            sigma = (mean_fall - mean_normal) / std_normal
            print(f"\n  Sigma separation: {sigma:.2f}σ")
    else:
        print("  Insufficient data for evaluation")


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V2-JEPA Le2i Training & Evaluation")
    parser.add_argument("--mode", default="all", choices=["train", "eval", "all"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data_root", default="datasets/Le2i")
    parser.add_argument("--checkpoint", default="checkpoints/v2_jepa_le2i.pt")
    args = parser.parse_args()

    cfg = Config()
    cfg.__post_init__(args)

    if args.mode in ("train", "all"):
        train(cfg)

    if args.mode in ("eval", "all"):
        evaluate(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
