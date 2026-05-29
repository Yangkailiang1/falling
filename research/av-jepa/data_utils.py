"""
AV-JEPA Data Utilities
-----------------------
Synthetic data generation + video/audio loading utilities.

For initial testing, we generate synthetic "normal" and "fall" data.
When real datasets are available, replace this with actual data loading.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import random


def generate_synthetic_embeddings(
    batch_size: int = 32,
    embed_dim: int = 256,
    num_modes: int = 5,
    noise_std: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic "normal" AV embeddings.
    Simulates different activity modes (walking, sitting, standing, etc.).

    Each mode is a random Gaussian cluster. Sequences are smooth transitions
    within a mode, with occasional switches.

    Args:
        batch_size: number of samples
        embed_dim: embedding dimension
        num_modes: number of activity modes
        noise_std: within-mode variation

    Returns:
        ctx_embed: (B, embed_dim) "current" embedding
        tgt_embed: (B, embed_dim) "future" embedding (nearby in same mode)
    """
    # Generate mode centers
    mode_centers = torch.randn(num_modes, embed_dim) * 2.0

    # Assign each sample to a mode
    modes = torch.randint(0, num_modes, (batch_size,))

    # Generate context embeddings (mode center + noise)
    ctx_embed = mode_centers[modes] + torch.randn(batch_size, embed_dim) * noise_std

    # Target is nearby in embedding space (within-mode variation)
    tgt_embed = mode_centers[modes] + torch.randn(batch_size, embed_dim) * noise_std

    return ctx_embed, tgt_embed


def generate_synthetic_fall_embeddings(
    batch_size: int = 32,
    embed_dim: int = 256,
    normal_modes: int = 5,
    fall_displacement: float = 5.0,
    noise_std: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic "fall" AV embeddings.
    A fall is a sudden large displacement in embedding space:
    the target is very far from the context.

    Args:
        batch_size: number of samples
        embed_dim: embedding dimension
        normal_modes: number of normal activity modes
        fall_displacement: how far the fall shifts the embedding
        noise_std: within-mode variation

    Returns:
        ctx_embed: (B, embed_dim) "pre-fall" embedding (normal)
        tgt_embed: (B, embed_dim) "fall" embedding (far away)
    """
    # Pre-fall: normal mode
    mode_centers = torch.randn(normal_modes, embed_dim) * 2.0
    modes = torch.randint(0, normal_modes, (batch_size,))
    ctx_embed = mode_centers[modes] + torch.randn(batch_size, embed_dim) * noise_std

    # Fall: large random displacement
    fall_direction = torch.randn(batch_size, embed_dim)
    fall_direction = F.normalize(fall_direction, dim=-1)
    tgt_embed = ctx_embed + fall_direction * fall_displacement

    # Add some noise to the target
    tgt_embed = tgt_embed + torch.randn(batch_size, embed_dim) * noise_std

    return ctx_embed, tgt_embed


class SyntheticAVDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset for AV-JEPA testing.

    Generates embeddings directly (bypassing video/audio loading)
    for fast iteration during development.

    Normal samples: smooth within-mode transitions
    Fall samples: sudden large embedding displacement
    """

    def __init__(
        self,
        num_samples: int = 1000,
        embed_dim: int = 256,
        normal_ratio: float = 0.8,
        num_modes: int = 5,
        noise_std: float = 0.05,
        fall_displacement: float = 5.0,
    ):
        self.num_samples = num_samples
        self.embed_dim = embed_dim
        self.normal_ratio = normal_ratio
        self.num_modes = num_modes

        # Determine split
        self.num_normal = int(num_samples * normal_ratio)
        self.num_fall = num_samples - self.num_normal

        # Pre-generate all data
        self.ctx_embeds = []
        self.tgt_embeds = []
        self.labels = []  # 0 = normal, 1 = fall

        if self.num_normal > 0:
            ctx, tgt = generate_synthetic_embeddings(
                self.num_normal, embed_dim, num_modes, noise_std
            )
            self.ctx_embeds.append(ctx)
            self.tgt_embeds.append(tgt)
            self.labels.extend([0] * self.num_normal)

        if self.num_fall > 0:
            ctx, tgt = generate_synthetic_fall_embeddings(
                self.num_fall, embed_dim, num_modes, fall_displacement, noise_std
            )
            self.ctx_embeds.append(ctx)
            self.tgt_embeds.append(tgt)
            self.labels.extend([1] * self.num_fall)

        self.ctx_embeds = torch.cat(self.ctx_embeds, dim=0)
        self.tgt_embeds = torch.cat(self.tgt_embeds, dim=0)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.ctx_embeds[idx], self.tgt_embeds[idx], self.labels[idx]


def generate_synthetic_video_audio(
    batch_size: int = 8,
    num_frames: int = 8,
    frame_size: int = 224,
    audio_len: int = 144000,  # 3s @ 48kHz
    mode: str = "normal",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic video frames and audio for end-to-end testing.

    This is placeholder data — replace with real video/audio loading
    when datasets are available.

    Args:
        batch_size: number of clips
        num_frames: frames per clip
        frame_size: spatial size
        audio_len: audio samples
        mode: "normal" or "fall"

    Returns:
        frames: (B, T, C, H, W)
        audio: (B, audio_len)
    """
    if mode == "normal":
        # Normal: smooth video + ambient audio
        base_frame = torch.rand(batch_size, 1, 3, frame_size, frame_size)
        # Each subsequent frame is base + small noise
        noise = torch.randn(batch_size, num_frames - 1, 3, frame_size, frame_size) * 0.02
        frames = torch.cat([base_frame, base_frame.expand(-1, num_frames - 1, -1, -1, -1) + noise], dim=1)

        # Audio: low-frequency hum
        t = torch.linspace(0, 3, audio_len)
        audio = 0.1 * torch.sin(2 * np.pi * 100 * t).unsqueeze(0).expand(batch_size, -1)
        audio = audio + torch.randn(batch_size, audio_len) * 0.01

    else:
        # Fall: sudden frame change + loud impact
        pre_frames = torch.rand(batch_size, num_frames // 2, 3, frame_size, frame_size) * 0.5
        # Sudden visual change (simulating fall)
        post_frames = torch.rand(batch_size, num_frames - num_frames // 2, 3, frame_size, frame_size) * 1.0 + 0.5
        frames = torch.cat([pre_frames, post_frames], dim=1)

        # Audio: quiet then sudden impact
        first_half = audio_len // 2
        audio = torch.zeros(batch_size, audio_len)
        audio[:, :first_half] = torch.randn(batch_size, first_half) * 0.01
        # Impact: loud short burst
        impact_len = audio_len // 10
        audio[:, first_half:first_half + impact_len] = torch.randn(batch_size, impact_len) * 0.5
        audio[:, first_half + impact_len:] = torch.randn(batch_size, audio_len - first_half - impact_len) * 0.05

    return frames, audio
