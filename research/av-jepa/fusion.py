"""
AV-JEPA Fusion Module
---------------------
Project concatenated AV embeddings into a compact joint space,
then apply SIGReg regularization to prevent representation collapse.

Inspired by LeWorldModel's SIGReg and Fall-Mamba's cross-attention fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AVProjector(nn.Module):
    """
    Project concatenated [video_embed | audio_embed] into a compact joint space.

    This is equivalent to the "fusion" step in PE-AV's AlignModalities module.
    """

    def __init__(
        self,
        video_dim: int = 512,
        audio_dim: int = 512,
        joint_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()
        input_dim = video_dim + audio_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, joint_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, video_dim + audio_dim) concatenated AV embeddings

        Returns:
            z: (B, joint_dim) compact joint representation
        """
        return self.proj(x)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion: video attends to audio, audio attends to video.
    More expressive than simple concatenation, inspired by Fall-Mamba.

    Args:
        dim: embedding dimension for each modality
        num_heads: attention heads
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, joint_dim: int = 256):
        super().__init__()
        self.cross_attn_v2a = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=0.1
        )
        self.cross_attn_a2v = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=0.1
        )
        self.norm_v = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim * 2, joint_dim)

    def forward(
        self, v_embed: torch.Tensor, a_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            v_embed: (B, dim) video embeddings
            a_embed: (B, dim) audio embeddings

        Returns:
            joint: (B, joint_dim) fused embeddings
        """
        # Add sequence dim: (B, 1, dim)
        v = v_embed.unsqueeze(1)
        a = a_embed.unsqueeze(1)

        # Video attends to audio
        v_enhanced, _ = self.cross_attn_v2a(v, a, a)
        v_out = self.norm_v(v + v_enhanced)

        # Audio attends to video
        a_enhanced, _ = self.cross_attn_a2v(a, v, v)
        a_out = self.norm_a(a + a_enhanced)

        # Concatenate and project
        joint = torch.cat([v_out.squeeze(1), a_out.squeeze(1)], dim=-1)
        return self.output_proj(joint)


class SIGRegLoss(nn.Module):
    """
    SIGReg: enforce isotropic Gaussian prior on latent representations.

    From LeWorldModel:
    "SIGReg regularizes the latent space toward an isotropic Gaussian
     distribution to prevent representation collapse."

    Loss = mean( (std(z) - target_std)^2 ) where z is the joint embedding.
    """

    def __init__(self, target_std: float = 1.0):
        super().__init__()
        self.target_std = target_std

    def forward(self, z: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss for both context and target embeddings.

        Args:
            z: (B, D) context embeddings
            z_target: (B, D) target embeddings

        Returns:
            loss: scalar SIGReg loss
        """
        B = z.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z.device)

        # Standard deviation across batch dimension
        std_z = torch.std(z, dim=0).mean()
        std_zt = torch.std(z_target, dim=0).mean()

        loss_z = (std_z - self.target_std).pow(2)
        loss_zt = (std_zt - self.target_std).pow(2)

        return (loss_z + loss_zt) * 0.5


def create_fusion(
    video_dim: int,
    audio_dim: int,
    joint_dim: int,
    method: str = "concat",
) -> nn.Module:
    """
    Factory: create fusion module.

    Args:
        method: "concat" or "cross_attn"
    """
    if method == "cross_attn":
        assert audio_dim == video_dim, "Cross-attn requires equal dims"
        return CrossAttentionFusion(dim=video_dim, joint_dim=joint_dim)
    else:
        return AVProjector(
            video_dim=video_dim,
            audio_dim=audio_dim,
            joint_dim=joint_dim,
        )
