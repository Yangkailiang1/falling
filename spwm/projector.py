"""
VL-JEPA + TC-JEPA Text-Conditioned Projector

Maps predicted future latent states (z_future) into LLM embedding space (z_text).

VL-JEPA role: bridging JEPA latent space → LLM vocabulary space
TC-JEPA role: conditioning predictions on current text description to reduce
              uncertainty (text provides context constraints)

Architecture:
  z_future (1024) + text_condition (3584) → CrossAttn → MLP → z_text (3584)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import TextProjectorConfig


class TextConditionedProjector(nn.Module):
    """
    TC-JEPA style projector: conditions future state prediction on text context.

    Maps from JEPA latent space to LLM embedding space, with text conditioning
    to provide context awareness. The output z_text can be directly compared to
    actual text embeddings or used with a phrase retriever.

    Training: MSE(z_text, actual_future_text_embedding) + InfoNCE contrastive
    """

    def __init__(self, config: TextProjectorConfig):
        super().__init__()
        self.config = config

        # Project z_future from JEPA dim to output dim
        self.z_proj = nn.Sequential(
            nn.Linear(config.jepa_dim, config.projector_hidden),
            nn.LayerNorm(config.projector_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.projector_hidden, config.output_dim),
        )

        # Project text condition to output dim for cross-attention
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_dim, config.projector_hidden),
            nn.LayerNorm(config.projector_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.projector_hidden, config.output_dim),
        )

        # Cross-attention: z_future (query) attends to text (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.output_dim,
            num_heads=config.cross_attn_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Output refinement
        self.output = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim),
        )

        # Output normalization (cosine similarity expects unit norm)
        self.output_norm = nn.LayerNorm(config.output_dim)

    def forward(self, z_future: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_future: (B, 1024) predicted future latent state from JEPA
            text_embed: (B, 3584) current event text embedding (TC-JEPA condition)

        Returns:
            z_text: (B, 3584) predicted future text embedding in LLM space
        """
        # Project inputs to output dimension
        z = self.z_proj(z_future).unsqueeze(1)    # (B, 1, out_dim)
        t = self.text_proj(text_embed).unsqueeze(1)  # (B, 1, out_dim)

        # Cross-attention: z looks at text for context constraints
        conditioned, attn_weights = self.cross_attn(z, t, t)
        conditioned = conditioned.squeeze(1)  # (B, out_dim)

        # Output refinement
        z_text = self.output(conditioned)
        z_text = self.output_norm(z_text)

        return z_text


class VLJEPAPhraseProjector(nn.Module):
    """
    VL-JEPA projector that maps JEPA embeddings to a shared phrase embedding space.

    This is a dual-encoder design:
      - JEPA encoder: z_future → phrase-like embedding
      - Text encoder: text description → phrase-like embedding

    The two encodings are trained to align in a shared space using:
      1. MSE loss (direct alignment)
      2. InfoNCE contrastive loss (discriminative alignment)
      3. SIGReg (anti-collapse)

    At inference, z_text from JEPA is compared against a phrase library
    using cosine similarity → semantic verification (Gate 2).
    """

    def __init__(self, config: TextProjectorConfig):
        super().__init__()
        self.config = config

        # Text embedding normalization layer
        self.text_norm = nn.LayerNorm(config.text_dim)

        # TC-JEPA projector
        self.tc_projector = TextConditionedProjector(config)

        # Temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(
        self,
        z_future: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_future: (B, 1024) predicted future state
            text_embed: (B, 3584) current text condition
        Returns:
            z_text: (B, 3584) predicted text embedding
        """
        z_text = self.tc_projector(z_future, text_embed)
        return z_text

    def compute_contrastive_loss(
        self,
        z_text: torch.Tensor,
        target_text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss between predicted text embeddings and targets.

        Args:
            z_text: (B, D) predicted text embeddings
            target_text_embeds: (B, D) actual text embeddings
        Returns:
            scalar loss
        """
        # Normalize
        z_text = F.normalize(z_text, dim=-1)
        target_text_embeds = F.normalize(target_text_embeds, dim=-1)

        # Cosine similarity matrix
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * z_text @ target_text_embeds.T  # (B, B)

        # Labels: diagonal is positive
        labels = torch.arange(z_text.shape[0], device=z_text.device)

        # Symmetric InfoNCE
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2.0

    def compute_alignment_loss(
        self,
        z_text: torch.Tensor,
        target_text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Direct MSE alignment loss.
        """
        return F.mse_loss(z_text, target_text_embeds)
