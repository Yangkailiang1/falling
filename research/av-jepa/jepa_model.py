"""
AV-JEPA Core Model
------------------
Joint Embedding Predictive Architecture for Audio-Visual data.

Architecture:
    Context AV (t) ──→ [Frozen Encoder + Projector] ──→ z_ctx
    Target AV(t+k) ──→ [EMA Encoder + Projector]    ──→ z_tgt
    z_ctx ──→ [Transformer Predictor] ──→ z_pred

    Loss = L2(z_pred, z_tgt) + λ * SIGReg(z_ctx, z_tgt)

Inspired by:
    - V-JEPA 2 (Meta): latent-space prediction, EMA target
    - LeWorldModel: SIGReg regularization, lightweight design
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class JEPAPredictor(nn.Module):
    """
    Transformer-based predictor: z_ctx → z_pred (should match z_tgt).

    Lightweight design: 3 Transformer layers, ~2M parameters.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Context token + optional positional encoding
        self.context_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection: transformer output → predicted target embedding
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
        """
        Predict target embedding from context embedding.

        Args:
            z_ctx: (B, embed_dim) context joint AV embedding

        Returns:
            z_pred: (B, embed_dim) predicted target joint AV embedding
        """
        # Add sequence dim: (B, 1, embed_dim)
        B = z_ctx.shape[0]
        x = z_ctx.unsqueeze(1)

        # Prepend context token
        ctx_token = self.context_token.expand(B, -1, -1)
        x = torch.cat([ctx_token, x], dim=1)  # (B, 2, embed_dim)

        # Transformer forward
        x = self.transformer(x)  # (B, 2, embed_dim)

        # Extract prediction from context token position
        z_pred = self.output_proj(x[:, 0, :])  # (B, embed_dim)
        return z_pred


class AVJEPA(nn.Module):
    """
    Audio-Visual Joint Embedding Predictive Architecture.

    Components:
        1. AV Encoder (frozen CLIP + CLAP)
        2. Projector (concat → joint space)
        3. Predictor (Transformer, ~2M params)
        4. Target Encoder (EMA of encoder + projector)

    Training:
        - Freeze encoder
        - Train projector + predictor
        - EMA update target projector
        - Loss = L2(pred, target) + SIGReg
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        enc_cfg = config.encoder
        jepa_cfg = config.jepa

        # Encoder (frozen pretrained models + trainable projector)
        try:
            from .encoders import AVEncoder
        except ImportError:
            from encoders import AVEncoder
        self.encoder = AVEncoder(enc_cfg)

        # Projector: [v_emb | a_emb] → joint embedding
        try:
            from .fusion import AVProjector
        except ImportError:
            from fusion import AVProjector
        self.projector = AVProjector(
            video_dim=enc_cfg.video_embed_dim,
            audio_dim=enc_cfg.audio_embed_dim,
            joint_dim=enc_cfg.joint_embed_dim,
        )

        # Target projector (EMA of projector — LeWorldModel style)
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False

        # Encoder is frozen — reuse same encoder for target (no deep copy needed)

        # Predictor
        self.predictor = JEPAPredictor(
            embed_dim=enc_cfg.joint_embed_dim,
            num_heads=jepa_cfg.predictor_num_heads,
            num_layers=jepa_cfg.predictor_num_layers,
            hidden_dim=jepa_cfg.predictor_hidden_dim,
            dropout=jepa_cfg.predictor_dropout,
        )

        # SIGReg loss
        try:
            from .fusion import SIGRegLoss
        except ImportError:
            from fusion import SIGRegLoss
        self.sigreg = SIGRegLoss(target_std=jepa_cfg.sigreg_target_std)

        self.joint_dim = enc_cfg.joint_embed_dim
        self.time_horizon = jepa_cfg.time_horizon
        self.ema_decay = jepa_cfg.ema_decay
        self.ema_end_decay = jepa_cfg.ema_end_decay
        self.sigreg_weight = jepa_cfg.sigreg_weight

    @torch.no_grad()
    def _encode_target(self, frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Encode target (future) frames+audio with frozen encoder."""
        # Reuse frozen encoder (no separate copy needed)
        self.encoder.eval()
        target_raw = self.encoder(frames, audio)
        z_target = self.target_projector(target_raw)
        return z_target

    def forward_context(self, frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Encode context (current) frames+audio."""
        ctx_raw = self.encoder(frames, audio)
        z_ctx = self.projector(ctx_raw)
        return z_ctx

    def forward(
        self,
        ctx_frames: torch.Tensor,
        ctx_audio: torch.Tensor,
        tgt_frames: torch.Tensor,
        tgt_audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            ctx_frames: (B, T, C, H, W) context video frames
            ctx_audio: (B, audio_len) context audio
            tgt_frames: (B, T, C, H, W) target video frames (future)
            tgt_audio: (B, audio_len) target audio (future)

        Returns:
            z_pred: predicted target embedding
            z_target: actual target embedding
            z_ctx: context embedding (for SIGReg)
        """
        z_ctx = self.forward_context(ctx_frames, ctx_audio)
        z_pred = self.predictor(z_ctx)
        z_target = self._encode_target(tgt_frames, tgt_audio)
        return z_pred, z_target, z_ctx

    def compute_loss(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        z_ctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute JEPA training loss = prediction_loss + SIGReg.

        Returns:
            total_loss: scalar
            loss_dict: breakdown of individual losses
        """
        # Prediction loss (L2)
        pred_loss = F.mse_loss(z_pred, z_target)

        # SIGReg regularization
        sigreg_loss = self.sigreg(z_ctx, z_target)

        # Total loss
        total_loss = pred_loss + self.sigreg_weight * sigreg_loss

        loss_dict = {
            "pred_loss": pred_loss.item(),
            "sigreg_loss": sigreg_loss.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, loss_dict

    @torch.no_grad()
    def compute_surprise(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        future_frames: torch.Tensor,
        future_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute surprise score = ||predicted - actual||².

        During inference: high surprise → anomaly (potential fall).

        Args:
            frames, audio: current observation
            future_frames, future_audio: actual future observation

        Returns:
            surprise: (B,) scalar per sample
        """
        z_pred, z_target, _ = self.forward(
            frames, audio, future_frames, future_audio
        )
        surprise = F.mse_loss(z_pred, z_target, reduction="none").mean(dim=-1)
        return surprise

    @torch.no_grad()
    def compute_modality_errors(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        future_frames: torch.Tensor,
        future_audio: torch.Tensor,
    ):
        """
        Compute per-modality and joint prediction errors.

        Returns a dict with:
            video_error: how much video embedding changes between ctx and tgt
            audio_error: how much audio embedding changes between ctx and tgt
            joint_error: combined JEPA prediction error
        """
        self.eval()

        # Raw encoder outputs (before projector)
        ctx_raw = self.encoder(frames, audio)
        tgt_raw = self.encoder(future_frames, future_audio)

        # Split: first half = video, second half = audio
        v_dim = self.config.encoder.video_embed_dim
        v_ctx, a_ctx = ctx_raw[:, :v_dim], ctx_raw[:, v_dim:]
        v_tgt, a_tgt = tgt_raw[:, :v_dim], tgt_raw[:, v_dim:]

        video_error = F.mse_loss(v_ctx, v_tgt, reduction="none").mean(dim=-1)
        audio_error = F.mse_loss(a_ctx, a_tgt, reduction="none").mean(dim=-1)

        # Joint JEPA prediction error
        z_ctx = self.projector(ctx_raw)
        z_pred = self.predictor(z_ctx)
        z_target = self.target_projector(tgt_raw)
        joint_error = F.mse_loss(z_pred, z_target, reduction="none").mean(dim=-1)

        return {
            "video_error": video_error,
            "audio_error": audio_error,
            "joint_error": joint_error,
        }

    @torch.no_grad()
    def update_target_encoder(self, step: int, total_steps: int):
        """
        EMA update of target encoder and projector.

        Uses cosine schedule for EMA decay (from V-JEPA 2).
        """
        # Cosine schedule
        tau = (
            self.ema_end_decay
            - (self.ema_end_decay - self.ema_decay)
            * (math.cos(math.pi * step / total_steps) + 1)
            / 2
        )

        # Update target projector (EMA)
        for p_tgt, p_src in zip(
            self.target_projector.parameters(), self.projector.parameters()
        ):
            p_tgt.data.copy_(tau * p_tgt.data + (1 - tau) * p_src.data)
