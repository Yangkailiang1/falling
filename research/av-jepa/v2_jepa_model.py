"""
AV-JEPA v2 Core Model
----------------------
Cross-modal token-level fusion with V-JEPA 2 + WavJEPA backbones.

Architecture:
  ┌── Video Branch (V-JEPA 2 ViT-L, frozen) ──┐
  │  frames → multi-token → QueryPooler → v_tokens (8×1024)
  │                                            │
  ├── Audio Branch (WavJEPA, frozen) ─────────┤
  │  audio  → multi-token → QueryPooler → a_tokens (4×768)
  │                                            │
  ├── Cross-Modal Fusion ──────────────────────┤
  │  v_tokens ↔ CrossAttn ↔ a_tokens          │
  │  → joint_tokens (12×fusion_dim)            │
  │                                            │
  ├── Dual JEPA Predictor ────────────────────┤
  │  joint_ctx → Transformer → joint_pred      │
  │  Loss = L2(pred, target) + SIGReg          │
  │                                            │
  └── Target: EMA of projection layers ────────┘
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


# ─── Dimension Adapters ──────────────────────────────────────────────────

class DimAdapter(nn.Module):
    """Project modality-specific dim to fusion dim."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─── Cross-Modal Token Fusion ────────────────────────────────────────────

class CrossModalFusion(nn.Module):
    """
    Bidirectional cross-attention fusion at the token level.

    Video tokens attend to audio tokens (and vice versa), producing
    modality-enhanced representations. Final output concatenates
    both enhanced token sequences.

      v_tokens: (B, Qv, D)    a_tokens: (B, Qa, D)
          │                        │
    CrossAttn(v → a)      CrossAttn(a → v)
          │                        │
          ├──── concat ────────────┤
          └──→ joint_tokens: (B, Qv+Qa, D)
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # Bidirectional cross-attention layers
        self.v2a_layers = nn.ModuleList([
            CrossAttnBlock(dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.a2v_layers = nn.ModuleList([
            CrossAttnBlock(dim, num_heads, dropout) for _ in range(num_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        v_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            v_tokens: (B, Qv, D) video query tokens
            a_tokens: (B, Qa, D) audio query tokens

        Returns:
            joint_tokens: (B, Qv+Qa, D) fused multimodal tokens
        """
        v_enhanced = v_tokens
        a_enhanced = a_tokens

        for v2a, a2v in zip(self.v2a_layers, self.a2v_layers):
            # Video attends to audio
            v_enhanced = v2a(v_enhanced, a_enhanced)
            # Audio attends to video
            a_enhanced = a2v(a_enhanced, v_enhanced)

        # Concatenate enhanced tokens
        joint = torch.cat([v_enhanced, a_enhanced], dim=1)
        return self.output_norm(joint)


class CrossAttnBlock(nn.Module):
    """Single cross-attention block with residual connection."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )
        self.norm_out = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Cross-attention
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)
        x = q + attn_out

        # FFN
        x = x + self.ffn(self.norm_out(x))
        return x


# ─── Dual JEPA Predictor ─────────────────────────────────────────────────

class DualJEPAPredictor(nn.Module):
    """
    Multi-token JEPA predictor: joint_ctx → joint_pred.

    Takes fused context tokens from CrossModalFusion and predicts
    the fused tokens that the EMA target branch would produce from
    future observations.

    Architecture: Transformer encoder with context prefix token.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable context aggregation token
        self.ctx_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Position encoding for up to 64 tokens
        self.pos_embed = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_ctx: (B, S, embed_dim) context joint tokens

        Returns:
            z_pred: (B, S, embed_dim) predicted future joint tokens
        """
        B, S, D = z_ctx.shape

        # Prepend context token
        ctx = self.ctx_token.expand(B, -1, -1)
        x = torch.cat([ctx, z_ctx], dim=1)  # (B, 1+S, D)

        # Add position encoding
        if S + 1 <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :S + 1, :]

        # Transformer
        x = self.transformer(x)  # (B, 1+S, D)

        # Extract token predictions (skip context token)
        z_pred = self.output_head(x[:, 1:, :])  # (B, S, D)
        return z_pred


# ─── Full AV-JEPA v2 Model ───────────────────────────────────────────────

class AVJEPAv2(nn.Module):
    """
    AV-JEPA v2: Cross-modal JEPA for audio-visual fall detection.

    Components:
      1. V-JEPA 2 ViT-L    → video encoder (frozen)
      2. WavJEPA           → audio encoder (frozen)
      3. DimAdapters       → project modality dims to fusion dim
      4. QueryTokenPooler  → compress multi-token → fixed queries
      5. CrossModalFusion  → bidirectional cross-attention fusion
      6. DualJEPAPredictor → predict future joint tokens
      7. EMA target branch → slow-moving copy of 3+4+5
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        enc = config.encoder
        fus = config.fusion
        pred = config.predictor

        # ── Encoders (frozen) ──
        from v2_encoders import VJEPA2VideoEncoder, WavJEPAAudioEncoder, QueryTokenPooler
        self.video_encoder = VJEPA2VideoEncoder(
            model_name=enc.vjepa2_model,
            checkpoint_path=enc.vjepa2_checkpoint,
            img_size=enc.video_size,
            num_frames=enc.video_num_frames,
            embed_dim=enc.video_embed_dim,
        )
        self.audio_encoder = WavJEPAAudioEncoder(
            model_name=enc.wavjepa_model,
            sample_rate=enc.audio_sample_rate,
            embed_dim=enc.audio_embed_dim,
        )

        # ── Dimension adapters ──
        self.video_adapter = DimAdapter(enc.video_embed_dim, fus.fusion_dim)
        self.audio_adapter = DimAdapter(enc.audio_embed_dim, fus.fusion_dim)

        # ── Query token poolers ──
        self.video_pooler = QueryTokenPooler(
            embed_dim=fus.fusion_dim,
            num_queries=enc.num_video_queries,
        )
        self.audio_pooler = QueryTokenPooler(
            embed_dim=fus.fusion_dim,
            num_queries=enc.num_audio_queries,
        )

        # ── Cross-modal fusion ──
        self.fusion = CrossModalFusion(
            dim=fus.fusion_dim,
            num_heads=fus.num_heads,
            num_layers=fus.num_layers,
            dropout=fus.dropout,
        )

        # ── EMA target branch ──
        self.target_video_adapter = copy.deepcopy(self.video_adapter)
        self.target_audio_adapter = copy.deepcopy(self.audio_adapter)
        self.target_video_pooler = copy.deepcopy(self.video_pooler)
        self.target_audio_pooler = copy.deepcopy(self.audio_pooler)
        self.target_fusion = copy.deepcopy(self.fusion)
        for p in self._target_params():
            p.requires_grad = False

        # ── Predictor ──
        self.predictor = DualJEPAPredictor(
            embed_dim=fus.fusion_dim,
            num_layers=pred.predictor_num_layers,
            num_heads=pred.predictor_num_heads,
            hidden_dim=pred.predictor_hidden_dim,
            dropout=pred.predictor_dropout,
        )

        # SIGReg
        from fusion import SIGRegLoss
        self.sigreg = SIGRegLoss(target_std=pred.sigreg_target_std)

        # Config refs
        self.sigreg_weight = pred.sigreg_weight
        self.ema_decay = pred.ema_decay
        self.ema_end_decay = pred.ema_end_decay
        self.fusion_dim = fus.fusion_dim
        self.num_total_tokens = enc.num_video_queries + enc.num_audio_queries

    def _target_params(self):
        """All parameters in the EMA target branch."""
        for m in [self.target_video_adapter, self.target_audio_adapter,
                  self.target_video_pooler, self.target_audio_pooler,
                  self.target_fusion]:
            yield from m.parameters()

    def _encode_branch(self, frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Encode context: video+audio → fused joint tokens."""
        # Video
        v_tokens = self.video_encoder(frames)
        v_tokens = self.video_adapter(v_tokens)
        v_tokens = self.video_pooler(v_tokens)

        # Audio
        a_tokens = self.audio_encoder(audio)
        a_tokens = self.audio_adapter(a_tokens)
        a_tokens = self.audio_pooler(a_tokens)

        # Fusion
        return self.fusion(v_tokens, a_tokens)

    @torch.no_grad()
    def _encode_target_branch(self, frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Encode target (future): video+audio → fused joint tokens (EMA branch)."""
        # Video
        v_tokens = self.video_encoder(frames)
        v_tokens = self.target_video_adapter(v_tokens)
        v_tokens = self.target_video_pooler(v_tokens)

        # Audio
        a_tokens = self.audio_encoder(audio)
        a_tokens = self.target_audio_adapter(a_tokens)
        a_tokens = self.target_audio_pooler(a_tokens)

        # Fusion
        return self.target_fusion(v_tokens, a_tokens)

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
            ctx_frames: (B, T_v, 3, H, W) context video
            ctx_audio:  (B, audio_len)     context audio
            tgt_frames: (B, T_v, 3, H, W) target (future) video
            tgt_audio:  (B, audio_len)     target (future) audio

        Returns:
            z_pred:   (B, S, fusion_dim) predicted future joint tokens
            z_target: (B, S, fusion_dim) actual future joint tokens (EMA)
            z_ctx:    (B, S, fusion_dim) context joint tokens
        """
        z_ctx = self._encode_branch(ctx_frames, ctx_audio)
        z_pred = self.predictor(z_ctx)
        z_target = self._encode_target_branch(tgt_frames, tgt_audio)
        return z_pred, z_target, z_ctx

    def compute_loss(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        z_ctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        JEPA loss = prediction loss + λ * SIGReg.

        Args:
            z_pred:   (B, S, D) predicted tokens
            z_target: (B, S, D) target tokens (EMA)
            z_ctx:    (B, S, D) context tokens

        Returns:
            total_loss: scalar
            loss_dict: breakdown
        """
        pred_loss = F.mse_loss(z_pred, z_target)
        sigreg_loss = self.sigreg(
            z_ctx.mean(dim=1), z_target.mean(dim=1)
        )
        total = pred_loss + self.sigreg_weight * sigreg_loss

        return total, {
            "pred_loss": pred_loss.item(),
            "sigreg_loss": sigreg_loss.item() if isinstance(sigreg_loss, torch.Tensor) else sigreg_loss,
            "total_loss": total.item(),
        }

    @torch.no_grad()
    def compute_surprise(
        self,
        ctx_frames: torch.Tensor,
        ctx_audio: torch.Tensor,
        tgt_frames: torch.Tensor,
        tgt_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sample surprise score for anomaly detection.

        Returns:
            surprise: (B,) scalar per sample
        """
        z_pred, z_target, _ = self.forward(ctx_frames, ctx_audio, tgt_frames, tgt_audio)
        return F.mse_loss(z_pred, z_target, reduction="none").mean(dim=(-1, -2))

    @torch.no_grad()
    def compute_modality_surprise(
        self,
        ctx_frames: torch.Tensor,
        ctx_audio: torch.Tensor,
        tgt_frames: torch.Tensor,
        tgt_audio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-modality and joint surprise scores.

        Returns:
            {"video_surprise": (B,), "audio_surprise": (B,), "joint_surprise": (B,)}
        """
        self.eval()

        # Full forward for joint
        z_pred, z_target, z_ctx = self.forward(ctx_frames, ctx_audio, tgt_frames, tgt_audio)
        joint_surprise = F.mse_loss(z_pred, z_target, reduction="none").mean(dim=(-1, -2))

        # Video-only: compare context vs target raw tokens (encoder space)
        v_ctx = self.video_encoder(ctx_frames)
        v_tgt = self.video_encoder(tgt_frames)
        video_surprise = F.mse_loss(v_ctx, v_tgt, reduction="none").mean(dim=(-1, -2))

        # Audio-only
        a_ctx = self.audio_encoder(ctx_audio)
        a_tgt = self.audio_encoder(tgt_audio)
        audio_surprise = F.mse_loss(a_ctx, a_tgt, reduction="none").mean(dim=(-1, -2))

        return {
            "video_surprise": video_surprise,
            "audio_surprise": audio_surprise,
            "joint_surprise": joint_surprise,
        }

    @torch.no_grad()
    def update_target_encoder(self, step: int, total_steps: int):
        """Cosine-scheduled EMA update of target branch."""
        tau = (
            self.ema_end_decay
            - (self.ema_end_decay - self.ema_decay)
            * (math.cos(math.pi * step / max(total_steps, 1)) + 1) / 2
        )
        for p_tgt, p_src in zip(self._target_params(),
                                 self._source_params()):
            p_tgt.data.copy_(tau * p_tgt.data + (1.0 - tau) * p_src.data)

    def _source_params(self):
        """All parameters in the online branch (for EMA copy)."""
        for m in [self.video_adapter, self.audio_adapter,
                  self.video_pooler, self.audio_pooler, self.fusion]:
            yield from m.parameters()

    def trainable_parameters(self):
        """Return all trainable parameters (excludes frozen encoders)."""
        params = []
        for m in [self.video_adapter, self.audio_adapter,
                  self.video_pooler, self.audio_pooler,
                  self.fusion, self.predictor]:
            params.extend(list(m.parameters()))
        return params

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())
