"""
T-JEPA Fusion Module

M3-JEPA Multi-Gate MoE Fusion + Fall-Mamba Cross-Attention refinement.

Architecture (two-layer fusion):
  1. M3-JEPA MoE: 4 modality gates → 4 experts + 1 shared expert → coarse fusion
  2. Fall-Mamba Cross-Attention: Cross-modal attention refinement
  3. Final fusion projection → z_fused (1024)

Both layers are trainable (~8M params).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from .config import M3JEPAFusionConfig, FallMambaConfig


class M3JEPAFusion(nn.Module):
    """
    M3-JEPA style multi-modal fusion with MoE gating.

    4 input modalities → per-modality projection → MoE gating → fused embedding.

    Key insight (from M3-JEPA): the gating mechanism decouples modality-specific
    information from cross-modal shared information.
    Learning objective: maximize I(z_fused; targets) - minimize H(z_fused | inputs)

    Supports modality toggles: disabled modalities get zero gate weights,
    so they contribute nothing to the fused representation.
    """

    def __init__(self, config: M3JEPAFusionConfig):
        super().__init__()
        self.config = config
        dim = config.unified_dim
        experts = config.num_experts

        # ━━ Per-modality projection to unified dimension ━━
        self.map_v = nn.Sequential(
            nn.Linear(config.video_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.map_a = nn.Sequential(
            nn.Linear(config.audio_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.map_s = nn.Sequential(
            nn.Linear(config.skeleton_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.map_t = nn.Sequential(
            nn.Linear(config.text_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # ━━ MoE Gates (per-modality, output weights for experts + shared) ━━
        n_total = experts + config.num_shared_experts  # 4 + 1 = 5
        self.gate_v = nn.Linear(dim, n_total)
        self.gate_a = nn.Linear(dim, n_total)
        self.gate_s = nn.Linear(dim, n_total)
        self.gate_t = nn.Linear(dim, n_total)

        # ━━ Experts (per-modality processing) ━━
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.expert_hidden_dim, dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(dim, dim),
            )
            for _ in range(experts)
        ])

        # ━━ Shared Expert (cross-modal common information) ━━
        self.shared_expert = nn.Sequential(
            nn.Linear(config.expert_hidden_dim, dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim, dim),
        )

        # ━━ Expert input projection ━━
        # Project modality vectors to expert input dimension
        self.expert_input_proj = nn.Linear(dim, config.expert_hidden_dim)
        self.shared_input_proj = nn.Linear(dim, config.expert_hidden_dim)

        # ━━ Load balancing (M3-JEPA) ━━
        # Encourage uniform expert usage
        self.load_balance_loss_weight = 0.01

        # ━━ Fusion projection ━━
        self.fusion_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def _pool_modality(self, tokens: torch.Tensor, method: str = 'mean') -> torch.Tensor:
        """Pool multi-token modality to a single vector."""
        if tokens.dim() == 2:
            return tokens  # already pooled
        if method == 'mean':
            return tokens.mean(dim=1)
        elif method == 'max':
            return tokens.max(dim=1)[0]
        elif method == 'cls':
            return tokens[:, 0, :]
        else:
            return tokens.mean(dim=1)

    def forward(
        self,
        v_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
        s_tokens: torch.Tensor,
        t_embed: torch.Tensor,
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> Dict:
        """
        Args:
            v_tokens: (B, N_v, 1024) or (B, 1024)
            a_tokens: (B, N_a, 768)  or (B, 768)
            s_tokens: (B, N_s, 256)  or (B, 256)
            t_embed:  (B, 3584)      pooled text embedding
            modality_mask: dict with keys 'video','audio','skeleton','text' → bool.
                           Disabled modalities get zero gate weights.

        Returns:
            dict with keys: z_fused, z_moe, gate_weights (for analysis)
        """
        if modality_mask is None:
            modality_mask = {'video': True, 'audio': True, 'skeleton': True, 'text': True}

        B = v_tokens.shape[0]

        # ━━ 1. Pool multi-token → single vector ━━
        v = self._pool_modality(v_tokens)  # (B, 1024)
        a = self._pool_modality(a_tokens)  # (B, 768)
        s = self._pool_modality(s_tokens)  # (B, 256)

        # ━━ 2. Project to unified dimension ━━
        v = self.map_v(v)  # (B, dim)
        a = self.map_a(a)  # (B, dim)
        s = self.map_s(s)  # (B, dim)
        t = self.map_t(t_embed)  # (B, dim)

        # ━━ 3. MoE Gating (zero out disabled modality gates) ━━
        g_v = F.softmax(self.gate_v(v), dim=-1)
        g_a = F.softmax(self.gate_a(a), dim=-1)
        g_s = F.softmax(self.gate_s(s), dim=-1)
        g_t = F.softmax(self.gate_t(t), dim=-1)

        if not modality_mask.get('video', True):
            g_v = torch.zeros_like(g_v)
        if not modality_mask.get('audio', True):
            g_a = torch.zeros_like(g_a)
        if not modality_mask.get('skeleton', True):
            g_s = torch.zeros_like(g_s)
        if not modality_mask.get('text', True):
            g_t = torch.zeros_like(g_t)

        # ━━ 4. Expert processing ━━
        v_exp = self.expert_input_proj(v)
        a_exp = self.expert_input_proj(a)
        s_exp = self.expert_input_proj(s)
        t_exp = self.expert_input_proj(t)

        fused = torch.zeros(B, self.config.unified_dim, device=v.device)
        gate_weights = {
            'v': g_v.detach(),
            'a': g_a.detach(),
            's': g_s.detach(),
            't': g_t.detach(),
        }

        for i, expert in enumerate(self.experts):
            e_v = expert(v_exp) * g_v[:, i:i + 1]
            e_a = expert(a_exp) * g_a[:, i:i + 1]
            e_s = expert(s_exp) * g_s[:, i:i + 1]
            e_t = expert(t_exp) * g_t[:, i:i + 1]
            fused += (e_v + e_a + e_s + e_t)

        # ━━ 5. Shared expert ━━
        shared_input = torch.zeros_like(self.shared_input_proj(v))
        if modality_mask.get('video', True):
            shared_input += self.shared_input_proj(v)
        if modality_mask.get('audio', True):
            shared_input += self.shared_input_proj(a)
        if modality_mask.get('skeleton', True):
            shared_input += self.shared_input_proj(s)
        if modality_mask.get('text', True):
            shared_input += self.shared_input_proj(t)

        shared_output = self.shared_expert(shared_input)
        z_moe = fused + shared_output

        # ━━ 6. Final fusion projection ━━
        z_fused = self.fusion_proj(z_moe)

        return {
            'z_fused': z_fused,
            'z_moe': z_moe,
            'gate_weights': gate_weights,
        }


class CrossAttentionRefinement(nn.Module):
    """
    Fall-Mamba Optimization 1: Cross-Attention refinement layer.

    After M3-JEPA MoE coarse fusion, apply cross-attention between modality pairs
    to refine the fused representation. This provides mid-level fusion (proven
    11% better than early fusion, 7% better than late fusion in Fall-Mamba).

    Cross-attention pairs:
      - Audio attends to Video (audio=Q, video=KV)
      - Video attends to Audio (video=Q, audio=KV)
      - Skeleton attends to Fused AV (skeleton=Q, fused_av=KV)
    """

    def __init__(self, config: M3JEPAFusionConfig):
        super().__init__()
        dim = config.unified_dim
        heads = config.cross_attn_heads
        self.dim = dim

        self.v_to_q = nn.Linear(dim, dim)
        self.a_to_q = nn.Linear(dim, dim)
        self.a_to_kv = nn.Linear(dim, dim)
        self.v_to_kv = nn.Linear(dim, dim)

        self.v2a_cross_attn = nn.MultiheadAttention(
            dim, heads, dropout=config.dropout, batch_first=True
        )
        self.a2v_cross_attn = nn.MultiheadAttention(
            dim, heads, dropout=config.dropout, batch_first=True
        )

        # Skeleton to AV cross-attention
        self.s_to_q = nn.Linear(dim, dim)
        self.av_to_kv = nn.Linear(dim, dim)
        self.s2av_cross_attn = nn.MultiheadAttention(
            dim, heads, dropout=config.dropout, batch_first=True
        )

        # Output projections
        self.v_out = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.a_out = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.s_out = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(
        self,
        v: torch.Tensor,  # (B, dim) video feature
        a: torch.Tensor,  # (B, dim) audio feature
        s: torch.Tensor,  # (B, dim) skeleton feature
    ) -> Dict[str, torch.Tensor]:
        B = v.shape[0]

        # Audio attends to Video (Fall-Mamba: audio=Q, video=KV)
        a_q = self.a_to_q(a).unsqueeze(1)    # (B, 1, dim)
        v_kv = self.v_to_kv(v).unsqueeze(1)  # (B, 1, dim)
        a_refined, _ = self.a2v_cross_attn(a_q, v_kv, v_kv)
        a_refined = self.a_out(a_refined.squeeze(1))

        # Video attends to Audio
        v_q = self.v_to_q(v).unsqueeze(1)
        a_kv = self.a_to_kv(a).unsqueeze(1)
        v_refined, _ = self.v2a_cross_attn(v_q, a_kv, a_kv)
        v_refined = self.v_out(v_refined.squeeze(1))

        # Skeleton attends to fused AV
        av_fused = torch.cat([v_refined.unsqueeze(1), a_refined.unsqueeze(1)], dim=1)  # (B, 2, dim)
        s_q = self.s_to_q(s).unsqueeze(1)
        av_kv = self.av_to_kv(av_fused)
        s_refined, _ = self.s2av_cross_attn(s_q, av_kv, av_kv)
        s_refined = self.s_out(s_refined.squeeze(1))

        return {
            'v_refined': v_refined,
            'a_refined': a_refined,
            's_refined': s_refined,
        }


class HybridFusion(nn.Module):
    """
    Hybrid fusion: M3-JEPA MoE (coarse) + Fall-Mamba Cross-Attention (refinement).

    This is the recommended fusion module for T-JEPA.
    Two-layer fusion provides:
      - MoE: decouples modality-specific vs shared information
      - Cross-Attention: mid-level cross-modal refinement (Fall-Mamba validated)
    """

    def __init__(self, config: M3JEPAFusionConfig, modality_mask: dict = None):
        super().__init__()
        self.config = config

        # Layer 1: M3-JEPA MoE coarse fusion
        self.moe_fusion = M3JEPAFusion(config)

        # Layer 2: Fall-Mamba Cross-Attention refinement
        self.use_cross_attn = config.use_cross_attention
        if self.use_cross_attn:
            self.cross_attn_refine = CrossAttentionRefinement(config)

        # Compute combined dimension from active cross-attention modalities
        # cross-attention handles: video, audio, skeleton (+ text not in cross-attn)
        if modality_mask is None:
            n_cross_attn_mods = 3  # default: video, audio, skeleton
        else:
            n_cross_attn_mods = sum([
                modality_mask.get('video', True),
                modality_mask.get('audio', True),
                modality_mask.get('skeleton', True),
            ])
        # combined = z_moe (output_dim) + cross-attn refined (output_dim * n_mods)
        combined_dim = config.output_dim * (1 + n_cross_attn_mods)

        # Final fusion projection
        self.final_proj = nn.Sequential(
            nn.Linear(combined_dim, config.output_dim * 2),
            nn.LayerNorm(config.output_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim * 2, config.output_dim),
            nn.LayerNorm(config.output_dim),
        )

    def _pool_modality(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 2:
            return tokens
        return tokens.mean(dim=1)

    def forward(
        self,
        v_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
        s_tokens: torch.Tensor,
        t_embed: torch.Tensor,
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> Dict:
        """
        Hybrid fusion: MoE + Cross-Attention → z_fused.

        Returns:
            dict with z_fused, z_moe, gate_weights
        """
        # Pool for per-modality vectors
        v = self._pool_modality(v_tokens)
        a = self._pool_modality(a_tokens)
        s = self._pool_modality(s_tokens)

        # Layer 1: M3-JEPA MoE
        moe_output = self.moe_fusion(v_tokens, a_tokens, s_tokens, t_embed, modality_mask)
        z_moe = moe_output['z_moe']  # (B, dim)

        if self.use_cross_attn:
            # Project pooled modalities for cross-attention
            v_proj = self.moe_fusion.map_v(v)
            a_proj = self.moe_fusion.map_a(a)
            s_proj = self.moe_fusion.map_s(s)

            # Layer 2: Cross-Attention refinement
            refined = self.cross_attn_refine(v_proj, a_proj, s_proj)

            # Combine MoE + Cross-Attention (only enabled modalities)
            parts = [z_moe]
            if modality_mask is None or modality_mask.get('video', True):
                parts.append(refined['v_refined'])
            if modality_mask is None or modality_mask.get('audio', True):
                parts.append(refined['a_refined'])
            if modality_mask is None or modality_mask.get('skeleton', True):
                parts.append(refined['s_refined'])
            combined = torch.cat(parts, dim=-1)
            z_fused = self.final_proj(combined)
        else:
            z_fused = z_moe

        return {
            'z_fused': z_fused,
            'z_moe': z_moe,
            'gate_weights': moe_output['gate_weights'],
        }


class SIGRegLoss(nn.Module):
    """
    SIGReg (Spectral Information Gap Regularization) from LeWorldModel/NOVA.

    Prevents representation collapse by encouraging the latent space to be
    isotropic Gaussian. Computes covariance matrix and penalizes off-diagonal
    elements + encourages diagonal elements to be close to 1.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D) latent representations
        Returns:
            scalar loss
        """
        B, D = z.shape
        if B < 2:
            return torch.tensor(0.0, device=z.device)

        # Center
        z_centered = z - z.mean(dim=0, keepdim=True)

        # Covariance matrix
        cov = (z_centered.T @ z_centered) / (B - 1)  # (D, D)

        # Off-diagonal penalty
        mask = torch.eye(D, device=z.device)
        off_diag = (cov * (1 - mask)).pow(2).sum()

        # Diagonal: encourage variance ≈ 1
        diag = (cov.diag() - 1.0).pow(2).sum()

        return (off_diag + diag) / D
