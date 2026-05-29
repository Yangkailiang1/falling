"""
T-JEPA Predictor Module

Predicts future latent states from current fused representations.

Two predictor variants:
  1. V-JEPA 2 Predictor: 3-layer Transformer (frozen, from V-JEPA 2 paper)
  2. Bidirectional Mamba Predictor: SSM-based (Fall-Mamba optimization 2)
     - O(N) linear complexity vs Transformer's O(N²)
     - 3-5x faster for long sequences, 50% less memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .config import PredictorConfig


# ═══════════════════════════════════════════════════════════════
# V-JEPA 2 Transformer Predictor (frozen)
# ═══════════════════════════════════════════════════════════════

class VJEPA2Predictor(nn.Module):
    """
    V-JEPA 2 style predictor: 3-layer Transformer that predicts future
    latent states from current fused embedding.

    Input:  z_fused (B, 1024)
    Output: z_future (B, 1024)

    The predictor is typically frozen (pre-trained V-JEPA 2 weights).
    """

    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.config = config
        dim = config.input_dim

        # Learned query token
        self.query_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.transformer_n_heads,
            dim_feedforward=config.transformer_hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_n_layers,
        )

        # Output projection
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, config.output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.query_token, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, z_fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_fused: (B, 1024) current fused latent state
        Returns:
            z_future: (B, 1024) predicted future latent state
        """
        B = z_fused.shape[0]

        # Prepare input: query token + context latent
        query = self.query_token.expand(B, -1, -1)  # (B, 1, 1024)
        context = z_fused.unsqueeze(1)               # (B, 1, 1024)

        # Transformer predicts future from context
        x = torch.cat([query, context], dim=1)      # (B, 2, 1024)
        x = self.transformer(x)

        # Take query output as prediction
        z_future = x[:, 0, :]                        # (B, 1024)
        z_future = self.output_norm(z_future)
        z_future = self.output_proj(z_future)

        return z_future


# ═══════════════════════════════════════════════════════════════
# Mamba Block (SSM-based, from Fall-Mamba optimization 2)
# ═══════════════════════════════════════════════════════════════

class MambaBlock(nn.Module):
    """
    Simplified Mamba SSM block.

    Implements selective state space model with:
      - Input projection (expanded by expand factor)
      - 1D convolution over sequence
      - SiLU activation
      - Selective SSM (parameterized by input)
      - Residual connection

    Note: This is a functional approximation of the Mamba architecture.
    For full Mamba performance, use `mamba-ssm` package.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = d_model * expand

        # Input projection (doubles channel count for gate mechanism)
        self.in_proj = nn.Linear(d_model, d_inner * 2)

        # 1D convolution over sequence dimension
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,  # depthwise
        )

        # Selective SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.rand(d_inner, d_state) * 0.1))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Delta (step size) projection — input-dependent
        self.dt_proj = nn.Linear(d_inner, d_inner)

        # SSM input projection B and output projection C
        self.B_proj = nn.Linear(d_inner, d_state)
        self.C_proj = nn.Linear(d_inner, d_state)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model)

        # Normalization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _selective_scan(self, u: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Simplified selective scan (SSM core).

        In full Mamba, this uses hardware-optimized parallel scans.
        Here we use a simplified recurrence for compatibility.

        Args:
            u: (B, L, d_inner) input sequence
            delta: (B, L, d_inner) time step
        Returns:
            y: (B, L, d_inner) output sequence
        """
        B, L, D = u.shape
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Discretize A
        delta_unsqueeze = delta.unsqueeze(-1)  # (B, L, d_inner, 1)
        A_discrete = torch.exp(delta_unsqueeze * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)

        # B and C
        B_disc = self.B_proj(u)  # (B, L, d_state)
        C_disc = self.C_proj(u)  # (B, L, d_state)

        # Simplified recurrence (not full hardware-aware scan)
        # State: (B, D, d_state)
        state = torch.zeros(B, D, self.d_state, device=u.device)
        outputs = []

        for t in range(L):
            # Update state
            u_t = u[:, t, :].unsqueeze(-1)  # (B, D, 1)
            B_t = B_disc[:, t, :].unsqueeze(1)  # (B, 1, d_state)
            state = A_discrete[:, t, :, :] * state + B_t * u_t

            # Output
            C_t = C_disc[:, t, :].unsqueeze(1)  # (B, 1, d_state)
            y_t = (state * C_t).sum(dim=-1)  # (B, D)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, D)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) input sequence
        Returns:
            out: (B, L, d_model) output sequence
        """
        residual = x
        x = self.norm(x)

        # Input projection with gate
        x_and_gate = self.in_proj(x)  # (B, L, 2*d_inner)
        x_proj, gate = x_and_gate.chunk(2, dim=-1)

        # 1D convolution
        x_conv = x_proj.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv[:, :, :x_conv.size(2) - (self.conv1d.kernel_size[0] - 1)]  # remove padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)

        # SiLU activation
        x_act = F.silu(x_conv)

        # Delta (input-dependent step size)
        delta = F.softplus(self.dt_proj(x_act))

        # Selective scan
        y = self._selective_scan(x_act, delta)

        # Gating
        y = y * F.silu(gate)

        # Output with skip connection
        y = self.out_proj(y)
        out = self.dropout(y) + residual

        return out


class MambaTemporalPredictor(nn.Module):
    """
    Bidirectional Mamba temporal predictor (Fall-Mamba optimization 2).

    Uses forward + backward Mamba blocks for bidirectional sequence modeling.
    O(N) complexity, 3-5x faster than Transformer on 32+ frame sequences.

    Input:  (B, 1, 1024) or sequence of states
    Output: (B, 1024) predicted future state
    """

    def __init__(self, config: PredictorConfig):
        super().__init__()
        dim = config.input_dim

        # Input projection
        self.input_proj = nn.Linear(dim, dim)

        # Forward Mamba blocks
        self.forward_blocks = nn.ModuleList([
            MambaBlock(dim, config.mamba_d_state, config.mamba_d_conv, config.mamba_expand)
            for _ in range(config.mamba_n_layers)
        ])

        # Backward Mamba blocks
        self.backward_blocks = nn.ModuleList([
            MambaBlock(dim, config.mamba_d_state, config.mamba_d_conv, config.mamba_expand)
            for _ in range(config.mamba_n_layers)
        ])

        # Bidirectional fusion
        self.output_proj = nn.Linear(dim * 2, config.output_dim)
        self.output_norm = nn.LayerNorm(config.output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, z_fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_fused: (B, 1024) current fused latent state
        Returns:
            z_future: (B, 1024) predicted future latent state
        """
        B = z_fused.shape[0]

        # Add sequence dimension: (B, 1, dim)
        x = self.input_proj(z_fused).unsqueeze(1)

        # Forward pass
        fwd = x
        for block in self.forward_blocks:
            fwd = block(fwd)

        # Backward pass (flip sequence)
        bwd = torch.flip(x, dims=[1])
        for block in self.backward_blocks:
            bwd = block(bwd)
        bwd = torch.flip(bwd, dims=[1])

        # Take last time step from both directions
        fwd_last = fwd[:, -1, :]  # (B, dim)
        bwd_last = bwd[:, -1, :]  # (B, dim)

        # Bidirectional fusion
        bi = torch.cat([fwd_last, bwd_last], dim=-1)  # (B, dim*2)
        z_future = self.output_proj(bi)
        z_future = self.output_norm(z_future)

        return z_future


# ═══════════════════════════════════════════════════════════════
# Predictor Factory
# ═══════════════════════════════════════════════════════════════

def build_predictor(config: PredictorConfig) -> nn.Module:
    """
    Factory function to create the appropriate predictor.

    Returns MambaTemporalPredictor if use_mamba=True,
    otherwise VJEPA2Predictor (Transformer).
    """
    if config.use_mamba:
        return MambaTemporalPredictor(config)
    else:
        return VJEPA2Predictor(config)
