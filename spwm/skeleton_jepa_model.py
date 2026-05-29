"""
Skeleton-JEPA: Joint Embedding Predictive Architecture for 2D keypoints.

Architecture:
  Context skeleton (T_ctx frames) → SkeletonEncoder (Transformer) → z_ctx (d_model)
  Target skeleton (T_tgt frames) → SkeletonEncoder (EMA)           → z_tgt (d_model)
  Predictor: z_ctx → MLP → z_pred
  Loss: MSE(z_pred, z_tgt)

After JEPA pretraining, freeze encoder and add a lightweight MLP classifier
for supervised fall detection — same pattern as the video+audio JEPAClassifier.

Design for longer prediction horizon:
  - context: 16 frames (0.64s @ 25fps)
  - gap: 16+ frames (≥0.64s skipped)
  - target: 16 frames (0.64s being predicted)
  → predicts skeleton states 0.64-1.28s (or further) into the future
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from copy import deepcopy


class SkeletonEncoder(nn.Module):
    """Transformer encoder for 17-keypoint skeleton sequences.

    Input:  (B, T, 17, 3)  →  (x, y, confidence) per keypoint per frame
    Output: (B, T, d_model) → per-frame latent representations
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        input_dim: int = 3,          # (x, y, confidence) or (x, y, z)
        d_model: int = 256,
        n_head: int = 8,
        n_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_frames: int = 512,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        self.input_dim = input_dim

        in_features = num_keypoints * input_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.pos_embed = nn.Parameter(torch.randn(1, max_frames, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        x = x.reshape(B, T, -1)                     # (B, T, 51)
        x = self.input_proj(x)                       # (B, T, d_model)
        x = x + self.pos_embed[:, :T, :]             # add position
        x = self.transformer(x)                      # (B, T, d_model), batch_first
        x = self.norm(x)
        return x

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pool encoded sequence to single vector."""
        tokens = self.forward(x)                     # (B, T, d_model)
        return tokens.mean(dim=1)                    # (B, d_model)


class SkeletonPredictor(nn.Module):
    """MLP predictor: context representation → predicted target representation.

    Lightweight design: d_model → hidden → ... → hidden → d_model.
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = d_model
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else d_model
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim) if out_dim == hidden_dim else nn.Identity(),
                nn.GELU() if i < n_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < n_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        return self.mlp(z_ctx)


class SkeletonJEPA(nn.Module):
    """Joint Embedding Predictive Architecture for skeleton keypoints.

    Self-supervised pretraining: given context skeleton, predict the latent
    representation of the future target skeleton. This forces the encoder to
    learn temporal dynamics predictive of future states.

    Usage:
        model = SkeletonJEPA()
        z_pred, z_tgt = model(ctx_kp, tgt_kp)
        loss = F.mse_loss(z_pred, z_tgt)
        model.update_target_encoder(momentum=0.996)
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        d_model: int = 256,
        n_head: int = 8,
        n_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        predictor_hidden: int = 512,
        predictor_n_layers: int = 3,
        ema_start: float = 0.996,
    ):
        super().__init__()
        self.d_model = d_model
        self.ema_start = ema_start

        self.context_encoder = SkeletonEncoder(
            num_keypoints=num_keypoints,
            input_dim=3,
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.target_encoder = SkeletonEncoder(
            num_keypoints=num_keypoints,
            input_dim=3,
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.predictor = SkeletonPredictor(
            d_model=d_model,
            hidden_dim=predictor_hidden,
            n_layers=predictor_n_layers,
            dropout=dropout,
        )

        # Initialize target encoder = context encoder, freeze
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(
        self,
        ctx_kp: torch.Tensor,
        tgt_kp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for JEPA pretraining.

        Args:
            ctx_kp: (B, T_ctx, 17, 3) context keypoints
            tgt_kp: (B, T_tgt, 17, 3) target keypoints

        Returns:
            z_pred: (B, d_model) predicted target representation
            z_tgt:  (B, d_model) target encoder output (detached)
        """
        z_ctx_pooled = self.context_encoder.pooled(ctx_kp)
        z_pred = self.predictor(z_ctx_pooled)

        with torch.no_grad():
            z_tgt_pooled = self.target_encoder.pooled(tgt_kp)

        return z_pred, z_tgt_pooled

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None):
        """EMA update target encoder weights toward context encoder."""
        if momentum is None:
            momentum = self.ema_start
        for ctx_p, tgt_p in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            tgt_p.data = momentum * tgt_p.data + (1.0 - momentum) * ctx_p.data

    def get_encoder(self) -> SkeletonEncoder:
        """Return context encoder for downstream use (classifier training)."""
        return self.context_encoder


class SkeletonClassifier(nn.Module):
    """Lightweight classifier on top of a frozen SkeletonEncoder.

    Same pattern as JEPAClassifier: freeze encoder → fusion MLP → classifier.

    Architecture:
      Skeleton (T, 17, 3) → Frozen SkeletonEncoder → mean pool → (d_model)
        → Fusion MLP: d_model → 512 → 256
        → Classifier: 256 → 1 → sigmoid → P(fall in next future_frames)
    """

    def __init__(
        self,
        encoder: SkeletonEncoder,
        d_model: int = 256,
        fusion_hidden: int = 512,
        fusion_out: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.fusion = nn.Sequential(
            nn.Linear(d_model, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_out),
            nn.BatchNorm1d(fusion_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(fusion_out, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass.

        Args:
            x: (B, T, 17, 3) skeleton keypoint sequence

        Returns:
            logits: (B, 1) raw logits
            info: dict with intermediate features
        """
        with torch.no_grad():
            z = self.encoder(x)                      # (B, T, d_model)
            z_pooled = z.mean(dim=1)                 # (B, d_model)

        fused = self.fusion(z_pooled)                # (B, fusion_out)
        logits = self.classifier(fused)              # (B, 1)

        info = {"embedding": z_pooled, "fused": fused}
        return logits, info

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict fall probability."""
        logits, _ = self.forward(x)
        return torch.sigmoid(logits).squeeze(-1)

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_skeleton_jepa(
    num_keypoints: int = 17,
    d_model: int = 256,
    n_head: int = 8,
    n_layers: int = 4,
    dropout: float = 0.1,
    predictor_hidden: int = 512,
) -> SkeletonJEPA:
    """Factory: create SkeletonJEPA with default configuration."""
    return SkeletonJEPA(
        num_keypoints=num_keypoints,
        d_model=d_model,
        n_head=n_head,
        n_layers=n_layers,
        mlp_ratio=4,
        dropout=dropout,
        predictor_hidden=predictor_hidden,
        predictor_n_layers=3,
        ema_start=0.996,
    )


def create_skeleton_classifier(
    encoder: SkeletonEncoder,
    d_model: int = 256,
    fusion_hidden: int = 512,
    fusion_out: int = 256,
    dropout: float = 0.3,
) -> SkeletonClassifier:
    """Factory: create SkeletonClassifier from pretrained encoder."""
    return SkeletonClassifier(
        encoder=encoder,
        d_model=d_model,
        fusion_hidden=fusion_hidden,
        fusion_out=fusion_out,
        dropout=dropout,
    )
