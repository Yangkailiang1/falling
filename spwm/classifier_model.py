"""
Lightweight JEPA Classifier for Fall Detection.

Instead of the complex MoE Fusion + Mamba Predictor + EMA Target pipeline,
this uses frozen V-JEPA 2 + WavJEPA encoders as feature extractors, with a
simple fusion MLP + binary classifier on top.

Architecture:
  Context frames (16) → V-JEPA 2 ViT-L (frozen) → mean pool → v_feat (1024)
  Context audio         → WavJEPA (frozen)        → mean pool → a_feat (768)
                              ↓
                     Concat → (1792)
                              ↓
                Fusion MLP: 1792 → 512 → 256
                              ↓
                Classifier: 256 → 1 → P(fall in next 0.3s)

Trainable: ~1.1M params (Fusion MLP + Classifier only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from pathlib import Path

from .config import TJEPSConfig, VideoEncoderConfig, AudioEncoderConfig
from .encoders import VJEPA2VideoEncoder, AudioJEPEncoder


class JEPAClassifier(nn.Module):
    """Lightweight classifier on top of frozen JEPA encoders."""

    def __init__(
        self,
        video_config: Optional[VideoEncoderConfig] = None,
        audio_config: Optional[AudioEncoderConfig] = None,
        video_dim: int = 1024,
        audio_dim: int = 768,
        fusion_hidden: int = 512,
        fusion_out: int = 256,
        dropout: float = 0.3,
        use_audio: bool = True,
        encoder_device: str = "cpu",
    ):
        super().__init__()
        self.use_audio = use_audio
        self.encoder_device = encoder_device
        self.video_dim = video_dim
        self.audio_dim = audio_dim

        # Encoders (lazy init, frozen)
        self.video_encoder = VJEPA2VideoEncoder(
            video_config or VideoEncoderConfig()
        )
        self.audio_encoder = AudioJEPEncoder(
            audio_config or AudioEncoderConfig()
        ) if use_audio else None

        # Fusion input dimension
        fusion_input_dim = video_dim + (audio_dim if use_audio else 0)

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_out),
            nn.BatchNorm1d(fusion_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Classifier head
        self.classifier = nn.Linear(fusion_out, 1)

        # State
        self._encoders_placed = False

    def _place_encoders(self, device: str):
        """Move frozen encoders to target device."""
        if self._encoders_placed:
            return
        self._encoders_placed = True

        target = torch.device(device)
        if self.video_encoder is not None:
            self.video_encoder = self.video_encoder.to(target)
            self.video_encoder.eval()
            for p in self.video_encoder.parameters():
                p.requires_grad = False

        if self.audio_encoder is not None:
            self.audio_encoder = self.audio_encoder.to(target)
            self.audio_encoder.eval()
            for p in self.audio_encoder.parameters():
                p.requires_grad = False

    def _pool_video(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mean pool V-JEPA 2 tokens (B, N_v, D) → (B, D)."""
        if tokens.dim() == 3:
            return tokens.mean(dim=1)
        return tokens

    def _pool_audio(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mean pool WavJEPA tokens (B, N_a, D) → (B, D)."""
        if tokens.dim() == 3:
            return tokens.mean(dim=1)
        return tokens

    def forward(
        self,
        frames: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            frames: (B, T, C, H, W) video frames
            audio:  (B, L) audio waveform (optional)

        Returns:
            logits: (B, 1) raw logits for BCEWithLogitsLoss
            info:   dict with pooled features
        """
        B = frames.shape[0]
        enc_dev = self.encoder_device

        # Video encoding
        frames_enc = frames.to(enc_dev) if frames.device != enc_dev else frames
        v_tokens = self.video_encoder(frames_enc)  # (B, N_v, 1024)
        v_feat = self._pool_video(v_tokens)  # (B, 1024)
        if v_feat.device != frames.device:
            v_feat = v_feat.to(frames.device)

        # Audio encoding
        if self.use_audio and self.audio_encoder is not None and audio is not None:
            audio_enc = audio.to(enc_dev) if audio.device != enc_dev else audio
            a_tokens = self.audio_encoder(audio_enc)  # (B, N_a, 768)
            a_feat = self._pool_audio(a_tokens)  # (B, 768)
            if a_feat.device != frames.device:
                a_feat = a_feat.to(frames.device)
        else:
            a_feat = torch.zeros(B, self.audio_dim, device=frames.device)

        # Fusion
        if self.use_audio:
            fused_in = torch.cat([v_feat, a_feat], dim=-1)  # (B, video_dim + audio_dim)
        else:
            fused_in = v_feat  # video-only
        fused_out = self.fusion(fused_in)  # (B, fusion_out)
        logits = self.classifier(fused_out)  # (B, 1)

        info = {
            'v_feat': v_feat,
            'a_feat': a_feat,
            'fused': fused_out,
        }
        return logits, info

    @torch.no_grad()
    def predict(self, frames: torch.Tensor, audio: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict fall probability."""
        logits, _ = self.forward(frames, audio)
        return torch.sigmoid(logits).squeeze(-1)  # (B,)

    @torch.no_grad()
    def extract_features(
        self,
        frames: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract video and audio features for external use."""
        enc_dev = self.encoder_device
        frames_enc = frames.to(enc_dev) if frames.device != enc_dev else frames
        v_tokens = self.video_encoder(frames_enc)
        v_feat = self._pool_video(v_tokens)

        a_feat = None
        if self.use_audio and self.audio_encoder is not None and audio is not None:
            audio_enc = audio.to(enc_dev) if audio.device != enc_dev else audio
            a_tokens = self.audio_encoder(audio_enc)
            a_feat = self._pool_audio(a_tokens)

        return v_feat, a_feat


def create_classifier_from_config(config: TJEPSConfig) -> JEPAClassifier:
    """Factory: create JEPAClassifier from TJEPSConfig."""
    return JEPAClassifier(
        video_config=config.video,
        audio_config=config.audio,
        video_dim=config.video.embed_dim,
        audio_dim=config.audio.embed_dim,
        fusion_hidden=config.fusion.expert_hidden_dim,  # reuse: 512
        dropout=config.fusion.dropout,
        use_audio=config.use_audio,
        encoder_device=config.training.encoder_device,
    )
