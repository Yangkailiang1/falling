"""
AV-JEPA v2 Configuration
-------------------------
V2 upgrades:
  - Video: V-JEPA 2 ViT-L (300M, pretrained world model)
  - Audio: WavJEPA base (200M, pretrained world model)
  - Fusion: Cross-modal token-level attention
  - Multi-token JEPA predictor
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class V2EncoderConfig:
    """V-JEPA 2 video + WavJEPA audio encoder settings."""

    # ── Video: V-JEPA 2 ──
    vjepa2_model: str = "vit_large"      # "vit_large" | "vit_huge" | "vit_giant"
    vjepa2_checkpoint: str = ""          # path to pretrained .pt, "" = auto-resolve from spwm/model_weights/
    video_embed_dim: int = 1024          # ViT-L hidden dim
    video_num_frames: int = 16           # frames per clip (match V-JEPA 2 pretrain)
    video_size: int = 224                # frame resize for ViT-L (256 for ViT-H/g)
    video_patch_size: int = 16
    video_tubelet_size: int = 2

    # ── Audio: WavJEPA ──
    wavjepa_model: str = ""              # "" = auto-load from spwm/model_weights/wavjepa-base/
    audio_embed_dim: int = 768           # WavJEPA output dim
    audio_sample_rate: int = 16000
    audio_duration: float = 3.0          # seconds per audio clip

    # ── Joint ──
    joint_embed_dim: int = 256           # shared latent space

    # ── Token pooling ──
    num_video_queries: int = 8           # learnable queries for video token pooling
    num_audio_queries: int = 4           # learnable queries for audio token pooling


@dataclass
class V2FusionConfig:
    """Cross-modal fusion settings."""

    fusion_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2              # cross-attention layers
    dropout: float = 0.1


@dataclass
class V2PredictorConfig:
    """Multi-token joint predictor settings."""

    predictor_hidden_dim: int = 512
    predictor_num_layers: int = 4
    predictor_num_heads: int = 8
    predictor_dropout: float = 0.1

    # SIGReg
    sigreg_weight: float = 0.3
    sigreg_target_std: float = 1.0

    # EMA
    ema_decay: float = 0.996
    ema_end_decay: float = 1.0

    # Loss
    prediction_loss: str = "l2"      # "l2" | "cosine"


@dataclass
class V2TrainingConfig:
    """Training loop settings."""

    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"

    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500

    data_path: Optional[str] = None
    feature_cache: Optional[str] = None  # path to pre-extracted features
    num_workers: int = 4
    device: str = "cuda"


@dataclass
class AVJEPAv2Config:
    """Master v2 configuration."""

    encoder: V2EncoderConfig = field(default_factory=V2EncoderConfig)
    fusion: V2FusionConfig = field(default_factory=V2FusionConfig)
    predictor: V2PredictorConfig = field(default_factory=V2PredictorConfig)
    training: V2TrainingConfig = field(default_factory=V2TrainingConfig)

    # Detection
    anomaly_threshold_sigma: float = 3.0
    anomaly_fusion_weights: tuple = (0.6, 0.4)  # (video_weight, audio_weight)

    def __post_init__(self):
        import torch
        if self.training.device == "cuda" and not torch.cuda.is_available():
            self.training.device = "cpu"
