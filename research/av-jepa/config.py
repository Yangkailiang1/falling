"""
AV-JEPA Configuration
--------------------
Hyperparameters for the Audio-Visual Joint Embedding Predictive Architecture.
Inspired by LeWorldModel (SIGReg) and V-JEPA 2.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

# Resolve project root and model weights path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_WEIGHTS = os.path.join(_PROJECT_ROOT, "spwm", "model_weights")


@dataclass
class EncoderConfig:
    """Video and audio encoder settings."""

    # Video encoder: CLIP ViT (loaded from local model_weights/)
    clip_model: str = os.path.join(_MODEL_WEIGHTS, "clip-vit-base-patch16")
    video_embed_dim: int = 768  # CLIP ViT-B/16 hidden_size
    video_frames: int = 8  # number of frames per clip
    video_size: int = 224  # frame resize

    # Audio encoder: CLAP (loaded from local model_weights/)
    clap_model: str = os.path.join(_MODEL_WEIGHTS, "clap-htsat-unfused")
    audio_embed_dim: int = 512  # CLAP audio output dim
    audio_sample_rate: int = 48000
    audio_duration: float = 3.0  # seconds per audio clip

    # Joint embedding
    joint_embed_dim: int = 256  # fused AV embedding dim


@dataclass
class JEPAConfig:
    """JEPA predictor and training settings."""

    # Predictor architecture
    predictor_hidden_dim: int = 512
    predictor_num_layers: int = 3
    predictor_num_heads: int = 8
    predictor_dropout: float = 0.1

    # SIGReg regularization (from LeWorldModel)
    sigreg_weight: float = 0.3
    sigreg_target_std: float = 1.0  # target standard deviation for embeddings

    # Prediction
    time_horizon: int = 4  # predict N steps ahead

    # EMA target encoder
    ema_decay: float = 0.996
    ema_end_decay: float = 1.0

    # Loss
    prediction_loss: str = "l2"  # "l2" or "cosine"


@dataclass
class TrainingConfig:
    """Training loop settings."""

    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # LR schedule
    lr_scheduler: str = "cosine"  # "cosine" or "linear"

    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500

    # Data
    data_path: Optional[str] = None  # path to video+audio dataset
    num_workers: int = 4

    # Device
    device: str = "cuda"  # "cuda", "mps", or "cpu"


@dataclass
class AVJEPAConfig:
    """Master configuration combining all sub-configs."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    jepa: JEPAConfig = field(default_factory=JEPAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Anomaly detection
    anomaly_threshold_percentile: float = 95.0  # top N% prediction error = anomaly

    def __post_init__(self):
        import torch
        if self.training.device == "cuda" and not torch.cuda.is_available():
            self.training.device = "cpu"
        if self.training.device == "mps" and not torch.backends.mps.is_available():
            self.training.device = "cpu"
