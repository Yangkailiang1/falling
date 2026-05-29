"""
T-JEPA Configuration

All configuration dataclasses for the four-modality T-JEPA architecture:
  Video (V-JEPA 2 ViT-L) + Audio (Audio-JEPA ViT) + Skeleton (S-JEPA) + Text (Qwen2.5)
  → M3-JEPA MoE Fusion → VL-JEPA Text Alignment → 3-Tier Anomaly Gate

Fall-Mamba optimizations are parameterized via flags in the config.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Encoder Configs (all frozen at inference time)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class VideoEncoderConfig:
    """V-JEPA 2 ViT-L video encoder (300M params, frozen)."""
    model_name: str = "spwm/model_weights/vitl.pt"  # local checkpoint path
    local_repo: str = "spwm/model_weights/vjepa2"   # local vjepa2 source code
    image_size: int = 224
    num_frames: int = 16  # 8 context + 8 target
    patch_size: int = 16
    embed_dim: int = 1024  # output embedding dimension
    num_tokens: int = 1568  # (224/16)^2 * 16 = 196*8 = 1568 + CLS
    frozen: bool = True


@dataclass
class AudioEncoderConfig:
    """Audio-JEPA ViT encoder (~85M params, frozen).

    Fall-Mamba optimization: can fall back to Mel-Spectrogram + lightweight CNN
    for edge deployment.
    """
    model_name: str = "spwm/model_weights/clap-htsat-unfused"  # local CLAP path
    sample_rate: int = 16000
    duration: float = 3.0  # 3 seconds of audio
    mel_bins: int = 128
    embed_dim: int = 768
    num_tokens: int = 300  # approximate tokens from 3s @ 16kHz
    frozen: bool = True
    use_mel_fallback: bool = False  # edge-device fallback


@dataclass
class SkeletonEncoderConfig:
    """S-JEPA Transformer encoder for skeleton sequences (~22M params, frozen)."""
    model_name: str = "s-jepa-transformer"  # placeholder
    num_keypoints: int = 17  # COCO keypoints
    input_dim: int = 3  # (x, y, confidence)
    num_frames: int = 16
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    frozen: bool = True
    extraction_method: Literal["yolov8-pose", "mediapipe"] = "yolov8-pose"


@dataclass
class TextEncoderConfig:
    """Qwen2.5 embedding layer (frozen)."""
    model_name: str = "Qwen/Qwen2.5-7B"
    embed_dim: int = 3584  # Qwen2.5 embedding dimension
    max_length: int = 128
    frozen: bool = True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# M3-JEPA MoE Fusion Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class M3JEPAFusionConfig:
    """M3-JEPA Multi-Gate MoE fusion (trainable, ~8M params)."""
    # Input dimensions from each encoder
    video_dim: int = 1024
    audio_dim: int = 768
    skeleton_dim: int = 256
    text_dim: int = 3584

    # Unified projection dimension (all modalities projected here first)
    unified_dim: int = 1024

    # MoE configuration
    num_experts: int = 4  # per-modality experts
    num_shared_experts: int = 1  # cross-modal shared expert
    expert_hidden_dim: int = 512  # intermediate dim in expert MLP

    # Output
    output_dim: int = 1024  # z_fused dimension

    # Fall-Mamba: Cross-Attention refinement
    use_cross_attention: bool = True
    cross_attn_heads: int = 8

    # Regularization
    sigreg_lambda: float = 0.1
    dropout: float = 0.1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JEPA Predictor Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PredictorConfig:
    """V-JEPA 2 predictor for future state prediction (frozen)."""
    input_dim: int = 1024
    output_dim: int = 1024

    # Fall-Mamba: Bidirectional Mamba option
    use_mamba: bool = True  # use Mamba instead of Transformer for temporal prediction
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_n_layers: int = 4

    # Transformer fallback (if use_mamba=False)
    transformer_n_layers: int = 3
    transformer_n_heads: int = 8
    transformer_hidden_dim: int = 512

    dropout: float = 0.1
    frozen: bool = True  # V-JEPA 2 predictor weights are frozen


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VL-JEPA + TC-JEPA Projector Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TextProjectorConfig:
    """VL-JEPA + TC-JEPA text-conditioned projector (trainable, ~4M params)."""
    jepa_dim: int = 1024   # z_future dimension
    text_dim: int = 3584   # text embedding dimension
    output_dim: int = 3584  # z_text dimension (matches LLM embedding space)

    # Cross-attention for TC-JEPA text conditioning
    cross_attn_heads: int = 8
    projector_hidden: int = 2048  # MLP bottleneck

    dropout: float = 0.1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Anomaly Gate Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AnomalyGateConfig:
    """3-tier anomaly gating configuration."""
    # Gate 1: Statistical anomaly detection
    sigma_threshold: float = 2.0  # ||z_future - mu|| / sigma threshold
    gate1_dim: int = 1024  # latent dim for anomaly scoring

    # Gate 2: Semantic verification
    min_similarity: float = 0.5  # minimum cosine sim for fall phrase
    top_k_search: int = 3

    # Gate 3: LLM report (optional)
    enable_llm: bool = False
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Calibration
    calibration_samples: int = 1000


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fall-Mamba Optimization Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class FallMambaConfig:
    """Fall-Mamba paper optimizations applied to T-JEPA."""
    # Optimization 1: Cross-Attention refinement
    enable_cross_attention: bool = True

    # Optimization 2: Bidirectional Mamba (SSM) predictor
    enable_bidirectional_mamba: bool = True

    # Optimization 3: Frame Masking
    frame_mask_ratio: float = 0.2  # random mask 20% of frames during training

    # Optimization 4: DropPathway (single-modality fault tolerance)
    drop_modality_prob: float = 0.15  # probability of dropping a modality

    # Optimization 5: Mel-Spectrogram audio fallback
    enable_mel_fallback: bool = False  # for edge devices


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TrainingConfig:
    """Multi-stage training configuration."""
    # General
    device: str = "cuda"
    encoder_device: str = "cpu"  # frozen encoders on CPU to save GPU VRAM
    seed: int = 42
    mixed_precision: bool = True  # fp16/bf16
    grad_clip: float = 1.0

    # Data
    batch_size: int = 1
    num_workers: int = 2
    num_frames_context: int = 8
    num_frames_target: int = 8
    audio_duration: float = 3.0  # seconds
    sample_rate: int = 16000

    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)

    # Learning rate schedule
    warmup_steps: int = 1000
    lr_schedule: Literal["cosine", "linear"] = "cosine"

    # EMA (exponential moving average for target encoder)
    ema_start_decay: float = 0.996
    ema_end_decay: float = 1.0

    # Loss weights
    mse_weight: float = 1.0
    sigreg_weight: float = 0.1
    contrastive_weight: float = 0.3  # InfoNCE for text alignment
    mutual_info_weight: float = 0.3  # M3-JEPA mutual information

    # Stage-specific
    stage1_epochs: int = 100  # M3-JEPA fusion pre-training
    stage2_epochs: int = 100  # VL-JEPA text alignment
    stage3_epochs: int = 50   # OmniFall fine-tuning

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Realtime Inference Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class RealtimeConfig:
    """24fps real-time inference pipeline configuration."""
    # Frame rate
    fps: int = 24
    frame_budget_ms: float = 1000.0 / 24  # ~41.67ms

    # JEPA inference frequency
    jepa_interval_frames: int = 8  # run JEPA every 8 frames (3Hz)
    context_window_frames: int = 8
    target_window_frames: int = 8

    # Multi-threading
    use_threading: bool = True
    num_worker_threads: int = 2

    # Motion/audio event detection thresholds
    motion_threshold: float = 0.05  # minimum frame difference
    audio_event_threshold: float = 0.1

    # Skeleton extraction
    skeleton_method: Literal["yolov8-pose", "mediapipe"] = "mediapipe"

    # Async LLM (Gate 3)
    enable_async_llm: bool = False
    llm_timeout: int = 30  # seconds

    # Hardware
    device: str = "cuda"
    use_fp16: bool = True
    edge_deployment: bool = False  # Jetson Orin NX mode


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Master Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TJEPSConfig:
    """Master configuration for T-JEPA."""
    video: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)
    audio: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    skeleton: SkeletonEncoderConfig = field(default_factory=SkeletonEncoderConfig)
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    fusion: M3JEPAFusionConfig = field(default_factory=M3JEPAFusionConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    projector: TextProjectorConfig = field(default_factory=TextProjectorConfig)
    anomaly_gate: AnomalyGateConfig = field(default_factory=AnomalyGateConfig)
    fall_mamba: FallMambaConfig = field(default_factory=FallMambaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)

    # Modality toggles — disable modalities that lack pretrained weights or aren't needed
    use_video: bool = True
    use_audio: bool = True
    use_skeleton: bool = False   # no pretrained weights yet
    use_text: bool = False       # requires HF download, not needed for Stage 1

    # Paths
    model_weights_dir: str = "spwm/model_weights"
    datasets_dir: str = "datasets"
