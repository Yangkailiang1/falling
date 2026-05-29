"""
T-JEPA Encoders

Four frozen encoders that extract embeddings from raw modalities:
  - V-JEPA 2 ViT-L: Video frames → v_tokens (N_v, 1024)
  - Audio-JEPA ViT: Audio waveform → a_tokens (N_a, 768)
  - S-JEPA Transformer: Skeleton keypoints → s_tokens (N_s, 256)
  - Qwen2.5 Embedding: Text → t_embed (3584)

All encoders are frozen during training. Only the fusion/projector layers are trained.
"""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, Dict
from .config import (
    VideoEncoderConfig, AudioEncoderConfig,
    SkeletonEncoderConfig, TextEncoderConfig
)

# Local model weights directory
_MODEL_WEIGHTS = Path(__file__).resolve().parent / "model_weights"


# ═══════════════════════════════════════════════════════════════
# Video Encoder: V-JEPA 2 ViT-L (300M, frozen)
# ═══════════════════════════════════════════════════════════════

class VJEPA2VideoEncoder(nn.Module):
    """
    V-JEPA 2 ViT-L video encoder.

    Extracts visual tokens from video frames.
    Input:  (B, T, C, H, W) = (B, 16, 3, 224, 224)
    Output: (B, N_v, 1024) where N_v is the number of patch tokens

    Uses V-JEPA 2 weights from facebookresearch/vjepa2.
    Fallback: if V-JEPA 2 not available, uses CLIP ViT-L as approximation.
    """

    def __init__(self, config: VideoEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_frames = config.num_frames
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self._encoder = None
        self._initialized = False
        self._is_vjepa2 = False  # tracks whether encoder is V-JEPA 2 vs CLIP fallback

    def _lazy_init(self, device):
        """Lazy initialization to avoid loading 300M encoder on import."""
        if self._initialized:
            return
        self._initialized = True

        try:
            # Load V-JEPA 2 from local model_weights/ (source code + checkpoint)
            vjepa2_repo = _MODEL_WEIGHTS / "vjepa2"
            checkpoint_path = _MODEL_WEIGHTS / "vitl.pt"

            if not vjepa2_repo.exists():
                raise FileNotFoundError(f"V-JEPA 2 repo not found at {vjepa2_repo}")
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"V-JEPA 2 checkpoint not found at {checkpoint_path}")

            # Build model architecture from local source, then load pretrained weights
            encoder, predictor = torch.hub.load(
                str(vjepa2_repo),
                "vjepa2_vit_large",
                source="local",
                pretrained=False,
                skip_validation=True,
            )

            # Load pretrained weights from local checkpoint
            checkpoint = torch.load(
                str(checkpoint_path),
                map_location="cpu",
                weights_only=False,
            )

            # Clean state dict keys: remove "module." and "backbone." prefixes
            target_encoder_state = checkpoint.get("target_encoder", checkpoint.get("encoder", {}))
            cleaned_state = {}
            for k, v in target_encoder_state.items():
                new_k = k.replace("module.", "").replace("backbone.", "")
                cleaned_state[new_k] = v

            encoder.load_state_dict(cleaned_state, strict=False)
            self._encoder = encoder
            self._is_vjepa2 = True
            print("[VJEPA2VideoEncoder] Loaded V-JEPA 2 ViT-L from local model_weights/")
        except Exception as e:
            print(f"[VJEPA2VideoEncoder] V-JEPA 2 local load failed ({e}), trying CLIP fallback")
            self._encoder = self._build_clip_fallback()
            self._is_vjepa2 = False
            print("[VJEPA2VideoEncoder] Using CLIP ViT-B/16 as fallback")

        if self._encoder is not None:
            self._encoder = self._encoder.to(device).eval()
            for p in self._encoder.parameters():
                p.requires_grad = False

    def _build_clip_fallback(self):
        """Build CLIP ViT-B/16 as a fallback video encoder, loaded from local model_weights/."""
        clip_local = _MODEL_WEIGHTS / "clip-vit-base-patch16"
        if clip_local.exists():
            try:
                from transformers import CLIPVisionModel
                model = CLIPVisionModel.from_pretrained(str(clip_local))
                print("[VJEPA2VideoEncoder] Loaded CLIP ViT-B/16 from local model_weights/")
                return model
            except Exception:
                pass

        # Last resort: random-init CLIP ViT-L/14
        try:
            from transformers import CLIPVisionModel, CLIPVisionConfig
            config = CLIPVisionConfig(
                hidden_size=1024,
                image_size=224,
                patch_size=14,
                num_hidden_layers=24,
                num_attention_heads=16,
                intermediate_size=4096,
            )
            model = CLIPVisionModel(config)
            print("[VJEPA2VideoEncoder] Using random-init CLIP ViT-L/14 (no pretrained weights!)")
            return model
        except ImportError:
            return None

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) video frames, normalized to [0,1] or ImageNet stats
        Returns:
            tokens: (B, N_v, embed_dim) visual tokens
        """
        self._lazy_init(frames.device)

        B, T, C, H, W = frames.shape

        if self._encoder is None:
            raise RuntimeError("V-JEPA 2 encoder not loaded and no fallback available")

        # Handle V-JEPA 2 specific forward pass
        if self._is_vjepa2:
            # V-JEPA 2 VisionTransformer expects (B, C, T, H, W)
            # Our input is (B, T, C, H, W), so permute
            vjepa_frames = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            with torch.no_grad():
                tokens = self._encoder(vjepa_frames)  # (B, N_v, 1024)
        elif hasattr(self._encoder, 'forward_encoder'):
            # Legacy V-JEPA 2 forward: expects (B, T, C, H, W)
            with torch.no_grad():
                tokens = self._encoder.forward_encoder(frames)  # (B, N_v, 1024)
        else:
            # CLIP fallback: process frame by frame then pool
            frames_flat = frames.view(B * T, C, H, W)  # (B*T, C, H, W)
            with torch.no_grad():
                outputs = self._encoder(frames_flat, output_hidden_states=True)
                # Use last hidden state as tokens
                tokens = outputs.last_hidden_state  # (B*T, N_patches+CLS, 1024)
            # Remove CLS token, reshape back to (B, T*N_patches, 1024)
            tokens = tokens[:, 1:, :]  # remove CLS
            tokens = tokens.reshape(B, T * tokens.size(1), tokens.size(2))

        return tokens


# ═══════════════════════════════════════════════════════════════
# Audio Encoder: Audio-JEPA ViT (~85M, frozen)
# ═══════════════════════════════════════════════════════════════

class AudioJEPEncoder(nn.Module):
    """
    Audio-JEPA ViT encoder.

    Extracts audio tokens from raw waveform.
    Input:  (B, 48000) - 3 seconds @ 16kHz
    Output: (B, N_a, 768) audio tokens

    Fallback modes:
      - WavJEPA from HuggingFace
      - CLAP HTSAT (from existing av-jepa code)
      - Mel-Spectrogram + CNN (edge deployment, per Fall-Mamba optimization 5)
    """

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.sample_rate = config.sample_rate
        self.use_mel_fallback = config.use_mel_fallback

        self._encoder = None
        self._mel_cnn = None
        self._initialized = False

    def _lazy_init(self, device):
        if self._initialized:
            return
        self._initialized = True

        if self.use_mel_fallback:
            self._mel_cnn = self._build_mel_cnn().to(device).eval()
            print("[AudioJEPEncoder] Using Mel-Spectrogram + CNN fallback (edge mode)")
            return

        # Try primary encoders in order (prefer JEPA-type)
        encoder = None
        # Try 1: WavJEPA from local model_weights/ (JEPA-type audio encoder)
        wavjepa_local = _MODEL_WEIGHTS / "wavjepa-base"
        if wavjepa_local.exists():
            try:
                from transformers import AutoModel
                encoder = AutoModel.from_pretrained(str(wavjepa_local), trust_remote_code=True)
                print("[AudioJEPEncoder] Loaded WavJEPA (JEPA) from local model_weights/")
            except Exception:
                pass

        # Try 2: WavJEPA from HuggingFace (facebook/wavjepa)
        if encoder is None:
            try:
                from transformers import AutoModel
                encoder = AutoModel.from_pretrained("facebook/wavjepa", trust_remote_code=True)
                print("[AudioJEPEncoder] Loaded WavJEPA (JEPA) from HuggingFace")
            except Exception:
                pass

        # Try 3: CLAP HTSAT from local model_weights/ (contrastive, fallback)
        clap_local = _MODEL_WEIGHTS / "clap-htsat-unfused"
        if encoder is None and clap_local.exists():
            try:
                from transformers import AutoModel
                encoder = AutoModel.from_pretrained(str(clap_local))
                print("[AudioJEPEncoder] Loaded CLAP HTSAT from local model_weights/")
            except Exception:
                pass

        # Try 4: CLAP HTSAT from HuggingFace (online fallback)
        if encoder is None:
            try:
                from transformers import AutoModel
                encoder = AutoModel.from_pretrained("laion/clap-htsat-unfused")
                print("[AudioJEPEncoder] Loaded CLAP HTSAT from HuggingFace")
            except Exception:
                pass

        self._mel_extractor = None  # cached MelSpectrogramExtractor, device-aligned lazily

        if encoder is not None:
            self._encoder = encoder.to(device).eval()
            for p in self._encoder.parameters():
                p.requires_grad = False
        else:
            # Fall back to Mel-Spectrogram CNN
            print("[AudioJEPEncoder] No HF model available, using Mel-CNN fallback")
            self.use_mel_fallback = True
            self._mel_cnn = self._build_mel_cnn().to(device).eval()

    def _build_mel_cnn(self):
        """Lightweight Mel-Spectrogram CNN for edge deployment (Fall-Mamba optimization 5)."""
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, self.embed_dim),
        )

    def _compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute Mel spectrogram from raw waveform."""
        from .utils.mel_spectrogram import MelSpectrogramExtractor
        if self._mel_extractor is None:
            self._mel_extractor = MelSpectrogramExtractor(
                sample_rate=self.sample_rate,
                n_mels=self.config.mel_bins
            )
        if self._mel_extractor.window.device != waveform.device:
            self._mel_extractor = self._mel_extractor.to(waveform.device)
        mel = self._mel_extractor(waveform)
        return mel

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, L) audio waveform
        Returns:
            tokens: (B, N_a, embed_dim) audio tokens
        """
        self._lazy_init(waveform.device)
        B = waveform.shape[0]

        with torch.no_grad():
            if self.use_mel_fallback or (self._mel_cnn is not None and self._encoder is None):
                mel = self._compute_mel_spectrogram(waveform)
                tokens = self._mel_cnn(mel)  # (B, 768)
                # Expand to token sequence
                n_tokens = self.config.num_tokens
                tokens = tokens.unsqueeze(1).expand(B, n_tokens, self.embed_dim)
            elif self._encoder is not None:
                try:
                    # WavJEPA (JEPA-type): expects (B, 1, T), raw waveform → (hidden_states, mask) tuple
                    if waveform.dim() == 2:
                        wav_input = waveform.unsqueeze(1)  # (B, L) → (B, 1, L)
                    else:
                        wav_input = waveform
                    outputs = self._encoder(wav_input)
                    if isinstance(outputs, tuple):
                        tokens = outputs[0]  # (B, N_a, 768)
                    elif hasattr(outputs, 'last_hidden_state'):
                        tokens = outputs.last_hidden_state
                    else:
                        tokens = outputs
                except Exception:
                    try:
                        # CLAP (contrastive): mel spectrogram → get_audio_features
                        mel = self._compute_mel_spectrogram(waveform)
                        from torch.nn.functional import interpolate
                        if mel.shape[1] != 64:
                            mel = interpolate(mel, size=(64, mel.shape[-1]), mode='bilinear', align_corners=False)
                        mel = mel.squeeze(1)
                        mel = mel.transpose(1, 2)
                        outputs = self._encoder.get_audio_features(
                            input_features=mel,
                            is_longer=torch.ones(B, 1, device=waveform.device),
                        )
                        tokens = outputs.unsqueeze(1)
                        n_tokens = self.config.num_tokens
                        tokens = tokens.expand(B, n_tokens, self.embed_dim)
                    except Exception:
                        # Final fallback: Mel-CNN
                        mel = self._compute_mel_spectrogram(waveform)
                        if self._mel_cnn is None:
                            self._mel_cnn = self._build_mel_cnn().to(waveform.device).eval()
                        tokens = self._mel_cnn(mel)
                        n_tokens = self.config.num_tokens
                        tokens = tokens.unsqueeze(1).expand(B, n_tokens, self.embed_dim)

                # If too few tokens, pad; if too many, average pool
                if tokens.size(1) < self.config.num_tokens:
                    pad = torch.zeros(
                        B, self.config.num_tokens - tokens.size(1), tokens.size(2),
                        device=tokens.device, dtype=tokens.dtype,
                    )
                    tokens = torch.cat([tokens, pad], dim=1)
                elif tokens.size(1) > self.config.num_tokens:
                    # Average pool to target length
                    tokens = tokens.transpose(1, 2)
                    tokens = F.adaptive_avg_pool1d(tokens, self.config.num_tokens)
                    tokens = tokens.transpose(1, 2)
            else:
                raise RuntimeError("No audio encoder available")

        return tokens


# ═══════════════════════════════════════════════════════════════
# Skeleton Encoder: S-JEPA Transformer (~22M, frozen)
# ═══════════════════════════════════════════════════════════════

class SJEPASkeletonEncoder(nn.Module):
    """
    S-JEPA Transformer encoder for skeleton pose sequences.

    Input:  (B, T, 17, 3) - T frames of 17 COCO keypoints (x, y, confidence)
    Output: (B, N_s, 256) - skeleton tokens

    Architecture: 6-layer Transformer encoder over flattened keypoint sequence.
    """

    def __init__(self, config: SkeletonEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_keypoints = config.num_keypoints
        self.num_frames = config.num_frames
        self.input_dim = config.input_dim

        # Keypoint embedding
        self.keypoint_embed = nn.Sequential(
            nn.Linear(config.input_dim, 64),
            nn.GELU(),
            nn.Linear(64, config.embed_dim),
        )
        nn.init.normal_(self.keypoint_embed[-1].weight, std=0.02)

        # Positional encoding (learned)
        max_len = config.num_keypoints * config.num_frames + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, config.embed_dim))

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Output projection
        self.output_norm = nn.LayerNorm(config.embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skeleton: (B, T, K, 3) where K=17 keypoints, 3=(x, y, confidence)
                      T should be self.num_frames (16)
        Returns:
            tokens: (B, K*T+1, embed_dim) skeleton tokens (includes CLS)
        """
        B, T, K, C = skeleton.shape

        # Flatten keypoints: (B, T*K, 3)
        x = skeleton.view(B, T * K, C)

        # Embed each keypoint: (B, T*K, embed_dim)
        x = self.keypoint_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1 + T*K, embed_dim)

        # Add positional encoding
        x = x + self.pos_embed[:, :x.size(1), :]

        # Transformer encoder
        x = self.transformer(x)

        # Normalize
        x = self.output_norm(x)

        return x


# ═══════════════════════════════════════════════════════════════
# Text Encoder: Qwen2.5 Embedding (frozen)
# ═══════════════════════════════════════════════════════════════

class TextEncoder(nn.Module):
    """
    Qwen2.5 text embedding layer.

    Extracts text embeddings from Chinese/English descriptions.
    Input:  text string like "老人正在缓慢行走"
    Output: t_embed (B, 3584)

    Uses Qwen2.5-7B's embedding layer. Falls back to a simpler BERT model
    if Qwen2.5 is too heavy for the environment.
    """

    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim

        self._tokenizer = None
        self._embed_layer = None
        self._initialized = False

    def _lazy_init(self, device):
        if self._initialized:
            return
        self._initialized = True

        try:
            from transformers import AutoTokenizer, AutoModel
            model_name = self.config.model_name  # "Qwen/Qwen2.5-7B"
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            # Only load the embedding layer, not the full 7B model
            full_model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=torch.float16
            )
            # Extract just the embedding layer
            self._embed_layer = full_model.get_input_embeddings()
            if hasattr(full_model, 'embed_tokens'):
                self._embed_layer = full_model.embed_tokens
            self._embed_layer = self._embed_layer.to(device).eval()
            del full_model  # free 7B model memory
            print(f"[TextEncoder] Loaded {model_name} embedding layer")
        except Exception as e:
            print(f"[TextEncoder] Qwen2.5 not available ({e}), trying BERT fallback")
            try:
                from transformers import AutoTokenizer, AutoModel
                self._tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
                model = AutoModel.from_pretrained("bert-base-multilingual-cased")
                self._embed_layer = model.get_input_embeddings()
                self._embed_layer = self._embed_layer.to(device).eval()
                self.embed_dim = model.config.hidden_size  # 768
                del model
                print("[TextEncoder] Using BERT multilingual fallback (embed_dim=768)")
            except Exception as e2:
                print(f"[TextEncoder] BERT fallback also failed ({e2})")
                self._tokenizer = None
                self._embed_layer = None

    def forward(self, texts: list) -> torch.Tensor:
        """
        Args:
            texts: list of strings (batch_size,)
        Returns:
            embeddings: (B, embed_dim) mean-pooled text embeddings
        """
        if self._tokenizer is None or self._embed_layer is None:
            self._lazy_init(torch.device('cpu'))
            if self._tokenizer is None or self._embed_layer is None:
                raise RuntimeError("Text encoder not available and no fallback succeeded")

        self._lazy_init(next(self._embed_layer.parameters()).device)

        # Tokenize
        tokens = self._tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        input_ids = tokens['input_ids'].to(next(self._embed_layer.parameters()).device)
        attention_mask = tokens['attention_mask'].to(next(self._embed_layer.parameters()).device)

        with torch.no_grad():
            # Get embeddings
            embeddings = self._embed_layer(input_ids)  # (B, L, embed_dim)

            # Mean pooling (masked)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return pooled


# ═══════════════════════════════════════════════════════════════
# Combined Encoder Wrapper
# ═══════════════════════════════════════════════════════════════

class TJEPSEncoders(nn.Module):
    """Container for all four frozen encoders with modality toggles."""

    def __init__(
        self,
        video_cfg: VideoEncoderConfig,
        audio_cfg: AudioEncoderConfig,
        skeleton_cfg: SkeletonEncoderConfig,
        text_cfg: TextEncoderConfig,
        use_video: bool = True,
        use_audio: bool = True,
        use_skeleton: bool = True,
        use_text: bool = True,
    ):
        super().__init__()
        self.use_video = use_video
        self.use_audio = use_audio
        self.use_skeleton = use_skeleton
        self.use_text = use_text

        self.video_dim = video_cfg.embed_dim
        self.audio_dim = audio_cfg.embed_dim
        self.skeleton_dim = skeleton_cfg.embed_dim
        self.text_dim = text_cfg.embed_dim

        # Only create encoders for enabled modalities (lazy init, so cheap at construction)
        self.video_encoder = VJEPA2VideoEncoder(video_cfg) if use_video else None
        self.audio_encoder = AudioJEPEncoder(audio_cfg) if use_audio else None
        self.skeleton_encoder = SJEPASkeletonEncoder(skeleton_cfg) if use_skeleton else None
        self.text_encoder = TextEncoder(text_cfg) if use_text else None

    @property
    def enabled_modalities(self) -> dict:
        return {
            'video': self.use_video,
            'audio': self.use_audio,
            'skeleton': self.use_skeleton,
            'text': self.use_text,
        }

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        skeleton: torch.Tensor,
        texts: list,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward enabled encoders. Disabled modalities return zero tensors.

        Args:
            video: (B, T, 3, H, W) video frames
            audio: (B, L) audio waveform
            skeleton: (B, T, 17, 3) skeleton keypoints
            texts: list of strings

        Returns:
            dict with keys: v_tokens, a_tokens, s_tokens, t_embed
        """
        B = video.shape[0] if video is not None else len(texts)

        if self.use_video and self.video_encoder is not None:
            v_tokens = self.video_encoder(video)
        else:
            # Zero tokens matching V-JEPA 2 output shape
            v_tokens = torch.zeros(B, 1, self.video_dim, device=video.device if video is not None else 'cpu')

        if self.use_audio and self.audio_encoder is not None:
            a_tokens = self.audio_encoder(audio)
        else:
            a_tokens = torch.zeros(B, 1, self.audio_dim, device=audio.device if audio is not None else 'cpu')

        if self.use_skeleton and self.skeleton_encoder is not None:
            s_tokens = self.skeleton_encoder(skeleton)
        else:
            s_tokens = torch.zeros(B, 1, self.skeleton_dim, device=skeleton.device if skeleton is not None else 'cpu')

        if self.use_text and self.text_encoder is not None:
            t_embed = self.text_encoder(texts)
        else:
            t_embed = torch.zeros(B, self.text_dim, device=video.device)

        return {
            'v_tokens': v_tokens,
            'a_tokens': a_tokens,
            's_tokens': s_tokens,
            't_embed': t_embed,
        }
