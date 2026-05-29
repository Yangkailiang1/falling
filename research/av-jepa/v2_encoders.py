"""
AV-JEPA v2 Encoders
-------------------
Video: V-JEPA 2 ViT-L (frozen) — multi-token transformer features
Audio: WavJEPA (frozen) — multi-token waveform features

Shape summary:
  Video in: (B, 16, 3, 224, 224) → Video out: (B, 1568, 1024) [16/2 * 14 * 14]
  Audio in: (B, 3*16000=48000)   → Audio out: (B, 300, 768) [~50Hz * 3s]
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional
from pathlib import Path

# Local model weights root: research/av-jepa/ → research/ → project_root/ → spwm/model_weights/
_MODEL_WEIGHTS = Path(__file__).resolve().parent.parent.parent / "spwm" / "model_weights"


# ─── V-JEPA 2 Video Encoder ──────────────────────────────────────────────

class VJEPA2VideoEncoder(nn.Module):
    """
    V-JEPA 2 ViT-L as video encoder.
    Extracts multi-token (patch-level) features from video clips.

    Loads V-JEPA 2 source code + pretrained weights from spwm/model_weights/.
    """

    def __init__(
        self,
        model_name: str = "vit_large",
        checkpoint_path: str = "",
        img_size: int = 224,
        num_frames: int = 16,
        patch_size: int = 16,
        tubelet_size: int = 2,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.img_size = img_size

        # Build encoder from local V-JEPA 2 source via torch.hub (handles packaging)
        _vjepa_repo = str(_MODEL_WEIGHTS / "vjepa2")

        encoder, _predictor = torch.hub.load(
            _vjepa_repo,
            "vjepa2_vit_large",
            source="local",
            pretrained=False,
            skip_validation=True,
        )

        self.encoder = encoder

        # Load pretrained weights
        local_ckpt = _MODEL_WEIGHTS / "vitl.pt"
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        elif local_ckpt.exists():
            self._load_checkpoint(str(local_ckpt))
        else:
            self._load_from_torch_hub(model_name)

        # Freeze
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("target_encoder", ckpt.get("encoder", ckpt))
        state = {k.replace("module.", "").replace("backbone.", ""): v
                 for k, v in state.items()}
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        if missing:
            print(f"[VJEPA2] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[VJEPA2] Unexpected keys: {len(unexpected)}")
        else:
            print(f"[VJEPA2] Loaded weights from {path}")

    def _load_from_torch_hub(self, model_name: str):
        """Load from PyTorch Hub (internet fallback)."""
        mapping = {
            "vit_large": "vjepa2_vit_large",
            "vit_huge": "vjepa2_vit_huge",
            "vit_giant": "vjepa2_vit_giant",
            "vit_giant_384": "vjepa2_vit_giant_384",
        }
        hub_name = mapping.get(model_name, "vjepa2_vit_large")
        try:
            encoder, _ = torch.hub.load(
                "facebookresearch/vjepa2", hub_name,
                trust_repo=True, force_reload=False,
            )
            self.encoder.load_state_dict(encoder.state_dict())
            print(f"[VJEPA2] Loaded {hub_name} from torch.hub")
        except Exception as e:
            print(f"[VJEPA2] torch.hub failed ({e}), using random init")
            print("[VJEPA2] Download checkpoint from: "
                  "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt")

    @property
    def num_patches(self) -> int:
        """Number of output tokens per video clip."""
        t_tokens = self.num_frames // 2  # tubelet_size=2
        s_tokens = (self.img_size // 16) ** 2  # 14x14=196 @ 224
        return t_tokens * s_tokens

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W)  e.g., (B, 16, 3, 224, 224)

        Returns:
            features: (B, num_patches, embed_dim)  e.g., (B, 1568, 1024)
        """
        # V-JEPA 2 VisionTransformer expects (B, C, T, H, W)
        frames = frames.permute(0, 2, 1, 3, 4)  # (B,T,C,H,W) -> (B,C,T,H,W)
        # Normalize to ImageNet stats (V-JEPA 2 expects this)
        frames = (frames - 0.5) / 0.5
        return self.encoder(frames)


# ─── WavJEPA Audio Encoder ───────────────────────────────────────────────

class WavJEPAAudioEncoder(nn.Module):
    """
    WavJEPA as audio encoder.
    Extracts multi-token (time-frame-level) features from raw waveforms.

    Loads from local spwm/model_weights/wavjepa-base/ (fallback to HuggingFace).
    """

    def __init__(
        self,
        model_name: str = "",  # empty = use local default
        sample_rate: int = 16000,
        embed_dim: int = 768,
    ):
        super().__init__()
        from transformers import AutoModel

        # Use local model_weights/ if available, otherwise fall back to HF
        local_path = str(_MODEL_WEIGHTS / "wavjepa-base")
        if not model_name:
            if Path(local_path).exists():
                model_name = local_path
            else:
                model_name = "labhamlet/wavjepa-base"
                print("[WavJEPA] Local weights not found, using HuggingFace")

        print(f"[WavJEPA] Loading from: {model_name}")
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self.sample_rate = sample_rate
        self.embed_dim = embed_dim

        # Freeze
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (B, samples) raw waveform, 16kHz
                   e.g., (B, 48000) for 3 seconds

        Returns:
            features: (B, num_frames, embed_dim)
                      e.g., (B, ~300, 768) for 3s
        """
        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        output = self.model(audio)

        # WavJEPA returns various structures; extract last_hidden_state
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        elif isinstance(output, tuple):
            return output[0]
        elif isinstance(output, torch.Tensor):
            if output.dim() == 3:
                return output
            elif output.dim() == 2:
                return output.unsqueeze(1)  # (B, D) -> (B, 1, D)
        else:
            raise ValueError(f"Unexpected WavJEPA output: {type(output)}")


# ─── Query-based Token Pooling ───────────────────────────────────────────

class QueryTokenPooler(nn.Module):
    """
    Pool a sequence of tokens into a fixed number of learned query tokens
    via cross-attention. More expressive than simple mean pooling.

    Args:
        embed_dim: input token dimension
        num_queries: number of output tokens
        num_heads: attention heads
    """

    def __init__(self, embed_dim: int, num_queries: int = 8, num_heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) token sequence

        Returns:
            pooled: (B, num_queries, D)
        """
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)
        out, _ = self.cross_attn(q, x, x)
        return self.norm(out + q)


def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int = 16000) -> torch.Tensor:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    try:
        import torchaudio.functional as F
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        return F.resample(audio, orig_freq=orig_sr, new_freq=target_sr).squeeze(0)
    except ImportError:
        # Fallback: simple decimation/interpolation
        from scipy import signal as sp_signal
        audio_np = audio.cpu().numpy()
        num = int(len(audio_np) * target_sr / orig_sr)
        resampled = sp_signal.resample(audio_np, num)
        return torch.from_numpy(resampled).float()
