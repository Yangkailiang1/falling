"""
AV-JEPA Encoders
----------------
Video encoder: CLIP ViT (frozen, pretrained)
Audio encoder: CLAP (frozen, pretrained)

Both produce embeddings that are fused and fed into the JEPA predictor.
When PE-AV becomes available, these can be replaced with a single PE-AV call.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class VideoEncoder(nn.Module):
    """
    Video encoder using CLIP ViT.
    Encodes individual frames and pools them into a single video embedding.

    Args:
        model_name: HuggingFace CLIP model ID
        embed_dim: output embedding dimension
        num_frames: number of frames to expect per clip
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        embed_dim: int = 512,
        num_frames: int = 8,
    ):
        super().__init__()
        from transformers import CLIPVisionModel, CLIPImageProcessor

        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # Freeze the pretrained encoder
        for param in self.vision_model.parameters():
            param.requires_grad = False
        self.vision_model.eval()

        # Frame-level to video-level pooling: learnable temporal projection
        self.temporal_proj = nn.Sequential(
            nn.Linear(embed_dim * num_frames, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    @torch.no_grad()
    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of video frames.

        Args:
            frames: (B, T, C, H, W) tensor of video frames

        Returns:
            frame_embeddings: (B, T, embed_dim)
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)
        outputs = self.vision_model(pixel_values=frames_flat)
        pooler = outputs.pooler_output  # (B*T, embed_dim)
        return pooler.view(B, T, self.embed_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: frames → video embedding.

        Args:
            frames: (B, T, C, H, W)

        Returns:
            video_embed: (B, embed_dim)
        """
        frame_embeddings = self._encode_frames(frames)  # (B, T, D)
        B, T, D = frame_embeddings.shape
        flat = frame_embeddings.reshape(B, T * D)
        return self.temporal_proj(flat)  # (B, D)


class AudioEncoder(nn.Module):
    """
    Audio encoder using CLAP.

    Args:
        model_name: HuggingFace CLAP model ID
        embed_dim: output embedding dimension
    """

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        embed_dim: int = 512,
    ):
        super().__init__()
        from transformers import ClapModel, ClapProcessor

        self.clap_model = ClapModel.from_pretrained(model_name)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.embed_dim = embed_dim

        # Freeze the pretrained encoder
        for param in self.clap_model.parameters():
            param.requires_grad = False
        self.clap_model.eval()

        # Project to target dimension if different
        if embed_dim != self.clap_model.config.projection_dim:
            self.proj = nn.Linear(self.clap_model.config.projection_dim, embed_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: raw audio → audio embedding.

        Args:
            audio_input: (B, audio_length) float tensor (preprocessed by processor)
                         or a dict with input_features from ClapProcessor

        Returns:
            audio_embed: (B, embed_dim)
        """
        if isinstance(audio_input, dict):
            outputs = self.clap_model.get_audio_features(**audio_input)
        else:
            outputs = self.clap_model.get_audio_features(audio_input)
        return self.proj(outputs)


class AVEncoder(nn.Module):
    """
    Combined audio-visual encoder.
    Produces joint AV embeddings from video frames + audio.

    This is a stand-in for PE-AV; when PE-AV is available,
    replace this with a single PEAudioVisual call.
    """

    def __init__(self, config):
        super().__init__()
        self.video_encoder = VideoEncoder(
            model_name=config.clip_model,
            embed_dim=config.video_embed_dim,
            num_frames=config.video_frames,
        )
        self.audio_encoder = AudioEncoder(
            model_name=config.clap_model,
            embed_dim=config.audio_embed_dim,
        )
        self.joint_dim = config.joint_embed_dim
        self.video_dim = config.video_embed_dim
        self.audio_dim = config.audio_embed_dim

    def forward(
        self, frames: torch.Tensor, audio: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode video frames + audio into a joint embedding.

        Args:
            frames: (B, T, C, H, W) video frames
            audio: (B, audio_length) or dict from ClapProcessor

        Returns:
            joint_embed: (B, joint_embed_dim)
        """
        v_embed = self.video_encoder(frames)  # (B, video_dim)
        a_embed = self.audio_encoder(audio)    # (B, audio_dim)
        joint = torch.cat([v_embed, a_embed], dim=-1)  # (B, video_dim + audio_dim)
        return joint


def create_av_encoder(config) -> AVEncoder:
    """Factory function: create AV encoder from config."""
    return AVEncoder(config)
