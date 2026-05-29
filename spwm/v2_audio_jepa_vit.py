"""
Audio-JEPA ViT Encoder

Builds a ViT-B backbone matching ltuncay/Audio-JEPA checkpoint and
loads pretrained weights. Used as a frozen audio feature extractor.

Architecture: Standard ViT with 16x16 patch embedding on mel spectrograms.
  Input:  (B, 1, 256, 128) mel spectrogram
  Output: (B, 128, 768) patch tokens (no CLS token)
"""

import torch
import torch.nn as nn
import math
from typing import OrderedDict


def build_audio_jepa_vit(state_dict: OrderedDict) -> nn.Module:
    """
    Build and load Audio-JEPA ViT from checkpoint state dict.

    The checkpoint contains:
      encoder.xxx     - ViT context/target encoder (EMA teacher)
      target_encoder.xxx - EMA target encoder (same structure)
      predictor.xxx   - JEPA predictor (mask token + position embeds)

    We use the 'encoder' weights for feature extraction.
    """
    sd_enc = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            sd_enc[k[len('encoder.'):]] = v

    # Infer dimensions from checkpoint
    embed_dim = sd_enc['patch_embed.proj.weight'].shape[0]  # 768
    num_patches = sd_enc['pos_embed'].shape[1]  # 128
    patch_size = sd_enc['patch_embed.proj.weight'].shape[2]  # 16

    # Count number of transformer blocks
    num_blocks = 0
    while f'blocks.{num_blocks}.norm1.weight' in sd_enc:
        num_blocks += 1

    # Infer number of heads from qkv weight
    qkv_dim = sd_enc['blocks.0.attn.qkv.weight'].shape[0]  # 2304 = 3 * 768
    num_heads = 12  # 768 / 64 = 12 (standard ViT-B)

    model = AudioJEPAViT(
        img_height=256, img_width=128,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        mlp_ratio=4.0,
    )

    # Load weights
    missing, unexpected = model.load_state_dict(sd_enc, strict=True)
    if missing:
        print(f"[AudioJEPA] Missing keys: {missing}")
    if unexpected:
        print(f"[AudioJEPA] Unexpected keys: {unexpected}")

    return model


class PatchEmbed(nn.Module):
    """2D patch embedding for spectrograms: Conv2d in_channels -> embed_dim."""

    def __init__(self, in_channels=1, embed_dim=768, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, D, H/16, W/16)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H/16*W/16, D)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim, num_heads=12, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    """Transformer block: pre-norm + attention + MLP."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.ModuleDict({
            'fc1': nn.Linear(dim, mlp_hidden),
            'act': nn.GELU(),
            'fc2': nn.Linear(mlp_hidden, dim),
        })

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp['fc2'](self.mlp['act'](self.mlp['fc1'](self.norm2(x))))
        return x


class AudioJEPAViT(nn.Module):
    """
    Audio-JEPA ViT encoder.

    Pure patch-based (no CLS token), matches the ltuncay/Audio-JEPA architecture.
    """

    def __init__(self, img_height=256, img_width=128, patch_size=16,
                 embed_dim=768, num_heads=12, num_blocks=12, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        num_patches = (img_height // patch_size) * (img_width // patch_size)

        self.patch_embed = PatchEmbed(in_channels=1, embed_dim=embed_dim, patch_size=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) mel spectrogram
        Returns:
            tokens: (B, num_patches, embed_dim)
        """
        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed[:, :x.shape[1], :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


# ── Quick test ──
if __name__ == '__main__':
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download('ltuncay/Audio-JEPA', 'JEPA.ckpt')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = build_audio_jepa_vit(ckpt['state_dict'])
    model.eval()
    print(f'Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')

    # Test forward
    dummy = torch.randn(2, 1, 256, 128)
    with torch.no_grad():
        out = model(dummy)
    print(f'Input:  {tuple(dummy.shape)}')
    print(f'Output: {tuple(out.shape)}')
    assert out.shape == (2, 128, 768), f'Expected (2, 128, 768), got {out.shape}'
    print('Test passed!')
