#!/usr/bin/env python3
"""
M3-JEPA Video+Audio Fusion Test on Le2i Dataset

Implements the M3-JEPA Multi-gate MoE approach from the paper:
  "M3-JEPA: Multimodal Alignment via Multi-gate MoE based on JEPA"

Architecture (simplified for Video+Audio):
  1. Frozen encoders: CLIP ViT-B/16 (video) + CLAP HTSAT (audio)
  2. M3-JEPA MoE Fusion: modality-specific gates + shared experts
  3. JEPA Predictor: 3-layer Transformer context → target
  4. Loss = α * L2_reg + (1-α) * InfoNCE + λ * SIGReg

Training: only normal activity clips (unsupervised)
Detection: falls via prediction error (surprise)

Usage:
  python spwm/test_m3jepa_le2i.py --data_root datasets/Le2i --device cuda --epochs 10
"""

import os
import sys
import time
import math
import random
import argparse
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

class Config:
    video_encoder = "openai/clip-vit-base-patch16"
    audio_encoder = "laion/clap-htsat-unfused"
    video_dim = 512   # CLIP ViT-B
    audio_dim = 512   # CLAP HTSAT

    num_context_frames = 8
    num_target_frames = 8
    frame_size = 224
    audio_sample_rate = 16000
    audio_duration = 1.0  # seconds per clip half

    # M3-JEPA MoE
    unified_dim = 512
    num_experts = 4
    num_modalities = 2
    top_k = 2

    # Loss weights (M3-JEPA: α=0.5 is optimal)
    alpha_reg = 0.5       # weight for L2 regularization
    alpha_cl = 0.5        # weight for contrastive
    sigreg_weight = 0.1

    # Training
    batch_size = 2
    epochs = 20
    lr = 1e-4
    device = "cuda"
    stride = 8
    normal_only = True    # train only on normal activities
    grad_clip = 1.0
    mixed_precision = False  # disabled on small GPU

    # Paths
    data_root = "datasets/Le2i"
    checkpoint_dir = "checkpoints"

cfg = Config()


# ═══════════════════════════════════════════════════════════
# M3-JEPA MoE Fusion (from paper Section 3.3)
# ═══════════════════════════════════════════════════════════

class M3MoE(nn.Module):
    """
    M3-JEPA Multi-gate Mixture of Experts (MMoE) Predictor.

    Follows Equation 12-13 from the paper:
      - M modalities × N experts per modality = M*N total experts
      - Gate: G = softmax(g · [e₁ ⊕ e₂])
        where e_m is the modality-specific embedding
      - Two parallel gates (L=2): one for L_reg, one for L_cl
      - Top-K sparsification for efficiency

    The gating function automatically disentangles:
      - Modality-specific paths (expert EM_n + gating embedding e_m)
      - Shared representation (projection matrix g creates common subspace)
    """

    def __init__(self, input_dims: List[int], unified_dim: int,
                 num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.unified_dim = unified_dim
        self.num_experts = num_experts
        self.num_modalities = len(input_dims)
        self.top_k = top_k
        self.num_total_experts = self.num_modalities * num_experts

        # Per-modality projection to unified dim
        self.modality_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, unified_dim),
                nn.LayerNorm(unified_dim),
                nn.GELU(),
            )
            for d in input_dims
        ])

        # Modality-specific gating embeddings (e_m in paper Eq.13)
        self.gate_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(1, unified_dim) * 0.02)
            for _ in range(self.num_modalities)
        ])

        # Shared gate projection matrix (g in Eq.13)
        # Input: concat of all modality projections + gate embeddings
        gate_input_dim = unified_dim * self.num_modalities + unified_dim
        self.gate_proj_reg = nn.Linear(gate_input_dim, self.num_total_experts)
        self.gate_proj_cl = nn.Linear(gate_input_dim, self.num_total_experts)

        # Experts: M*N feed-forward networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(unified_dim + unified_dim, unified_dim * 2),  # concat input pair
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(unified_dim * 2, unified_dim),
            )
            for _ in range(self.num_total_experts)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(unified_dim),
            nn.Linear(unified_dim, unified_dim),
        )

    def forward(self, modality_embeds: List[torch.Tensor]) -> Dict:
        """
        Args:
            modality_embeds: list of (B, D_i) tensors, one per modality

        Returns:
            dict with z_fused, z_reg, z_cl, gate_weights
        """
        B = modality_embeds[0].shape[0]

        # Step 1: Project each modality to unified dim
        projected = []
        for i, embed in enumerate(modality_embeds):
            if embed.dim() > 2:
                embed = embed.mean(dim=1)  # pool tokens
            p = self.modality_proj[i](embed)  # (B, unified_dim)
            projected.append(p)

        # Step 2: Build gating input (Eq.13: G = softmax(g · [e₁⊕e₂⊕... ⊕ gate_embs]))
        # Concatenate modality projections + modality-specific gate embeddings
        gate_cat = projected + [self.gate_embeddings[i].expand(B, -1) for i in range(self.num_modalities)]
        gate_input = torch.cat(gate_cat, dim=-1)  # (B, unified_dim * 2M)

        # Two parallel gates: one for L_reg, one for L_cl
        gate_reg = F.softmax(self.gate_proj_reg(gate_input), dim=-1)  # (B, N_total)
        gate_cl = F.softmax(self.gate_proj_cl(gate_input), dim=-1)    # (B, N_total)

        # Step 3: Expert computation with Top-K
        # Each expert takes ALL modality projections as input
        expert_input = torch.cat(projected, dim=-1)  # (B, unified_dim * M)

        # L_reg pathway
        if self.top_k < self.num_total_experts:
            topk_reg_vals, topk_reg_idx = gate_reg.topk(self.top_k, dim=-1)
            topk_reg_gates = F.softmax(topk_reg_vals, dim=-1)
        else:
            topk_reg_idx = torch.arange(self.num_total_experts, device=gate_reg.device).expand(B, -1)
            topk_reg_gates = gate_reg

        z_reg = torch.zeros(B, self.unified_dim, device=expert_input.device)
        for k in range(min(self.top_k, self.num_total_experts)):
            expert_idx = topk_reg_idx[:, k]  # (B,)
            for b in range(B):
                e_out = self.experts[expert_idx[b]](expert_input[b:b+1])
                z_reg[b] += topk_reg_gates[b, k] * e_out.squeeze(0)

        # L_cl pathway
        if self.top_k < self.num_total_experts:
            topk_cl_vals, topk_cl_idx = gate_cl.topk(self.top_k, dim=-1)
            topk_cl_gates = F.softmax(topk_cl_vals, dim=-1)
        else:
            topk_cl_idx = torch.arange(self.num_total_experts, device=gate_cl.device).expand(B, -1)
            topk_cl_gates = gate_cl

        z_cl = torch.zeros(B, self.unified_dim, device=expert_input.device)
        for k in range(min(self.top_k, self.num_total_experts)):
            expert_idx = topk_cl_idx[:, k]
            for b in range(B):
                e_out = self.experts[expert_idx[b]](expert_input[b:b+1])
                z_cl[b] += topk_cl_gates[b, k] * e_out.squeeze(0)

        z_reg = self.output_proj(z_reg)
        z_cl = self.output_proj(z_cl)

        # Final fused: average of both pathways
        z_fused = (z_reg + z_cl) / 2.0

        return {
            'z_fused': z_fused,
            'z_reg': z_reg,
            'z_cl': z_cl,
            'gate_reg': gate_reg.detach(),
            'gate_cl': gate_cl.detach(),
        }


# ═══════════════════════════════════════════════════════════
# JEPA Predictor (simple 3-layer Transformer)
# ═══════════════════════════════════════════════════════════

class JEPAPredictor(nn.Module):
    """Simple 3-layer Transformer predictor: z_fused → z_future"""

    def __init__(self, dim: int = 512, n_layers: int = 3, n_heads: int = 8):
        super().__init__()
        self.query_token = nn.Parameter(torch.zeros(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.query_token, std=0.02)

    def forward(self, z_fused: torch.Tensor) -> torch.Tensor:
        B = z_fused.shape[0]
        query = self.query_token.expand(B, -1, -1)
        context = z_fused.unsqueeze(1)
        x = torch.cat([query, context], dim=1)
        x = self.transformer(x)
        z_pred = x[:, 0, :]
        return self.out_proj(self.out_norm(z_pred))


# ═══════════════════════════════════════════════════════════
# M3-JEPA Model (Video + Audio)
# ═══════════════════════════════════════════════════════════

class M3JEPAModel(nn.Module):
    """M3-JEPA: Video + Audio fusion via Multi-gate MoE."""

    def __init__(self):
        super().__init__()
        dim = cfg.unified_dim

        # Frozen encoders
        self.video_encoder = None   # lazy init
        self.audio_encoder = None   # lazy init
        self._encoders_loaded = False

        # Trainable: M3-JEPA MoE fusion
        self.moe_fusion = M3MoE(
            input_dims=[cfg.video_dim, cfg.audio_dim],
            unified_dim=dim,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
        )

        # Trainable: JEPA predictor
        self.predictor = JEPAPredictor(dim=dim)

        # SIGReg
        self.sigreg = SIGRegLoss()

        self._init_encoders()

    def _init_encoders(self):
        if self._encoders_loaded:
            return
        self._encoders_loaded = True

        # Use offline mode to skip HuggingFace network requests
        import os
        os.environ.setdefault('HF_HUB_OFFLINE', '1')

        # Video: CLIP ViT-B/16
        try:
            from transformers import CLIPVisionModel
            self.video_encoder = CLIPVisionModel.from_pretrained(
                cfg.video_encoder, local_files_only=True
            )
            for p in self.video_encoder.parameters():
                p.requires_grad = False
            print(f"[M3-JEPA] Video encoder: CLIP ViT-B/16 (loaded from cache)")
        except Exception as e:
            print(f"[M3-JEPA] Video encoder failed: {e}")

        # Audio: CLAP HTSAT
        try:
            from transformers import AutoModel
            self.audio_encoder = AutoModel.from_pretrained(
                cfg.audio_encoder, local_files_only=True
            )
            for p in self.audio_encoder.parameters():
                p.requires_grad = False
            print(f"[M3-JEPA] Audio encoder: CLAP HTSAT (loaded from cache)")
        except Exception as e:
            print(f"[M3-JEPA] Audio encoder failed: {e}, using Mel-CNN")
            self.audio_encoder = MelCNNEncoder(cfg.audio_dim)

    def _encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode video frames → pooled embedding (B, 512)."""
        if self.video_encoder is None:
            return torch.randn(frames.shape[0], cfg.video_dim, device=frames.device)

        with torch.no_grad():
            B, T, C, H, W = frames.shape
            frames_flat = frames.view(B * T, C, H, W)
            outputs = self.video_encoder(frames_flat)
            v_embed = outputs.pooler_output  # (B*T, 512)
            v_embed = v_embed.view(B, T, -1).mean(dim=1)  # (B, 512)
        return v_embed

    def _encode_audio(self, waveform: torch.Tensor, orig_sr: int = 48000) -> torch.Tensor:
        """Encode audio → pooled embedding (B, 512)."""
        if self.audio_encoder is None:
            return torch.randn(waveform.shape[0], cfg.audio_dim, device=waveform.device)

        with torch.no_grad():
            if isinstance(self.audio_encoder, nn.Module) and not hasattr(self.audio_encoder, 'config'):
                # MelCNN fallback
                mel = compute_mel(waveform, orig_sr)
                a_embed = self.audio_encoder(mel)
            else:
                # CLAP: expects mel spectrogram input
                mel = compute_mel(waveform, orig_sr, n_mels=64)
                mel = mel.squeeze(1).transpose(1, 2)  # (B, T, 64)
                outputs = self.audio_encoder.get_audio_features(
                    input_features=mel,
                    is_longer=torch.ones(waveform.shape[0], 1, device=waveform.device),
                )
                a_embed = outputs
        return a_embed

    def forward(self, batch: Dict, return_loss: bool = True) -> Dict:
        """
        Forward pass for M3-JEPA.

        args:
            batch: dict with ctx_frames, tgt_frames, ctx_audio, tgt_audio
        returns:
            dict with z_fused, z_pred, z_target, loss, ...
        """
        ctx_frames = batch['ctx_frames']
        tgt_frames = batch['tgt_frames']
        ctx_audio = batch['ctx_audio']
        tgt_audio = batch['tgt_audio']

        B = ctx_frames.shape[0]
        device = ctx_frames.device

        # ━━ Encode context ━━
        v_ctx = self._encode_video(ctx_frames)  # (B, 512)
        a_ctx = self._encode_audio(ctx_audio, orig_sr=cfg.audio_sample_rate)  # (B, 512)

        # ━━━ M3-JEPA MoE Fusion ━━━
        moe_out = self.moe_fusion([v_ctx, a_ctx])
        z_fused = moe_out['z_fused']  # (B, 512)

        # ━━━ JEPA Predict ━━━
        z_pred = self.predictor(z_fused)  # (B, 512)

        result = {
            'z_fused': z_fused,
            'z_pred': z_pred,
            'gate_reg': moe_out['gate_reg'],
            'gate_cl': moe_out['gate_cl'],
        }

        # ━━━ Encode target (for loss) ━━━
        if return_loss:
            v_tgt = self._encode_video(tgt_frames)
            a_tgt = self._encode_audio(tgt_audio, orig_sr=cfg.audio_sample_rate)

            # Fuse target modalities (simple concat → linear)
            z_target = torch.cat([v_tgt, a_tgt], dim=-1)
            z_target = F.normalize(
                nn.Linear(cfg.video_dim + cfg.audio_dim, cfg.unified_dim,
                          device=device, dtype=z_target.dtype)(z_target),
                dim=-1
            )
            z_target_detached = z_target.detach()

            # ━━━ M3-JEPA Losses ━━━
            # L_reg: L2 distance (Eq.9)
            loss_reg = F.mse_loss(z_pred, z_target_detached)

            # L_cl: InfoNCE contrastive (Eq.10)
            z_pred_norm = F.normalize(z_pred, dim=-1)
            z_target_norm = F.normalize(z_target_detached, dim=-1)
            logits = z_pred_norm @ z_target_norm.T / 0.07  # τ=0.07
            labels = torch.arange(B, device=device)
            loss_cl = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

            # SIGReg anti-collapse
            loss_sigreg = self.sigreg(z_fused) + self.sigreg(z_pred)

            # Total (Eq.11: α * L_reg + (1-α) * L_cl)
            loss = cfg.alpha_reg * loss_reg + cfg.alpha_cl * loss_cl + cfg.sigreg_weight * loss_sigreg

            result['loss'] = loss
            result['loss_reg'] = loss_reg
            result['loss_cl'] = loss_cl
            result['loss_sigreg'] = loss_sigreg
            result['z_target'] = z_target_detached

        return result

    def anomaly_score(self, z_pred: torch.Tensor, normal_stats: Dict) -> torch.Tensor:
        """Compute anomaly score: Mahalanobis distance from normal distribution."""
        centered = z_pred - normal_stats['mean']
        score = torch.sqrt((centered * normal_stats['inv_var'] * centered).sum(dim=-1) + 1e-8)
        return score / (normal_stats['std_mean'] + 1e-8)


class MelCNNEncoder(nn.Module):
    """Lightweight Mel-Spectrogram CNN encoder for audio fallback."""
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.net(mel)


class SIGRegLoss(nn.Module):
    """SIGReg: covariance regularization to prevent collapse."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, D = z.shape
        if B < 2:
            return torch.tensor(0.0, device=z.device)
        z_c = z - z.mean(dim=0, keepdim=True)
        cov = (z_c.T @ z_c) / (B - 1)
        mask = torch.eye(D, device=z.device)
        off_diag = (cov * (1 - mask)).pow(2).sum()
        diag = (cov.diag() - 1.0).pow(2).sum()
        return (off_diag + diag) / D


def compute_mel(waveform: torch.Tensor, orig_sr: int,
                n_mels: int = 128, n_fft: int = 1024,
                hop_length: int = 512, target_sr: int = 16000) -> torch.Tensor:
    """Compute Mel spectrogram from raw waveform."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    B = waveform.shape[0]

    # Resample to 16kHz if needed
    if orig_sr != target_sr and waveform.shape[1] > 0:
        waveform = F.interpolate(
            waveform.unsqueeze(1),
            size=int(waveform.shape[1] * target_sr / orig_sr),
            mode='linear', align_corners=False,
        ).squeeze(1)

    # STFT
    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(
        waveform, n_fft=n_fft, hop_length=hop_length,
        window=window, center=True, return_complex=True,
    )
    mag = torch.abs(stft)  # (B, n_fft//2+1, T)

    # Simple mel-scale approximation: log-spaced averaging
    n_freqs = mag.shape[1]
    mel_indices = torch.logspace(0, math.log10(n_freqs - 1), n_mels, base=10).long()
    mel_indices = torch.clamp(mel_indices, 0, n_freqs - 1)

    mel = torch.zeros(B, n_mels, mag.shape[-1], device=waveform.device)
    for i in range(n_mels - 1):
        start, end = mel_indices[i], mel_indices[i + 1]
        if end > start:
            mel[:, i, :] = mag[:, start:end, :].mean(dim=1)
    mel[:, -1, :] = mag[:, mel_indices[-1]:, :].mean(dim=1)

    mel = torch.log(mel + 1e-10)
    mel = (mel - mel.mean(dim=(-2, -1), keepdim=True)) / (mel.std(dim=(-2, -1), keepdim=True) + 1e-8)
    return mel.unsqueeze(1)  # (B, 1, n_mels, T)


# ═══════════════════════════════════════════════════════════
# Le2i Dataset Loader
# ═══════════════════════════════════════════════════════════

class Le2iAVDataset(Dataset):
    """
    Le2i dataset loader for M3-JEPA training.
    Returns (context_frames, target_frames, context_audio, target_audio).
    """

    def __init__(self, data_root: str, train: bool = True, normal_only: bool = True):
        self.data_root = data_root
        self.train = train
        self.normal_only = normal_only
        self.ctx_frames = cfg.num_context_frames
        self.tgt_frames = cfg.num_target_frames
        self.total_frames = self.ctx_frames + self.tgt_frames
        self.stride = cfg.stride

        # Scan for videos and annotations
        self.samples = []
        self._scan()

        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * 0.8)
        if train:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

        print(f"[Le2iAV] {'Train' if train else 'Eval'}: {len(self.samples)} clips "
              f"(normal_only={normal_only})")

    def _scan(self):
        """Find AVI videos and annotations."""
        import glob

        # Find annotation files and map to videos
        anno_patterns = [
            os.path.join(self.data_root, '**', 'Annotation_files', '*.txt'),
            os.path.join(self.data_root, '**', 'annotations', '*.txt'),
        ]

        video_map = {}  # base_name → video_path
        for root, dirs, files in os.walk(self.data_root):
            for f in files:
                if f.endswith('.avi'):
                    base = os.path.splitext(f)[0]  # "video (1)"
                    video_map[base] = os.path.join(root, f)

        # Parse annotations
        for pattern in anno_patterns:
            for anno_path in glob.glob(pattern, recursive=True):
                base = os.path.splitext(os.path.basename(anno_path))[0]
                if base not in video_map:
                    continue

                video_path = video_map[base]
                # Parse: frame, label, x1, y1, x2, y2
                fall_frames = set()
                with open(anno_path) as fh:
                    for line in fh:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            frame = int(parts[0])
                            label = int(parts[1])
                            # Labels: 8=fall, others=normal activities
                            if label == 8:
                                fall_frames.add(frame)

                if not fall_frames and self.normal_only:
                    # No falls → all frames are normal → generate clips
                    self._generate_clips(video_path, fall_frames, 99999)
                elif fall_frames:
                    # Generate both normal and fall clips
                    if not self.normal_only:
                        self._generate_clips(video_path, fall_frames, 99999)

    def _generate_clips(self, video_path: str, fall_frames: set, max_frames: int):
        """Generate clip samples from a video, avoiding or including falls."""
        # Cap maximum frames per video
        actual_max = min(max_frames, 500)

        # Sample clips at stride intervals
        for start in range(0, actual_max - self.total_frames, self.stride):
            clip_range = set(range(start, start + self.total_frames))
            has_fall = bool(clip_range & fall_frames)

            if self.normal_only and has_fall:
                continue
            if not self.normal_only and not has_fall:
                continue  # For fall detection eval, only keep fall clips

            self.samples.append({
                'video_path': video_path,
                'start_frame': start,
                'has_fall': has_fall,
            })

            # Limit to ~50 clips per video max
            if len([s for s in self.samples if s['video_path'] == video_path]) >= 50:
                break

    def _load_video(self, video_path: str, start: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Load video frames and audio from disk."""
        import av
        try:
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            fps = float(video_stream.average_rate) if video_stream.average_rate else 25.0

            # Decode frames
            all_frames = []
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format='rgb24')
                all_frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
                if len(all_frames) >= start + self.total_frames + 2:
                    break

            # Decode audio
            all_audio = []
            try:
                audio_stream = container.streams.audio[0]
                for frame in container.decode(audio=0):
                    arr = frame.to_ndarray()
                    all_audio.append(torch.from_numpy(arr.astype('float32')))
                    # Stop when we have enough audio
                    audio_len = sum(a.shape[-1] for a in all_audio)
                    audio_per_frame = audio_len / max(len(all_frames), 1)
                    if audio_per_frame * (start + self.total_frames) > 48000 * 3:
                        break
            except Exception:
                pass

            container.close()

            # Extract relevant frames
            frames = torch.stack(all_frames) if all_frames else torch.zeros(1, 3, 224, 224)
            end = min(start + self.total_frames, len(frames))
            clip_frames = frames[start:end]

            # Pad if needed
            if clip_frames.shape[0] < self.total_frames:
                pad = clip_frames[-1:].repeat(self.total_frames - clip_frames.shape[0], 1, 1, 1)
                clip_frames = torch.cat([clip_frames, pad], dim=0)

            # Resize: (T, C, H, W) → (T*C, H, W) → resize → reshape back
            if clip_frames.shape[-1] != cfg.frame_size or clip_frames.shape[-2] != cfg.frame_size:
                T, C, H, W = clip_frames.shape
                clip_frames = F.interpolate(
                    clip_frames.view(T * C, 1, H, W) if C > 1 else clip_frames.view(T, 1, H, W),
                    size=(cfg.frame_size, cfg.frame_size),
                    mode='bilinear', align_corners=False,
                )
                clip_frames = clip_frames.view(T, C, cfg.frame_size, cfg.frame_size)

            # Audio
            if all_audio:
                audio = torch.cat(all_audio, dim=-1).mean(dim=0)  # mono
                # Extract audio corresponding to our frame segment
                audio_per_frame = len(audio) / max(len(frames), 1)
                a_start = int(start * audio_per_frame)
                a_end = int(min(start + self.total_frames, len(frames)) * audio_per_frame)
                audio = audio[a_start:a_end]
            else:
                audio = torch.zeros(int(cfg.audio_duration * 2 * cfg.audio_sample_rate))

            return clip_frames, audio, fps

        except Exception as e:
            print(f"  Error loading {video_path}: {e}")
            return (torch.zeros(self.total_frames, 3, cfg.frame_size, cfg.frame_size),
                    torch.zeros(int(cfg.audio_duration * 2 * cfg.audio_sample_rate)),
                    25.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        frames, audio, fps = self._load_video(sample['video_path'], sample['start_frame'])

        # Split into context and target
        ctx_frames = frames[:self.ctx_frames].clone()
        tgt_frames = frames[self.ctx_frames:self.ctx_frames + self.tgt_frames].clone()

        # Split audio
        audio_mid = len(audio) // 2
        ctx_audio = audio[:audio_mid].clone()
        tgt_audio = audio[audio_mid:min(audio_mid * 2, len(audio))].clone()

        # Audio to target length - handle empty audio
        target_audio_len = int(cfg.audio_duration * cfg.audio_sample_rate)
        if len(ctx_audio) == 0:
            ctx_audio = torch.zeros(target_audio_len)
        elif len(ctx_audio) < target_audio_len:
            ctx_audio = F.pad(ctx_audio, (0, target_audio_len - len(ctx_audio)))
        else:
            ctx_audio = ctx_audio[:target_audio_len]
        if len(tgt_audio) == 0:
            tgt_audio = torch.zeros(target_audio_len)
        elif len(tgt_audio) < target_audio_len:
            tgt_audio = F.pad(tgt_audio, (0, target_audio_len - len(tgt_audio)))
        else:
            tgt_audio = tgt_audio[:target_audio_len]

        # Validate: ensure we have non-empty frames
        if ctx_frames.shape[0] < self.ctx_frames or tgt_frames.shape[0] < self.tgt_frames:
            # Fallback: random data as placeholder (shouldn't happen with good videos)
            ctx_frames = torch.randn(self.ctx_frames, 3, cfg.frame_size, cfg.frame_size)
            tgt_frames = torch.randn(self.tgt_frames, 3, cfg.frame_size, cfg.frame_size)

        return {
            'ctx_frames': ctx_frames,
            'tgt_frames': tgt_frames,
            'ctx_audio': ctx_audio,
            'tgt_audio': tgt_audio,
            'has_fall': sample['has_fall'],
            'video_path': sample['video_path'],
        }


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════

def train(args):
    """M3-JEPA training on Le2i dataset."""
    print("\n" + "=" * 60)
    print("M3-JEPA Video+Audio Fusion Training on Le2i")
    print("=" * 60)
    print(f"  Alpha (reg/cl): {cfg.alpha_reg}/{cfg.alpha_cl}")
    print(f"  MoE experts: {cfg.num_experts} per modality, Top-K={cfg.top_k}")
    print(f"  Train: normal only, {cfg.epochs} epochs, batch={cfg.batch_size}")
    print(f"  Device: {args.device}")

    # Apply args
    cfg.data_root = args.data_root
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.device = args.device
    if args.alpha is not None:
        cfg.alpha_reg = args.alpha
        cfg.alpha_cl = 1.0 - args.alpha

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # Dataset
    train_dataset = Le2iAVDataset(cfg.data_root, train=True, normal_only=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)

    eval_dataset = Le2iAVDataset(cfg.data_root, train=False, normal_only=False)
    eval_loader = DataLoader(eval_dataset, batch_size=1,
                             shuffle=False, num_workers=0)

    print(f"  Train clips: {len(train_dataset)}, Eval clips: {len(eval_dataset)}")

    # Model
    model = M3JEPAModel().to(device)

    # Count params
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * cfg.epochs,
    )

    # Training loop
    model.train()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        epoch_loss_reg = 0.0
        epoch_loss_cl = 0.0
        t0 = time.time()

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            try:
                output = model(batch, return_loss=True)
            except Exception as e:
                print(f"  Error at step {global_step}: {e}")
                continue

            loss = output['loss']
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()
            epoch_loss_reg += output['loss_reg'].item()
            epoch_loss_cl += output['loss_cl'].item()

            if global_step % 20 == 0:
                gate_reg_mean = output['gate_reg'].mean(dim=0)
                gate_usage = (gate_reg_mean > 0.01).sum().item()
                print(f"  E{epoch} S{global_step}: loss={loss.item():.4f} "
                      f"(reg={output['loss_reg'].item():.4f}, cl={output['loss_cl'].item():.4f}, "
                      f"sig={output['loss_sigreg'].item():.4f}) | experts_used={gate_usage}/{cfg.num_experts * 2}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}: avg_loss={avg_loss:.4f} "
              f"(reg={epoch_loss_reg/max(len(train_loader),1):.4f}, "
              f"cl={epoch_loss_cl/max(len(train_loader),1):.4f}) "
              f"| {elapsed:.1f}s")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(cfg.checkpoint_dir, 'm3jepa_le2i_best.pt'))

    # ━━━ Evaluation: anomaly detection on fall clips ━━━
    print("\n" + "=" * 60)
    print("Evaluation: Anomaly Detection on Fall Clips")
    print("=" * 60)

    model.eval()

    # Calibrate: collect normal stats from training set
    normal_z_preds = []
    print("  Calibrating normal distribution...")
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= 50:  # 50 samples enough for calibration
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            try:
                output = model(batch, return_loss=False)
                normal_z_preds.append(output['z_pred'].cpu())
            except Exception:
                continue

    if normal_z_preds:
        normal_z = torch.cat(normal_z_preds, dim=0)
        normal_mean = normal_z.mean(dim=0)
        normal_var = normal_z.var(dim=0) + 1e-8
        normal_std_mean = normal_var.mean().sqrt()
        normal_stats = {
            'mean': normal_mean.to(device),
            'inv_var': (1.0 / normal_var).to(device),
            'std_mean': normal_std_mean.to(device),
        }
        print(f"  Normal stats: mean_norm={normal_mean.norm():.2f}, "
              f"std_mean={normal_std_mean:.4f}")
    else:
        normal_stats = {
            'mean': torch.zeros(cfg.unified_dim, device=device),
            'inv_var': torch.ones(cfg.unified_dim, device=device),
            'std_mean': torch.tensor(1.0, device=device),
        }

    # Test on fall clips
    results = []
    with torch.no_grad():
        for batch in eval_loader:
            if len(results) >= 100:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            try:
                output = model(batch, return_loss=False)
                score = model.anomaly_score(output['z_pred'], normal_stats)
                results.append({
                    'score': score.item(),
                    'has_fall': bool(batch['has_fall'].item() if isinstance(batch['has_fall'], torch.Tensor) else batch['has_fall']),
                    'video': os.path.basename(batch['video_path'][0]) if isinstance(batch['video_path'], list) else os.path.basename(batch['video_path']),
                })
            except Exception as e:
                continue

    # Analyze
    if results:
        fall_scores = [r['score'] for r in results if r['has_fall']]
        normal_scores = [r['score'] for r in results if not r['has_fall']]

        print(f"\n  Evaluated {len(results)} clips "
              f"({len(fall_scores)} falls, {len(normal_scores)} normal)")

        if fall_scores and normal_scores:
            fall_mean = sum(fall_scores) / len(fall_scores)
            normal_mean = sum(normal_scores) / len(normal_scores)
            separation = abs(fall_mean - normal_mean) / (max(fall_scores + normal_scores) - min(fall_scores + normal_scores) + 1e-8)

            # Simple threshold detection at 2σ
            threshold = 2.0
            tp = sum(1 for s in fall_scores if s > threshold)
            fp = sum(1 for s in normal_scores if s > threshold)
            recall = tp / len(fall_scores) if fall_scores else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            print(f"  Fall scores:   mean={fall_mean:.2f} [{min(fall_scores):.2f}, {max(fall_scores):.2f}]")
            print(f"  Normal scores: mean={normal_mean:.2f} [{min(normal_scores):.2f}, {max(normal_scores):.2f}]")
            print(f"  Separation:    {fall_mean/normal_mean if normal_mean else 0:.1f}x")
            print(f"  Recall@2σ:     {recall:.1%}")
            print(f"  Precision@2σ:  {precision:.1%}")
            print(f"  TP={tp}, FP={fp}, FN={len(fall_scores)-tp}")
        else:
            print("  Not enough data for evaluation")

    print(f"\nBest model saved to {cfg.checkpoint_dir}/m3jepa_le2i_best.pt")
    return model


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="M3-JEPA Video+Audio Fusion on Le2i")
    parser.add_argument('--data_root', type=str, default='datasets/Le2i')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--alpha', type=float, default=None,
                        help='L_reg weight (0-1). Default: 0.5 (optimal per M3-JEPA)')
    args = parser.parse_args()

    # Update global config
    cfg.data_root = args.data_root
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.device = args.device
    if args.alpha is not None:
        cfg.alpha_reg = args.alpha
        cfg.alpha_cl = 1.0 - args.alpha

    train(args)


if __name__ == '__main__':
    main()
