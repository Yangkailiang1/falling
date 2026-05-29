"""
T-JEPA: Full Model Assembly

Combines all T-JEPA components into an end-to-end model:
  Encoders (frozen) → HybridFusion (trainable) → Predictor → Projector → AnomalyGate

Supports multi-stage training:
  Stage 1: Train fusion (M3-JEPA pre-training)
  Stage 2: Train projector (VL-JEPA text alignment)
  Stage 3: Fine-tune on OmniFall
  Stage 4: Calibrate anomaly gate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Literal
from pathlib import Path

from .config import TJEPSConfig
from .encoders import TJEPSEncoders
from .fusion import HybridFusion, M3JEPAFusion, SIGRegLoss
from .predictor import build_predictor
from .projector import VLJEPAPhraseProjector
from .anomaly_gate import TJEPSAnomalyGate
from .phrase_retriever import PhraseLibrary
from .utils.frame_masking import frame_masking_augment
from .utils.drop_pathway import drop_modality


class TJEPS(nn.Module):
    """
    T-JEPA: Temporal Joint Embedding Predictive Architecture.

    Full model for multimodal fall detection with predictive text output.

    Architecture flow:
      1. Encode 4 modalities (video, audio, skeleton, text) → tokens
      2. Hybrid fusion (M3-JEPA MoE + Cross-Attention) → z_fused
      3. Temporal prediction (Transformer or Mamba) → z_future
      4. Text-aligned projection (VL-JEPA + TC-JEPA) → z_text
      5. 3-tier anomaly gating → fall detection + text report
    """

    def __init__(self, config: TJEPSConfig):
        super().__init__()
        self.config = config

        # Modality mask for fusion gating
        self.modality_mask = {
            'video': config.use_video,
            'audio': config.use_audio,
            'skeleton': config.use_skeleton,
            'text': config.use_text,
        }

        # ━━ 1. Frozen encoders ━━
        self.encoders = TJEPSEncoders(
            config.video, config.audio, config.skeleton, config.text,
            use_video=config.use_video,
            use_audio=config.use_audio,
            use_skeleton=config.use_skeleton,
            use_text=config.use_text,
        )

        # ━━ 2. Trainable fusion ━━
        if config.fusion.use_cross_attention:
            self.fusion = HybridFusion(config.fusion, modality_mask=self.modality_mask)
        else:
            self.fusion = M3JEPAFusion(config.fusion)

        # ━━ 3. Predictor ━━
        self.predictor = build_predictor(config.predictor)

        # ━━ 4. Text projector (VL-JEPA + TC-JEPA) ━━
        self.projector = VLJEPAPhraseProjector(config.projector)

        # ━━ 5. Anomaly gate ━━
        self.anomaly_gate = TJEPSAnomalyGate(config.anomaly_gate)

        # ━━ 6. Phrase library (built at training time) ━━
        self.phrase_library = PhraseLibrary(embed_dim=config.text.embed_dim)

        # ━━ Regularization ━━
        self.sigreg_loss = SIGRegLoss()

        # ━━ State ━━
        self.training_stage: Literal['stage1', 'stage2', 'stage3', 'inference'] = 'stage1'

    # ═══════════════════════════════════════════════════════════
    # Training Mode Configuration
    # ═══════════════════════════════════════════════════════════

    def set_stage(self, stage: str):
        """
        Configure trainable parameters for each training stage.

        Stage 1 (M3-JEPA fusion pre-training):
          Train: fusion module
          Frozen: encoders, predictor, projector

        Stage 2 (VL-JEPA text alignment):
          Train: projector
          Frozen: encoders, predictor
          Optional: fine-tune fusion last layers

        Stage 3 (OmniFall fine-tuning):
          Train: fusion gates, projector
          Frozen: encoders, predictor
        """
        self.training_stage = stage

        # Freeze everything first
        for p in self.parameters():
            p.requires_grad = False

        if stage == 'stage1':
            # Train fusion + predictor
            for p in self.fusion.parameters():
                p.requires_grad = True
            for p in self.predictor.parameters():
                p.requires_grad = True

        elif stage == 'stage2':
            # Train projector
            for p in self.projector.parameters():
                p.requires_grad = True
            # Optionally train fusion projection layers
            for p in self.fusion.parameters():
                p.requires_grad = True

        elif stage == 'stage3':
            # Train fusion + projector (full except encoders + predictor)
            for p in self.fusion.parameters():
                p.requires_grad = True
            for p in self.projector.parameters():
                p.requires_grad = True

        elif stage == 'inference':
            pass  # All frozen

        # Move frozen encoders to CPU to save GPU VRAM
        self._place_encoders()

    def _place_encoders(self):
        """Move frozen encoders to encoder_device (cpu) to free GPU memory."""
        encoder_device = self.config.training.encoder_device
        self.encoders = self.encoders.to(encoder_device)

    # ═══════════════════════════════════════════════════════════
    # Forward Pass
    # ═══════════════════════════════════════════════════════════

    def forward(
        self,
        ctx_frames: torch.Tensor,
        ctx_audio: torch.Tensor,
        ctx_skeleton: torch.Tensor,
        text_condition: List[str],
        tgt_frames: Optional[torch.Tensor] = None,
        tgt_audio: Optional[torch.Tensor] = None,
        tgt_skeleton: Optional[torch.Tensor] = None,
        target_text: Optional[List[str]] = None,
        apply_frame_mask: bool = False,
        apply_drop_pathway: bool = False,
    ) -> Dict:
        """
        Full T-JEPA forward pass.

        Args:
            ctx_frames: (B, T_ctx, C, H, W) context video frames
            ctx_audio: (B, L_ctx) context audio
            ctx_skeleton: (B, T_ctx, 17, 3) context skeleton
            text_condition: list of strings, current state description
            tgt_frames: (B, T_tgt, C, H, W) target video frames (training only)
            tgt_audio: (B, L_tgt) target audio (training only)
            tgt_skeleton: (B, T_tgt, 17, 3) target skeleton (training only)
            target_text: list of strings, future state description (training only)
            apply_frame_mask: enable Fall-Mamba frame masking augmentation
            apply_drop_pathway: enable Fall-Mamba DropPathway regularization

        Returns:
            dict with z_fused, z_future, z_text, losses, etc.
        """
        B = ctx_frames.shape[0]

        # ━━ Fall-Mamba: Frame Masking ━━
        if apply_frame_mask and self.training:
            ctx_frames = frame_masking_augment(
                ctx_frames, self.config.fall_mamba.frame_mask_ratio
            )
            if tgt_frames is not None:
                tgt_frames = frame_masking_augment(
                    tgt_frames, self.config.fall_mamba.frame_mask_ratio
                )

        # ━━ Fall-Mamba: DropPathway ━━
        if apply_drop_pathway and self.training:
            ctx_frames, ctx_audio, ctx_skeleton, _, _ = drop_modality(
                ctx_frames, ctx_audio, ctx_skeleton,
                torch.zeros(B, self.config.text.embed_dim, device=ctx_frames.device),
                drop_prob=self.config.fall_mamba.drop_modality_prob,
            )

        # ━━ 1. Encode context ━━
        encoder_device = self.config.training.encoder_device
        train_device = ctx_frames.device

        # Move inputs to encoder device for frozen encoder inference (saves GPU VRAM)
        ctx_video_input = ctx_frames[:, :self.config.training.num_frames_context] if ctx_frames.dim() == 5 else ctx_frames
        encodings = self.encoders(
            video=ctx_video_input.to(encoder_device) if ctx_video_input is not None else None,
            audio=ctx_audio.to(encoder_device) if ctx_audio is not None else None,
            skeleton=ctx_skeleton.to(encoder_device) if ctx_skeleton is not None else None,
            texts=text_condition,
        )
        # Move token outputs back to training device
        encodings = {k: v.to(train_device) if isinstance(v, torch.Tensor) else v
                     for k, v in encodings.items()}

        # ━━ 2. Fusion (M3-JEPA + Cross-Attention) ━━
        fusion_output = self.fusion(
            encodings['v_tokens'],
            encodings['a_tokens'],
            encodings['s_tokens'],
            encodings['t_embed'],
            modality_mask=self.modality_mask,
        )
        z_fused = fusion_output['z_fused']  # (B, 1024)

        # ━━ 3. Predict future ━━
        z_future = self.predictor(z_fused)  # (B, 1024)

        # ━━ 4. Project to text space ━━
        z_text = self.projector(z_future, encodings['t_embed'])  # (B, 3584)

        result = {
            'z_fused': z_fused,
            'z_future': z_future,
            'z_text': z_text,
            'encodings': encodings,
            'fusion_output': fusion_output,
        }

        # ━━ 5. Compute losses (if targets provided) ━━
        if tgt_frames is not None and tgt_audio is not None and tgt_skeleton is not None:
            losses = self._compute_losses(
                z_fused, z_future, z_text,
                encodings['t_embed'],
                tgt_frames, tgt_audio, tgt_skeleton,
                target_text,
            )
            result['losses'] = losses
            result['loss'] = sum(losses.values()) if losses else torch.tensor(0.0)

        return result

    def _compute_losses(
        self,
        z_fused: torch.Tensor,
        z_future: torch.Tensor,
        z_text: torch.Tensor,
        text_embed: torch.Tensor,
        tgt_frames: torch.Tensor,
        tgt_audio: torch.Tensor,
        tgt_skeleton: torch.Tensor,
        target_text: Optional[List[str]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective training loss.

        Loss components:
          1. JEPA prediction loss: MSE(z_future, z_target)
          2. SIGReg anti-collapse: covariance regularization
          3. Contrastive text alignment: InfoNCE (Stage 2)
          4. Mutual information: M3-JEPA MI maximization (Stage 1)
        """
        losses = {}

        # ━━ Encode target ━━
        edev = self.config.training.encoder_device
        tdev = z_fused.device
        with torch.no_grad():
            target_enc = self.encoders(
                video=tgt_frames.to(edev) if tgt_frames is not None else None,
                audio=tgt_audio.to(edev) if tgt_audio is not None else None,
                skeleton=tgt_skeleton.to(edev) if tgt_skeleton is not None else None,
                texts=[''] * (tgt_frames.shape[0] if tgt_frames is not None else 1),
            )
            target_enc = {k: v.to(tdev) if isinstance(v, torch.Tensor) else v
                          for k, v in target_enc.items()}
            # Fuse target
            target_fusion = self.fusion(
                target_enc['v_tokens'],
                target_enc['a_tokens'],
                target_enc['s_tokens'],
                target_enc['t_embed'],
                modality_mask=self.modality_mask,
            )
            z_target = target_fusion['z_fused']  # (B, 1024)

        # 1. JEPA prediction loss
        if self.training_stage in ('stage1', 'stage3'):
            jepa_loss = F.mse_loss(z_future, z_target)
            losses['jepa_mse'] = jepa_loss * self.config.training.mse_weight

        # 2. SIGReg
        sigreg = self.sigreg_loss(z_fused) + self.sigreg_loss(z_future)
        losses['sigreg'] = sigreg * self.config.training.sigreg_weight

        # 3. Mutual information loss (M3-JEPA)
        if self.training_stage in ('stage1', 'stage3'):
            mi_loss = self._mutual_info_loss(z_fused, z_target)
            losses['mutual_info'] = mi_loss * self.config.training.mutual_info_weight

        # 4. Text alignment loss (VL-JEPA) — only when text modality is enabled
        if self.config.use_text and self.training_stage in ('stage2', 'stage3') and target_text is not None:
            # Encode target text
            with torch.no_grad():
                target_text_embed = self.encoders.text_encoder(target_text)

            # Contrastive loss
            contrastive = self.projector.compute_contrastive_loss(z_text, target_text_embed)
            losses['contrastive'] = contrastive * self.config.training.contrastive_weight

            # Alignment loss
            alignment = self.projector.compute_alignment_loss(z_text, target_text_embed)
            losses['alignment'] = alignment * 0.5

        return losses

    def _mutual_info_loss(self, z: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """
        Simplified mutual information maximization loss (M3-JEPA).

        Maximizes I(z; z_target) by minimizing the variance-normalized MSE,
        which encourages z to capture information predictive of z_target.
        """
        z_norm = F.normalize(z, dim=-1)
        t_norm = F.normalize(z_target, dim=-1)
        return -(z_norm * t_norm).sum(dim=-1).mean()  # Negative cosine → maximize similarity

    # ═══════════════════════════════════════════════════════════
    # Inference
    # ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def detect(
        self,
        ctx_frames: torch.Tensor,
        ctx_audio: torch.Tensor,
        ctx_skeleton: torch.Tensor,
        text_condition: str,
    ) -> Dict:
        """
        End-to-end fall detection inference.

        Args:
            ctx_frames: (T_ctx, C, H, W) or (1, T_ctx, C, H, W) context frames
            ctx_audio: (L,) or (1, L) context audio
            ctx_skeleton: (T_ctx, 17, 3) or (1, T_ctx, 17, 3) context skeleton
            text_condition: str text description

        Returns:
            dict with is_fall, anomaly_score, top_phrase, etc.
        """
        # Add batch dim
        if ctx_frames.dim() == 4:
            ctx_frames = ctx_frames.unsqueeze(0)
        if ctx_audio.dim() == 1:
            ctx_audio = ctx_audio.unsqueeze(0)
        if ctx_skeleton.dim() == 3:
            ctx_skeleton = ctx_skeleton.unsqueeze(0)

        # Forward pass (no targets)
        output = self.forward(
            ctx_frames=ctx_frames,
            ctx_audio=ctx_audio,
            ctx_skeleton=ctx_skeleton,
            text_condition=[text_condition],
        )

        # Gate evaluation
        gate_result = self.anomaly_gate.step(
            z_future=output['z_future'].squeeze(0),
            z_text=output['z_text'].squeeze(0),
            phrase_embeddings=self.phrase_library.get_embedding_tensor(),
            phrase_labels=self.phrase_library.get_labels(),
        )

        return {**output, **gate_result}

    # ═══════════════════════════════════════════════════════════
    # Build Phrase Library
    # ═══════════════════════════════════════════════════════════

    def build_phrase_library(self, use_chinese: bool = True):
        """Build the Gate 2 phrase library using the text encoder."""
        self.phrase_library.build(
            text_encoder=self.encoders.text_encoder,
            use_chinese=use_chinese,
        )

    # ═══════════════════════════════════════════════════════════
    # Save / Load
    # ═══════════════════════════════════════════════════════════

    def save(self, path: str):
        """Save trainable parameters and calibration state."""
        state = {
            'fusion': self.fusion.state_dict(),
            'projector': self.projector.state_dict(),
            'config': self.config,
            'training_stage': self.training_stage,
        }
        # Only save predictor if it was trained
        if self.training_stage == 'stage1':
            pass  # predictor is frozen
        else:
            state['predictor'] = self.predictor.state_dict()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        print(f"[TJEPS] Model saved to {path}")

    def load(self, path: str, strict: bool = False):
        """Load trainable parameters."""
        state = torch.load(path, map_location='cpu')
        self.fusion.load_state_dict(state['fusion'], strict=strict)
        self.projector.load_state_dict(state['projector'], strict=strict)
        if 'predictor' in state:
            self.predictor.load_state_dict(state['predictor'], strict=strict)
        self.training_stage = state.get('training_stage', 'inference')
        print(f"[TJEPS] Model loaded from {path}")
