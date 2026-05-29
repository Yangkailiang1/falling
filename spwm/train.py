"""
T-JEPA Training Scripts

Multi-stage training pipeline:
  Stage 1: M3-JEPA Fusion Pre-training (V1-33K or any generic data)
  Stage 2: VL-JEPA Text Alignment Pre-training
  Stage 3: OmniFall Fine-tuning
  Stage 4: Anomaly Gate Calibration

Supports:
  - Mixed precision training (AMP)
  - Gradient accumulation
  - EMA target encoder (cosine schedule)
  - Fall-Mamba augmentations (Frame Masking, DropPathway)
  - Checkpointing and logging
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spwm.config import TJEPSConfig
from spwm.tjepa_model import TJEPS
from spwm.data.le2i_dataset import Le2iDataset
from spwm.data.text_annotations import TextAnnotator
from spwm.data.skeleton_extractor import SkeletonExtractor
from torch.utils.data import DataLoader


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def create_optimizer(model: TJEPS, config: TJEPSConfig) -> AdamW:
    """Create optimizer for trainable parameters."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"Trainable parameters: {n_params:,}")

    return AdamW(
        trainable,
        lr=config.training.lr,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay,
    )


def create_scheduler(
    optimizer: AdamW,
    config: TJEPSConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create learning rate scheduler with warmup."""
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.training.warmup_steps,
    )

    if config.training.lr_schedule == 'cosine':
        main = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - config.training.warmup_steps,
            eta_min=config.training.lr * 0.01,
        )
    else:
        main = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=total_steps - config.training.warmup_steps,
        )

    return SequentialLR(
        optimizer,
        schedulers=[warmup, main],
        milestones=[config.training.warmup_steps],
    )


def ema_cosine_decay(step: int, total_steps: int, start: float = 0.996, end: float = 1.0) -> float:
    """Cosine EMA decay schedule (from V-JEPA 2)."""
    if total_steps <= 0:
        return end
    progress = min(step / total_steps, 1.0)
    return end + (start - end) * (1 + torch.cos(torch.tensor(progress * torch.pi)).item()) / 2


def update_ema_target(student: nn.Module, teacher: nn.Module, decay: float):
    """Update EMA target encoder parameters."""
    with torch.no_grad():
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.data.mul_(decay).add_(s_param.data, alpha=1 - decay)


# ═══════════════════════════════════════════════════════════════
# Stage 1: M3-JEPA Fusion Pre-training
# ═══════════════════════════════════════════════════════════════

def train_stage1(args):
    """
    Train M3-JEPA fusion module.

    Only the fusion layers are trainable. Encoders and predictor are frozen.
    Loss = MSE(z_future, z_target) + SIGReg + Mutual Info

    Training data: Le2i (normal clips only) or any generic video dataset.
    """
    print("\n" + "=" * 60)
    print("Stage 1: M3-JEPA Fusion Pre-training")
    print("=" * 60)

    config = TJEPSConfig()
    config.training.stage1_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.lr = args.lr

    # Modality toggles
    config.use_skeleton = args.use_skeleton
    config.use_text = args.use_text

    # Model
    model = TJEPS(config)
    model.set_stage('stage1')
    model = model.to(args.device)

    # Data
    train_dataset = Le2iDataset(
        data_root=args.data_root,
        num_context_frames=config.training.num_frames_context,
        num_target_frames=config.training.num_frames_target,
        train=True,
        normal_only=True,
        stride=args.stride,
        skeleton_extractor=SkeletonExtractor(method='mediapipe') if config.use_skeleton else None,
        text_annotator=TextAnnotator() if config.use_text else None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    optimizer = create_optimizer(model, config)
    total_steps = len(train_loader) * config.training.stage1_epochs
    scheduler = create_scheduler(optimizer, config, total_steps)

    # Training
    scaler = GradScaler(enabled=config.training.mixed_precision)
    model.train()
    global_step = 0

    print(f"Training: {config.training.stage1_epochs} epochs, "
          f"{len(train_loader)} steps/epoch, batch_size={config.training.batch_size}")

    for epoch in range(config.training.stage1_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with autocast(enabled=config.training.mixed_precision):
                output = model(
                    ctx_frames=batch['ctx_frames'],
                    ctx_audio=batch['ctx_audio'],
                    ctx_skeleton=batch['ctx_skeleton'],
                    text_condition=batch['text_condition'],
                    tgt_frames=batch['tgt_frames'],
                    tgt_audio=batch['tgt_audio'],
                    tgt_skeleton=batch['tgt_skeleton'],
                    apply_frame_mask=config.fall_mamba.frame_mask_ratio > 0,
                    apply_drop_pathway=config.fall_mamba.drop_modality_prob > 0,
                )
                loss = output['loss']

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if config.training.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()

            # Log
            if batch_idx % config.training.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                loss_str = ', '.join(f"{k}={v.item():.4f}" for k, v in output.get('losses', {}).items())
                print(f"E{epoch} S{global_step} | loss={loss.item():.4f} ({loss_str}) | lr={lr:.2e}")

        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch:3d} avg_loss={avg_loss:.4f} time={format_time(elapsed)}")

        # Save checkpoint
        if epoch % config.training.save_interval == 0 or epoch == config.training.stage1_epochs - 1:
            ckpt_path = os.path.join(config.training.checkpoint_dir, f'stage1_epoch{epoch}.pt')
            model.save(ckpt_path)

    print(f"\nStage 1 completed. Total steps: {global_step}")
    return model


# ═══════════════════════════════════════════════════════════════
# Stage 2: VL-JEPA Text Alignment
# ═══════════════════════════════════════════════════════════════

def train_stage2(args):
    """
    Train VL-JEPA + TC-JEPA text alignment projector.

    Maps z_future → LLM embedding space (z_text).
    Uses contrastive loss (InfoNCE) + alignment loss (MSE).

    Requires: model from Stage 1 checkpoint.
    """
    print("\n" + "=" * 60)
    print("Stage 2: VL-JEPA Text Alignment")
    print("=" * 60)

    config = TJEPSConfig()
    config.training.stage2_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.lr = args.lr * 0.5  # Lower LR for fine-tuning

    # Modality toggles
    config.use_skeleton = args.use_skeleton
    config.use_text = args.use_text

    # Model
    model = TJEPS(config)
    if args.resume:
        print(f"Loading Stage 1 checkpoint: {args.resume}")
        model.load(args.resume)
    else:
        print("Warning: No Stage 1 checkpoint. Starting from scratch.")
    model.set_stage('stage2')
    model = model.to(args.device)

    # Build phrase library if text is enabled
    if config.use_text:
        model.build_phrase_library(use_chinese=True)

    # Data
    train_dataset = Le2iDataset(
        data_root=args.data_root,
        num_context_frames=config.training.num_frames_context,
        num_target_frames=config.training.num_frames_target,
        train=True,
        normal_only=True,
        stride=args.stride,
        skeleton_extractor=SkeletonExtractor(method='mediapipe') if config.use_skeleton else None,
        text_annotator=TextAnnotator() if config.use_text else None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = create_optimizer(model, config)
    total_steps = len(train_loader) * config.training.stage2_epochs
    scheduler = create_scheduler(optimizer, config, total_steps)
    scaler = GradScaler(enabled=config.training.mixed_precision)

    model.train()
    global_step = 0

    for epoch in range(config.training.stage2_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with autocast(enabled=config.training.mixed_precision):
                output = model(
                    ctx_frames=batch['ctx_frames'],
                    ctx_audio=batch['ctx_audio'],
                    ctx_skeleton=batch['ctx_skeleton'],
                    text_condition=batch['text_condition'],
                    tgt_frames=batch['tgt_frames'],
                    tgt_audio=batch['tgt_audio'],
                    tgt_skeleton=batch['tgt_skeleton'],
                    target_text=batch['target_text'],  # VL-JEPA target
                )
                loss = output['loss']

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()

            if batch_idx % config.training.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                loss_str = ', '.join(f"{k}={v.item():.4f}" for k, v in output.get('losses', {}).items())
                print(f"E{epoch} S{global_step} | loss={loss.item():.4f} ({loss_str}) | lr={lr:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch:3d} avg_loss={avg_loss:.4f} time={format_time(elapsed)}")

        if epoch % config.training.save_interval == 0 or epoch == config.training.stage2_epochs - 1:
            ckpt_path = os.path.join(config.training.checkpoint_dir, f'stage2_epoch{epoch}.pt')
            model.save(ckpt_path)

    # Save phrase library
    model.phrase_library.save(
        os.path.join(config.training.checkpoint_dir, 'phrase_library.pt')
    )

    print(f"\nStage 2 completed. Total steps: {global_step}")
    return model


# ═══════════════════════════════════════════════════════════════
# Stage 3: OmniFall Fine-tuning
# ═══════════════════════════════════════════════════════════════

def train_stage3(args):
    """
    Fine-tune on OmniFall dataset.

    Fine-tunes the fusion and projector layers on fall-specific data.
    Includes both normal and fall clips for calibration.
    """
    print("\n" + "=" * 60)
    print("Stage 3: OmniFall Fine-tuning")
    print("=" * 60)

    config = TJEPSConfig()
    config.training.stage3_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.lr = args.lr * 0.2  # Even lower LR

    # Modality toggles
    config.use_skeleton = args.use_skeleton
    config.use_text = args.use_text

    model = TJEPS(config)
    if args.resume:
        model.load(args.resume)
    model.set_stage('stage3')
    model = model.to(args.device)

    # Data - include both normal and fall clips for calibration
    train_dataset = Le2iDataset(
        data_root=args.data_root,
        num_context_frames=config.training.num_frames_context,
        num_target_frames=config.training.num_frames_target,
        train=True,
        normal_only=False,  # Include falls for calibration
        stride=args.stride,
        skeleton_extractor=SkeletonExtractor(method='mediapipe') if config.use_skeleton else None,
        text_annotator=TextAnnotator() if config.use_text else None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = create_optimizer(model, config)
    total_steps = len(train_loader) * config.training.stage3_epochs
    scheduler = create_scheduler(optimizer, config, total_steps)
    scaler = GradScaler(enabled=config.training.mixed_precision)

    model.train()
    global_step = 0

    for epoch in range(config.training.stage3_epochs):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with autocast(enabled=config.training.mixed_precision):
                output = model(
                    ctx_frames=batch['ctx_frames'],
                    ctx_audio=batch['ctx_audio'],
                    ctx_skeleton=batch['ctx_skeleton'],
                    text_condition=batch['text_condition'],
                    tgt_frames=batch['tgt_frames'],
                    tgt_audio=batch['tgt_audio'],
                    tgt_skeleton=batch['tgt_skeleton'],
                    target_text=batch.get('target_text'),
                )
                loss = output['loss']

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()

            if batch_idx % config.training.log_interval == 0:
                loss_str = ', '.join(f"{k}={v.item():.4f}" for k, v in output.get('losses', {}).items())
                print(f"E{epoch} S{global_step} | loss={loss.item():.4f} ({loss_str})")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch:3d} avg_loss={avg_loss:.4f}")

        if epoch % config.training.save_interval == 0 or epoch == config.training.stage3_epochs - 1:
            ckpt_path = os.path.join(config.training.checkpoint_dir, f'stage3_epoch{epoch}.pt')
            model.save(ckpt_path)

    print(f"\nStage 3 completed. Total steps: {global_step}")
    return model


# ═══════════════════════════════════════════════════════════════
# Stage 4: Gate Calibration
# ═══════════════════════════════════════════════════════════════

def calibrate_gate(args):
    """
    Calibrate Gate 1 anomaly thresholds.

    Collects z_future samples from normal activity to compute
    normal distribution statistics (mean, std, covariance).
    """
    print("\n" + "=" * 60)
    print("Stage 4: Anomaly Gate Calibration")
    print("=" * 60)

    config = TJEPSConfig()

    # Modality toggles
    config.use_skeleton = args.use_skeleton
    config.use_text = args.use_text

    model = TJEPS(config)
    if args.resume:
        model.load(args.resume)
    model.set_stage('inference')
    model = model.to(args.device)
    model.eval()

    # Use only normal clips for calibration
    calib_dataset = Le2iDataset(
        data_root=args.data_root,
        num_context_frames=config.training.num_frames_context,
        num_target_frames=config.training.num_frames_target,
        train=True,  # Use training split (no falls)
        normal_only=True,
        stride=4,
        skeleton_extractor=SkeletonExtractor(method='mediapipe') if config.use_skeleton else None,
        text_annotator=TextAnnotator() if config.use_text else None,
    )
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    # Collect z_future samples from normal activity
    samples = []
    n_samples = min(len(calib_dataset), config.anomaly_gate.calibration_samples)

    print(f"Collecting {n_samples} normal activity samples...")
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            if len(samples) >= n_samples:
                break

            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            output = model(
                ctx_frames=batch['ctx_frames'],
                ctx_audio=batch['ctx_audio'],
                ctx_skeleton=batch['ctx_skeleton'],
                text_condition=batch['text_condition'],
            )

            z_future = output['z_future'].cpu()  # (B, 1024)
            samples.append(z_future)

            if (i + 1) % 10 == 0:
                print(f"  Collected {len(samples) * config.training.batch_size} samples...")

    samples = torch.cat(samples, dim=0)[:n_samples]
    print(f"Collected {samples.shape[0]} normal samples, dim={samples.shape[1]}")

    # Compute statistics
    mean = samples.mean(dim=0)
    std = samples.std(dim=0).clamp(min=1e-6)
    centered = samples - mean
    cov = (centered.T @ centered) / (samples.shape[0] - 1)

    print(f"Normal statistics: mean_norm={mean.norm():.3f}, "
          f"mean_std={std.mean():.3f}, "
          f"cov_trace={cov.trace():.3f}")

    # Compute anomaly scores for ALL samples to verify threshold
    from spwm.anomaly_gate import AnomalyDetector
    detector = AnomalyDetector(config.anomaly_gate)
    detector.calibrate(samples)

    # Score calibration samples
    scores = []
    for i in range(samples.shape[0]):
        _, sigma = detector.compute_anomaly_score(samples[i])
        scores.append(sigma.item())

    scores = sorted(scores)
    p50 = scores[len(scores) // 2]
    p95 = scores[int(len(scores) * 0.95)]
    p99 = scores[int(len(scores) * 0.99)]

    print(f"\nScore distribution on normal data:")
    print(f"  Median sigma:  {p50:.3f}")
    print(f"  P95 sigma:     {p95:.3f}")
    print(f"  P99 sigma:     {p99:.3f}")
    print(f"  Threshold 2σ:  {2.0:.3f}")

    threshold_rec = 2.0
    if p99 > 2.0:
        print(f"\n  Warning: P99 > 2.0. Consider setting threshold > {p99:.1f}")
        threshold_rec = max(2.5, p99 * 1.1)

    # Save
    calib_path = os.path.join(config.training.checkpoint_dir, 'gate_calibration.pt')
    detector.save_calibration(calib_path)
    print(f"\nCalibration saved to {calib_path}")

    # Also save to model weights dir
    weights_path = os.path.join(config.model_weights_dir, 'gate_calibration.pt')
    detector.save_calibration(weights_path)
    print(f"Calibration saved to {weights_path}")

    return threshold_rec


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="T-JEPA Multi-Stage Training")
    parser.add_argument('--stage', type=str, required=True,
                        choices=['stage1', 'stage2', 'stage3', 'calibrate', 'all'],
                        help='Training stage to run')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root (Le2i)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--stride', type=int, default=4,
                        help='Frame stride for clip generation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to resume checkpoint')
    parser.add_argument('--extract_skeleton', action='store_true',
                        help='Enable skeleton extraction')
    parser.add_argument('--use_skeleton', action='store_true', default=False,
                        help='Enable skeleton modality (requires pretrained weights)')
    parser.add_argument('--use_text', action='store_true', default=False,
                        help='Enable text modality (requires HuggingFace download)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint save directory')

    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if not torch.cuda.is_available():
        args.device = 'cpu'
        print("CUDA not available, using CPU")

    if args.stage == 'stage1' or args.stage == 'all':
        train_stage1(args)

    if args.stage == 'stage2' or args.stage == 'all':
        if args.stage == 'stage2' and not args.resume:
            # Auto-find latest stage1 checkpoint
            stage1_ckpts = sorted(Path(args.checkpoint_dir).glob('stage1_epoch*.pt'))
            if stage1_ckpts:
                args.resume = str(stage1_ckpts[-1])
                print(f"Auto-resuming from: {args.resume}")
        train_stage2(args)

    if args.stage == 'stage3' or args.stage == 'all':
        if args.stage == 'stage3' and not args.resume:
            stage2_ckpts = sorted(Path(args.checkpoint_dir).glob('stage2_epoch*.pt'))
            if stage2_ckpts:
                args.resume = str(stage2_ckpts[-1])
                print(f"Auto-resuming from: {args.resume}")
        train_stage3(args)

    if args.stage == 'calibrate' or args.stage == 'all':
        if args.stage == 'calibrate' and not args.resume:
            stage3_ckpts = sorted(Path(args.checkpoint_dir).glob('stage3_epoch*.pt'))
            if stage3_ckpts:
                args.resume = str(stage3_ckpts[-1])
                print(f"Auto-resuming from: {args.resume}")
        calibrate_gate(args)


if __name__ == '__main__':
    main()
