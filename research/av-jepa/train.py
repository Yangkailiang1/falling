"""
AV-JEPA Training Script
------------------------
Trains the JEPA predictor on normal activity data.
Only needs normal samples — anomaly detection uses prediction error.

Usage:
    # Quick test with synthetic data
    python train.py --synthetic --epochs 50

    # Train on real data (when available)
    python train.py --data_path /path/to/normal_videos/
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from config import AVJEPAConfig, EncoderConfig, JEPAConfig, TrainingConfig
from jepa_model import AVJEPA
from data_utils import SyntheticAVDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train AV-JEPA")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--data_path", type=str, default=None, help="Path to real data")
    parser.add_argument("--save_path", type=str, default="checkpoints/av_jepa.pt")
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, epoch, total_steps):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_sigreg = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, (ctx_emb, tgt_emb, _) in enumerate(pbar):
        ctx_emb = ctx_emb.to(device)
        tgt_emb = tgt_emb.to(device)

        # Forward pass (embeddings directly — shortcut for synthetic data)
        z_ctx = model.projector(ctx_emb)
        z_pred = model.predictor(z_ctx)
        z_target = model.target_projector(tgt_emb)

        # Compute loss
        loss, loss_dict = model.compute_loss(z_pred, z_target, z_ctx)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), model.config.training.max_grad_norm
        )
        optimizer.step()

        # EMA update target encoder
        model.update_target_encoder(total_steps + step, total_steps + len(dataloader) * args.epochs)

        # Track metrics
        total_loss += loss_dict["total_loss"]
        total_pred_loss += loss_dict["pred_loss"]
        total_sigreg += loss_dict["sigreg"]
        n_batches += 1

        if step % 20 == 0:
            pbar.set_postfix({
                "loss": f"{loss_dict['total_loss']:.4f}",
                "pred": f"{loss_dict['pred_loss']:.4f}",
                "sigreg": f"{loss_dict['sigreg']:.4f}",
            })

    return {
        "loss": total_loss / n_batches,
        "pred_loss": total_pred_loss / n_batches,
        "sigreg": total_sigreg / n_batches,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate prediction error on normal vs. fall data."""
    model.eval()
    normal_errors = []
    fall_errors = []

    for ctx_emb, tgt_emb, labels in dataloader:
        ctx_emb = ctx_emb.to(device)
        tgt_emb = tgt_emb.to(device)

        z_ctx = model.projector(ctx_emb)
        z_pred = model.predictor(z_ctx)
        z_target = model.target_projector(tgt_emb)

        # Per-sample prediction error
        error = torch.nn.functional.mse_loss(
            z_pred, z_target, reduction="none"
        ).mean(dim=-1)  # (B,)

        for i, label in enumerate(labels):
            if label == 0:
                normal_errors.append(error[i].item())
            else:
                fall_errors.append(error[i].item())

    normal_mean = np.mean(normal_errors) if normal_errors else 0
    normal_std = np.std(normal_errors) if normal_errors else 0
    fall_mean = np.mean(fall_errors) if fall_errors else 0
    fall_std = np.std(fall_errors) if fall_errors else 0

    return {
        "normal_mean": normal_mean,
        "normal_std": normal_std,
        "fall_mean": fall_mean,
        "fall_std": fall_std,
        "separation": (fall_mean - normal_mean) / (normal_std + 1e-8),
    }


def main():
    args = parse_args()

    # Config
    config = AVJEPAConfig(
        encoder=EncoderConfig(),
        jepa=JEPAConfig(),
        training=TrainingConfig(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device,
        ),
    )
    device = torch.device(args.device)
    config.training.device = args.device

    print(f"Device: {device}")
    print(f"Config: {config}")

    # Data
    if args.synthetic:
        raw_embed_dim = config.encoder.video_embed_dim + config.encoder.audio_embed_dim
        train_dataset = SyntheticAVDataset(
            num_samples=2000,
            embed_dim=raw_embed_dim,
            normal_ratio=0.85,
            num_modes=8,
            noise_std=0.05,
            fall_displacement=5.0,
        )
        eval_dataset = SyntheticAVDataset(
            num_samples=500,
            embed_dim=raw_embed_dim,
            normal_ratio=0.5,  # 50% normal, 50% fall for eval
            num_modes=8,
            noise_std=0.05,
            fall_displacement=5.0,
        )
    else:
        raise NotImplementedError(
            "Real data loading not yet implemented. Use --synthetic for now."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Model
    model = AVJEPA(config).to(device)
    param_count = sum(p.numel() for p in model.projector.parameters()) + \
                  sum(p.numel() for p in model.predictor.parameters())
    print(f"Trainable params: {param_count:,}")

    # Optimizer
    trainable_params = list(model.projector.parameters()) + \
                       list(model.predictor.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=config.training.weight_decay,
    )

    # Training
    print("\n=== Training AV-JEPA ===")
    total_steps = 0
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, train_loader, optimizer, device, epoch, total_steps)
        total_steps += len(train_loader)

        print(f"  Epoch {epoch:3d} | loss={metrics['loss']:.4f} "
              f"pred={metrics['pred_loss']:.4f} sigreg={metrics['sigreg']:.4f}")

        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            eval_metrics = evaluate(model, eval_loader, device)
            print(f"  --- Eval ---")
            print(f"    Normal error: {eval_metrics['normal_mean']:.4f} ± {eval_metrics['normal_std']:.4f}")
            print(f"    Fall error:   {eval_metrics['fall_mean']:.4f} ± {eval_metrics['fall_std']:.4f}")
            print(f"    Separation:   {eval_metrics['separation']:.2f}σ")
            print()

    # Final evaluation
    print("\n=== Final Evaluation ===")
    eval_metrics = evaluate(model, eval_loader, device)
    print(f"  Normal prediction error: {eval_metrics['normal_mean']:.4f} ± {eval_metrics['normal_std']:.4f}")
    print(f"  Fall prediction error:   {eval_metrics['fall_mean']:.4f} ± {eval_metrics['fall_std']:.4f}")
    print(f"  Separation (σ):          {eval_metrics['separation']:.2f}")
    print(f"  Fall > Normal:           {eval_metrics['fall_mean'] > eval_metrics['normal_mean']}")

    if eval_metrics['separation'] > 2.0:
        print("  ✅ SUCCESS: Model clearly separates falls from normal activity!")
    elif eval_metrics['separation'] > 1.0:
        print("  ⚠️  MODERATE: Some separation, may need more training/data.")
    else:
        print("  ❌ FAIL: No clear separation. Check data/hyperparameters.")

    # Save
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else ".", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "eval_metrics": eval_metrics,
    }, args.save_path)
    print(f"\nModel saved to {args.save_path}")


if __name__ == "__main__":
    main()
