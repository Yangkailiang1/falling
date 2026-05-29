"""
NTU120 Skeleton-JEPA Pretraining.

Pretrain SkeletonEncoder on 114K NTU skeleton sequences via JEPA objective,
then transfer to Le2i for fall detection classification.

Usage:
  python3 -m spwm.ntu_jepa_pretrain --device cuda:0
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from .skeleton_jepa_model import create_skeleton_jepa
from .data.ntu_dataset import NTUSkeletonDataset, create_ntu_jepa_loaders


def parse_args():
    p = argparse.ArgumentParser(description="NTU120 Skeleton-JEPA Pretraining")
    p.add_argument("--data_root", type=str, default="NTU120_skeleton/nturgbd_skeletons")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--predictor_hidden", type=int, default=512)
    p.add_argument("--context_frames", type=int, default=16)
    p.add_argument("--target_frames", type=int, default=16)
    p.add_argument("--gap_frames", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema_start", type=float, default=0.996)
    p.add_argument("--ema_end", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.02)
    p.add_argument("--max_files", type=int, default=None, help="Limit files for quick test")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, device):
    model.train()
    model.context_encoder.train()
    total_loss = 0.0
    for step, (ctx, tgt) in enumerate(loader):
        ctx = ctx.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()
        z_pred, z_tgt = model(ctx, tgt)

        # Cosine distance loss
        z_pred_norm = F.normalize(z_pred, dim=-1)
        z_tgt_norm = F.normalize(z_tgt, dim=-1)
        loss = 2.0 - 2.0 * (z_pred_norm * z_tgt_norm).sum(dim=-1).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.context_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for ctx, tgt in loader:
        ctx = ctx.to(device)
        tgt = tgt.to(device)
        z_pred, z_tgt = model(ctx, tgt)
        z_pred_norm = F.normalize(z_pred, dim=-1)
        z_tgt_norm = F.normalize(z_tgt, dim=-1)
        loss = 2.0 - 2.0 * (z_pred_norm * z_tgt_norm).sum(dim=-1).mean()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[NTU-JEPA] Device: {device}")
    print(f"[NTU-JEPA] Config: d={args.d_model}, L={args.n_layers}, ctx={args.context_frames}, gap={args.gap_frames}, tgt={args.target_frames}")
    print(f"[NTU-JEPA] Horizon: {args.gap_frames*40}-{(args.gap_frames+args.target_frames)*40}ms into future")

    # Data
    train_loader, val_loader = create_ntu_jepa_loaders(
        data_root=args.data_root,
        num_context_frames=args.context_frames,
        num_target_frames=args.target_frames,
        gap_frames=args.gap_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        max_files=args.max_files,
        seed=args.seed,
    )
    print(f"[NTU-JEPA] Train: ~{len(train_loader.dataset)} windows, Val: ~{len(val_loader.dataset)} windows")
    print(f"[NTU-JEPA] Batches/epoch: {len(train_loader)}")

    # Model
    model = create_skeleton_jepa(
        d_model=args.d_model,
        n_head=args.n_head,
        n_layers=args.n_layers,
        dropout=args.dropout,
        predictor_hidden=args.predictor_hidden,
    )
    model = model.to(device)
    trainable = sum(p.numel() for p in model.context_encoder.parameters()) + \
                sum(p.numel() for p in model.predictor.parameters())
    print(f"[NTU-JEPA] Trainable params: {trainable:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        ema = args.ema_start + (args.ema_end - args.ema_start) * (epoch / args.epochs)
        model.update_target_encoder(momentum=ema)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"[NTU-JEPA Epoch {epoch+1:3d}/{args.epochs}] "
              f"T-loss={train_loss:.6f} V-loss={val_loss:.6f} "
              f"EMA={ema:.4f} [{elapsed:.0f}s]")

        if val_loss < best_loss:
            best_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, "ntu_jepa_best.pt")
            torch.save({
                "epoch": epoch + 1,
                "context_encoder": model.context_encoder.state_dict(),
                "predictor": model.predictor.state_dict(),
                "target_encoder": model.target_encoder.state_dict(),
                "best_loss": best_loss,
                "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
            }, best_path)
            print(f"          [*] Best saved (V-loss={best_loss:.6f})")

    # Final save
    final_path = os.path.join(args.checkpoint_dir, "ntu_jepa_final.pt")
    torch.save({
        "epoch": args.epochs,
        "context_encoder": model.context_encoder.state_dict(),
        "predictor": model.predictor.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
    }, final_path)
    print(f"[NTU-JEPA] Final: {final_path}")


if __name__ == "__main__":
    main()
