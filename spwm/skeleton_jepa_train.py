"""
Skeleton-JEPA Training Script.

Two-phase training (same pattern as video+audio JEPA Classifier):

  Phase 1 — JEPA Pretraining (self-supervised):
    Train SkeletonEncoder to predict future skeleton states from context.
    Normal activity only. MSE loss between predictor output and target EMA encoding.
    python3 -m spwm.skeleton_jepa_train --phase jepa --epochs 100 --device cuda

  Phase 2 — Supervised Classification:
    Freeze pretrained SkeletonEncoder, train fusion MLP + classifier head.
    BCE loss on fall vs non-fall.
    python3 -m spwm.skeleton_jepa_train --phase classify --epochs 200 --device cuda

  Both phases:
    python3 -m spwm.skeleton_jepa_train --phase both --jepa_epochs 100 --cls_epochs 200 --device cuda
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

from .skeleton_jepa_model import (
    SkeletonJEPA, SkeletonClassifier, SkeletonEncoder,
    create_skeleton_jepa, create_skeleton_classifier,
)
from .data.skeleton_dataset import (
    SkeletonDataset, jepa_collate, classification_collate,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Skeleton-JEPA for Fall Detection")

    # Phase selection
    p.add_argument("--phase", type=str, default="both",
                   choices=["jepa", "classify", "both"],
                   help="Training phase")
    p.add_argument("--resume_jepa", type=str, default="",
                   help="Path to pretrained JEPA checkpoint for classification phase")

    # Data
    p.add_argument("--data_root", type=str, default="le2i_keypoints")
    p.add_argument("--split_json", type=str, default="le2i_split.json")
    p.add_argument("--context_frames", type=int, default=16,
                   help="Context window size (frames, 0.64s @ 25fps)")
    p.add_argument("--target_frames", type=int, default=16,
                   help="Target window size for JEPA (frames)")
    p.add_argument("--future_frames", type=int, default=32,
                   help="Prediction horizon for classification (frames, 1.28s @ 25fps)")
    p.add_argument("--gap_frames", type=int, default=16,
                   help="Gap between context and target (frames, ≥16 for longer prediction)")
    p.add_argument("--neg_stride", type=int, default=16,
                   help="Stride for negative sample windows from non-fall videos")

    # JEPA pretraining
    p.add_argument("--jepa_epochs", type=int, default=100)
    p.add_argument("--jepa_batch_size", type=int, default=16)
    p.add_argument("--jepa_lr", type=float, default=1e-4)
    p.add_argument("--ema_start", type=float, default=0.996,
                   help="EMA momentum for target encoder update")
    p.add_argument("--ema_end", type=float, default=1.0,
                   help="Final EMA momentum (1.0 = no update)")

    # Model hyperparams
    p.add_argument("--d_model", type=int, default=256,
                   help="Transformer hidden dimension")
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--predictor_hidden", type=int, default=512)

    # Classifier training
    p.add_argument("--cls_epochs", type=int, default=200)
    p.add_argument("--cls_batch_size", type=int, default=8)
    p.add_argument("--cls_lr", type=float, default=1e-4)
    p.add_argument("--pos_weight", type=float, default=5.0,
                   help="Positive class weight for BCE (depends on neg_stride)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--finetune_encoder", action="store_true",
                   help="Fine-tune encoder with small LR during classification")
    p.add_argument("--finetune_lr", type=float, default=1e-5,
                   help="Learning rate for encoder fine-tuning")
    p.add_argument("--fusion_hidden", type=int, default=512)
    p.add_argument("--fusion_out", type=int, default=256)

    # General
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--overfit_test", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_normalize", action="store_true",
                   help="Disable keypoint normalization")
    p.add_argument("--no_augment", action="store_true",
                   help="Disable data augmentation")

    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(labels, probs):
    """Compute classification metrics."""
    preds = (probs > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

    if len(np.unique(labels)) > 1:
        metrics["auc"] = roc_auc_score(labels, probs)
        metrics["ap"] = average_precision_score(labels, probs)
    else:
        metrics["auc"] = 0.5
        metrics["ap"] = 0.0

    cm = confusion_matrix(labels, preds)
    if cm.size == 4:
        metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()
    else:
        metrics["tn"] = metrics["fp"] = metrics["fn"] = metrics["tp"] = 0

    return metrics


def compute_per_leak_breakdown(labels, probs, leaks):
    """Compute accuracy per leak level."""
    breakdown = {}
    for leak_val in range(17):
        mask = (labels == 1) & (leaks == leak_val)
        n = mask.sum()
        if n == 0:
            continue
        preds = (probs[mask] > 0.5).astype(int)
        correct = (preds == labels[mask]).sum()
        breakdown[leak_val] = {"n": int(n), "correct": int(correct), "acc": float(correct / n)}

    pos_mask = labels == 1
    n_pos = pos_mask.sum()
    if n_pos > 0:
        pos_preds = (probs[pos_mask] > 0.5).astype(int)
        pos_correct = (pos_preds == labels[pos_mask]).sum()
        breakdown["all_pos"] = {"n": int(n_pos), "correct": int(pos_correct), "acc": float(pos_correct / n_pos)}
    return breakdown


# ============================================================================
# Phase 1: JEPA Pretraining
# ============================================================================

def train_jepa_epoch(model, loader, optimizer, device):
    """Train one JEPA epoch."""
    model.train()
    model.context_encoder.train()
    total_loss = 0.0

    for step, (ctx_kp, tgt_kp) in enumerate(loader):
        ctx_kp = ctx_kp.to(device)
        tgt_kp = tgt_kp.to(device)

        optimizer.zero_grad()

        z_pred, z_tgt = model(ctx_kp, tgt_kp)

        # MSE loss in latent space (L2 normalized for stability)
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
def validate_jepa(model, loader, device):
    """Validate JEPA prediction error."""
    model.eval()
    total_loss = 0.0

    for ctx_kp, tgt_kp in loader:
        ctx_kp = ctx_kp.to(device)
        tgt_kp = tgt_kp.to(device)

        z_pred, z_tgt = model(ctx_kp, tgt_kp)
        z_pred_norm = F.normalize(z_pred, dim=-1)
        z_tgt_norm = F.normalize(z_tgt, dim=-1)
        loss = 2.0 - 2.0 * (z_pred_norm * z_tgt_norm).sum(dim=-1).mean()
        total_loss += loss.item()

    return total_loss / len(loader)


def run_jepa_pretraining(args, device):
    """JEPA self-supervised pretraining on normal activity only."""
    print("=" * 65)
    print("[JEPA Phase 1] Self-supervised pretraining on normal activity")
    print(f"  Context: {args.context_frames} frames  |  Gap: {args.gap_frames} frames  |  Target: {args.target_frames} frames")
    print(f"  Prediction horizon: {args.gap_frames * 40:.0f}-{(args.gap_frames + args.target_frames) * 40:.0f}ms into the future")
    print("=" * 65)

    # Datasets
    train_ds = SkeletonDataset(
        data_root=args.data_root, split_json=args.split_json,
        split="train", mode="jepa",
        num_context_frames=args.context_frames,
        num_target_frames=args.target_frames,
        gap_frames=args.gap_frames,
        normalize=not args.no_normalize,
        augment=not args.no_augment,
        jepa_normal_only=True,
        seed=args.seed,
    )
    test_ds = SkeletonDataset(
        data_root=args.data_root, split_json=args.split_json,
        split="test", mode="jepa",
        num_context_frames=args.context_frames,
        num_target_frames=args.target_frames,
        gap_frames=args.gap_frames,
        normalize=not args.no_normalize,
        jepa_normal_only=True,
        seed=args.seed,
    )

    if args.overfit_test:
        from torch.utils.data import Subset
        n = min(40, len(train_ds))
        train_ds = Subset(train_ds, range(n))
        test_ds = Subset(train_ds, range(n))  # overfit: same data for train/val
        print(f"[JEPA] Overfit test: {len(train_ds)} train, {len(test_ds)} val (same data)")

    print(f"[JEPA] Train samples: {len(train_ds)}, Val samples: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.jepa_batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=jepa_collate,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        test_ds, batch_size=args.jepa_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=jepa_collate,
        pin_memory=True,
    )

    # Model
    model = create_skeleton_jepa(
        d_model=args.d_model, n_head=args.n_head, n_layers=args.n_layers,
        dropout=args.dropout, predictor_hidden=args.predictor_hidden,
    )
    model = model.to(device)
    trainable = sum(p.numel() for p in model.context_encoder.parameters()) + \
                sum(p.numel() for p in model.predictor.parameters())
    print(f"[JEPA] Trainable params: {trainable:,}")

    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.jepa_lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.jepa_epochs)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.jepa_epochs):
        t0 = time.time()

        train_loss = train_jepa_epoch(model, train_loader, optimizer, device)
        val_loss = validate_jepa(model, val_loader, device)

        # EMA update
        ema = args.ema_start + (args.ema_end - args.ema_start) * (epoch / args.jepa_epochs)
        model.update_target_encoder(momentum=ema)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"[JEPA Epoch {epoch+1:3d}/{args.jepa_epochs}] "
              f"T-loss={train_loss:.6f} V-loss={val_loss:.6f} EMA={ema:.4f} [{elapsed:.1f}s]")

        if val_loss < best_loss:
            best_loss = val_loss
            jepa_path = os.path.join(args.checkpoint_dir, "skeleton_jepa_best.pt")
            torch.save({
                "epoch": epoch + 1,
                "context_encoder": model.context_encoder.state_dict(),
                "predictor": model.predictor.state_dict(),
                "target_encoder": model.target_encoder.state_dict(),
                "best_loss": best_loss,
                "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
            }, jepa_path)
            print(f"          [*] Best JEPA model saved (loss={best_loss:.6f})")

    # Final save
    final_path = os.path.join(args.checkpoint_dir, "skeleton_jepa_final.pt")
    torch.save({
        "epoch": args.jepa_epochs,
        "context_encoder": model.context_encoder.state_dict(),
        "predictor": model.predictor.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
    }, final_path)
    print(f"[JEPA] Final model: {final_path}")

    return model


# ============================================================================
# Phase 2: Supervised Classification
# ============================================================================

def train_cls_epoch(model, loader, optimizer, criterion, device, log_interval):
    """Train one classification epoch."""
    model.train()
    model.fusion.train()
    model.classifier.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for step, (ctx_kp, labels, _leaks) in enumerate(loader):
        ctx_kp = ctx_kp.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(ctx_kp)
        loss = criterion(logits.squeeze(-1), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_probs.extend(probs.tolist() if hasattr(probs, "tolist") else [float(probs)])
            all_labels.extend(labels.cpu().numpy().tolist())

        if (step + 1) % log_interval == 0:
            print(f"  Step {step+1}/{len(loader)}: loss={loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def validate_cls(model, loader, criterion, device):
    """Validate classification with per-leak breakdown."""
    model.eval()
    total_loss = 0.0
    all_labels, all_probs, all_leaks = [], [], []

    for ctx_kp, labels, leaks in loader:
        ctx_kp = ctx_kp.to(device)
        labels = labels.to(device)

        logits, _ = model(ctx_kp)
        loss = criterion(logits.squeeze(-1), labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_probs.extend(probs.tolist() if hasattr(probs, "tolist") else [float(probs)])
        all_labels.extend(labels.cpu().numpy().tolist())
        all_leaks.extend(leaks.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    metrics["loss"] = avg_loss
    metrics["per_leak"] = compute_per_leak_breakdown(
        np.array(all_labels), np.array(all_probs), np.array(all_leaks)
    )
    return metrics


def run_classifier_training(args, device, encoder_state_dict=None):
    """Supervised classifier training on frozen JEPA encoder."""
    print("=" * 65)
    print("[Classifier Phase 2] Supervised training on frozen SkeletonEncoder")
    print(f"  Context: {args.context_frames} frames  |  Prediction horizon: {args.future_frames} frames ({args.future_frames * 40:.0f}ms)")
    print("=" * 65)

    # Datasets
    train_ds = SkeletonDataset(
        data_root=args.data_root, split_json=args.split_json,
        split="train", mode="classification",
        num_context_frames=args.context_frames,
        num_future_frames=args.future_frames,
        neg_stride=args.neg_stride,
        normalize=not args.no_normalize,
        augment=not args.no_augment,
        seed=args.seed,
    )
    test_ds = SkeletonDataset(
        data_root=args.data_root, split_json=args.split_json,
        split="test", mode="classification",
        num_context_frames=args.context_frames,
        num_future_frames=args.future_frames,
        neg_stride=args.neg_stride,
        normalize=not args.no_normalize,
        augment=False,
        seed=args.seed,
    )

    if args.overfit_test:
        from torch.utils.data import Subset
        n = min(40, len(train_ds))
        train_ds = Subset(train_ds, range(n))
        test_ds = Subset(train_ds, range(n))  # overfit: same data for train/val
        print(f"[Classifier] Overfit test: {len(train_ds)} train, {len(test_ds)} val (same data)")

    # Compute class balance
    labels = torch.tensor([train_ds[i]["label"].item() for i in range(len(train_ds))])
    n_pos = int(labels.sum().item())
    n_neg = len(labels) - n_pos
    print(f"[Classifier] Train: {len(train_ds)} ({n_pos} pos, {n_neg} neg), Test: {len(test_ds)}")

    # WeightedRandomSampler for imbalanced data
    sample_weights = torch.where(labels == 1, float(n_neg) / max(n_pos, 1), 1.0)
    train_loader = DataLoader(
        train_ds, batch_size=args.cls_batch_size,
        sampler=WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True),
        num_workers=args.num_workers, collate_fn=classification_collate,
        pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.cls_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=classification_collate,
        pin_memory=True,
    )

    # Build encoder and load pretrained weights if provided
    encoder = SkeletonEncoder(
        num_keypoints=17, d_model=args.d_model,
        n_head=args.n_head, n_layers=args.n_layers,
        dropout=args.dropout,
    )
    if encoder_state_dict is not None:
        encoder.load_state_dict(encoder_state_dict, strict=True)
        print("[Classifier] Loaded pretrained encoder weights from JEPA phase")

    # Optionally unfreeze encoder for fine-tuning
    if args.finetune_encoder:
        for p in encoder.parameters():
            p.requires_grad = True
        encoder.train()
        print(f"[Classifier] Encoder UNFROZEN for fine-tuning (LR={args.finetune_lr})")

    model = create_skeleton_classifier(
        encoder, d_model=args.d_model,
        fusion_hidden=args.fusion_hidden, fusion_out=args.fusion_out,
        dropout=args.dropout,
    )
    model = model.to(device)

    # If fine-tuning, the encoder was already unfrozen; create_skeleton_classifier
    # would have frozen it again, so we need to re-unfreeze
    if args.finetune_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = True
    print(f"[Classifier] Trainable params: {model.trainable_params():,}")

    auto_pos_weight = float(n_neg) / max(n_pos, 1) if n_pos > 0 else 1.0
    pos_weight = torch.tensor([args.pos_weight if args.pos_weight > 0 else auto_pos_weight], device=device)
    print(f"[Classifier] pos_weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if args.finetune_encoder:
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": args.finetune_lr},
            {"params": model.fusion.parameters(), "lr": args.cls_lr},
            {"params": model.classifier.parameters(), "lr": args.cls_lr},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.cls_lr, weight_decay=args.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cls_epochs)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(args.cls_epochs):
        t0 = time.time()

        train_metrics = train_cls_epoch(model, train_loader, optimizer, criterion, device, args.log_interval)
        val_metrics = validate_cls(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        print(f"[Cls Epoch {epoch+1:3d}/{args.cls_epochs}] "
              f"T-loss={train_metrics['loss']:.4f} V-loss={val_metrics['loss']:.4f} "
              f"T-F1={train_metrics['f1']:.3f} V-F1={val_metrics['f1']:.3f} "
              f"V-Acc={val_metrics['accuracy']:.3f} V-AUC={val_metrics['auc']:.3f} "
              f"[{elapsed:.0f}s]")
        print(f"          TN={val_metrics['tn']} FP={val_metrics['fp']} "
              f"FN={val_metrics['fn']} TP={val_metrics['tp']}")

        pl = val_metrics.get("per_leak", {})
        if pl:
            all_pos = pl.pop("all_pos", None)
            items = sorted(pl.items())
            for i in range(0, len(items), 4):
                chunk = items[i:i+4]
                parts = [f"leak={leak:2d}:{info['acc']:.2f}(n={info['n']})" for leak, info in chunk]
                print("          " + "  ".join(parts))
            if all_pos:
                print(f"          All positive: acc={all_pos['acc']:.3f} (n={all_pos['n']})")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_path = os.path.join(args.checkpoint_dir, "skeleton_classifier_best.pt")
            torch.save({
                "epoch": epoch + 1,
                "encoder_state_dict": encoder.state_dict(),
                "fusion_state_dict": model.fusion.state_dict(),
                "classifier_state_dict": model.classifier.state_dict(),
                "best_f1": best_f1,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
            }, best_path)
            print(f"          [*] Best classifier saved (F1={best_f1:.4f})")

    print(f"[Classifier] Best F1: {best_f1:.4f}")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    print(f"[Train] Phase: {args.phase}")

    encoder_state_dict = None

    if args.phase in ("jepa", "both"):
        jepa_model = run_jepa_pretraining(args, device)
        encoder_state_dict = jepa_model.context_encoder.state_dict()

    if args.phase in ("classify", "both"):
        if args.resume_jepa and encoder_state_dict is None:
            ckpt = torch.load(args.resume_jepa, map_location="cpu")
            encoder_state_dict = ckpt.get("context_encoder", ckpt)
            print(f"[Classifier] Loaded JEPA checkpoint from {args.resume_jepa}")

        run_classifier_training(args, device, encoder_state_dict)


if __name__ == "__main__":
    main()
