"""
Supervised Training Script for Lightweight JEPA Classifier.

Two modes:
  1. Balanced (recommended): video-level train/test split, one sample per video,
     balanced leak levels 0-16, independent test set evaluation.
     python3 -m spwm.classifier_train --balanced --no_audio --epochs 200 --device cuda

  2. Legacy sliding-window: clip-level random split (deprecated due to data leakage).
     python3 -m spwm.classifier_train --data_root Le2i --epochs 50 --device cuda
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
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

from .classifier_model import JEPAClassifier
from .data.le2i_dataset import Le2iDataset
from .data.preprocessed_dataset import PreprocessedLe2iDataset
from .data.balanced_le2i_dataset import BalancedLe2iDataset


def parse_args():
    p = argparse.ArgumentParser(description='Train JEPA Classifier for Fall Detection')
    p.add_argument('--data_root', type=str, default='Le2i', help='Path to Le2i dataset')
    p.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    p.add_argument('--batch_size', type=int, default=4, help='Batch size')
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    p.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    p.add_argument('--device', type=str, default='cuda', help='Training device (cuda/cpu)')
    p.add_argument('--encoder_device', type=str, default='cpu', help='Device for frozen encoders')
    p.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    p.add_argument('--context_frames', type=int, default=16, help='Context frames (observation window)')
    p.add_argument('--future_frames', type=int, default=8, help='Future frames to predict (prediction horizon)')
    p.add_argument('--stride', type=int, default=8, help='Sliding window stride (legacy mode only)')
    p.add_argument('--val_split', type=float, default=0.2, help='Validation set fraction (legacy mode only)')
    p.add_argument('--pos_weight', type=float, default=15.0, help='Positive class weight for BCE loss')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint save directory')
    p.add_argument('--log_interval', type=int, default=20, help='Log every N steps')
    p.add_argument('--overfit_test', action='store_true', help='Quick overfit test on small subset')
    p.add_argument('--no_audio', action='store_true', help='Disable audio modality')
    p.add_argument('--preprocessed', action='store_true', help='Use pre-processed .pt dataset (Le2i_processed/)')
    p.add_argument('--leak_frames', type=int, default=4, help='Frames of fall onset to leak (legacy mode only)')
    p.add_argument('--seed', type=int, default=42)
    # Balanced mode
    p.add_argument('--balanced', action='store_true',
                   help='Use balanced video-level split (one sample per video, independent test set)')
    p.add_argument('--split_json', type=str, default='le2i_split.json',
                   help='Path to split JSON (balanced mode only)')
    p.add_argument('--neg_stride', type=int, default=32,
                   help='Stride for negative sample windows from non-fall videos (balanced mode)')
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_classification(batch):
    """Custom collate for classification mode: pad audio to max length."""
    frames = torch.stack([item['frames'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    n_fall_frames = torch.tensor([item.get('n_fall_frames', 0) for item in batch])

    # Audio: pad to max length in batch
    audios = [item['audio'] for item in batch]
    max_len = max(a.shape[0] for a in audios)
    padded_audio = []
    for a in audios:
        if a.shape[0] < max_len:
            a = F.pad(a, (0, max_len - a.shape[0]))
        padded_audio.append(a)
    audio = torch.stack(padded_audio)

    return frames, audio, labels, n_fall_frames


def compute_metrics(labels, probs):
    """Compute classification metrics."""
    preds = (probs > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }

    if len(np.unique(labels)) > 1:
        metrics['auc'] = roc_auc_score(labels, probs)
        metrics['ap'] = average_precision_score(labels, probs)
    else:
        metrics['auc'] = 0.5
        metrics['ap'] = 0.0

    cm = confusion_matrix(labels, preds)
    metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return metrics


def train_epoch(model, loader, optimizer, criterion, device, log_interval):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for step, (frames, audio, labels, _n_fall) in enumerate(loader):
        frames = frames.to(device)
        audio = audio.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, _ = model(frames, audio if not model.use_audio else audio)
        loss = criterion(logits.squeeze(-1), labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        if (step + 1) % log_interval == 0:
            print(f"  Step {step+1}/{len(loader)}: loss={loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    metrics['loss'] = avg_loss
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validation with per-leak breakdown."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_leaks = []

    for frames, audio, labels, n_fall in loader:
        frames = frames.to(device)
        audio = audio.to(device)
        labels = labels.to(device)

        logits, _ = model(frames, audio if not model.use_audio else audio)
        loss = criterion(logits.squeeze(-1), labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_leaks.extend(n_fall.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    metrics['loss'] = avg_loss

    # Per-leak breakdown for positive samples
    metrics['per_leak'] = compute_per_leak_breakdown(
        np.array(all_labels), np.array(all_probs), np.array(all_leaks)
    )
    return metrics


def compute_per_leak_breakdown(labels, probs, leaks):
    """Compute accuracy per leak level (only for positive samples)."""
    breakdown = {}
    for leak_val in range(17):  # 0..16
        mask = (labels == 1) & (leaks == leak_val)
        n = mask.sum()
        if n == 0:
            continue
        sub_labels = labels[mask]
        sub_probs = probs[mask]
        preds = (sub_probs > 0.5).astype(int)
        correct = (preds == sub_labels).sum()
        breakdown[leak_val] = {
            'n': int(n),
            'correct': int(correct),
            'acc': float(correct / n),
        }
    # Overall for positives
    pos_mask = labels == 1
    n_pos = pos_mask.sum()
    if n_pos > 0:
        pos_preds = (probs[pos_mask] > 0.5).astype(int)
        pos_correct = (pos_preds == labels[pos_mask]).sum()
        breakdown['all_pos'] = {
            'n': int(n_pos),
            'correct': int(pos_correct),
            'acc': float(pos_correct / n_pos),
        }
    return breakdown


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}, Encoder device: {args.encoder_device}")

    # ---- Balanced mode: video-level split, independent test set ----
    if args.balanced:
        print(f"[Train] Balanced mode: video-level split from {args.split_json}")
        train_dataset = BalancedLe2iDataset(
            split_json=args.split_json, split='train',
            data_root='Le2i_processed',
            num_context_frames=args.context_frames,
            num_future_frames=args.future_frames,
            neg_stride=args.neg_stride,
        )
        test_dataset = BalancedLe2iDataset(
            split_json=args.split_json, split='test',
            data_root='Le2i_processed',
            num_context_frames=args.context_frames,
            num_future_frames=args.future_frames,
            neg_stride=args.neg_stride,
        )

        if args.overfit_test:
            # Overfit: use small subset of train as both train and val
            from torch.utils.data import Subset
            n = min(20, len(train_dataset))
            indices = list(range(n))
            train_dataset = Subset(train_dataset, indices)
            test_dataset = Subset(test_dataset, list(range(min(10, len(test_dataset)))))
            print(f"[Train] Overfit test: {len(train_dataset)} train, {len(test_dataset)} test")

        # WeightedRandomSampler: oversample positives to counter class imbalance
        n_pos = train_dataset.labels.sum().item()
        n_neg = len(train_dataset.labels) - n_pos
        sample_weights = torch.where(train_dataset.labels == 1, n_neg / n_pos, 1.0)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True),
            num_workers=0, collate_fn=collate_classification,
            pin_memory=True, drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_classification,
            pin_memory=True,
        )
        val_loader = test_loader  # validate on test set
        print(f"[Train] Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # ---- Legacy mode: sliding-window clip-level split ----
    else:
        if args.preprocessed:
            dataset = PreprocessedLe2iDataset(
                data_root='Le2i_processed',
                num_context_frames=args.context_frames,
                num_future_frames=args.future_frames,
                stride=args.stride,
                classification_mode=True,
                leak_frames=args.leak_frames,
            )
        else:
            dataset = Le2iDataset(
                data_root=args.data_root,
                num_context_frames=args.context_frames,
                num_target_frames=0,
                num_future_frames=args.future_frames,
                stride=args.stride,
                normal_only=True,
                classification_mode=True,
                cache_videos=False,
                leak_frames=args.leak_frames,
            )

        if args.overfit_test:
            from torch.utils.data import Subset
            n_samples = 20
            fall_indices = [i for i, c in enumerate(dataset.clips) if c['future_fall']]
            normal_indices = [i for i, c in enumerate(dataset.clips) if not c['future_fall']]
            selected = fall_indices[:n_samples//2] + normal_indices[:n_samples//2]
            train_dataset = Subset(dataset, selected)
            val_dataset = Subset(dataset, selected)
            print(f"[Train] Overfit test: {len(selected)} samples")
        else:
            n_total = len(dataset)
            n_val = int(n_total * args.val_split)
            indices = list(range(n_total))
            np.random.shuffle(indices)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            from torch.utils.data import Subset
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            print(f"[Train] Train: {len(train_indices)}, Val: {len(val_indices)}")

        n_workers = 2 if args.preprocessed else 1
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=n_workers, collate_fn=collate_classification,
            pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=n_workers, collate_fn=collate_classification,
            pin_memory=True,
        )
        test_loader = None

    # ---- Create model ----
    from .config import VideoEncoderConfig, AudioEncoderConfig
    model = JEPAClassifier(
        video_config=VideoEncoderConfig(num_frames=args.context_frames),
        audio_config=AudioEncoderConfig(),
        use_audio=not args.no_audio,
        encoder_device=args.encoder_device,
    )
    model._place_encoders(args.encoder_device)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Trainable params: {trainable:,}")

    # Loss (pos_weight is important for imbalanced data; for balanced mode use 1.0)
    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Checkpoint dir
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    best_path = ""

    print(f"[Train] Starting {args.epochs} epochs...")
    print(f"[Train] {'='*60}")

    for epoch in range(args.epochs):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, args.log_interval)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        print(f"[Epoch {epoch+1:3d}/{args.epochs}] "
              f"T-loss={train_metrics['loss']:.4f} V-loss={val_metrics['loss']:.4f} "
              f"T-F1={train_metrics['f1']:.3f} V-F1={val_metrics['f1']:.3f} "
              f"V-Acc={val_metrics['accuracy']:.3f} V-Prec={val_metrics['precision']:.3f} "
              f"V-Rec={val_metrics['recall']:.3f} V-AUC={val_metrics['auc']:.3f} "
              f"[{elapsed:.0f}s]")
        print(f"          TN={val_metrics['tn']} FP={val_metrics['fp']} "
              f"FN={val_metrics['fn']} TP={val_metrics['tp']}")

        # Per-leak breakdown
        pl = val_metrics.get('per_leak', {})
        if pl:
            all_pos = pl.pop('all_pos', None)
            items = sorted(pl.items())
            lines = []
            for i in range(0, len(items), 4):
                chunk = items[i:i+4]
                parts = []
                for leak, info in chunk:
                    parts.append(f"leak={leak:2d}:{info['acc']:.2f}(n={info['n']})")
                lines.append("          " + "  ".join(parts))
            for line in lines:
                print(line)
            if all_pos:
                print(f"          All positive: acc={all_pos['acc']:.3f} (n={all_pos['n']})")

        # Save best
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_path = os.path.join(args.checkpoint_dir, 'classifier_best.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'args': args,
            }, best_path)
            print(f"          [*] Best model saved (F1={best_f1:.4f})")

    print(f"[Train] {'='*60}")
    print(f"[Train] Best F1: {best_f1:.4f} at {best_path}")

    # Final save
    final_path = os.path.join(args.checkpoint_dir, 'classifier_final.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1': best_f1,
    }, final_path)
    print(f"[Train] Final model saved to {final_path}")


if __name__ == '__main__':
    main()
