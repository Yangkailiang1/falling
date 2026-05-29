"""
AV-JEPA Training on Le2i Dataset
---------------------------------
Trains the AV-JEPA world model on Le2i normal activity videos (unsupervised).
Evaluates fall detection via prediction error (surprise score).

Usage:
    python train_le2i.py --epochs 50 --batch_size 8 --device cpu
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from config import AVJEPAConfig, EncoderConfig, JEPAConfig, TrainingConfig
from jepa_model import AVJEPA
from le2i_dataset import Le2iDataset, create_le2i_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train AV-JEPA on Le2i")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--data_root", type=str,
                        default="/home/yangkailiang/.cache/kagglehub/datasets/tuyenldvn/falldataset-imvia/versions/2")
    parser.add_argument("--save_path", type=str,
                        default="checkpoints/av_jepa_le2i.pt")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--clip_duration", type=float, default=2.0)
    parser.add_argument("--context_duration", type=float, default=2.0)
    parser.add_argument("--target_gap", type=float, default=1.0)
    return parser.parse_args()


def filter_valid_samples(batch):
    """Filter out dummy samples (label=-1) that failed to decode."""
    ctx_f, ctx_a, tgt_f, tgt_a, labels = batch
    valid_mask = labels >= 0
    if not valid_mask.any():
        return None
    return (
        ctx_f[valid_mask],
        ctx_a[valid_mask],
        tgt_f[valid_mask],
        tgt_a[valid_mask],
        labels[valid_mask],
    )


def compute_audio_features(model, audio_tensor, device="cpu"):
    """Encode real audio using CLAP encoder (frozen)."""
    B = audio_tensor.shape[0]
    audio_encoder = model.encoder.audio_encoder

    with torch.no_grad():
        # Check if audio is valid (not dummy zeros)
        audio_np = audio_tensor.cpu().numpy()
        if audio_np.std() < 1e-6:
            # No real audio — return small noise to avoid dead input
            return torch.randn(B, audio_encoder.embed_dim, device=device) * 0.01

        # Preprocess with CLAP processor (handles resampling to 48kHz)
        inputs = audio_encoder.processor(
            audios=list(audio_np),
            sampling_rate=48000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        features = audio_encoder.clap_model.get_audio_features(**inputs)
        return audio_encoder.proj(features)


def train_epoch(model, dataloader, optimizer, device, epoch, total_steps, total_steps_all):
    """Train one epoch on Le2i data."""
    model.train()
    metrics = {"loss": 0, "pred_loss": 0, "sigreg": 0}
    n_valid = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Filter corrupt samples
        batch = filter_valid_samples(batch)
        if batch is None:
            continue

        ctx_f, ctx_a, tgt_f, tgt_a, labels = batch
        B = ctx_f.shape[0]
        if B == 0:
            continue

        ctx_f = ctx_f.to(device)
        ctx_a = ctx_a.to(device)
        tgt_f = tgt_f.to(device)
        tgt_a = tgt_a.to(device)

        # ── Get context embeddings via frozen encoders ──
        with torch.no_grad():
            ctx_v_emb = model.encoder.video_encoder(ctx_f)  # (B, 512)
            # Audio: placeholder until decoding is fixed
            ctx_a_emb = compute_audio_features(model, ctx_a, device=device)
            ctx_raw = torch.cat([ctx_v_emb, ctx_a_emb], dim=-1)  # (B, 1024)

            tgt_v_emb = model.encoder.video_encoder(tgt_f)
            tgt_a_emb = compute_audio_features(model, tgt_a, device=device)
            tgt_raw = torch.cat([tgt_v_emb, tgt_a_emb], dim=-1)

        # ── JEPA forward ──
        z_ctx = model.projector(ctx_raw)
        z_pred = model.predictor(z_ctx)
        z_target = model.target_projector(tgt_raw)

        loss, loss_dict = model.compute_loss(z_pred, z_target, z_ctx)

        # ── Backward ──
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), model.config.training.max_grad_norm
        )
        optimizer.step()

        model.update_target_encoder(total_steps, total_steps_all)

        metrics["loss"] += loss_dict.get("total_loss", loss_dict.get("pred_loss", 0)) * B
        metrics["pred_loss"] += loss_dict.get("pred_loss", 0) * B
        metrics["sigreg"] += loss_dict.get("sigreg_loss", 0) * B
        n_valid += B
        total_steps += 1

        pbar.set_postfix({
            "loss": f"{loss_dict['total_loss']:.4f}",
            "pred": f"{loss_dict['pred_loss']:.4f}",
        })

    for k in metrics:
        metrics[k] /= max(n_valid, 1)
    metrics["n_valid"] = n_valid
    return metrics, total_steps


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate prediction error on normal vs. fall clips, per modality."""
    model.eval()
    results = {"normal": {"video": [], "audio": [], "joint": []},
               "fall":   {"video": [], "audio": [], "joint": []}}
    v_dim = model.config.encoder.video_embed_dim

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = filter_valid_samples(batch)
        if batch is None:
            continue

        ctx_f, ctx_a, tgt_f, tgt_a, labels = batch
        B = ctx_f.shape[0]
        if B == 0:
            continue

        ctx_f, ctx_a = ctx_f.to(device), ctx_a.to(device)
        tgt_f, tgt_a = tgt_f.to(device), tgt_a.to(device)

        ctx_v_emb = model.encoder.video_encoder(ctx_f)
        ctx_a_emb = compute_audio_features(model, ctx_a, device=device)
        ctx_raw = torch.cat([ctx_v_emb, ctx_a_emb], dim=-1)

        tgt_v_emb = model.encoder.video_encoder(tgt_f)
        tgt_a_emb = compute_audio_features(model, tgt_a, device=device)
        tgt_raw = torch.cat([tgt_v_emb, tgt_a_emb], dim=-1)

        # Per-modality errors (raw encoder output change)
        video_err = F.mse_loss(ctx_v_emb, tgt_v_emb, reduction="none").mean(dim=-1)
        audio_err = F.mse_loss(ctx_a_emb, tgt_a_emb, reduction="none").mean(dim=-1)

        # Joint JEPA error
        z_ctx = model.projector(ctx_raw)
        z_pred = model.predictor(z_ctx)
        z_target = model.target_projector(tgt_raw)
        joint_err = F.mse_loss(z_pred, z_target, reduction="none").mean(dim=-1)

        for i, label in enumerate(labels):
            bucket = "normal" if label == 0 else "fall"
            results[bucket]["video"].append(video_err[i].item())
            results[bucket]["audio"].append(audio_err[i].item())
            results[bucket]["joint"].append(joint_err[i].item())

    def stats(arr):
        arr = np.array(arr)
        return arr.mean(), arr.std() if len(arr) > 1 else 0

    return {
        "normal": {k: stats(v) for k, v in results["normal"].items()},
        "fall":   {k: stats(v) for k, v in results["fall"].items()},
        "separation": {
            m: (stats(results["fall"][m])[0] - stats(results["normal"][m])[0])
               / (stats(results["normal"][m])[1] + 1e-8)
            for m in ["video", "audio", "joint"]
        },
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("AV-JEPA Training on Le2i Dataset")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch:  {args.batch_size}")

    # ── Config ──
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

    # ── Data ──
    train_loader, eval_loader = create_le2i_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        clip_duration=args.clip_duration,
        context_duration=args.context_duration,
        target_gap=args.target_gap,
    )

    total_steps_all = len(train_loader) * args.epochs

    # ── Model ──
    model = AVJEPA(config).to(device)
    trainable = list(model.projector.parameters()) + list(model.predictor.parameters())
    param_count = sum(p.numel() for p in trainable)
    print(f"\nTrainable params: {param_count:,}")
    print(f"Training clips:   {len(train_loader.dataset)}")
    print(f"Eval clips:       {len(eval_loader.dataset)}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=config.training.weight_decay,
    )

    # ── Training ──
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    total_steps = 0
    best_separation = -float("inf")

    for epoch in range(1, args.epochs + 1):
        metrics, total_steps = train_epoch(
            model, train_loader, optimizer, device, epoch,
            total_steps, total_steps_all
        )

        print(f"  Epoch {epoch:3d} | loss={metrics['loss']:.4f} "
              f"pred={metrics['pred_loss']:.4f} sigreg={metrics['sigreg']:.4f} "
              f"samples={metrics['n_valid']}")

        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            print("  --- Evaluating ---")
            eval_metrics = evaluate(model, eval_loader, device)
            sep = eval_metrics["separation"]
            print(f"    {'':>10} {'Video':>12} {'Audio':>12} {'Joint':>12}")
            print(f"    {'Normal':>10} {eval_metrics['normal']['video'][0]:>12.4f} {eval_metrics['normal']['audio'][0]:>12.4f} {eval_metrics['normal']['joint'][0]:>12.4f}")
            print(f"    {'Fall':>10} {eval_metrics['fall']['video'][0]:>12.4f} {eval_metrics['fall']['audio'][0]:>12.4f} {eval_metrics['fall']['joint'][0]:>12.4f}")
            print(f"    {'Sep σ':>10} {sep['video']:>12.1f} {sep['audio']:>12.1f} {sep['joint']:>12.1f}")
            print()

            if sep['joint'] > best_separation:
                best_separation = sep['joint']
                os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else ".", exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "eval_metrics": eval_metrics,
                }, args.save_path)
                print(f"    ✅ Best model saved (separation={best_separation:.1f}σ)")

    # ── Final Evaluation ──
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    eval_metrics = evaluate(model, eval_loader, device)
    print(f"  {'':>10} {'Video':>12} {'Audio':>12} {'Joint':>12}")
    print(f"  {'Normal':>10} {eval_metrics['normal']['video'][0]:>12.4f} {eval_metrics['normal']['audio'][0]:>12.4f} {eval_metrics['normal']['joint'][0]:>12.4f}")
    print(f"  {'Fall':>10} {eval_metrics['fall']['video'][0]:>12.4f} {eval_metrics['fall']['audio'][0]:>12.4f} {eval_metrics['fall']['joint'][0]:>12.4f}")
    sep = eval_metrics["separation"]
    print(f"  {'Sep σ':>10} {sep['video']:>12.1f} {sep['audio']:>12.1f} {sep['joint']:>12.1f}")

    if eval_metrics['fall']['joint'][0] > eval_metrics['normal']['joint'][0] + 1.0 * eval_metrics['normal']['joint'][1]:
        print("  ✅ Fall events produce higher prediction error — world model works!")
    else:
        print("  ⚠️  Separation weak — try more epochs, different hyperparams, or fix audio")

    # Save final
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "epoch": args.epochs,
        "eval_metrics": eval_metrics,
    }, args.save_path.replace(".pt", "_final.pt"))
    print(f"\nFinal model saved to {args.save_path.replace('.pt', '_final.pt')}")


if __name__ == "__main__":
    main()
