"""
AV-JEPA v2 Training Script
---------------------------
Trains the cross-modal JEPA v2 model on Le2i features.

Two modes:
  1. Feature-based (default): pre-extracted V-JEPA 2 + WavJEPA features
  2. End-to-end: load encoders and process raw video+audio

Usage:
    # Feature-based (recommended)
    python v2_train.py --feature_cache features_v2/ --epochs 100 --device cuda

    # End-to-end
    python v2_train.py --data_root /path/to/Le2i --epochs 100 --device cuda
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))


# ─── Feature Dataset ─────────────────────────────────────────────────────

class FeatureDataset(Dataset):
    """Dataset for pre-extracted V-JEPA 2 + WavJEPA features."""

    def __init__(self, feature_list):
        self.features = feature_list

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        return (
            f["v_ctx"], f["a_ctx"],
            f["v_tgt"], f["a_tgt"],
            torch.tensor(f["label"], dtype=torch.long),
        )


def collate_variable_tokens(batch):
    """
    Collate batches with variable-length token sequences.
    Pads to max length in batch.
    """
    v_ctx_list, a_ctx_list = [], []
    v_tgt_list, a_tgt_list = [], []
    labels = []

    v_ctx_max = max(b[0].shape[0] for b in batch)
    a_ctx_max = max(b[1].shape[0] for b in batch)
    v_tgt_max = max(b[2].shape[0] for b in batch)
    a_tgt_max = max(b[3].shape[0] for b in batch)

    for v_ctx, a_ctx, v_tgt, a_tgt, label in batch:
        # Pad video context
        if v_ctx.shape[0] < v_ctx_max:
            pad = torch.zeros(v_ctx_max - v_ctx.shape[0], v_ctx.shape[1])
            v_ctx = torch.cat([v_ctx, pad], dim=0)
        v_ctx_list.append(v_ctx.unsqueeze(0))

        # Pad audio context
        if a_ctx.shape[0] < a_ctx_max:
            pad = torch.zeros(a_ctx_max - a_ctx.shape[0], a_ctx.shape[1])
            a_ctx = torch.cat([a_ctx, pad], dim=0)
        a_ctx_list.append(a_ctx.unsqueeze(0))

        # Pad targets
        if v_tgt.shape[0] < v_tgt_max:
            pad = torch.zeros(v_tgt_max - v_tgt.shape[0], v_tgt.shape[1])
            v_tgt = torch.cat([v_tgt, pad], dim=0)
        v_tgt_list.append(v_tgt.unsqueeze(0))

        if a_tgt.shape[0] < a_tgt_max:
            pad = torch.zeros(a_tgt_max - a_tgt.shape[0], a_tgt.shape[1])
            a_tgt = torch.cat([a_tgt, pad], dim=0)
        a_tgt_list.append(a_tgt.unsqueeze(0))

        labels.append(label)

    return (
        torch.cat(v_ctx_list, dim=0),
        torch.cat(a_ctx_list, dim=0),
        torch.cat(v_tgt_list, dim=0),
        torch.cat(a_tgt_list, dim=0),
        torch.tensor(labels),
    )


# ─── Training Loop ───────────────────────────────────────────────────────

def train_epoch_feature_based(model, dataloader, optimizer, device, epoch, total_steps, total_steps_all):
    """Train one epoch on pre-extracted features."""
    model.train()
    metrics = {"loss": 0, "pred_loss": 0, "sigreg": 0}
    n_total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        v_ctx, a_ctx, v_tgt, a_tgt, labels = batch
        B = v_ctx.shape[0]
        v_ctx = v_ctx.to(device)
        a_ctx = a_ctx.to(device)
        v_tgt = v_tgt.to(device)
        a_tgt = a_tgt.to(device)
        labels = labels.to(device)

        # ── Encode context ──
        v_c = model.video_adapter(v_ctx)
        v_c = model.video_pooler(v_c)
        a_c = model.audio_adapter(a_ctx)
        a_c = model.audio_pooler(a_c)
        z_ctx = model.fusion(v_c, a_c)

        # ── Predict ──
        z_pred = model.predictor(z_ctx)

        # ── Target (EMA) ──
        with torch.no_grad():
            v_t = model.target_video_adapter(v_tgt)
            v_t = model.target_video_pooler(v_t)
            a_t = model.target_audio_adapter(a_tgt)
            a_t = model.target_audio_pooler(a_t)
            z_target = model.target_fusion(v_t, a_t)

        # ── Loss ──
        loss, loss_dict = model.compute_loss(z_pred, z_target, z_ctx)

        # ── Backward ──
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.update_target_encoder(total_steps, total_steps_all)

        n_total += B
        metrics["loss"] += loss_dict["total_loss"] * B
        metrics["pred_loss"] += loss_dict["pred_loss"] * B
        metrics["sigreg"] += loss_dict.get("sigreg_loss", 0) * B
        total_steps += 1

        pbar.set_postfix({
            "loss": f"{loss_dict['total_loss']:.4f}",
            "pred": f"{loss_dict['pred_loss']:.4f}",
        })

    for k in metrics:
        metrics[k] /= max(n_total, 1)
    return metrics, total_steps


# ─── Evaluation ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_feature_based(model, dataloader, device):
    """Evaluate prediction error on normal vs. fall clips."""
    model.eval()
    results = {"normal": {"video": [], "audio": [], "joint": []},
               "fall":   {"video": [], "audio": [], "joint": []}}

    for batch in tqdm(dataloader, desc="Evaluating"):
        v_ctx, a_ctx, v_tgt, a_tgt, labels = batch
        B = v_ctx.shape[0]
        v_ctx = v_ctx.to(device)
        a_ctx = a_ctx.to(device)
        v_tgt = v_tgt.to(device)
        a_tgt = a_tgt.to(device)
        labels = labels.to(device)

        # Video-only raw error
        video_err = F.mse_loss(v_ctx, v_tgt, reduction="none").mean(dim=(-1, -2))

        # Audio-only raw error
        audio_err = F.mse_loss(a_ctx, a_tgt, reduction="none").mean(dim=(-1, -2))

        # Joint JEPA error
        v_c = model.video_adapter(v_ctx)
        v_c = model.video_pooler(v_c)
        a_c = model.audio_adapter(a_ctx)
        a_c = model.audio_pooler(a_c)
        z_ctx = model.fusion(v_c, a_c)

        z_pred = model.predictor(z_ctx)

        v_t = model.target_video_adapter(v_tgt)
        v_t = model.target_video_pooler(v_t)
        a_t = model.target_audio_adapter(a_tgt)
        a_t = model.target_audio_pooler(a_t)
        z_target = model.target_fusion(v_t, a_t)

        joint_err = F.mse_loss(z_pred, z_target, reduction="none").mean(dim=(-1, -2))

        for i in range(B):
            bucket = "normal" if labels[i] == 0 else "fall"
            results[bucket]["video"].append(video_err[i].item())
            results[bucket]["audio"].append(audio_err[i].item())
            results[bucket]["joint"].append(joint_err[i].item())

    def stats(arr):
        a = np.array(arr)
        return a.mean(), a.std() if len(a) > 1 else 0.0

    return {
        "normal": {k: stats(v) for k, v in results["normal"].items()},
        "fall": {k: stats(v) for k, v in results["fall"].items()},
        "separation": {
            m: (stats(results["fall"][m])[0] - stats(results["normal"][m])[0])
               / (stats(results["normal"][m])[1] + 1e-8)
            for m in ["video", "audio", "joint"]
        },
    }


# ─── Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train AV-JEPA v2")
    p.add_argument("--feature_cache", type=str, default="",
                   help="Path to pre-extracted features (train_features.pt)")
    p.add_argument("--data_root", type=str, default="",
                   help="Path to Le2i dataset (for end-to-end mode)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_path", type=str, default="checkpoints/av_jepa_v2.pt")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="features_v2")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from v2_config import AVJEPAv2Config, V2EncoderConfig, V2FusionConfig, V2PredictorConfig, V2TrainingConfig
    from v2_jepa_model import AVJEPAv2

    # ── Config ──
    config = AVJEPAv2Config(
        encoder=V2EncoderConfig(
            video_embed_dim=1024,
            audio_embed_dim=768,
            video_num_frames=16,
            video_size=224,
            num_video_queries=8,
            num_audio_queries=4,
        ),
        fusion=V2FusionConfig(
            fusion_dim=256,
            num_heads=8,
            num_layers=2,
        ),
        predictor=V2PredictorConfig(
            predictor_hidden_dim=512,
            predictor_num_layers=4,
            predictor_num_heads=8,
        ),
        training=V2TrainingConfig(
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device,
        ),
    )

    # ── Data ──
    if args.feature_cache:
        # Feature-based mode
        train_path = os.path.join(args.feature_cache, "train_features.pt")
        eval_path = os.path.join(args.feature_cache, "eval_features.pt")

        if not os.path.exists(train_path):
            print(f"Features not found at {train_path}")
            print("Run v2_extract_features.py first, or use --data_root for e2e mode")
            sys.exit(1)

        print(f"Loading features from {args.feature_cache}")
        train_data = torch.load(train_path, weights_only=False)
        eval_data = torch.load(eval_path, weights_only=False)

        train_dataset = FeatureDataset(train_data)
        eval_dataset = FeatureDataset(eval_data)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_variable_tokens, drop_last=True,
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_variable_tokens,
        )

        print(f"  Train (normal only): {len(train_dataset)} clips")
        print(f"  Eval (mixed):        {len(eval_dataset)} clips")
        e2e_mode = False
    else:
        print("No feature cache or data root provided.")
        print("Use --feature_cache for pre-extracted features, or --data_root for e2e.")
        sys.exit(1)

    # ── Model (no encoders loaded in feature-based mode) ──
    if not e2e_mode:
        # Build model but skip encoder loading (we use pre-extracted features)
        print("\nBuilding AV-JEPA v2 model (feature-based mode)...")
        # We need to create the model without loading heavy encoders
        from v2_jepa_model import (
            AVJEPAv2, DimAdapter, CrossModalFusion, DualJEPAPredictor
        )
        from v2_encoders import QueryTokenPooler
        from fusion import SIGRegLoss

        # Build model manually to skip encoder initialization
        model = AVJEPAv2.__new__(AVJEPAv2)
        nn.Module.__init__(model)
        model.config = config
        model.sigreg_weight = config.predictor.sigreg_weight
        model.ema_decay = config.predictor.ema_decay
        model.ema_end_decay = config.predictor.ema_end_decay
        model.fusion_dim = config.fusion.fusion_dim
        model.num_total_tokens = config.encoder.num_video_queries + config.encoder.num_audio_queries

        fus = config.fusion

        # Build trainable components
        model.video_adapter = DimAdapter(config.encoder.video_embed_dim, fus.fusion_dim)
        model.audio_adapter = DimAdapter(config.encoder.audio_embed_dim, fus.fusion_dim)
        model.video_pooler = QueryTokenPooler(fus.fusion_dim, config.encoder.num_video_queries)
        model.audio_pooler = QueryTokenPooler(fus.fusion_dim, config.encoder.num_audio_queries)
        model.fusion = CrossModalFusion(fus.fusion_dim, fus.num_heads, fus.num_layers, fus.dropout)

        # EMA targets
        import copy
        model.target_video_adapter = copy.deepcopy(model.video_adapter)
        model.target_audio_adapter = copy.deepcopy(model.audio_adapter)
        model.target_video_pooler = copy.deepcopy(model.video_pooler)
        model.target_audio_pooler = copy.deepcopy(model.audio_pooler)
        model.target_fusion = copy.deepcopy(model.fusion)
        for p in model._target_params():
            p.requires_grad = False

        # Predictor
        pred = config.predictor
        model.predictor = DualJEPAPredictor(
            fus.fusion_dim, pred.predictor_num_layers,
            pred.predictor_num_heads, pred.predictor_hidden_dim,
            pred.predictor_dropout,
        )

        # SIGReg
        model.sigreg = SIGRegLoss(target_std=pred.sigreg_target_std)
        model.to(device)

    trainable_count = model.count_trainable()
    print(f"  Trainable params: {trainable_count:,}")
    print(f"  Fusion dim:       {config.fusion.fusion_dim}")
    print(f"  Video queries:    {config.encoder.num_video_queries}")
    print(f"  Audio queries:    {config.encoder.num_audio_queries}")
    print(f"  Total tokens:     {model.num_total_tokens}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Training ──
    total_steps_all = len(train_loader) * args.epochs
    total_steps = 0
    best_sep = -float("inf")

    print(f"\n{'='*60}")
    print(f"Training AV-JEPA v2 ({args.epochs} epochs, BS={args.batch_size})")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        metrics, total_steps = train_epoch_feature_based(
            model, train_loader, optimizer, device, epoch, total_steps, total_steps_all
        )
        scheduler.step()

        print(f"  Epoch {epoch:3d} | loss={metrics['loss']:.4f} "
              f"pred={metrics['pred_loss']:.4f} sigreg={metrics['sigreg']:.4f}")

        if epoch % 5 == 0:
            eval_metrics = evaluate_feature_based(model, eval_loader, device)
            sep = eval_metrics["separation"]

            print(f"    {'':>10} {'Video':>12} {'Audio':>12} {'Joint':>12}")
            print(f"    {'Normal':>10} "
                  f"{eval_metrics['normal']['video'][0]:>12.4f} "
                  f"{eval_metrics['normal']['audio'][0]:>12.4f} "
                  f"{eval_metrics['normal']['joint'][0]:>12.4f}")
            print(f"    {'Fall':>10} "
                  f"{eval_metrics['fall']['video'][0]:>12.4f} "
                  f"{eval_metrics['fall']['audio'][0]:>12.4f} "
                  f"{eval_metrics['fall']['joint'][0]:>12.4f}")
            print(f"    {'Sep σ':>10} "
                  f"{sep['video']:>12.1f} "
                  f"{sep['audio']:>12.1f} "
                  f"{sep['joint']:>12.1f}")
            print()

            if sep["joint"] > best_sep:
                best_sep = sep["joint"]
                os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "eval_metrics": eval_metrics,
                }, args.save_path)
                print(f"    Best model saved (joint={best_sep:.1f}σ)")

    # ── Final ──
    print(f"\n{'='*60}")
    print(f"Training complete — Best separation: {best_sep:.1f}σ")
    print(f"{'='*60}")

    final_eval = evaluate_feature_based(model, eval_loader, device)
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "epoch": args.epochs,
        "eval_metrics": final_eval,
    }, args.save_path.replace(".pt", "_final.pt"))
    print(f"Final model: {args.save_path.replace('.pt', '_final.pt')}")


if __name__ == "__main__":
    main()
