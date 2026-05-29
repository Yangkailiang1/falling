"""
AV-JEPA v2 Fall Detection
--------------------------
Anomaly detection via cross-modal JEPA prediction error.

Usage:
    python v2_detect.py --checkpoint checkpoints/av_jepa_v2.pt \\
                        --feature_cache features_v2/ \\
                        --device cuda
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="AV-JEPA v2 Fall Detection")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--feature_cache", type=str, default="features_v2")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--threshold_sigma", type=float, default=3.0)
    return p.parse_args()


def build_model_from_checkpoint(ckpt_path: str, device: str):
    """Reconstruct model from checkpoint (feature-based mode)."""
    from v2_config import AVJEPAv2Config, V2EncoderConfig, V2FusionConfig, V2PredictorConfig, V2TrainingConfig
    from v2_jepa_model import (
        DimAdapter, CrossModalFusion, DualJEPAPredictor, AVJEPAv2
    )
    from v2_encoders import QueryTokenPooler
    from fusion import SIGRegLoss
    import copy

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Build model manually
    model = AVJEPAv2.__new__(AVJEPAv2)
    torch.nn.Module.__init__(model)
    model.config = config
    model.sigreg_weight = config.predictor.sigreg_weight
    model.ema_decay = config.predictor.ema_decay
    model.ema_end_decay = config.predictor.ema_end_decay
    model.fusion_dim = config.fusion.fusion_dim
    model.num_total_tokens = config.encoder.num_video_queries + config.encoder.num_audio_queries

    fus = config.fusion
    pred = config.predictor

    model.video_adapter = DimAdapter(config.encoder.video_embed_dim, fus.fusion_dim)
    model.audio_adapter = DimAdapter(config.encoder.audio_embed_dim, fus.fusion_dim)
    model.video_pooler = QueryTokenPooler(fus.fusion_dim, config.encoder.num_video_queries)
    model.audio_pooler = QueryTokenPooler(fus.fusion_dim, config.encoder.num_audio_queries)
    model.fusion = CrossModalFusion(fus.fusion_dim, fus.num_heads, fus.num_layers, fus.dropout)

    model.target_video_adapter = copy.deepcopy(model.video_adapter)
    model.target_audio_adapter = copy.deepcopy(model.audio_adapter)
    model.target_video_pooler = copy.deepcopy(model.video_pooler)
    model.target_audio_pooler = copy.deepcopy(model.audio_pooler)
    model.target_fusion = copy.deepcopy(model.fusion)
    for p in model._target_params():
        p.requires_grad = False

    model.predictor = DualJEPAPredictor(
        fus.fusion_dim, pred.predictor_num_layers,
        pred.predictor_num_heads, pred.predictor_hidden_dim,
        pred.predictor_dropout,
    )
    model.sigreg = SIGRegLoss(target_std=pred.sigreg_target_std)

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, config, ckpt


class V2FallDetector:
    """Fall detector using AV-JEPA v2 prediction error."""

    def __init__(self, model, threshold: float, device: str = "cuda"):
        self.model = model
        self.threshold = threshold
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def detect(
        self,
        v_ctx: torch.Tensor,
        a_ctx: torch.Tensor,
        v_tgt: torch.Tensor,
        a_tgt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect falls from feature pairs.

        Args:
            v_ctx, a_ctx: context video/audio features
            v_tgt, a_tgt: target video/audio features

        Returns:
            is_fall: (B,) bool
            error:   (B,) float
        """
        # Encode context
        v_c = self.model.video_adapter(v_ctx)
        v_c = self.model.video_pooler(v_c)
        a_c = self.model.audio_adapter(a_ctx)
        a_c = self.model.audio_pooler(a_c)
        z_ctx = self.model.fusion(v_c, a_c)

        z_pred = self.model.predictor(z_ctx)

        # Encode target (EMA)
        v_t = self.model.target_video_adapter(v_tgt)
        v_t = self.model.target_video_pooler(v_t)
        a_t = self.model.target_audio_adapter(a_tgt)
        a_t = self.model.target_audio_pooler(a_t)
        z_target = self.model.target_fusion(v_t, a_t)

        error = F.mse_loss(z_pred, z_target, reduction="none").mean(dim=(-1, -2))
        is_fall = error > self.threshold
        return is_fall, error

    def detect_with_details(
        self,
        v_ctx: torch.Tensor,
        a_ctx: torch.Tensor,
        v_tgt: torch.Tensor,
        a_tgt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Detect with per-modality breakdown."""
        # Joint
        is_fall, joint_err = self.detect(v_ctx, a_ctx, v_tgt, a_tgt)

        # Video-only
        video_err = F.mse_loss(v_ctx, v_tgt, reduction="none").mean(dim=(-1, -2))

        # Audio-only
        audio_err = F.mse_loss(a_ctx, a_tgt, reduction="none").mean(dim=(-1, -2))

        return {
            "is_fall": is_fall,
            "joint_error": joint_err,
            "video_error": video_err,
            "audio_error": audio_err,
        }


def evaluate_detector(
    detector: V2FallDetector,
    features_list: list,
    device: str,
) -> Dict:
    """Full evaluation with metrics."""
    from v2_train import FeatureDataset, collate_variable_tokens
    from torch.utils.data import DataLoader

    dataset = FeatureDataset(features_list)
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        collate_fn=collate_variable_tokens,
    )

    tp = tn = fp = fn = 0
    all_errors_normal = []
    all_errors_fall = []

    for batch in tqdm(loader, desc="Detecting"):
        v_ctx, a_ctx, v_tgt, a_tgt, labels = batch
        v_ctx = v_ctx.to(device)
        a_ctx = a_ctx.to(device)
        v_tgt = v_tgt.to(device)
        a_tgt = a_tgt.to(device)
        labels = labels.to(device)

        is_fall, errors = detector.detect(v_ctx, a_ctx, v_tgt, a_tgt)

        for i in range(len(labels)):
            pred = is_fall[i].item()
            true = labels[i].item()
            err = errors[i].item()

            if true == 0:
                all_errors_normal.append(err)
                if pred:
                    fp += 1
                else:
                    tn += 1
            else:
                all_errors_fall.append(err)
                if pred:
                    tp += 1
                else:
                    fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Separation
    normal_arr = np.array(all_errors_normal)
    fall_arr = np.array(all_errors_fall)
    separation = (fall_arr.mean() - normal_arr.mean()) / (normal_arr.std() + 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "separation_sigma": separation,
        "normal_mean": normal_arr.mean(),
        "normal_std": normal_arr.std(),
        "fall_mean": fall_arr.mean(),
        "fall_std": fall_arr.std(),
        "threshold": detector.threshold,
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config, ckpt = build_model_from_checkpoint(args.checkpoint, device)
    print(f"  Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")

    eval_metrics = ckpt.get("eval_metrics", {})
    print(f"  Saved metrics: {list(eval_metrics.keys()) if eval_metrics else 'none'}")

    # ── Set threshold ──
    normal_mean = eval_metrics.get("normal", {}).get("joint", (0.0, 0.0))[0]
    normal_std = eval_metrics.get("normal", {}).get("joint", (0.0, 1.0))[1]
    threshold = normal_mean + args.threshold_sigma * normal_std

    print(f"\n  Normal error: {normal_mean:.6f} ± {normal_std:.6f}")
    print(f"  Threshold ({args.threshold_sigma}σ): {threshold:.6f}")

    detector = V2FallDetector(model, threshold, device)

    # ── Evaluate ──
    if args.feature_cache and os.path.exists(
        os.path.join(args.feature_cache, "eval_features.pt")
    ):
        print(f"\nLoading eval features from {args.feature_cache}")
        eval_data = torch.load(
            os.path.join(args.feature_cache, "eval_features.pt"),
            weights_only=False,
        )
        print(f"  Total eval clips: {len(eval_data)}")

        metrics = evaluate_detector(detector, eval_data, device)

        print(f"\n{'='*60}")
        print("Fall Detection Results")
        print(f"{'='*60}")
        print(f"  Threshold:       {threshold:.6f}")
        print(f"  Accuracy:        {metrics['accuracy']:.2%}")
        print(f"  Precision:       {metrics['precision']:.2%}")
        print(f"  Recall:          {metrics['recall']:.2%}")
        print(f"  F1 Score:        {metrics['f1']:.2%}")
        print(f"  Separation σ:    {metrics['separation_sigma']:.1f}")
        print(f"  TP={metrics['tp']}, TN={metrics['tn']}, "
              f"FP={metrics['fp']}, FN={metrics['fn']}")
        print(f"\n  Normal error:    {metrics['normal_mean']:.4f} ± {metrics['normal_std']:.4f}")
        print(f"  Fall error:      {metrics['fall_mean']:.4f} ± {metrics['fall_std']:.4f}")

        if args.output:
            torch.save(metrics, args.output)
            print(f"\n  Saved to {args.output}")
    else:
        print(f"\nEval features not found at {args.feature_cache}/eval_features.pt")


if __name__ == "__main__":
    main()
