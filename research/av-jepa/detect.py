"""
AV-JEPA Fall Detection
-----------------------
Anomaly detection via prediction error.
Normal activity → low prediction error
Fall events → high prediction error (surprise)

Usage:
    python detect.py --checkpoint checkpoints/av_jepa.pt --data_path /path/to/test_videos/
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

from config import AVJEPAConfig
from jepa_model import AVJEPA


def parse_args():
    parser = argparse.ArgumentParser(description="AV-JEPA Fall Detection")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold_percentile", type=float, default=95.0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


class FallDetector:
    """
    Fall detector using AV-JEPA prediction error.

    Workflow:
        1. Load trained AV-JEPA model
        2. Establish baseline error distribution from normal data
        3. For each new sample:
           - Compute prediction error
           - Flag as fall if error > threshold
    """

    def __init__(self, model: AVJEPA, threshold: float):
        self.model = model
        self.threshold = threshold
        self.model.eval()

    @torch.no_grad()
    def detect(
        self,
        ctx_embed: torch.Tensor,
        tgt_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect falls from embedding pairs.

        Args:
            ctx_embed: (B, D) context embeddings
            tgt_embed: (B, D) target embeddings

        Returns:
            is_fall: (B,) boolean
            error: (B,) prediction error per sample
        """
        z_ctx = self.model.projector(ctx_embed)
        z_pred = self.model.predictor(z_ctx)
        z_target = self.model.target_projector(tgt_embed)

        error = F.mse_loss(z_pred, z_target, reduction="none").mean(dim=-1)
        is_fall = error > self.threshold

        return is_fall, error

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu"):
        """Load detector from checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=device)
        config = ckpt["config"]
        model = AVJEPA(config).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Estimate threshold from training metrics
        eval_metrics = ckpt.get("eval_metrics", {})
        normal_mean = eval_metrics.get("normal_mean", 0.01)
        normal_std = eval_metrics.get("normal_std", 0.005)
        threshold = normal_mean + 3.0 * normal_std  # 3-sigma rule

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Normal error: {normal_mean:.6f} ± {normal_std:.6f}")
        print(f"  Threshold (3σ): {threshold:.6f}")

        return cls(model, threshold)


def evaluate_detection(
    detector: FallDetector,
    ctx_embeds: torch.Tensor,
    tgt_embeds: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """
    Evaluate fall detection performance.

    Returns:
        metrics dict with accuracy, precision, recall, F1
    """
    is_fall_pred, errors = detector.detect(ctx_embeds, tgt_embeds)

    # Confusion matrix
    tp = ((is_fall_pred == 1) & (labels == 1)).sum().item()
    tn = ((is_fall_pred == 0) & (labels == 0)).sum().item()
    fp = ((is_fall_pred == 1) & (labels == 0)).sum().item()
    fn = ((is_fall_pred == 0) & (labels == 1)).sum().item()

    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    detector = FallDetector.from_checkpoint(args.checkpoint, device)

    # Quick test with synthetic data
    from data_utils import SyntheticAVDataset

    test_dataset = SyntheticAVDataset(
        num_samples=200,
        embed_dim=detector.model.config.encoder.video_embed_dim + detector.model.config.encoder.audio_embed_dim,
        normal_ratio=0.5,
        num_modes=5,
        noise_std=0.05,
        fall_displacement=5.0,
    )

    all_ctx, all_tgt, all_labels = [], [], []
    for i in range(len(test_dataset)):
        ctx, tgt, label = test_dataset[i]
        all_ctx.append(ctx)
        all_tgt.append(tgt)
        all_labels.append(label)

    ctx = torch.stack(all_ctx).to(device)
    tgt = torch.stack(all_tgt).to(device)
    labels = torch.tensor(all_labels).to(device)

    metrics = evaluate_detection(detector, ctx, tgt, labels)

    print(f"\n{'='*50}")
    print(f"Fall Detection Results")
    print(f"{'='*50}")
    print(f"  Accuracy:   {metrics['accuracy']:.2%}")
    print(f"  Precision:  {metrics['precision']:.2%}")
    print(f"  Recall:     {metrics['recall']:.2%}")
    print(f"  F1 Score:   {metrics['f1']:.2%}")
    print(f"  TP={metrics['tp']}, TN={metrics['tn']}, "
          f"FP={metrics['fp']}, FN={metrics['fn']}")


if __name__ == "__main__":
    main()
