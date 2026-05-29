"""
T-JEPA Anomaly Gate Module

3-tier cascade anomaly detection:
  Gate 1 (continuous, <1ms): Statistical anomaly — ||z_future - μ|| > 2σ
  Gate 2 (on anomaly, ~4ms): Semantic verification — cosine match against phrase library
  Gate 3 (on fall, async): LLM detailed report

The 3-tier design solves the "cat runs by" problem: Gate 1 triggers on any anomaly
(Gate 1 → false positive for cats), but Gate 2 performs semantic verification to
distinguish falls from other anomalies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from .config import AnomalyGateConfig


@dataclass
class NormalStats:
    """Statistics computed during calibration on normal activity."""
    mean: torch.Tensor          # (D,) mean of normal z_future distribution
    std: torch.Tensor           # (D,) std of normal z_future
    covariance: torch.Tensor    # (D, D) covariance matrix (for Mahalanobis)
    sample_count: int = 0

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.covariance = self.covariance.to(device)
        return self


class AnomalyDetector(nn.Module):
    """
    Gate 1: Statistical anomaly detection.

    Computes anomaly score as Mahalanobis distance (or L2 if no covariance).
    Threshold at sigma_threshold standard deviations from normal distribution mean.

    Runs continuously at 3Hz (every 8 frames). Latency <1ms.
    """

    def __init__(self, config: AnomalyGateConfig):
        super().__init__()
        self.config = config
        self.dim = config.gate1_dim  # 1024
        self.sigma_threshold = config.sigma_threshold

        # Normal statistics (set during calibration)
        self.register_buffer('normal_mean', torch.zeros(config.gate1_dim))
        self.register_buffer('normal_std', torch.ones(config.gate1_dim))
        self.register_buffer('normal_inv_cov', torch.eye(config.gate1_dim))
        self.n_samples = 0
        self._calibrated = False

    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate(
        self,
        normal_samples: torch.Tensor,
        use_mahalanobis: bool = True,
    ):
        """
        Compute statistics from samples of normal activity.

        Args:
            normal_samples: (N, D) z_future samples from normal activity
            use_mahalanobis: if True, use Mahalanobis distance (accounts for correlation)
        """
        N, D = normal_samples.shape
        self.n_samples = N
        self.normal_mean = normal_samples.mean(dim=0)
        self.normal_std = normal_samples.std(dim=0).clamp(min=1e-6)

        if use_mahalanobis and N > D:
            # Compute inverse covariance for Mahalanobis distance
            centered = normal_samples - self.normal_mean
            cov = (centered.T @ centered) / (N - 1)
            # Regularize for numerical stability
            cov += torch.eye(D, device=cov.device) * 1e-5
            try:
                self.normal_inv_cov = torch.linalg.inv(cov)
            except torch.linalg.LinAlgError:
                # Fall back to diag (L2 norm)
                self.normal_inv_cov = torch.diag(1.0 / (self.normal_std ** 2 + 1e-6))
        else:
            # L2 with per-dimension scaling
            self.normal_inv_cov = torch.diag(1.0 / (self.normal_std ** 2 + 1e-6))

        self._calibrated = True

    def compute_anomaly_score(self, z_future: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anomaly score (Mahalanobis distance from normal distribution).

        Args:
            z_future: (B, D) or (D,) predicted future state

        Returns:
            anomaly_score: (B,) Mahalanobis distance
            sigma_score: (B,) normalized by standard deviation
        """
        if not self._calibrated:
            # Uncalibrated: use simple norm
            diff = z_future - self.normal_mean
            anomaly_score = torch.norm(diff, dim=-1)
            sigma_score = anomaly_score / (self.normal_std.mean() + 1e-8)
            return anomaly_score, sigma_score

        centered = z_future - self.normal_mean  # (B, D)

        # Mahalanobis distance: sqrt((x-μ)^T Σ^{-1} (x-μ))
        if len(centered.shape) == 1:
            centered = centered.unsqueeze(0)

        mahalanobis = torch.sqrt(
            (centered @ self.normal_inv_cov * centered).sum(dim=-1) + 1e-8
        )

        # Sigma score: normalized by expected std
        sigma_score = mahalanobis / (self.normal_std.mean() + 1e-8)

        if mahalanobis.shape[0] == 1:
            mahalanobis = mahalanobis.squeeze(0)
            sigma_score = sigma_score.squeeze(0)

        return mahalanobis, sigma_score

    def is_anomaly(self, z_future: torch.Tensor) -> Tuple[bool, float, float]:
        """
        Gate 1 decision: is the future state anomalous?

        Returns:
            is_anomaly: bool
            anomaly_score: float
            sigma_score: float
        """
        anomaly_score, sigma_score = self.compute_anomaly_score(z_future)
        is_anomaly = sigma_score > self.sigma_threshold

        if isinstance(is_anomaly, torch.Tensor):
            is_anomaly = bool(is_anomaly.item())
            sigma_score = float(sigma_score.item())
            anomaly_score = float(anomaly_score.item())

        return is_anomaly, anomaly_score, sigma_score


class SemanticVerifier(nn.Module):
    """
    Gate 2: Semantic verification via phrase matching.

    After Gate 1 triggers, this verifies whether the anomaly corresponds
    to a fall by comparing z_text with a phrase library of known fall/non-fall
    descriptions using cosine similarity.
    """

    def __init__(self, config: AnomalyGateConfig):
        super().__init__()
        self.config = config
        self.min_similarity = config.min_similarity
        self.top_k = config.top_k_search

    def forward(
        self,
        z_text: torch.Tensor,
        phrase_embeddings: torch.Tensor,
        phrase_labels: list,
    ) -> Dict:
        """
        Verify if z_text matches fall phrases.

        Args:
            z_text: (B, D) or (D,) predicted text embedding
            phrase_embeddings: (N_phrases, D) phrase library embeddings
            phrase_labels: list of (phrase_text, is_fall) tuples

        Returns:
            dict with top_matches, is_fall, confidence
        """
        if z_text.dim() == 1:
            z_text = z_text.unsqueeze(0)

        # Normalize for cosine similarity
        z_norm = torch.nn.functional.normalize(z_text, dim=-1)
        p_norm = torch.nn.functional.normalize(phrase_embeddings, dim=-1)

        # Cosine similarity
        similarities = z_norm @ p_norm.T  # (B, N_phrases)

        # Top-k matches
        top_sims, top_indices = similarities.topk(
            min(self.top_k, phrase_embeddings.shape[0]), dim=-1
        )
        top_sims = top_sims[0]  # single batch
        top_indices = top_indices[0]

        top_matches = []
        for idx, sim in zip(top_indices, top_sims):
            phrase, is_fall = phrase_labels[int(idx)]
            top_matches.append({
                'phrase': phrase,
                'similarity': float(sim),
                'is_fall': is_fall,
            })

        # Decision: top-1 must be fall phrase AND similarity > threshold
        top_phrase = top_matches[0]
        is_fall = top_phrase['is_fall'] and top_phrase['similarity'] >= self.min_similarity

        return {
            'top_matches': top_matches,
            'is_fall': is_fall,
            'top_phrase': top_phrase['phrase'],
            'confidence': top_phrase['similarity'],
        }


class TJEPSAnomalyGate(nn.Module):
    """
    Complete 3-tier anomaly gate for T-JEPA.

    Combines:
      - Gate 1: Statistical anomaly detection (continuous)
      - Gate 2: Semantic verification via phrase matching
      - Gate 3: LLM detailed report (optional, async)
    """

    def __init__(self, config: AnomalyGateConfig):
        super().__init__()
        self.config = config
        self.anomaly_detector = AnomalyDetector(config)
        self.semantic_verifier = SemanticVerifier(config)
        self.enable_llm = config.enable_llm

    def calibrate(self, normal_samples: torch.Tensor):
        """Calibrate Gate 1 on normal activity."""
        self.anomaly_detector.calibrate(normal_samples)

    @torch.no_grad()
    def step(
        self,
        z_future: torch.Tensor,
        z_text: Optional[torch.Tensor] = None,
        phrase_embeddings: Optional[torch.Tensor] = None,
        phrase_labels: Optional[list] = None,
    ) -> Dict:
        """
        Single-step 3-tier gate evaluation.

        Args:
            z_future: (D,) predicted future state
            z_text: (D,) predicted text embedding (Gate 2 input)
            phrase_embeddings: (N, D) phrase library (Gate 2)
            phrase_labels: phrase metadata (Gate 2)

        Returns:
            dict with is_fall, tier results, anomaly score, latency info
        """
        result = {
            'is_fall': False,
            'skip': True,
            'tier': 0,
        }

        # ━━ Gate 1: Statistical anomaly ━━
        is_anomaly, anomaly_score, sigma_score = self.anomaly_detector.is_anomaly(
            z_future
        )
        result['anomaly_score'] = anomaly_score
        result['sigma_score'] = sigma_score
        result['tier'] = 1

        if not is_anomaly:
            # Normal → skip remaining gates
            result['output'] = f"Normal (sigma={sigma_score:.1f})"
            return result

        result['skip'] = False

        # ━━ Gate 2: Semantic verification ━━
        if z_text is None or phrase_embeddings is None:
            # No verifier available → conservative: treat as anomaly
            result['is_fall'] = True
            result['tier'] = 2
            result['output'] = f"⚠️ Anomaly (sigma={sigma_score:.1f}), no verifier"
            return result

        verification = self.semantic_verifier(z_text, phrase_embeddings, phrase_labels)
        result.update(verification)
        result['tier'] = 2

        if not result['is_fall']:
            # Anomaly but not fall (e.g., cat, running, lights off)
            result['output'] = (
                f"Anomaly but not fall: {result['top_phrase']} "
                f"(sim={result['confidence']:.0%}, sigma={sigma_score:.1f})"
            )
            return result

        # ━━ Gate 3: Fall detected → optional LLM report ━━
        result['tier'] = 3
        result['output'] = f"🚨 Fall detected: {result['top_phrase']} (conf={result['confidence']:.0%})"
        result['needs_llm_report'] = self.enable_llm

        return result

    def save_calibration(self, path: str):
        """Save Gate 1 calibration statistics."""
        stats = {
            'mean': self.anomaly_detector.normal_mean,
            'std': self.anomaly_detector.normal_std,
            'inv_cov': self.anomaly_detector.normal_inv_cov,
            'n_samples': self.anomaly_detector.n_samples,
        }
        torch.save(stats, path)

    def load_calibration(self, path: str):
        """Load Gate 1 calibration statistics."""
        stats = torch.load(path, map_location='cpu')
        self.anomaly_detector.normal_mean = stats['mean']
        self.anomaly_detector.normal_std = stats['std']
        self.anomaly_detector.normal_inv_cov = stats['inv_cov']
        self.anomaly_detector.n_samples = stats['n_samples']
        self.anomaly_detector._calibrated = True
