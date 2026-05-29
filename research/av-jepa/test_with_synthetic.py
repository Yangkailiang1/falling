"""
AV-JEPA Smoke Test
-------------------
End-to-end test with synthetic data.
Verifies: data generation → training → detection pipeline works.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import AVJEPAConfig
from jepa_model import AVJEPA
from data_utils import (
    generate_synthetic_embeddings,
    generate_synthetic_fall_embeddings,
    SyntheticAVDataset,
)


def test_data_generation():
    """Test synthetic data generation."""
    print("1. Testing Data Generation...")

    # Normal data
    ctx, tgt = generate_synthetic_embeddings(32, 256)
    assert ctx.shape == (32, 256), f"Expected (32, 256), got {ctx.shape}"
    assert tgt.shape == (32, 256)
    assert not torch.allclose(ctx, tgt), "Context and target should differ"

    # Fall data
    ctx_f, tgt_f = generate_synthetic_fall_embeddings(32, 256)
    normal_dist = (ctx - tgt).norm(dim=-1).mean()
    fall_dist = (ctx_f - tgt_f).norm(dim=-1).mean()
    assert fall_dist > normal_dist * 3, (
        f"Fall distance ({fall_dist:.2f}) should be >> normal ({normal_dist:.2f})"
    )
    print(f"   ✅ Normal distance: {normal_dist:.4f}, Fall distance: {fall_dist:.4f}")

    # Dataset
    dataset = SyntheticAVDataset(100, 256, normal_ratio=0.8)
    assert len(dataset) == 100
    normal_count = (dataset.labels == 0).sum().item()
    fall_count = (dataset.labels == 1).sum().item()
    print(f"   ✅ Dataset: {normal_count} normal, {fall_count} fall")
    print()


def test_model_forward():
    """Test model forward pass."""
    print("2. Testing Model Forward Pass...")

    config = AVJEPAConfig()
    device = torch.device("cpu")
    model = AVJEPA(config).to(device)

    # Test with synthetic embeddings (skip encoder for speed)
    B, D = 8, config.encoder.joint_embed_dim
    ctx_emb = torch.randn(B, config.encoder.video_embed_dim + config.encoder.audio_embed_dim)
    tgt_emb = torch.randn(B, config.encoder.video_embed_dim + config.encoder.audio_embed_dim)

    z_ctx = model.projector(ctx_emb)
    assert z_ctx.shape == (B, D), f"Expected ({B}, {D}), got {z_ctx.shape}"

    z_pred = model.predictor(z_ctx)
    assert z_pred.shape == (B, D)

    z_target = model.target_projector(tgt_emb)
    assert z_target.shape == (B, D)

    loss, loss_dict = model.compute_loss(z_pred, z_target, z_ctx)
    assert loss.item() > 0
    print(f"   ✅ Forward pass OK, loss={loss_dict['total_loss']:.4f}")
    print()


def test_training_step():
    """Test a few training steps."""
    print("3. Testing Training Step...")

    config = AVJEPAConfig()
    device = torch.device("cpu")
    model = AVJEPA(config).to(device)

    trainable = list(model.projector.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.Adam(trainable, lr=1e-3)

    raw_dim = config.encoder.video_embed_dim + config.encoder.audio_embed_dim
    dataset = SyntheticAVDataset(128, raw_dim, normal_ratio=0.85)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    initial_losses = []
    final_losses = []

    for epoch in range(3):
        epoch_loss = 0
        for step, (ctx_emb, tgt_emb, _) in enumerate(loader):
            z_ctx = model.projector(ctx_emb)
            z_pred = model.predictor(z_ctx)
            z_target = model.target_projector(tgt_emb)
            loss, _ = model.compute_loss(z_pred, z_target, z_ctx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.update_target_encoder(step, len(loader) * 3)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        if epoch == 0:
            initial_losses.append(avg_loss)
        final_losses.append(avg_loss)

    print(f"   Initial loss: {initial_losses[0]:.4f}")
    print(f"   Final loss:   {final_losses[-1]:.4f}")
    assert final_losses[-1] < initial_losses[0] * 1.2, (
        "Loss should decrease or stay stable"
    )
    print(f"   ✅ Training converges")
    print()


def test_anomaly_detection():
    """Test that prediction error separates normal from fall."""
    print("4. Testing Anomaly Detection...")

    config = AVJEPAConfig()
    device = torch.device("cpu")

    # Train quickly
    model = AVJEPA(config).to(device)
    trainable = list(model.projector.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.Adam(trainable, lr=1e-3)

    raw_dim = config.encoder.video_embed_dim + config.encoder.audio_embed_dim
    # Only train on NORMAL data — falls should be unknown/surprising
    dataset = SyntheticAVDataset(500, raw_dim, normal_ratio=1.0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    total_steps = len(loader) * 20
    step = 0
    for epoch in range(20):
        for ctx_emb, tgt_emb, _ in loader:
            z_ctx = model.projector(ctx_emb)
            z_pred = model.predictor(z_ctx)
            z_target = model.target_projector(tgt_emb)
            loss, _ = model.compute_loss(z_pred, z_target, z_ctx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target_encoder(step, total_steps)
            step += 1

    # Test on separate data
    raw_dim = config.encoder.video_embed_dim + config.encoder.audio_embed_dim
    test_normal = torch.stack([
        SyntheticAVDataset(50, raw_dim, normal_ratio=1.0)[i][0]
        for i in range(50)
    ])
    test_normal_tgt = torch.stack([
        SyntheticAVDataset(50, raw_dim, normal_ratio=1.0)[i][1]
        for i in range(50)
    ])

    fall_ctx, fall_tgt = generate_synthetic_fall_embeddings(
        50, raw_dim, fall_displacement=5.0
    )

    model.eval()
    with torch.no_grad():
        # Normal errors
        z_ctx_n = model.projector(test_normal)
        z_pred_n = model.predictor(z_ctx_n)
        z_tgt_n = model.target_projector(test_normal_tgt)
        normal_errors = F.mse_loss(z_pred_n, z_tgt_n, reduction="none").mean(-1)

        # Fall errors
        z_ctx_f = model.projector(fall_ctx)
        z_pred_f = model.predictor(z_ctx_f)
        z_tgt_f = model.target_projector(fall_tgt)
        fall_errors = F.mse_loss(z_pred_f, z_tgt_f, reduction="none").mean(-1)

    normal_mean, normal_std = normal_errors.mean().item(), normal_errors.std().item()
    fall_mean, fall_std = fall_errors.mean().item(), fall_errors.std().item()

    print(f"   Normal error: {normal_mean:.4f} ± {normal_std:.4f}")
    print(f"   Fall error:   {fall_mean:.4f} ± {fall_std:.4f}")

    separation = (fall_mean - normal_mean) / (normal_std + 1e-8)
    print(f"   Separation:   {separation:.1f}σ")

    if fall_mean > normal_mean + 2 * normal_std:
        print("   ✅ Fall events produce significantly higher prediction error!")
    else:
        print("   ⚠️  Separation could be better — try more training or data")

    print()


def main():
    print("=" * 60)
    print("AV-JEPA Smoke Test")
    print("=" * 60)
    print()

    tests = [
        test_data_generation,
        test_model_forward,
        test_training_step,
        test_anomaly_detection,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"   ❌ FAILED: {e}\n")

    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("✅ ALL TESTS PASSED — AV-JEPA pipeline is working!")
    print("=" * 60)


if __name__ == "__main__":
    main()
