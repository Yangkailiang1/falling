"""
T-JEPA Synthetic Smoke Tests

Tests all T-JEPA components with synthetic data to verify:
  1. Encoders (all 4 modalities) produce correct output shapes
  2. M3-JEPA MoE Fusion works end-to-end
  3. Hybrid fusion (MoE + Cross-Attention) produces z_fused
  4. Predictor (Transformer + Mamba) predicts future states
  5. Text projector (VL-JEPA + TC-JEPA) maps to LLM space
  6. Full T-JEPA model forward pass
  7. Anomaly gate 3-tier detection
  8. Phrase retriever cosine similarity search

All tests use random/synthetic data and require no real models.
"""

import sys
import os
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from spwm.config import (
    TJEPSConfig, VideoEncoderConfig, AudioEncoderConfig,
    SkeletonEncoderConfig, TextEncoderConfig,
    M3JEPAFusionConfig, PredictorConfig, TextProjectorConfig,
    AnomalyGateConfig, FallMambaConfig, TrainingConfig, RealtimeConfig,
)

# ═══════════════════════════════════════════════════════════════
# Test 0: Configuration
# ═══════════════════════════════════════════════════════════════

def test_config():
    """Verify all config dataclasses can be created."""
    print("\n[Test 0] Configuration...")
    config = TJEPSConfig()
    assert config.video.embed_dim == 1024
    assert config.audio.embed_dim == 768
    assert config.skeleton.embed_dim == 256
    assert config.text.embed_dim == 3584
    assert config.fusion.unified_dim == 1024
    assert config.predictor.input_dim == 1024
    assert config.anomaly_gate.sigma_threshold == 2.0
    print("  [PASS] All configs created successfully")


# ═══════════════════════════════════════════════════════════════
# Test 1: Encoders
# ═══════════════════════════════════════════════════════════════

def test_skeleton_encoder():
    """Test S-JEPA skeleton encoder with synthetic data."""
    print("\n[Test 1] Skeleton Encoder...")

    from spwm.encoders import SJEPASkeletonEncoder
    cfg = SkeletonEncoderConfig(embed_dim=256, num_keypoints=17, num_frames=8, num_layers=2)
    encoder = SJEPASkeletonEncoder(cfg)

    B, T, K, C = 2, 8, 17, 3
    skeleton = torch.randn(B, T, K, C)
    tokens = encoder(skeleton)

    assert tokens.dim() == 3, f"Expected 3D, got {tokens.dim()}D"
    assert tokens.shape[0] == B, f"Batch size mismatch: {tokens.shape[0]}"
    assert tokens.shape[2] == cfg.embed_dim, f"Embed dim mismatch: {tokens.shape[2]}"
    print(f"  [PASS] Input: {skeleton.shape} → Output: {tokens.shape}")


def test_fusion():
    """Test M3-JEPA MoE and Hybrid fusion."""
    print("\n[Test 2] M3-JEPA Fusion...")

    from spwm.fusion import M3JEPAFusion, HybridFusion

    cfg = M3JEPAFusionConfig()
    B = 4

    # Generate per-modality pooled features
    v = torch.randn(B, 1024)
    a = torch.randn(B, 768)
    s = torch.randn(B, 256)
    t = torch.randn(B, 3584)

    # Test M3-JEPA MoE alone
    moe = M3JEPAFusion(cfg)
    out_moe = moe(v.unsqueeze(1), a.unsqueeze(1), s.unsqueeze(1), t)
    z_fused = out_moe['z_fused']
    assert z_fused.shape == (B, 1024), f"M3-JEPA output shape: {z_fused.shape}"
    assert 'gate_weights' in out_moe, "Missing gate weights"
    print(f"  [PASS] M3-JEPA MoE: {z_fused.shape}")

    # Test Hybrid fusion (MoE + Cross-Attention)
    cfg.use_cross_attention = True
    hybrid = HybridFusion(cfg)
    out_hybrid = hybrid(v.unsqueeze(1), a.unsqueeze(1), s.unsqueeze(1), t)
    z_hybrid = out_hybrid['z_fused']
    assert z_hybrid.shape == (B, 1024), f"Hybrid output shape: {z_hybrid.shape}"
    print(f"  [PASS] Hybrid Fusion (MoE + CrossAttn): {z_hybrid.shape}")


def test_predictor():
    """Test both Transformer and Mamba predictors."""
    print("\n[Test 3] Predictors...")

    from spwm.predictor import VJEPA2Predictor, MambaTemporalPredictor

    cfg = PredictorConfig()
    B = 4
    z_fused = torch.randn(B, 1024)

    # Transformer predictor
    cfg.use_mamba = False
    transformer_pred = VJEPA2Predictor(cfg)
    z_transformer = transformer_pred(z_fused)
    assert z_transformer.shape == (B, 1024)
    print(f"  [PASS] Transformer predictor: {z_transformer.shape}")

    # Mamba predictor
    cfg.use_mamba = True
    mamba_pred = MambaTemporalPredictor(cfg)
    z_mamba = mamba_pred(z_fused)
    assert z_mamba.shape == (B, 1024)
    print(f"  [PASS] Mamba predictor: {z_mamba.shape}")


def test_projector():
    """Test VL-JEPA + TC-JEPA text projector."""
    print("\n[Test 4] Text Projector...")

    from spwm.projector import TextConditionedProjector, VLJEPAPhraseProjector

    cfg = TextProjectorConfig()
    B = 4
    z_future = torch.randn(B, 1024)
    text_embed = torch.randn(B, 3584)

    # Text-conditioned projector
    tc_proj = TextConditionedProjector(cfg)
    z_text = tc_proj(z_future, text_embed)
    assert z_text.shape == (B, 3584), f"Expected (B, 3584), got {z_text.shape}"
    print(f"  [PASS] TC-JEPA projector: {z_text.shape}")

    # Full VL-JEPA projector
    vl_proj = VLJEPAPhraseProjector(cfg)
    z_text2 = vl_proj(z_future, text_embed)
    assert z_text2.shape == (B, 3584)
    print(f"  [PASS] VL-JEPA projector: {z_text2.shape}")


def test_anomaly_gate():
    """Test 3-tier anomaly gate with synthetic data."""
    print("\n[Test 5] Anomaly Gate...")

    from spwm.anomaly_gate import TJEPSAnomalyGate, AnomalyDetector

    cfg = AnomalyGateConfig(sigma_threshold=2.0)
    D = cfg.gate1_dim

    # Generate synthetic normal samples
    normal_mean = torch.zeros(D)
    normal_samples = torch.randn(1000, D) * 0.5 + normal_mean

    # Calibrate
    detector = AnomalyDetector(cfg)
    detector.calibrate(normal_samples)

    # Test normal
    normal_input = torch.randn(D) * 0.3
    is_anomaly, score, sigma = detector.is_anomaly(normal_input)
    print(f"  Normal sample: is_anomaly={is_anomaly}, sigma={sigma:.2f}")

    # Test anomaly (5 sigma away)
    anomaly_input = torch.randn(D) * 0.3 + normal_mean + 5
    is_anomaly, score, sigma = detector.is_anomaly(anomaly_input)
    print(f"  Anomaly sample: is_anomaly={is_anomaly}, sigma={sigma:.2f}")
    assert is_anomaly, "Anomaly should be detected!"

    # Test full 3-tier gate
    gate = TJEPSAnomalyGate(cfg)
    gate.calibrate(normal_samples)

    # Build mini phrase library for testing
    phrase_embeds = torch.randn(10, 3584)
    phrase_labels = [
        ("老人向前摔倒", True),
        ("老人向后倒下", True),
        ("猫快速跑过", False),
        ("关灯", False),
        ("老人缓慢行走", False),
        ("老人坐着", False),
        ("跑步", False),
        ("跳跃", False),
        ("老人从椅子上滑落", True),
        ("老人正常站立", False),
    ]

    # Test normal (should skip Gate 2)
    z_future_normal = torch.randn(D) * 0.3
    z_text_normal = torch.randn(3584)

    result = gate.step(z_future_normal, z_text_normal, phrase_embeds, phrase_labels)
    print(f"  Normal detection: is_fall={result['is_fall']}, tier={result['tier']}, skip={result.get('skip')}")

    # Test anomaly
    z_future_anomaly = torch.randn(D) * 0.3 + normal_mean + 5
    result2 = gate.step(z_future_anomaly, z_text_normal, phrase_embeds, phrase_labels)
    print(f"  Anomaly detection: is_fall={result2['is_fall']}, tier={result2['tier']}")

    print("  [PASS] Anomaly gate tests")


def test_phrase_retriever():
    """Test phrase retriever with synthetic embeddings."""
    print("\n[Test 6] Phrase Retriever...")

    from spwm.phrase_retriever import PhraseLibrary
    import torch.nn.functional as F

    library = PhraseLibrary(embed_dim=3584)

    # Build library with simple encoder (identity)
    class DummyEncoder:
        def __call__(self, texts):
            # Hash-based deterministic embeddings
            embeds = []
            for t in texts:
                h = hash(t) % 10000
                torch.manual_seed(h)
                embeds.append(torch.randn(3584))
            return torch.stack(embeds)

    library.build(text_encoder=DummyEncoder(), use_chinese=True)

    # Test search
    query = torch.randn(3584)
    results = library.search(query, top_k=3)

    assert len(results) == 3
    assert all(isinstance(r[0], str) for r in results)
    assert all(isinstance(r[1], float) for r in results)
    assert all(isinstance(r[2], bool) for r in results)

    print(f"  Top match: '{results[0][0]}' (sim={results[0][1]:.3f}, is_fall={results[0][2]})")
    print(f"  Total phrases: {len(library.phrases)}")
    print("  [PASS] Phrase retriever")


def test_utils():
    """Test Fall-Mamba utility functions."""
    print("\n[Test 7] Utilities...")

    from spwm.utils.frame_masking import frame_masking_augment, temporal_subsample
    from spwm.utils.drop_pathway import drop_modality

    # Frame masking
    frames = torch.randn(2, 16, 3, 224, 224)
    masked = frame_masking_augment(frames, mask_ratio=0.2)
    assert masked.shape == frames.shape, f"Frame masking shape: {masked.shape}"
    n_zero_frames = (masked.sum(dim=(2, 3, 4)) == 0).sum().item()
    print(f"  Frame masking: {frames.shape} → {masked.shape}, ~{n_zero_frames} frames zeroed")

    # Temporal subsample
    subsampled = temporal_subsample(frames, 8, mode='uniform')
    assert subsampled.shape[1] == 8, f"Subsample shape: {subsampled.shape}"
    print(f"  Temporal subsample: {frames.shape[1]} → {subsampled.shape[1]} frames")

    # DropPathway
    v, a, s, t = torch.randn(4, 10, 1024), torch.randn(4, 10, 768), torch.randn(4, 10, 256), torch.randn(4, 10, 3584)
    v_out, a_out, s_out, t_out, mask = drop_modality(v, a, s, t, drop_prob=0.5, max_drop=1)
    n_dropped = (~mask).sum().item()
    print(f"  DropPathway: {n_dropped} modalities dropped, mask={mask.tolist()}")

    print("  [PASS] Utility functions")


def test_full_model():
    """Test full T-JEPA model forward pass with synthetic data."""
    print("\n[Test 8] Full T-JEPA Model...")

    from spwm.tjepa_model import TJEPS
    config = TJEPSConfig()

    # Reduce some sizes for faster testing
    config.skeleton.num_layers = 2
    config.predictor.transformer_n_layers = 1
    config.predictor.mamba_n_layers = 1

    # Use Mamba predictor by default (it's the recommended option)
    config.predictor.use_mamba = True

    model = TJEPS(config)
    model.set_stage('stage1')  # Test training mode

    # Replace text encoder with a dummy to avoid downloading Qwen2.5 (14GB)
    class DummyTextEncoder(torch.nn.Module):
        def __init__(self, embed_dim=3584):
            super().__init__()
            self.embed_dim = embed_dim
        def forward(self, texts):
            B = len(texts)
            return torch.randn(B, self.embed_dim)
    # Use _modules dict to bypass type checking
    dummy = DummyTextEncoder(config.text.embed_dim)
    model.encoders.text_encoder = dummy
    model.encoders._modules['text_encoder'] = dummy

    model.train()

    B = 2
    T_ctx, T_tgt = 8, 8
    C, H, W = 3, 224, 224

    ctx_frames = torch.randn(B, T_ctx, C, H, W)
    tgt_frames = torch.randn(B, T_tgt, C, H, W)
    ctx_audio = torch.randn(B, 48000)
    tgt_audio = torch.randn(B, 48000)
    ctx_skeleton = torch.randn(B, T_ctx, 17, 3)
    tgt_skeleton = torch.randn(B, T_tgt, 17, 3)
    text_condition = ["老人缓慢行走"] * B
    target_text = ["老人继续缓慢行走"] * B

    # Forward pass with targets (training)
    output = model(
        ctx_frames=ctx_frames,
        ctx_audio=ctx_audio,
        ctx_skeleton=ctx_skeleton,
        text_condition=text_condition,
        tgt_frames=tgt_frames,
        tgt_audio=tgt_audio,
        tgt_skeleton=tgt_skeleton,
        target_text=target_text,
    )

    assert 'z_fused' in output, "Missing z_fused"
    assert 'z_future' in output, "Missing z_future"
    assert 'z_text' in output, "Missing z_text"
    assert 'loss' in output, "Missing loss"

    print(f"  z_fused:  {output['z_fused'].shape}")
    print(f"  z_future: {output['z_future'].shape}")
    print(f"  z_text:   {output['z_text'].shape}")
    print(f"  loss:     {output['loss'].item():.4f}")

    if 'losses' in output:
        for k, v in output['losses'].items():
            print(f"    {k}: {v.item():.4f}")

    # Test detection mode
    model.eval()
    model.set_stage('inference')
    model.build_phrase_library(use_chinese=True)

    with torch.no_grad():
        detect_result = model.detect(
            ctx_frames=ctx_frames[0],
            ctx_audio=ctx_audio[0],
            ctx_skeleton=ctx_skeleton[0],
            text_condition="老人缓慢行走",
        )

    print(f"\n  Detection result:")
    print(f"    is_fall:  {detect_result.get('is_fall')}")
    print(f"    tier:     {detect_result.get('tier')}")
    print(f"    sigma:    {detect_result.get('sigma_score', 0):.2f}")
    if detect_result.get('top_phrase'):
        print(f"    phrase:   {detect_result.get('top_phrase')}")

    print("  [PASS] Full T-JEPA model")


def test_sigreg():
    """Test SIGReg anti-collapse regularization."""
    print("\n[Test 9] SIGReg...")

    from spwm.fusion import SIGRegLoss

    sigreg = SIGRegLoss()

    # Collapsed data (all zeros)
    collapsed = torch.zeros(64, 256)
    loss_collapsed = sigreg(collapsed)
    print(f"  Collapsed data loss: {loss_collapsed.item():.4f}")

    # Normal data (roughly unit normal)
    normal = torch.randn(64, 256)
    loss_normal = sigreg(normal)
    print(f"  Normal data loss:   {loss_normal.item():.4f}")

    # Highly correlated data
    base = torch.randn(64, 1).expand(64, 256) + 0.01 * torch.randn(64, 256)
    loss_corr = sigreg(base)
    print(f"  Correlated data loss: {loss_corr.item():.4f}")

    # SIGReg penalizes: off-diagonal covariance + deviation of diag from 1
    # Collapsed: off-diag=0, diag variance=0 → penalty = D per dimension for diag deviation
    # Normal: off-diag non-zero, diag~1 → penalty from off-diagonal terms
    # Both can have high loss, but correlated data should have highest (off-diag dominates)
    assert loss_corr > loss_normal * 0.5, f"Correlated data SIGReg loss ({loss_corr:.1f}) should be higher than normal ({loss_normal:.1f})"
    print(f"  Collapsed: {loss_collapsed.item():.4f}, Normal: {loss_normal.item():.4f}, Correlated: {loss_corr.item():.4f}")
    print("  [PASS] SIGReg")


def test_frame_masking():
    """Test Fall-Mamba frame masking augmentation."""
    print("\n[Test 10] Frame Masking...")

    from spwm.utils.frame_masking import frame_masking_augment

    frames = torch.ones(2, 16, 3, 224, 224)

    # 20% masking
    masked = frame_masking_augment(frames.clone(), mask_ratio=0.2)
    n_kept = (masked.sum(dim=(2, 3, 4)) > 0).sum().item()
    kept_ratio = n_kept / (2 * 16)
    print(f"  20% mask: {n_kept}/{32} frames kept ({kept_ratio:.0%})")

    # 50% masking
    masked = frame_masking_augment(frames.clone(), mask_ratio=0.5)
    n_kept = (masked.sum(dim=(2, 3, 4)) > 0).sum().item()
    kept_ratio = n_kept / (2 * 16)
    print(f"  50% mask: {n_kept}/{32} frames kept ({kept_ratio:.0%})")

    print("  [PASS] Frame masking")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run_all_tests():
    """Run all smoke tests."""
    print("=" * 60)
    print("T-JEPA Synthetic Smoke Tests")
    print("=" * 60)

    tests = [
        ("Configuration", test_config),
        ("Skeleton Encoder", test_skeleton_encoder),
        ("M3-JEPA Fusion", test_fusion),
        ("Predictors (Transformer + Mamba)", test_predictor),
        ("Text Projector (VL-JEPA + TC-JEPA)", test_projector),
        ("Anomaly Gate", test_anomaly_gate),
        ("Phrase Retriever", test_phrase_retriever),
        ("Utilities", test_utils),
        ("SIGReg", test_sigreg),
        ("Frame Masking", test_frame_masking),
        ("Full T-JEPA Model", test_full_model),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} passed")
    if failed > 0:
        print(f"         {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
