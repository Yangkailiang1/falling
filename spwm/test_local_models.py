#!/usr/bin/env python3
"""
Test loading all video+audio models from spwm/model_weights/
"""
import sys
import os
import torch

# Project root
sys.path.insert(0, "/Volumes/forya/falling")
os.chdir("/Volumes/forya/falling")

MODEL_WEIGHTS = "/Volumes/forya/falling/spwm/model_weights"

def test_vjepa2():
    """Test 1: V-JEPA 2 ViT-L video encoder from local."""
    print("\n" + "=" * 60)
    print("Test 1: V-JEPA 2 ViT-L")
    print("=" * 60)

    import sys
    # Add vjepa2 ROOT (not src/) so relative imports like "from ..models" work
    sys.path.insert(0, f"{MODEL_WEIGHTS}/vjepa2")

    from src.hub.backbones import vjepa2_vit_large, _clean_backbone_key

    # Build model architecture (V-JEPA 2 uses RoPE, input size is flexible)
    print("Building V-JEPA 2 ViT-L architecture...")
    encoder, predictor = vjepa2_vit_large(pretrained=False)
    print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M")

    # Load pretrained weights
    print("Loading pretrained weights from vitl.pt...")
    checkpoint = torch.load(
        f"{MODEL_WEIGHTS}/vitl.pt",
        map_location="cpu",
        weights_only=False,
    )
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")

    # Clean and load target_encoder weights
    target_state = checkpoint["target_encoder"]
    cleaned = {}
    for k, v in target_state.items():
        new_k = k.replace("module.", "").replace("backbone.", "")
        cleaned[new_k] = v

    missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
    print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # Quick forward pass (V-JEPA 2 expects (B, C, T, H, W))
    encoder = encoder.eval()
    dummy = torch.randn(1, 3, 16, 224, 224)  # (B, C, T, H, W)
    with torch.no_grad():
        out = encoder(dummy)
    print(f"  Output shape: {out.shape}")
    print("  [OK] V-JEPA 2 ViT-L loaded successfully from local")


def test_clip():
    """Test 2: CLIP ViT-B/16 video encoder from local."""
    print("\n" + "=" * 60)
    print("Test 2: CLIP ViT-B/16 (local)")
    print("=" * 60)

    from transformers import CLIPVisionModel, CLIPImageProcessor

    clip_path = f"{MODEL_WEIGHTS}/clip-vit-base-patch16"
    print(f"Loading from: {clip_path}")
    model = CLIPVisionModel.from_pretrained(clip_path)
    processor = CLIPImageProcessor.from_pretrained(clip_path)
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Quick forward pass
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print(f"  Pooler output shape: {out.pooler_output.shape}")
    print(f"  Last hidden shape: {out.last_hidden_state.shape}")
    print("  [OK] CLIP ViT-B/16 loaded successfully from local")


def test_clap():
    """Test 3: CLAP HTSAT audio encoder from local."""
    print("\n" + "=" * 60)
    print("Test 3: CLAP HTSAT (local)")
    print("=" * 60)

    from transformers import ClapModel, ClapProcessor

    clap_path = f"{MODEL_WEIGHTS}/clap-htsat-unfused"
    print(f"Loading from: {clap_path}")
    model = ClapModel.from_pretrained(clap_path)
    processor = ClapProcessor.from_pretrained(clap_path)
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Verify the model has expected methods (input format is dataset-specific)
    assert hasattr(model, 'get_audio_features'), "Missing get_audio_features"
    assert hasattr(model, 'get_text_features'), "Missing get_text_features"
    assert model.config.projection_dim == 512, f"Unexpected projection_dim: {model.config.projection_dim}"
    print(f"  CLAP API verified: get_audio_features, get_text_features, projection_dim=512")
    print("  [OK] CLAP HTSAT loaded successfully from local")


if __name__ == "__main__":
    results = []
    try:
        test_vjepa2()
        results.append(("V-JEPA 2 ViT-L (video)", "OK"))
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append(("V-JEPA 2 ViT-L (video)", f"FAIL: {e}"))

    try:
        test_clip()
        results.append(("CLIP ViT-B/16 (video)", "OK"))
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append(("CLIP ViT-B/16 (video)", f"FAIL: {e}"))

    try:
        test_clap()
        results.append(("CLAP HTSAT (audio)", "OK"))
    except Exception as e:
        print(f"  [FAIL] {e}")
        results.append(("CLAP HTSAT (audio)", f"FAIL: {e}"))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, status in results:
        print(f"  [{status[:3]}] {name}: {status}")
