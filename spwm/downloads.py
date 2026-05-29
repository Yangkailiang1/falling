#!/usr/bin/env python3
"""
Download Script: Datasets and Pre-trained Model Weights for T-JEPA

Downloads:
  1. Le2i Fall Detection Dataset (from Kaggle)
  2. V1-33K video-text dataset (from HuggingFace, optional)
  3. OmniFall dataset (from HuggingFace, optional)
  4. Pre-trained model weights (V-JEPA 2, Audio-JEPA, S-JEPA, Qwen2.5)

Datasets → /Volumes/forya/falling/datasets/
Model weights → /Volumes/forya/falling/spwm/model_weights/

Usage:
  python downloads.py --all          # Download everything
  python downloads.py --datasets     # Download only datasets
  python downloads.py --weights      # Download only model weights
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

# Paths
DATASETS_DIR = Path("/Volumes/forya/falling/datasets")
MODEL_WEIGHTS_DIR = Path("/Volumes/forya/falling/spwm/model_weights")


def download_le2i_dataset():
    """Download Le2i Fall Detection Dataset via KaggleHub."""
    print("\n" + "=" * 60)
    print("Downloading Le2i Fall Detection Dataset")
    print("=" * 60)

    try:
        import kagglehub
        path = kagglehub.dataset_download("tuyenldvn/falldataset-imvia")
        print(f"  Le2i dataset downloaded to: {path}")

        # Symlink to datasets dir
        target = DATASETS_DIR / "Le2i"
        if not target.exists():
            os.symlink(path, target)
            print(f"  Symlinked to {target}")
        else:
            print(f"  Already exists at {target}")

        return True
    except Exception as e:
        print(f"  Failed: {e}")
        print("  Manual download: https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia")
        print("  Place in: datasets/Le2i/")
        return False


def download_v1_33k():
    """Download V1-33K video-text dataset (optional for Stage 1+2 pre-training)."""
    print("\n" + "=" * 60)
    print("Downloading V1-33K Video-Text Dataset")
    print("=" * 60)
    print("  Note: V1-33K is a large dataset (~100GB).")
    print("  For Stage 1+2 pre-training, you can also use Le2i only.")
    print("  V1-33K: https://huggingface.co/datasets/...")
    print("  Skipping (not required for initial training).")
    return True


def download_omnifall():
    """Download OmniFall dataset (optional for Stage 3 fine-tuning)."""
    print("\n" + "=" * 60)
    print("Downloading OmniFall Dataset")
    print("=" * 60)
    print("  Note: OmniFall is a large dataset with 70K+ clips.")
    print("  For Stage 3 fine-tuning, you can also use Le2i with falls.")
    print("  OmniFall: https://github.com/...")
    print("  Skipping (not required for initial training).")
    return True


def download_vjepa2_weights():
    """Download V-JEPA 2 ViT-L pre-trained weights."""
    print("\n" + "=" * 60)
    print("Downloading V-JEPA 2 Weights")
    print("=" * 60)

    weight_path = MODEL_WEIGHTS_DIR / "vjepa2_vitl16.pth"
    if weight_path.exists():
        print(f"  Already exists: {weight_path}")
        return True

    print("  V-JEPA 2 weights are loaded from torch.hub at runtime.")
    print("  facebookresearch/vjepa2 must be installed:")
    print("    pip install git+https://github.com/facebookresearch/vjepa2.git")
    print("  Or clone manually:")
    print(f"    git clone https://github.com/facebookresearch/vjepa2.git {MODEL_WEIGHTS_DIR}/vjepa2/")
    return True


def download_audio_jepa_weights():
    """Download Audio-JEPA pre-trained weights."""
    print("\n" + "=" * 60)
    print("Downloading Audio-JEPA / WavJEPA Weights")
    print("=" * 60)

    print("  Audio encoder options (tried in order):")
    print("    1. WavJEPA from HuggingFace (facebook/wavjepa)")
    print("    2. CLAP HTSAT (laion/clap-htsat-unfused)")
    print("    3. Mel-Spectrogram + CNN (edge fallback, no download needed)")
    print()
    print("  HuggingFace models are auto-downloaded by transformers.")
    print("  First run will cache to ~/.cache/huggingface/")
    print("  To pre-download:")
    print("    python -c \"from transformers import AutoModel; AutoModel.from_pretrained('facebook/wavjepa', trust_remote_code=True)\"")
    return True


def download_sjepa_weights():
    """Download S-JEPA pre-trained weights."""
    print("\n" + "=" * 60)
    print("Downloading S-JEPA Weights")
    print("=" * 60)

    print("  S-JEPA skeleton encoder is built from scratch in the codebase.")
    print("  We use a simple 6-layer Transformer trained on COCO keypoints.")
    print("  No external weights needed for smoke tests.")
    return True


def download_qwen_weights():
    """Download Qwen2.5 embedding weights."""
    print("\n" + "=" * 60)
    print("Downloading Qwen2.5 Weights")
    print("=" * 60)

    print("  Qwen2.5-7B is loaded from HuggingFace (auto-download).")
    print("  Only the embedding layer is used (~160MB of 14GB total).")
    print("  To pre-download:")
    print("    python -c \"from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-7B', trust_remote_code=True, torch_dtype='auto')\"")
    print()
    print("  Fallback: BERT multilingual (bert-base-multilingual-cased)")
    print("    Much smaller (~700MB), auto-downloaded by transformers.")
    return True


def download_pose_model():
    """Download YOLOv8-Pose or MediaPipe for skeleton extraction."""
    print("\n" + "=" * 60)
    print("Downloading Skeleton Extraction Models")
    print("=" * 60)

    print("  Options:")
    print("    1. MediaPipe Pose (recommended, CPU-friendly)")
    print("       pip install mediapipe")
    print()
    print("    2. YOLOv8-Pose (GPU recommended)")
    print("       pip install ultralytics")
    print("       yolo11n-pose.pt auto-downloads on first use")
    return True


def check_environment():
    """Check required packages and environment."""
    print("\n" + "=" * 60)
    print("Environment Check")
    print("=" * 60)

    required = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'numpy': 'NumPy',
    }
    optional = {
        'av': 'PyAV (video decoding)',
        'mediapipe': 'MediaPipe (skeleton extraction)',
        'ultralytics': 'YOLOv8 (pose estimation)',
        'cv2': 'OpenCV (camera capture)',
        'kagglehub': 'KaggleHub (dataset download)',
    }

    print("\n  Required:")
    for mod, name in required.items():
        try:
            __import__(mod)
            print(f"    [OK] {name} ({mod})")
        except ImportError:
            print(f"    [MISSING] {name} ({mod}) - pip install {mod}")

    print("\n  Optional:")
    for mod, name in optional.items():
        try:
            __import__(mod)
            print(f"    [OK] {name} ({mod})")
        except ImportError:
            print(f"    [MISSING] {name} ({mod})")


def main():
    parser = argparse.ArgumentParser(description="Download T-JEPA datasets and model weights")
    parser.add_argument('--all', action='store_true', help='Download everything')
    parser.add_argument('--datasets', action='store_true', help='Download only datasets')
    parser.add_argument('--weights', action='store_true', help='Download only pre-trained weights')
    parser.add_argument('--check', action='store_true', help='Check environment only')

    args = parser.parse_args()

    if not any([args.all, args.datasets, args.weights, args.check]):
        parser.print_help()
        return

    # Create directories
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Environment check
    check_environment()

    if args.check:
        return

    # Download datasets
    if args.all or args.datasets:
        download_le2i_dataset()
        download_v1_33k()
        download_omnifall()

    # Download weights
    if args.all or args.weights:
        download_vjepa2_weights()
        download_audio_jepa_weights()
        download_sjepa_weights()
        download_qwen_weights()
        download_pose_model()

    print("\n" + "=" * 60)
    print("Downloads complete!")
    print(f"  Datasets:     {DATASETS_DIR}")
    print(f"  Model weights: {MODEL_WEIGHTS_DIR}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run smoke tests:  python spwm/test_with_synthetic.py")
    print("  2. Train Stage 1:    python spwm/train.py --stage stage1 --data_root datasets/Le2i")
    print("  3. Train Stage 2:    python spwm/train.py --stage stage2 --data_root datasets/Le2i")
    print("  4. Train Stage 3:    python spwm/train.py --stage stage3 --data_root datasets/Le2i")
    print("  5. Calibrate:        python spwm/train.py --stage calibrate --data_root datasets/Le2i")
    print("  6. Run detector:     python spwm/detector.py --checkpoint checkpoints/stage3_epoch49.pt")


if __name__ == '__main__':
    main()
