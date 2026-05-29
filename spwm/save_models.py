#!/usr/bin/env python3
"""
Save all required pretrained models to spwm/model_weights/ from local HF/torch cache.

Only video + audio modalities:
  - CLIP ViT-B/16 (V1 video)
  - CLAP HTSAT (V1 + spwm audio)
  - V-JEPA 2 ViT-L (spwm video) — vitl.pt + source code
"""

import os
import shutil
import torch
from pathlib import Path

MODEL_WEIGHTS = Path("/Volumes/forya/falling/spwm/model_weights")
MODEL_WEIGHTS.mkdir(parents=True, exist_ok=True)


def save_clip_vit_b16():
    """Save CLIP ViT-B/16 vision model to local dir."""
    dest = MODEL_WEIGHTS / "clip-vit-base-patch16"
    if dest.exists():
        print(f"[CLIP ViT-B/16] Already exists at {dest}")
        return True

    print("[CLIP ViT-B/16] Saving from HF cache...")
    from transformers import CLIPVisionModel, CLIPImageProcessor

    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

    model.save_pretrained(dest)
    processor.save_pretrained(dest)
    print(f"[CLIP ViT-B/16] Saved to {dest}")
    return True


def save_clap_htsat():
    """Save CLAP HTSAT audio model to local dir."""
    dest = MODEL_WEIGHTS / "clap-htsat-unfused"
    if dest.exists():
        print(f"[CLAP HTSAT] Already exists at {dest}")
        return True

    print("[CLAP HTSAT] Saving from HF cache...")
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    model.save_pretrained(dest)
    processor.save_pretrained(dest)
    print(f"[CLAP HTSAT] Saved to {dest}")
    return True


def copy_vjepa2():
    """Copy V-JEPA 2 checkpoint and source code to local dir."""
    dest_checkpoint = MODEL_WEIGHTS / "vitl.pt"
    dest_code = MODEL_WEIGHTS / "vjepa2"

    all_ok = True

    # Copy the checkpoint
    src_checkpoint = Path.home() / ".cache/torch/hub/checkpoints/vitl.pt"
    if dest_checkpoint.exists():
        print(f"[V-JEPA 2] Checkpoint already exists: {dest_checkpoint}")
    elif src_checkpoint.exists():
        print(f"[V-JEPA 2] Copying vitl.pt ({src_checkpoint.stat().st_size / 1e9:.1f} GB)...")
        shutil.copy2(src_checkpoint, dest_checkpoint)
        print(f"[V-JEPA 2] Checkpoint copied to {dest_checkpoint}")
    else:
        print(f"[V-JEPA 2] ERROR: vitl.pt not found at {src_checkpoint}")
        all_ok = False

    # Copy the source code
    src_code = Path.home() / ".cache/torch/hub/facebookresearch_vjepa2_main"
    if dest_code.exists():
        print(f"[V-JEPA 2] Source code already exists: {dest_code}")
    elif src_code.exists():
        print(f"[V-JEPA 2] Copying source code...")
        shutil.copytree(src_code, dest_code, dirs_exist_ok=True)
        print(f"[V-JEPA 2] Source code copied to {dest_code}")
    else:
        print(f"[V-JEPA 2] ERROR: source code not found at {src_code}")
        all_ok = False

    return all_ok


def main():
    print("=" * 60)
    print("Saving models to spwm/model_weights/")
    print("=" * 60)

    results = {
        "CLIP ViT-B/16 (video)": save_clip_vit_b16(),
        "CLAP HTSAT (audio)": save_clap_htsat(),
        "V-JEPA 2 ViT-L (video)": copy_vjepa2(),
    }

    print("\n" + "=" * 60)
    print("Summary:")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {name}")

    print(f"\nModels in: {MODEL_WEIGHTS}")
    total = sum(
        sum(f.stat().st_size for f in MODEL_WEIGHTS.rglob("*") if f.is_file())
    )
    print(f"Total size: {total / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
