# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multimodal fall detection for elderly people. Three approaches:

1. **JEPA Classifier** (`spwm/classifier_*.py`): Supervised. Frozen V-JEPA 2 + WavJEPA encoders → MLP → P(fall in next 0.3s). **1.05M trainable params**. F1=1.000, AUC=1.000 on test set.

2. **V-JEPA V2 Anomaly** (`spwm/v2_fall_detect.py`): Unsupervised. JEPA prediction error on normal activity → anomaly score. **Surprise ratio: 4.46x**.

3. **Skeleton-JEPA** (`spwm/skeleton_jepa_*.py`): Two-phase. Phase 1: JEPA pretraining on 2D keypoints → predict future skeleton states (0.64-1.28s ahead). Phase 2: Frozen encoder + MLP classifier. **~265K trainable params**. F1=0.58, AUC=0.89 (bottleneck: only 66 train positives).

## Data Files (gitignored — NOT in repo)

| Path | Size | Description | Source |
|------|------|-------------|--------|
| `Le2i/` | 17G | Raw AVI files (25fps, 320×240) with fall annotations | [Kaggle](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia) (`kagglehub.dataset_download("tuyenldvn/falldataset-imvia")`) |
| `checkpoints/` | ~4G | Model checkpoints (classifier_best.pt 1.9G, skeleton checkpoints ~70M) | Local training |
| `spwm/model_weights/` | 6.5G | Pretrained encoders: V-JEPA2 ViT-L, WavJEPA, CLIP, CLAP | See below |
| `logs/` | ~1.4M | Training logs | — |

## Model Weights (`spwm/model_weights/` — ~6.5G total, gitignored)

| Model | Path | Params | Input → Output |
|-------|------|--------|----------------|
| V-JEPA 2 ViT-L | `vitl.pt` + `vjepa2/` | 303.9M | `(B,C,T,H,W)` → `(B,1568,1024)` |
| WavJEPA-Base | `wavjepa-base/` | 196.3M | `(B,1,48000)` → `(B,299,768)` |
| CLIP ViT-B/16 | `clip-vit-base-patch16/` | 85.8M | `(B,3,H,W)` → `(B,768)` |
| CLAP HTSAT | `clap-htsat-unfused/` | 153.5M | mel spectrogram → `(B,512)` |

All load from local paths — no online download needed at runtime.

| Weight | Source |
|--------|--------|
| V-JEPA 2 ViT-L (vitl.pt) | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) |
| WavJEPA-Base | [facebookresearch/wavjepa](https://github.com/facebookresearch/wavjepa) |
| CLIP ViT-B/16 | HuggingFace `openai/clip-vit-base-patch16` |
| CLAP HTSAT | HuggingFace `laion/clap-htsat-unfused` |

See `spwm/downloads.py` for download instructions.

## Skeleton Data Sources

| Path | Size | Description | Source |
|------|------|-------------|--------|
| `le2i_keypoints/` | 8.3M | 127 clips, 17 COCO keypoints (YOLOv8-pose) | Generated via `spwm/data/skeleton_extractor.py` |
| `le2i_keypoints_25/` | 13M | 130 clips, 25 NTU-aligned keypoints (MediaPipe) | Generated via `spwm/data/extract_le2i_25kp.py` |
| `le2i_split.json` | 40K | Video-level train/test split (seed=42) | Generated via `spwm/data/prepare_le2i_split.py` |
| `research/` | ~30M | Surveyed papers | — |

**NTU RGB+D 120 skeleton data for large-scale JEPA pretraining**: downloaded to `NTU120_skeleton/` (gitignored, 35G).
- 89,314 sequences → 421K JEPA windows. Training via `spwm/ntu_jepa_pretrain.py`.

- Source: [ModelScope](https://www.modelscope.cn/datasets/yangkailiang12/NTU120_skeleton) (`git clone` with LFS)
- Status: Downloaded 2026-05-29, extracted to `NTU120_skeleton/` (35G on disk, gitignored)
- 89,314 sequences after filtering (min 48 frames), producing 421K JEPA context-target windows
- Training: `spwm/ntu_jepa_pretrain.py` — 3.84M params, ~1600 batches/epoch @ batch_size=256
- After pretraining: transfer encoder to Le2i classifier via `spwm/skeleton_jepa_train.py --phase classify --resume_jepa checkpoints/ntu_jepa_best.pt`

## JEPA Classifier (Video + Audio) — F1=1.000

```
Context 16 frames (0.64s) → V-JEPA 2 ViT-L (frozen) → mean pool → v_feat (1024)
Context Audio (0.64s)     → WavJEPA (frozen)        → mean pool → a_feat (768)
                                                                    ↓
                                                 Fusion MLP: 1792 → 512 → 256
                                                                    ↓
                                                 Classifier: 256 → 1 → P(fall in next 0.3s)
```

- Frozen encoders (500M params total), trainable MLP = 1.05M params
- BCEWithLogitsLoss with pos_weight, single-stage supervised
- Perfect test set at epoch 13: 0 FP, 0 FN (all 17 leak levels at 1.00)
- Checkpoint: `checkpoints/classifier_best.pt` (1.9 GB, includes all encoder weights)

```bash
# Training
python3 -m spwm.classifier_train --balanced --epochs 200 --batch_size 8 \
  --device cuda --encoder_device cuda --pos_weight 1.0 --neg_stride 32

# Overfit test
python3 -m spwm.classifier_train --data_root Le2i --epochs 30 --batch_size 8 \
  --overfit_test --device cuda --encoder_device cuda
```

## Skeleton-JEPA — F1=0.58, AUC=0.89

Two-phase approach on 2D skeleton keypoints. Supports both 17 COCO (`le2i_keypoints/`) and 25 NTU-aligned (`le2i_keypoints_25/`, matching NTU120 format for transfer).

### Phase 1: JEPA Pretraining (self-supervised, normal activity only)

```
Context 16 frames → SkeletonEncoder (Transformer, 3.3M) → z_ctx (256-d)
Target  16 frames → SkeletonEncoder (EMA)               → z_tgt (256-d)
Predictor: z_ctx → MLP → z_pred
Loss: 2 - 2·cos(z_pred, z_tgt)  [cosine distance]
```

- Gap: 16 frames (0.64s) → predicts 0.64-1.28s into the future
- Target encoder: EMA 0.996 → 1.0
- Best V-loss: 0.094 (cos θ ≈ 18°)

### Phase 2: Supervised Classifier

```
Skeleton (16,17,3) → Frozen SkeletonEncoder → mean pool → MLP(256→512→256) → P(fall in 1.28s)
```

- Trainable: 265K params (fusion MLP + classifier head)
- `future_frames=32` (1.28s, 4× further than video classifier)

### Current Performance (d=256, L=4, ns=16, f=32)

| | Train (646) | Test (172) |
|---|-------------|------------|
| Accuracy | 0.957 | 0.855 |
| Precision | 0.702 | 0.436 |
| Recall | 1.000 | 0.850 |
| F1 | 0.825 | 0.576 |
| AUC | 0.999 | 0.887 |

Per-leak (test): leak=1 50%, leak=2 0%, all others 100%.

### Bottleneck

Only 66 positive training samples from 127 videos. JEPA pretrained on 1133 windows from 27 non-fall videos — far too little to learn generalizable human motion representations. **Plan: pretrain on NTU RGB+D 120 (114K skeleton samples).**

### Training

```bash
# Overfit test
python3 -m spwm.skeleton_jepa_train --phase both --overfit_test \
  --jepa_epochs 100 --cls_epochs 100 --device cpu

# Full training (GPU)
python3 -m spwm.skeleton_jepa_train --phase both \
  --jepa_epochs 200 --cls_epochs 200 \
  --gap_frames 16 --future_frames 32 \
  --d_model 256 --n_layers 4 --device cuda
```

## Key spwm/ Files

| File | Role |
|------|------|
| `classifier_model.py` | `JEPAClassifier` — video+audio fusion MLP + classifier |
| `classifier_train.py` | Supervised BCE training with class balancing |
| `encoders.py` | `VJEPA2VideoEncoder`, `AudioJEPEncoder` |
| `skeleton_jepa_model.py` | `SkeletonEncoder`, `SkeletonPredictor`, `SkeletonJEPA`, `SkeletonClassifier` |
| `skeleton_jepa_train.py` | Two-phase training CLI: JEPA pretrain + supervised classifier |
| `data/skeleton_dataset.py` | `SkeletonDataset` — loads .npy keypoints, normalization, JEPA + classification modes |
| `data/le2i_dataset.py` | `Le2iDataset` — PyAV decoding, annotation parsing |
| `data/prepare_le2i_split.py` | Scene-stratified video-level split → `le2i_split.json` |
| `data/balanced_le2i_dataset.py` | `BalancedLe2iDataset` — one sample per video, balanced leak levels |
| `config.py` | Dataclass configs |
| `v2_fall_detect.py` | V2 unsupervised JEPA anomaly detection |
| `visualize_predictions.py` | P(fall) vs time static plots |
| `render_video_predictions.py` | MP4 videos with P(fall) overlay |
| `test_with_synthetic.py` | 11 smoke tests |

### Deprecated (kept for reference)

| File | Role |
|------|------|
| `tjepa_model.py` | Old `TJEPS` — MoE + Mamba + 3-tier gate |
| `fusion.py` | Old `M3JEPAFusion`, `HybridFusion` |
| `predictor.py` | Old `MambaBlock`, `VJEPA2Predictor` |
| `train.py` | Old 4-stage training CLI |
| `anomaly_gate.py`, `detector.py`, `projector.py` | Old T-JEPA components |

## Runtime Environment

- **Python**: Must use `python3` (system `python` is Python 2.7)
- **GPU**: 2× RTX 4090 (48GB VRAM each). GPU 0 for training, GPU 1 idle. CUDA 13.0, Driver 580.95.05.
- **VRAM**: 48GB per GPU — enough for frozen encoders + training on same GPU.

## Key Dependencies

- `torch` / `torchvision` (2.4.1+cu121)
- `transformers` (CLIP, CLAP, WavJEPA)
- `timm` + `einops` (V-JEPA 2 backend)
- `av` (PyAV — video/audio decoding)
- `kagglehub` (Le2i download)
- `tqdm`, `numpy`, `scikit-learn`

## ModelScope Deployment

Published at **[yangkailiang12/va-jepa](https://www.modelscope.cn/models/yangkailiang12/va-jepa)**.

```bash
pip install modelscope
modelscope upload yangkailiang12/va-jepa /tmp/va-jepa \
  --token "ms-..." \
  --exclude ".git" ".git/*" "__pycache__" "*.pyc"
```
