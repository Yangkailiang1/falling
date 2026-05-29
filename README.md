# Falling — Multimodal Fall Detection with JEPA

Joint Embedding Predictive Architecture (JEPA) for elderly fall detection on the [Le2i](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia) dataset. Three approaches: video+audio supervised classifier, unsupervised anomaly detection, and skeleton-based future prediction.

## Results

| Approach | Modality | Params | F1 | AUC | Prediction |
|----------|----------|--------|-----|-----|------------|
| **JEPA Classifier** | Video + Audio | 1.05M | **1.000** | **1.000** | 0.3s ahead |
| V2 Anomaly | Video + Audio | — | — | Surprise 4.46× | unsupervised |
| **Skeleton-JEPA** | 2D Keypoints | 265K | 0.576 | 0.887 | 1.28s ahead |

## Architecture

### 1. JEPA Classifier (Video + Audio)

Frozen V-JEPA 2 ViT-L (304M) + WavJEPA (196M) as feature extractors → lightweight fusion MLP → binary classifier.

```
Video 16 frames (0.64s) → V-JEPA 2 ViT-L → mean pool → 1024-d
Audio (0.64s)            → WavJEPA        → mean pool → 768-d
                                    ↓
                         Fusion MLP: 1792 → 512 → 256
                                    ↓
                         P(fall in next 0.3s)
```

### 2. Skeleton-JEPA (2D Keypoints)

Two-phase: JEPA self-supervised pretraining on normal activity → frozen encoder + supervised classifier.

```
Phase 1 (JEPA):
  Context skeleton 16 frames → Transformer Encoder → z_ctx
  Target skeleton 16 frames  → EMA Encoder         → z_tgt
  Predictor(z_ctx) → z_pred,  Loss: cosine_distance(z_pred, z_tgt)

Phase 2 (Classifier):
  Skeleton → Frozen Encoder → Fusion MLP → P(fall in next 1.28s)
```

## Project Structure

```
falling/
├── spwm/                     # Main source code
│   ├── classifier_model.py   # Video+audio classifier
│   ├── classifier_train.py   # Classifier training
│   ├── skeleton_jepa_model.py # Skeleton-JEPA model
│   ├── skeleton_jepa_train.py # Skeleton-JEPA training (two-phase)
│   ├── encoders.py           # V-JEPA2, WavJEPA encoders
│   ├── config.py             # Configuration dataclasses
│   ├── v2_fall_detect.py     # Unsupervised anomaly detection
│   ├── visualize_predictions.py  # P(fall) vs time plots
│   ├── render_video_predictions.py # MP4 with prediction overlay
│   └── data/
│       ├── le2i_dataset.py       # Raw video dataset (PyAV)
│       ├── skeleton_dataset.py   # Keypoint dataset (.npy)
│       ├── balanced_le2i_dataset.py  # Balanced video-level split
│       ├── prepare_le2i_split.py # Split generator
│       └── skeleton_extractor.py # YOLOv8-pose keypoint extraction
├── le2i_keypoints/           # 127 clips, 17 COCO keypoints (.npy)
├── le2i_split.json           # Train/test split (seed=42)
├── research/                 # Surveyed papers
├── checkpoints/              # Trained models (gitignored)
└── logs/                     # Training logs (gitignored)
```

## Quick Start

### Requirements

```bash
pip install torch torchvision transformers timm einops av numpy scikit-learn tqdm
```

### Data

```bash
# Download Le2i dataset
python -m spwm.downloads --datasets

# Extract keypoints (optional, for skeleton-jepa)
python -m spwm.data.skeleton_extractor
```

### Training

```bash
# Video+Audio Classifier (GPU, ~3 min to convergence)
python3 -m spwm.classifier_train --balanced --epochs 200 \
  --batch_size 8 --device cuda --encoder_device cuda \
  --pos_weight 1.0 --neg_stride 32

# Skeleton-JEPA (two-phase, GPU)
python3 -m spwm.skeleton_jepa_train --phase both \
  --jepa_epochs 200 --cls_epochs 200 \
  --d_model 256 --n_layers 4 \
  --gap_frames 16 --future_frames 32 \
  --device cuda

# Quick overfit test (CPU, ~1 min)
python3 -m spwm.skeleton_jepa_train --phase both --overfit_test --device cpu
```

### Pretrained Checkpoints

Model published on [ModelScope](https://www.modelscope.cn/models/yangkailiang12/va-jepa):

```bash
git clone https://www.modelscope.cn/yangkailiang12/va-jepa.git
cd va-jepa && pip install -r requirements.txt
python demo.py --test --device cuda
```

## Model Weights

Pretrained encoders are loaded from `spwm/model_weights/` (gitignored, ~6.5G):

| Encoder | Params | Source |
|---------|--------|--------|
| V-JEPA 2 ViT-L | 304M | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) |
| WavJEPA-Base | 196M | [facebookresearch/wavjepa](https://github.com/facebookresearch/wavjepa) |
| CLIP ViT-B/16 | 86M | HuggingFace `openai/clip-vit-base-patch16` |
| CLAP HTSAT | 154M | HuggingFace `laion/clap-htsat-unfused` |

See `spwm/downloads.py` for automated download.

## Environment

- Python 3.10+ (use `python3`)
- GPU: 2× RTX 4090 (48GB each), CUDA 13.0
- Ubuntu 22.04
