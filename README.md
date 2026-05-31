# Falling — Multimodal Fall Detection with JEPA

Joint Embedding Predictive Architecture (JEPA) for elderly fall detection. Three approaches:

| Approach | Modality | Trainable Params | F1 | AUC | VRAM | Prediction Horizon |
|---|---|---|---|---|---|---|
| **JEPA Classifier** | Video + Audio | 1.05M | **1.000** | **1.000** | ~6 GB | 0.3s |
| V2 Anomaly | Video + Audio | — | — | Surprise 4.46× | ~6 GB | unsupervised |
| **Skeleton-JEPA** | 2D Keypoints | 265K | 0.576 | 0.887 | **~0.5 GB** | 1.28s |

---

## Hardware Requirements

| Approach | GPU VRAM | Disk | External Data |
|---|---|---|---|
| **Skeleton-JEPA** (Le2i) | Any GPU (≥2 GB) or CPU | ~500 MB | **None** — data included in repo |
| Skeleton-JEPA + NTU120 pretrain | ≥4 GB | +35 GB | NTU120 skeleton dataset |
| Video+Audio Classifier | ≥10 GB | +17 GB + 6.5 GB | Le2i videos + pretrained encoders |

---

## Quick Start — Skeleton-JEPA (Recommended First Step)

This approach needs **zero external downloads**. All keypoint data is pre-extracted and included in this repository.

### 1. Clone and install

```bash
git clone https://github.com/Yangkailiang1/falling.git
cd falling
pip install -r requirements.txt
```

### 2. Verify with a fast smoke test (CPU, ~1 minute)

```bash
python3 -m spwm.skeleton_jepa_train --phase both --overfit_test --device cpu
```

This does a quick overfit on 4 videos. Should see loss dropping rapidly — confirms everything works.

### 3. Full training on Le2i keypoints (GPU recommended, ~4 minutes)

```bash
# With 17 COCO keypoints (YOLOv8-pose)
python3 -m spwm.skeleton_jepa_train --phase both \
  --jepa_epochs 200 --cls_epochs 200 \
  --d_model 256 --n_layers 4 \
  --gap_frames 16 --future_frames 32 \
  --device cuda

# With 25 NTU-aligned keypoints (MediaPipe) — better for NTU transfer
python3 -m spwm.skeleton_jepa_train --phase both \
  --data_root le2i_keypoints_25 --num_keypoints 25 \
  --jepa_epochs 200 --cls_epochs 200 \
  --d_model 256 --n_layers 4 \
  --gap_frames 16 --future_frames 32 \
  --device cuda
```

**Expected output on Le2i (25-joint):** F1≈0.58, AUC≈0.89. The bottleneck is only ~66 positive training samples from 127 videos — this is why NTU120 pretraining matters (see below).

### 4. (Optional) NTU120 large-scale JEPA pretraining

Pretraining on 114K NTU skeleton sequences dramatically improves the encoder. Requires downloading the NTU120 skeleton dataset.

```bash
# 1. Download NTU120 skeleton data (35 GB)
git clone https://www.modelscope.cn/datasets/yangkailiang12/NTU120_skeleton.git
# Place it at: NTU120_skeleton/nturgbd_skeletons/

# 2. Phase 1: Pretrain on NTU120 (~22 hours for 200 epochs on RTX 4090)
python3 -m spwm.ntu_jepa_pretrain \
  --data_root NTU120_skeleton/nturgbd_skeletons \
  --batch_size 64 --epochs 200 \
  --device cuda

# 3. Phase 2: Transfer to Le2i classifier
python3 -m spwm.skeleton_jepa_train --phase classify \
  --resume_jepa checkpoints/ntu_jepa_best.pt \
  --data_root le2i_keypoints_25 --num_keypoints 25 \
  --cls_epochs 200 --device cuda
```

**VRAM:** ~0.5 GB (batch_size=64). Any GPU with ≥2 GB works. CPU training is also viable (~20× slower per epoch).

---

## Video+Audio JEPA Classifier (F1=1.000)

This approach uses frozen V-JEPA2 ViT-L (304M) + WavJEPA (196M) as feature extractors with a lightweight trainable MLP (1.05M params).

### Prerequisites

```bash
# 1. Download Le2i raw videos (17 GB)
python3 -c "import kagglehub; kagglehub.dataset_download('tuyenldvn/falldataset-imvia')"
# Symlink or copy to: Le2i/

# 2. Download V-JEPA2 pretrained weights (~3 GB)
git clone https://github.com/facebookresearch/vjepa2.git spwm/model_weights/vjepa2/
# Download vitl.pt from facebookresearch/vjepa2 → spwm/model_weights/vitl.pt

# 3. WavJEPA auto-downloads from HuggingFace on first run
```

### Training

```bash
python3 -m spwm.classifier_train --balanced --epochs 200 \
  --batch_size 8 --device cuda --encoder_device cuda \
  --pos_weight 1.0 --neg_stride 32
```

**Converges in ~13 epochs** to perfect test-set performance (F1=1.000, AUC=1.000).

---

## Architecture

### 1. JEPA Classifier (Video + Audio)

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
  Context skeleton (16 frames) → Transformer Encoder → z_ctx (256-d)
  Target skeleton (16 frames)  → EMA Encoder         → z_tgt (256-d)
  Predictor(z_ctx) → z_pred,  Loss: cosine_distance(z_pred, z_tgt)
  Gap: 16 frames (0.64s) → predicts 0.64-1.28s into the future

Phase 2 (Classifier):
  Skeleton → Frozen Encoder → Fusion MLP (256→512→256) → P(fall in next 1.28s)
```

---

## Repository Structure

```
falling/
├── spwm/                          # Main source code
│   ├── classifier_model.py        # Video+audio JEPA classifier
│   ├── classifier_train.py        # Supervised BCE classifier training
│   ├── skeleton_jepa_model.py     # Skeleton encoder, predictor, JEPA, classifier
│   ├── skeleton_jepa_train.py     # Two-phase skeleton training CLI
│   ├── ntu_jepa_pretrain.py       # NTU120 skeleton JEPA pretraining
│   ├── encoders.py                # V-JEPA2, WavJEPA encoder wrappers
│   ├── config.py                  # Configuration dataclasses
│   ├── v2_fall_detect.py          # Unsupervised JEPA anomaly detection
│   ├── downloads.py               # Download helper for weights/datasets
│   ├── visualize_predictions.py   # Static P(fall) vs time plots
│   ├── render_video_predictions.py # MP4 videos with P(fall) overlay
│   ├── test_with_synthetic.py     # 11 smoke tests
│   └── data/
│       ├── le2i_dataset.py              # Raw video dataset (PyAV)
│       ├── skeleton_dataset.py          # Keypoint dataset loader (.npy)
│       ├── ntu_dataset.py               # NTU120 skeleton dataset loader
│       ├── balanced_le2i_dataset.py     # Class-balanced video-level sampler
│       ├── prepare_le2i_split.py        # Train/test split generator
│       ├── skeleton_extractor.py        # YOLOv8-pose keypoint extraction
│       └── extract_le2i_25kp.py         # MediaPipe 25-joint extraction
├── le2i_keypoints/                # 127 clips, 17 COCO keypoints (8.3 MB, in repo)
├── le2i_keypoints_25/             # 130 clips, 25 NTU-aligned keypoints (13 MB, in repo)
├── le2i_split.json                # Video-level train/test split (40 KB, in repo)
├── requirements.txt               # Python dependencies
├── research/                      # Surveyed papers and early prototypes
│
# Below are gitignored — must be downloaded separately:
├── Le2i/                          # Raw AVI videos (17 GB)
├── NTU120_skeleton/               # NTU120 skeleton data (35 GB)
├── checkpoints/                   # Trained model checkpoints
├── spwm/model_weights/            # Pretrained encoders (6.5 GB)
└── logs/                          # Training logs
```

---

## Python Environment

- **Python**: 3.10+ (use `python3` — system `python` may be 2.7)
- Tested with: torch 2.6.0+cu124, transformers 5.8.1, CUDA 12.4/13.0
- **Supports CPU-only training** for skeleton approaches (set `--device cpu`)

## Pretrained Checkpoints

Checkpoints are published on [ModelScope](https://www.modelscope.cn/models/yangkailiang12/va-jepa):

```bash
git clone https://www.modelscope.cn/yangkailiang12/va-jepa.git
cd va-jepa && pip install -r requirements.txt
python demo.py --test --device cuda
```

## Data Sources

| Data | Size | Source |
|------|------|--------|
| Le2i raw videos | 17 GB | [Kaggle](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia) |
| Le2i 17-joint keypoints | 8.3 MB | Included in repo (extracted via YOLOv8-pose) |
| Le2i 25-joint keypoints | 13 MB | Included in repo (extracted via MediaPipe) |
| NTU120 skeleton | 35 GB | [ModelScope](https://www.modelscope.cn/datasets/yangkailiang12/NTU120_skeleton) |
| V-JEPA 2 ViT-L | ~3 GB | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) |
| WavJEPA-Base | ~2 GB | HuggingFace `facebook/wavjepa` |
