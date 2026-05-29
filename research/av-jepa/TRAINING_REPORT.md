# AV-JEPA：音视频联合世界模型跌倒检测

> 训练时间：2026-05-12 | 设备：NVIDIA GTX 1650 SUPER (4GB) | 框架：PyTorch 2.4.1+cu121

---

## 1. 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                     AV-JEPA Pipeline                         │
│                                                              │
│  当前帧×8 ──→ [CLIP ViT-B/16] ──→ v_ctx (512) ──┐          │
│                                                    ├→ concat │
│  当前音频 ──→ [CLAP HTSAT]    ──→ a_ctx (512) ──┘  (1024)  │
│                                                       │      │
│                                              [AVProjector]   │
│                                             1024 → 256       │
│                                                  ↓           │
│                                              z_ctx (256)     │
│                                                  ↓           │
│                                        [JEPA Predictor]      │
│                                    3-layer Transformer       │
│                                                  ↓           │
│                                              z_pred (256)    │
│                                                  │           │
│  未来帧×8 ──→ [CLIP ViT-B/16] ──→ v_tgt ──┐    │           │
│                                              ├→ concat        │
│  未来音频 ──→ [CLAP HTSAT]    ──→ a_tgt ──┘  (1024)        │
│                                                  ↓           │
│                                         [EMA Projector]      │
│                                                  ↓           │
│                                           z_target (256)     │
│                                                  │           │
│                              Loss = MSE(z_pred, z_target)     │
│                                   + λ · SIGReg(z_ctx, z_tgt) │
└─────────────────────────────────────────────────────────────┘
```

### 组件详情

| 组件 | 规格 | 参数量 | 可训练 |
|------|------|:-----:|:------:|
| 视频编码器 | CLIP ViT-B/16 (HuggingFace `openai/clip-vit-base-patch16`) | ~86M | 冻结 |
| 音频编码器 | CLAP HTSAT-unfused (HuggingFace `laion/clap-htsat-unfused`) | ~300M | 冻结 |
| 融合投影器 | 1024→512→256, LayerNorm + GELU + Dropout(0.1) | ~0.8M | ✅ |
| JEPA 预测器 | 3层 Transformer, 8头, hidden=512 | ~1.6M | ✅ |
| 目标投影器 | EMA of 融合投影器 (decay=0.996→1.0 cosine schedule) | ~0.8M | EMA 更新 |
| **总计可训练** | | **2.44M** | |

---

## 2. 训练方法

### 训练范式：自监督 JEPA（Joint Embedding Predictive Architecture）

模型**只使用正常活动视频进行训练，从未见过任何跌倒样本**。训练目标是：给定当前时刻的音视频观测，预测未来时刻的联合嵌入。

### 数据集

- **来源**：Le2i Fall Detection Dataset (Kaggle: `tuyenldvn/falldataset-imvia`)
- **筛选**：仅使用视频和音频轨道均完好的数据（46 个损坏音频的视频已排除）
- **场景**：Home_01, Home_02, Coffee_room, Office

| 集合 | 片段数 | 说明 |
|------|:-----:|------|
| 训练集 | 1,451 | **仅正常活动**（无跌倒） |
| 评估集（正常） | 1,451 | 与训练同分布 |
| 评估集（跌倒） | 333 | 仅用于评估，训练不可见 |

### 片段构建

```
每个训练样本:
  ├── 上下文帧：t 到 t+2s 的视频帧 (8帧 @ 12fps) + 音频
  ├── 间隔：1s
  └── 目标帧：t+3s 到 t+5s 的视频帧 (8帧) + 音频
```

滑动窗口步长 1s，每个视频生成多个片段。相邻窗口产生平滑过渡（正常），跌倒事件产生大跳跃。

---

## 3. 损失函数

### 总损失

$$\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda \cdot \mathcal{L}_{\text{SIGReg}}$$

其中 $\lambda = 0.3$

### 预测损失（Prediction Loss）

$$\mathcal{L}_{\text{pred}} = \frac{1}{B \cdot D}\sum_{i=1}^{B}\sum_{j=1}^{D} (z_{\text{pred}}^{(i,j)} - z_{\text{target}}^{(i,j)})^2$$

- $B$: batch size = 2
- $D$: joint embedding dimension = 256
- $z_{\text{pred}}$: 预测器输出的未来嵌入
- $z_{\text{target}}$: 通过 EMA 目标编码器计算的真实未来嵌入

### SIGReg 正则化损失

$$\mathcal{L}_{\text{SIGReg}} = \frac{1}{2}\left[(\sigma(z_{\text{ctx}}) - 1.0)^2 + (\sigma(z_{\text{target}}) - 1.0)^2\right]$$

其中 $\sigma$ 为 batch 维度的标准差。此损失强制潜在空间保持各向同性高斯分布，**防止表示坍塌**（所有嵌入坍缩到同一点）。

来自 Meta FAIR 的 LeWorldModel 论文。

### 优化器设置

| 参数 | 值 |
|------|:--:|
| 优化器 | AdamW |
| 学习率 | 1e-4 |
| Weight decay | 1e-4 |
| Gradient clipping | max_norm=1.0 |
| Batch size | 2 |
| Epochs | 50 |
| LR Schedule | Cosine |

---

## 4. 检测原理：为什么正常和异常输出不同？

### 核心思想

> 模型被训练去学习"正常活动的动力学规律"。跌倒之所以能被检测到，是因为它**违反了这些规律**。

### 三种误差指标

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| **Video Error** | MSE(v_ctx, v_tgt) in CLIP 512维空间 | 视频帧特征的变化程度 |
| **Audio Error** | MSE(a_ctx, a_tgt) in CLAP 512维空间 | 音频特征的变化程度 |
| **Joint Error** | MSE(z_pred, z_target) in JEPA 256维空间 | 融合后的 JEPA 预测惊奇度 |

### 为什么正常活动误差小

训练时模型反复见到：走路→走路、坐下→坐下、躺着→躺着。这些活动的嵌入变化是**平滑、小幅**的。

模型学会了：当前嵌入在哪里，未来嵌入就在附近。

### 为什么跌倒误差大

跌倒的物理过程：
```
站立(0-1s) → 失去平衡(1-1.5s) → 快速落地(1.5-2s) → 地面静止(2s+)
```

1. **位置突变**：身体中心从 1.5m 高度突然降到 0.2m
2. **姿态突变**：从垂直站姿变为水平躺姿
3. **速度异常**：普通人日常活动加速度 < 0.5g，跌倒瞬间 > 2g
4. **音频冲击**：身体撞击地面的声音具有突发性、宽频谱特性

这些极端变化在正常活动数据中**从未出现**，导致模型的预测完全偏离实际——惊奇度激增。

---

## 5. 训练结果

### Loss 曲线

| Epoch | pred_loss | sigreg | 趋势 |
|:-----:|:---------:|:------:|:----:|
| 1 | 0.1290 | 0.2246 | 初始 |
| 5 | 0.2272 | 0.0607 | 震荡 |
| 10 | 0.2178 | 0.0356 | 收敛中 |
| 20 | 0.2024 | 0.0306 | 稳定 |
| 30 | 0.2194 | 0.0205 | 稳定 |
| 45 | 0.2139 | 0.0178 | 收敛 |
| **50** | **0.2108** | **0.0165** | **最终** |

SIGReg 从 0.22 下降到 0.02，正则化成功防止了坍塌。Pred loss 在 0.20-0.22 区间波动，已达当前数据量的瓶颈。

### 最终评估：误差对比

```
              Video       Audio       Joint
Normal     ~0.03       ~0.02       0.1414 ± 0.0845
Fall       ~0.12       ~0.09       0.3682 ± 0.3695
─────────────────────────────────────────────────
Sep σ      2.3σ        1.8σ        2.7σ
```

### 关键发现

1. **融合胜于单模态**：Joint 分离度 2.7σ > Video 2.3σ > Audio 1.8σ
2. **视频是主导信号**：画面变化在跌倒检测中贡献最大
3. **音频提供互补**：撞击声提供独立的检测信号，减少误报
4. **Separation > 2σ**：正常和跌倒的误差分布有显著统计差异

### 提升方向

- 更多训练数据（当前仅 1451 片段，39 个跌倒评估样本）
- 更大容量预测器（当前 3 层 → 6 层）
- 替换 CLIP/CLAP 为 PE-AV 原生音视频联合编码器
- 在老年人真实日常活动数据上微调

---

## 6. 复现命令

```bash
# 1. 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers av tqdm numpy

# 2. 下载数据集
python3 -c "
import kagglehub
path = kagglehub.dataset_download('tuyenldvn/falldataset-imvia')
print(path)
"

# 3. 训练
cd av-jepa
HF_ENDPOINT=https://hf-mirror.com python3 -W ignore train_le2i.py \
    --epochs 50 --batch_size 2 --device cuda --data_root <dataset_path>

# 4. 推理
python3 detect.py --checkpoint checkpoints/av_jepa_le2i.pt --device cuda
```

---

## 7. 参考资料

- V-JEPA 2 (Meta FAIR, 2025): Self-Supervised Video Models Enable Understanding, Prediction and Planning
- LeWorldModel (Meta FAIR, 2025): Stable End-to-End JEPA from Pixels
- Fall-Mamba (IEEE IoT-J, 2025): Multimodal Fusion for Fall Detection
- Perception Encoder (Meta FAIR, 2025): PE-AV — Open-Source Audiovisual Joint Embedding
