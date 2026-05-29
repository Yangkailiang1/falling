# 多模态+世界模型跌倒检测 —— 论文调研与代码整理

> 调研时间：2026年5月  
> 项目方向：老年人跌倒识别检测，多模态（视频+音频），世界模型架构  
> 说明：本目录按论文分类整理，每个子目录包含论文介绍、代码（如有）和数据集信息

---

## 目录总览

| 编号 | 文件夹 | 方向 | 代码 | 数据集 |
|------|--------|------|------|--------|
| 01 | [fall-mamba](#01-fall-mamba-多模态融合sota) | 多模态视频+音频融合 SOTA | ✅ 已克隆 | Le2i/URFD/Multicam（需单独下载） |
| 02 | [pricai-multimodal-attention](#02-pricai-multimodal-attention-注意力多模态融合) | 注意力多模态融合 | ❌ 未公开 | Le2i/UP-Fall（需单独下载） |
| 03 | [ds-evidence-theory-fusion](#03-ds-evidence-theory-fusion-d-s证据理论融合) | D-S 证据理论决策融合 | ❌ 未公开 | 自采集（未公开） |
| 04 | [egofalls](#04-egofalls-第一人称多模态数据集) | 第一人称多模态数据集 | ❌ 未公开 | DataverseNL 托管（~200GB） |
| 05 | [v-jepa2](#05-v-jepa2-meta-视频世界模型) | Meta 视频世界模型（backbone 候选） | ✅ 已克隆 | 预训练权重在 HuggingFace |
| 06 | [leworldmodel](#06-leworldmodel-轻量级-jepa-世界模型) | 轻量级 JEPA 世界模型 | ✅ 已克隆 | Push-T 基准（含在代码中） |
| 07 | [thinkjepa](#07-thinkjepa-双时间路径世界模型) | 双时间路径 JEPA + VLM | ❌ 未公开 | — |
| 08 | [sonata](#08-sonata-临床小样本世界模型) | 临床小样本 IMU 世界模型 | ❌ 未公开 | 9 个公开 IMU 数据集 |
| 09 | [bimp-dataset](#09-bimp-dataset-双模态碰撞数据集) | 视频+音频碰撞检测数据集 | ❌ 未公开 | IEEE DataPort（需单独下载） |
| 10 | [fallvision](#10-fallvision-大规模视频基准数据集) | 大规模视频基准数据集 | ❌ 未公开 | Harvard Dataverse（公开） |
| 11 | [up-fall](#11-up-fall-多模态跌倒检测数据集) | 多模态经典数据集（2024 更新） | ❌ 配套代码 | Zenodo（公开，含 3D 骨架） |
| 12 | [perception-models](#12-perception-models-meta-音视频感知模型族) | Meta 音视频感知模型族（PE + PLM） | ✅ 已克隆 | HuggingFace 预训练权重（公开） |
| 13 | [videoprism](#13-videoprism-google-视频基础编码器) | Google 视频基础编码器 | ✅ 已克隆 | HuggingFace 预训练权重（公开） |
| 14 | [gemma-4](#14-gemma-4-google-开源多模态vlm推理层) | Google 开源多模态 VLM（推理层） | ✅ 权重公开 | HuggingFace（Apache 2.0） |
| — | [av-jepa](#av-jepa-自研-音视频联合世界模型实验代码) | 🧪 自研 AV-JEPA 实验代码 | ✅ 本仓库自研 | 待定 |

---

## 01 fall-mamba（多模态融合 SOTA）

### 论文信息

- **标题**: Fall-Mamba: A Multimodal Fusion and Masked Mamba-Based Approach for Fall Detection
- **发表**: IEEE Internet of Things Journal, Vol. 12(8), April 2025
- **作者**: Xuebin Zhang, Qicheng Xu, Fuyuan Feng, Xiaochen Lu, Longting Xu 等
- **机构**: DHU (东华大学) Speech Lab

### 核心方法

Fall-Mamba 是目前**视频+音频多模态跌倒检测的最强基线（99.63% 准确率）**，核心创新包括：

1. **Cross-Attention 多模态融合**：通过交叉注意力机制融合视频特征和音频特征（梅尔频谱图），动态学习模态间的互补信息
2. **Bidirectional Mamba 时序建模**：使用双向状态空间模型（SSM/Mamba）替代传统 LSTM/Transformer，实现高效的时序依赖建模
3. **Multi-Head Temporal Attention**：多头时序注意力，强化关键帧的贡献
4. **Frame Masking 策略**：随机遮蔽部分帧，增强模型在遮挡和低光照条件下的鲁棒性

### 代码结构（已克隆）

```
01-fall-mamba/
├── videomamba.py          # 核心模型：双向 Mamba + Cross-Attention 融合
├── kinetics_sparse.py     # Kinetics 数据集加载器（稀疏采样）
├── mamba/                 # Mamba SSM 模块实现
├── causal-conv1d/         # 因果卷积加速模块
├── requirements.txt       # Python 依赖
└── README.md              # 官方说明
```

- **框架**: PyTorch
- **预训练**: 模型在 Kinetics-400/600 上预训练后微调
- **关键依赖**: `causal-conv1d`, `mamba-ssm`, `torch`

### 数据集

论文使用了三个公开数据集的合并：
- **Le2i**: 办公室+居家场景，RGB 视频+音频
- **URFD** (UR Fall Detection): 30 个跌倒序列，RGB+深度+加速度计
- **Multicam**: 多视角跌倒视频

> ⚠️ 需要自行从各数据集官网下载

### 与本项目的关联

- ✅ 多模态融合架构可作为本项目 baseline
- ✅ 注意力融合思路可直接借鉴
- ⚠️ 传统有监督方法，不是世界模型路线
- 💡 建议：将此作为准确率上限参考，同时用世界模型实现无监督版本

---

## 02 pricai-multimodal-attention（注意力多模态融合）

### 论文信息

- **标题**: Video-Audio Multimodal Fall Detection Method
- **发表**: PRICAI 2024 (Pacific Rim International Conference on AI), LNCS Vol. 15284, 2025 年出版
- **作者**: Mahtab Jamali, Paul Davidsson, Reza Khoshkangini 等
- **机构**: Malmö University, Sweden

### 核心方法

1. **视频流 + 音频流分离处理**：各自通过 CNN 提取特征
2. **加性注意力机制**：在视频流和音频流中分别引入 Additive Attention，让模型自动关注跌倒事件中最关键的帧和频谱特征
3. **特征级融合**：将视频和音频的注意力加权特征进行拼接融合后分类
4. **目标**：解决单一模态在恶劣环境（遮挡、噪声、低光照）下的失效问题

### 代码

❌ 未公开代码

### 数据集

- **Le2i**: 办公室+居家场景跌倒视频
- **UP-Fall**: 多模态数据集（17 名受试者，11 类活动）

> 两个数据集均需从官网申请/下载

### 与本项目的关联

- ✅ 注意力机制动态加权多模态的思路值得借鉴
- ✅ 证明了视频+音频互补的有效性（尤其在遮挡和噪声场景下）
- 💡 可将此注意力融合思路嵌入世界模型的编码器

---

## 03 ds-evidence-theory-fusion（D-S 证据理论融合）

### 论文信息

- **标题**: Multimodal Fall Detection for Solitary Individuals Based on Audio-Video Decision Fusion Processing
- **发表**: Heliyon, Vol. 10(8), April 2024（开放获取）
- **作者**: Shiqin Jiao, Guoqi Li, Guiyang Zhang 等
- **机构**: Beihang University（北京航空航天大学）

### 核心方法

1. **视频流**: YOLOv7-Pose 提取 17 个 COCO 骨骼关键点 → Two-Stream ST-GCN（空间+时间图卷积）分类
2. **音频流**: Log-Mel Spectrogram → MobileNetV2 轻量级分类
3. **决策级融合**: 
   - 线性加权融合（Linear Weighting）
   - **Dempster-Shafer (D-S) 证据理论**：处理决策不确定性，在模态冲突时更有优势

### 性能

| 模态 | 灵敏度 (Sensitivity) |
|------|:---------------------:|
| 仅视频 | 81.67% |
| 仅音频 | — |
| 线性加权融合 | 95.00% |
| **D-S 证据理论融合** | **97.50%** |

### 代码

❌ 未公开代码

### 数据集

自采集数据集（未公开），但方法在公开数据集上可复现

### 与本项目的关联

- ✅ 骨骼关键点 + GCN 是高效的视频表示方法
- ✅ D-S 证据理论适合处理多模态决策不确定性
- 💡 可考虑 ST-GCN 作为世界模型的视频编码器前端

---

## 04 egofalls（第一人称多模态数据集）

### 论文信息

- **标题**: EGOFALLS: A Visual-Audio Dataset and Benchmark for Fall Detection Using Egocentric Cameras
- **发表**: ICPR 2024 (International Conference on Pattern Recognition), 2025 年 LNCS 出版
- **作者**: Xueyi Wang 等
- **机构**: University of Groningen, Netherlands

### 数据集特点

| 属性 | 详情 |
|------|------|
| **样本量** | 10,948 个视频片段 |
| **受试者** | 14 人（12 年轻人 + **2 老年人**）⭐ |
| **模态** | 视频 + 音频（同步） |
| **视角** | 第一人称（腰部/颈部安装的 OnReal G1 & CAMMHD Bodycam） |
| **场景** | 14 个室外 + 15 个室内 |
| **跌倒类型** | 20 种（不同方向、不同物体交互） |
| **数据量** | ~200 GB |
| **特色** | **目前唯一包含真实老年人数据的第一人称多模态跌倒数据集** |

### 方法

- 多模态描述符提取（视觉特征 + 音频特征）
- 后期决策融合（Late Decision Fusion）
- 详细的消融实验：单特征 vs 视觉融合 vs 音视频融合

### 代码与数据获取

- ❌ 官方代码未公开
- 📦 数据集托管于 **DataverseNL**: [research.rug.nl](https://research.rug.nl/en/datasets/egofalls-a-visual-audio-dataset-and-benchmark-for-fall-detection-/)（需申请访问）

### 与本项目的关联

- ✅ **唯一包含老年人真实数据的第一人称多模态数据集**，对本项目极其重要
- ✅ 同步视频+音频，适合训练多模态世界模型
- ⚠️ 第一人称视角 vs 固定摄像头视角的差异需注意（本项目可能更偏向固定摄像头）

---

## 05 v-jepa2（Meta 视频世界模型）⭐

### 论文信息

- **标题**: V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
- **发表**: arXiv 2506.09985, June 2025
- **作者**: Mahmoud Assran, Adrien Bardes, Yann LeCun, Michael Rabbat, Nicolas Ballas 等
- **机构**: Meta FAIR

### 核心方法

V-JEPA 2 是 Meta 发布的大规模视频世界模型，基于 JEPA（联合嵌入预测架构）：

1. **自监督学习**：在 100 万+ 小时互联网视频（2200 万片段）上训练，**无需任何标注**
2. **潜在空间预测**：不预测像素，而是预测视频缺失部分的抽象语义表示（Embedding）
3. **3D-RoPE 位置编码**：三维旋转位置嵌入，同时建模空间和时间维度
4. **多尺寸模型**：ViT-L (300M) 到 ViT-g (1B) 参数

### 关键 Benchmark 表现

| 任务 | 数据集 | 指标 | 成绩 | 对比 |
|------|--------|------|:----:|:----:|
| 动作预测 | Epic-Kitchens-100 | Recall@5 | **39.7** | 此前最佳 27.6 |
| 运动理解 | Something-Something v2 | Top-1 | **77.3** | — |
| 时序推理 | TempCompass | Acc | **76.9** | — |
| 物理推理 | PerceptionTest | Acc | **84.0** | — |

### 代码结构（已克隆）

```
05-v-jepa2/
├── src/                    # 核心源码（模型、训练、评估）
│   ├── models/             # JEPA 模型定义
│   ├── trainer.py          # 训练循环
│   └── ...
├── configs/                # 训练/评估配置文件（YAML）
│   ├── train_2_1/          # 预训练配置
│   └── eval_2_1/           # 评估配置
├── evals/                  # 下游任务评估脚本
├── notebooks/              # Jupyter 示例
├── app/                    # Gradio 演示应用
├── setup.py                # 安装脚本
└── hubconf.py              # PyTorch Hub 加载入口
```

**快速使用**:
```python
import torch
processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
model = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')
```

### 模型权重

预训练权重托管在 HuggingFace:
- `facebook/vjepa2-vitl-fpc32-256` (ViT-L, 3 亿参数)
- `facebook/vjepa2-vitg-fpc64-256` (ViT-g, 10 亿参数)
- V-JEPA 2-AC 系列（机器人操作后训练版本）

### 与本项目的关联 ⭐⭐⭐

这是本项目**最推荐的世界模型 backbone 候选**，原因如下：

1. **无监督异常检测天然契合**：在正常日常活动视频上微调后，跌倒会作为"不可预测事件"被检测
2. **动作预测 SOTA**：能准确预测下一帧动作，跌倒 = 预测误差激增
3. **不需要标注跌倒数据**：仅需正常活动视频即可训练
4. **隐私友好**：在潜在空间操作，不需要存储/传输原始视频
5. **已在 HuggingFace Transformers 集成**：部署方便

> 💡 建议技术路线：加载 V-JEPA 2 预训练权重 → 在正常老年人日常活动视频上微调 → 用预测误差（Surprise Score）做跌倒异常检测

---

## 06 leworldmodel（轻量级 JEPA 世界模型）⭐

### 论文信息

- **标题**: LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels
- **发表**: 2025/2026（Yann LeCun 团队）
- **作者**: Lucas Maes, Quentin Le Lidec, Damien Scieur, Randall Balestriero 等
- **机构**: Meta FAIR / AMI Labs

### 核心方法

LeWorldModel 解决了 JEPA 训练中的一个核心难题——**表示坍塌（Representation Collapse）**：

1. **极简 JEPA**：仅 15M 参数，单 GPU（如 NVIDIA L40S）数小时即可完成训练
2. **SIGReg 正则化**：引入各向同性高斯分布潜变量正则化，强制潜在空间保持语义分集，彻底解决表示坍塌
3. **端到端训练**：无需预训练编码器，不需要像素重构
4. **违反预期检验（Violation-of-Expectation Test）**：模型对物理不合理事件（如物体瞬移）会自动产生高"惊奇度"

### 性能

| 指标 | LeWorldModel | DINO-WM（对比） |
|------|:-----------:|:--------------:|
| Push-T 成功率 | **96%** | — |
| 推理速度 | **<1 秒** | ~47 秒（快 48 倍） |
| 物理异常检测 | ✅ 可靠 | — |

### 代码结构（已克隆）

```
06-leworldmodel/
├── jepa.py                 # JEPA 核心模块（编码器/预测器/目标编码器）
├── module.py               # SIGReg 正则化模块
├── train.py                # 训练脚本
├── eval.py                 # 评估脚本（含 Violation-of-Expectation 测试）
├── utils.py                # 工具函数
├── config/                 # 配置参数
├── assets/                 # 演示素材
├── LICENSE                 # MIT License
└── README.md               # 官方文档
```

- **框架**: PyTorch
- **依赖**: 轻量，无特殊依赖
- **硬件**: 单 GPU 即可训练

### 与本项目的关联 ⭐⭐⭐

LeWorldModel 是**本项目最值得精读和复现的世界模型论文**：

1. **物理异常检测机制**：VoE 测试天然适合跌倒检测——跌倒就是"人体瞬移"类型的物理异常
2. **超轻量**：15M 参数，可部署在边缘设备（家庭监控摄像头配套的小型计算设备）
3. **训练成本低**：单 GPU 数小时，适合学术研究和快速迭代
4. **SIGReg 正则化**：解决了 JEPA 训练不稳定的核心痛点，代码可直接复用

> 💡 建议：将 LeWorldModel 的训练方法作为本项目世界模型训练的起点，在其 VoE 机制上适配跌倒场景

---

## 07 thinkjepa（双时间路径世界模型）

### 论文信息

- **标题**: ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
- **发表**: arXiv 2603.22281, March 2026
- **作者**: Haichao Zhang, Yijiang Li, Shwai He, Tushar Nagarajan, Mingfei Chen, Jianglin Lu, Ang Li, Yun Fu
- **机构**: Northeastern University, UC San Diego, University of Maryland, UT Austin, University of Washington

### 核心方法

ThinkJEPA 提出了**双时间路径（Dual-Temporal Pathway）**架构：

1. **密集 JEPA 分支**（Dense Pathway）：以高帧率采样，捕捉精细动作和交互（如跌倒瞬间的快速加速度变化）
2. **稀疏 VLM "思考者" 分支**（Thinker Pathway）：以更长时间步长均匀采样，使用 Qwen3-VL Thinking 模型提取长程上下文和高层语义（如场景中的环境风险识别）
3. **分层金字塔表示提取**：聚合 VLM 的多个中间层特征，而非仅用最后一层
4. **冻结 VLM**：保持 VLM 权重不变，只训练 JEPA 分支和融合模块

### 优势

- 解决了传统世界模型在长序列预测中的**语义漂移**问题
- 同时捕捉瞬时物理事件和长期行为趋势
- 对跌倒检测意义：不仅能识别"跌倒"这个动作，还能理解跌倒前的"步态不稳""徘徊"等先兆行为

### 代码

❌ 未公开（2026 年 3 月刚发布，代码可能尚未整理完成）

### 与本项目的关联

- 💡 该架构的"双时间路径"思路对本项目非常合适：密集路径捕捉跌倒瞬间物理冲击，稀疏路径分析长程行为先兆
- ⚠️ 当前无代码，可先基于 V-JEPA 2 + Qwen-VL 自行实现类似架构
- 📌 持续关注代码发布

---

## 08 sonata（临床小样本世界模型）

### 论文信息

- **标题**: Sonata: A Hybrid World Model for Inertial Kinematics under Clinical Data Scarcity
- **发表**: arXiv 2604.18058, April 2026
- **作者**: Salil Patel, Dr.
- **机构**: —

### 核心方法

1. **超轻量世界模型**：仅 3.77M 参数，可部署在可穿戴设备上
2. **潜在状态预测优于信号重构**：核心实证结论——预测未来潜在状态比重构原始信号效果好得多
3. **跨队列迁移学习（Cross-cohort Transfer）**：在 14-arm 评估套件中展现强泛化能力
4. **小样本场景**：专为临床数据稀缺的情况设计（几十到几百名患者）

### 训练数据

- 9 个公开 IMU 数据集的协调语料库
- 739 名受试者，190,000 个时间窗口
- 在 14-arm 评估套件上进行跨队列评估

### 代码

❌ 未公开（2026 年 4 月刚发布）

### 与本项目的关联

- ✅ "潜在状态预测 > 原始信号重构"的核心结论对视频世界模型同样适用
- ✅ 小样本学习能力对跌倒检测很重要（真实老年人跌倒数据非常稀缺）
- ✅ 跨队列迁移方法可借鉴来弥合"年轻人模拟跌倒 → 真实老年人跌倒"的 Sim-to-Real 鸿沟
- ⚠️ 该论文面向 IMU 数据，需要将其方法迁移到视频领域

---

## 09 bimp-dataset（双模态碰撞数据集）

### 论文信息

- **标题**: Bi-Modal Multiperspective Percussive (BiMP) Dataset for Visual and Audio Human Fall Detection
- **发表**: IEEE Access, January 2025
- **作者**: —

### 数据集特点

| 属性 | 详情 |
|------|------|
| **样本量** | 1,300 个同步视频+音频样本 |
| **受试者** | 26 人，居住环境 |
| **核心特色** | 强调**碰撞冲击声**的声学特征 |
| **视角** | 多视角 |
| **基线方法** | GoogLeNet, LSTM, CWT, STFT |
| **基线准确率** | 高达 96% |

### 特色

- 重点研究了人体跌倒时**不同地面材质（瓷砖、地毯、地板）**对撞击声的影响
- 通过音频碰撞特征区分"人体倒地"和"书籍掉落"等误报源

### 数据获取

📦 IEEE DataPort（需单独申请/下载）

### 与本项目的关联

- ✅ 视频+音频精确同步，适合多模态融合训练
- ✅ 声学碰撞特征研究对降低误报率很重要
- ⚠️ 数据集规模较小（仅 1,300 样本），需数据增强

---

## 10 fallvision（大规模视频基准数据集）

### 论文信息

- **标题**: FallVision: A Benchmark Video Dataset for Fall Detection
- **发表**: Data in Brief, Vol. 59, Article 111440, April 2025（开放获取）
- **作者**: Rahman N.N., Mahi A.B.S., Mistry D. 等
- **机构**: University of Asia Pacific, Dhaka, Bangladesh

### 数据集特点

| 属性 | 详情 |
|------|------|
| **总视频数** | **11,732**（6,002 跌倒 + 5,730 非跌倒） |
| **受试者** | 58 名志愿者（22-27 岁） |
| **类别** | 3 种跌倒类型（床上、椅子上、站立） |
| **视角** | 前、后、左、右、地板、天花板（6 视角） |
| **格式** | MP4 原始视频 + CSV 关键点文件（17 COCO 关键点/帧） |
| **分辨率** | 720p-1080p, ≥30fps |
| **标注** | YOLOv7-Pose 提取的骨骼关键点，逐帧标注 |

### 数据获取

📦 **Harvard Dataverse**（公开，免费下载）: 
🔗 https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/75QPKK

### 与本项目的关联

- ✅ **即开即用**：公开可下载，规模大（>11K 视频），多视角
- ✅ 骨骼关键点标注可以直接用于世界模型的姿态特征输入
- ⚠️ 视频仅含视觉模态（无音频），如需多模态需搭配其他数据集
- ⚠️ 年轻人模拟数据，需注意 Sim-to-Real 问题

---

## 11 up-fall（多模态经典数据集，2024 更新）

### 论文信息

- **原始论文**: UP-Fall Detection Dataset: A Multimodal Approach, Sensors (MDPI), 2019
- **2024 更新**: Improved 3D Skeletons UP-Fall Dataset, July 2024
- **机构**: Universidad Panamericana, Mexico City

### 原始数据集 (2019)

| 属性 | 详情 |
|------|------|
| **受试者** | 17 名健康年轻成人 |
| **活动类别** | 11 种（6 ADL + 5 跌倒类型） |
| **模态** | 5 个 IMU + 1 个 EEG 头带 + 6 个红外传感器 + 2 个 RGB 摄像头 |
| **数据量** | ~850 GB（含原始数据和特征集） |

### 2024 更新版 ⭐

| 改进 | 说明 |
|------|------|
| **3D 骨架** | 33 个 MediaPipe BlazePose 关键点 (x, y, z + visibility) |
| **标签验证** | 人工验证修正标签错误 |
| **主体分离** | 从多人帧中分离主要受试者 |
| **影响检测** | 2g 阈值区分真正跌倒 vs 滑动/近跌倒 |
| **性能** | SVM 99.5%, LSTM 98.5% |

### 数据获取

- 📦 **Zenodo** (2024 更新版): https://zenodo.org/records/12773013
- 📦 **GitHub** (含训练代码): https://github.com/Tresor-Koffi/3D_skeletons-UP-Fall-Dataset
- 📦 **原始数据**: http://sites.google.com/up.edu.mx/har-up/

### 与本项目的关联

- ✅ 最丰富的多模态基准（RGB+深度+IMU+红外），适合跨模态验证
- ✅ 2024 更新版的 3D 骨架质量很高，可直接用于姿态世界模型的输入
- ⚠️ 年轻人模拟数据（在床垫上跌倒），不是真实老年人

---

## 12 perception-models（Meta 音视频感知模型族）⭐

### 论文信息

- **标题**:
  - Perception Encoder: The Best Visual Embeddings Are Not at the Output of the Network (arXiv 2504.13181, 2025)
  - PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding (arXiv 2504.13180, 2025)
  - Pushing the Frontier of Audiovisual Perception with Large-Scale Multimodal Correspondence Learning (Meta Blog, Dec 2025)
- **作者**: Daniel Bolya, Po-Yao Huang, Jang Hyun Cho, Andrea Madotto, Piotr Dollár, Christoph Feichtenhofer 等
- **机构**: Meta FAIR
- **License**: Apache 2.0（PE） / FAIR Research License（PLM）

### 核心方法

这是 Meta 2025 年发布的一套 SOTA 感知模型，包含两个互补组件：

**1. Perception Encoder (PE) — 视觉/音频编码器**

| 类型 | 能力 | 超越/对标 |
|------|------|-----------|
| **PE Core** | 图像+视频+文本 CLIP 式对齐（T→G，300M→1B 参数） | 超越 SigLIP2（图像）、InternVideo2（视频） |
| **PE Lang** | LLM 对齐的编码器，直接接入 VLM | 驱动 PLM 超越 QwenVL2.5、InternVL3 |
| **PE Spatial** | 密集预测（分割/检测/跟踪） | 超越 DINOv2 |
| **PE-AV（⭐重点）** | **音视频+文本联合嵌入**，将音频、视频、音视频、文本映射到同一个共享嵌入空间 | 首个原生多模态联合嵌入 |
| **PE-A-Frame** | 音频事件时序定位，输出 (start, end) 时间戳 | — |

**2. Perception Language Model (PLM) — 多模态 VLM**

- 1B / 3B / 8B 三种规格，基于 Llama-3.x-Instruct
- **核心能力**：
  - **细粒度视频理解**：MVBench 77.1, PerceptionTest（物理推理）82.7
  - **时序定位（RTLoc）**：精确定位视频中某个事件的时间区间
  - **密集时序描述（RDCap）**：对视频中人体活动进行逐帧级描述
  - **区域视频描述（RCap）**：针对视频中特定区域/主体进行描述

**配套数据集**：
- **PVD**：100 万高质量运动中心视频（含第一人称+第三人称视角）
- **PLM-Video-Human**：人类标注的细粒度视频问答 + 时序定位 + 密集描述数据集

### 代码结构（已克隆）

```
12-perception-models/
├── core/
│   ├── vision_encoder/          # PE Core/Lang/Spatial 视觉编码器
│   │   ├── pe.py                # CLIP 模型定义（from_config 加载）
│   │   ├── config.py            # 模型配置（PE-Core-G14-448 等）
│   │   ├── transforms.py        # 图像/视频预处理
│   │   └── rope.py              # 2D RoPE 位置编码
│   ├── audio_visual_encoder/    # PE-AV / PE-A-Frame 音视频编码器
│   │   ├── pe.py                # PEAudioVisual / PEAudioFrame 类
│   │   ├── audio_codec.py       # DAC-VAE 音频编解码
│   │   ├── aligner.py           # 模态对齐模块（AlignModalities）
│   │   ├── transformer.py       # Transformer backbone
│   │   └── transforms.py        # 多模态预处理
│   ├── vision_projector/        # VLM 视觉投影器
│   └── data/                    # 数据加载/混合/预处理
├── apps/
│   ├── pe/                      # PE 评估/基准测试
│   ├── plm/                     # PLM 训练/推理/评估
│   │   ├── generate.py          # 推理生成脚本
│   │   ├── train.py             # 训练脚本
│   │   └── notebook_demos/      # Colab 示例
│   └── detection/               # DETA/Detectron2 检测集成
├── setup.py                     # pip install -e .
└── README.md                    # 官方文档
```

**快速使用**:

```python
# PE-AV: 音视频联合嵌入
from core.audio_visual_encoder import PEAudioVisual, PEAudioVisualTransform
model = PEAudioVisual.from_config("pe-av-large", pretrained=True)
transform = PEAudioVisualTransform.from_config("pe-av-large")

video_files = ["example.mp4"]
descriptions = ["a person falling down"]
outputs = transform(videos=video_files, audio=video_files, text=descriptions)
# outputs.audio_visual_embeds  # 联合嵌入
# outputs.visual_embeds        # 单独视觉嵌入
# outputs.audio_embeds         # 单独音频嵌入

# PE Core: 视觉编码器
from core.vision_encoder import pe
model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
```

**可用预训练权重**（HuggingFace）:

| 系列 | 规格 | HuggingFace ID |
|------|------|----------------|
| PE Core | T/16, S/16, B/16, L/14, G/14 | `facebook/PE-Core-{size}-{px}` |
| PE Lang | L/14, G/14（含 Tiling 版） | `facebook/PE-Lang-{size}-{px}` |
| PE Spatial | T/16→G/14（蒸馏版） | `facebook/PE-Spatial-{size}-{px}` |
| PE-AV | S, B, L（16帧/全帧） | `facebook/pe-av-{size}` |
| PE-A-Frame | S, B, L | `facebook/pe-a-frame-{size}` |
| PLM | 1B, 3B, 8B | `facebook/Perception-LM-{size}B` |

### 与本项目的关联 ⭐⭐⭐

**1. PE-AV 替代手工多模态融合（直接影响 01/02/03 号条目）**

PE-AV 是首个**原生训练的音视频联合嵌入模型**——在预训练阶段就学习了音频、视频、音视频、文本四种模态的共享嵌入。这意味着：
- 不再需要 Fall-Mamba 的 Cross-Attention 手工融合
- 不再需要 D-S Evidence Theory 的决策级融合
- 跌倒检测可以直接使用 `audio_visual_embeds` 作为特征，在下游接一个简单的分类头

**2. PLM 作为世界模型的推理组件（影响 07 ThinkJEPA）**

PLM 可以直接替代 ThinkJEPA 中的 Qwen3-VL "思考者"分支：
- **RTLoc（时序定位）**：在长监控视频中精确定位跌倒事件的时间区间
- **FGQA（细粒度问答）**："视频中的人是否倒在了地上？"
- **RDCap（密集时序描述）**：逐帧描述人体姿态变化，提供可解释的跌倒检测输出
- PLM 的 PerceptionTest 82.7 说明其物理推理能力强，能理解人体倒地是否符合物理规律

**3. PE Core 可作为世界模型的视觉 backbone（影响 05 V-JEPA 2）**

- PE Core 在 Kinetics-400 零样本分类上超越 InternVideo2（76.9 vs —），可作为视频编码 backbone
- 与 V-JEPA 2 形成互补：PE 做语义特征提取 + JEPA 做预测异常检测
- 轻量化模型（T/16 仅 ~10M 参数）可直接部署在边缘设备

**4. PVD 数据集可用作大规模预训练源**

- 100 万运动中心视频，覆盖第一/第三人称
- 如果你需要在大规模视频上预训练世界模型，PVD 是高质量选择
- 包含人类活动标注，可筛选出包含"人体运动"相关的子集

**5. PE-A-Frame 音频事件定位 — 跌倒撞击声检测**

- 自动在音频流中定位异常撞击声的时间戳
- 可与其他传感器（如震动传感器）联合判断
- (start, end) 输出格式天然适合跌倒事件的时域标注

**局限与注意事项**：

- PE/PLM 是**对比学习/有监督**范式，不是 JEPA 式的预测世界模型
- 不能直接做"预测误差 → 异常检测"这条你核心关注的无监督路线
- PLM 的 FAIR Research License 对商业使用有限制，需注意
- PE-AV 模型较大（L 规格约 300M+ 参数），边缘部署需考虑 PE-AV-Small 或 PE-Core-T/16
- 预训练数据不特意包含跌倒场景，需要在跌倒数据上微调

> 💡 **核心定位**：Perception Models 是本项目**第二重要的工具库**（仅次于 V-JEPA 2/LeWorldModel）。它的角色不是取代世界模型的预测架构，而是提供世界模型所需的**高质量多模态特征提取前端**和**高层语义推理后端**。推荐架构：PE-AV（音视频编码）→ JEPA/LeWorldModel（预测异常检测）→ PLM（事件定位 + 可解释输出）。

---

## 13 videoprism（Google 视频基础编码器）⭐

### 论文信息

- **标题**: VideoPrism: A Foundational Visual Encoder for Video Understanding
- **发表**: ICML 2024（arXiv v3 更新于 2025.06）
- **作者**: Long Zhao, Nitesh B. Gundavarapu, Liangzhe Yuan 等
- **机构**: Google DeepMind

### 核心方法

VideoPrism 是 Google DeepMind 的视频基础编码器，**冻结状态下在 31/33 个公开视频理解基准上达到 SOTA**：

1. **ViViT 因子化编码器**：空间 ViT（CoCa 初始化）+ 4 层时序注意力 → 同时捕捉空间外观和时序动态
2. **两阶段预训练**：3600 万高质量视频-文本对 → 5.82 亿带噪声并行文本（ASR/元数据/VLM 描述）的视频片段
3. **双塔对比学习**：视频编码器 + 文本编码器，支持视频-文本跨模态检索

### 模型规格

| 规格 | 参数量 | 输入 | HuggingFace ID |
|------|:---:|------|----------------|
| Base | 114M | 16帧×288 | `google/videoprism-base-f16r288` |
| Large | 354M | 16帧×288 | `google/videoprism-large` |
| LVT Base | 248M | 视频+文本双塔 | `google/videoprism-lvt-base` |
| LVT Large | 580M | 视频+文本双塔 | `google/videoprism-lvt-large` |

### 覆盖任务

| 类别 | 具体任务 |
|------|----------|
| 视频分类 | Kinetics-400/600/700, SSv2, Charades, AVA |
| 时序定位 | ActivityNet, Charades-STA |
| 视频-文本检索 | MSRVTT, VATEX, DiDeMo, LSMDC |
| 零样本分类 | 开放集动作识别 |

### 代码结构（已克隆）

```
13-videoprism/（需通过代理克隆）
├── videoprism/              # 核心模型（JAX/Flax）
├── checkpoints/             # 预训练权重加载
├── configs/                 # 实验配置文件
└── README.md
```

- **框架**: JAX/Flax（HuggingFace 提供 PyTorch 互操作）
- **依赖**: `tensorflow-text`, `scenic`（Google 内部框架）

### 与本项目的关联 ⭐⭐⭐

**1. Google 系的视频 backbone（直接对标 PE Core）**

VideoPrism 是 Google 生态中**最成熟的开放视频编码器**，与 Meta 的 PE Core 形成互补：
- PE Core 侧重 CLIP 式全局语义对齐
- VideoPrism 侧重细粒度时序理解（ViViT 的时序注意力 + 空间-时间解耦）
- **两者可做集成对比实验**：PE Core 做全局场景理解 + VideoPrism 做细粒度动作建模

**2. 填补项目 Google 生态空白**

目前项目 backbone 全部是 Meta 系（V-JEPA 2 / PE），引入 VideoPrism 提供跨机构的方法多样性，增强论文说服力。

**3. 时序定位能力对跌倒检测很有价值**

Charades-STA 时序动作定位能力可用于"在长监控视频中定位跌倒发生的精确时刻"——这直接对应对 PLM RTLoc 的验证。

**4. 零样本分类可用于快速原型**

冻结 VideoPrism → 零样本测试 "falling down" vs "walking" → 不改代码直接出 baseline。

### 局限

- JAX/Flax 框架，PyTorch 加载需要 HuggingFace 互操作层，有一定转换成本
- **纯视频编码器，无音频**：如需多模态需单独接入音频编码器
- 不是世界模型，不做预测
- 两阶段预训练数据不特意包含跌倒场景

> 💡 **定位**: VideoPrism = Google 版的 PE Core。适合作为第二视频 backbone 验证，或替换 PE Core 做对比实验。不替代 JEPA/世界模型路线。

---

## 14 gemma-4（Google 开源多模态 VLM — 推理层）⭐

### 论文信息

- **标题**: Gemma 4（技术报告未单独发布，模型卡见 Google AI for Developers）
- **发布**: 2026.04.02
- **机构**: Google DeepMind
- **License**: Apache 2.0（完全商用自由）

### 核心方法

Gemma 4 是 Google 最强开源多模态模型族，基于 Gemini 3 研究，四个规格覆盖从手机到服务器：

| 规格 | 有效参数 | 总参数 | 模态 | 部署 |
|------|:---:|:---:|------|------|
| **E2B** ⭐ | 2.3B | 5.1B | 文本+图像+**音频** | 手机/Raspberry Pi |
| **E4B** | 4.5B | 8B | 文本+图像+**音频** | 笔记本/Jetson |
| 26B A4B | 3.8B 活跃 | 25.2B | 文本+图像+视频 | 单 GPU |
| 31B | 30.7B | 30.7B | 文本+图像+视频 | H100/A100 |

**关键架构创新**：
- **MatFormer**：弹性推理，动态调整计算量
- **Per-Layer Embeddings (PLE)**：有效参数仅 2.3B，总容量 5.1B
- **交替注意力**：局部滑动窗口 + 全局全上下文交替
- **共享 KV Cache**：尾部层复用前层 KV，减少 74% 显存
- **音频编码器**：USM Conformer (~300M)，支持多语种语音识别
- **视频能力**：最高 60s @ ~1fps，但视觉编码器约 150M 参数

### 关键 Benchmark

| Benchmark | 31B | E4B | E2B |
|-----------|:---:|:---:|:---:|
| MMMU Pro（视觉推理） | **76.9%** | 52.6% | 44.2% |
| MATH-Vision | **85.6%** | 59.5% | 52.4% |
| AIME 2026（数学推理） | **89.2%** | 42.5% | 37.5% |
| LMArena ELO | **~1452** (#3 开源) | — | — |

**推理输出示例（Thinking 模式）**：

```
输入: 10秒监控视频片段
Gemma 4:
  "观察到：0-3s 人物站立并缓慢行走，步态稳定；
   3-4s 人物左腿突然失去支撑，身体重心快速右移；
   4-5s 人物完全接触地面，下肢呈不自然弯曲；
   5-10s 人物无自主起身动作。
   结论：高度疑似跌倒事件（confidence: 0.94）"
```

### 代码获取

```bash
# HuggingFace 直接加载
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("google/gemma-4-e2b-it")
# 或通过 Ollama / vLLM / llama.cpp
```

### 与本项目的关联 ⭐⭐⭐

**1. VLM 推理层 — 替代 PLM 的"思考者"角色**

在 ThinkJEPA 的双时间路径架构中，Gemma 4 可完美替代 Qwen3-VL 作为稀疏 VLM 分支：
- 跌倒事件确认（是/否判断 + 置信度）
- 生成自然语言解释（"该人物因绊倒导致前向跌倒，摔倒后 5 秒未起身"）
- Apache 2.0 比 PLM 的 FAIR Research License 更自由

**2. 边缘部署 — 隐私保护的最后一公里**

E2B 版本 2-bit 量化 < 1.5GB，可在树莓派上运行：
```
监控摄像头 → [AV-JEPA 异常检测] → 异常帧 → [Gemma 4 E2B 本地确认] → 告警
                                                      ↑
                                              视频数据不出设备
```
这是跌倒检测产品化的关键优势——老年人隐私保护是第一优先级。

**3. 原生音频输入**

E2B/E4B 是 Google 系**唯一原生支持音频**的开源 VLM，可直接处理撞击声、呼救声：
- 视频帧 + 倒地撞击声 → 联合判断 → 减少误报

**4. Function Calling 结构化输出**

```python
# Gemma 4 直接输出 JSON
{"fall_detected": true, "type": "forward_fall",
 "confidence": 0.94, "timestamp_start": 3.2, "timestamp_end": 4.8}
```

### 致命局限（对于做主模型）

| 致命问题 | 详细说明 |
|----------|----------|
| **帧率仅 ~1fps** | 跌倒事件 0.5-1.5s 内完成 → 整个过程仅有 0-2 帧可分析 → 极易漏检 |
| **不是流式模型** | Prompt-Response 模式，无法连续监控 → 需滑动窗口分段推理，效率极低 |
| **没有预测能力** | 不理解物理演化，无法做"预测误差异常检测"——恰好是你项目的核心方向 |
| **视觉编码器偏弱** | 150M 参数，远弱于 VideoPrism(354M)/PE Core(1B)/V-JEPA 2(1B) |
| **推理延迟** | Even E2B 在树莓派上 decode 仅 7.6 tok/s，完整视频分析需数秒 |

### 局限

> 💡 **核心定位**: Gemma 4 不是世界模型，不能做主 backbone。它的最佳角色是**二级推理确认层**——当 AV-JEPA 检测到异常时，Gemma 4 以低帧率确认是否真跌倒，并生成可解释报告。配合 Apache 2.0 许可，是产品化的理想选择。

---

## av-jepa（🧪 自研 — 音视频联合世界模型实验代码）⭐

这是本项目**自研的核心实验代码**，是上述所有论文调研的落脚点。目标：**构建世界上首个 Audio-Visual JEPA（AV-JEPA）跌倒检测系统**。

### 架构设计

```
┌────────────────────────────────────────────────────────┐
│                    AV-JEPA Pipeline                      │
│                                                          │
│  视频帧(t) ──→ [CLIP ViT] ──┐                           │
│                              ├→ [Fusion] → [JEPA Pred] → Predicted AV Emb(t+Δt)
│  音频段(t) ──→ [CLAP]   ──┘                    ↓        │
│                                          Surprise Score   │
│                                          (L2 distance)    │
│                                               ↓           │
│                                          > threshold?     │
│                                          → 跌倒异常检测    │
└────────────────────────────────────────────────────────┘
```

**设计思想：**
1. **编码器复用 HuggingFace 预训练模型**（CLIP + CLAP），零成本获取高质量多模态特征
2. **JEPA 预测器从零训练**（3 层 Transformer，仅 ~2M 参数），遵循 LeWorldModel 的 SIGReg 正则化防坍塌
3. **仅需正常活动数据训练**——跌倒样本只在推理阶段用于验证
4. **后续可无缝替换为 PE-AV 原生编码器**（当有 GPU 机器时）

### 代码结构

```
av-jepa/
├── config.py              # 训练/模型配置 dataclass
├── encoders.py            # CLIP(ViT) 视频编码 + CLAP 音频编码 封装
├── fusion.py              # 多模态融合（拼接 + 投影 / 交叉注意力）
├── jepa_model.py          # JEPA 预测器 + SIGReg 正则化
├── data_utils.py          # 视频帧提取 + 音频加载
├── train.py               # 训练循环（JEPA + SIGReg loss）
├── detect.py              # 跌倒检测（预测误差评估）
├── test_with_synthetic.py # 合成数据冒烟测试
└── README.md              # 使用说明
```

### 当前状态

- [x] 代码框架搭建完成
- [x] 合成数据冒烟测试通过
- [ ] 真实跌倒数据集训练（待下载 FallVision/UP-Fall）
- [ ] 替换为 PE-AV 编码器（待 GPU 机到位）
- [ ] 多卡分布式训练

### 与本项目的关联

这是本项目所有论文调研的**代码兑现**：
- 世界模型架构来自 **LeWorldModel (06)**
- 多模态融合思路来自 **Fall-Mamba (01)** 和 **PE-AV (12)**
- JEPA 预测理念来自 **V-JEPA 2 (05)**
- 异常检测机制来自 **LeWorldModel 的 VoE 测试 (06)**
- 后续 VLM 推理层可接入 **Gemma 4 (14)** 或 **PLM (12)**

> 💡 **核心创新**: AV-JEPA 填补了"音频-视频联合世界模型"的研究空白。在 2026 年初，没有任何已知工作将 JEPA 架构同时应用于视频和音频模态。这很可能成为**该方向的首篇论文**。

---

## 总推荐优先级

### 如果要快速出成果（3-6 个月）

```
1. 数据集: FallVision（立刻可下载） + UP-Fall 2024（立刻可下载）
2. Baseline: Fall-Mamba（有代码，SOTA）
3. 世界模型: LeWorldModel（15M，单GPU，有代码）
4. 方案: LeWorldModel 在正常活动视频上训练 → 跌倒时 VoE 异常检测
```

### 如果要做多模态识别（3-6 个月，偏工程/实用）

```
1. 特征提取: PE-AV（音视频联合嵌入，12号条目）
2. 推理确认: Gemma 4 E2B（边缘部署，Apache 2.0，14号条目）
3. Baseline: Fall-Mamba（01号，99.63% 准确率作为上限参考）
4. 数据集: FallVision + UP-Fall 2024 + BiMP
5. 方案: PE-AV 提取联合嵌入 → 下游二分类器 → Gemma 4 做边端推理确认
```

### 如果要深耕世界模型方向（6-12 个月）

```
1. Backbone: V-JEPA 2（1B参数，需要中大算力）或 VideoPrism（354M，Google 系，13号条目）
2. 方法论: LeWorldModel + SIGReg 训练技巧
3. 架构参考: ThinkJEPA 的双时间路径（需自行实现）
4. 特征前端: PE-AV 音视频联合编码（12号条目）
5. 推理后端: Gemma 4 E2B 边端推理 + 可解释输出（14号条目，Apache 2.0）
6. 多模态: PE-AV 原生联合嵌入 + Fall-Mamba Cross-Attention 作为对比
7. 数据集: 组合 FallVision + UP-Fall + PVD（预训练）+ 自采老年人数据
8. 核心实验: AV-JEPA 自研代码（本仓库 av-jepa/）
9. 创新点: 世界模型 × 多模态 × 跌倒检测（完全空白的研究方向）
```

---

## 参考文献

1. Zhang et al., "Fall-Mamba: A Multimodal Fusion and Masked Mamba-Based Approach for Fall Detection," IEEE IoT-J, 2025.
2. Jamali et al., "Video-Audio Multimodal Fall Detection Method," PRICAI 2024, LNCS 2025.
3. Jiao et al., "Multimodal Fall Detection for Solitary Individuals Based on Audio-Video Decision Fusion Processing," Heliyon, 2024.
4. Wang et al., "EGOFALLS: A Visual-Audio Dataset and Benchmark for Fall Detection Using Egocentric Cameras," ICPR 2024.
5. Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning," arXiv 2506.09985, 2025.
6. Maes et al., "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels," 2025/2026.
7. Zhang et al., "ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model," arXiv 2603.22281, 2026.
8. Patel, "Sonata: A Hybrid World Model for Inertial Kinematics under Clinical Data Scarcity," arXiv 2604.18058, 2026.
9. "BiMP: Bi-Modal Multiperspective Percussive Dataset for Visual and Audio Human Fall Detection," IEEE Access, 2025.
10. Rahman et al., "FallVision: A Benchmark Video Dataset for Fall Detection," Data in Brief, 2025.
11. Martínez-Villaseñor et al., "UP-Fall Detection Dataset: A Multimodal Approach," Sensors, 2019. (Updated 2024)
12. Bolya et al., "Perception Encoder: The Best Visual Embeddings Are Not at the Output of the Network," arXiv 2504.13181, 2025.
13. Cho et al., "PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding," arXiv 2504.13180, 2025.
14. Zhao et al., "VideoPrism: A Foundational Visual Encoder for Video Understanding," ICML 2024 (updated 2025).
15. Google DeepMind, "Gemma 4: Frontier Multimodal Intelligence on Device," 2026.

---

> 📌 **核心结论**: "多模态世界模型 × 跌倒检测" 在 2026 年初仍是一片**尚未被探索的研究空白**。JEPA 架构在通用视频理解和医疗健康领域已取得爆发式进展，但还没有任何人将其应用于跌倒检测。Meta Perception Models（PE-AV + PLM）的发布为多模态融合提供了基础设施，Google VideoPrism + Gemma 4 进一步补齐了视频编码和边缘推理的生态。AV-JEPA 是本项目的核心实验代码，目标是填补"音频-视频联合世界模型"这一研究空白，力争成为该方向的**首篇论文**。
