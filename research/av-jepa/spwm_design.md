# T-JEPA: Temporal JEPA for Multimodal Fall Detection with Predictive Text Output

> **四模态输入 → 异常门控 → 文本输出**
>
> Video + Audio + Skeleton + Text → M3-JEPA 融合 → VL-JEPA 对齐 → 只在异常时生成文字
>
> **Fall-Mamba 优化**：Cross-Attention 跨模态融合 + Bidirectional Mamba 时序预测 + Frame Masking 鲁棒训练 + DropPathway 单模态容错

---

## 〇、文献角色分配

| 论文 | 角色 | 在 T-JEPA 中的位置 |
|------|------|-------------------|
| **V-JEPA 2** | 视频编码器 | 冻结的 ViT-L，输出视频 token |
| **Audio-JEPA** | 音频编码器 | 冻结的 ViT，输出音频 token |
| **S-JEPA** | 骨骼编码器 | 冻结的 Transformer，输出骨骼 token |
| **M3-JEPA** | **多模态融合核心** | MoE Predictor：4 路编码 → 门控融合 → 联合 embedding |
| **VL-JEPA** | **嵌入→文本桥梁** | Predictor：联合 embedding → LLM 词汇空间 embedding |
| **TC-JEPA** | **文本条件化** | 用过去事件的文本描述条件化预测，降低不确定性 |
| **NOVA** | 对齐损失 | MSE + SIGReg，稳定训练，无需负样本 |
| **Event-VStream** | 实时策略 | 异常门控：只在 embedding 异常时触发文本生成 |
| **NEP / V1-33K** | 预训练数据 | 前半视频 → 未来描述 |
| **OmniFall** | 微调 + 评估 | 16 类跌倒数据 + 人数统计公平性 |

---

## 一、核心架构：四模态异常门控 T-JEPA

```
╔══════════════════════════════════════════════════════════════════════════╗
║     T-JEPA: 四模态输入 → M3-JEPA 融合 → 异常门控 → 文本输出              ║
╚══════════════════════════════════════════════════════════════════════════╝


  输入 (全部冻结的编码器，来自各 JEPA 论文的预训练权重)
  ═══════════════════════════════════════════════════════════════

  🎬 Video          🔊 Audio          🦴 Skeleton         📝 Text
  (B,16,3,224,224)  (B, 48000)       (B, T, 17, 3)       "老人正在缓慢行走"
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
  V-JEPA 2 ViT-L    Audio-JEPA ViT    S-JEPA Transformer   Qwen2.5 Embedding
  (300M, 冻结)      (~85M, 冻结)      (~22M, 冻结)         (embed layer, 冻结)
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
  v_tokens          a_tokens          s_tokens           t_embed
  (N_v, 1024)       (N_a, 768)        (N_s, 256)          (3584)


  M3-JEPA 多模态融合 (核心可训练模块)
  ═══════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────┐
  │  Multi-Gate MoE Predictor (来自 M3-JEPA, ~8M 参数)           │
  │                                                             │
  │  v_tokens ──→ Gate_v ──→ Expert_v ──┐                      │
  │  a_tokens ──→ Gate_a ──→ Expert_a ──┤                      │
  │  s_tokens ──→ Gate_s ──→ Expert_s ──┼──→ z_fused (1024)   │
  │                                      │                      │
  │  每个 Gate 学习: 多少信息是模态特有的?                      │
  │               多少信息是跨模态共享的?                       │
  │                                                             │
  │  M3-JEPA 的核心: 门控机制解耦模态特有/共享信息               │
  │  → 最大化互信息 → 最小化条件熵 → 最优融合                    │
  └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                       z_fused (1024)
                    融合后的联合 embedding


  V-JEPA 2 Predictor (冻结，预测未来)
  ═══════════════════════════════════════════════════════════════

  z_fused(1024) ──→ V-JEPA 2 Predictor ──→ z_future(1024)
                                            预测的未来联合状态


  ┌─────────────────────────────────────────────────────────────┐
  │                   ⚡ 异常门控 (Event-Driven Gate)              │
  │                                                             │
  │  正常状态 embedding 分布: N(μ, σ²)                           │
  │                                                             │
  │  if ||z_future - μ|| < 2σ:                                  │
  │      → 正常 → 跳过文本生成 → 继续监控 → <1ms               │
  │                                                             │
  │  if ||z_future - μ|| > 2σ:                                  │
  │      → 异常！→ 触发文本推演 → ~5ms                          │
  │                                                             │
  │  关键: 这里的"异常"只是触发条件,                            │
  │       不是最终判断。后面有 VL-JEPA 语义验证                  │
  └─────────────────────────────────────────────────────────────┘
                              │
                              ▼ 异常触发
  ═══════════════════════════════════════════


  VL-JEPA 文本嵌入预测 + TC-JEPA 文本条件化
  ═══════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────┐
  │  Temporal Alignment Projector (可训练, ~4M)                  │
  │                                                             │
  │  输入: z_future(1024)                                       │
  │        + text_condition(3584)  ← 当前时刻的文本描述         │
  │          (TC-JEPA: 文本条件化减少预测不确定性)               │
  │                                                             │
  │  z_future + t_cond ──→ CrossAttn(文本条件化)                │
  │                     ──→ MLP(1024+3584 → 2048 → 3584)       │
  │                     ──→ z_text(3584)                        │
  │                                                             │
  │  z_text: 预测的未来在 LLM 词汇空间中的 embedding              │
  │          "老人失去平衡向前摔倒" ← LLM 可以读懂               │
  └─────────────────────────────────────────────────────────────┘
                              │
                              ▼


  Tier 1: 快速文本检索 (相似度匹配)
  ═══════════════════════════════════════════════════════════════

  z_text(3584) ──→ cos(z_text, pre_embed[i]) for i in 短语库
       │
       ├── "老人向前摔倒"     : 0.89  ← 最高
       ├── "老人缓慢行走"     : 0.12
       ├── "猫快速跑过"       : 0.03
       └── "老人向后倒下"     : 0.78

  if top-1 ∈ FALL_PHRASES:
      → 🚨 确认跌倒 + 输出文本描述


  Tier 2: LLM 详细生成 (仅跌倒触发时调用)
  ═══════════════════════════════════════════════════════════════

  z_text → Qwen2.5 prefix embedding → 自回归生成:

  "预测: 老人将在 0.5 秒内向前摔倒。
   当前姿态: 左脚失稳，身体重心前倾 15°，双臂本能前伸。
   骨架分析: 髋关节 y 坐标正在以 0.3m/s 速度下降。
   音频检测: 未检测到撞击声，摔倒尚未完成。
   风险评估: 前方 1 米有茶几，可能头部撞击。
   建议: 立即查看。"
```

---

## 二、关键设计：异常门控（解决"猫跑过"问题的新答案）

### 2.1 之前的方案为什么不 work

```
AV-JEPA:     embedding 偏离正常分布 → 直接报告跌倒 → 猫、跑动都会触发
SPWM/VQ:     离散码本 → 表达能力有限
PWLM:        LLM 对齐 → 猫不会匹配跌倒短语 → 但每次都跑文本生成,浪费
```

### 2.2 新方案：三级级联

```
          输入 (视频+音频+骨骼+文本)
                    │
    ┌───────────────▼────────────────────┐
    │  Gate 1: 异常 detection (连续运行)  │  ← 门控, ~1ms
    │  ||z_future - μ|| > 2σ ?           │
    └───────────────┬────────────────────┘
                    │
         ┌──────────┴──────────┐
         │ 正常                  │ 异常
         ▼                      ▼
    跳过,继续监控    ┌────────────────────────────┐
                    │  Gate 2: 语义验证 (~4ms)     │  ← 可训练
                    │  z_text → cos() vs 短语库   │
                    │  → 匹配到跌倒短语?           │
                    └────────────┬───────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ 猫/跑动         │ 老人摔倒
                    ▼                 ▼
                "非跌倒事件"    ┌────────────────────┐
                不触发          │  Gate 3: LLM 报告   │  ← 按需, 10-20s
                                │  详细描述 + 建议    │
                                └────────────────────┘
```

### 2.3 猫跑过的完整路径

```
猫跑过:
  Gate 1: z_future 偏离正常? → 是 (画面变化大) → 进入 Gate 2

  Gate 2: VL-JEPA 语义对齐:
    z_text → cos("猫快速跑过") = 0.91  ← 匹配到非跌倒短语
    z_text → cos("老人向前摔倒") = 0.03  ← 远离跌倒短语
    top-1 ∉ FALL_PHRASES → 不触发 ✅

对比纯 AV-JEPA:
  同一场景 → surprise 高 → 直接报警 ❌
```

---

## 三、M3-JEPA 模态融合细节

### 3.1 MoE 门控机制

```python
class M3JEPAFusion(nn.Module):
    """
    M3-JEPA 风格的多模态融合。
    4 路输入编码 → MoE 门控 → 融合联合 embedding。

    关键: 门控函数解耦模态特有 vs 跨模态共享信息
    训练目标: 最大化互信息 I(z_fused; targets) - 最小化条件熵 H(z_fused|inputs)
    """

    def __init__(self):
        # 每个模态一个 Gate + 一组 Expert
        self.gate_v = nn.Linear(1024, 4)    # 视频门控
        self.gate_a = nn.Linear(768, 4)     # 音频门控
        self.gate_s = nn.Linear(256, 4)     # 骨骼门控
        self.gate_t = nn.Linear(3584, 4)    # 文本门控

        # 4 个 Expert (MLP)，每个 Expert 处理一种特征模式
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(2048, 1024), nn.GELU())
            for _ in range(4)
        ])

        # 共享 Expert (跨模态公共信息)
        self.shared_expert = nn.Sequential(
            nn.Linear(2048, 1024), nn.GELU()
        )

        self.fusion_proj = nn.Linear(1024, 1024)  # 最终融合投影

    def forward(self, v, a, s, t):
        """
        v: (B, 1024)  视频 embedding
        a: (B, 768)   音频 embedding
        s: (B, 256)   骨骼 embedding
        t: (B, 3584)  文本 embedding

        Returns: z_fused (B, 1024)
        """
        # 投影到统一维度
        v = self.map_v(v)  # (B, 1024) → keep
        a = self.map_a(a)  # (B, 768)  → (B, 1024)
        s = self.map_s(s)  # (B, 256)  → (B, 1024)
        t = self.map_t(t)  # (B, 3584) → (B, 1024)

        # M3-JEPA MoE 门控
        # 每个模态的门控权重 (B, 5) ← 4 experts + 1 shared
        g_v = F.softmax(self.gate_v(v), dim=-1)
        g_a = F.softmax(self.gate_a(a), dim=-1)
        g_s = F.softmax(self.gate_s(s), dim=-1)
        g_t = F.softmax(self.gate_t(t), dim=-1)

        # 每个模态通过 MoE
        fused = 0
        for i, expert in enumerate(self.experts):
            e_v = expert(v) * g_v[:, i:i+1]
            e_a = expert(a) * g_a[:, i:i+1]
            e_s = expert(s) * g_s[:, i:i+1]
            e_t = expert(t) * g_t[:, i:i+1]
            fused += (e_v + e_a + e_s + e_t)

        # 共享信息 + 最终融合
        shared = self.shared_expert(v + a + s + t)
        return self.fusion_proj(fused + shared)
```

### 3.2 信息论训练目标（来自 M3-JEPA）

```python
# M3-JEPA 训练目标: 能量函数最小化
# E = L_contrastive + λ · L_reg

# 对比损失: 拉近正例 (同一样本的跨模态表示)
# 规则化损失: 防止表示坍塌
# 互信息最大: I(z_fused; target) → max
# 条件熵最小: H(z_fused | inputs) → min

loss = (
    contrastive_loss(z_fused, targets)      # InfoNCE
    + 0.1 * sigreg_loss(z_fused)            # 防坍塌
    + 0.3 * mutual_info_loss(z_fused)       # 互信息最大化
)
```

---

## 四、TC-JEPA 文本条件化

### 4.1 为什么需要文本条件化

```
没有文本条件化:
  z_future → Projector → "一个人在...做什么?"
  预测不确定性高，可能和实际未来描述不匹配

TC-JEPA 条件化:
  z_future + text("老人正在慢慢走向沙发")
  → Projector → "老人会在沙发坐下" 或 "老人会失去平衡摔倒"
  文本提供了上下文约束，预测更准确
```

### 4.2 实现

```python
class TextConditionedProjector(nn.Module):
    """
    TC-JEPA 风格: 用当前时刻的文本描述条件化未来预测。
    """

    def __init__(self, jepa_dim=1024, text_dim=3584, out_dim=3584):
        super().__init__()
        # 跨注意力: z_future 作为 query, text 作为 key/value
        self.cross_attn = nn.MultiheadAttention(
            out_dim, 8, batch_first=True
        )
        self.z_proj = nn.Linear(jepa_dim, out_dim)
        self.text_proj = nn.Linear(text_dim, out_dim)
        self.output = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, z_future, text_embed):
        """
        z_future: (B, 1024)  预测的未来联合状态
        text_embed: (B, 3584) 当前时刻事件的文本 embedding
        Returns: z_text (B, 3584)  在 LLM 空间的未来文本 embedding
        """
        z = self.z_proj(z_future).unsqueeze(1)   # (B, 1, 3584)
        t = self.text_proj(text_embed).unsqueeze(1)

        # 跨注意力: z 看向 t 来获取上下文约束
        conditioned, _ = self.cross_attn(z, t, t)
        return self.output(conditioned.squeeze(1))
```

---

## 五、完整训练流程

```
Stage 1: M3-JEPA 融合预训练 (V1-33K, 自监督)
  ═══════════════════════════════════════
  输入: 视频前半 + 音频 + 骨骼 + 文本描述
  目标: 学习跨模态融合表示
  训练: 对比损失 + SIGReg + 互信息最大
  训练量: 1 天, ~8M 可训练参数


Stage 2: VL-JEPA 文本对齐预训练 (V1-33K)
  ═══════════════════════════════════════
  输入: z_future + text_condition
  目标: z_future → LLM embedding space
  训练: MSE(z_text, e_future_caption)
        + InfoNCE(对比损失)
        + SIGReg(防坍塌)
  训练量: 1 天, ~4M 可训练参数


Stage 3: OmniFall 跌倒专项微调
  ═══════════════════════════════════════
  数据: OF-Staged (含 Le2i) + OF-ITW
  微调: MoE gates (最后几层) + Projector
  新增: Anomaly Gate (正常分布统计)
  训练量: 1 天


Stage 4: 异常门控校准
  ═══════════════════════════════════════
  在 OF-Staged 正常活动上计算 μ, σ
  设置 Gate 1 阈值: threshold = μ + 2σ
  (低阈值: 宁可多触发 Gate 2, 不能漏掉跌倒)
```

---

## 六、推理管线

```python
class TJEPADetector:
    """T-JEPA: 四模态异常门控跌倒检测"""

    def __init__(self, fusion, projector, normal_stats, retriever, llm=None):
        self.fusion = fusion          # M3-JEPA 融合
        self.projector = projector    # VL-JEPA + TC-JEPA 对齐
        self.normal_mean = normal_stats['mean']   # normal μ
        self.normal_std = normal_stats['std']     # normal σ
        self.retriever = retriever    # 短语检索库
        self.llm = llm

    @torch.no_grad()
    def step(self, video, audio, skeleton, text_embed):
        """
        单次推理。四模态输入 → 异常门控 → 可选文本输出。
        """

        # ── M3-JEPA 融合 ──
        z_fused = self.fusion(video, audio, skeleton, text_embed)

        # ── 未来预测 ──
        z_future = vjepa2_predictor(z_fused)

        # ── ⚡ Gate 1: 异常检测 (<1ms) ──
        anomaly_score = ((z_future - self.normal_mean) ** 2).sum() ** 0.5

        if anomaly_score / self.normal_std < 2.0:
            # 正常 → 跳过，节省资源
            return {"is_fall": False, "skip": True, "anomaly": anomaly_score}

        # ── Gate 2: VL-JEPA 语义验证 (~4ms) ──
        z_text = self.projector(z_future, text_embed)
        top_matches = self.retriever.search(z_text, top_k=3)
        top_phrase, top_sim = top_matches[0]

        if not self.retriever.is_fall(top_phrase) or top_sim < 0.5:
            # 不是跌倒 → "猫跑过" / "关灯" / 其他异常
            return {
                "is_fall": False,
                "anomaly": anomaly_score,
                "top_phrase": top_phrase,
                "output": f"异常但非跌倒: {top_phrase} (相似度 {top_sim:.0%})"
            }

        # ── Gate 3: LLM 详细报告 (异步, 10-20s) ──
        if self.llm:
            report = self.llm.generate_from_embedding(z_text, prompt)
        else:
            report = top_phrase

        return {
            "is_fall": True,
            "anomaly": anomaly_score,
            "text_prediction": top_phrase,
            "report": report,
        }
```

---

## 七、各方案最终对比

| | AV-JEPA | 骨骼+音频 | VL-JEPA | T-JEPA (本方案) |
|---|---|---|---|---|
| 视频输入 | ✅ | ❌ | ✅ | ✅ |
| 音频输入 | ✅ | ✅ | ❌ | ✅ |
| 骨骼输入 | ❌ | ✅ | ❌ | ✅ |
| 文本输入 | ❌ | ❌ | ❌ | ✅ |
| 融合方式 | 简单 concat | 后期融合 | N/A | **M3-JEPA MoE** |
| 预测未来 | ✅ embedding | ❌ | ❌ | **✅ embedding→文本** |
| 异常分离 | ❌ surprise | N/A | N/A | **✅ 三级门控** |
| 文本输出 | ❌ | ❌ | ✅ | **✅ 自然语言** |
| 实时性 | ✅ 10ms | ✅ 5ms | ✅ 100ms | **✅ 正常<1ms, 异常~5ms** |
| 猫跑过误报 | ❌ | ✅ | N/A | **✅** |
| 训练数据 | Le2i | 标注数据 | 通用 | V1-33K + OmniFall |
| 论文定位 | 首篇 AV-JEPA | 工程方案 | Meta FAIR | **首篇 Predictive VL-JEPA** |

---

## 八、实施计划

| 步骤 | 任务 | 数据 | 工作量 |
|------|------|------|--------|
| 1 | 提取骨骼特征 (S-JEPA 或 YOLOv8-Pose) | Le2i + OmniFall | 1 天 |
| 2 | 实现 M3-JEPA MoE 融合模块 | — | 2 天 |
| 3 | V1-33K 预训练 (Stage 1+2) | V1-33K (31K) | 2 天 |
| 4 | OmniFall 微调 (Stage 3) | OmniFall (70K+) | 1 天 |
| 5 | 异常门控校准 + 短语库构建 | — | 半天 |
| 6 | 端到端评估 + 跨人群测试 | OmniFall | 1 天 |
| 7 | 非跌倒异常场景测试 | 自制 (猫/跑动/关灯) | 1 天 |
| 8 | 集成 Tier 3 LLM 报告生成 | Qwen2.5 | 1 天 |
| 9 | 论文撰写 | — | 2 周 |

---

## 九、24fps 实时推理实现方案

### 9.1 核心原则：不是每帧都跑完整管线

```
24fps = 每帧 42ms 预算

不可能每帧跑:
  V-JEPA 2 encoder (300M):    ~3ms
  Audio-JEPA encoder (85M):   ~2ms
  S-JEPA encoder (22M):       ~1ms
  M3-JEPA MoE fusion (8M):    ~2ms
  V-JEPA 2 Predictor:         ~0.5ms
  VL-JEPA Projector (4M):     ~0.5ms
  ─────────────────────────
  总计:                       ~9ms
  × 24fps = 216ms/秒 → 需要 5 倍加速才能实时

正确做法: 分层运行，不同频率
  帧级 (24Hz):    帧缓冲 + 骨骼提取
  子采样级 (3Hz):  JEPA 编码 + 预测 + 门控
  触发级 (按需):   LLM 文本生成
```

### 9.2 时间预算分配

```
每 1 秒 = 24 帧，分为 3 个 JEPA 推理窗口 (每 8 帧一次)

时间线 (1 秒):
  Frame 0:  ─┐
  Frame 1:   ├─ 帧缓冲 (轻量预处理, 每帧 <1ms)
  Frame 2:   │
  Frame 3:   │
  Frame 4:   │
  Frame 5:   │
  Frame 6:   │
  Frame 7:  ─┘ → JEPA 推理 #1 (8 帧上下文 + 8 帧目标)
  Frame 8:  ─┐
  Frame 9:   │
  Frame 10:  │
  Frame 11:  │
  Frame 12:  │
  Frame 13:  │
  Frame 14:  │
  Frame 15: ─┘ → JEPA 推理 #2
  Frame 16: ─┐
  ...
  Frame 23: ─┘ → JEPA 推理 #3

每秒 3 次 JEPA 推理 × 9ms = 27ms/秒 → 仅占 2.7% 的 GPU 时间
```

### 9.3 完整管线实现

```python
class RealtimeTJEPA:
    """
    24fps 实时 T-JEPA 跌倒检测管线。

    核心策略:
      - 帧级 (24Hz): 轻量帧缓冲 + YOLOv8-Pose 骨骼提取
      - 子采样级 (3Hz): 每次 8 帧上下文 + 8 帧目标, JEPA 推理
      - 触发级: LLM 详细报告
    """

    def __init__(self, device="cuda:0"):
        self.device = device

        # ━━━ 24fps 层: 轻量实时组件 ━━━
        self.frame_buffer = torch.zeros(16, 3, 224, 224)  # 环形缓冲 16 帧
        self.audio_buffer = deque(maxlen=48000 * 3)        # 3 秒音频
        self.skeleton_buffer = deque(maxlen=16)            # 16 帧骨骼

        # YOLOv8-Pose: 轻量骨骼提取，每帧 ~3ms (GPU) 或 ~10ms (CPU)
        # 可用 MediaPipe 替代 (<5ms CPU)
        self.pose_extractor = YOLOv8Pose("yolov8n-pose.pt")
        self.pose_extractor.to(device).eval()

        # 音频特征提取器: 短时能量 + 频谱变化检测
        self.audio_detector = AudioChangeDetector()

        # 帧间运动检测器
        self.motion_detector = FrameMotionDetector()

        # ━━━ 3Hz 层: JEPA 推理 ━━━
        self.video_encoder = load_vjepa2_encoder().to(device).eval()
        self.audio_encoder = load_audio_jepa_encoder().to(device).eval()
        self.skeleton_encoder = load_sjepa_encoder().to(device).eval()
        self.fusion = M3JEPAFusion().to(device)          # 可训练
        self.predictor = load_vjepa2_predictor().to(device).eval()
        self.projector = TextConditionedProjector().to(device)  # 可训练

        # ━━━ 触发级: LLM ━━━
        self.llm = None  # 延迟加载

        # ━━━ 状态 ━━━
        self.frame_idx = 0
        self.normal_mean = load_calibration("normal_mean.pt")
        self.normal_std = load_calibration("normal_std.pt")
        self.retriever = load_phrase_retriever()

        # 流对象
        self.cap = cv2.VideoCapture(0)
        self.audio_stream = pyaudio.PyAudio().open(...)


    # ══════════════════════════════════════════════════════
    # 24fps 循环: 主循环，每帧调用
    # ══════════════════════════════════════════════════════

    def run_24fps_loop(self):
        """主循环: 模拟 24fps 实时管线"""
        print("T-JEPA 24fps 实时跌倒检测启动...")

        while True:
            t0 = time.perf_counter()

            # ──── Step 1: 帧捕获 (硬件 I/O, ~1ms) ────
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.resize(frame, (224, 224))
            frame_tensor = torch.from_numpy(frame).permute(2,0,1).float() / 255.0

            audio_chunk = self.audio_stream.read(1024)

            # ──── Step 2: 帧缓冲 (环形, <0.1ms) ────
            buf_idx = self.frame_idx % 16
            self.frame_buffer[buf_idx] = frame_tensor
            self.audio_buffer.extend(audio_chunk)

            # ──── Step 3: 骨骼提取 (每帧, ~3ms) ────
            pose_keypoints = self.pose_extractor(frame_tensor)
            self.skeleton_buffer.append(pose_keypoints)

            # ──── Step 4: 帧间变化检测 (<0.5ms) ────
            # 只在新帧出现显著变化时才考虑触发 JEPA
            has_motion = self.motion_detector.update(frame_tensor)
            has_audio_event = self.audio_detector.update(audio_chunk)

            self.frame_idx += 1

            # ──── Step 5: 每 8 帧触发一次 JEPA 推理 ────
            if self.frame_idx % 8 == 0 and self.frame_idx >= 16:
                if has_motion or has_audio_event:
                    # 过去 8 帧 → 未来 8 帧 的世界模型预测
                    jepa_result = self._run_jepa_inference()

                    if jepa_result["is_fall"]:
                        alarm(jepa_result)

            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000

            # 帧率控制: 如果这帧处理太快，补齐到 42ms
            target_frame_time = 1000.0 / 24  # 41.67ms
            if elapsed_ms < target_frame_time:
                time.sleep((target_frame_time - elapsed_ms) / 1000.0)

            if self.frame_idx % 240 == 0:
                actual_ms = (time.perf_counter() - t0) * 1000
                print(f"Frame {self.frame_idx}: {actual_ms:.1f}ms "
                      f"(budget {target_frame_time:.1f}ms)")


    # ══════════════════════════════════════════════════════
    # 3Hz JEPA 推理: 每 8 帧调用一次
    # ══════════════════════════════════════════════════════

    @torch.no_grad()
    def _run_jepa_inference(self):
        """完整的 T-JEPA 三级门控推理 (<10ms)"""

        # ━━━ 数据准备 ━━━
        # 上下文帧: 最近的 8 帧 (buffer idx 8-15)
        ctx_frames = self._get_buffer_frames("context")
        # 目标帧: 未来 8 帧 → 实际应用中来自后续缓冲
        # 在实时场景中，我们预测未来，不需要等待实际目标帧

        ctx_audio = torch.tensor(list(self.audio_buffer)).float()
        ctx_audio = resample_audio(ctx_audio, 48000, 16000)

        ctx_skeleton = torch.stack(list(self.skeleton_buffer)[-8:])

        # 文本条件: 用一个轻量描述符生成当前状态文本
        text_condition = self._get_text_condition(ctx_skeleton)
        text_embed = self.text_encoder.encode(text_condition)

        # ━━━ 编码 ━━━
        v_tokens = self.video_encoder(ctx_frames)          # (~2.5ms)
        a_tokens = self.audio_encoder(ctx_audio)           # (~1.5ms)
        s_tokens = self.skeleton_encoder(ctx_skeleton)     # (~1ms)

        # ━━━ M3-JEPA 融合 + JEPA 预测 ━━━
        z_fused = self.fusion(v_tokens, a_tokens, s_tokens, text_embed)  # (~2ms)
        z_future = self.predictor(z_fused)                                # (~0.5ms)

        # ━━━ ⚡ Gate 1: 异常门控 ━━━
        anomaly_score = torch.norm(z_future - self.normal_mean).item()
        sigma_score = anomaly_score / (self.normal_std + 1e-8)

        if sigma_score < 2.0:
            # 正常 → 不触发后续
            return {
                "is_fall": False,
                "skip": True,
                "sigma": sigma_score,
                "latency_ms": 8.0,
            }

        # ━━━ Gate 2: VL-JEPA 语义验证 ━━━
        z_text = self.projector(z_future, text_embed)              # (~0.5ms)
        top_matches = self.retriever.search(z_text, top_k=3)       # (~0.2ms)
        top_phrase, top_sim = top_matches[0]

        if not self.retriever.is_fall(top_phrase) or top_sim < 0.5:
            # 异常但非跌倒
            return {
                "is_fall": False,
                "anomaly_sigma": sigma_score,
                "top_phrase": top_phrase,
                "similarity": top_sim,
                "output": f"异常: {top_phrase} (不是跌倒，置信度 {top_sim:.0%})",
                "latency_ms": 9.0,
            }

        # ━━━ Gate 3: LLM 详细报告 (后台异步) ━━━
        self._trigger_llm_report(z_text, top_phrase)

        return {
            "is_fall": True,
            "anomaly_sigma": sigma_score,
            "text_prediction": top_phrase,
            "similarity": top_sim,
            "output": f"🚨 检测到跌倒: {top_phrase}",
            "latency_ms": 9.0,
        }


    # ══════════════════════════════════════════════════════
    # 辅助函数
    # ══════════════════════════════════════════════════════

    def _get_buffer_frames(self, mode="context"):
        """从环形缓冲取 8 帧 (B, 8, C, H, W)"""
        if mode == "context":
            # 最近的 8 帧
            indices = [(self.frame_idx - 16 + i) % 16 for i in range(8, 16)]
        else:
            indices = [(self.frame_idx - 8 + i) % 16 for i in range(8)]
        frames = torch.stack([self.frame_buffer[i] for i in indices])
        return frames.unsqueeze(0).to(self.device)

    def _get_text_condition(self, skeleton):
        """从骨骼状态生成文本条件 (TC-JEPA)"""
        heights = skeleton[:, :, 1]  # y 坐标
        avg_height = heights.mean()

        if skeleton.shape[0] < 2:
            return "老人在监控画面中"

        # 简单启发式 (训练后由 TC-JEPA 的 Projector 替代)
        velocity_y = (heights[-1] - heights[-2]).mean()
        if velocity_y > 5:
            return "老人正在向上移动"
        elif velocity_y < -5:
            return "老人身体正在下移"
        elif skeleton[:, :, :2].std() < 5:
            return "老人保持静止"
        else:
            return "老人在缓慢活动"

    def _trigger_llm_report(self, z_text, phrase):
        """后台线程调用 LLM 生成详细报告"""
        if self.llm is None:
            self.llm = load_qwen25_7b()
        prompt = f"""检测到可能的跌倒事件: {phrase}。
请生成详细的老人跌倒报告，包括: 1.动作描述 2.姿态分析 3.风险评估 4.建议"""
        threading.Thread(
            target=lambda: self._generate_and_alert(z_text, prompt)
        ).start()

    def _generate_and_alert(self, z_text, prompt):
        report = self.llm.generate_from_embedding(z_text, prompt)
        print(f"\n{'='*60}")
        print("T-JEPA 跌倒报告:")
        print(report)
        print(f"{'='*60}\n")
```

### 9.4 每帧详细时间预算

```
Frame N (42ms 预算):
┌──────────────────────────────────────────────────────────────┐
│  帧捕获 (cv2.VideoCapture.read)            ~0.5ms            │
│  帧预处理 (resize, normalize)              ~0.3ms            │
│  骨骼提取 (YOLOv8-Pose, GPU)              ~3.0ms            │
│  运动检测 (<0.5ms)                        ~0.2ms            │
│  音频检测 (<0.5ms)                        ~0.2ms            │
│  ───────────────────────────────────────                     │
│  每帧固定开销:                            ~4.2ms  (10%)      │
│                                                              │
│  如果是第 8 帧 + 有运动/音频事件:                             │
│    JEPA 推理 (V-JEPA2 + Audio-JEPA +                         │
│              S-JEPA + M3 fusion +                            │
│              Predictor + Projector + Gate)  ~9ms             │
│    ────────────────────────────────────                       │
│    总开销:                                ~13ms  (31%)       │
│    剩余:                                  29ms  (69%)       │
│                                                              │
│  如果不是第 8 帧 (87.5% 的帧):                               │
│    总开销:                                ~4.2ms (10%)      │
│    剩余:                                 ~37.5ms (90%)      │
└──────────────────────────────────────────────────────────────┘

总结:
  87.5% 的帧: 4.2ms → 远低于 42ms 预算
  12.5% 的帧: 13ms → 也远低于 42ms 预算

  平均每秒开销:
    21 帧 × 4.2ms + 3 帧 × 13ms = 88.2ms + 39ms = 127ms
    → GPU 利用率仅 12.7%，还有大量余量
```

### 9.5 硬件选择建议

| 组件 | 选项 A (边缘) | 选项 B (服务器) |
|------|-------------|----------------|
| GPU | Jetson Orin NX (100 TOPS) | RTX 4090 (82 TFLOPS) |
| 骨骼提取 | MediaPipe CPU (<5ms) | YOLOv8-Pose GPU (<3ms) |
| JEPA 推理 | 量化 INT8 (~15ms) | FP16 (~9ms) |
| 帧率 | 24fps ✅ | 24fps ✅ |
| 功耗 | ~25W | ~450W |
| 成本 | ~$500 | ~$2000 |
| LLM Tier 3 | 不可用 (送到云端) | Qwen2.5-7B 本地 |

### 9.6 延迟 vs 覆盖率权衡

```
JEPA 推理频率    每帧耗时      覆盖率      跌倒检测延迟
────────────────────────────────────────────────────
每帧 (24Hz)     33ms   ❌     100%        1 帧 (42ms)
每 4 帧 (6Hz)   13ms   ✅     100%        4 帧 (167ms)
每 8 帧 (3Hz)   13ms   ✅     100%        8 帧 (333ms) ← 推荐
每 16 帧 (1.5Hz) 13ms  ✅      87.5%      16 帧 (667ms)
每 32 帧 (0.75Hz)13ms  ⚠️     75%        32 帧 (1.3s)

推荐 3Hz (每 8 帧): 延迟 333ms 对于跌倒检测足够
(跌倒通常持续 1-3 秒，333ms 意味着在跌倒过程中就能触发)
```

### 9.7 关键: 多线程流水线

```
线程 1 (主线程, 24Hz):  帧捕获 → 预处理 → buffer 写入
线程 2 (GPU, 3Hz):      buffer 读取 → JEPA 推理 → 门控判断
线程 3 (后台, 按需):    LLM 文本生成 → 报告输出

┌─ Thread 1 ─────────────────────────────────────────────────┐
│ Frame0  Frame1  Frame2  Frame3  Frame4  Frame5  Frame6  Frame7 │
│  ↓prep   ↓prep   ↓prep   ↓prep   ↓prep   ↓prep   ↓prep   ↓prep │
└──────────┬──────────────────────────────────────────────────┘
           │ 每帧写入 buffer
           ▼
┌─ Thread 2 (每 8 帧触发) ───────────────────────────────────┐
│                        [Read 16 frames] → JEPA → Gate → result │
│                        耗时 ~9ms, 在下一帧到来前完成            │
└────────────────────────────────────────────────────────────┘
           │ 如果跌倒
           ▼
┌─ Thread 3 (后台) ──────────────────────────────────────────┐
│                        z_text → Qwen2.5 → 详细报告            │
│                        耗时 10-20s, 不影响主线程帧率           │
└────────────────────────────────────────────────────────────┘
```

---

## 十、Fall-Mamba 论文优化：对我们方案的 5 个具体改进

> 来源: Zhang et al., "Fall-Mamba: A Multimodal Fusion and Masked Mamba-Based Approach for Fall Detection", IEEE IoT Journal, 2025.
> 在 Le2i+URFD+Multicam 上达到 **99.63% 准确率**，已验证的工程级方案。

### 10.1 Fall-Mamba 关键设计

```
Fall-Mamba 架构:

  视频帧 (ResNet 特征提取)          音频 (Mel-Spectrogram)
       │                                  │
  视频特征 tokens                   音频特征 tokens
       │                                  │
       └──────┬───────────────────────────┘
              ▼
       Cross-Attention 跨模态融合
       音频 → Query, 视频 → Key/Value
       (mid-level fusion, 远优于 concat/加权)
              │
              ▼
       Temporal Mamba Block
       Multi-head Temporal Attention
       + Bidirectional Mamba (SSM)
       (线性复杂度 vs Transformer 平方复杂度)
              │
              ▼
       MLP → Binary: Fall / Normal
```

### 10.2 五项优化

#### 优化 1: Cross-Attention 替代简单 Concat 融合

**当前 T-JEPA 的 M3-JEPA 融合**：每个模态通过独立 Gate + Expert + 共享 Expert → 求和融合。

**Fall-Mamba 的做法**：Cross-Attention，其中一种模态做 Query，另一种模态做 Key/Value，让模型主动选择关注对方模态的哪些部分。

**改进方案**：在 M3-JEPA 的 MoE 融合之后，加一层 Cross-Attention 精炼：

```python
class HybridFusion(nn.Module):
    """M3-JEPA MoE + Fall-Mamba Cross-Attention 双层融合"""

    def __init__(self):
        # 第一层: M3-JEPA MoE 粗融合 (保留)
        self.moe_fusion = M3JEPAFusion()

        # 第二层: Fall-Mamba Cross-Attention 精炼 (新增)
        self.v2a_cross_attn = nn.MultiheadAttention(1024, 8, batch_first=True)
        self.a2v_cross_attn = nn.MultiheadAttention(1024, 8, batch_first=True)

        # 第三层: Skeleton ↔ AV Cross-Attention
        self.s2av_cross_attn = nn.MultiheadAttention(1024, 8, batch_first=True)

    def forward(self, v, a, s, t):
        # Step 1: MoE 粗融合
        z_moe = self.moe_fusion(v, a, s, t)  # (B, 1024)

        # Step 2: Cross-Attention 精炼
        # Audio attends to Video (Fall-Mamba: audio=Q, video=KV)
        a_refined, _ = self.a2v_cross_attn(a_q, v_kv, v_kv)

        # Video attends to Audio
        v_refined, _ = self.v2a_cross_attn(v_q, a_kv, a_kv)

        # Skeleton attends to fused AV
        s_refined, _ = self.s2av_cross_attn(
            s_q, torch.cat([v_refined, a_refined], dim=1),
            torch.cat([v_refined, a_refined], dim=1)
        )

        # 最终融合
        return self.final_proj(z_moe + v_refined + a_refined + s_refined)
```

**预期收益**：Fall-Mamba 的消融实验显示 Cross-Attention 中期融合比 Early Fusion 高 11%，比 Late Fusion 高 7%。

#### 优化 2: Bidirectional Mamba 替代 Transformer 时序预测器

**当前 T-JEPA**：TemporalStatePredictor 使用 Transformer（平方复杂度 O(N²)）。

**Fall-Mamba 的做法**：使用 Bidirectional Mamba（SSM），线性复杂度 O(N)，在处理长序列时更快、更省内存。

**改进方案**：用 Bidirectional Mamba 替换 Temporal 部分的 Transformer：

```python
class MambaTemporalPredictor(nn.Module):
    """
    Fall-Mamba 风格的时序预测器。
    Bidirectional Mamba (SSM) 替代 Transformer。
    优势: O(N) 线性复杂度，长序列更友好。
    """

    def __init__(self, d_model=256, d_state=16, n_layers=4):
        super().__init__()
        # Forward Mamba blocks
        self.forward_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(n_layers)
        ])
        # Backward Mamba blocks
        self.backward_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        """
        x: (B, T, d_model)  过去 T 个状态
        Returns: (B, d_model) 下一时刻预测
        """
        # Forward pass
        fwd = x
        for block in self.forward_blocks:
            fwd = block(fwd)

        # Backward pass (flip sequence)
        bwd = torch.flip(x, dims=[1])
        for block in self.backward_blocks:
            bwd = block(bwd)
        bwd = torch.flip(bwd, dims=[1])

        # Bidirectional fusion
        bi = torch.cat([fwd[:, -1, :], bwd[:, -1, :]], dim=-1)
        return self.output_proj(bi)
```

**预期收益**：处理 32+ 帧的长序列时，推理速度比 Transformer 快 3-5 倍，内存占用降低 50%。

#### 优化 3: Frame Masking 鲁棒训练

**Fall-Mamba 的做法**：训练时随机 mask 部分帧（10-30%），模拟真实场景中的帧丢失/遮挡。**准确率从 98.88% 提升到 99.63%。**

**改进方案**：在 T-JEPA 预训练和微调中都加入 Frame Masking：

```python
def frame_masking_augment(frames, mask_ratio=0.2):
    """
    Fall-Mamba 风格的帧级随机 mask。
    mask 掉部分帧 → 模型学会从不完整数据中预测。

    frames: (B, T, C, H, W)
    mask_ratio: 0.1 ~ 0.3
    """
    B, T, C, H, W = frames.shape
    # 随机选择要 mask 的帧
    mask = torch.rand(B, T, 1, 1, 1) > mask_ratio
    # 被 mask 的帧替换为零 (或 learnable mask token)
    return frames * mask.float()

# 训练时:
ctx_frames = frame_masking_augment(ctx_frames, mask_ratio=0.2)
```

**预期收益**：
- 对遮挡、帧丢失更鲁棒
- 对暗光环境更鲁棒（Fall-Mamba 在 Dark 条件下仍保持 98.14%）
- 有助于防止过拟合

#### 优化 4: DropPathway 单模态容错

**Fall-Mamba 的做法**：如果某一模态缺数据（如没有音频），只用另一模态继续训练，模型不崩溃。

**改进方案**：训练时随机丢弃某一模态（概率 10-20%），让模型学会单模态推理：

```python
def drop_modality(v, a, s, t, drop_prob=0.15):
    """随机丢弃一个模态 (训练时正则化)"""
    modality_mask = torch.rand(4) > drop_prob  # 4 模态随机保留
    if not modality_mask[0]: v = torch.zeros_like(v)  # drop video
    if not modality_mask[1]: a = torch.zeros_like(a)  # drop audio
    if not modality_mask[2]: s = torch.zeros_like(s)  # drop skeleton
    if not modality_mask[3]: t = torch.zeros_like(t)  # drop text
    return v, a, s, t, modality_mask
```

**预期收益**：在某一模态缺失时（如夜间无视频、嘈杂环境无音频），系统继续工作而不崩溃。

#### 优化 5: Mel-Spectrogram 作为音频备选

**Fall-Mamba 的做法**：音频 → FFT → Mel 滤波器组 → Mel-Spectrogram → 作为 "2D 图像" 与视频帧对齐。

**为什么有用**：WavJEPA 需要原始波形作为输入（200M 参数），在实时场景下可能过重。Mel-Spectrogram + 轻量 CNN 作为备选方案可以在边缘设备 (Jetson) 上运行。

```python
# 轻量音频路径 (备选，用于边缘设备)
audio → Mel-Spectrogram (256×256) → MobileNetV2 → (B, 768)
# 替代: WavJEPA (~200M) → 仅服务器端使用
```

---

### 10.3 优化后的完整架构对比

```
优化前 (T-JEPA):                          优化后 (T-JEPA + Fall-Mamba):

Video ──┐                                 Video ──┐
Audio ──┤                                 Audio ──┤
Skel ──┼─ MoE gate → Expert → sum         Skel ──┼─ MoE → Cross-Attn 精炼
Text ──┘                                  Text ──┘
       │                                         │
  Simple fusion                            MoE + Cross-Attn 双层融合
       │                                         │
       ▼                                         ▼
  Transformer Predictor                    Bidirectional Mamba Predictor
  O(N²)                                    O(N), 快 3-5x
       │                                         │
       ▼                                         ▼
  正常训练                                  + Frame Masking (鲁棒性)
                                           + DropPathway (容错)
                                           + Mel-Spec 音频备选 (边缘部署)
```

### 10.4 Fall-Mamba 实验结果（可直接引用为 Baseline）

| 模型 | Le2i 准确率 | 备注 |
|------|------------|------|
| Fall-Mamba tiny (video only) | 91.15% | 最轻量 |
| Fall-Mamba small (video only) | 94.44% | 适中 |
| Fall-Mamba middle (video only) | 96.77% | 单模态最佳 |
| Fall-Mamba tiny (video+audio) | 95.01% | Cross-Attn 融合 |
| Fall-Mamba middle (video+audio, 8帧) | 98.88% | 多模态 |
| **Fall-Mamba middle + Frame Masking** | **99.63%** | 最佳结果 |
| Fall-Mamba middle (Dark 环境) | 98.14% | 暗光下仍然高精度 |
| Fall-Mamba (30% 混合干扰) | 92.62% | 极端干扰下可用 |

