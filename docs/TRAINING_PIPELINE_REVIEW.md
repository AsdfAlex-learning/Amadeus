# Amadeus 训练与微调 Pipeline 深度审查报告

> **审查日期**: 2026-06-07  
> **审查范围**: 模型架构 (`model.py`, `inference.py`)、训练循环 (`train.py`)、数据集 (`dataset.py`)、LoRA 微调 (`lora.py`)、预处理管线 (`preprocess/`)、性能参数 (`performance.py`)、推理路径 (`inference.py`)  
> **审查文件**: 14 个核心文件，~2500 行代码  
> **项目状态**: Roadmap Phase 7（数据预处理）已完成，Phase 8（模型训练与数据管线验证）待开始

---

## 目录

- [严重性总览](#严重性总览)
- [Critical — 阻断训练](#critical-—-阻断训练)
  - [C1: Sigmoid 输出层与 ε-prediction 训练目标不兼容](#c1-sigmoid-输出层与-ε-prediction-训练目标不兼容)
  - [C2: FPS 时空对齐错误 — 25fps 数据 vs 50 帧模型输出](#c2-fps-时空对齐错误--25fps-数据-vs-50-帧模型输出)
- [High — 推理阶段会失败](#high-—-推理阶段会失败)
  - [H1: DDIM 推理步骤不符合标准公式](#h1-ddim-推理步骤不符合标准公式)
  - [H2: LoRA 推理加载路径缺失](#h2-lora-推理加载路径缺失)
- [Medium — 质量退化](#medium-—-质量退化)
  - [M1: 训练时视觉帧始终为零](#m1-训练时视觉帧始终为零)
  - [M2: weight_decay 参数未传入优化器](#m2-weight_decay-参数未传入优化器)
  - [M3: 推理时 T=50 硬编码](#m3-推理时-t50-硬编码)
  - [M4: Learning rate warmup 未实现](#m4-learning-rate-warmup-未实现)
- [Low — 最佳实践缺失](#low-—-最佳实践缺失)
  - [L1: 无数据增强](#l1-无数据增强)
  - [L2: 无 EMA 权重](#l2-无-ema-权重)
  - [L3: 无早停机制](#l3-无早停机制)
  - [L4: 检查点不保存优化器状态](#l4-检查点不保存优化器状态)
  - [L5: Legacy 数据集路径的时间对齐错误](#l5-legacy-数据集路径的时间对齐错误)
- [模型架构审查](#模型架构审查)
  - [架构总览](#架构总览)
  - [架构关注点](#架构关注点)
- [LoRA 微调审查](#lora-微调审查)
  - [实现完整性](#实现完整性)
  - [LoRA 关注点](#lora-关注点)
- [预处理管线审查](#预处理管线审查)
  - [管线完整性](#管线完整性)
  - [预处理关注点](#预处理关注点)
- [训练循环审查](#训练循环审查)
  - [已有功能](#已有功能)
  - [缺失功能](#缺失功能)
- [ARKit→Live2D 映射审查](#arkitlive2d-映射审查)
- [修复优先级路线图](#修复优先级路线图)
- [附录: 完整文件清单与行号索引](#附录-完整文件清单与行号索引)

---

## 严重性总览

| 严重性 | 数量 | 阻断训练? | 阻断推理? |
|--------|------|-----------|-----------|
| 🔴 Critical | 2 | ✅ 是 | — |
| 🟠 High | 2 | — | ✅ 是 |
| 🟡 Medium | 4 | ❌ 质量退化 | ❌ |
| 🔵 Low | 5 | ❌ | ❌ |

---

## Critical — 阻断训练

### C1: Sigmoid 输出层与 ε-prediction 训练目标不兼容

**文件**: `src/motion/model.py:347`, `src/motion/training/train.py:149`

**问题**: 模型输出端使用 `nn.Sigmoid()`（输出范围 [0, 1]），但训练损失函数使用 ε-prediction：

```python
# model.py:338-347 — 输出头以 Sigmoid 结尾
self.output_head = nn.Sequential(
    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=5, padding=2),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Conv1d(hidden_dim // 2, num_params, kernel_size=5, padding=2),
    nn.Sigmoid(),  # ← 输出范围 [0, 1]
)

# train.py:149 — 训练目标是标准正态噪声
loss = criterion(pred, noise)  # noise ~ N(0, 1)
```

标准正态噪声 ε ~ N(0, 1) 有约 68% 的值落在 [-1, 1] 区间之外。Sigmoid 的输出范围 [0, 1] 使得模型**数学上不可能**正确预测目标 — 它被要求学习一个它无法表达的函数。

这不是 x-prediction（此时目标应该是 `motion` 而非 `noise`），也不是 v-prediction（此时目标是无界的 v = √α_t·ε - √(1-α_t)·x_0）。

**影响**: 训练无法收敛。模型会产生饱和输出（Sigmoid 两端），梯度在饱和区消失。

**修复方案（二选一）**:

| 方案 | 改动 | 推荐度 | 说明 |
|------|------|--------|------|
| **A: 切换到 x-prediction** | 保留 Sigmoid，修改 `loss = criterion(pred, motion)` | ⭐⭐⭐ | Live2D 参数天然在 [0, 1]，Sigmoid 作为约束合理 |
| B: 移除 Sigmoid | 替换为恒等映射，推理时对输出使用 clip | ⭐⭐ | 输出无界，需要额外约束 |

**方案 A 具体修改**:

```python
# train.py — 正向过程不变，修改 loss target
# 旧: loss = criterion(pred, noise)
# 新:
loss = criterion(pred, motion)  # 预测 x_0，而非 ε
```

```python
# inference.py — 重写 DDIM 采样为 x-prediction 公式
# 见 H1 修复方案
```

**工作量**: 1-4 小时（需与 H1 一起修复）

---

### C2: FPS 时空对齐错误 — 25fps 数据 vs 50 帧模型输出

**文件**: `src/motion/preprocess/pipeline.py:33`, `src/motion/training/dataset.py:62`

**问题**: 预处理管线默认以 `target_fps=25` 提取帧数据，但 `MotionDataset` 假设音频特征率映射到恒定的 50 帧/秒：

```python
# dataset.py:36
AUDIO_FEATURES_PER_SEC = 50  # Hubert stride=320 at 16kHz → 50 features/sec

# dataset.py:62
self.chunk_motion_frames = int(chunk_duration * self.AUDIO_FEATURES_PER_SEC)
# = int(1.0 * 50) = 50
```

而预处理输出的运动数据为每秒 25 帧：

```python
# pipeline.py:33 (默认 target_fps=25)
# config/default.yaml:142 — chunk_duration: 1.0
```

在 `_get_npz_item` 中，25 帧的数据被 zero-pad 到 50 帧：

```python
# dataset.py:131-134
if len(motion_chunk) < self.chunk_motion_frames:  # 25 < 50
    pad_len = self.chunk_motion_frames - len(motion_chunk)  # = 25
    motion_chunk = np.pad(motion_chunk, ((0, pad_len), (0, 0)))  # 后半全是零！
```

**影响**: 每个训练样本 50% 都是零填充。模型学到"后半段总是零"的模式，推理时输出会倾向在后半段衰减到静默状态。

**数据流全链路分析**:

```
视频 (25fps) → FaceLandmarker (25fps blendshapes) → ARKitToLive2D (25fps × 45 params)
  → NPZ 存储 (25fps)
  → MotionDataset (chunk_motion_frames=50) → 50% zero padding
  → Model (output: 50 frames × 45 params)
  → Inference (hardcoded T=50)
```

**修复方案（三选一）**:

| 方案 | 做法 | 推荐度 | 工作量 |
|------|------|--------|--------|
| **A: 数据集端插值** | 在 `_get_npz_item` 中将 25fps 插值到 50fps | ⭐⭐⭐ | 1-2h |
| **B: 预处理提取 50fps** | 将 `target_fps` 从 25 改为 50 | ⭐⭐ | 需重新处理所有视频 |
| C: 修改模型帧率 | T=25 对齐 25fps 数据 | ⭐ | 破坏 Hubert 对齐 |

**方案 A 具体修改**:

```python
# dataset.py — _get_npz_item 中，padding 之前添加插值
from scipy.interpolate import interp1d

if motion_chunk.shape[0] < self.chunk_motion_frames:
    # 插值而非 zero-pad
    t_in = np.linspace(0, 1, motion_chunk.shape[0])
    t_out = np.linspace(0, 1, self.chunk_motion_frames)
    motion_chunk = interp1d(t_in, motion_chunk, axis=0,
                            kind='linear', fill_value='extrapolate')(t_out)
    # head_angles 同理
    head_chunk = interp1d(t_in, head_chunk, axis=0,
                          kind='linear', fill_value='extrapolate')(t_out)
```

**工作量**: 1-4 小时

---

## High — 推理阶段会失败

### H1: DDIM 推理步骤不符合标准公式

**文件**: `src/motion/inference.py:152-195`

**问题**: 推理代码使用了一个自定义的"DDIM"步骤，但数学上与标准 DDIM 不一致：

```python
# inference.py 当前实现 (line 188-195)
params = (params - (beta_t / sqrt(1 - alpha_t)) * pred) / sqrt(1 - beta_t) + sigma_t * noise
```

标准 DDIM ε-prediction 公式为：

```
pred_x0 = (x_t - √(1-α_t) · ε_θ) / √(α_t)
dir_xt  = √(1 - α_{t-1} - σ²) · ε_θ
x_{t-1} = √(α_{t-1}) · pred_x0 + dir_xt + σ_t · noise
```

当前代码使用 `beta_t`（单步 β）代替 `1 - alpha_cumprod_t`（累积 α），分母使用 `√(1 - beta_t)` 而非正确的 DDIM 推导。此公式既不符合 DDPM 也不符合 DDIM。

**影响**: 推理产出质量低下或不稳定。但此问题与 C1 互相依赖 — 如果修复 C1 切换到 x-prediction，推理代码需要整体重写为 x-prediction 的 DDIM 步骤。

**修复**: 必须与 C1 一起修复。x-prediction 的 DDIM 步骤：

```python
# 模型直接预测 x_0 ∈ [0,1]（经过 Sigmoid）
pred_x0 = model(user_audio, tts_audio, visual, prompts, identity, t_tensor, x_t)

# 从 x_0 和 x_t 计算估计噪声
sqrt_alpha_t = torch.sqrt(alpha_cumprod_t)
sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_cumprod_t)
pred_eps = (x_t - sqrt_alpha_t * pred_x0) / sqrt_one_minus_alpha_t

# DDIM 采样步骤
alpha_prev = alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0)
sigma_t = eta * torch.sqrt(
    (1 - alpha_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_prev)
)
dir_xt = torch.sqrt(1 - alpha_prev - sigma_t**2) * pred_eps
x_t_prev = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma_t * noise
```

**工作量**: 2-4 小时（与 C1 一起修复）

---

### H2: LoRA 推理加载路径缺失

**文件**: `src/motion/inference.py:57-83`

**问题**: `DiffusionMotionInference.load_model()` 只加载基础模型 (`full_duplex_dit.pt`)，没有任何 LoRA 适配器加载逻辑：

```python
def load_model(self) -> bool:
    # ...
    checkpoint_path = self.model_path / "full_duplex_dit.pt"
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=self._device)
        self._model.load_state_dict(state)
    # ← 没有 load_lora() 调用
    # ← 没有 character_id 到 LoRA 文件的映射
    # ← 没有 LoRA 热加载机制
```

而 `lora.py` 已经实现了完整的 `load_lora()` API。

**影响**: 用户训练完角色 LoRA 后，无法在运行时加载它。角色特化功能完全不可用。

**修复**: 在 `load_model()` 后添加 LoRA 加载逻辑：

```python
from src.motion.training.lora import apply_lora, load_lora

# 加载基础模型后
lora_dir = self.model_path / "lora" / character_name
lora_path = lora_dir / "lora_adapter.pt"
if lora_path.exists():
    apply_lora(self._model, {"lora_rank": 8, "lora_alpha": 16})
    load_lora(self._model, lora_path)
```

**工作量**: 2-4 小时（包含 character 管理逻辑）

---

## Medium — 质量退化

### M1: 训练时视觉帧始终为零

**文件**: `src/motion/training/dataset.py:137-139`, `src/motion/training/train.py:130`

**问题**: `MotionDataset` 的 NPZ 和 legacy 路径都构造零视觉帧：

```python
# dataset.py:137-139
visual_frames = np.zeros(
    (self.num_visual_frames, 3, self.visual_size, self.visual_size), dtype=np.float32
)
```

训练时代码使用：

```python
# train.py:130
visual = batch.get("visual_frames", torch.zeros(B, 5, 3, 224, 224, device=device))
```

结果：`VisualEncoder`（MobileNetV3 + proj）对零输入产生无意义特征向量，被注入到 Listen 模式的 cross-attention 中。模型会学到"忽略视觉输入"，导致推理时即使输入真实相机帧，模型也无法正确利用。

**影响**: 视觉通路闲置（浪费 2.5M+ 参数），Listen 模式 cross-attention 混入不可靠特征。

**修复**: 添加 modality dropout — 在训练时随机将视觉特征置零，让模型学会在有/无视觉时都能工作：

```python
# 在训练循环中，计算 visual_feat 后
if self.training and torch.rand(1).item() < 0.5:
    visual_feat = torch.zeros_like(visual_feat)
```

对于 MVP，如果确实没有视觉训练数据，考虑完全移除视觉通路。

**工作量**: <1 小时（dropout）/ 2-4 小时（移除视觉通路）

---

### M2: weight_decay 参数未传入优化器

**文件**: `src/motion/training/train.py:116`, `config/default.yaml:103`

**问题**: 配置文件指定 `weight_decay: 0.01`，但优化器创建时未使用：

```python
# train.py:116
optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
# ← 缺少 weight_decay 参数，AdamW 默认值为 0.0
```

```yaml
# default.yaml:103
training:
  weight_decay: 0.01  # 从未被使用
```

**影响**: 无 L2 正则化，在小数据集上容易出现过拟合。

**修复**:

```python
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=learning_rate,
    weight_decay=config.get("weight_decay", 0.01)  # 从配置读取
)
```

**工作量**: <30 分钟

---

### M3: 推理时 T=50 硬编码

**文件**: `src/motion/inference.py:173`

**问题**:

```python
T = 50  # 硬编码！不是从音频编码器输出推导
params = torch.randn(B, T, self.num_params, device=self._device)
```

T=50 仅在音频恰好为 1 秒时正确（Hubert stride=320, 16000/320=50）。如果音频长度不同（如 VAD 产生的变长音频），帧率不匹配导致动作与音频不同步。

**修复**: 从 Hubert 编码器输出推导 T：

```python
with torch.no_grad():
    audio_feat = self._model.audio_encoder(user_wav)
T = audio_feat.shape[1]  # Hubert 自动计算
```

**工作量**: <1 小时

---

### M4: Learning rate warmup 未实现

**文件**: `src/motion/training/train.py:117`, `config/default.yaml:104`

**问题**: 配置文件指定 `warmup_steps: 100`，但训练代码使用了 `CosineAnnealingLR`（无 warmup 参数）：

```python
# train.py:117
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
# ← 没有 warmup
```

```yaml
# default.yaml:104
warmup_steps: 100  # 从未被使用
```

**修复**: 使用 `torch.optim.lr_scheduler.SequentialLR` 组合 warmup + cosine：

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_steps)
scheduler = SequentialLR(optimizer, [warmup, cosine], [warmup_steps])
```

**工作量**: 1-2 小时

---

## Low — 最佳实践缺失

### L1: 无数据增强

训练数据无任何增强 — 无音频噪声注入、无时间拉伸、无频谱增强。当前仅有的"增强"是随机块起始点选择（`start_chunk = np.random.randint(...)`），本质上是时间裁剪不是增强。

**建议**: 优先添加 modality dropout（M1 的一部分），然后添加：
- 音频高斯噪声注入（SNR 20-30dB）
- 随机时间拉伸（±5%）
- 随机模态丢弃（text prompt 置空等）

**工作量**: 1-2 天（完整增强管线）

---

### L2: 无 EMA 权重

扩散模型训练中，权重的指数移动平均（EMA, 衰减率 0.999）显著提升采样质量。当前实现直接使用最末步权重作为推理模型。

**建议**: 添加简单 EMA：

```python
from torch_ema import ExponentialMovingAverage
ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
# 训练循环中: ema.update(model.parameters())
# 推理时: ema.store(model.parameters()); ema.copy_to(model.parameters())
```

**工作量**: 1-2 小时

---

### L3: 无早停机制

训练循环仅在固定 epoch 数后停止，不会在验证损失持续上升时提前终止。

```python
# train.py:122 — 简单循环，无早停
for epoch in range(start_epoch, num_epochs):
    ...
```

**建议**: 添加 patience=10 的早停，基于验证损失：

```python
best_val_loss = float('inf')
patience_counter = 0
for epoch in range(start_epoch, num_epochs):
    ...
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存最佳模型
    else:
        patience_counter += 1
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
```

**工作量**: <1 小时

---

### L4: 检查点不保存优化器状态

```python
# train.py:208
torch.save(model.state_dict(), checkpoint_path)
# ← 仅模型权重，不包含优化器、调度器、epoch 状态
```

这导致无法完整恢复训练状态（优化器动量、学习率调度器状态等）。

**修复**:

```python
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'epoch': epoch,
    'scaler': scaler.state_dict() if scaler else None,
}, checkpoint_path)
```

**工作量**: <1 小时

---

### L5: Legacy 数据集路径的时间对齐错误

**文件**: `src/motion/training/dataset.py:151-173`

Legacy 路径 `_get_legacy_item` 假设音频和运动数据共享相同的时间轴：

```python
motion_chunk = motion[start:start + self.chunk_samples]  # chunk_samples=16000
# 但 motion 形状为 (T, 45)，16000 个帧是荒谬的
```

由于 NPZ 格式是主路径，此问题在正常流程中不会触发。

**建议**: 标记为 `@deprecated` 或修复帧率对齐逻辑。

**工作量**: 1-2 小时

---

## 模型架构审查

### 架构总览

```
FullDuplexDiT (432行, ~124M 参数)
├── HubertEncoder (frozen, ~94M) — 768-dim 音频特征 @50Hz
│   └── facebook/hubert-base-ls960
├── VisualEncoder (MobileNetV3-Small, frozen, ~2.5M) — 512-dim 视觉特征
│   └── proj: Linear(576 → 320)
├── TextEncoder (BERT-tiny, frozen, ~4.4M) — 128-dim 文本特征
│   └── proj: Linear(128 → 320)
├── Projections
│   ├── audio_proj: Linear(768 → 320) — 共享于 Listen/Speak 模式
│   ├── cross_proj: Linear(640 → 320) — 拼接条件投影
│   └── mode_embedding: Embedding(2 → 320) — Listen/Speak 模式标识
├── TimestepEmbedding (Sinusoidal + MLP, 320-dim)
├── IdentityEmbedding (16 → 320) — 角色身份标识
├── DiT Blocks ×4 (Interleaved Listen/Speak)
│   ├── AdaLN (LayerNorm + scale/shift from timestep)
│   ├── Self-Attention (8 heads, dim=320)
│   ├── Cross-Attention (8 heads, dim=320)
│   └── FFN (320 → 1280 → 320, GELU + Dropout)
└── Output Head (Conv1d×3 + Sigmoid → 45 params)
    ├── Conv1d(320, 320, k=5, p=2) + GELU + Dropout
    ├── Conv1d(320, 160, k=5, p=2) + GELU + Dropout
    └── Conv1d(160, 45, k=5, p=2) + Sigmoid
```

**参数分布**:

| 组件 | 参数量 | 可训练 | 占比 |
|------|--------|--------|------|
| HubertEncoder | ~94M | ❌ (frozen) | 75.8% |
| VisualEncoder | ~2.5M | 仅 proj | 2.0% |
| TextEncoder | ~4.4M | 仅 proj | 3.5% |
| DiT Blocks | ~15M | ✅ | 12.1% |
| Output Head | ~6M | ✅ | 4.8% |
| Projections | ~0.3M | ✅ | 0.2% |
| Identity Embedding | ~5K | ✅ | <0.01% |
| **Total** | **~124M** | **~24M** | — |

### 架构关注点

| 项目 | 描述 | 严重性 |
|------|------|--------|
| Sigmoid + ε-prediction | 见 C1 | 🔴 |
| 共享 HubertEncoder | Listen 和 Speak 模式共享同一个 Hubert，但输入分布截然不同（用户语音 vs TTS 合成语音）。Hubert 在两种分布上的特征提取可能不均 | 🟡 |
| Cross-attention 条件拼接 | `cross_kv = cat([audio_feat, visual/text], dim=-1)` → `cross_proj(640→320)` — 信息从 640 维被投影到 320 维，可能成为瓶颈 | 🔵 |
| Mode embedding 仅加法 | `x = x + (listen_emb or speak_emb)` — 仅一个 embedding 向量区分两种模式，对于模式行为的强差异可能不够 | 🟡 |
| 无残差连接到输出 | DiT blocks 输出直接送进 Conv head，无 skip connection from input noisy params | 🔵 可接受（标准 DiT 做法） |
| Gradient checkpointing | 默认在 CUDA/MPS 上启用 | ✅ |
| Xavier 初始化 | 仅对 dim≥2 参数，偏置和 1D 参数使用 PyTorch 默认 | ✅ |

---

## LoRA 微调审查

### 实现完整性

`src/motion/training/lora.py` (495行) 实现了完整的 LoRA 生命周期：

| 功能 | 状态 | 质量 | 备注 |
|------|------|------|------|
| `LoRALinear` (nn.Linear 替换) | ✅ | 良好 | Kaiming 初始化 A，零初始化 B；scaling = α/r |
| `LoRAConv1d` (nn.Conv1d 替换) | ✅ | 良好 | 正确的 channel-wise 分解 `(out, r) × (r, in*kernel)` |
| `apply_lora` | ✅ | 良好 | 支持通配符模式匹配，自动冻结基础模型 |
| `remove_lora` | ✅ | 良好 | 恢复原始层，丢弃 LoRA 贡献 |
| `merge_lora` | ✅ | 良好 | 权重合并进基础层，LoRA 矩阵零化 |
| `save_lora` | ✅ | 良好 | 仅保存 A/B 矩阵 + config，文件小 |
| `load_lora` | ✅ | 良好 | 自动 apply_lora + 权重加载 |
| 多模块类型支持 | ✅ | Linear + Conv1d | 覆盖模型的两种主要层类型 |
| Dropout 正则化 | ✅ | LoRA path 上 | 默认 0.0 |

### LoRA 关注点

| 项目 | 问题 | 严重性 |
|------|------|--------|
| **推理加载路径缺失** | `inference.py` 无 `load_lora()` 调用，角色特化无法使用 | 🟠 High (见 H2) |
| LoRA config 保存格式 | `torch.save(lora_config, ...)` 使用 `.pt` 后缀，但内容是 dict — 建议改为 `.json` 或 `.yaml` | 🔵 Low |
| Target 模块默认值 | 默认包含 `audio_proj` 和 `cross_proj`（单线性层）— LoRA 在小层上增益有限 | 🟡 Medium |
| merge 后性能 | `merge_lora` 后 LoRA 矩阵零化但包装器仍在，运行时有少量额外计算开销 | 🔵 Low |

---

## 预处理管线审查

### 管线完整性

| 组件 | 文件 | 状态 | Fallback |
|------|------|------|----------|
| VideoReader | `video_reader.py` | ✅ | ffmpeg → cv2 |
| FaceLandmarkerExtractor | `face_landmarker.py` | ✅ | Tasks API → legacy FaceMesh → zeros |
| ARKitToLive2DMapper | `arkit_to_live2d.py` | ✅ | YAML 配置驱动 |
| Pipeline orchestrator | `pipeline.py` | ✅ | CLI 入口点 |
| Bad frame 插值 | `pipeline.py:176-198` | ✅ | 线性均值插值 |
| BodySkeletonExtractor | `body_skeleton.py` | ⚠️ Stub | 返回全零（文档标注 deferred）|

### 预处理关注点

| 项目 | 问题 | 严重性 |
|------|------|--------|
| **目标帧率默认 25fps** | 与模型 50fps 对齐不匹配 (见 C2) | 🔴 Critical |
| 无并行视频处理 | 串行处理，大型数据集速度慢 | 🔵 Low |
| bad_frames 用 list 而非 set | 插值时 O(n) 查找，长视频性能差 | 🔵 Low |
| 无输出验证 | 保存 .npz 时不验证形状/范围一致性 | 🔵 Low |
| 无视频级元数据聚合 | 每个视频独立处理，无全局统计 | 🔵 Low |

---

## 训练循环审查

### 已有功能

| 功能 | 实现 | 质量 |
|------|------|------|
| 优化器 | AdamW | ✅ 但缺 weight_decay |
| 学习率调度器 | CosineAnnealingLR | ✅ 但缺 warmup |
| 混合精度 | AMP + GradScaler (CUDA) | ✅ |
| 梯度裁剪 | max_norm=1.0 | ✅ |
| 梯度累积 | grad_accum_steps=4 | ✅ |
| 验证集切分 | 10% random_split | ✅ |
| Epoch 检查点 | 每 10 epoch | ✅ 但不完整 |
| 恢复训练 | resume_from 参数 | ⚠️ epoch 从文件名解析 — 脆弱 |
| LoRA 训练 | --use_lora 参数 | ✅ |
| LoRA 检查点 | lora_adapter_epoch_NNNN.pt | ✅ |

### 缺失功能

| 功能 | 严重性 | 说明 |
|------|--------|------|
| Learning rate warmup | 🟡 Medium | config 有 `warmup_steps` 但代码未实现 |
| EMA weights | 🔵 Low | 扩散模型的推荐实践 |
| Early stopping | 🔵 Low | 无验证损失提前终止 |
| TensorBoard/WandB | 🔵 Low | 仅 loguru 文本日志 |
| 分布式 (DDP) | 🔵 Low | 单 GPU 可用于 MVP |
| 完整检查点 | 🟡 Medium | 不保存优化器/调度器状态 |
| 随机种子固定 | 🔵 Low | 无全局种子设置，不可复现 |

---

## ARKit→Live2D 映射审查

| Live2D 参数 | 映射类型 | 源 | 质量 |
|-------------|----------|------|------|
| ParamAngleX/Y/Z | head_angle | yaw/pitch/roll × scale + bias | ✅ 合理 |
| ParamBodyAngleX/Y/Z | head_angle | 同上 × 较小 scale | ✅ 身体跟随头部 |
| ParamEyeLOpen/R | blendshape | eyeBlink × -1 + 1.0 | ✅ 取反映射正确 |
| ParamEyeBallX/Y | blendshape | eyeLook × weight + bias | ✅ |
| ParamBrowLX/LY/RX/RY | blendshape | 多源加权 | ✅ |
| ParamMouthOpenY | blendshape | jawOpen - mouthClose | ✅ |
| ParamMouthForm | blendshape | smile - frown + bias=0.5 | ✅ |
| ParamCheek | blendshape | cheekPuff | ✅ |
| ParamBreath | constant | 0.0 | ✅ 不可从静态数据学习 |
| ParamArmX/Y (4个) | constant | 0.0 | ⚠️ 无 body skeleton 输入 |
| ParamHairX (4个) | constant | 0.0 | ✅ 物理驱动 |
| 其余 (眼泪/腮红/舌/耳/尾/翅/道具) | constant 或 简单 blendshape | 0.0 或 简单映射 | ⚠️ 45 参数中仅 ~15 个有非平凡映射 |

**关注点**: 45 个 Live2D 参数中，约 30 个为 constant(0.0) 或非常简单的映射。这意味着模型有 67% 的输出维度在训练时始终为 0，Sigmoid 会让它们趋向 0.5 的中间值（而非训练目标的 0），加剧训练不稳定。

---

## 修复优先级路线图

### Phase 1: 训练阻断修复 (1-2 天)

**必须在任何训练尝试之前完成。**

| # | 问题 | 修复 | 工作量 | 依赖 |
|---|------|------|--------|------|
| C1 | Sigmoid + ε-prediction | 切换到 x-prediction: `loss = criterion(pred, motion)` | 1h | 无 |
| C2 | FPS 对齐错误 | 数据集端插值 25fps→50fps | 1-2h | 无 |
| H1 | DDIM 推理错误 | 重写为 x-prediction DDIM | 2-3h | C1 |
| M2 | weight_decay 缺失 | 传入优化器参数 | 15min | 无 |

### Phase 2: 质量提升 (1-2 天)

**显著提升模型质量。**

| # | 问题 | 修复 | 工作量 | 依赖 |
|---|------|------|--------|------|
| M1 | 零视觉帧 | 添加 modality dropout | <1h | 无 |
| M3 | T 硬编码 | 从 Hubert 输出推导 | <1h | 无 |
| M4 | Warmup 未实现 | SequentialLR(warmup + cosine) | 1-2h | 无 |
| L3 | 无早停 | 添加 patience=10 早停 | <1h | 无 |
| L4 | 不完整检查点 | 保存 optimizer/scheduler/epoch | <1h | 无 |

### Phase 3: 推理完整性 (2-3 天)

**使系统端到端可用。**

| # | 问题 | 修复 | 工作量 | 依赖 |
|---|------|------|--------|------|
| H2 | LoRA 推理加载 | 添加 load_lora 到 inference.py | 2-3h | 无 |
| L2 | 无 EMA | 添加 ExponentialMovingAverage | 1-2h | 无 |
| L5 | Legacy 路径 | 标记 deprecated 或修复 | 1-2h | 无 |

---

## 附录: 完整文件清单与行号索引

| 文件 | 行数 | 关键审查点 |
|------|------|-----------|
| `src/motion/model.py` | 432 | L11-56: HubertEncoder; L58-111: VisualEncoder; L114-170: TextEncoder; L178-198: TimestepEmbedding; L201-211: AdaLN; L214-271: DiTBlock; L279-432: FullDuplexDiT; **L338-347: output_head + Sigmoid** |
| `src/motion/inference.py` | 227 | **L152-195: DDIM 推理步骤（错误公式）**; L173: T=50 硬编码; L57-83: load_model 无 LoRA |
| `src/motion/training/train.py` | 277 | **L116: AdamW 缺 weight_decay**; L117: 无 warmup; L148-149: ε-prediction loss; L208: 不完整检查点 |
| `src/motion/training/dataset.py` | 203 | **L62: chunk_motion_frames=50; L131-134: 零填充（应为插值）**; L137-139: 零视觉帧 |
| `src/motion/training/lora.py` | 495 | L35-95: LoRALinear; L97-170: LoRAConv1d; L221-308: apply_lora + 管理 API; L398-433: save/load |
| `src/motion/performance.py` | 300 | L38-72: PerformanceConfig; L125-300: PerformanceEngine |
| `src/motion/preprocess/pipeline.py` | 247 | L33: target_fps=25（应为 50）; L176-198: bad frame 插值 |
| `src/motion/preprocess/face_landmarker.py` | 302 | L44-302: FaceLandmarkerExtractor; L172-220: blendshape 近似 |
| `src/motion/preprocess/arkit_to_live2d.py` | 127 | L58-108: map() — YAML 配置驱动映射 |
| `src/motion/preprocess/video_reader.py` | 298 | L16-298: VideoReader — ffmpeg + cv2 fallback |
| `src/motion/preprocess/body_skeleton.py` | 49 | L11-49: Stub — 返回零 |
| `src/motion/preprocess/mappings/default.yaml` | 304 | L1-304: 完整 45 参数映射配置 |
| `config/default.yaml` | 156 | L96-121: 训练配置; L124-142: 预处理配置; L73-88: 模型架构配置 |

---

*报告结束。建议按 Phase 1 → Phase 2 → Phase 3 顺序修复，Phase 1 完成后即可开始 Base Model 训练。*