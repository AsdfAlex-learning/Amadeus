# Amadeus 项目复盘审查报告

> **审查对象**: `D:\Whoami\Amadeus` @ `fix/training-pipeline-issues` (7070c9c)
> **审查日期**: 2026-09-15
> **代码规模**: 37 个源文件 / 4,501 LOC（`src/`）+ 2 个脚本 + `demo.py`
> **审查方式**: 主 agent 逐文件通读 + 文档-代码一致性交叉核对
> **审查范围**: `src/` 全部 Python 文件、`scripts/`、`demo.py`、`config/`、`docs/`、`.github/workflows/`

---

## 修复状态（2026-09-15，分支 `fix/runtime-p0`）

| 缺陷 | 状态 | 提交 | 验证方式 |
|---|---|---|---|
| C-9 应用无法启动（Qt6 导入） | ✅ 已修 | `306ff29` | 导入冒烟 37/37 |
| C-3 `--resume` 崩溃 | ✅ 已修 | `9c5446f` | 契约测试 RED→GREEN |
| C-4 视觉帧 HWC/CHW | ✅ 已修 | `f2e3927` | 2 个契约测试 RED→GREEN |
| C-5 TTS 采样率不一致 | ✅ 已修 | `4687fe3` | 契约测试 RED→GREEN |
| C-6 Speak 依赖麦克风输入 | ✅ 已修 | `529c279` | 契约测试 RED→GREEN |
| C-8 PerformanceEngine 未接线 | ✅ 已修 | `6e7f6ec` | 契约测试 RED→GREEN |
| C-2 fp16 路径必然崩溃 | ✅ 已修 | `3e7ae79` | Layer B 测试 + **CUDA 实跑** `(1,49,45) float16` |
| **C-1 推理跑在实时回调线程** | ⚠️ **部分** | `2a9bee2` | 回调已仅入队（实测 500ms→<50ms）。但 T1 风扇出使 VAD 与推理共用线程，**Stage 1.4 未完成**，慢推理仍会阻塞 VAD |
| C-7 假流式生成 | ✅ **已解决（改道）** | 见 docs/adr/0005 | 已实测确认：pinned `transformers 4.32.1` 的 `TextIteratorStreamer` 只在 `end()` 时冲刷缓冲，无法增量输出（3 个 token、间隔 0.25s，实测首块在 1.14s 后一次性到达）。修复需要升级 transformers 或换流式机制，两者都应单独评估 |
| H-1/H-2/H-6/H-8 等 | 部分已修 | `3ac8b8c`(H-8) | H-2/H-6 属 Stage 2 |

测试：`pytest tests/` → **75 passed**（conda `amadeus`，Python 3.11.5；hermetic，约 5 秒）。

修复过程中 C-2 又暴露出 **3 个此前未记录的 dtype 缺陷**（`TextEncoder.to_half` 的 `_loaded` 守卫、`TimestepEmbedding` 用 float32 特征喂 fp16 MLP、`_resample_time` 的 float32 `linspace` 静默提升 dtype）——最后一个在训练中不可见、且当编码器输出长度恰好等于 T 时被掩盖。

---

## 1. 总体结论

**一句话**：这是一个**训练管线已经被认真打磨过、但运行时（runtime）从未被端到端跑通**的项目。所有已知问题都属于"推理/部署侧"，而 `docs/TRAINING_PIPELINE_REVIEW.md` 已经把训练侧审查得很彻底。

| 维度 | 评分 | 说明 |
|---|---|---|
| 架构设计 | ★★★★☆ | 分层清晰、职责分明、降级策略完善；理念（Live2D 参数空间而非像素）正确 |
| 训练管线 | ★★★★☆ | x-prediction / EMA / 全量快照 / 早停 / LoRA 生命周期完整，有 ADR 记录 |
| **运行时正确性** | ★☆☆☆☆ | **存在 8 个确定性崩溃/bug，端到端从未跑通** |
| 线程与实时性 | ★☆☆☆☆ | 扩散模型跑在 PortAudio 实时回调线程中，架构性错误 |
| 测试 | ★☆☆☆☆ | 已建立 hermetic 测试套件（`tests/`，7 RED 契约测试，5s 跑完）；CI 仍未覆盖当前分支 |
| 文档一致性 | ★★☆☆☆ | 架构理念文档优秀，但 README 的行数/状态/命令大量失真 |
| 依赖管理 | ★★☆☆☆ | 4 处依赖声明互不一致 |
| 安全 | ★★☆☆☆ | `weights_only=False` + `trust_remote_code=True` + 无校验下载 |

---

## 2. 🔴 Critical —— 阻断运行时（确定性缺陷）

这些是通读中发现的、**此前两份审查文档均未记录**的问题。

### C-1 扩散推理跑在 PortAudio 实时音频回调线程内

- `src/app/window.py:124` 把 `motion_model.process_user_audio` 直接注册为原始音频回调
- `src/audio/pipeline.py:68-70` `on_raw_audio` → `capture.on_audio(callback)`
- `src/audio/capture.py:61-71` `_audio_callback` 是 PortAudio 的实时回调，**同步**调用所有回调
- `src/motion/inference.py:193-197 → 209-236 → 251-307`：Hubert 编码 + 4 步 DiT 前向

回调必须在约 64ms（1024 samples @16kHz）内返回。把这套计算放进去，在 CPU（项目声明的推理目标 M2 / R7 8854HS）上会稳定产生音频丢帧（xrun）。

**同一根因**还导致 ASR：`src/audio/pipeline.py:110` 的 `self.asr.transcribe()` 在 `_on_audio_chunk` 里同步执行，Whisper 会阻塞音频线程数百毫秒到数秒。

> 缺少的核心组件：音频回调 → 队列 → 工作线程。这是运行时最根本的缺陷。

### C-2 `to_half()` 漏掉 `input_proj` → CUDA 上 fp16 推理必然崩溃

- `src/motion/model.py:376-391` 的 `to_half()` 列表包含 `audio_proj / cross_proj / mode_embedding / time_embed / identity_embedding / output_head` + `dit_blocks`，**唯独缺 `input_proj`**
- `src/motion/model.py:466` `x = self.input_proj(noisy_params)` → 保持 fp32
- `src/motion/model.py:474` `x = x + listen_emb`（fp16）→ 后续进入 fp16 权重的 `nn.MultiheadAttention`

结果：`RuntimeError: expected scalar type Half but found Float`。而 `config/default.yaml:87 use_fp16: true` + `src/motion/inference.py:92-93` 会在非 CPU 设备上启用 `to_half()`。**即 CUDA/GPU 推理路径 100% 崩溃**，只有 CPU 路径可用（而 CPU 路径又被 C-1 判了死刑）。

### C-3 `--resume` 完整快照必然崩溃：`scaler` 先用后定义

- `src/motion/training/train.py:163-178` 的恢复块在尝试 `optimizer/scheduler/scaler` 的状态恢复
- `src/motion/training/train.py:174` `if scaler is not None and "scaler" in _resume_state:`
- `src/motion/training/train.py:193` 才定义 `scaler = torch.amp.GradScaler(...)`

`scaler` 是函数局部变量且在 174 行之前从未赋值 → 只要 `_resume_state is not None` 就抛 `UnboundLocalError`。**任何通过 `--resume` 从完整快照恢复的训练都会崩溃**，而 `README.md:127-129` 明确宣称 "resume is lossless"。讽刺的是该块上方的注释还写着 "Must run AFTER these objects are constructed"。

### C-4 视觉帧 shape 不一致：HWC vs CHW —— 开摄像头即崩

- 相机输出 `(224, 224, 3)`：`src/perception/camera.py:92-94`
- `src/motion/inference.py:241` 空缓冲时返回 `(1, 3, 224, 224)`（CHW，正确）
- `src/motion/inference.py:247/249` 有帧时返回 `np.stack(...)` → `(5, 224, 224, 3)`（**HWC**）
- `src/motion/model.py:103` `B, T, C, H, W = frames.shape` → C=224
- `src/motion/model.py:104` `frames.view(B*T, C, H, W)` → 送 224 通道给 `mobilenet_v3_small` → 崩溃

训练侧 `src/motion/training/train.py:207` 用的是 `(B,5,3,224,224)`（CHW），与推理侧不一致。**结论：带摄像头的推理路径从未被运行过。**

### C-5 TTS 采样率与运动模型采样率不一致（口型必然不同步）

- `config/default.yaml:71` `tts.sample_rate: 22050`
- `src/motion/inference.py:17` `self.sample_rate = config["audio"]["sample_rate"]` = **16000**
- `src/motion/inference.py:199-202` 把 22050Hz 样本塞进 `_tts_buffer`，而 `src/motion/inference.py:210,216` 按 16000/秒 消费

结果：TTS 音频被以 1.38× 速度喂给模型，模型看到的"语音"时长被压缩 27.6%。`src/tts/engine.py:119-121` 的 `_silence()` 又硬编码 16000，第三处采样率假设。

### C-6 Speak 状态的运动依赖麦克风持续输入（违反"always-on"设计）

- `src/motion/inference.py:199-202` `process_tts_audio` **只做缓冲，不触发推理**
- 推理只由 `process_user_audio`（`inference.py:193-197`）驱动，而它由麦克风回调驱动

因此 TTS 播放期间若麦克风停止（`window.py:238` `audio_pipeline.stop()`），角色立刻静止。这与 `CONTEXT.md` 宣称的"引擎 always-on、状态由音频存在驱动"相矛盾。且 Speak 与 Listen 用的是同一路 `user_audio` 触发，状态区分实际上不成立。

### C-7 `DialogueModel.generate()` 是假流式：阻塞至完整生成后才逐字符吐出

- `src/dialogue/model.py:69-92`：开线程 → `thread.join()`（立即 join，线程毫无意义）→ `yield from full_text`

`yield from` 一个字符串会**逐字符**产出。首字节延迟 = 完整生成时间（最多 512 tokens）。UI 上的"流式打字"其实是生成完毕后的事后动画。`max_new_tokens=512` 硬编码，`src/dialogue/model.py:14` 读入的 `self.max_tokens` 从未使用。

### C-8 `PerformanceEngine` 完全未接线 —— persona 性能参数是死配置

- `src/motion/performance.py`（300 行）**未被 `window.py` / `inference.py` 任何地方导入**
- `config/default.yaml:143-150` 的 `performance:` 段只被 `src/config.py:48-58 get_performance_config()` 读取，而该函数**从未被调用**
- `CONTEXT.md` 把 `gesture_scale / react_speed / expressiveness / mouth_open_max / head_motion_range / idle_energy` 记录为生效参数

即：6 个 persona 参数、300 行引擎代码、一整段配置，全部是死的。`inference.py:44 self._overlap = 0.2` 硬编码也从未被 performance 覆盖。

### C-9 应用无法启动：Qt6 下 `QOpenGLWidget` 导入路径错误

> 本节由 `tests/test_import_smoke.py` 在建立测试套件时发现（2026-09-15）。

- `src/app/live2d_widget.py:6` `from PySide6.QtWidgets import QOpenGLWidget`
- PySide6 6.11.0（`environment.yml:17`）中 `QOpenGLWidget` 已移至 **`PySide6.QtOpenGLWidgets`**

实测：

```
>>> from PySide6.QtWidgets import QOpenGLWidget
ImportError: cannot import name 'QOpenGLWidget' from 'PySide6.QtWidgets'
>>> from PySide6.QtOpenGLWidgets import QOpenGLWidget   # OK
```

传播路径：`src/app/window.py:72` 在 `_setup_ui()` 内 `from src.app.live2d_widget import Live2DWidget` → `AmadeusWindow.__init__` 抛 `ImportError` → **`python -m src.main` 立即崩溃，应用根本无法启动**。

这是"运行时从未端到端跑通"最直接的证据：连窗口都建不起来。也意味着此前所有关于渲染、TTS 播放、摄像头接线的验证都从未发生过。

---

## 3. 🟠 High —— 功能性错误

### H-1 感知子系统（人脸/注视/表情）计算结果被完全丢弃

`src/perception/camera.py:67 on_perception` 从未被注册 —— `window.py:133` 只调用了 `on_frame`。而 `camera.py:102` 判断 `if self._perception_callbacks and ...`，列表为空 → **`_process_face_mesh` 一次都不会执行**。`ARCHITECTURE.md:204` 宣称 "These results are mapped to Live2D parameters in the window" —— 不存在。

### H-2 `_try_infer` 的 overlap 逻辑是 no-op

`src/motion/inference.py:225-231`：`popleft()` 出 `keep_n` 个元素后又 `extendleft(reversed(keep))` 原样放回 —— 完全等价于什么都不做。`ARCHITECTURE.md:187` 宣称 "Streaming ... inference with overlap-add" 并未实现。

### H-3 5 路多模态实际只有 1 路

`src/motion/training/dataset.py:150-158`：`visual_frames` = 全零、`tts_audio` = 全零、`text_prompt` = `""`。训练时 `train.py:206-209` 直接吃这些零。所以模型实际只从 `user_audio`（+ `identity_id`）学习。但推理时却喂 `tts_audio`（`window.py:192`）并期望"说话动作"。**训练分布与推理分布不一致** —— 这是 Speak 表现不可能好的结构性原因。

### H-4 45 个参数中约 30 个恒为常量 0，与 Sigmoid 输出冲突

`src/motion/preprocess/mappings/default.yaml`：仅约 15 个参数有非平凡映射，其余为 `constant: 0.0`。而 `src/motion/model.py:355` 的 `nn.Sigmoid()` 输出 ∈ (0,1) 永不为 0，梯度持续把输出压向 0，与 45 维中的恒定 0 目标形成系统性张力。`performance.py:266-268` 又把 head 参数围绕 **0** 缩放，但 `default.yaml:70` 的 head 参数 bias 是 **0.5** —— 两处对"中性值"的定义直接冲突。

### H-5 Live2D 参数被全局夹到 [0,1]（真实模型语义错误）

`src/live2d/renderer.py:62` `clamped = max(0.0, min(1.0, float(value)))`。真实 Live2D 中 `ParamAngleX ∈ [-30,30]`、`ParamBodyAngleX ∈ [-10,10]`。全量夹到 [0,1] 会让人物头部永远无法正常转动（只能贴 0 或 1 极值）。`renderer.py:39` 构建的 `_param_range` 字典从未被读取（死代码）。

### H-6 参数队列产出/消费速率不匹配 → 可见抖动

模型 1 秒产出约 49 帧（50Hz），而 `src/app/live2d_widget.py:40` 每个 paint（60fps）只弹 1 帧 → 队列（`maxlen=120`）约 2 秒排空，随后回落 `_default_params`（`live2d_widget.py:44`）直到下一 chunk。同时 49 帧以 60fps 播放 → 动作比实时快约 1.2×。

### H-7 `DiTBlock` 每次前向重复计算同一 AdaLN 三次

`src/motion/model.py:248-250`：`self.attn_norm1(x, c)` 被连写三遍作为 Q/K/V，即 3× LayerNorm + Linear 开销。同时 `mask` 参数（`model.py:243, 267`）在所有注意力中均未使用 —— 模型是非因果的，chunk 之间无法保持时序一致。

### H-8 每 chunk 重复做一次 Hubert 前向

`src/motion/inference.py:272-274` 先编码一次仅为取 `T`，随后 `inference.py:291` 模型内部再编码一次 → 每 chunk 2× Hubert 成本。

### H-9 `context.py` 的"摘要"名不副实

`src/dialogue/context.py:44-48,52-72`：`_summary` 每次被**覆盖**为"最后一条被裁剪的消息的前 200 字符"，`_inject_summary` 又替换既有摘要 → 永远只有一条被截断的消息，不是累积摘要。`ARCHITECTURE.md:114` 宣称 "automatic summarization on overflow"。`_estimate_tokens = len//2`（`context.py:76`）；`get_messages` 浅拷贝（dict 共享）；`add_system_message` 从未使用。

### H-10 TTS 默认引擎必然静默失败

`config/default.yaml:65 engine: cosyvoice2` → `src/tts/engine.py:45` 把 `CosyVoice2` **模型对象**存为 `self._model`，而 `engine.py:36` 执行 `self._model(text)`；CosyVoice2 不可这样调用 → 抛异常 → `engine.py:39` 返回 `_silence(1.0)`，且因 `_loaded=True`（`engine.py:29`）**不会降级到 pyttsx3**。用户听到的是一片静音，且无任何错误提示。

### H-11 关闭应用时的跨线程竞争

`src/app/window.py:195-196` 在**生成工作线程**里惰性创建 `self._pyaudio`，而 `window.py:247-248` 在**主线程** `closeEvent` 里 `self._pyaudio.terminate()`。同时 `_speak_response` 每次说话都新建/销毁一个输出流（`window.py:197-205`），无中断机制。

### H-12 生成完成信号早于 TTS 播放

`src/app/window.py:169-173`：`finally` 块先 `emit(response_done)`（把 `_generating=False`），**之后**才调 `_speak_response`。UI 会在角色还在说话时显示"就绪"，且新语音可与 TTS 播放并发。

### H-13 CI 存在但从未覆盖当前分支，且类型检查必然红

- `.github/workflows/ci.yml:3-7` 仅在 **`main`** 分支的 push/PR 触发
- 实际开发全在 `fix/training-pipeline-issues`，而 `main` 落后 19 个 commit
- 因此**当前全部代码从未被 CI 验证过**
- `ci.yml:67` `basedpyright src/` 无任何 `pyrightconfig.json`，对 `renderer.py:45 self._live2d.init()` 这类写法会大量报错

### H-14 零测试

无 `tests/`、无 `conftest.py`、无 pytest 配置。CI 只做 ruff + basedpyright + AST 语法 + import 冒烟，**不执行任何测试**（`pytest` 只在 `pyproject.toml:27` 的 dev extra 里躺着）。而 `README.md:308` 写着 `pytest tests/` —— 该目录不存在，命令必失败。

---

## 4. 🟡 Medium

| # | 位置 | 问题 |
|---|---|---|
| M-1 | `src/perception/camera.py:88-91` | `if not ret: continue` → 摄像头故障时 CPU 100% 忙等自旋 |
| M-2 | `src/perception/camera.py:13` | `enable_face` 标志被存储但从未使用（150/204 行用的是 expression/gaze） |
| M-3 | `src/audio/pipeline.py:82-89` | VAD 起始帧被**重复采样两次**（先入 pre-speech buffer 再 append） |
| M-4 | `src/audio/pipeline.py:83` | VAD 仅用 RMS 单阈值 0.02，无迟滞，无自适应噪声底 |
| M-5 | `src/audio/capture.py:20` | `pyaudio.PyAudio()` 在构造函数中初始化 —— 仅枚举设备（`demo.py:48`）也会占用音频子系统 |
| M-6 | `src/dialogue/model.py:39-43` | `device_map="mps"` 非法（应为 `"auto"` 或 dict）；`load_in_8bit` + `torch_dtype=float16` + `device_map` 组合需 `BitsAndBytesConfig` |
| M-7 | `src/dialogue/model.py:45-47` | `trust_remote_code=True` —— 任意代码执行风险 |
| M-8 | `src/dialogue/model.py:21-28`、`src/motion/inference.py:61-67`、`src/config.py:61-79` | 同一套 device 探测逻辑**重复三份**，且行为不一致（config 版支持 `prefer`，另两份不支持） |
| M-9 | `src/dialogue/model.py:98,111`、`demo.py:265` | Ollama URL 硬编码 `localhost:11434` 三处；`urlopen` 无超时 → 可永久挂起 |
| M-10 | `src/dialogue/persona.py:39-45` | `format_prompt(history)` 完全忽略 `history`；`window.py:159` 传 `[]` |
| M-11 | `src/motion/inference.py:83,90` | `torch.load(..., weights_only=False)` 加载不可信 `.pt` → 任意代码执行 |
| M-12 | `src/motion/preprocess/face_landmarker.py:47-52,103` | `MODEL_FILENAME` 是 `..._v2_with_blendshapes.task`，但下载 URL 与落地文件名都是 `face_landmarker.task` —— 若资产不含 blendshapes，所有帧会被判为 bad frame 而静默产出全零 |
| M-13 | `src/motion/preprocess/face_landmarker.py:65-75` | Tasks API 与 legacy 两条路径都失败时**只警告**，之后所有帧静默返回全零 —— 训练会"成功"但学到零 |
| M-14 | `src/motion/model.py:35,87,149` | 三个编码器加载失败均静默降级为 stub（Hubert 退化为 `expand(-1,-1,768)` 的伪特征），模型会在垃圾特征上"正常"训练 |
| M-15 | `src/motion/preprocess/pipeline.py:59-66` | 同一视频被 `VideoReader` 打开两次（一次抽音频、一次抽帧） |
| M-16 | `src/motion/preprocess/pipeline.py:86-88` | `assert` 做校验，`python -O` 下失效 |
| M-17 | `src/motion/training/train.py:58-60` | 扩散 schedule 硬编码 `1e-4/0.02`，忽略 `config.training.beta_start/beta_end` |
| M-18 | `src/motion/training/train.py:306` | `(epoch+1) % 10` 硬编码，忽略 `training.checkpoint_interval` |
| M-19 | `src/motion/training/train.py:237-253` | `len(dataloader)` 不被 `grad_accum_steps` 整除时，最后一个累积组的梯度被丢弃 |
| M-20 | 全局 | 无随机种子设置 → 训练不可复现（其自身 roadmap 已承认） |
| M-21 | `src/audio/asr.py:42-49` | `urlretrieve` 无超时、无校验和、无进度；在首次使用时静默下载数百 MB |
| M-22 | `demo.py:96` | `torch.cuda.get_device_properties(0).total_mem` —— 属性名应为 `total_memory`，异常被吞后误报为"torch未安装" |
| M-23 | `src/motion/model.py:456-457` | `_resample_time` 已把 `visual_feat` 对齐到 T，紧随其后的 repeat/tile 分支**永不触发**（死代码，注释仍描述旧行为） |
| M-24 | `src/app/window.py:156` | 每次生成都新建 daemon 线程，无取消、无 join；关闭应用时线程仍在访问 Qt 对象 |

---

## 5. 文档 vs 代码一致性（文档漂移）

| 宣称 | 实际 |
|---|---|
| `README.md:9` Platform: **macOS / Linux** | 仓库含 `scripts/setup_windows.ps1`、`run_overnight.bat`，全部训练在 RTX 4060 Ti (Windows) |
| `README.md:183-211` 逐文件行数（`window.py 261`、`live2d_widget.py 79`、`renderer.py 115`、`model.py 432`） | 实际 **249 / 109 / 107 / 489** —— 全部过期 |
| `README.md:127-129` "resume is lossless" | `train.py:174` `scaler` 未定义 → resume 必崩（C-3） |
| `README.md:265` "x-prediction (model predicts x₀ via Sigmoid)" + 50Hz 输出 | 训练目标中 30/45 维恒为 0（H-4）；`tts/visual/text` 三路输入恒为零（H-3） |
| `README.md:308` "Run tests: `pytest tests/`" | `tests/` 不存在 |
| `README.md:320-327` Phase 8 "First end-to-end training run" 未勾选 | 正文却报告了完成的 3-epoch HDTF 结果，且 `models/motion/overnight_run/` 存有 epoch 110–190 检查点 |
| `ARCHITECTURE.md:187` "inference with overlap-add" | 未实现（H-2） |
| `ARCHITECTURE.md:114` "automatic summarization on overflow" | 只有单条截断消息（H-9） |
| `ARCHITECTURE.md:204` 感知结果映射到 Live2D 参数 | `on_perception` 从未注册（H-1） |
| `ARCHITECTURE.md:118` "Five input streams" | 实际训练时 4 路为常量（H-3） |
| `CONTEXT.md` persona 性能参数表 | 引擎未接线（C-8） |

**根本原因**：文档是按"设计意图"写的，而代码是增量演进的；没有机制在合并时校验二者一致性。

---

## 6. 依赖管理混乱

同一组依赖在四处声明且互不一致：

| 包 | `pyproject.toml` | `requirements.txt` | `environment.yml` | 实际被 import |
|---|---|---|---|---|
| `whisper-cpp-python` | ✅ :17 | ❌ | ✅ :32 | `audio/asr.py:21` |
| `pyttsx3` | ❌ | ✅ :13 | ✅ :33 | `tts/engine.py:91` |
| `librosa` | ❌ | ❌ | ❌ | `dataset.py:236`、`video_reader.py:163` |
| `matplotlib` | ❌ | ❌ | ❌ | `training/visualize.py:20` |
| `soundfile` | ✅ :21 | ✅ :11 | ✅ :27 | 多处 |
| `basedpyright` | ❌（CI 用） | ❌ | ❌ | `ci.yml:64` |

`environment.yml` 与 `requirements-lock.txt` 是两套独立的锁定，`ci.yml:64` 用后者装依赖 → 与 `environment.yml` 的实际验证环境不同源。

---

## 7. 安全

1. `src/motion/inference.py:83` / `train.py:90` / `lora.py:453` — `torch.load(..., weights_only=False)`：加载 `.pt` 即任意代码执行
2. `src/dialogue/model.py:45-47` — `trust_remote_code=True`
3. `src/audio/asr.py:42-49`、`src/motion/preprocess/face_landmarker.py:105` — 运行时自动下载模型，无校验和/签名
4. `src/dialogue/model.py:111` — 无超时的阻塞网络调用（可被本地进程 DoS）
5. 无 secrets 泄漏（`.gitignore` 覆盖 `.env`），`opencode.yml:31` 正确使用 `secrets.*`

---

## 8. 值得肯定的部分

- **架构理念正确且论证充分**：在 Live2D 参数空间（45 维）而非像素空间生成，`docs/MODEL_TRAINING_ROADMAP.md` 对"为什么"和"风险在哪"（Listen/Silence 的数据问题）的推演是高质量的
- **渐进降级设计完善**：ffmpeg→cv2、Tasks API→legacy FaceMesh、CosyVoice→ChatTTS→Fish→pyttsx3、Hubert 缺失→stub —— 思路对，虽然实现上有的降级过于安静（M-14）
- **训练侧工程化程度高**：x-prediction 自洽、全量快照、EMA、早停、per-step warmup+cosine、LoRA 完整生命周期（`LoRAConv1d` 的 channel-wise 分解是正确写法）、`--resume` 设计意图明确（只是实现有 C-3 的 bug）
- **ADR 与审查文档文化**：`docs/adr/0001、0002` + `TRAINING_PIPELINE_REVIEW.md`（831 行）显示有把决策沉淀成文档的习惯
- **提交信息规范**：`fix(train): ... (M4)` 这类带问题 ID 的原子提交是好的工程实践

---

## 9. 改进优先级

### P0 —— 让运行时能跑（1-3 天）

> **测试状态（2026-09-15）**：`tests/` 已建立，`pytest tests/` → **7 failed, 36 passed in 5.09s**。
> 下列第 1、3、4、5、6、7、8 项各自已有对应的 RED 测试作为回归保护（见附录 C）。

0. 修正 `live2d_widget.py:6` 的导入为 `PySide6.QtOpenGLWidgets`（C-9）—— **应用当前完全无法启动**
1. 引入音频工作线程 + 队列，把 `process_user_audio` 和 ASR 移出 PortAudio 回调（C-1）
2. `model.py:376-391` 补上 `input_proj.half()`（C-2）
3. `train.py` 把 `scaler` 定义移到恢复块之前（C-3）
4. 统一视觉帧为 CHW，相机输出改 `(3,224,224)` 并统一 `_prepare_visual_frames` 两条分支（C-4）
5. 统一 TTS/音频采样率（C-5）
6. 让 `process_tts_audio` 也能驱动推理，实现真正的 always-on（C-6）
7. 接上 `PerformanceEngine`（C-8）

### P1 —— 让运行时正确（3-5 天）

7. 接上 `PerformanceEngine`（C-8）
8. 注册 `on_perception`，打通感知→参数（H-1）
9. 修正 `renderer.py:62` 的参数范围处理，按参数名查真实 range（H-5）
10. 实现 chunk 间 overlap 与 widget 侧插值/时间戳（H-2、H-6）
11. 修 `cosyvoice2` 调用方式或让失败触发降级（H-10）

### P2 —— 建测试与 CI（2-3 天）

12. 建 `tests/`：优先为 `arkit_to_live2d.map()`、`MotionDataset._resample_motion`、`lora.apply/merge/remove`、`context._trim`、`inference._diffusion_infer` 的形状契约写单测 —— 上述一半 bug 会被这类测试直接抓到
13. `ci.yml` 触发分支加上工作分支；补 `pyrightconfig.json`；加 `pytest` 步骤
14. 在任何大改动前先写一个"端到端冒烟"测试：全组件降级模式下跑通 `audio→…→params`

### P3 —— 一致性与可复现（1 天）

15. 用 `README.md` 的行数/平台/命令做一次全量校准（写个脚本自动更新行数）
16. 统一依赖声明，选一个 source of truth
17. 加全局随机种子 + run manifest（其 roadmap Phase 2 已规划）

---

## 10. 结论

训练侧的问题已经被 `TRAINING_PIPELINE_REVIEW.md` 覆盖得很彻底，而且修得不错。**但那份报告止步于 `train.py` 的边界** —— 它没有审查 `window.py` 如何把各组件接起来，于是全部运行时缺陷被漏掉了。

本报告发现的 8 个 Critical 中，有 **6 个只在"组件被真正连接起来运行时"才会暴露**（C-1 线程、C-4 视觉 shape、C-5 采样率、C-6 Speak 触发、C-7 假流式、C-8 未接线），这正是"训练管线可跑 ≠ 系统可跑"的体现。仓库现状也印证了这点：

- `assets/live2d/` 只有 `.gitkeep` → 渲染从未接过真实模型
- `models/whisper/` 不存在 → ASR 从未真正转写过
- `DEMO_MISSING.md` 自己列出的状态是：Live2D ❌、语音输入 ❌、语音输出 ❌、动作 ❌、摄像头 ❌

**下一个最有价值的动作不是继续训练，而是把 `docs/MODEL_TRAINING_ROADMAP.md` 的 Phase 0（非神经基线）落地，并写一个端到端冒烟测试。** 有了基线，上述 bug 会在几秒钟内全部显形，而不是等到第一次真实对话才逐个爆炸。

---

## 附录 A：本次会话附带变更

上一轮为满足"把所有历史 commit 作者改为 `AsdfAlex-learning`"的要求，对 `fix/training-pipeline-issues`（56 commits）与 `main`（37 commits）执行了 `git filter-branch` 并 `--force-with-lease` 推送。含义：

- **所有 commit hash 已变更，任何其他克隆必须重新 clone**（不能 pull）
- 旧 hash 的引用（issue / PR / 外部链接）已失效
- `docs/MODEL_TRAINING_ROADMAP.md` 仍为未跟踪状态（`git status` 显示 `??`）

## 附录 B：审查覆盖清单

已通读的文件（36/37 个 Python 源文件 + 配置 + 文档）：

```
src/main.py                      src/config.py
src/app/window.py                src/app/live2d_widget.py
src/live2d/renderer.py           src/audio/{capture,asr,pipeline}.py
src/dialogue/{model,persona,context}.py
src/motion/model.py              src/motion/inference.py
src/motion/performance.py        src/motion/__init__.py
src/motion/preprocess/{pipeline,face_landmarker,arkit_to_live2d,video_reader,body_skeleton}.py
src/motion/preprocess/mappings/default.yaml
src/motion/training/{train,dataset,lora,ema,visualize,__init__}.py
src/tts/engine.py                src/perception/camera.py
scripts/train_base.py            demo.py
config/default.yaml              pyproject.toml
requirements.txt                 environment.yml
.github/workflows/{ci,opencode}.yml
docs/{ARCHITECTURE,TRAINING_PIPELINE_REVIEW,MODEL_TRAINING_ROADMAP}.md
README.md                        DEMO_MISSING.md
```

未逐行阅读：`scripts/download_models.py`（外链下载脚本，非核心）、各包的空 `__init__.py`。

---

## 附录 C：运行时契约测试套件（2026-09-15 建立）

分支 `fix/runtime-p0`，Python 环境为 conda `amadeus`（Python 3.11.5）。

```
tests/
├── conftest.py                 # hermetic 环境守卫 + 共享 fixture
├── test_import_smoke.py        # 37 个模块的导入冒烟
└── test_runtime_contracts.py   # C-3/C-4/C-5/C-6/C-8 的契约测试
```

**运行**：`pytest tests/` → `7 failed, 36 passed in 5.09s`

### 设计要点

- **Hermetic**：`HF_HOME` 指向临时目录 + `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`，Hubert 与 BERT 因此走各自的内置 stub 分支；`QT_QPA_PLATFORM=offscreen` 避免需要显示设备。
- **封堵 torchvision 的漏洞**：`mobilenet_v3_small(weights="DEFAULT")` 会无视 HF 离线设置，直接从 `download.pytorch.org` 下载 ~10MB。conftest 的 autouse fixture 将其钉死为 `weights=None`，日志可见 `MobileNetV3 using random init (no cached weights)`。仅权重改变，架构与所有 tensor shape 不变。
- **停在模型边界而非 mock 内部**：测试只替换 `_diffusion_infer`（扩散模型是权重与本组契约无关的组件，构造耗时 <0.5s），其余走真实调用链 `process_user_audio → _try_infer → _prepare_visual_frames`。

### RED 清单与对应缺陷

| 测试 | 缺陷 | 失败原因 |
|---|---|---|
| `test_module_imports_cleanly[src.app.live2d_widget]` | C-9 | `ImportError: cannot import name 'QOpenGLWidget'` |
| `test_model_receives_channel_first_visual_frames` | C-4 | 收到 `(5,224,224,3)`，期望 `(5,3,224,224)` |
| `test_visual_layout_agrees_with_and_without_camera` | C-4 | 无相机分支 `(1,3,224,224)` vs 有相机分支 `(5,224,224,3)` |
| `test_one_second_of_tts_audio_occupies_one_second_of_motion_time` | C-5 | 22050 样本残留 6050 未消费 |
| `test_tts_audio_alone_drives_motion_inference` | C-6 | 喂 TTS 音频产生 0 次推理步 |
| `test_training_resumes_from_a_full_snapshot` | C-3 | `UnboundLocalError: scaler` @ `train.py:174` |
| `test_persona_performance_params_change_emitted_motion` | C-8 | gesture_scale=0.0 与 1.0 输出完全一致 |

### 尚未覆盖（需 Layer B / Layer C）

- **C-1**（音频回调内跑扩散推理）与 **H-1**（感知回调未注册）是 `AmadeusWindow` 的接线缺陷，需构造窗口才能行为化验证 —— 而窗口构造目前被 C-9 阻断。
- **C-2**（CUDA + fp16 dtype 崩溃）需要 Layer B 的真实模型前向；本机 CUDA 可用，可立即复现。
- **C-7**（假流式）需要 `DialogueModel` 的真实生成路径。
