# Amadeus 运行时修复报告

> **分支**: `fix/runtime-p0`（基于 `fix/training-pipeline-issues` @ `7070c9c`）
> **提交数**: 23
> **测试**: 151 passing（hermetic，约 20 秒）
> **状态**: 已合并进 main（PR #1，2026-11-26，merge commit `db04bae`）；文档已随 PR 入库
> **更新（2026-10-12）**: C-7 已改道解决（见 §6）；分支现有 29 个提交、186 个测试。
> **更新（2026-11-26）**: PR #1 合并（74 commits + merge commit，234 tests）。清理评审（PR #2）修复落库，按内容记录（该评审的 M-编号与 PROJECT_REVIEW_REPORT 的 M 目录是两套编号，勿混用）：warmup_steps=0 真禁用、训练可复现种子（目录 M-20）、torch.load 可信注记（`5254805`）、motion 对比图 gt 重采样（`d71cdfa`）、梯度累积尾批（目录 M-19，`0c37efb`）；SilencePromptGenerator 已接线（`5aac7c4`）；basedpyright 归零并成为硬门禁。目录 M-5（构造器里建 PyAudio）/M-8（设备检测三处重复）/M-9（demo.py 硬编码 Ollama 地址）仍为开放项。已知未处理（有意保留待定夺）：训练 loss 未按 `grad_accum_steps` 缩放（累积组有效步长偏大）；`head_motion_range` 围绕 0 缩放而映射 YAML 的角度以 0.5 为中性。
> **日期**: 2026-09-15

---

## 0. 一页摘要

这份工作的起点是一次全项目审查，结论是：**训练管线已被认真打磨，但运行时从未被端到端跑通过。**

最直接的证据是第一次建立测试套件时发现的 **C-9**：`src/app/live2d_widget.py` 用了 Qt5 的导入路径（`PySide6.QtWidgets.QOpenGLWidget`），而 PySide6 6.11 已把它移到 `QtOpenGLWidgets`。因为 `window.py` 在 `_setup_ui()` 内部才 import 它，**`python -m src.main` 会立刻崩溃 —— 应用根本无法启动**。这意味着此前关于渲染、TTS 播放、摄像头接线的所有"验证"都从未发生过。

修复分两阶段进行，每个阶段都先出设计、再经 Oracle 独立评审、并按评审意见修订后才实施：

| | 内容 | 结果 |
|---|---|---|
| **Stage 1** | 线程所有权模型：音频回调仅入队、ASR/运动/播放各自拥有线程、关闭协议 | 1.1–1.5 完成 |
| **Stage 2** | 同步与降级：渲染排空策略、摄像头交接、播放时钟、中断、降级路径 | S2.1–S2.6 完成 |

**9 个 Critical 中 8 个已修复并验证**（C-1 至 C-6、C-8、C-9）；**C-7 已改道解决**（2026-10-12，见 §6）。

**诚实边界**：全部验证都是 hermetic 的 —— 测试不接触真实麦克风、摄像头、Live2D 模型、Whisper 权重或音频设备。因此 **"151 tests passing" 不等于 "应用已端到端跑通"**。真实设备上的首次运行仍未发生（见 §7）。

---

## 1. 如何运行

```bash
conda activate amadeus          # Python 3.11.5
pytest                          # 约 20 秒，151 passed
```

**环境说明**：本机 `amadeus` 环境是手工搭建的（`torch 2.6.0+cu124`），**与 `environment.yml` 锁定的 `torch 2.5.1` 不一致**。因此**不要**用它重建环境 —— 那会降级 torch 并很可能破坏 CUDA。本次只向该环境装入了 `pytest`。

**测试为何是 hermetic 的**（`tests/conftest.py`）：

- `HF_HOME` 指向临时目录 + `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`，使 Hubert 与 BERT 走各自的内置 stub 分支；
- **torchvision 不受 HF 离线设置约束**，`mobilenet_v3_small(weights="DEFAULT")` 仍会从 `download.pytorch.org` 下载约 10 MB —— 用 autouse fixture 把它钉死为随机初始化；
- `QT_QPA_PLATFORM=offscreen`，无需显示器。

---

## 2. 缺陷清单与修复状态

### 2.1 Critical

| ID | 缺陷 | 状态 | 提交 | 验证方式 |
|---|---|---|---|---|
| **C-9** | 应用无法启动（Qt6 导入路径） | ✅ | `306ff29` | 导入冒烟 37/37 |
| **C-2** | fp16 路径必然崩溃 | ✅ | `3e7ae79` | Layer B 测试 + **CUDA 实跑** |
| **C-3** | `--resume` 必崩（`UnboundLocalError: scaler`） | ✅ | `9c5446f` | 契约测试 RED→GREEN |
| **C-4** | 视觉帧 HWC/CHW 不一致 | ✅ | `f2e3927` | 2 个契约测试 RED→GREEN |
| **C-5** | TTS 采样率 22050 vs 运动引擎 16000 | ✅ | `4687fe3` | 契约测试 RED→GREEN |
| **C-6** | Speak 态依赖麦克风输入 | ✅ | `529c279` | 契约测试 RED→GREEN |
| **C-8** | `PerformanceEngine` 完全未接线 | ✅ | `6e7f6ec` | 契约测试 RED→GREEN |
| **C-1** | 扩散推理跑在 PortAudio 实时回调线程内 | ✅ | `2a9bee2` … `35e346c` | 实测 500.3 ms → <50 ms |
| **C-7** | 假流式生成（首字节延迟 = 完整生成时间） | ✅ **已解决（改道）** | `cdf58d6`、`ceb1df5` | 见 §6 |

### 2.2 值得单独说明的三组发现

**① C-2 牵出 3 个此前未记录的 dtype 缺陷。** 修 fp16 时，契约测试暴露出的不止是 `input_proj` 漏加：

- `TextEncoder.to_half()` 被 `_loaded` 守卫住，stub 编码器的投影停留在 float32；
- `TimestepEmbedding` 用 float32 计算正弦特征后直接喂给 fp16 MLP；
- **`_resample_time` 用 float32 `linspace` 静默把半精度条件提升为 float32** —— 这个在训练中完全不可见（训练全程 float32），且当编码器输出长度恰好等于 T 时被掩盖。由于 5 帧视觉特征总是要插值到 T，它在**每一次** fp16 前向都会触发。

修复方式不是往手维护的列表里补一项，而是用 `nn.Module.half()` 递归整棵树。

**② 测试套件抓到了应用根本无法启动。** 导入冒烟测试遍历 `src/` 下全部模块。这不是形式主义 —— 应用的 UI 模块是从 `AmadeusWindow.__init__` 内部惰性 import 的，所以叶子模块的导入错误不会在包导入时暴露，只会表现为"启动即崩溃"，而这正是"从未跑通"会掩盖的东西。

**③ S2.6 的真正价值是发现了一个组合缺陷。** Speak 态被播放位置门控（S2.4）、渲染器在下溢时保持末帧（S2.1）。一旦播放断路器跳闸，位置永不推进 → 运动停止 → 组件**永远**保持最后那个说话姿态。**三个分别验证通过的改动，合成出一个不可恢复的雕像。** 这是 Oracle 在评审里专门标出的 M6，也是分阶段验证之所以必要的最好例子。

---

## 3. 架构改动

### 3.1 Stage 1：线程所有权模型

修复前的状态：PortAudio 回调线程**同步**调用所有消费者，包括 4 步扩散前向与 Whisper 转写；`DiffusionMotionInference` 被四个线程无锁改写。

改动后的线程所有权：

| 线程 | 拥有 | 消费 | 产出 |
|---|---|---|---|
| T0 PortAudio 回调 | 无 | 设备缓冲 | `AudioRing`（仅拷贝 + 信号） |
| T1 音频排空 worker | VAD 状态、语句缓冲 | `AudioRing` | `asr_q`、`motion_in` |
| T2 ASR worker | `WhisperASR` | `asr_q`（FIFO，容量 3） | 文本 → Qt 信号 |
| T3 运动 worker | `DiffusionMotionInference` 全部 | `motion_in`、`tts_pending`、`visual_slot` | 参数帧 → `render_q` |
| T4 摄像头 worker | `VideoCapture`、FaceMesh | 摄像头 | `visual_slot` |
| T5 生成线程（每次生成） | LLM 流、TTS 合成 | 文本事件、逐代 token | UI 分片；音频 → T7 |
| T6 GUI 线程 | 全部控件、渲染器 | `render_q` | 用户输入 |
| T7 播放 worker | 输出流、自有 PyAudio | `play_q` | 扬声器；**播放位置** |

**实测证据**：修复前，单个休眠消费者让实时回调阻塞 **500.3 ms**（预算 64 ms）；修复后 <50 ms。

### 3.2 Stage 2：同步与降级

| 阶段 | 内容 | 提交 |
|---|---|---|
| S2.1 | `render_q` 独立定时器限速排空、**应用最后弹出的一帧**、下溢保持末帧 + 400 ms 空闲超时衰减回中性、溢出计数、credit 钳制 | `5f53759` |
| S2.2 | 摄像头→T3 的 `FrameBuffer` 交接（修跨线程竞争）、入库时间戳 + 按龄淘汰、取**最新 5 帧**、源头降频 4 Hz、失败设备不再忙等自旋 | `f09a503` |
| S2.3 | 抽取 T7 `PlaybackWorker`：分块写入 200 ms、递增位置、异步中断 + 80 ms 淡出、自有 PyAudio | `8dbb3d4` |
| S2.4 | Speak 态由播放时钟驱动：保留流 + 游标**门控** + max-lag 跳帧 + 单位统一 + Speak 下 mic 置零 | `3391bc0` |
| S2.5 | 逐代取消 token、`response_done` 代际门控、中断（VAD onset）、保留截断轮次 | `210b09b` |
| S2.6 | `TTSEngine.cleanup`、T1 断路器、**T7 失败降级** | `76f4316` |

---

## 4. 关键设计决策（含评审修正）

两个 ADR 都先自行出方案，再交 Oracle 独立评审，**并按评审意见修订后才实施**。评审确实改变了设计：

### 4.1 ADR 0003：我原设计里的 4 个策略性错误

| 我原本的设计 | 评审结论 | 修正后 |
|---|---|---|
| `tts_slot` 用 latest-wins | **错**。TTS 是一条有限流，整个波形按序驱动嘴型；"积压的 TTS 音频没用"不成立 | 有界 FIFO + 与 T7 播放时钟耦合 |
| `render_q` "只保留最新帧" | **错**。等于把 49 帧序列塌缩成每次推理一次姿态瞬移 | 限速排空 + **应用最后弹出的一帧** |
| 关闭只 join T1–T4 | **不完整**。T5 句柄被丢弃，**原理上无法 join**，复现了该 ADR 自己引用的那个 use-after-terminate | 保存句柄 + 取消 + 在拆资源前 join |
| `asr_slot` 用 latest-wins | **错**。静默删掉用户语句，是数据丢失而非陈旧性权衡 | 有界 FIFO（容量 3） |

评审还指出：`_prev_params` 从未被读取、overlap 块是 no-op —— 即**根本不存在跨 chunk 连续性机制**；`AudioRing` 的"覆盖最旧样本"要求写者推进读索引，**破坏 SPSC 不变式**。

### 4.2 ADR 0004：D1 的方向对、机制错

我原本写 `audio[played : played + chunk]`。评审指出这会**跳过**游标与播放位置之间的每一个未推理窗口 —— **正是我同一段里声称"绝不会发生"的句中跳帧**。而且"有界滞后"其实没有常数界：CPU 上按每 chunk 3 秒算，嘴型每播放 1 秒落后约 0.67 秒，30 秒的句子结束时嘴还在演 20 秒前的音频。

修正后的结构：**播放门控推进，而非索引推进**。T3 自持 `inferred` 游标，仅当 `inferred + chunk <= min(played + lookahead, len(audio))` 时才前进一步；另加 max-lag 跳帧（默认 1.5 s）并计数。

评审还找出了另外 4 个 Critical，包括：共享取消标志会连带中止**替代的那次生成**；被取代者的完成回调会抹掉替代者的 `_generating`（可 spawn 出第三条并发线程）；`T7` 在计划里根本不存在却同时是 D1/D5 的前置。

### 4.3 我修正过的自己两处不严谨表述

- ADR 0003 原文写"超出预算 100×+"，改为有依据的 **8–47×**；
- ASR 的注册行是 `pipeline.py:46` 而非 70。

---

## 5. 验证方式与诚实边界

### 5.1 每个修复都有对应的验证

- **契约测试**：先用 RED 测试描述"应该是什么"，再改代码使其转绿。6 个 Layer A 契约测试 + 2 个 Layer B（fp16）。
- **不是断言形状，而是实跑**：C-2 除测试外，直接在 CUDA 上跑了 4 层模型，确认从"抛 dtype 错误"变为返回 `(1, 49, 45) float16`。
- **量测而非假定**：C-1 的 500.3 ms → <50 ms 是实测的。
- **并发测试**：`FrameBuffer` 有并发读写测试**复现**了原来的竞争；`AudioRing` 有 SPSC 冒烟测试断言绝不重排或复制样本。

### 5.2 过程中有 5 处失败断言是**我的测试错了**

包括浮点边界（`0.02 × 50` 得 0.99999 错过帧边界）、遗忘的采样率换算（20 秒样本被按 22050 Hz 解释成 14.5 秒）、在 `finally: cleanup()` 清空缓冲**之后**才断言、以及推入了整整一秒麦克风音频而它在 Speak 门控关闭后**合法地**产生了一次 Listen 步。

每一处我都**改测试的取值或构造，而不是放松策略断言**。

### 5.3 边界（重要）

**全流程 151 个测试都是 hermetic 的**：不碰真麦克风、摄像头、Live2D 模型、Whisper 权重、音频设备。所以：

> **"151 tests passing" 不等于 "应用已经端到端跑通"。**

真实设备上的首次运行**仍未发生**。仓库现状也印证这点：`assets/live2d/` 只有 `.gitkeep`（无 Live2D 模型）、`models/whisper/` 不存在、TTS 后端未安装（`cosyvoice2`/`ChatTTS`/`fishspeech` 均未安装）。

---

## 6. C-7：为什么我选择回退而不是交付

C-7 是"假流式"：`DialogueModel.generate()` 把整段生成跑完后才 `yield from full_text`，即**逐字符**吐出已完成的字符串。首字节延迟等于完整生成时间。

我按标准做法改用 `TextIteratorStreamer`，测试立刻给出反证：3 个 token、间隔 0.25 s，首块却在 **1.14 s 后一次性**到达。于是读已安装的源码 —— transformers **4.32.1**（`environment.yml` 锁定）的 `TextIteratorStreamer` 只定义了 `__init__ / on_finalized_text / __iter__ / __next__`，它继承 `put()` 且**只在 `end()` 时冲刷缓冲**。

我**回退了它**，把实测结论写进代码注释与报告，并把生成参数（`max_new_tokens` / `temperature` / `top_p` / `repetition_penalty`）改成配置驱动。

**解除 C-7 需要**：升级 transformers（会牵动 4.32 锁定的 Qwen/Hubert 栈），或改用另一种流式机制。两者都值得单独评估。

> **（2026-10-12 更新）** C-7 已通过**改道**解决。接手复测发现假流式所在的本地路径是双重死路：4.32.1
> 加载不了 Qwen2.5（`KeyError: 'qwen2'`），且没有 `apply_chat_template`——该路径从未真正运行过。
> 经决策删除本地路径，对话改为 **OpenAI 兼容 + Ollama 双协议 API 客户端**（api_base + model +
> 环境变量密钥），真流式 + 平滑吐字（词/标点聚批 + 200 ms 兜底）+ 静默期可取消 + 错误带外显示
> （不再污染对话历史/朗读）。另经回环实测与真实冒烟：Windows 上 closesocket 不可靠唤醒阻塞 recv
> （0–5 s 不等），读线程以 `select()` 预检可读性（0.25 s），既不依赖唤醒也不依赖读超时（一次读超时
> 会永久毒化 SocketIO）。冒烟另抓到两个 hermetic 盲区缺陷（中文请求体 latin-1 崩、读超时毒化），
> 已修复并回归钉死；真实冒烟（DeepSeek `deepseek-flash`）：首块 0.953 s、5 个平滑 chunk、总 1.047 s。
> 设计与 Oracle 评审记录见
> `docs/adr/0005-dialogue-api-client-and-streaming.md`；提交 `cdf58d6`（客户端 + 31 测试）、
> `ceb1df5`（window 集成 + 3 测试）、`c7e37d3`（冒烟发现的两缺陷修复 + 1 测试），全量 186 passed。
> 原"解除 C-7 的两个选项"均已不需要。**本地 Qwen2.5 也不是必需项**：对话已 API 化，
> 本地模型可经 Ollama 走现有 `provider: ollama` 路径；`transformers` 仍只服务 motion
> 编码器（Hubert/BERT），升级到 >=4.37 仅对"重新加入 in-process HF 本地路径"有意义，
> 属可选项。真正待处理的是 `requirements.txt`（transformers>=4.36.0）与
> `requirements-lock.txt`（4.32.1）的版本声明不一致。

---

## 7. 下一步建议

> **2026-11-10 更新（CI 修复后）**：三类 CI 失败已修复并提交为 `e79ddf2`（未 push）。
> 本地复验：ruff 0.17.4 check/format/AST 全 0；186 passed；CI 导入脚本 PASS；
> lock+pytest 与 lock+basedpyright 的 pip 解析均 exit 0。真实 CI 仍需 push 后确认。
> 最新优先级计划见 §7.3；§7.1/§7.2 作为历史建议保留。

### 7.1 最高价值：真实设备上的首次端到端运行

所有修复都只在 hermetic 层面验证过。下一步应该是补上：

- **`docs/MODEL_TRAINING_ROADMAP.md` 的 Phase 0（非神经基线）** —— RMS 驱动口型 + 眨眼 + 呼吸的规则实现。它让应用在没有神经模型时也能"活着"，并提供一个质量下限来判断神经模型是否真的更好。
- 一次真实设备冒烟运行（麦克风 + 摄像头 + 一个 Live2D 模型 + Ollama）。
- **`SilencePromptGenerator`**：✅ 已接线（`5aac7c4`）：窗口以定时器生成静默提示词并经 `set_silence_prompt()` 下发，"静默时该想什么"的缺口已闭合。

### 7.2 其余已知未修项

| 项 | 状态 |
|---|---|
| `Silence` 状态无驱动力 | ✅ 已修（`664b8b1`）：运动 worker 改为定时 tick，空闲窗口后以**零输入 + `silence` 模式**出步，使引擎真正 always-on；可用 `silence_enabled` 关闭、`silence_interval_seconds` 限速 |
| `_prev_params` 死状态 / overlap 是 no-op | ✅ 已移除（`7d19047`）：原 overlap 是"从队首弹出若干个、再按原序放回"的 no-op，既没跨 chunk 携带音频也没让缓冲变短，却暗示了并不存在的连续性。现场留下 NOTE 说明为何**没有**连续性机制，避免后人把 no-op 重新加回 |
| `basedpyright` 无配置 | ✅ 已配置（`485304c`）：未配置时 **203 errors / 2384 warnings**（几乎全是注解覆盖类噪音）；配置后 **31 errors / 55 warnings**（2026-11-10 实测；14 个 optional-access、5 个 `video_reader.py` generator 返回类型、2 个 `dataset.py` dict 返回类型、2 个 `SequentialLR` 参数，其余为 torch stub / model optionality）。CI 步骤已于 2026-11-26 归零（0 errors）并从 `continue-on-error` 转为硬门禁 |
| CI 未覆盖工作分支 | ✅ 已修（`7d19047`）：改为 `on: push / pull_request` 覆盖所有分支，并新增执行 `pytest` 的 `test` job（2026-11-26 更新：`push` 收窄至 main 以消除 PR 双跑；分支推送由其 pull_request 运行覆盖，未开 PR 的分支推送不再触发 CI） |
| README Phase 8 漂移 | ✅ 已修（`7d19047`）：改为如实描述（HDTF 是人类视频而非角色，仅作管线冒烟测试），并新增"运行加固"一节记录测试套件与"未在真实硬件验证"的边界 |
| `SilencePromptGenerator` | ⏳ **未实现**：`set_silence_prompt()` 只有定义和测试调用，生产代码无调用方；生成器本身（结合感知 + 静默时长 + 情绪延续）仍未建 |
| `BodySkeletonExtractor` | ⏳ **stub**：`src/motion/preprocess/body_skeleton.py` 返回全零 body 参数；YOLOv8 身体路径未实现 |
| **C-7 真流式生成** | ✅ **已解决（改道）**（2026-10-12）：见 §6 与 `docs/adr/0005` |
| **Phase 0 非神经基线** | ❌ **未做**：属新功能而非缺陷（见 §7.1） |
| **真实设备端到端运行** | ❌ **未做**：本环境无法进行（见 §5.3） |
| **本地 Qwen2.5 / transformers** | ❌ **不需要**：对话已 API 化；本地模型可经 Ollama；`transformers` 仅服务 motion 编码器（Hubert/BERT）。升级 >=4.37 仅对重新加入 in-process HF 路径有意义，属 P3 可选项 |
| **requirements 与 lock 不一致** | ⚠️ `requirements.txt` 要求 `transformers>=4.36.0`，lock 固定 `4.32.1`；需对齐声明，不是 Qwen2.5 的需求 |
| **CI 工具版本未固定** | ⚠️ `ruff` / `pytest` / `basedpyright` 均未 pin；未来版本可能让绿色流水线再次变红 |
| **lint/format/AST 覆盖** | ⚠️ 只检查 `src/`；`tests/`、`scripts/`、`demo.py` 未覆盖；AST 的 `read_text()` 缺少显式 UTF-8 |
| **打包/依赖卫生** | ⚠️ `requires-python == 3.11.5` 过严；`opencv-python` 与 `opencv-contrib-python` 同时安装；lock 中 jax/jaxlib/librosa/matplotlib/scipy/numba 等未固定 |
| **whisper-cpp-python 构建** | ⚠️ sdist 在本机需要 MSVC/cpp；Linux CI 有工具链，Windows 开发环境需要预编译 wheel 或降级路径 |
| **PR/合并 + 文档** | ⏳ 用户 gate：分支领先 `origin/main` 49 个提交，含堆叠 training 分支；6 份 docs 未跟踪（见 §8.4） |
| **API key 轮换** | ⏳ 交接文档仍提醒；若未做，先轮换 DeepSeek key |

### 7.3 优化计划（2026-11-10 更新，按优先级）

**P0 — 发布 gate（先做）**

1. Push `e79ddf2`，确认 lint / test / type-check 三 job。当前 `fix/runtime-p0` 领先 `origin/fix/runtime-p0` 1 个提交，领先 `origin/main` 49 个提交。
2. 定 PR/合并范围：49 个提交包含堆叠的 `fix/training-pipeline-issues`；决定 base、是否拆 PR/squash。
3. 决定 6 份未跟踪 docs 的去留（见 §8.4）。
4. 若未做，轮换 DeepSeek API key。

**P1 — 功能 backlog（有代码证据）**

1. `SilencePromptGenerator`：实现"静默时长 + 对话上下文 + 情绪延续 + 微行为建议"的提示词生成器，并接线到 `set_silence_prompt()`；当前生产代码无调用方。
2. `BodySkeletonExtractor`：实现 YOLOv8 身体骨架 → Live2D body 参数；当前 `body_skeleton.py` 是返回零值的 stub。
3. 真实设备 e2e：麦克风 + 摄像头 + Live2D + Whisper/Ollama 的一次完整冒烟；现有 186 个测试是 hermetic 的。
4. `docs/MODEL_TRAINING_ROADMAP.md` Phase 0 非神经基线：RMS 口型 + 眨眼 + 呼吸，作为无模型时的质量下限。

**P2 — 工程质量 / CI（可并行）**

1. basedpyright：31 errors / 55 warnings（2026-11-10 实测）。优先清 optional-access 与 `video_reader.py` generator 返回类型；清零后移除 `continue-on-error`。
2. 固定 CI 工具版本：`ruff==0.17.4`、`pytest==9.1.1`、`basedpyright==1.40.x`；当前都是 unpinned。
3. 扩大 lint/format/AST 覆盖到 `tests/`、`scripts/`、`demo.py`；AST 脚本的 `read_text()` 加 `encoding="utf-8"`。
4. 打包/依赖：
   - `requires-python` 从 `==3.11.5` 改为 `>=3.11,<3.12`（或 `>=3.11`）。
   - 只保留一个 OpenCV provider；当前 `opencv-python==4.10.0.84` 与 mediapipe 拉入的 `opencv-contrib-python==4.11.0.86` 同时提供 `cv2`。
   - 对齐 `requirements.txt`（`transformers>=4.36.0`）与 lock（`4.32.1`）；不要把"升级 transformers"当成 Qwen2.5 的需求。
   - lock 不是完整锁：`jax/jaxlib/librosa/matplotlib/scipy/numba` 等未固定；建议用 `pip-compile`/`uv pip compile` 生成全量锁。
   - `whisper-cpp-python` 的 sdist 需要 MSVC/cpp；Windows 开发环境提供预编译 wheel 或明确降级路径。
5. CI 性能/安全：加 `concurrency` 取消旧 run + pip cache；考虑 Windows test job（覆盖 socket/路径/编码）；GitHub Actions 固定 SHA；`opencode.yml` 的 `@latest` + 评论触发 + secret 需权限审查。

**P3 — 可选 / 暂不做**

1. **in-process HF Qwen2.5：不需要**。当前对话是 API-only；本地模型用 Ollama 即可，无需升级 transformers。只有重新引入 in-process 本地路径时才做，工作量包括恢复代码 + 升级 transformers + 复测 motion encoders。
2. pre-commit hooks（防 lint 回归；CI 已覆盖所有分支，可选）。
3. coverage baseline / test-count gate（可选）。

---

## 8. 附录

### 8.1 全部提交（26）

```
485304c  chore(ci): configure basedpyright and make its step advisory rather than fictional
7d19047  chore: make CI cover the working branch, and remove verified dead code
664b8b1  feat(motion): give the Silence state a driver so the engine is actually always-on
76f4316  fix: complete shutdown and add the degradation path that prevents a permanent freeze
210b09b  feat(app): make generations supersedable and implement interruption
3391bc0  fix(motion): drive Speak state from the playback clock, gated rather than indexed
8dbb3d4  feat(runtime): extract playback into its own worker (T7) with an incremental position
f09a503  fix(perception): synchronise, timestamp and rate-gate the camera->motion handoff
5f53759  fix(app): drain the render queue at the model rate, stop snapping to neutral
9ca582a  chore(dialogue): make generation params configurable, document the C-7 blocker
81d95a6  docs: correct claims that did not match the code, and declare missing deps
3ac8b8c  perf(motion): stop encoding the user waveform twice per diffusion step
24144b0  fix(app): join the generation thread before tearing down PortAudio
35e346c  fix(motion): give inference its own owner thread
9cba58c  fix(audio): run transcription on a dedicated worker, off the audio path
2a9bee2  fix(audio): move microphone fan-out off the PortAudio real-time thread
3e7ae79  fix(model): make the fp16 path work end-to-end
6e7f6ec  fix(motion): wire PerformanceEngine into the inference path
529c279  fix(motion): let TTS audio drive inference so Speak state is always-on
4687fe3  fix(tts): resample TTS audio to the motion pipeline's sample rate
f2e3927  fix(motion): deliver camera frames to the model as channel-first
9c5446f  fix(train): construct AMP scaler before the resume block
da71e07  feat(runtime): add owner-thread concurrency primitives with tests
ac8af72  test(runtime): add P0 runtime contract tests for C-3/C-4/C-5/C-6/C-8
306ff29  fix(app): import QOpenGLWidget from PySide6.QtOpenGLWidgets for Qt6
12e840e  test(runtime): add hermetic pytest harness, config, and src import smoke test
```

> 2026-11-10 追加：`e79ddf2` — CI 修复（ruff check/format 全量修复 + `protobuf==4.25.9`）。
> 上面的 26 个提交是 §6 运行修复的提交列表，不含后续 C-7 与 CI 提交。

### 8.2 新增与修改的源文件

**新增**

```
src/runtime/__init__.py          导出并发原语
src/runtime/concurrency.py       AudioRing / Mailbox / BoundedDropOldest / FrameBuffer
src/runtime/playback.py          PlaybackWorker (T7)
```

**修改**

```
src/app/window.py                线程生命周期、逐代 token、中断、关闭协议
src/app/live2d_widget.py         排空策略、空闲超时、丢弃计数
src/audio/capture.py             回调仅入队 + 排空 worker + 断路器
src/audio/pipeline.py            ASR worker + asr_q + 语音起始信号
src/motion/inference.py          运动 worker、视觉交接、Speak 时钟、降级
src/motion/model.py              to_half 递归化、TimestepEmbedding、_resample_time dtype
src/motion/training/train.py     scaler 顺序修正
src/perception/camera.py         源头降频 + 忙等修正
src/tts/engine.py                cleanup + _silence 采样率修正
```

### 8.3 测试文件（17）

```
tests/conftest.py                        hermetic 环境守卫 + qapp/config fixture
tests/test_import_smoke.py               37 个模块的导入冒烟
tests/test_runtime_contracts.py          C-3/C-4/C-5/C-6/C-8 契约测试
tests/test_motion_model_dtypes.py        Layer B：fp16 参数覆盖 + 半精度前向
tests/test_concurrency.py                AudioRing / Mailbox / BoundedDropOldest
tests/test_audio_capture_threading.py    实时回调仅入队 + 排空 worker
tests/test_audio_pipeline_threading.py   ASR 线程 + 连续缓冲
tests/test_motion_worker_threading.py    运动 worker + 断路器
tests/test_audio_encoder_reuse.py        H-8：编码器调用计数
tests/test_visual_pipeline_threading.py  FrameBuffer + 并发读写 + 最新 5 帧 + 降频
tests/test_live2d_widget_drain.py        排空策略 / 末帧 / 空闲超时 / 丢弃计数
tests/test_playback_worker.py            T7：分块写入 / 递增位置 / 中断 / 淡出 / 失败
tests/test_speak_state_clock.py          S2.4：游标门控 / max-lag / 单位 / mic 置零
tests/test_generation_lifecycle.py       逐代 token + 代际门控
tests/test_interruption.py               语音起始 + 中断端到端
tests/test_window_shutdown.py            关闭顺序 + 幂等
tests/test_shutdown_and_degradation.py   TTS 清理 + T1 断路器 + T7 降级
```

### 8.4 文档（2026-11-26 起已随 PR #1 入库；下文保留当时记录）

```
docs/PROJECT_REVIEW_REPORT.md                              复盘审查报告
docs/RUNTIME_FIX_REPORT.md                                 运行修复报告（含本优化计划）
docs/MODEL_TRAINING_ROADMAP.md                             训练路线图（原本就未跟踪）
docs/adr/0003-runtime-threading-and-realtime-architecture.md  Stage 1 设计（含评审记录）
docs/adr/0004-speak-state-and-render-synchronisation.md       Stage 2 设计（含评审记录与 Q1–Q6 裁定）
docs/adr/0005-dialogue-api-client-and-streaming.md          C-7 API 客户端与流式设计（含 Oracle 评审）
```

这六份是本次工作的设计依据与结论记录，**建议随 PR 一起提交以便审阅**。

---

## 9. 一句话结论

训练侧的问题此前已被 `docs/TRAINING_PIPELINE_REVIEW.md` 覆盖得很彻底，但那份报告止步于 `train.py` 的边界 —— 它没有审查各组件是如何被连接起来的。本次的 8 个 Critical 中有 6 个**只在"组件被真正连接运行时"才会暴露**（C-1 线程、C-4 视觉 shape、C-5 采样率、C-6 Speak 触发、C-8 未接线、C-9 导入）。这是"训练管线可跑 ≠ 系统可跑"的具体体现。

**下一个最有价值的动作：先 push `e79ddf2` 确认三个 CI job，然后让它在真实设备上第一次真正跑起来。本地 Qwen2.5 不是这条路上的前置条件。**
