# ADR 0005: Dialogue Model over OpenAI-Compatible / Ollama APIs with Smooth Streaming

## Status

Accepted — revised after Oracle review (2026-10-12, Oracle: Mimo v2.6 pro)。
标注 (rev) 的段落为评审后修订。

## Context

C-7（见 `docs/PROJECT_REVIEW_REPORT.md`、`docs/RUNTIME_FIX_REPORT.md` §6）的原表述是"假流式生成"。
本轮接手后取证（2026-10-12，均为实测）：

1. **本地路径双重死路**：锁定的 `transformers 4.32.1` 加载不了配置中的 Qwen2.5
   （`KeyError: 'qwen2'`，Qwen2 架构 4.37 才支持）；且 4.32.1 **没有 `apply_chat_template`**
   （4.34 引入）——即使模型能加载，`generate()` 也会崩。本地 transformers 路径在当前栈上从未跑通过，
   程序实际一直走 Ollama 兜底。
2. **Ollama 兜底同样是死的**：`_generate_api` 硬编码 `http://localhost:11434/api/chat` 与
   `qwen2.5:{size}`，且开发机 Ollama 未运行（11434 不通）。
3. **假流式机制已溯源**（安装版源码 `transformers/generation/streamers.py`）：
   `TextStreamer.put()` 每次都调 `on_finalized_text`，但冲刷量受启发式控制——换行→全冲；
   CJK 末字符→逐字冲；**其余只冲到最后一个空格**（`rfind(" ") == -1` 时 emit 空串），
   直到 `end()` 才一次性冲出。即"1.14 s 一次性到达"的机制。
4. **Qwen2.5-3B-Instruct 权重（~5.9 GB）已完整缓存在本地**，但被 transformers 版本卡死。
5. **用户决策（2026-10-12）**：放弃本地模型路径，改为 **api_base + model + 环境变量密钥** 接外部模型；
   **OpenAI 兼容 + Ollama 兼容双轨**；配置默认**留空**（不预设 Ollama）；**平滑模式**吐字
   （按词/标点聚批 + 200 ms 强制兜底）；验证 = hermetic mock 测试 + 真实冒烟；
   transformers 升级不在本次范围（Hubert/BERT 回归无 hermetic 覆盖，风险敞口不闭合）。

## Decision

### D1 — 删除本地 transformers 路径

删除 `load()` 的模型加载逻辑、`_detect_device()`、`_resolve_model_id()`、transformers/torch 依赖。
`dialogue` 模块只做 API 客户端。`generate()` 对外契约不变（消费 `Generator[str, None, None]`），
window.py 的调用点形态保持 `for chunk in ...generate(messages)`。

注意：`model.py:11-12` 现以 `[]` 强读 `model_type`/`model_path`，删键时构造函数必须同步改为 `.get()`。

### D2 — 双协议 HTTP 客户端（`provider: openai | ollama`）

- **openai**：`POST {api_base}/chat/completions`，SSE 流，取 `choices[0].delta.content`。
  覆盖 DeepSeek/Kimi/智谱/OpenRouter/vLLM 等一切 OpenAI 兼容服务。
- **ollama**：`POST {api_base}/api/chat`，NDJSON 流，取 `message.content`，`done: true` 结束。
- `api_base` 按 OpenAI SDK 惯例**原样拼接**（rstrip `/` 后追加固定路径）：
  openai 示例 `https://api.deepseek.com/v1`、`https://openrouter.ai/api/v1`；
  ollama 示例 `http://localhost:11434`。
- 仅用标准库（不引入 `requests`）。(rev) **使用 `http.client.HTTPConnection` 并持有
  `conn.sock`**——评审 F-1 实测（见 Review outcome）：`urllib` 响应对象的 `close()` 在读线程阻塞时
  自身会被 `BufferedReader` 内部锁卡住且不唤醒读线程，只有**底层 socket close** 能打断阻塞读。
- 鉴权：`api_key_env` 指向的环境变量非空时发 `Authorization: Bearer ...`；空则不发
  （本地 Ollama 无需鉴权）。**密钥本体永不进仓库/配置文件。**

**(rev) 流解析规约**（评审 F-8）：

- SSE：行按 UTF-8 解码后 `rstrip("\r\n")`；空行结束一个事件、多行 `data:` 以 `\n` 拼接；
  `:` 开头为注释/心跳，跳过；`data: [DONE]` 结束；坏行记日志跳过；`error` 字段 → 失败；
  `choices: []` 或 `delta.content` 缺失/为 null → 跳过；**EOF 未见 `[DONE]`** → 正常收尾（冲残余）。
- NDJSON：逐行 JSON；坏行记日志跳过；`error` 字段 → 失败；`done: true` 结束；
  EOF 无 `done` → 正常收尾。
- HTTP 层按行分帧，UTF-8 多字节不跨行（延续字节恒非 0x0A），无需 token 级 `\ufffd` 挂起。

### D3 — 平滑模式聚批（(rev) 最后边界符 + 200 ms 截止时间兜底）

网络读线程（`amadeus-dialogue-stream`，daemon）解析协议并把增量推入队列；`generate()` 消费并聚批：

- **(rev)** 收到增量 → 入缓冲；在缓冲内找**最后一个边界符**，冲出该边界符（含）之前的部分，**余尾保留**
  （原稿"只看缓冲末尾字符"会导致英文纯词增量永远不触发边界、退化为 200 ms 定长切块——评审 F-5）。
  边界符集：ASCII 空白 + `.,!?;:` 与 `，。！？；：、…` 及收尾引号括号（详见实现，注意原稿有重复逗号笔误）。
- **(rev) 兜底语义为截止时间锚定**：`deadline = last_flush + 200 ms`，`队列.get` 超时取
  `max(小超时, deadline - now)`；**自上次冲出起满 200 ms 必冲**当前缓冲（哪怕无边界符）。
  原稿"队列空转达 200 ms"若按到达即重置计时实现，190 ms/块的滴流会**饿死**兜底（评审 F-6）。
- 流结束 → 冲出残余缓冲。
- **(rev) 消息进入文本增量、缓冲、yield 的通路不吞取消信号**——见 D4。

### D4 — (rev) 取消与线程生命周期（推翻原稿机制，评审 F-1/F-2）

原稿"`response.close()` 解开读线程阻塞"被回环实测**证伪**（close() 自身阻塞且不唤醒读线程）。
修订机制：

- 读线程持有 `HTTPConnection` 与响应；**取消/收尾走 `conn.sock.close()`（尽力而为）**，
  socket 关闭后再 `join` 读线程（~2 s 超时，daemon 兜底），随后 `conn.close()`。
  **不在读线程阻塞期间调用响应对象的 `close()`。**
  **（实现期实测修正）** `sock.close()` 立即返回但**不可靠唤醒**阻塞 recv——实测 0–5 s 不等、
  甚至仅在 socket 超时后才醒（探针 v3）；故读线程**不依赖唤醒**：以 `select()` 每 0.25 s
  预检可读性并检查 stop，`readline` 只在数据就绪时调用。亦**不可**用"读超时+重试"轮询：
  一次读超时永久毒化 makefile 响应（`SocketIO._timeout_occurred` →
  "cannot read from timed out object"）。消费循环的队列短轮询（50 ms）不变。
- **(rev) 流静默期可取消**（原稿只在 chunk 边界查取消，首 token 延迟/中途停顿期间打断与关停失灵）：
  `generate(messages, cancel_event: threading.Event | None = None)`；消费循环以短超时轮询队列并
  检查 `cancel_event`；置位 → socket close + 读线程收尾 + 生成器返回（残余缓冲丢弃）。
  window 为**每个 generation_id 建独立 Event**（supersede 安全），`_cancel_generation` 置位。
- **测试构造改动（诚实记录）**：window 调用变为 `generate(messages, cancel_event=...)`，
  既有 monkeypatch 单参 stub（9 处）需机械改为 `lambda _m, **_: ...`；**断言全部不动**。
  （Oracle 建议"kwarg 默认 None 即保契约零改动"未计及 stub 签名——此为其评审中一处自身错误。）
- 读线程 try/except 全包，异常以标记入队，绝不裸抛。符合 ADR 0003 的 T5 归属（读线程是 T5
  生成线程的内部助手，join 在生成器收尾内完成）。

### D5 — 配置 schema（`dialogue` 段）

```yaml
dialogue:
  # 留空即禁用对话模型；load() 返回 False，generate() 抛 DialogueUnavailableError
  provider: ""                 # "openai" | "ollama"
  api_base: ""                 # 例：https://api.deepseek.com/v1 或 http://localhost:11434
  model: ""                    # 例：deepseek-chat 或 qwen2.5:3b
  api_key_env: "AMADEUS_API_KEY"   # 密钥所在环境变量的【名字】，勿在此写密钥
  request_timeout_seconds: 60
  max_context_tokens: 4096     # (rev) 历史截断已存在（ConversationContext._trim），此处为预算上限
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  repetition_penalty: 1.1      # 仅 ollama 协议（请求体映射见 D7）
  persona_name: "Amadeus"
  persona_file: ""             # (rev) persona.py 在读，原稿漏列
  persona: |
    ...
```

删除键：`model_type`、`model_path`、`model_size`、`quantization`。
`persona` 系键归 `Persona` 类（`src/dialogue/persona.py`），不动。

**(rev) `max_context_tokens` 表述更正**（评审 F-7）：历史截断**并非**"暂不实现"——
`ConversationContext._trim()` 已按预算裁剪并注入摘要。残留缺口（记录为跟进项，不在本次实现）：
token 估算粗糙（`len(text)//2`，中文易低估）、未为 `max_new_tokens` 预留输出预算、
persona system prompt 不计入预算。

### D6 — `load()` / `is_loaded` = 配置校验

`load()` 校验 `provider ∈ {openai, ollama}` 且 `api_base`、`model` 非空，返回 bool（不再加载权重）。
`is_loaded` 沿用（无外部消费者，全仓核实）。window 提示语从"请安装 Ollama"改为通用配置指引。

### D7 — (rev) 生成参数映射

- openai 请求体：`model`、`messages`、`stream: true`、`max_tokens`、`temperature`、`top_p`
  （**不发** `repetition_penalty`——非 OpenAI 标准字段，严格实现拒未知字段 400；此决策经评审维持）。
- ollama 请求体：`model`、`messages`、`stream: true`、
  `options: {num_predict, temperature, top_p, repeat_penalty}`。
  **(rev) 修正**：Ollama 官方参数名为 `repeat_penalty`（docs.ollama.com/modelfile 实证），
  原稿误写 `repetition_penalty`（HF 命名），Ollama 对未知键**静默忽略**，错名会使配置悄悄失效。
  配置键名保留 `repetition_penalty`，仅请求体映射改名。

### D8 — (rev) 错误语义：带外化，不进历史、不进 TTS

原稿"yield 友好提示文本"经评审（F-4）确认会污染：window.py:231 无条件
`add_assistant_message(full_response)` 且随后送 TTS——错误提示会进对话历史、被朗读，
未配置态每轮反复污染。修订：

- 新增 `DialogueUnavailableError(RuntimeError)`；网络/协议/未配置错误**抛出**而非 yield。
- 错误前已 yield 的部分文本保留在 `full_response`（既有"截断轮次保留"政策：历史留、TTS 仍按既有
  finally 逻辑处理）。
- window 在 generic `except Exception` 前捕获 `DialogueUnavailableError` → **仅显示**其消息
  （走现有 `response_chunk` 显示通道，不进 `full_response`）。既有 generic except 保持兜底。
- **(rev) F-9**：空 `full_response` 不 `add_assistant_message`（取消发生在首 chunk 前的空轮次
  不得进历史——空 content 消息部分严格端拒收）。

## Alternatives considered

- **升级 transformers**（可顺带解锁缓存 Qwen2.5）：被否——Hubert/BERT 真实回归无 hermetic 覆盖，
  依赖连带变动风险敞口不闭合；且用户明确要走 API 路线。留作独立待办。
- **自定义 TextStreamer 子类**（原 C-7 计划方案 A）：本地路径删除后失去宿主，作废。
- **引入 requests**：作废——stdlib 足够，且可通过注入连接缝合点 mock。
- **(rev) `response.close()` 取消**：被实测证伪（F-1），作废。
- **(rev) 错误文本内联 yield**：污染历史/TTS（F-4），作废。

## Consequences

- 对话子系统在任何 OpenAI 兼容 / Ollama 服务上即配即用；真流式 + 平滑吐字；取消/关停在
  首 token 等待期与流静默期均有效。
- 本地 GPU 生成移除；将来若复活（升级 transformers 或 llama.cpp 后端），可在 `generate()`
  契约后无损挂回。
- OpenAI 协议下无 `repetition_penalty`；`max_context_tokens` 预算的残留缺口见 D5 (rev)。
- 既有 window 层 monkeypatch 测试 9 处 stub 签名机械调整（断言不动）。

## Non-goals

- 为 `max_new_tokens` 预留输出预算 / 改进 token 估算（跟进项）；按句 TTS 流式管线（此前已裁定
  范围外）；重试/多端点故障转移；密钥管理 UI；Ollama `truncate: true` 兜底（跟进项）。

## Test plan（hermetic，mock 连接缝合点；一处真实回环）

`tests/test_dialogue_api_streaming.py`（+ window 层补充）：

1. **openai 协议解析**：SSE 多增量拼合；空 `delta`/`content: null` 跳过；`[DONE]` 终止；
   **(rev)** CRLF 行尾、多行 `data:` 拼接、`:` 心跳行、`choices: []` 终块、坏行容错、
   EOF 无 `[DONE]` 收尾、流中 `error` 字段。
2. **ollama 协议解析**：NDJSON `message.content` 拼合、`done` 终止、`error` 对象、坏行容错、
   EOF 无 `done` 收尾。
3. **平滑聚批 (rev)**：按**最后边界符**冲出（词增量也要触发）；截止时间兜底；
   **滴流反饿死**（每 50 ms 无边界小增量，断言 ≤200 ms 必有 yield）。
4. **交错性（C-7 反证）**：首个 chunk 在后续增量到达前已 yield（事件同步，不依赖真实时钟）。
5. **取消 (rev)**：①逻辑层——流静默期置 `cancel_event`，生成器有限时间退出、残余不吐、读线程退出；
   ②**真实回环 socket**——mock `close()` 无法暴露锁阻塞（F-1），回环验证 socket-close 解阻塞与
   线程回收（回环不违反 hermetic）。
6. **错误 (rev)**：连接失败/读超时/流中 error → **抛 `DialogueUnavailableError`**（不 yield、不进
   历史）；部分文本已吐时历史保留截断轮次；window 捕获后仅显示、不入历史、不送 TTS。
7. **未配置**：`load()` False；`generate()` 抛 `DialogueUnavailableError`（含配置指引）；
   **(rev)** 连续两轮不累积指引文本（历史零污染）。
8. **参数透传 (rev)**：openai 体含 `max_tokens/temperature/top_p` 且**无** `repetition_penalty`；
   ollama 体 `options` 含 **`repeat_penalty`**、`num_predict`。
9. **鉴权**：环境变量非空 → `Authorization: Bearer`；空 → 无该头。
10. **URL 拼接**：api_base 尾斜杠规范化。
11. **(rev) F-9**：空轮次（首 chunk 前取消）不进历史。

回归：151 既有测试全绿（9 处 stub 签名机械调整后）；import 冒烟；CI ruff/basedpyright 不回退。

## Staged plan

1. Seg 1：schema + 删本地路径 + `DialogueUnavailableError` + window（提示语/带外错误/F-9/取消事件）
   + 核心测试批次（RED→GREEN）
2. Seg 2：openai/ollama 流式 + 平滑聚批 + 取消机制 + 协议边界测试（RED→GREEN）
3. Seg 3：回环取消测试 + 滴流反饿死 + 遗漏边界补齐
4. Seg 4：文档同步（`RUNTIME_FIX_REPORT` §6/§7.2 等，保持 untracked 不提交）
5. 真实冒烟：用户提供 base_url + model（key 经 `AMADEUS_API_KEY` 环境变量），实测首块延迟

## Review outcome (Oracle, Mimo v2.6 pro, 2026-10-12)

评审产出 2 Critical / 2 High / 4 Medium / 2 Low。逐条经**独立复验**后采纳：

| # | 级别 | 结论 | 复验方式 | 裁决 |
|---|---|---|---|---|
| F-1 | Critical | `resp.close()` 不解阻塞且自身死锁于 BufferedReader 锁；仅 `sock.close()` 有效 | 本人重跑 Oracle 留存的回环探针（`Temp\opencode\probe_close_unblock.py`）：close 模式读线程 8 s 后才醒且 close 阻塞 7.5 s；shutdown 模式 4 s 仍阻塞；sock_close 数秒内 WinError 10058 | **采纳**，D4 重写 |
| F-2 | Critical | 流静默期（首 token 等待/中途停顿）取消失灵 | 代码走查 window.py:220-221（逐 chunk 才查取消）+ D3 原稿措辞，逻辑成立 | **采纳**；但 Oracle 建议的修法自身有误（kwarg 会打破 9 处单参 stub），修正如 D4 (rev) |
| F-3 | High | ollama 参数名应为 `repeat_penalty` | docs.ollama.com/modelfile 实证（repeat_penalty / num_predict） | **采纳** |
| F-4 | High | 错误/指引文本进历史+TTS | 代码走查 window.py:224-238，成立 | **采纳**，D8 带外化 |
| F-5 | Medium | 边界规则只看末尾字符，英文退化 200 ms 切块；边界符笔误 | 逻辑复核成立 | **采纳** |
| F-6 | Medium | 兜底语义二义（滴流饿死风险） | 逻辑复核成立 | **采纳** |
| F-7 | Medium | "历史截断暂不实现"不实 | 走查 context.py `_trim()` 确认已实现 | **采纳**，更正表述 |
| F-8 | Medium | SSE/NDJSON 边界未列全 | 协议常识 + 实现核对 | **采纳** |
| F-9 | Low | 空轮次进历史 | 走查 window.py:231 | **采纳** |
| F-10 | Low | schema 漏 `persona_file` | persona.py:24 确认在读 | **采纳** |

评审方自身错误 1 处：F-2 修法声称"kwarg 默认 None 保契约零改动"，未计及 monkeypatch stub 签名
（其事实表第 6 项自己列出该契约）。按院内惯例，stub 签名属**测试构造**，机械调整、断言不动，
并在此诚实记录。

### 实现期补充（2026-10-12，真实冒烟发现 2 个 hermetic 盲区缺陷，均已修复并回归钉死）

1. **中文请求体崩**：`http.client` 对 str 请求体按 latin-1 编码，persona 系统提示词（中文）直接
   `UnicodeEncodeError`。hermetic 测试全 ASCII + mock 绕过真实编码层，故未暴露。修复：请求体
   预编码 UTF-8 bytes。回归：中文消息走真实 `conn.request` 的回环测试 + body 类型单测。
2. **读超时毒化**："读超时+重试"式轮询不可行——一次读超时永久毒化 makefile 响应
   （`SocketIO._timeout_occurred`）；且 closesocket 唤醒不可靠（探针 v3：`sock.close()` 返回后
   读线程 5 s 才靠 socket 超时醒）。终稿：`select()` 可读性预检（见 D4 修正）。

**真实冒烟**（DeepSeek `deepseek-flash`，经真实 `DialogueModel.generate()` 全路径）：
首块延迟 **0.953 s**、5 个平滑聚批 chunk、总 1.047 s。
