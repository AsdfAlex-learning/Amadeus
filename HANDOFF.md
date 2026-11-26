# Handoff — Amadeus C-7 改道：Dialogue API 化改造（2026-10-12，终态）

**Repo:** `D:\Whoami\Amadeus`（GitHub: `AsdfAlex-learning/Amadeus`）
**Branch:** `fix/runtime-p0` — **已推送**（origin/fix/runtime-p0，共 48 commits，HEAD `c7e37d3`）
**PR 入口:** https://github.com/AsdfAlex-learning/Amadeus/pull/new/fix/runtime-p0
**下一会话焦点:** 确认 CI 三 job 结果 → 与用户定 PR/合并方案（用户流程门槛）→ backlog。

> 注：本文件为工作区副本。早前两份 temp 副本（含 `HANDOFF_amadeus_runtime-p0.md`
> 旧 handoff）已被系统清理；旧 handoff 的环境铁律/身份修复史已吸收进 §2/§3。

---

## 1. 工作记录在哪（读这些，不要重复本文件）

| 产物 | 内容 |
|---|---|
| `docs/adr/0005-dialogue-api-client-and-streaming.md`（untracked） | 本次设计 D1–D8 + Oracle 评审 10 条裁决（含其自身 1 处错误）+ 实现期实测补记 |
| `docs/RUNTIME_FIX_REPORT.md` §6（untracked，已更新） | C-7 从"假流式"到"改道 API 化"的完整叙事、冒烟数字 |
| `docs/PROJECT_REVIEW_REPORT.md`（untracked，C-7 行已更新） | 原始缺陷清单 |
| 3 个新 commit 的 message | `cdf58d6`（客户端）、`ceb1df5`（window 集成）、`c7e37d3`（冒烟发现 2 缺陷）各自的 what/why/诚实记录 |
| `docs/RUNTIME_FIX_REPORT.md` §8.1 | 此前 26 个 commit 的清单（注意：其中 hash 为身份重写前旧值，以 subject 为准） |

关键源码：`src/dialogue/model.py`、`src/app/window.py`、`config/default.yaml`（dialogue 新 schema）、
`tests/test_dialogue_api_streaming.py`。

## 2. 环境（硬事实）

- Python: `C:\Users\Sama608\.conda\envs\amadeus\python.exe`（勿动、勿重建、勿混环境；
  手工搭建 torch 2.6.0+cu124，与 environment.yml 的 2.5.1 不一致——别重建）。
- Shell 是 **PowerShell 5.1**（`VAR=1 cmd` 前缀无效，用 `$env:VAR='1'`）。
  git 命令前缀 `$env:GIT_MASTER='1'`；CI 式调用再加 `GIT_TERMINAL_PROMPT='0'`、`GIT_EDITOR=':'`、
  `GIT_PAGER='cat'`、`CI=true`。
- **网络**：github.com 直连 443 不通；**本机 `127.0.0.1:7897` 代理可用**（本次推送即经它完成，
  仅用 `git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=...` 会话级参数，**未写入持久配置**）。
  后续 push/gh 操作同样需要 `-c` 参数或先做持久配置（需用户点头）。
- **Windows socket 铁律（3 轮探针实测）**：`resp.close()` 对并发读者死锁；`sock.shutdown()` 不唤醒
  阻塞 recv；`sock.close()` 不可靠唤醒（0–5s 不等）；一次读超时**永久毒化** SocketIO
  （`_timeout_occurred`）。读线程用 `select()` 预检可读性——**不要改回超时轮询**。
- transformers 4.32.1 加载不了 Qwen2（`KeyError: 'qwen2'`）且无 `apply_chat_template`——本地
  LLM 路径双重死路（已删）。Qwen2.5-3B 权重 ~5.9GB 仍在 HF 缓存，升级 transformers 可解锁。
  ruff 0.17.3 已装入 amadeus 环境。
- 终态基线：**186 passed** / basedpyright **31 errors** / ruff check **18 errors**（后两者全预存、
  零新增；清零 basedpyright 后移除 CI 的 continue-on-error）。
- DeepSeek API：`/models` 返回 `deepseek-flash`（即 v4.1 flash）与 `deepseek-v4-pro`。
  冒烟脚本留存：`Temp\opencode\smoke_deepseek.py`（若被清理可按 ADR 0005 测试计划重建；
  key 只经 `AMADEUS_API_KEY` 环境变量注入）。
- **⚠️ 密钥事件**：用户曾将 DeepSeek key 明文贴进对话。已扫描确认**推送内容零泄露**（见 §3），
  并已建议用户在 DeepSeek 控制台轮换。日常使用 `setx AMADEUS_API_KEY`，config 只存变量名。

## 3. Git 状态

- 已推送：`fix/runtime-p0` → `origin/fix/runtime-p0`（48 commits，含堆叠的
  `fix/training-pipeline-issues` 分支工作）。本会话 3 个新 commit 见 §1。
- **推送前密钥扫描（已做，全干净）**：commit messages / `origin/main..HEAD` diff 新增行 /
  HEAD 全文件树三面扫描 key 全文与分段片段 + 通用 `sk-*` 模式，零匹配。后续如再引用该 key，
  只用前缀指代，勿复述全文/片段。
- CI（lint / type-check advisory / test）随推送首跑本分支——**下一会话先确认三 job 结果**。
- **6 份文档 + 本 handoff 均为 untracked**（用户既定"报告先不提交"）：两报告、三 ADR、
  路线图、HANDOFF.md。**未经用户明示不要 commit。**
- 身份：作者/提交者均为个人全局身份（`AsdfAlex-learning`；本地 bot 覆盖早已清除，历史已重写
  并 force-push，勿重做）。

## 4. 已完成（摘要，细节在 ADR/报告/commit）

C-7 **改道解决**：删除死的本地 transformers 路径，对话改为 OpenAI 兼容 + Ollama 双协议 API 客户端
（api_base + model + 环境变量密钥；配置默认留空，Ollama 非默认）。真流式 + 平滑吐字（词/标点
聚批 + 200ms 截止时间兜底）+ 首 token 等待/流静默期可取消 + 错误带外（不进历史不朗读）。
Oracle（Mimo v2.6 pro）评审 2C/2H/4M/2L 全采纳并独立复验（其 F-2 修法自身有 stub 签名错误，
已记录）。真实冒烟通过（DeepSeek `deepseek-flash`：首块 0.953s / 5 个平滑 chunk / 总 1.047s），
冒烟另抓 2 个 hermetic 盲区缺陷（中文请求体 latin-1 崩、读超时毒化）已修复钉死。

## 5. 剩余工作

1. **CI 结果确认**（下一会话首要动作）：三 job 绿则进入 2；lint/test 若挂则修（本分支新增代码
   本地 ruff/pytest 干净，理论上绿；type-check advisory 不阻塞）。
2. **PR/合并决策**（用户流程门槛："允许必要的架构改造，但是需要新建分支，确认完成之后再考虑合并"）：
   与用户确认是否开 PR、PR 范围（48 commits 含堆叠 training 分支）、是否顺带提交文档。
3. Backlog（范围外）：
   - `SilencePromptGenerator`——`set_silence_prompt()` 钩子存在但**至今无调用方**，接线也在待办；
     动工前先 brainstorm。
   - Phase 0 非神经基线（RMS 口型+眨眼+呼吸）——先 brainstorm。
   - basedpyright 31 errors 清零 → 移除 CI continue-on-error。
   - 真实设备端到端冒烟（mic/camera/Live2D 模型/Whisper 权重——`assets/live2d/` 只有 .gitkeep，
     `models/whisper/` 不存在；"186 tests ≠ app runs" 的诚实边界仍在）。
   - transformers 升级（可解锁本地 Qwen2.5；需回归 Hubert/BERT，测试是 stub 覆盖不到真实路径）。
4. **密钥轮换**再提醒用户（若未做）。

## 6. 约束（用户原话与既定流程）

- "我使用了conda管理环境，不要混用其他环境"——勿重建 amadeus 环境。
- "报告先不提交，文档漂移待统一修正"——文档保持 untracked。
- "允许必要的架构改造，但是需要新建分支，确认完成之后再考虑合并"。
- "只需要代码与测试提交"。
- Oracle 用 Mimo v2.6 pro（Opencode Go 限额，不用 gpt-5.6-luna）。
- 有决策点先通俗易懂讲解，再用 question 工具提问，直到决策完成再继续（用户明确要求）。
- 工作流先例（继续遵守）：自设计 → Oracle 评审 → 核实断言后修订 → 分段 RED→GREEN →
  每段一 commit；测试断言错则改构造不放松策略；不交付不可验证的半成品。
- **未经用户明示不 push/merge/reset**（本会话 push 是用户明示的例外）。

## 7. Suggested skills

- `verification-before-completion` — 声称完成前跑全量并展示输出（惯例：186 passed / 31E / 18E）。
- `finishing-a-development-branch` — 用户谈 PR/合并时用（下一步大概率触发）。
- `receiving-code-review` — 任何评审意见先核实再采纳（Oracle 自身也出过错）。
- `systematic-debugging` — 修 bug 前先钉死机制（socket 收尾探针 3 轮才定案）。
- `brainstorming` — SilencePromptGenerator / Phase 0 动工前强制。
- `git-guardrails-claude-code` — 若需 push/merge 保护钩子。
