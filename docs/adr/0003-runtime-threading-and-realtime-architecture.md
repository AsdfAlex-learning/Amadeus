# ADR 0003: Runtime Threading & Real-Time Audio Architecture

> **Status**: Proposed — revised after independent review
> **Date**: 2026-09-15
> **Branch**: `fix/runtime-p0`
> **Related**: `docs/PROJECT_REVIEW_REPORT.md` §C-1, §C-5, §C-6, §H-1, §H-2, §H-6, §H-8
> **Review**: independent critique (Oracle, 2026-09-15) — see "Review history"

## Status

Accepted (implemented; merged via PR #1 on 2026-11-26)

## Review history

This ADR was submitted to an independent reviewer before any implementation. The
review verified every factual claim about the existing code (**all 7 confirmed**,
with four imprecisions) and found **12 flaws in the proposed design**, four of
which were policy errors that would have shipped new bugs:

| # | Reviewed proposal | Correction now in this ADR |
|---|---|---|
| 1 | `tts_slot` as single-slot **latest-wins** | **Wrong.** TTS is a finite stream whose whole content matters for lip-sync. Now a bounded FIFO with a stream id, consumed against the **playback clock**, replaced only on explicit interruption. See "TTS stream". |
| 2 | `render_q` drains and applies **only the newest** frame | **Wrong.** `_emit_params` emits ~49 frames per chunk; newest-only collapses a sequence into a per-inference pose teleport. Now a bounded queue with **rate-matched sequential drain + hold-last-params**. See "Render queue". |
| 3 | Shutdown joins T1–T4 | **Incomplete.** T5's handle is discarded at `window.py:156`, so it *cannot* be joined — reproducing the exact `_pyaudio` use-after-terminate this ADR cites as motivation. Now joined before PyAudio teardown, with an escalation policy. See "Shutdown". |
| 4 | `asr_slot` as **latest-wins** | **Wrong.** Transcripts are not interchangeable; latest-wins silently deletes a user utterance. Now a bounded FIFO (cap 3). See "ASR queue". |

Also adopted from the review: the **staged plan** (§Staged plan) — land the
minimal enqueue-only fix first, because it removes the most severe defect class
with a tiny diff and is strictly compatible with the full design.

## Context

The system is documented as an **always-on** performance engine: the character
reacts while the user speaks, while it speaks, and while idle. The runtime is a
PySide6 application that wires six subsystems together in
`src/app/window.py:104-134`.

The current wiring cannot satisfy that goal, for two independent reasons.

### 1. Heavy compute runs on the PortAudio real-time thread

`MicrophoneCapture` opens the input stream in **callback mode**
(`src/audio/capture.py:31-39`). Its `_audio_callback` (`capture.py:61-71`) is
invoked by PortAudio's real-time thread and fans out **synchronously** to every
registered callback:

```python
for cb in self._callbacks:
    cb(audio_data)          # capture.py:66-70 — inline, no queue
```

Two heavy consumers are on that path:

| Consumer | Registered at | Work performed inside the callback |
|---|---|---|
| `motion_model.process_user_audio` | `window.py:124` → `pipeline.py:68-70` | Hubert encode + 4-step DDIM forward (`inference.py:193-236`, `251-307`) |
| `pipeline._on_audio_chunk` → `_finalize_utterance` → `asr.transcribe` | `pipeline.py:46` → `pipeline.py:110` | Whisper.cpp transcription (hundreds of ms to seconds) |

Three further violations sit on the same thread: the callback allocates twice per
chunk (`np.frombuffer(...).astype(np.float32) / 32768.0`, `capture.py:64`), can
call `logger.error` (`capture.py:70`), and **emits Qt signals** — `window.py:105-106`
register `_on_text`/`_on_pipeline_state`, which emit `_bridge` signals from
`pipeline.py:113-115,124-127`, i.e. from T0.

The callback budget is one chunk: `chunk_size / sample_rate` = 1024 / 16000 ≈
**64 ms** (`config/default.yaml:38,40`). A 124 M-parameter, 4-step diffusion
forward pass plus two Hubert encodes cannot fit in it. Measured estimates put the
CPU target (M2 / R7 8854HS) at roughly 0.5–3 s per 1 s chunk — i.e. **8–47× over
budget**. (An earlier draft of this ADR said "100×+"; that number was not
grounded and has been removed.) The conclusion is unchanged: no chunk size makes
a diffusion model fit a 64 ms real-time callback.

### 2. No state ownership model

`DiffusionMotionInference` is mutated from four threads with no synchronisation:

| Thread | Mutates |
|---|---|
| PortAudio callback (T0) | `_audio_buffer`, `_tts_buffer` (via `_try_infer`), `_prev_params` |
| Camera thread | `_visual_buffer` (`process_visual_frame`) |
| Generation thread | `_tts_buffer`, `_current_prompt`, `_pyaudio` |
| GUI thread | `_model`, `_loaded`, buffers (`load_model`, `cleanup`) |

Two more shared objects the review added that an earlier draft missed:

- `_param_queue` (`live2d_widget.py:24`) is **appended from T0** — `window.py:123`
  registers `push_params` as an `on_params` callback, invoked from
  `_emit_params` on the PortAudio thread — and popped from the GUI thread.
- `ConversationContext` is written by the GUI thread (`window.py:144`) and read
  by T5 (`window.py:160`) concurrently.

The concrete `_tts_buffer` race: T0 executes `min(len(self._tts_buffer), …)`
(`inference.py:216`), the GUI thread runs `cleanup()` (`window.py:245` →
`inference.py:331` `_tts_buffer.clear()`) between the check and the
`popleft()` loop (`inference.py:219-221`) → `IndexError` on an empty deque,
swallowed by `capture.py:67-70`, leaving `_audio_buffer` half-drained. The same
class applies to `_audio_buffer`, which `cleanup()` clears while `_try_infer`
pops from it on T0.

### Constraints

- **C1** — The audio callback must return in microseconds. No unbounded
  allocation, no blocking, no contended lock, no logging, no signal emission.
- **C2** — Inference may be far slower than real time. The design must *bound
  latency by discarding work*, not by buffering it.
- **C3** — PySide6/Qt objects may only be touched from the GUI thread.
- **C4** — Public surfaces must be preserved: `process_user_audio`,
  `process_tts_audio`, `process_visual_frame`, `on_params`, `on_text`,
  `on_state_change`, `PipelineState`. *(See "Known C4 conflicts" — two of these
  cannot be preserved unchanged.)*
- **C5** — Shutdown must be deterministic: no thread may touch a resource after
  it has been released.

### Goals

1. The PortAudio callback copies samples and signals — nothing else.
2. Each piece of mutable state has exactly **one** owning thread.
3. Latency stays bounded when inference cannot keep up.
4. Shutdown joins every thread — including per-utterance generation — before
   releasing resources.
5. The capture→compute seam is observable from a test.

## Decision

Adopt an **owner-thread + bounded-queue** architecture (option B), staged: land
the minimal fix (option D) first, then upgrade.

### Threads and ownership

| ID | Thread | Owns | Consumes | Produces |
|---|---|---|---|---|
| T0 | PortAudio callback (external) | nothing | device buffer | `AudioRing` |
| T1 | `AudioWorker` (new) | VAD state, utterance buffer | `AudioRing` | utterance → `asr_q`; raw audio → `motion_in` |
| T2 | `ASRWorker` (new) | `WhisperASR` | `asr_q` (**FIFO, cap 3**) | text → Qt signal |
| T3 | `MotionWorker` (new) | **all** of `DiffusionMotionInference` | `motion_in`, `tts_stream`, `visual_slot`, `cmd_q` | frames → `render_q` |
| T4 | `CameraWorker` (exists, `camera.py:37,85-108`) | `cv2.VideoCapture`, FaceMesh | camera | `visual_slot` |
| T5 | Generation thread (exists, `window.py:156`) | `DialogueModel` stream, TTS synthesis | text event, `gen_cancel` | UI chunks; TTS → `tts_stream` |
| T6 | GUI thread (Qt) | widgets, `Live2DRenderer` | `render_q` | user input |
| T7 | Playback thread (new) | output `pyaudio.Stream` | `play_q` | speaker; publishes **playback position** |

### Mechanisms

**`AudioRing`** — preallocated `np.float32` ring, single producer (T0), single
consumer (T1). The int16→float32 conversion may stay on T0 (~µs for 1024
samples) or move to T1; this ADR puts it on T1 to honour goal 1 strictly. T0
performs a bare copy plus `Event.set()`; T1 is woken via that event (accepting
that `Event.set` takes an internal, effectively-uncontended lock — the ADR
should say "no *contended* lock", not "no locks").

*Overflow policy (corrected):* the reviewed draft said "overwrite the oldest
samples", which requires the **writer** to advance the read index — a data race
with the consumer's index updates, defeating the SPSC invariant. Instead: the
consumer detects lag via an **epoch/sequence counter** and resyncs, dropping to
the newest audio; overflow is counted and logged as an anomaly, not treated as
routine.

**`Mailbox[T]`** — single-slot, latest-value-wins, short-lived `Lock`. Used
**only** for `visual_slot`. Rationale: a stale camera frame is worthless, so
staleness *is* worse than loss. This rationale does **not** extend to audio or
transcripts — see below.

**`motion_in`** — mic audio into T3 is **bounded and latest-wins** (drop to the
newest complete chunk). Because the model is memoryless per chunk (see
"Continuity"), a FIFO would only grow without bound and drift ever further from
reality. Dropping keeps the character approximately in sync.

**`tts_stream` (corrected)** — TTS audio is **not** a mailbox. It is a finite
stream whose entire content drives the mouth sequentially; "a backlog of TTS
audio is useless" was false. Design:

- a **bounded FIFO** carrying `(stream_id, sample_rate, audio)`;
- T5 publishes the whole utterance for a given `stream_id`;
- T3's read position is derived from **T7's playback position**, not from
  inference cadence — otherwise T7 (playing at 1×) and T3 (eating 1 s per
  inference) are two unsynchronized consumers of one stream, and under C2 the
  mouth lags the speaker without bound;
- replacement discards only when a **new utterance** supersedes the current one
  (explicit interruption), never mid-utterance.

**`asr_q` (corrected)** — a bounded FIFO, cap 3, drop-oldest + log on overflow.
Latest-wins would delete a user utterance whenever a second utterance arrives
while T2 is still transcribing the first.

**`render_q` (corrected)** — bounded (≈3 s of frames), and drained
**sequentially and rate-matched**: per paint, pop up to the number of frames
that wall-clock has advanced, and on underflow **hold the last applied
parameters** — do not reset to `_default_params`. The actual defect in
`live2d_widget.py:40-44` is the reset, not the deque:

> the reset is **structural**, not a "when behind" phenomenon: at
> `update_interval: 0.016` (`default.yaml:33`) the consumer takes 62.5 frames/s
> while the producer emits ~49/s, and those 49 arrive as one burst per second.

The reviewed draft's "apply only the newest frame" would have discarded 48 of
every 49 frames — teleporting the character once per inference instead of
animating it. Bound `play_q` likewise.

**`visual_slot` / camera (corrected)** — throttle **at the source** in T4 to
2–5 Hz (time-gated, not frame-count). FaceMesh at 30 fps costs 10–30 ms/frame of
CPU competing with T3, and `_prepare_visual_frames` only ever samples 5 frames
(`inference.py:239`). Also cap `_visual_buffer`: retaining 100 frames of
`224×224×3` float32 is **≈57 MB of stale data** (`inference.py:204-207`) that is
never time-evicted. Cap at ~5–10 with timestamp eviction.

**Continuity (corrected framing)** — an earlier draft claimed dropping chunks
"interacts badly with overlap-add". There is nothing to interact with:
`_prev_params` (`inference.py:45`, written at `235`) is **never read** — dead
state — and the overlap block (`inference.py:225-231`) pops `keep_n` items from
the *front* then `extendleft(reversed(keep))` puts them back in original order, a
**verified no-op** (§H-2). The model is memoryless per chunk. So the honest
statement is: *there is no cross-chunk continuity mechanism; dropped chunks are
motion-time holes whose 49-frame sequences are then played at an unrelated
wall-clock rate.*

Two consequences the design must respect:

- `_try_infer` pairs each mic chunk **positionally** with the next 1 s of TTS
  (`inference.py:216-221`). If `motion_in` drops mic chunks while the TTS stream
  is FIFO, chunk N's mic audio (t=[5,6)) gets paired with TTS from t=[0,1). The
  pairing must be **time-aligned, not positional**.
- **Constraint on future work:** as soon as real continuity/conditioning is
  implemented (§H-2, §P1-10), drop-to-newest becomes actively harmful — it would
  condition on chunk N's parameters while feeding chunk N+5's audio, breaking the
  causal assumption. Any such implementation must revisit this policy.

**Command queue (corrected)** — `set_text_prompt`, `set_character_id`,
`load_model` route to T3 as commands. **Known C4 conflict:** `load_model`'s
synchronous `bool` return is *load-bearing* — `window.py:121-125` decides whether
to wire `on_params`/`on_raw_audio` based on it. Making it async therefore
requires a completion signal and moving that wiring into the completion handler.
This is a deliberate, documented C4 break.

### Shutdown protocol (corrected)

Ordered, with an explicit escalation policy:

1. **Store the T5 handle** (currently discarded at `window.py:156`) and add a
   `gen_cancel` event checked between streamed chunks
   (`window.py:163`). Note the local-transformers path
   (`dialogue/model.py:72-92`) completes generation in one blocking
   `model.generate` call before yielding, so between-chunk cancellation only
   works on the Ollama streaming path (`model.py:111-119`); the local path needs
   a `StoppingCriteria` or must accept a long join.
2. `capture.stop()` — T0 stops firing.
3. Set `stop_event`; wake and join T1, T2, T3, T4.
4. **Join T5 before touching PyAudio.**
5. Terminate **both** PyAudio instances — the capture-side one
   (`capture.py:20`, via `AudioPipeline.cleanup()`/`capture.cleanup()`,
   `pipeline.py:58-60`) and the playback one (`window.py:196`) — with
   `capture.cleanup()` explicitly placed in the order.
6. Tear down T7's stream; add `TTSEngine` cleanup (currently **no one** cleans it
   up — not `closeEvent`, not any protocol).
7. Stop the GUI timer, then `renderer.cleanup()`.

**Escalation on join timeout (required):** `join(timeout=2.0)` is *expected* to
time out — a single 4-step DDIM forward on CPU exceeds 2 s, and Whisper `base` on
a 15 s utterance (`pipeline.py:21`) can too. On timeout: **log and leak** the
thread (keep daemon=True so process exit reaps it) and **do not release the
resources that thread owns**. Proceeding to release anyway violates C5. "Each
join bounded" without an escalation path fails silently exactly when it matters.

**Camera ordering:** set `stop_event` **first**, then call `camera.stop()`.
`camera.stop()` (`camera.py:43-51`) joins with its own 2 s timeout then releases
`_cap`; on timeout, `_capture_loop` can hit `_cap.read()` on `None` between the
`_cap is not None` check (`camera.py:88`) and the read (`camera.py:89`).

**Test note:** `test_shutdown_joins_all_workers` must include T5 — as specified
in the reviewed draft, that test would pass while T5 still crashed the process.

### Failure semantics (new)

- **Worker error policy:** T3's loop catches, logs, and continues, with a
  consecutive-failure **circuit breaker**. Today exceptions are swallowed
  per-callback (`capture.py:67-70`); without a policy one bad frame silently
  kills motion forever.
- **`PipelineState` (must be decided, not inherited):** today `PROCESSING` spans
  synchronous transcription (`pipeline.py:108-119`) and, because
  `_on_audio_chunk` only buffers in `LISTENING` (`pipeline.py:85`), **all speech
  during the 1–2 s transcription window is discarded**. That is an existing
  data-loss bug adjacent to C-1. Under this design T1 buffers **continuously**
  and states become UI labels; `PROCESSING` must therefore stop meaning "VAD
  paused". This is a deliberate semantics change, contradicting the reviewed
  draft's claim of "identical semantics" — accepted deliberately, because the
  old semantics lose user input.
- **Interruption (newly identified, currently absent):**
  `window.py:149-151` returns early while `_generating` is true, so a user
  utterance arriving during a response is appended to context (`window.py:144`)
  but **never generates a reply**. `CONTEXT.md`'s interruption spec has no
  mechanism. T7 needs a stop-and-flush command coordinated with `tts_stream`
  replacement. This is in scope for "always-on".

## Alternatives considered

| Option | Summary | Verdict |
|---|---|---|
| **B** — Owner threads + bounded queues | As above | **Chosen**, with the four policy corrections |
| **A** — Single-threaded `asyncio` | `stream.read()` in a blocking thread; asyncio orchestration | **Fairly dismissed.** Qt-centred system, synchronous CPU-bound torch; `run_in_executor` re-adds every thread, and blocking-mode reads add jitter vs. callbacks |
| **C** — Motion model in a separate **process** | shared memory for audio/params | **Fair conclusion, thin reasoning.** "Ops largely release the GIL" was asserted, never measured — and Hubert is a `transformers` model whose forward is many small ops with real Python-side GIL time. The genuine trigger is likely **core starvation** (torch intra-op threads vs. T1/T2/T4), not GIL contention *per se* — same remedy, different diagnosis. B's seams are already process-boundary-shaped, so C is an extraction later *if* the benchmark demands it |
| **D** — Minimal fix | Callback only enqueues; ASR moves off-thread; T3 stays driven by the audio queue | **Underplayed in the reviewed draft.** D fixes the most severe defect class (C-1) with a tiny diff and is strictly compatible with B's seams. **Now the first stage.** Note: D alone leaves C-6 (Speak dead without mic input) unfixed |

## Staged plan

| Stage | Content | Closes | Size |
|---|---|---|---|
| **0** | Commit the C-9 import fix (`live2d_widget.py:6` → `PySide6.QtOpenGLWidgets`) — **already written**, uncommitted | C-9 | Quick |
| **1 (D)** | `_audio_callback` enqueue-only; dedicated T2 ASR thread; `asr_q` FIFO; store T5 handle + join before PyAudio teardown | C-1 (most severe class) | Short |
| **2 (B)** | Full owner-thread model: T1/T3/T7, `tts_stream` + playback-clock coupling, rate-matched `render_q` + hold-last-params, `AudioRing` epoch resync, failure semantics, `load_model` completion signal, camera throttle, interruption, dead-code removal (`capture.get_buffer` `capture.py:73`, `CHUNK_MS` `capture.py:11`) | C-1, C-5, C-6, H-1, H-2, H-6 | 1–2 d |

**Prerequisite, separate change:** remove the duplicate Hubert forward (§H-8 —
`inference.py:272-274` encodes purely to derive `T`, then `model.py:407-408`
encodes the same audio again). That is a free ~2× on the cost the entire design
is organized around, and it is cheaper than any architectural work.

## Consequences

### Positive
- Audio capture becomes real-time safe; dropouts stop.
- State ownership becomes stateable in one table, making races reviewable.
- Latency is bounded by construction; dropped work is counted and logged.
- The capture→compute seam becomes testable without hardware.

### Negative / accepted costs
- **Inference remains slower than real time on the CPU target.** The design makes
  this visible and bounded; it does not fix it. On CPU, Listen mode will still
  produce occasional snaps, and Speak mode needs the playback-clock coupling to
  avoid unbounded mouth lag. The product answer for motion quality is the
  procedural baseline in `docs/MODEL_TRAINING_ROADMAP.md` Phase 0 — a separate
  change.
- `motion_in` dropping chunks is incompatible with any future continuity
  mechanism (see "Continuity").
- More threads means more shutdown surface; mitigated by the explicit protocol
  and an escalation policy, plus a join-verification test that includes T5.
- Two documented C4 breaks: `load_model` becomes async, and `PROCESSING` stops
  meaning "VAD paused".

## Settling evidence to gather (cheap)

To decide later whether option C (process isolation) is warranted:

1. p99 T1 ring-handoff latency with T3 idle vs. saturated.
2. `_diffusion_infer` wall-clock on the CPU target with `torch.set_num_threads`
   capped vs. uncapped, while T1/T2/T4 run.

Escalate to C if T1's p99 exceeds ~50 ms, or if inference inflates >30 % under
co-load.

## Test plan

| Test | Asserts |
|---|---|
| `test_audio_callback_returns_within_budget` | T0 path performs no work beyond copy + signal (no logging, no Qt emit, no inference) |
| `test_mailbox_latest_wins` | Writing twice yields the second value; no growth |
| `test_asr_queue_preserves_utterances` | Two utterances queued back-to-back are both delivered (guards the latest-wins regression) |
| `test_tts_consumption_tracks_playback_clock` | T3's TTS read position follows T7's playback position |
| `test_render_queue_plays_frames_in_order` | A 49-frame burst is consumed sequentially, not collapsed to the newest |
| `test_widget_holds_last_params_on_underflow` | Empty queue holds parameters; does **not** reset to `_default_params` |
| `test_motion_input_drops_to_newest_when_behind` | Bounded, no unbounded growth |
| `test_speak_state_advances_without_microphone` | TTS alone advances T3 (replaces the RED `test_tts_audio_alone_drives_motion_inference`) |
| `test_shutdown_joins_all_workers` | After `stop()`, T1–T5 are all dead and resources released exactly once |
| `test_tts_audio_resampled_to_pipeline_rate` | 1 s of TTS occupies 1 s of motion time (replaces the RED `test_one_second_of_tts_audio_occupies_one_second_of_motion_time`) |

Note: the C-5 resample-at-ingest must know each buffer's **true** source rate —
`TTSEngine._silence()` hardcodes 16000 while the engine advertises 22050
(`engine.py:119-121`), so the silence fallback would otherwise be
time-stretched.
