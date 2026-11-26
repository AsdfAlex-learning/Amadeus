# ADR 0004: Speak-State and Render Synchronisation (ADR 0003 Stage 2)

> **Status**: Proposed — for review
> **Date**: 2026-09-15
> **Branch**: `fix/runtime-p0`
> **Depends on**: ADR 0003 (Stage 1 landed: owner threads for capture / ASR / motion, generation-thread join)
> **Related**: `docs/PROJECT_REVIEW_REPORT.md` §H-2, §H-6, §C-7, and the review flaws recorded in ADR 0003

## Status

Accepted (implemented; merged via PR #1 on 2026-11-26). D1 was revised after Oracle review - see "Review outcome".

## Alternatives considered

- **Indexing the speech window at the playback position** (`audio[played : played + chunk]`): rejected - it skips every un-inferred window between the cursor and playback, the mid-utterance skip the design exists to avoid. Playback gates the advance; it does not index it.
- **The original D1** (SpeechStream consumed purely against the playback clock) was revised after Oracle review to the gated-cursor design recorded under "D1 revised".
- **Rate-matched drain vs frame-drop for `render_q`**: resolved as D2 (drain rate-matched to the display, hold the last applied frame, decay to neutral on sustained underflow).

## Context

ADR 0003 Stage 1 is implemented: the PortAudio callback only enqueues, ASR runs on
its own thread fed by a bounded FIFO, inference runs on a dedicated `MotionWorker`,
and the generation thread is retained and joined before PortAudio is torn down.

That fixed C-1 (compute on the real-time thread) but deliberately left four
correctness problems that Stage 1 does not address. They are all *synchronisation*
problems, not threading problems, which is why they need their own design.

### P1 — TTS audio is consumed at inference cadence, not playback cadence

`process_tts_audio` buffers resampled speech and `_try_infer` drains exactly
`chunk_size` seconds of it per inference step. Playback (`window._speak_response`)
independently writes the *same* audio to the output stream at 1x.

So one utterance has **two unsynchronised consumers**: T7 plays it in real time,
T3 eats it one second per inference. On the CPU target, where inference is far
slower than real time, the mouth drifts ever further behind the spoken audio. The
review flagged this as the single most serious flaw in ADR 0003's first draft
(flaw 1), and Stage 1 did nothing about it.

### P2 — `render_q` snaps to the default pose

`live2d_widget.py` holds a `deque(maxlen=120)` and pops one frame per paint:

```python
params = self._param_queue.popleft() if self._param_queue else None
if params is not None:
    self.renderer.set_parameters(params)
else:
    self.renderer.set_parameters(self._default_params)   # <- the defect
```

At `update_interval: 0.016` the consumer takes ~62.5 frames/s while the producer
emits ~49 frames/s *in one burst per second*, so the queue underflows
structurally — not only when inference is slow. On underflow the character resets
to `_default_params`, which is what produces visible snapping between chunks.
The review's proposed "keep only the newest frame" would have been worse (it
collapses a 49-frame sequence into a per-inference pose teleport); the correct
policy is sequential, rate-matched consumption plus *hold last applied*.

### P3 — Interruption does not exist

`window._start_generation` returns early while `_generating` is true, so a user
utterance arriving during a spoken response is appended to the context but
**never generates a reply**. `CONTEXT.md` specifies the opposite ("user speech
during Speak → immediate switch"). There is no mechanism, not merely a weak one.

### P4 — Resource and lifecycle gaps

- Two `PyAudio` instances exist (capture, playback). Shutdown *does* terminate both —
  but it terminates the playback instance **unconditionally, even after a join has
  leaked a generation thread still inside `stream.write()`**, which violates the
  ADR 0003 escalation policy and reintroduces the exact use-after-terminate crash the
  policy exists to prevent. *(Corrected during review: an earlier revision of this ADR
  claimed only one instance was terminated. That was stale pre-Stage-1 text and had the
  bug backwards.)*
- `TTSEngine` is never cleaned up by anyone — not `closeEvent`, not any protocol.
- The camera runs FaceMesh at ~30 fps (10–30 ms/frame) competing with inference,
  and `_visual_buffer` retains 100 frames of `224x224x3` float32 ≈ **57 MB**, of
  which `_prepare_visual_frames` only ever samples 5.

## Decision

### D1 — A `SpeechStream` consumed against the playback clock

Introduce an explicit value object and make playback the timing master:

```python
@dataclass(frozen=True)
class SpeechStream:
    stream_id: int
    sample_rate: int
    audio: np.ndarray          # float32, mono, at sample_rate
    started_at: float          # monotonic seconds, set when playback begins
```

- T5 publishes a `SpeechStream` into a **bounded FIFO** (capacity 2). Not a
  mailbox: TTS is a finite stream whose whole content drives the mouth
  sequentially, so a backlog is what the mouth *should* be working through.
- T7 owns the output stream and publishes `played_samples(stream_id)` — a
  monotonic count of samples written for the current stream.
- T3 derives its read window from that position:
  `window = audio[played : played + chunk_samples]`, rather than from its own
  inference cadence.
- Consequence: if T3 is slower than real time it falls behind by a bounded,
  *visible* amount, and the mouth never skips mid-utterance. This is strictly
  better than today's unbounded drift.

### D2 — `render_q`: rate-matched drain, hold last applied

- `render_q` becomes a bounded deque (capacity ≈ 3 s at 50 Hz = 150 frames).
- Drain on a `QTimer` at the model's **nominal 50 Hz**, popping up to the number
  of frames wall-clock has advanced, and applying the newest available.
- **On underflow, hold the last applied parameters.** `_default_params` is
  applied only before the first frame ever arrives.
- The `maxlen` overflow path drops oldest and counts it.

### D3 — Camera throttle and bounded visual buffer

- Time-gate the camera loop to `perception.fps` (default 4 Hz). FaceMesh at 30 fps
  is pure CPU contention against inference and 29 of 30 frames are discarded.
- Cap `_visual_buffer` at 10 entries with timestamp eviction (~6 MB instead of ~57 MB).

### D4 — `load_model` becomes asynchronous with a completion signal

*(Accepted C4 break.)* `load_model()`'s synchronous `bool` return is load-bearing:
`window._load_motion_model` decides whether to register `on_params`/`on_raw_audio`
from it. Under an owner thread the load happens on T3, so:

- add `load_model_async()` plus `on_loaded(callback)`;
- move the wiring into the completion callback;
- keep the synchronous `load_model()` for tests and standalone use.

### D5 — Interruption

On user speech onset while `Speak` is active:

1. `T7.stop_and_flush()` — output stream stopped and its queue drained;
2. `SpeechStream` for the current `stream_id` is superseded (explicit interrupt);
3. the in-flight generation is cancelled (`_gen_cancel`);
4. a new generation starts immediately.

`_start_generation` therefore stops early-returning on `_generating` and instead
cancels and replaces. A short release (configurable, default 0 ms) is available
if hard cuts prove harsh.

### D6 — Shutdown completeness

Ordered: stop capture → `stop_event` → join T1/T2/T3/T4 → **join T5** →
terminate *both* PyAudio instances → **`TTSEngine.cleanup()`** (new) → stop GUI
timer → renderer cleanup. Each join keeps the ADR 0003 escalation policy
(log and leak on timeout; never release what a live thread may use).

### D7 — Uniform failure semantics

Every worker loop catches, logs and continues, with a consecutive-failure circuit
breaker that stops the worker so a persistent fault is *visible*. Already
implemented for motion; extend to ASR (partially done) and playback.

## Consequences

**Positive**
- Mouth motion tracks the audio actually being heard, within a bounded lag.
- The character stops snapping to a neutral pose between chunks.
- Interruption works, which is what "always-on conversation" requires.
- ~50 MB of resident stale frames removed; camera CPU contention drops ~7x.

**Negative / accepted**
- D1 couples T3 to T7: if playback stalls, motion correctly stalls too. That is
  intended, but it means a playback fault is now a motion fault — the circuit
  breaker in D7 must cover T7.
- D4 and the `PROCESSING` change (already landed in Stage 1.3) are two documented
  breaks of "public interfaces preserved".
- Bounded lag is now explicit rather than hidden. On CPU the character is still
  behind; the design makes that measurable instead of pretending otherwise.

## Non-goals

- **Real-time inference on CPU.** That needs model work (smaller chunk, fewer DDIM
  steps, distillation) and is untouched here.
- **Cross-chunk continuity (§H-2).** `_prev_params` is dead and the overlap block
  is a verified no-op, so there is no continuity mechanism today. Adding one is a
  separate change, and ADR 0003 records that `motion_in`'s drop-to-newest policy
  must be revisited if it lands.
- **C-7 (real token streaming).** Blocked on the pinned transformers 4.32.1, whose
  `TextIteratorStreamer` buffers until `end()`. Separate decision.

## Open questions for review

1. **Clock ownership (D1).** Is "T7 publishes `played_samples`" the right master, or
   should there be an explicit clock object both T3 and T7 read? What happens if
   the output stream underruns and playback effectively stalls?
2. **Asymmetric drop.** `motion_in` drops to the newest mic chunk while TTS follows
   the playback clock. Is that combination coherent, or should mic audio also be
   clocked (to *arrival* time) to avoid pairing mic audio from one moment with TTS
   from another?
3. **Drain policy (D2).** A 50 Hz `QTimer` versus a paint-driven accumulator: which
   is more robust when `paintGL` is throttled (window occluded/minimised)? How do we
   bound `render_q` when the widget stops painting entirely?
4. **Interruption latency (D5).** Is a hard cut acceptable at the audio level, or is
   a short release (e.g. 80 ms) needed to avoid a click? Should the *context* keep
   the truncated assistant turn or discard it?
5. **Camera rate (D3).** The model consumes 5 frames per inference step (~1 Hz). Is
   4 Hz the right source rate, or is there value in higher-rate sampling with
   timestamp-based selection instead?
6. **Ordering risk.** D1 makes motion depend on playback. Does that create a
   deadlock or priority-inversion path with the existing `_generating` single-flight
   guard, given that T5 both synthesises and triggers playback?

## Staged plan (after review)

| Stage | Content | Closes | Risk |
|---|---|---|---|
| S2.1 | `render_q` rate-matched drain + hold-last-params + bound | H-6, P2 | Low — self-contained in `live2d_widget.py` + `window.py`, and it is the most visible defect |
| S2.2 | Camera throttle + `_visual_buffer` cap | P4 (partly) | Low |
| S2.3 | `SpeechStream` + playback-clock coupling | P1 (Oracle flaw 1) | **High** — cross-thread clock; needs the review's answers to Q1/Q2 first |
| S2.4 | Interruption | P3 | Medium — state-machine change |
| S2.5 | Shutdown completeness + `load_model` async + failure-semantics sweep | P4, D4, D6, D7 | Medium |

S2.1 and S2.2 are independent of the clock design and can land before the harder
questions are settled.

## Test plan

| Test | Asserts |
|---|---|
| `test_render_queue_plays_frames_in_order` | A 49-frame burst is consumed sequentially, not collapsed to the newest |
| `test_widget_holds_last_params_on_underflow` | Underflow holds parameters; never resets to `_default_params` |
| `test_render_queue_is_bounded_when_painting_stops` | Producer cannot grow the queue without bound |
| `test_camera_publishes_at_the_configured_rate` | Time-gated, not frame-count-gated |
| `test_visual_buffer_is_capped` | ≤10 frames retained |
| `test_tts_window_follows_playback_position` | T3's read window tracks `played_samples` |
| `test_new_utterance_supersedes_the_current_stream` | Explicit interrupt drops the old stream; no mid-utterance skip |
| `test_user_speech_during_response_starts_a_new_generation` | P3: interruption actually produces a reply |
| `test_shutdown_terminates_both_pyaudio_instances` | Capture-side and playback-side |
| `test_shutdown_cleans_up_tts_engine` | `TTSEngine.cleanup()` is invoked exactly once |

---

## Review outcome (Oracle, 2026-09-15) — D1 revised

Reviewed before implementation. Verdict: **D1 is the right approach with the wrong
mechanism**, plus 5 Critical, 4 High and 6 Medium flaws across D1–D7.

### D1 revised (supersedes the D1 text above)

- **Playback *gates* inference; it does not *index* it.** T3 owns a per-stream
  `inferred` cursor and each step reads `audio[inferred : inferred + chunk]`, advancing
  only while `inferred + chunk <= min(played + lookahead, len(audio))`. The original
  `audio[played : played + chunk]` skips every un-inferred window between T3's cursor
  and `played` — precisely the "mid-utterance skip" this ADR claimed was impossible.
- **A max-lag skip policy is required.** The lag is bounded by utterance length, not by
  a constant: at 3 s per chunk the mouth falls ~0.67 s behind per second of playback and
  keeps animating after the audio has ended, into the user's next turn. Add: if
  `played - inferred > max_lag`, jump the cursor to `played - max_lag` and **count** the
  dropped window.
- **Prerequisite: extract T7 with chunked writes (≤200 ms) that publish
  `(stream_id, played_samples)` per chunk.** A single blocking `stream.write` of the
  whole utterance makes the position visible only at the end, which reproduces today's
  behaviour plus a delay. A callback-mode output stream is the alternative.
- **Unit mismatch (the C-5 bug class, again).** `chunk_samples` is at 16 kHz and
  `SpeechStream.audio` at 22050 Hz; indexing one with the other is an off-by-1.38.
  Store the stream resampled to the pipeline rate, or publish position in *seconds*.
- **FIFO overflow must supersede *pending*, never the active stream.**
  `BoundedDropOldest` drops the oldest — i.e. the stream T3 is working on. Wrong policy
  for speech. Also drop `started_at`: redundant with `played_samples`, and it would be
  written by a different thread than the one that publishes the (frozen) dataclass.
- **`SpeechStream` must carry the backend's *true* rate and dtype.** `synthesize`
  currently offers no way to report either, and the pyttsx3 path reads via `sf.read`,
  which returns float64 — `tobytes()` would play that as `paFloat32` garbage.
- **During an active stream, feed `user_audio = zeros`.** That *is* CONTEXT.md's Speak
  state. The current code can feed user *and* TTS audio non-zero simultaneously — a
  fourth state that appears in no specification.

### Other adopted corrections

- **D2** — "apply the newest available" must read **apply the last frame *popped***;
  otherwise it is the per-inference pose teleport that ADR 0003's review already killed.
  Add an **idle timeout** (>300–500 ms of underflow → decay to `_default_params`),
  because hold-last-forever combined with having no Silence driver is a permanent statue.
  `deque(maxlen=…)` does not count drops, and a 20 ms timer needs `Qt.PreciseTimer`.
- **D5** — `_gen_cancel` is one process-wide `Event` shared with shutdown; reusing it for
  interruption makes the *replacement* generation observe the flag and abort immediately,
  so the interrupting utterance gets no reply. Needs a **per-generation token**, and
  `_on_response_done` must be **generation-id-gated**, or the superseded generation's
  completion clobbers the new generation's `_generating` and spawns a third concurrent
  thread sharing `self._pyaudio`. This is the concrete answer to Q6 — a lost-update race,
  not a deadlock. Also, the transcript arrives seconds *after* speech onset (VAD +
  Whisper), so what can start immediately is the *reaction* (stop playback, Listen pose),
  not a new generation.
- **D6** — must join **T7** (not only T5), and must **skip PyAudio termination when a
  join leaked a live thread**. The landed Stage 1 code violates its own escalation
  policy today.
- **D3** — capping the buffer does not fix the actual defect: the camera thread
  appends/pops `_visual_buffer` while T3 reads it with `np.stack` over indices, with **no
  synchronisation at all**. This is an *unlanded Stage 1 gap* (ADR 0003 promised a
  `visual_slot` Mailbox), not a cap change. Frames also carry no timestamps, so
  "timestamp eviction" is unimplementable without stamping at ingest. Sample the
  **newest 5** frames, not `range(0, num, step)[:5]`.
- **D7** — add breakers for T1 (drain worker) and T5; on T7 failure **supersede the
  stream** so T3 falls back to Listen/Silence. T7's breaker + D1's coupling + D2's
  hold-last otherwise compose into an unrecoverable statue.

### Corrected factual claims

- **P4 "shutdown terminates one of them [PyAudio]" — false** (see the inline correction above).
- **P4's FaceMesh CPU claim — false as wired.** No `on_perception` callback is registered
  anywhere, so `_process_face_mesh` never executes; the "~7× contention drop" is
  ungrounded. The throttle is still right for the per-frame `cvtColor`+`resize`.
- **P3's framing — wrong for the playback phase.** `_generating` is cleared *before*
  playback begins (`response_done` is emitted before `_speak_response`), so a new
  utterance during playback does start a second generation thread. The real bug there is
  concurrent `_speak_response` calls sharing `self._pyaudio` with overlapping audio — not
  "never answered".
- **"T7" does not exist yet.** Playback is inline on T5; both D1 and D5 depend on
  extracting it, and that extraction is the actual unit of work.
- **§(e) The Silence state has no driver at all.** `_try_infer` requires ≥1 s in some
  buffer, so idle motion never runs today: the "always-on" engine is a two-state
  implementation. Named here because D2 makes idle *worse* (statue vs. neutral pose).

### Q1–Q6 answers

1. **No shared clock object** — a third mutable thing to own. T7 publishing
   `(stream_id, played_samples)` as a GIL-atomic int suffices *provided writes are
   chunked*. Prefer chunked blocking writes (smaller change); expose an underrun counter.
2. **Decide, don't hedge:** during an active stream feed `user_audio = zeros`.
   Interruption is the state transition; time-aligning two live streams for a memoryless
   1 s-chunk model is over-engineering.
3. **QTimer, unambiguously** — a paint-driven drain stops when the widget is hidden.
   Boundedness was never the defect (`maxlen` already bounds memory); the defects were
   reset-on-underflow and rate mismatch.
4. **Non-zero fade, ~80 ms default** (a hard cut clicks). **Keep the truncated assistant
   turn**, marked truncated — the user heard part of it, and deleting it makes the next
   reply reference a statement the model believes it never made. Fix the generation races
   first.
5. **4–5 Hz with index sampling**, stamped at ingest, sampling the newest 5.
6. **No deadlock** — T3's wait is poll-based and there is no cycle T3→T7→T5→T3 — *provided
   the SpeechStream publish stays non-blocking*. A blocking put livelocks T5 once T3's
   breaker trips. The real hazards are the generation races and the D7 starvation composite.

### S2.1 / S2.2 independence

**Confirmed**, with two caveats: (i) the idle-timeout decision belongs in **S2.1**,
because it determines idle behaviour under *every* clock design, so deferring it means
revisiting the widget; (ii) **S2.2 is not merely a cap change** — it requires the
camera→T3 handoff fix, an unlanded Stage 1 gap.

### Revised staged plan (dependency order changed)

| Stage | Content | Effort |
|---|---|---|
| S2.1 | `render_q` drain policy + "last popped" + **idle timeout** + drop counting | Quick |
| S2.2 | Camera→T3 **Mailbox handoff** + ingest stamping + newest-5 + throttle 4–5 Hz | Short |
| S2.3 | **Extract T7** (chunked writes, publish position) — prerequisite for S2.4 and S2.5 | Short |
| S2.4 | D1 revised: gated cursor + max-lag skip + unit fix + supersede-pending | Medium (1–2 d) |
| S2.5 | Per-generation token + `response_done` gating + interruption (fade) + truncated-turn policy | Short |
| S2.6 | Shutdown: join T7, skip terminate on leak, `TTSEngine.cleanup`, T1/T5 breakers | Short |

T7 extraction is now explicit and precedes D1/D5, which the review showed cannot be
built without it.
