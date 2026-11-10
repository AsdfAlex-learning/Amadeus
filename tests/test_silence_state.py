"""The Silence state must actually run.

CONTEXT.md describes three states — Listen, Speak, Silence — but Silence had no
driver at all. `_try_infer` requires a full chunk in some buffer, so with nothing
to listen to and nothing to perform there were zero inference steps; the
"always-on" engine was on only while someone was talking, and the only idle
behaviour came from the renderer decaying to its neutral pose after its 400 ms
timeout.

Silence motion is emitted with both audio inputs zero and the ``silence`` mode, so
PerformanceEngine applies ``idle_energy`` rather than speech expressiveness.
"""

from __future__ import annotations

import time

import numpy as np


class _RecordingEngine:
    """Builds an engine whose model boundary records (user, tts, prompt, mode)."""

    def __init__(self, config: dict) -> None:
        from src.motion.inference import DiffusionMotionInference

        self.steps: list[tuple[np.ndarray, np.ndarray, str | None]] = []
        self.modes: list[str] = []

        def _infer(user_chunk, tts_chunk, _visual, prompt=None):
            self.steps.append((np.array(user_chunk), np.array(tts_chunk), prompt))
            return np.full((5, 45), 0.5, dtype=np.float32)

        engine = DiffusionMotionInference(config)
        engine._loaded = True
        engine._model = object()
        engine._diffusion_infer = _infer  # type: ignore[method-assign]

        original_apply = engine._apply_performance

        def _apply(params, mode):
            self.modes.append(mode)
            return original_apply(params, mode)

        engine._apply_performance = _apply  # type: ignore[method-assign]
        self.engine = engine


def test_silence_emits_an_idle_step_when_nothing_else_is_happening(config: dict) -> None:
    """With no input at all, the engine must still produce motion."""
    rec = _RecordingEngine(config)
    rec.engine._silence_idle_s = 0.0
    rec.engine._silence_interval_s = 0.0

    rec.engine._maybe_run_silence()

    assert rec.steps, "no idle step ran, so the character would be motionless"
    user_chunk, tts_chunk, _prompt = rec.steps[0]
    assert float(np.abs(user_chunk).sum()) == 0.0, "Silence must feed zero user audio"
    assert float(np.abs(tts_chunk).sum()) == 0.0, "Silence must feed zero TTS audio"
    assert rec.modes[-1] == "silence", (
        f"expected the silence performance mode (idle_energy), got {rec.modes[-1]!r}"
    )


def test_silence_waits_for_the_idle_window(config: dict) -> None:
    """It must not fire while a real interaction is still warm."""
    rec = _RecordingEngine(config)
    rec.engine._silence_idle_s = 5.0
    rec.engine._silence_interval_s = 0.0
    rec.engine._last_activity_at = time.monotonic()

    rec.engine._maybe_run_silence()

    assert not rec.steps, "idle motion started while an interaction was still recent"


def test_silence_is_rate_limited(config: dict) -> None:
    """It must not run on every tick."""
    rec = _RecordingEngine(config)
    rec.engine._silence_idle_s = 0.0
    rec.engine._silence_interval_s = 10.0
    rec.engine._last_activity_at = 0.0  # long ago

    rec.engine._maybe_run_silence()
    rec.engine._maybe_run_silence()

    assert len(rec.steps) == 1, f"expected one rate-limited step, got {len(rec.steps)}"


def test_silence_yields_to_speak_state(config: dict) -> None:
    """While an utterance is being performed, Silence must not interfere."""
    rec = _RecordingEngine(config)
    rec.engine._silence_idle_s = 0.0
    rec.engine._silence_interval_s = 0.0

    class _Idle:
        played_samples = 0

    rec.engine.attach_playback(_Idle(), playback_rate=int(config["tts"]["sample_rate"]))
    # No lookahead, so the utterance stays retained instead of being consumed
    # immediately (a 1 s utterance with 1 s of lookahead completes at once).
    rec.engine.SPEECH_LOOKAHEAD_SECONDS = 0.0
    rec.engine.process_tts_audio(np.zeros(16000, dtype=np.float32), sample_rate=16000)
    assert rec.engine.speech_active

    before = len(rec.steps)
    rec.engine._maybe_run_silence()

    assert len(rec.steps) == before, "idle motion ran while the character was speaking"


def test_silence_yields_to_queued_listen_audio(config: dict) -> None:
    """Queued microphone audio belongs to the Listen state."""
    rec = _RecordingEngine(config)
    rec.engine._silence_idle_s = 0.0
    rec.engine._silence_interval_s = 0.0
    rec.engine._audio_buffer.extend([0.2] * int(rec.engine.chunk_size * rec.engine.sample_rate))

    rec.engine._maybe_run_silence()

    assert not rec.steps, "idle motion ran while Listen audio was queued"


def test_silence_can_be_disabled(config: dict) -> None:
    rec = _RecordingEngine(config)
    rec.engine._silence_enabled = False
    rec.engine._silence_idle_s = 0.0
    rec.engine._silence_interval_s = 0.0

    rec.engine._maybe_run_silence()

    assert not rec.steps, "silence_enabled=False still produced idle motion"


def test_silence_uses_its_own_prompt(config: dict) -> None:
    """Idle behaviour is described by perception and elapsed time, not by dialogue."""
    rec = _RecordingEngine(config)
    rec.engine._silence_idle_s = 0.0
    rec.engine._silence_interval_s = 0.0
    rec.engine.set_text_prompt("what did the user say")
    rec.engine.set_silence_prompt("user has been quiet for 15 s")

    rec.engine._maybe_run_silence()

    _user, _tts, prompt = rec.steps[0]
    assert prompt == "user has been quiet for 15 s", (
        f"the silence prompt was not used; got {prompt!r}"
    )


def test_the_worker_ticks_so_silence_runs_without_input(config: dict) -> None:
    """End to end: a running worker emits idle motion with nothing published."""
    rec = _RecordingEngine(config)
    rec.engine._silence_idle_s = 0.0
    rec.engine._silence_interval_s = 0.05

    rec.engine.start_worker()
    try:
        deadline = time.time() + 5.0
        while not rec.steps and time.time() < deadline:
            time.sleep(0.02)
    finally:
        rec.engine.stop_worker()

    assert rec.steps, (
        "the worker emitted no motion with no input published - the engine is on "
        "only while someone is talking"
    )
