"""Speak state must follow the playback clock, not the inference cadence (S2.4).

The design that was reviewed and rejected read its window as
``audio[played : played + chunk]``. That *skips* every un-inferred window between
the cursor and the playback position — the mid-utterance skip the design exists to
avoid. Playback **gates** the advance; it does not **index** it. These tests pin
that distinction, plus the unit conversion and the Speak-state input contract.
"""

from __future__ import annotations

import numpy as np
import pytest

RATE = 16_000


class _FakePlayback:
    """Stands in for T7: reports a playback position and its own sample rate."""

    def __init__(self, played: int = 0) -> None:
        self.played_samples = played


def _engine(config: dict, steps: list[tuple[np.ndarray, np.ndarray]]):
    """Engine with the model boundary replaced, recording (user, tts) chunks."""
    from src.motion.inference import DiffusionMotionInference

    def _infer(user_chunk, tts_chunk, _visual_frames):
        steps.append((np.array(user_chunk), np.array(tts_chunk)))
        return None

    engine = DiffusionMotionInference(config)
    engine._loaded = True
    engine._model = object()
    engine._diffusion_infer = _infer  # type: ignore[method-assign]
    return engine


def _ramp(seconds: float) -> np.ndarray:
    n = int(RATE * seconds)
    return (np.arange(n, dtype=np.float32) / n).astype(np.float32)


def test_cursor_reads_at_its_own_position_not_at_playback(config: dict) -> None:
    """The read window must start at the cursor, never at the playback position."""
    steps: list[tuple[np.ndarray, np.ndarray]] = []
    engine = _engine(config, steps)
    # Playback is 2 s ahead, but the lag bound is generous so no skip occurs.
    engine.attach_playback(_FakePlayback(played=2 * RATE), playback_rate=RATE)
    engine._max_lag_s = 10.0

    engine.process_tts_audio(_ramp(4.0))

    assert steps, "no inference step ran"
    _user, tts_chunk = steps[0]
    # The first window must be audio[0:1s] (values near 0), not audio[2s:3s]
    # (values near 0.5) which is what indexing at `played` would produce.
    assert tts_chunk[0] == pytest.approx(0.0, abs=1e-6), (
        f"the first window started at {tts_chunk[0]:.3f}; indexing at the playback "
        f"position instead of the cursor would start near 0.5"
    )


def test_advance_is_gated_by_playback_with_a_bounded_lookahead(config: dict) -> None:
    """With nothing played, only the lookahead's worth of audio may be inferred."""
    steps: list[tuple[np.ndarray, np.ndarray]] = []
    engine = _engine(config, steps)
    engine.attach_playback(_FakePlayback(played=0), playback_rate=RATE)
    engine.SPEECH_LOOKAHEAD_SECONDS = 1.0

    engine.process_tts_audio(_ramp(4.0))

    assert len(steps) == 1, (
        f"with 0 samples played and 1 s of lookahead, exactly one 1 s step should "
        f"have run; got {len(steps)}"
    )
    engine._motion_wake  # noqa: B018 - attribute presence only


def test_more_playback_unlocks_more_inference(config: dict) -> None:
    """Advancing the playback position must let the next window run."""
    steps: list[tuple[np.ndarray, np.ndarray]] = []
    engine = _engine(config, steps)
    playback = _FakePlayback(played=0)
    engine.attach_playback(playback, playback_rate=RATE)
    engine.SPEECH_LOOKAHEAD_SECONDS = 1.0

    engine.process_tts_audio(_ramp(4.0))
    assert len(steps) == 1

    playback.played_samples = RATE  # one second played
    # A single sample is enough to wake the engine; queueing a full second would
    # legitimately add a Listen step once the Speak gate closes.
    engine.process_user_audio(np.zeros(1, dtype=np.float32))
    assert len(steps) == 2, (
        f"after 1 s of playback the next window should be inferable; got {len(steps)} steps"
    )


def test_excessive_lag_skips_forward_and_is_counted(config: dict) -> None:
    """Lag must be bounded relative to what the user hears, and the loss visible."""
    steps: list[tuple[np.ndarray, np.ndarray]] = []
    engine = _engine(config, steps)
    engine.attach_playback(_FakePlayback(played=9 * RATE), playback_rate=RATE)
    engine._max_lag_s = 1.5

    # Declare the true source rate, otherwise the 20 s of samples are treated as
    # 22050 Hz audio and resampled to a 14.5 s stream.
    engine.process_tts_audio(_ramp(20.0), sample_rate=RATE)

    # The cursor should jump to played - max_lag = 7.5 s, not to 9 s.
    assert engine.speech_skipped_samples == pytest.approx(7.5 * RATE, abs=2 * RATE), (
        f"expected a bounded skip to 7.5 s, counted {engine.speech_skipped_samples} samples"
    )
    _user, tts_chunk = steps[0]
    assert tts_chunk[0] == pytest.approx(7.5 / 20.0, abs=0.01), (
        f"the resumed window starts at {tts_chunk[0]:.3f}, expected ~0.375 (7.5 s in)"
    )


def test_playback_position_is_converted_to_the_pipeline_rate(config: dict) -> None:
    """The playback position is counted at the TTS rate and must be converted.

    Mixing the two rates is the C-5 bug class: 1 s of playback at 22050 Hz is not
    1 s at 16000 Hz.
    """
    steps: list[tuple[np.ndarray, np.ndarray]] = []
    engine = _engine(config, steps)
    tts_rate = int(config["tts"]["sample_rate"])
    # One second of playback at the TTS rate == one second at the pipeline rate.
    engine.attach_playback(_FakePlayback(played=tts_rate), playback_rate=tts_rate)
    engine.SPEECH_LOOKAHEAD_SECONDS = 0.0  # only played audio may be inferred

    engine.process_tts_audio(_ramp(4.0))

    assert len(steps) == 1, (
        f"1 s of playback at {tts_rate} Hz should unlock exactly 1 s at "
        f"{RATE} Hz; got {len(steps)} steps (a unit mix-up would unlock ~1.38 s "
        f"or only ~0.73 s)"
    )


def test_microphone_is_zeroed_while_speaking(config: dict) -> None:
    """During an utterance the model must get user_audio = 0 (the Speak state).

    Feeding mic and TTS audio simultaneously is a fourth state that appears in no
    specification (CONTEXT.md allows at most one non-zero input).
    """
    steps: list[tuple[np.ndarray, np.ndarray]] = []
    engine = _engine(config, steps)
    engine.attach_playback(_FakePlayback(played=0), playback_rate=RATE)
    engine.SPEECH_LOOKAHEAD_SECONDS = 1.0

    engine.process_tts_audio(_ramp(2.0))
    # Queue mic audio that would otherwise trigger a Listen step.
    engine._audio_buffer.extend([0.5] * RATE * 4)
    engine.process_user_audio(np.zeros(RATE, dtype=np.float32))

    speaking_steps = [s for s in steps if float(np.abs(s[1]).sum()) > 0]
    assert speaking_steps, "no Speak step ran"
    for user_chunk, _tts in speaking_steps:
        assert float(np.abs(user_chunk).sum()) == 0.0, (
            "microphone audio was fed to the model while the character was speaking"
        )


def test_a_new_utterance_supersedes_a_pending_one_not_the_active_one(config: dict) -> None:
    """Dropping the *active* stream would jump mid-utterance."""
    steps: list[tuple[np.ndarray, np.ndarray]] = []
    engine = _engine(config, steps)
    engine.attach_playback(_FakePlayback(played=0), playback_rate=RATE)
    engine.SPEECH_LOOKAHEAD_SECONDS = 1.0

    engine.process_tts_audio(_ramp(4.0))  # stream 1 promoted
    assert engine.speech_active
    engine.process_tts_audio(np.zeros(RATE, dtype=np.float32))  # stream 2 pending

    pending = engine._tts_pending.peek()
    assert pending is not None and pending[0] == 2, (
        "the pending utterance should be superseded, never silently dropped"
    )
    assert engine._speech is not None and engine._speech[0] == 1, (
        "the active stream was replaced; that would skip mid-utterance"
    )
