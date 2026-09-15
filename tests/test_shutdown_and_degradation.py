"""Shutdown and degradation completeness (S2.6).

Three gaps, one of which is a composite that only appears when earlier fixes are
combined:

* ``TTSEngine`` had no ``cleanup`` at all and no caller touched it, so a backend
  holding loaded models was abandoned at shutdown.
* The audio drain worker (T1) logged a failing consumer forever instead of giving
  up visibly.
* **Playback failure froze motion permanently.** Speak state is gated on the
  playback position (S2.4), and the renderer holds its last applied pose on
  underflow (S2.1). Together with a tripped playback breaker those produce an
  unrecoverable statue: position stops advancing, motion stops, and the widget
  holds the final speaking pose forever.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from PySide6.QtGui import QCloseEvent

from src.audio.capture import MicrophoneCapture


# ── TTS engine lifecycle ─────────────────────────────────────────────────────


def test_tts_cleanup_releases_the_backend(config: dict) -> None:
    from src.tts.engine import TTSEngine

    engine = TTSEngine(config)
    engine._model = object()
    engine._loaded = True

    engine.cleanup()

    assert engine._model is None
    assert engine.is_loaded is False, "the engine still reports itself as loaded"


def test_tts_cleanup_calls_a_backend_closer_when_present(config: dict) -> None:
    """Backends needing real teardown expose close(); it must be invoked."""
    from src.tts.engine import TTSEngine

    class _Closable:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    engine = TTSEngine(config)
    backend = _Closable()
    engine._model = backend
    engine._loaded = True

    engine.cleanup()

    assert backend.closed, "the backend closer was not called"


def test_silence_uses_the_engine_sample_rate(config: dict) -> None:
    """A regression guard: silence must match the advertised rate, not 16000."""
    from src.tts.engine import TTSEngine

    engine = TTSEngine(config)
    silence = engine._silence(1.0)
    assert silence.size == engine.sample_rate, (
        f"1 s of silence is {silence.size} samples at {engine.sample_rate} Hz"
    )


# ── T1 (drain worker) breaker ────────────────────────────────────────────────


def test_drain_worker_stops_after_repeated_consumer_failures(config: dict) -> None:
    """A permanently failing consumer must be surfaced, not logged forever."""
    capture = MicrophoneCapture(config)
    capture._running = True

    def broken(_audio: np.ndarray) -> None:
        raise RuntimeError("consumer is broken")

    capture.on_audio(broken)
    capture._start_worker()
    worker = capture._worker
    assert worker is not None

    try:
        deadline = time.time() + 10.0
        while worker.is_alive() and time.time() < deadline:
            capture._ring.write(np.ones(capture.chunk_size, dtype=np.float32))
            capture._wake.set()
            time.sleep(0.01)
    finally:
        capture.cleanup()

    assert not worker.is_alive(), (
        "the drain worker kept running after its consumer failed on every chunk"
    )


def test_drain_worker_survives_an_occasional_failure(config: dict) -> None:
    """A transient failure must not stop the worker."""
    capture = MicrophoneCapture(config)
    capture._running = True
    delivered: list[np.ndarray] = []
    calls = {"n": 0}

    def flaky(audio: np.ndarray) -> None:
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise RuntimeError("occasional failure")
        delivered.append(audio)

    capture.on_audio(flaky)
    capture._start_worker()
    worker = capture._worker
    assert worker is not None

    try:
        for _ in range(12):
            capture._ring.write(np.ones(capture.chunk_size, dtype=np.float32))
            capture._wake.set()
            time.sleep(0.02)
    finally:
        capture.cleanup()

    assert delivered, "healthy deliveries stopped after transient failures"


# ── playback failure must not freeze motion ──────────────────────────────────


class _FailedPlayback:
    played_samples = 0
    failed = True


def test_playback_failure_drops_the_speech_stream(config: dict) -> None:
    """A dead playback worker must not leave motion gated forever."""
    from src.motion.inference import DiffusionMotionInference

    steps: list[int] = []
    engine = DiffusionMotionInference(config)
    engine._loaded = True
    engine._model = object()
    engine._diffusion_infer = lambda *_a, **_k: (steps.append(1), None)[1]  # type: ignore[method-assign]
    engine.attach_playback(_FailedPlayback(), playback_rate=int(config["tts"]["sample_rate"]))

    engine.process_tts_audio(np.zeros(16000, dtype=np.float32))
    assert not engine.speech_active, (
        "the utterance was retained even though playback can never advance, so "
        "motion would hold its last frame forever"
    )
    assert not steps, "an inference step ran for an utterance that cannot be played"


def test_playback_failure_falls_back_to_listen(config: dict) -> None:
    """After degradation the engine must still respond to the microphone."""
    from src.motion.inference import DiffusionMotionInference

    steps: list[tuple[np.ndarray, np.ndarray]] = []

    def _infer(user_chunk, tts_chunk, _visual):
        steps.append((np.array(user_chunk), np.array(tts_chunk)))
        return None

    engine = DiffusionMotionInference(config)
    engine._loaded = True
    engine._model = object()
    engine._diffusion_infer = _infer  # type: ignore[method-assign]
    engine.attach_playback(_FailedPlayback(), playback_rate=int(config["tts"]["sample_rate"]))

    engine.process_tts_audio(np.zeros(16000, dtype=np.float32))
    engine.process_user_audio(np.full(16000, 0.3, dtype=np.float32))

    assert steps, "the engine stopped responding after playback failed"
    user_chunk, tts_chunk = steps[-1]
    assert float(np.abs(tts_chunk).sum()) == 0.0, "a TTS chunk was fed after playback failed"
    assert float(np.abs(user_chunk).sum()) > 0.0, "the microphone was not used as a fallback"


# ── window shutdown ──────────────────────────────────────────────────────────


@pytest.fixture
def window(config: dict, qapp, monkeypatch):
    from src.app.window import AmadeusWindow

    monkeypatch.setattr(AmadeusWindow, "_load_dialogue_model", lambda self: None)
    monkeypatch.setattr(AmadeusWindow, "_load_motion_model", lambda self: None)
    monkeypatch.setattr(AmadeusWindow, "_load_tts", lambda self: None)
    monkeypatch.setattr(AmadeusWindow, "_connect_camera", lambda self: None)
    win = AmadeusWindow(config)
    yield win
    win.closeEvent(QCloseEvent())


def test_shutdown_cleans_up_the_tts_engine(window, monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(window.tts_engine, "cleanup", lambda: calls.append(1))

    window.closeEvent(QCloseEvent())

    assert calls == [1], "TTSEngine.cleanup was not called exactly once at shutdown"


def test_shutdown_leaves_no_worker_threads(window) -> None:
    """Every worker must be joined, not abandoned."""
    workers = {
        "playback": window._playback._thread,
        "audio-drain": window.audio_pipeline.capture._worker,
        "asr": window.audio_pipeline._asr_worker,
    }

    window.closeEvent(QCloseEvent())

    alive = {name: t for name, t in workers.items() if t is not None and t.is_alive()}
    assert not alive, f"threads outlived shutdown: {sorted(alive)}"
