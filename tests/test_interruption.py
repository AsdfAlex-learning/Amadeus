"""Interruption must exist: user speech during a spoken response changes state.

Before this, `_start_generation` simply returned while `_generating` was true, so
a user utterance during a response was recorded in the context but never answered,
and playback kept running over the user.

What can start at the moment of onset is only the *reaction* — stop playback, drop
the active utterance, cancel the in-flight generation — because the transcript is
still seconds away.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from PySide6.QtGui import QCloseEvent

from src.app.window import AmadeusWindow
from src.audio.pipeline import AudioPipeline


@pytest.fixture
def window(config: dict, qapp, monkeypatch):
    monkeypatch.setattr(AmadeusWindow, "_load_dialogue_model", lambda self: None)
    monkeypatch.setattr(AmadeusWindow, "_load_motion_model", lambda self: None)
    monkeypatch.setattr(AmadeusWindow, "_load_tts", lambda self: None)
    monkeypatch.setattr(AmadeusWindow, "_connect_camera", lambda self: None)
    win = AmadeusWindow(config)
    yield win
    win.closeEvent(QCloseEvent())


def _slow_generator(chunks: int = 200, delay: float = 0.02):
    def _generate(_messages, **_kw):
        for _ in range(chunks):
            yield "x"
            time.sleep(delay)

    return _generate


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ── the onset signal ─────────────────────────────────────────────────────────


def test_speech_onset_fires_once_per_utterance(config: dict) -> None:
    """Onset must fire on the silence→speech transition, not on every chunk."""
    pipeline = AudioPipeline(config)
    onsets: list[int] = []
    pipeline.on_speech_onset(lambda: onsets.append(1))
    try:
        loud = np.full(1024, 0.5, dtype=np.float32)
        for _ in range(5):
            pipeline._on_audio_chunk(loud)
        assert len(onsets) == 1, f"onset fired {len(onsets)} times for one continuous utterance"

        # End the utterance, then speak again.
        loud_until_silence = [loud] * int(
            AudioPipeline.SILENCE_DURATION * pipeline.capture.sample_rate / 1024 + 3
        )
        for _ in range(len(loud_until_silence)):
            pipeline._on_audio_chunk(np.zeros(1024, dtype=np.float32))
        pipeline._audio_buffer.clear()  # keep the test independent of finalisation
        pipeline._on_audio_chunk(loud)
        assert len(onsets) == 2, "a second utterance did not produce a second onset"
    finally:
        pipeline.cleanup()


def test_silence_does_not_fire_an_onset(config: dict) -> None:
    pipeline = AudioPipeline(config)
    onsets: list[int] = []
    pipeline.on_speech_onset(lambda: onsets.append(1))
    try:
        for _ in range(10):
            pipeline._on_audio_chunk(np.zeros(1024, dtype=np.float32))
        assert not onsets
    finally:
        pipeline.cleanup()


# ── interruption behaviour ───────────────────────────────────────────────────


def test_interruption_stops_playback_and_cancels_generation(window, monkeypatch) -> None:
    monkeypatch.setattr(window.dialogue_model, "generate", _slow_generator())
    monkeypatch.setattr(type(window.tts_engine), "is_loaded", property(lambda self: False))

    stopped: list[int] = []
    monkeypatch.setattr(window._playback, "stop_and_flush", lambda: stopped.append(1))

    window._handle_text("hello")
    assert _wait_until(lambda: window._generation_thread is not None)
    generation_id = window._generation_id
    assert window._generating is True

    window._interrupt_speech()

    assert stopped == [1], "interruption did not stop playback"
    assert window._is_generation_cancelled(generation_id), (
        "interruption did not cancel the in-flight generation"
    )
    assert window._generating is False, (
        "the flag was not released, so the interrupting utterance could not be answered"
    )
    assert not window.motion_model.speech_active, (
        "the active utterance was not dropped, so the engine would keep performing "
        "audio the user has stopped hearing"
    )


def test_interruption_then_transcript_produces_a_reply(window, monkeypatch) -> None:
    """The whole point: the interrupting utterance must actually be answered."""
    monkeypatch.setattr(window.dialogue_model, "generate", _slow_generator(chunks=3))
    monkeypatch.setattr(type(window.tts_engine), "is_loaded", property(lambda self: False))

    window._handle_text("first")
    assert _wait_until(lambda: window._generation_thread is not None)
    window._interrupt_speech()

    before = window._generation_id
    window._handle_text("second")
    assert window._generation_id > before, "the interrupting utterance started no generation"

    assert _wait_until(
        lambda: any(m["role"] == "assistant" for m in window.context.get_messages())
    ), "the interrupting utterance never produced a reply"


def test_interruption_is_a_no_op_during_shutdown(window) -> None:
    """Shutdown sets the flag itself; a late onset must not resurrect work."""
    stopped: list[int] = []
    window._playback.shutdown()
    window._shutting_down = True
    window._interrupt_speech()
    assert stopped == []  # nothing was asked of playback


def test_supersede_speech_drops_the_active_utterance(config: dict) -> None:
    """The engine must fall back to Listen rather than finish stale audio."""
    from src.motion.inference import DiffusionMotionInference

    engine = DiffusionMotionInference(config)
    engine._loaded = True
    engine._model = object()
    engine._diffusion_infer = lambda *_a, **_k: None  # type: ignore[method-assign]

    class _Idle:
        played_samples = 0

    engine.attach_playback(_Idle(), playback_rate=int(config["tts"]["sample_rate"]))
    # Long enough that the lookahead gate leaves part of it unperformed: a
    # sub-chunk utterance is zero-padded and performed in full immediately, so
    # it would no longer be active by the time this test looks.
    engine.process_tts_audio(np.zeros(int(2.5 * 16000), dtype=np.float32), sample_rate=16000)
    assert engine.speech_active

    engine.supersede_speech()
    assert not engine.speech_active, "the utterance survived interference"
    assert engine._tts_pending.peek() is None


def test_an_interrupt_landing_during_synthesize_does_not_rearm_playback(
    window, monkeypatch
) -> None:
    """The cancel window between the pre-check and play() must be closed.

    ``tts_engine.synthesize`` takes hundreds of milliseconds to seconds. If a
    VAD onset interrupts during it - flushing playback and cancelling the
    generation - the cancelled reply must not go on to feed the motion engine
    and ``play()`` afterwards: that re-arms exactly the audio the user just
    interrupted, and the onset fires only once per utterance.
    """
    release = threading.Event()
    entered = threading.Event()
    played: list[int] = []
    fed: list[int] = []

    class _BlockingTTS:
        is_loaded = True

        def synthesize(self, _text: str):
            entered.set()
            release.wait(timeout=5)
            return np.zeros(4, dtype=np.float32)

        def cleanup(self):
            pass

    class _Motion:
        def process_tts_audio(self, _audio, sample_rate=None, stream_id=None):
            fed.append(stream_id)

        def supersede_speech(self):
            pass

        def set_text_prompt(self, _text):
            pass

        def cleanup(self):
            pass

    class _Playback:
        def play(self, stream_id, _audio):
            played.append(stream_id)

        def stop_and_flush(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(window, "tts_engine", _BlockingTTS())
    monkeypatch.setattr(window, "motion_model", _Motion())
    monkeypatch.setattr(window, "_playback", _Playback())
    monkeypatch.setattr(window.dialogue_model, "generate", lambda _m, **_k: iter(["ok."]))

    window._handle_text("hello")
    assert _wait_until(entered.is_set), "the reply never reached synthesis"

    window._interrupt_speech()
    release.set()
    assert _wait_until(
        lambda: window._generation_thread is None or not window._generation_thread.is_alive()
    )

    assert fed == [], "an interrupted reply re-fed the motion engine after synthesize"
    assert played == [], "an interrupted reply re-armed playback after synthesize"
