"""Generation lifecycle: one shared cancel flag and one un-gated completion
callback are both insufficient once a generation can be superseded.

Two concrete races, both found in review:

* **Shared cancel flag.** Interruption sets the flag, the old generation breaks,
  a replacement starts — and immediately observes the *same* flag still set, so
  it aborts on its first chunk. The interrupting utterance gets no reply.
* **Un-gated completion.** The superseded generation's ``finally`` emits
  ``response_done``, which clears ``_generating`` — the flag that now belongs to
  its replacement. A third utterance can then start a third concurrent generation
  sharing playback and the conversation context.
"""

from __future__ import annotations

import threading
import time

import pytest
from PySide6.QtGui import QCloseEvent

from src.app.window import AmadeusWindow
from src.dialogue.model import DialogueUnavailableError


@pytest.fixture
def window(config: dict, qapp, monkeypatch):
    """AmadeusWindow with the model-loading paths stubbed (see test_window_shutdown)."""
    monkeypatch.setattr(AmadeusWindow, "_load_dialogue_model", lambda self: None)
    monkeypatch.setattr(AmadeusWindow, "_load_motion_model", lambda self: None)
    monkeypatch.setattr(AmadeusWindow, "_load_tts", lambda self: None)
    monkeypatch.setattr(AmadeusWindow, "_connect_camera", lambda self: None)
    win = AmadeusWindow(config)
    yield win
    win.closeEvent(QCloseEvent())


def _slow_generator(chunks: int = 50, delay: float = 0.02):
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


def test_speak_response_feeds_motion_and_playback_the_same_stream_id(window, monkeypatch) -> None:
    """The Speak-state clock and playback must agree on stream identity.

    The motion engine tells "still playing" from "played out" by matching its
    utterance id against the playback worker's published stream ids. If the two
    sides numbered streams independently, every id comparison would compare
    unrelated counters and the zero-padded tail drain could never fire.
    """
    import numpy as np

    fed_ids: list[int | None] = []
    played_ids: list[int] = []

    class _TTS:
        is_loaded = True

        def synthesize(self, _text: str):
            return np.zeros(2, dtype=np.float32)

        def cleanup(self):
            pass

    class _Motion:
        def process_tts_audio(self, _audio, sample_rate=None, stream_id=None):
            fed_ids.append(stream_id)

        def cleanup(self):
            pass

    class _Playback:
        def play(self, stream_id, _audio):
            played_ids.append(stream_id)

        def shutdown(self):
            pass

    monkeypatch.setattr(window, "tts_engine", _TTS())
    monkeypatch.setattr(window, "motion_model", _Motion())
    monkeypatch.setattr(window, "_playback", _Playback())

    window._speak_response("hello")
    window._speak_response("world")

    assert fed_ids == played_ids, (
        f"motion got stream ids {fed_ids} while playback got {played_ids}; "
        f"both clocks must be keyed by the same id"
    )
    assert len(set(played_ids)) == 2, "consecutive utterances must get distinct ids"


def test_a_replacement_generation_is_not_cancelled_by_the_previous_cancel(
    window, monkeypatch
) -> None:
    """Cancelling generation N must not abort generation N+1.

    With a single shared Event, the replacement sees the flag still set and
    breaks on its first chunk, so the interrupting utterance is never answered.
    """
    monkeypatch.setattr(window.dialogue_model, "generate", _slow_generator())
    monkeypatch.setattr(type(window.tts_engine), "is_loaded", property(lambda self: False))

    window._start_generation()
    first_id = window._generation_id
    assert _wait_until(lambda: window._generation_thread is not None)

    window._cancel_generation(first_id)

    # A replacement generation, as interruption would start.
    window._generating = False
    window._start_generation()
    replacement_id = window._generation_id
    assert replacement_id != first_id

    assert not window._is_generation_cancelled(replacement_id), (
        "the replacement generation inherited the previous generation's cancellation"
    )

    # It must actually produce output rather than breaking on its first chunk.
    assert _wait_until(lambda: window._generation_thread is not None), "no thread"


def test_a_superseded_completion_does_not_clear_the_current_flag(window, monkeypatch) -> None:
    """The C4 race: the old generation's completion must not release the new one."""
    monkeypatch.setattr(window.dialogue_model, "generate", _slow_generator())
    monkeypatch.setattr(type(window.tts_engine), "is_loaded", property(lambda self: False))

    window._start_generation()
    superseded_id = window._generation_id

    # Supersede it.
    window._cancel_generation(superseded_id)
    window._generating = False
    window._start_generation()
    current_id = window._generation_id
    assert current_id != superseded_id

    # The superseded generation's completion callback arrives (queued signal).
    window._on_response_done(superseded_id)

    assert window._generating is True, (
        "a superseded generation's completion cleared the current generation's flag; "
        "a third utterance could now start a third concurrent generation"
    )


def test_the_current_generation_completion_does_clear_the_flag(window, monkeypatch) -> None:
    """The gating must not break the normal path."""
    monkeypatch.setattr(window.dialogue_model, "generate", lambda _m, **_kw: iter(["hi"]))
    monkeypatch.setattr(type(window.tts_engine), "is_loaded", property(lambda self: False))

    window._start_generation()
    current_id = window._generation_id
    assert _wait_until(lambda: not window._generation_thread.is_alive())

    window._on_response_done(current_id)
    assert window._generating is False


def test_cancelling_does_not_affect_a_generation_that_has_not_started(window) -> None:
    """Ids are per-generation, so cancelling a stale id is harmless."""
    window._cancel_generation(999)
    assert not window._is_generation_cancelled(window._generation_id)


def test_shutdown_cancels_every_generation(window, monkeypatch) -> None:
    monkeypatch.setattr(window.dialogue_model, "generate", _slow_generator())
    monkeypatch.setattr(type(window.tts_engine), "is_loaded", property(lambda self: False))
    window._start_generation()
    generation_id = window._generation_id

    window.closeEvent(QCloseEvent())

    assert window._is_generation_cancelled(generation_id)


def test_a_truncated_turn_is_kept_in_context(window, monkeypatch) -> None:
    """The user heard part of it; dropping it falsifies the model's history."""
    monkeypatch.setattr(window.dialogue_model, "generate", _slow_generator(chunks=200))
    monkeypatch.setattr(type(window.tts_engine), "is_loaded", property(lambda self: False))

    window._handle_text("hello")
    assert _wait_until(lambda: window._generation_thread is not None)
    time.sleep(0.1)  # let it emit a few chunks

    window._cancel_generation(window._generation_id)
    assert _wait_until(lambda: not window._generation_thread.is_alive())

    assistant_turns = [m for m in window.context.get_messages() if m["role"] == "assistant"]
    assert assistant_turns, "the truncated assistant turn was discarded from context"


def test_backend_error_is_displayed_out_of_band(window, monkeypatch, qapp) -> None:
    """ADR 0005 D8: backend diagnostics are shown but never enter the history
    as if the character had said them (and are never spoken)."""

    def boom(_messages, **_kw):
        raise DialogueUnavailableError("测试原因")
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(window.dialogue_model, "generate", boom)
    monkeypatch.setattr(type(window.tts_engine), "is_loaded", property(lambda self: False))

    window._start_generation()
    assert _wait_until(
        lambda: window._generation_thread is not None and not window._generation_thread.is_alive()
    )
    window._on_response_done(window._generation_id)
    qapp.processEvents()

    assert "测试原因" in window.conversation_display.toPlainText()
    assert window.context.get_messages() == [], (
        "a backend diagnostic was written into the conversation history"
    )


def test_empty_turn_is_not_added_to_history(window, monkeypatch) -> None:
    """A cancelled-before-first-chunk turn is not a turn (ADR 0005, F-9)."""
    monkeypatch.setattr(window.dialogue_model, "generate", lambda _m, **_kw: iter([]))
    monkeypatch.setattr(type(window.tts_engine), "is_loaded", property(lambda self: False))

    window._start_generation()
    assert _wait_until(
        lambda: window._generation_thread is not None and not window._generation_thread.is_alive()
    )
    assert window.context.get_messages() == [], (
        "an empty turn reached the history as an empty assistant message"
    )


def test_cancel_generation_sets_the_stream_cancel_event(window, monkeypatch) -> None:
    """The per-generation event must be created, handed to generate(), set on
    cancel, and cleaned up afterwards."""
    release = threading.Event()
    seen: dict = {}

    def fake(_messages, cancel_event=None, **_kw):
        seen["event"] = cancel_event
        yield "x"
        release.wait(timeout=5)

    monkeypatch.setattr(window.dialogue_model, "generate", fake)
    monkeypatch.setattr(type(window.tts_engine), "is_loaded", property(lambda self: False))

    window._start_generation()
    generation_id = window._generation_id
    assert _wait_until(lambda: seen.get("event") is not None), "cancel_event not passed"

    window._cancel_generation(generation_id)
    assert seen["event"].is_set(), "_cancel_generation did not set the stream cancel event"

    release.set()
    assert _wait_until(
        lambda: window._generation_thread is not None and not window._generation_thread.is_alive()
    )
    assert generation_id not in window._generation_cancel_events, "cancel event not cleaned up"


def test_a_second_turn_while_a_generation_is_in_flight_is_answered_not_dropped(
    window, monkeypatch
) -> None:
    """A turn arriving mid-generation must supersede, not vanish.

    The second utterance's onset fires before its transcript exists, so it
    interrupts without starting anything; the late first transcript then starts
    a generation, and the second transcript must take over instead of hitting
    the busy guard - a dropped turn is a user message recorded in the context
    that never receives any answer.
    """
    monkeypatch.setattr(type(window.tts_engine), "is_loaded", property(lambda self: False))
    calls: list[list[str]] = []

    def slow(_messages, **_kw):
        calls.append([m["content"] for m in _messages])
        for _ in range(50):
            yield "x"
            time.sleep(0.02)

    monkeypatch.setattr(window.dialogue_model, "generate", slow)

    window._handle_text("first")
    assert _wait_until(lambda: len(calls) == 1), "the first turn never started a generation"
    first_id = window._generation_id

    window._handle_text("second")
    assert _wait_until(lambda: len(calls) == 2, timeout=2.0), (
        "the second turn never started a generation; its user message is recorded "
        "but will never be answered"
    )
    assert "second" in calls[1], "the replacement generation must answer the second turn"
    assert window._is_generation_cancelled(first_id), "the superseded generation was not cancelled"
