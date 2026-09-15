"""AmadeusWindow shutdown must join the generation thread before PortAudio dies.

``window.py`` started the per-utterance generation thread with
``Thread(...).start()`` and discarded the handle, so nothing could ever join it.
``closeEvent`` then terminated the playback ``PyAudio`` instance while that thread
could be inside a blocking ``stream.write()`` — PortAudio use-after-terminate,
which is a native crash rather than a Python exception.

These are Layer C tests: they construct the real window offscreen with the
model-loading paths stubbed, because the subject is the wiring and the shutdown
sequence, not the models.
"""

from __future__ import annotations

import threading
import time

import pytest
from PySide6.QtGui import QCloseEvent

from src.app.window import AmadeusWindow


@pytest.fixture
def window(config: dict, qapp, monkeypatch):
    """An AmadeusWindow whose model loading is stubbed out.

    Dialogue, motion and TTS loading all pull in model weights and are covered
    elsewhere; what this module tests is how the window wires and tears down.
    """
    monkeypatch.setattr(AmadeusWindow, "_load_dialogue_model", lambda self: None)
    monkeypatch.setattr(AmadeusWindow, "_load_motion_model", lambda self: None)
    monkeypatch.setattr(AmadeusWindow, "_load_tts", lambda self: None)
    monkeypatch.setattr(AmadeusWindow, "_connect_camera", lambda self: None)
    win = AmadeusWindow(config)
    yield win
    win.closeEvent(QCloseEvent())


def _install_slow_generation(window, monkeypatch) -> threading.Thread:
    """Start a generation that yields slowly and can be cancelled between chunks."""
    monkeypatch.setattr(
        type(window.tts_engine), "is_loaded", property(lambda self: False)
    )

    def slow_generate(_messages):
        for _ in range(200):
            yield "x"
            time.sleep(0.05)

    monkeypatch.setattr(window.dialogue_model, "generate", slow_generate)
    window._start_generation()
    thread = window._generation_thread
    assert thread is not None, "the generation thread handle must be retained"
    time.sleep(0.2)
    assert thread.is_alive(), "generation never started"
    return thread


def test_generation_thread_handle_is_retained(window, monkeypatch) -> None:
    """Without the handle, joining is impossible in principle."""
    monkeypatch.setattr(window.dialogue_model, "generate", lambda _m: iter(["hi"]))
    window._start_generation()

    assert window._generation_thread is not None
    window._generation_thread.join(timeout=5.0)


def test_shutdown_joins_generation_before_stopping_playback(window, monkeypatch) -> None:
    """T7 must not be stopped while the generation thread is still running.

    T5 is the thread that submits audio to playback, so tearing playback down first
    would let a live generation publish into a worker that is being closed. This is
    the same ordering requirement that previously applied to terminating PyAudio —
    playback simply owns its own stream now.
    """
    thread = _install_slow_generation(window, monkeypatch)

    # Record whether the generation thread was still running when playback stopped.
    alive_when_stopped: list[bool] = []
    monkeypatch.setattr(
        window._playback,
        "shutdown",
        lambda: alive_when_stopped.append(thread.is_alive()),
    )

    window.closeEvent(QCloseEvent())

    assert not thread.is_alive(), "shutdown left the generation thread running"
    assert alive_when_stopped == [False], (
        f"playback was stopped while the generation thread was alive "
        f"(observed {alive_when_stopped})"
    )


def test_shutdown_is_idempotent(window, monkeypatch) -> None:
    """A second close must be safe, and must leave no worker running."""
    _install_slow_generation(window, monkeypatch)
    worker = window._playback._thread

    window.closeEvent(QCloseEvent())
    window.closeEvent(QCloseEvent())  # must not raise

    assert worker is not None and not worker.is_alive(), (
        "the playback worker outlived a double shutdown"
    )
