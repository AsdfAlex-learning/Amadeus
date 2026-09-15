"""Playback (T7) must publish an incremental position and be interruptible.

The previous implementation synthesised a whole utterance and then did one
blocking ``stream.write`` on the generation thread. Two consequences:

* playback position was only visible when the write *returned*, so nothing could
  use it as a clock — it appeared as a single 0 -> N jump;
* interrupting playback meant stopping another thread's blocking write on a
  shared stream, which is not safe, so interruption did not exist at all.

These tests use an injected stream, so no audio device is required.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from src.runtime.playback import PlaybackWorker

RATE = 16_000


class _FakeStream:
    """Records writes; can be made slow so position can be observed mid-stream."""

    def __init__(self, rate: int, write_delay: float = 0.0) -> None:
        self.rate = rate
        self.writes: list[np.ndarray] = []
        self.closed = False
        self._write_delay = write_delay

    def write(self, payload: bytes) -> None:
        if self._write_delay:
            time.sleep(self._write_delay)
        self.writes.append(np.frombuffer(payload, dtype=np.float32))

    def close(self) -> None:
        self.closed = True


class _Factory:
    def __init__(self, write_delay: float = 0.0) -> None:
        self.rates: list[int] = []
        self.streams: list[_FakeStream] = []
        self._write_delay = write_delay

    def __call__(self, rate: int) -> _FakeStream:
        self.rates.append(rate)
        stream = _FakeStream(rate, self._write_delay)
        self.streams.append(stream)
        return stream


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_playback_writes_in_chunks_at_the_stream_rate() -> None:
    """Writes must be chunked, not one blocking write for the whole utterance."""
    factory = _Factory()
    worker = PlaybackWorker(RATE, stream_factory=factory, chunk_seconds=0.2)
    worker.start()
    try:
        one_second = np.ones(RATE, dtype=np.float32)
        worker.play(7, one_second)
        assert _wait_until(lambda: factory.streams and factory.streams[0].closed), (
            "playback never completed"
        )
    finally:
        worker.shutdown()

    stream = factory.streams[0]
    assert stream.rate == RATE
    assert len(stream.writes) == 5, (
        f"1 s of audio at 0.2 s per chunk should be 5 writes, got {len(stream.writes)}"
    )
    assert sum(w.size for w in stream.writes) == RATE


def test_position_advances_during_playback_not_only_at_the_end() -> None:
    """The position is the clock for Speak-state motion, so it must be incremental."""
    factory = _Factory(write_delay=0.05)
    worker = PlaybackWorker(RATE, stream_factory=factory, chunk_seconds=0.2)
    worker.start()
    observed: list[int] = []
    try:
        worker.play(1, np.ones(RATE, dtype=np.float32))  # 5 chunks x 50 ms
        assert _wait_until(lambda: worker.played_samples > 0)
        observed.append(worker.played_samples)
        assert _wait_until(lambda: factory.streams and factory.streams[0].closed)
    finally:
        worker.shutdown()

    assert observed[0] > 0, "position was still 0 while audio was already being written"


def test_stop_and_flush_interrupts_an_in_progress_utterance() -> None:
    """Interruption must be possible while playback is running."""
    factory = _Factory(write_delay=0.05)
    worker = PlaybackWorker(RATE, stream_factory=factory, chunk_seconds=0.2)
    worker.start()
    try:
        long_audio = np.ones(RATE * 5, dtype=np.float32)  # 5 s, 25 chunks
        worker.play(3, long_audio)
        assert _wait_until(lambda: worker.played_samples > 0)
        worker.stop_and_flush()
        assert _wait_until(lambda: worker.played_samples == 0, timeout=5.0)
    finally:
        worker.shutdown()

    stream = factory.streams[0]
    written = sum(w.size for w in stream.writes)
    assert written < long_audio.size, (
        f"stop_and_flush let the whole 5 s utterance play ({written} samples)"
    )


def test_interruption_writes_a_decaying_tail_rather_than_cutting_hard() -> None:
    """A hard cut clicks; the tail must fade toward silence."""
    factory = _Factory(write_delay=0.05)
    worker = PlaybackWorker(
        RATE, stream_factory=factory, chunk_seconds=0.2, fade_ms=80.0
    )
    worker.start()
    try:
        worker.play(4, np.ones(RATE * 5, dtype=np.float32))
        assert _wait_until(lambda: worker.played_samples > 0)
        worker.stop_and_flush()
        assert _wait_until(lambda: worker.played_samples == 0, timeout=5.0)
    finally:
        worker.shutdown()

    stream = factory.streams[0]
    fade = stream.writes[-1]
    assert fade.size == int(RATE * 0.08), (
        f"expected an 80 ms fade tail, got {fade.size} samples"
    )
    assert fade[0] > fade[-1], "the fade tail does not decay"
    assert fade[-1] == pytest.approx(0.0, abs=1e-6)


def test_a_new_utterance_supersedes_a_pending_one() -> None:
    """A superseded utterance must not be played after the new one."""
    factory = _Factory()
    worker = PlaybackWorker(RATE, stream_factory=factory, chunk_seconds=0.5)
    # Do not start the worker: we are asserting the queueing policy.
    worker.play(1, np.ones(RATE, dtype=np.float32))
    worker.play(2, np.full(RATE, 0.5, dtype=np.float32))

    pending = worker._take_pending()
    assert pending is not None
    stream_id, audio = pending
    assert stream_id == 2, f"expected only the newest utterance to remain, got {stream_id}"
    assert float(audio[0]) == pytest.approx(0.5)
    assert worker._take_pending() is None


def test_shutdown_joins_the_worker() -> None:
    factory = _Factory()
    worker = PlaybackWorker(RATE, stream_factory=factory)
    worker.start()
    thread = worker._thread
    assert thread is not None and thread.is_alive()

    worker.shutdown()

    assert not thread.is_alive(), "playback worker outlived shutdown()"


def test_a_failing_stream_does_not_kill_the_worker_immediately() -> None:
    """One bad write must not silently mute the character forever."""
    attempts: list[int] = []

    def flaky_factory(rate: int):
        attempts.append(rate)
        if len(attempts) == 1:
            raise RuntimeError("device busy")
        return _FakeStream(rate)

    worker = PlaybackWorker(RATE, stream_factory=flaky_factory, chunk_seconds=0.2)
    worker.start()
    try:
        worker.play(1, np.ones(RATE // 2, dtype=np.float32))
        assert _wait_until(lambda: worker.failures == 1), "failure was not recorded"
        worker.play(2, np.ones(RATE // 2, dtype=np.float32))
        assert _wait_until(lambda: len(attempts) >= 2), "worker did not retry after a failure"
    finally:
        worker.shutdown()

    assert attempts, "the factory was never invoked"
