"""Playback worker (T7): the timing master for Speak-state motion.

Playback is the only clock the user's ears follow, so Speak-state motion must be
derived from the audio actually being heard rather than from the inference
cadence. That requires two things the previous inline implementation could not
provide:

* **An incremental position.** A single blocking ``stream.write`` of a whole
  utterance makes the position visible only at the end — one atomic 0 → N jump —
  which reproduces the existing behaviour plus a delay. Writes are therefore
  chunked (≤200 ms by default) and ``played_samples`` advances per chunk.
* **A stoppable stream.** You cannot safely interrupt another thread's single
  blocking write on a shared stream, so playback owns its own thread and stream,
  and interruption is an async flag checked between chunks (with a short fade so
  stopping does not click).

Owning the stream here also removes the old check-then-create race on a shared
``pyaudio.PyAudio`` instance between concurrent generation threads.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger


class PyAudioOutputStream:
    """An output stream that owns its own ``PyAudio`` instance.

    Creating it on the playback thread keeps device setup off the generation
    thread, and means there is exactly one owner to tear down.
    """

    def __init__(self, sample_rate: int) -> None:
        import pyaudio

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            output=True,
        )

    def write(self, payload: bytes) -> None:
        self._stream.write(payload)

    def close(self) -> None:
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._pa.terminate()


class PlaybackWorker:
    """Owns the output stream and publishes ``(stream_id, played_samples)``."""

    DEFAULT_CHUNK_SECONDS = 0.2
    #: Interruption fade. A hard cut clicks; 80 ms is inaudible as a cut.
    DEFAULT_FADE_MS = 80.0
    MAX_CONSECUTIVE_FAILURES = 5
    _JOIN_TIMEOUT_S = 2.0
    _POLL_S = 0.02

    def __init__(
        self,
        sample_rate: int,
        stream_factory: Callable[[int], Any] | None = None,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        fade_ms: float = DEFAULT_FADE_MS,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self._chunk_samples = max(1, int(self.sample_rate * float(chunk_seconds)))
        self._fade_samples = max(0, int(self.sample_rate * float(fade_ms) / 1000.0))
        self._factory: Callable[[int], Any] = stream_factory or PyAudioOutputStream

        self._lock = threading.Lock()
        self._pending: tuple[int, np.ndarray] | None = None
        self._wake = threading.Event()
        self._stop_all = threading.Event()
        self._interrupt = threading.Event()
        self._thread: threading.Thread | None = None

        self._played = 0
        self._current_stream_id: int | None = None
        self._failures = 0
        self._failed = False
        self._underruns = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_all.clear()
        self._thread = threading.Thread(
            target=self._loop, name="amadeus-playback", daemon=True
        )
        self._thread.start()

    def shutdown(self) -> None:
        """Stop playback and join the worker.

        Escalation policy (ADR 0003): on timeout, leak the daemon rather than let
        a caller release a stream it may still be writing to.
        """
        self._interrupt.set()
        self._stop_all.set()
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is None:
            return
        thread.join(timeout=self._JOIN_TIMEOUT_S)
        if thread.is_alive():
            logger.warning(
                "Playback worker did not stop within %.0f s; leaking it (daemon)",
                self._JOIN_TIMEOUT_S,
            )
            self._thread = thread

    # ── producer API ─────────────────────────────────────────────────────────

    def play(self, stream_id: int, audio: np.ndarray) -> None:
        """Queue an utterance, superseding any utterance still pending."""
        fused = np.asarray(audio, dtype=np.float32)
        with self._lock:
            self._pending = (int(stream_id), fused)
        self._wake.set()

    def stop_and_flush(self) -> None:
        """Interrupt the current utterance and drop anything pending.

        Asynchronous on purpose: the caller is often the GUI thread, and blocking
        it for the remainder of a write would freeze the UI.
        """
        with self._lock:
            self._pending = None
        self._interrupt.set()
        self._wake.set()

    # ── consumer state ───────────────────────────────────────────────────────

    @property
    def played_samples(self) -> int:
        """Samples written for the current stream. Advances per chunk."""
        return self._played

    @property
    def current_stream_id(self) -> int | None:
        return self._current_stream_id

    @property
    def failures(self) -> int:
        return self._failures

    @property
    def failed(self) -> bool:
        """True once the worker has given up, so consumers can stop waiting on it.

        Without this, a dead playback worker freezes ``played_samples`` — and since
        Speak-state motion is gated on that position, the character would hold its
        last frame indefinitely.
        """
        return self._failed

    # ── worker ───────────────────────────────────────────────────────────────

    def _take_pending(self) -> tuple[int, np.ndarray] | None:
        with self._lock:
            item, self._pending = self._pending, None
        return item

    def _loop(self) -> None:
        while not self._stop_all.is_set():
            if not self._wake.wait(timeout=self._POLL_S):
                continue
            self._wake.clear()
            if self._interrupt.is_set():
                self._interrupt.clear()
                self._played = 0
                self._current_stream_id = None

            item = self._take_pending()
            if item is None:
                continue
            try:
                self._play_stream(*item)
            except Exception as e:
                self._failures += 1
                logger.error(
                    f"Playback failed ({self._failures}/{self.MAX_CONSECUTIVE_FAILURES}): {e}"
                )
                if self._failures >= self.MAX_CONSECUTIVE_FAILURES:
                    self._failed = True
                    logger.error(
                        "Playback failed repeatedly; stopping the worker so the fault "
                        "is visible instead of silently muting the character"
                    )
                    break
            else:
                self._failures = 0

    def _play_stream(self, stream_id: int, audio: np.ndarray) -> None:
        stream = None
        self._played = 0
        self._current_stream_id = int(stream_id)
        total = int(audio.size)
        try:
            stream = self._factory(self.sample_rate)
            while self._played < total:
                if self._stop_all.is_set():
                    return
                if self._interrupt.is_set():
                    # Consume the flag and decay the remaining audio so the stop
                    # is not audible as a click.
                    self._interrupt.clear()
                    self._write_fade_out(stream, audio)
                    return
                end = min(self._played + self._chunk_samples, total)
                stream.write(audio[self._played : end].tobytes())
                self._played = end
        finally:
            self._current_stream_id = None
            self._played = 0
            if stream is not None:
                try:
                    stream.close()
                except Exception as e:
                    logger.debug(f"Playback stream close failed: {e}")

    def _write_fade_out(self, stream: Any, audio: np.ndarray) -> None:
        """Write a linearly decaying tail covering :attr:`_fade_samples`."""
        if self._fade_samples <= 0:
            return
        remaining = audio[self._played : self._played + self._fade_samples]
        if remaining.size == 0:
            return
        ramp = np.linspace(1.0, 0.0, remaining.size, dtype=np.float32)
        stream.write((remaining * ramp).tobytes())
