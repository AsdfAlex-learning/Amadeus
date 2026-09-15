"""Microphone capture with a real-time-safe PortAudio callback.

``_audio_callback`` runs on PortAudio's own thread with a hard budget of
``chunk_size / sample_rate`` (about 64 ms at 16 kHz with a 1024-frame chunk).
Nothing that can block may run there, so the callback does the minimum: convert
the int16 chunk, publish it to a lock-free ring, and signal a worker. All
consumer code — VAD, ASR, model inference — runs on the drain worker instead.

See ``docs/adr/0003-runtime-threading-and-realtime-architecture.md``.
"""

import threading
from collections.abc import Callable

import numpy as np
import pyaudio
from loguru import logger

from src.runtime.concurrency import AudioRing

#: How long to wait for the drain worker to exit before giving up on it.
_WORKER_JOIN_TIMEOUT_S = 2.0
#: Wake-up polling interval for the drain worker.
_WORKER_POLL_S = 0.1


class MicrophoneCapture:
    #: Stop the drain worker after this many consecutive failing chunks, rather
    #: than logging the same error forever.
    MAX_CONSECUTIVE_CALLBACK_FAILURES = 20

    def __init__(self, config: dict):
        audio_cfg = config["audio"]
        self.sample_rate = int(audio_cfg["sample_rate"])
        self.channels = int(audio_cfg["channels"])
        self.chunk_size = int(audio_cfg["chunk_size"])
        self.device_index = audio_cfg.get("input_device")

        self._audio = pyaudio.PyAudio()
        self._stream: pyaudio.Stream | None = None
        self._running = False
        self._callbacks: list[Callable[[np.ndarray], None]] = []

        # Single-producer (PortAudio thread) / single-consumer (drain worker)
        # ring. Sized for ~2 s so a briefly slow consumer does not lose audio.
        self._ring = AudioRing(capacity=max(self.sample_rate * 2, self.chunk_size * 8))
        self._wake = threading.Event()
        self._worker: threading.Thread | None = None
        self._worker_stop = threading.Event()
        self._reported_dropped = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._start_worker()
        self._stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )
        self._stream.start_stream()
        logger.info(
            f"Microphone started: {self.sample_rate}Hz, {self.channels}ch, chunk={self.chunk_size}"
        )

    def stop(self):
        # Stop the producer first so no further writes race with teardown.
        self._running = False
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        self._stop_worker()
        logger.info("Microphone stopped")

    def cleanup(self):
        self.stop()
        self._audio.terminate()
        self._callbacks.clear()

    def on_audio(self, callback: Callable[[np.ndarray], None]):
        """Register a consumer. Invoked on the drain worker, never on the callback."""
        self._callbacks.append(callback)

    # ── real-time path ───────────────────────────────────────────────────────

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PortAudio callback. Budget ~64 ms; may not block or run consumer code.

        The int16 -> float32 conversion is a single vectorised op over 1024
        samples (microseconds) and the ring write never blocks, so the only
        other work is signalling the worker.
        """
        if not self._running:
            return (None, pyaudio.paAbort)
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        samples *= 1.0 / 32768.0
        self._ring.write(samples)
        self._wake.set()
        return (None, pyaudio.paContinue)

    # ── drain worker (owns fan-out) ──────────────────────────────────────────

    def _start_worker(self):
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker_stop.clear()
        self._worker = threading.Thread(
            target=self._drain_loop, name="amadeus-audio-drain", daemon=True
        )
        self._worker.start()

    def _stop_worker(self):
        worker, self._worker = self._worker, None
        if worker is None:
            return
        self._worker_stop.set()
        self._wake.set()
        worker.join(timeout=_WORKER_JOIN_TIMEOUT_S)
        if worker.is_alive():
            # Escalation policy (ADR 0003): never release resources a live thread
            # may still touch. Leak it instead - it is a daemon, so process exit
            # reaps it - and keep the handle so the situation is observable.
            logger.warning(
                "Audio drain worker did not stop within %.0f s; leaking it rather "
                "than releasing resources it may still be using",
                _WORKER_JOIN_TIMEOUT_S,
            )
            self._worker = worker
        self._wake.clear()

    def _drain_loop(self):
        """Consume the ring and fan out to registered consumers.

        This is the thread that owns fan-out, so consumer latency can no longer
        affect the real-time callback.
        """
        consecutive_failures = 0
        while not self._worker_stop.is_set():
            if not self._wake.wait(timeout=_WORKER_POLL_S):
                continue
            self._wake.clear()
            while not self._worker_stop.is_set():
                chunk = self._ring.read(self.chunk_size)
                if chunk.size == 0:
                    break
                failed = False
                for callback in list(self._callbacks):
                    try:
                        callback(chunk)
                    except Exception as e:
                        failed = True
                        logger.error(f"Audio callback error: {e}")
                if failed:
                    consecutive_failures += 1
                    if consecutive_failures >= self.MAX_CONSECUTIVE_CALLBACK_FAILURES:
                        logger.error(
                            f"A consumer failed on {consecutive_failures} consecutive "
                            f"chunks; stopping the audio drain worker so the fault is "
                            f"visible instead of logging forever"
                        )
                        return
                else:
                    consecutive_failures = 0
            self._report_drops()

    def _report_drops(self):
        """Surface lapping once per occurrence rather than every loop iteration."""
        dropped = self._ring.dropped
        if dropped > self._reported_dropped:
            logger.warning(
                f"Audio ring lapped its consumer: {dropped - self._reported_dropped} "
                f"sample(s) lost (consumer is slower than {self.sample_rate} Hz)"
            )
            self._reported_dropped = dropped

    # ── introspection ────────────────────────────────────────────────────────

    def get_audio_devices(self) -> list[dict]:
        devices = []
        for i in range(self._audio.get_device_count()):
            info = self._audio.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                devices.append(
                    {
                        "index": i,
                        "name": info["name"],
                        "channels": info["maxInputChannels"],
                        "sample_rate": int(info["defaultSampleRate"]),
                    }
                )
        return devices

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def dropped_samples(self) -> int:
        """Samples lost because the consumer could not keep up (monotonic)."""
        return self._ring.dropped
