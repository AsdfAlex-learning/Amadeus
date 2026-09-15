"""The PortAudio callback is a real-time context and must stay trivial.

``MicrophoneCapture._audio_callback`` runs on PortAudio's own thread with a
budget of ``chunk_size / sample_rate`` — about 64 ms at 16 kHz with a 1024-frame
chunk. It used to invoke every registered consumer inline, so a 124 M-parameter
diffusion forward pass or a Whisper transcription executed inside that budget and
the stream stalled or dropped samples.

These tests pin the two halves of the contract: the callback itself does almost
nothing, and the fan-out to consumers happens on a worker thread instead.
"""

from __future__ import annotations

import threading
import time

import numpy as np

CHUNK = 1024
RT_BUDGET_MS = CHUNK / 16_000 * 1000  # ~64 ms


def test_audio_callback_does_not_run_consumers(config: dict) -> None:
    """A slow consumer must not delay the real-time callback.

    The callback's own work must be a conversion plus a non-blocking write, so a
    consumer that blocks for 0.5 s must not show up in the callback's latency.
    """
    from src.audio.capture import MicrophoneCapture

    capture = MicrophoneCapture(config)
    capture._running = True

    def slow_consumer(_audio: np.ndarray) -> None:
        time.sleep(0.5)  # stands in for model inference / transcription

    capture.on_audio(slow_consumer)
    payload = np.zeros(CHUNK, dtype=np.int16).tobytes()

    try:
        started = time.perf_counter()
        capture._audio_callback(payload, CHUNK, None, 0)
        elapsed = time.perf_counter() - started
    finally:
        capture.cleanup()

    assert elapsed < 0.05, (
        f"the real-time callback blocked for {elapsed * 1000:.1f} ms; its budget is "
        f"{RT_BUDGET_MS:.0f} ms and it must only enqueue, never run consumers"
    )


def test_audio_callback_publishes_samples_for_the_worker(config: dict) -> None:
    """What the callback publishes must be the normalised audio the consumer sees."""
    from src.audio.capture import MicrophoneCapture

    capture = MicrophoneCapture(config)
    capture._running = True

    try:
        payload = np.full(CHUNK, 16384, dtype=np.int16).tobytes()
        capture._audio_callback(payload, CHUNK, None, 0)
        samples = capture._ring.read(CHUNK)
    finally:
        capture.cleanup()

    assert samples.shape == (CHUNK,)
    # 16384 / 32768 == 0.5
    assert np.allclose(samples, 0.5, atol=1e-4), (
        f"expected normalised int16 samples (~0.5), got min={samples.min():.3f} "
        f"max={samples.max():.3f}"
    )


def test_worker_delivers_audio_to_registered_consumers(config: dict) -> None:
    """Fan-out happens on the worker, so consumers still receive audio."""
    from src.audio.capture import MicrophoneCapture

    capture = MicrophoneCapture(config)
    capture._running = True
    received: list[np.ndarray] = []
    delivered = threading.Event()

    def consumer(audio: np.ndarray) -> None:
        received.append(audio)
        delivered.set()

    capture.on_audio(consumer)
    capture._start_worker()

    try:
        payload = np.full(CHUNK, 8192, dtype=np.int16).tobytes()
        capture._audio_callback(payload, CHUNK, None, 0)
        assert delivered.wait(timeout=3.0), "the worker never delivered audio to the consumer"
    finally:
        capture.cleanup()

    assert received[0].shape == (CHUNK,)
    assert np.allclose(received[0], 0.25, atol=1e-4)


def test_a_failing_consumer_does_not_stop_the_worker(config: dict) -> None:
    """One broken consumer must not silence the others or kill the worker."""
    from src.audio.capture import MicrophoneCapture

    capture = MicrophoneCapture(config)
    capture._running = True
    good_calls: list[np.ndarray] = []
    delivered = threading.Event()

    def broken(_audio: np.ndarray) -> None:
        raise RuntimeError("consumer exploded")

    def good(audio: np.ndarray) -> None:
        good_calls.append(audio)
        delivered.set()

    capture.on_audio(broken)
    capture.on_audio(good)
    capture._start_worker()

    try:
        capture._audio_callback(np.zeros(CHUNK, dtype=np.int16).tobytes(), CHUNK, None, 0)
        assert delivered.wait(timeout=3.0), "a failing consumer suppressed a healthy one"
    finally:
        capture.cleanup()

    assert good_calls, "healthy consumer was never invoked"


def test_worker_stops_on_cleanup(config: dict) -> None:
    """Shutdown must join the worker rather than abandon it."""
    from src.audio.capture import MicrophoneCapture

    capture = MicrophoneCapture(config)
    capture._running = True
    capture._start_worker()
    worker = capture._worker
    assert worker is not None and worker.is_alive()

    capture.cleanup()

    assert not worker.is_alive(), "audio worker outlived cleanup()"
