"""Motion inference must run on its own owner thread.

``process_user_audio`` was wired directly to the capture drain worker, so a
124 M-parameter, 4-step diffusion forward pass ran on the same thread that runs
VAD. On the CPU target one pass costs roughly 0.5-3 s per 1 s of audio, which
starves the VAD and overflows the audio ring — the ring then reports lapped
sample loss, but the real problem is that speech onset detection stops running.

ADR 0003 Stage 1.4 gives inference its own thread, so the capture worker only ever
publishes audio into a bounded queue.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from src.motion.inference import DiffusionMotionInference

CHUNK = 16_000  # one second at 16 kHz
LONG_INFERENCE_S = 0.5


def _engine(config: dict) -> DiffusionMotionInference:
    engine = DiffusionMotionInference(config)
    engine._loaded = True
    engine._model = object()  # sentinel: only _diffusion_infer would dereference it
    return engine


def test_process_user_audio_returns_immediately_when_a_worker_runs(config: dict) -> None:
    """Handing audio over must not run inference on the caller's thread."""

    def slow_infer(*_args, **_kwargs):
        time.sleep(LONG_INFERENCE_S)
        return None

    engine = _engine(config)
    engine._diffusion_infer = slow_infer  # type: ignore[method-assign]
    engine.start_worker()
    try:
        started = time.perf_counter()
        engine.process_user_audio(np.zeros(CHUNK, dtype=np.float32))
        elapsed = time.perf_counter() - started
    finally:
        engine.stop_worker()

    assert elapsed < 0.1, (
        f"process_user_audio blocked for {elapsed:.3f}s while a worker was running; "
        f"inference belongs to the worker, not the caller"
    )


def test_inference_runs_off_the_calling_thread(config: dict) -> None:
    """Parameter callbacks must be delivered from the worker thread."""
    emitter_threads: set[str] = set()
    ran = threading.Event()

    def infer(*_args, **_kwargs):
        return np.full((5, 45), 0.5, dtype=np.float32)

    engine = _engine(config)
    engine._diffusion_infer = infer  # type: ignore[method-assign]

    def on_params(_params: dict[str, float]) -> None:
        emitter_threads.add(threading.current_thread().name)
        ran.set()

    engine.on_params(on_params)
    engine.start_worker()
    try:
        engine.process_user_audio(np.zeros(CHUNK, dtype=np.float32))
        assert ran.wait(timeout=5.0), "inference never ran"
    finally:
        engine.stop_worker()

    assert emitter_threads, "no parameters were emitted"
    assert all("motion" in name for name in emitter_threads), (
        f"parameters were emitted on {emitter_threads}; expected the motion worker"
    )


def test_a_single_failure_does_not_kill_the_worker(config: dict) -> None:
    """One bad frame must not silently stop motion forever."""
    attempts = {"count": 0}
    recovered = threading.Event()

    def flaky(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("simulated inference failure")
        recovered.set()
        return None

    engine = _engine(config)
    engine._diffusion_infer = flaky  # type: ignore[method-assign]
    engine.start_worker()
    try:
        engine.process_user_audio(np.zeros(CHUNK, dtype=np.float32))
        time.sleep(LONG_INFERENCE_S)
        engine.process_user_audio(np.zeros(CHUNK, dtype=np.float32))
        assert recovered.wait(timeout=5.0), "the worker died after a single failure"
    finally:
        engine.stop_worker()


def test_worker_stops_on_cleanup(config: dict) -> None:
    """Shutdown must join the motion worker rather than abandon it."""
    engine = _engine(config)
    engine._diffusion_infer = lambda *_a, **_k: None  # type: ignore[method-assign]
    engine.start_worker()
    engine.process_user_audio(np.zeros(CHUNK, dtype=np.float32))
    worker = engine._motion_worker
    assert worker is not None and worker.is_alive()

    engine.cleanup()

    assert not worker.is_alive(), "motion worker outlived cleanup()"
