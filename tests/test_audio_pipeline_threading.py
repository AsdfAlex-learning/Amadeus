"""Transcription must not run on the thread that drains the microphone ring.

``AudioPipeline._finalize_utterance`` used to call ``asr.transcribe()`` inline.
That executed a Whisper pass on whichever thread was consuming microphone audio,
so the ring stalled for hundreds of milliseconds to seconds and audio was lost.
It also gated buffering on ``PipelineState.LISTENING``, so anything said while a
transcription was in flight was discarded outright.

Per ADR 0003 Stage 1, transcription moves to a dedicated worker fed by a bounded
FIFO, and ``PipelineState`` stops meaning "VAD paused" — it becomes a UI label.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from src.audio.pipeline import AudioPipeline, PipelineState

TRANSCRIBE_DELAY_S = 0.6


class _SlowStubASR:
    """Stands in for Whisper: slow, and records what it was asked to transcribe."""

    def __init__(self, text: str = "hello") -> None:
        self.text = text
        self.transcribed: list[np.ndarray] = []
        self.loaded = False

    def load_model(self) -> bool:
        self.loaded = True
        return True

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        self.transcribed.append(audio)
        time.sleep(TRANSCRIBE_DELAY_S)
        return self.text

    @property
    def is_loaded(self) -> bool:
        return self.loaded


def _pipeline_with_stub(config: dict) -> tuple[AudioPipeline, _SlowStubASR]:
    pipeline = AudioPipeline(config)
    stub = _SlowStubASR()
    pipeline.asr = stub
    return pipeline, stub


def test_finalize_utterance_does_not_transcribe_inline(config: dict) -> None:
    """Queueing an utterance must be fast, regardless of how slow ASR is."""
    pipeline, _ = _pipeline_with_stub(config)
    pipeline._audio_buffer = [np.ones(pipeline.capture.sample_rate, dtype=np.float32)]

    try:
        started = time.perf_counter()
        pipeline._finalize_utterance()
        elapsed = time.perf_counter() - started
    finally:
        pipeline.cleanup()

    assert elapsed < 0.2, (
        f"_finalize_utterance blocked for {elapsed:.3f}s while ASR takes "
        f"{TRANSCRIBE_DELAY_S}s; transcription must be handed to a worker, not run "
        f"on the audio path"
    )


def test_transcription_result_reaches_text_callbacks(config: dict) -> None:
    """The worker must still deliver the transcript to registered consumers."""
    pipeline, _ = _pipeline_with_stub(config)
    results: list[str] = []
    delivered = threading.Event()

    def on_text(text: str) -> None:
        results.append(text)
        delivered.set()

    pipeline.on_text(on_text)
    pipeline._audio_buffer = [np.ones(pipeline.capture.sample_rate, dtype=np.float32)]

    try:
        pipeline._finalize_utterance()
        assert delivered.wait(timeout=5.0), "the transcript never reached the callbacks"
    finally:
        pipeline.cleanup()

    assert results == ["hello"]


def test_speech_is_buffered_while_a_transcription_is_pending(config: dict) -> None:
    """``PROCESSING`` must no longer mean "VAD paused".

    Buffering used to happen only in ``LISTENING``, so speech arriving during the
    transcription window was dropped on the floor — a data-loss bug independent of
    which thread ran the transcription.
    """
    pipeline, _ = _pipeline_with_stub(config)
    pipeline._state = PipelineState.PROCESSING
    chunk = np.full(1024, 0.5, dtype=np.float32)  # well above the RMS threshold

    try:
        pipeline._on_audio_chunk(chunk)
        # Snapshot before teardown: cleanup() clears the buffer by design.
        buffered = list(pipeline._audio_buffer)
    finally:
        pipeline.cleanup()

    assert buffered, (
        "speech that arrived while a transcription was pending was discarded"
    )


def test_back_to_back_utterances_are_not_overwritten(config: dict) -> None:
    """Two utterances queued in quick succession must both be transcribed.

    A latest-wins slot here would silently delete the first utterance, which is
    data loss rather than a staleness trade-off.
    """
    pipeline, stub = _pipeline_with_stub(config)
    one_second = np.ones(pipeline.capture.sample_rate, dtype=np.float32)

    try:
        for _ in range(2):
            pipeline._audio_buffer = [one_second]
            pipeline._finalize_utterance()

        deadline = time.time() + 6.0
        while len(stub.transcribed) < 2 and time.time() < deadline:
            time.sleep(0.05)
    finally:
        pipeline.cleanup()

    assert len(stub.transcribed) == 2, (
        f"only {len(stub.transcribed)} of 2 utterances reached ASR; the transcript "
        f"queue must preserve order and completeness"
    )


def test_asr_worker_stops_on_cleanup(config: dict) -> None:
    """Shutdown must join the ASR worker rather than abandon it."""
    pipeline, _ = _pipeline_with_stub(config)
    pipeline._audio_buffer = [np.ones(pipeline.capture.sample_rate, dtype=np.float32)]
    pipeline._finalize_utterance()
    worker = pipeline._asr_worker
    assert worker is not None and worker.is_alive()

    pipeline.cleanup()

    assert not worker.is_alive(), "ASR worker outlived cleanup()"
