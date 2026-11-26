"""Voice activity detection and speech-to-text orchestration.

Threading contract (ADR 0003):

* ``_on_audio_chunk`` runs on the capture drain worker. It is the only thread
  that touches the VAD state and the utterance buffer.
* ``_asr_loop`` owns ``WhisperASR`` and runs transcription, so a slow Whisper pass
  can never stall the thread that drains the microphone ring.
* ``PipelineState`` is a *UI label*, not a gate. Buffering is unconditional;
  gating it on ``LISTENING`` used to discard anything said while a transcription
  was in flight.
"""

import threading
from collections import deque
from collections.abc import Callable
from enum import Enum, auto

import numpy as np
from loguru import logger

from src.audio.asr import WhisperASR
from src.audio.capture import MicrophoneCapture
from src.runtime.concurrency import BoundedDropOldest


class PipelineState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()


class AudioPipeline:
    SILENCE_DURATION = 1.2
    SPEECH_THRESHOLD_RMS = 0.02
    MAX_SPEECH_DURATION = 15.0
    PRE_SPEECH_BUFFER_SEC = 0.3
    #: Utterances are queued, not collapsed: dropping one loses user input.
    ASR_QUEUE_CAPACITY = 3
    _WORKER_JOIN_TIMEOUT_S = 2.0
    _WORKER_POLL_S = 0.1

    def __init__(self, config: dict):
        self.config = config
        self.capture = MicrophoneCapture(config)
        self.asr = WhisperASR(config)
        self._state = PipelineState.IDLE
        self._text_callbacks: list[Callable[[str], None]] = []
        self._state_callbacks: list[Callable[[PipelineState], None]] = []
        self._audio_callbacks: list[Callable[[np.ndarray], None]] = []
        self._onset_callbacks: list[Callable[[], None]] = []

        self._audio_buffer: list[np.ndarray] = []
        self._pre_speech_buffer: deque[np.ndarray] = deque(
            maxlen=int(
                self.PRE_SPEECH_BUFFER_SEC * self.capture.sample_rate / self.capture.chunk_size
            )
        )
        self._silence_counter = 0
        self._speech_duration = 0.0

        self._asr_q: BoundedDropOldest[np.ndarray] = BoundedDropOldest(self.ASR_QUEUE_CAPACITY)
        self._asr_worker: threading.Thread | None = None
        self._asr_stop = threading.Event()
        self._asr_wake = threading.Event()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        if self._state != PipelineState.IDLE:
            return
        self._ensure_asr_worker()
        self.capture.on_audio(self._on_audio_chunk)
        self.capture.start()
        self._set_state(PipelineState.LISTENING)
        logger.info("Audio pipeline started")

    def stop(self):
        self.capture.stop()
        self._stop_asr_worker()
        self._set_state(PipelineState.IDLE)
        self._audio_buffer.clear()
        self._pre_speech_buffer.clear()
        logger.info("Audio pipeline stopped")

    def cleanup(self):
        self.stop()
        self.capture.cleanup()
        self._text_callbacks.clear()
        self._state_callbacks.clear()
        self._audio_callbacks.clear()
        self._onset_callbacks.clear()

    # ── registration ─────────────────────────────────────────────────────────

    def on_text(self, callback: Callable[[str], None]):
        self._text_callbacks.append(callback)

    def on_state_change(self, callback: Callable[[PipelineState], None]):
        self._state_callbacks.append(callback)

    def on_speech_onset(self, callback: Callable[[], None]):
        """Called when the VAD transitions from silence to speech.

        Interruption needs the *onset*: the transcript does not exist for another
        second or more (VAD silence window plus transcription), and waiting for it
        would leave playback running through the user's whole utterance.
        """
        self._onset_callbacks.append(callback)

    def on_raw_audio(self, callback: Callable[[np.ndarray], None]):
        """Register a raw-audio consumer.

        Note: the callback runs on the capture drain worker, so a slow consumer
        here delays VAD. ADR 0003 Stage 1.4 moves model inference onto its own
        thread to remove that coupling.
        """
        self._audio_callbacks.append(callback)
        self.capture.on_audio(callback)

    # ── VAD (capture drain worker) ───────────────────────────────────────────

    def _on_audio_chunk(self, audio: np.ndarray):
        """Energy-based VAD, run continuously.

        Deliberately **not** gated on ``PipelineState``: buffering used to happen
        only in ``LISTENING``, so speech arriving while a transcription was in
        flight was discarded. ``PipelineState`` now reports intent to the UI
        rather than gating the VAD.
        """
        rms = float(np.sqrt(np.mean(audio**2)))
        self._pre_speech_buffer.append(audio)
        is_speech = rms > self.SPEECH_THRESHOLD_RMS

        if is_speech:
            if not self._audio_buffer:
                # Start a new utterance by adopting the pre-speech context, which
                # already contains this chunk plus ~300 ms of lead-in, so the onset
                # is not clipped. Appending again here would duplicate the chunk.
                self._audio_buffer.extend(self._pre_speech_buffer)
                self._pre_speech_buffer.clear()
                self._notify_speech_onset()
            else:
                self._audio_buffer.append(audio)
            self._silence_counter = 0
            self._speech_duration += len(audio) / self.capture.sample_rate
            if self._speech_duration > self.MAX_SPEECH_DURATION:
                self._finalize_utterance()
        elif self._audio_buffer:
            self._audio_buffer.append(audio)
            self._silence_counter += 1
            silence_sec = self._silence_counter * len(audio) / self.capture.sample_rate
            if silence_sec >= self.SILENCE_DURATION:
                self._finalize_utterance()

    def _notify_speech_onset(self):
        for callback in list(self._onset_callbacks):
            try:
                callback()
            except Exception as e:
                logger.error(f"Speech onset callback error: {e}")

    def _finalize_utterance(self):
        """Hand the completed utterance to the ASR worker.

        Must stay fast: this runs on the thread draining the microphone ring, and
        a Whisper pass takes hundreds of milliseconds to seconds.
        """
        if not self._audio_buffer:
            return
        audio = np.concatenate(self._audio_buffer)
        self._audio_buffer.clear()
        self._silence_counter = 0
        self._speech_duration = 0.0

        self._ensure_asr_worker()
        before = self._asr_q.dropped
        self._asr_q.put(audio)
        if self._asr_q.dropped > before:
            logger.warning(
                "ASR queue full: an earlier utterance was dropped (transcription is "
                "falling behind speech)"
            )
        self._asr_wake.set()
        self._set_state(PipelineState.PROCESSING)

    # ── ASR worker ───────────────────────────────────────────────────────────

    def _ensure_asr_worker(self):
        if self._asr_worker is not None and self._asr_worker.is_alive():
            return
        self._asr_stop.clear()
        self._asr_worker = threading.Thread(target=self._asr_loop, name="amadeus-asr", daemon=True)
        self._asr_worker.start()

    def _stop_asr_worker(self):
        worker, self._asr_worker = self._asr_worker, None
        if worker is None:
            return
        self._asr_stop.set()
        self._asr_wake.set()
        worker.join(timeout=self._WORKER_JOIN_TIMEOUT_S)
        if worker.is_alive():
            # Escalation policy (ADR 0003): a Whisper pass can exceed the join
            # timeout. Log and leak rather than releasing what it still uses.
            logger.warning(
                "ASR worker did not stop within %.0f s; leaking it (daemon)",
                self._WORKER_JOIN_TIMEOUT_S,
            )
            self._asr_worker = worker
        self._asr_wake.clear()

    def _asr_loop(self):
        # Model loading is slow (hundreds of MB); do it here, never on the audio path.
        self._load_asr()
        while not self._asr_stop.is_set():
            if not self._asr_wake.wait(timeout=self._WORKER_POLL_S):
                continue
            self._asr_wake.clear()
            while not self._asr_stop.is_set():
                utterance = self._asr_q.get_nowait()
                if utterance is None:
                    break
                try:
                    text = self.asr.transcribe(utterance, self.capture.sample_rate)
                except Exception as e:
                    logger.error(f"Transcription failed: {e}")
                    continue
                if text:
                    logger.info(f"ASR: {text}")
                    for callback in list(self._text_callbacks):
                        try:
                            callback(text)
                        except Exception as e:
                            logger.error(f"Text callback error: {e}")
            if len(self._asr_q) == 0 and self._state == PipelineState.PROCESSING:
                self._set_state(PipelineState.LISTENING)

    def _load_asr(self):
        success = self.asr.load_model()
        if not success:
            logger.warning(
                "Whisper.cpp not available. Install: pip install whisper-cpp-python. "
                "ASR will return empty strings."
            )

    def _set_state(self, state: PipelineState):
        if state != self._state:
            self._state = state
            for callback in list(self._state_callbacks):
                try:
                    callback(state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def dropped_utterances(self) -> int:
        """Utterances discarded because transcription could not keep up."""
        return self._asr_q.dropped
