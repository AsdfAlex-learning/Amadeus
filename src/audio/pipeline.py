from collections import deque
from collections.abc import Callable
from enum import Enum, auto

import numpy as np
from loguru import logger

from src.audio.asr import WhisperASR
from src.audio.capture import MicrophoneCapture


class PipelineState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()


class AudioPipeline:
    SILENCE_DURATION = 1.2
    SPEECH_THRESHOLD_RMS = 0.02
    MAX_SPEECH_DURATION = 15.0
    PRE_SPEECH_BUFFER_SEC = 0.3

    def __init__(self, config: dict):
        self.config = config
        self.capture = MicrophoneCapture(config)
        self.asr = WhisperASR(config)
        self._state = PipelineState.IDLE
        self._text_callbacks: list[Callable[[str], None]] = []
        self._state_callbacks: list[Callable[[PipelineState], None]] = []
        self._audio_callbacks: list[Callable[[np.ndarray], None]] = []

        self._audio_buffer: list[np.ndarray] = []
        self._pre_speech_buffer: deque[np.ndarray] = deque(
            maxlen=int(
                self.PRE_SPEECH_BUFFER_SEC * self.capture.sample_rate / self.capture.chunk_size
            )
        )
        self._silence_counter = 0
        self._speech_duration = 0.0

    def start(self):
        if self._state != PipelineState.IDLE:
            return
        self._load_asr()
        self.capture.on_audio(self._on_audio_chunk)
        self.capture.start()
        self._set_state(PipelineState.LISTENING)
        logger.info("Audio pipeline started")

    def stop(self):
        self.capture.stop()
        self._set_state(PipelineState.IDLE)
        self._audio_buffer.clear()
        self._pre_speech_buffer.clear()
        logger.info("Audio pipeline stopped")

    def cleanup(self):
        self.stop()
        self.capture.cleanup()

    def on_text(self, callback: Callable[[str], None]):
        self._text_callbacks.append(callback)

    def on_state_change(self, callback: Callable[[PipelineState], None]):
        self._state_callbacks.append(callback)

    def on_raw_audio(self, callback: Callable[[np.ndarray], None]):
        self._audio_callbacks.append(callback)
        self.capture.on_audio(callback)

    def _load_asr(self):
        success = self.asr.load_model()
        if not success:
            logger.warning(
                "Whisper.cpp not available. Install: pip install whisper-cpp-python. "
                "ASR will return empty strings."
            )

    def _on_audio_chunk(self, audio: np.ndarray):
        rms = float(np.sqrt(np.mean(audio**2)))
        self._pre_speech_buffer.append(audio)
        is_speech = rms > self.SPEECH_THRESHOLD_RMS

        if self._state == PipelineState.LISTENING:
            if is_speech:
                self._audio_buffer.extend(self._pre_speech_buffer)
                self._pre_speech_buffer.clear()
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

    def _finalize_utterance(self):
        if not self._audio_buffer:
            return
        audio = np.concatenate(self._audio_buffer)
        self._audio_buffer.clear()
        self._silence_counter = 0
        self._speech_duration = 0.0
        self._set_state(PipelineState.PROCESSING)
        try:
            text = self.asr.transcribe(audio, self.capture.sample_rate)
            if text:
                logger.info(f"ASR: {text}")
                for cb in self._text_callbacks:
                    try:
                        cb(text)
                    except Exception as e:
                        logger.error(f"Text callback error: {e}")
        finally:
            self._set_state(PipelineState.LISTENING)

    def _set_state(self, state: PipelineState):
        if state != self._state:
            self._state = state
            for cb in self._state_callbacks:
                try:
                    cb(state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")

    @property
    def state(self) -> PipelineState:
        return self._state
