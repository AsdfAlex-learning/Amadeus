import threading
from collections import deque
from typing import Callable

import numpy as np
import pyaudio
from loguru import logger


class MicrophoneCapture:
    CHUNK_MS = 30

    def __init__(self, config: dict):
        audio_cfg = config["audio"]
        self.sample_rate = int(audio_cfg["sample_rate"])
        self.channels = int(audio_cfg["channels"])
        self.chunk_size = int(audio_cfg["chunk_size"])
        self.device_index = audio_cfg.get("input_device")

        self._audio = pyaudio.PyAudio()
        self._stream: pyaudio.Stream | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._callbacks: list[Callable[[np.ndarray], None]] = []
        self._buffer = deque(maxlen=int(self.sample_rate * 5 / self.chunk_size))

    def start(self):
        if self._running:
            return
        self._running = True
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
            f"Microphone started: {self.sample_rate}Hz, "
            f"{self.channels}ch, chunk={self.chunk_size}"
        )

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        logger.info("Microphone stopped")

    def cleanup(self):
        self.stop()
        self._audio.terminate()
        self._callbacks.clear()

    def on_audio(self, callback: Callable[[np.ndarray], None]):
        self._callbacks.append(callback)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if not self._running:
            return (None, pyaudio.paAbort)
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self._buffer.append(audio_data)
        for cb in self._callbacks:
            try:
                cb(audio_data)
            except Exception as e:
                logger.error(f"Audio callback error: {e}")
        return (None, pyaudio.paContinue)

    def get_buffer(self, duration_seconds: float) -> np.ndarray:
        chunks_needed = max(1, int(duration_seconds * self.sample_rate / self.chunk_size))
        items = list(self._buffer)[-chunks_needed:]
        if not items:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(items)

    def get_audio_devices(self) -> list[dict]:
        devices = []
        for i in range(self._audio.get_device_count()):
            info = self._audio.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                    "sample_rate": int(info["defaultSampleRate"]),
                })
        return devices

    @property
    def is_running(self) -> bool:
        return self._running
