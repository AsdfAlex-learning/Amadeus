"""Phase 0 non-neural performance baseline.

This is the deterministic floor described in ``docs/MODEL_TRAINING_ROADMAP.md``:
audio energy drives mouth openness, a seeded scheduler blinks, a slow sine
breathes, and small head/body oscillations keep the character alive. It is used
when no FullDuplexDiT checkpoint is available, so the application can run and be
judged before any model has been trained.

The output is the same ``(T, 45)`` Live2D-parameter array the neural path emits,
in the canonical ``LIVE2D_PARAM_NAMES`` order.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from src.motion.preprocess.arkit_to_live2d import LIVE2D_PARAM_NAMES

_INDEX = {name: index for index, name in enumerate(LIVE2D_PARAM_NAMES)}


class Phase0Baseline:
    """Generate rule-based Live2D parameters from audio and elapsed time."""

    def __init__(self, config: dict | None = None, sample_rate: int = 16000) -> None:
        motion_cfg = (config or {}).get("motion", {}) or {}
        cfg: dict[str, Any] = motion_cfg.get("baseline", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.fps = max(1.0, float(cfg.get("fps", 50.0)))
        self.sample_rate = int(sample_rate)
        self.mouth_gain = float(cfg.get("mouth_open_gain", 3.0))
        self.blink_duration = max(0.02, float(cfg.get("blink_duration_seconds", 0.12)))
        self.breath_bpm = max(1.0, float(cfg.get("breath_cycles_per_minute", 14.0)))
        self.head_amplitude = float(cfg.get("head_motion_amplitude", 0.15))
        self.smoothing = float(np.clip(cfg.get("expression_smoothing", 0.2), 0.0, 1.0))
        interval = cfg.get("blink_interval_seconds", [3.0, 6.0])
        if isinstance(interval, (int, float)):
            low = high = float(interval)
        else:
            values = [float(value) for value in interval]
            low, high = (values[0], values[-1]) if values else (3.0, 6.0)
        if low > high:
            low, high = high, low
        self.blink_interval = (max(0.1, low), max(0.1, high))
        self._rng = random.Random(cfg.get("seed"))
        self._time_offset = 0.0
        self._next_blink_at = self.blink_interval[0]
        self._blink_until = -1.0
        self._prev = np.full(len(LIVE2D_PARAM_NAMES), 0.5, dtype=np.float32)

    def num_frames_for(self, audio_length: int) -> int:
        """Number of output frames for an audio buffer at the configured rate."""
        return max(1, int(round(audio_length / self.sample_rate * self.fps)))

    def reset(self) -> None:
        self._time_offset = 0.0
        self._next_blink_at = self.blink_interval[0]
        self._blink_until = -1.0
        self._prev = np.full(len(LIVE2D_PARAM_NAMES), 0.5, dtype=np.float32)

    def generate(
        self,
        user_audio: np.ndarray,
        tts_audio: np.ndarray,
        mode: str = "listen",
        num_frames: int | None = None,
    ) -> np.ndarray:
        """Return ``(T, 45)`` parameters for the given audio and state."""
        user = np.asarray(user_audio, dtype=np.float32).reshape(-1)
        tts = np.asarray(tts_audio, dtype=np.float32).reshape(-1)
        if mode == "speak":
            active = tts
        elif mode == "listen":
            active = user
        else:
            active = np.zeros(0, dtype=np.float32)

        frames = (
            max(1, int(num_frames)) if num_frames is not None else self.num_frames_for(len(active))
        )
        times = self._time_offset + np.arange(frames, dtype=np.float32) / self.fps
        mouth = np.clip(self._rms_per_frame(active, frames) * self.mouth_gain, 0.0, 1.0)

        breath = 0.5 + 0.5 * np.sin(2.0 * math.pi * self.breath_bpm / 60.0 * times)
        head_x = 0.5 + self.head_amplitude * np.sin(2.0 * math.pi * 0.12 * times)
        head_y = 0.5 + self.head_amplitude * np.sin(2.0 * math.pi * 0.09 * times + 1.0)
        head_z = 0.5 + self.head_amplitude * 0.5 * np.sin(2.0 * math.pi * 0.07 * times + 2.0)
        blink = self._blink_openness(times)

        params = np.full((frames, len(LIVE2D_PARAM_NAMES)), 0.5, dtype=np.float32)
        params[:, _INDEX["ParamAngleX"]] = head_x
        params[:, _INDEX["ParamAngleY"]] = head_y
        params[:, _INDEX["ParamAngleZ"]] = head_z
        params[:, _INDEX["ParamBodyAngleX"]] = 0.5 + (head_x - 0.5) * 0.3
        params[:, _INDEX["ParamBodyAngleY"]] = 0.5 + (head_y - 0.5) * 0.3
        params[:, _INDEX["ParamBodyAngleZ"]] = 0.5 + (head_z - 0.5) * 0.3
        params[:, _INDEX["ParamEyeLOpen"]] = blink
        params[:, _INDEX["ParamEyeROpen"]] = blink
        params[:, _INDEX["ParamEyeBallX"]] = 0.5 + self.head_amplitude * 0.3 * np.sin(
            2.0 * math.pi * 0.05 * times
        )
        params[:, _INDEX["ParamEyeBallY"]] = 0.5 + self.head_amplitude * 0.3 * np.sin(
            2.0 * math.pi * 0.04 * times + 0.5
        )
        params[:, _INDEX["ParamBrowLY"]] = 0.5 + self.head_amplitude * 0.2 * np.sin(
            2.0 * math.pi * 0.06 * times
        )
        params[:, _INDEX["ParamBrowRY"]] = params[:, _INDEX["ParamBrowLY"]]
        params[:, _INDEX["ParamMouthOpenY"]] = mouth
        params[:, _INDEX["ParamMouthForm"]] = (
            0.5 + 0.05 * np.sin(2.0 * math.pi * 0.1 * times) + (0.05 if mode == "speak" else 0.0)
        )
        params[:, _INDEX["ParamCheek"]] = 0.5 + 0.05 * np.sin(2.0 * math.pi * 0.08 * times + 0.7)
        params[:, _INDEX["ParamBreath"]] = breath
        params[:, _INDEX["ParamArmLX"]] = 0.5 + 0.03 * np.sin(2.0 * math.pi * 0.05 * times)
        params[:, _INDEX["ParamArmRX"]] = 0.5 - 0.03 * np.sin(2.0 * math.pi * 0.05 * times)
        params = np.clip(params, 0.0, 1.0)

        if self.smoothing > 0.0:
            alpha = self.smoothing
            state = self._prev.copy()
            for index in range(frames):
                state = state + alpha * (params[index] - state)
                params[index] = state
            self._prev = state

        self._time_offset = float(times[-1] + 1.0 / self.fps)
        if self._time_offset > 1000.0:
            self._time_offset = 0.0
        return params.astype(np.float32, copy=False)

    def _rms_per_frame(self, audio: np.ndarray, frames: int) -> np.ndarray:
        if audio.size == 0:
            return np.zeros(frames, dtype=np.float32)
        starts = np.floor(np.linspace(0, audio.size, frames, endpoint=False)).astype(np.int64)
        window = max(1, int(self.sample_rate / self.fps))
        values = np.empty(frames, dtype=np.float32)
        for index, start in enumerate(starts):
            end = min(audio.size, int(start) + window)
            chunk = audio[int(start) : end]
            values[index] = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        return values

    def _blink_openness(self, times: np.ndarray) -> np.ndarray:
        openness = np.ones(times.shape, dtype=np.float32)
        for index, now in enumerate(times):
            now = float(now)
            if now >= self._next_blink_at and now >= self._blink_until:
                self._blink_until = now + self.blink_duration
                self._next_blink_at = self._blink_until + self._rng.uniform(*self.blink_interval)
            if now < self._blink_until:
                start = self._blink_until - self.blink_duration
                progress = (now - start) / self.blink_duration
                openness[index] = 1.0 - math.sin(math.pi * float(np.clip(progress, 0.0, 1.0)))
        return np.clip(openness, 0.0, 1.0)
