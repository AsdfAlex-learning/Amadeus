"""Persona Performance Parameters — post-processing multipliers for Live2D output.

After the FullDuplexDiT model outputs raw Live2D parameters (shape: T×45 or
B×T×45), PerformanceEngine applies per-persona style multipliers to control
gesture amplitude, facial expressiveness, mouth openness, head motion range,
reaction speed, and idle energy.

These multipliers are NOT retrained — they are runtime post-processing knobs
that adjust the character's "feel" without modifying model weights. Different
personas (Kurisu, etc.) can have different PerformanceConfig values loaded from
YAML.

Usage:
    from src.motion.performance import PerformanceConfig, PerformanceEngine

    config = PerformanceConfig(gesture_scale=0.7, expressiveness=0.8)
    engine = PerformanceEngine(config, mapping_path="src/motion/preprocess/mappings/default.yaml")
    adjusted = engine.apply(raw_params, mode="speak")  # numpy array T×45 or B×T×45
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════


@dataclass
class PerformanceConfig:
    """Per-persona performance parameter multipliers.

    All values are in [0.0, 1.0] range:
      - 0.0 = no effect (minimal motion)
      - 0.5 = moderate (default)
      - 1.0 = maximum effect (exaggerated motion)
    """

    gesture_scale: float = 0.5  # Overall movement amplitude
    react_speed: float = 0.5  # Reaction speed (temporal smoothing)
    expressiveness: float = 0.5  # Facial expression exaggeration
    mouth_open_max: float = 0.5  # Maximum mouth openness cap
    head_motion_range: float = 0.5  # Head/body angle range
    idle_energy: float = 0.5  # Energy level during silence

    def __post_init__(self):
        """Clamp all values to [0.0, 1.0]."""
        for fld in [
            "gesture_scale", "react_speed", "expressiveness",
            "mouth_open_max", "head_motion_range", "idle_energy",
        ]:
            val = getattr(self, fld)
            if val < 0.0 or val > 1.0:
                logger.warning(f"PerformanceConfig.{fld}={val} clamped to [0, 1]")
                setattr(self, fld, max(0.0, min(1.0, val)))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PerformanceConfig:
        """Create from a config dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f) for f in self.__dataclass_fields__.values()}


# ═══════════════════════════════════════════════════════════════
# Parameter Name Mapping
# ═══════════════════════════════════════════════════════════════


# Fallback heuristic groups when mapping file is unavailable
_FACE_PARAMS = set(range(0, 31))  # mouth, eye, brow params
_HEAD_PARAMS = set(range(31, 40))  # angle & body params
_OTHER_PARAMS = set(range(40, 45))  # misc params

# Known Live2D param name patterns for classification
_MOUTH_PATTERNS = (
    "ParamMouthOpenY", "ParamMouthSmile", "ParamMouthForm",
    "ParamMouth", "ParamJawOpen", "ParamLip",
)
_EYE_PATTERNS = (
    "ParamEyeLOpen", "ParamEyeROpen", "ParamEyeL", "ParamEyeR",
    "ParamEyeBallX", "ParamEyeBallY", "ParamBrowL", "ParamBrowR",
    "ParamEyeSmile",
)
_HEAD_PATTERNS = (
    "ParamAngleX", "ParamAngleY", "ParamAngleZ",
    "ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ",
    "ParamBody",
)


def _classify_param_name(name: str) -> str:
    """Classify a Live2D param name into a category.

    Returns: 'mouth', 'eye', 'head', or 'other'
    """
    name_upper = name.upper()
    for pattern in _MOUTH_PATTERNS:
        if pattern.upper() in name_upper:
            return "mouth"
    for pattern in _EYE_PATTERNS:
        if pattern.upper() in name_upper:
            return "eye"
    for pattern in _HEAD_PATTERNS:
        if pattern.upper() in name_upper:
            return "head"
    return "other"


# ═══════════════════════════════════════════════════════════════
# Performance Engine
# ═══════════════════════════════════════════════════════════════


class PerformanceEngine:
    """Applies persona performance multipliers to model output parameters.

    The engine maintains state for temporal smoothing (react_speed) and
    operates on numpy arrays in a vectorized manner — no Python per-element
    loops.
    """

    def __init__(
        self,
        config: PerformanceConfig | None = None,
        mapping_path: str | Path | None = None,
    ):
        self.config = config or PerformanceConfig()
        self._param_indices: dict[str, int] = {}  # name → index
        self._mouth_indices: list[int] = []
        self._eye_indices: list[int] = []
        self._face_indices: list[int] = []  # mouth + eye + brow
        self._head_indices: list[int] = []
        self._other_indices: list[int] = []
        self._prev_params: np.ndarray | None = None  # for react_speed smoothing
        self._num_params: int = 45

        if mapping_path is not None:
            self._load_mapping(mapping_path)
        else:
            self._use_heuristic_mapping()

    def _load_mapping(self, path: str | Path) -> None:
        """Load parameter name→index mapping from YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Mapping file not found: {path}, using heuristic")
            self._use_heuristic_mapping()
            return

        with open(path) as f:
            data = yaml.safe_load(f)

        params = data.get("params", {})
        for name, spec in params.items():
            idx = spec.get("target_index")
            if idx is not None:
                self._param_indices[name] = int(idx)

        if not self._param_indices:
            logger.warning(f"No params with target_index found in {path}, using heuristic")
            self._use_heuristic_mapping()
            return

        self._num_params = max(self._param_indices.values()) + 1 if self._param_indices else 45

        # Classify indices by category
        for name, idx in self._param_indices.items():
            cat = _classify_param_name(name)
            if cat == "mouth":
                self._mouth_indices.append(idx)
            elif cat == "eye":
                self._eye_indices.append(idx)
            elif cat == "head":
                self._head_indices.append(idx)
            else:
                self._other_indices.append(idx)

        self._face_indices = self._mouth_indices + self._eye_indices
        logger.info(
            f"Loaded param mapping: {len(self._mouth_indices)} mouth, "
            f"{len(self._eye_indices)} eye, {len(self._head_indices)} head, "
            f"{len(self._other_indices)} other"
        )

    def _use_heuristic_mapping(self) -> None:
        """Fall back to heuristic index-based mapping."""
        self._mouth_indices = [i for i in range(6)]  # first 6: mouth-related
        self._eye_indices = [i for i in range(6, 20)]  # eyes, brows
        self._face_indices = list(range(0, 20))
        self._head_indices = list(range(31, 40))
        self._other_indices = list(range(20, 31)) + list(range(40, 45))
        self._num_params = 45
        logger.info("Using heuristic param mapping (45 params)")

    def reset(self) -> None:
        """Clear temporal smoothing state."""
        self._prev_params = None

    def apply(
        self,
        params: np.ndarray,
        mode: str = "speak",
    ) -> np.ndarray:
        """Apply performance multipliers to model output.

        Args:
            params: Live2D parameters, shape (T, 45) or (B, T, 45).
                    Values should be in [0, 1] range.
            mode: "speak", "listen", or "silence".
                  - speak/listen: apply gesture_scale, expressiveness, etc.
                  - silence: apply idle_energy instead of expressiveness.

        Returns:
            Adjusted parameters, same shape as input.
        """
        was_3d = params.ndim == 3
        if was_3d:
            B, T, P = params.shape
            # Process each batch item independently
            result = np.stack([self._apply_single(params[b], mode) for b in range(B)])
            return result
        else:
            return self._apply_single(params, mode)

    def _apply_single(self, params: np.ndarray, mode: str) -> np.ndarray:
        """Apply performance multipliers to a single sequence (T, P)."""
        result = params.copy()
        T, P = result.shape

        # 1. gesture_scale: scale all params around center (0.5)
        #    new_val = 0.5 + (val - 0.5) * gesture_scale
        result = 0.5 + (result - 0.5) * self.config.gesture_scale

        # 2. expressiveness: scale face params (mouth, eye, brow) around center
        if self._face_indices:
            face_idx = np.array(self._face_indices)
            face_vals = result[:, face_idx]
            if mode == "silence":
                # In silence mode, use idle_energy for face variation
                face_mean = face_vals.mean(axis=0, keepdims=True)
                result[:, face_idx] = face_mean + (face_vals - face_mean) * self.config.idle_energy
            else:
                # In speak/listen mode, scale face expressiveness around 0.5
                result[:, face_idx] = 0.5 + (face_vals - 0.5) * self.config.expressiveness

        # 3. mouth_open_max: cap mouth parameters
        if self._mouth_indices:
            mouth_idx = np.array(self._mouth_indices)
            result[:, mouth_idx] = np.clip(
                result[:, mouth_idx], 0.0, self.config.mouth_open_max
            )

        # 4. head_motion_range: scale head/body angles around 0.0
        #    These params are neutral at 0, so: new_val = val * head_motion_range
        if self._head_indices:
            head_idx = np.array(self._head_indices)
            result[:, head_idx] = result[:, head_idx] * self.config.head_motion_range

        # 5. react_speed: temporal smoothing (EMA-like)
        #    new_val = prev + react_speed * (current - prev)
        if self.config.react_speed < 1.0 and T > 1:
            result = self._apply_smoothing(result)

        # Clip final values to [0, 1]
        result = np.clip(result, 0.0, 1.0)

        return result

    def _apply_smoothing(self, params: np.ndarray) -> np.ndarray:
        """Apply exponential moving average smoothing based on react_speed.

        Higher react_speed = faster reaction (less smoothing).
        Lower react_speed = slower reaction (more smoothing).
        """
        result = params.copy()
        alpha = self.config.react_speed  # 0 = no change, 1 = instant

        T = result.shape[0]

        if self._prev_params is not None and self._prev_params.shape == result.shape[1:]:
            result[0] = self._prev_params + alpha * (result[0] - self._prev_params)

        for t in range(1, T):
            result[t] = result[t - 1] + alpha * (result[t] - result[t - 1])

        # Store last frame for next call
        self._prev_params = result[-1].copy()

        return result