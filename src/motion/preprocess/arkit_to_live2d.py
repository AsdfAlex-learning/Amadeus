"""Map 52 ARKit blendshapes + 3 head angles to 45 Live2D parameters.

Mapping is configured via YAML files in src/motion/preprocess/mappings/.
Each Live2D param can have multiple ARKit sources with weights and optional bias.
"""

from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from src.motion.preprocess.face_landmarker import ARKIT_BLENDSHAPE_NAMES, _ARKIT_NAME_TO_IDX

# Canonical 45 Live2D parameter names — MUST match live2d_widget.py push_params LUT
LIVE2D_PARAM_NAMES = [
    "ParamAngleX", "ParamAngleY", "ParamAngleZ",
    "ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ",
    "ParamEyeLOpen", "ParamEyeROpen", "ParamEyeBallX", "ParamEyeBallY",
    "ParamBrowLX", "ParamBrowLY", "ParamBrowRX", "ParamBrowRY",
    "ParamMouthOpenY", "ParamMouthForm", "ParamCheek", "ParamBreath",
    "ParamArmLX", "ParamArmLY", "ParamArmRX", "ParamArmRY",
    "ParamHairFront", "ParamHairBack", "ParamHairSideL", "ParamHairSideR",
    "ParamTear", "ParamBlush", "ParamNose",
    "ParamLipUpper", "ParamLipLower", "ParamTongue",
    "ParamEarL", "ParamEarR", "ParamTail", "ParamWingL", "ParamWingR",
    "ParamItem1", "ParamItem2", "ParamItem3",
    "ParamExtra1", "ParamExtra2", "ParamExtra3", "ParamExtra4", "ParamExtra5",
]

NUM_LIVE2D_PARAMS = len(LIVE2D_PARAM_NAMES)  # 45


class ARKitToLive2DMapper:
    """Map ARKit blendshapes to Live2D parameters using YAML configuration."""

    DEFAULT_MAPPING_PATH = Path(__file__).parent / "mappings" / "default.yaml"

    def __init__(self, mapping_path: str | Path | None = None):
        if mapping_path is None:
            mapping_path = self.DEFAULT_MAPPING_PATH
        self._mapping_path = Path(mapping_path)
        self._mappings: dict = {}
        self._live2d_order: list[str] = []
        self._load_mapping()

    def _load_mapping(self):
        if not self._mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {self._mapping_path}")
        with open(self._mapping_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._live2d_order = data.get("live2d_param_order", LIVE2D_PARAM_NAMES)
        self._mappings = data.get("mappings", {})
        warnings = self.validate()
        for w in warnings:
            logger.warning(w)

    def map(self, blendshapes: np.ndarray, head_angles: np.ndarray) -> np.ndarray:
        """Map ARKit blendshapes + head angles to Live2D parameters.

        Args:
            blendshapes: (..., 52) float32 ARKit blendshape values in [0, 1].
            head_angles: (..., 3) float32 [pitch, yaw, roll] in degrees.

        Returns:
            (..., 45) float32 Live2D parameter values in [0, 1].
        """
        single = blendshapes.ndim == 1
        if single:
            blendshapes = blendshapes.unsqueeze(0) if hasattr(blendshapes, 'unsqueeze') else blendshapes[np.newaxis, :]
            head_angles = head_angles[np.newaxis, :]

        T = blendshapes.shape[0]
        result = np.zeros((T, NUM_LIVE2D_PARAMS), dtype=np.float32)

        for param_idx, param_name in enumerate(self._live2d_order):
            rule = self._mappings.get(param_name)
            if rule is None:
                continue  # unmapped → stays 0

            rule_type = rule.get("type", "constant")

            if rule_type == "constant":
                result[:, param_idx] = float(rule.get("value", 0.0))

            elif rule_type == "head_angle":
                source = rule.get("source", "yaw")
                angle_idx = {"pitch": 0, "yaw": 1, "roll": 2}[source]
                # Support both "weight" and "scale" keys for compatibility
                weight = float(rule.get("weight", rule.get("scale", 1.0)))
                bias = float(rule.get("bias", 0.0))
                result[:, param_idx] = np.clip(
                    head_angles[:, angle_idx] * weight + bias, 0.0, 1.0)

            elif rule_type == "blendshape":
                sources = rule.get("sources", [])
                value = np.full(T, float(rule.get("bias", 0.0)), dtype=np.float32)
                for src in sources:
                    name = src.get("name", "")
                    idx = _ARKIT_NAME_TO_IDX.get(name)
                    if idx is not None:
                        weight = float(src.get("weight", 1.0))
                        value += blendshapes[:, idx] * weight
                result[:, param_idx] = np.clip(value, 0.0, 1.0)

        if single:
            result = result[0]
        return result

    def validate(self) -> list[str]:
        """Validate that all 45 Live2D params have mappings.

        Returns list of warning strings for unmapped params.
        """
        warnings = []
        for name in self._live2d_order:
            if name not in self._mappings:
                warnings.append(f"No mapping for Live2D param: {name}")
        return warnings

    @property
    def live2d_param_names(self) -> list[str]:
        return list(self._live2d_order)

    @property
    def missing_mappings(self) -> list[str]:
        return [n for n in self._live2d_order if n not in self._mappings]
