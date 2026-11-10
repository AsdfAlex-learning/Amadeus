"""YOLOv8-pose body keypoints to the seven body Live2D parameters.

The face path already produces 45 Live2D parameters. This module adds the body
half of the Skeleton-to-Live2D mapping described in CONTEXT.md. ``ultralytics``
is optional: without it the extractor is unavailable and the preprocessing
pipeline simply omits ``body_params``.

The output is deliberately rule-based and normalized to [0, 1]:
``arm_l_x/y`` and ``arm_r_x/y`` are wrist offsets from the shoulder,
``body_angle_x`` is lateral lean, ``body_angle_y`` is a head-over-shoulder pitch
proxy, and ``body_angle_z`` is shoulder-line roll. These are training targets,
not display-ready Cubism angles.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

BODY_PARAM_KEYS = (
    "arm_l_x",
    "arm_l_y",
    "arm_r_x",
    "arm_r_y",
    "body_angle_x",
    "body_angle_y",
    "body_angle_z",
)

COCO_KEYPOINTS = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}

_MIN_KEYPOINT_CONF = 0.2


class BodySkeletonExtractor:
    """Extract body parameters from frames with YOLOv8-pose."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf: float = 0.25,
        enabled: bool = True,
        model: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.conf = float(conf)
        self.enabled = bool(enabled)
        self._model = model
        self._available = model is not None
        if not self.enabled or model is not None:
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(model_path or "yolov8n-pose.pt"))
            self._available = True
        except ImportError:
            logger.warning("ultralytics not installed; body skeleton extraction disabled")
        except Exception as e:
            logger.warning(f"Could not load YOLOv8-pose model: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    def process_frame(self, frame: np.ndarray) -> dict[str, float] | None:
        """Return one frame's body parameters, or None when no person is found."""
        if not self._available or self._model is None or self.enabled is False:
            return None
        if not isinstance(frame, np.ndarray) or frame.ndim < 2 or frame.shape[0] < 2:
            return None
        try:
            results = self._model.predict(frame, verbose=False, conf=self.conf)
        except Exception as e:
            logger.error(f"YOLO pose inference failed: {e}")
            return None
        if not results:
            return None
        keypoints = getattr(results[0], "keypoints", None)
        if keypoints is None:
            return None
        xy = np.asarray(getattr(keypoints, "xy", []), dtype=np.float32)
        if xy.ndim != 3 or xy.shape[0] == 0:
            return None
        conf = np.asarray(getattr(keypoints, "conf", []), dtype=np.float32)
        if conf.ndim != 2 or conf.shape[0] != xy.shape[0]:
            conf = np.ones(xy.shape[:2], dtype=np.float32)
        person = int(np.argmax(conf.mean(axis=1)))
        return self._map_keypoints(xy[person], conf[person], frame.shape)

    @staticmethod
    def _map_keypoints(
        keypoints: np.ndarray, confidence: np.ndarray, frame_shape: tuple[int, ...]
    ) -> dict[str, float] | None:
        height, width = frame_shape[:2]
        if width <= 0 or height <= 0:
            return None

        def point(name: str) -> tuple[float, float] | None:
            index = COCO_KEYPOINTS[name]
            if index >= len(keypoints):
                return None
            if index < len(confidence) and float(confidence[index]) < _MIN_KEYPOINT_CONF:
                return None
            x, y = keypoints[index][:2]
            if not np.isfinite(x) or not np.isfinite(y):
                return None
            return float(x), float(y)

        left_shoulder = point("left_shoulder")
        right_shoulder = point("right_shoulder")
        left_hip = point("left_hip")
        right_hip = point("right_hip")
        nose = point("nose")
        if left_shoulder is None or right_shoulder is None or left_hip is None or right_hip is None:
            return None

        def clip(value: float) -> float:
            return float(np.clip(value, 0.0, 1.0))

        def arm(shoulder: tuple[float, float], wrist_name: str) -> tuple[float, float]:
            wrist = point(wrist_name)
            if wrist is None:
                return 0.5, 0.5
            return (
                clip(0.5 + (wrist[0] - shoulder[0]) / width),
                clip(0.5 + (wrist[1] - shoulder[1]) / height),
            )

        arm_l_x, arm_l_y = arm(left_shoulder, "left_wrist")
        arm_r_x, arm_r_y = arm(right_shoulder, "right_wrist")

        shoulder_cx = (left_shoulder[0] + right_shoulder[0]) * 0.5
        shoulder_cy = (left_shoulder[1] + right_shoulder[1]) * 0.5
        hip_cx = (left_hip[0] + right_hip[0]) * 0.5
        body_angle_x = clip(0.5 + (shoulder_cx - hip_cx) / width)

        if nose is not None:
            body_angle_y = clip(0.5 + ((nose[1] - shoulder_cy) / height + 0.15))
        else:
            body_angle_y = 0.5

        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        shoulder_angle = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-6 else 0.0
        body_angle_z = clip(0.5 + shoulder_angle / math.pi)

        return {
            "arm_l_x": arm_l_x,
            "arm_l_y": arm_l_y,
            "arm_r_x": arm_r_x,
            "arm_r_y": arm_r_y,
            "body_angle_x": body_angle_x,
            "body_angle_y": body_angle_y,
            "body_angle_z": body_angle_z,
        }

    def process_video(self, video_path: str | Path, fps: float | None = None) -> dict[str, Any]:
        """Extract body parameters from a video, sampling at ``fps``."""
        target_fps = float(fps) if fps else 30.0
        empty = {
            "body_params": np.zeros((0, len(BODY_PARAM_KEYS)), dtype=np.float32),
            "fps": target_fps,
            "detected_frames": 0,
        }
        if not self._available or not self.enabled:
            return empty
        try:
            import cv2
        except ImportError:
            logger.warning("opencv-python not installed; body skeleton extraction disabled")
            return empty

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video for body extraction: {video_path}")
            return empty
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or target_fps)
        step = max(1, int(round(source_fps / max(target_fps, 1e-6))))
        rows: list[np.ndarray] = []
        detected = 0
        index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % step == 0:
                    mapped = self.process_frame(frame)
                    if mapped is None:
                        rows.append(np.zeros(len(BODY_PARAM_KEYS), dtype=np.float32))
                    else:
                        rows.append(
                            np.array([mapped[key] for key in BODY_PARAM_KEYS], dtype=np.float32)
                        )
                        detected += 1
                index += 1
        finally:
            cap.release()

        params = (
            np.stack(rows).astype(np.float32)
            if rows
            else np.zeros((0, len(BODY_PARAM_KEYS)), dtype=np.float32)
        )
        return {"body_params": params, "fps": target_fps, "detected_frames": detected}

    def cleanup(self) -> None:
        self._model = None
        self._available = False
