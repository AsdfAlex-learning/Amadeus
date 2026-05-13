import threading
from collections.abc import Callable

import numpy as np
from loguru import logger


class CameraPerception:
    def __init__(self, config: dict):
        perception_cfg = config["perception"]
        self.enable_face = bool(perception_cfg.get("enable_face_detection", True))
        self.enable_gaze = bool(perception_cfg.get("enable_gaze_estimation", True))
        self.enable_expression = bool(perception_cfg.get("enable_expression_recognition", True))
        self.device_index = perception_cfg.get("camera_device")

        self._cap = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._face_mesh = None
        self._callbacks: list[Callable[[np.ndarray], None]] = []

    def start(self):
        if self._running:
            return
        try:
            import cv2
            self._cap = cv2.VideoCapture(self.device_index if self.device_index is not None else 0)
            if not self._cap.isOpened():
                logger.error("Could not open camera")
                return
            self._load_mediapipe()
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            logger.info("Camera perception started")
        except ImportError:
            logger.warning("opencv-python not installed")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera perception stopped")

    def cleanup(self):
        self.stop()
        self._callbacks.clear()

    def on_frame(self, callback: Callable[[np.ndarray], None]):
        self._callbacks.append(callback)

    def _load_mediapipe(self):
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("MediaPipe loaded")
        except ImportError:
            logger.warning("mediapipe not installed")

    def _capture_loop(self):
        import cv2
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (224, 224))
            normalized = resized.astype(np.float32) / 255.0
            for cb in self._callbacks:
                try:
                    cb(normalized)
                except Exception as e:
                    logger.error(f"Frame callback error: {e}")

    @property
    def is_running(self) -> bool:
        return self._running
