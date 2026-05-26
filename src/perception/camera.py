import math
import threading
from collections.abc import Callable
from typing import Any

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
        self._perception_callbacks: list[Callable[[dict[str, Any]], None]] = []

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
        self._perception_callbacks.clear()
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception as e:
                logger.error(f"FaceMesh close error: {e}")
            self._face_mesh = None

    def on_frame(self, callback: Callable[[np.ndarray], None]):
        self._callbacks.append(callback)

    def on_perception(self, callback: Callable[[dict[str, Any]], None]):
        self._perception_callbacks.append(callback)

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

            if self._perception_callbacks and self._face_mesh is not None:
                perception = self._process_face_mesh(frame_rgb)
                for cb in self._perception_callbacks:
                    try:
                        cb(perception)
                    except Exception as e:
                        logger.error(f"Perception callback error: {e}")

    def _process_face_mesh(self, frame_rgb: np.ndarray) -> dict[str, Any]:
        result: dict[str, Any] = {
            "face_detected": False,
            "face_landmarks": None,
            "gaze_direction": "center",
            "mouth_openness": 0.0,
            "eye_openness_l": 0.0,
            "eye_openness_r": 0.0,
            "brow_raise": 0.0,
            "smile": 0.0,
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "head_roll": 0.0,
        }
        if self._face_mesh is None:
            return result
        try:
            mp_result = self._face_mesh.process(frame_rgb)
        except Exception as e:
            logger.error(f"FaceMesh process error: {e}")
            return result

        if not mp_result.multi_face_landmarks:
            return result

        face = mp_result.multi_face_landmarks[0]
        h, w = frame_rgb.shape[:2]
        landmarks = np.array(
            [[lm.x * w, lm.y * h, lm.z * w] for lm in face.landmark],
            dtype=np.float32,
        )
        result["face_detected"] = True
        result["face_landmarks"] = landmarks

        # Face vertical extent (forehead lm 10 to chin lm 152)
        face_height = float(np.linalg.norm(landmarks[10][:2] - landmarks[152][:2]))
        if face_height <= 1e-6:
            return result

        # Mouth openness: distance between upper inner lip (13) and lower inner lip (14)
        if self.enable_expression:
            mouth_gap = float(np.linalg.norm(landmarks[13][:2] - landmarks[14][:2]))
            result["mouth_openness"] = float(np.clip(mouth_gap / (face_height * 0.15), 0.0, 1.0))

            # Eye openness: vertical distance between upper/lower eyelid
            # Left eye: upper 159, lower 145
            left_eye_gap = float(np.linalg.norm(landmarks[159][:2] - landmarks[145][:2]))
            # Right eye: upper 386, lower 374
            right_eye_gap = float(np.linalg.norm(landmarks[386][:2] - landmarks[374][:2]))
            result["eye_openness_l"] = float(
                np.clip(left_eye_gap / (face_height * 0.05), 0.0, 1.0)
            )
            result["eye_openness_r"] = float(
                np.clip(right_eye_gap / (face_height * 0.05), 0.0, 1.0)
            )

            # Brow raise: vertical distance between brow (105 left, 334 right) and eye
            # Use left brow tip 105 to left eye top 159
            brow_eye_l = float(landmarks[159][1] - landmarks[105][1])
            brow_eye_r = float(landmarks[386][1] - landmarks[334][1])
            avg_brow = (brow_eye_l + brow_eye_r) * 0.5
            # Larger gap = more raised. Normalize against a baseline (~face_height * 0.08)
            result["brow_raise"] = float(np.clip(avg_brow / (face_height * 0.12), 0.0, 1.0))

            # Smile: mouth corner elevation relative to mouth center
            # Mouth corners: 61 (left), 291 (right). Mouth center upper: 13
            mouth_center_y = landmarks[13][1]
            corner_l_y = landmarks[61][1]
            corner_r_y = landmarks[291][1]
            # Negative (corners above center) = smile
            avg_corner_lift = mouth_center_y - (corner_l_y + corner_r_y) * 0.5
            result["smile"] = float(np.clip(avg_corner_lift / (face_height * 0.05), 0.0, 1.0))

        # Head pose: nose tip (1) relative to face center (between cheeks 234, 454)
        nose_tip = landmarks[1]
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        face_center_x = (left_cheek[0] + right_cheek[0]) * 0.5
        face_center_y = (landmarks[10][1] + landmarks[152][1]) * 0.5
        face_width = float(np.linalg.norm(left_cheek[:2] - right_cheek[:2]))
        if face_width > 1e-6:
            # Yaw: nose horizontal offset from face center
            yaw_offset = (nose_tip[0] - face_center_x) / (face_width * 0.5)
            result["head_yaw"] = float(np.clip(yaw_offset * 45.0, -90.0, 90.0))
            # Pitch: nose vertical offset from face center
            pitch_offset = (nose_tip[1] - face_center_y) / (face_height * 0.5)
            result["head_pitch"] = float(np.clip(pitch_offset * 45.0, -90.0, 90.0))
            # Roll: angle between the two cheeks
            dx = right_cheek[0] - left_cheek[0]
            dy = right_cheek[1] - left_cheek[1]
            result["head_roll"] = float(np.clip(math.degrees(math.atan2(dy, dx)), -90.0, 90.0))

        # Gaze: iris landmarks (refine_landmarks=True provides 468-477)
        # Left iris center: 468, Right iris center: 473
        if self.enable_gaze and landmarks.shape[0] >= 478:
            # Left eye corners: 33 (outer), 133 (inner)
            left_eye_outer = landmarks[33]
            left_eye_inner = landmarks[133]
            left_iris = landmarks[468]
            left_eye_width = float(np.linalg.norm(left_eye_outer[:2] - left_eye_inner[:2]))
            if left_eye_width > 1e-6:
                # Position of iris along eye width (0 outer .. 1 inner)
                rel = (left_iris[0] - left_eye_outer[0]) / (left_eye_inner[0] - left_eye_outer[0] + 1e-6)
                if rel < 0.35:
                    result["gaze_direction"] = "right"  # mirrored: outer is screen-right for left eye
                elif rel > 0.65:
                    result["gaze_direction"] = "left"
                else:
                    result["gaze_direction"] = "center"

        return result

    @property
    def is_running(self) -> bool:
        return self._running
