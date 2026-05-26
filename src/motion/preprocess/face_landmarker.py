"""MediaPipe FaceLandmarker: extract 52 ARKit blendshapes + head pose from video frames.

Uses the Tasks API (face_landmarker_v2_with_blendshapes.task).
Auto-downloads the model asset if not cached locally.
Falls back to legacy solutions.FaceMesh when Tasks API is unavailable.
"""

import math
import urllib.request
from pathlib import Path

import numpy as np
from loguru import logger


ARKIT_BLENDSHAPE_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]

NUM_BLENDSHAPES = len(ARKIT_BLENDSHAPE_NAMES)  # 52
_ARKIT_NAME_TO_IDX = {name: i for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES)}


class FaceLandmarkerExtractor:
    """Extract ARKit blendshapes and head pose from video frames."""

    MODEL_FILENAME = "face_landmarker_v2_with_blendshapes.task"
    FALLBACK_MODEL_FILENAME = "face_landmarker.task"
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )

    def __init__(self, model_dir: str | Path = "models/mediapipe", delegate: str = "CPU"):
        self._model_dir = Path(model_dir)
        self._delegate = delegate
        self._landmarker = None
        self._use_tasks_api = False
        self._legacy_face_mesh = None
        self._init_landmarker()

    def _init_landmarker(self):
        try:
            self._init_tasks_api()
            self._use_tasks_api = True
            logger.info("FaceLandmarker initialized (Tasks API)")
            return
        except Exception as e:
            logger.debug(f"Tasks API FaceLandmarker failed: {e}")
        try:
            self._init_legacy()
            self._use_tasks_api = False
            logger.info("FaceLandmarker initialized (legacy FaceMesh fallback)")
        except Exception as e:
            logger.warning(f"FaceLandmarker unavailable: {e}. All extractions will return zeros.")

    def _init_tasks_api(self):
        from mediapipe.tasks.python import BaseOptions, vision
        model_path = self._ensure_model()
        base_options = BaseOptions(model_asset_path=str(model_path), delegate=self._delegate)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def _init_legacy(self):
        import mediapipe as mp
        self._legacy_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    def _ensure_model(self) -> Path:
        self._model_dir.mkdir(parents=True, exist_ok=True)
        for filename in [self.MODEL_FILENAME, self.FALLBACK_MODEL_FILENAME]:
            p = self._model_dir / filename
            if p.exists() and p.stat().st_size > 0:
                return p
        logger.info(f"Downloading FaceLandmarker model to {self._model_dir}...")
        target = self._model_dir / self.FALLBACK_MODEL_FILENAME
        try:
            urllib.request.urlretrieve(self.MODEL_URL, str(target))
            logger.info(f"Model downloaded: {target}")
            return target
        except Exception as e:
            logger.warning(f"Model download failed: {e}")
            raise RuntimeError(
                f"Could not download FaceLandmarker model. "
                f"Manually download from {self.MODEL_URL} to {target}"
            ) from e

    # ── Per-frame extraction ──

    def process_frame(self, frame: np.ndarray) -> dict:
        """Process a single BGR frame and extract blendshapes + head pose.

        Args:
            frame: (H, W, 3) uint8 BGR numpy array.

        Returns:
            dict with keys: blendshapes (52,), head_pitch, head_yaw, head_roll, face_detected
        """
        default = {
            "blendshapes": np.zeros(NUM_BLENDSHAPES, dtype=np.float32),
            "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
            "face_detected": False,
        }
        if self._use_tasks_api and self._landmarker is not None:
            return self._process_frame_tasks(frame, default)
        elif self._legacy_face_mesh is not None:
            return self._process_frame_legacy(frame, default)
        return default

    def _process_frame_tasks(self, frame: np.ndarray, default: dict) -> dict:
        import mediapipe as mp
        frame_rgb = frame[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_blendshapes:
            return default
        blendshapes = np.zeros(NUM_BLENDSHAPES, dtype=np.float32)
        for i, bs in enumerate(result.face_blendshapes[0]):
            if i < NUM_BLENDSHAPES:
                blendshapes[i] = float(bs.score)
        head_pitch, head_yaw, head_roll = 0.0, 0.0, 0.0
        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
            head_pitch, head_yaw, head_roll = self._decompose_rotation(matrix)
        return {"blendshapes": blendshapes, "head_pitch": head_pitch,
                "head_yaw": head_yaw, "head_roll": head_roll, "face_detected": True}

    def _process_frame_legacy(self, frame: np.ndarray, default: dict) -> dict:
        import cv2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result = self._legacy_face_mesh.process(frame_rgb)
        if not mp_result.multi_face_landmarks:
            return default
        face = mp_result.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        landmarks = np.array(
            [[lm.x * w, lm.y * h, lm.z * w] for lm in face.landmark], dtype=np.float32)
        blendshapes = self._landmarks_to_blendshapes(landmarks)
        head_pitch, head_yaw, head_roll = self._estimate_head_pose(landmarks)
        return {"blendshapes": blendshapes, "head_pitch": head_pitch,
                "head_yaw": head_yaw, "head_roll": head_roll, "face_detected": True}

    # ── Blendshape approximation from landmarks (legacy fallback) ──

    def _landmarks_to_blendshapes(self, lm: np.ndarray) -> np.ndarray:
        """Approximate 52 ARKit blendshapes from 478 MediaPipe landmarks."""
        bs = np.zeros(NUM_BLENDSHAPES, dtype=np.float32)
        face_h = max(1e-6, float(np.linalg.norm(lm[10][:2] - lm[152][:2])))
        face_w = max(1e-6, float(np.linalg.norm(lm[234][:2] - lm[454][:2])))

        def dist(a: int, b: int) -> float:
            return float(np.linalg.norm(lm[a][:2] - lm[b][:2]))

        # eyeBlinkLeft (8), eyeBlinkRight (9)
        bs[8] = float(np.clip(1.0 - dist(159, 145) / (face_h * 0.05), 0, 1))
        bs[9] = float(np.clip(1.0 - dist(386, 374) / (face_h * 0.05), 0, 1))
        # browInnerUp (2)
        brow_l = float(lm[159][1] - lm[107][1])
        brow_r = float(lm[386][1] - lm[336][1])
        bs[2] = float(np.clip((brow_l + brow_r) * 0.5 / (face_h * 0.08), 0, 1))
        # browDownLeft (0), browDownRight (1)
        bs[0] = float(np.clip(1.0 - brow_l / (face_h * 0.08), 0, 1)) * 0.3
        bs[1] = float(np.clip(1.0 - brow_r / (face_h * 0.08), 0, 1)) * 0.3
        # jawOpen (24)
        bs[24] = float(np.clip(dist(13, 14) / (face_h * 0.12), 0, 1))
        # mouthSmileLeft (43), mouthSmileRight (44)
        mc_y = lm[13][1]
        bs[43] = float(np.clip(max(0, mc_y - lm[61][1]) / (face_h * 0.04), 0, 1))
        bs[44] = float(np.clip(max(0, mc_y - lm[291][1]) / (face_h * 0.04), 0, 1))
        # mouthFrownLeft (30), mouthFrownRight (31)
        bs[30] = float(np.clip(max(0, lm[61][1] - mc_y) / (face_h * 0.04), 0, 1))
        bs[31] = float(np.clip(max(0, lm[291][1] - mc_y) / (face_h * 0.04), 0, 1))
        # mouthPucker (37), mouthFunnel (32)
        mw = dist(61, 291)
        bs[37] = float(np.clip(1.0 - mw / (face_w * 0.4), 0, 1)) * 0.5
        bs[32] = bs[37] * 0.7
        # mouthClose (27)
        bs[27] = float(np.clip(1.0 - bs[24], 0, 1)) * 0.3
        # cheekPuff (5)
        bs[5] = float(np.clip(dist(116, 345) / face_w - 0.7, 0, 0.3)) * 2.0
        # noseSneerLeft (50), noseSneerRight (51)
        bs[50] = float(np.clip((lm[66][1] - lm[105][1]) / (face_h * 0.03), 0, 1))
        bs[51] = float(np.clip((lm[297][1] - lm[334][1]) / (face_h * 0.03), 0, 1))
        # eyeWideLeft (20), eyeWideRight (21)
        bs[20] = float(np.clip(bs[2] * 0.6, 0, 1))
        bs[21] = bs[20]
        # mouthLowerDownLeft (33), mouthLowerDownRight (34)
        bs[33] = float(np.clip((lm[14][1] - lm[13][1]) / (face_h * 0.04), 0, 1))
        bs[34] = bs[33]
        # mouthUpperUpLeft (48), mouthUpperUpRight (49)
        bs[48] = bs[33] * 0.5
        bs[49] = bs[48]
        return np.clip(bs, 0.0, 1.0)

    # ── Head pose ──

    @staticmethod
    def _decompose_rotation(matrix: np.ndarray) -> tuple[float, float, float]:
        """Decompose 4x4 transformation matrix to euler angles (degrees)."""
        r = matrix[:3, :3]
        yaw = float(np.arctan2(-r[2, 0], np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)))
        pitch = float(np.arctan2(r[2, 1], r[2, 2]))
        roll = float(np.arctan2(r[1, 0], r[0, 0]))
        return (float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll)))

    @staticmethod
    def _estimate_head_pose(landmarks: np.ndarray) -> tuple[float, float, float]:
        """Estimate head pose from face landmarks."""
        nose = landmarks[1]
        lc, rc = landmarks[234], landmarks[454]
        fh, chin = landmarks[10], landmarks[152]
        cx = (lc[0] + rc[0]) * 0.5
        cy = (fh[1] + chin[1]) * 0.5
        fw = max(1e-6, float(np.linalg.norm(lc[:2] - rc[:2])))
        fht = max(1e-6, float(np.linalg.norm(fh[:2] - chin[:2])))
        yaw = float(np.clip((nose[0] - cx) / (fw * 0.5) * 45.0, -90, 90))
        pitch = float(np.clip((nose[1] - cy) / (fht * 0.5) * 45.0, -90, 90))
        dx, dy = rc[0] - lc[0], rc[1] - lc[1]
        roll = float(np.clip(math.degrees(math.atan2(dy, dx)), -90, 90))
        return pitch, yaw, roll

    # ── Batch video processing ──

    def process_video(self, video_path: str | Path, fps: float | None = None) -> dict:
        """Process entire video, extracting blendshapes for every frame."""
        from src.motion.preprocess.video_reader import VideoReader
        video_path = Path(video_path)
        reader = VideoReader(video_path, fps=fps)
        all_blendshapes = []
        all_head_angles = []
        bad_frames = []
        frame_idx = 0
        for frame in reader.iter_frames(fps=fps):
            result = self.process_frame(frame)
            all_blendshapes.append(result["blendshapes"])
            all_head_angles.append(
                [result["head_pitch"], result["head_yaw"], result["head_roll"]])
            if not result["face_detected"]:
                bad_frames.append(frame_idx)
            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx} frames...")
        if not all_blendshapes:
            logger.warning(f"No frames processed from {video_path}")
            return {
                "blendshapes": np.zeros((0, NUM_BLENDSHAPES), dtype=np.float32),
                "head_angles": np.zeros((0, 3), dtype=np.float32),
                "bad_frames": [], "fps": fps or 25.0,
            }
        logger.info(f"Processed {frame_idx} frames from {video_path.name} "
                     f"({len(bad_frames)} bad frames)")
        return {
            "blendshapes": np.stack(all_blendshapes),
            "head_angles": np.stack(all_head_angles).astype(np.float32),
            "bad_frames": bad_frames,
            "fps": fps or 25.0,
        }

    @property
    def blendshape_names(self) -> list[str]:
        return list(ARKIT_BLENDSHAPE_NAMES)

    def cleanup(self):
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
        if self._legacy_face_mesh is not None:
            try:
                self._legacy_face_mesh.close()
            except Exception:
                pass
            self._legacy_face_mesh = None
