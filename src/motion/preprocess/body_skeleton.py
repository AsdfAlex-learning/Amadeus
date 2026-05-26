"""Body skeleton extraction from video (STUB — not yet implemented).

Future: YOLOv8-pose for body keypoint extraction -> body Live2D params.
Anime source videos have unreliable body framing, so this is deferred from MVP.
"""

import numpy as np
from loguru import logger


class BodySkeletonExtractor:
    """Extract body keypoints from video frames. (STUB — returns zeros.)"""

    def __init__(self):
        logger.debug("BodySkeletonExtractor is a stub — returns zeros")

    def process_frame(self, frame: np.ndarray) -> dict | None:
        """Stub: returns zeros for body Live2D params.

        Args:
            frame: (H, W, 3) uint8 numpy array (unused).

        Returns:
            dict with zeroed body params.
        """
        return {
            "arm_l_x": 0.0,
            "arm_l_y": 0.0,
            "arm_r_x": 0.0,
            "arm_r_y": 0.0,
            "body_angle_x": 0.0,
            "body_angle_y": 0.0,
            "body_angle_z": 0.0,
        }

    def process_video(self, video_path: str, fps: float | None = None) -> dict:
        """Stub: returns empty body data.

        Args:
            video_path: Path to video file (unused).
            fps: Target FPS (unused).

        Returns:
            dict with zeroed body params array.
        """
        return {
            "body_params": np.zeros((0, 7), dtype=np.float32),
            "fps": fps or 30.0,
        }
