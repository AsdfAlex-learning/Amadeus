"""End-to-end video -> training data preprocessing pipeline.

Orchestrates: VideoReader -> FaceLandmarkerExtractor -> ARKitToLive2DMapper -> .npz output.

Usage:
    python -m src.motion.preprocess.pipeline input_video.mp4 --output_dir data/preprocessed
    python -m src.motion.preprocess.pipeline input_dir/ --output_dir data/preprocessed --manifest manifest.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from src.motion.preprocess.arkit_to_live2d import ARKitToLive2DMapper, NUM_LIVE2D_PARAMS
from src.motion.preprocess.face_landmarker import FaceLandmarkerExtractor
from src.motion.preprocess.video_reader import VideoReader

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}


class PreprocessPipeline:
    """End-to-end video -> training data preprocessing pipeline."""

    def __init__(
        self,
        mapping_path: str | Path | None = None,
        model_dir: str | Path = "models/mediapipe",
        target_fps: float = 25.0,
    ):
        self.target_fps = target_fps
        self.video_reader: VideoReader | None = None
        self.landmarker = FaceLandmarkerExtractor(model_dir=model_dir)
        self.mapper = ARKitToLive2DMapper(mapping_path=mapping_path)

    def process_video(
        self, video_path: str | Path, output_dir: str | Path, identity_id: int = 0
    ) -> Path | None:
        """Process a single video file -> .npz + _meta.json.

        Steps:
        1. Extract audio to 16kHz WAV
        2. Extract frames at target_fps via FaceLandmarker
        3. Map ARKit -> Live2D params
        4. Save .npz and _meta.json

        Returns path to the .npz file, or None if no face detected.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing: {video_path.name}")

        # 1. Extract audio
        reader = VideoReader(video_path, fps=self.target_fps)
        audio_path = reader.extract_audio(
            output_dir / f"{video_path.stem}_audio.wav", sample_rate=16000
        )
        logger.info(f"Audio extracted: {audio_path}")

        # 2. Extract blendshapes from video
        face_data = self.landmarker.process_video(video_path, fps=self.target_fps)
        blendshapes = face_data["blendshapes"]  # (T, 52)
        head_angles = face_data["head_angles"]  # (T, 3)
        bad_frames = face_data["bad_frames"]

        T = blendshapes.shape[0]
        if T == 0:
            logger.warning(f"No frames extracted from {video_path}, skipping")
            return None

        # Check if too many bad frames
        bad_ratio = len(bad_frames) / T if T > 0 else 1.0
        if bad_ratio > 0.9:
            logger.warning(
                f"Too many bad frames ({bad_ratio:.0%}) in {video_path}, skipping"
            )
            return None

        # 3. Map ARKit -> Live2D params
        live2d_params = self.mapper.map(blendshapes, head_angles)  # (T, 45)
        assert live2d_params.shape == (T, NUM_LIVE2D_PARAMS), (
            f"Expected ({T}, {NUM_LIVE2D_PARAMS}), got {live2d_params.shape}"
        )

        # Interpolate bad frames (replace with average of neighbors)
        if bad_frames:
            live2d_params = self._interpolate_bad_frames(live2d_params, bad_frames)

        # 4. Get video metadata
        meta = reader.get_metadata()
        duration_sec = meta.get("duration_sec", T / self.target_fps)

        # 5. Save .npz
        npz_path = output_dir / f"{video_path.stem}.npz"
        np.savez(
            npz_path,
            live2d_params=live2d_params.astype(np.float32),
            blendshapes=blendshapes.astype(np.float32),
            head_angles=head_angles.astype(np.float32),
            bad_frames=np.array(bad_frames, dtype=np.int64),
            fps=np.float32(self.target_fps),
            duration_sec=np.float32(duration_sec),
            identity_id=np.int64(identity_id),
            source_video=str(video_path),
        )
        logger.info(f"Saved: {npz_path} ({T} frames, {len(bad_frames)} bad)")

        # 6. Save _meta.json
        meta_path = output_dir / f"{video_path.stem}_meta.json"
        meta_data = {
            "source_video": str(video_path),
            "identity_id": identity_id,
            "mapping_file": str(self.mapper._mapping_path),
            "extraction_fps": self.target_fps,
            "num_frames": T,
            "duration_sec": duration_sec,
            "bad_frame_count": len(bad_frames),
            "live2d_param_count": NUM_LIVE2D_PARAMS,
            "created_at": datetime.now().isoformat(),
            "amadeus_version": "0.1.0",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)

        return npz_path

    def process_directory(
        self,
        video_dir: str | Path,
        output_dir: str | Path,
        manifest_path: str | Path | None = None,
    ) -> list[Path]:
        """Process all videos in a directory.

        If manifest_path is provided, read it for video->identity_id mappings.
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)

        # Find video files
        videos = sorted(
            p for p in video_dir.iterdir()
            if p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            logger.warning(f"No video files found in {video_dir}")
            return []

        # Load manifest if provided
        identity_map: dict[str, int] = {}
        default_identity = 0
        if manifest_path is not None:
            manifest_path = Path(manifest_path)
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                identity_map = manifest.get("identities", {})
                default_identity = manifest.get("default_identity", 0)

        results = []
        for video in videos:
            identity_id = identity_map.get(video.name, default_identity)
            npz_path = self.process_video(video, output_dir, identity_id)
            if npz_path is not None:
                results.append(npz_path)

        logger.info(f"Processed {len(results)}/{len(videos)} videos successfully")
        return results

    @staticmethod
    def _interpolate_bad_frames(
        params: np.ndarray, bad_frames: list[int]
    ) -> np.ndarray:
        """Replace bad frames with interpolated values from neighbors."""
        params = params.copy()
        for idx in bad_frames:
            # Find nearest good frame before and after
            prev_good = idx - 1
            while prev_good >= 0 and prev_good in bad_frames:
                prev_good -= 1
            next_good = idx + 1
            while next_good < len(params) and next_good in bad_frames:
                next_good += 1

            if prev_good >= 0 and next_good < len(params):
                # Average of neighbors
                params[idx] = (params[prev_good] + params[next_good]) * 0.5
            elif prev_good >= 0:
                params[idx] = params[prev_good]
            elif next_good < len(params):
                params[idx] = params[next_good]
            # else: leave as-is (all bad)
        return params

    def cleanup(self):
        self.landmarker.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Amadeus Video -> Training Data Pipeline"
    )
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument(
        "--output_dir", default="data/preprocessed", help="Output directory"
    )
    parser.add_argument(
        "--mapping", default=None, help="ARKit->Live2D mapping YAML path"
    )
    parser.add_argument(
        "--manifest", default=None, help="Video->identity manifest JSON"
    )
    parser.add_argument(
        "--fps", type=float, default=25.0, help="Target extraction FPS"
    )
    parser.add_argument(
        "--model_dir", default="models/mediapipe", help="MediaPipe model dir"
    )
    args = parser.parse_args()

    pipeline = PreprocessPipeline(
        mapping_path=args.mapping,
        model_dir=args.model_dir,
        target_fps=args.fps,
    )

    input_path = Path(args.input)
    try:
        if input_path.is_file():
            pipeline.process_video(input_path, args.output_dir)
        elif input_path.is_dir():
            pipeline.process_directory(
                input_path, args.output_dir, manifest_path=args.manifest
            )
        else:
            logger.error(f"Input not found: {input_path}")
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
