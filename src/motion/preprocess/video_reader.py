"""Video I/O utilities: demux audio + extract frames from video files.

Uses subprocess calls to ffmpeg/ffprobe (must be on PATH or configured).
Falls back to cv2.VideoCapture when ffmpeg is unavailable.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger


class VideoReader:
    """Read video frames and extract audio from a video file using ffmpeg."""

    def __init__(self, video_path: str | Path, fps: float | None = None):
        self.video_path = Path(video_path)
        self.target_fps = fps
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        self._metadata: dict | None = None

    def get_metadata(self) -> dict:
        """Return video metadata via ffprobe.

        Returns:
            dict with keys: fps, duration_sec, width, height, num_frames,
            audio_sample_rate, has_audio, codec
        """
        if self._metadata is not None:
            return self._metadata

        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_format", "-show_streams", str(self.video_path),
                ],
                capture_output=True, text=True, timeout=10,
            )
            probe = json.loads(result.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            logger.warning(f"ffprobe unavailable or failed ({e}), using cv2 fallback")
            return self._metadata_fallback()

        video_stream = None
        audio_stream = None
        for s in probe.get("streams", []):
            if s["codec_type"] == "video" and video_stream is None:
                video_stream = s
            elif s["codec_type"] == "audio" and audio_stream is None:
                audio_stream = s

        if video_stream is None:
            raise ValueError(f"No video stream in {self.video_path}")

        # Parse FPS from r_frame_rate (e.g. "30/1")
        fps_str = video_stream.get("r_frame_rate", "30/1")
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 30.0
        except (ValueError, ZeroDivisionError):
            fps = 30.0

        duration = float(probe.get("format", {}).get("duration", 0))
        if duration == 0:
            duration = float(video_stream.get("duration", 0))

        self._metadata = {
            "fps": fps,
            "duration_sec": duration,
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "num_frames": int(video_stream.get("nb_frames", 0)),
            "has_audio": audio_stream is not None,
            "audio_sample_rate": (
                int(audio_stream["sample_rate"]) if audio_stream else 0
            ),
            "codec": video_stream.get("codec_name", "unknown"),
        }
        return self._metadata

    def _metadata_fallback(self) -> dict:
        """Fallback metadata extraction using cv2."""
        try:
            import cv2

            cap = cv2.VideoCapture(str(self.video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            duration = frames / fps if fps > 0 else 0
            cap.release()
            self._metadata = {
                "fps": fps,
                "duration_sec": duration,
                "width": width,
                "height": height,
                "num_frames": frames,
                "has_audio": False,
                "audio_sample_rate": 0,
                "codec": "unknown",
            }
        except ImportError:
            self._metadata = {
                "fps": 30.0, "duration_sec": 0, "width": 0, "height": 0,
                "num_frames": 0, "has_audio": False, "audio_sample_rate": 0,
                "codec": "unknown",
            }
        return self._metadata

    def extract_audio(self, output_path: str | Path | None = None, sample_rate: int = 16000) -> Path:
        """Demux and resample audio to 16kHz mono wav.

        Args:
            output_path: Where to save the wav. None = temp file.
            sample_rate: Target sample rate (default 16000 for Hubert).

        Returns:
            Path to the extracted wav file.
        """
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".wav"))
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(self.video_path),
                    "-vn", "-ar", str(sample_rate), "-ac", "1",
                    "-sample_fmt", "s16", str(output_path),
                ],
                capture_output=True, text=True, timeout=120,
            )
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Audio extracted: {output_path}")
                return output_path
        except FileNotFoundError:
            logger.warning("ffmpeg not found, trying cv2 audio extraction")
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timed out during audio extraction")

        # Fallback: try to load audio with soundfile from the container directly
        return self._extract_audio_fallback(output_path, sample_rate)

    def _extract_audio_fallback(self, output_path: Path, sample_rate: int) -> Path:
        """Attempt audio extraction without ffmpeg."""
        try:
            import soundfile as sf

            # Some video containers allow direct audio read via soundfile
            audio, sr = sf.read(str(self.video_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != sample_rate:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                except ImportError:
                    logger.warning("librosa not installed, skipping resample")
            sf.write(str(output_path), audio.astype(np.float32), sample_rate)
            return output_path
        except Exception as e:
            logger.warning(f"Audio extraction fallback failed: {e}")
            # Write silence
            duration = self.get_metadata().get("duration_sec", 1.0)
            silence = np.zeros(int(duration * sample_rate), dtype=np.float32)
            try:
                import soundfile as sf
                sf.write(str(output_path), silence, sample_rate)
            except ImportError:
                np.save(str(output_path).replace(".wav", ".npy"), silence)
            return output_path

    def extract_frames(self, output_dir: str | Path, fps: float | None = None) -> list[Path]:
        """Extract frames as PNG files at target fps using ffmpeg.

        Args:
            output_dir: Directory to save frames.
            fps: Target FPS. None = use source fps or constructor target_fps.

        Returns:
            List of paths to extracted frame PNGs.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target_fps = fps or self.target_fps

        fps_filter = f"fps={target_fps}" if target_fps else "copy"

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(self.video_path),
                    "-vf", fps_filter,
                    str(output_dir / "frame_%06d.png"),
                ],
                capture_output=True, text=True, timeout=300,
            )
            frames = sorted(output_dir.glob("frame_*.png"))
            if frames:
                logger.info(f"Extracted {len(frames)} frames to {output_dir}")
                return frames
        except FileNotFoundError:
            logger.warning("ffmpeg not found, falling back to cv2")
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timed out during frame extraction")

        return self._extract_frames_cv2(output_dir, target_fps)

    def _extract_frames_cv2(self, output_dir: Path, fps: float | None) -> list[Path]:
        """Fallback frame extraction using cv2."""
        import cv2

        cap = cv2.VideoCapture(str(self.video_path))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip = max(1, int(source_fps / (fps or source_fps)))

        frames = []
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % skip == 0:
                path = output_dir / f"frame_{idx:06d}.png"
                cv2.imwrite(str(path), frame)
                frames.append(path)
            idx += 1
        cap.release()
        logger.info(f"Extracted {len(frames)} frames (cv2 fallback)")
        return frames

    def iter_frames(self, fps: float | None = None) -> np.ndarray:
        """Yield frames as numpy arrays without writing to disk.

        Uses ffmpeg pipe output for streaming extraction.

        Args:
            fps: Target FPS for frame sampling.

        Yields:
            np.ndarray of shape (H, W, 3), dtype uint8, BGR format.
        """
        target_fps = fps or self.target_fps
        meta = self.get_metadata()
        width = meta.get("width", 640) or 640
        height = meta.get("height", 480) or 480

        fps_arg = ["-vf", f"fps={target_fps}"] if target_fps else []

        try:
            proc = subprocess.Popen(
                [
                    "ffmpeg", "-i", str(self.video_path),
                    *fps_arg,
                    "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-v", "quiet", "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            frame_size = width * height * 3
            while True:
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                yield frame

            proc.wait()
        except FileNotFoundError:
            yield from self._iter_frames_cv2(target_fps)

    def _iter_frames_cv2(self, fps: float | None) -> np.ndarray:
        """Fallback frame iteration using cv2."""
        import cv2

        cap = cv2.VideoCapture(str(self.video_path))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip = max(1, int(source_fps / (fps or source_fps)))

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % skip == 0:
                yield frame
            idx += 1
        cap.release()
