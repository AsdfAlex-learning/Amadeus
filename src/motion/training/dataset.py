"""Training dataset for FullDuplexDiT motion model.

Supports two data formats:
1. Legacy: paired .wav + .npy files (audio-sample-aligned, flat directory)
2. Preprocessed: .npz files from the video->training-data pipeline

The .npz format provides multimodal inputs:
    - live2d_params: (T, 45) float32 — ground truth motion
    - blendshapes: (T, 52) float32 — raw ARKit (for reference)
    - head_angles: (T, 3) float32 — pitch/yaw/roll
    - bad_frames: (N,) int — indices where face detect failed
    - fps: float — extraction fps
    - identity_id: int — character index
    - source_video: str — original video path

Audio is loaded from a companion _audio.wav file extracted during preprocessing.
"""

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    """Multimodal motion dataset for FullDuplexDiT training.

    Supports both legacy (.wav + .npy) and preprocessed (.npz) data formats.
    """

    SUPPORTED_DATASETS = ["mead", "biwi", "vocaset", "preprocessed"]
    # Hubert-base: stride=320 at 16kHz -> 50 features/sec
    # For 1-sec audio: T_audio_features = 50
    AUDIO_FEATURES_PER_SEC = 50
    SAMPLE_RATE = 16000

    def __init__(
        self,
        data_dir: str | Path,
        dataset_type: str = "preprocessed",
        sample_rate: int = 16000,
        chunk_duration: float = 1.0,
        audio_suffix: str = ".wav",
        motion_suffix: str = ".npy",
        num_live2d_params: int = 45,
        num_visual_frames: int = 5,
        visual_size: int = 224,
    ):
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type.lower()
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.audio_suffix = audio_suffix
        self.motion_suffix = motion_suffix
        self.num_live2d_params = num_live2d_params
        self.num_visual_frames = num_visual_frames
        self.visual_size = visual_size
        # Motion frames per chunk: derived from audio feature rate
        self.chunk_motion_frames = int(chunk_duration * self.AUDIO_FEATURES_PER_SEC)

        self._samples: list[dict] = []
        self._scan()

    def _scan(self):
        if self.dataset_type == "preprocessed":
            self._scan_npz()
        else:
            self._scan_legacy()

    def _scan_npz(self):
        npz_files = sorted(self.data_dir.glob("*.npz"))
        for npz_path in npz_files:
            if "_meta" in npz_path.stem:
                continue
            audio_path = npz_path.with_suffix(".wav")
            if not audio_path.exists():
                audio_path = npz_path.parent / f"{npz_path.stem}_audio.wav"
            if not audio_path.exists():
                logger.warning(f"No audio for {npz_path.name}, skipping")
                continue
            self._samples.append({"type": "npz", "npz_path": npz_path, "audio_path": audio_path})
        logger.info(f"Found {len(self._samples)} preprocessed samples in {self.data_dir}")

    def _scan_legacy(self):
        audio_paths = sorted(self.data_dir.glob(f"*{self.audio_suffix}"))
        for ap in audio_paths:
            mp = ap.with_suffix(self.motion_suffix)
            if not mp.exists():
                mp = ap.parent / f"{ap.stem}_motion{self.motion_suffix}"
            if mp.exists():
                self._samples.append({"type": "legacy", "audio_path": ap, "motion_path": mp})
        logger.info(f"Found {len(self._samples)} legacy samples in {self.data_dir}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self._samples[idx]
        if sample["type"] == "npz":
            return self._get_npz_item(sample)
        return self._get_legacy_item(sample)

    def _get_npz_item(self, sample: dict) -> dict[str, torch.Tensor]:
        data = np.load(str(sample["npz_path"]), allow_pickle=False)
        live2d_params = data["live2d_params"]  # (T, 45)
        head_angles = data.get("head_angles", np.zeros((live2d_params.shape[0], 3), dtype=np.float32))
        identity_id = int(data.get("identity_id", 0))
        audio = self._load_audio(sample["audio_path"])

        T_motion = live2d_params.shape[0]
        T_audio_chunks = len(audio) // self.chunk_samples
        max_chunks = min(T_audio_chunks, T_motion // self.chunk_motion_frames)

        if max_chunks == 0:
            audio_chunk = np.zeros(self.chunk_samples, dtype=np.float32)
            motion_chunk = np.zeros((self.chunk_motion_frames, self.num_live2d_params), dtype=np.float32)
            head_chunk = np.zeros((self.chunk_motion_frames, 3), dtype=np.float32)
        else:
            start_chunk = np.random.randint(0, max(1, max_chunks))
            audio_start = start_chunk * self.chunk_samples
            motion_start = start_chunk * self.chunk_motion_frames
            audio_chunk = audio[audio_start:audio_start + self.chunk_samples]
            motion_chunk = live2d_params[motion_start:motion_start + self.chunk_motion_frames]
            head_chunk = head_angles[motion_start:motion_start + self.chunk_motion_frames]

        # Pad if needed
        if len(audio_chunk) < self.chunk_samples:
            audio_chunk = np.pad(audio_chunk, (0, self.chunk_samples - len(audio_chunk)))
        if len(motion_chunk) < self.chunk_motion_frames:
            pad_len = self.chunk_motion_frames - len(motion_chunk)
            motion_chunk = np.pad(motion_chunk, ((0, pad_len), (0, 0)))
            head_chunk = np.pad(head_chunk, ((0, pad_len), (0, 0)))

        visual_frames = np.zeros(
            (self.num_visual_frames, 3, self.visual_size, self.visual_size), dtype=np.float32
        )

        return {
            "user_audio": torch.from_numpy(audio_chunk).float(),
            "tts_audio": torch.zeros(self.chunk_samples, dtype=torch.float32),
            "visual_frames": torch.from_numpy(visual_frames).float(),
            "text_prompt": "",
            "identity_id": torch.tensor(identity_id, dtype=torch.long),
            "motion": torch.from_numpy(motion_chunk).float(),
            "head_angles": torch.from_numpy(head_chunk).float(),
        }

    def _get_legacy_item(self, sample: dict) -> dict[str, torch.Tensor]:
        audio = self._load_audio(sample["audio_path"])
        motion = np.load(str(sample["motion_path"])).astype(np.float32)
        min_len = min(len(audio), len(motion))
        audio = audio[:min_len]
        motion = motion[:min_len]
        start = np.random.randint(0, max(1, min_len - self.chunk_samples))
        audio_chunk = audio[start:start + self.chunk_samples]
        motion_chunk = motion[start:start + self.chunk_samples]
        if len(audio_chunk) < self.chunk_samples:
            audio_chunk = np.pad(audio_chunk, (0, self.chunk_samples - len(audio_chunk)))
            motion_chunk = np.pad(motion_chunk, ((0, self.chunk_samples - len(motion_chunk)), (0, 0)))

        return {
            "user_audio": torch.from_numpy(audio_chunk).float(),
            "tts_audio": torch.zeros(self.chunk_samples, dtype=torch.float32),
            "visual_frames": torch.zeros(
                self.num_visual_frames, 3, self.visual_size, self.visual_size, dtype=torch.float32
            ),
            "text_prompt": "",
            "identity_id": torch.tensor(0, dtype=torch.long),
            "motion": torch.from_numpy(motion_chunk).float(),
            "head_angles": torch.zeros(motion_chunk.shape[0], 3, dtype=torch.float32),
        }

    def _load_audio(self, path: Path) -> np.ndarray:
        try:
            import soundfile as sf
            audio, sr = sf.read(str(path))
            if sr != self.sample_rate:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                except ImportError:
                    logger.warning("librosa not installed, skipping resample")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32)
        except ImportError:
            return np.zeros(self.chunk_samples, dtype=np.float32)

    @staticmethod
    def from_video_landmarks(video_dir: Path, output_dir: Path, num_params: int = 45):
        """Convert videos to training data using the preprocessing pipeline.

        Use: python -m src.motion.preprocess.pipeline <video_dir> --output_dir <output_dir>
        """
        from src.motion.preprocess.pipeline import PreprocessPipeline
        pipeline = PreprocessPipeline()
        results = pipeline.process_directory(video_dir, output_dir)
        pipeline.cleanup()
        logger.info(f"Preprocessed {len(results)} videos to {output_dir}")
        return results
