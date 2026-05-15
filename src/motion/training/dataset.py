from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    SUPPORTED_DATASETS = ["mead", "biwi", "vocaset"]

    def __init__(
        self,
        data_dir: str | Path,
        dataset_type: str = "mead",
        sample_rate: int = 16000,
        chunk_duration: float = 1.0,
        audio_suffix: str = ".wav",
        motion_suffix: str = ".npy",
    ):
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type.lower()
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.audio_suffix = audio_suffix
        self.motion_suffix = motion_suffix
        self._samples: list[tuple[Path, Path]] = []
        self._scan()

    def _scan(self):
        audio_paths = sorted(self.data_dir.glob(f"*{self.audio_suffix}"))
        for ap in audio_paths:
            mp = ap.with_suffix(self.motion_suffix)
            if not mp.exists():
                mp = ap.parent / f"{ap.stem}_motion{self.motion_suffix}"
            if mp.exists():
                self._samples.append((ap, mp))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio_path, motion_path = self._samples[idx]
        audio = self._load_audio(audio_path)
        motion = np.load(str(motion_path)).astype(np.float32)
        min_len = min(len(audio), len(motion))
        audio = audio[:min_len]
        motion = motion[:min_len]
        start = np.random.randint(0, max(1, min_len - self.chunk_samples))
        audio_chunk = audio[start : start + self.chunk_samples]
        motion_chunk = motion[start : start + self.chunk_samples]
        if len(audio_chunk) < self.chunk_samples:
            audio_chunk = np.pad(audio_chunk, (0, self.chunk_samples - len(audio_chunk)))
            motion_chunk = np.pad(
                motion_chunk, ((0, self.chunk_samples - len(motion_chunk)), (0, 0))
            )
        return (
            torch.from_numpy(audio_chunk).float(),
            torch.from_numpy(motion_chunk).float(),
        )

    def _load_audio(self, path: Path) -> np.ndarray:
        try:
            import soundfile as sf

            audio, sr = sf.read(str(path))
            if sr != self.sample_rate:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32)
        except ImportError:
            return np.zeros(self.chunk_samples, dtype=np.float32)

    @staticmethod
    def from_video_landmarks(
        video_dir: Path,
        output_dir: Path,
        num_params: int = 45,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting motion from videos in {video_dir}")
        print(f"Output: {output_dir}")
        print("This requires face detection + landmark extraction pipeline.")
        print("Use MediaPipe Face Mesh or OpenFace for landmark extraction.")
        print("Then map landmarks to Live2D parameter space.")
        print(f"Target: {num_params} Live2D parameters per frame.")
