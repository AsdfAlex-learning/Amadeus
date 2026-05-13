from pathlib import Path

import numpy as np


class WhisperASR:
    SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, config: dict):
        asr_cfg = config["asr"]
        self.model_size = asr_cfg["model_size"]
        self.language = asr_cfg.get("language", "zh")
        self.model_dir = Path(asr_cfg.get("model_dir", "models/whisper"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._loaded = False

    def load_model(self) -> bool:
        try:
            from whisper_cpp_python import Whisper

            model_path = str(self.model_dir / f"ggml-{self.model_size}.bin")
            if not Path(model_path).exists():
                self._download_model()
            self._model = Whisper(model_path=model_path)
            self._loaded = True
            return True
        except ImportError:
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if not self._loaded:
            return ""
        result = self._model.transcribe(
            audio.astype(np.float32),
            language=self.language if self.language != "auto" else None,
            sample_rate=sample_rate,
        )
        return result.get("text", "").strip()

    def _download_model(self):
        model_url = (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
            f"ggml-{self.model_size}.bin"
        )
        model_path = self.model_dir / f"ggml-{self.model_size}.bin"
        import urllib.request

        urllib.request.urlretrieve(model_url, str(model_path))

    @property
    def is_loaded(self) -> bool:
        return self._loaded
