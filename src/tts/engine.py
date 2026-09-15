from pathlib import Path

import numpy as np
from loguru import logger


class TTSEngine:
    def __init__(self, config: dict):
        tts_cfg = config["tts"]
        self.engine = tts_cfg["engine"]
        self.voice = tts_cfg.get("voice", "default")
        self.speed = float(tts_cfg.get("speed", 1.0))
        self.sample_rate = int(tts_cfg.get("sample_rate", 22050))
        self.model_path = Path(tts_cfg.get("model_path", "models/tts"))
        self._model = None
        self._loaded = False
        self._fallback = None

    def load(self) -> bool:
        loaded = False
        if self.engine == "cosyvoice2":
            loaded = self._load_cosyvoice2()
        elif self.engine == "chattts":
            loaded = self._load_chattts()
        elif self.engine == "fishspeech":
            loaded = self._load_fishspeech()
        if not loaded:
            loaded = self._load_pyttsx3()
        self._loaded = loaded
        return loaded

    def synthesize(self, text: str) -> np.ndarray:
        if not self._loaded:
            return self._silence(0.5)
        try:
            return self._model(text)
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return self._silence(1.0)

    def _load_cosyvoice2(self) -> bool:
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2

            self._model = CosyVoice2(str(self.model_path))
            logger.info("CosyVoice2 loaded")
            return True
        except ImportError:
            logger.debug("CosyVoice2 not installed")
            return False
        except Exception as e:
            logger.warning(f"CosyVoice2 load failed: {e}")
            return False

    def _load_chattts(self) -> bool:
        try:
            import ChatTTS

            chat = ChatTTS.Chat()
            chat.load(source="local" if self.model_path.exists() else "huggingface")

            def _synthesize(text: str) -> np.ndarray:
                wavs = chat.infer([text], use_decoder=True)
                return wavs[0].squeeze()

            self._model = _synthesize
            logger.info("ChatTTS loaded")
            return True
        except ImportError:
            logger.debug("ChatTTS not installed")
            return False

    def _load_fishspeech(self) -> bool:
        try:
            from fish_speech.inference_engine import TTSInferenceEngine

            engine = TTSInferenceEngine(checkpoint_dir=str(self.model_path))

            def _synthesize(text: str) -> np.ndarray:
                return engine.inference(text)

            self._model = _synthesize
            logger.info("Fish-Speech loaded")
            return True
        except ImportError:
            logger.debug("Fish-Speech not installed")
            return False

    def _load_pyttsx3(self) -> bool:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)
            engine.setProperty("rate", int(150 * self.speed))

            def _synthesize(text: str) -> np.ndarray:
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_path = f.name
                engine.save_to_file(text, tmp_path)
                engine.runAndWait()
                import soundfile as sf

                audio, _ = sf.read(tmp_path)
                Path(tmp_path).unlink()
                return audio.astype(np.float32)

            self._model = _synthesize
            logger.info("pyttsx3 loaded as TTS fallback")
            return True
        except ImportError:
            logger.debug("pyttsx3 not installed")
            return False

    def cleanup(self) -> None:
        """Release the backend.

        No caller did this before — `closeEvent` never touched the TTS engine — so a
        backend holding loaded models was simply abandoned at shutdown. Backends
        that need more than dropping a reference (a speech driver thread, a decoder
        process) should expose a `close()` from their loader and have it called here.
        """
        model = self._model
        self._model = None
        self._fallback = None
        self._loaded = False
        closer = getattr(model, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as e:
                logger.debug(f"TTS backend close failed: {e}")

    def _silence(self, duration: float) -> np.ndarray:
        """Silence at *this engine's* advertised rate.

        Hardcoding 16000 while advertising ``sample_rate`` (22050) made the
        fallback the wrong duration as soon as anything downstream trusted the
        declared rate. (C-5)
        """
        return np.zeros(int(duration * self.sample_rate), dtype=np.float32)
        """Silence at *this engine's* advertised rate.

        Hardcoding 16000 while advertising ``sample_rate`` (22050) made the
        fallback the wrong duration as soon as anything downstream trusted the
        declared rate. (C-5)
        """
        return np.zeros(int(duration * self.sample_rate), dtype=np.float32)

    @property
    def is_loaded(self) -> bool:
        return self._loaded
