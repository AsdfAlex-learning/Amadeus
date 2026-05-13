# Changelog

All notable changes to Amadeus will be documented in this file.

## [0.1.0] - 2026-05-12

### Added

- **Phase 1**: Project skeleton with PySide6 window and Live2D rendering via live2d-py
- **Phase 2**: Audio pipeline (PyAudio capture, VAD, Whisper.cpp ASR)
- **Phase 3**: Dialogue model (Qwen2.5 3B inference, persona management, context window)
- **Phase 4**: Mini-LPM motion model (Hubert + Transformer + CNN, 109M/15M params)
- **Phase 5**: TTS engine (CosyVoice2, ChatTTS, Fish-Speech, pyttsx3 fallback)
- **Phase 6**: Camera perception (MediaPipe face detection, gaze, expression)
- Config file (`config/default.yaml`) for all module parameters
- Locked environment files (`environment.yml`, `requirements-lock.txt`, `.python-version`)

### Technical

- Python 3.11.5 with 39 pinned dependencies
- 26 source files, 1,706 lines of Python
- All ASR/TTS/Motion modules support fallback when optional deps unavailable

[0.1.0]: https://github.com/AsdfAlex-learning/Amadeus/releases/tag/v0.1.0
