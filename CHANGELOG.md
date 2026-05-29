# Changelog

All notable changes to Amadeus will be documented in this file.

## [Unreleased] - 2026-05-29

### Added

- **Data Preprocessing Pipeline** (`src/motion/preprocess/`): Video → training data workflow
  - `face_landmarker.py`: MediaPipe FaceLandmarker Tasks API with 52 ARKit blendshapes + head pose
  - `arkit_to_live2d.py`: YAML-configurable weight mapping from 52 ARKit → 45 Live2D parameters
  - `mappings/default.yaml`: Complete 45-parameter mapping configuration
  - `video_reader.py`: FFmpeg audio extraction (16kHz WAV) + frame extraction
  - `pipeline.py`: End-to-end orchestrator with CLI entry point
  - `body_skeleton.py`: Stub for YOLOv8-pose body keypoint extraction (deferred from MVP)
- **LoRA Training Module** (`src/motion/training/lora.py`): Character-specific fine-tuning
  - `LoRALinear` / `LoRAConv1d`: Drop-in LoRA wrappers with Kaiming init
  - `apply_lora()`: Monkey-patch target modules in frozen base model
  - `remove_lora()` / `merge_lora()` / `save_lora()` / `load_lora()`: Full lifecycle API
  - `get_lora_target_modules()` / `get_lora_param_count()`: Inspection utilities
  - Target modules default: `dit_blocks.*`, `output_head.*`, `audio_proj.*`, `cross_proj.*`
- **LoRA Integration into train.py**:
  - New params: `--use_lora`, `--lora_rank`, `--lora_alpha`, `--lora_dropout`, `--lora_target_modules`
  - LoRA mode: optimizer only trains LoRA A/B matrices; checkpoints saved via `save_lora()`
  - Auto-saves `lora_config.pt` alongside checkpoints for reproducibility
- **PerformanceEngine** (`src/motion/performance.py`): Persona-based post-processing
  - `PerformanceConfig` dataclass: 6 parameters (gesture_scale, react_speed, expressiveness, mouth_open_max, head_motion_range, idle_energy)
  - `PerformanceEngine`: Vectorized numpy ops, 2D/3D array support, silence/speak/listen modes
  - `react_speed`: EMA temporal smoothing with state tracking
  - Configurable YAML mapping or heuristic parameter grouping
- **Config Unification** (`config/default.yaml`):
  - New sections: `training`, `training.lora`, `preprocess`, `performance`
  - Typed accessors in `src/config.py`: `get_training_config()`, `get_lora_config()`, `get_preprocess_config()`, `get_performance_config()`
  - All performance params validated to [0, 1] range with clamping

### Changed

- **MotionDataset** rewritten: Multimodal dict format (7 keys), 50Hz alignment, .npz + legacy support
- **train.py** enhanced: Dict batch format, `--dataset_type` arg, `--val_split`, `--resume`, epoch checkpointing
- **DialogueModel device priority**: CUDA > MPS > CPU via `_detect_device()` method
- **CameraPerception**: FaceMesh + perception callback pipeline
- **AudioPipeline**: Added PROCESSING state between LISTENING and IDLE

### Fixed

- `demo.py`: `has_camera` NameError when camera unavailable
- `camera.py`: FaceMesh callback wiring for perception events
- `context.py`: `_summary` attribute injection for early conversation truncation
- `pipeline.py`: PROCESSING state transition during ASR
- `inference.py`: DDPM timestep T=49 corrected to T=50 (Hubert stride alignment)
- `train.py`: DDPM noise schedule using correct `sqrt(1 - alpha_cumprod)` factor

### Infrastructure

- `scripts/setup_windows.ps1`: Windows 11 environment setup script (conda env, package verification, device detection)
- opencode default model switched to `deepseek-v4-pro` for better quota availability

## [0.1.0] - 2026-05-12

### Added

- **Phase 1**: Project skeleton with PySide6 window and Live2D rendering via live2d-py
- **Phase 2**: Audio pipeline (PyAudio capture, VAD, Whisper.cpp ASR)
- **Phase 3**: Dialogue model (Qwen2.5 3B inference, persona management, context window)
- **Phase 4**: FullDuplexDiT motion model (Hubert + DiT + CNN, 124M/24M params)
- **Phase 5**: TTS engine (CosyVoice2, ChatTTS, Fish-Speech, pyttsx3 fallback)
- **Phase 6**: Camera perception (MediaPipe face detection, gaze, expression)
- Config file (`config/default.yaml`) for all module parameters
- Locked environment files (`environment.yml`, `requirements-lock.txt`, `.python-version`)

### Technical

- Python 3.11.5 with 39 pinned dependencies
- 26 source files, 1,706 lines of Python
- All ASR/TTS/Motion modules support fallback when optional deps unavailable

[0.1.0]: https://github.com/AsdfAlex-learning/Amadeus/releases/tag/v0.1.0
