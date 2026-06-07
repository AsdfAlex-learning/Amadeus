# Changelog

All notable changes to Amadeus will be documented in this file.

## [Unreleased] - 2026-06-07

### Critical (Training-Blocking) Fixes

After a full review of the training and inference pipeline
(`docs/TRAINING_PIPELINE_REVIEW.md`, 11 architecture diagrams in
`docs/ARCHITECTURE_DIAGRAMS.md`), the following issues were identified
and fixed in the `fix/training-pipeline-issues` branch (10 atomic
commits, each independently revertible):

- **C1: x-prediction diffusion** — `src/motion/training/train.py`.
  The output head ends in `nn.Sigmoid()` (range [0, 1]) but the loss
  was MSE against standard normal noise — mathematically unrepresentable
  (≈68% of `N(0, 1)` falls outside [0, 1]). The model now predicts the
  clean sample `x_0` directly. Inference switched to the matching
  x-prediction DDIM step formula.
- **H1: x-prediction DDIM inference** — `src/motion/inference.py`.
  Replaced the custom non-standard DDIM step with the canonical
  x-prediction formulation
  `x_{t-1} = √ᾱ_{t-1}·pred_x0 + √(1-ᾱ_{t-1}−σ²)·pred_eps + σ·noise`.
- **C2: 25→50 fps motion alignment** — `src/motion/training/dataset.py`.
  The preprocessing pipeline extracts at 25 fps but the model expects
  50 Hz motion. Previously the dataset zero-padded shorter segments, so
  every training sample was 50% zeros. Motion (and head_angles) are
  now linearly resampled to `AUDIO_FEATURES_PER_SEC` (50) using the
  `fps` field stored in each .npz.

### High Severity Fixes

- **H2: LoRA inference path** — `src/motion/inference.py`. Trained
  character LoRA adapters were silently ignored at inference. Added a
  `lora` block to the motion config, automatic adapter load from
  `models/lora/<character_id>/lora_adapter.pt`, and runtime character
  swap via `set_character_id()` with `remove_lora` / `apply_lora` /
  `load_lora` / `merge_lora`.

### Medium Severity Improvements

- **M1: Visual modality dropout** — `src/motion/model.py`. Training
  data has no real camera frames; the VisualEncoder therefore outputs
  garbage that pollutes cross-attention. The visual pathway is now
  randomly zeroed out 50% of the time during training so the model
  learns to produce motion from audio alone and to use vision when it
  is actually present at inference.
- **M2: `weight_decay` plumbed to AdamW** — `src/motion/training/train.py`.
  The YAML config specified `weight_decay: 0.01` but the optimizer was
  created with the AdamW default (0.0). The value is now passed
  through `train()` and the new `--weight_decay` CLI argument.
- **M3: Dynamic T from audio encoder** — `src/motion/inference.py`.
  The 4-step DDIM loop previously hardcoded `T = 50`. `T` is now
  derived from the Hubert encoder's output length, so inference
  remains correctly aligned for variable-length audio.
- **M4: Per-step warmup + cosine** — `src/motion/training/train.py`.
  The config specified `warmup_steps: 100` but it was ignored. The
  scheduler is now `SequentialLR(LinearLR + CosineAnnealingLR)` over
  total optimization steps, stepped after every optimizer update.

### Low Severity Improvements

- **L2: EMA of trainable parameters** — new file
  `src/motion/training/ema.py`; integrated into `train.py`. Self-
  contained, no third-party dependency. Enabled by `--ema_decay`
  (0 = disabled, 0.999 typical). EMA weights are used for validation
  and for the final checkpoint.
- **L3: Early stopping** — `src/motion/training/train.py`. Stops
  training when validation loss fails to improve for
  `--early_stopping_patience` epochs (0 = disabled).
- **L4: Complete checkpoint snapshots** — `src/motion/training/train.py`.
  Checkpoints now store model, optimizer, scheduler, AMP scaler, EMA
  shadow, epoch counter, and training config. The legacy raw-`state_dict`
  format is still accepted on resume.
- **L5: Legacy dataset path fix** — `src/motion/training/dataset.py`.
  The `.wav + .npy` loader indexed the motion array with `chunk_samples`
  (16 000) frames, which only made sense if motion were recorded
  sample-by-sample alongside audio. Resample to 50 Hz first, then chunk
  with `chunk_motion_frames`.

### Documentation

- New: `docs/TRAINING_PIPELINE_REVIEW.md` — full review report covering
  11 issues, severity table, fix plan.
- New: `docs/ARCHITECTURE_DIAGRAMS.md` plus `docs/diagrams/*.png` —
  11 Mermaid diagrams of system data flow, model architecture, training,
  inference, LoRA, preprocessing, performance, FPS alignment, state
  machine, and issue impact chain.
- New: `docs/adr/0002-x-prediction-and-50hz-alignment.md` — ADR
  documenting the C1 and C2 decisions.

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
