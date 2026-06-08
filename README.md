# Amadeus

**LPM-inspired real-time interactive AI companion with Live2D avatar.**

Amadeus is a desktop application that combines a customizable Live2D character with real-time multimodal AI: voice conversation, facial expression recognition, and motion synthesis driven by a Mini-LPM (Large Performance Model). Inspired by MiHoYo's LPM 1.0 paper and the concept of Amadeus from Steins;Gate.

![Python](https://img.shields.io/badge/Python-3.11.5-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey)

## Architecture

```
 Microphone ──→ Whisper.cpp ASR ──→ Qwen2.5 3B ──→ TTS ──→ Speaker
     │                                    │
     └──→ FullDuplexDiT (124M) ──→ Live2D Params (45)
              │                               │
     Camera Perception ────→ Live2D Renderer (PySide6 + live2d-py)
              │
     Preprocessing Pipeline ──→ Training Data (.npz)
              │
     LoRA Adapters ──→ Character-specific Performance
```

## Features

- **Real-time Conversation**: Voice-driven dialogue with a local LLM (Qwen2.5 3B)
- **Live2D Avatar**: Character rendering with live2d-py + PySide6/OpenGL
- **FullDuplexDiT Motion Model**: Multimodal diffusion transformer (Hubert + DiT + CNN, 124M/24M params)
  - Three states: Listen (user audio), Speak (TTS audio), Silence (camera context)
  - **x-prediction diffusion** (model outputs `x_0` directly, naturally compatible with the `[0, 1]` Live2D parameter range)
  - 4-step x-prediction DDIM inference at 50Hz parameter output
  - 50% visual modality dropout during training so the model learns to ignore the absent camera stream
- **Character LoRA**: Low-rank fine-tuning (~MB per character) for character-specific motion style
  - **Hot-swappable at inference** — `set_character_id()` removes the previous adapter and applies the new one in place
- **Persona Performance Parameters**: Gesture scale, react speed, expressiveness, and more — configurable per persona
- **Video → Training Data Pipeline**: MediaPipe FaceLandmarker + ARKit → Live2D parameter mapping
  - Linear resampling to 50 Hz at load time, regardless of the original extraction fps
- **Camera Perception**: FaceMesh face detection, gaze estimation, expression analysis
- **Pluggable Characters**: Framework-first design — swap Live2D models and personas
- **Full Local**: All models run locally on consumer hardware (M2 / RTX 4060)

## Quick Start

### Prerequisites

- Python 3.11.5
- macOS (Apple Silicon) or Linux
- Conda (recommended) or venv

### Installation

```bash
# Clone the repository
git clone https://github.com/AsdfAlex-learning/Amadeus.git
cd amadeus

# Create environment (Conda)
conda env create -f environment.yml
conda activate amadeus

# Or use venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

### Running

```bash
# Start Ollama for dialogue model
ollama pull qwen2.5:3b

# Launch Amadeus
python -m src.main
```

### Training the Motion Model

```bash
# 1. Prepare data via preprocessing pipeline (any fps — dataset will resample to 50 Hz)
python -m src.motion.preprocess.pipeline --input video.mp4 --output data/preprocessed/

# 2. Train base model (x-prediction diffusion) — convenience script
python scripts/train_base.py \
    --data_dir data/preprocessed \
    --output_dir models/motion/base_model \
    --num_epochs 200 \
    --warmup_steps 200 \
    --ema_decay 0.999 \
    --early_stopping_patience 50

# 3. Or use the low-level CLI directly:
python -m src.motion.training.train \
    --data_dir data/preprocessed \
    --output_dir models/motion \
    --num_params 45 \
    --epochs 100 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --ema_decay 0.999 \
    --early_stopping_patience 20

# 4. LoRA fine-tuning for a specific character
python -m src.motion.training.train \
    --data_dir data/character/ \
    --output_dir models/lora/kurisu \
    --use_lora --lora_rank 8 --lora_alpha 16 \
    --epochs 50

# 5. Resume training from a full snapshot (restores optimizer + scheduler + EMA + epoch)
python -m src.motion.training.train \
    --data_dir data/preprocessed \
    --output_dir models/motion \
    --resume models/motion/full_duplex_dit_epoch_0050.pt
```

**Training pipeline guarantees** (after the v0.2 fixes documented in
`docs/TRAINING_PIPELINE_REVIEW.md`):

- **x-prediction** — the model output's Sigmoid is preserved and the
  loss target is the clean motion signal (in [0, 1]).
- **fps-correct alignment** — every chunk contains 50 motion frames,
  no zero-padding.
- **EMA weights** (when `--ema_decay > 0`) drive validation and the
  final checkpoint.
- **Full-snapshot checkpoints** — model, optimizer, scheduler, AMP
  scaler, EMA shadow, epoch, and config are all persisted; resume is
  lossless.
- **Character LoRA is hot-swappable** at inference via
  `DiffusionMotionInference.set_character_id()`.

### Verified Dataset: HDTF

The [HDTF](https://github.com/MRzzm/HDTF) (High-definition Talking Face)
dataset has been tested end-to-end with the Amadeus training pipeline:

| Step | Detail |
|------|--------|
| **Download** | `clips.zip` (4.76 GB) from HuggingFace `Guangtian/HDTF` |
| **Format** | 16,914 clips, 390×390, 25 fps, 81 frames each (~3.24 s) |
| **Preprocessing** | MediaPipe FaceLandmarker → 52 ARKit blendshapes → 45 Live2D params |
| **Result** | 300 clips processed in 104 s, 0 failures, 13 MB `.npz` output |

**3-epoch smoke test** (300 clips, RTX 4060 Ti 8 GB):

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1     | 0.0320    | 0.3172   |
| 2     | 0.0094    | 0.1492   |
| 3     | 0.0081    | 0.0946   |

Loss decreases on both splits; the model learns motion dynamics from
the preprocessed data. Full 200-epoch training takes ~2.3 hours on an
RTX 4060 Ti. Results are visualized via:

```bash
# Loss curve (auto-generated by train_base.py)
python -m src.motion.training.visualize loss \
    --log models/motion/base_model/train.log \
    --out models/motion/base_model/loss_curve.png

# Motion parameter comparison (ground truth vs predicted)
python -m src.motion.training.visualize motion \
    --gt_npz data/preprocessed/sample.npz \
    --pred_npy models/motion/base_model/sample_pred.npy \
    --out models/motion/base_model/comparison.png

# Animated GIF of one motion sequence
python -m src.motion.training.visualize gif \
    --input data/preprocessed/sample.npz \
    --out models/motion/base_model/sample.gif
```

## Project Structure

```
Amadeus/
├── src/
│   ├── main.py                  # Application entry point
│   ├── config.py                # YAML configuration loader + typed accessors
│   ├── app/
│   │   ├── window.py            # PySide6 main window (261 lines)
│   │   └── live2d_widget.py     # OpenGL Live2D widget (79 lines)
│   ├── live2d/
│   │   └── renderer.py          # live2d-py wrapper (115 lines)
│   ├── audio/
│   │   ├── capture.py           # PyAudio microphone capture
│   │   ├── asr.py               # Whisper.cpp speech recognition
│   │   └── pipeline.py          # VAD + ASR pipeline orchestration
│   ├── dialogue/
│   │   ├── model.py             # LLM inference (local/API)
│   │   ├── persona.py           # Character persona management
│   │   └── context.py           # Conversation context window
│   ├── motion/
│   │   ├── model.py             # FullDuplexDiT architecture (Hubert+DiT+CNN, 432 lines)
│   │   ├── inference.py         # Real-time inference pipeline (4-step DDIM)
│   │   ├── performance.py       # PerformanceEngine (persona param post-processing)
│   │   ├── preprocess/
│   │   │   ├── face_landmarker.py   # MediaPipe FaceLandmarker + ARKit blendshapes
│   │   │   ├── arkit_to_live2d.py   # YAML mapping 52 ARKit → 45 Live2D
│   │   │   ├── video_reader.py      # FFmpeg video/audio extraction
│   │   │   ├── body_skeleton.py     # YOLOv8-pose stub (deferred)
│   │   │   ├── pipeline.py          # End-to-end preprocessing orchestrator
│   │   │   └── mappings/default.yaml  # 45-parameter mapping config
│   │   └── training/
│   │       ├── dataset.py       # MotionDataset (multimodal dict, 50Hz alignment, fps resample)
│   │       ├── train.py         # Training script (x-prediction, LoRA, EMA, val split, full checkpoint, early stop)
│   │       ├── lora.py          # LoRA module (495 lines, full lifecycle API)
│   │       ├── ema.py           # EMA of trainable parameters (no third-party deps)
│   │       └── visualize.py     # Loss curves, motion comparison, animated GIFs
│   ├── tts/
│   │   └── engine.py            # TTS engine (multi-backend)
│   └── perception/
│       └── camera.py            # Camera + MediaPipe FaceMesh perception
├── config/
│   └── default.yaml             # Default configuration (app/live2d/audio/asr/dialogue/tts/motion/perception/training/lora/preprocess/performance)
├── docs/
│   ├── ARCHITECTURE.md          # Detailed architecture documentation
│   ├── ARCHITECTURE_DIAGRAMS.md # 11 Mermaid diagrams + PNGs
│   ├── TRAINING_PIPELINE_REVIEW.md  # Full training & inference review report
│   └── adr/
│       ├── 0001-base-model-lora-architecture.md
│       └── 0002-x-prediction-and-50hz-alignment.md
├── scripts/
│   ├── download_models.py       # Download encoder weights (Hubert+MobileNet+BERT)
│   ├── train_base.py            # Convenience wrapper for base model training
│   ├── run_overnight.bat        # Batch script for overnight runs (Windows)
│   └── setup_windows.ps1        # Windows 11 environment setup
├── environment.yml              # Conda locked environment
├── requirements-lock.txt        # Pip locked requirements
├── requirements.txt             # Pip flexible requirements
└── pyproject.toml               # Project metadata
```

## Configuration

All module parameters are in `config/default.yaml`:

- **live2d**: Model path, default parameters, update interval
- **audio**: Sample rate, channels, chunk size
- **asr**: Whisper model size (`tiny`/`base`/`small`/`medium`/`large`), language
- **dialogue**: Model type, persona system prompt, quantization, context tokens
- **tts**: Engine selection, voice, speed
- **motion**: Mini-LPM architecture, output params count, chunk size
- **perception**: Camera device, detection targets
- **logging**: Level, format, rotation

## Research Background

This project is inspired by MiHoYo's **[LPM 1.0: Video-based Character Performance Model](https://arxiv.org/abs/2604.07823)** (arXiv:2604.07823), which addresses the "performance trilemma" — jointly achieving expressiveness, real-time inference, and long-horizon identity stability. LPM 1.0 is a 17B Diffusion Transformer distilled into a causal streaming generator for real-time conversational character performance.

Our Mini-LPM adapts this concept to operate in **Live2D parameter space** instead of pixel space, outputting 30-50 character motion parameters (facial expressions, mouth shapes, body pose) rather than video frames. This makes real-time inference feasible on consumer hardware.

## Model Details

### Mini-LPM (Motion Model)

| Property | Value |
|---|---|
| Architecture | Hubert encoder + 4-layer Interlaced DiT (Listen/Speak) + CNN decoder |
| Total Params | 124M (94M frozen Hubert + 24M trainable DiT + CNN) |
| Input | 5-stream multimodal: user audio, TTS audio, camera frames, text prompt, character ID |
| Output | 50 frames × 45 Live2D parameters (per second of audio) |
| Diffusion formulation | **x-prediction** (model predicts `x_0` ∈ [0, 1] via Sigmoid) |
| Loss | MSE between `pred_x0` and ground-truth motion |
| LoRA Params | ~500K per character (rank=8), ~MB per character |
| Inference | 4-step x-prediction DDIM, real-time at 50 Hz |
| Optional training features | EMA weights (`--ema_decay`), full-snapshot checkpoints, early stopping, warmup + cosine LR, weight decay |
| Reference | NVIDIA Audio2Face-3D v2.3 (40M, real-time) |

### Dialogue Model

- Base: Qwen2.5-3B-Instruct (INT8 quantized)
- Fallback: Ollama API (`qwen2.5:3b`)
- Context: Sliding window with automatic summarization on overflow
- Persona: YAML-based character profiles with tone and trait control

## Dependencies

### Core

| Package | Version | Purpose |
|---|---|---|
| PySide6 | 6.11.0 | Qt6 GUI framework |
| live2d-py | 0.6.1.1 | Live2D Cubism 3.0+ rendering |
| torch | 2.5.1 | Mini-LPM / LLM inference |
| transformers | 4.32.1 | Hubert / Qwen2.5 |
| whisper-cpp-python | 0.2.0 | Local speech recognition |
| mediapipe | 0.10.18 | Face detection & analysis |
| opencv-python | 4.10.0.84 | Camera capture & processing |

See `requirements-lock.txt` for the complete pinned dependency tree (39 packages).

## Development

```bash
# Install dev dependencies
pip install pytest black ruff

# Run linting
ruff check src/

# Run type checking
basedpyright src/

# Run tests
pytest tests/
```

## Roadmap

- [x] Phase 1: Project skeleton + Live2D rendering
- [x] Phase 2: Audio pipeline (capture + ASR + VAD)
- [x] Phase 3: Dialogue model (LLM + persona + context)
- [x] Phase 4: FullDuplexDiT motion model (architecture + inference + training)
- [x] Phase 5: TTS engine (multi-backend)
- [x] Phase 6: Camera perception (face + gaze + expression)
- [x] Phase 7: Data preprocessing (video → facial landmarks → Live2D parameters)
- [x] **Phase 7.5: Training pipeline audit & fixes** (branch `fix/training-pipeline-issues`)
  - 10 atomic commits: x-prediction, fps alignment, DDIM rewrite, LoRA
    inference path, visual dropout, weight decay, warmup, EMA, full
    checkpoints, legacy path fix
  - See `docs/TRAINING_PIPELINE_REVIEW.md` and `docs/adr/0002-*.md`
- [ ] Phase 8: First end-to-end training run on real character data
- [ ] Phase 9: Memory system integration (LLMChatFlow)
- [ ] Phase 10: Multi-character support & live swapping

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [LPM 1.0](https://arxiv.org/abs/2604.07823) by MiHoYo — foundational research
- [NVIDIA Audio2Face-3D](https://github.com/NVIDIA/Audio2Face-3D) — architecture reference
- [live2d-py](https://github.com/Arkueid/live2d-py) — Python Live2D bindings
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) — on-device ASR
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) — dialogue model base
