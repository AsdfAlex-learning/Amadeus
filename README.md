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
     └──→ Mini-LPM (15M) ──→ Live2D Params
              │                         │
     Camera Perception ────→ Live2D Renderer (PySide6 + live2d-py)
```

## Features

- **Real-time Conversation**: Voice-driven dialogue with a distilled local LLM (Qwen2.5 3B)
- **Live2D Avatar**: Character rendering with live2d-py + PySide6/OpenGL
- **Mini-LPM Motion Model**: Audio-to-Live2D-parameters generation (Hubert + Transformer, 15M trainable params)
- **Camera Perception**: Face detection, gaze estimation, expression recognition via MediaPipe
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
# Prepare data: .wav audio files + .npy motion parameter files in a directory
# Then train:
python -m src.motion.training.train \
    --data_dir data/mead \
    --output_dir models/motion \
    --num_params 45 \
    --epochs 100
```

## Project Structure

```
Amadeus/
├── src/
│   ├── main.py                  # Application entry point
│   ├── config.py                # YAML configuration loader
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
│   │   ├── model.py             # Mini-LPM architecture (Hubert+Transformer)
│   │   ├── inference.py         # Real-time inference pipeline
│   │   └── training/
│   │       ├── dataset.py       # Training data loader
│   │       └── train.py         # Training script
│   ├── tts/
│   │   └── engine.py            # TTS engine (multi-backend)
│   └── perception/
│       └── camera.py            # Camera + MediaPipe perception
├── config/
│   └── default.yaml             # Default configuration
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
| Architecture | Hubert encoder + 4-layer Transformer + CNN decoder |
| Total Params | 109M (94M frozen Hubert + 15M trainable) |
| Input | 1-sec audio @ 16kHz (16,000 samples) |
| Output | 49 frames × 45 Live2D parameters |
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
- [x] Phase 4: Mini-LPM motion model (architecture + inference + training)
- [x] Phase 5: TTS engine (multi-backend)
- [x] Phase 6: Camera perception (face + gaze + expression)
- [ ] Phase 7: Data preprocessing (video → facial landmarks → Live2D parameters)
- [ ] Phase 8: Model training & distillation pipeline
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
