# Amadeus Architecture

## System Overview

Amadeus is a modular pipeline that processes multimodal input (voice, camera) through a series of independent components, drives a Live2D character with synthesized motion, and outputs voice responses through TTS.

```
                         ┌─────────────────┐
                         │   config.yaml    │
                         └────────┬────────┘
                                  │
    ┌──────────┐   ┌──────────┐   │   ┌──────────┐   ┌──────────┐
    │  Camera  │   │   Mic    │   │   │ Dialogue │   │   TTS    │
    │Perception│   │ Capture  │   │   │  Model   │   │  Engine  │
    └────┬─────┘   └────┬─────┘   │   └────┬─────┘   └────┬─────┘
         │              │         │        │              │
         ▼              ▼         │        ▼              │
    ┌─────────┐   ┌──────────┐    │   ┌──────────┐       │
    │MediaPipe│   │    VAD    │    │   │ Persona  │       │
    │  Face   │   │(energy)   │    │   │ Context  │       │
    └────┬─────┘   └────┬─────┘    │   └──────────┘       │
         │              │         │                       │
         │         ┌────▼────┐    │                       │
         │         │Whisper  │    │                       │
         │         │  .cpp   │    │                       │
         │         └────┬────┘    │                       │
         │              │         │                       │
         │              ▼         ▼                       │
         │         ┌─────────────────┐                   │
         │         │   AudioPipeline │                   │
         │         └────────┬────────┘                   │
         │                  │ text                       │
         │                  ▼                            │
         │         ┌────────────────┐                    │
         │         │ AmadeusWindow  │                    │
         │         │  (PySide6/Qt6) │◄───────────────────┘
         │         └───────┬────────┘   stream response
         │                 │
         │    ┌────────────┼────────────┐
         │    │            │            │
         ▼    ▼            ▼            ▼
    ┌──────────┐  ┌────────────┐  ┌──────────┐
    │  Camera  │  │  Motion    │  │ Live2D   │
    │  →Params │  │ Inference  │  │ Renderer │
    └──────────┘  └─────┬──────┘  └──────────┘
                        │
                   ┌────▼─────┐
                   │ Mini-LPM │
                   │  Model   │
                   └──────────┘
```

## Component Details

### 1. Application Layer (`src/app/`)

**AmadeusWindow** (`window.py`): PySide6 QMainWindow orchestrating the full application lifecycle. Manages all sub-components via `_PipelineBridge` (thread-safe Qt signal bridge for audio→UI communication).

**Live2DWidget** (`live2d_widget.py`): QOpenGLWidget with 60fps timer. Maintains a parameter queue consumed each frame. Maps generic parameter indices to Live2D Cubism parameter names via a lookup table.

### 2. Live2D Rendering (`src/live2d/`)

**Live2DRenderer** (`renderer.py`): Wraps `live2d-py`. Handles model loading (Cubism 3.0+), parameter setting, OpenGL rendering, and cleanup. Falls back to headless mode when no model is configured.

### 3. Audio Pipeline (`src/audio/`)

**MicrophoneCapture** (`capture.py`): PyAudio streaming with callback-based audio delivery. 16kHz mono, configurable chunk size. Exposes device enumeration.

**WhisperASR** (`asr.py`): Whisper.cpp Python bindings wrapper. Auto-downloads GGML model files from HuggingFace. Supports `tiny` through `large` model sizes.

**AudioPipeline** (`pipeline.py`): State machine orchestrating VAD → buffering → ASR:

```
IDLE → LISTENING → (speech detected) → buffering → (silence 1.2s) → ASR → text callback
```

VAD is energy-based (RMS threshold). Includes pre-speech buffer (300ms) to avoid clipping start of utterances.

### 4. Dialogue (`src/dialogue/`)

**DialogueModel** (`model.py`): Dual-mode LLM inference:
- **Local**: transformers `AutoModelForCausalLM` with INT8/INT4 quantization, MPS (Apple Silicon) support
- **API fallback**: Ollama-compatible HTTP streaming endpoint

**Persona** (`persona.py`): YAML-based character profile with system prompt, tone, and trait vectors. Format prompt as conversation prefix.

**ConversationContext** (`context.py`): Sliding window message history with token estimation and automatic truncation (FIFO, preserving system message).

### 5. Motion Model (`src/motion/`)

**MiniLPM** (`model.py`): Core architecture:

```
Audio (16kHz, 1sec)
    │
    ▼
┌─────────────────┐
│ Hubert Encoder  │  ← facebook/hubert-base-ls960 (frozen, 94M)
│ (768-dim @50Hz) │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Linear  │  768 → 512
    └────┬────┘
         │
    ┌────▼────────┐
    │ Positional  │
    │  Encoding   │
    └────┬────────┘
         │
    ┌────▼────────────┐
    │ Transformer      │  4 layers, 8 heads, 512-dim, GELU
    │ Encoder          │
    └────┬────────────┘
         │
    ┌────▼────────┐
    │ CNN Decoder  │  Conv1d ×3 → Sigmoid
    │ (45 params)  │
    └─────────────┘
         │
         ▼
   Live2D Parameters (49 frames × 45 values)
```

**MotionInference** (`inference.py`): Streaming inference with overlap-add. Maintains an audio buffer, processes chunks of `chunk_size` seconds, and emits parameter dictionaries frame-by-frame to callbacks.

**Training** (`training/`): PyTorch dataset and training loop. Expects paired `.wav` (16kHz mono) and `.npy` (frames × params) files. Uses MSE loss + AdamW + cosine annealing.

### 6. TTS (`src/tts/`)

**TTSEngine** (`engine.py`): Multi-backend TTS with graceful degradation:

1. CosyVoice2 → 2. ChatTTS → 3. Fish-Speech → 4. pyttsx3 (system TTS fallback)

Each backend is loaded lazily, returns `np.ndarray` of float32 audio.

### 7. Perception (`src/perception/`)

**CameraPerception** (`camera.py`): OpenCV capture + MediaPipe processing in a background thread. Produces per-frame perception results:
- `face_detected`: boolean
- `gaze`: angle + direction (left/center/right)
- `expression`: mouth openness, smile detection, brow raise

These results are mapped to Live2D parameters (gaze → ParamAngleY, smile → ParamMouthForm, etc.) in the window.

## Data Flow

### Conversation Loop

```
1. Mic captures audio → PyAudio callback
2. VAD detects speech → buffers audio until silence
3. Whisper.cpp transcribes → emits text
4. Text added to ConversationContext → sent to DialogueModel
5. LLM generates response → streamed to UI display
6. Full response → TTS synthesizes audio → playback
```

### Motion Loop

```
1. Mic raw audio → MotionInference.process_audio()
2. Audio buffered → chunk assembled
3. Mini-LPM forward pass → parameter sequence
4. Parameters emitted frame-by-frame → Live2DWidget.push_params()
5. Each paintGL frame consumes one parameter set → renderer.set_parameters()
```

### Perception Loop

```
1. Camera captures frame → OpenCV
2. MediaPipe FaceMesh processes → landmarks
3. Gaze/expression estimation → perception result
4. Result mapped to Live2D parameters → Live2DWidget.push_params()
```

## Threading Model

| Thread | Responsibility |
|---|---|
| Main (Qt) | GUI event loop, OpenGL rendering |
| Audio callback | Microphone data, VAD |
| ASR (sync) | Whisper.cpp transcription |
| Generation | LLM inference (daemon) |
| Camera | OpenCV capture + MediaPipe (daemon) |

Qt `Signal`/`Slot` bridge (`_PipelineBridge`) handles thread-safe communication from audio/generation threads to the UI thread.

## Configuration

All runtime parameters are in `config/default.yaml`. The config is loaded once at startup and passed to all components. Each component reads its own section, allowing independent configuration.
