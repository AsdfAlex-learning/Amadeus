# Amadeus Architecture

## System Overview

Amadeus is a modular pipeline that processes multimodal input (voice, camera) through a series of independent components, drives a Live2D character with synthesized motion, and outputs voice responses through TTS.

```mermaid
flowchart TB
    CFG[("config.yaml")]

    subgraph INPUTS["Inputs"]
        CAM["Camera<br/>Perception"]
        MIC["Mic<br/>Capture"]
        DLG["Dialogue<br/>Model"]
        TTS["TTS<br/>Engine"]
    end

    subgraph PROCESS["Processing"]
        MP["MediaPipe<br/>Face"]
        VAD["VAD<br/>(energy)"]
        WSP["Whisper<br/>.cpp"]
    end

    subgraph DIALOGUE["Dialogue Subsystem"]
        PERS["Persona +<br/>Context"]
    end

    AP["AudioPipeline"]
    WIN["AmadeusWindow<br/>(PySide6/Qt6)"]

    subgraph OUTPUTS["Outputs"]
        CPARAMS["Camera<br/>→ Params"]
        MINF["Motion<br/>Inference"]
        REND["Live2D<br/>Renderer"]
    end

    LPM[("Mini-LPM<br/>Model")]

    CFG -.-> WIN

    CAM --> MP
    MIC --> VAD --> WSP --> AP
    AP -->|"text"| WIN
    WSP -.->|"perception events"| WIN

    DLG --> PERS --> WIN
    TTS -.->|"stream response"| WIN
    TTS -.->|"TTS audio"| MINF

    MP --> CPARAMS
    MP -->|"gaze/expression<br/>face_detected"| MINF
    WIN --> MINF
    WIN --> REND
    CPARAMS --> WIN
    MINF --> LPM
    LPM --> MINF
    MINF -->|"params"| WIN

    style WIN fill:#1a1a2e,stroke:#4a90d9,color:#fff
    style LPM fill:#d94a90,stroke:#8b2c5e,color:#fff
    style AP fill:#4a90d9,stroke:#2b5d8e,color:#fff
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

```mermaid
stateDiagram-v2
    direction LR
    [*] --> IDLE
    IDLE --> LISTENING : mic opened, VAD armed
    LISTENING --> BUFFERING : speech detected<br/>(RMS > threshold)
    BUFFERING --> PROCESSING : silence 1.2s
    BUFFERING --> BUFFERING : continue capturing<br/>(max 15s)
    PROCESSING --> EMIT : ASR returns text
    EMIT --> LISTENING : ready for next utterance
    EMIT --> IDLE : shutdown / pause

    note right of BUFFERING
      Pre-speech buffer
      300 ms (avoid
      clipping start
      of utterances)
    end note
```

VAD is energy-based (RMS threshold). Includes pre-speech buffer (300ms) to avoid clipping start of utterances.

### 4. Dialogue (`src/dialogue/`)

**DialogueModel** (`model.py`): Dual-mode LLM inference:
- **Local**: transformers `AutoModelForCausalLM` with INT8/INT4 quantization, MPS (Apple Silicon) support
- **API fallback**: Ollama-compatible HTTP streaming endpoint

**Persona** (`persona.py`): YAML-based character profile with system prompt, tone, and trait vectors. Format prompt as conversation prefix.

**ConversationContext** (`context.py`): Sliding window message history with token estimation and automatic truncation (FIFO, preserving system message).

### 5. Motion Model (`src/motion/`)

**FullDuplexDiT** (`model.py`): Core architecture — multimodal Diffusion Transformer with interlaced Listen/Speak layers. Five input streams: user audio, TTS audio, camera frames, text prompt, character identity. Output: 50 frames × 45 Live2D parameters.

```mermaid
flowchart TB
    subgraph INPUTS["5-Stream Inputs"]
        UA["User Audio<br/>(16kHz, 1s)"]
        TA["TTS Audio<br/>(16kHz, 1s)"]
        VF["Camera Frames<br/>(5×224×224)"]
        TP["Text Prompt"]
        ID["Identity ID"]
    end

    subgraph ENC["Frozen Encoders"]
        HE["Hubert Encoder<br/>(94M frozen)<br/>768-dim @50Hz"]
        VE["MobileNetV3-Small<br/>(2.5M frozen)<br/>512-dim @50Hz"]
    end

    UA --> HE
    TA --> HE
    VF --> VE

    HE -->|"768→320"| AP1["Audio proj 1<br/>(listen)"]
    HE -->|"768→320"| AP2["Audio proj 2<br/>(speak)"]
    VE -->|"512→320"| VPROJ["Visual proj"]

    subgraph DIT["DiT Interlaced Blocks ×4 (dim=320, 8 heads)"]
        direction TB
        B0["Block 0: Listen<br/>self-attn + cross-attn<br/>(user audio + visual)"]
        B1["Block 1: Speak<br/>self-attn + cross-attn<br/>(TTS audio + text)"]
        B2["Block 2: Listen"]
        B3["Block 3: Speak"]
        B0 --> B1 --> B2 --> B3
    end

    AP1 -->|"listen_feat"| B0
    AP2 -->|"speak_feat"| B1
    VPROJ -->|"visual_feat"| B0
    VPROJ -->|"visual_feat"| B2
    TP -->|"text_feat"| B1
    TP -->|"text_feat"| B3
    ID -->|"identity_emb"| TIME["Time Embedding<br/>+ identity"]
    TIME --> B0

    B3 --> DEC["CNN Decoder<br/>Conv1d×3 + Sigmoid"]
    DEC -->|"50 frames × 45"| OUT[("Live2D<br/>Parameters<br/>∈ [0, 1]")]

    style HE fill:#ffe066,stroke:#e67700,color:#000
    style DEC fill:#d0bfff,stroke:#7950f2,color:#000
    style OUT fill:#d94a90,stroke:#8b2c5e,color:#fff
    style AP1 fill:#d3f9d8,stroke:#2b8a3e,color:#000
    style AP2 fill:#d0ebff,stroke:#1971c2,color:#000
```

**PerformanceEngine** (`performance.py`): Persona-based post-processing on model output. Six configurable parameters applied as multiplicative adjustments in vectorized numpy. EMA temporal smoothing for `react_speed`. Supports 2D/3D arrays + silence/speak/listen modes.

**Preprocessing Pipeline** (`preprocess/`): Video → training data workflow:
- `face_landmarker.py`: MediaPipe FaceLandmarker → 52 ARKit blendshapes + head pose
- `arkit_to_live2d.py`: YAML-configurable weight mapping → 45 Live2D params
- `video_reader.py`: FFmpeg audio extraction + frame extraction
- `pipeline.py`: End-to-end orchestrator with CLI entry point
- `body_skeleton.py`: YOLOv8-pose stub (deferred from MVP)

**Training** (`training/`):
- `dataset.py`: `MotionDataset` — multimodal dict format, 50Hz alignment, fps resampling (linear interp to 50Hz regardless of source fps), .npz + legacy .npy/.wav support
- `train.py`: **x-prediction** training loop. Features: val split, **full-snapshot checkpointing** (model + optimizer + scheduler + AMP scaler + EMA + epoch), resume (lossless from full snapshots, legacy raw-state_dict supported), `--dataset_type`, and full LoRA integration (`--use_lora`, `--lora_rank`, `--lora_alpha`). New flags: `--weight_decay`, `--warmup_steps`, `--ema_decay`, `--early_stopping_patience`.
- `lora.py`: LoRA training module — `LoRALinear` / `LoRAConv1d` wrappers, `apply/remove/merge/save/load_lora` lifecycle API, monkey-patching approach (no model.py modification)
- `ema.py`: Self-contained EMA of trainable parameters. No third-party dependency. Used for validation loss and final checkpoint save when `--ema_decay > 0`.

**Inference** (`inference.py`): Streaming x-prediction diffusion inference with overlap-add. Maintains audio/visual/tts buffers, processes chunks of `chunk_size` seconds via **4-step x-prediction DDIM** (η=0 deterministic), emits parameter dictionaries frame-by-frame to callbacks. **T is dynamically derived from the Hubert encoder output length** (was previously hardcoded to 50). Loads and **hot-swaps character LoRA adapters** via `set_character_id()` — looks up `models/lora/<id>/lora_adapter.pt` and merges in place.

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

```mermaid
flowchart LR
    A1["Mic captures audio<br/>PyAudio callback"] --> A2
    A2["VAD detects speech<br/>buffers until silence 1.2s"] --> A3
    A3["Whisper.cpp transcribes<br/>emits text"] --> A4
    A4["Text added to<br/>ConversationContext"] --> A5
    A5["Sent to DialogueModel<br/>(Qwen2.5-3B)"] --> A6
    A6["LLM streams response<br/>to UI display"] --> A7
    A7["Full response sent to<br/>TTS for synthesis"] --> A8
    A8["TTS audio plays<br/>via PyAudio"]

    style A1 fill:#4a90d9,color:#fff
    style A5 fill:#d94a90,color:#fff
    style A8 fill:#51cf66,color:#fff
```

### Motion Loop

```mermaid
flowchart LR
    M1["Mic raw audio<br/>MotionInference.process_audio()"] --> M2
    M2["Audio buffered<br/>chunk assembled (1s)"] --> M3
    M3["Mini-LPM forward pass<br/>4-step x-DDIM"] --> M4
    M4["Parameter sequence emitted<br/>frame-by-frame (50Hz)"] --> M5
    M5["Live2DWidget.push_params()<br/>consumer queue"] --> M6
    M6["Each paintGL frame<br/>renderer.set_parameters()"]

    style M3 fill:#d94a90,color:#fff
    style M6 fill:#51cf66,color:#fff
```

### Perception Loop

```mermaid
flowchart LR
    P1["Camera captures frame<br/>OpenCV @ 30fps"] --> P2
    P2["MediaPipe FaceMesh<br/>processes landmarks"] --> P3
    P3["Gaze / expression<br/>estimation"] --> P4
    P4["Perception result<br/>(text_prompt + visual)"] --> P5
    P5["Mapped to Live2D params<br/>Live2DWidget.push_params()"]

    style P2 fill:#ffd43b,color:#000
    style P5 fill:#51cf66,color:#fff
```

## Threading Model

```mermaid
flowchart TB
    subgraph MAIN["Main Thread (Qt)"]
        GUI["GUI event loop<br/>OpenGL rendering"]
    end

    subgraph WORKERS["Worker Threads"]
        AUDIO["Audio Callback<br/>(PyAudio thread)"]
        ASR["ASR (sync)<br/>Whisper.cpp"]
        GEN["Generation (daemon)<br/>LLM inference"]
        CAM["Camera (daemon)<br/>OpenCV + MediaPipe"]
    end

    BR["_PipelineBridge<br/>(QObject signals/slots)"]

    AUDIO -- "raw frames<br/>(Signal)" --> BR
    ASR -- "transcript<br/>(Signal)" --> BR
    GEN -- "streamed tokens<br/>(Signal)" --> BR
    CAM -- "perception result<br/>(Signal)" --> BR
    BR -- "queued Slot<br/>(UI thread)" --> GUI

    style MAIN fill:#1a1a2e,stroke:#4a90d9,color:#fff
    style BR fill:#d94a90,stroke:#8b2c5e,color:#fff
```

Qt `Signal`/`Slot` bridge (`_PipelineBridge`) handles thread-safe communication from audio/generation threads to the UI thread.

## Configuration

All runtime parameters are in `config/default.yaml`. The config is loaded once at startup and passed to all components. Each component reads its own section, allowing independent configuration.
