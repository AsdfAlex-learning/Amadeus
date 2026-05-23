# Amadeus — Domain Glossary

## Core Entities

**Amadeus**: The desktop application system. Contains all subsystems (rendering, audio, dialogue, perception). Not a character.

**Character (角色)**: The visual Live2D avatar displayed on screen. Pluggable — different `.model3.json` files produce different characters.

**Persona (人格)**: The personality configuration for dialogue generation. Defined via YAML files (`system_prompt`, `tone`, `traits`). Default persona: **Kurisu (红莉栖)**.

**Performance (表演)**: The sequence of Live2D parameters inferred by the FullDuplexDiT model from multimodal input. Model-driven, not preset. Contrast with **Animation (动画)**: preset or state-machine-driven behavior.

## Subsystems

| Term | Code Location | Description |
|---|---|---|
| **Audio Pipeline** | `src/audio/` | Microphone capture, VAD, Whisper.cpp ASR |
| **Dialogue Model** | `src/dialogue/` | LLM inference, persona management, conversation context |
| **Performance Engine** | `src/motion/` | FullDuplexDiT — multimodal diffusion model that generates Performance |
| **TTS Engine** | `src/tts/` | Synthesizes speech from dialogue text |
| **Perception** | `src/perception/` | Camera capture — raw frames fed to Performance Engine |
| **Renderer** | `src/live2d/` | live2d-py wrapper — consumes Performance parameters and renders the Character |

## Performance Model Architecture

**Base Model (基础模型)**: FullDuplexDiT trained on broad multimodal datasets (MEAD, BIWI, VOCASET). Learns universal human motion patterns — how people move when speaking, listening, and emoting. Not character-specific.

**Character Weight (角色权重)**: A fine-tuned weight delta derived from character-specific video data. Extracted via YOLOv8 skeleton tracking on source material (e.g., Steins;Gate anime scenes of Kurisu). Loaded on top of the Base Model to specialize Performance for a specific Character.

**Workflow**:
1. Base Model trained on broad data → universal motion understanding
2. Character videos → YOLOv8 skeleton extraction → skeleton-to-Live2D-parameter mapping
3. Character-specific LoRA fine-tuning → weight file (~MB per character)
4. At runtime: load Base Model + load Character LoRA = Character-specific Performance

**Persona YAML**: Secondary mechanism for performance control. Text-based style hints (`react_style`, `gesture_scale`) injected as text prompts into the Performance Engine. Lower priority than Character Weights for motion quality.

**Base Model Training Data**: Mix of real human datasets (MEAD, BIWI, VOCASET — foundation for natural motion) and anime character data (higher expressiveness for certain motion patterns).

**Training Data Format**: Ground truth = Live2D parameter values (45 floats per frame). Input = audio waveform + camera frames. Skeleton extraction (YOLOv8/MediaPipe) is an intermediate step — skeleton coordinates are mapped to Live2D parameter space before training.

## Key Distinctions

- **Performance ≠ Animation**: Performance is model-inferred. Animation is hardcoded.
- **Amadeus ≠ Character**: The system hosts Characters. Characters have Personas.
- **Persona ≠ Character**: Persona is the personality config. Character is the visual avatar.
