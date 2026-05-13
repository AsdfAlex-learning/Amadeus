# Contributing to Amadeus

## Development Environment

```bash
conda env create -f environment.yml
conda activate amadeus
pip install pytest black ruff basedpyright
```

## Code Style

- **Formatter**: Black (line-length=100)
- **Linter**: Ruff (target Python 3.11)
- **Type Checker**: basedpyright
- **Commit Style**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`)

## Pull Request Process

1. Fork the repository and create a feature branch
2. Ensure code passes linting and type checking
3. Add tests for new functionality
4. Update documentation if needed
5. Submit a PR with a clear description

## Project Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Module Overview

| Module | Responsibility |
|---|---|
| `src/app/` | PySide6 window and Live2D widget |
| `src/live2d/` | live2d-py wrapper with parameter control |
| `src/audio/` | Microphone capture, VAD, ASR pipeline |
| `src/dialogue/` | LLM inference, persona, context management |
| `src/motion/` | Mini-LPM model, inference, training |
| `src/tts/` | Text-to-speech engine (multi-backend) |
| `src/perception/` | Camera capture and computer vision |

## Adding a New Character

1. Place Live2D model files (`.model3.json`, `.moc3`, textures) in `assets/live2d/`
2. Create a persona YAML file:

```yaml
name: "Character Name"
system_prompt: "Character persona description..."
tone: "warm and friendly"
traits:
  - empathetic
  - curious
  - playful
```

3. Update `config/default.yaml`:
   - `live2d.model_name` → your model file name
   - `dialogue.persona_file` → path to your persona YAML

## Adding a New TTS Backend

Implement the `_load_<engine>()` method in `src/tts/engine.py` following the existing pattern:

```python
def _load_myengine(self) -> bool:
    try:
        # import and initialize
        def _synthesize(text: str) -> np.ndarray:
            # ... return float32 numpy array
            pass
        self._model = _synthesize
        return True
    except ImportError:
        return False
```
