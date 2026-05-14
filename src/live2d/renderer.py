from pathlib import Path

from loguru import logger


class Live2DRenderer:
    def __init__(self, config: dict):
        self.config = config
        self._model_loaded = False
        self._params: dict[str, float] = {}
        self._param_range: dict[str, tuple[float, float]] = {}
        self._live2d = None

    def initialize(self) -> bool:
        model_path = self._resolve_model_path()
        if model_path is None:
            logger.info("No Live2D model configured — running in headless mode")
            self._init_default_params()
            return False
        return self._load_model(model_path)

    def _resolve_model_path(self) -> Path | None:
        model_name = self.config["live2d"].get("model_name")
        if not model_name:
            return None
        model_dir = Path(self.config["live2d"]["model_dir"])
        if not model_dir.is_absolute():
            model_dir = Path.cwd() / model_dir
        model_file = model_dir / model_name
        if not model_file.exists():
            logger.error(f"Live2D model file not found: {model_file}")
            return None
        return model_file

    def _init_default_params(self):
        defaults = self.config["live2d"]["default_parameters"]
        for name, value in defaults.items():
            self._params[name] = float(value)
            self._param_range[name] = (0.0, 1.0)

    def _load_model(self, model_path: Path) -> bool:
        try:
            import live2d.v3 as live2d

            self._live2d = live2d
            self._live2d.init()
            self._live2d.LoadModel(str(model_path))
            self._model_loaded = True
            self._init_default_params()
            logger.info(f"Live2D model loaded: {model_path}")
            return True
        except ImportError:
            logger.error(
                "live2d-py not installed. Run: pip install live2d-py"
            )
            self._init_default_params()
            return False
        except Exception as e:
            logger.error(f"Failed to load Live2D model: {e}")
            self._init_default_params()
            return False

    def set_parameter(self, name: str, value: float):
        clamped = max(0.0, min(1.0, float(value)))
        self._params[name] = clamped
        if self._model_loaded:
            try:
                self._live2d.SetParameterValue(name, clamped)
            except Exception:
                pass

    def set_parameters(self, values: dict[str, float]):
        for name, value in values.items():
            self.set_parameter(name, value)

    def get_parameter(self, name: str) -> float:
        return self._params.get(name, 0.0)

    def render(self, width: int, height: int):
        if not self._model_loaded:
            return
        try:
            self._live2d.Update()
            self._live2d.Draw(width, height)
        except Exception as e:
            logger.debug(f"Render error: {e}")

    def resize(self, width: int, height: int):
        if self._model_loaded:
            try:
                self._live2d.Resize(width, height)
            except Exception as e:
                logger.debug(f"Resize error: {e}")

    def cleanup(self):
        if self._model_loaded:
            try:
                self._live2d.Terminate()
            except Exception as e:
                logger.debug(f"Cleanup error: {e}")
        self._model_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def parameter_names(self) -> list[str]:
        return list(self._params.keys())
