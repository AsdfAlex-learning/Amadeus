from collections import deque

from PySide6.QtWidgets import QOpenGLWidget
from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
from loguru import logger

from src.live2d.renderer import Live2DRenderer


class Live2DWidget(QOpenGLWidget):
    def __init__(self, config: dict):
        fmt = QSurfaceFormat()
        fmt.setAlphaBufferSize(8)
        fmt.setSamples(4)
        QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__()
        self.config = config
        self.renderer: Live2DRenderer | None = None
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self.update)
        self._initialized = False
        self._param_queue: deque[dict[str, float]] = deque(maxlen=120)
        self._default_params = dict(config["live2d"]["default_parameters"])

    def initializeGL(self):
        self.renderer = Live2DRenderer(self.config)
        success = self.renderer.initialize()
        if not success:
            logger.warning("Live2D renderer initialized without a model")
        self._initialized = True
        interval_ms = int(self.config["live2d"]["update_interval"] * 1000)
        self._update_timer.start(interval_ms)
        logger.info("Live2D OpenGL widget initialized")

    def paintGL(self):
        if self.renderer is None:
            return
        params = self._param_queue.popleft() if self._param_queue else None
        if params is not None:
            self.renderer.set_parameters(params)
        else:
            self.renderer.set_parameters(self._default_params)
        self.renderer.render(self.width(), self.height())

    def resizeGL(self, width: int, height: int):
        if self.renderer is not None:
            self.renderer.resize(width, height)

    def push_params(self, params: dict[str, float]):
        mapped = {}
        lut = {
            "param_0": "ParamMouthOpenY",
            "param_1": "ParamEyeLOpen",
            "param_2": "ParamEyeROpen",
            "param_3": "ParamBrowLY",
            "param_4": "ParamBrowRY",
            "param_5": "ParamAngleX",
            "param_6": "ParamAngleY",
            "param_7": "ParamAngleZ",
            "param_8": "ParamBodyAngleX",
            "param_9": "ParamBodyAngleY",
            "param_10": "ParamBodyAngleZ",
            "param_11": "ParamMouthForm",
            "param_12": "ParamCheek",
            "param_13": "ParamEyeBallX",
            "param_14": "ParamEyeBallY",
        }
        for key, value in params.items():
            mapped_key = lut.get(key, key)
            mapped[mapped_key] = value
        self._param_queue.append(mapped)

    def cleanup(self):
        self._update_timer.stop()
        if self.renderer is not None:
            self.renderer.cleanup()
            self.renderer = None
