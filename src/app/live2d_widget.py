from collections import deque

from loguru import logger
from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QOpenGLWidget

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
            "param_0": "ParamAngleX",
            "param_1": "ParamAngleY",
            "param_2": "ParamAngleZ",
            "param_3": "ParamBodyAngleX",
            "param_4": "ParamBodyAngleY",
            "param_5": "ParamBodyAngleZ",
            "param_6": "ParamEyeLOpen",
            "param_7": "ParamEyeROpen",
            "param_8": "ParamEyeBallX",
            "param_9": "ParamEyeBallY",
            "param_10": "ParamBrowLX",
            "param_11": "ParamBrowLY",
            "param_12": "ParamBrowRX",
            "param_13": "ParamBrowRY",
            "param_14": "ParamMouthOpenY",
            "param_15": "ParamMouthForm",
            "param_16": "ParamCheek",
            "param_17": "ParamBreath",
            "param_18": "ParamArmLX",
            "param_19": "ParamArmLY",
            "param_20": "ParamArmRX",
            "param_21": "ParamArmRY",
            "param_22": "ParamHairFront",
            "param_23": "ParamHairBack",
            "param_24": "ParamHairSideL",
            "param_25": "ParamHairSideR",
            "param_26": "ParamTear",
            "param_27": "ParamBlush",
            "param_28": "ParamNose",
            "param_29": "ParamLipUpper",
            "param_30": "ParamLipLower",
            "param_31": "ParamTongue",
            "param_32": "ParamEarL",
            "param_33": "ParamEarR",
            "param_34": "ParamTail",
            "param_35": "ParamWingL",
            "param_36": "ParamWingR",
            "param_37": "ParamItem1",
            "param_38": "ParamItem2",
            "param_39": "ParamItem3",
            "param_40": "ParamExtra1",
            "param_41": "ParamExtra2",
            "param_42": "ParamExtra3",
            "param_43": "ParamExtra4",
            "param_44": "ParamExtra5",
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
