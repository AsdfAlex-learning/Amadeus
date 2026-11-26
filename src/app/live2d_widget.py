"""OpenGL Live2D widget: parameter queue, rate-matched drain, and rendering.

The widget decouples three rates that are all different:

* the model emits ~50 parameter frames per second, in one burst per inference
  step (every 0.5-3 s on the CPU target);
* the drain timer consumes at the model's nominal rate;
* `paintGL` runs at whatever rate Qt gives us.

An earlier revision drained one frame inside `paintGL` and reset to
`_default_params` on underflow. That underflows *structurally* — the consumer
took ~62.5 frames/s while the producer emitted ~50 — so the character snapped to
a neutral pose between chunks. Draining in `paintGL` also stops entirely when the
widget is hidden or occluded.

This implementation drains on its own timer, applies the **last frame popped**
(consuming intermediate frames without rendering them is the point — a 50 Hz
stream cannot be rendered frame-by-frame through a 60 Hz paint loop), holds the
last applied pose through a brief underflow, and only decays to the neutral pose
after sustained silence.

See `docs/adr/0004-speak-state-and-render-synchronisation.md` §S2.1.
"""

import time
from collections import deque

from loguru import logger
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from src.live2d.renderer import Live2DRenderer

#: Index -> Live2D Cubism parameter name. The model emits generic indices.
_PARAM_LUT = {
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


class Live2DWidget(QOpenGLWidget):
    #: Frames the model produces per second of audio.
    DEFAULT_MODEL_FPS = 50.0
    #: Seconds of frames to retain. Older frames are dropped, and counted.
    BUFFER_SECONDS = 3.0
    #: Underflow longer than this decays the pose back to neutral. Without it,
    #: holding the last frame forever would freeze the character on its final
    #: speaking pose (mouth possibly open) for the whole idle period.
    DEFAULT_IDLE_TIMEOUT_MS = 400.0

    def __init__(self, config: dict):
        fmt = QSurfaceFormat()
        fmt.setAlphaBufferSize(8)
        fmt.setSamples(4)
        QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__()
        self.config = config
        self.renderer: Live2DRenderer | None = None

        live2d_cfg = config["live2d"]
        self.model_fps = float(live2d_cfg.get("model_fps", self.DEFAULT_MODEL_FPS))
        self.idle_timeout_ms = float(
            live2d_cfg.get("idle_timeout_ms", self.DEFAULT_IDLE_TIMEOUT_MS)
        )
        self._capacity = max(1, int(self.model_fps * self.BUFFER_SECONDS))

        self._param_queue: deque[dict[str, float]] = deque(maxlen=self._capacity)
        self._default_params = dict(live2d_cfg["default_parameters"])
        self._applied_params: dict[str, float] = dict(self._default_params)
        self._dropped_frames = 0

        # Drain clock state.
        self._credit = 0.0
        self._last_drain_at: float | None = None
        self._last_frame_at: float | None = None

        self._initialized = False
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self.update)

        # The drain runs on its own timer, NOT in paintGL: paintGL stops when the
        # widget is hidden or occluded, a QTimer does not.
        self._drain_timer = QTimer(self)
        self._drain_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._drain_timer.timeout.connect(self._drain)

    # ── Qt GL lifecycle ──────────────────────────────────────────────────────

    def initializeGL(self):
        self.renderer = Live2DRenderer(self.config)
        success = self.renderer.initialize()
        if not success:
            logger.warning("Live2D renderer initialized without a model")
        self._initialized = True

        interval_ms = int(self.config["live2d"]["update_interval"] * 1000)
        self._update_timer.start(interval_ms)
        drain_ms = max(1, int(round(1000.0 / self.model_fps)))
        self._drain_timer.start(drain_ms)
        logger.info("Live2D OpenGL widget initialized")

    def paintGL(self):
        """Render the most recently applied pose. Never blocks, never drains."""
        if self.renderer is None:
            return
        self.renderer.set_parameters(self._applied_params)
        self.renderer.render(self.width(), self.height())

    def resizeGL(self, width: int, height: int):
        if self.renderer is not None:
            self.renderer.resize(width, height)

    # ── producer ─────────────────────────────────────────────────────────────

    def push_params(self, params: dict[str, float]):
        """Queue one frame of model output. Called from the motion worker thread."""
        mapped = {_PARAM_LUT.get(key, key): value for key, value in params.items()}
        if len(self._param_queue) >= self._capacity:
            # deque(maxlen) silently discards; count it so the loss is visible.
            self._dropped_frames += 1
        self._param_queue.append(mapped)

    # ── consumer ─────────────────────────────────────────────────────────────

    def _drain(self, now: float | None = None):
        """Consume up to the number of frames wall-clock has advanced.

        `now` is injectable so tests can drive the clock deterministically.
        """
        if now is None:
            now = time.perf_counter()

        if self._last_drain_at is None:
            self._last_drain_at = now
            self._last_frame_at = now
            return
        elapsed = max(0.0, now - self._last_drain_at)
        self._last_drain_at = now
        self._credit = min(self._credit + elapsed * self.model_fps, float(self._capacity))

        applied = False
        while self._credit >= 1.0 and self._param_queue:
            # Applying the LAST frame popped is the intended behaviour: the
            # intermediate frames cannot be rendered through a slower paint loop,
            # and applying queue[-1] instead would be a pose teleport.
            self._applied_params = self._param_queue.popleft()
            self._credit -= 1.0
            applied = True

        if applied:
            self._last_frame_at = now
            return

        # Underflow: hold the last applied pose briefly, then decay to neutral so
        # an idle character is not frozen on its final speaking frame.
        if self._last_frame_at is not None:
            if (now - self._last_frame_at) * 1000.0 >= self.idle_timeout_ms:
                self._applied_params = dict(self._default_params)

    def cleanup(self):
        self._update_timer.stop()
        self._drain_timer.stop()
        if self.renderer is not None:
            self.renderer.cleanup()
            self.renderer = None

    # ── introspection (used by tests and diagnostics) ─────────────────────────

    @property
    def applied_params(self) -> dict[str, float]:
        """The pose currently being rendered."""
        return self._applied_params

    @property
    def dropped_frames(self) -> int:
        """Frames discarded because the buffer was full."""
        return self._dropped_frames

    @property
    def queued_frames(self) -> int:
        return len(self._param_queue)

    @property
    def capacity(self) -> int:
        return self._capacity
