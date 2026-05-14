from threading import Thread

from loguru import logger
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.audio.pipeline import AudioPipeline, PipelineState
from src.dialogue.context import ConversationContext
from src.dialogue.model import DialogueModel
from src.dialogue.persona import Persona
from src.motion.inference import DiffusionMotionInference
from src.perception.camera import CameraPerception
from src.tts.engine import TTSEngine


class _PipelineBridge(QObject):
    text_received = Signal(str)
    state_changed = Signal(int)
    response_chunk = Signal(str)
    response_done = Signal()


class AmadeusWindow(QMainWindow):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.audio_pipeline = AudioPipeline(config)
        self.dialogue_model = DialogueModel(config)
        self.motion_model = DiffusionMotionInference(config)
        self.tts_engine = TTSEngine(config)
        self.camera = CameraPerception(config)
        self.persona = Persona(config)
        self.context = ConversationContext(
            max_tokens=int(config["dialogue"].get("max_context_tokens", 4096))
        )
        self._bridge = _PipelineBridge()
        self._generating = False
        self._pyaudio = None
        self._setup_window()
        self._setup_ui()
        self._connect_pipeline()
        self._load_dialogue_model()
        self._load_motion_model()
        self._load_tts()
        self._connect_camera()
        logger.debug("Main window initialized")

    def _setup_window(self):
        window_cfg = self.config["app"]["window"]
        self.setWindowTitle(window_cfg["title"])
        self.resize(window_cfg["width"], window_cfg["height"])
        if window_cfg["fullscreen"]:
            self.showFullScreen()
        if window_cfg.get("transparent_background", False):
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setStyleSheet("background: transparent;")

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        from src.app.live2d_widget import Live2DWidget
        self.live2d_widget = Live2DWidget(self.config)
        layout.addWidget(self.live2d_widget, stretch=3)

        self.conversation_display = QTextEdit()
        self.conversation_display.setReadOnly(True)
        self.conversation_display.setMaximumHeight(150)
        self.conversation_display.setFont(QFont("PingFang SC", 12))
        self.conversation_display.setStyleSheet(
            "QTextEdit { color: white; background: rgba(0,0,0,150); "
            "border: none; border-radius: 8px; padding: 8px; }"
        )
        layout.addWidget(self.conversation_display)

        self.status_label = QLabel("● 就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "QLabel { color: #888; background: rgba(0,0,0,120); "
            "padding: 4px 16px; border-radius: 10px; font-size: 13px; }"
        )
        layout.addWidget(self.status_label)

        self.toggle_btn = QPushButton("🎤 开始对话")
        self.toggle_btn.setStyleSheet(
            "QPushButton { background: #4a90d9; color: white; border: none; "
            "padding: 8px 24px; border-radius: 6px; font-size: 14px; }"
            "QPushButton:hover { background: #357abd; }"
        )
        self.toggle_btn.clicked.connect(self._toggle_pipeline)
        layout.addWidget(self.toggle_btn)

    def _connect_pipeline(self):
        self.audio_pipeline.on_text(self._on_text)
        self.audio_pipeline.on_state_change(self._on_pipeline_state)
        self._bridge.text_received.connect(self._handle_text)
        self._bridge.state_changed.connect(self._handle_state)
        self._bridge.response_chunk.connect(self._append_response)
        self._bridge.response_done.connect(self._on_response_done)

    def _load_dialogue_model(self):
        success = self.dialogue_model.load()
        if not success:
            self.conversation_display.append(
                "<span style='color:#888'>💡 对话模型未加载。请安装 Ollama 并运行: "
                "ollama pull qwen2.5:3b</span>"
            )

    def _load_motion_model(self):
        success = self.motion_model.load_model()
        if success:
            self.motion_model.on_params(self.live2d_widget.push_params)
            self.audio_pipeline.on_raw_audio(self.motion_model.process_user_audio)
            logger.info("Motion model connected — unified full-duplex path active")

    def _load_tts(self):
        success = self.tts_engine.load()
        if not success:
            logger.info("TTS not loaded — voice output disabled")

    def _connect_camera(self):
        self.camera.on_frame(self.motion_model.process_visual_frame)
        logger.info("Camera connected — visual frames fed to motion model")

    def _on_text(self, text: str):
        self._bridge.text_received.emit(text)

    def _on_pipeline_state(self, state: PipelineState):
        self._bridge.state_changed.emit(state.value)

    def _handle_text(self, text: str):
        self.conversation_display.append(
            f"<span style='color:#4a90d9'><b>你:</b> {text}</span>"
        )
        self.context.add_user_message(text)
        self.motion_model.set_text_prompt(text)
        self._start_generation()

    def _start_generation(self):
        if self._generating:
            return
        self._generating = True
        self.status_label.setText("◉ 思考中...")
        self.conversation_display.append(
            f"<span style='color:#d94a90'><b>{self.persona.name}:</b> </span>"
        )
        Thread(target=self._run_generation, daemon=True).start()

    def _run_generation(self):
        messages = [{"role": "system", "content": self.persona.format_prompt([])}]
        messages.extend(self.context.get_messages())
        full_response = ""
        try:
            for chunk in self.dialogue_model.generate(messages):
                full_response += chunk
                self._bridge.response_chunk.emit(chunk)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            self._bridge.response_chunk.emit("（抱歉，出了点问题...）")
        finally:
            self.context.add_assistant_message(full_response)
            self._bridge.response_done.emit()
            if self.tts_engine.is_loaded and full_response.strip():
                self._speak_response(full_response)

    def _append_response(self, text: str):
        cursor = self.conversation_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        scrollbar = self.conversation_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_response_done(self):
        self._generating = False
        state_label = "● 聆听中..." if self.audio_pipeline.state != PipelineState.IDLE else "● 就绪"
        self.status_label.setText(state_label)

    def _speak_response(self, text: str):
        try:
            audio_data = self.tts_engine.synthesize(text)
            if audio_data is None or len(audio_data) == 0:
                return
            self.motion_model.process_tts_audio(audio_data)
            import pyaudio
            if self._pyaudio is None:
                self._pyaudio = pyaudio.PyAudio()
            stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.tts_engine.sample_rate,
                output=True,
            )
            stream.write(audio_data.tobytes())
            stream.stop_stream()
            stream.close()
        except Exception as e:
            logger.error(f"TTS playback error: {e}")

    def _handle_state(self, state_val: int):
        state = PipelineState(state_val)
        labels = {
            PipelineState.IDLE: "● 就绪",
            PipelineState.LISTENING: "● 聆听中...",
            PipelineState.PROCESSING: "◉ 识别中...",
        }
        if not self._generating:
            self.status_label.setText(labels.get(state, "● 就绪"))
        if state == PipelineState.IDLE:
            self.toggle_btn.setText("🎤 开始对话")
            self.toggle_btn.setStyleSheet(
                "QPushButton { background: #4a90d9; color: white; border: none; "
                "padding: 8px 24px; border-radius: 6px; font-size: 14px; }"
                "QPushButton:hover { background: #357abd; }"
            )
        else:
            self.toggle_btn.setText("⏹ 停止对话")
            self.toggle_btn.setStyleSheet(
                "QPushButton { background: #d94a4a; color: white; border: none; "
                "padding: 8px 24px; border-radius: 6px; font-size: 14px; }"
                "QPushButton:hover { background: #bd3535; }"
            )

    def _toggle_pipeline(self):
        if self.audio_pipeline.state == PipelineState.IDLE:
            self.audio_pipeline.start()
            self.camera.start()
        else:
            self.audio_pipeline.stop()
            self.camera.stop()

    def closeEvent(self, event):
        logger.info("Shutting down Amadeus...")
        self.camera.cleanup()
        self.audio_pipeline.cleanup()
        self.motion_model.cleanup()
        self.live2d_widget.cleanup()
        if self._pyaudio is not None:
            self._pyaudio.terminate()
        super().closeEvent(event)
