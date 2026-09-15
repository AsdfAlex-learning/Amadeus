import threading
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
from src.runtime.playback import PlaybackWorker
from src.tts.engine import TTSEngine


class _PipelineBridge(QObject):
    text_received = Signal(str)
    state_changed = Signal(int)
    response_chunk = Signal(str)
    # Carries the generation id: a superseded generation's completion must not be
    # confused with the current one's.
    response_done = Signal(int)


class AmadeusWindow(QMainWindow):
    #: How long to wait for the per-utterance generation thread at shutdown.
    GENERATION_JOIN_TIMEOUT_S = 5.0

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
        # Each generation gets its own id. One shared cancel Event is not enough:
        # cancelling generation N would also abort generation N+1, which starts
        # immediately afterwards and would still see the flag set - so the
        # interrupting utterance would get no reply at all.
        self._generation_id = 0
        self._cancelled_generations: set[int] = set()
        self._shutting_down = False
        self._generation_thread: threading.Thread | None = None
        # T7: owns the output stream and publishes playback position. It creates
        # its own PyAudio instance, so no shared instance is raced by concurrent
        # generation threads.
        self._playback = PlaybackWorker(int(config["tts"].get("sample_rate", 22050)))
        self._playback.start()
        self._stream_seq = 0
        # Speak-state motion follows the audio the user actually hears (S2.4),
        # rather than the inference cadence. The position is counted at the TTS
        # rate and converted inside the engine.
        self.motion_model.attach_playback(
            self._playback, int(config["tts"].get("sample_rate", 22050))
        )
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
        # Interruption triggers on the VAD onset, not on the transcript: waiting
        # for the transcript would leave the character talking over the user for
        # the whole utterance.
        self.audio_pipeline.on_speech_onset(self._interrupt_speech)
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
            # Stage 1.4: inference gets its own owner thread. Without this,
            # process_user_audio runs inline on the capture drain worker and a
            # 0.5-3 s forward pass stalls VAD for its duration.
            self.motion_model.start_worker()
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
        self.conversation_display.append(f"<span style='color:#4a90d9'><b>你:</b> {text}</span>")
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
        # Keep the handle: shutdown must join this thread before PortAudio is torn
        # down, otherwise a generation thread inside a blocking stream.write()
        # races PyAudio.terminate(). That is PortAudio use-after-terminate, which
        # is a native crash rather than a Python exception.
        self._generation_id += 1
        generation_id = self._generation_id
        self._generation_thread = threading.Thread(
            target=self._run_generation,
            args=(generation_id,),
            daemon=True,
            name="amadeus-generation",
        )
        self._generation_thread.start()

    def _is_generation_cancelled(self, generation_id: int) -> bool:
        return self._shutting_down or generation_id in self._cancelled_generations

    def _cancel_generation(self, generation_id: int | None = None) -> None:
        """Cancel one generation, leaving any later one untouched."""
        target = self._generation_id if generation_id is None else generation_id
        if target > 0:
            self._cancelled_generations.add(target)
        if len(self._cancelled_generations) > 64:
            # Only the newest id can still be in flight.
            self._cancelled_generations = {max(self._cancelled_generations)}

    def _run_generation(self, generation_id: int):
        messages = [{"role": "system", "content": self.persona.format_prompt([])}]
        messages.extend(self.context.get_messages())
        full_response = ""
        try:
            for chunk in self.dialogue_model.generate(messages):
                if self._is_generation_cancelled(generation_id):
                    break  # superseded, interrupted, or shutting down
                full_response += chunk
                self._bridge.response_chunk.emit(chunk)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            self._bridge.response_chunk.emit("（抱歉，出了点问题...）")
        finally:
            # A truncated turn is KEPT: the user already heard part of it, and
            # deleting it would make the next reply reference a statement the
            # model believes it never made.
            self.context.add_assistant_message(full_response)
            self._bridge.response_done.emit(generation_id)
            if (
                self.tts_engine.is_loaded
                and full_response.strip()
                and not self._is_generation_cancelled(generation_id)
            ):
                self._speak_response(full_response)

    def _append_response(self, text: str):
        cursor = self.conversation_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        scrollbar = self.conversation_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_response_done(self, generation_id: int):
        # A superseded generation's completion must not clear the flag that now
        # belongs to its replacement. Doing so would let a third utterance spawn a
        # third concurrent generation sharing playback and the conversation
        # context - and the replacement's reply would never see the partial it
        # superseded.
        if generation_id != self._generation_id:
            return
        self._generating = False
        state_label = "● 聆听中..." if self.audio_pipeline.state != PipelineState.IDLE else "● 就绪"
        self.status_label.setText(state_label)

    def _speak_response(self, text: str):
        try:
            audio_data = self.tts_engine.synthesize(text)
            if audio_data is None or len(audio_data) == 0:
                return
            # Feed the motion engine the spoken audio (its Speak conditioning).
            self.motion_model.process_tts_audio(audio_data)
            # Playback runs on its own thread (T7) so that it can publish an
            # incremental playback position and be interrupted. Previously this
            # did one blocking stream.write() on the generation thread, which made
            # the position visible only at the end and made interruption unsafe.
            self._stream_seq += 1
            self._playback.play(self._stream_seq, audio_data)
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

    def _interrupt_speech(self):
        """React to the user speaking while the character is talking.

        Only the *reaction* can start now: the transcript is seconds away (VAD
        silence window plus Whisper), so this stops playback, drops the active
        utterance so the engine falls back to Listen, and cancels the in-flight
        generation. The replacement reply follows when the transcript arrives,
        which `_handle_text` starts in the usual way.
        """
        if self._shutting_down:
            return
        self._playback.stop_and_flush()
        self.motion_model.supersede_speech()
        self._cancel_generation()
        self._generating = False

    def closeEvent(self, event):
        logger.info("Shutting down Amadeus...")
        # Stop accepting work, cancel and join every worker, then release
        # resources. The generation thread must be joined BEFORE playback is torn
        # down, because it is the thread that submits audio to it.
        self._shutting_down = True
        self._cancel_generation()
        self._join_generation_thread()
        # T7 is stopped after the generation thread is joined (T5 is what submits
        # audio to it) and before the motion engine and renderer are released.
        # Its own stream and PyAudio instance are its responsibility to tear down,
        # and it leaks rather than closing them if it cannot stop in time.
        self._playback.shutdown()
        self.camera.cleanup()
        self.audio_pipeline.cleanup()
        self.motion_model.cleanup()
        self.live2d_widget.cleanup()
        super().closeEvent(event)

    def _join_generation_thread(self):
        thread, self._generation_thread = self._generation_thread, None
        if thread is None:
            return
        thread.join(timeout=self.GENERATION_JOIN_TIMEOUT_S)
        if thread.is_alive():
            # Escalation policy (ADR 0003): the local-transformers path runs the
            # whole generation inside one blocking model.generate call, so it
            # cannot honour cancellation between chunks. Leak the daemon rather
            # than release resources it may still be using.
            logger.warning(
                "Generation thread did not stop within %.0f s; leaking it (daemon)",
                self.GENERATION_JOIN_TIMEOUT_S,
            )
            self._generation_thread = thread
