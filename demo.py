#!/usr/bin/env python3
"""Amadeus Demo — 最小可运行演示，逐组件降级。"""

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def check_component(name: str, import_path: str) -> tuple[bool, str]:
    try:
        __import__(import_path)
        return True, "✅"
    except Exception as e:
        return False, f"❌ {str(e)[:60]}"


def check_hardware(name: str) -> tuple[bool, str]:
    checks = {
        "麦克风": ("src.audio.capture", "MicrophoneCapture"),
        "摄像头": ("src.perception.camera", "CameraPerception"),
        "Live2D 模型": ("assets/live2d", None),
        "Whisper 模型": ("models/whisper", None),
        "动作模型权重": ("models/motion/full_duplex_dit.pt", None),
        "Ollama": ("http://localhost:11434", None),
    }
    if name == "麦克风":
        try:
            from src.audio.capture import MicrophoneCapture
            from src.config import load_config
            cfg = load_config()
            mic = MicrophoneCapture(cfg)
            devices = mic.get_audio_devices()
            mic.cleanup()
            if devices:
                return True, f"✅ {len(devices)}个设备"
            return False, "❌ 无可用设备"
        except Exception as e:
            return False, f"❌ {str(e)[:40]}"
    elif name == "摄像头":
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            ok = cap.isOpened()
            cap.release()
            return (True, "✅") if ok else (False, "❌ 无摄像头")
        except Exception:
            return False, "❌ cv2不可用"
    elif name in ("Live2D 模型", "Whisper 模型", "动作模型权重"):
        p = Path(name.split()[-1] if name == "动作模型权重" else
                 "assets/live2d" if "Live2D" in name else "models/whisper")
        if name == "动作模型权重":
            p = Path("models/motion/full_duplex_dit.pt")
        if name == "Whisper 模型":
            p = Path("models/whisper")
            if p.exists() and list(p.glob("ggml-*")):
                return True, "✅"
            return False, "❌ 未下载"
        if name == "Live2D 模型":
            p = Path("assets/live2d")
            if p.exists() and list(p.glob("*.model3.json")):
                return True, "✅"
            return False, "❌ 无模型文件"
        if p.exists():
            return True, "✅"
        return False, "❌ 未找到"
    elif name == "Ollama":
        try:
            from urllib import request
            req = request.Request("http://localhost:11434/api/tags")
            request.urlopen(req, timeout=2)
            return True, "✅"
        except Exception:
            return False, "❌ 未运行"


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Amadeus Demo — 组件状态检查")
        self.resize(900, 650)
        self._setup_ui()
        self._run_checks()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Amadeus v0.1.0 — Demo")
        title.setFont(QFont("PingFang SC", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("实时多模态 AI 伙伴 · Live2D 角色 · FullDuplexDiT 动作模型")
        subtitle.setFont(QFont("PingFang SC", 10))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #888;")
        layout.addWidget(subtitle)

        self.status_area = QTextEdit()
        self.status_area.setReadOnly(True)
        self.status_area.setFont(QFont("Menlo", 11))
        self.status_area.setStyleSheet(
            "QTextEdit { background: #1a1a2e; color: #e0e0e0; border: 1px solid #333; border-radius: 6px; padding: 12px; }"
        )
        layout.addWidget(self.status_area, stretch=2)

        chat_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("输入消息 (回车发送，文本 fallback 模式)...")
        self.chat_input.setFont(QFont("PingFang SC", 13))
        self.chat_input.setStyleSheet(
            "QLineEdit { background: #2a2a3e; color: white; border: 1px solid #555; border-radius: 4px; padding: 8px; }"
        )
        self.chat_input.returnPressed.connect(self._send_text)
        chat_row.addWidget(self.chat_input)

        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self._send_text)
        self.send_btn.setStyleSheet(
            "QPushButton { background: #4a90d9; color: white; border: none; padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background: #357abd; }"
        )
        chat_row.addWidget(self.send_btn)
        layout.addLayout(chat_row)

        self.ollama_status = QLabel("")
        self.ollama_status.setFont(QFont("Menlo", 10))
        self.ollama_status.setStyleSheet("color: #888; padding: 4px;")
        layout.addWidget(self.ollama_status)

    def _run_checks(self):
        checks = [
            ("Python 库", [
                ("PySide6", check_component("PySide6", "PySide6")),
                ("live2d-py", check_component("live2d-py", "live2d")),
                ("numpy", check_component("numpy", "numpy")),
                ("pyaudio", check_component("pyaudio", "pyaudio")),
                ("opencv", check_component("opencv", "cv2")),
                ("mediapipe", check_component("mediapipe", "mediapipe")),
                ("torch", check_component("torch", "torch")),
                ("transformers", check_component("transformers", "transformers")),
                ("pyyaml", check_component("pyyaml", "yaml")),
                ("soundfile", check_component("soundfile", "soundfile")),
                ("pyttsx3", check_component("pyttsx3", "pyttsx3")),
                ("loguru", check_component("loguru", "loguru")),
            ]),
            ("硬件 & 模型", [
                ("麦克风", check_hardware("麦克风")),
                ("摄像头", check_hardware("摄像头")),
                ("Live2D 模型", check_hardware("Live2D 模型")),
                ("Whisper 模型", check_hardware("Whisper 模型")),
                ("动作模型权重", check_hardware("动作模型权重")),
                ("Ollama (对话)", check_hardware("Ollama")),
            ]),
        ]

        lines = []
        for section, items in checks:
            lines.append(f"\n{'═' * 50}")
            lines.append(f"  {section}")
            lines.append(f"{'═' * 50}")
            for label, (ok, msg) in items:
                icon = "  ✅" if ok else "  ❌"
                lines.append(f"{icon} {label:<18} {msg}")
            ok_count = sum(1 for _, (o, _) in items if o)
            lines.append(f"  ── {ok_count}/{len(items)} 可用")

        lines.append(f"\n{'═' * 50}")
        lines.append(f"  当前可用功能")
        lines.append(f"{'═' * 50}")

        has_live2d = any("Live2D" in l and "✅" in l for l in lines)
        has_audio = any("麦克风" in l and "✅" in l for l in lines)
        has_ollama = any("Ollama" in l and "✅" in l for l in lines)
        has_camera = any("摄像头" in l and "✅" in l for l in lines)

        if has_live2d:
            lines.append("  ✅ Live2D 角色渲染")
        else:
            lines.append("  ⚠  Live2D 头显模式 (无模型文件)")

        if has_ollama:
            lines.append("  ✅ 文本对话 (Ollama)")
            self.ollama_status.setText("💬 对话就绪 — 在下方输入框打字聊天")
            self.ollama_status.setStyleSheet("color: #4a90d9; padding: 4px;")
        else:
            lines.append("  ⚠  对话不可用 (Ollama 未运行)")
            self.ollama_status.setText("⚠ Ollama 未运行 — 对话不可用，启动: ollama serve && ollama pull qwen2.5:3b")
            self.ollama_status.setStyleSheet("color: #d94a4a; padding: 4px;")

        if has_audio:
            lines.append("  ✅ 语音输入 (麦克风检测到)")
        else:
            lines.append("  ⚠  语音输入不可用")
        if has_camera:
            lines.append("  ✅ 摄像头感知")
        else:
            lines.append("  ⚠  摄像头不可用")

        lines.append("")
        lines.append("  文本 fallback 模式已激活 — 下方输入框可打字互动")

        self.status_area.setText("\n".join(lines))

    def _send_text(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        self.chat_input.clear()
        self.status_area.append(f"\n🧑 你: {text}")

        if check_hardware("Ollama")[0]:
            self._ollama_chat(text)
        else:
            self.status_area.append(f"🤖 Amadeus: (Ollama 未运行，无法回复)")

    def _ollama_chat(self, text: str):
        from threading import Thread

        def _chat():
            try:
                import json
                from urllib import request

                payload = json.dumps({
                    "model": "qwen2.5:3b",
                    "messages": [{"role": "user", "content": text}],
                    "stream": False,
                }).encode()
                req = request.Request("http://localhost:11434/api/chat", data=payload)
                req.add_header("Content-Type", "application/json")
                with request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                    reply = data.get("message", {}).get("content", "(无回复)")
                self.status_area.append(f"🤖 Amadeus: {reply}")
            except Exception as e:
                self.status_area.append(f"🤖 Amadeus: (错误: {e})")

        Thread(target=_chat, daemon=True).start()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QMainWindow { background-color: #12122a; }
        QWidget { background-color: #12122a; color: #e0e0e0; }
    """)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
