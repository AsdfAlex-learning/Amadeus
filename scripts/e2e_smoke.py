"""Run the closest available end-to-end smoke for the Amadeus runtime.

Usage:
    python scripts/e2e_smoke.py --json [--camera] [--tts] [--dialogue]
    python scripts/e2e_smoke.py --checkpoint-dir models/motion/first_run

Core checks always run: config, audio devices, Live2D package/model presence,
and the motion engine (Phase 0 baseline or a real checkpoint). Hardware and
network checks are opt-in so the script can run on a headless machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _result(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def check_config(config: dict) -> dict:
    required = ("app", "live2d", "audio", "asr", "dialogue", "tts", "motion", "perception")
    missing = [key for key in required if key not in config]
    if missing:
        return _result("config", "FAIL", f"missing sections: {missing}")
    return _result("config", "PASS", f"{len(config)} sections loaded")


def check_audio_devices(config: dict) -> dict:
    from src.audio.capture import MicrophoneCapture

    mic = MicrophoneCapture(config)
    try:
        devices = mic.get_audio_devices()
    finally:
        mic.cleanup()
    if not devices:
        return _result("audio", "SKIP", "no input devices exposed")
    return _result("audio", "PASS", f"{len(devices)} input device(s)")


def check_camera(config: dict, open_camera: bool) -> dict:
    from src.perception.camera import CameraPerception

    camera = CameraPerception(config)
    if not open_camera:
        return _result(
            "camera",
            "SKIP",
            f"not opened (device={config['perception'].get('camera_device')!r}); pass --camera",
        )
    thread = threading.Thread(target=camera.start, daemon=True)
    thread.start()
    thread.join(timeout=5.0)
    try:
        if not camera.is_running:
            return _result("camera", "SKIP", "no camera available or open timed out")
        return _result("camera", "PASS", "camera opened and capture thread running")
    finally:
        camera.cleanup()


def check_live2d(config: dict) -> dict:
    try:
        import live2d.v3 as live2d
    except Exception as e:
        return _result("live2d", "SKIP", f"live2d-py unavailable: {e}")
    model_name = config["live2d"].get("model_name")
    if not model_name:
        return _result(
            "live2d",
            "SKIP",
            f"package {getattr(live2d, 'LIVE2D_VERSION', '?')} available; no model_name configured",
        )
    model_path = Path(config["live2d"]["model_dir"]) / str(model_name)
    if not model_path.exists():
        return _result("live2d", "SKIP", f"model file missing: {model_path}")
    from src.live2d.renderer import Live2DRenderer

    renderer = Live2DRenderer(config)
    try:
        loaded = renderer.initialize()
    finally:
        renderer.cleanup()
    if not loaded:
        return _result("live2d", "SKIP", f"renderer could not load {model_path} headlessly")
    return _result("live2d", "PASS", f"loaded {model_path}")


def check_tts(config: dict, load_tts: bool) -> dict:
    if not load_tts:
        return _result("tts", "SKIP", "pass --tts to initialize the configured engine")
    from src.tts.engine import TTSEngine

    engine = TTSEngine(config)
    try:
        loaded = engine.load()
    finally:
        engine.cleanup()
    if not loaded:
        return _result("tts", "SKIP", f"engine {config['tts'].get('engine')} unavailable")
    return _result("tts", "PASS", f"engine {config['tts'].get('engine')} loaded")


def check_dialogue(config: dict, use_dialogue: bool) -> dict:
    dialogue = config.get("dialogue", {}) or {}
    provider = dialogue.get("provider")
    api_base = dialogue.get("api_base")
    model = dialogue.get("model")
    key_env = dialogue.get("api_key_env", "AMADEUS_API_KEY")
    if not provider or not api_base or not model:
        return _result("dialogue", "SKIP", "provider/api_base/model not configured")
    if not os.environ.get(key_env):
        return _result("dialogue", "SKIP", f"{key_env} is not set")
    if not use_dialogue:
        return _result(
            "dialogue", "SKIP", f"configured ({provider}/{model}); pass --dialogue to call it"
        )
    from src.dialogue.model import DialogueModel

    model_client = DialogueModel(config)
    cancel = threading.Event()
    chunks: list[str] = []
    error: list[str] = []
    thread = threading.Thread(
        target=lambda: _collect_dialogue(model_client, cancel, chunks, error), daemon=True
    )
    thread.start()
    thread.join(timeout=30.0)
    if thread.is_alive():
        cancel.set()
        return _result("dialogue", "FAIL", "generation did not finish within 30 s")
    if error:
        return _result("dialogue", "FAIL", error[0])
    text = "".join(chunks).strip()
    return _result("dialogue", "PASS" if text else "FAIL", f"{len(text)} chars returned")


def _collect_dialogue(client, cancel, chunks, error) -> None:
    try:
        for chunk in client.generate(
            [{"role": "user", "content": "Reply with one short sentence."}],
            cancel_event=cancel,
        ):
            chunks.append(chunk)
    except Exception as e:
        error.append(str(e))


def check_motion(config: dict, checkpoint_dir: str | None) -> dict:
    from src.motion.inference import DiffusionMotionInference

    cfg = json.loads(json.dumps(config))
    if checkpoint_dir:
        cfg["motion"]["model_path"] = checkpoint_dir
    engine = DiffusionMotionInference(cfg)
    params: list[dict] = []
    try:
        if not engine.load_model():
            return _result("motion", "FAIL", "load_model returned False")
        engine.on_params(params.append)
        times = np.arange(16000) / 16000
        tts = (0.5 * np.sin(2.0 * np.pi * 180.0 * times)).astype(np.float32)
        engine.process_tts_audio(tts, sample_rate=16000)
        backend = "Phase 0 baseline" if engine._use_baseline else "neural checkpoint"
        if not params:
            return _result("motion", "FAIL", f"{backend} produced no parameter frames")
        frames = len(params)
        values = np.array([[frame[f"param_{i}"] for i in range(45)] for frame in params])
        if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
            return _result("motion", "FAIL", f"{backend} produced out-of-range parameters")
        return _result(
            "motion",
            "PASS",
            f"{backend}; {frames} frames x 45 params; range [{values.min():.3f}, {values.max():.3f}]",
        )
    finally:
        engine.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description="Amadeus end-to-end smoke")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--camera", action="store_true", help="open the configured camera")
    parser.add_argument("--tts", action="store_true", help="initialize the configured TTS engine")
    parser.add_argument("--dialogue", action="store_true", help="call the configured dialogue API")
    parser.add_argument("--checkpoint-dir", default=None, help="motion checkpoint directory")
    args = parser.parse_args()

    from src.config import load_config

    config = load_config(ROOT / "config" / "default.yaml")
    checks = [
        check_config(config),
        check_audio_devices(config),
        check_camera(config, args.camera),
        check_live2d(config),
        check_tts(config, args.tts),
        check_dialogue(config, args.dialogue),
        check_motion(config, args.checkpoint_dir),
    ]
    core = {"config", "motion"}
    failures = [item for item in checks if item["name"] in core and item["status"] == "FAIL"]
    if args.json:
        print(json.dumps({"checks": checks, "ok": not failures}, ensure_ascii=False, indent=2))
    else:
        for item in checks:
            print(f"[{item['status']}] {item['name']}: {item['detail']}")
        print("RESULT:", "PASS" if not failures else "FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
