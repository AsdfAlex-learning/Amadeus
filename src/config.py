from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


def load_config(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def _validate_range(value: float, name: str, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a numeric value to [lo, hi] with a warning if out of range."""
    if value < lo or value > hi:
        import warnings
        warnings.warn(f"{name}={value} out of [{lo}, {hi}], clamped")
        return max(lo, min(hi, value))
    return value


def get_training_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the training config section with defaults."""
    if config is None:
        config = load_config()
    return config.get("training", {})


def get_lora_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the LoRA config section (nested under training.lora)."""
    training = get_training_config(config)
    return training.get("lora", {})


def get_preprocess_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the preprocess config section with defaults."""
    if config is None:
        config = load_config()
    return config.get("preprocess", {})


def get_performance_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the performance config section, with values validated to [0, 1]."""
    if config is None:
        config = load_config()
    perf = config.get("performance", {})
    # Clamp all performance params to [0.0, 1.0]
    for key in ("gesture_scale", "react_speed", "expressiveness",
                "mouth_open_max", "head_motion_range", "idle_energy"):
        if key in perf:
            perf[key] = _validate_range(perf[key], f"performance.{key}", 0.0, 1.0)
    return perf


def detect_device(prefer: str | None = None) -> str:
    """Auto-detect best available device.

    Priority: explicit prefer > CUDA > MPS > CPU.
    Returns: "cuda", "mps", or "cpu".
    """
    if prefer and prefer != "auto":
        return prefer

    try:
        import torch  # lazy import — config may be loaded before torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"