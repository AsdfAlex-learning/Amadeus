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
