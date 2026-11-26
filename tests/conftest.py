"""Shared fixtures and hermetic-environment guard for the Amadeus test suite.

These tests must not touch the network, the microphone, the camera, or the
~400 MB of pretrained encoder weights. We therefore force the HuggingFace /
transformers stack into offline mode *before* it is imported anywhere, so the
motion model's frozen encoders (Hubert, MobileNetV3, BERT) fall back to their
built-in deterministic stubs.

That keeps the contract tests hermetic and fast. It does not weaken them: the
defects they target (tensor layout, sample-rate mismatch, dtype promotion,
threading) are all independent of the real encoder weights.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Point HuggingFace at an empty scratch cache *and* force offline mode, so the
# pretrained encoders always take their "not cached" path. This must happen
# before transformers / huggingface_hub are imported by any test module.
_HF_SCRATCH = tempfile.mkdtemp(prefix="amadeus-test-hf-")
os.environ["HF_HOME"] = _HF_SCRATCH
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# Tests must never require a display, a GPU surface, or a window manager.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _force_offline_encoders(monkeypatch):
    """Keep the three frozen encoders off the network.

    Hubert and BERT already fall back to their built-in stubs when the
    HuggingFace cache is empty (see the module docstring). torchvision has no
    such fallback: ``mobilenet_v3_small(weights="DEFAULT")`` will cheerfully
    download ~10 MB from download.pytorch.org even in "offline" mode. We pin
    it to ``weights=None`` so the visual encoder is deterministic random-init.

    Only the *weights* change; the architecture and every tensor shape are
    identical, and no test here asserts on encoder outputs.
    """
    import torchvision.models as tvm

    real_mobilenet = tvm.mobilenet_v3_small

    def _no_download(*_args, **kwargs):
        if kwargs.get("weights") is not None:
            raise RuntimeError("weight downloads are disabled in tests")
        return real_mobilenet(weights=None)

    monkeypatch.setattr(tvm, "mobilenet_v3_small", _no_download)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root."""
    return PROJECT_ROOT


@pytest.fixture
def config() -> dict:
    """The project's default configuration, loaded by explicit path.

    We pass the path explicitly because ``load_config()`` defaults to a
    CWD-relative ``config/default.yaml``; tests must not depend on the CWD.
    """
    from src.config import load_config

    return load_config(PROJECT_ROOT / "config" / "default.yaml")


@pytest.fixture(scope="session")
def qapp():
    """A single offscreen QApplication for the session.

    Qt allows only one QApplication per process, so this is session-scoped.
    ``QT_QPA_PLATFORM=offscreen`` is set at import time above, so no display is
    required.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
