"""Smoke test: every module under ``src/`` must be importable.

The application entry point (``python -m src.main``) imports its UI modules
lazily from inside ``AmadeusWindow.__init__``. A broken import in a leaf module
therefore does not surface at package-import time -- it surfaces as an
immediate crash when the app starts, which is exactly the failure mode that
"has never been run end-to-end" hides.

This test closes that gap by importing every module in the package. It is the
cheapest possible end-to-end smoke check of the whole codebase, and it asserts
nothing about behavior -- only that the codebase can be loaded at all.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"


def _discover_modules() -> list[str]:
    """Return the dotted name of every module under ``src/``.

    Discovered from the filesystem rather than ``pkgutil.walk_packages`` so
    that collection has no import side effects.
    """
    names: set[str] = set()
    for py_file in SRC_ROOT.rglob("*.py"):
        relative = py_file.relative_to(SRC_ROOT.parent).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.add(".".join(parts))
    return sorted(names)


@pytest.mark.parametrize("module_name", _discover_modules())
def test_module_imports_cleanly(module_name: str) -> None:
    """Every module in the package must be importable without error."""
    importlib.import_module(module_name)
