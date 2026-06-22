"""Compatibility shim for Zelda research tools now owned by ``afs_scawful``."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from afs_scawful.agents.zelda_tools import ZeldaToolkit, _extract_labels_from_asm

_EXTENSION_MODULE = "afs_scawful.agents.zelda_tools"
_ERROR = (
    "Zelda research tools moved out of core AFS. Install/enable the "
    "afs_scawful extension repo and import afs_scawful.agents.zelda_tools."
)


def _load_extension_module() -> Any:
    try:
        return importlib.import_module(_EXTENSION_MODULE)
    except Exception as exc:  # pragma: no cover - compatibility path
        raise RuntimeError(_ERROR) from exc


def __getattr__(name: str) -> Any:
    return getattr(_load_extension_module(), name)


__all__ = ["ZeldaToolkit", "_extract_labels_from_asm"]
