"""Compatibility shim for extension-owned Oracle orchestration."""

from __future__ import annotations

import importlib

_ERROR = (
    "oracle.orchestrator requires the afs_scawful extension repo. "
    "Enable afs_scawful or see docs/EXTENSION_MIGRATION.md."
)

try:  # pragma: no cover - compatibility path when extension is installed
    _module = importlib.import_module("afs_scawful.oracle.orchestrator")
except Exception as exc:  # pragma: no cover - default path without extension
    raise RuntimeError(_ERROR) from exc

TriforceOrchestrator = _module.TriforceOrchestrator

__all__ = ["TriforceOrchestrator"]
