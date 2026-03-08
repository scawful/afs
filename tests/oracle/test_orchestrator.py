from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest


def _has_extension(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def test_oracle_orchestrator_is_extension_owned() -> None:
    sys.modules.pop("afs.oracle", None)
    sys.modules.pop("afs.oracle.orchestrator", None)

    if _has_extension("afs_scawful.oracle"):
        module = importlib.import_module("afs.oracle.orchestrator")
        assert hasattr(module, "TriforceOrchestrator")
        return

    with pytest.raises(RuntimeError, match="afs-scawful extension"):
        importlib.import_module("afs.oracle.orchestrator")
