"""Compatibility coverage for Zelda tooling that now lives in afs_scawful."""

from __future__ import annotations

import importlib.util

import pytest

from afs.agents import list_agents


def _has_extension(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def test_zelda_tools_are_extension_owned() -> None:
    if _has_extension("afs_scawful.agents.zelda_tools"):
        from afs.agents.zelda_tools import ZeldaToolkit

        assert ZeldaToolkit is not None
        return

    with pytest.raises(RuntimeError, match="afs_scawful extension repo"):
        from afs.agents.zelda_tools import ZeldaToolkit  # noqa: F401


def test_zelda_label_extractor_is_extension_owned() -> None:
    if _has_extension("afs_scawful.agents.zelda_tools"):
        from afs.agents.zelda_tools import _extract_labels_from_asm

        assert callable(_extract_labels_from_asm)
        return

    with pytest.raises(RuntimeError, match="afs_scawful extension repo"):
        from afs.agents.zelda_tools import _extract_labels_from_asm  # noqa: F401


def test_claude_orchestrator_is_not_a_core_agent() -> None:
    names = {spec.name for spec in list_agents()}
    assert "claude-orchestrator" not in names
