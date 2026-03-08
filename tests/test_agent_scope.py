from __future__ import annotations

import importlib.util

import pytest

from afs.agent.models import ModelConfig
from afs.agent.tools import TRIFORCE_TOOLS, create_triforce_tools


def _has_extension(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def test_domain_presets_require_extension() -> None:
    if _has_extension("afs_scawful.agent_model_presets"):
        config = ModelConfig.majora_lmstudio()
        assert config.model_id
        return

    with pytest.raises(RuntimeError, match="afs-scawful extension"):
        ModelConfig.majora_lmstudio()


def test_triforce_tools_are_disabled_without_extension() -> None:
    if _has_extension("afs_scawful.agent_tools"):
        tools = create_triforce_tools()
        assert tools
        return

    assert TRIFORCE_TOOLS == []
    with pytest.raises(RuntimeError, match="afs-scawful extension"):
        create_triforce_tools()
