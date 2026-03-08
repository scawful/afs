from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest

from afs.knowledge import PersonalKnowledgeGraph


def _has_extension(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def test_core_knowledge_graph_stays_in_core() -> None:
    graph = PersonalKnowledgeGraph()
    assert isinstance(graph, PersonalKnowledgeGraph)


def test_zelda_entity_extractor_is_extension_owned() -> None:
    sys.modules.pop("afs.knowledge.entity_extractor", None)

    if _has_extension("afs_scawful.knowledge"):
        module = importlib.import_module("afs.knowledge.entity_extractor")
        assert hasattr(module, "EntityExtractor")
        return

    with pytest.raises(RuntimeError, match="afs-scawful extension"):
        importlib.import_module("afs.knowledge.entity_extractor")
