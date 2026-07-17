from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from afs.cli.embeddings import _resolve_knowledge_root
from afs.context_layout import scaffold_v2
from afs.schema import AFSConfig, GeneralConfig


def _args(context_root: Path, *, project: str = "alpha", knowledge_path: str | None = None):
    return argparse.Namespace(
        context_root=str(context_root),
        project=project,
        knowledge_path=knowledge_path,
        knowledge_dir=None,
    )


def test_default_embedding_index_rejects_unscoped_v2_legacy_path(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)

    with pytest.raises(ValueError, match="afs search --semantic --rebuild"):
        _resolve_knowledge_root(_args(context_root), AFSConfig())

    assert not (context_root / "knowledge" / "common" / "alpha").exists()


def test_embedding_index_preserves_v1_and_explicit_override_paths(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    config = AFSConfig(general=GeneralConfig(context_root=context_root))

    assert _resolve_knowledge_root(_args(context_root), config) == (
        context_root / "knowledge" / "alpha"
    )

    explicit = tmp_path / "external-index"
    assert _resolve_knowledge_root(
        _args(context_root, project="../ignored", knowledge_path=str(explicit)),
        config,
    ) == explicit


def test_explicit_embedding_override_remains_available_in_v2(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    explicit = tmp_path / "external-index"

    assert _resolve_knowledge_root(
        _args(context_root, knowledge_path=str(explicit)),
        AFSConfig(),
    ) == explicit
