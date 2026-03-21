from __future__ import annotations

import json
from pathlib import Path

from afs.context_index import ContextSQLiteIndex
from afs.embeddings import build_embedding_index
from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import AFSConfig, GeneralConfig, SensitivityConfig
from afs.sensitivity import filtered_tree_copy, matches_path_rules


def test_matches_path_rules_supports_relative_and_absolute(tmp_path: Path) -> None:
    candidate = tmp_path / "knowledge" / "private" / "note.md"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("secret", encoding="utf-8")

    assert matches_path_rules(
        candidate,
        relative_path="knowledge/private/note.md",
        patterns=["knowledge/private/*"],
    )
    assert matches_path_rules(
        candidate,
        relative_path="knowledge/private/note.md",
        patterns=[str(candidate)],
    )
    assert not matches_path_rules(
        candidate,
        relative_path="knowledge/private/note.md",
        patterns=["knowledge/public/*"],
    )


def test_context_index_respects_never_index_rules(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    config = AFSConfig(
        general=GeneralConfig(context_root=context_root),
        sensitivity=SensitivityConfig(never_index=["knowledge/private/*"]),
    )
    manager = AFSManager(config=config)
    manager.ensure(context_root=context_root)
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    (knowledge_root / "private").mkdir(parents=True, exist_ok=True)
    (knowledge_root / "public").mkdir(parents=True, exist_ok=True)
    (knowledge_root / "private" / "secret.md").write_text("secret", encoding="utf-8")
    (knowledge_root / "public" / "guide.md").write_text("guide", encoding="utf-8")

    index = ContextSQLiteIndex(manager, context_root)
    index.rebuild(mount_types=[MountType.KNOWLEDGE])
    results = index.query(mount_types=[MountType.KNOWLEDGE], limit=10)

    relative_paths = {item["relative_path"] for item in results}
    assert "public/guide.md" in relative_paths
    assert "private/secret.md" not in relative_paths


def test_embedding_index_respects_skip_path(tmp_path: Path) -> None:
    source_root = tmp_path / "knowledge"
    source_root.mkdir()
    (source_root / "guide.md").write_text("guide", encoding="utf-8")
    (source_root / "private.md").write_text("private", encoding="utf-8")
    output_root = tmp_path / "index"

    result = build_embedding_index(
        [source_root],
        output_root,
        skip_path=lambda root, path: matches_path_rules(
            path,
            relative_path=path.relative_to(root).as_posix(),
            patterns=["private.md"],
        ),
    )

    index = json.loads((output_root / "embedding_index.json").read_text(encoding="utf-8"))
    documents = {key: value for key, value in index.items() if key != "_metadata"}
    assert result.indexed == 1
    assert len(documents) == 1
    only_doc = next(iter(documents))
    assert "guide.md" in only_doc


def test_prepare_export_root_filters_sensitive_paths(tmp_path: Path) -> None:
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    (memory_root / "public.json").write_text("{}", encoding="utf-8")
    (memory_root / "secret.json").write_text("{}", encoding="utf-8")

    with filtered_tree_copy(memory_root, ["secret.json"]) as export_root:
        assert (export_root / "public.json").exists()
        assert not (export_root / "secret.json").exists()
