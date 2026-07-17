from __future__ import annotations

from pathlib import Path

import pytest

from afs.scratchpad import ScratchpadStore


def test_draft_names_are_readable_unique_and_scoped(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    store = ScratchpadStore(context, scope_id="project:prj_alpha")

    first = store.create(title="Review the search plan", body="first")
    second = store.create(title="Review the search plan", body="second")

    assert first.path != second.path
    assert "review-the-search-plan" in first.path.name
    assert first.path.parent == (
        context / "scratchpad" / "projects" / "prj_alpha" / "notes"
    ).resolve()
    assert store.read(first.metadata.artifact_id) == first


def test_promote_is_idempotent_and_preserves_draft(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    store = ScratchpadStore(context, scope_id="project:prj_alpha")
    draft = store.create(
        title="Keep this decision",
        body="Durable rationale.",
        project_id="prj_alpha",
        agent_name="codex",
    )

    promoted = store.promote(draft.metadata.artifact_id)
    repeated = store.promote(draft.metadata.artifact_id)

    assert draft.path.exists()
    assert promoted.path == repeated.path
    assert promoted.body == draft.body
    assert promoted.metadata.provenance is not None
    assert promoted.metadata.provenance["source_artifact_id"] == draft.metadata.artifact_id
    assert promoted.path.parent == (
        context / "memory" / "projects" / "prj_alpha" / "notes"
    ).resolve()


def test_archive_is_explicit_idempotent_and_still_readable(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    store = ScratchpadStore(context)
    draft = store.create(title="Temporary investigation", body="working")

    archived = store.archive(draft.metadata.artifact_id)
    repeated = store.archive(draft.metadata.artifact_id)

    assert not draft.path.exists()
    assert archived.path == repeated.path
    assert archived.path.parent == (context / "scratchpad" / "common" / "archive").resolve()
    assert store.list() == []
    assert store.list(archived=True) == [archived]
    assert store.read(draft.metadata.artifact_id) == archived


def test_archive_rejects_unknown_identifier(tmp_path: Path) -> None:
    store = ScratchpadStore(tmp_path / ".context")
    with pytest.raises(FileNotFoundError, match="draft note not found"):
        store.archive("missing")
