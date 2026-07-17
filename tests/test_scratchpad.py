from __future__ import annotations

import os
from pathlib import Path

import pytest

import afs.scratchpad as scratchpad_module
from afs.context_layout import scaffold_v2
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


@pytest.mark.parametrize("target_kind", ["other-project", "outside"])
def test_scratchpad_store_rejects_symlinked_project_collection(
    tmp_path: Path,
    target_kind: str,
) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    target = (
        context / "scratchpad" / "projects" / "prj_beta" / "notes"
        if target_kind == "other-project"
        else tmp_path / "outside-drafts"
    )
    target.mkdir(parents=True)
    collection = context / "scratchpad" / "projects" / "prj_alpha" / "notes"
    collection.parent.mkdir(parents=True)
    try:
        collection.symlink_to(target, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        ScratchpadStore(context, scope_id="project:prj_alpha")
    assert list(target.iterdir()) == []


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


@pytest.mark.skipif(os.name != "posix", reason="descriptor-relative rename is POSIX-only")
def test_archive_uses_pinned_directory_after_regular_root_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = tmp_path / ".context"
    store = ScratchpadStore(context)
    draft = store.create(title="Pinned archive", body="working")
    detached_active = store.active_root.with_name("notes-detached")
    original_rename = scratchpad_module._rename_between_directories

    def rename_after_swap(*args, **kwargs) -> None:
        store.active_root.rename(detached_active)
        store.active_root.mkdir()
        original_rename(*args, **kwargs)

    monkeypatch.setattr(scratchpad_module, "_rename_between_directories", rename_after_swap)

    archived = store.archive(draft.metadata.artifact_id)

    assert archived.path.name == draft.path.name
    assert not (store.active_root / draft.path.name).exists()
    assert not (detached_active / draft.path.name).exists()
    assert (store.archive_root / draft.path.name).is_file()


def test_archive_rejects_unknown_identifier(tmp_path: Path) -> None:
    store = ScratchpadStore(tmp_path / ".context")
    with pytest.raises(FileNotFoundError, match="draft note not found"):
        store.archive("missing")
