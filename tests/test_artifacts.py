"""Tests for immutable, human-readable AFS artifacts."""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from afs.artifacts import MarkdownArtifactCodec, NoteStore, default_artifact_root
from afs.context_layout import scaffold_v2


def test_markdown_artifact_roundtrip_and_filename_contract(tmp_path: Path) -> None:
    codec = MarkdownArtifactCodec(tmp_path / "artifacts")
    artifact = codec.create(
        kind="note",
        title="A readable title / with unsafe path text " + "x" * 100,
        body="# Body\n\nUseful context.",
        scope_id="project:prj_a1b2c3",
        project_id="prj_a1b2c3",
        task_id="task-7",
        agent_name="codex",
        author_kind="agent",
        provenance={"source": "test", "inputs": ["one", "two"]},
    )

    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{6}Z--[a-z0-9-]{1,60}--[0-9a-f]{10}\.md",
        artifact.path.name,
    )
    assert artifact.path.stat().st_mode & 0o777 == 0o600
    assert artifact.path.parent == (tmp_path / "artifacts").resolve()

    restored = codec.read(artifact.path)
    assert restored.metadata == artifact.metadata
    assert restored.body == "# Body\n\nUseful context.\n"
    rendered = artifact.path.read_text(encoding="utf-8")
    assert rendered.startswith('+++\nschema_version = "1"')
    assert "[provenance]" in rendered


def test_artifact_explicit_id_never_overwrites(tmp_path: Path) -> None:
    codec = MarkdownArtifactCodec(tmp_path / "artifacts")
    artifact_id = "12345678-1234-5678-1234-567812345678"
    first = codec.create(
        kind="note",
        title="Collision test",
        body="first",
        scope_id="common",
        artifact_id=artifact_id,
        created_at="2026-07-17T00:00:00Z",
    )

    with pytest.raises(FileExistsError):
        codec.create(
            kind="note",
            title="Different filename, same identifier",
            body="second",
            scope_id="common",
            artifact_id=artifact_id,
            created_at="2026-07-17T00:00:00Z",
        )
    assert codec.read(first.path).body == "first\n"


def test_artifact_rejects_unsafe_metadata_and_paths(tmp_path: Path) -> None:
    codec = MarkdownArtifactCodec(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="scope_id"):
        codec.create(kind="note", title="x", body="x", scope_id="../other")
    with pytest.raises(ValueError, match="relative_dir"):
        codec.create(
            kind="note",
            title="x",
            body="x",
            scope_id="common",
            relative_dir="../../escape",
        )
    with pytest.raises(ValueError, match="provenance keys"):
        codec.create(
            kind="note",
            title="x",
            body="x",
            scope_id="common",
            provenance={"../unsafe": True},
        )


def test_artifact_rejects_relative_directory_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "stream").symlink_to(outside, target_is_directory=True)
    codec = MarkdownArtifactCodec(root)

    with pytest.raises(ValueError, match="outside the artifact root"):
        codec.create(
            kind="handoff",
            title="No escape",
            body="body",
            scope_id="common",
            relative_dir="stream",
        )
    assert list(outside.iterdir()) == []


def test_artifact_codec_rejects_symlinked_collection_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    collection = tmp_path / "artifacts"
    try:
        collection.symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        MarkdownArtifactCodec(collection)
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="descriptor-relative publication is POSIX-only")
@pytest.mark.parametrize("swap_root", [False, True], ids=["intermediate", "root"])
def test_artifact_publication_rejects_directory_swap(
    tmp_path: Path,
    swap_root: bool,
) -> None:
    root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    outside.mkdir()

    class SwappingCodec(MarkdownArtifactCodec):
        swapped = False

        @staticmethod
        def _write_exclusive_at(directory_fd: int, filename: str, payload: str) -> None:
            if not SwappingCodec.swapped:
                SwappingCodec.swapped = True
                if swap_root:
                    root.rename(tmp_path / "parked-root")
                    root.symlink_to(outside, target_is_directory=True)
                else:
                    (root / "stream").rename(root / "parked-stream")
                    (root / "stream").symlink_to(outside, target_is_directory=True)
            MarkdownArtifactCodec._write_exclusive_at(directory_fd, filename, payload)

    codec = SwappingCodec(root)
    with pytest.raises(OSError, match="changed during publication"):
        codec.create(
            kind="handoff",
            title="No swapped publication",
            body="body",
            scope_id="common",
            relative_dir="stream",
        )

    assert list(outside.iterdir()) == []
    parked = tmp_path / "parked-root" if swap_root else root / "parked-stream"
    assert list(parked.rglob("*.md")) == []


def test_note_store_uses_canonical_project_scope(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    context.mkdir()
    note = NoteStore(context, scope_id="project:prj_42").create(
        title="Design decision",
        body="Prefer immutable records.",
        author_kind="human",
    )

    expected = context / "memory" / "projects" / "prj_42" / "notes"
    assert note.path.parent == expected.resolve()
    assert note.metadata.scope_id == "project:prj_42"
    assert note.metadata.project_id == "prj_42"
    assert note.metadata.provenance == {"source": "afs.note"}


@pytest.mark.parametrize("target_kind", ["other-project", "outside"])
def test_note_store_rejects_symlinked_project_collection(
    tmp_path: Path,
    target_kind: str,
) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    target = (
        context / "memory" / "projects" / "prj_beta" / "notes"
        if target_kind == "other-project"
        else tmp_path / "outside-notes"
    )
    target.mkdir(parents=True)
    collection = context / "memory" / "projects" / "prj_alpha" / "notes"
    collection.parent.mkdir(parents=True)
    try:
        collection.symlink_to(target, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        NoteStore(context, scope_id="project:prj_alpha")
    assert list(target.iterdir()) == []


def test_note_store_common_scope_and_custom_resolver(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    context.mkdir()
    common = NoteStore(context).create(title="Common note", body="shared")
    assert common.path.parent == (context / "memory" / "common" / "notes").resolve()

    def resolver(
        context_path: Path,
        *,
        scope_id: str,
        collection: str,
        config: object = None,
    ) -> Path:
        del context_path, config
        return tmp_path / "custom" / scope_id.replace(":", "-") / collection

    custom = NoteStore(
        context,
        scope_id="project:prj_custom",
        root_resolver=resolver,
    ).create(title="Custom root", body="custom")
    assert custom.path.parent == (tmp_path / "custom" / "project-prj_custom" / "notes").resolve()


def test_note_store_lists_and_reads_by_id_or_contained_path(tmp_path: Path) -> None:
    store = NoteStore(tmp_path / ".context")
    first = store.create(title="First", body="one")
    store.create(title="Second", body="two")

    assert [item.metadata.title for item in store.list(limit=2)] == ["Second", "First"]
    assert store.read(first.metadata.artifact_id) == first
    assert store.read(first.path.name) == first
    assert store.read("../../outside.md") is None
    assert store.read("00000000-0000-0000-0000-000000000000") is None


def test_default_artifact_root_rejects_noncanonical_project_scope(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    context.mkdir()
    with pytest.raises(ValueError, match="scope_id"):
        default_artifact_root(context, scope_id="prj_42", collection="notes")
    with pytest.raises(ValueError, match="project scope id"):
        default_artifact_root(
            context,
            scope_id="project:../../escape",
            collection="notes",
        )
    with pytest.raises(ValueError, match="project_id must match"):
        MarkdownArtifactCodec(tmp_path / "artifacts").create(
            kind="note",
            title="Mismatch",
            body="body",
            scope_id="project:prj_42",
            project_id="prj_other",
        )


def test_concurrent_note_creation_is_unique_and_lossless(tmp_path: Path) -> None:
    store = NoteStore(tmp_path / ".context")

    def create(index: int) -> tuple[str, str]:
        artifact = store.create(title="Shared title", body=f"body {index}")
        return artifact.path.name, artifact.metadata.artifact_id

    with ThreadPoolExecutor(max_workers=12) as pool:
        created = list(pool.map(create, range(64)))

    assert len({name for name, _ in created}) == 64
    assert len({artifact_id for _, artifact_id in created}) == 64
    assert len(list(store.root.glob("*.md"))) == 64
