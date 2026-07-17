"""Readable, scoped draft notes with explicit promotion and archival."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import (
    MarkdownArtifact,
    MarkdownArtifactCodec,
    NoteStore,
    validate_scope_id,
)
from .context_paths import resolve_mount_root
from .models import MountType


def _scope_directory(scope_id: str) -> Path:
    canonical = validate_scope_id(scope_id)
    if canonical == "common":
        return Path("common")
    return Path("projects") / canonical.removeprefix("project:")


class ScratchpadStore:
    """Manage immutable draft files and their explicit lifecycle.

    Draft contents are never rewritten. Promotion copies a draft into durable
    memory with provenance and leaves the source in place. Archival is the
    only operation that moves a draft, and it is always an explicit call.
    """

    def __init__(
        self,
        context_path: Path,
        *,
        scope_id: str = "common",
        config: Any = None,
    ) -> None:
        self.context_path = context_path.expanduser().resolve()
        self.scope_id = validate_scope_id(scope_id)
        scratchpad_root = resolve_mount_root(
            self.context_path,
            MountType.SCRATCHPAD,
            config=config,
        )
        scoped = scratchpad_root / _scope_directory(self.scope_id)
        self.active_root = scoped / "notes"
        self.archive_root = scoped / "archive"
        self._active = MarkdownArtifactCodec(self.active_root)
        self._archive = MarkdownArtifactCodec(self.archive_root)
        self._config = config

    def create(
        self,
        *,
        title: str,
        body: str,
        project_id: str = "",
        task_id: str = "",
        agent_name: str = "",
        author_kind: str = "human",
        sensitivity: str = "internal",
        provenance: Mapping[str, Any] | None = None,
    ) -> MarkdownArtifact:
        return self._active.create(
            kind="draft",
            title=title,
            body=body,
            scope_id=self.scope_id,
            project_id=project_id,
            task_id=task_id,
            agent_name=agent_name,
            author_kind=author_kind,
            sensitivity=sensitivity,
            provenance=provenance or {"source": "afs.notes.draft"},
        )

    def list(self, *, archived: bool = False, limit: int = 100) -> list[MarkdownArtifact]:
        if limit <= 0:
            return []
        codec = self._archive if archived else self._active
        drafts = list(codec.iter_artifacts(kind="draft"))
        drafts.sort(
            key=lambda item: (item.metadata.created_at, item.metadata.artifact_id),
            reverse=True,
        )
        return drafts[:limit]

    def read(
        self,
        identifier: str | Path,
        *,
        include_archived: bool = True,
    ) -> MarkdownArtifact | None:
        artifact = self._read_from(self._active, identifier)
        if artifact is not None or not include_archived:
            return artifact
        return self._read_from(self._archive, identifier)

    def is_archived(self, identifier: str | Path) -> bool:
        return self._read_from(self._archive, identifier) is not None

    def archive(self, identifier: str | Path) -> MarkdownArtifact:
        artifact = self._read_from(self._active, identifier)
        if artifact is None:
            archived = self._read_from(self._archive, identifier)
            if archived is not None:
                return archived
            raise FileNotFoundError(f"draft note not found: {identifier}")

        _rename_between_directories(
            self._active,
            self._archive,
            artifact.path.name,
        )
        archived = self._archive.read(self.archive_root / artifact.path.name)
        return archived

    def promote(self, identifier: str | Path) -> MarkdownArtifact:
        draft = self.read(identifier)
        if draft is None:
            raise FileNotFoundError(f"draft note not found: {identifier}")

        notes = NoteStore(
            self.context_path,
            scope_id=self.scope_id,
            config=self._config,
        )
        for note in notes.list(limit=1_000_000):
            provenance = note.metadata.provenance or {}
            if provenance.get("source_artifact_id") == draft.metadata.artifact_id:
                return note

        provenance = dict(draft.metadata.provenance or {})
        provenance.update(
            {
                "source": "afs.notes.promote",
                "source_artifact_id": draft.metadata.artifact_id,
                "source_path": str(draft.path),
            }
        )
        return notes.create(
            title=draft.metadata.title,
            body=draft.body,
            project_id=draft.metadata.project_id,
            task_id=draft.metadata.task_id,
            agent_name=draft.metadata.agent_name,
            author_kind=draft.metadata.author_kind,
            sensitivity=draft.metadata.sensitivity,
            provenance=provenance,
        )

    @staticmethod
    def _read_from(
        codec: MarkdownArtifactCodec,
        identifier: str | Path,
    ) -> MarkdownArtifact | None:
        raw = str(identifier).strip()
        if not raw:
            return None
        for artifact in codec.iter_artifacts(kind="draft"):
            if artifact.metadata.artifact_id == raw or artifact.path.name == raw:
                return artifact
            if str(artifact.path) == raw:
                return artifact
        return None


def _rename_between_directories(
    source: MarkdownArtifactCodec,
    destination: MarkdownArtifactCodec,
    filename: str,
) -> None:
    """Rename one contained file through the codecs' pinned directory handles."""

    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise ValueError("draft filename must be a contained basename")
    source_fd = source._duplicate_root_fd()
    destination_fd = destination._duplicate_root_fd()
    if source_fd is not None and destination_fd is not None:
        try:
            os.rename(
                filename,
                filename,
                src_dir_fd=source_fd,
                dst_dir_fd=destination_fd,
            )
        finally:
            os.close(source_fd)
            os.close(destination_fd)
        return
    if source_fd is not None:
        os.close(source_fd)
    if destination_fd is not None:
        os.close(destination_fd)

    # Platforms without descriptor-relative I/O retain a best-effort static
    # check immediately before the rename.
    source._assert_root_binding()
    destination._assert_root_binding()
    source_path = (source.root / filename).resolve()
    destination_path = (destination.root / filename).resolve(strict=False)
    source_path.relative_to(source.root.resolve())
    destination_path.relative_to(destination.root.resolve())
    os.replace(source_path, destination_path)


__all__ = ["ScratchpadStore"]
