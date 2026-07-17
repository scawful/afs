"""Safe, human-readable Markdown artifacts for durable AFS records.

Artifacts use TOML front matter and an immutable, sortable filename.  The
codec deliberately owns filename creation so callers cannot accidentally
overwrite another note or smuggle path separators through a title.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import tomlkit

from .context_paths import resolve_mount_root
from .models import MountType
from .toml_compat import tomllib

ARTIFACT_SCHEMA_VERSION = "1"
_FRONTMATTER_DELIMITER = "+++"
_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SAFE_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PROVENANCE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_ALLOWED_AUTHOR_KINDS = frozenset({"agent", "human", "import", "system"})
_ALLOWED_SENSITIVITIES = frozenset({"internal", "private", "public", "restricted"})


class ArtifactRootResolver(Protocol):
    """Resolve the directory for a scoped artifact collection."""

    def __call__(
        self,
        context_path: Path,
        *,
        scope_id: str,
        collection: str,
        config: Any = None,
    ) -> Path: ...


def _validate_safe_segment(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not _SAFE_SEGMENT_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{field_name} must be a safe identifier containing only letters, "
            "numbers, '.', '_' or '-'"
        )
    if normalized in {".", ".."}:
        raise ValueError(f"{field_name} cannot be '.' or '..'")
    return normalized


def validate_scope_id(value: str) -> str:
    """Validate the canonical ``common`` or ``project:<id>`` scope grammar."""

    normalized = value.strip()
    if normalized == "common":
        return normalized
    prefix, separator, project_id = normalized.partition(":")
    if prefix != "project" or not separator:
        raise ValueError("scope_id must be 'common' or 'project:<project-id>'")
    safe_project_id = _validate_safe_segment(project_id, field_name="project scope id")
    return f"project:{safe_project_id}"


def _validate_text(value: str, *, field_name: str, required: bool = False) -> str:
    normalized = value.strip()
    if required and not normalized:
        raise ValueError(f"{field_name} is required")
    if _CONTROL_PATTERN.search(normalized):
        raise ValueError(f"{field_name} cannot contain control characters")
    return normalized


def _normalize_artifact_id(value: str | None) -> str:
    if value is None:
        return uuid.uuid4().hex
    raw = value.strip().lower()
    try:
        normalized = uuid.UUID(raw).hex
    except (ValueError, AttributeError) as exc:
        raise ValueError("artifact_id must be a full UUID") from exc
    if not _ID_PATTERN.fullmatch(normalized):  # pragma: no cover - UUID guarantees this
        raise ValueError("artifact_id must be a full UUID")
    return normalized


def _normalize_created_at(value: str | datetime | None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    else:
        raw = value.strip()
        if raw.endswith("Z"):
            raw = f"{raw[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError("created_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at must include a timezone")
    parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_provenance_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        raise ValueError("provenance cannot be nested more than 6 levels")
    if isinstance(value, str):
        if "\x00" in value:
            raise ValueError("provenance strings cannot contain NUL characters")
        return value
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("provenance numbers must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not _PROVENANCE_KEY_PATTERN.fullmatch(key):
                raise ValueError("provenance keys must be safe identifiers")
            result[key] = _validate_provenance_value(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_validate_provenance_value(item, depth=depth + 1) for item in value]
    raise ValueError(f"unsupported provenance value type: {type(value).__name__}")


def _slugify(title: str, *, max_length: int = 60) -> str:
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")
    slug = slug[:max_length].rstrip("-")
    return slug or "artifact"


def default_artifact_root(
    context_path: Path,
    *,
    scope_id: str,
    collection: str,
    config: Any = None,
) -> Path:
    """Resolve the canonical common or project-scoped artifact directory."""

    canonical_scope = validate_scope_id(scope_id)
    safe_collection = _validate_safe_segment(collection, field_name="collection")
    memory_root = resolve_mount_root(context_path, MountType.MEMORY, config=config)
    if canonical_scope == "common":
        return memory_root / "common" / safe_collection
    project_id = canonical_scope.removeprefix("project:")
    return memory_root / "projects" / project_id / safe_collection


def infer_scope_id(context_path: Path) -> str:
    """Return the conservative compatibility scope.

    Paths and worktrees are aliases, not durable project identity.  Callers
    with a project registry should pass its canonical ``project:<id>`` scope;
    legacy callers safely fall back to the shared ``common`` scope.
    """

    del context_path
    return "common"


@dataclass(frozen=True)
class ArtifactMetadata:
    """Validated metadata shared by all human-facing AFS artifacts."""

    schema_version: str
    artifact_id: str
    kind: str
    title: str
    created_at: str
    scope_id: str
    project_id: str = ""
    task_id: str = ""
    agent_name: str = ""
    author_kind: str = "agent"
    sensitivity: str = "internal"
    provenance: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        title: str,
        scope_id: str,
        project_id: str = "",
        task_id: str = "",
        agent_name: str = "",
        author_kind: str = "agent",
        sensitivity: str = "internal",
        provenance: Mapping[str, Any] | None = None,
        artifact_id: str | None = None,
        created_at: str | datetime | None = None,
    ) -> ArtifactMetadata:
        safe_kind = _validate_safe_segment(kind, field_name="kind")
        safe_scope = validate_scope_id(scope_id)
        safe_title = _validate_text(title, field_name="title", required=True)
        if len(safe_title) > 240:
            raise ValueError("title cannot exceed 240 characters")
        if author_kind not in _ALLOWED_AUTHOR_KINDS:
            raise ValueError(f"author_kind must be one of {sorted(_ALLOWED_AUTHOR_KINDS)}")
        if sensitivity not in _ALLOWED_SENSITIVITIES:
            raise ValueError(f"sensitivity must be one of {sorted(_ALLOWED_SENSITIVITIES)}")
        normalized_provenance = _validate_provenance_value(dict(provenance or {}))
        safe_project_id = _validate_text(project_id, field_name="project_id")
        if safe_scope.startswith("project:"):
            scoped_project_id = safe_scope.removeprefix("project:")
            if safe_project_id and safe_project_id != scoped_project_id:
                raise ValueError("project_id must match the project scope")
            safe_project_id = scoped_project_id
        return cls(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            artifact_id=_normalize_artifact_id(artifact_id),
            kind=safe_kind,
            title=safe_title,
            created_at=_normalize_created_at(created_at),
            scope_id=safe_scope,
            project_id=safe_project_id,
            task_id=_validate_text(task_id, field_name="task_id"),
            agent_name=_validate_text(agent_name, field_name="agent_name"),
            author_kind=author_kind,
            sensitivity=sensitivity,
            provenance=normalized_provenance,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactMetadata:
        required = {
            "schema_version",
            "artifact_id",
            "kind",
            "title",
            "created_at",
            "scope_id",
            "project_id",
            "task_id",
            "agent_name",
            "author_kind",
            "sensitivity",
            "provenance",
        }
        missing = sorted(required.difference(data))
        if missing:
            raise ValueError(f"artifact metadata is missing: {', '.join(missing)}")
        if str(data["schema_version"]) != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported artifact schema_version: {data['schema_version']}")
        provenance = data.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance must be a TOML table")
        metadata = cls.create(
            kind=str(data["kind"]),
            title=str(data["title"]),
            scope_id=str(data["scope_id"]),
            project_id=str(data["project_id"]),
            task_id=str(data["task_id"]),
            agent_name=str(data["agent_name"]),
            author_kind=str(data["author_kind"]),
            sensitivity=str(data["sensitivity"]),
            provenance=dict(provenance),
            artifact_id=str(data["artifact_id"]),
            created_at=str(data["created_at"]),
        )
        return metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "title": self.title,
            "created_at": self.created_at,
            "scope_id": self.scope_id,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "author_kind": self.author_kind,
            "sensitivity": self.sensitivity,
            "provenance": dict(self.provenance or {}),
        }


@dataclass(frozen=True)
class MarkdownArtifact:
    metadata: ArtifactMetadata
    body: str
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "body": self.body,
            "path": str(self.path),
        }


class MarkdownArtifactCodec:
    """Create and read immutable Markdown artifacts beneath one root."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    @staticmethod
    def filename(metadata: ArtifactMetadata) -> str:
        created = datetime.fromisoformat(metadata.created_at.replace("Z", "+00:00"))
        timestamp = created.astimezone(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        return f"{timestamp}--{_slugify(metadata.title)}--{metadata.artifact_id[:10]}.md"

    def create(
        self,
        *,
        kind: str,
        title: str,
        body: str,
        scope_id: str,
        project_id: str = "",
        task_id: str = "",
        agent_name: str = "",
        author_kind: str = "agent",
        sensitivity: str = "internal",
        provenance: Mapping[str, Any] | None = None,
        artifact_id: str | None = None,
        created_at: str | datetime | None = None,
        relative_dir: str | None = None,
    ) -> MarkdownArtifact:
        if not isinstance(body, str):
            raise TypeError("body must be a string")
        if "\x00" in body:
            raise ValueError("body cannot contain NUL characters")

        directory = self.root
        if relative_dir is not None:
            directory = self.root / _validate_safe_segment(relative_dir, field_name="relative_dir")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory = directory.resolve()
        try:
            directory.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("relative_dir resolves outside the artifact root") from exc

        explicit_id = artifact_id is not None
        attempts = 1 if explicit_id else 16
        for _ in range(attempts):
            metadata = ArtifactMetadata.create(
                kind=kind,
                title=title,
                scope_id=scope_id,
                project_id=project_id,
                task_id=task_id,
                agent_name=agent_name,
                author_kind=author_kind,
                sensitivity=sensitivity,
                provenance=provenance,
                artifact_id=artifact_id,
                created_at=created_at,
            )
            try:
                claim_path = self._claim_artifact_id(metadata.artifact_id)
            except FileExistsError:
                if explicit_id:
                    raise
                artifact_id = None
                continue
            canonical_body = body.rstrip("\n") + "\n"
            rendered = (
                f"{_FRONTMATTER_DELIMITER}\n"
                f"{tomlkit.dumps(metadata.to_dict()).rstrip()}\n"
                f"{_FRONTMATTER_DELIMITER}\n\n"
                f"{canonical_body}"
            )
            path = directory / self.filename(metadata)
            try:
                self._write_exclusive(path, rendered)
            except FileExistsError:
                claim_path.unlink(missing_ok=True)
                if explicit_id:
                    raise
                artifact_id = None
                continue
            except BaseException:
                claim_path.unlink(missing_ok=True)
                raise
            return MarkdownArtifact(metadata=metadata, body=canonical_body, path=path)
        raise FileExistsError("could not allocate a unique artifact filename")

    def read(self, path: Path) -> MarkdownArtifact:
        resolved = path.expanduser().resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("artifact path is outside the configured root") from exc
        payload = resolved.read_text(encoding="utf-8")
        lines = payload.splitlines(keepends=True)
        if not lines or lines[0].strip() != _FRONTMATTER_DELIMITER:
            raise ValueError("artifact is missing TOML front matter")
        closing = next(
            (
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.strip() == _FRONTMATTER_DELIMITER
            ),
            None,
        )
        if closing is None:
            raise ValueError("artifact TOML front matter is not terminated")
        frontmatter = "".join(lines[1:closing])
        try:
            parsed = tomllib.loads(frontmatter)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError("artifact TOML front matter is invalid") from exc
        metadata = ArtifactMetadata.from_dict(parsed)
        if resolved.name != self.filename(metadata):
            raise ValueError("artifact filename does not match its metadata")
        body = "".join(lines[closing + 1 :])
        if body.startswith("\n"):
            body = body[1:]
        return MarkdownArtifact(metadata=metadata, body=body, path=resolved)

    def iter_artifacts(self, *, kind: str | None = None) -> Iterator[MarkdownArtifact]:
        for path in sorted(self.root.rglob("*.md")):
            try:
                artifact = self.read(path)
            except (OSError, ValueError):
                continue
            if kind is None or artifact.metadata.kind == kind:
                yield artifact

    @staticmethod
    def _write_exclusive(path: Path, payload: str) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            encoded = payload.encode("utf-8")
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short write while creating artifact")
                view = view[written:]
            os.fsync(fd)
        except BaseException:
            os.close(fd)
            path.unlink(missing_ok=True)
            raise
        else:
            os.close(fd)

    def _claim_artifact_id(self, artifact_id: str) -> Path:
        claim_root = self.root / ".artifact_ids"
        claim_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        claim_root = claim_root.resolve()
        try:
            claim_root.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("artifact id registry resolves outside the artifact root") from exc
        claim_path = claim_root / artifact_id
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(claim_path, flags, 0o600)
        os.close(fd)
        return claim_path


class NoteStore:
    """Create durable scoped notes using the common artifact contract."""

    def __init__(
        self,
        context_path: Path,
        *,
        scope_id: str | None = None,
        config: Any = None,
        root_resolver: ArtifactRootResolver | None = None,
    ) -> None:
        self.context_path = context_path.expanduser().resolve()
        self.scope_id = validate_scope_id(scope_id or infer_scope_id(self.context_path))
        resolver = root_resolver or default_artifact_root
        self.root = (
            resolver(
                self.context_path,
                scope_id=self.scope_id,
                collection="notes",
                config=config,
            )
            .expanduser()
            .resolve()
        )
        self._codec = MarkdownArtifactCodec(self.root)

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
        return self._codec.create(
            kind="note",
            title=title,
            body=body,
            scope_id=self.scope_id,
            project_id=project_id,
            task_id=task_id,
            agent_name=agent_name,
            author_kind=author_kind,
            sensitivity=sensitivity,
            provenance=provenance or {"source": "afs.note"},
        )

    def list(self, *, limit: int = 100) -> list[MarkdownArtifact]:
        """List the newest notes in this scope."""

        if limit <= 0:
            return []
        notes = list(self._codec.iter_artifacts(kind="note"))
        notes.sort(
            key=lambda artifact: (
                artifact.metadata.created_at,
                artifact.metadata.artifact_id,
            ),
            reverse=True,
        )
        return notes[:limit]

    def read(self, identifier: str | Path) -> MarkdownArtifact | None:
        """Read a note by full artifact id or by a contained path."""

        raw = str(identifier).strip()
        if not raw:
            return None
        try:
            artifact_id = _normalize_artifact_id(raw)
        except ValueError:
            artifact_id = ""
        if artifact_id:
            for artifact in self._codec.iter_artifacts(kind="note"):
                if artifact.metadata.artifact_id == artifact_id:
                    return artifact
            return None

        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            artifact = self._codec.read(candidate)
        except (OSError, ValueError):
            return None
        return artifact if artifact.metadata.kind == "note" else None


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactMetadata",
    "ArtifactRootResolver",
    "MarkdownArtifact",
    "MarkdownArtifactCodec",
    "NoteStore",
    "default_artifact_root",
    "infer_scope_id",
    "validate_scope_id",
]
