"""Provider-neutral context source adapter models.

Core AFS owns the record/schema and filesystem landing zone. Concrete source
connectors live in extensions and can register providers through an extension
manifest without baking a vendor adapter into core.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

CONTEXT_SOURCE_KINDS = (
    "task",
    "ticket",
    "review",
    "doc",
    "message",
    "test",
    "hook",
    "trace",
)


def normalize_source_kind(value: str) -> str:
    """Return a stable source kind, falling back to ``doc`` for unknown input."""
    normalized = (value or "doc").strip().lower().replace("_", "-")
    if normalized in CONTEXT_SOURCE_KINDS:
        return normalized
    return "doc"


def safe_source_id(value: str) -> str:
    """Return a filesystem-safe identifier for a source record."""
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", (value or "item").strip()).strip(".-_")
    return normalized[:120] or "item"


@dataclass(frozen=True)
class ContextSourceRecord:
    """Normalized record emitted by a context-source provider."""

    id: str
    kind: str
    title: str
    body: str = ""
    provider: str = ""
    source: str = ""
    uri: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", safe_source_id(self.id))
        object.__setattr__(self, "kind", normalize_source_kind(self.kind))
        object.__setattr__(self, "title", self.title.strip() or self.id)
        object.__setattr__(self, "provider", self.provider.strip())
        object.__setattr__(self, "source", self.source.strip())
        object.__setattr__(self, "uri", self.uri.strip())
        object.__setattr__(self, "created_at", self.created_at.strip())
        object.__setattr__(self, "updated_at", self.updated_at.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "body": self.body,
            "provider": self.provider,
            "source": self.source,
            "uri": self.uri,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContextSourceRecord":
        return cls(
            id=str(payload.get("id", "")).strip(),
            kind=str(payload.get("kind", "doc")).strip(),
            title=str(payload.get("title", "")).strip(),
            body=str(payload.get("body", "")).strip(),
            provider=str(payload.get("provider", "")).strip(),
            source=str(payload.get("source", "")).strip(),
            uri=str(payload.get("uri", "")).strip(),
            created_at=str(payload.get("created_at", "")).strip(),
            updated_at=str(payload.get("updated_at", "")).strip(),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )

    def render_markdown(self) -> str:
        """Render this record into a context-indexable markdown document."""
        frontmatter = {
            key: value
            for key, value in self.to_dict().items()
            if key not in {"body", "metadata"} and value not in ("", None)
        }
        if self.metadata:
            frontmatter["metadata"] = dict(self.metadata)
        lines = ["---", json.dumps(frontmatter, sort_keys=True), "---", "", f"# {self.title}", ""]
        if self.body.strip():
            lines.append(self.body.strip())
            lines.append("")
        return "\n".join(lines)


@dataclass(frozen=True)
class SourceProviderSpec:
    """Manifest-declared context-source provider spec."""

    name: str
    module: str
    factory: str = "register_context_source_provider"
    description: str = ""
    kinds: tuple[str, ...] = ()
    extension: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", safe_source_id(self.name))
        object.__setattr__(self, "module", self.module.strip())
        object.__setattr__(self, "factory", (self.factory or "register_context_source_provider").strip())
        object.__setattr__(
            self,
            "kinds",
            tuple(normalize_source_kind(kind) for kind in self.kinds if str(kind).strip()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "factory": self.factory,
            "description": self.description,
            "kinds": list(self.kinds),
            "extension": self.extension,
        }


@dataclass(frozen=True)
class SourceSyncResult:
    provider: str
    dry_run: bool
    target_dir: Path
    records: tuple[ContextSourceRecord, ...]
    written_paths: tuple[Path, ...] = ()
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "dry_run": self.dry_run,
            "target_dir": str(self.target_dir),
            "record_count": len(self.records),
            "records": [record.to_dict() for record in self.records],
            "written_paths": [str(path) for path in self.written_paths],
            "status": self.status,
        }


@runtime_checkable
class ContextSourceProvider(Protocol):
    """Minimal provider interface implemented by extension adapters."""

    name: str
    kinds: tuple[str, ...]

    def status(self) -> dict[str, Any]:
        """Return provider health/capability state."""
        ...

    def sync(self, *, query: str = "", limit: int = 50) -> list[ContextSourceRecord] | list[dict[str, Any]]:
        """Return records to materialize into AFS context items."""
        ...
