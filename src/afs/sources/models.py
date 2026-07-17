"""Provider-neutral context source adapter models.

Core AFS owns the record/schema and filesystem landing zone. Concrete source
connectors live in extensions and can register providers through an extension
manifest without baking a vendor adapter into core.
"""

from __future__ import annotations

import ipaddress
import json
import math
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

MAX_RESEARCH_RESULTS = 50
MAX_RESEARCH_TIMEOUT_SECONDS = 120.0
MAX_RESEARCH_BYTES = 10 * 1024 * 1024
MAX_RESEARCH_QUERY_CHARS = 2_000
_RESEARCH_DOMAIN = re.compile(
    r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z"
)
_NON_PUBLIC_RESEARCH_SUFFIXES = frozenset(
    {
        "home",
        "internal",
        "invalid",
        "lan",
        "local",
        "localdomain",
        "localhost",
        "test",
    }
)


def _is_public_research_domain(domain: str) -> bool:
    """Reject literal/private-style hosts before a provider receives consent.

    This is deliberately a syntax-level guard. Selected providers remain
    responsible for resolving DNS and rejecting private destinations after
    redirects or rebinding.
    """

    try:
        ipaddress.ip_address(domain)
    except ValueError:
        pass
    else:
        return False
    labels = domain.split(".")
    return not (
        all(label.isdigit() for label in labels)
        or labels[-1] in _NON_PUBLIC_RESEARCH_SUFFIXES
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
    def from_dict(cls, payload: dict[str, Any]) -> ContextSourceRecord:
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


@dataclass(frozen=True)
class ResearchRequest:
    """Bounded request passed to an extension-owned research provider.

    Core AFS does not silently enable network access. A caller must opt in and
    name at least one allowed domain before a provider may browse. Providers
    remain responsible for enforcing transport-level timeout, redirect, and
    private-address protections; the bounds here form the portable contract.
    """

    query: str
    network_allowed: bool = False
    allowed_domains: tuple[str, ...] = ()
    max_results: int = 10
    timeout_seconds: float = 20.0
    max_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        query = self.query.strip()
        if not query:
            raise ValueError("research query must be non-empty")
        if len(query) > MAX_RESEARCH_QUERY_CHARS:
            raise ValueError(
                f"research query must be no more than {MAX_RESEARCH_QUERY_CHARS} characters"
            )
        if not isinstance(self.network_allowed, bool):
            raise ValueError("research network_allowed must be a boolean")
        if (
            not isinstance(self.max_results, int)
            or isinstance(self.max_results, bool)
            or not 1 <= self.max_results <= MAX_RESEARCH_RESULTS
        ):
            raise ValueError(
                f"research max_results must be between 1 and {MAX_RESEARCH_RESULTS}"
            )
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= MAX_RESEARCH_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "research timeout_seconds must be greater than 0 and no more than "
                f"{MAX_RESEARCH_TIMEOUT_SECONDS:g}"
            )
        if (
            not isinstance(self.max_bytes, int)
            or isinstance(self.max_bytes, bool)
            or not 1 <= self.max_bytes <= MAX_RESEARCH_BYTES
        ):
            raise ValueError(
                f"research max_bytes must be between 1 and {MAX_RESEARCH_BYTES}"
            )

        domains: list[str] = []
        for raw_domain in self.allowed_domains:
            if not isinstance(raw_domain, str):
                raise ValueError("research allowed_domains must contain host names")
            domain = raw_domain.strip().lower().rstrip(".")
            if (
                not domain
                or "://" in domain
                or "/" in domain
                or domain.startswith(".")
                or "*" in domain
                or not _RESEARCH_DOMAIN.fullmatch(domain)
                or not _is_public_research_domain(domain)
            ):
                raise ValueError(
                    "research allowed_domains must contain public plain host names "
                    "without IP literals, private-style suffixes, schemes, paths, "
                    "or wildcards"
                )
            if domain not in domains:
                domains.append(domain)
        if self.network_allowed and not domains:
            raise ValueError(
                "network research requires at least one explicitly allowed domain"
            )

        object.__setattr__(self, "query", query)
        object.__setattr__(self, "allowed_domains", tuple(domains))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "network_allowed": self.network_allowed,
            "allowed_domains": list(self.allowed_domains),
            "max_results": self.max_results,
            "timeout_seconds": self.timeout_seconds,
            "max_bytes": self.max_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResearchRequest:
        allowed = {
            "query",
            "network_allowed",
            "allowed_domains",
            "max_results",
            "timeout_seconds",
            "max_bytes",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(
                "research request contains unknown fields: " + ", ".join(unknown)
            )
        domains = payload.get("allowed_domains", [])
        if not isinstance(domains, list):
            raise ValueError("research allowed_domains must be an array")
        query = payload.get("query")
        if not isinstance(query, str):
            raise ValueError("research query must be a string")
        return cls(
            query=query,
            network_allowed=payload.get("network_allowed", False),
            allowed_domains=tuple(domains),
            max_results=payload.get("max_results", 10),
            timeout_seconds=payload.get("timeout_seconds", 20.0),
            max_bytes=payload.get("max_bytes", 1_000_000),
        )


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


@runtime_checkable
class ResearchSourceProvider(Protocol):
    """Opt-in bounded research interface implemented by extensions.

    Merely discovering a provider must not perform network I/O. AFS passes a
    ``ResearchRequest`` only after an explicit caller grants network access
    and domain scope.
    """

    name: str

    def research(
        self,
        request: ResearchRequest,
    ) -> list[ContextSourceRecord] | list[dict[str, Any]]:
        """Return normalized evidence records within the request bounds."""
        ...
