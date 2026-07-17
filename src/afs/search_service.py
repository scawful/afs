"""Reusable project-scoped hybrid search service.

The CLI is only one consumer of scoped retrieval.  This module owns source
assembly, index readiness, semantic opt-in, and export filtering so agents and
other callers cannot accidentally bypass the same project and sensitivity
boundaries enforced by ``afs search``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context_layout import LAYOUT_VERSION, detect_layout_version, resolve_system_path
from .embeddings import DEFAULT_GEMINI_MODEL
from .hybrid_search import (
    HybridBuildResult,
    HybridSearchEngine,
    HybridSearchResponse,
    HybridSource,
    SourcePolicy,
    hybrid_hit_blocked,
    source_policy_digest,
)
from .models import ContextCategory
from .path_safety import assert_no_linklike_components
from .project_registry import COMMON_SCOPE_ID, ProjectRegistry
from .schema import AFSConfig


@dataclass(frozen=True)
class ScopedSearchRequest:
    """One bounded retrieval request against a project and its visible context."""

    context_root: Path
    project_path: Path
    query: str
    mode: str = "text"
    limit: int = 10
    rebuild: bool = False
    semantic: bool = False
    embedding_provider: str = "gemini"
    embedding_model: str | None = None
    all_projects: bool = False
    excluded_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        query = self.query.strip()
        if not query:
            raise ValueError("search query cannot be empty")
        if self.mode not in {"text", "symbol"}:
            raise ValueError("search mode must be 'text' or 'symbol'")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("search limit must be an integer")
        if not 1 <= self.limit <= 100:
            raise ValueError("search limit must be between 1 and 100")
        for field_name in ("rebuild", "semantic", "all_projects"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"search {field_name} must be a boolean")
        provider = self.embedding_provider.strip()
        if not provider:
            raise ValueError("embedding provider cannot be empty")
        model = self.embedding_model.strip() if self.embedding_model else None
        excluded_paths = tuple(
            path.expanduser().resolve() for path in self.excluded_paths
        )
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "embedding_provider", provider)
        object.__setattr__(self, "embedding_model", model)
        object.__setattr__(self, "excluded_paths", excluded_paths)


@dataclass(frozen=True)
class ScopedSearchSources:
    """Authorized sources and identity resolved for a search request."""

    sources: tuple[HybridSource, ...]
    scope_id: str
    project_id: str


@dataclass
class ScopedSearchResult:
    """Search response plus index and scope provenance."""

    response: HybridSearchResponse
    context_root: Path
    project_path: Path
    scope_id: str
    project_id: str
    index_root: Path
    semantic_requested: bool
    embedding_provider: str
    embedding_model: str
    build: HybridBuildResult | None = None

    @property
    def rebuilt(self) -> bool:
        return self.build is not None

    def to_dict(self) -> dict[str, Any]:
        """Return the compatibility JSON shape emitted by ``afs search``."""

        payload = self.response.to_dict()
        payload.update(
            {
                "context_root": str(self.context_root),
                "project_path": str(self.project_path),
                "project_id": self.project_id,
                "index_root": str(self.index_root),
                "semantic_requested": self.semantic_requested,
                "embedding_provider": (
                    self.embedding_provider if self.semantic_requested else ""
                ),
                "embedding_model": (
                    self.embedding_model if self.semantic_requested else ""
                ),
                "rebuilt": self.rebuilt,
                "build": _build_payload(self.build),
            }
        )
        return payload


def build_scoped_sources(
    context_root: Path,
    project_path: Path,
    *,
    semantic: bool,
    all_projects: bool,
    previously_consented_scopes: frozenset[str] = frozenset(),
    never_index: tuple[str, ...] = (),
    never_embed: tuple[str, ...] = (),
    excluded_paths: tuple[Path, ...] = (),
) -> ScopedSearchSources:
    """Assemble only sources authorized for the requesting project.

    Version 2 project access comes exclusively from the central registry.
    Registered child projects are excluded from parent codebase traversal, and
    semantic transmission remains disabled unless the request opted in.
    """

    context = context_root.expanduser().resolve()
    project = project_path.expanduser().resolve()

    def policy(*prefixes: str) -> SourcePolicy:
        return SourcePolicy(
            never_index=never_index,
            never_embed=never_embed,
            sensitivity_prefixes=tuple(prefixes),
        )

    sources: list[HybridSource] = []
    current_scope = COMMON_SCOPE_ID
    current_project_id = ""
    if detect_layout_version(context) == LAYOUT_VERSION:
        registry = ProjectRegistry(context)
        current = registry.resolve(project)
        if current is None:
            raise PermissionError(f"project is not registered in central context: {project}")
        current_scope = current.scope_id
        current_project_id = current.project_id
        records = registry.all_records() if all_projects else [current]
        for record in records:
            active = record.project_id == current.project_id
            source_path = Path(record.path).expanduser().resolve()
            excluded_roots = registry.codebase_excluded_roots(
                source_path,
                project_id=record.project_id,
            )
            embed_allowed = semantic and (
                active or all_projects or record.scope_id in previously_consented_scopes
            )
            sources.append(
                HybridSource(
                    source_path,
                    scope_id=record.scope_id,
                    project_id=record.project_id,
                    project_terms=(record.name,),
                    active=active,
                    excluded_roots=(*excluded_roots, *excluded_paths),
                    policy=policy(),
                    embed_allowed=embed_allowed,
                )
            )
            for category in ContextCategory:
                scoped_root = context / category.value / "projects" / record.project_id
                try:
                    scoped_root = assert_no_linklike_components(
                        scoped_root,
                        boundary=context,
                    )
                except ValueError:
                    if active or all_projects:
                        raise
                    continue
                if scoped_root.is_dir():
                    sources.append(
                        HybridSource(
                            scoped_root,
                            scope_id=record.scope_id,
                            project_id=record.project_id,
                            project_terms=(record.name, category.value),
                            active=active,
                            excluded_roots=excluded_paths,
                            policy=policy(
                                category.value,
                                f"{category.value}/projects/{record.project_id}",
                            ),
                            embed_allowed=embed_allowed,
                        )
                    )
        for category in ContextCategory:
            common_root = assert_no_linklike_components(
                context / category.value / "common",
                boundary=context,
            )
            if common_root.is_dir():
                sources.append(
                    HybridSource(
                        common_root,
                        scope_id=COMMON_SCOPE_ID,
                        project_terms=("common", category.value),
                        embed_allowed=semantic,
                        excluded_roots=excluded_paths,
                        policy=policy(
                            category.value,
                            f"{category.value}/common",
                        ),
                    )
                )
    else:
        sources.append(
            HybridSource(
                project,
                scope_id=COMMON_SCOPE_ID,
                project_terms=(project.name,),
                embed_allowed=semantic,
                excluded_roots=excluded_paths,
                policy=policy(),
            )
        )
        for category in ContextCategory:
            category_root = context / category.value
            if category_root.is_dir():
                sources.append(
                    HybridSource(
                        category_root,
                        scope_id=COMMON_SCOPE_ID,
                        project_terms=(category.value,),
                        embed_allowed=semantic,
                        excluded_roots=excluded_paths,
                        policy=policy(category.value),
                    )
                )

    return ScopedSearchSources(
        sources=tuple(sources),
        scope_id=current_scope,
        project_id=current_project_id,
    )


def search_scoped(
    request: ScopedSearchRequest,
    *,
    config: AFSConfig,
) -> ScopedSearchResult:
    """Build or reuse the scoped index and execute one local-first search."""

    context = request.context_root.expanduser().resolve()
    project = request.project_path.expanduser().resolve()
    index_root = resolve_system_path(context, "search")
    engine = HybridSearchEngine(index_root)
    has_index = engine.current_path.is_file() or engine.metadata_path.is_file()
    requested_model = request.embedding_model or (
        DEFAULT_GEMINI_MODEL if request.embedding_provider == "gemini" else "nomic-embed-text"
    )

    previous_metadata = _read_metadata(engine) if has_index else {}
    previously_consented_scopes: frozenset[str] = frozenset()
    collection = previous_metadata.get("collection", {})
    if (
        request.semantic
        and isinstance(collection, dict)
        and collection.get("provider") == request.embedding_provider
        and collection.get("model") == requested_model
    ):
        previously_consented_scopes = _scope_coverage(
            previous_metadata,
            "intended_scope_ids",
        )

    scoped = build_scoped_sources(
        context,
        project,
        semantic=request.semantic,
        all_projects=request.all_projects,
        previously_consented_scopes=previously_consented_scopes,
        never_index=tuple(config.sensitivity.never_index),
        never_embed=(
            *config.sensitivity.never_embed,
            *config.sensitivity.never_export,
        ),
        excluded_paths=request.excluded_paths,
    )
    required_semantic_scopes = _required_semantic_scopes(
        scoped.sources,
        current_scope=scoped.scope_id,
        all_projects=request.all_projects,
    )
    policy_index_ready = bool(previous_metadata) and previous_metadata.get(
        "sensitivity_policy_digest"
    ) == source_policy_digest(scoped.sources)
    semantic_index_ready = _semantic_index_ready(
        previous_metadata,
        requested=request.semantic and has_index,
        provider=request.embedding_provider,
        model=requested_model,
        required_scopes=required_semantic_scopes,
    )

    # This is the network/privacy boundary: no embedding backend is even
    # constructed unless the caller explicitly requested semantic retrieval.
    embed_fn = None
    if request.semantic:
        # Resolve the factory at the semantic boundary instead of module import
        # time.  Besides making the opt-in explicit, this keeps provider
        # selection patchable by library consumers and tests.
        from .embeddings import create_embed_fn

        embed_fn = create_embed_fn(
            request.embedding_provider,
            model=requested_model,
        )
    build: HybridBuildResult | None = None
    if (
        request.rebuild
        or not has_index
        or not policy_index_ready
        or (request.semantic and not semantic_index_ready)
    ):
        build = engine.build(scoped.sources, embed_fn=embed_fn)

    response = engine.search(
        request.query,
        scope_ids=(scoped.scope_id,) if scoped.scope_id != COMMON_SCOPE_ID else (),
        include_common=True,
        all_projects=request.all_projects,
        mode="hybrid" if request.semantic else request.mode,
        top_k=request.limit,
        recreate_query_embedder=request.semantic,
        required_scope_ids=sorted(
            {source.scope_id for source in scoped.sources if source.path.exists()}
        ),
    )
    export_patterns = [
        *config.sensitivity.never_index,
        *config.sensitivity.never_export,
    ]
    response.results = [
        hit
        for hit in response.results
        if not hybrid_hit_blocked(
            hit,
            context_root=context,
            patterns=export_patterns,
        )
    ]
    return ScopedSearchResult(
        response=response,
        context_root=context,
        project_path=project,
        scope_id=scoped.scope_id,
        project_id=scoped.project_id,
        index_root=index_root,
        semantic_requested=request.semantic,
        embedding_provider=request.embedding_provider,
        embedding_model=requested_model,
        build=build,
    )


def _read_metadata(engine: HybridSearchEngine) -> dict[str, Any]:
    try:
        payload = json.loads(engine.metadata_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _scope_coverage(metadata: dict[str, Any], field: str) -> frozenset[str]:
    coverage = metadata.get("scope_coverage", {})
    if not isinstance(coverage, dict):
        return frozenset()
    values = coverage.get(field, [])
    if not isinstance(values, list):
        return frozenset()
    return frozenset(value for value in values if isinstance(value, str) and value.strip())


def _required_semantic_scopes(
    sources: tuple[HybridSource, ...],
    *,
    current_scope: str,
    all_projects: bool,
) -> frozenset[str]:
    if all_projects:
        return frozenset(source.scope_id for source in sources if source.path.exists())
    required = {current_scope}
    if any(source.scope_id == COMMON_SCOPE_ID and source.path.exists() for source in sources):
        required.add(COMMON_SCOPE_ID)
    return frozenset(required)


def _semantic_index_ready(
    metadata: dict[str, Any],
    *,
    requested: bool,
    provider: str,
    model: str,
    required_scopes: frozenset[str],
) -> bool:
    if not requested:
        return False
    collection = metadata.get("collection", {})
    stats = metadata.get("stats", {})
    if not isinstance(collection, dict) or not isinstance(stats, dict):
        return False
    try:
        vector_count = int(stats.get("vectors", 0) or 0)
    except (TypeError, ValueError):
        return False
    return (
        collection.get("health") == "healthy"
        and collection.get("provider") == provider
        and collection.get("model") == model
        and vector_count > 0
        and required_scopes.issubset(_scope_coverage(metadata, "intended_scope_ids"))
    )


def _build_payload(build: HybridBuildResult | None) -> dict[str, Any] | None:
    if build is None:
        return None
    return {
        "total_files": build.total_files,
        "indexed_files": build.indexed_files,
        "chunks_written": build.chunks_written,
        "skipped": build.skipped,
        "vector_count": build.vector_count,
        "vector_dimension": build.vector_dimension,
        "semantic_status": build.semantic_status,
        "source_scope_ids": build.source_scope_ids,
        "intended_scope_ids": build.intended_scope_ids,
        "embedded_scope_ids": build.embedded_scope_ids,
        "capped_sources": build.capped_sources,
        "denied": build.denied,
        "errors": build.errors,
    }


__all__ = [
    "ScopedSearchRequest",
    "ScopedSearchResult",
    "ScopedSearchSources",
    "build_scoped_sources",
    "search_scoped",
]
