"""Project-scoped hybrid retrieval for the AFS v2 context namespace.

The filesystem remains authoritative.  This module owns only rebuildable
SQLite FTS metadata and a compact normalized float32 vector collection.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import (
    DEFAULT_GEMINI_DIMENSION,
    GEMINI_QUERY_TASK,
    EmbeddingCollectionMetadata,
    create_embed_fn,
)
from .index_storage import (
    atomic_write_text,
    fsync_directory,
    fsync_file,
    index_file_lock,
    read_generation_id,
)
from .path_safety import assert_no_linklike_components, lexical_absolute
from .sensitivity import SensitivityRuleSet

HYBRID_INDEX_VERSION = 1
RRF_K = 60
DATABASE_FILENAME = "search.sqlite3"
VECTOR_FILENAME = "vectors.npy"
METADATA_FILENAME = "hybrid_index.json"
CURRENT_FILENAME = "CURRENT"
GENERATIONS_DIRNAME = "generations"
BUILD_LOCK_FILENAME = ".hybrid-build.lock"
PUBLICATION_LOCK_FILENAME = ".hybrid-publication.lock"

_HARD_DENY_DIRS = frozenset(
    {
        ".git",
        ".afs",
        ".hg",
        ".svn",
        ".cache",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
        "vendor",
        "build",
        "dist",
        "target",
    }
)
_HARD_DENY_GLOBS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.crt",
    "*.cer",
    "id_rsa*",
    "id_ed25519*",
    "credentials*.json",
    "service-account*.json",
    "*.sqlite-wal",
    "*.sqlite-shm",
)
_SECRET_CONTENT = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"^\s*(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[^\s#]{8,}",
    re.IGNORECASE | re.MULTILINE,
)
_WORD = re.compile(r"[A-Za-z0-9_./:-]+")
_SYMBOL = re.compile(
    r"(?m)^(?:\s*)(?:class|def|enum|struct|func|function|fn|interface|trait)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)|^#{1,6}\s+(.+?)\s*$"
)


@dataclass(frozen=True)
class SourcePolicy:
    """Local indexing and remote-embedding safety policy."""

    max_bytes: int = 1_048_576
    max_files_active: int = 10_000
    max_files_inactive: int = 5_000
    max_total_bytes_active: int = 100 * 1_048_576
    max_total_bytes_inactive: int = 50 * 1_048_576
    include_globs: tuple[str, ...] = ()
    exclude_globs: tuple[str, ...] = ()
    respect_gitignore: bool = True
    allow_hidden: bool = False
    never_index: tuple[str, ...] = ()
    never_embed: tuple[str, ...] = ()
    sensitivity_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class HybridSource:
    """A registered source and the scope applied before retrieval ranking."""

    path: Path
    scope_id: str
    project_id: str | None = None
    project_terms: tuple[str, ...] = ()
    embed_allowed: bool = False
    active: bool = True
    max_files: int | None = None
    excluded_roots: tuple[Path, ...] = ()
    policy: SourcePolicy = field(default_factory=SourcePolicy)

    def __post_init__(self) -> None:
        if not self.scope_id.strip():
            raise ValueError("HybridSource.scope_id cannot be empty")
        if self.max_files is not None and self.max_files <= 0:
            raise ValueError("HybridSource.max_files must be positive")


@dataclass
class HybridBuildResult:
    total_files: int = 0
    indexed_files: int = 0
    chunks_written: int = 0
    skipped: int = 0
    vector_count: int = 0
    vector_dimension: int | None = None
    semantic_status: str = "disabled"
    source_scope_ids: list[str] = field(default_factory=list)
    intended_scope_ids: list[str] = field(default_factory=list)
    embedded_scope_ids: list[str] = field(default_factory=list)
    capped_sources: list[str] = field(default_factory=list)
    denied: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    sensitivity_policy_digest: str = ""

    @property
    def healthy(self) -> bool:
        return not self.errors and self.semantic_status != "unhealthy"


@dataclass
class HybridSearchHit:
    doc_id: str
    score: float
    source_path: str
    relative_path: str
    scope_id: str
    project_id: str | None
    text_preview: str
    chunk_index: int
    line_start: int
    line_end: int
    signals: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": self.score,
            "source_path": self.source_path,
            "relative_path": self.relative_path,
            "scope_id": self.scope_id,
            "project_id": self.project_id,
            "text_preview": self.text_preview,
            "chunk_index": self.chunk_index,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "signals": self.signals,
        }


@dataclass
class HybridSearchResponse:
    query: str
    mode: str
    results: list[HybridSearchHit] = field(default_factory=list)
    semantic_status: str = "not_requested"
    semantic_reason: str | None = None
    searched_scopes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "semantic_status": self.semantic_status,
            "semantic_reason": self.semantic_reason,
            "searched_scopes": self.searched_scopes,
            "results": [result.to_dict() for result in self.results],
        }


class HybridScopeCoverageError(ValueError):
    """The active immutable generation does not cover an authorized scope."""


@dataclass
class _DocumentRow:
    row_id: int
    doc_id: str
    source_path: str
    relative_path: str
    scope_id: str
    project_id: str | None
    title: str
    body: str
    symbols: str
    project_terms: str
    text_preview: str
    chunk_index: int
    line_start: int
    line_end: int
    vector_row: int | None


@dataclass
class _ScopeBudget:
    max_files: int
    max_bytes: int
    used_files: int = 0
    used_bytes: int = 0


class HybridSearchEngine:
    """Build and query a deterministic project-scoped hybrid index."""

    def __init__(self, index_root: Path) -> None:
        candidate = lexical_absolute(index_root)
        self._trusted_boundary: Path | None = None
        if candidate.name == "search" and candidate.parent.name == ".afs":
            self._trusted_boundary = candidate.parent.parent
            candidate = assert_no_linklike_components(
                candidate,
                boundary=self._trusted_boundary,
            )
        self.index_root = candidate

    def _safe_index_root(self) -> Path:
        if self._trusted_boundary is None:
            return self.index_root
        return assert_no_linklike_components(
            self.index_root,
            boundary=self._trusted_boundary,
        )

    def _safe_internal_path(
        self,
        path: Path,
        *,
        allow_missing: bool = True,
    ) -> Path:
        """Validate one managed v2 index path while preserving legacy roots."""

        if self._trusted_boundary is None:
            return path
        # Revalidate the search root on every operation.  Passing the search
        # root itself as the helper boundary would trust a root replaced after
        # engine construction, so first bind it to the context boundary.
        search_root = self._safe_index_root()
        return assert_no_linklike_components(
            path,
            boundary=search_root,
            allow_missing=allow_missing,
        )

    def _safe_database_bundle(self, database_path: Path) -> Path:
        """Validate SQLite's database leaf and predictable sidecar files."""

        safe_database = self._safe_internal_path(database_path)
        for suffix in ("-journal", "-shm", "-wal"):
            self._safe_internal_path(Path(f"{safe_database}{suffix}"))
        return safe_database

    @property
    def database_path(self) -> Path:
        return self._safe_database_bundle(
            self._active_generation_root() / DATABASE_FILENAME
        )

    @property
    def vector_path(self) -> Path:
        return self._safe_internal_path(
            self._active_generation_root() / VECTOR_FILENAME
        )

    @property
    def metadata_path(self) -> Path:
        return self._safe_internal_path(
            self._active_generation_root() / METADATA_FILENAME
        )

    @property
    def current_path(self) -> Path:
        return self._safe_internal_path(
            self._safe_index_root() / CURRENT_FILENAME
        )

    @property
    def generations_root(self) -> Path:
        return self._safe_internal_path(
            self._safe_index_root() / GENERATIONS_DIRNAME
        )

    @property
    def build_lock_path(self) -> Path:
        return self._safe_internal_path(
            self._safe_index_root() / BUILD_LOCK_FILENAME
        )

    @property
    def publication_lock_path(self) -> Path:
        return self._safe_internal_path(
            self._safe_index_root() / PUBLICATION_LOCK_FILENAME
        )

    @staticmethod
    def discover(root: Path) -> list[Path]:
        """Discover only documented v2 locations in stable order."""
        resolved = root.expanduser().resolve()
        candidates = [resolved, resolved / ".afs" / "search"]
        found: dict[str, Path] = {}
        for candidate in candidates:
            if not (
                (candidate / CURRENT_FILENAME).is_file()
                or (candidate / METADATA_FILENAME).is_file()
            ):
                continue
            engine = HybridSearchEngine(candidate)
            try:
                with index_file_lock(engine.publication_lock_path, shared=True):
                    if engine.metadata_path.is_file() and engine.database_path.is_file():
                        found[str(candidate)] = candidate
            except (OSError, ValueError):
                continue
        return [found[key] for key in sorted(found)]

    def build(
        self,
        sources: Iterable[HybridSource],
        *,
        embed_fn: Callable[[str], list[float]] | None = None,
        collection: EmbeddingCollectionMetadata | None = None,
        chunk_tokens: int = 800,
        chunk_overlap: int = 120,
    ) -> HybridBuildResult:
        """Serialize builders while readers continue using the current generation."""
        self._safe_index_root().mkdir(parents=True, exist_ok=True)
        self._safe_index_root()
        with index_file_lock(self.build_lock_path):
            return self._build_locked(
                sources,
                embed_fn=embed_fn,
                collection=collection,
                chunk_tokens=chunk_tokens,
                chunk_overlap=chunk_overlap,
            )

    def _build_locked(
        self,
        sources: Iterable[HybridSource],
        *,
        embed_fn: Callable[[str], list[float]] | None = None,
        collection: EmbeddingCollectionMetadata | None = None,
        chunk_tokens: int = 800,
        chunk_overlap: int = 120,
    ) -> HybridBuildResult:
        """Rebuild the index from registered sources.

        ``embed_allowed`` must be true on a source before any text from it is
        passed to ``embed_fn``. Keyword indexing remains local.
        """
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_tokens:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_tokens")

        self._safe_index_root().mkdir(parents=True, exist_ok=True)
        self._safe_index_root()
        result = HybridBuildResult()
        documents: list[_DocumentRow] = []
        vectors: list[np.ndarray] = []
        expected_dimension = _embedding_dimension(embed_fn, collection)
        semantic_attempts = 0
        semantic_failures = 0
        embedded_scope_ids: set[str] = set()
        next_row_id = 1
        budgets: dict[str, _ScopeBudget] = {}
        registered_sources = sorted(sources, key=lambda item: (item.scope_id, str(item.path)))
        result.sensitivity_policy_digest = source_policy_digest(registered_sources)
        for source in registered_sources:
            _scope_budget(source, budgets)

        for source in registered_sources:
            budget = budgets[source.project_id or source.scope_id]
            for root, path, relative_path in _iter_safe_source_files(
                source,
                result,
                budget=budget,
                excluded_root=self._safe_index_root(),
            ):
                result.total_files += 1
                text = _safe_read_text(path, source.policy, result)
                if text is None:
                    result.skipped += 1
                    continue
                if _SECRET_CONTENT.search(text):
                    result.skipped += 1
                    result.denied.append(f"{path}: secret-like content denied")
                    continue

                chunks = _chunk_words(text, chunk_tokens, chunk_overlap)
                if not chunks:
                    result.skipped += 1
                    continue
                title = _title_for(relative_path, text)
                symbols = " ".join(_extract_symbols(text))
                project_terms = " ".join(
                    _dedupe(
                        [
                            source.project_id or "",
                            *(source.project_terms or ()),
                            *Path(relative_path).parts,
                        ]
                    )
                )
                file_had_row = False
                embed_path_allowed = not _sensitivity_blocked(
                    path,
                    relative_path,
                    patterns=source.policy.never_embed,
                    prefixes=source.policy.sensitivity_prefixes,
                )
                for chunk_index, (line_start, line_end, chunk_text) in enumerate(chunks):
                    vector_row: int | None = None
                    if source.embed_allowed and embed_path_allowed and embed_fn is not None:
                        semantic_attempts += 1
                        try:
                            raw_vector = embed_fn(chunk_text)
                            vector = _normalize_vector(raw_vector)
                        except Exception as exc:  # noqa: BLE001 - extension callback boundary
                            result.errors.append(
                                f"{path} chunk {chunk_index}: embed failed ({exc})"
                            )
                            semantic_failures += 1
                            vector = None
                        if vector is None:
                            if not result.errors or "embed failed" not in result.errors[-1]:
                                result.errors.append(
                                    f"{path} chunk {chunk_index}: embedder returned an empty vector"
                                )
                                semantic_failures += 1
                        elif expected_dimension is None:
                            expected_dimension = int(vector.shape[0])
                        elif int(vector.shape[0]) != expected_dimension:
                            result.errors.append(
                                f"{path} chunk {chunk_index}: embedding dimension "
                                f"{vector.shape[0]} != collection dimension {expected_dimension}"
                            )
                            semantic_failures += 1
                            vector = None
                        if vector is not None:
                            vector_row = len(vectors)
                            vectors.append(vector)
                            embedded_scope_ids.add(source.scope_id)

                    doc_id = _document_id(source, root, relative_path, chunk_index)
                    documents.append(
                        _DocumentRow(
                            row_id=next_row_id,
                            doc_id=doc_id,
                            source_path=str(path),
                            relative_path=relative_path,
                            scope_id=source.scope_id,
                            project_id=source.project_id,
                            title=title,
                            body=chunk_text,
                            symbols=symbols,
                            project_terms=project_terms,
                            text_preview=chunk_text[:1000].strip(),
                            chunk_index=chunk_index,
                            line_start=line_start,
                            line_end=line_end,
                            vector_row=vector_row,
                        )
                    )
                    next_row_id += 1
                    result.chunks_written += 1
                    file_had_row = True
                if file_had_row:
                    result.indexed_files += 1

        if embed_fn is None or semantic_attempts == 0:
            semantic_status = "disabled"
        elif semantic_failures:
            semantic_status = "unhealthy"
        else:
            semantic_status = "healthy"
        result.semantic_status = semantic_status
        result.vector_count = len(vectors)
        result.vector_dimension = expected_dimension
        result.source_scope_ids = sorted({source.scope_id for source in registered_sources})
        result.intended_scope_ids = sorted(
            {
                source.scope_id
                for source in registered_sources
                if embed_fn is not None and source.embed_allowed
            }
        )
        result.embedded_scope_ids = sorted(embedded_scope_ids)

        resolved_collection = _collection_metadata(
            embed_fn,
            collection,
            dimension=expected_dimension,
            health=semantic_status,
        )
        self._write_index(documents, vectors, resolved_collection, result)
        return result

    def search(
        self,
        query: str,
        *,
        scope_ids: Sequence[str] = (),
        include_common: bool = True,
        common_scope: str = "common",
        all_projects: bool = False,
        mode: str = "hybrid",
        top_k: int = 10,
        query_embed_fn: Callable[[str], list[float]] | None = None,
        recreate_query_embedder: bool = False,
        required_scope_ids: Sequence[str] = (),
    ) -> HybridSearchResponse:
        """Search one immutable generation under a shared publication lock."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        with index_file_lock(self.publication_lock_path, shared=True):
            return self._search_locked(
                query,
                scope_ids=scope_ids,
                include_common=include_common,
                common_scope=common_scope,
                all_projects=all_projects,
                mode=mode,
                top_k=top_k,
                query_embed_fn=query_embed_fn,
                recreate_query_embedder=recreate_query_embedder,
                required_scope_ids=required_scope_ids,
            )

    def _search_locked(
        self,
        query: str,
        *,
        scope_ids: Sequence[str] = (),
        include_common: bool = True,
        common_scope: str = "common",
        all_projects: bool = False,
        mode: str = "hybrid",
        top_k: int = 10,
        query_embed_fn: Callable[[str], list[float]] | None = None,
        recreate_query_embedder: bool = False,
        required_scope_ids: Sequence[str] = (),
    ) -> HybridSearchResponse:
        """Search after applying scope constraints, then fuse ranked signals."""
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"hybrid", "text", "semantic", "symbol"}:
            raise ValueError("mode must be hybrid, text, semantic, or symbol")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        allowed_scopes = _resolve_scopes(
            scope_ids,
            include_common=include_common,
            common_scope=common_scope,
            all_projects=all_projects,
        )
        metadata = self._load_metadata()
        coverage = metadata.get("scope_coverage", {})
        source_scope_ids = {
            value
            for value in (
                coverage.get("source_scope_ids", [])
                if isinstance(coverage, dict)
                else []
            )
            if isinstance(value, str) and value.strip()
        }
        missing_scopes = sorted(
            {
                value
                for value in required_scope_ids
                if isinstance(value, str) and value.strip()
            }
            - source_scope_ids
        )
        if missing_scopes:
            missing = ", ".join(missing_scopes)
            raise HybridScopeCoverageError(
                "hybrid index does not cover requested scope(s): "
                f"{missing}; rebuild the index from the requested project"
            )
        collection = _collection_from_hybrid_metadata(metadata)

        database_path = self.database_path
        with sqlite3.connect(database_path) as connection:
            connection.row_factory = sqlite3.Row
            documents = _load_scoped_documents(connection, allowed_scopes)
            by_id = {document.doc_id: document for document in documents}
            ranks: dict[str, list[tuple[str, float]]] = {}

            semantic_status = "not_requested"
            semantic_reason: str | None = None
            semantic_requested = normalized_mode in {"hybrid", "semantic"}
            if semantic_requested:
                vector_rank, semantic_status, semantic_reason = self._semantic_rank(
                    query,
                    documents,
                    collection,
                    query_embed_fn=query_embed_fn,
                    recreate_query_embedder=recreate_query_embedder,
                )
                if vector_rank:
                    ranks["semantic"] = vector_rank

            use_text_fallback = normalized_mode == "semantic" and semantic_status != "ready"
            if normalized_mode in {"hybrid", "text"} or use_text_fallback:
                ranks["text"] = _fts_rank(connection, query, allowed_scopes)
                ranks.update(_association_ranks(query, documents, include_project=True))
            elif normalized_mode == "symbol":
                ranks.update(_association_ranks(query, documents, include_project=False))

        fused = _rrf_fuse(ranks, by_id, top_k=top_k)
        return HybridSearchResponse(
            query=query,
            mode=normalized_mode,
            results=fused,
            semantic_status=semantic_status,
            semantic_reason=semantic_reason,
            searched_scopes=[] if allowed_scopes is None else sorted(allowed_scopes),
        )

    def _semantic_rank(
        self,
        query: str,
        documents: list[_DocumentRow],
        collection: EmbeddingCollectionMetadata,
        *,
        query_embed_fn: Callable[[str], list[float]] | None,
        recreate_query_embedder: bool,
    ) -> tuple[list[tuple[str, float]], str, str | None]:
        if collection.health == "unhealthy":
            return [], "unhealthy", "embedding collection is unhealthy"
        if collection.health in {"disabled", "keyword_only"} or not collection.dimension:
            return [], "disabled", "embedding collection has no usable vectors"
        if query_embed_fn is None and recreate_query_embedder:
            try:
                query_embed_fn = _create_query_embedder(collection)
            except (RuntimeError, ValueError) as exc:
                return [], "fallback", f"query embedder unavailable: {exc}"
        if query_embed_fn is None:
            return [], "fallback", "query embedder was not provided"
        try:
            query_vector = _normalize_vector(query_embed_fn(query))
        except Exception as exc:  # noqa: BLE001 - extension callback boundary
            return [], "fallback", f"query embedding failed: {exc}"
        if query_vector is None:
            return [], "fallback", "query embedder returned an empty vector"
        if int(query_vector.shape[0]) != collection.dimension:
            return (
                [],
                "dimension_mismatch",
                f"query dimension {query_vector.shape[0]} != collection dimension "
                f"{collection.dimension}",
            )
        # Resolve the managed leaf outside the decoder error boundary so a
        # link/reparse rejection cannot be downgraded to ordinary index
        # corruption and silently hidden behind text fallback.
        vector_path = self.vector_path
        try:
            matrix = np.load(vector_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            return [], "unhealthy", f"vector collection unreadable: {exc}"
        if matrix.ndim != 2 or matrix.shape[1] != collection.dimension:
            return [], "unhealthy", "vector collection shape does not match metadata"

        ranked: list[tuple[str, float]] = []
        for document in documents:
            if document.vector_row is None:
                continue
            if document.vector_row < 0 or document.vector_row >= matrix.shape[0]:
                return [], "unhealthy", "vector row points outside the collection"
            score = float(np.dot(matrix[document.vector_row], query_vector))
            ranked.append((document.doc_id, score))
        if not ranked:
            return [], "disabled", "no vectors are available in the selected scopes"
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked, "ready", None

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self.metadata_path
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Hybrid index not found: {metadata_path}") from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid hybrid index metadata: {metadata_path}") from exc
        if not isinstance(payload, dict) or payload.get("version") != HYBRID_INDEX_VERSION:
            raise ValueError(f"Unsupported hybrid index metadata: {metadata_path}")
        generation_id = read_generation_id(self.current_path)
        if generation_id is not None and payload.get("generation_id") != generation_id:
            raise ValueError("Hybrid index metadata does not match CURRENT generation")
        return payload

    def _active_generation_root(self) -> Path:
        generation_id = read_generation_id(self.current_path)
        if generation_id is None:
            return self._safe_index_root()
        generation_root = self._safe_internal_path(
            self.generations_root / generation_id
        )
        if not generation_root.is_dir():
            raise ValueError(f"Missing hybrid index generation: {generation_root}")
        self._safe_internal_path(generation_root, allow_missing=False)
        return generation_root

    def _write_index(
        self,
        documents: list[_DocumentRow],
        vectors: list[np.ndarray],
        collection: EmbeddingCollectionMetadata,
        result: HybridBuildResult,
    ) -> None:
        generation_id = uuid.uuid4().hex
        generations_root = self.generations_root
        generation_root = self._safe_internal_path(generations_root / generation_id)
        generation_root.mkdir(parents=True, exist_ok=False)
        self._safe_internal_path(generation_root, allow_missing=False)
        database_path = self._safe_database_bundle(
            generation_root / DATABASE_FILENAME
        )
        vector_path = self._safe_internal_path(generation_root / VECTOR_FILENAME)
        metadata_path = self._safe_internal_path(
            generation_root / METADATA_FILENAME
        )
        try:
            database_path = self._safe_database_bundle(database_path)
            with sqlite3.connect(database_path) as connection:
                connection.execute("PRAGMA synchronous=FULL")
                _create_schema(connection)
                for document in documents:
                    connection.execute(
                        """
                        INSERT INTO documents (
                            row_id, doc_id, source_path, relative_path, scope_id,
                            project_id, title, body, symbols, project_terms,
                            text_preview, chunk_index, line_start, line_end, vector_row
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document.row_id,
                            document.doc_id,
                            document.source_path,
                            document.relative_path,
                            document.scope_id,
                            document.project_id,
                            document.title,
                            document.body,
                            document.symbols,
                            document.project_terms,
                            document.text_preview,
                            document.chunk_index,
                            document.line_start,
                            document.line_end,
                            document.vector_row,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO documents_fts(rowid, body, title, path, symbols, project_terms) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            document.row_id,
                            document.body,
                            document.title,
                            document.relative_path,
                            document.symbols,
                            document.project_terms,
                        ),
                    )
                connection.commit()

            dimension = collection.dimension or 0
            matrix = (
                np.vstack(vectors).astype(np.float32, copy=False)
                if vectors
                else np.empty((0, dimension), dtype=np.float32)
            )
            vector_path = self._safe_internal_path(vector_path)
            with vector_path.open("wb") as handle:
                np.save(handle, matrix, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            metadata = {
                "version": HYBRID_INDEX_VERSION,
                "generation_id": generation_id,
                "built_at": datetime.now(timezone.utc).isoformat(),
                "layout": {
                    "database": DATABASE_FILENAME,
                    "vectors": VECTOR_FILENAME,
                },
                "collection": collection.to_dict(),
                "scope_coverage": {
                    "source_scope_ids": result.source_scope_ids,
                    "intended_scope_ids": result.intended_scope_ids,
                    "embedded_scope_ids": result.embedded_scope_ids,
                },
                "sensitivity_policy_digest": result.sensitivity_policy_digest,
                "stats": {
                    "files": result.indexed_files,
                    "chunks": result.chunks_written,
                    "vectors": result.vector_count,
                    "skipped": result.skipped,
                    "errors": len(result.errors),
                    "denied": len(result.denied),
                    "capped_sources": sorted(result.capped_sources),
                },
            }
            metadata_path = self._safe_internal_path(metadata_path)
            atomic_write_text(
                metadata_path,
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            )
            fsync_file(self._safe_database_bundle(database_path))
            fsync_directory(
                self._safe_internal_path(generation_root, allow_missing=False)
            )
            fsync_directory(self.generations_root)

            with index_file_lock(self.publication_lock_path):
                previous_generation = read_generation_id(self.current_path)
                retain_previous = False
                if previous_generation:
                    previous_metadata = self._safe_internal_path(
                        self.generations_root / previous_generation / METADATA_FILENAME
                    )
                    try:
                        previous_payload = json.loads(
                            previous_metadata.read_text(encoding="utf-8")
                        )
                        retain_previous = (
                            previous_payload.get("sensitivity_policy_digest")
                            == result.sensitivity_policy_digest
                        )
                    except (OSError, json.JSONDecodeError):
                        retain_previous = False
                self._validated_generation_directories()
                atomic_write_text(self.current_path, generation_id + "\n")
                if previous_generation is None:
                    for legacy_name in (
                        DATABASE_FILENAME,
                        VECTOR_FILENAME,
                        METADATA_FILENAME,
                    ):
                        self._safe_internal_path(
                            self._safe_index_root() / legacy_name
                        ).unlink(missing_ok=True)
                self._gc_generations(
                    generation_id,
                    previous_generation if retain_previous else None,
                    result,
                )
        except Exception:
            try:
                published = read_generation_id(self.current_path) == generation_id
            except (OSError, ValueError):
                published = False
            if not published:
                try:
                    safe_generation_root = self._safe_internal_path(
                        generation_root,
                        allow_missing=False,
                    )
                except (OSError, ValueError):
                    pass
                else:
                    shutil.rmtree(safe_generation_root, ignore_errors=True)
            raise

    def _validated_generation_directories(self) -> list[Path]:
        """Return generation directories only after rejecting linked entries."""

        generations_root = self.generations_root
        if not generations_root.exists():
            return []
        self._safe_internal_path(generations_root, allow_missing=False)
        generations: list[Path] = []
        for path in generations_root.iterdir():
            safe_path = self._safe_internal_path(path)
            if safe_path.is_dir():
                generations.append(safe_path)
        return sorted(generations, key=lambda path: path.name)

    def _gc_generations(
        self,
        current_id: str,
        previous_id: str | None,
        result: HybridBuildResult,
    ) -> None:
        try:
            generations = self._validated_generation_directories()
        except OSError as exc:
            result.errors.append(f"generation cleanup scan failed: {exc}")
            return
        keep = {current_id}
        if previous_id:
            keep.add(previous_id)
        for generation in generations:
            if generation.name in keep:
                continue
            try:
                shutil.rmtree(
                    self._safe_internal_path(generation, allow_missing=False)
                )
            except OSError as exc:
                result.errors.append(f"{generation}: generation cleanup failed ({exc})")


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE documents (
            row_id INTEGER PRIMARY KEY,
            doc_id TEXT NOT NULL UNIQUE,
            source_path TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            project_id TEXT,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            symbols TEXT NOT NULL,
            project_terms TEXT NOT NULL,
            text_preview TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            line_start INTEGER NOT NULL,
            line_end INTEGER NOT NULL,
            vector_row INTEGER
        );
        CREATE INDEX documents_scope_idx ON documents(scope_id);
        CREATE INDEX documents_project_idx ON documents(project_id);
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            body, title, path, symbols, project_terms, tokenize='unicode61'
        );
        """
    )


def source_policy_digest(sources: Iterable[HybridSource]) -> str:
    """Hash sensitivity inputs that determine persisted hybrid-index content."""

    payload = [
        {
            "path": str(source.path.expanduser().resolve()),
            "scope_id": source.scope_id,
            "never_index": list(source.policy.never_index),
            "never_embed": list(source.policy.never_embed),
            "sensitivity_prefixes": list(source.policy.sensitivity_prefixes),
            "excluded_roots": sorted(
                str(path.expanduser().resolve()) for path in source.excluded_roots
            ),
        }
        for source in sorted(sources, key=lambda item: (item.scope_id, str(item.path)))
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hybrid_hit_blocked(
    hit: HybridSearchHit,
    *,
    context_root: Path,
    patterns: Iterable[str],
) -> bool:
    """Apply export rules to a hybrid hit using v1 and v2 path spellings."""

    source = Path(hit.source_path)
    relative_paths = [hit.relative_path]
    try:
        context_relative = source.resolve().relative_to(context_root.expanduser().resolve())
    except (OSError, ValueError):
        context_relative = None
    if context_relative is not None:
        relative_paths.append(context_relative.as_posix())
        parts = context_relative.parts
        if len(parts) >= 3 and parts[0] in {
            "history",
            "memory",
            "scratchpad",
            "knowledge",
            "tools",
            "human",
        }:
            offset = 2 if parts[1] == "common" else 3 if parts[1] == "projects" else 1
            if len(parts) > offset:
                relative_paths.append("/".join((parts[0], *parts[offset:])))
    return SensitivityRuleSet.from_patterns(patterns).blocked(
        source,
        relative_paths=relative_paths,
    )


def _scope_budget(source: HybridSource, budgets: dict[str, _ScopeBudget]) -> _ScopeBudget:
    key = source.project_id or source.scope_id
    max_files = (
        source.policy.max_files_active if source.active else source.policy.max_files_inactive
    )
    max_bytes = (
        source.policy.max_total_bytes_active
        if source.active
        else source.policy.max_total_bytes_inactive
    )
    if max_files <= 0 or max_bytes <= 0:
        raise ValueError("source policy file and byte caps must be positive")
    budget = budgets.get(key)
    if budget is None:
        budget = _ScopeBudget(max_files=max_files, max_bytes=max_bytes)
        budgets[key] = budget
    else:
        # Conflicting registrations resolve conservatively; splitting one
        # project across mounts cannot multiply its crawl allowance.
        budget.max_files = min(budget.max_files, max_files)
        budget.max_bytes = min(budget.max_bytes, max_bytes)
    return budget


def _is_generated_path(path: Path, source_root: Path, excluded_root: Path) -> bool:
    resolved_path = path.resolve()
    resolved_source = source_root.resolve()
    resolved_excluded = excluded_root.resolve()
    if resolved_excluded.name == "search" and resolved_excluded.parent.name == ".afs":
        context_root = resolved_excluded.parent.parent
        if (
            resolved_source != context_root
            and _is_within(context_root, resolved_source)
            and _is_within(resolved_path, context_root)
        ):
            # A central context may be configured under a registered project
            # with a visible directory name. Never crawl that managed tree as
            # project source content and relabel another scope's artifacts.
            return True
    if resolved_source != resolved_excluded and _is_within(resolved_excluded, resolved_source):
        return _is_within(resolved_path, resolved_excluded)
    if resolved_source != resolved_excluded:
        try:
            source_relative = resolved_source.relative_to(resolved_excluded)
        except ValueError:
            return False
        return bool(source_relative.parts) and source_relative.parts[0] in {
            GENERATIONS_DIRNAME,
            DATABASE_FILENAME,
            VECTOR_FILENAME,
            METADATA_FILENAME,
        }
    try:
        relative = resolved_path.relative_to(resolved_source)
    except ValueError:
        return False
    if not relative.parts:
        return False
    return relative.parts[0] in {
        CURRENT_FILENAME,
        GENERATIONS_DIRNAME,
        DATABASE_FILENAME,
        VECTOR_FILENAME,
        METADATA_FILENAME,
        BUILD_LOCK_FILENAME,
        PUBLICATION_LOCK_FILENAME,
    }


def _iter_safe_source_files(
    source: HybridSource,
    result: HybridBuildResult,
    *,
    budget: _ScopeBudget,
    excluded_root: Path,
) -> Iterable[tuple[Path, Path, str]]:
    requested = source.path.expanduser()
    try:
        root = requested.resolve(strict=True)
    except OSError as exc:
        result.errors.append(f"{requested}: source unavailable ({exc})")
        return
    excluded_source_roots = tuple(
        path.expanduser().resolve() for path in source.excluded_roots
    )
    if root.is_file():
        container = root.parent
        if (
            not _is_generated_path(root, container, excluded_root)
            and not any(
                _is_within(root, denied_root) for denied_root in excluded_source_roots
            )
            and _path_allowed(root, container, root.name, source.policy, [])
            and budget.used_files < budget.max_files
            and budget.used_bytes + root.stat().st_size <= budget.max_bytes
        ):
            budget.used_files += 1
            budget.used_bytes += root.stat().st_size
            yield container, root, root.name
        else:
            result.skipped += 1
        return
    if not root.is_dir():
        result.errors.append(f"{root}: source is not a file or directory")
        return

    gitignore = _load_ignore_file(root / ".afsignore")
    if source.policy.respect_gitignore:
        gitignore.extend(_load_ignore_file(root / ".gitignore"))
    hard_files = (
        source.policy.max_files_active if source.active else source.policy.max_files_inactive
    )
    max_files = min(source.max_files or hard_files, hard_files)
    yielded_files = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        base_path = Path(base)
        safe_dirs: list[str] = []
        for dirname in sorted(dirs):
            candidate = base_path / dirname
            rel = candidate.relative_to(root).as_posix()
            if dirname in _HARD_DENY_DIRS or _is_hidden(rel, source.policy):
                continue
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                continue
            if (
                not _is_within(resolved, root)
                or _is_generated_path(resolved, root, excluded_root)
                or any(
                    _is_within(resolved, denied_root)
                    for denied_root in excluded_source_roots
                )
                or _gitignored(rel + "/", gitignore)
            ):
                continue
            safe_dirs.append(dirname)
        dirs[:] = safe_dirs
        for filename in sorted(files):
            path = base_path / filename
            rel = path.relative_to(root).as_posix()
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                result.errors.append(f"{path}: cannot resolve ({exc})")
                result.skipped += 1
                continue
            if not _is_within(resolved, root):
                result.denied.append(f"{path}: escaping symlink denied")
                result.skipped += 1
                continue
            if any(
                _is_within(resolved, denied_root)
                for denied_root in excluded_source_roots
            ):
                result.skipped += 1
                continue
            if not _path_allowed(resolved, root, rel, source.policy, gitignore):
                result.skipped += 1
                continue
            try:
                size_bytes = resolved.stat().st_size
            except OSError as exc:
                result.errors.append(f"{resolved}: cannot stat ({exc})")
                result.skipped += 1
                continue
            if (
                yielded_files >= max_files
                or budget.used_files >= budget.max_files
                or budget.used_bytes + size_bytes > budget.max_bytes
            ):
                source_label = f"{source.scope_id}:{root}"
                if source_label not in result.capped_sources:
                    result.capped_sources.append(source_label)
                return
            yielded_files += 1
            budget.used_files += 1
            budget.used_bytes += size_bytes
            yield root, resolved, rel


def _path_allowed(
    path: Path,
    root: Path,
    relative_path: str,
    policy: SourcePolicy,
    gitignore: list[tuple[str, bool]],
) -> bool:
    if not _is_within(path, root):
        return False
    if any(part in _HARD_DENY_DIRS for part in Path(relative_path).parts):
        return False
    if _is_hidden(relative_path, policy):
        return False
    name = path.name.lower()
    if any(fnmatch.fnmatch(name, pattern.lower()) for pattern in _HARD_DENY_GLOBS):
        return False
    if policy.exclude_globs and any(
        fnmatch.fnmatch(relative_path, pattern) for pattern in policy.exclude_globs
    ):
        return False
    if _sensitivity_blocked(
        path,
        relative_path,
        patterns=policy.never_index,
        prefixes=policy.sensitivity_prefixes,
    ):
        return False
    if policy.include_globs and not any(
        fnmatch.fnmatch(relative_path, pattern) for pattern in policy.include_globs
    ):
        return False
    return not (policy.respect_gitignore and _gitignored(relative_path, gitignore))


def _sensitivity_blocked(
    path: Path,
    relative_path: str,
    *,
    patterns: tuple[str, ...],
    prefixes: tuple[str, ...],
) -> bool:
    if not patterns:
        return False
    relative_paths = [relative_path]
    relative_paths.extend(
        f"{prefix.strip('/')}/{relative_path}" for prefix in prefixes if prefix.strip("/")
    )
    return SensitivityRuleSet.from_patterns(patterns).blocked(
        path,
        relative_paths=relative_paths,
    )


def _safe_read_text(path: Path, policy: SourcePolicy, result: HybridBuildResult) -> str | None:
    try:
        with path.open("rb") as handle:
            raw = handle.read(policy.max_bytes + 1)
    except OSError as exc:
        result.errors.append(f"{path}: read failed ({exc})")
        return None
    if len(raw) > policy.max_bytes or b"\x00" in raw:
        return None
    return raw.decode("utf-8", errors="replace")


def _load_ignore_file(path: Path) -> list[tuple[str, bool]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    patterns: list[tuple[str, bool]] = []
    for line in lines:
        pattern = line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        negated = pattern.startswith("!")
        if negated:
            pattern = pattern[1:]
        patterns.append((pattern.lstrip("/"), negated))
    return patterns


def _gitignored(relative_path: str, patterns: list[tuple[str, bool]]) -> bool:
    ignored = False
    rel = relative_path.rstrip("/")
    for pattern, negated in patterns:
        directory_pattern = pattern.endswith("/")
        candidate = pattern.rstrip("/")
        if "/" not in candidate:
            matched = any(fnmatch.fnmatch(part, candidate) for part in Path(rel).parts)
        else:
            matched = fnmatch.fnmatch(rel, candidate) or fnmatch.fnmatch(rel, candidate + "/**")
        if directory_pattern:
            matched = matched or rel.startswith(candidate + "/")
        if matched:
            ignored = not negated
    return ignored


def _is_hidden(relative_path: str, policy: SourcePolicy) -> bool:
    return not policy.allow_hidden and any(
        part.startswith(".") for part in Path(relative_path).parts
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _chunk_words(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    lines = text.splitlines()
    words: list[tuple[str, int]] = []
    for line_number, line in enumerate(lines, start=1):
        words.extend((match.group(0), line_number) for match in re.finditer(r"\S+", line))
    if not words:
        return []
    step = size - overlap
    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(words):
        selected = words[start : start + size]
        chunks.append((selected[0][1], selected[-1][1], " ".join(word for word, _ in selected)))
        if start + size >= len(words):
            break
        start += step
    return chunks


def _title_for(relative_path: str, text: str) -> str:
    for line in text.splitlines()[:20]:
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title[:200]
    return Path(relative_path).stem[:200]


def _extract_symbols(text: str) -> list[str]:
    symbols: list[str] = []
    for match in _SYMBOL.finditer(text):
        value = (match.group(1) or match.group(2) or "").strip()
        if value:
            symbols.append(value[:120])
    return _dedupe(symbols)[:200]


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _document_id(source: HybridSource, root: Path, rel: str, chunk_index: int) -> str:
    identity = f"{source.scope_id}\0{source.project_id or ''}\0{root}\0{rel}\0{chunk_index}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _normalize_vector(values: Sequence[float] | np.ndarray) -> np.ndarray | None:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return None
    return vector / norm


def _embedding_dimension(
    embed_fn: Callable[[str], list[float]] | None,
    collection: EmbeddingCollectionMetadata | None,
) -> int | None:
    if collection and collection.dimension:
        return collection.dimension
    if embed_fn is None:
        return None
    value = getattr(embed_fn, "_afs_embedding_dimension", None)
    return int(value) if isinstance(value, int) and value > 0 else None


def _collection_metadata(
    embed_fn: Callable[[str], list[float]] | None,
    collection: EmbeddingCollectionMetadata | None,
    *,
    dimension: int | None,
    health: str,
) -> EmbeddingCollectionMetadata:
    if collection is not None:
        return replace(collection, dimension=dimension, normalized=True, health=health)
    if embed_fn is None:
        return EmbeddingCollectionMetadata(
            provider="none", dimension=dimension, normalized=True, health=health
        )
    provider = str(getattr(embed_fn, "_afs_embedding_provider", "custom"))
    instruction = str(getattr(embed_fn, "_afs_embedding_instruction", ""))
    return EmbeddingCollectionMetadata(
        provider=provider,
        model=str(getattr(embed_fn, "_afs_embedding_model", "callable")),
        dimension=dimension,
        document_instruction=instruction,
        query_instruction=GEMINI_QUERY_TASK if provider == "gemini" else instruction,
        normalized=True,
        health=health,
    )


def _collection_from_hybrid_metadata(payload: dict[str, Any]) -> EmbeddingCollectionMetadata:
    wrapper = {"_metadata": {"collection": payload.get("collection", {})}}
    return EmbeddingCollectionMetadata.from_index(wrapper)


def _create_query_embedder(
    collection: EmbeddingCollectionMetadata,
) -> Callable[[str], list[float]]:
    if collection.provider in {"none", "custom", ""} or not collection.model:
        raise ValueError(f"provider {collection.provider!r} cannot be recreated")
    kwargs: dict[str, Any] = {
        "model": collection.model,
        "task_type": collection.query_instruction,
        "dimension": collection.dimension,
    }
    if collection.provider == "gemini":
        kwargs.update(
            task_type=collection.query_instruction or GEMINI_QUERY_TASK,
            output_dimensionality=collection.dimension or DEFAULT_GEMINI_DIMENSION,
        )
    return create_embed_fn(collection.provider, **kwargs)


def _resolve_scopes(
    scope_ids: Sequence[str],
    *,
    include_common: bool,
    common_scope: str,
    all_projects: bool,
) -> set[str] | None:
    if all_projects:
        return None
    allowed = {scope.strip() for scope in scope_ids if scope.strip()}
    if include_common and common_scope.strip():
        allowed.add(common_scope.strip())
    if not allowed:
        raise ValueError("At least one scope_id is required unless all_projects=True")
    return allowed


def _load_scoped_documents(
    connection: sqlite3.Connection, allowed_scopes: set[str] | None
) -> list[_DocumentRow]:
    sql = "SELECT * FROM documents"
    params: list[str] = []
    if allowed_scopes is not None:
        placeholders = ",".join("?" for _ in allowed_scopes)
        sql += f" WHERE scope_id IN ({placeholders})"
        params.extend(sorted(allowed_scopes))
    sql += " ORDER BY doc_id"
    return [_row_from_sql(row) for row in connection.execute(sql, params)]


def _fts_rank(
    connection: sqlite3.Connection, query: str, allowed_scopes: set[str] | None
) -> list[tuple[str, float]]:
    terms = _query_terms(query)
    if not terms:
        return []
    match_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
    sql = (
        "SELECT d.doc_id, bm25(documents_fts) AS rank "
        "FROM documents_fts JOIN documents d ON d.row_id = documents_fts.rowid "
        "WHERE documents_fts MATCH ?"
    )
    params: list[Any] = [match_query]
    if allowed_scopes is not None:
        placeholders = ",".join("?" for _ in allowed_scopes)
        sql += f" AND d.scope_id IN ({placeholders})"
        params.extend(sorted(allowed_scopes))
    rows = [(str(row[0]), -float(row[1])) for row in connection.execute(sql, params)]
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows


def _association_ranks(
    query: str, documents: list[_DocumentRow], *, include_project: bool
) -> dict[str, list[tuple[str, float]]]:
    terms = [term.lower() for term in _query_terms(query)]
    if not terms:
        return {
            "path": [],
            "symbol": [],
            "exact": [],
            **({"project": []} if include_project else {}),
        }
    ranks: dict[str, list[tuple[str, float]]] = {"path": [], "symbol": [], "exact": []}
    if include_project:
        ranks["project"] = []
    normalized_query = " ".join(terms)
    for document in documents:
        path = document.relative_path.lower()
        symbols = document.symbols.lower().split()
        project = f"{document.project_id or ''} {document.project_terms}".lower()
        path_score = sum(term in path for term in terms)
        symbol_score = sum(term in symbols for term in terms)
        project_score = sum(term in project for term in terms)
        exact_score = float(
            normalized_query == Path(path).stem.lower()
            or normalized_query in symbols
            or normalized_query == path
        )
        if path_score:
            ranks["path"].append((document.doc_id, float(path_score)))
        if symbol_score:
            ranks["symbol"].append((document.doc_id, float(symbol_score)))
        if exact_score:
            ranks["exact"].append((document.doc_id, exact_score))
        if include_project and project_score:
            ranks["project"].append((document.doc_id, float(project_score)))
    for items in ranks.values():
        items.sort(key=lambda item: (-item[1], item[0]))
    return ranks


def _query_terms(query: str) -> list[str]:
    return _dedupe(match.group(0).lower() for match in _WORD.finditer(query))


def _rrf_fuse(
    ranks: dict[str, list[tuple[str, float]]],
    documents: dict[str, _DocumentRow],
    *,
    top_k: int,
) -> list[HybridSearchHit]:
    scores: dict[str, float] = {}
    signals: dict[str, dict[str, dict[str, float | int]]] = {}
    for signal_name in sorted(ranks):
        for rank, (doc_id, raw_score) in enumerate(ranks[signal_name], start=1):
            if doc_id not in documents:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
            signals.setdefault(doc_id, {})[signal_name] = {
                "rank": rank,
                "score": raw_score,
            }
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]
    results: list[HybridSearchHit] = []
    for doc_id in ordered:
        document = documents[doc_id]
        results.append(
            HybridSearchHit(
                doc_id=doc_id,
                score=scores[doc_id],
                source_path=document.source_path,
                relative_path=document.relative_path,
                scope_id=document.scope_id,
                project_id=document.project_id,
                text_preview=document.text_preview,
                chunk_index=document.chunk_index,
                line_start=document.line_start,
                line_end=document.line_end,
                signals=signals[doc_id],
            )
        )
    return results


def _row_from_sql(row: sqlite3.Row) -> _DocumentRow:
    return _DocumentRow(
        row_id=int(row["row_id"]),
        doc_id=str(row["doc_id"]),
        source_path=str(row["source_path"]),
        relative_path=str(row["relative_path"]),
        scope_id=str(row["scope_id"]),
        project_id=str(row["project_id"]) if row["project_id"] is not None else None,
        title=str(row["title"]),
        body=str(row["body"]),
        symbols=str(row["symbols"]),
        project_terms=str(row["project_terms"]),
        text_preview=str(row["text_preview"]),
        chunk_index=int(row["chunk_index"]),
        line_start=int(row["line_start"]),
        line_end=int(row["line_end"]),
        vector_row=int(row["vector_row"]) if row["vector_row"] is not None else None,
    )
