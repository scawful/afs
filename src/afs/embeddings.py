"""Embedding indexing and search helpers for AFS."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EmbeddingFactory = Callable[..., Callable[[str], list[float]]]
_EMBEDDING_BACKENDS: dict[str, EmbeddingFactory] = {}

EMBEDDING_INDEX_VERSION = 2
DEFAULT_GEMINI_MODEL = "gemini-embedding-2"
DEFAULT_GEMINI_DIMENSION = 768
GEMINI_DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
GEMINI_QUERY_TASK = "RETRIEVAL_QUERY"


@dataclass(frozen=True)
class EmbeddingCollectionMetadata:
    """Versioned contract for vectors stored in an embedding collection."""

    version: int = EMBEDDING_INDEX_VERSION
    provider: str = "none"
    model: str = ""
    dimension: int | None = None
    document_instruction: str = ""
    query_instruction: str = ""
    normalized: bool = False
    health: str = "keyword_only"

    @classmethod
    def from_index(cls, payload: dict[str, Any]) -> EmbeddingCollectionMetadata:
        raw = payload.get("_metadata", {})
        collection = raw.get("collection", {}) if isinstance(raw, dict) else {}
        if not isinstance(collection, dict):
            collection = {}
        dimension = collection.get("dimension")
        return cls(
            version=_coerce_positive_int(collection.get("version"), default=1),
            provider=str(collection.get("provider") or "none"),
            model=str(collection.get("model") or ""),
            dimension=_coerce_positive_int(dimension, default=None),
            document_instruction=str(collection.get("document_instruction") or ""),
            query_instruction=str(collection.get("query_instruction") or ""),
            normalized=bool(collection.get("normalized", False)),
            health=str(collection.get("health") or "legacy"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "provider": self.provider,
            "model": self.model,
            "dimension": self.dimension,
            "document_instruction": self.document_instruction,
            "query_instruction": self.query_instruction,
            "normalized": self.normalized,
            "health": self.health,
        }


@dataclass
class EmbeddingSearchResponse:
    """Search results plus an honest account of semantic retrieval state."""

    results: list[SearchResult] = field(default_factory=list)
    semantic_status: str = "not_requested"
    semantic_reason: str | None = None
    collection: EmbeddingCollectionMetadata | None = None


@dataclass
class EmbeddingIndexResult:
    total_files: int = 0
    indexed: int = 0
    skipped: int = 0
    reused: int = 0
    removed: int = 0
    orphans_removed: int = 0
    chunks_written: int = 0
    semantic_status: str = "keyword_only"
    mode: str = "full"
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"total={self.total_files} indexed={self.indexed} "
            f"skipped={self.skipped} errors={len(self.errors)}"
        ]
        if self.mode == "incremental":
            parts.append(f"reused={self.reused} removed={self.removed}")
        if self.orphans_removed:
            parts.append(f"orphans_removed={self.orphans_removed}")
        if self.chunks_written:
            parts.append(f"chunks={self.chunks_written}")
        parts.append(f"semantic={self.semantic_status}")
        return " ".join(parts)


@dataclass
class SearchResult:
    doc_id: str
    score: float
    source_path: str
    text_preview: str
    chunk_index: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "doc_id": self.doc_id,
            "score": self.score,
            "source_path": self.source_path,
            "text_preview": self.text_preview,
        }
        if self.chunk_index is not None:
            d["chunk_index"] = self.chunk_index
        if self.line_start is not None:
            d["line_start"] = self.line_start
        if self.line_end is not None:
            d["line_end"] = self.line_end
        if self.char_start is not None:
            d["char_start"] = self.char_start
        if self.char_end is not None:
            d["char_end"] = self.char_end
        return d


@dataclass
class EmbeddingEvalCase:
    query: str
    expected_doc_ids: list[str] = field(default_factory=list)
    expected_path_contains: list[str] = field(default_factory=list)
    case_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class EmbeddingEvalResult:
    total: int = 0
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    mrr: float = 0.0
    avg_hit_score: float = 0.0
    cases: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"total={self.total} hits={self.hits} "
            f"misses={self.misses} hit_rate={self.hit_rate:.3f} "
            f"mrr={self.mrr:.3f} avg_hit_score={self.avg_hit_score:.3f}"
        )


def register_embedding_backend(name: str, factory: EmbeddingFactory) -> None:
    """Register an embedding backend factory."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Backend name cannot be empty")
    _EMBEDDING_BACKENDS[normalized] = factory


def create_embed_fn(provider: str, **kwargs) -> Callable[[str], list[float]]:
    """Create an embedding function from a registered backend."""
    normalized = provider.strip().lower()
    factory = _EMBEDDING_BACKENDS.get(normalized)
    if not factory:
        raise ValueError(f"Unknown embedding provider: {provider}")
    embed_fn = factory(**kwargs)
    provider_defaults: dict[str, tuple[str, str, int | None]] = {
        "gemini": (DEFAULT_GEMINI_MODEL, GEMINI_DOCUMENT_TASK, DEFAULT_GEMINI_DIMENSION),
        "ollama": ("nomic-embed-text", "", None),
    }
    default_model, default_task, default_dimension = provider_defaults.get(
        normalized, ("", "", None)
    )
    model = str(kwargs.get("model") or default_model)
    task_type = str(kwargs.get("task_type") or default_task)
    dimension = _coerce_positive_int(
        kwargs.get("output_dimensionality") or kwargs.get("dimension"),
        default=default_dimension,
    )
    # Functions are intentionally annotated instead of wrapped so provider
    # implementations retain their normal exception and tracing behavior.
    try:
        embed_fn._afs_embedding_provider = normalized  # type: ignore[attr-defined]
        embed_fn._afs_embedding_model = model  # type: ignore[attr-defined]
        embed_fn._afs_embedding_dimension = dimension  # type: ignore[attr-defined]
        embed_fn._afs_embedding_instruction = task_type  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        backend = embed_fn

        def embed_fn(text: str) -> list[float]:
            return backend(text)

        embed_fn._afs_embedding_provider = normalized  # type: ignore[attr-defined]
        embed_fn._afs_embedding_model = model  # type: ignore[attr-defined]
        embed_fn._afs_embedding_dimension = dimension  # type: ignore[attr-defined]
        embed_fn._afs_embedding_instruction = task_type  # type: ignore[attr-defined]
    return embed_fn


def create_ollama_embed_fn(
    model: str = "nomic-embed-text",
    host: str = "http://localhost:11435",
) -> Callable[[str], list[float]]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx not installed") from exc

    def embed(text: str) -> list[float]:
        response = httpx.post(
            f"{host}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding", [])
        return [float(value) for value in embedding]

    return embed


def create_openai_embed_fn(
    model: str,
    *,
    base_url: str = "https://api.openai.com/v1",
    api_key: str | None = None,
    timeout: float = 30.0,
) -> Callable[[str], list[float]]:
    """Create an OpenAI-compatible embeddings backend."""
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx not installed") from exc

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("Missing API key for OpenAI-compatible embeddings backend")

    base = base_url.rstrip("/")
    url = f"{base}/embeddings" if base.endswith("/v1") else f"{base}/v1/embeddings"

    def embed(text: str) -> list[float]:
        response = httpx.post(
            url,
            json={"model": model, "input": text},
            headers={"Authorization": f"Bearer {resolved_api_key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        rows = data.get("data", [])
        if not rows:
            return []
        embedding = rows[0].get("embedding", [])
        return [float(value) for value in embedding]

    return embed


def create_gemini_embed_fn(
    model: str = DEFAULT_GEMINI_MODEL,
    *,
    api_key: str | None = None,
    task_type: str = GEMINI_DOCUMENT_TASK,
    output_dimensionality: int = DEFAULT_GEMINI_DIMENSION,
) -> Callable[[str], list[float]]:
    """Create a Gemini embeddings backend.

    Uses the ``google-genai`` SDK when available, falling back to a
    plain HTTP request against the Gemini REST API otherwise.

    The stable ``gemini-embedding-2`` collection defaults to 768 dimensions.
    Document and query callers must use their respective retrieval task type.
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "Missing API key for Gemini embeddings. "
            "Set GEMINI_API_KEY or pass --gemini-api-key."
        )

    # Normalize model name — SDK expects "models/" prefix
    qualified_model = model if model.startswith("models/") else f"models/{model}"

    # Try the official SDK first; fall back to raw HTTP.
    try:
        from google import genai  # type: ignore[import-untyped]

        client = genai.Client(api_key=resolved_key)

        def _embed_sdk(text: str) -> list[float]:
            result = client.models.embed_content(
                model=qualified_model,
                contents=text,
                config={
                    "task_type": task_type,
                    "output_dimensionality": output_dimensionality,
                },
            )
            values = result.embeddings[0].values
            return [float(v) for v in values]

        return _embed_sdk
    except ImportError:
        pass

    # Fallback: plain HTTP via httpx or requests.
    try:
        import httpx as _http_mod
    except ImportError:
        try:
            import requests as _http_mod  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError(
                "Neither google-genai nor httpx/requests installed for Gemini embeddings"
            ) from exc

    # Strip "models/" prefix for the URL path since it's already in the endpoint
    url_model = model.removeprefix("models/")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{url_model}:embedContent"
        f"?key={resolved_key}"
    )

    def _embed_http(text: str) -> list[float]:
        body = {
            "model": qualified_model,
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
            "outputDimensionality": output_dimensionality,
        }
        if hasattr(_http_mod, "post"):
            resp = _http_mod.post(url, json=body, timeout=30.0)  # type: ignore[union-attr]
        else:
            resp = _http_mod.post(url, json=body, timeout=30)  # type: ignore[union-attr]
        resp.raise_for_status()
        data = resp.json()
        values = data.get("embedding", {}).get("values", [])
        return [float(v) for v in values]

    return _embed_http


def create_hf_embed_fn(
    model: str,
    *,
    device: str | None = None,
    max_tokens: int | None = 512,
    pooling: str = "mean",
    normalize: bool = True,
    token: str | None = None,
) -> Callable[[str], list[float]]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers not installed") from exc

    pooling = (pooling or "mean").strip().lower()
    if pooling not in {"mean", "cls"}:
        raise ValueError("pooling must be 'mean' or 'cls'")

    resolved_device = _resolve_torch_device(device, torch)
    auth_token = _resolve_hf_token(token)

    tokenizer = AutoTokenizer.from_pretrained(model, token=auth_token)
    model_instance = AutoModel.from_pretrained(model, token=auth_token)
    model_instance.to(resolved_device)
    model_instance.eval()

    def embed(text: str) -> list[float]:
        if not isinstance(text, str):
            text_value = str(text)
        else:
            text_value = text
        if not text_value.strip():
            return []

        token_kwargs = {"return_tensors": "pt", "truncation": True}
        if max_tokens:
            token_kwargs["max_length"] = max_tokens
        encoded = tokenizer(text_value, **token_kwargs)
        encoded = {key: val.to(resolved_device) for key, val in encoded.items()}

        with torch.no_grad():
            outputs = model_instance(**encoded)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                if isinstance(outputs, (tuple, list)) and outputs:
                    hidden = outputs[0]
                else:
                    raise RuntimeError("Unexpected model output for embeddings")

            if pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                mask = encoded.get("attention_mask")
                if mask is None:
                    pooled = hidden.mean(dim=1)
                else:
                    mask = mask.unsqueeze(-1).expand(hidden.size()).float()
                    summed = (hidden * mask).sum(dim=1)
                    denom = mask.sum(dim=1).clamp(min=1e-9)
                    pooled = summed / denom

            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        return pooled[0].detach().cpu().tolist()

    return embed


def load_embedding_collection_metadata(index_root: Path) -> EmbeddingCollectionMetadata:
    """Load collection metadata while preserving legacy index readability."""
    index_path = index_root / "embedding_index.json"
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Index not found: {index_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid index: {index_path}") from exc
    if not isinstance(payload, dict):
        # Very early indexes were occasionally emitted as JSON arrays. They
        # contain no usable provider contract but remain discoverable/readable.
        return EmbeddingCollectionMetadata(version=1, health="legacy")
    return EmbeddingCollectionMetadata.from_index(payload)


def discover_embedding_indexes(root: Path, *, max_depth: int = 2) -> list[Path]:
    """Discover documented embedding index roots in a bounded, stable order.

    AFS checks the supplied directory, ``.afs/search``, ``knowledge``, and
    their direct project children. It intentionally does not recursively scan
    an entire home directory.
    """
    resolved = root.expanduser().resolve()
    candidates = [resolved, resolved / ".afs" / "search", resolved / "knowledge"]
    found: dict[str, Path] = {}
    for candidate in candidates:
        _record_embedding_index(candidate, found)
        if max_depth <= 0 or not candidate.is_dir():
            continue
        try:
            children = sorted(candidate.iterdir(), key=lambda path: path.name)
        except OSError:
            continue
        for child in children:
            if child.is_dir():
                _record_embedding_index(child, found)
    return [found[key] for key in sorted(found)]


def create_query_embed_fn_from_index(
    index_root: Path,
    *,
    api_key: str | None = None,
) -> Callable[[str], list[float]]:
    """Recreate the query-side embedder declared by a versioned collection."""
    metadata = load_embedding_collection_metadata(index_root)
    if metadata.version < EMBEDDING_INDEX_VERSION:
        raise ValueError("Legacy embedding index has no query embedder contract")
    if metadata.provider in {"", "none", "custom"} or not metadata.model:
        raise ValueError(
            f"Embedding provider {metadata.provider!r} cannot be recreated from metadata"
        )
    kwargs: dict[str, Any] = {
        "model": metadata.model,
        "task_type": metadata.query_instruction,
        "dimension": metadata.dimension,
    }
    if metadata.provider == "gemini":
        kwargs.update(
            task_type=metadata.query_instruction or GEMINI_QUERY_TASK,
            output_dimensionality=metadata.dimension or DEFAULT_GEMINI_DIMENSION,
            api_key=api_key,
        )
    elif api_key is not None:
        kwargs["api_key"] = api_key
    return create_embed_fn(metadata.provider, **kwargs)


def _record_embedding_index(path: Path, found: dict[str, Path]) -> None:
    index_path = path / "embedding_index.json"
    if index_path.is_file():
        found[str(path)] = path


def _coerce_positive_int(value: object, *, default: int | None) -> int | None:
    if isinstance(value, bool):
        return default
    try:
        resolved = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return resolved if resolved > 0 else default


def _embed_fn_collection_metadata(
    embed_fn: Callable[[str], list[float]] | None,
) -> dict[str, Any]:
    if embed_fn is None:
        return {
            "provider": "none",
            "model": "",
            "dimension": None,
            "document_instruction": "",
            "query_instruction": "",
        }
    provider = str(getattr(embed_fn, "_afs_embedding_provider", "custom"))
    instruction = str(getattr(embed_fn, "_afs_embedding_instruction", ""))
    return {
        "provider": provider,
        "model": str(getattr(embed_fn, "_afs_embedding_model", "callable")),
        "dimension": _coerce_positive_int(
            getattr(embed_fn, "_afs_embedding_dimension", None), default=None
        ),
        "document_instruction": instruction,
        "query_instruction": GEMINI_QUERY_TASK if provider == "gemini" else instruction,
    }


def _load_existing_index(output_dir: Path) -> tuple[dict[str, str], dict[str, dict]]:
    """Load existing index and per-doc metadata (size_bytes, modified_at)."""
    index_path = output_dir / "embedding_index.json"
    embeddings_dir = output_dir / "embeddings"
    old_index: dict[str, str] = {}
    old_meta: dict[str, dict] = {}
    if not index_path.exists():
        return old_index, old_meta
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            old_index = {k: v for k, v in raw.items() if k != "_metadata"}
    except (json.JSONDecodeError, OSError):
        return old_index, old_meta
    for doc_id, filename in old_index.items():
        entry_path = embeddings_dir / filename
        try:
            data = json.loads(entry_path.read_text(encoding="utf-8"))
            old_meta[doc_id] = {
                "size_bytes": data.get("size_bytes", 0),
                "modified_at": data.get("modified_at"),
                "chunked": bool(data.get("chunked", False)),
            }
        except (json.JSONDecodeError, OSError):
            pass
    return old_index, old_meta


def _referenced_entry_files(embeddings_dir: Path, filename: str) -> set[str]:
    """Return a legacy entry and any chunk payloads it references."""
    referenced = {filename}
    try:
        payload = json.loads((embeddings_dir / filename).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return referenced
    if isinstance(payload, dict) and payload.get("chunked"):
        referenced.update(
            name for name in payload.get("chunks", []) if isinstance(name, str)
        )
    return referenced


def _index_vectors_are_compatible(
    index: dict[str, Any], embeddings_dir: Path, expected_dimension: int
) -> bool:
    """Fail closed when a supposedly semantic collection is partial."""
    vector_count = 0
    for doc_id, filename in index.items():
        if doc_id == "_metadata":
            continue
        if not isinstance(filename, str):
            return False
        try:
            payload = json.loads((embeddings_dir / filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        rows: list[dict[str, Any]] = []
        if isinstance(payload, dict) and payload.get("chunked"):
            for chunk_name in payload.get("chunks", []):
                if not isinstance(chunk_name, str):
                    return False
                try:
                    chunk = json.loads(
                        (embeddings_dir / chunk_name).read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    return False
                if not isinstance(chunk, dict):
                    return False
                rows.append(chunk)
        elif isinstance(payload, dict):
            rows.append(payload)
        else:
            return False
        for row in rows:
            vector = row.get("embedding")
            if not isinstance(vector, list) or len(vector) != expected_dimension:
                return False
            vector_count += 1
    return vector_count > 0


def _file_changed(path: Path, old_size: int, old_mtime: str | None) -> bool:
    """Check if file changed based on size+mtime comparison."""
    try:
        stat = path.stat()
    except OSError:
        return True
    if stat.st_size != old_size:
        return True
    if old_mtime:
        current_mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
        if current_mtime != old_mtime:
            return True
    return False


def _split_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int, str]]:
    """Split *text* into overlapping char-boundary chunks.

    Returns a list of ``(char_start, char_end, chunk_text)`` tuples.
    """
    if not text:
        return []
    step = max(1, chunk_size - overlap)
    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append((start, end, text[start:end]))
        if end == len(text):
            break
        start += step
    return chunks


def _count_lines_before(text: str, char_offset: int) -> int:
    """Return the 1-based line number at *char_offset* in *text*."""
    return text[:char_offset].count("\n") + 1


def build_embedding_index(
    sources: Iterable[Path],
    output_dir: Path,
    *,
    include_patterns: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = None,
    max_files: int | None = 10000,
    preview_chars: int = 1000,
    embed_chars: int = 2000,
    max_bytes: int | None = 2_000_000,
    embed_fn: Callable[[str], list[float]] | None = None,
    include_hidden: bool = False,
    skip_path: Callable[[Path, Path], bool] | None = None,
    incremental: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int = 200,
) -> EmbeddingIndexResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "embedding_index.json"

    index: dict[str, str] = {}
    result = EmbeddingIndexResult()
    result.mode = "incremental" if incremental else "full"
    collection = _embed_fn_collection_metadata(embed_fn)
    expected_dimension = _coerce_positive_int(collection.get("dimension"), default=None)
    semantic_attempts = 0
    semantic_failures = 0
    referenced_files: set[str] = set()

    old_index: dict[str, str] = {}
    old_meta: dict[str, dict] = {}
    previous_collection: EmbeddingCollectionMetadata | None = None
    if incremental:
        old_index, old_meta = _load_existing_index(output_dir)
        try:
            previous_collection = load_embedding_collection_metadata(output_dir)
        except (OSError, ValueError):
            previous_collection = None
        if previous_collection and previous_collection.version >= EMBEDDING_INDEX_VERSION:
            current_identity = (collection["provider"], collection["model"], expected_dimension)
            previous_identity = (
                previous_collection.provider,
                previous_collection.model,
                previous_collection.dimension,
            )
            if current_identity != previous_identity:
                # Reusing vectors from a different collection would make
                # cosine scores meaningless. Treat this incremental request
                # as a clean rebuild instead.
                old_index = {}
                old_meta = {}

    include = list(include_patterns) if include_patterns else []
    exclude = list(exclude_patterns) if exclude_patterns else []

    seen_doc_ids: set[str] = set()
    for source_root, path in _iter_source_files(
        sources, include, exclude, max_files, include_hidden
    ):
        result.total_files += 1
        if skip_path is not None and skip_path(source_root, path):
            result.skipped += 1
            continue

        doc_id = _build_doc_id(source_root, path)
        seen_doc_ids.add(doc_id)
        filename = f"{_hash_doc_id(doc_id)}.json"

        # Incremental: reuse unchanged files
        if incremental and doc_id in old_index and doc_id in old_meta:
            meta = old_meta[doc_id]
            if not _file_changed(path, meta.get("size_bytes", 0), meta.get("modified_at")):
                index[doc_id] = old_index[doc_id]
                referenced_files.update(
                    _referenced_entry_files(embeddings_dir, old_index[doc_id])
                )
                result.reused += 1
                continue

        try:
            text = _read_text_file(path, max_bytes=max_bytes)
        except OSError as exc:
            result.skipped += 1
            result.errors.append(f"{path}: {exc}")
            continue
        if text is None:
            result.skipped += 1
            continue

        stat = None
        if path.exists():
            try:
                stat = path.stat()
            except OSError:
                stat = None
        size_bytes = stat.st_size if stat else 0
        modified_at = (
            datetime.fromtimestamp(stat.st_mtime).isoformat() if stat else None
        )

        # --- Chunked mode ---
        if chunk_size is not None and len(text) > chunk_size:
            base_hash = _hash_doc_id(doc_id)
            chunks = _split_into_chunks(text, chunk_size, chunk_overlap)
            chunk_filenames: list[str] = []
            chunk_ok = True
            for chunk_idx, (char_start, char_end, chunk_text) in enumerate(chunks):
                chunk_doc_id = f"{doc_id}::chunk:{chunk_idx}"
                chunk_filename = f"{base_hash}_c{chunk_idx}.json"
                chunk_preview = chunk_text[:preview_chars].strip()
                chunk_embedding: list[float] = []
                if embed_fn:
                    semantic_attempts += 1
                    embed_failed = False
                    try:
                        chunk_embedding = embed_fn(chunk_text[:embed_chars])
                    except Exception as exc:
                        result.errors.append(f"{path} chunk {chunk_idx}: embed failed ({exc})")
                        semantic_failures += 1
                        embed_failed = True
                    if chunk_embedding:
                        if expected_dimension is None:
                            expected_dimension = len(chunk_embedding)
                        elif len(chunk_embedding) != expected_dimension:
                            result.errors.append(
                                f"{path} chunk {chunk_idx}: embedding dimension "
                                f"{len(chunk_embedding)} != collection dimension "
                                f"{expected_dimension}"
                            )
                            chunk_embedding = []
                            semantic_failures += 1
                    elif not embed_failed:
                        semantic_failures += 1
                line_start = _count_lines_before(text, char_start)
                line_end = _count_lines_before(text, char_end - 1) if char_end > char_start else line_start
                chunk_payload = {
                    "id": chunk_doc_id,
                    "parent_doc_id": doc_id,
                    "source_path": str(path),
                    "chunk_index": chunk_idx,
                    "line_start": line_start,
                    "line_end": line_end,
                    "char_start": char_start,
                    "char_end": char_end,
                    "text_preview": chunk_preview,
                    "search_text": chunk_text[:embed_chars],
                    "embedding": chunk_embedding,
                    "size_bytes": size_bytes,
                    "modified_at": modified_at,
                }
                try:
                    (embeddings_dir / chunk_filename).write_text(
                        json.dumps(chunk_payload, ensure_ascii=True) + "\n",
                        encoding="utf-8",
                    )
                    chunk_filenames.append(chunk_filename)
                    referenced_files.add(chunk_filename)
                    result.chunks_written += 1
                except OSError as exc:
                    result.errors.append(f"{path} chunk {chunk_idx}: write failed ({exc})")
                    chunk_ok = False
                    break

            if not chunk_ok:
                referenced_files.difference_update(chunk_filenames)
                result.skipped += 1
                continue

            manifest_filename = f"{base_hash}_chunks.json"
            chunk_manifest = {
                "id": doc_id,
                "source_path": str(path),
                "size_bytes": size_bytes,
                "modified_at": modified_at,
                "chunked": True,
                "chunk_count": len(chunk_filenames),
                "chunks": chunk_filenames,
            }
            try:
                (embeddings_dir / manifest_filename).write_text(
                    json.dumps(chunk_manifest, ensure_ascii=True) + "\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                referenced_files.difference_update(chunk_filenames)
                result.skipped += 1
                result.errors.append(f"{path}: chunk manifest write failed ({exc})")
                continue
            index[doc_id] = manifest_filename
            referenced_files.add(manifest_filename)
            result.indexed += 1
            continue

        # --- Single-file mode (original path) ---
        preview = text[:preview_chars].strip()
        embed_text = text[:embed_chars]
        embedding: list[float] = []
        if embed_fn:
            semantic_attempts += 1
            embed_failed = False
            try:
                embedding = embed_fn(embed_text)
            except Exception as exc:
                result.errors.append(f"{path}: embed failed ({exc})")
                embedding = []
                embed_failed = True
            if embedding:
                if expected_dimension is None:
                    expected_dimension = len(embedding)
                elif len(embedding) != expected_dimension:
                    result.errors.append(
                        f"{path}: embedding dimension {len(embedding)} != "
                        f"collection dimension {expected_dimension}"
                    )
                    embedding = []
                    semantic_failures += 1
            elif not embed_failed:
                semantic_failures += 1

        payload = {
            "id": doc_id,
            "source_path": str(path),
            "text_preview": preview,
            "search_text": embed_text,
            "embedding": embedding,
            "size_bytes": size_bytes,
            "modified_at": modified_at,
        }
        try:
            (embeddings_dir / filename).write_text(
                json.dumps(payload, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            result.skipped += 1
            result.errors.append(f"{path}: write failed ({exc})")
            continue
        index[doc_id] = filename
        referenced_files.add(filename)
        result.indexed += 1

    # Detect deletions in incremental mode
    if incremental:
        for old_doc_id, old_filename in old_index.items():
            if old_doc_id not in seen_doc_ids:
                result.removed += 1
                stale_path = embeddings_dir / old_filename
                try:
                    stale_path.unlink(missing_ok=True)
                except OSError:
                    pass

    # Remove payloads that are no longer reachable from the new manifest.
    # This also cleans old chunk files after a document shrinks or is deleted.
    for payload_path in embeddings_dir.glob("*.json"):
        if payload_path.name in referenced_files:
            continue
        try:
            payload_path.unlink()
            result.orphans_removed += 1
        except OSError as exc:
            result.errors.append(f"{payload_path}: orphan cleanup failed ({exc})")

    if embed_fn is None:
        collection_health = "keyword_only"
    elif (
        semantic_attempts == 0
        and result.reused
        and previous_collection is not None
        and previous_collection.health == "healthy"
    ):
        collection_health = "healthy"
    elif semantic_attempts == 0 or semantic_failures or result.errors:
        collection_health = "unhealthy"
    else:
        collection_health = "healthy"
    collection.update(
        version=EMBEDDING_INDEX_VERSION,
        dimension=expected_dimension,
        normalized=False,
        health=collection_health,
    )
    result.semantic_status = collection_health

    # Write _metadata block into index
    metadata_block: dict[str, Any] = {
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "mode": result.mode,
        "collection": collection,
        "stats": {
            "total_files": result.total_files,
            "reused": result.reused,
            "re_embedded": result.indexed,
            "removed": result.removed,
            "orphans_removed": result.orphans_removed,
            "semantic_attempts": semantic_attempts,
            "semantic_failures": semantic_failures,
        },
    }
    index_with_meta: dict[str, Any] = {"_metadata": metadata_block}
    index_with_meta.update(index)

    index_path.write_text(
        json.dumps(index_with_meta, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    try:
        from .history import log_embedding_event
        log_embedding_event("index_build", metadata={
            "output_dir": str(output_dir),
            "total_files": result.total_files,
            "indexed": result.indexed,
            "skipped": result.skipped,
            "reused": result.reused,
            "removed": result.removed,
            "mode": result.mode,
            "errors_count": len(result.errors),
        })
    except Exception:
        pass
    return result


def search_embedding_index(
    index_root: Path,
    query: str,
    *,
    embed_fn: Callable[[str], list[float]] | None = None,
    recreate_query_embedder: bool = False,
    top_k: int = 5,
    min_score: float = 0.3,
) -> list[SearchResult]:
    """Search a legacy or versioned embedding index.

    This compatibility wrapper preserves the historic list return value. New
    callers should use :func:`search_embedding_index_detailed` so fallback is
    visible rather than mislabeled as semantic retrieval.
    """
    return search_embedding_index_detailed(
        index_root,
        query,
        embed_fn=embed_fn,
        recreate_query_embedder=recreate_query_embedder,
        top_k=top_k,
        min_score=min_score,
    ).results


def search_embedding_index_detailed(
    index_root: Path,
    query: str,
    *,
    embed_fn: Callable[[str], list[float]] | None = None,
    recreate_query_embedder: bool = False,
    top_k: int = 5,
    min_score: float = 0.3,
) -> EmbeddingSearchResponse:
    """Search with strict vector compatibility and explicit fallback state."""
    index_path = index_root / "embedding_index.json"
    embeddings_dir = index_root / "embeddings"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid index: {index_path}") from exc
    if not isinstance(index, dict):
        raise ValueError(f"Invalid index object: {index_path}")

    collection = EmbeddingCollectionMetadata.from_index(index)
    semantic_requested = embed_fn is not None or recreate_query_embedder
    semantic_status = "not_requested"
    semantic_reason: str | None = None

    if semantic_requested and collection.health == "unhealthy":
        semantic_status = "fallback"
        semantic_reason = "embedding collection is unhealthy"
        embed_fn = None
    elif recreate_query_embedder and embed_fn is None:
        try:
            embed_fn = create_query_embed_fn_from_index(index_root)
        except (OSError, RuntimeError, ValueError) as exc:
            semantic_status = "fallback"
            semantic_reason = f"query embedder unavailable: {exc}"

    query_embedding: list[float] | None = None
    if embed_fn:
        try:
            query_embedding = embed_fn(query)
        except Exception as exc:
            query_embedding = None
            semantic_status = "fallback"
            semantic_reason = f"query embedding failed: {exc}"
        if query_embedding:
            expected_dimension = collection.dimension
            if expected_dimension and len(query_embedding) != expected_dimension:
                semantic_status = "fallback"
                semantic_reason = (
                    f"query dimension {len(query_embedding)} != collection dimension "
                    f"{expected_dimension}"
                )
                query_embedding = None
            elif not _index_vectors_are_compatible(index, embeddings_dir, len(query_embedding)):
                semantic_status = "fallback"
                semantic_reason = "embedding collection contains missing or incompatible vectors"
                query_embedding = None
            else:
                semantic_status = "ready"
        elif semantic_status == "not_requested":
            semantic_status = "fallback"
            semantic_reason = "query embedder returned no vector"

    results: list[SearchResult] = []
    for doc_id, filename in index.items():
        if doc_id == "_metadata":
            continue
        entry_path = embeddings_dir / filename
        try:
            data = json.loads(entry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        # --- Chunked manifest: score each chunk, keep best ---
        if data.get("chunked"):
            best: SearchResult | None = None
            for chunk_filename in data.get("chunks", []):
                chunk_path = embeddings_dir / chunk_filename
                try:
                    chunk_data = json.loads(chunk_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                chunk_text_preview = chunk_data.get("text_preview", "")
                chunk_source = str(chunk_data.get("source_path", ""))
                chunk_embedding = chunk_data.get("embedding", [])
                if query_embedding and chunk_embedding:
                    score = _cosine_similarity(query_embedding, chunk_embedding)
                else:
                    search_text = chunk_data.get("search_text") or chunk_text_preview
                    score = _keyword_score(query, search_text, doc_id)
                if score >= min_score:
                    candidate = SearchResult(
                        doc_id=str(chunk_data.get("id", doc_id)),
                        score=float(score),
                        source_path=chunk_source,
                        text_preview=str(chunk_text_preview),
                        chunk_index=chunk_data.get("chunk_index"),
                        line_start=chunk_data.get("line_start"),
                        line_end=chunk_data.get("line_end"),
                        char_start=chunk_data.get("char_start"),
                        char_end=chunk_data.get("char_end"),
                    )
                    if best is None or candidate.score > best.score:
                        best = candidate
            if best is not None:
                results.append(best)
            continue

        # --- Single-file entry ---
        text_preview = data.get("text_preview", "")
        source_path = str(data.get("source_path", ""))
        embedding = data.get("embedding", [])

        if query_embedding and embedding:
            score = _cosine_similarity(query_embedding, embedding)
        else:
            search_text = data.get("search_text") or text_preview
            score = _keyword_score(query, search_text, doc_id)

        if score >= min_score:
            results.append(
                SearchResult(
                    doc_id=str(doc_id),
                    score=float(score),
                    source_path=source_path,
                    text_preview=str(text_preview),
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)
    return EmbeddingSearchResponse(
        results=results[:top_k],
        semantic_status=semantic_status,
        semantic_reason=semantic_reason,
        collection=collection,
    )


def load_embedding_eval_cases(path: Path) -> list[EmbeddingEvalCase]:
    cases: list[EmbeddingEvalCase] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                case = _parse_eval_case(payload)
                if case:
                    cases.append(case)
    except OSError:
        return cases
    return cases


def evaluate_embedding_index(
    index_root: Path,
    cases: Iterable[EmbeddingEvalCase],
    *,
    embed_fn: Callable[[str], list[float]] | None = None,
    top_k: int = 5,
    min_score: float = 0.3,
    match_mode: str = "any",
    include_cases: bool = False,
) -> EmbeddingEvalResult:
    result = EmbeddingEvalResult()
    match_mode = (match_mode or "any").strip().lower()

    total_mrr = 0.0
    hit_scores: list[float] = []

    for case in cases:
        if not case.query:
            continue
        result.total += 1
        results = search_embedding_index(
            index_root,
            case.query,
            embed_fn=embed_fn,
            top_k=top_k,
            min_score=min_score,
        )

        hit_rank = None
        hit_score = None
        for idx, item in enumerate(results, start=1):
            if _matches_eval_case(item, case, match_mode):
                hit_rank = idx
                hit_score = item.score
                break

        if hit_rank is not None:
            result.hits += 1
            total_mrr += 1.0 / hit_rank
            if hit_score is not None:
                hit_scores.append(hit_score)
        else:
            result.misses += 1

        if include_cases:
            result.cases.append(
                {
                    "id": case.case_id,
                    "query": case.query,
                    "hit": hit_rank is not None,
                    "rank": hit_rank,
                    "score": hit_score,
                    "results": [res.to_dict() for res in results],
                }
            )

    if result.total:
        result.hit_rate = result.hits / result.total
        result.mrr = total_mrr / result.total
    if hit_scores:
        result.avg_hit_score = sum(hit_scores) / len(hit_scores)
    return result


def _iter_source_files(
    sources: Iterable[Path],
    include_patterns: list[str],
    exclude_patterns: list[str],
    max_files: int | None,
    include_hidden: bool,
) -> Iterable[tuple[Path, Path]]:
    count = 0
    for source in sources:
        root = source.expanduser().resolve()
        if not root.exists():
            continue
        if root.is_file():
            if _matches_patterns(
                root, root.parent, include_patterns, exclude_patterns, include_hidden
            ):
                yield root.parent, root
                count += 1
        else:
            for base, dirs, files in os.walk(root):
                if not include_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in files:
                    path = Path(base) / name
                    if _matches_patterns(path, root, include_patterns, exclude_patterns, include_hidden):
                        yield root, path
                        count += 1
                        if max_files and count >= max_files:
                            return
        if max_files and count >= max_files:
            return


def _parse_eval_case(payload: dict) -> EmbeddingEvalCase | None:
    if not isinstance(payload, dict):
        return None
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return None

    expected_doc_ids: list[str] = []
    expected_paths: list[str] = []
    expected = payload.get("expected")

    if isinstance(expected, str):
        expected_doc_ids.append(expected)
        expected_paths.append(expected)
    elif isinstance(expected, list):
        for item in expected:
            if isinstance(item, str):
                expected_doc_ids.append(item)
                expected_paths.append(item)
            elif isinstance(item, dict):
                doc_id = item.get("doc_id")
                path_contains = item.get("path_contains") or item.get("path")
                if isinstance(doc_id, str):
                    expected_doc_ids.append(doc_id)
                if isinstance(path_contains, str):
                    expected_paths.append(path_contains)

    doc_id = payload.get("expected_doc_id")
    if isinstance(doc_id, str):
        expected_doc_ids.append(doc_id)
    path_contains = payload.get("expected_path_contains") or payload.get("expected_path")
    if isinstance(path_contains, str):
        expected_paths.append(path_contains)

    case_id = payload.get("id")
    metadata = payload.get("metadata")
    meta = metadata if isinstance(metadata, dict) else {}

    return EmbeddingEvalCase(
        query=query.strip(),
        expected_doc_ids=_dedupe_values(expected_doc_ids),
        expected_path_contains=_dedupe_values(expected_paths),
        case_id=str(case_id) if case_id is not None else None,
        metadata={str(k): str(v) for k, v in meta.items()},
    )


def _matches_eval_case(
    result: SearchResult,
    case: EmbeddingEvalCase,
    match_mode: str,
) -> bool:
    if match_mode == "doc_id":
        return _match_doc_id(result.doc_id, case.expected_doc_ids)
    if match_mode == "path":
        return _match_path(result.source_path, case.expected_path_contains)
    return _match_doc_id(result.doc_id, case.expected_doc_ids) or _match_path(
        result.source_path, case.expected_path_contains
    )


def _match_doc_id(doc_id: str, expected: list[str]) -> bool:
    if not expected:
        return False
    for item in expected:
        if doc_id == item:
            return True
    return False


def _match_path(source_path: str, expected: list[str]) -> bool:
    if not expected:
        return False
    lowered = source_path.lower()
    for item in expected:
        if item.lower() in lowered:
            return True
    return False


def _dedupe_values(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for item in items:
        item = item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _matches_patterns(
    path: Path,
    root: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
    include_hidden: bool,
) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.name
    if not include_hidden and any(part.startswith(".") for part in Path(rel).parts):
        return False
    if rel in ("", "."):
        rel = path.name
    if exclude_patterns and any(fnmatch.fnmatch(rel, pattern) for pattern in exclude_patterns):
        return False
    if not include_patterns:
        return True
    return any(fnmatch.fnmatch(rel, pattern) for pattern in include_patterns)


def _read_text_file(path: Path, *, max_bytes: int | None) -> str | None:
    if max_bytes is not None and max_bytes > 0:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
        if len(raw) > max_bytes:
            return None
    else:
        raw = path.read_bytes()
    if b"\x00" in raw:
        return None
    return raw.decode("utf-8", errors="replace")


def _build_doc_id(root: Path, path: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return f"{root}::{rel}"


def _hash_doc_id(doc_id: str) -> str:
    return hashlib.sha1(doc_id.encode("utf-8")).hexdigest()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    size = len(a)
    if size == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(size))
    norm_a = math.sqrt(sum(a[i] * a[i] for i in range(size)))
    norm_b = math.sqrt(sum(b[i] * b[i] for i in range(size)))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_score(query: str, text: str, doc_id: str) -> float:
    query_terms = {term for term in query.lower().split() if term}
    if not query_terms:
        return 0.0
    text_lower = text.lower()
    doc_lower = doc_id.lower()
    score = 0.0
    for term in query_terms:
        if term in text_lower:
            score += 1.0
        if term in doc_lower:
            score += 0.5
    return score / len(query_terms)


def _resolve_torch_device(device: str | None, torch_module) -> str:
    device = (device or "auto").strip().lower()
    if device not in {"auto", "cpu", "cuda", "mps"}:
        return device
    if device == "cpu":
        return "cpu"
    if device == "cuda" and torch_module.cuda.is_available():
        return "cuda"
    if device == "mps":
        mps = getattr(torch_module.backends, "mps", None)
        if mps and mps.is_available():
            return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps and mps.is_available():
        return "mps"
    return "cpu"


def _resolve_hf_token(token: str | None) -> str | None:
    if token:
        return token
    return os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")


register_embedding_backend(
    "ollama",
    lambda model="nomic-embed-text", host="http://localhost:11435", **_: create_ollama_embed_fn(
        model=model,
        host=host,
    ),
)
register_embedding_backend(
    "hf",
    lambda model, device=None, max_tokens=512, pooling="mean", normalize=True, token=None, **_: create_hf_embed_fn(
        model=model,
        device=device,
        max_tokens=max_tokens,
        pooling=pooling,
        normalize=normalize,
        token=token,
    ),
)
register_embedding_backend(
    "openai",
    lambda model, base_url="https://api.openai.com/v1", api_key=None, timeout=30.0, **_: create_openai_embed_fn(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    ),
)
register_embedding_backend(
    "gemini",
    lambda model=DEFAULT_GEMINI_MODEL,
    api_key=None,
    task_type=GEMINI_DOCUMENT_TASK,
    output_dimensionality=DEFAULT_GEMINI_DIMENSION,
    **_: create_gemini_embed_fn(
        model=model,
        api_key=api_key,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
    ),
)
