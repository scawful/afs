"""Embedding indexing and search helpers for AFS."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable


@dataclass
class EmbeddingIndexResult:
    total_files: int = 0
    indexed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"total={self.total_files} indexed={self.indexed} "
            f"skipped={self.skipped} errors={len(self.errors)}"
        )


@dataclass
class SearchResult:
    doc_id: str
    score: float
    source_path: str
    text_preview: str

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "score": self.score,
            "source_path": self.source_path,
            "text_preview": self.text_preview,
        }


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
) -> EmbeddingIndexResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "embedding_index.json"

    index: dict[str, str] = {}
    result = EmbeddingIndexResult()

    include = list(include_patterns) if include_patterns else []
    exclude = list(exclude_patterns) if exclude_patterns else []

    for source_root, path in _iter_source_files(
        sources, include, exclude, max_files, include_hidden
    ):
        result.total_files += 1
        try:
            text = _read_text_file(path, max_bytes=max_bytes)
        except OSError as exc:
            result.skipped += 1
            result.errors.append(f"{path}: {exc}")
            continue
        if text is None:
            result.skipped += 1
            continue
        preview = text[:preview_chars].strip()
        embed_text = text[:embed_chars]
        embedding: list[float] = []
        if embed_fn:
            try:
                embedding = embed_fn(embed_text)
            except Exception as exc:
                result.errors.append(f"{path}: embed failed ({exc})")
                embedding = []

        doc_id = _build_doc_id(source_root, path)
        filename = f"{_hash_doc_id(doc_id)}.json"
        stat = None
        if path.exists():
            try:
                stat = path.stat()
            except OSError:
                stat = None
        payload = {
            "id": doc_id,
            "source_path": str(path),
            "text_preview": preview,
            "embedding": embedding,
            "size_bytes": stat.st_size if stat else 0,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
            if stat
            else None,
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
        result.indexed += 1

    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return result


def search_embedding_index(
    index_root: Path,
    query: str,
    *,
    embed_fn: Callable[[str], list[float]] | None = None,
    top_k: int = 5,
    min_score: float = 0.3,
) -> list[SearchResult]:
    index_path = index_root / "embedding_index.json"
    embeddings_dir = index_root / "embeddings"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid index: {index_path}") from exc

    query_embedding: list[float] | None = None
    if embed_fn:
        try:
            query_embedding = embed_fn(query)
        except Exception:
            query_embedding = None

    results: list[SearchResult] = []
    for doc_id, filename in index.items():
        entry_path = embeddings_dir / filename
        try:
            data = json.loads(entry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        text_preview = data.get("text_preview", "")
        source_path = str(data.get("source_path", ""))
        embedding = data.get("embedding", [])

        if query_embedding and embedding:
            score = _cosine_similarity(query_embedding, embedding)
        else:
            score = _keyword_score(query, text_preview, doc_id)

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
    return results[:top_k]


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


def _matches_patterns(
    path: Path,
    root: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
    include_hidden: bool,
) -> bool:
    if not include_hidden and any(part.startswith(".") for part in path.parts):
        return False
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.name
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
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
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
