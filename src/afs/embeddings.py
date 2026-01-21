"""Embedding indexing and search helpers for AFS."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


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
