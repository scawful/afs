"""Embedding index CLI commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config_model
from ..context_paths import resolve_mount_root
from ..core import resolve_context_root
from ..embeddings import (
    build_embedding_index,
    create_embed_fn,
    evaluate_embedding_index,
    load_embedding_eval_cases,
    search_embedding_index,
)
from ..models import MountType


def embeddings_index_command(args: argparse.Namespace) -> int:
    """Build an embedding index for a source path."""
    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    try:
        index_root = _resolve_knowledge_root(args, config)
    except ValueError as exc:
        print(str(exc))
        return 1

    sources = _resolve_sources(args, index_root)
    if not sources:
        print("No sources provided.")
        return 1

    embed_fn = _resolve_embed_fn(args)
    if embed_fn is None and args.provider != "none":
        return 1

    result = build_embedding_index(
        sources,
        index_root,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        max_files=args.max_files,
        preview_chars=args.preview_chars,
        embed_chars=args.embed_chars,
        max_bytes=args.max_bytes,
        embed_fn=embed_fn,
        include_hidden=args.include_hidden,
    )

    if args.json:
        payload = {
            "index_root": str(index_root),
            "sources": [str(path) for path in sources],
            "summary": result.summary(),
            "total_files": result.total_files,
            "indexed": result.indexed,
            "skipped": result.skipped,
            "errors": result.errors,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"index_root: {index_root}")
    print(result.summary())
    if result.errors:
        for error in result.errors[:10]:
            print(f"error: {error}")
        if len(result.errors) > 10:
            print(f"errors: {len(result.errors)} (truncated)")
    return 0


def embeddings_search_command(args: argparse.Namespace) -> int:
    """Search an embedding index."""
    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    try:
        index_root = _resolve_knowledge_root(args, config)
    except ValueError as exc:
        print(str(exc))
        return 1

    embed_fn = _resolve_embed_fn(args)
    if embed_fn is None and args.provider != "none":
        return 1

    try:
        results = search_embedding_index(
            index_root,
            args.query,
            embed_fn=embed_fn,
            top_k=args.top_k,
            min_score=args.min_score,
        )
    except (OSError, ValueError) as exc:
        print(str(exc))
        return 1

    if args.json:
        payload = {
            "index_root": str(index_root),
            "query": args.query,
            "results": [result.to_dict() for result in results],
        }
        print(json.dumps(payload, indent=2))
        return 0

    for result in results:
        print(f"{result.score:.3f}\t{result.doc_id}\t{result.source_path}")
        if args.preview and result.text_preview:
            print(result.text_preview)
    return 0


def embeddings_eval_command(args: argparse.Namespace) -> int:
    """Evaluate embedding retrieval quality."""
    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    try:
        index_root = _resolve_knowledge_root(args, config)
    except ValueError as exc:
        print(str(exc))
        return 1

    query_path = Path(args.query_file).expanduser().resolve()
    cases = load_embedding_eval_cases(query_path)
    if not cases:
        print(f"No eval cases loaded from {query_path}")
        return 1

    embed_fn = _resolve_embed_fn(args)
    if embed_fn is None and args.provider != "none":
        return 1

    result = evaluate_embedding_index(
        index_root,
        cases,
        embed_fn=embed_fn,
        top_k=args.top_k,
        min_score=args.min_score,
        match_mode=args.match,
        include_cases=args.details,
    )

    if args.json:
        payload = {
            "index_root": str(index_root),
            "query_file": str(query_path),
            "summary": {
                "total": result.total,
                "hits": result.hits,
                "misses": result.misses,
                "hit_rate": result.hit_rate,
                "mrr": result.mrr,
                "avg_hit_score": result.avg_hit_score,
            },
            "cases": result.cases if args.details else [],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"index_root: {index_root}")
    print(result.summary())
    if args.details and result.cases:
        for case in result.cases:
            status = "hit" if case.get("hit") else "miss"
            rank = case.get("rank")
            label = f"{status}"
            if rank:
                label += f"@{rank}"
            print(f"{label}\t{case.get('query')}")
    return 0


def _resolve_knowledge_root(args: argparse.Namespace, config) -> Path:
    knowledge_path = args.knowledge_path or getattr(args, "knowledge_dir", None)
    if knowledge_path:
        return Path(knowledge_path).expanduser().resolve()
    if not args.project:
        raise ValueError("Missing --project or --knowledge-path/--knowledge-dir")
    if args.context_root:
        context_root = Path(args.context_root).expanduser().resolve()
    else:
        context_root = resolve_context_root(config, None)
    return resolve_mount_root(context_root, MountType.KNOWLEDGE, config=config) / args.project


def _resolve_sources(args: argparse.Namespace, index_root: Path) -> list[Path]:
    sources: list[Path] = []
    raw_sources = args.source or []
    for source in raw_sources:
        sources.append(Path(source).expanduser().resolve())

    if not sources and (args.knowledge_path or getattr(args, "knowledge_dir", None)):
        sources.append(index_root)

    return sources


def _resolve_embed_fn(args: argparse.Namespace):
    if args.provider == "none":
        return None

    kwargs: dict[str, object] = {"model": args.model}

    if args.provider == "ollama":
        kwargs["host"] = args.host
    elif args.provider == "hf":
        kwargs["device"] = args.hf_device
        kwargs["max_tokens"] = args.hf_max_tokens
        kwargs["pooling"] = args.hf_pooling
        kwargs["normalize"] = not args.hf_no_normalize
    elif args.provider == "openai":
        kwargs["base_url"] = args.openai_base_url
        kwargs["api_key"] = args.openai_api_key

    try:
        return create_embed_fn(args.provider, **kwargs)
    except (RuntimeError, ValueError) as exc:
        print(str(exc))
        return None


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register embeddings command parsers."""
    emb_parser = subparsers.add_parser(
        "embeddings", help="Embedding index management."
    )
    emb_sub = emb_parser.add_subparsers(dest="embeddings_command")

    def add_provider_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--provider",
            choices=["none", "ollama", "hf", "openai"],
            default="none",
            help="Embedding provider.",
        )
        parser.add_argument(
            "--model",
            default="nomic-embed-text",
            help="Embedding model (provider-specific).",
        )
        parser.add_argument(
            "--host", default="http://localhost:11435", help="Ollama host."
        )
        parser.add_argument(
            "--hf-device",
            default="auto",
            help="HF device (auto, cpu, cuda, mps).",
        )
        parser.add_argument(
            "--hf-max-tokens",
            type=int,
            default=512,
            help="HF max token length.",
        )
        parser.add_argument(
            "--hf-pooling",
            choices=["mean", "cls"],
            default="mean",
            help="HF pooling strategy.",
        )
        parser.add_argument(
            "--hf-no-normalize",
            action="store_true",
            help="Disable L2 normalization for HF embeddings.",
        )
        parser.add_argument(
            "--openai-base-url",
            default="https://api.openai.com/v1",
            help="OpenAI-compatible embeddings base URL.",
        )
        parser.add_argument("--openai-api-key", help="OpenAI-compatible API key override.")

    emb_index = emb_sub.add_parser("index", help="Build embedding index.")
    emb_index.add_argument("--config", help="Config path.")
    emb_index.add_argument("--project", help="Project name for knowledge index.")
    emb_index.add_argument("--knowledge-path", help="Knowledge base root override.")
    emb_index.add_argument("--knowledge-dir", help="Alias for --knowledge-path.")
    emb_index.add_argument("--context-root", help="Context root override.")
    emb_index.add_argument(
        "--source",
        action="append",
        help="Source path to index (repeatable). Defaults to knowledge dir.",
    )
    emb_index.add_argument(
        "--include",
        action="append",
        help="Glob patterns to include (repeatable).",
    )
    emb_index.add_argument(
        "--exclude",
        action="append",
        help="Glob patterns to exclude (repeatable).",
    )
    emb_index.add_argument("--max-files", type=int, default=10000, help="Max files.")
    emb_index.add_argument(
        "--preview-chars", type=int, default=1000, help="Preview length."
    )
    emb_index.add_argument(
        "--embed-chars", type=int, default=2000, help="Embedding text length."
    )
    emb_index.add_argument("--max-bytes", type=int, default=2_000_000, help="Max bytes.")
    add_provider_args(emb_index)
    emb_index.add_argument(
        "--include-hidden", action="store_true", help="Include hidden files."
    )
    emb_index.add_argument("--json", action="store_true", help="Output JSON.")
    emb_index.set_defaults(func=embeddings_index_command)

    emb_search = emb_sub.add_parser("search", help="Search embedding index.")
    emb_search.add_argument("--config", help="Config path.")
    emb_search.add_argument("--project", help="Project name for knowledge index.")
    emb_search.add_argument("--knowledge-path", help="Knowledge base root override.")
    emb_search.add_argument("--knowledge-dir", help="Alias for --knowledge-path.")
    emb_search.add_argument("--context-root", help="Context root override.")
    emb_search.add_argument("query", help="Search query.")
    emb_search.add_argument("--top-k", type=int, default=5, help="Top results.")
    emb_search.add_argument("--min-score", type=float, default=0.3, help="Min score.")
    add_provider_args(emb_search)
    emb_search.add_argument("--preview", action="store_true", help="Show preview.")
    emb_search.add_argument("--json", action="store_true", help="Output JSON.")
    emb_search.set_defaults(func=embeddings_search_command)

    emb_eval = emb_sub.add_parser("eval", help="Evaluate retrieval quality.")
    emb_eval.add_argument("--config", help="Config path.")
    emb_eval.add_argument("--project", help="Project name for knowledge index.")
    emb_eval.add_argument("--knowledge-path", help="Knowledge base root override.")
    emb_eval.add_argument("--knowledge-dir", help="Alias for --knowledge-path.")
    emb_eval.add_argument("--context-root", help="Context root override.")
    emb_eval.add_argument("--query-file", required=True, help="JSONL eval file.")
    emb_eval.add_argument("--top-k", type=int, default=5, help="Top results.")
    emb_eval.add_argument("--min-score", type=float, default=0.3, help="Min score.")
    emb_eval.add_argument(
        "--match",
        choices=["any", "doc_id", "path"],
        default="any",
        help="Match mode for expected targets.",
    )
    add_provider_args(emb_eval)
    emb_eval.add_argument(
        "--details", action="store_true", help="Include per-case details."
    )
    emb_eval.add_argument("--json", action="store_true", help="Output JSON.")
    emb_eval.set_defaults(func=embeddings_eval_command)
