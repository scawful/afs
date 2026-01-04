"""Embedding index CLI commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config_model
from ..core import resolve_context_root
from ..embeddings import (
    build_embedding_index,
    create_ollama_embed_fn,
    search_embedding_index,
)


def embeddings_index_command(args: argparse.Namespace) -> int:
    """Build an embedding index for a source path."""
    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    try:
        index_root = _resolve_knowledge_root(args, config)
    except ValueError as exc:
        print(str(exc))
        return 1
    sources = [Path(source).expanduser().resolve() for source in args.source]
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


def _resolve_knowledge_root(args: argparse.Namespace, config) -> Path:
    if args.knowledge_path:
        return Path(args.knowledge_path).expanduser().resolve()
    if not args.project:
        raise ValueError("Missing --project or --knowledge-path")
    if args.context_root:
        context_root = Path(args.context_root).expanduser().resolve()
    else:
        context_root = resolve_context_root(config, None)
    return context_root / "knowledge" / args.project


def _resolve_embed_fn(args: argparse.Namespace):
    if args.provider == "none":
        return None
    if args.provider == "ollama":
        try:
            return create_ollama_embed_fn(model=args.model, host=args.host)
        except RuntimeError as exc:
            print(str(exc))
            return None
    print(f"Unknown provider: {args.provider}")
    return None


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register embeddings command parsers."""
    emb_parser = subparsers.add_parser(
        "embeddings", help="Embedding index management."
    )
    emb_sub = emb_parser.add_subparsers(dest="embeddings_command")

    emb_index = emb_sub.add_parser("index", help="Build embedding index.")
    emb_index.add_argument("--config", help="Config path.")
    emb_index.add_argument("--project", help="Project name for knowledge index.")
    emb_index.add_argument("--knowledge-path", help="Knowledge base root override.")
    emb_index.add_argument("--context-root", help="Context root override.")
    emb_index.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source path to index (repeatable).",
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
    emb_index.add_argument(
        "--provider",
        choices=["none", "ollama"],
        default="none",
        help="Embedding provider.",
    )
    emb_index.add_argument("--model", default="nomic-embed-text", help="Ollama model.")
    emb_index.add_argument(
        "--host", default="http://localhost:11435", help="Ollama host."
    )
    emb_index.add_argument(
        "--include-hidden", action="store_true", help="Include hidden files."
    )
    emb_index.add_argument("--json", action="store_true", help="Output JSON.")
    emb_index.set_defaults(func=embeddings_index_command)

    emb_search = emb_sub.add_parser("search", help="Search embedding index.")
    emb_search.add_argument("--config", help="Config path.")
    emb_search.add_argument("--project", help="Project name for knowledge index.")
    emb_search.add_argument("--knowledge-path", help="Knowledge base root override.")
    emb_search.add_argument("--context-root", help="Context root override.")
    emb_search.add_argument("query", help="Search query.")
    emb_search.add_argument("--top-k", type=int, default=5, help="Top results.")
    emb_search.add_argument("--min-score", type=float, default=0.3, help="Min score.")
    emb_search.add_argument(
        "--provider",
        choices=["none", "ollama"],
        default="none",
        help="Embedding provider.",
    )
    emb_search.add_argument("--model", default="nomic-embed-text", help="Ollama model.")
    emb_search.add_argument(
        "--host", default="http://localhost:11435", help="Ollama host."
    )
    emb_search.add_argument("--preview", action="store_true", help="Show preview.")
    emb_search.add_argument("--json", action="store_true", help="Output JSON.")
    emb_search.set_defaults(func=embeddings_search_command)
