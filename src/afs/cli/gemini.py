"""Gemini integration CLI commands.

Provides setup, status, and context generation for Gemini Code Assist
and the Gemini CLI. Manages settings.json, MCP registration, and
knowledge-based context injection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..config import load_config_model
from ..context_paths import resolve_mount_root
from ..core import resolve_context_root
from ..embeddings import SearchResult, create_embed_fn, search_embedding_index
from ..health.mcp_registration import find_afs_mcp_registrations
from ..models import MountType

# --------------------------------------------------------------------------- #
# settings.json management
# --------------------------------------------------------------------------- #


def _gemini_settings_candidates() -> list[Path]:
    home = Path.home()
    return [
        home / ".gemini" / "settings.json",
        home / ".config" / "gemini" / "settings.json",
    ]


def _find_gemini_settings() -> Path | None:
    """Return the first existing Gemini settings.json, or None."""
    for candidate in _gemini_settings_candidates():
        if candidate.exists():
            return candidate
    return None


def _default_gemini_settings_path() -> Path:
    """Return the preferred path for a new settings.json."""
    return Path.home() / ".gemini" / "settings.json"


def _build_afs_mcp_entry() -> dict[str, Any]:
    """Build the AFS MCP server entry for settings.json."""
    python = sys.executable
    return {
        "command": python,
        "args": ["-m", "afs.mcp_server"],
    }


def _read_settings(path: Path) -> dict[str, Any]:
    """Read and parse a settings.json file."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_settings(path: Path, data: dict[str, Any]) -> None:
    """Write settings.json with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# gemini setup
# --------------------------------------------------------------------------- #


def gemini_setup_command(args: argparse.Namespace) -> int:
    """Set up Gemini integration: settings.json + MCP registration."""
    settings_path = _find_gemini_settings() or _default_gemini_settings_path()
    if args.settings_path:
        settings_path = Path(args.settings_path).expanduser().resolve()

    data = _read_settings(settings_path) if settings_path.exists() else {}

    if "mcpServers" not in data:
        data["mcpServers"] = {}

    mcp_entry = _build_afs_mcp_entry()

    if "afs" in data["mcpServers"]:
        if args.force:
            data["mcpServers"]["afs"] = mcp_entry
            print(f"Updated AFS MCP entry in {settings_path}")
        else:
            print(f"AFS MCP already registered in {settings_path}")
            print("Use --force to overwrite.")
    else:
        data["mcpServers"]["afs"] = mcp_entry
        print(f"Added AFS MCP entry to {settings_path}")

    _write_settings(settings_path, data)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        key_name = (
            "GEMINI_API_KEY" if os.getenv("GEMINI_API_KEY") else "GOOGLE_API_KEY"
        )
        print(f"API key: set (via {key_name})")
    else:
        print(
            "API key: NOT SET — export GEMINI_API_KEY to enable embeddings and generation"
        )

    return 0


# --------------------------------------------------------------------------- #
# gemini status
# --------------------------------------------------------------------------- #


def gemini_status_command(args: argparse.Namespace) -> int:
    """Check Gemini integration status."""
    checks: list[tuple[str, bool, str]] = []
    config = _load_cli_config(args)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    checks.append(("API key", bool(api_key), "GEMINI_API_KEY or GOOGLE_API_KEY"))

    try:
        from google import genai  # type: ignore[import-untyped]  # noqa: F401

        sdk_ok = True
    except ImportError:
        sdk_ok = False
    checks.append(("google-genai SDK", sdk_ok, "pip install google-genai"))

    settings_path = _find_gemini_settings()
    checks.append(
        ("settings.json", settings_path is not None, str(settings_path or "not found"))
    )

    registrations = find_afs_mcp_registrations()
    gemini_registered = bool(registrations.get("gemini"))
    checks.append(
        (
            "MCP registered",
            gemini_registered,
            ", ".join(registrations.get("gemini", [])) or "not registered",
        )
    )

    indexed_roots = _indexed_knowledge_roots(_candidate_knowledge_roots(args, config))
    has_index = bool(indexed_roots)
    if indexed_roots:
        total_docs = sum(_index_doc_count(root) for root in indexed_roots)
        index_info = (
            f"{len(indexed_roots)} index roots, {total_docs} docs "
            f"under {indexed_roots[0]}"
        )
    else:
        index_info = "no embedding index found"
    checks.append(("Embeddings indexed", has_index, index_info))

    embed_ok = True
    embed_info = "skipped"
    if api_key and sdk_ok and not args.skip_ping:
        try:
            from ..embeddings import create_gemini_embed_fn

            fn = create_gemini_embed_fn()
            result = fn("test")
            embed_ok = True
            embed_info = f"{len(result)}-dim vectors"
        except Exception as exc:  # pragma: no cover - live API path
            embed_ok = False
            embed_info = f"failed: {exc}"
    checks.append(("Embedding ping", embed_ok, embed_info))

    if args.json:
        payload = {
            "checks": [
                {"name": name, "ok": ok, "detail": detail}
                for name, ok, detail in checks
            ]
        }
        print(json.dumps(payload, indent=2))
        return 0

    all_ok = True
    for name, ok, detail in checks:
        icon = "ok" if ok else "MISSING"
        print(f"  [{icon:>7s}] {name}: {detail}")
        if not ok:
            all_ok = False

    return 0 if all_ok else 1


# --------------------------------------------------------------------------- #
# gemini context
# --------------------------------------------------------------------------- #


def gemini_context_command(args: argparse.Namespace) -> int:
    """Generate a context document from knowledge base for Gemini sessions."""
    config = _load_cli_config(args)
    accessible_roots = _candidate_knowledge_roots(args, config)
    indexed_roots = _indexed_knowledge_roots(accessible_roots)

    if not accessible_roots:
        print("No accessible knowledge mount found.")
        return 1

    if args.query:
        if not indexed_roots:
            print("No embedding index found. Run: afs embeddings index first.")
            return 1
        return _context_search(indexed_roots, args)

    for root in accessible_roots:
        if (root / "INDEX.md").exists():
            return _context_full(root)
    return _context_full(accessible_roots[-1])


def _context_search(knowledge_roots: Iterable[Path], args: argparse.Namespace) -> int:
    """Generate context by searching one or more indexed knowledge roots."""
    embed_fn = None
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            embed_fn = create_embed_fn(
                "gemini",
                model="gemini-embedding-001",
                api_key=api_key,
                task_type="RETRIEVAL_QUERY",
            )
        except (RuntimeError, ValueError):
            pass

    results = _search_across_knowledge_roots(
        knowledge_roots,
        args.query,
        embed_fn=embed_fn,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    if not results:
        print("No relevant documents found.")
        return 0

    if args.json:
        payload = {"query": args.query, "documents": []}
        for result in results:
            doc: dict[str, Any] = {
                "doc_id": result.doc_id,
                "score": result.score,
                "source_path": result.source_path,
            }
            if args.include_content:
                try:
                    content = Path(result.source_path).read_text(encoding="utf-8")
                    doc["content"] = content
                except OSError:
                    doc["content"] = result.text_preview
            payload["documents"].append(doc)
        print(json.dumps(payload, indent=2))
        return 0

    print(f"# Context for: {args.query}\n")
    for result in results:
        rel_path = (
            result.doc_id.split("::")[-1] if "::" in result.doc_id else result.doc_id
        )
        print(f"## {rel_path} (score: {result.score:.3f})\n")
        if args.include_content:
            try:
                content = Path(result.source_path).read_text(encoding="utf-8")
                print(content)
            except OSError:
                print(result.text_preview)
        else:
            print(result.text_preview)
        print()

    return 0


def _context_full(knowledge_root: Path) -> int:
    """Generate full context summary from knowledge INDEX.md."""
    index_md = knowledge_root / "INDEX.md"
    if index_md.exists():
        print(index_md.read_text(encoding="utf-8"))
        return 0

    md_files = sorted(knowledge_root.rglob("*.md"))
    if not md_files:
        print("No knowledge documents found.")
        return 1

    print(f"# Knowledge Base ({len(md_files)} documents)\n")
    for path in md_files:
        rel = path.relative_to(knowledge_root)
        size = path.stat().st_size
        print(f"- `{rel}` ({size:,} bytes)")

    return 0


def _load_cli_config(args: argparse.Namespace):
    config_path = getattr(args, "config", None)
    resolved_config_path = (
        Path(config_path).expanduser().resolve() if config_path else None
    )
    return load_config_model(config_path=resolved_config_path, merge_user=True)


def _candidate_knowledge_roots(args: argparse.Namespace, config) -> list[Path]:
    if getattr(args, "knowledge_path", None):
        base_root = Path(args.knowledge_path).expanduser().resolve()
    else:
        context_override = getattr(args, "context_root", None)
        context_root = (
            Path(context_override).expanduser().resolve()
            if context_override
            else resolve_context_root(config, None)
        )
        base_root = resolve_mount_root(context_root, MountType.KNOWLEDGE, config=config)
        project = getattr(args, "project", None)
        if project:
            base_root = base_root / project
    return _expand_knowledge_roots(base_root)


def _expand_knowledge_roots(base_root: Path) -> list[Path]:
    resolved_root = base_root.expanduser().resolve()
    if not resolved_root.exists():
        return []

    roots = [resolved_root]
    try:
        for child in sorted(resolved_root.iterdir()):
            if child.is_dir():
                roots.append(child.resolve())
    except OSError:
        pass
    return roots


def _indexed_knowledge_roots(roots: Iterable[Path]) -> list[Path]:
    indexed: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        marker = str(root)
        if marker in seen:
            continue
        seen.add(marker)
        if (root / "embedding_index.json").exists():
            indexed.append(root)
    return indexed


def _index_doc_count(index_root: Path) -> int:
    try:
        payload = json.loads(
            (index_root / "embedding_index.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return 0
    return len(payload) if isinstance(payload, list) else 0


def _search_across_knowledge_roots(
    knowledge_roots: Iterable[Path],
    query: str,
    *,
    embed_fn,
    top_k: int,
    min_score: float,
) -> list[SearchResult]:
    merged: dict[tuple[str, str], SearchResult] = {}
    for root in knowledge_roots:
        try:
            results = search_embedding_index(
                root,
                query,
                embed_fn=embed_fn,
                top_k=top_k,
                min_score=min_score,
            )
        except (OSError, ValueError):
            continue
        for result in results:
            key = (result.doc_id, result.source_path)
            previous = merged.get(key)
            if previous is None or result.score > previous.score:
                merged[key] = result
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:top_k]


# --------------------------------------------------------------------------- #
# Parser registration
# --------------------------------------------------------------------------- #


def _add_knowledge_root_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Config path.")
    parser.add_argument("--context-root", help="Context root override.")
    parser.add_argument("--knowledge-path", help="Knowledge root override.")
    parser.add_argument("--project", help="Knowledge project under the context root.")


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register gemini command parsers."""
    gemini_parser = subparsers.add_parser(
        "gemini", help="Gemini integration management."
    )
    gemini_sub = gemini_parser.add_subparsers(dest="gemini_command")

    setup = gemini_sub.add_parser(
        "setup", help="Set up Gemini settings.json and MCP registration."
    )
    setup.add_argument(
        "--settings-path",
        help="Override settings.json path (default: ~/.gemini/settings.json).",
    )
    setup.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing AFS MCP entry.",
    )
    setup.set_defaults(func=gemini_setup_command)

    status = gemini_sub.add_parser(
        "status", help="Check Gemini integration health."
    )
    _add_knowledge_root_args(status)
    status.add_argument("--json", action="store_true", help="JSON output.")
    status.add_argument(
        "--skip-ping",
        action="store_true",
        help="Skip live embedding ping test.",
    )
    status.set_defaults(func=gemini_status_command)

    ctx = gemini_sub.add_parser(
        "context",
        help="Generate context from knowledge base for Gemini sessions.",
    )
    _add_knowledge_root_args(ctx)
    ctx.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query to find relevant context (omit for full index).",
    )
    ctx.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to include (default: 5).",
    )
    ctx.add_argument(
        "--min-score",
        type=float,
        default=0.3,
        help="Minimum relevance score (default: 0.3).",
    )
    ctx.add_argument(
        "--include-content",
        action="store_true",
        help="Include full document content (not just preview).",
    )
    ctx.add_argument("--json", action="store_true", help="JSON output.")
    ctx.set_defaults(func=gemini_context_command)
