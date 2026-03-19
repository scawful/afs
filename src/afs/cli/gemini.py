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
from pathlib import Path
from typing import Any

from ..config import load_config_model
from ..context_paths import resolve_mount_root
from ..core import resolve_context_root
from ..health.mcp_registration import find_afs_mcp_registrations
from ..models import MountType
from ..profiles import resolve_active_profile


# --------------------------------------------------------------------------- #
# settings.json management
# --------------------------------------------------------------------------- #

_GEMINI_SETTINGS_CANDIDATES = [
    Path.home() / ".gemini" / "settings.json",
    Path.home() / ".config" / "gemini" / "settings.json",
]


def _find_gemini_settings() -> Path | None:
    """Return the first existing Gemini settings.json, or None."""
    for candidate in _GEMINI_SETTINGS_CANDIDATES:
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

    # Ensure mcpServers section exists
    if "mcpServers" not in data:
        data["mcpServers"] = {}

    # Register AFS MCP server
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

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        print(f"API key: set (via {'GEMINI_API_KEY' if os.getenv('GEMINI_API_KEY') else 'GOOGLE_API_KEY'})")
    else:
        print("API key: NOT SET — export GEMINI_API_KEY to enable embeddings and generation")

    return 0


# --------------------------------------------------------------------------- #
# gemini status
# --------------------------------------------------------------------------- #

def gemini_status_command(args: argparse.Namespace) -> int:
    """Check Gemini integration status."""
    checks: list[tuple[str, bool, str]] = []

    # 1. API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    checks.append(("API key", bool(api_key), "GEMINI_API_KEY or GOOGLE_API_KEY"))

    # 2. google-genai SDK
    try:
        from google import genai  # type: ignore[import-untyped]  # noqa: F401
        sdk_ok = True
    except ImportError:
        sdk_ok = False
    checks.append(("google-genai SDK", sdk_ok, "pip install google-genai"))

    # 3. Settings.json
    settings_path = _find_gemini_settings()
    checks.append(("settings.json", settings_path is not None, str(settings_path or "not found")))

    # 4. MCP registration
    registrations = find_afs_mcp_registrations()
    gemini_registered = bool(registrations.get("gemini"))
    checks.append((
        "MCP registered",
        gemini_registered,
        ", ".join(registrations.get("gemini", [])) or "not registered",
    ))

    # 5. Embeddings index
    config = load_config_model(merge_user=True)
    profile = resolve_active_profile(config)
    has_index = False
    index_info = "no knowledge mounts"
    for mount in profile.knowledge_mounts:
        mount_path = Path(mount).expanduser()
        idx = mount_path / "embedding_index.json"
        if idx.exists():
            try:
                index_data = json.loads(idx.read_text(encoding="utf-8"))
                count = len(index_data)
                has_index = True
                index_info = f"{count} docs at {mount_path}"
            except (OSError, json.JSONDecodeError):
                pass
    checks.append(("Embeddings indexed", has_index, index_info))

    # 6. Test embedding (if API key and SDK available)
    embed_ok = True  # treat skipped as OK
    embed_info = "skipped"
    if api_key and sdk_ok and not args.skip_ping:
        try:
            from ..embeddings import create_gemini_embed_fn
            fn = create_gemini_embed_fn()
            result = fn("test")
            embed_ok = True
            embed_info = f"{len(result)}-dim vectors"
        except Exception as exc:
            embed_info = f"failed: {exc}"
    checks.append(("Embedding ping", embed_ok, embed_info))

    # Print results
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
    config = load_config_model(merge_user=True)
    profile = resolve_active_profile(config)

    if not profile.knowledge_mounts:
        print("No knowledge mounts in active profile.")
        return 1

    # Find knowledge roots with embedding indexes, falling back to any accessible mount
    indexed_roots: list[Path] = []
    accessible_roots: list[Path] = []
    for mount in profile.knowledge_mounts:
        mount_path = Path(mount).expanduser()
        if mount_path.exists():
            accessible_roots.append(mount_path)
            if (mount_path / "embedding_index.json").exists():
                indexed_roots.append(mount_path)

    if not accessible_roots:
        print("No accessible knowledge mount found.")
        return 1

    if args.query:
        # Semantic/keyword search mode — prefer indexed mounts
        knowledge_root = indexed_roots[0] if indexed_roots else accessible_roots[0]
        if not indexed_roots:
            print(f"No embedding index found. Run: afs embeddings index first.")
            return 1
        return _context_search(knowledge_root, args)
    else:
        # Full context mode — prefer mount with INDEX.md, then any with .md files
        for root in accessible_roots:
            if (root / "INDEX.md").exists():
                return _context_full(root, args)
        return _context_full(accessible_roots[-1], args)


def _context_search(knowledge_root: Path, args: argparse.Namespace) -> int:
    """Generate context by searching the knowledge base."""
    from ..embeddings import build_embedding_index, search_embedding_index, create_embed_fn

    # Check if index exists
    index_path = knowledge_root / "embedding_index.json"
    if not index_path.exists():
        print(f"No embedding index at {knowledge_root}. Run: afs embeddings index first.")
        return 1

    # Try to use Gemini embeddings for search, fall back to keyword
    embed_fn = None
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            embed_fn = create_embed_fn("gemini", model="gemini-embedding-001", api_key=api_key)
        except (RuntimeError, ValueError):
            pass

    results = search_embedding_index(
        knowledge_root,
        args.query,
        embed_fn=embed_fn,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    if not results:
        print("No relevant documents found.")
        return 0

    if args.json:
        payload = {
            "query": args.query,
            "documents": [],
        }
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

    # Markdown output for Gemini context injection
    print(f"# Context for: {args.query}\n")
    for result in results:
        rel_path = result.doc_id.split("::")[-1] if "::" in result.doc_id else result.doc_id
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


def _context_full(knowledge_root: Path, args: argparse.Namespace) -> int:
    """Generate full context summary from knowledge INDEX.md."""
    index_md = knowledge_root / "INDEX.md"
    if index_md.exists():
        content = index_md.read_text(encoding="utf-8")
        print(content)
        return 0

    # Fallback: list all .md files
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


# --------------------------------------------------------------------------- #
# Parser registration
# --------------------------------------------------------------------------- #

def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register gemini command parsers."""
    gemini_parser = subparsers.add_parser(
        "gemini", help="Gemini integration management."
    )
    gemini_sub = gemini_parser.add_subparsers(dest="gemini_command")

    # afs gemini setup
    setup = gemini_sub.add_parser(
        "setup", help="Set up Gemini settings.json and MCP registration."
    )
    setup.add_argument(
        "--settings-path",
        help="Override settings.json path (default: ~/.gemini/settings.json).",
    )
    setup.add_argument(
        "--force", action="store_true",
        help="Overwrite existing AFS MCP entry.",
    )
    setup.set_defaults(func=gemini_setup_command)

    # afs gemini status
    status = gemini_sub.add_parser(
        "status", help="Check Gemini integration health."
    )
    status.add_argument("--json", action="store_true", help="JSON output.")
    status.add_argument(
        "--skip-ping", action="store_true",
        help="Skip live embedding ping test.",
    )
    status.set_defaults(func=gemini_status_command)

    # afs gemini context
    ctx = gemini_sub.add_parser(
        "context",
        help="Generate context from knowledge base for Gemini sessions.",
    )
    ctx.add_argument(
        "query", nargs="?", default=None,
        help="Search query to find relevant context (omit for full index).",
    )
    ctx.add_argument(
        "--top-k", type=int, default=5,
        help="Number of documents to include (default: 5).",
    )
    ctx.add_argument(
        "--min-score", type=float, default=0.3,
        help="Minimum relevance score (default: 0.3).",
    )
    ctx.add_argument(
        "--include-content", action="store_true",
        help="Include full document content (not just preview).",
    )
    ctx.add_argument("--json", action="store_true", help="JSON output.")
    ctx.set_defaults(func=gemini_context_command)
