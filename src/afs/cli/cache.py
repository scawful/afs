"""Cache management CLI commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._utils import load_manager, resolve_context_paths


def cache_clear_command(args: argparse.Namespace) -> int:
    """Clear session pack cache files."""
    from ..context_pack import clear_pack_cache, _resolve_session_pack_cache_dir

    config_path = Path(args.config).expanduser().resolve() if getattr(args, "config", None) else None
    manager = load_manager(config_path)

    context_path: Path | None = None
    if getattr(args, "path", None) or getattr(args, "context_root", None):
        _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
            args, manager
        )

    removed = clear_pack_cache(context_path, config=manager.config)
    use_json = getattr(args, "json", False)
    if use_json:
        print(json.dumps({
            "removed": removed,
            "context_path": str(context_path) if context_path else None,
            "cache_dir": str(_resolve_session_pack_cache_dir(manager.config)),
        }))
    else:
        scope = f" for {context_path}" if context_path else ""
        print(f"Cleared {removed} session pack cache file(s){scope}.")
    return 0


def cache_status_command(args: argparse.Namespace) -> int:
    """Show session pack cache status."""
    from ..context_pack import _resolve_session_pack_cache_dir

    config_path = Path(args.config).expanduser().resolve() if getattr(args, "config", None) else None
    manager = load_manager(config_path)

    cache_cfg = manager.config.session_pack_cache
    cache_dir = _resolve_session_pack_cache_dir(manager.config)
    count = 0
    total_bytes = 0
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.json"):
            if cache_file.is_file():
                count += 1
                try:
                    total_bytes += cache_file.stat().st_size
                except OSError:
                    pass

    use_json = getattr(args, "json", False)
    if use_json:
        print(json.dumps({
            "enabled": cache_cfg.enabled,
            "ttl_seconds": cache_cfg.ttl_seconds,
            "cache_dir": str(cache_dir),
            "files": count,
            "total_bytes": total_bytes,
        }))
    else:
        status = "enabled" if cache_cfg.enabled else "disabled"
        print(f"Session pack cache: {status}")
        print(f"TTL: {cache_cfg.ttl_seconds}s")
        print(f"Directory: {cache_dir}")
        print(f"Files: {count} ({total_bytes} bytes)")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register cache management command parsers."""
    cache_parser = subparsers.add_parser("cache", help="Manage session pack cache.")
    cache_sub = cache_parser.add_subparsers(dest="cache_command")

    # cache clear
    clear_parser = cache_sub.add_parser("clear", help="Clear session pack cache files.")
    clear_parser.add_argument("--config", help="Config path.")
    clear_parser.add_argument("--path", help="Project path (scope clear to this context).")
    clear_parser.add_argument("--context-root", help="Context root override.")
    clear_parser.add_argument("--context-dir", help="Context directory name.")
    clear_parser.add_argument("--json", action="store_true", help="Output JSON.")
    clear_parser.set_defaults(func=cache_clear_command)

    # cache status
    status_parser = cache_sub.add_parser("status", help="Show cache status.")
    status_parser.add_argument("--config", help="Config path.")
    status_parser.add_argument("--json", action="store_true", help="Output JSON.")
    status_parser.set_defaults(func=cache_status_command)
