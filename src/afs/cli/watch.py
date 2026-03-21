"""Watch context directories for changes and trigger index rebuilds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import MountType
from ._utils import load_manager, resolve_context_paths


def _collect_watch_paths(manager, context_path: Path) -> list[Path]:
    """Resolve knowledge, tools, scratchpad mount roots for watching."""
    paths: list[Path] = []
    seen: set[str] = set()
    for mount_type in (MountType.KNOWLEDGE, MountType.TOOLS, MountType.SCRATCHPAD):
        try:
            root = manager.resolve_mount_root(context_path, mount_type)
        except Exception:
            continue
        key = str(root)
        if key in seen or not root.exists():
            continue
        seen.add(key)
        paths.append(root)
    return paths


def _snapshot_paths(paths: list[Path]) -> dict[str, tuple[float, int]]:
    """Take a stat snapshot of all files under watched paths."""
    snapshot: dict[str, tuple[float, int]] = {}
    for root in paths:
        if not root.exists():
            continue
        try:
            children = sorted(root.iterdir(), key=lambda item: item.name)
        except OSError:
            continue
        for child in children:
            if child.name == ".keep":
                continue
            if child.is_file():
                try:
                    stat = child.stat()
                    snapshot[str(child)] = (stat.st_mtime, stat.st_size)
                except OSError:
                    continue
            if not child.is_dir():
                continue

            if child.is_symlink():
                try:
                    scan_root = child.resolve()
                except OSError:
                    continue
                prefix = child.name
            else:
                scan_root = child
                prefix = ""

            if not scan_root.exists() or not scan_root.is_dir():
                continue

            for entry in scan_root.rglob("*"):
                if entry.name == ".keep" or not entry.is_file():
                    continue
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                if prefix:
                    key = str(root / prefix / entry.relative_to(scan_root))
                else:
                    key = str(entry)
                snapshot[key] = (stat.st_mtime, stat.st_size)
    return snapshot


def _diff_snapshots(
    previous: dict[str, tuple[float, int]],
    current: dict[str, tuple[float, int]],
) -> list[str]:
    """Return list of changed file paths."""
    changed: list[str] = []
    all_keys = set(previous) | set(current)
    for key in sorted(all_keys):
        if previous.get(key) != current.get(key):
            changed.append(key)
    return changed


def _trigger_actions(
    manager,
    context_path: Path,
    changed: list[str],
    *,
    on_change: str | None = None,
) -> dict[str, Any]:
    """Rebuild index and optionally run on-change command."""
    result: dict[str, Any] = {
        "changed_files": len(changed),
        "index_rebuild": None,
        "on_change": None,
        "hivemind_notified": False,
    }

    # Rebuild index
    try:
        rebuild_result = manager.rebuild_context_index(context_path)
        result["index_rebuild"] = rebuild_result
    except Exception as exc:
        result["index_rebuild"] = {"error": str(exc)}

    # Send hivemind notification
    try:
        from ..hivemind import HivemindBus

        bus = HivemindBus(context_path, config=manager.config)
        bus.send(
            "afs-watch",
            "status",
            {"changed_count": len(changed), "files": changed[:10]},
            topic="context:repair",
        )
        result["hivemind_notified"] = True
    except Exception:
        pass

    # Run on-change command
    if on_change:
        try:
            proc = subprocess.run(
                on_change,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            result["on_change"] = {
                "returncode": proc.returncode,
                "stdout": proc.stdout[:500] if proc.stdout else "",
                "stderr": proc.stderr[:500] if proc.stderr else "",
            }
        except Exception as exc:
            result["on_change"] = {"error": str(exc)}

    return result


def _watch_with_watchfiles(
    paths: list[Path],
    *,
    debounce_seconds: float = 2.0,
) -> list[str]:
    """Use watchfiles library for efficient file watching. Returns changed paths."""
    from watchfiles import watch

    watcher = watch(
        *[str(p) for p in paths],
        debounce=max(int(debounce_seconds * 1000), 100),
        yield_on_timeout=True,
        rust_timeout=30000,
    )
    changes = next(watcher)
    return [str(path) for _change_type, path in changes] if changes else []


def _watch_with_polling(
    paths: list[Path],
    previous_snapshot: dict[str, tuple[float, int]],
    *,
    poll_seconds: float = 30.0,
) -> tuple[list[str], dict[str, tuple[float, int]]]:
    """Fallback stat-based polling. Returns (changed_paths, new_snapshot)."""
    time.sleep(poll_seconds)
    current = _snapshot_paths(paths)
    changed = _diff_snapshots(previous_snapshot, current)
    return changed, current


def watch_command(args: argparse.Namespace) -> int:
    """Watch context directories for changes."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    watch_paths = _collect_watch_paths(manager, context_path)
    if not watch_paths:
        print("No watchable paths found.", file=sys.stderr)
        return 1

    debounce = args.debounce
    on_change = args.on_change

    # Check for watchfiles availability
    has_watchfiles = False
    try:
        import watchfiles  # noqa: F401

        has_watchfiles = True
    except ImportError:
        pass

    if not has_watchfiles:
        print(
            json.dumps(
                {
                    "event": "start",
                    "mode": "polling",
                    "paths": [str(p) for p in watch_paths],
                    "poll_seconds": debounce,
                }
            ),
            flush=True,
        )
    else:
        print(
            json.dumps(
                {
                    "event": "start",
                    "mode": "watchfiles",
                    "paths": [str(p) for p in watch_paths],
                    "debounce_seconds": debounce,
                }
            ),
            flush=True,
        )

    snapshot = _snapshot_paths(watch_paths)

    try:
        while True:
            if has_watchfiles:
                changed = _watch_with_watchfiles(
                    watch_paths, debounce_seconds=debounce
                )
            else:
                changed, snapshot = _watch_with_polling(
                    watch_paths, snapshot, poll_seconds=debounce
                )

            if not changed:
                continue

            actions = _trigger_actions(
                manager, context_path, changed, on_change=on_change
            )
            event = {
                "event": "change",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "changed_count": len(changed),
                "changed_files": changed[:20],
                "actions": actions,
            }
            print(json.dumps(event), flush=True)

            # Update snapshot for polling mode
            if not has_watchfiles:
                snapshot = _snapshot_paths(watch_paths)

    except KeyboardInterrupt:
        print(
            json.dumps(
                {
                    "event": "stop",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ),
            flush=True,
        )
        return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register watch command parser."""
    watch_parser = subparsers.add_parser(
        "watch",
        help="Watch context directories for changes and trigger index rebuilds.",
    )
    watch_parser.add_argument("--config", help="Config path.")
    watch_parser.add_argument("--path", help="Project path.")
    watch_parser.add_argument("--context-root", help="Context root override.")
    watch_parser.add_argument("--context-dir", help="Context directory name.")
    watch_parser.add_argument(
        "--debounce",
        type=float,
        default=30.0,
        help="Debounce/poll interval in seconds (default: 30).",
    )
    watch_parser.add_argument(
        "--on-change",
        help="Shell command to run after each change batch.",
    )
    watch_parser.set_defaults(func=watch_command)
