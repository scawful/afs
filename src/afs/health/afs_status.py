"""Focused AFS health diagnostics used by `afs health`."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    psutil = None

from ..config import load_config_model
from ..core import find_root, resolve_context_root
from ..history import iter_history_events
from ..manager import AFSManager
from ..mcp_server import get_mcp_status
from ..monorepo_bridge import (
    DEFAULT_ACTIVE_WORKSPACE_STALE_SECONDS,
    get_workspace_bridge_status,
)
from ..plugins import discover_extension_manifests, load_enabled_extensions
from ..profiles import merge_extension_hooks, resolve_active_profile
from .mcp_registration import SUPPORTED_MCP_CLIENTS, find_afs_mcp_registrations

EMBEDDINGS_STALE_SECONDS = 24 * 3600
HOOK_EVENTS = (
    "before_context_read",
    "after_context_write",
    "before_agent_dispatch",
)


def collect_afs_health(config_path: Path | None = None) -> dict[str, Any]:
    """Collect AFS-focused health diagnostics."""
    config = load_config_model(config_path=config_path, merge_user=True)
    manager = AFSManager(config=config)
    profile = resolve_active_profile(config)
    linked_root = None if config_path else find_root(Path.cwd())
    context_root = resolve_context_root(config, linked_root)

    mounts: dict[str, int] = {}
    total_mounts = 0
    if context_root.exists():
        try:
            context = manager.list_context(context_path=context_root)
            mounts = {
                mount_type.value: len(items)
                for mount_type, items in context.mounts.items()
            }
            total_mounts = context.total_mounts
        except Exception:
            mounts = {}

    bridge_status = get_workspace_bridge_status(context_root, manager=manager)
    embeddings = _embedding_index_health(profile.knowledge_mounts)
    mcp_status = get_mcp_status(config_path=config_path)
    extensions = _extension_health(config, profile, mcp_status)
    hooks = _hook_health(config, profile, context_root)
    mcp = _mcp_health(mcp_status)

    return {
        "profile": {
            "active": profile.name,
            "configured": config.profiles.active_profile,
            "policies": profile.policies,
            "extensions": profile.enabled_extensions,
        },
        "context": {
            "path": str(context_root),
            "exists": context_root.exists(),
            "total_mounts": total_mounts,
            "mounts": mounts,
        },
        "monorepo_bridge": {
            "path": str(bridge_status.path),
            "exists": bridge_status.exists,
            "age_seconds": bridge_status.age_seconds,
            "stale": bridge_status.stale,
            "stale_after_seconds": DEFAULT_ACTIVE_WORKSPACE_STALE_SECONDS,
            "modified_at": bridge_status.modified_at,
        },
        "embeddings": embeddings,
        "extensions": extensions,
        "hooks": hooks,
        "mcp": mcp,
    }


def render_afs_health(snapshot: dict[str, Any]) -> str:
    """Render human-readable health output."""
    profile = snapshot["profile"]
    context = snapshot["context"]
    bridge = snapshot["monorepo_bridge"]
    embeddings = snapshot["embeddings"]
    extensions = snapshot["extensions"]
    hooks = snapshot["hooks"]
    mcp = snapshot["mcp"]

    lines = []
    lines.append("AFS Health")
    lines.append(f"profile: {profile['active']} (configured={profile['configured']})")
    lines.append(f"context: {context['path']} (exists={str(context['exists']).lower()})")
    lines.append(f"mounts: total={context['total_mounts']}")
    if context["mounts"]:
        mount_pairs = ", ".join(
            f"{name}={count}" for name, count in sorted(context["mounts"].items())
        )
        lines.append(f"mount_breakdown: {mount_pairs}")
    else:
        lines.append("mount_breakdown: (none)")

    bridge_age = _format_age(bridge["age_seconds"])
    lines.append(
        "monorepo_bridge: "
        f"exists={str(bridge['exists']).lower()} "
        f"stale={str(bridge['stale']).lower()} "
        f"age={bridge_age}"
    )

    lines.append(
        "embeddings: "
        f"indices={embeddings['count']} stale={embeddings['stale_count']} "
        f"newest={_format_age(embeddings['newest_age_seconds'])}"
    )

    lines.append(
        "extensions: "
        f"enabled={len(extensions['enabled'])} loaded={len(extensions['loaded'])} "
        f"missing={len(extensions['missing'])} mcp_errors={len(extensions['mcp_load_errors'])}"
    )

    hook_parts = []
    for event, info in hooks["events"].items():
        hook_parts.append(
            f"{event}:registered={info['registered_count']} "
            f"last_run={info['last_run'] or 'never'}"
        )
    lines.append("hooks: " + " | ".join(hook_parts))

    lines.append(
        "mcp: "
        f"running={str(mcp['running']).lower()} "
        f"registered_clients={','.join(mcp['registered_client_names']) or 'none'} "
        f"tools={len(mcp['tools'])}"
    )
    for client in SUPPORTED_MCP_CLIENTS:
        hits = mcp["config_hits"].get(client, [])
        if hits:
            lines.append(f"{client}_config_hits: " + ", ".join(hits))
    return "\n".join(lines)


def _embedding_index_health(knowledge_mounts: list[Path]) -> dict[str, Any]:
    indices: list[Path] = []
    seen: set[str] = set()

    for root in knowledge_mounts:
        resolved = root.expanduser().resolve()
        if not resolved.exists():
            continue
        candidates: list[Path] = []
        candidates.append(resolved / "embedding_index.json")
        try:
            for child in resolved.iterdir():
                if child.is_dir():
                    candidates.append(child / "embedding_index.json")
        except OSError:
            pass
        for candidate in candidates:
            marker = str(candidate)
            if candidate.exists() and marker not in seen:
                seen.add(marker)
                indices.append(candidate)

    ages: list[float] = []
    now = datetime.now(timezone.utc)
    stale = 0
    for index in indices:
        age = _file_age_seconds(index, now=now)
        if age is None:
            continue
        ages.append(age)
        if age > EMBEDDINGS_STALE_SECONDS:
            stale += 1

    newest = min(ages) if ages else None
    oldest = max(ages) if ages else None
    return {
        "count": len(indices),
        "stale_count": stale,
        "newest_age_seconds": newest,
        "oldest_age_seconds": oldest,
        "paths": [str(path) for path in indices],
        "stale_after_seconds": EMBEDDINGS_STALE_SECONDS,
    }


def _extension_health(
    config,
    profile,
    mcp_status: dict[str, Any],
) -> dict[str, Any]:
    discovered = discover_extension_manifests(config=config)
    enabled = sorted(
        set(config.extensions.enabled_extensions).union(profile.enabled_extensions)
    )
    loaded = load_enabled_extensions(config=config, requested=enabled) if enabled else {}
    loaded_names = sorted(loaded.keys())
    missing = sorted([name for name in enabled if name not in loaded_names])

    mcp_errors = mcp_status.get("load_errors", {})

    return {
        "enabled": enabled,
        "loaded": loaded_names,
        "missing": missing,
        "discovered": sorted(discovered.keys()),
        "mcp_load_errors": mcp_errors,
    }


def _hook_health(config, profile, context_root: Path) -> dict[str, Any]:
    history_root = context_root / "history"
    last_run: dict[str, str | None] = dict.fromkeys(HOOK_EVENTS, None)
    if history_root.exists():
        for event in iter_history_events(
            history_root,
            event_types={"hook"},
            include_payloads=False,
        ):
            op = event.get("op")
            timestamp = event.get("timestamp")
            if not isinstance(op, str) or op not in last_run:
                continue
            if not isinstance(timestamp, str):
                continue
            previous = last_run.get(op)
            if previous is None or timestamp > previous:
                last_run[op] = timestamp

    by_event: dict[str, dict[str, Any]] = {}
    for event in HOOK_EVENTS:
        commands = merge_extension_hooks(config, profile, event)
        by_event[event] = {
            "registered_count": len(commands),
            "registered": commands,
            "last_run": last_run.get(event),
        }

    return {"events": by_event}


def _mcp_health(mcp_status: dict[str, Any]) -> dict[str, Any]:
    running, running_details = _detect_mcp_running()
    registrations = find_afs_mcp_registrations()
    registered_clients = {
        client: bool(registrations.get(client))
        for client in SUPPORTED_MCP_CLIENTS
    }
    registered_client_names = [
        client for client, is_registered in registered_clients.items() if is_registered
    ]
    return {
        "running": running,
        "running_details": running_details,
        "registered_clients": registered_clients,
        "registered_client_names": registered_client_names,
        "config_hits": registrations,
        "registered_with_gemini": registered_clients.get("gemini", False),
        "gemini_config_hits": registrations.get("gemini", []),
        "registered_with_claude": registered_clients.get("claude", False),
        "claude_config_hits": registrations.get("claude", []),
        "registered_with_codex": registered_clients.get("codex", False),
        "codex_config_hits": registrations.get("codex", []),
        "tools": mcp_status.get("tools", []),
        "extension_status": mcp_status.get("extension_status", []),
        "load_errors": mcp_status.get("load_errors", {}),
    }


def _detect_mcp_running() -> tuple[bool, list[str]]:
    matches: list[str] = []
    if psutil is not None:
        try:
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    cmdline = proc.info.get("cmdline") or []
                    joined = " ".join(cmdline)
                except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, OSError):
                    continue
                if _looks_like_afs_mcp_command(joined):
                    matches.append(f"pid={proc.info.get('pid')}")
        except (psutil.AccessDenied, PermissionError, OSError):
            matches = []
        if matches:
            return True, matches

    try:
        result = subprocess.run(
            ["pgrep", "-af", "afs"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False, matches

    if result.returncode == 0:
        pids: list[str] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            pid, _, command = line.partition(" ")
            if pid and _looks_like_afs_mcp_command(command):
                pids.append(pid)
        if pids:
            matches.extend([f"pid={pid}" for pid in pids])
            return True, matches
    return False, matches


def _looks_like_afs_mcp_command(command: str) -> bool:
    normalized = command.strip()
    if not normalized:
        return False
    if "afs.mcp_server" in normalized:
        return True
    if " -m afs mcp serve" in f" {normalized}":
        return True
    return "afs mcp serve" in normalized


def _file_age_seconds(path: Path, *, now: datetime | None = None) -> float | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    current = now or datetime.now(timezone.utc)
    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return max((current - modified).total_seconds(), 0.0)


def _format_age(age_seconds: float | None) -> str:
    if age_seconds is None:
        return "n/a"
    if age_seconds < 60:
        return f"{int(age_seconds)}s"
    minutes = age_seconds / 60
    if minutes < 60:
        return f"{int(minutes)}m"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    days = hours / 24
    return f"{days:.1f}d"
