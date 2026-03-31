"""Focused AFS health diagnostics used by `afs health`."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    psutil = None

from ..config import load_runtime_config_model
from ..context_paths import resolve_agent_output_root, resolve_mount_root
from ..core import find_root, resolve_context_root
from ..event_log import summarize_mcp_tool_usage
from ..history import iter_history_events
from ..manager import AFSManager
from ..mcp_server import get_mcp_status
from ..models import MountType
from ..plugins import discover_extension_manifests, load_enabled_extensions
from ..profiles import merge_extension_hooks, resolve_active_profile
from ..services.manager import ServiceManager
from .mcp_registration import SUPPORTED_MCP_CLIENTS, find_afs_mcp_registrations

EMBEDDINGS_STALE_SECONDS = 24 * 3600
HOOK_EVENTS = (
    "before_context_read",
    "after_context_write",
    "before_agent_dispatch",
    "session_start",
    "session_end",
    "user_prompt_submit",
    "turn_started",
    "turn_completed",
    "turn_failed",
    "task_created",
    "task_progress",
    "task_completed",
    "task_failed",
)


def collect_afs_health(config_path: Path | None = None) -> dict[str, Any]:
    """Collect AFS-focused health diagnostics."""
    config, _resolved_config_path = load_runtime_config_model(
        config_path=config_path,
        merge_user=True,
        start_dir=Path.cwd(),
    )
    manager = AFSManager(config=config)
    profile = resolve_active_profile(config)
    linked_root = None if config_path else find_root(Path.cwd())
    context_root = resolve_context_root(config, linked_root)

    mounts: dict[str, int] = {}
    total_mounts = 0
    mount_health: dict[str, Any] = {
        "healthy": False,
        "missing_dirs": [],
        "broken_mounts": [],
        "duplicate_mount_sources": [],
        "profile": {
            "name": profile.name,
            "managed_mounts": 0,
            "missing_mounts": [],
            "missing_sources": [],
            "mismatched_mounts": [],
        },
        "suggested_actions": [],
    }
    if context_root.exists():
        try:
            context = manager.list_context(context_path=context_root)
            mounts = {
                mount_type.value: len(items)
                for mount_type, items in context.mounts.items()
            }
            total_mounts = context.total_mounts
            mount_health = manager.context_health(context_root, profile_name=profile.name)
        except Exception:
            mounts = {}

    embeddings = _embedding_index_health(profile.knowledge_mounts)
    mcp_status = get_mcp_status(config_path=config_path)
    extensions = _extension_health(config, profile, mcp_status)
    hooks = _hook_health(config, profile, context_root)
    mcp = _mcp_health(mcp_status, context_root=context_root, config=config)
    maintenance = _maintenance_health(config, context_root)

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
            "mount_health": mount_health,
        },
        "embeddings": embeddings,
        "extensions": extensions,
        "hooks": hooks,
        "mcp": mcp,
        "maintenance": maintenance,
    }


def render_afs_health(snapshot: dict[str, Any]) -> str:
    """Render human-readable health output."""
    profile = snapshot["profile"]
    context = snapshot["context"]
    embeddings = snapshot["embeddings"]
    extensions = snapshot["extensions"]
    hooks = snapshot["hooks"]
    mcp = snapshot["mcp"]
    maintenance = snapshot["maintenance"]

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

    mount_health = context["mount_health"]
    mount_health_parts: list[str] = []
    if mount_health["broken_mounts"]:
        mount_health_parts.append(f"broken={len(mount_health['broken_mounts'])}")
    if mount_health["duplicate_mount_sources"]:
        mount_health_parts.append(
            f"duplicates={len(mount_health['duplicate_mount_sources'])}"
        )
    profile_health = mount_health["profile"]
    if profile_health["missing_mounts"]:
        mount_health_parts.append(
            f"profile_missing={len(profile_health['missing_mounts'])}"
        )
    if profile_health["missing_sources"]:
        mount_health_parts.append(
            f"profile_sources_missing={len(profile_health['missing_sources'])}"
        )
    if profile_health["mismatched_mounts"]:
        mount_health_parts.append(
            f"profile_mismatched={len(profile_health['mismatched_mounts'])}"
        )
    lines.append(
        "mount_health: "
        + (", ".join(mount_health_parts) if mount_health_parts else "ok")
    )
    if mount_health["suggested_actions"]:
        lines.append("mount_actions: " + "; ".join(mount_health["suggested_actions"]))

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
    workflow_usage = mcp["workflow_usage"]
    lines.append(
        "mcp_workflow: "
        f"proactive={str(workflow_usage['proactive']).lower()} "
        + " ".join(
            f"{name}={tool['count']}"
            for name, tool in sorted(workflow_usage["tools"].items())
        )
    )
    warm = maintenance["reports"]["context_warm"]
    watch = maintenance["reports"]["context_watch"]
    supervisor_report = maintenance["reports"]["agent_supervisor"]
    history_memory = maintenance["reports"]["history_memory"]
    doctor_snapshot = maintenance["reports"]["doctor_snapshot"]
    supervisor_audit = maintenance["supervisor"]
    lines.append(
        "maintenance: "
        f"context_warm={warm['status'] or 'unknown'} "
        f"context_watch={watch['status'] or 'unknown'} "
        f"agent_supervisor={supervisor_report['status'] or 'unknown'} "
        f"history_memory={history_memory['status'] or 'unknown'} "
        f"doctor={doctor_snapshot['status'] or 'unknown'} "
        f"age={_format_age(warm['age_seconds'])} "
        f"degraded_contexts={maintenance['degraded_contexts']} "
        f"remapped_mounts={maintenance['remapped_mounts']}"
    )
    lines.append(
        "agents: "
        f"running={supervisor_audit['counts']['running']} "
        f"failed={supervisor_audit['counts']['failed']} "
        f"stopped={supervisor_audit['counts']['stopped']} "
        f"manual_stop={supervisor_audit['counts']['manual_stop']}"
    )
    brief = maintenance["reports"]["gemini_workspace_brief"]
    if brief["available"]:
        lines.append(
            "gemini_workspace_brief: "
            f"status={brief['status'] or 'unknown'} "
            f"age={_format_age(brief['age_seconds'])}"
        )
    service_states = ", ".join(
        f"{name}={state}"
        for name, state in sorted(maintenance["services"].items())
    )
    lines.append(f"maintenance_services: {service_states or 'none'}")
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
    history_root = resolve_mount_root(context_root, MountType.HISTORY, config=config)
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


def _mcp_health(mcp_status: dict[str, Any], *, context_root: Path, config) -> dict[str, Any]:
    running, running_details = _detect_mcp_running()
    registrations = find_afs_mcp_registrations()
    registered_clients = {
        client: bool(registrations.get(client))
        for client in SUPPORTED_MCP_CLIENTS
    }
    registered_client_names = [
        client for client, is_registered in registered_clients.items() if is_registered
    ]
    workflow_usage = summarize_mcp_tool_usage(
        context_root,
        tool_names=(
            "afs.session.bootstrap",
            "context.status",
            "context.diff",
            "context.query",
            "session.pack",
        ),
        config=config,
    )
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
        "workflow_usage": workflow_usage,
    }

def _maintenance_health(config, context_root: Path) -> dict[str, Any]:
    agent_output_dir = resolve_agent_output_root(context_root, config=config)
    reports = {
        "context_warm": _load_agent_report(agent_output_dir / "context_warm.json"),
        "context_watch": _load_agent_report(agent_output_dir / "context_watch.json"),
        "agent_supervisor": _load_agent_report(agent_output_dir / "agent_supervisor.json"),
        "history_memory": _load_agent_report(agent_output_dir / "history_memory.json"),
        "doctor_snapshot": _load_agent_report(agent_output_dir / "doctor_snapshot.json"),
        "gemini_workspace_brief": _load_agent_report(
            agent_output_dir / "gemini_workspace_brief.json"
        ),
    }
    degraded_contexts = 0
    remapped_mounts = 0
    warm_payload = reports["context_warm"]["payload"] if reports["context_warm"]["available"] else {}
    if isinstance(warm_payload, dict):
        audits = warm_payload.get("context_health", [])
        if isinstance(audits, list):
            degraded_contexts = sum(
                1
                for audit in audits
                if isinstance(audit, dict)
                and isinstance(audit.get("mount_health"), dict)
                and not audit["mount_health"].get("healthy", False)
            )
            remapped_mounts = sum(
                len(audit.get("repair", {}).get("remapped_mounts", []))
                for audit in audits
                if isinstance(audit, dict)
            )

    service_manager = ServiceManager(config=config)
    services: dict[str, str] = {}
    for name in (
        "context-warm",
        "context-watch",
        "agent-supervisor",
        "history-memory",
        "gemini-workspace-brief",
    ):
        definition = service_manager.get_definition(name)
        if definition is None:
            continue
        services[name] = service_manager.status(name).state.value

    try:
        from ..agents.supervisor import AgentSupervisor

        supervisor = AgentSupervisor(config=config)
        supervisor_audit = supervisor.audit()
    except Exception:
        supervisor_audit = {
            "state_dir": str(agent_output_dir / "supervisor"),
            "counts": {
                "running": 0,
                "failed": 0,
                "stopped": 0,
                "manual_stop": 0,
                "configured": 0,
            },
            "stale_pid_files": [],
            "agents": [],
        }

    return {
        "reports": reports,
        "services": services,
        "degraded_contexts": degraded_contexts,
        "remapped_mounts": remapped_mounts,
        "supervisor": supervisor_audit,
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


def _load_agent_report(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "available": path.exists(),
        "path": str(path),
        "status": None,
        "started_at": None,
        "finished_at": None,
        "age_seconds": _file_age_seconds(path) if path.exists() else None,
        "metrics": {},
        "notes": [],
        "payload": {},
    }
    if not path.exists():
        return payload
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload["status"] = "invalid"
        return payload

    if not isinstance(raw, dict):
        payload["status"] = "invalid"
        return payload
    payload["status"] = raw.get("status") if isinstance(raw.get("status"), str) else None
    payload["started_at"] = raw.get("started_at") if isinstance(raw.get("started_at"), str) else None
    payload["finished_at"] = raw.get("finished_at") if isinstance(raw.get("finished_at"), str) else None
    if isinstance(raw.get("metrics"), dict):
        payload["metrics"] = raw["metrics"]
    if isinstance(raw.get("notes"), list):
        payload["notes"] = [note for note in raw["notes"] if isinstance(note, str)]
    if isinstance(raw.get("payload"), dict):
        payload["payload"] = raw["payload"]
    return payload


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
