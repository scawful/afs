"""Build a proactive AFS session bootstrap summary for agents and humans."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context_freshness import (
    MountFreshness,
    context_diff_since_session,
    mount_freshness,
    save_context_snapshot,
)
from .context_index import ContextSQLiteIndex, count_mount_files
from .context_paths import resolve_agent_output_root, resolve_mount_root
from .manager import AFSManager
from .models import MountType

logger = logging.getLogger(__name__)

SESSION_BOOTSTRAP_JSON = "session_bootstrap.json"
SESSION_BOOTSTRAP_MARKDOWN = "session_bootstrap.md"
_MAX_TEXT_CHARS = 1500
_MAX_LIST_ITEMS = 8


def _estimate_tokens(text: str) -> int:
    """Approximate token count cheaply for bootstrap budgeting."""
    if not text or not text.strip():
        return 0
    return max(1, (len(text) + 3) // 4)

# Priority ordering for token budgeting (highest priority first).
# Sections at the top survive truncation; sections at the bottom are
# dropped first when the bootstrap exceeds the token budget.
_SECTION_PRIORITY = [
    "handoff",          # critical: what happened last session
    "scratchpad",       # high: current working state
    "tasks",            # high: pending work items
    "session_changes",  # medium-high: what changed since last session
    "diff",             # medium: recent index drift
    "mount_freshness",  # medium: per-mount freshness scores
    "memory",           # medium: durable context
    "hivemind",         # low: cross-agent messages
    "agent_reports",    # low: background agent status
]


def collect_context_status(manager: AFSManager, context_path: Path) -> dict[str, Any]:
    """Return the same context status summary used by MCP and session bootstrap."""
    context_path = context_path.expanduser().resolve()
    settings = manager.config.context_index
    mount_health = manager.context_health(context_path)

    mount_counts: dict[str, int] = {}
    total_files = 0
    for mount_type in MountType:
        mount_dir = manager.resolve_mount_root(context_path, mount_type)
        if not mount_dir.exists():
            continue
        count = count_mount_files(mount_dir)
        if count > 0:
            mount_counts[mount_type.value] = count
            total_files += count

    index_info: dict[str, Any] = {"enabled": settings.enabled}
    if settings.enabled:
        db_path = manager.resolve_mount_root(context_path, MountType.GLOBAL) / settings.db_filename
        if db_path.exists():
            try:
                index = ContextSQLiteIndex(manager, context_path)
                has_entries = index.has_entries()
                index_info["built"] = True
                index_info["has_entries"] = has_entries
                index_info["total_entries"] = index.total_entries
                index_info["stale"] = index.needs_refresh() if has_entries else False
                index_info["db_size_bytes"] = db_path.stat().st_size
                index_info["db_path"] = str(db_path)
            except Exception:
                index_info["error"] = "failed to read index"
        else:
            index_info["built"] = False

    return {
        "context_path": str(context_path),
        "profile": manager.config.profiles.active_profile,
        "mount_counts": mount_counts,
        "total_files": total_files,
        "mount_health": mount_health,
        "actions": list(mount_health.get("suggested_actions", [])),
        "index": index_info,
    }


def collect_context_diff(
    manager: AFSManager,
    context_path: Path,
    *,
    mount_types: list[MountType] | None = None,
    item_limit: int = _MAX_LIST_ITEMS,
) -> dict[str, Any]:
    """Return a trimmed diff summary for bootstrap and MCP."""
    context_path = context_path.expanduser().resolve()
    settings = manager.config.context_index
    if not settings.enabled:
        return {
            "context_path": str(context_path),
            "available": False,
            "error": "index disabled",
            "added": [],
            "modified": [],
            "deleted": [],
            "total_changes": 0,
        }

    index = ContextSQLiteIndex(manager, context_path)
    if not index.has_entries(mount_types=mount_types):
        return {
            "context_path": str(context_path),
            "available": False,
            "error": "index empty — run context.index.rebuild first",
            "added": [],
            "modified": [],
            "deleted": [],
            "total_changes": 0,
        }

    diff = index.diff(mount_types=mount_types)
    trimmed = {
        "context_path": diff["context_path"],
        "available": True,
        "error": "",
        "total_changes": diff["total_changes"],
        "added": diff["added"][:item_limit],
        "modified": diff["modified"][:item_limit],
        "deleted": diff["deleted"][:item_limit],
    }
    truncated_counts = {
        "added": max(0, len(diff["added"]) - len(trimmed["added"])),
        "modified": max(0, len(diff["modified"]) - len(trimmed["modified"])),
        "deleted": max(0, len(diff["deleted"]) - len(trimmed["deleted"])),
    }
    is_truncated = any(v > 0 for v in truncated_counts.values())
    trimmed["truncated"] = truncated_counts
    trimmed["is_truncated"] = is_truncated
    if is_truncated:
        omitted = sum(truncated_counts.values())
        logger.info(
            "context diff truncated: %d of %d total changes omitted (limit=%d)",
            omitted,
            diff["total_changes"],
            item_limit,
        )
    return trimmed


def build_session_bootstrap(
    manager: AFSManager,
    context_path: Path,
    *,
    task_limit: int = 10,
    message_limit: int = 10,
    agent_name: str = "cli",
    record_event: bool = True,
    token_budget: int = 0,
) -> dict[str, Any]:
    """Build a structured startup packet for a context-aware agent session.

    When *token_budget* > 0, sections are truncated from lowest priority
    first until the rendered bootstrap fits within the budget.  Priority
    ordering (highest first): handoff, scratchpad, tasks, diff, memory,
    hivemind, agent_reports.  The status header and startup_sequence are
    always included.
    """
    context_path = context_path.expanduser().resolve()
    manager.register_agent(agent_name, context_path)
    context = manager.list_context(context_path=context_path)
    status = collect_context_status(manager, context_path)
    diff = collect_context_diff(manager, context_path)
    scratchpad = _collect_scratchpad(manager, context_path)
    tasks = _collect_tasks(context_path, limit=task_limit)
    hivemind = _collect_hivemind(context_path, limit=message_limit)
    memory = _collect_memory(manager, context_path)
    reports = _collect_agent_reports(manager, context_path)

    handoff = _collect_latest_handoff(context_path, config=manager.config)

    # Compute per-mount freshness (filesystem-based, no DB needed)
    decay_hours = manager.config.context_index.decay_hours
    freshness_map: dict[str, MountFreshness] = {}
    stale_mounts: list[str] = []
    try:
        freshness_map = mount_freshness(
            context_path,
            config=manager.config,
            decay_hours=decay_hours,
        )
        stale_mounts = [
            name for name, mf in freshness_map.items() if mf.stale
        ]
    except Exception:
        pass

    # Fall back to index-based freshness when filesystem scan is empty
    if not freshness_map:
        try:
            index = ContextSQLiteIndex(manager, context_path)
            if index.has_entries():
                idx_freshness = index.freshness_scores(decay_hours=decay_hours)
                stale_mounts = [
                    mount for mount, score in idx_freshness["mount_scores"].items()
                    if score < 0.3
                ]
        except Exception:
            pass

    # Session-aware diff: compare against previous snapshot if one exists
    session_changes: dict[str, Any] = {"available": False}
    try:
        session_diff = context_diff_since_session(
            context_path, config=manager.config
        )
        if session_diff is not None:
            session_changes = {
                "available": True,
                **session_diff.to_dict(),
            }
    except Exception:
        pass

    summary = {
        "context_path": str(context_path),
        "project": context.project_name,
        "profile": status["profile"],
        "startup_sequence": [
            "Review context health and recent drift first.",
            "Read scratchpad state and deferred notes before editing.",
            "Check pending tasks and recent hivemind messages for handoffs.",
            "Use context.query before asking for context that may already be in memory or knowledge.",
            "Write updates back to scratchpad, tasks, or hivemind before handoff.",
        ],
        "status": status,
        "diff": diff,
        "scratchpad": scratchpad,
        "tasks": tasks,
        "hivemind": hivemind,
        "memory": memory,
        "agent_reports": reports,
        "handoff": handoff,
        "stale_mounts": stale_mounts,
        "mount_freshness": {
            k: v.to_dict() for k, v in freshness_map.items()
        },
        "session_changes": session_changes,
    }
    summary["recommended_actions"] = _build_recommendations(summary)

    # Apply token budget truncation if requested
    if token_budget > 0:
        summary = _apply_token_budget(summary, token_budget)

    if record_event:
        try:
            from .history import log_session_event
            log_session_event(
                "bootstrap",
                metadata={"context_path": str(context_path)},
                context_root=context_path,
            )
        except Exception:
            pass

        # Save a context snapshot for the next session to diff against
        import os
        session_id = os.getenv("AFS_SESSION_ID", "").strip()
        if not session_id:
            session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            save_context_snapshot(
                context_path, session_id, config=manager.config
            )
        except Exception:
            pass
    return summary


def _apply_token_budget(summary: dict[str, Any], budget: int) -> dict[str, Any]:
    """Truncate low-priority sections until the summary fits within budget."""
    rendered = json.dumps(summary, default=str)
    current_tokens = _estimate_tokens(rendered)
    if current_tokens <= budget:
        return summary

    # Record what was truncated for observability
    truncated_sections: list[str] = []

    # Walk sections from lowest priority to highest, replacing with stubs
    for section_key in reversed(_SECTION_PRIORITY):
        if current_tokens <= budget:
            break
        if section_key not in summary:
            continue
        section_data = summary[section_key]
        # Skip sections that are already minimal
        section_json = json.dumps(section_data, default=str)
        section_tokens = _estimate_tokens(section_json)
        if section_tokens < 20:
            continue
        summary[section_key] = {"truncated": True, "reason": "token_budget"}
        truncated_sections.append(section_key)
        rendered = json.dumps(summary, default=str)
        current_tokens = _estimate_tokens(rendered)

    if truncated_sections:
        summary["_budget_info"] = {
            "token_budget": budget,
            "estimated_tokens": current_tokens,
            "truncated_sections": truncated_sections,
        }
    return summary


def render_session_bootstrap(summary: dict[str, Any]) -> str:
    """Render a bootstrap packet as markdown/text for CLI and MCP prompts."""
    status = summary["status"]
    diff = summary["diff"]
    scratchpad = summary["scratchpad"]
    tasks = summary["tasks"]
    hivemind = summary["hivemind"]
    memory = summary["memory"]
    reports = summary["agent_reports"]

    lines = [
        f"# AFS Session Bootstrap: {summary['project']}",
        f"Context: {summary['context_path']}",
        f"Profile: {summary['profile']}",
        "",
        "## Startup Sequence",
    ]
    for index, step in enumerate(summary["startup_sequence"], start=1):
        lines.append(f"{index}. {step}")

    lines.extend(["", "## Context Health"])
    lines.append(
        f"- mounts: {status['total_files']} files across {len(status['mount_counts'])} active mount types"
    )
    healthy = status["mount_health"].get("healthy", False)
    lines.append(f"- mount_health: {'healthy' if healthy else 'needs repair'}")
    index_info = status["index"]
    if not index_info.get("enabled", False):
        lines.append("- index: disabled")
    elif not index_info.get("built", index_info.get("has_entries", False)):
        lines.append("- index: not built")
    else:
        stale = index_info.get("stale")
        stale_text = "stale" if stale else "fresh"
        lines.append(
            f"- index: {index_info.get('total_entries', 0)} entries, {stale_text}"
        )
    if status["actions"]:
        lines.append("- suggested_actions:")
        for action in status["actions"]:
            lines.append(f"  - {action}")

    stale_mounts = summary.get("stale_mounts", [])
    if stale_mounts:
        lines.append(f"- stale_mounts: {', '.join(stale_mounts)}")

    mount_freshness_data = summary.get("mount_freshness", {})
    if mount_freshness_data:
        lines.extend(["", "## Mount Freshness"])
        for mt_name, mf_data in sorted(mount_freshness_data.items()):
            score = mf_data.get("freshness_score", 0)
            stale_flag = " [STALE]" if mf_data.get("stale") else ""
            iso = mf_data.get("newest_mtime_iso", "n/a")
            lines.append(
                f"- {mt_name}: score={score:.2f}, files={mf_data.get('file_count', 0)}, newest={iso}{stale_flag}"
            )

    session_changes = summary.get("session_changes", {})
    if session_changes.get("available"):
        lines.extend(["", "## Changes Since Last Session"])
        lines.append(f"- {session_changes.get('change_summary', 'unknown')}")
        per_mount = session_changes.get("per_mount_summary", {})
        if per_mount:
            lines.append("- per_mount:")
            for mt_name, counts in sorted(per_mount.items()):
                parts = [f"{k}={v}" for k, v in sorted(counts.items()) if v > 0]
                if parts:
                    lines.append(f"  - {mt_name}: {', '.join(parts)}")

    lines.extend(["", "## Recent Drift"])
    if diff["available"]:
        lines.append(f"- total_changes: {diff['total_changes']}")
        if diff.get("is_truncated"):
            lines.append("- truncated: true (some changes omitted)")
        for label in ("added", "modified", "deleted"):
            items = diff[label]
            if items:
                lines.append(f"- {label}:")
                for item in items:
                    lines.append(f"  - {item['mount_type']}/{item['relative_path']}")
                extra = diff["truncated"].get(label, 0)
                if extra:
                    lines.append(f"  - ... and {extra} more")
    else:
        lines.append(f"- unavailable: {diff['error']}")

    lines.extend(["", "## Scratchpad"])
    lines.append(f"- path: {scratchpad['path']}")
    if scratchpad["state_text"]:
        lines.append("- state:")
        lines.extend(_indent_block(scratchpad["state_text"]))
    if scratchpad["deferred_text"]:
        lines.append("- deferred:")
        lines.extend(_indent_block(scratchpad["deferred_text"]))
    if scratchpad["other_files"]:
        lines.append("- other_files:")
        for name in scratchpad["other_files"]:
            lines.append(f"  - {name}")
    if scratchpad.get("agent_namespaces"):
        lines.append("- agent_namespaces:")
        for ns in scratchpad["agent_namespaces"]:
            lines.append(f"  - {ns['agent_name']}: {ns['file_count']} files, {ns['size_bytes']} bytes")
    if not scratchpad["state_text"] and not scratchpad["deferred_text"] and not scratchpad["other_files"]:
        lines.append("- empty")

    lines.extend(["", "## Tasks"])
    lines.append(f"- total: {tasks['total']}")
    if tasks["counts"]:
        counts_line = ", ".join(f"{name}={count}" for name, count in sorted(tasks["counts"].items()))
        lines.append(f"- counts: {counts_line}")
    if tasks["items"]:
        lines.append("- top_items:")
        for item in tasks["items"]:
            assigned = f" -> {item['assigned_to']}" if item.get("assigned_to") else ""
            lines.append(
                f"  - [{item['status']}] p{item['priority']} {item['title']}{assigned}"
            )

    lines.extend(["", "## Hivemind"])
    lines.append(f"- recent_messages: {hivemind['recent_count']}")
    if hivemind["messages"]:
        for message in hivemind["messages"]:
            to_part = f" -> {message['to']}" if message.get("to") else ""
            lines.append(
                f"  - {message['timestamp'][:19]} [{message['type']}] {message['from']}{to_part}"
            )

    lines.extend(["", "## Durable Memory"])
    lines.append(f"- entries_count: {memory['entries_count']}")
    manifest = memory.get("memory_manifest", [])
    if manifest:
        lines.append(f"- topics ({len(manifest)}):")
        for topic in manifest:
            lines.append(
                f"  - {topic['topic']}: {topic['entry_count']} entries, latest {topic['latest'][:10]}"
            )
    if memory["latest_markdown_path"]:
        lines.append(f"- latest_summary: {memory['latest_markdown_path']}")
        if memory["latest_markdown_excerpt"]:
            lines.extend(_indent_block(memory["latest_markdown_excerpt"], bullet="- excerpt:"))
    else:
        lines.append("- latest_summary: none")

    lines.extend(["", "## Agent Reports"])
    for report in reports["reports"]:
        age = report["age_seconds"]
        age_label = f"{age}s" if age is not None else "n/a"
        lines.append(
            f"- {report['name']}: {report['status'] or 'unknown'} age={age_label}"
        )

    handoff = summary.get("handoff", {})
    if handoff.get("available"):
        lines.extend(["", "## Latest Handoff"])
        lines.append(f"- session_id: {handoff.get('session_id', '')}")
        lines.append(f"- agent: {handoff.get('agent_name', '')}")
        lines.append(f"- timestamp: {handoff.get('timestamp', '')}")
        if handoff.get("accomplished"):
            lines.append("- accomplished:")
            for item in handoff["accomplished"]:
                lines.append(f"  - {item}")
        if handoff.get("blocked"):
            lines.append("- blocked:")
            for item in handoff["blocked"]:
                lines.append(f"  - {item}")
        if handoff.get("next_steps"):
            lines.append("- next_steps:")
            for item in handoff["next_steps"]:
                lines.append(f"  - {item}")

    lines.extend(["", "## Recommended Actions"])
    for action in summary["recommended_actions"]:
        lines.append(f"- {action}")

    artifact_paths = summary.get("artifact_paths") or {}
    if artifact_paths:
        lines.extend(["", "## Artifact Paths"])
        for label, value in artifact_paths.items():
            lines.append(f"- {label}: {value}")

    return "\n".join(lines)


def write_session_bootstrap_artifacts(
    manager: AFSManager,
    context_path: Path,
    summary: dict[str, Any],
) -> dict[str, str]:
    """Persist the latest bootstrap snapshot for wrappers and handoff tools."""
    output_root = resolve_agent_output_root(context_path, config=manager.config)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / SESSION_BOOTSTRAP_JSON
    markdown_path = output_root / SESSION_BOOTSTRAP_MARKDOWN

    payload = dict(summary)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["artifact_paths"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    markdown = render_session_bootstrap(payload)

    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    return payload["artifact_paths"]


def _collect_scratchpad(manager: AFSManager, context_path: Path) -> dict[str, Any]:
    scratchpad_root = resolve_mount_root(context_path, MountType.SCRATCHPAD, config=manager.config)
    state_text = _read_text(scratchpad_root / "state.md")
    deferred_text = _read_text(scratchpad_root / "deferred.md")
    other_files: list[str] = []
    if scratchpad_root.exists():
        try:
            for candidate in sorted(scratchpad_root.iterdir()):
                if not candidate.is_file():
                    continue
                if candidate.name in {"state.md", "deferred.md"}:
                    continue
                other_files.append(candidate.name)
                if len(other_files) >= _MAX_LIST_ITEMS:
                    break
        except OSError:
            pass

    agent_namespaces: list[dict[str, Any]] = []
    agents_dir = scratchpad_root / "agents"
    if agents_dir.exists():
        try:
            for agent_dir in sorted(agents_dir.iterdir()):
                if not agent_dir.is_dir() or agent_dir.name.startswith("."):
                    continue
                files: list[str] = []
                size_bytes = 0
                for f in sorted(agent_dir.rglob("*")):
                    if f.is_file():
                        files.append(str(f.relative_to(agent_dir)))
                        try:
                            size_bytes += f.stat().st_size
                        except OSError:
                            pass
                agent_namespaces.append({
                    "agent_name": agent_dir.name,
                    "file_count": len(files),
                    "size_bytes": size_bytes,
                    "files": files[:_MAX_LIST_ITEMS],
                })
        except OSError:
            pass

    return {
        "path": str(scratchpad_root),
        "state_text": state_text,
        "deferred_text": deferred_text,
        "other_files": other_files,
        "agent_namespaces": agent_namespaces,
    }


def _collect_tasks(context_path: Path, *, limit: int) -> dict[str, Any]:
    try:
        from .tasks import TaskQueue

        queue = TaskQueue(context_path)
        tasks = queue.list()
    except Exception as exc:
        return {"total": 0, "counts": {}, "items": [], "error": str(exc)}

    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.status] = counts.get(task.status, 0) + 1
    return {
        "total": len(tasks),
        "counts": counts,
        "items": [task.to_dict() for task in tasks[: max(1, limit)]],
    }


def _collect_hivemind(context_path: Path, *, limit: int) -> dict[str, Any]:
    try:
        from .hivemind import HivemindBus

        bus = HivemindBus(context_path)
        messages = bus.read(limit=max(1, limit))
    except Exception as exc:
        return {"recent_count": 0, "messages": [], "error": str(exc)}
    return {
        "recent_count": len(messages),
        "messages": [message.to_dict() for message in messages],
    }


def _collect_memory(manager: AFSManager, context_path: Path) -> dict[str, Any]:
    pipeline_status: dict[str, Any] = {}
    try:
        from .memory_consolidation import memory_status

        pipeline_status = memory_status(context_path, config=manager.config)
    except Exception:
        pipeline_status = {}

    memory_root = resolve_mount_root(context_path, MountType.MEMORY, config=manager.config)
    entries_path = memory_root / manager.config.memory_consolidation.entries_filename
    summary_dir = memory_root / manager.config.memory_consolidation.summary_dir_name
    entries_count = 0
    entries: list[dict[str, Any]] = []
    if entries_path.exists():
        try:
            for line in entries_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                entries_count += 1
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        except OSError:
            entries_count = 0

    # Build memory manifest: unique topics from tags/domains, most recent first
    manifest = _build_memory_manifest(entries)

    latest_markdown_path = None
    latest_markdown_excerpt = ""
    if summary_dir.exists():
        try:
            candidates = [path for path in summary_dir.glob("*.md") if path.is_file()]
        except OSError:
            candidates = []
        if candidates:
            latest = max(candidates, key=lambda item: item.stat().st_mtime)
            latest_markdown_path = latest
            latest_markdown_excerpt = _read_text(latest)

    return {
        "path": str(memory_root),
        "entries_path": str(entries_path),
        "summary_dir": str(summary_dir),
        "entries_count": entries_count,
        "memory_manifest": manifest,
        "latest_markdown_path": str(latest_markdown_path) if latest_markdown_path else "",
        "latest_markdown_excerpt": latest_markdown_excerpt,
        "status": pipeline_status,
    }


_MAX_MANIFEST_TOPICS = 20


def _build_memory_manifest(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a manifest of memory topics from entries, most recent first.

    Scans entries for unique tags and domains, returning up to
    _MAX_MANIFEST_TOPICS topics with entry counts and latest timestamps.
    This allows agents to make targeted memory.search calls instead of
    broad queries.
    """
    topic_stats: dict[str, dict[str, Any]] = {}

    for entry in entries:
        created_at = entry.get("created_at", "")
        domain = entry.get("domain")
        tags = entry.get("tags", [])

        # Collect topic keys from domain and tags
        topic_keys: list[str] = []
        if isinstance(domain, str) and domain.strip():
            topic_keys.append(f"domain:{domain.strip()}")
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    key = f"tag:{tag.strip()}"
                    if key not in topic_keys:
                        topic_keys.append(key)

        for key in topic_keys:
            if key not in topic_stats:
                topic_stats[key] = {"topic": key, "entry_count": 0, "latest": ""}
            topic_stats[key]["entry_count"] += 1
            if isinstance(created_at, str) and created_at > topic_stats[key]["latest"]:
                topic_stats[key]["latest"] = created_at

    # Sort by latest timestamp descending, then by entry count descending
    sorted_topics = sorted(
        topic_stats.values(),
        key=lambda t: (t["latest"], t["entry_count"]),
        reverse=True,
    )
    return sorted_topics[:_MAX_MANIFEST_TOPICS]


def _collect_agent_reports(manager: AFSManager, context_path: Path) -> dict[str, Any]:
    output_root = resolve_agent_output_root(context_path, config=manager.config)
    reports: list[dict[str, Any]] = []
    for name in (
        "context_warm",
        "context_watch",
        "agent_supervisor",
        "history_memory",
        "gemini_workspace_brief",
    ):
        path = output_root / f"{name}.json"
        payload: dict[str, Any] = {}
        status = ""
        age_seconds = None
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            status = str(payload.get("status", "")).strip()
            try:
                age_seconds = int(
                    max(0.0, datetime.now(timezone.utc).timestamp() - path.stat().st_mtime)
                )
            except OSError:
                age_seconds = None
        reports.append(
            {
                "name": name,
                "path": str(path),
                "available": path.exists(),
                "status": status,
                "age_seconds": age_seconds,
            }
        )
    return {"path": str(output_root), "reports": reports}


def _collect_latest_handoff(context_path: Path, *, config: Any = None) -> dict[str, Any]:
    """Collect the latest handoff packet if available."""
    try:
        from .handoff import HandoffStore

        store = HandoffStore(context_path, config=config)
        packet = store.read()
        if packet is None:
            return {"available": False}
        result = packet.to_dict()
        result["available"] = True
        return result
    except Exception:
        return {"available": False}


def _build_recommendations(summary: dict[str, Any]) -> list[str]:
    status = summary["status"]
    diff = summary["diff"]
    scratchpad = summary["scratchpad"]
    tasks = summary["tasks"]
    hivemind = summary["hivemind"]
    memory = summary["memory"]

    recommendations: list[str] = []
    if not status["mount_health"].get("healthy", False):
        recommendations.append("Run `context.repair` before editing because mount health is degraded.")

    index_info = status["index"]
    if not index_info.get("enabled", False):
        recommendations.append("Context indexing is disabled; rely on direct filesystem reads or enable the index.")
    elif not index_info.get("built", index_info.get("has_entries", False)):
        recommendations.append("Build the SQLite index before relying on `context.query`.")
    elif index_info.get("stale", False):
        recommendations.append("Refresh the stale SQLite index before trusting search results.")

    if diff.get("available") and diff.get("total_changes", 0) > 0:
        recommendations.append("Review `context.diff` before editing because the workspace has unreviewed drift.")

    stale_mounts = summary.get("stale_mounts", [])
    if stale_mounts:
        recommendations.append(
            "Review low-freshness mounts before trusting older context summaries or search results."
        )

    if scratchpad["state_text"] or scratchpad["deferred_text"]:
        recommendations.append("Read scratchpad state and deferred notes before making changes.")

    if tasks.get("total", 0) > 0:
        recommendations.append("Check pending items tasks before creating parallel work.")

    if hivemind.get("recent_count", 0) > 0:
        recommendations.append("Review recent hivemind messages for cross-agent handoffs.")

    if memory.get("latest_markdown_path"):
        recommendations.append("Scan the latest durable memory summary before asking for already-known context.")
    else:
        recommendations.append("No durable memory summary exists yet; run `afs memory consolidate` if handoff context matters.")
    if memory.get("status", {}).get("stale"):
        recommendations.append("Memory consolidation looks stale; run `afs memory consolidate` before relying on durable summaries.")

    session_changes = summary.get("session_changes", {})
    if session_changes.get("available") and session_changes.get("total_changes", 0) > 0:
        recommendations.append(
            "Review changes since last session before assuming context is current."
        )

    handoff = summary.get("handoff", {})
    if handoff.get("available") and handoff.get("blocked"):
        recommendations.append("Review blocked items from last handoff before starting new work.")

    recommendations.append("Prefer scratchpad updates, task queue entries, and hivemind notes for handoff instead of ad hoc chat summaries.")
    return recommendations


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if len(text) > _MAX_TEXT_CHARS:
        return text[:_MAX_TEXT_CHARS].rstrip() + "\n..."
    return text


def _indent_block(text: str, *, bullet: str | None = None) -> list[str]:
    if not text:
        return []
    lines = text.splitlines()
    output: list[str] = []
    if bullet is not None:
        output.append(bullet)
    output.extend(f"  {line}" for line in lines)
    return output
