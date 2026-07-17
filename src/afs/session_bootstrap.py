"""Build a proactive AFS session bootstrap summary for agents and humans."""

from __future__ import annotations

import json
import logging
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .codebase_explorer import build_codebase_summary, render_codebase_summary
from .context_freshness import (
    MountFreshness,
    context_diff_since_session,
    mount_freshness,
    save_context_snapshot,
)
from .context_index import ContextSQLiteIndex, count_mount_files
from .context_layout import LAYOUT_VERSION, detect_layout_version
from .context_paths import resolve_agent_output_root, resolve_mount_root
from .manager import AFSManager
from .models import MountType
from .profiles import resolve_active_profile
from .scopes import resolve_scope
from .skills import (
    MAX_SKILL_BODIES_CHARS,
    MAX_SKILL_BODY_CHARS,
    MAX_SKILL_BODY_MATCHES,
    MAX_SKILL_MATCHES,
    build_skill_matches,
    resolve_skill_roots,
    truncate_skill_body,
)

logger = logging.getLogger(__name__)

SESSION_BOOTSTRAP_JSON = "session_bootstrap.json"
SESSION_BOOTSTRAP_MARKDOWN = "session_bootstrap.md"
_MAX_TEXT_CHARS = 1500
_MAX_LIST_ITEMS = 8
_MAX_SKILL_SIGNAL_CHARS = 8_000

# Private test seam; production reads the platform controlling terminal.
_ENGAGE_READER = None


def build_agent_discovery_path(context_path: Path) -> dict[str, Any]:
    """Return the deterministic, low-noise AFS discovery path for agents."""
    context = context_path.expanduser().resolve()
    return {
        "principle": "Start with the smallest read-only context surface, then route richer intents through named CLI or slash-command flows.",
        "default_mcp_tools": [
            "context.status",
            "context.query",
            "context.read",
            "context.list",
            "context.write",
        ],
        "steps": [
            {
                "step": "status",
                "tool": "context.status",
                "when": "first AFS touch in a workspace",
                "next": "continue if mounts are healthy; repair only when health reports a concrete issue",
            },
            {
                "step": "query",
                "tool": "context.query",
                "when": "the user asks for context, history, scratchpad notes, or prior decisions",
                "next": "read exact sources only when the query result points to them",
            },
            {
                "step": "read",
                "tool": "context.read/context.list",
                "when": "a specific scratchpad, handoff, knowledge, or .context file is relevant",
                "next": "keep memory and knowledge read-only unless explicitly requested",
            },
            {
                "step": "scratchpad",
                "tool": "context.write",
                "when": "a local note, checkpoint, or handoff draft is explicitly useful",
                "next": f"default writes under {context / 'scratchpad'}; keep memory/knowledge deliberate",
            },
            {
                "step": "route",
                "tool": "CLI/slash command",
                "when": "tasks, handoffs, work preflight, verification, refresh, repair, or session packs are needed",
                "next": "use the named command instead of expanding the MCP catalog",
            },
        ],
        "routed_flows": {
            "next": "afs next --intent <intent> --path <workspace> --json",
            "tasks": "afs tasks list --path <workspace>",
            "handoff": "afs session handoff list --path <workspace> --json",
            "work_preflight": "afs work communication preflight --path <workspace> --json",
            "verify": "afs verify plan --cwd <workspace> --json",
            "refresh": "afs context repair --path <workspace> --dry-run --json",
            "pack": "afs session pack --path <workspace> --json",
            "human_manager": "afs manager open --path <workspace>",
        },
        "do_not_default": [
            "session.pack",
            "context.diff",
            "context.freshness",
            "task.*",
            "handoff.*",
            "memory.*",
            "agent.*",
            "events.*",
            "embeddings.*",
        ],
    }


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
    "missions",         # critical: durable in-flight goals and human done criteria
    "scratchpad",       # high: current working state
    "agent_manifest",   # high: shared harness/source-of-truth state
    "skills",           # high: task-matched local operating instructions
    "agent_jobs",       # high: file-backed background work queue
    "tasks",            # high: pending work items
    "work_assistant",   # high: external-write approvals and people context
    "codebase",         # high: quick repo orientation
    "session_changes",  # medium-high: what changed since last session
    "diff",             # medium: recent index drift
    "mount_freshness",  # medium: per-mount freshness scores
    "memory",           # medium: durable context
    "agent_runs",       # medium-low: recent recorded agent activity
    "hivemind",         # low: compatibility key for scoped messages
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
                index_info["stale"] = index.needs_health_refresh() if has_entries else False
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
        "discovery_path": build_agent_discovery_path(context_path),
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
    project_path: Path | None = None,
    task_limit: int = 10,
    message_limit: int = 10,
    agent_name: str = "cli",
    record_event: bool = True,
    token_budget: int = 0,
    skills_prompt: str = "",
    skills_top_k: int = 5,
    include_skills: bool = True,
) -> dict[str, Any]:
    """Build a structured startup packet for a context-aware agent session.

    When *token_budget* > 0, sections are truncated from lowest priority
    first until the rendered bootstrap fits within the budget.  Priority
    ordering is defined by ``_SECTION_PRIORITY``. Skill bodies are removed
    before whole sections, while the status header and startup sequence are
    always included.
    """
    context_path = context_path.expanduser().resolve()
    resolved_scope = resolve_scope(context_path, requester_path=project_path)
    visible_scopes = [resolved_scope.scope_id]
    if resolved_scope.scope_id != "common":
        visible_scopes.append("common")
    manager.register_agent(agent_name, context_path)
    context = manager.list_context(context_path=context_path)
    status = collect_context_status(manager, context_path)
    diff = collect_context_diff(manager, context_path)
    scratchpad = _collect_scratchpad(
        manager,
        context_path,
        scope_ids=visible_scopes,
    )
    tasks = _collect_tasks(context_path, limit=task_limit)
    agent_jobs = _collect_agent_jobs(context_path, limit=task_limit)
    agent_runs = _collect_agent_runs(context_path, limit=task_limit)
    agent_manifest = _collect_agent_manifest()
    work_assistant = _collect_work_assistant(manager, context_path, limit=task_limit)
    missions = _collect_missions(manager, context_path, limit=task_limit)
    handoff = _collect_latest_handoff(
        context_path,
        config=manager.config,
        scope_ids=visible_scopes,
    )
    skill_focus = skills_prompt.strip()
    skill_prompt_source = "explicit" if skill_focus else "session_state"
    if not skill_focus:
        skill_focus = _skill_signal(handoff=handoff, missions=missions, tasks=tasks)
    if not skill_focus:
        skill_prompt_source = "none"
    skills = _collect_skills(
        manager,
        prompt=skill_focus,
        prompt_source=skill_prompt_source,
        top_k=skills_top_k,
        enabled=include_skills,
    )
    messages = _collect_messages(
        context_path,
        scope_id=resolved_scope.scope_id,
        limit=message_limit,
        config=manager.config,
    )
    memory = _collect_memory(manager, context_path)
    reports = _collect_agent_reports(manager, context_path)
    codebase = build_codebase_summary(resolved_scope.requester_path or context_path)

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
        "project": resolved_scope.project_name or context.project_name,
        "scope_id": resolved_scope.scope_id,
        "project_id": resolved_scope.project_id,
        "project_description": getattr(context.metadata, "description", "") or "",
        "profile": status["profile"],
        "startup_sequence": [
            "Review context health and recent drift first.",
            "Check the agent manifest for shared harness, skill, and MCP routing before editing harness config.",
            "Read scratchpad state and deferred notes before editing.",
            "Check pending tasks, agent jobs, recent run records, and scoped messages for handoffs.",
            "Check `afs work` for people, review routes, activity, and pending approval-gated external writes.",
            "Use `afs context overview` / `afs.context.overview` for a fast repo map before deeper grep/query passes.",
            "Use `afs context query` / `context.query` before asking for context that may already be in memory or knowledge.",
            "Write updates back to notes, tasks, or scoped messages before handoff.",
        ],
        "status": status,
        "diff": diff,
        "scratchpad": scratchpad,
        "tasks": tasks,
        "agent_manifest": agent_manifest,
        "agent_jobs": agent_jobs,
        "agent_runs": agent_runs,
        "work_assistant": work_assistant,
        "missions": missions,
        "skills": skills,
        "codebase": codebase,
        # Keep the v1 key for one compatibility cycle. User-facing renderers
        # call this section Messages.
        "hivemind": messages,
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

    truncated_sections: list[str] = []
    budget_info = {
        "token_budget": budget,
        "estimated_tokens": 0,
        "truncated_sections": truncated_sections,
    }
    summary["_budget_info"] = budget_info

    def refresh_estimate() -> int:
        """Include observability metadata in the reported token estimate."""
        estimate = 0
        for _ in range(4):
            estimate = _estimate_tokens(json.dumps(summary, default=str))
            if budget_info["estimated_tokens"] == estimate:
                break
            budget_info["estimated_tokens"] = estimate
        return _estimate_tokens(json.dumps(summary, default=str))

    current_tokens = refresh_estimate()

    # Preserve compact match pointers before dropping the whole skills section.
    skills = summary.get("skills")
    if isinstance(skills, dict):
        matches = skills.get("matches")
        removed_body = False
        if isinstance(matches, list):
            for match in matches:
                if isinstance(match, dict) and match.get("body"):
                    match["body"] = ""
                    match["body_chars"] = 0
                    match["body_omitted"] = "token_budget"
                    removed_body = True
        if removed_body:
            truncated_sections.append("skills.bodies")
            current_tokens = refresh_estimate()
            if current_tokens <= budget:
                return summary

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
        current_tokens = refresh_estimate()

    refresh_estimate()
    return summary


def render_session_bootstrap(summary: dict[str, Any]) -> str:
    """Render a bootstrap packet as markdown/text for CLI and MCP prompts."""
    status = summary["status"]
    diff = summary["diff"]
    scratchpad = summary["scratchpad"]
    tasks = summary["tasks"]
    missions = summary.get("missions", {})
    agent_manifest = summary.get("agent_manifest", {})
    skills = summary.get("skills", {})
    agent_jobs = summary.get("agent_jobs", {})
    agent_runs = summary.get("agent_runs", {})
    work_assistant = summary.get("work_assistant", {})
    messages = summary["hivemind"]
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

    codebase = summary.get("codebase", {})
    if isinstance(codebase, dict) and codebase:
        lines.extend(["", "## Codebase"])
        lines.extend(_indent_block(render_codebase_summary(codebase)))

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
    if scratchpad.get("drafts"):
        lines.append("- scoped_drafts:")
        for draft in scratchpad["drafts"]:
            lines.append(
                f"  - {draft['title']} ({draft['artifact_id']}) [{draft['scope_id']}]"
            )
    if (
        not scratchpad["state_text"]
        and not scratchpad["deferred_text"]
        and not scratchpad["other_files"]
        and not scratchpad.get("drafts")
    ):
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

    lines.extend(["", "## Active Missions"])
    if missions.get("truncated"):
        lines.append("- truncated: token_budget")
    elif not missions.get("available", True):
        lines.append(f"- unavailable: {missions.get('error', 'unknown error')}")
    else:
        active_missions = missions.get("active") or []
        lines.append(f"- total: {missions.get('active_count', len(active_missions))}")
        if not active_missions:
            lines.append("- none")
        for mission in active_missions:
            if not isinstance(mission, dict):
                continue
            owner = f" -> {mission['owner']}" if mission.get("owner") else ""
            lines.append(
                f"- [{mission.get('status', 'active')}] "
                f"{mission.get('title', '')} ({mission.get('mission_id', '?')}){owner}"
            )
            if (
                mission.get("acceptance_human_confirmed") is True
                and mission.get("acceptance")
            ):
                lines.append(f"  - done_when: {mission['acceptance']}")
            if mission.get("summary"):
                lines.append(f"  - summary: {mission['summary']}")
            for step in mission.get("next_steps") or []:
                lines.append(f"  - next: {step}")
            for blocker in mission.get("blockers") or []:
                lines.append(f"  - blocker: {blocker}")

    lines.extend(["", "## Work Assistant"])
    if work_assistant.get("available", True):
        summary_counts = work_assistant.get("summary") or {}
        initialized = "yes" if work_assistant.get("initialized") else "no"
        lines.append(f"- initialized: {initialized}")
        lines.append(f"- database: {work_assistant.get('db_path', '')}")
        lines.append(
            "- counts: "
            f"people={summary_counts.get('people', 0)}, "
            f"relationships={summary_counts.get('relationships', 0)}, "
            f"review_routes={summary_counts.get('review_routes', 0)}, "
            f"approvals={summary_counts.get('approvals', 0)}, "
            f"communication_samples={summary_counts.get('communication_samples', 0)}, "
            f"activity={summary_counts.get('activity', 0)}"
        )
        pending = work_assistant.get("pending_approvals") or []
        if pending:
            lines.append("- pending_approvals:")
            for approval in pending:
                lines.append(
                    f"  - {approval.get('approval_id')}: "
                    f"{approval.get('target_system')}/{approval.get('action')} - "
                    f"{approval.get('summary')}"
                )
        samples = work_assistant.get("communication_samples") or []
        if samples:
            lines.append("- communication_samples:")
            for sample in samples[:3]:
                label = sample.get("purpose") or sample.get("channel") or "work_communication"
                excerpt = str(sample.get("text_excerpt") or "").strip().replace("\n", " ")
                if len(excerpt) > 120:
                    excerpt = excerpt[:117].rstrip() + "..."
                lines.append(f"  - {label}: {excerpt}")
        guidance = work_assistant.get("communication_guidance") or {}
        guidance_lines = guidance.get("guidance") if isinstance(guidance, dict) else []
        if guidance_lines:
            lines.append("- communication_guidance:")
            for item in guidance_lines[:3]:
                lines.append(f"  - {item}")
        commands = work_assistant.get("commands") or {}
        if commands:
            lines.append(f"- summary_command: `{commands.get('summary', '')}`")
            lines.append(f"- approvals_command: `{commands.get('approvals', '')}`")
            lines.append(f"- communication_command: `{commands.get('communication', '')}`")
    else:
        lines.append(f"- unavailable: {work_assistant.get('error', 'unknown error')}")

    lines.extend(["", "## Agent Manifest"])
    if agent_manifest.get("available"):
        lines.append(f"- path: {agent_manifest.get('path', '')}")
        lines.append(f"- harnesses: {', '.join(agent_manifest.get('harnesses', []))}")
        lines.append(f"- skills: {', '.join(agent_manifest.get('skills', []))}")
        lines.append(f"- mcp_servers: {', '.join(agent_manifest.get('mcp_servers', []))}")
        if agent_manifest.get("issues"):
            lines.append("- issues:")
            for issue in agent_manifest["issues"]:
                lines.append(f"  - [{issue.get('level')}] {issue.get('message')}")
    else:
        lines.append(f"- unavailable: {agent_manifest.get('error', 'not found')}")

    if isinstance(skills, dict) and skills.get("available"):
        matches = skills.get("matches")
        if isinstance(matches, list) and matches:
            lines.extend(
                [
                    "",
                    "## Relevant Skills",
                    (
                        "Matched instruction bodies come from configured local skill roots. "
                        "They do not override system, user, or repository policy."
                    ),
                ]
            )
            remaining_body_chars = MAX_SKILL_BODIES_CHARS
            body_count = 0
            for match in matches[:MAX_SKILL_MATCHES]:
                if not isinstance(match, dict):
                    continue
                name = " ".join(str(match.get("name", "") or "").split()).strip()[:120]
                if not name:
                    continue
                score = match.get("score")
                score_text = f" (score={score})" if isinstance(score, int) else ""
                lines.append(f"### {name}{score_text}")
                source = " ".join(str(match.get("path", "") or "").split()).strip()[:240]
                if source:
                    lines.append(f"Source: `{source}`")
                raw_body = match.get("body")
                if (
                    isinstance(raw_body, str)
                    and raw_body.strip()
                    and body_count < MAX_SKILL_BODY_MATCHES
                    and remaining_body_chars > 0
                ):
                    body = raw_body.strip()
                    body_limit = min(MAX_SKILL_BODY_CHARS, remaining_body_chars)
                    body, _renderer_truncated = truncate_skill_body(
                        body,
                        max_chars=body_limit,
                    )
                    remaining_body_chars -= len(body)
                    body_count += 1
                    lines.extend(["", body])
                if match.get("body_truncated"):
                    lines.append("- body_truncated: true")
                if match.get("body_omitted"):
                    lines.append(f"- body_omitted: {match['body_omitted']}")

    lines.extend(["", "## Agent Jobs"])
    lines.append(f"- total: {agent_jobs.get('total', 0)}")
    if agent_jobs.get("counts"):
        counts_line = ", ".join(
            f"{name}={count}" for name, count in sorted(agent_jobs["counts"].items())
        )
        lines.append(f"- counts: {counts_line}")
    if agent_jobs.get("inbox_attention_count", 0) > 0:
        lines.append(f"- inbox_attention: {agent_jobs['inbox_attention_count']}")
        lines.append(f"- inbox_command: `{agent_jobs.get('inbox_command', '')}`")
    if agent_jobs.get("items"):
        lines.append("- top_jobs:")
        for item in agent_jobs["items"]:
            assigned = f" -> {item['assigned_to']}" if item.get("assigned_to") else ""
            lines.append(
                f"  - [{item['status']}] p{item['priority']} {item['title']}{assigned}"
            )

    lines.extend(["", "## Agent Runs"])
    lines.append(f"- recent_count: {agent_runs.get('recent_count', 0)}")
    if agent_runs.get("items"):
        lines.append("- recent:")
        for item in agent_runs["items"]:
            lines.append(
                f"  - [{item['status']}] {item.get('harness') or '-'} {item['task']} ({item['id']})"
            )

    lines.extend(["", "## Messages"])
    lines.append(f"- recent_messages: {messages['recent_count']}")
    if messages["messages"]:
        for message in messages["messages"]:
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


def _collect_scratchpad(
    manager: AFSManager,
    context_path: Path,
    *,
    scope_ids: list[str] | None = None,
) -> dict[str, Any]:
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

    scoped_drafts: list[dict[str, Any]] = []
    try:
        from .scratchpad import ScratchpadStore

        for scope_id in dict.fromkeys(scope_ids or ["common"]):
            store = ScratchpadStore(
                context_path,
                scope_id=scope_id,
                config=manager.config,
            )
            for draft in store.list(limit=_MAX_LIST_ITEMS):
                scoped_drafts.append(
                    {
                        "artifact_id": draft.metadata.artifact_id,
                        "title": draft.metadata.title,
                        "created_at": draft.metadata.created_at,
                        "scope_id": draft.metadata.scope_id,
                        "path": str(draft.path),
                    }
                )
        scoped_drafts.sort(
            key=lambda item: (item["created_at"], item["artifact_id"]),
            reverse=True,
        )
        scoped_drafts = scoped_drafts[:_MAX_LIST_ITEMS]
    except Exception:
        scoped_drafts = []

    return {
        "path": str(scratchpad_root),
        "state_text": state_text,
        "deferred_text": deferred_text,
        "other_files": other_files,
        "agent_namespaces": agent_namespaces,
        "drafts": scoped_drafts,
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
    open_statuses = {"pending", "claimed", "in_progress"}
    ordered_tasks = [task for task in tasks if task.status in open_statuses]
    ordered_tasks.extend(task for task in tasks if task.status not in open_statuses)
    return {
        "total": len(tasks),
        "counts": counts,
        "items": [task.to_dict() for task in ordered_tasks[: max(1, limit)]],
    }


def _collect_agent_manifest() -> dict[str, Any]:
    try:
        from .agent_manifest import (
            default_manifest_path,
            load_manifest,
            summarize_manifest,
            validate_manifest,
        )

        path = default_manifest_path().expanduser()
        data = load_manifest(path)
        issues = validate_manifest(data)
        summary = summarize_manifest(data)
        return {
            "available": True,
            "path": str(path),
            "issues": [issue.to_dict() for issue in issues],
            **summary,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc), "issues": []}


def _collect_agent_jobs(context_path: Path, *, limit: int) -> dict[str, Any]:
    try:
        from .agent_job_inbox import build_agent_job_inbox
        from .agent_jobs import AgentJobQueue

        queue = AgentJobQueue(context_path)
        jobs = [
            job
            for job in queue.list()
            if job.status in {"queue", "running", "done", "failed"}
        ]
        inbox = build_agent_job_inbox(context_path, limit=max(1, limit))
    except Exception as exc:
        return {"total": 0, "counts": {}, "items": [], "error": str(exc)}

    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.status] = counts.get(job.status, 0) + 1
    return {
        "total": len(jobs),
        "counts": counts,
        "items": [job.to_dict() for job in jobs[: max(1, limit)]],
        "inbox_attention_count": inbox.get("attention_count", 0),
        "inbox_command": inbox.get("command", ""),
        "inbox": inbox,
    }


def _collect_agent_runs(context_path: Path, *, limit: int) -> dict[str, Any]:
    try:
        from .agent_runs import AgentRunStore

        runs = AgentRunStore(context_path).list(limit=max(1, limit))
    except Exception as exc:
        return {"recent_count": 0, "items": [], "error": str(exc)}

    return {
        "recent_count": len(runs),
        "items": [run.to_dict() for run in runs],
    }


def _collect_work_assistant(
    manager: AFSManager,
    context_path: Path,
    *,
    limit: int,
) -> dict[str, Any]:
    quoted_context = shlex.quote(str(context_path.expanduser().resolve()))
    commands = {
        "summary": f"afs work --context-root {quoted_context}",
        "approvals": f"afs work approvals list --context-root {quoted_context}",
        "communication": f"afs work communication preflight --context-root {quoted_context}",
        "communication_guide": f"afs work communication guide --context-root {quoted_context}",
        "communication_list": f"afs work communication list --context-root {quoted_context}",
        "activity": f"afs work activity list --context-root {quoted_context}",
    }
    try:
        from .work_assistant import DEFAULT_DB_FILENAME, WorkAssistantStore

        global_root = resolve_mount_root(context_path, MountType.GLOBAL, config=manager.config)
        db_path = global_root / DEFAULT_DB_FILENAME
        empty_summary = {
            "people": 0,
            "relationships": 0,
            "review_routes": 0,
            "approvals": 0,
            "activity": 0,
            "communication_samples": 0,
            "pending_approvals": 0,
            "db_path": str(db_path),
        }
        if not db_path.exists():
            return {
                "available": True,
                "initialized": False,
                "db_path": str(db_path),
                "summary": empty_summary,
                "people": [],
                "relationships": [],
                "pending_approvals": [],
                "communication_samples": [],
                "communication_guidance": {},
                "communication_preflight": {},
                "recent_activity": [],
                "commands": commands,
            }

        store = WorkAssistantStore(context_path, config=manager.config, db_path=db_path)
        return {
            "available": True,
            "initialized": True,
            "db_path": str(db_path),
            "summary": store.summary(),
            "people": store.list_people(limit=max(1, min(limit, _MAX_LIST_ITEMS))),
            "relationships": store.list_relationships(
                limit=max(1, min(limit, _MAX_LIST_ITEMS))
            ),
            "pending_approvals": store.list_approvals(status="pending", limit=max(1, limit)),
            "communication_samples": store.list_communication_samples(
                limit=max(1, min(limit, _MAX_LIST_ITEMS))
            ),
            "communication_guidance": store.communication_style_summary(
                limit=max(1, min(limit, _MAX_LIST_ITEMS))
            ),
            "communication_preflight": store.communication_preflight(
                limit=max(1, min(limit, _MAX_LIST_ITEMS)),
                approval_limit=max(1, min(limit, _MAX_LIST_ITEMS)),
                context_path=context_path,
            ),
            "recent_activity": store.list_activity(limit=max(1, min(limit, _MAX_LIST_ITEMS))),
            "commands": commands,
        }
    except Exception as exc:
        return {
            "available": False,
            "initialized": False,
            "summary": {},
            "people": [],
            "relationships": [],
            "pending_approvals": [],
            "communication_samples": [],
            "communication_guidance": {},
            "communication_preflight": {},
            "recent_activity": [],
            "commands": commands,
            "error": str(exc),
        }


def _collect_missions(
    manager: AFSManager, context_path: Path, *, limit: int
) -> dict[str, Any]:
    """Collect in-flight background missions so a resumed session sees them.

    Defensive: missions are optional; any failure yields an empty, well-shaped
    result rather than breaking bootstrap.
    """
    try:
        from .missions import MissionStore

        store = MissionStore(context_path, config=manager.config)
        active = store.active(limit=max(1, limit))
    except Exception as exc:
        return {"available": False, "active": [], "active_count": 0, "error": str(exc)}
    def bounded_text(value: Any, *, max_chars: int) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def bounded_list(value: Any, *, limit: int, max_chars: int) -> list[str]:
        if not isinstance(value, list):
            return []
        return [
            bounded_text(item, max_chars=max_chars)
            for item in value[:limit]
            if str(item or "").strip()
        ]

    active_payload: list[dict[str, Any]] = []
    for mission in active:
        # Bootstrap is a prompt surface, not a mission backup.  Exclude
        # unbounded logs, metadata, and link histories; retain only the
        # actionable fields a resumed agent needs.
        active_payload.append(
            {
                "mission_id": bounded_text(mission.mission_id, max_chars=80),
                "title": bounded_text(mission.title, max_chars=240),
                "status": bounded_text(mission.status, max_chars=32),
                "owner": bounded_text(mission.owner, max_chars=120),
                "summary": bounded_text(mission.summary, max_chars=_MAX_TEXT_CHARS),
                "acceptance": (
                    bounded_text(mission.acceptance, max_chars=_MAX_TEXT_CHARS)
                    if mission.acceptance_human_confirmed is True
                    else ""
                ),
                "acceptance_human_confirmed": (
                    mission.acceptance_human_confirmed is True
                ),
                "next_steps": bounded_list(
                    mission.next_steps, limit=5, max_chars=400
                ),
                "blockers": bounded_list(mission.blockers, limit=5, max_chars=400),
                "tags": bounded_list(mission.tags, limit=10, max_chars=80),
            }
        )
    return {
        "available": True,
        "active": active_payload,
        "active_count": len(active_payload),
    }


def _collect_skills(
    manager: AFSManager,
    *,
    prompt: str,
    prompt_source: str,
    top_k: int,
    enabled: bool,
) -> dict[str, Any]:
    """Collect bounded skill bodies matched to an explicit task prompt."""
    if not enabled:
        return {"available": False, "roots": [], "matches": []}
    try:
        profile = resolve_active_profile(manager.config)
        roots = resolve_skill_roots(list(profile.skill_roots))
        return {
            "available": True,
            "profile": profile.name,
            "prompt_source": prompt_source,
            "roots": [str(path) for path in roots],
            "matches": build_skill_matches(
                prompt,
                roots,
                profile=profile.name,
                top_k=top_k,
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "prompt_source": prompt_source,
            "roots": [],
            "matches": [],
            "error": str(exc),
        }


def _skill_signal(
    *,
    handoff: dict[str, Any],
    missions: dict[str, Any],
    tasks: dict[str, Any],
) -> str:
    """Build a bounded continuation signal used only to select trusted skills."""
    parts: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            normalized = " ".join(value.split()).strip()
            if normalized:
                parts.append(normalized)

    for step in handoff.get("next_steps") or []:
        add(step)
    for mission in missions.get("active") or []:
        if not isinstance(mission, dict):
            continue
        add(mission.get("title"))
        add(mission.get("summary"))
        for step in mission.get("next_steps") or []:
            add(step)
    for item in tasks.get("items") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "pending") not in {"pending", "claimed", "in_progress"}:
            continue
        add(item.get("title"))

    return " ".join(parts)[:_MAX_SKILL_SIGNAL_CHARS]


def _collect_messages(
    context_path: Path,
    *,
    scope_id: str,
    limit: int,
    config: Any = None,
) -> dict[str, Any]:
    try:
        from .messages import MessageBus

        bus = MessageBus(
            context_path,
            scope_id=scope_id,
            config=config,
            include_legacy=detect_layout_version(context_path) != LAYOUT_VERSION,
        )
        messages = bus.read(limit=max(1, limit))
    except Exception as exc:
        return {"recent_count": 0, "messages": [], "error": str(exc)}
    return {
        "recent_count": len(messages),
        "messages": [message.to_dict() for message in messages],
    }


def _collect_hivemind(context_path: Path, *, limit: int) -> dict[str, Any]:
    """Compatibility wrapper for callers that have not adopted project scopes."""

    return _collect_messages(
        context_path,
        scope_id="common",
        limit=limit,
    )


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


def _collect_latest_handoff(
    context_path: Path,
    *,
    config: Any = None,
    scope_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Collect the latest handoff packet if available."""
    try:
        from .handoff import HandoffStore

        packets = []
        for scope_id in dict.fromkeys(scope_ids or ["common"]):
            store = HandoffStore(context_path, config=config, scope_id=scope_id)
            packet = store.read()
            if packet is not None:
                packets.append(packet)
        packet = max(
            packets,
            key=lambda item: (item.timestamp, item.revision_id),
            default=None,
        )
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
    agent_manifest = summary.get("agent_manifest", {})
    agent_jobs = summary.get("agent_jobs", {})
    agent_runs = summary.get("agent_runs", {})
    work_assistant = summary.get("work_assistant", {})
    messages = summary["hivemind"]
    memory = summary["memory"]

    recommendations: list[str] = []
    if not status["mount_health"].get("healthy", False):
        recommendations.append("Run `context.repair` before editing because mount health is degraded.")

    index_info = status["index"]
    if not index_info.get("enabled", False):
        recommendations.append("Context indexing is disabled; rely on direct filesystem reads or enable the index.")
    elif not index_info.get("built", index_info.get("has_entries", False)):
        recommendations.append("Run `afs index rebuild --path <workspace>` before relying on `afs context query`.")
    elif index_info.get("stale", False):
        recommendations.append("Refresh the stale SQLite index with `afs index rebuild --path <workspace>` before trusting `afs context query` results.")

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

    if agent_manifest.get("issues"):
        recommendations.append("Run `afs agent-manifest validate --check-paths` before editing harness config.")

    if agent_jobs.get("inbox_attention_count", 0) > 0:
        recommendations.append(
            f"Review agent job inbox with `{agent_jobs.get('inbox_command', 'afs agent-jobs inbox')}`."
        )
    elif agent_jobs.get("total", 0) > 0:
        recommendations.append("Review `afs agent-jobs status` before spawning background work.")

    if agent_runs.get("recent_count", 0) > 0:
        recommendations.append("Review recent `afs agent-runs list` output for replayable prior agent state.")

    pending_approvals = work_assistant.get("pending_approvals") or []
    if pending_approvals:
        command = (work_assistant.get("commands") or {}).get("approvals", "afs work approvals list")
        recommendations.append(f"Review pending external-write approvals with `{command}`.")
    work_summary = work_assistant.get("summary") if isinstance(work_assistant, dict) else {}
    if isinstance(work_summary, dict) and work_summary.get("communication_samples", 0) == 0:
        command = (work_assistant.get("commands") or {}).get(
            "communication",
            "afs work communication preflight",
        )
        recommendations.append(
            "For work-context writing, inspect or capture user communication samples before "
            f"matching tone (`{command}`)."
        )

    if messages.get("recent_count", 0) > 0:
        recommendations.append("Review recent scoped messages for cross-agent handoffs.")

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

    recommendations.append(
        "Prefer scoped notes, task entries, and messages for handoff instead of ad hoc chat summaries."
    )
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

# -- engage mode ---------------------------------------------------------------


def top_priority_item(summary: dict[str, Any]) -> str:
    """The single item a human triaging this session should name first.

    Precedence mirrors how the rendered bootstrap leads: the latest handoff's
    first next step, then the newest active mission, then the first open task.
    """
    handoff = summary.get("handoff") or {}
    for step in handoff.get("next_steps") or []:
        text = str(step).strip()
        if text:
            return text
    for mission in (summary.get("missions") or {}).get("active") or []:
        if isinstance(mission, dict):
            title = str(mission.get("title") or "").strip()
            if title:
                return title
    for task in (summary.get("tasks") or {}).get("items") or []:
        if isinstance(task, dict):
            title = str(task.get("title") or "").strip()
            if title:
                return title
    return ""


def _prediction_matches(predicted: str, actual: str) -> bool:
    left = " ".join(predicted.lower().split())
    right = " ".join(actual.lower().split())
    if not left or not right:
        return False
    return left in right or right in left


def run_engage_prediction(
    context_path: Path,
    summary: dict[str, Any],
    *,
    config: Any = None,
) -> dict[str, Any] | None:
    """Predict-before-reveal micro-question for session bootstrap ``--engage``.

    Asks the human to guess the top queued item before the packet is shown,
    then logs prediction vs actual to the calibration trail. Generation before
    reveal is the engagement point; skipping (empty input, no tty) is always
    allowed and never blocks the bootstrap itself.
    """
    from .calibration import (
        human_prediction_scope,
        record_human_prediction,
    )
    from .human_provenance import _broker_for_reader

    kind = "bootstrap_top_priority"
    actual = top_priority_item(summary)
    try:
        result = _broker_for_reader(_ENGAGE_READER).read_line(
            "Before the reveal — what is the top queued item right now? ",
            scope=lambda response: human_prediction_scope(
                context_path,
                kind=kind,
                predicted=response.strip(),
                actual=actual,
                match=(
                    _prediction_matches(response.strip(), actual)
                    if actual
                    else None
                ),
                config=config,
            ),
        )
    except (EOFError, KeyboardInterrupt):
        return None
    if result is None:
        print("engage: skipped (requires an interactive controlling terminal)")
        return None
    predicted_raw, authorization = result
    predicted = predicted_raw.strip()
    if not predicted:
        return None
    match = _prediction_matches(predicted, actual) if actual else None
    try:
        entry = record_human_prediction(
            context_path,
            kind=kind,
            predicted=predicted,
            actual=actual,
            match=match,
            authorization=authorization,
            config=config,
        )
    except Exception as exc:
        logger.warning("Failed to record engage prediction: %s", exc)
        entry = None
    marker = {True: "matched", False: "different", None: "unresolved"}[match]
    print(f"engage: you predicted {predicted!r}; the queue says {actual!r} ({marker})")
    return entry
