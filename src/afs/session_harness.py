"""Session-preparation helpers for client wrappers and external harnesses."""

from __future__ import annotations

import json
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chat_registry import load_chat_registry
from .context_pack import build_context_pack, write_context_pack_artifacts
from .context_paths import resolve_agent_output_root
from .manager import AFSManager
from .model_prompts import build_model_system_prompt
from .profiles import resolve_active_profile
from .repo_policy import evaluate_repo_policy, load_repo_policy
from .session_bootstrap import build_session_bootstrap, write_session_bootstrap_artifacts
from .session_workflows import build_session_execution_profile
from .skills import discover_skills, resolve_skill_roots, score_skill_relevance
from .verification import build_structured_guidance, build_verification_plan

_RECENT_ACTIVITY_LIMIT = 20
_PROMPT_PREVIEW_CHARS = 180
_SUMMARY_PREVIEW_CHARS = 160
_SYSTEM_PROMPT_PREVIEW_CHARS = 320
_DEFAULT_SYSTEM_PROMPT_TOKEN_BUDGET = 2000
_WORKFLOW_SNAPSHOT_STEP_LIMIT = 6
_WORKFLOW_SNAPSHOT_SKILL_LIMIT = 4
_VERIFICATION_RECORD_LIMIT = 10
_DEFAULT_BASE_PROMPTS = {
    "generic": (
        "You are a context-aware assistant operating inside the Agentic File System. "
        "Use AFS session artifacts, cited repo context, and the declared workflow/tool "
        "profile before assuming missing facts."
    ),
    "codex": (
        "You are Codex operating inside the Agentic File System. "
        "Bias toward concrete files, minimal patches, and fast verification while using "
        "AFS session artifacts as the source of repo context."
    ),
    "claude": (
        "You are Claude operating inside the Agentic File System. "
        "Keep the loop structured, use AFS session artifacts first, and ground claims in "
        "the current repo context before expanding scope."
    ),
    "gemini": (
        "You are Gemini operating inside the Agentic File System. "
        "Use the prepared AFS session artifacts to stay retrieval-first, path-oriented, "
        "and concise."
    ),
    "vscode": (
        "You are the VS Code AFS harness. "
        "Keep filesystem and context operations aligned with the prepared AFS session "
        "contract and report progress through session lifecycle events."
    ),
}

_SESSION_ACTIVITY_EVENT_SPECS: tuple[dict[str, str], ...] = (
    {
        "name": "session_start",
        "phase": "session",
        "description": "Session wrapper or harness started and context is live.",
    },
    {
        "name": "session_end",
        "phase": "session",
        "description": "Session wrapper or harness ended.",
    },
    {
        "name": "user_prompt_submit",
        "phase": "turn",
        "description": "A new user prompt was submitted into the host loop.",
    },
    {
        "name": "turn_started",
        "phase": "turn",
        "description": "A model/tool turn started for the active prompt.",
    },
    {
        "name": "turn_completed",
        "phase": "turn",
        "description": "A model/tool turn completed successfully.",
    },
    {
        "name": "turn_failed",
        "phase": "turn",
        "description": "A model/tool turn failed.",
    },
    {
        "name": "task_created",
        "phase": "task",
        "description": "A background or delegated task was created.",
    },
    {
        "name": "task_progress",
        "phase": "task",
        "description": "A tracked task emitted progress.",
    },
    {
        "name": "task_completed",
        "phase": "task",
        "description": "A tracked task completed successfully.",
    },
    {
        "name": "task_failed",
        "phase": "task",
        "description": "A tracked task failed.",
    },
    {
        "name": "verification_recorded",
        "phase": "verification",
        "description": "A verification command or result was recorded for the session.",
    },
)


def _client_slug(client: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", client.strip().lower()).strip("_") or "client"


def _client_output_paths(
    manager: AFSManager,
    context_path: Path,
    client: str,
) -> dict[str, Path]:
    output_root = resolve_agent_output_root(context_path, config=manager.config)
    output_root.mkdir(parents=True, exist_ok=True)
    slug = _client_slug(client)
    return {
        "payload_json": output_root / f"session_client_{slug}.json",
        "skills_json": output_root / f"session_skills_{slug}.json",
        "prompt_json": output_root / f"session_system_prompt_{slug}.json",
        "prompt_text": output_root / f"session_system_prompt_{slug}.txt",
    }


def supported_session_activity_events() -> list[dict[str, str]]:
    """Return the stable event contract for client-session activity updates."""
    return [dict(spec) for spec in _SESSION_ACTIVITY_EVENT_SPECS]


def _truncate_text(text: str, *, limit: int) -> str:
    value = text.strip()
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 0)].rstrip() + "..."


def _initial_activity_state() -> dict[str, Any]:
    return {
        "updated_at": "",
        "session_state": {"status": "prepared"},
        "current_prompt": {},
        "current_turn": {},
        "active_tasks": [],
        "recent_events": [],
        "counters": {},
        "last_event": {},
        "last_task": {},
        "verification": _initial_verification_state(),
        "feedback": _initial_feedback_state(),
    }


def _integration_contract() -> dict[str, Any]:
    return {
        "version": 1,
        "notify_command": "afs session event",
        "hook_command": "afs session hook",
        "supported_events": supported_session_activity_events(),
        "payload_file": "",
    }


def _build_cli_hints(
    *,
    workspace_path: Path,
    bootstrap_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_workspace = workspace_path.expanduser().resolve()
    quoted_workspace = shlex.quote(str(resolved_workspace))
    notes: list[str] = []
    summary = bootstrap_state or {}
    status = summary.get("status") or {}
    index_state = status.get("index") or {}
    stale_mounts = [str(value).strip() for value in (summary.get("stale_mounts") or []) if str(value).strip()]

    if bool(index_state.get("enabled")) and (
        bool(index_state.get("stale")) or not bool(index_state.get("has_entries"))
    ):
        notes.append(
            "Indexed retrieval may be stale; run `afs index rebuild` before trusting query results."
        )
    if stale_mounts:
        notes.append(
            "Bootstrap reported low-freshness mounts; prefer recent scratchpad/history context over older summaries."
        )

    return {
        "workspace_path": str(resolved_workspace),
        "query_shortcut": f"afs query <text> --path {quoted_workspace}",
        "query_canonical": f"afs context query <text> --path {quoted_workspace}",
        "index_rebuild": f"afs index rebuild --path {quoted_workspace}",
        "notes": notes,
    }


def _default_client_session_payload(
    *,
    context_path: Path,
    client: str,
    session_id: str = "",
    config_path: Path | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    resolved_cwd = (cwd or Path.cwd()).expanduser().resolve()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "client": client,
        "session_id": session_id,
        "context_path": str(context_path.expanduser().resolve()),
        "config_path": str(config_path) if config_path else "",
        "cwd": str(resolved_cwd),
        "bootstrap": {"available": False, "artifact_paths": {}},
        "pack": {"available": False, "artifact_paths": {}},
        "skills": {"available": False, "artifact_paths": {}},
        "prompt": {"available": False, "artifact_paths": {}},
        "repo_policy": {"available": False},
        "verification_plan": {"available": False},
        "structured_guidance": {},
        "cli_hints": _build_cli_hints(workspace_path=resolved_cwd),
        "integration": _integration_contract(),
        "activity": _initial_activity_state(),
        "artifact_paths": {},
    }


def _ensure_client_session_payload_shape(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    cwd_value = str(normalized.get("cwd", "")).strip()
    resolved_cwd = Path(cwd_value).expanduser().resolve() if cwd_value else Path.cwd()
    activity = normalized.get("activity")
    if not isinstance(activity, dict):
        activity = _initial_activity_state()
    activity.setdefault("updated_at", "")
    activity.setdefault("session_state", {"status": "prepared"})
    activity.setdefault("current_prompt", {})
    activity.setdefault("current_turn", {})
    activity.setdefault("active_tasks", [])
    activity.setdefault("recent_events", [])
    activity.setdefault("counters", {})
    activity.setdefault("last_event", {})
    activity.setdefault("last_task", {})
    activity.setdefault("verification", _initial_verification_state())
    activity.setdefault("feedback", _initial_feedback_state())
    normalized["activity"] = activity

    integration = normalized.get("integration")
    if not isinstance(integration, dict):
        integration = _integration_contract()
    integration.setdefault("version", 1)
    integration.setdefault("notify_command", "afs session event")
    integration.setdefault("hook_command", "afs session hook")
    integration.setdefault("supported_events", supported_session_activity_events())
    integration.setdefault("payload_file", "")
    normalized["integration"] = integration

    normalized.setdefault("bootstrap", {"available": False, "artifact_paths": {}})
    normalized.setdefault("pack", {"available": False, "artifact_paths": {}})
    normalized.setdefault("skills", {"available": False, "artifact_paths": {}})
    normalized.setdefault("prompt", {"available": False, "artifact_paths": {}})
    normalized.setdefault("repo_policy", {"available": False})
    normalized.setdefault("verification_plan", {"available": False})
    normalized.setdefault("structured_guidance", {})
    cli_hints = normalized.get("cli_hints")
    merged_cli_hints = _build_cli_hints(workspace_path=resolved_cwd)
    if isinstance(cli_hints, dict):
        for key, value in cli_hints.items():
            if key == "notes" and isinstance(value, list):
                merged_cli_hints[key] = list(value)
            elif value not in (None, ""):
                merged_cli_hints[key] = value
    normalized["cli_hints"] = merged_cli_hints
    normalized.setdefault("artifact_paths", {})
    return normalized


def resolve_client_session_payload_path(
    manager: AFSManager,
    context_path: Path,
    *,
    client: str,
) -> Path:
    """Resolve the canonical client-session payload artifact path."""
    return _client_output_paths(manager, context_path, client)["payload_json"]


def _load_client_session_payload(
    manager: AFSManager,
    context_path: Path,
    *,
    client: str,
    session_id: str = "",
    config_path: Path | None = None,
    cwd: Path | None = None,
    payload_file: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    payload_path = (
        Path(payload_file).expanduser().resolve()
        if payload_file
        else resolve_client_session_payload_path(manager, context_path, client=client)
    )
    if payload_path.exists():
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = _default_client_session_payload(
                context_path=context_path,
                client=client,
                session_id=session_id,
                config_path=config_path,
                cwd=cwd,
            )
    else:
        payload = _default_client_session_payload(
            context_path=context_path,
            client=client,
            session_id=session_id,
            config_path=config_path,
            cwd=cwd,
        )
    return _ensure_client_session_payload_shape(payload), payload_path


def _activity_event_summary(event_name: str, event_payload: dict[str, Any]) -> str:
    client = str(event_payload.get("client", "")).strip()
    turn_id = str(event_payload.get("turn_id", "")).strip()
    task_id = str(event_payload.get("task_id", "")).strip()
    task_title = str(event_payload.get("task_title", "")).strip()
    summary = str(event_payload.get("summary", "")).strip()
    reason = str(event_payload.get("reason", "")).strip()
    prompt = str(event_payload.get("prompt", "")).strip()

    if event_name == "session_start":
        return f"Session started for {client}".strip()
    if event_name == "session_end":
        if reason:
            return f"Session ended ({reason})"
        return "Session ended"
    if event_name == "user_prompt_submit":
        return f"Prompt submitted: {_truncate_text(prompt, limit=_PROMPT_PREVIEW_CHARS)}"
    if event_name == "turn_started":
        return f"Turn {turn_id or '?'} started"
    if event_name == "turn_completed":
        return f"Turn {turn_id or '?'} completed"
    if event_name == "turn_failed":
        return f"Turn {turn_id or '?'} failed"
    if event_name == "task_created":
        return f"Task {task_id or '?'} created: {task_title or summary or '(untitled)'}"
    if event_name == "task_progress":
        return f"Task {task_id or '?'} progress: {summary or task_title or '(no update)'}"
    if event_name == "task_completed":
        return f"Task {task_id or '?'} completed"
    if event_name == "task_failed":
        return f"Task {task_id or '?'} failed"
    if event_name == "verification_recorded":
        return f"Verification recorded: {summary or '(no summary)'}"
    return summary or event_name


def _task_map(active_tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for entry in active_tasks:
        if not isinstance(entry, dict):
            continue
        task_id = str(entry.get("task_id", "")).strip()
        if task_id:
            tasks[task_id] = dict(entry)
    return tasks


def _compact_event_names(entries: list[dict[str, Any]]) -> list[str]:
    compact: list[str] = []
    previous = ""
    for entry in entries[-_WORKFLOW_SNAPSHOT_STEP_LIMIT:]:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("event", "")).strip()
        if not name or name == previous:
            continue
        compact.append(name)
        previous = name
    return compact


def _initial_verification_state() -> dict[str, Any]:
    return {
        "required": False,
        "status": "not_required",
        "expected": [],
        "records": [],
        "record_count": 0,
        "commands": [],
        "last_record": {},
        "message": "",
        "updated_at": "",
        "mode": "",
    }


def _initial_feedback_state() -> dict[str, Any]:
    return {
        "signals": [],
        "counts": {},
        "message": "",
        "updated_at": "",
    }


def _normalize_verification_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"passed", "pass", "verified", "ok", "success"}:
        return "passed"
    if text in {"failed", "fail", "error"}:
        return "failed"
    if text in {"skipped", "skip"}:
        return "skipped"
    if text in {"missing", "pending", "not_required"}:
        return text
    if text.startswith("verification_"):
        return _normalize_verification_status(text.removeprefix("verification_"))
    return ""


def _verification_requirements(payload: dict[str, Any]) -> dict[str, Any]:
    verification_plan = (
        payload.get("verification_plan")
        if isinstance(payload.get("verification_plan"), dict)
        else {}
    )
    selected_checks = (
        verification_plan.get("selected_checks")
        if isinstance(verification_plan.get("selected_checks"), list)
        else []
    )
    if selected_checks:
        expected: list[str] = []
        required = False
        for check in selected_checks:
            if not isinstance(check, dict):
                continue
            name = str(check.get("name", "")).strip()
            required = required or bool(check.get("required"))
            commands = check.get("commands") if isinstance(check.get("commands"), list) else []
            if commands:
                for command in commands:
                    text = str(command).strip()
                    if not text:
                        continue
                    expected.append(f"{name}: {text}" if name else text)
            elif name:
                expected.append(f"{name}: review changed scope")
        return {
            "required": required,
            "workflow": str(verification_plan.get("workflow", "")).strip(),
            "tool_profile": str(verification_plan.get("tool_profile", "")).strip(),
            "expected": expected,
        }

    pack = payload.get("pack") if isinstance(payload.get("pack"), dict) else {}
    prompt = payload.get("prompt") if isinstance(payload.get("prompt"), dict) else {}
    skills = payload.get("skills") if isinstance(payload.get("skills"), dict) else {}

    workflow = str(pack.get("workflow") or prompt.get("workflow") or "").strip()
    tool_profile = str(pack.get("tool_profile") or prompt.get("tool_profile") or "").strip()
    model_family = str(prompt.get("model_family") or payload.get("client") or "generic").strip()

    workflow_contract: list[str] = []
    if workflow:
        try:
            execution_profile = build_session_execution_profile(
                model=model_family or "generic",
                workflow=workflow,
                tool_profile=tool_profile or None,
            )
        except Exception:
            execution_profile = {}
        workflow_contract = [
            str(item).strip()
            for item in (execution_profile.get("verification_contract") or [])
            if str(item).strip()
        ]

    skill_contract: list[str] = []
    matches = skills.get("matches") if isinstance(skills, dict) else []
    if isinstance(matches, list):
        for match in matches[:_WORKFLOW_SNAPSHOT_SKILL_LIMIT + 2]:
            if not isinstance(match, dict):
                continue
            name = str(match.get("name", "")).strip()
            verification = match.get("verification")
            if not isinstance(verification, list):
                continue
            for item in verification[:2]:
                text = str(item).strip()
                if not text:
                    continue
                skill_contract.append(f"{name}: {text}" if name else text)

    required = bool(skill_contract) or tool_profile == "edit_and_verify" or workflow in {
        "edit_fast",
        "root_cause_deep",
    }

    expected: list[str] = []
    seen: set[str] = set()
    for item in workflow_contract + skill_contract:
        marker = item.lower()
        if marker in seen:
            continue
        seen.add(marker)
        expected.append(item)

    return {
        "required": required,
        "workflow": workflow,
        "tool_profile": tool_profile,
        "expected": expected,
    }


def _verification_record_from_event(
    *,
    event_name: str,
    event_payload: dict[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    status = _normalize_verification_status(event_payload.get("verification_status"))
    command = str(event_payload.get("verification_command", "")).strip()
    summary = str(event_payload.get("summary", "")).strip()

    if not status:
        status = _normalize_verification_status(event_payload.get("status"))

    if not status and command:
        if event_name in {"turn_failed", "task_failed"}:
            status = "failed"
        else:
            status = "passed"

    if not status:
        return {}

    record = {
        "timestamp": timestamp,
        "event": event_name,
        "status": status,
        "command": command,
        "summary": _truncate_text(summary, limit=_SUMMARY_PREVIEW_CHARS) if summary else "",
        "turn_id": str(event_payload.get("turn_id", "")).strip(),
        "task_id": str(event_payload.get("task_id", "")).strip(),
        "task_title": str(event_payload.get("task_title", "")).strip(),
    }
    return {key: value for key, value in record.items() if value not in ("", None, [], {})}


def _build_session_verification_state(
    payload: dict[str, Any],
    *,
    final: bool,
    updated_at: str = "",
) -> dict[str, Any]:
    activity = payload.get("activity") if isinstance(payload.get("activity"), dict) else {}
    current = activity.get("verification") if isinstance(activity, dict) else {}
    current = current if isinstance(current, dict) else {}
    records = current.get("records") if isinstance(current.get("records"), list) else []

    requirements = _verification_requirements(payload)
    required = bool(requirements.get("required"))
    expected = list(requirements.get("expected") or [])

    normalized_records: list[dict[str, Any]] = []
    commands: list[str] = []
    command_seen: set[str] = set()
    for entry in records[-_VERIFICATION_RECORD_LIMIT:]:
        if not isinstance(entry, dict):
            continue
        status = _normalize_verification_status(entry.get("status"))
        if not status:
            continue
        record = dict(entry)
        record["status"] = status
        normalized_records.append(record)
        command = str(record.get("command", "")).strip()
        if command and command not in command_seen:
            command_seen.add(command)
            commands.append(command)

    statuses = [str(entry.get("status", "")).strip() for entry in normalized_records]
    if not required:
        status = "not_required"
        message = "Verification not required for this workflow."
    elif "failed" in statuses:
        status = "failed"
        message = "Recorded verification failed."
    elif "passed" in statuses:
        status = "passed"
        message = "Verification recorded."
    elif "skipped" in statuses:
        status = "skipped"
        message = "Verification was explicitly skipped."
    elif final:
        status = "missing"
        message = "Required verification was not recorded before session end."
    else:
        status = "pending"
        message = "Verification still pending."

    last_record = normalized_records[-1] if normalized_records else {}
    state = {
        "required": required,
        "status": status,
        "expected": expected,
        "records": normalized_records,
        "record_count": len(normalized_records),
        "commands": commands,
        "last_record": last_record,
        "message": message,
        "updated_at": updated_at or str(current.get("updated_at", "")).strip(),
        "workflow": str(requirements.get("workflow", "")).strip(),
        "tool_profile": str(requirements.get("tool_profile", "")).strip(),
    }
    return {key: value for key, value in state.items() if value not in ("", None, [], {})}


def _build_session_feedback_state(
    payload: dict[str, Any],
    *,
    updated_at: str = "",
) -> dict[str, Any]:
    activity = payload.get("activity") if isinstance(payload.get("activity"), dict) else {}
    verification = (
        activity.get("verification")
        if isinstance(activity.get("verification"), dict)
        else {}
    )
    policy = payload.get("repo_policy") if isinstance(payload.get("repo_policy"), dict) else {}

    counts: dict[str, int] = {}
    signals: list[str] = []

    matched_risks = policy.get("matched_risks") if isinstance(policy.get("matched_risks"), list) else []
    anti_pattern_hits = (
        policy.get("anti_pattern_hits")
        if isinstance(policy.get("anti_pattern_hits"), list)
        else []
    )
    if matched_risks:
        counts["review_risk"] = len(matched_risks)
        signals.append("review_risk")
    if anti_pattern_hits:
        counts["policy_violation"] = len(anti_pattern_hits)
        signals.append("policy_violation")

    verification_status = str(verification.get("status", "")).strip()
    if verification_status in {"failed", "missing", "skipped"}:
        counts[f"verification_{verification_status}"] = 1
        signals.append(f"verification_{verification_status}")

    message_parts: list[str] = []
    if counts.get("review_risk"):
        message_parts.append(f"{counts['review_risk']} repo risk alerts")
    if counts.get("policy_violation"):
        message_parts.append(f"{counts['policy_violation']} policy violations")
    if verification_status in {"failed", "missing", "skipped"}:
        message_parts.append(f"verification {verification_status}")

    return {
        "signals": signals,
        "counts": counts,
        "message": "; ".join(message_parts),
        "updated_at": updated_at,
    }


def build_session_activity_snapshot(
    payload: dict[str, Any] | None,
    *,
    event_name: str,
) -> dict[str, Any]:
    """Build a compact workflow snapshot from the current session payload."""
    if not isinstance(payload, dict):
        return {}

    activity = payload.get("activity")
    if not isinstance(activity, dict):
        return {}

    pack = payload.get("pack") if isinstance(payload.get("pack"), dict) else {}
    skills = payload.get("skills") if isinstance(payload.get("skills"), dict) else {}
    current_prompt = (
        activity.get("current_prompt")
        if isinstance(activity.get("current_prompt"), dict)
        else {}
    )
    current_turn = (
        activity.get("current_turn")
        if isinstance(activity.get("current_turn"), dict)
        else {}
    )
    last_task = (
        activity.get("last_task")
        if isinstance(activity.get("last_task"), dict)
        else {}
    )
    session_state = (
        activity.get("session_state")
        if isinstance(activity.get("session_state"), dict)
        else {}
    )
    verification = (
        activity.get("verification")
        if isinstance(activity.get("verification"), dict)
        else {}
    )
    feedback = (
        activity.get("feedback")
        if isinstance(activity.get("feedback"), dict)
        else {}
    )
    counters = activity.get("counters") if isinstance(activity.get("counters"), dict) else {}
    recent_events = (
        activity.get("recent_events")
        if isinstance(activity.get("recent_events"), list)
        else []
    )

    matched_skills: list[str] = []
    matches = skills.get("matches") if isinstance(skills, dict) else []
    if isinstance(matches, list):
        for entry in matches[:_WORKFLOW_SNAPSHOT_SKILL_LIMIT]:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            if name:
                matched_skills.append(name)

    failed_events = int(counters.get("turn_failed", 0)) + int(counters.get("task_failed", 0))
    completed_events = int(counters.get("turn_completed", 0)) + int(counters.get("task_completed", 0))
    if event_name in {"turn_failed", "task_failed"}:
        outcome = "failed"
    elif event_name in {"turn_completed", "task_completed"}:
        outcome = "completed"
    elif event_name == "session_end":
        outcome = "failed" if failed_events else "completed"
    else:
        outcome = ""

    snapshot = {
        "client": str(payload.get("client", "")).strip(),
        "session_id": str(payload.get("session_id", "")).strip(),
        "workflow": str(pack.get("workflow", "")).strip(),
        "tool_profile": str(pack.get("tool_profile", "")).strip(),
        "pack_mode": str(pack.get("pack_mode", "")).strip(),
        "prompt_preview": str(current_prompt.get("preview", "")).strip(),
        "turn_id": str(current_turn.get("turn_id", "")).strip(),
        "turn_status": str(current_turn.get("status", "")).strip(),
        "task_id": str(last_task.get("task_id", "")).strip(),
        "task_title": str(last_task.get("task_title", "")).strip(),
        "session_status": str(session_state.get("status", "")).strip(),
        "event_name": event_name,
        "outcome": outcome,
        "workflow_steps": _compact_event_names(recent_events),
        "matched_skills": matched_skills,
        "completed_events": completed_events,
        "failed_events": failed_events,
        "verification_status": str(verification.get("status", "")).strip(),
        "verification_required": bool(verification.get("required")),
        "verification_record_count": int(verification.get("record_count", 0) or 0),
        "feedback_signals": list(feedback.get("signals") or []),
    }
    return {
        key: value
        for key, value in snapshot.items()
        if value not in ("", None, [], {})
    }


def _estimate_tokens(text: str) -> int:
    if not text.strip():
        return 0
    return max(1, (len(text) + 3) // 4)


def _bootstrap_bundle(
    manager: AFSManager,
    context_path: Path,
    *,
    client: str,
    write_artifacts: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = build_session_bootstrap(
        manager,
        context_path,
        agent_name=f"{client}-client",
        record_event=False,
    )
    artifact_paths = (
        write_session_bootstrap_artifacts(manager, context_path, summary)
        if write_artifacts
        else summary.get("artifact_paths") or {}
    )
    payload = {
        "available": True,
        "context_path": summary["context_path"],
        "project": summary["project"],
        "profile": summary["profile"],
        "stale_mounts": list(summary.get("stale_mounts", [])),
        "recommended_actions": list(summary.get("recommended_actions", [])),
        "artifact_paths": artifact_paths,
    }
    return summary, payload


def _bootstrap_summary(
    manager: AFSManager,
    context_path: Path,
    *,
    client: str,
    write_artifacts: bool,
) -> dict[str, Any]:
    _summary, payload = _bootstrap_bundle(
        manager,
        context_path,
        client=client,
        write_artifacts=write_artifacts,
    )
    return payload


def _pack_summary(
    manager: AFSManager,
    context_path: Path,
    *,
    query: str,
    task: str,
    model: str,
    workflow: str,
    tool_profile: str,
    pack_mode: str,
    token_budget: int | None,
    include_content: bool,
    max_query_results: int,
    max_embedding_results: int,
    write_artifacts: bool,
) -> dict[str, Any]:
    pack = build_context_pack(
        manager,
        context_path,
        query=query,
        task=task,
        model=model,
        workflow=workflow,
        tool_profile=tool_profile,
        pack_mode=pack_mode,
        token_budget=token_budget,
        include_content=include_content,
        max_query_results=max_query_results,
        max_embedding_results=max_embedding_results,
    )
    artifact_paths = pack.get("artifact_paths") or {}
    if write_artifacts and not bool((pack.get("cache") or {}).get("hit")):
        artifact_paths = write_context_pack_artifacts(manager, context_path, pack)
    return {
        "available": True,
        "query": query,
        "task": task,
        "model": pack.get("model", model),
        "workflow": (pack.get("execution_profile") or {}).get("workflow", workflow),
        "tool_profile": (
            ((pack.get("execution_profile") or {}).get("tool_profile") or {}).get("name")
            or tool_profile
        ),
        "pack_mode": pack.get("pack_mode", pack_mode),
        "estimated_tokens": pack.get("estimated_tokens"),
        "cache": dict(pack.get("cache") or {}),
        "artifact_paths": artifact_paths,
    }


def _skills_summary(
    manager: AFSManager,
    context_path: Path,
    *,
    client: str,
    prompt: str,
    top_k: int,
    write_artifacts: bool,
) -> dict[str, Any]:
    profile = resolve_active_profile(manager.config)
    roots = resolve_skill_roots(
        list(profile.skill_roots),
        afs_root=os.getenv("AFS_ROOT", "").strip() or None,
    )
    ranked: list[tuple[int, Any]] = []
    for skill in discover_skills(roots, profile=profile.name):
        score = score_skill_relevance(prompt, skill)
        if score > 0:
            ranked.append((score, skill))
    ranked.sort(key=lambda item: item[0], reverse=True)

    payload = {
        "available": True,
        "client": client,
        "profile": profile.name,
        "prompt": prompt,
        "roots": [str(path) for path in roots],
        "matches": [
            {
                "score": score,
                "name": skill.name,
                "path": str(skill.path),
                "triggers": skill.triggers,
                "requires": skill.requires,
                "enforcement": skill.enforcement,
                "verification": skill.verification,
            }
            for score, skill in ranked[: max(top_k, 0)]
        ],
        "artifact_paths": {},
    }
    if write_artifacts:
        paths = _client_output_paths(manager, context_path, client)
        payload["artifact_paths"] = {"json": str(paths["skills_json"])}
        paths["skills_json"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _resolve_client_base_prompt(
    manager: AFSManager,
    *,
    client: str,
    model: str,
    config_path: Path | None,
) -> tuple[str, str]:
    try:
        registry = load_chat_registry(config=manager.config, config_path=config_path)
    except Exception:
        registry = None

    if registry is not None:
        for candidate in (client, model):
            entry = registry.models.get(candidate)
            if entry and entry.system_prompt.strip():
                return entry.system_prompt.strip(), f"chat_registry:{candidate}"

    for candidate in (client, model, "generic"):
        base_prompt = _DEFAULT_BASE_PROMPTS.get(candidate, "").strip()
        if base_prompt:
            return base_prompt, f"builtin:{candidate}"

    return "", ""


def _prompt_summary(
    manager: AFSManager,
    context_path: Path,
    *,
    client: str,
    model: str,
    workflow: str,
    tool_profile: str,
    token_budget: int | None,
    config_path: Path | None,
    session_state: dict[str, Any],
    pack_state: dict[str, Any],
    skills_state: dict[str, Any],
    verification_state: dict[str, Any],
    policy_state: dict[str, Any],
    structured_guidance: dict[str, Any],
    write_artifacts: bool,
) -> dict[str, Any]:
    base_prompt, base_prompt_source = _resolve_client_base_prompt(
        manager,
        client=client,
        model=model,
        config_path=config_path,
    )
    budget = token_budget if isinstance(token_budget, int) and token_budget > 0 else _DEFAULT_SYSTEM_PROMPT_TOKEN_BUDGET
    prompt_text = build_model_system_prompt(
        base_prompt=base_prompt,
        model_family=model,
        role=client,
        session_state=session_state,
        pack_state=pack_state,
        skills_state=skills_state,
        verification_state=verification_state,
        policy_state=policy_state,
        structured_guidance=structured_guidance,
        workflow=workflow,
        tool_profile=tool_profile,
        token_budget=budget,
    )

    artifact_paths: dict[str, str] = {}
    preview = _truncate_text(prompt_text, limit=_SYSTEM_PROMPT_PREVIEW_CHARS) if prompt_text else ""
    estimated_tokens = _estimate_tokens(prompt_text) if prompt_text else 0

    if write_artifacts and prompt_text:
        paths = _client_output_paths(manager, context_path, client)
        prompt_payload = {
            "client": client,
            "model_family": model,
            "role": client,
            "workflow": workflow,
            "tool_profile": tool_profile,
            "base_prompt_source": base_prompt_source,
            "estimated_tokens": estimated_tokens,
            "preview": preview,
            "recommended_schema": str(structured_guidance.get("recommended_schema", "")).strip(),
            "text": prompt_text,
        }
        paths["prompt_text"].write_text(prompt_text + "\n", encoding="utf-8")
        paths["prompt_json"].write_text(json.dumps(prompt_payload, indent=2) + "\n", encoding="utf-8")
        artifact_paths = {
            "text": str(paths["prompt_text"]),
            "json": str(paths["prompt_json"]),
        }

    return {
        "available": bool(prompt_text.strip()),
        "client": client,
        "model_family": model,
        "role": client,
        "workflow": workflow,
        "tool_profile": tool_profile,
        "base_prompt_source": base_prompt_source,
        "estimated_tokens": estimated_tokens,
        "preview": preview,
        "artifact_paths": artifact_paths,
    }


def write_client_session_payload_artifact(
    manager: AFSManager,
    context_path: Path,
    *,
    client: str,
    payload: dict[str, Any],
    payload_path: str | Path | None = None,
) -> dict[str, str]:
    """Persist the latest client-session payload for harness consumers."""
    resolved_payload, resolved_path = _load_client_session_payload(
        manager,
        context_path,
        client=client,
        session_id=str(payload.get("session_id", "")).strip(),
        config_path=Path(str(payload.get("config_path", ""))).expanduser().resolve()
        if str(payload.get("config_path", "")).strip()
        else None,
        cwd=Path(str(payload.get("cwd", ""))).expanduser().resolve()
        if str(payload.get("cwd", "")).strip()
        else None,
        payload_file=payload_path,
    )
    resolved_payload.update(payload)
    resolved_payload = _ensure_client_session_payload_shape(resolved_payload)
    resolved_payload["artifact_paths"] = {"json": str(resolved_path)}
    resolved_payload["integration"]["payload_file"] = str(resolved_path)
    resolved_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    cli_hints = resolved_payload.get("cli_hints")
    if isinstance(cli_hints, dict):
        cli_hints["verify_run"] = f"afs verify run --payload-file {shlex.quote(str(resolved_path))} --json"
        cli_hints["verify_plan"] = f"afs verify plan --payload-file {shlex.quote(str(resolved_path))} --json"
    resolved_path.write_text(json.dumps(resolved_payload, indent=2) + "\n", encoding="utf-8")
    payload.clear()
    payload.update(resolved_payload)
    return {"json": str(resolved_path)}


def record_client_session_activity(
    manager: AFSManager,
    context_path: Path,
    *,
    client: str,
    event_name: str,
    event_payload: dict[str, Any],
    payload_file: str | Path | None = None,
    session_id: str = "",
    config_path: Path | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Update the live session payload artifact with prompt/turn/task activity."""
    payload, payload_path = _load_client_session_payload(
        manager,
        context_path,
        client=client,
        session_id=session_id or str(event_payload.get("session_id", "")).strip(),
        config_path=config_path,
        cwd=cwd,
        payload_file=payload_file,
    )
    now = datetime.now(timezone.utc).isoformat()
    activity = payload["activity"]
    summary = _activity_event_summary(event_name, event_payload)
    prompt = str(event_payload.get("prompt", "")).strip()
    turn_id = str(event_payload.get("turn_id", "")).strip()
    task_id = str(event_payload.get("task_id", "")).strip()
    task_title = str(event_payload.get("task_title", "")).strip()
    status = str(event_payload.get("status", "")).strip()
    reason = str(event_payload.get("reason", "")).strip()

    event_entry = {
        "timestamp": now,
        "event": event_name,
        "summary": summary,
        "client": str(event_payload.get("client", client)).strip(),
        "session_id": str(event_payload.get("session_id", session_id)).strip(),
        "turn_id": turn_id,
        "task_id": task_id,
        "task_title": task_title,
        "status": status,
        "reason": reason,
    }
    activity["updated_at"] = now
    counters = activity.get("counters", {})
    counters[event_name] = int(counters.get(event_name, 0)) + 1
    activity["counters"] = counters
    recent_events = list(activity.get("recent_events", []))
    recent_events.append(event_entry)
    activity["recent_events"] = recent_events[-_RECENT_ACTIVITY_LIMIT:]
    activity["last_event"] = event_entry

    if event_name == "session_start":
        session_state = dict(activity.get("session_state") or {})
        session_state.update({"status": "active", "started_at": now})
        activity["session_state"] = session_state
    elif event_name == "session_end":
        session_state = dict(activity.get("session_state") or {})
        session_state.update(
            {
                "status": "ended",
                "ended_at": now,
                "reason": reason,
                "exit_code": event_payload.get("exit_code"),
            }
        )
        activity["session_state"] = session_state

    if event_name == "user_prompt_submit":
        activity["current_prompt"] = {
            "text": prompt,
            "preview": _truncate_text(prompt, limit=_PROMPT_PREVIEW_CHARS),
            "submitted_at": now,
            "turn_id": turn_id,
        }

    if event_name == "turn_started":
        activity["current_turn"] = {
            "turn_id": turn_id,
            "status": "running",
            "started_at": now,
            "summary": _truncate_text(summary, limit=_SUMMARY_PREVIEW_CHARS),
        }
    elif event_name in {"turn_completed", "turn_failed"}:
        current_turn = dict(activity.get("current_turn") or {})
        current_turn.update(
            {
                "turn_id": turn_id or str(current_turn.get("turn_id", "")),
                "status": "completed" if event_name == "turn_completed" else "failed",
                "completed_at": now,
                "summary": _truncate_text(summary, limit=_SUMMARY_PREVIEW_CHARS),
                "reason": reason,
            }
        )
        if "started_at" not in current_turn:
            current_turn["started_at"] = now
        activity["current_turn"] = current_turn

    verification_state = (
        activity.get("verification")
        if isinstance(activity.get("verification"), dict)
        else _initial_verification_state()
    )
    verification_records = list(verification_state.get("records", []))
    verification_record = _verification_record_from_event(
        event_name=event_name,
        event_payload=event_payload,
        timestamp=now,
    )
    if verification_record:
        verification_records.append(verification_record)
        verification_records = verification_records[-_VERIFICATION_RECORD_LIMIT:]
    verification_state = dict(verification_state)
    verification_state["records"] = verification_records
    verification_state["updated_at"] = now
    activity["verification"] = verification_state

    tasks = _task_map(list(activity.get("active_tasks", [])))
    if event_name in {"task_created", "task_progress", "task_completed", "task_failed"} and task_id:
        task = tasks.get(
            task_id,
            {
                "task_id": task_id,
                "task_title": task_title,
                "status": "created",
                "summary": "",
                "turn_id": turn_id,
                "created_at": now,
            },
        )
        if task_title:
            task["task_title"] = task_title
        if turn_id:
            task["turn_id"] = turn_id
        if summary:
            task["summary"] = summary
        task["updated_at"] = now
        if reason:
            task["reason"] = reason
        if event_name == "task_created":
            task["status"] = status or "created"
            tasks[task_id] = task
        elif event_name == "task_progress":
            task["status"] = status or "running"
            tasks[task_id] = task
        else:
            task["status"] = "completed" if event_name == "task_completed" else "failed"
            task["completed_at"] = now
            activity["last_task"] = task
            tasks.pop(task_id, None)

    activity["active_tasks"] = sorted(
        tasks.values(),
        key=lambda entry: str(entry.get("updated_at", entry.get("created_at", ""))),
        reverse=True,
    )
    payload["activity"] = activity
    activity["verification"] = _build_session_verification_state(
        payload,
        final=event_name == "session_end",
        updated_at=now,
    )
    activity["feedback"] = _build_session_feedback_state(
        payload,
        updated_at=now,
    )
    payload["activity"] = activity
    payload["updated_at"] = now
    write_client_session_payload_artifact(
        manager,
        context_path,
        client=client,
        payload=payload,
        payload_path=payload_path,
    )
    return payload


def build_client_session_payload(
    manager: AFSManager,
    context_path: Path,
    *,
    client: str,
    session_id: str = "",
    config_path: Path | None = None,
    cwd: Path | None = None,
    query: str = "",
    task: str = "",
    model: str = "codex",
    workflow: str = "general",
    tool_profile: str = "default",
    pack_mode: str = "focused",
    token_budget: int | None = None,
    include_content: bool = False,
    max_query_results: int = 6,
    max_embedding_results: int = 4,
    include_pack: bool = True,
    skills_prompt: str = "",
    include_skills: bool = True,
    skills_top_k: int = 10,
    changed_paths: list[str] | None = None,
    verification_profile: str = "",
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Build a client-ready session payload with reusable artifact paths."""
    resolved_context = context_path.expanduser().resolve()
    resolved_cwd = (cwd or Path.cwd()).expanduser().resolve()
    bootstrap_state, bootstrap_payload = _bootstrap_bundle(
        manager,
        resolved_context,
        client=client,
        write_artifacts=write_artifacts,
    )
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "client": client,
        "session_id": session_id,
        "context_path": str(resolved_context),
        "config_path": str(config_path) if config_path else "",
        "cwd": str(resolved_cwd),
        "bootstrap": bootstrap_payload,
        "pack": {"available": False, "artifact_paths": {}},
        "skills": {"available": False, "artifact_paths": {}},
        "prompt": {"available": False, "artifact_paths": {}},
        "repo_policy": {"available": False},
        "verification_plan": {"available": False},
        "structured_guidance": {},
        "cli_hints": _build_cli_hints(
            workspace_path=resolved_cwd,
            bootstrap_state=bootstrap_state,
        ),
        "integration": _integration_contract(),
        "activity": _initial_activity_state(),
        "artifact_paths": {},
    }

    if include_pack:
        payload["pack"] = _pack_summary(
            manager,
            resolved_context,
            query=query or "",
            task=task or "",
            model=model,
            workflow=workflow,
            tool_profile=tool_profile,
            pack_mode=pack_mode,
            token_budget=token_budget,
            include_content=include_content,
            max_query_results=max_query_results,
            max_embedding_results=max_embedding_results,
            write_artifacts=write_artifacts,
        )

    if include_skills:
        payload["skills"] = _skills_summary(
            manager,
            resolved_context,
            client=client,
            prompt=skills_prompt,
            top_k=skills_top_k,
            write_artifacts=write_artifacts,
        )

    repo_policy = load_repo_policy(start_dir=resolved_cwd)
    payload["verification_plan"] = build_verification_plan(
        config=manager.config,
        cwd=resolved_cwd,
        workflow=workflow,
        tool_profile=tool_profile,
        matched_skills=((payload.get("skills") or {}) if isinstance(payload.get("skills"), dict) else {}).get("matches"),
        changed_paths=changed_paths,
        verification_profile=verification_profile,
        policy_summary=None,
    )
    repo_root = Path(
        str((payload.get("verification_plan") or {}).get("repo_root", resolved_cwd))
    ).expanduser().resolve()
    payload["repo_policy"] = evaluate_repo_policy(
        repo_policy,
        repo_root=repo_root,
        changed_paths=list((payload.get("verification_plan") or {}).get("changed_paths") or []),
    )
    payload["verification_plan"] = build_verification_plan(
        config=manager.config,
        cwd=repo_root,
        workflow=workflow,
        tool_profile=tool_profile,
        matched_skills=((payload.get("skills") or {}) if isinstance(payload.get("skills"), dict) else {}).get("matches"),
        changed_paths=list((payload.get("verification_plan") or {}).get("changed_paths") or []),
        verification_profile=verification_profile,
        policy_summary=payload["repo_policy"],
    )
    payload["structured_guidance"] = build_structured_guidance(
        model=model,
        workflow=workflow,
        policy_summary=payload["repo_policy"],
    )

    payload["prompt"] = _prompt_summary(
        manager,
        resolved_context,
        client=client,
        model=model,
        workflow=workflow,
        tool_profile=tool_profile,
        token_budget=token_budget,
        config_path=config_path,
        session_state=bootstrap_state,
        pack_state=dict(payload.get("pack") or {}),
        skills_state=dict(payload.get("skills") or {}),
        verification_state=dict(payload.get("verification_plan") or {}),
        policy_state=dict(payload.get("repo_policy") or {}),
        structured_guidance=dict(payload.get("structured_guidance") or {}),
        write_artifacts=write_artifacts,
    )

    if write_artifacts:
        write_client_session_payload_artifact(
            manager,
            resolved_context,
            client=client,
            payload=payload,
        )

    return payload
