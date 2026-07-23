"""Session-preparation helpers for client wrappers and external harnesses."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chat_registry import load_chat_registry
from .context_layout import LAYOUT_VERSION, _atomic_write_text, detect_layout_version
from .context_pack import build_context_pack, write_context_pack_artifacts
from .context_paths import resolve_agent_output_root
from .manager import AFSManager
from .model_profiles import profile_for_client_model
from .model_prompts import build_model_system_prompt
from .path_safety import assert_no_linklike_components, lexical_absolute
from .profiles import resolve_active_profile
from .repo_policy import evaluate_repo_policy, load_repo_policy
from .scopes import resolve_scope
from .session_bootstrap import build_session_bootstrap, write_session_bootstrap_artifacts
from .session_workflows import build_session_execution_profile
from .skills import build_skill_matches, resolve_skill_roots
from .verification import (
    build_structured_guidance,
    build_verification_plan,
    redact_legacy_verification_command,
    redact_verification_argv,
    redact_verification_plan,
    verification_item_id,
)

_RECENT_ACTIVITY_LIMIT = 20
_PROMPT_PREVIEW_CHARS = 180
_SUMMARY_PREVIEW_CHARS = 160
_SYSTEM_PROMPT_PREVIEW_CHARS = 320
_DEFAULT_SYSTEM_PROMPT_TOKEN_BUDGET = 4000
_WORKFLOW_SNAPSHOT_STEP_LIMIT = 6
_WORKFLOW_SNAPSHOT_SKILL_LIMIT = 4
_VERIFICATION_RECORD_LIMIT = 10
logger = logging.getLogger(__name__)


def _as_dict(value: Any) -> dict[str, Any]:
    """Return a mapping-shaped payload value without unsafe chained access."""
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    """Return a list-shaped payload value without unsafe chained access."""
    return value if isinstance(value, list) else []


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    """Return only mapping entries from an untrusted payload list."""
    return [entry for entry in _as_list(value) if isinstance(entry, dict)]


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
    *,
    scope_id: str = "common",
) -> dict[str, Path]:
    output_root = resolve_agent_output_root(
        context_path,
        config=getattr(manager, "config", None),
        scope_id=scope_id,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if detect_layout_version(context_path) == LAYOUT_VERSION:
        output_root = assert_no_linklike_components(
            output_root,
            boundary=context_path.expanduser().resolve(),
            allow_missing=False,
        )
    slug = _client_slug(client)
    return {
        "payload_json": output_root / f"session_client_{slug}.json",
        "skills_json": output_root / f"session_skills_{slug}.json",
        "prompt_json": output_root / f"session_system_prompt_{slug}.json",
        "prompt_text": output_root / f"session_system_prompt_{slug}.txt",
    }


def _write_client_artifact(
    context_path: Path,
    output_root: Path,
    path: Path,
    text: str,
) -> None:
    """Write a client artifact without following v2 namespace links."""
    if detect_layout_version(context_path) != LAYOUT_VERSION:
        path.write_text(text, encoding="utf-8")
        return

    trusted_root = assert_no_linklike_components(
        output_root,
        boundary=context_path.expanduser().resolve(),
        allow_missing=False,
    )
    safe_path = assert_no_linklike_components(path, boundary=trusted_root)
    if safe_path == trusted_root:
        raise ValueError("client artifact path must name a file within its output root")
    _atomic_write_text(safe_path, text)


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
    agent_jobs = summary.get("agent_jobs") or {}
    work_assistant = summary.get("work_assistant") or {}
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
    if int(agent_jobs.get("inbox_attention_count", 0) or 0) > 0:
        notes.append(
            "Agent job inbox has background output to review before starting more work."
        )
    if work_assistant.get("pending_approvals"):
        notes.append(
            "Work assistant has pending external-write approvals to review before using connector write tools."
        )
    work_summary = work_assistant.get("summary") if isinstance(work_assistant, dict) else {}
    if work_assistant and isinstance(work_summary, dict) and work_summary.get("communication_samples", 0) == 0:
        notes.append(
            "For work-context writing, collect or inspect communication samples before imitating the user's tone."
        )

    return {
        "workspace_path": str(resolved_workspace),
        "query_shortcut": f"afs query <text> --path {quoted_workspace}",
        "query_canonical": f"afs context query <text> --path {quoted_workspace}",
        "index_rebuild": f"afs index rebuild --path {quoted_workspace}",
        "agent_jobs_inbox": f"afs agent-jobs inbox --path {quoted_workspace}",
        "work_summary": f"afs work --path {quoted_workspace}",
        "work_approvals": f"afs work approvals list --path {quoted_workspace}",
        "work_communication": f"afs work communication preflight --path {quoted_workspace}",
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
        "model_profile": profile_for_client_model(client, client).to_dict(),
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
    pack_state = _as_dict(normalized.get("pack"))
    normalized.setdefault(
        "model_profile",
        profile_for_client_model(
            str(normalized.get("client", "generic")),
            str(pack_state.get("model", normalized.get("client", "generic"))),
        ).to_dict(),
    )
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
    project_path: Path | None = None,
) -> Path:
    """Resolve the canonical client-session payload artifact path."""
    scoped = resolve_scope(context_path, requester_path=project_path)
    return _client_output_paths(
        manager,
        context_path,
        client,
        scope_id=scoped.scope_id,
    )["payload_json"]


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
    scoped = resolve_scope(context_path, requester_path=cwd)
    output_paths = _client_output_paths(
        manager,
        context_path,
        client,
        scope_id=scoped.scope_id,
    )
    output_root = output_paths["payload_json"].parent
    if payload_file:
        if scoped.layout_version == LAYOUT_VERSION:
            candidate = lexical_absolute(Path(payload_file))
            try:
                candidate.relative_to(output_root)
            except ValueError:
                try:
                    candidate.relative_to(context_path.expanduser().resolve())
                except ValueError:
                    pass
                else:
                    raise PermissionError(
                        "client payload scope does not match the requester scope"
                    ) from None
            payload_path = assert_no_linklike_components(
                candidate,
                boundary=output_root,
            )
            if payload_path == output_root:
                raise ValueError(
                    "client payload path must name a file within its authorized output root"
                )
        else:
            payload_path = Path(payload_file).expanduser().resolve()
    else:
        payload_path = output_paths["payload_json"]

    if scoped.layout_version == LAYOUT_VERSION:
        payload_path = assert_no_linklike_components(
            payload_path,
            boundary=output_root,
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
    persisted_scope = str(payload.get("scope_id", "") or "").strip()
    if cwd is not None and scoped.layout_version == LAYOUT_VERSION and persisted_scope not in {
        "",
        scoped.scope_id,
    }:
        raise PermissionError(
            f"session payload scope {persisted_scope!r} does not match {scoped.scope_id!r}"
        )
    payload.setdefault("scope_id", scoped.scope_id)
    payload.setdefault("project_id", scoped.project_id)
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
    if text in {"blocked", "policy_blocked"}:
        return "blocked"
    if text in {"missing", "pending", "not_required"}:
        return text
    if text.startswith("verification_"):
        return _normalize_verification_status(text.removeprefix("verification_"))
    return ""


def _verification_requirements(payload: dict[str, Any]) -> dict[str, Any]:
    verification_plan = _as_dict(payload.get("verification_plan"))
    selected_checks = _as_list(verification_plan.get("selected_checks"))
    preflight_required = bool(verification_plan.get("preflight_required"))
    if verification_plan.get("available") and (selected_checks or preflight_required):
        planned_expected: list[str] = []
        required = preflight_required
        required_checks: dict[str, int] = {}
        required_items: dict[str, str] = {}
        plan_items: dict[str, str] = {}
        allow_legacy_shell = bool(verification_plan.get("allow_legacy_shell", False))
        preflight_item_id = str(
            verification_plan.get("preflight_item_id", "")
        ).strip()
        if preflight_item_id:
            plan_items[preflight_item_id] = "verification-preflight"
            if preflight_required:
                required_items[preflight_item_id] = "verification-preflight"
                required_checks["verification-preflight"] = 1
        for check in selected_checks:
            if not isinstance(check, dict):
                continue
            name = str(check.get("name", "")).strip()
            check_required = bool(check.get("required"))
            required = required or check_required
            required_item_count = 0
            executions = _as_list(check.get("executions"))
            commands = _as_list(check.get("commands"))
            execution_item_ids = list(check.get("execution_item_ids") or [])
            command_item_ids = list(check.get("command_item_ids") or [])
            rendered = False
            for item_index, execution in enumerate(executions):
                if not isinstance(execution, dict):
                    continue
                argv = execution.get("argv")
                if not isinstance(argv, list) or not argv:
                    continue
                text = shlex.join(
                    redact_verification_argv(
                        argv,
                        execution.get("redact_argv_indices"),
                    )
                )
                planned_expected.append(f"{name}: {text}" if name else text)
                rendered = True
                required_item_count += 1
                item_id = (
                    str(execution_item_ids[item_index])
                    if item_index < len(execution_item_ids)
                    else verification_item_id(name, "execution", item_index)
                )
                plan_items[item_id] = name
                if check_required and name:
                    required_items[item_id] = name
            for item_index, command in enumerate(commands):
                text = str(command).strip()
                if not text:
                    continue
                legacy_status = (
                    "deprecated; explicit opt-in enabled"
                    if allow_legacy_shell
                    else "deprecated; blocked; explicit opt-in required"
                )
                rendered_command = (
                    f"legacy shell ({legacy_status}): "
                    f"{redact_legacy_verification_command(text)}"
                )
                planned_expected.append(
                    f"{name}: {rendered_command}" if name else rendered_command
                )
                rendered = True
                required_item_count += 1
                item_id = (
                    str(command_item_ids[item_index])
                    if item_index < len(command_item_ids)
                    else verification_item_id(name, "legacy", item_index)
                )
                plan_items[item_id] = name
                if check_required and name:
                    required_items[item_id] = name
            if not rendered and name:
                planned_expected.append(f"{name}: review changed scope")
                if check_required:
                    item_id = str(
                        check.get("review_item_id")
                        or verification_item_id(name, "review", 0)
                    )
                    required_items[item_id] = name
                else:
                    item_id = str(
                        check.get("review_item_id")
                        or verification_item_id(name, "review", 0)
                    )
                plan_items[item_id] = name
            if check_required and name:
                required_checks[name] = required_checks.get(name, 0) + max(
                    1, required_item_count
                )
        return {
            "required": required,
            "workflow": str(verification_plan.get("workflow", "")).strip(),
            "tool_profile": str(verification_plan.get("tool_profile", "")).strip(),
            "expected": planned_expected,
            "required_checks": required_checks,
            "required_items": required_items,
            "plan_items": plan_items,
            "verification_run_id": str(
                verification_plan.get("verification_run_id", "")
            ).strip(),
        }

    pack = _as_dict(payload.get("pack"))
    prompt = _as_dict(payload.get("prompt"))
    skills = _as_dict(payload.get("skills"))

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
        except (KeyError, TypeError, ValueError):
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
        "check_name": str(event_payload.get("verification_check", "")).strip(),
        "item_id": str(event_payload.get("verification_item_id", "")).strip(),
    }
    return {key: value for key, value in record.items() if value not in ("", None, [], {})}


def _build_session_verification_state(
    payload: dict[str, Any],
    *,
    final: bool,
    updated_at: str = "",
) -> dict[str, Any]:
    activity = _as_dict(payload.get("activity"))
    current = _as_dict(activity.get("verification"))
    records = _as_list(current.get("records"))

    requirements = _verification_requirements(payload)
    required = bool(requirements.get("required"))
    expected = list(requirements.get("expected") or [])
    required_checks = {
        str(name): int(count)
        for name, count in dict(requirements.get("required_checks") or {}).items()
        if str(name).strip() and isinstance(count, int) and count > 0
    }
    required_items = {
        str(item_id): str(check_name)
        for item_id, check_name in dict(
            requirements.get("required_items") or {}
        ).items()
        if str(item_id).strip() and str(check_name).strip()
    }
    plan_items = {
        str(item_id): str(check_name)
        for item_id, check_name in dict(requirements.get("plan_items") or {}).items()
        if str(item_id).strip() and str(check_name).strip()
    }
    verification_run_id = str(requirements.get("verification_run_id", "")).strip()

    retained_indices = set(range(max(0, len(records) - _VERIFICATION_RECORD_LIMIT), len(records)))
    if plan_items:
        latest_plan_record: dict[str, int] = {}
        for index, entry in enumerate(records):
            if not isinstance(entry, dict):
                continue
            item_id = str(entry.get("item_id", "")).strip()
            if item_id in plan_items:
                latest_plan_record[item_id] = index
        retained_indices.update(latest_plan_record.values())

    normalized_records: list[dict[str, Any]] = []
    commands: list[str] = []
    command_seen: set[str] = set()
    for index in sorted(retained_indices):
        entry = records[index]
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

    relevant_records = normalized_records
    if plan_items:
        relevant_records = [
            entry
            for entry in normalized_records
            if str(entry.get("item_id", "")).strip() in plan_items
            or (
                not verification_run_id
                and len(required_items) == 1
                and not str(entry.get("item_id", "")).strip()
            )
        ]
    statuses = [str(entry.get("status", "")).strip() for entry in relevant_records]
    passed_item_ids: set[str] = set()
    generic_passed = 0
    for entry in relevant_records:
        if entry.get("status") != "passed":
            continue
        item_id = str(entry.get("item_id", "")).strip()
        if item_id in required_items:
            passed_item_ids.add(item_id)
        elif not item_id:
            generic_passed += 1
    if not verification_run_id and len(required_items) == 1 and generic_passed:
        passed_item_ids.add(next(iter(required_items)))
    completed_required_checks = sorted(
        name
        for name in required_checks
        if {
            item_id
            for item_id, check_name in required_items.items()
            if check_name == name
        }.issubset(passed_item_ids)
    )
    required_coverage_complete = bool(required_items) and set(required_items).issubset(
        passed_item_ids
    )
    if "blocked" in statuses:
        status = "blocked"
        message = "Verification was blocked by policy."
    elif "failed" in statuses:
        status = "failed"
        message = "Recorded verification failed."
    elif not required:
        status = "not_required"
        message = "Verification not required for this workflow."
    elif required_checks and required_coverage_complete:
        status = "passed"
        message = "Verification recorded."
    elif required_checks and final:
        status = "missing"
        message = "Required verification checks were not all recorded before session end."
    elif required_checks:
        status = "pending"
        message = "Required verification checks are still pending."
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
        "required_checks": required_checks,
        "required_items": sorted(required_items),
        "completed_required_items": sorted(passed_item_ids),
        "verification_run_id": verification_run_id,
        "completed_required_checks": completed_required_checks,
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
    activity = _as_dict(payload.get("activity"))
    verification = _as_dict(activity.get("verification"))
    policy = _as_dict(payload.get("repo_policy"))

    counts: dict[str, int] = {}
    signals: list[str] = []

    matched_risks = _as_list(policy.get("matched_risks"))
    anti_pattern_hits = _as_list(policy.get("anti_pattern_hits"))
    if matched_risks:
        counts["review_risk"] = len(matched_risks)
        signals.append("review_risk")
    if anti_pattern_hits:
        counts["policy_violation"] = len(anti_pattern_hits)
        signals.append("policy_violation")

    verification_status = str(verification.get("status", "")).strip()
    if verification_status in {"blocked", "failed", "missing", "skipped"}:
        counts[f"verification_{verification_status}"] = 1
        signals.append(f"verification_{verification_status}")

    message_parts: list[str] = []
    if counts.get("review_risk"):
        message_parts.append(f"{counts['review_risk']} repo risk alerts")
    if counts.get("policy_violation"):
        message_parts.append(f"{counts['policy_violation']} policy violations")
    if verification_status in {"blocked", "failed", "missing", "skipped"}:
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

    pack = _as_dict(payload.get("pack"))
    skills = _as_dict(payload.get("skills"))
    current_prompt = _as_dict(activity.get("current_prompt"))
    current_turn = _as_dict(activity.get("current_turn"))
    last_task = _as_dict(activity.get("last_task"))
    session_state = _as_dict(activity.get("session_state"))
    verification = _as_dict(activity.get("verification"))
    feedback = _as_dict(activity.get("feedback"))
    counters = _as_dict(activity.get("counters"))
    recent_events = _as_dict_list(activity.get("recent_events"))

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
    project_path: Path | None = None,
    skills_prompt: str = "",
    skills_top_k: int = 5,
    include_skills: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = build_session_bootstrap(
        manager,
        context_path,
        project_path=project_path,
        agent_name=f"{client}-client",
        record_event=False,
        skills_prompt=skills_prompt,
        skills_top_k=skills_top_k,
        include_skills=include_skills,
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
        "scope_id": summary.get("scope_id", "common"),
        "project_id": summary.get("project_id", ""),
        "profile": summary["profile"],
        "stale_mounts": list(summary.get("stale_mounts", [])),
        "work_assistant": summary.get("work_assistant", {}),
        "skills": summary.get("skills", {}),
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
    semantic: bool,
    max_query_results: int,
    max_embedding_results: int,
    write_artifacts: bool,
    project_path: Path | None = None,
) -> dict[str, Any]:
    pack = build_context_pack(
        manager,
        context_path,
        project_path=project_path,
        query=query,
        task=task,
        model=model,
        workflow=workflow,
        tool_profile=tool_profile,
        pack_mode=pack_mode,
        token_budget=token_budget,
        include_content=include_content,
        semantic=semantic,
        max_query_results=max_query_results,
        max_embedding_results=max_embedding_results,
    )
    artifact_paths = pack.get("artifact_paths") or {}
    if write_artifacts and not bool((pack.get("cache") or {}).get("hit")):
        artifact_paths = write_context_pack_artifacts(manager, context_path, pack)
    return {
        "available": True,
        "scope_id": pack.get("scope_id", "common"),
        "project_id": pack.get("project_id", ""),
        "project_path": pack.get("project_path", ""),
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
    scope_id: str = "common",
    fallback_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = resolve_active_profile(manager.config)
    roots = resolve_skill_roots(
        list(profile.skill_roots),
        afs_root=os.getenv("AFS_ROOT", "").strip() or None,
    )
    matches = build_skill_matches(
        prompt,
        roots,
        profile=profile.name,
        top_k=top_k,
    )
    prompt_source = "explicit" if prompt.strip() else "none"
    if not matches and not prompt.strip() and isinstance(fallback_state, dict):
        fallback_matches = fallback_state.get("matches")
        if isinstance(fallback_matches, list):
            matches = [dict(match) for match in fallback_matches if isinstance(match, dict)]
            prompt_source = str(fallback_state.get("prompt_source") or "session_state")

    payload = {
        "available": True,
        "client": client,
        "profile": profile.name,
        "prompt": prompt,
        "prompt_source": prompt_source,
        "roots": [str(path) for path in roots],
        "matches": matches,
        "artifact_paths": {},
    }
    if write_artifacts:
        paths = _client_output_paths(
            manager,
            context_path,
            client,
            scope_id=scope_id,
        )
        payload["artifact_paths"] = {"json": str(paths["skills_json"])}
        _write_client_artifact(
            context_path,
            paths["skills_json"].parent,
            paths["skills_json"],
            json.dumps(payload, indent=2) + "\n",
        )
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
    except Exception:  # noqa: BLE001 - registry config is an optional external boundary.
        logger.debug("Unable to load chat registry", exc_info=True)
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
    scope_id: str = "common",
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
        paths = _client_output_paths(
            manager,
            context_path,
            client,
            scope_id=scope_id,
        )
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
        _write_client_artifact(
            context_path,
            paths["prompt_text"].parent,
            paths["prompt_text"],
            prompt_text + "\n",
        )
        _write_client_artifact(
            context_path,
            paths["prompt_json"].parent,
            paths["prompt_json"],
            json.dumps(prompt_payload, indent=2) + "\n",
        )
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
    requester_path = (
        Path(str(payload.get("cwd", ""))).expanduser()
        if str(payload.get("cwd", "")).strip()
        else None
    )
    resolved_payload, resolved_path = _load_client_session_payload(
        manager,
        context_path,
        client=client,
        session_id=str(payload.get("session_id", "")).strip(),
        config_path=Path(str(payload.get("config_path", ""))).expanduser().resolve()
        if str(payload.get("config_path", "")).strip()
        else None,
        cwd=requester_path.resolve() if requester_path is not None else None,
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
    scoped = resolve_scope(
        context_path,
        requester_path=requester_path,
    )
    output_root = resolve_agent_output_root(
        context_path,
        config=getattr(manager, "config", None),
        scope_id=scoped.scope_id,
    )
    _write_client_artifact(
        context_path,
        output_root,
        resolved_path,
        json.dumps(resolved_payload, indent=2) + "\n",
    )
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
    semantic: bool = False,
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
    skills_focus = skills_prompt.strip() or task.strip() or query.strip()
    bootstrap_state, bootstrap_payload = _bootstrap_bundle(
        manager,
        resolved_context,
        client=client,
        write_artifacts=write_artifacts,
        project_path=resolved_cwd,
        skills_prompt=skills_focus if include_skills else "",
        skills_top_k=skills_top_k,
        include_skills=include_skills,
    )
    model_profile = profile_for_client_model(client, model).to_dict()
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "client": client,
        "session_id": session_id,
        "model_profile": model_profile,
        "context_path": str(resolved_context),
        "scope_id": bootstrap_state.get("scope_id", "common"),
        "project_id": bootstrap_state.get("project_id", ""),
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
            semantic=semantic,
            max_query_results=max_query_results,
            max_embedding_results=max_embedding_results,
            write_artifacts=write_artifacts,
            project_path=resolved_cwd,
        )

    if include_skills:
        payload["skills"] = _skills_summary(
            manager,
            resolved_context,
            client=client,
            prompt=skills_focus,
            top_k=skills_top_k,
            write_artifacts=write_artifacts,
            scope_id=str(bootstrap_state.get("scope_id", "common") or "common"),
            fallback_state=(
                bootstrap_state.get("skills")
                if isinstance(bootstrap_state.get("skills"), dict)
                else None
            ),
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
    discovery_error = str(
        (payload.get("verification_plan") or {}).get("discovery_error", "")
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
        discovery_error=discovery_error,
    )
    payload["verification_plan"] = redact_verification_plan(
        payload["verification_plan"]
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
        scope_id=str(bootstrap_state.get("scope_id", "common") or "common"),
    )

    if write_artifacts:
        write_client_session_payload_artifact(
            manager,
            resolved_context,
            client=client,
            payload=payload,
        )

    return payload
