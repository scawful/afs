"""Session-preparation helpers for client wrappers and external harnesses."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context_pack import build_context_pack, write_context_pack_artifacts
from .context_paths import resolve_agent_output_root
from .manager import AFSManager
from .profiles import resolve_active_profile
from .session_bootstrap import build_session_bootstrap, write_session_bootstrap_artifacts
from .skills import discover_skills, resolve_skill_roots, score_skill_relevance

_RECENT_ACTIVITY_LIMIT = 20
_PROMPT_PREVIEW_CHARS = 180
_SUMMARY_PREVIEW_CHARS = 160

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
    }


def _integration_contract() -> dict[str, Any]:
    return {
        "version": 1,
        "notify_command": "afs session event",
        "hook_command": "afs session hook",
        "supported_events": supported_session_activity_events(),
        "payload_file": "",
    }


def _default_client_session_payload(
    *,
    context_path: Path,
    client: str,
    session_id: str = "",
    config_path: Path | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "client": client,
        "session_id": session_id,
        "context_path": str(context_path.expanduser().resolve()),
        "config_path": str(config_path) if config_path else "",
        "cwd": str((cwd or Path.cwd()).expanduser().resolve()),
        "bootstrap": {"available": False, "artifact_paths": {}},
        "pack": {"available": False, "artifact_paths": {}},
        "skills": {"available": False, "artifact_paths": {}},
        "integration": _integration_contract(),
        "activity": _initial_activity_state(),
        "artifact_paths": {},
    }


def _ensure_client_session_payload_shape(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
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


def _bootstrap_summary(
    manager: AFSManager,
    context_path: Path,
    *,
    client: str,
    write_artifacts: bool,
) -> dict[str, Any]:
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
    return {
        "available": True,
        "context_path": summary["context_path"],
        "project": summary["project"],
        "profile": summary["profile"],
        "stale_mounts": list(summary.get("stale_mounts", [])),
        "recommended_actions": list(summary.get("recommended_actions", [])),
        "artifact_paths": artifact_paths,
    }


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
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Build a client-ready session payload with reusable artifact paths."""
    resolved_context = context_path.expanduser().resolve()
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "client": client,
        "session_id": session_id,
        "context_path": str(resolved_context),
        "config_path": str(config_path) if config_path else "",
        "cwd": str((cwd or Path.cwd()).expanduser().resolve()),
        "bootstrap": _bootstrap_summary(
            manager,
            resolved_context,
            client=client,
            write_artifacts=write_artifacts,
        ),
        "pack": {"available": False, "artifact_paths": {}},
        "skills": {"available": False, "artifact_paths": {}},
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

    if write_artifacts:
        write_client_session_payload_artifact(
            manager,
            resolved_context,
            client=client,
            payload=payload,
        )

    return payload
