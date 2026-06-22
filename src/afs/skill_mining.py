"""Mine repeated successful session traces into reviewable skill candidates."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .event_log import build_session_replay, list_recorded_sessions
from .models import MountType

_DEFAULT_LOOKBACK_HOURS = 24 * 7
_DEFAULT_MAX_SESSIONS = 50
_DEFAULT_MIN_OCCURRENCES = 2
_DEFAULT_MAX_CANDIDATES = 10
_DEFAULT_REPLAY_LIMIT = 200
_DEFAULT_REVIEW_LIMIT = 10
_REVIEW_STATE_FILENAME = "skill_candidate_review_state.json"
_MAX_TOOL_STEPS = 6
_MAX_PROMPT_EXAMPLES = 3
_MAX_TRIGGER_TERMS = 6
_MAX_PATH_HINTS = 5
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "agent",
        "agents",
        "against",
        "build",
        "change",
        "changes",
        "check",
        "common",
        "continue",
        "debug",
        "does",
        "done",
        "from",
        "have",
        "into",
        "investigate",
        "make",
        "need",
        "next",
        "prompt",
        "review",
        "session",
        "should",
        "some",
        "status",
        "task",
        "that",
        "them",
        "this",
        "turn",
        "using",
        "want",
        "what",
        "when",
        "where",
        "with",
        "work",
        "workflow",
    }
)


def mine_skill_candidates(
    context_path: Path,
    *,
    lookback_hours: int = _DEFAULT_LOOKBACK_HOURS,
    max_sessions: int = _DEFAULT_MAX_SESSIONS,
    min_occurrences: int = _DEFAULT_MIN_OCCURRENCES,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    replay_limit: int = _DEFAULT_REPLAY_LIMIT,
    config: Any = None,
) -> dict[str, Any]:
    """Mine repeated successful session traces into candidate skill summaries."""
    resolved_context = context_path.expanduser().resolve()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max(1, lookback_hours))

    session_rows = list_recorded_sessions(
        resolved_context,
        config=config,
        limit=max(max_sessions, 1) * 4,
    )

    replayed = 0
    analyzed: list[dict[str, Any]] = []
    for row in session_rows:
        if len(analyzed) >= max(max_sessions, 1):
            break
        session_id = str(row.get("session_id", "")).strip()
        ended_at = _parse_timestamp(str(row.get("end", "")).strip())
        if not session_id or ended_at is None or ended_at < cutoff:
            continue
        replay = build_session_replay(
            resolved_context,
            session_id=session_id,
            limit=max(replay_limit, 0),
            include_payloads=False,
            config=config,
        )
        replayed += 1
        summary = _summarize_session_trace(replay)
        if summary is not None:
            analyzed.append(summary)

    successful = [summary for summary in analyzed if summary["successful"]]
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for summary in successful:
        signature = tuple(summary["tool_signature"])
        if len(signature) < 2:
            continue
        grouped[signature].append(summary)

    candidates: list[dict[str, Any]] = []
    for signature, cluster in grouped.items():
        if len(cluster) < max(1, min_occurrences):
            continue
        candidates.append(_build_candidate(signature, cluster))

    candidates.sort(
        key=lambda item: (
            -float(item.get("confidence", 0.0)),
            -int(item.get("occurrences", 0)),
            str(item.get("name", "")),
        )
    )
    candidates = candidates[: max(max_candidates, 0)]

    return {
        "generated_at": now.isoformat(),
        "context_path": str(resolved_context),
        "lookback_hours": max(1, lookback_hours),
        "max_sessions": max(max_sessions, 1),
        "min_occurrences": max(1, min_occurrences),
        "max_candidates": max(max_candidates, 0),
        "sessions_considered": len(session_rows),
        "sessions_replayed": replayed,
        "sessions_analyzed": len(analyzed),
        "successful_sessions": len(successful),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "artifact_paths": {},
    }


def write_skill_candidate_artifacts(
    manager: Any,
    context_path: Path,
    payload: dict[str, Any],
) -> dict[str, str]:
    """Write mined skill candidates into scratchpad review artifacts."""
    scratchpad_root = manager.resolve_mount_root(context_path, MountType.SCRATCHPAD)
    output_root = scratchpad_root / "skill_candidates"
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_root / f"skill_candidates_{stamp}.json"
    markdown_path = output_root / f"skill_candidates_{stamp}.md"

    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(payload) + "\n", encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }


def _skill_candidate_output_root(manager: Any, context_path: Path) -> Path:
    scratchpad_root = manager.resolve_mount_root(context_path, MountType.SCRATCHPAD)
    output_root = scratchpad_root / "skill_candidates"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _skill_candidate_review_state_path(manager: Any, context_path: Path) -> Path:
    return _skill_candidate_output_root(manager, context_path) / _REVIEW_STATE_FILENAME


def load_skill_candidate_review_state(
    manager: Any,
    context_path: Path,
) -> dict[str, Any]:
    """Load persisted review state for mined candidates."""
    state_path = _skill_candidate_review_state_path(manager, context_path)
    if not state_path.exists():
        return {"updated_at": "", "entries": {}}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"updated_at": "", "entries": {}}
    if not isinstance(payload, dict):
        return {"updated_at": "", "entries": {}}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    return {
        "updated_at": str(payload.get("updated_at", "")).strip(),
        "entries": {
            str(key).strip(): value
            for key, value in entries.items()
            if str(key).strip() and isinstance(value, dict)
        },
        "state_path": str(state_path),
    }


def record_skill_candidate_review_state(
    manager: Any,
    context_path: Path,
    *,
    candidate_id: str,
    status: str,
    artifact_path: str = "",
    skill_name: str = "",
    skill_path: str = "",
    root: str = "",
) -> dict[str, Any]:
    """Persist review state for a candidate decision."""
    resolved_context = context_path.expanduser().resolve()
    state = load_skill_candidate_review_state(manager, resolved_context)
    state_path = Path(
        str(state.get("state_path") or _skill_candidate_review_state_path(manager, resolved_context))
    ).expanduser().resolve()
    normalized_id = str(candidate_id or "").strip()
    if not normalized_id:
        raise ValueError("Candidate id is required to record review state.")

    normalized_status = str(status or "").strip().lower() or "pending"
    now = datetime.now(timezone.utc).isoformat()
    entries = dict(state.get("entries") or {})
    existing = entries.get(normalized_id)
    entry = dict(existing) if isinstance(existing, dict) else {}
    entry.update(
        {
            "candidate_id": normalized_id,
            "status": normalized_status,
            "artifact_path": str(artifact_path or entry.get("artifact_path", "")).strip(),
            "skill_name": str(skill_name or entry.get("skill_name", "")).strip(),
            "skill_path": str(skill_path or entry.get("skill_path", "")).strip(),
            "root": str(root or entry.get("root", "")).strip(),
            "updated_at": now,
        }
    )
    if normalized_status in {"promoted", "rejected", "archived"}:
        entry[f"{normalized_status}_at"] = now
    entries[normalized_id] = entry

    payload = {"updated_at": now, "entries": entries}
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    entry["state_path"] = str(state_path)
    return entry


def list_skill_candidate_artifact_paths(
    manager: Any,
    context_path: Path,
) -> list[Path]:
    """List scratchpad skill-candidate JSON artifacts newest-first."""
    output_root = _skill_candidate_output_root(manager, context_path)
    return sorted(
        output_root.glob("skill_candidates_*.json"),
        key=lambda path: path.name,
        reverse=True,
    )


def load_skill_candidate_artifact(path: Path | str) -> dict[str, Any]:
    """Load a skill-candidate artifact from disk."""
    artifact_path = Path(path).expanduser().resolve()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Skill candidate artifact is not a JSON object: {artifact_path}")
    payload["artifact_path"] = str(artifact_path)
    return payload


def review_skill_candidates(
    manager: Any,
    context_path: Path,
    *,
    artifact_path: str | Path | None = None,
    candidate_id: str = "",
    status_filter: str = "",
    limit: int = _DEFAULT_REVIEW_LIMIT,
) -> dict[str, Any]:
    """Load and filter mined skill candidates for review."""
    resolved_context = context_path.expanduser().resolve()
    artifacts = list_skill_candidate_artifact_paths(manager, resolved_context)
    requested_id = str(candidate_id or "").strip()
    requested_status = str(status_filter or "").strip().lower()
    selected_path = (
        Path(artifact_path).expanduser().resolve()
        if artifact_path
        else (artifacts[0] if artifacts else None)
    )
    if selected_path is None:
        return {
            "context_path": str(resolved_context),
            "artifact_path": "",
            "artifact_count": 0,
            "available_artifacts": [],
            "candidate_filter": requested_id,
            "status_filter": requested_status,
            "candidate_count": 0,
            "total_candidate_count": 0,
            "status_counts": {},
            "review_state_path": str(_skill_candidate_review_state_path(manager, resolved_context)),
            "candidates": [],
        }

    payload = load_skill_candidate_artifact(selected_path)
    state = load_skill_candidate_review_state(manager, resolved_context)
    entries = state.get("entries") if isinstance(state.get("entries"), dict) else {}
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        candidates = []

    status_counts: Counter[str] = Counter()
    filtered: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        enriched = dict(candidate)
        identifier = str(enriched.get("id", "")).strip() or str(enriched.get("name", "")).strip()
        review_entry = entries.get(identifier)
        if isinstance(review_entry, dict):
            enriched["review_state"] = dict(review_entry)
            enriched["status"] = str(review_entry.get("status", "")).strip() or "pending"
        else:
            enriched["review_state"] = {}
            enriched["status"] = "pending"
        status_counts[enriched["status"]] += 1
        if requested_id:
            name = str(enriched.get("name", "")).strip()
            if requested_id not in {name, identifier}:
                continue
        if requested_status and enriched["status"] != requested_status:
            continue
        filtered.append(enriched)

    shown = filtered[: max(limit, 0)]
    return {
        "context_path": str(resolved_context),
        "artifact_path": str(selected_path),
        "generated_at": str(payload.get("generated_at", "")).strip(),
        "artifact_count": len(artifacts),
        "available_artifacts": [str(path) for path in artifacts[:5]],
        "candidate_filter": requested_id,
        "status_filter": requested_status,
        "candidate_count": len(shown),
        "total_candidate_count": len(filtered),
        "status_counts": dict(sorted(status_counts.items())),
        "review_state_path": str(state.get("state_path", "")),
        "sessions_analyzed": int(payload.get("sessions_analyzed", 0) or 0),
        "successful_sessions": int(payload.get("successful_sessions", 0) or 0),
        "candidates": shown,
    }


def normalize_promoted_skill_name(name: str) -> str:
    """Normalize a promoted skill name into a stable directory slug."""
    return _slugify(name) or "workflow-candidate"


def render_promoted_skill_markdown(
    candidate: dict[str, Any],
    *,
    skill_name: str,
    profile_name: str,
    artifact_path: str = "",
) -> str:
    """Render a starter SKILL.md from a reviewed candidate."""
    normalized_name = normalize_promoted_skill_name(skill_name)
    title = _title_from_skill_name(normalized_name)
    triggers = _promotion_triggers(candidate)
    profiles = _promotion_profiles(profile_name)
    suggested_skill = (
        candidate.get("suggested_skill")
        if isinstance(candidate.get("suggested_skill"), dict)
        else {}
    )
    source_sessions = candidate.get("source_sessions")
    prompt_examples = candidate.get("prompt_examples")
    touched_paths = candidate.get("touched_path_hints")
    tool_sequence = candidate.get("tool_sequence")

    lines = [
        "---",
        f"name: {normalized_name}",
    ]
    if triggers:
        lines.append("triggers:")
        for trigger in triggers:
            lines.append(f"  - {trigger}")
    if profiles:
        lines.append("profiles:")
        for profile in profiles:
            lines.append(f"  - {profile}")
    lines.extend(
        [
            "requires:",
            "  - afs",
            "---",
            "",
            f"# {title}",
            "",
            "Generated from mined AFS session traces. Review and tighten this draft before broad reuse.",
            "",
            "## Provenance",
            "",
            f"- Candidate: {candidate.get('id', normalized_name)}",
            f"- Confidence: {candidate.get('confidence', 0.0)}",
            f"- Occurrences: {candidate.get('occurrences', 0)}",
        ]
    )
    if artifact_path:
        lines.append(f"- Source artifact: {artifact_path}")
    if isinstance(source_sessions, list) and source_sessions:
        lines.append(f"- Source sessions: {', '.join(source_sessions)}")

    lines.extend(
        [
            "",
            "## Observed Pattern",
            "",
            f"- Tool sequence: {' -> '.join(tool_sequence) if isinstance(tool_sequence, list) else '-'}",
            f"- Clients: {', '.join(candidate.get('clients', [])) if isinstance(candidate.get('clients'), list) else '-'}",
            f"- Touched paths: {', '.join(touched_paths) if isinstance(touched_paths, list) and touched_paths else '-'}",
            "",
        ]
    )

    notes = suggested_skill.get("notes") if isinstance(suggested_skill, dict) else []
    if isinstance(notes, list) and notes:
        lines.extend(
            [
                "## Draft Notes",
                "",
            ]
        )
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    if isinstance(prompt_examples, list) and prompt_examples:
        lines.extend(
            [
                "## Prompt Examples",
                "",
            ]
        )
        for prompt in prompt_examples:
            lines.append(f"- {prompt}")
        lines.append("")

    lines.extend(
        [
            "## Next Pass",
            "",
            "- Replace the observed pattern summary with concrete steps, verification, and failure handling.",
            "- Trim triggers to the smallest reliable set before treating this as a stable reusable skill.",
        ]
    )

    return "\n".join(lines).rstrip()


def write_promoted_skill(
    root: Path | str,
    *,
    skill_name: str,
    content: str,
    force: bool = False,
) -> dict[str, Any]:
    """Write a promoted skill into a skill root."""
    root_path = Path(root).expanduser().resolve()
    normalized_name = normalize_promoted_skill_name(skill_name)
    skill_dir = root_path / normalized_name
    skill_path = skill_dir / "SKILL.md"
    existed = skill_path.exists()
    if existed and not force:
        raise FileExistsError(
            f"Skill already exists at {skill_path}. Use --force to overwrite."
        )
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return {
        "root": str(root_path),
        "skill_dir": str(skill_dir),
        "skill_path": str(skill_path),
        "skill_name": normalized_name,
        "existed": existed,
        "overwritten": existed and force,
    }


def _summarize_session_trace(replay: dict[str, Any]) -> dict[str, Any] | None:
    events = replay.get("events")
    if not isinstance(events, list) or not events:
        return None

    tool_sequence: list[str] = []
    session_sequence: list[str] = []
    prompt_examples: list[str] = []
    clients: set[str] = set()
    touched_paths: list[str] = []
    mount_types: Counter[str] = Counter()
    workflow_names: set[str] = set()
    tool_profiles: set[str] = set()
    matched_skills: list[str] = []
    completion_signals = 0
    failure_signals = 0

    for event in events:
        if not isinstance(event, dict):
            continue
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        event_type = str(event.get("type", "")).strip()
        op = str(event.get("op", "")).strip()

        if event_type == "session":
            client = str(metadata.get("client", "")).strip()
            if client:
                clients.add(client)
            workflow = str(metadata.get("workflow", "")).strip()
            if workflow:
                workflow_names.add(workflow)
            tool_profile = str(metadata.get("tool_profile", "")).strip()
            if tool_profile:
                tool_profiles.add(tool_profile)
            workflow_steps = metadata.get("workflow_steps")
            if isinstance(workflow_steps, list):
                for step in workflow_steps:
                    step_name = str(step).strip()
                    if step_name:
                        session_sequence.append(step_name)
            skill_list = metadata.get("matched_skills")
            if isinstance(skill_list, list):
                for item in skill_list:
                    skill_name = str(item).strip()
                    if skill_name:
                        matched_skills.append(skill_name)
            if op == "user_prompt_submit":
                prompt_preview = str(metadata.get("prompt_preview", "")).strip()
                if prompt_preview:
                    prompt_examples.append(prompt_preview)
            elif op in {"turn_completed", "task_completed"}:
                completion_signals += 1
            elif op in {"turn_failed", "task_failed"}:
                failure_signals += 1
            elif op == "session_end":
                exit_code = metadata.get("exit_code")
                if isinstance(exit_code, int) and exit_code != 0:
                    failure_signals += 1
                elif prompt_examples or session_sequence:
                    completion_signals += 1
            if op:
                session_sequence.append(op)
        elif event_type == "mcp_tool":
            tool_name = str(metadata.get("tool_name", "")).strip()
            if tool_name:
                tool_sequence.append(tool_name)
            if metadata.get("ok") is False or metadata.get("error"):
                failure_signals += 1
        elif event_type in {"fs", "context"}:
            relative = str(
                metadata.get("relative_path") or metadata.get("alias") or ""
            ).strip()
            if relative:
                touched_paths.append(relative)
            mount_type = str(metadata.get("mount_type", "")).strip()
            if mount_type:
                mount_types[mount_type] += 1

    compact_tools = _compact_sequence(tool_sequence, limit=_MAX_TOOL_STEPS)
    compact_session = _compact_sequence(session_sequence, limit=_MAX_TOOL_STEPS)
    fallback_signature: list[str] = []
    for workflow in sorted(workflow_names):
        fallback_signature.append(f"workflow:{workflow}")
    for tool_profile in sorted(tool_profiles):
        fallback_signature.append(f"profile:{tool_profile}")
    for skill_name in _unique_strings(matched_skills, limit=2):
        fallback_signature.append(f"skill:{skill_name}")
    fallback_signature.extend(compact_session)
    compact_fallback = _compact_sequence(fallback_signature, limit=_MAX_TOOL_STEPS)

    signature = compact_tools or compact_fallback
    if not signature:
        return None

    return {
        "session_id": str(replay.get("session_id", "")).strip(),
        "started_at": str(replay.get("started_at", "")).strip(),
        "ended_at": str(replay.get("ended_at", "")).strip(),
        "event_count": int(replay.get("count", 0) or 0),
        "successful": completion_signals > 0 and failure_signals == 0,
        "completion_signals": completion_signals,
        "failure_signals": failure_signals,
        "tool_signature": signature,
        "prompt_examples": _unique_strings(prompt_examples, limit=_MAX_PROMPT_EXAMPLES),
        "clients": sorted(clients),
        "touched_paths": touched_paths,
        "mount_types": dict(sorted(mount_types.items())),
    }


def _build_candidate(
    signature: tuple[str, ...],
    cluster: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_examples: list[str] = []
    source_sessions: list[str] = []
    client_counter: Counter[str] = Counter()
    path_counter: Counter[str] = Counter()
    completion_total = 0

    for summary in cluster:
        prompt_examples.extend(summary.get("prompt_examples", []))
        source_sessions.append(str(summary.get("session_id", "")).strip())
        client_counter.update(summary.get("clients", []))
        completion_total += int(summary.get("completion_signals", 0) or 0)
        for raw_path in summary.get("touched_paths", []):
            hint = _path_hint(str(raw_path))
            if hint:
                path_counter[hint] += 1

    occurrences = len(cluster)
    common_terms = _common_prompt_terms(prompt_examples, cluster_size=occurrences)
    name = _suggest_candidate_name(signature, common_terms)
    confidence = _candidate_confidence(occurrences=occurrences, steps=len(signature))
    top_paths = [hint for hint, _count in path_counter.most_common(_MAX_PATH_HINTS)]

    return {
        "id": name,
        "name": name,
        "occurrences": occurrences,
        "confidence": confidence,
        "tool_sequence": list(signature),
        "trigger_terms": common_terms,
        "clients": [name for name, _count in client_counter.most_common()],
        "prompt_examples": _unique_strings(prompt_examples, limit=_MAX_PROMPT_EXAMPLES),
        "touched_path_hints": top_paths,
        "source_sessions": [session_id for session_id in source_sessions if session_id],
        "completion_signals": completion_total,
        "suggested_skill": {
            "name": name,
            "triggers": common_terms,
            "notes": [
                f"Observed in {occurrences} successful session traces.",
                f"Typical tool sequence: {' -> '.join(signature)}",
            ],
        },
    }


def _candidate_confidence(*, occurrences: int, steps: int) -> float:
    score = 0.4
    score += min(max(occurrences - 2, 0) * 0.12, 0.36)
    score += min(max(steps - 2, 0) * 0.05, 0.2)
    return round(min(score, 0.98), 2)


def _common_prompt_terms(
    prompts: list[str],
    *,
    cluster_size: int,
) -> list[str]:
    counter: Counter[str] = Counter()
    for prompt in prompts:
        counter.update(set(_tokenize_prompt(prompt)))

    threshold = max(1, min(cluster_size, 2))
    common = [
        token
        for token, count in counter.most_common()
        if count >= threshold
    ]
    return common[:_MAX_TRIGGER_TERMS]


def _tokenize_prompt(prompt: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[a-z][a-z0-9_-]{2,}", prompt.lower()):
        if token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _suggest_candidate_name(
    signature: tuple[str, ...],
    trigger_terms: list[str],
) -> str:
    parts: list[str] = []
    if trigger_terms:
        parts.extend(trigger_terms[:2])
    for tool_name in signature[:3]:
        parts.append(tool_name.replace(".", "-"))
    slug_parts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = _slugify(part)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        slug_parts.append(normalized)
    slug = "-".join(slug_parts)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return f"workflow-{slug or 'candidate'}"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _title_from_skill_name(value: str) -> str:
    words = [part for part in value.replace("_", "-").split("-") if part]
    if not words:
        return "Workflow Candidate"
    return " ".join(word.capitalize() for word in words)


def _promotion_profiles(profile_name: str) -> list[str]:
    normalized = str(profile_name or "").strip().lower()
    if normalized in {"", "default", "general"}:
        return ["general"]
    return [normalized]


def _promotion_triggers(candidate: dict[str, Any]) -> list[str]:
    raw = candidate.get("trigger_terms")
    if isinstance(raw, list):
        triggers = [
            str(item).strip()
            for item in raw
            if str(item).strip()
        ]
        if triggers:
            return triggers

    fallback: list[str] = []
    tool_sequence = candidate.get("tool_sequence")
    if isinstance(tool_sequence, list):
        for item in tool_sequence:
            token = str(item).strip()
            if ":" in token:
                token = token.split(":", 1)[1]
            elif "." in token:
                token = token.rsplit(".", 1)[-1]
            token = token.replace("_", "-")
            normalized = _slugify(token)
            if normalized and normalized not in fallback:
                fallback.append(normalized)
            if len(fallback) >= _MAX_TRIGGER_TERMS:
                break
    return fallback


def _path_hint(value: str) -> str:
    path = Path(value)
    if path.suffix:
        return f"*{path.suffix.lower()}"
    parts = [part for part in path.parts if part not in (".", "")]
    if parts:
        return f"{parts[0]}/*"
    return ""


def _compact_sequence(values: list[str], *, limit: int) -> list[str]:
    compact: list[str] = []
    previous = ""
    for value in values:
        item = str(value).strip()
        if not item or item == previous:
            continue
        compact.append(item)
        previous = item
        if len(compact) >= limit:
            break
    return compact


def _unique_strings(values: list[str], *, limit: int) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def _parse_timestamp(raw: str) -> datetime | None:
    if not raw:
        return None
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Skill Candidates",
        "",
        f"- Generated: {payload.get('generated_at', '')}",
        f"- Context: {payload.get('context_path', '')}",
        f"- Sessions analyzed: {payload.get('sessions_analyzed', 0)}",
        f"- Successful sessions: {payload.get('successful_sessions', 0)}",
        f"- Candidate count: {payload.get('candidate_count', 0)}",
        "",
    ]

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        lines.append("No repeated successful session traces met the mining threshold.")
        return "\n".join(lines)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        lines.extend(
            [
                f"## {candidate.get('name', 'workflow-candidate')}",
                "",
                f"- Confidence: {candidate.get('confidence', 0.0)}",
                f"- Occurrences: {candidate.get('occurrences', 0)}",
                f"- Tool sequence: {' -> '.join(candidate.get('tool_sequence', []))}",
                f"- Trigger terms: {', '.join(candidate.get('trigger_terms', [])) or '-'}",
                f"- Clients: {', '.join(candidate.get('clients', [])) or '-'}",
                f"- Touched paths: {', '.join(candidate.get('touched_path_hints', [])) or '-'}",
                f"- Source sessions: {', '.join(candidate.get('source_sessions', [])) or '-'}",
                "",
            ]
        )
        prompt_examples = candidate.get("prompt_examples")
        if isinstance(prompt_examples, list) and prompt_examples:
            lines.append("Prompt examples:")
            for prompt in prompt_examples:
                lines.append(f"- {prompt}")
            lines.append("")

    return "\n".join(lines).rstrip()


def render_skill_candidate_review(payload: dict[str, Any]) -> str:
    """Render a compact human-readable review summary."""
    lines = [
        f"artifact: {payload.get('artifact_path', '') or '(none)'}",
    ]
    generated_at = str(payload.get("generated_at", "")).strip()
    if generated_at:
        lines.append(f"generated_at: {generated_at}")
    lines.append(
        f"candidates: {payload.get('candidate_count', 0)}/{payload.get('total_candidate_count', 0)} shown"
    )
    status_counts = payload.get("status_counts")
    if isinstance(status_counts, dict) and status_counts:
        lines.append(
            "status_counts: "
            + ", ".join(f"{name}={count}" for name, count in status_counts.items())
        )
    sessions_analyzed = int(payload.get("sessions_analyzed", 0) or 0)
    successful_sessions = int(payload.get("successful_sessions", 0) or 0)
    if sessions_analyzed or successful_sessions:
        lines.append(
            f"sessions: analyzed={sessions_analyzed} successful={successful_sessions}"
        )

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        lines.append("(no candidates)")
        return "\n".join(lines)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        lines.extend(
            [
                "",
                f"{candidate.get('name', 'workflow-candidate')}",
                f"  status={candidate.get('status', 'pending')} confidence={candidate.get('confidence', 0.0)} occurrences={candidate.get('occurrences', 0)}",
                f"  tool_sequence={' -> '.join(candidate.get('tool_sequence', [])) or '-'}",
                f"  trigger_terms={', '.join(candidate.get('trigger_terms', [])) or '-'}",
                f"  clients={', '.join(candidate.get('clients', [])) or '-'}",
                f"  source_sessions={', '.join(candidate.get('source_sessions', [])) or '-'}",
            ]
        )
        review_state = candidate.get("review_state")
        if isinstance(review_state, dict) and review_state:
            skill_path = str(review_state.get("skill_path", "")).strip()
            if skill_path:
                lines.append(f"  promoted_to={skill_path}")
        prompt_examples = candidate.get("prompt_examples")
        if isinstance(prompt_examples, list) and prompt_examples:
            lines.append(f"  prompts={ ' | '.join(prompt_examples) }")

    return "\n".join(lines)
