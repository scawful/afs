"""Native AFS work-assistant state for people, reviews, and approvals."""

from __future__ import annotations

import json
import os
import re
import shlex
import sqlite3
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context_paths import resolve_mount_root
from .models import MountType
from .schema import AFSConfig

DEFAULT_DB_FILENAME = "work_assistant.sqlite3"
COMMUNICATION_PREFLIGHT_STEPS = (
    "Load opt-in personal work context when a mode is provided.",
    "Inspect stored work communication samples before drafting.",
    "Inspect relevant scratchpad/history/context hits for the concrete topic.",
    "Draft locally and state missing style evidence instead of inventing preferences.",
    "Create an AFS work approval before any external post, send, submit, or edit.",
)
EXTERNAL_WRITE_ACTIONS = frozenset(
    {
        "add_doc_comment",
        "assign_task",
        "assign_ticket",
        "change_due_date",
        "change_sheet_formula",
        "change_ticket_priority",
        "change_ticket_status",
        "create_task",
        "edit_doc",
        "edit_sheet",
        "post_ticket_comment",
        "post_code_review_comment",
        "post_doc_comment",
        "post_pr_comment",
        "post_pull_request_review",
        "publish_comment",
        "reply_to_comment",
        "send_email",
        "send_message",
        "share_doc",
        "submit_review",
        "update_sheet_cell",
    }
)
REVIEW_RELATIONSHIP_TYPES = frozenset(
    {"owner", "dri", "maintainer", "reviewer", "approver", "stakeholder"}
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    return [value]


def _str_list(value: Any) -> list[str]:
    items: list[str] = []
    for item in _as_list(value):
        text = str(item).strip()
        if text and text not in items:
            items.append(text)
    return items


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _personal_context_summary(personal_context: Any | None) -> dict[str, Any]:
    if personal_context is None:
        return {
            "loaded": False,
            "mode": "",
            "work_context": False,
            "tone": "",
            "bias_warning": None,
            "style_instructions": [],
            "communication_sources": [],
            "posting_policy": "",
            "files": [],
            "missing": [],
        }

    files: list[dict[str, str]] = []
    for item in _as_list(getattr(personal_context, "files", [])):
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        rel, content = item
        files.append({"path": str(rel), "excerpt": _truncate_text(content, 800)})

    return {
        "loaded": True,
        "mode": str(getattr(personal_context, "mode", "") or ""),
        "work_context": bool(getattr(personal_context, "work_context", False)),
        "tone": str(getattr(personal_context, "tone", "") or ""),
        "bias_warning": getattr(personal_context, "bias_warning", None),
        "style_instructions": _str_list(getattr(personal_context, "style_instructions", [])),
        "communication_sources": _str_list(getattr(personal_context, "communication_sources", [])),
        "posting_policy": str(getattr(personal_context, "posting_policy", "") or ""),
        "files": files,
        "missing": _str_list(getattr(personal_context, "missing", [])),
    }


def _merge_lists(*values: Iterable[Any]) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for value in values:
        for item in value:
            key = _json_dumps(item) if isinstance(item, dict) else str(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _merge_dicts(*values: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        for key, item in value.items():
            if item not in (None, "", [], {}):
                merged[key] = item
    return merged


def _stable_id(prefix: str, *parts: Any) -> str:
    cleaned = [
        str(part).strip().lower()
        for part in parts
        if str(part).strip()
    ]
    if not cleaned:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"
    raw = "|".join(cleaned)
    safe = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    if 4 <= len(safe) <= 64:
        return f"{prefix}_{safe}"
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, raw).hex[:16]}"


def _normalize_handle_key(key: str) -> str:
    lowered = key.strip().lower()
    aliases = {
        "mail": "email",
        "e-mail": "email",
        "github_login": "github",
        "chat_id": "chat",
    }
    return aliases.get(lowered, lowered)


def _person_id_from_record(record: dict[str, Any]) -> str:
    explicit = str(record.get("person_id") or record.get("id") or "").strip()
    if explicit:
        return explicit
    handles = record.get("handles") if isinstance(record.get("handles"), dict) else {}
    if "email" in handles:
        return _stable_id("person", "email", handles["email"])
    for key in sorted(handles):
        if handles[key]:
            return _stable_id("person", key, handles[key])
    return _stable_id("person", record.get("display_name") or record.get("name") or "unknown")


def _normalize_person(raw: Any, *, role: str | None = None, target_type: str | None = None) -> dict[str, Any]:
    if isinstance(raw, dict):
        data = dict(raw)
    else:
        text = str(raw).strip()
        data = {"display_name": text}
        if "@" in text:
            data["handles"] = {"email": text}

    handles = data.get("handles") if isinstance(data.get("handles"), dict) else {}
    normalized_handles = {
        _normalize_handle_key(str(key)): str(value).strip()
        for key, value in handles.items()
        if str(value).strip()
    }
    for key in ("email", "github", "chat", "ticket_user_id", "document_account"):
        if key in data and str(data[key]).strip():
            normalized_handles[_normalize_handle_key(key)] = str(data[key]).strip()

    display_name = str(
        data.get("display_name")
        or data.get("name")
        or normalized_handles.get("email")
        or "Unknown person"
    ).strip()

    roles = _str_list(data.get("roles") or data.get("role"))
    if role and role not in roles:
        roles.append(role)

    permissions = _str_list(data.get("permissions") or data.get("permission_notes"))
    if role in {"owner", "approver"} and target_type:
        permissions.append(f"can approve {target_type}")
    if role in {"reviewer", "maintainer", "owner"} and target_type:
        permissions.append(f"can review {target_type}")

    return {
        "person_id": _person_id_from_record(
            {**data, "display_name": display_name, "handles": normalized_handles}
        ),
        "display_name": display_name,
        "handles": normalized_handles,
        "organization": str(data.get("organization") or "").strip(),
        "team": str(data.get("team") or "").strip(),
        "roles": roles,
        "permissions": _str_list(permissions),
        "provenance": _as_list(data.get("provenance")),
        "confidence": float(data.get("confidence") or 0.5),
    }


class WorkAssistantStore:
    """SQLite-backed work-assistant state under a context's global mount."""

    def __init__(
        self,
        context_root: Path,
        *,
        config: AFSConfig | None = None,
        db_path: Path | None = None,
    ) -> None:
        self.context_root = context_root.expanduser().resolve()
        if db_path is None:
            global_root = resolve_mount_root(self.context_root, MountType.GLOBAL, config=config)
            db_path = global_root / DEFAULT_DB_FILENAME
        self.db_path = db_path.expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS people (
                    person_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    handles_json TEXT NOT NULL,
                    organization TEXT NOT NULL DEFAULT '',
                    team TEXT NOT NULL DEFAULT '',
                    roles_json TEXT NOT NULL,
                    permissions_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS relationships (
                    relationship_id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    scope_type TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    allowed_review_targets_json TEXT NOT NULL,
                    permission_class TEXT NOT NULL DEFAULT '',
                    provenance_json TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0,
                    expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES people(person_id)
                );

                CREATE TABLE IF NOT EXISTS review_routes (
                    route_id TEXT PRIMARY KEY,
                    scope_type TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0,
                    provenance_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES people(person_id)
                );

                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    target_system TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    preview_json TEXT NOT NULL,
                    affected_people_json TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    permission_required TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    approved_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    result_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS activity (
                    activity_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_id TEXT,
                    activity_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    target_system TEXT NOT NULL DEFAULT '',
                    target_id TEXT NOT NULL DEFAULT '',
                    actor TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS communication_samples (
                    sample_id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL DEFAULT '',
                    source_system TEXT NOT NULL DEFAULT '',
                    source_id TEXT NOT NULL DEFAULT '',
                    channel TEXT NOT NULL DEFAULT '',
                    purpose TEXT NOT NULL DEFAULT '',
                    text_excerpt TEXT NOT NULL,
                    style_notes_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_relationships_scope
                    ON relationships(scope_type, scope_id, relationship_type);
                CREATE INDEX IF NOT EXISTS idx_review_routes_target
                    ON review_routes(scope_type, scope_id, target_type);
                CREATE INDEX IF NOT EXISTS idx_approvals_status
                    ON approvals(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_activity_timestamp
                    ON activity(timestamp);
                CREATE INDEX IF NOT EXISTS idx_communication_samples_purpose
                    ON communication_samples(purpose, updated_at);
                CREATE INDEX IF NOT EXISTS idx_communication_samples_person
                    ON communication_samples(person_id, updated_at);
                """
            )

    def upsert_person(self, person: dict[str, Any]) -> str:
        normalized = _normalize_person(person)
        person_id = normalized["person_id"]
        now = _now()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM people WHERE person_id = ?", (person_id,)
            ).fetchone()
            if existing:
                handles = _merge_dicts(
                    _json_loads(existing["handles_json"], {}),
                    normalized["handles"],
                )
                roles = _merge_lists(
                    _json_loads(existing["roles_json"], []),
                    normalized["roles"],
                )
                permissions = _merge_lists(
                    _json_loads(existing["permissions_json"], []),
                    normalized["permissions"],
                )
                provenance = _merge_lists(
                    _json_loads(existing["provenance_json"], []),
                    normalized["provenance"],
                )
                connection.execute(
                    """
                    UPDATE people
                    SET display_name = ?, handles_json = ?, organization = ?, team = ?,
                        roles_json = ?, permissions_json = ?, provenance_json = ?,
                        confidence = ?, updated_at = ?
                    WHERE person_id = ?
                    """,
                    (
                        normalized["display_name"] or existing["display_name"],
                        _json_dumps(handles),
                        normalized["organization"] or existing["organization"],
                        normalized["team"] or existing["team"],
                        _json_dumps(roles),
                        _json_dumps(permissions),
                        _json_dumps(provenance),
                        max(float(existing["confidence"] or 0), normalized["confidence"]),
                        now,
                        person_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO people (
                        person_id, display_name, handles_json, organization, team,
                        roles_json, permissions_json, provenance_json,
                        confidence, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        person_id,
                        normalized["display_name"],
                        _json_dumps(normalized["handles"]),
                        normalized["organization"],
                        normalized["team"],
                        _json_dumps(normalized["roles"]),
                        _json_dumps(normalized["permissions"]),
                        _json_dumps(normalized["provenance"]),
                        normalized["confidence"],
                        now,
                        now,
                    ),
                )
        return person_id

    def list_people(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM people ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._person_row_to_dict(row) for row in rows]

    def upsert_relationship(
        self,
        *,
        person_id: str,
        scope_type: str,
        scope_id: str,
        relationship_type: str,
        allowed_review_targets: Iterable[str] | None = None,
        permission_class: str = "",
        provenance: Iterable[Any] | None = None,
        confidence: float = 0.5,
        expires_at: str | None = None,
    ) -> str:
        relationship_id = _stable_id(
            "rel", person_id, scope_type, scope_id, relationship_type
        )
        targets = _str_list(list(allowed_review_targets or []))
        now = _now()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM relationships WHERE relationship_id = ?",
                (relationship_id,),
            ).fetchone()
            if existing:
                merged_targets = _merge_lists(
                    _json_loads(existing["allowed_review_targets_json"], []),
                    targets,
                )
                merged_provenance = _merge_lists(
                    _json_loads(existing["provenance_json"], []),
                    list(provenance or []),
                )
                connection.execute(
                    """
                    UPDATE relationships
                    SET allowed_review_targets_json = ?, permission_class = ?,
                        provenance_json = ?, confidence = ?, expires_at = ?,
                        updated_at = ?
                    WHERE relationship_id = ?
                    """,
                    (
                        _json_dumps(merged_targets),
                        permission_class or existing["permission_class"],
                        _json_dumps(merged_provenance),
                        max(float(existing["confidence"] or 0), confidence),
                        expires_at or existing["expires_at"],
                        now,
                        relationship_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO relationships (
                        relationship_id, person_id, scope_type, scope_id,
                        relationship_type, allowed_review_targets_json,
                        permission_class, provenance_json, confidence,
                        expires_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        relationship_id,
                        person_id,
                        scope_type,
                        scope_id,
                        relationship_type,
                        _json_dumps(targets),
                        permission_class,
                        _json_dumps(list(provenance or [])),
                        confidence,
                        expires_at,
                        now,
                        now,
                    ),
                )
        return relationship_id

    def list_relationships(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT relationships.*, people.display_name
                FROM relationships
                LEFT JOIN people ON people.person_id = relationships.person_id
                ORDER BY relationships.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._relationship_row_to_dict(row) for row in rows]

    def upsert_review_route(
        self,
        *,
        person_id: str,
        scope_type: str,
        scope_id: str,
        target_type: str,
        reason: str = "",
        confidence: float = 0.5,
        provenance: Iterable[Any] | None = None,
    ) -> str:
        route_id = _stable_id("route", person_id, scope_type, scope_id, target_type)
        now = _now()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM review_routes WHERE route_id = ?", (route_id,)
            ).fetchone()
            if existing:
                merged_provenance = _merge_lists(
                    _json_loads(existing["provenance_json"], []),
                    list(provenance or []),
                )
                connection.execute(
                    """
                    UPDATE review_routes
                    SET reason = ?, confidence = ?, provenance_json = ?, updated_at = ?
                    WHERE route_id = ?
                    """,
                    (
                        reason or existing["reason"],
                        max(float(existing["confidence"] or 0), confidence),
                        _json_dumps(merged_provenance),
                        now,
                        route_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO review_routes (
                        route_id, scope_type, scope_id, target_type, person_id,
                        reason, confidence, provenance_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        route_id,
                        scope_type,
                        scope_id,
                        target_type,
                        person_id,
                        reason,
                        confidence,
                        _json_dumps(list(provenance or [])),
                        now,
                        now,
                    ),
                )
        return route_id

    def suggest_reviewers(
        self,
        *,
        target_type: str,
        scope_type: str | None = None,
        scope_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        reviewers = self._suggest_reviewers_from_routes(
            target_type=target_type,
            scope_type=scope_type,
            scope_id=scope_id,
            limit=limit,
        )
        if reviewers:
            return reviewers
        return self._suggest_reviewers_from_relationships(
            target_type=target_type,
            scope_type=scope_type,
            scope_id=scope_id,
            limit=limit,
        )

    def create_approval(
        self,
        *,
        target_system: str,
        target_id: str,
        action: str,
        summary: str,
        preview: Any | None = None,
        affected_people: Iterable[Any] | None = None,
        risk_level: str = "medium",
        permission_required: str = "",
        requested_by: str = "agent",
        expires_at: str | None = None,
        dedupe_key: str | None = None,
    ) -> str:
        approval_id = dedupe_key or f"approval_{uuid.uuid4().hex[:12]}"
        now = _now()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT approval_id FROM approvals WHERE approval_id = ?", (approval_id,)
            ).fetchone()
            if existing:
                return str(existing["approval_id"])
            connection.execute(
                """
                INSERT INTO approvals (
                    approval_id, status, target_system, target_id, action, summary,
                    preview_json, affected_people_json, risk_level, permission_required,
                    requested_by, approved_by, created_at, updated_at, expires_at, result_json
                ) VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    target_system,
                    target_id,
                    action,
                    summary,
                    _json_dumps(preview or {}),
                    _json_dumps(list(affected_people or [])),
                    risk_level,
                    permission_required,
                    requested_by,
                    now,
                    now,
                    expires_at,
                    _json_dumps({}),
                ),
            )
        return approval_id

    def list_approvals(
        self,
        *,
        status: str | None = "pending",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM approvals"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._approval_row_to_dict(row) for row in rows]

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        return self._approval_row_to_dict(row) if row else None

    def approve(self, approval_id: str, *, approved_by: str = "human") -> bool:
        return self._set_approval_status(approval_id, "approved", approved_by=approved_by)

    def reject(self, approval_id: str, *, rejected_by: str = "human") -> bool:
        return self._set_approval_status(approval_id, "rejected", approved_by=rejected_by)

    def record_approval_result(
        self,
        approval_id: str,
        *,
        result: dict[str, Any],
        status: str | None = None,
    ) -> bool:
        now = _now()
        if status:
            statement = """
                UPDATE approvals
                SET result_json = ?, status = ?, updated_at = ?
                WHERE approval_id = ?
            """
            params: tuple[Any, ...] = (_json_dumps(result), status, now, approval_id)
        else:
            statement = """
                UPDATE approvals
                SET result_json = ?, updated_at = ?
                WHERE approval_id = ?
            """
            params = (_json_dumps(result), now, approval_id)
        with self._connect() as connection:
            cursor = connection.execute(statement, params)
            return cursor.rowcount > 0

    def record_activity(
        self,
        *,
        activity_type: str,
        summary: str,
        event_id: str | None = None,
        target_system: str = "",
        target_id: str = "",
        actor: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        activity_id = f"activity_{uuid.uuid4().hex[:12]}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO activity (
                    activity_id, timestamp, event_id, activity_type, summary,
                    target_system, target_id, actor, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    activity_id,
                    _now(),
                    event_id,
                    activity_type,
                    summary,
                    target_system,
                    target_id,
                    actor,
                    _json_dumps(metadata or {}),
                ),
            )
        return activity_id

    def list_activity(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM activity ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._activity_row_to_dict(row) for row in rows]

    def record_communication_sample(
        self,
        *,
        text: str,
        person_id: str = "",
        source_system: str = "",
        source_id: str = "",
        channel: str = "",
        purpose: str = "",
        style_notes: Iterable[Any] | None = None,
        provenance: Iterable[Any] | None = None,
        confidence: float = 0.5,
        dedupe_key: str | None = None,
    ) -> str:
        """Store a work communication sample for later style grounding."""
        text_excerpt = str(text or "").strip()
        if not text_excerpt:
            return ""
        if len(text_excerpt) > 1200:
            text_excerpt = text_excerpt[:1197].rstrip() + "..."

        sample_id = dedupe_key or _stable_id(
            "comm",
            person_id,
            source_system,
            source_id,
            channel,
            purpose,
            text_excerpt[:120],
        )
        now = _now()
        notes = _str_list(style_notes or [])
        sample_provenance = _as_list(provenance)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM communication_samples WHERE sample_id = ?",
                (sample_id,),
            ).fetchone()
            if existing:
                merged_notes = _merge_lists(
                    _json_loads(existing["style_notes_json"], []),
                    notes,
                )
                merged_provenance = _merge_lists(
                    _json_loads(existing["provenance_json"], []),
                    sample_provenance,
                )
                connection.execute(
                    """
                    UPDATE communication_samples
                    SET person_id = ?, source_system = ?, source_id = ?, channel = ?,
                        purpose = ?, text_excerpt = ?, style_notes_json = ?,
                        provenance_json = ?, confidence = ?, updated_at = ?
                    WHERE sample_id = ?
                    """,
                    (
                        person_id or existing["person_id"],
                        source_system or existing["source_system"],
                        source_id or existing["source_id"],
                        channel or existing["channel"],
                        purpose or existing["purpose"],
                        text_excerpt or existing["text_excerpt"],
                        _json_dumps(merged_notes),
                        _json_dumps(merged_provenance),
                        max(float(existing["confidence"] or 0), confidence),
                        now,
                        sample_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO communication_samples (
                        sample_id, person_id, source_system, source_id, channel,
                        purpose, text_excerpt, style_notes_json, provenance_json,
                        confidence, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample_id,
                        person_id,
                        source_system,
                        source_id,
                        channel,
                        purpose,
                        text_excerpt,
                        _json_dumps(notes),
                        _json_dumps(sample_provenance),
                        confidence,
                        now,
                        now,
                    ),
                )
        return sample_id

    def list_communication_samples(
        self,
        *,
        person_id: str | None = None,
        purpose: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if person_id:
            clauses.append("communication_samples.person_id = ?")
            params.append(person_id)
        if purpose:
            clauses.append("communication_samples.purpose = ?")
            params.append(purpose)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT communication_samples.*, people.display_name
                FROM communication_samples
                LEFT JOIN people ON people.person_id = communication_samples.person_id
                {where}
                ORDER BY communication_samples.updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._communication_sample_row_to_dict(row) for row in rows]

    def communication_style_summary(
        self,
        *,
        person_id: str | None = None,
        purpose: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Return compact work-writing guidance from stored communication samples."""
        samples = self.list_communication_samples(
            person_id=person_id,
            purpose=purpose,
            limit=limit,
        )
        purposes: dict[str, int] = {}
        channels: dict[str, int] = {}
        notes: list[str] = []
        for sample in samples:
            sample_purpose = str(sample.get("purpose") or "work_communication")
            purposes[sample_purpose] = purposes.get(sample_purpose, 0) + 1
            channel = str(sample.get("channel") or "").strip()
            if channel:
                channels[channel] = channels.get(channel, 0) + 1
            for note in _str_list(sample.get("style_notes")):
                if note not in notes:
                    notes.append(note)

        guidance: list[str] = []
        if samples:
            guidance.append(
                "Use these work communication samples as grounding before drafting "
                "documentation, design docs, technical requirements, or replies."
            )
            if notes:
                guidance.append(f"Observed style notes: {', '.join(notes[:8])}.")
            else:
                guidance.append("No explicit style notes were stored; infer conservatively from excerpts.")
        else:
            guidance.append(
                "No work communication samples are stored for this filter; state that "
                "style evidence is missing before imitating the user."
            )
        guidance.append(
            "Never post, send, submit, or edit an external work system on the user's "
            "behalf without explicit approval; draft locally or create an AFS work approval first."
        )

        return {
            "sample_count": len(samples),
            "person_id": person_id or "",
            "purpose": purpose or "",
            "purposes": purposes,
            "channels": channels,
            "style_notes": notes,
            "samples": samples,
            "guidance": guidance,
        }

    def communication_preflight(
        self,
        *,
        person_id: str | None = None,
        purpose: str | None = None,
        limit: int = 20,
        approval_limit: int = 10,
        personal_context: Any | None = None,
        context_path: Path | None = None,
    ) -> dict[str, Any]:
        """Return the mandatory work-writing preflight context and guardrails."""
        style = self.communication_style_summary(
            person_id=person_id,
            purpose=purpose,
            limit=limit,
        )
        personal = _personal_context_summary(personal_context)
        pending_approvals = self.list_approvals(status="pending", limit=approval_limit)

        personal_style_items = [
            *personal["style_instructions"],
            *[entry.get("excerpt", "") for entry in personal["files"] if entry.get("excerpt")],
        ]
        has_personal_style = bool(personal["tone"] or personal_style_items)
        missing_style_evidence = style["sample_count"] == 0 and not has_personal_style

        guidance = list(style["guidance"])
        if personal["loaded"]:
            guidance.append(
                f"Personal context mode '{personal['mode']}' was explicitly loaded; use it as opt-in work-style grounding."
            )
            if personal["tone"]:
                guidance.append(f"Personal context tone: {personal['tone']}.")
            if personal["style_instructions"]:
                guidance.append(
                    "Personal style instructions: "
                    + ", ".join(personal["style_instructions"][:8])
                    + "."
                )
            if personal["communication_sources"]:
                guidance.append(
                    "Required communication sources to inspect: "
                    + ", ".join(personal["communication_sources"][:8])
                    + "."
                )
            if personal["posting_policy"]:
                guidance.append(f"Personal posting policy: {personal['posting_policy']}")
        else:
            guidance.append(
                "No personal context mode was loaded; use --personal-mode only when the user opted into that context."
            )
        if missing_style_evidence:
            guidance.append(
                "Style evidence is missing; ask for samples or draft neutrally instead of imitating a voice."
            )

        commands: dict[str, str] = {}
        if context_path is not None:
            quoted_context = shlex.quote(str(context_path.expanduser().resolve()))
            commands = {
                "communication_preflight": f"afs work communication preflight --context-root {quoted_context}",
                "communication_guide": f"afs work communication guide --context-root {quoted_context}",
                "communication_list": f"afs work communication list --context-root {quoted_context}",
                "approval_request": (
                    "afs work approvals request --context-root "
                    f"{quoted_context} --target-system <system> --target-id <id> "
                    "--action <action> --summary <summary> --preview-json '<preview>'"
                ),
                "approvals_list": f"afs work approvals list --context-root {quoted_context}",
            }

        return {
            "context_path": str(context_path) if context_path is not None else "",
            "person_id": person_id or "",
            "purpose": purpose or "",
            "style": style,
            "personal_context": personal,
            "pending_approvals": pending_approvals,
            "pending_approval_count": len(pending_approvals),
            "missing_style_evidence": missing_style_evidence,
            "approval_guardrail": {
                "requires_explicit_approval": True,
                "ready_to_post": False,
                "policy": (
                    "Do not post, send, submit, or edit an external work system unless "
                    "the user explicitly approves the exact target, action, and preview."
                ),
            },
            "checklist": [
                {
                    "step": COMMUNICATION_PREFLIGHT_STEPS[0],
                    "status": "done" if personal["loaded"] else "not_loaded",
                    "detail": (
                        f"Loaded mode {personal['mode']}."
                        if personal["loaded"]
                        else "Personal context is opt-in and was not requested."
                    ),
                },
                {
                    "step": COMMUNICATION_PREFLIGHT_STEPS[1],
                    "status": "done" if style["sample_count"] else "missing",
                    "detail": f"{style['sample_count']} communication sample(s) available.",
                },
                {
                    "step": COMMUNICATION_PREFLIGHT_STEPS[2],
                    "status": "required",
                    "detail": "Run context.query/session pack against the concrete doc, ticket, PR, or comment thread.",
                },
                {
                    "step": COMMUNICATION_PREFLIGHT_STEPS[3],
                    "status": "required",
                    "detail": "Keep the draft local until the user reviews it.",
                },
                {
                    "step": COMMUNICATION_PREFLIGHT_STEPS[4],
                    "status": "required",
                    "detail": "Use work.approvals.request for the proposed external write; this preflight never executes it.",
                },
            ],
            "guidance": guidance,
            "commands": commands,
        }

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            counts = {
                name: int(
                    connection.execute(f"SELECT COUNT(*) AS count FROM {name}").fetchone()[
                        "count"
                    ]
                )
                for name in (
                    "people",
                    "relationships",
                    "review_routes",
                    "approvals",
                    "activity",
                    "communication_samples",
                )
            }
            pending = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM approvals WHERE status = 'pending'"
                ).fetchone()["count"]
            )
        counts["pending_approvals"] = pending
        counts["db_path"] = str(self.db_path)
        return counts

    def _set_approval_status(
        self,
        approval_id: str,
        status: str,
        *,
        approved_by: str,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE approvals
                SET status = ?, approved_by = ?, updated_at = ?
                WHERE approval_id = ? AND status = 'pending'
                """,
                (status, approved_by, _now(), approval_id),
            )
            return cursor.rowcount > 0

    def _suggest_reviewers_from_routes(
        self,
        *,
        target_type: str,
        scope_type: str | None,
        scope_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        clauses = ["review_routes.target_type = ?"]
        params: list[Any] = [target_type]
        if scope_type:
            clauses.append("review_routes.scope_type = ?")
            params.append(scope_type)
        if scope_id:
            clauses.append("review_routes.scope_id = ?")
            params.append(scope_id)
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT review_routes.*, people.display_name, people.handles_json
                FROM review_routes
                LEFT JOIN people ON people.person_id = review_routes.person_id
                WHERE {' AND '.join(clauses)}
                ORDER BY review_routes.confidence DESC, review_routes.updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._review_route_row_to_dict(row) for row in rows]

    def _suggest_reviewers_from_relationships(
        self,
        *,
        target_type: str,
        scope_type: str | None,
        scope_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        clauses = [
            "relationships.relationship_type IN ({})".format(
                ",".join("?" for _ in REVIEW_RELATIONSHIP_TYPES)
            )
        ]
        params: list[Any] = list(REVIEW_RELATIONSHIP_TYPES)
        if scope_type:
            clauses.append("relationships.scope_type = ?")
            params.append(scope_type)
        if scope_id:
            clauses.append("relationships.scope_id = ?")
            params.append(scope_id)
        params.append(limit * 3)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT relationships.*, people.display_name, people.handles_json
                FROM relationships
                LEFT JOIN people ON people.person_id = relationships.person_id
                WHERE {' AND '.join(clauses)}
                ORDER BY relationships.confidence DESC, relationships.updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        reviewers: list[dict[str, Any]] = []
        for row in rows:
            targets = _json_loads(row["allowed_review_targets_json"], [])
            if targets and target_type not in targets:
                continue
            reviewers.append(
                {
                    "person_id": row["person_id"],
                    "display_name": row["display_name"],
                    "handles": _json_loads(row["handles_json"], {}),
                    "scope_type": row["scope_type"],
                    "scope_id": row["scope_id"],
                    "target_type": target_type,
                    "reason": f"{row['relationship_type']} for {row['scope_type']}:{row['scope_id']}",
                    "confidence": row["confidence"],
                    "source": "relationship",
                }
            )
            if len(reviewers) >= limit:
                break
        return reviewers

    def _person_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "person_id": row["person_id"],
            "display_name": row["display_name"],
            "handles": _json_loads(row["handles_json"], {}),
            "organization": row["organization"],
            "team": row["team"],
            "roles": _json_loads(row["roles_json"], []),
            "permissions": _json_loads(row["permissions_json"], []),
            "provenance": _json_loads(row["provenance_json"], []),
            "confidence": row["confidence"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _relationship_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "relationship_id": row["relationship_id"],
            "person_id": row["person_id"],
            "display_name": row["display_name"],
            "scope_type": row["scope_type"],
            "scope_id": row["scope_id"],
            "relationship_type": row["relationship_type"],
            "allowed_review_targets": _json_loads(row["allowed_review_targets_json"], []),
            "permission_class": row["permission_class"],
            "provenance": _json_loads(row["provenance_json"], []),
            "confidence": row["confidence"],
            "expires_at": row["expires_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _review_route_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "route_id": row["route_id"],
            "scope_type": row["scope_type"],
            "scope_id": row["scope_id"],
            "target_type": row["target_type"],
            "person_id": row["person_id"],
            "display_name": row["display_name"],
            "handles": _json_loads(row["handles_json"], {}),
            "reason": row["reason"],
            "confidence": row["confidence"],
            "provenance": _json_loads(row["provenance_json"], []),
            "source": "review_route",
        }

    def _approval_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "approval_id": row["approval_id"],
            "status": row["status"],
            "target_system": row["target_system"],
            "target_id": row["target_id"],
            "action": row["action"],
            "summary": row["summary"],
            "preview": _json_loads(row["preview_json"], {}),
            "affected_people": _json_loads(row["affected_people_json"], []),
            "risk_level": row["risk_level"],
            "permission_required": row["permission_required"],
            "requested_by": row["requested_by"],
            "approved_by": row["approved_by"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "expires_at": row["expires_at"],
            "result": _json_loads(row["result_json"], {}),
        }

    def _activity_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "activity_id": row["activity_id"],
            "timestamp": row["timestamp"],
            "event_id": row["event_id"],
            "activity_type": row["activity_type"],
            "summary": row["summary"],
            "target_system": row["target_system"],
            "target_id": row["target_id"],
            "actor": row["actor"],
            "metadata": _json_loads(row["metadata_json"], {}),
        }

    def _communication_sample_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "sample_id": row["sample_id"],
            "person_id": row["person_id"],
            "display_name": row["display_name"] if "display_name" in row.keys() else "",
            "source_system": row["source_system"],
            "source_id": row["source_id"],
            "channel": row["channel"],
            "purpose": row["purpose"],
            "text_excerpt": row["text_excerpt"],
            "style_notes": _json_loads(row["style_notes_json"], []),
            "provenance": _json_loads(row["provenance_json"], []),
            "confidence": row["confidence"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


def enrich_logged_event(context_root: Path | None, event: dict[str, Any]) -> dict[str, int]:
    """Enrich native work-assistant state from a logged context/history event."""
    if context_root is None or os.getenv("AFS_WORK_ASSISTANT_ENRICH_DISABLED") == "1":
        return _empty_counts()

    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    if not _looks_work_relevant(event, metadata, payload):
        return _empty_counts()

    store = WorkAssistantStore(context_root)
    counts = _empty_counts()

    target_type = str(metadata.get("target_type") or payload.get("target_type") or "").strip()
    scope_type = str(metadata.get("scope_type") or metadata.get("target_system") or "project").strip()
    scope_id = str(
        metadata.get("scope_id")
        or metadata.get("project")
        or metadata.get("target_id")
        or metadata.get("path")
        or payload.get("scope_id")
        or "default"
    ).strip()

    people = _collect_people(metadata, payload, target_type=target_type or None)
    person_ids: dict[str, str] = {}
    for person in people:
        person_id = store.upsert_person(person)
        person_ids[person.get("source_key", person_id)] = person_id
        counts["people"] += 1

        for role in person.get("roles", []):
            if not role:
                continue
            store.upsert_relationship(
                person_id=person_id,
                scope_type=scope_type,
                scope_id=scope_id,
                relationship_type=role,
                allowed_review_targets=[target_type] if target_type else [],
                permission_class=_permission_class_for_role(role),
                provenance=[_event_provenance(event)],
                confidence=float(person.get("confidence") or 0.5),
            )
            counts["relationships"] += 1

            if role in REVIEW_RELATIONSHIP_TYPES and target_type:
                store.upsert_review_route(
                    person_id=person_id,
                    scope_type=scope_type,
                    scope_id=scope_id,
                    target_type=target_type,
                    reason=f"inferred {role} from logged context",
                    confidence=float(person.get("confidence") or 0.5),
                    provenance=[_event_provenance(event)],
                )
                counts["review_routes"] += 1

    known_person_ids = set(person_ids.values())

    for relationship in _as_list(metadata.get("relationships") or payload.get("relationships")):
        if not isinstance(relationship, dict):
            continue
        person = relationship.get("person")
        person_id = str(relationship.get("person_id") or "").strip()
        if not person_id and person:
            person_id = store.upsert_person(_normalize_person(person, target_type=target_type or None))
            counts["people"] += 1
        if not person_id:
            continue
        store.upsert_relationship(
            person_id=person_id,
            scope_type=str(relationship.get("scope_type") or scope_type),
            scope_id=str(relationship.get("scope_id") or scope_id),
            relationship_type=str(relationship.get("relationship_type") or relationship.get("role") or "collaborator"),
            allowed_review_targets=_str_list(
                relationship.get("allowed_review_targets") or ([target_type] if target_type else [])
            ),
            permission_class=str(relationship.get("permission_class") or ""),
            provenance=_as_list(relationship.get("provenance")) or [_event_provenance(event)],
            confidence=float(relationship.get("confidence") or 0.5),
            expires_at=relationship.get("expires_at"),
        )
        counts["relationships"] += 1

    for route in _as_list(metadata.get("review_routes") or payload.get("review_routes")):
        if not isinstance(route, dict):
            continue
        person_id = str(route.get("person_id") or "").strip()
        if not person_id and route.get("person"):
            person_id = store.upsert_person(_normalize_person(route["person"], target_type=target_type or None))
            counts["people"] += 1
        if not person_id:
            continue
        store.upsert_review_route(
            person_id=person_id,
            scope_type=str(route.get("scope_type") or scope_type),
            scope_id=str(route.get("scope_id") or scope_id),
            target_type=str(route.get("target_type") or target_type or "work"),
            reason=str(route.get("reason") or "declared review route"),
            confidence=float(route.get("confidence") or 0.5),
            provenance=_as_list(route.get("provenance")) or [_event_provenance(event)],
        )
        counts["review_routes"] += 1

    approval = _approval_from_event(metadata, payload, event)
    if approval:
        store.create_approval(**approval)
        counts["approvals"] += 1

    for sample in _communication_samples_from_event(
        metadata,
        payload,
        event,
        person_ids=person_ids,
        target_type=target_type,
    ):
        sample_person = sample.pop("person", None)
        if not sample.get("person_id") and isinstance(sample_person, dict):
            sample["person_id"] = store.upsert_person(
                _normalize_person(sample_person, target_type=target_type or None)
            )
            if sample["person_id"] not in known_person_ids:
                known_person_ids.add(sample["person_id"])
                counts["people"] += 1
        if store.record_communication_sample(**sample):
            counts["communication_samples"] += 1

    store.record_activity(
        activity_type="context_logged",
        event_id=str(event.get("id") or ""),
        summary=_event_summary(event, metadata),
        target_system=str(metadata.get("target_system") or payload.get("target_system") or ""),
        target_id=str(metadata.get("target_id") or metadata.get("path") or payload.get("target_id") or ""),
        actor=str(metadata.get("actor") or metadata.get("agent_name") or event.get("source") or ""),
        metadata={
            "event_type": event.get("type"),
            "op": event.get("op"),
            "people": counts["people"],
            "relationships": counts["relationships"],
            "review_routes": counts["review_routes"],
            "approvals": counts["approvals"],
            "communication_samples": counts["communication_samples"],
        },
    )
    counts["activity"] += 1
    return counts


def _empty_counts() -> dict[str, int]:
    return {
        "people": 0,
        "relationships": 0,
        "review_routes": 0,
        "approvals": 0,
        "activity": 0,
        "communication_samples": 0,
    }


def _looks_work_relevant(
    event: dict[str, Any],
    metadata: dict[str, Any],
    payload: dict[str, Any],
) -> bool:
    if event.get("type") == "work_assistant":
        return True
    keys = {
        "people",
        "person",
        "owner",
        "assignee",
        "requester",
        "reviewers",
        "relationships",
        "review_routes",
        "approval_request",
        "communication_sample",
        "communication_samples",
        "requires_approval",
        "style_notes",
        "target_system",
        "target_type",
        "work_item",
        "writing_sample",
        "writing_samples",
    }
    return any(key in metadata or key in payload for key in keys)


def _collect_people(
    metadata: dict[str, Any],
    payload: dict[str, Any],
    *,
    target_type: str | None,
) -> list[dict[str, Any]]:
    people: list[dict[str, Any]] = []
    for raw in _as_list(metadata.get("people") or payload.get("people")):
        person = _normalize_person(raw, target_type=target_type)
        person["source_key"] = person["person_id"]
        people.append(person)

    role_keys = {
        "owner": "owner",
        "author": "author",
        "actor": "actor",
        "assignee": "assignee",
        "requester": "requester",
        "approver": "approver",
    }
    for key, role in role_keys.items():
        raw = metadata.get(key, payload.get(key))
        if raw:
            person = _normalize_person(raw, role=role, target_type=target_type)
            person["source_key"] = key
            people.append(person)

    for raw in _as_list(metadata.get("reviewers") or payload.get("reviewers")):
        person = _normalize_person(raw, role="reviewer", target_type=target_type)
        person["source_key"] = person["person_id"]
        people.append(person)

    deduped: dict[str, dict[str, Any]] = {}
    for person in people:
        person_id = str(person.get("person_id") or "")
        if not person_id:
            continue
        if person_id in deduped:
            deduped[person_id]["roles"] = _merge_lists(
                deduped[person_id].get("roles", []),
                person.get("roles", []),
            )
            continue
        deduped[person_id] = person
    return list(deduped.values())


def _approval_from_event(
    metadata: dict[str, Any],
    payload: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, Any] | None:
    raw = metadata.get("approval_request") or payload.get("approval_request")
    if raw and isinstance(raw, dict):
        target_system = str(raw.get("target_system") or metadata.get("target_system") or "external")
        target_id = str(raw.get("target_id") or metadata.get("target_id") or metadata.get("path") or "")
        action = str(raw.get("action") or metadata.get("action") or event.get("op") or "external_write")
        return {
            "target_system": target_system,
            "target_id": target_id,
            "action": action,
            "summary": str(raw.get("summary") or "Approval requested by logged context"),
            "preview": raw.get("preview") or payload.get("preview") or {},
            "affected_people": _as_list(raw.get("affected_people") or metadata.get("affected_people")),
            "risk_level": str(raw.get("risk_level") or metadata.get("risk_level") or "medium"),
            "permission_required": str(raw.get("permission_required") or metadata.get("permission_required") or ""),
            "requested_by": str(raw.get("requested_by") or metadata.get("actor") or event.get("source") or "agent"),
            "expires_at": raw.get("expires_at"),
            "dedupe_key": raw.get("dedupe_key")
            or _stable_id("approval", event.get("id"), target_system, target_id, action),
        }

    action = str(metadata.get("action") or event.get("op") or "").strip()
    requires_approval = bool(metadata.get("requires_approval")) or action in EXTERNAL_WRITE_ACTIONS
    if not requires_approval:
        return None
    target_system = str(metadata.get("target_system") or payload.get("target_system") or "external")
    target_id = str(metadata.get("target_id") or metadata.get("path") or payload.get("target_id") or "")
    summary = str(metadata.get("summary") or f"Approve {action or 'external write'} on {target_system}")
    return {
        "target_system": target_system,
        "target_id": target_id,
        "action": action or "external_write",
        "summary": summary,
        "preview": payload.get("preview") or metadata.get("preview") or {},
        "affected_people": _as_list(metadata.get("affected_people") or []),
        "risk_level": str(metadata.get("risk_level") or "medium"),
        "permission_required": str(metadata.get("permission_required") or "human approval"),
        "requested_by": str(metadata.get("actor") or event.get("source") or "agent"),
        "dedupe_key": metadata.get("approval_dedupe_key")
        or _stable_id("approval", event.get("id"), target_system, target_id, action),
    }


def _communication_samples_from_event(
    metadata: dict[str, Any],
    payload: dict[str, Any],
    event: dict[str, Any],
    *,
    person_ids: dict[str, str],
    target_type: str,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    raw_values: list[Any] = []
    for key in ("communication_samples", "writing_samples"):
        raw_values.extend(_as_list(metadata.get(key) or payload.get(key)))
    for key in ("communication_sample", "writing_sample"):
        raw = metadata.get(key) if key in metadata else payload.get(key)
        if raw:
            raw_values.extend(_as_list(raw))

    provenance = _event_provenance(event)
    for raw in raw_values:
        if isinstance(raw, dict):
            record = dict(raw)
        else:
            record = {"text": str(raw)}

        text = str(
            record.get("text")
            or record.get("content")
            or record.get("excerpt")
            or record.get("body")
            or record.get("draft")
            or ""
        ).strip()
        if not text:
            continue

        source_system = str(
            record.get("source_system")
            or metadata.get("target_system")
            or payload.get("target_system")
            or event.get("source")
            or ""
        ).strip()
        source_id = str(
            record.get("source_id")
            or record.get("target_id")
            or metadata.get("target_id")
            or metadata.get("path")
            or payload.get("target_id")
            or event.get("id")
            or ""
        ).strip()
        channel = str(
            record.get("channel")
            or metadata.get("channel")
            or target_type
            or metadata.get("target_type")
            or ""
        ).strip()
        purpose = str(
            record.get("purpose")
            or metadata.get("purpose")
            or metadata.get("action")
            or event.get("op")
            or "work_communication"
        ).strip()

        style_notes = _merge_lists(
            _str_list(record.get("style_notes")),
            _str_list(record.get("style")),
            _str_list(record.get("tone")),
        )
        person_id = str(record.get("person_id") or "").strip()
        if not person_id and not isinstance(record.get("person"), dict):
            for source_key in ("author", "actor", "requester", "owner"):
                if source_key in person_ids:
                    person_id = person_ids[source_key]
                    break

        sample: dict[str, Any] = {
            "person_id": person_id,
            "source_system": source_system,
            "source_id": source_id,
            "channel": channel,
            "purpose": purpose or "work_communication",
            "text": text,
            "style_notes": style_notes,
            "provenance": _as_list(record.get("provenance")) or [provenance],
            "confidence": float(record.get("confidence") or metadata.get("confidence") or 0.5),
            "dedupe_key": record.get("dedupe_key")
            or _stable_id("comm", event.get("id"), source_system, source_id, purpose, text[:80]),
        }
        if isinstance(record.get("person"), dict):
            sample["person"] = record["person"]
        samples.append(sample)

    return samples


def _permission_class_for_role(role: str) -> str:
    if role in {"owner", "approver"}:
        return "can_approve"
    if role in {"reviewer", "maintainer"}:
        return "can_review"
    if role in {"stakeholder", "observer"}:
        return "informational"
    return ""


def _event_provenance(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event.get("id"),
        "event_type": event.get("type"),
        "source": event.get("source"),
        "op": event.get("op"),
        "timestamp": event.get("timestamp"),
    }


def _event_summary(event: dict[str, Any], metadata: dict[str, Any]) -> str:
    summary = str(metadata.get("summary") or "").strip()
    if summary:
        return summary
    pieces = [
        str(event.get("type") or "event"),
        str(event.get("source") or ""),
        str(event.get("op") or ""),
    ]
    return " ".join(piece for piece in pieces if piece).strip()
