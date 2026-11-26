"""Inter-agent message bus.

Messages, recipient matches, and ack labels live in SQLite
(``messages.sqlite3`` on the hivemind mount). Scratchpad documents stay
files. Legacy per-message JSON is imported once and not written again.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .agent_scope import assert_mount_allowed
from .config import load_config_model
from .context_layout import LAYOUT_VERSION, detect_layout_version
from .context_paths import resolve_mount_root
from .message_store import MessageStore
from .models import MountType
from .path_safety import assert_no_linklike_components

logger = logging.getLogger(__name__)


def _record_history(
    op: str,
    agent_name: str,
    context_root: Path,
    *,
    msg_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append a hivemind history event; failures are logged, never raised."""
    try:
        from .history import log_hivemind_event

        log_hivemind_event(
            op,
            agent_name,
            msg_id=msg_id,
            metadata=metadata,
            context_root=context_root,
        )
    except Exception:  # noqa: BLE001 - history is telemetry; the bus mutation already happened
        logger.warning("hivemind history append failed (op=%s)", op, exc_info=True)


def _validate_agent_name(agent_name: str) -> str:
    """Return an agent name that is safe to use as one path component."""
    if (
        not isinstance(agent_name, str)
        or not agent_name
        or agent_name in {".", ".."}
        or "/" in agent_name
        or "\\" in agent_name
        or "\x00" in agent_name
    ):
        raise ValueError("agent name must be a non-empty single path component")
    return agent_name


BROADCAST_RECIPIENTS = frozenset({"all", "*", "everyone"})
ACK_STATUSES = ("read", "done")
SESSION_RETENTION_DAYS = 14
_RECIPIENT_SEPARATORS = re.compile(r"[,/;]+")


def normalize_agent_name(name: str) -> str:
    """Return the comparable form of an agent label ("Codex RC Planning" -> "codex-rc-planning")."""
    return re.sub(r"[\s_]+", "-", str(name).strip().lower()).strip("-")


def _agent_family(name: str) -> str:
    return name.split("-", 1)[0]


def recipient_matches(to: str | None, recipient: str) -> bool:
    """Return whether a ``to`` field addresses *recipient*.

    ``to`` may list several recipients separated by ``,``, ``/`` or ``;``.
    A harness name and its session names share a family, so ``claude`` and
    ``claude-kydrog`` both receive messages sent to either label.
    """
    if to is None:
        return True
    wanted = normalize_agent_name(recipient)
    if not wanted:
        return False
    for raw in _RECIPIENT_SEPARATORS.split(to):
        token = normalize_agent_name(raw)
        if not token:
            continue
        if token in BROADCAST_RECIPIENTS or token == wanted:
            return True
        if _agent_family(token) == wanted or token == _agent_family(wanted):
            return True
    return False


def message_summary(payload: Any, width: int = 120) -> str:
    """One-line preview of a message payload (``text``/``message``/``summary``/``ask`` first)."""
    if isinstance(payload, dict):
        for key in ("text", "message", "summary", "ask"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                text = " ".join(value.split())
                break
        else:
            text = json.dumps(payload, ensure_ascii=False)
    else:
        text = str(payload)
    return text if len(text) <= width else text[: width - 3] + "..."


@dataclass
class HivemindMessage:
    id: str
    from_agent: str
    to: str | None
    msg_type: str  # "finding", "request", "status"
    payload: dict[str, Any]
    timestamp: str
    topic: str | None = None
    expires_at: str | None = None
    scope_id: str = ""
    reply_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "from": self.from_agent,
            "to": self.to,
            "type": self.msg_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }
        if self.topic is not None:
            d["topic"] = self.topic
        if self.expires_at is not None:
            d["expires_at"] = self.expires_at
        if self.scope_id:
            d["scope_id"] = self.scope_id
        if self.reply_to is not None:
            d["reply_to"] = self.reply_to
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HivemindMessage:
        return cls(
            id=str(data.get("id", "")),
            from_agent=str(data.get("from", "")),
            to=data.get("to"),
            msg_type=str(data.get("type", "status")),
            payload=data.get("payload") or {},
            timestamp=str(data.get("timestamp", "")),
            topic=data.get("topic"),
            expires_at=data.get("expires_at"),
            scope_id=str(data.get("scope_id", "") or ""),
            reply_to=data.get("reply_to"),
        )


@dataclass
class HivemindSubscription:
    agent_name: str
    topics: list[str]
    created_at: str
    updated_at: str
    ttl_hours: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "agent_name": self.agent_name,
            "topics": list(self.topics),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.ttl_hours is not None:
            d["ttl_hours"] = self.ttl_hours
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HivemindSubscription:
        ttl_hours_raw = data.get("ttl_hours")
        ttl_hours = (
            int(ttl_hours_raw)
            if isinstance(ttl_hours_raw, (int, float)) and int(ttl_hours_raw) > 0
            else None
        )
        return cls(
            agent_name=str(data.get("agent_name", "")),
            topics=list(data.get("topics", [])),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            ttl_hours=ttl_hours,
        )


class HivemindBus:
    """SQLite message bus rooted at the hivemind mount directory."""

    def __init__(self, context_path: Path, *, config: Any = None) -> None:
        assert_mount_allowed(MountType.HIVEMIND, operation="access")
        self._context_path = context_path.expanduser().resolve()
        self._harden_paths = (
            detect_layout_version(self._context_path) == LAYOUT_VERSION
        )
        self._config = config or load_config_model()
        self._root = resolve_mount_root(
            self._context_path,
            MountType.HIVEMIND,
            config=self._config,
        )
        self._safe_path(self._root)

    def _store(self) -> MessageStore:
        self._reject_linked_queue_entries()
        return MessageStore(self._root)

    def _reject_linked_queue_entries(self) -> None:
        """Refuse a symlinked queue root or agent directory before opening SQLite."""
        self._safe_path(self._root)
        if not self._root.exists():
            return
        self._safe_path(self._root, allow_missing=False)
        for child in self._root.iterdir():
            if child.name.startswith("."):
                continue
            if child.name == "messages.sqlite3" or child.name.startswith("messages.sqlite3-"):
                continue
            self._safe_path(child)

    def _load(self, row: dict[str, Any]) -> HivemindMessage:
        return HivemindMessage.from_dict(row)

    def _safe_path(self, path: Path, *, allow_missing: bool = True) -> Path:
        """Reject link-like v2 queue paths without changing v1 mount behavior."""

        if not self._harden_paths:
            return path
        return assert_no_linklike_components(
            path,
            boundary=self._context_path,
            allow_missing=allow_missing,
        )

    def send(
        self,
        from_agent: str,
        msg_type: str,
        payload: dict[str, Any] | None = None,
        *,
        to: str | None = None,
        topic: str | None = None,
        ttl_hours: int | None = None,
        scope_id: str = "",
        reply_to: str | None = None,
    ) -> HivemindMessage:
        """Write one message.

        Directed messages (``to`` set) get no default expiry: they stay
        visible until the recipient marks them done or an explicit
        ``ttl_hours`` elapses.
        """
        safe_from_agent = _validate_agent_name(from_agent)
        if reply_to is not None:
            reply_to = _validate_agent_name(reply_to)
        now = datetime.now(timezone.utc)
        msg_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        effective_ttl = _normalize_ttl_hours(ttl_hours)
        if effective_ttl is None and to is None:
            effective_ttl = _normalize_ttl_hours(
                getattr(getattr(self._config, "hivemind", None), "default_ttl_hours", None)
            )
        expires_at = (
            (now + timedelta(hours=effective_ttl)).isoformat()
            if effective_ttl is not None
            else None
        )
        message = HivemindMessage(
            id=msg_id,
            from_agent=safe_from_agent,
            to=to,
            msg_type=msg_type,
            payload=payload or {},
            timestamp=now.isoformat(),
            topic=topic,
            expires_at=expires_at,
            scope_id=scope_id.strip(),
            reply_to=reply_to,
        )

        self._store().insert(message)
        _record_history(
            "send",
            safe_from_agent,
            self._context_path,
            msg_id=msg_id,
            metadata={
                "to": to,
                "msg_type": msg_type,
                "topic": topic,
                "ttl_hours": effective_ttl,
                "expires_at": expires_at,
                "scope_id": message.scope_id,
            },
        )
        return message

    def read(
        self,
        *,
        agent_name: str | None = None,
        since: datetime | None = None,
        msg_type: str | None = None,
        topic: str | None = None,
        limit: int = 50,
        scope_ids: set[str] | None = None,
        include_unscoped: bool = True,
        require_scope: bool = False,
        recipient: str | None = None,
        include_broadcasts: bool = True,
        exclude_from_norm: str | None = None,
    ) -> list[HivemindMessage]:
        if agent_name is not None:
            _validate_agent_name(agent_name)
        if not self._root.exists():
            return []
        store = self._store()
        store.ensure_imported()
        recipient_norm = normalize_agent_name(recipient) if recipient else None
        rows = store.query(
            from_agent=agent_name,
            msg_type=msg_type,
            topic=topic,
            since_iso=since.isoformat() if since is not None else None,
            scope_ids=scope_ids,
            include_unscoped=include_unscoped,
            require_scope=require_scope,
            recipient_norm=recipient_norm or None,
            recipient_family=_agent_family(recipient_norm) if recipient_norm else None,
            include_broadcasts=include_broadcasts,
            exclude_from_norm=exclude_from_norm,
            limit=limit,
        )
        return [self._load(row) for row in rows]

    def read_for(
        self,
        recipient: str,
        *,
        agent_name: str | None = None,
        since: datetime | None = None,
        msg_type: str | None = None,
        topic: str | None = None,
        limit: int = 50,
    ) -> list[HivemindMessage]:
        """Read messages addressed to a specific agent (see ``recipient_matches``)."""
        return self.read(
            agent_name=agent_name,
            since=since,
            msg_type=msg_type,
            topic=topic,
            limit=limit,
            recipient=recipient,
        )

    def inbox(
        self,
        recipient: str,
        *,
        agent_name: str | None = None,
        msg_type: str | None = None,
        topic: str | None = None,
        include_done: bool = False,
        include_broadcasts: bool = True,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return messages for *recipient* with that recipient's ack state.

        Each entry is the message dict plus ``state`` (``unread``, ``read``
        or ``done``). The recipient's own messages are excluded. Sender
        (*agent_name*), *msg_type* and *topic* filters apply before *limit*.
        ``since`` skips older messages before their ack state is read.
        """
        me = normalize_agent_name(recipient)
        if not me or not self._root.exists():
            return []
        store = self._store()
        store.ensure_imported()
        rows = store.query(
            from_agent=_validate_agent_name(agent_name) if agent_name is not None else None,
            msg_type=msg_type,
            topic=topic,
            since_iso=since.isoformat() if since is not None else None,
            recipient_norm=me,
            recipient_family=_agent_family(me),
            include_broadcasts=include_broadcasts,
            exclude_from_norm=me,
            exclude_done_for=None if include_done else me,
            limit=limit,
            **self._scope_kwargs(),
        )
        states = store.states_for([str(row["id"]) for row in rows], me)
        return [
            {**self._load(row).to_dict(), "state": states.get(str(row["id"]), "unread")}
            for row in rows
        ]

    def _scope_kwargs(self) -> dict[str, Any]:
        allowed = getattr(self, "allowed_scope_ids", None)
        if allowed is None:
            return {}
        all_projects = bool(getattr(self, "all_projects", False))
        include_legacy = bool(getattr(self, "include_legacy", False))
        return {
            "scope_ids": None if all_projects else set(allowed),
            "include_unscoped": include_legacy,
            "require_scope": all_projects and not include_legacy,
        }

    def _find_message(self, msg_id: str) -> HivemindMessage | None:
        safe_id = _validate_agent_name(msg_id)
        if not self._root.exists():
            return None
        store = self._store()
        store.ensure_imported()
        row = store.get(safe_id)
        if row is None:
            return None
        return self._load(row)

    def acks(self, msg_id: str) -> list[dict[str, str]]:
        """Return the ack records for one message, oldest first."""
        if not self._root.exists():
            return []
        return self._store().acks(_validate_agent_name(msg_id))

    def ack_state(self, msg_id: str, recipient: str) -> str:
        me = normalize_agent_name(recipient)
        if not me or not self._root.exists():
            return "unread"
        return self._store().ack_state(_validate_agent_name(msg_id), me)

    def ack(self, msg_id: str, by: str, *, status: str = "read", note: str = "") -> dict[str, Any]:
        """Record that *by* read or finished a message. Returns the message with acks."""
        if status not in ACK_STATUSES:
            raise ValueError(f"status must be one of {', '.join(ACK_STATUSES)}")
        actor = normalize_agent_name(by)
        if not actor:
            raise ValueError("ack requires a non-empty agent name")
        message = self._find_message(msg_id)
        if message is None:
            raise KeyError(f"message not found: {msg_id}")
        records = self._store().add_ack(
            message.id,
            actor,
            status=status,
            note=note,
            at=datetime.now(timezone.utc).isoformat(),
        )
        _record_history(
            "ack",
            actor,
            self._context_path,
            msg_id=message.id,
            metadata={"status": status},
        )
        return {**message.to_dict(), "acks": records}

    def register_session(
        self,
        host: str,
        session_id: str,
        *,
        scope_id: str | None = None,
        cwd: str = "",
        source: str = "",
    ) -> dict[str, str]:
        """Record a live host session (Claude session, Codex thread) for this queue.

        Re-registering the same session (resume, compact) moves ``registered_at``
        forward; push delivery only covers messages newer than that time.
        Records idle for ``SESSION_RETENTION_DAYS`` are pruned here.
        """
        store = self._store()
        store.ensure_imported()
        store.prune_sessions(retention_days=SESSION_RETENTION_DAYS)
        record = {
            "host": normalize_agent_name(host),
            "session_id": _validate_agent_name(session_id),
            "scope_id": (scope_id or "").strip(),
            "cwd": cwd,
            "source": source,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        return store.upsert_session(record)

    def session_record(self, host: str, session_id: str) -> dict[str, Any] | None:
        store = self._store()
        store.ensure_imported()
        return store.get_session(
            normalize_agent_name(host),
            _validate_agent_name(session_id),
        )

    def update_session(
        self, host: str, session_id: str, fields: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Merge *fields* into an existing session record; ``None`` when there is none."""
        record = self.session_record(host, session_id)
        if record is None:
            return None
        record.update(fields)
        return self._store().upsert_session(record)

    def live_sessions(self, host: str, *, scope_id: str | None = None) -> list[dict[str, Any]]:
        """Return recorded sessions for *host*, most recently registered first."""
        store = self._store()
        store.ensure_imported()
        return store.list_sessions(normalize_agent_name(host), scope_id=scope_id)

    def delivered_ids(self, host: str, session_id: str) -> set[str]:
        """Return the message ids already claimed for one session."""
        store = self._store()
        store.ensure_imported()
        return store.delivered_ids(
            normalize_agent_name(host),
            _validate_agent_name(session_id),
        )

    def claim_delivery(self, host: str, session_id: str, msg_id: str) -> bool:
        """Claim one message for one session; only the first claim returns True."""
        store = self._store()
        store.ensure_imported()
        return store.claim_delivery(
            normalize_agent_name(host),
            _validate_agent_name(session_id),
            _validate_agent_name(msg_id),
            claimed_at=datetime.now(timezone.utc).isoformat(),
        )

    def subscribe(
        self,
        agent_name: str,
        topics: list[str],
        *,
        ttl_hours: int | None = None,
    ) -> HivemindSubscription:
        """Merge topics into an agent's subscription."""
        safe_agent_name = _validate_agent_name(agent_name)
        now = datetime.now(timezone.utc).isoformat()
        store = self._store()
        store.ensure_imported()
        existing = self.get_subscriptions(safe_agent_name)
        normalized_topics = _merge_topics(existing.topics if existing else [], topics)
        normalized_ttl = _normalize_ttl_hours(ttl_hours)
        if existing:
            created_at = existing.created_at
            effective_ttl = existing.ttl_hours if normalized_ttl is None else normalized_ttl
        else:
            created_at = now
            effective_ttl = normalized_ttl
        store.upsert_subscription(
            safe_agent_name,
            normalized_topics,
            created_at=created_at,
            updated_at=now,
            ttl_hours=effective_ttl,
        )
        sub = HivemindSubscription(
            agent_name=safe_agent_name,
            topics=normalized_topics,
            created_at=created_at,
            updated_at=now,
            ttl_hours=effective_ttl,
        )
        _record_history(
            "subscribe",
            safe_agent_name,
            self._context_path,
            metadata={"topics": sub.topics, "ttl_hours": sub.ttl_hours},
        )
        return sub

    def unsubscribe(self, agent_name: str, topics: list[str]) -> HivemindSubscription:
        """Remove topics from an agent's subscription."""
        safe_agent_name = _validate_agent_name(agent_name)
        now = datetime.now(timezone.utc).isoformat()
        store = self._store()
        store.ensure_imported()
        existing = self.get_subscriptions(safe_agent_name)
        remaining = [t for t in (existing.topics if existing else []) if t not in set(topics)]
        created_at = existing.created_at if existing else now
        ttl_hours = existing.ttl_hours if existing else None
        store.upsert_subscription(
            safe_agent_name,
            remaining,
            created_at=created_at,
            updated_at=now,
            ttl_hours=ttl_hours,
        )
        sub = HivemindSubscription(
            agent_name=safe_agent_name,
            topics=remaining,
            created_at=created_at,
            updated_at=now,
            ttl_hours=ttl_hours,
        )
        _record_history(
            "unsubscribe",
            safe_agent_name,
            self._context_path,
            metadata={"topics": list(topics), "remaining_topics": sub.topics},
        )
        return sub

    def get_subscriptions(self, agent_name: str) -> HivemindSubscription | None:
        """Read an agent's subscription."""
        safe_agent_name = _validate_agent_name(agent_name)
        store = self._store()
        store.ensure_imported()
        data = store.get_subscription(safe_agent_name)
        return HivemindSubscription.from_dict(data) if data is not None else None

    def read_subscribed(
        self,
        agent_name: str,
        *,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[HivemindMessage]:
        """Read messages matching agent's subscribed topics + direct + topicless."""
        sub = self.get_subscriptions(agent_name)
        subscribed_topics = set(sub.topics) if sub else set()
        effective_since = since
        if effective_since is None and sub and sub.ttl_hours:
            effective_since = datetime.now(timezone.utc) - timedelta(hours=sub.ttl_hours)
        all_msgs = self.read(since=effective_since, limit=0)
        matched = []
        for m in all_msgs:
            # Direct messages to this agent
            if m.to is not None and recipient_matches(m.to, agent_name):
                matched.append(m)
            # Topicless broadcasts
            elif m.topic is None and m.to is None:
                matched.append(m)
            # Topic matches subscription
            elif m.topic and m.topic in subscribed_topics:
                matched.append(m)
        matched.sort(key=lambda m: m.timestamp)
        return matched[-limit:] if len(matched) > limit else matched

    def cleanup(self, *, max_age_hours: int = 24) -> int:
        """Remove messages older than max_age_hours. Returns count removed."""
        return int(self.cleanup_stats(max_age_hours=max_age_hours)["removed_count"])

    def reap(
        self,
        *,
        max_age_hours: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Reap expired or aged-out messages using the configured/default retention."""
        effective_hours = max_age_hours
        if effective_hours is None:
            effective_hours = getattr(
                getattr(self._config, "hivemind", None),
                "default_ttl_hours",
                24,
            )
        return self.cleanup_stats(max_age_hours=int(effective_hours), dry_run=dry_run)

    def cleanup_stats(self, *, max_age_hours: int = 24, dry_run: bool = False) -> dict[str, Any]:
        """Cleanup with detailed statistics. Returns removal breakdown."""
        empty: dict[str, Any] = {
            "removed_count": 0,
            "remaining_count": 0,
            "expired_count": 0,
            "aged_out_count": 0,
            "oldest_remaining": None,
            "per_agent": {},
        }
        if not self._root.exists():
            return empty
        store = self._store()
        store.ensure_imported()
        summary = store.cleanup_stats(max_age_hours=max_age_hours, dry_run=dry_run)
        if summary["removed_count"] > 0 and not dry_run:
            _record_history(
                "reap",
                "system",
                self._context_path,
                metadata={
                    "removed_count": summary["removed_count"],
                    "expired_count": summary["expired_count"],
                    "aged_out_count": summary["aged_out_count"],
                },
            )
        return summary


def _normalize_ttl_hours(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and int(value) > 0:
        return int(value)
    return None


def _merge_topics(existing: list[str], updates: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for raw_topic in [*existing, *updates]:
        topic = str(raw_topic).strip()
        if not topic or topic in seen:
            continue
        seen.add(topic)
        merged.append(topic)
    return merged
