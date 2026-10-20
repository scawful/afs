"""Hivemind message bus for inter-agent communication."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .agent_scope import assert_mount_allowed
from .atomic_io import atomic_write_text as _atomic_write_text
from .config import load_config_model
from .context_layout import LAYOUT_VERSION, detect_layout_version
from .context_paths import resolve_mount_root
from .models import MountType
from .path_safety import assert_no_linklike_components


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
    """File-based message bus using the hivemind mount directory."""

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

    def _safe_path(self, path: Path, *, allow_missing: bool = True) -> Path:
        """Reject link-like v2 queue paths without changing v1 mount behavior."""

        if not self._harden_paths:
            return path
        return assert_no_linklike_components(
            path,
            boundary=self._context_path,
            allow_missing=allow_missing,
        )

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        self._safe_path(path.parent, allow_missing=False)
        self._safe_path(path)
        _atomic_write_text(path, json.dumps(payload, indent=2))

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

        agent_dir = self._root / safe_from_agent
        self._safe_path(agent_dir)
        agent_dir.mkdir(parents=True, exist_ok=True)
        self._safe_path(agent_dir, allow_missing=False)
        msg_path = agent_dir / f"{msg_id}.json"
        self._write_json(msg_path, message.to_dict())
        try:
            from .history import log_hivemind_event

            log_hivemind_event(
                "send",
                safe_from_agent,
                msg_id=msg_id,
                metadata={
                    "to": to,
                    "msg_type": msg_type,
                    "topic": topic,
                    "ttl_hours": effective_ttl,
                    "expires_at": expires_at,
                    "scope_id": message.scope_id,
                },
                context_root=self._context_path,
            )
        except Exception:
            pass
        return message

    def read(
        self,
        *,
        agent_name: str | None = None,
        since: datetime | None = None,
        msg_type: str | None = None,
        topic: str | None = None,
        limit: int = 50,
    ) -> list[HivemindMessage]:
        messages: list[HivemindMessage] = []
        now = datetime.now(timezone.utc)

        if agent_name is not None:
            agent_dir = self._root / _validate_agent_name(agent_name)
            self._safe_path(agent_dir)
            scan_dirs = [agent_dir]
        else:
            self._safe_path(self._root)
            if not self._root.exists():
                return []
            self._safe_path(self._root, allow_missing=False)
            scan_dirs = []
            for candidate in sorted(self._root.iterdir()):
                if candidate.name.startswith("."):
                    continue
                self._safe_path(candidate)
                if candidate.is_dir():
                    scan_dirs.append(candidate)

        for agent_dir in scan_dirs:
            self._safe_path(agent_dir)
            if not agent_dir.exists():
                continue
            self._safe_path(agent_dir)
            for msg_file in sorted(agent_dir.glob("*.json")):
                self._safe_path(msg_file)
                try:
                    data = json.loads(msg_file.read_text(encoding="utf-8"))
                    msg = HivemindMessage.from_dict(data)
                except (json.JSONDecodeError, OSError):
                    continue
                if _message_is_expired(msg, now):
                    continue

                if since and msg.timestamp < since.isoformat():
                    continue
                if msg_type and msg.msg_type != msg_type:
                    continue
                if topic and msg.topic != topic:
                    continue
                messages.append(msg)

        messages.sort(key=lambda m: m.timestamp)
        return messages[-limit:] if len(messages) > limit else messages

    def read_for(
        self,
        recipient: str,
        *,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[HivemindMessage]:
        """Read messages addressed to a specific agent (see ``recipient_matches``)."""
        all_msgs = self.read(since=since, limit=0)
        matched = [m for m in all_msgs if recipient_matches(m.to, recipient)]
        matched.sort(key=lambda m: m.timestamp)
        return matched[-limit:] if len(matched) > limit else matched

    def inbox(
        self,
        recipient: str,
        *,
        include_done: bool = False,
        include_broadcasts: bool = True,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return messages for *recipient* with that recipient's ack state.

        Each entry is the message dict plus ``state`` (``unread``, ``read``
        or ``done``). The recipient's own messages are excluded.
        """
        me = normalize_agent_name(recipient)
        entries: list[dict[str, Any]] = []
        for message in self.read_for(recipient, limit=0):
            if normalize_agent_name(message.from_agent) == me:
                continue
            if message.to is None and not include_broadcasts:
                continue
            state = self.ack_state(message.id, recipient)
            if state == "done" and not include_done:
                continue
            entries.append({**message.to_dict(), "state": state})
        return entries[-limit:] if limit and len(entries) > limit else entries

    def _acks_dir(self) -> Path:
        ack_dir = self._root / ".acks"
        self._safe_path(ack_dir)
        return ack_dir

    def _find_message(self, msg_id: str) -> HivemindMessage | None:
        safe_id = _validate_agent_name(msg_id)
        self._safe_path(self._root)
        if not self._root.exists():
            return None
        for agent_dir in sorted(self._root.iterdir()):
            if agent_dir.name.startswith(".") or not agent_dir.is_dir():
                continue
            candidate = agent_dir / f"{safe_id}.json"
            self._safe_path(candidate)
            if not candidate.is_file():
                continue
            try:
                return HivemindMessage.from_dict(
                    json.loads(candidate.read_text(encoding="utf-8"))
                )
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def acks(self, msg_id: str) -> list[dict[str, str]]:
        """Return the ack records for one message, oldest first."""
        path = self._acks_dir() / f"{_validate_agent_name(msg_id)}.json"
        self._safe_path(path)
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        records = data.get("acks") if isinstance(data, dict) else None
        if not isinstance(records, list):
            return []
        return [r for r in records if isinstance(r, dict)]

    def ack_state(self, msg_id: str, recipient: str) -> str:
        me = normalize_agent_name(recipient)
        state = "unread"
        for record in self.acks(msg_id):
            if normalize_agent_name(str(record.get("by", ""))) != me:
                continue
            if record.get("status") == "done":
                return "done"
            state = "read"
        return state

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
        ack_dir = self._acks_dir()
        ack_dir.mkdir(parents=True, exist_ok=True)
        records = self.acks(message.id)
        record: dict[str, str] = {
            "by": actor,
            "status": status,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        if note:
            record["note"] = note
        records.append(record)
        self._write_json(ack_dir / f"{message.id}.json", {"msg_id": message.id, "acks": records})
        try:
            from .history import log_hivemind_event

            log_hivemind_event(
                "ack",
                actor,
                msg_id=message.id,
                metadata={"status": status},
                context_root=self._context_path,
            )
        except Exception:
            pass
        return {**message.to_dict(), "acks": records}

    def _subscriptions_dir(self) -> Path:
        sub_dir = self._root / ".subscriptions"
        self._safe_path(sub_dir)
        sub_dir.mkdir(parents=True, exist_ok=True)
        self._safe_path(sub_dir, allow_missing=False)
        return sub_dir

    def subscribe(
        self,
        agent_name: str,
        topics: list[str],
        *,
        ttl_hours: int | None = None,
    ) -> HivemindSubscription:
        """Merge topics into an agent's subscription file."""
        safe_agent_name = _validate_agent_name(agent_name)
        now = datetime.now(timezone.utc).isoformat()
        existing = self.get_subscriptions(safe_agent_name)
        normalized_topics = _merge_topics(existing.topics if existing else [], topics)
        normalized_ttl = _normalize_ttl_hours(ttl_hours)
        if existing:
            existing.topics = normalized_topics
            existing.updated_at = now
            if normalized_ttl is not None:
                existing.ttl_hours = normalized_ttl
            sub = existing
        else:
            sub = HivemindSubscription(
                agent_name=safe_agent_name,
                topics=normalized_topics,
                created_at=now,
                updated_at=now,
                ttl_hours=normalized_ttl,
            )
        sub_path = self._subscriptions_dir() / f"{safe_agent_name}.json"
        self._write_json(sub_path, sub.to_dict())
        try:
            from .history import log_hivemind_event

            log_hivemind_event(
                "subscribe",
                safe_agent_name,
                metadata={"topics": sub.topics, "ttl_hours": sub.ttl_hours},
                context_root=self._context_path,
            )
        except Exception:
            pass
        return sub

    def unsubscribe(self, agent_name: str, topics: list[str]) -> HivemindSubscription:
        """Remove topics from an agent's subscription."""
        safe_agent_name = _validate_agent_name(agent_name)
        now = datetime.now(timezone.utc).isoformat()
        existing = self.get_subscriptions(safe_agent_name)
        if existing:
            existing.topics = [t for t in existing.topics if t not in set(topics)]
            existing.updated_at = now
            sub = existing
        else:
            sub = HivemindSubscription(
                agent_name=safe_agent_name,
                topics=[],
                created_at=now,
                updated_at=now,
            )
        sub_path = self._subscriptions_dir() / f"{safe_agent_name}.json"
        self._write_json(sub_path, sub.to_dict())
        try:
            from .history import log_hivemind_event

            log_hivemind_event(
                "unsubscribe",
                safe_agent_name,
                metadata={"topics": list(topics), "remaining_topics": sub.topics},
                context_root=self._context_path,
            )
        except Exception:
            pass
        return sub

    def get_subscriptions(self, agent_name: str) -> HivemindSubscription | None:
        """Read an agent's subscription file."""
        safe_agent_name = _validate_agent_name(agent_name)
        sub_path = self._subscriptions_dir() / f"{safe_agent_name}.json"
        self._safe_path(sub_path)
        if not sub_path.exists():
            return None
        self._safe_path(sub_path)
        try:
            data = json.loads(sub_path.read_text(encoding="utf-8"))
            return HivemindSubscription.from_dict(data)
        except (json.JSONDecodeError, OSError):
            return None

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
        self._safe_path(self._root)
        if not self._root.exists():
            return {
                "removed_count": 0,
                "remaining_count": 0,
                "expired_count": 0,
                "aged_out_count": 0,
                "oldest_remaining": None,
                "per_agent": {},
            }
        self._safe_path(self._root, allow_missing=False)
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        now = datetime.now(timezone.utc)
        removed_count = 0
        remaining_count = 0
        expired_count = 0
        aged_out_count = 0
        oldest_remaining: float | None = None
        per_agent: dict[str, dict[str, int]] = {}
        for agent_dir in self._root.iterdir():
            if agent_dir.name.startswith("."):
                continue
            self._safe_path(agent_dir)
            if not agent_dir.is_dir():
                continue
            agent_name = agent_dir.name
            agent_stats = {"removed": 0, "remaining": 0, "expired": 0, "aged_out": 0}
            for msg_file in agent_dir.glob("*.json"):
                self._safe_path(msg_file)
                try:
                    mtime = msg_file.stat().st_mtime
                except OSError:
                    continue
                expired = False
                awaiting_recipient = False
                try:
                    msg = HivemindMessage.from_dict(
                        json.loads(msg_file.read_text(encoding="utf-8"))
                    )
                    expired = _message_is_expired(msg, now)
                    awaiting_recipient = msg.to is not None and not any(
                        record.get("status") == "done" for record in self.acks(msg.id)
                    )
                except (json.JSONDecodeError, OSError):
                    pass
                aged_out = mtime < cutoff and not awaiting_recipient
                if expired or aged_out:
                    if not dry_run:
                        try:
                            self._safe_path(msg_file)
                            msg_file.unlink()
                        except OSError:
                            continue
                        ack_file = self._acks_dir() / msg_file.name
                        self._safe_path(ack_file)
                        ack_file.unlink(missing_ok=True)
                    agent_stats["removed"] += 1
                    removed_count += 1
                    if expired:
                        agent_stats["expired"] += 1
                        expired_count += 1
                    if aged_out:
                        agent_stats["aged_out"] += 1
                        aged_out_count += 1
                else:
                    agent_stats["remaining"] += 1
                    remaining_count += 1
                    if oldest_remaining is None or mtime < oldest_remaining:
                        oldest_remaining = mtime
            per_agent[agent_name] = agent_stats
            if not dry_run and agent_dir.exists() and not any(agent_dir.iterdir()):
                try:
                    self._safe_path(agent_dir)
                    agent_dir.rmdir()
                except OSError:
                    pass
        oldest_iso = None
        if oldest_remaining is not None:
            oldest_iso = datetime.fromtimestamp(oldest_remaining, tz=timezone.utc).isoformat()
        summary = {
            "removed_count": removed_count,
            "remaining_count": remaining_count,
            "expired_count": expired_count,
            "aged_out_count": aged_out_count,
            "oldest_remaining": oldest_iso,
            "per_agent": per_agent,
        }
        if removed_count > 0 and not dry_run:
            try:
                from .history import log_hivemind_event

                log_hivemind_event(
                    "reap",
                    "system",
                    metadata={
                        "removed_count": removed_count,
                        "expired_count": expired_count,
                        "aged_out_count": aged_out_count,
                    },
                    context_root=self._context_path,
                )
            except Exception:
                pass
        return summary


def _normalize_ttl_hours(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and int(value) > 0:
        return int(value)
    return None


def _message_is_expired(message: HivemindMessage, now: datetime | None = None) -> bool:
    if not message.expires_at:
        return False
    current = now or datetime.now(timezone.utc)
    try:
        expires_at = datetime.fromisoformat(message.expires_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at <= current.astimezone(timezone.utc)


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
