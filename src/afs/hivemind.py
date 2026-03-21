"""Hivemind message bus for inter-agent communication."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .agent_scope import assert_mount_allowed
from .config import load_config_model
from .context_paths import resolve_mount_root
from .models import MountType


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
        ttl_hours = int(ttl_hours_raw) if isinstance(ttl_hours_raw, (int, float)) and int(ttl_hours_raw) > 0 else None
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
        self._config = config or load_config_model()
        self._root = resolve_mount_root(
            self._context_path,
            MountType.HIVEMIND,
            config=self._config,
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
    ) -> HivemindMessage:
        now = datetime.now(timezone.utc)
        msg_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        effective_ttl = _normalize_ttl_hours(ttl_hours)
        if effective_ttl is None:
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
            from_agent=from_agent,
            to=to,
            msg_type=msg_type,
            payload=payload or {},
            timestamp=now.isoformat(),
            topic=topic,
            expires_at=expires_at,
        )

        agent_dir = self._root / from_agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        msg_path = agent_dir / f"{msg_id}.json"
        msg_path.write_text(
            json.dumps(message.to_dict(), indent=2), encoding="utf-8"
        )
        try:
            from .history import log_hivemind_event
            log_hivemind_event(
                "send",
                from_agent,
                msg_id=msg_id,
                metadata={
                    "to": to,
                    "msg_type": msg_type,
                    "topic": topic,
                    "ttl_hours": effective_ttl,
                    "expires_at": expires_at,
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

        if agent_name:
            scan_dirs = [self._root / agent_name]
        elif self._root.exists():
            scan_dirs = sorted(
                d for d in self._root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
        else:
            return []

        for agent_dir in scan_dirs:
            if not agent_dir.exists():
                continue
            for msg_file in sorted(agent_dir.glob("*.json")):
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
        """Read messages addressed to a specific agent."""
        all_msgs = self.read(since=since, limit=0)
        matched = [
            m for m in all_msgs
            if m.to == recipient or m.to is None
        ]
        matched.sort(key=lambda m: m.timestamp)
        return matched[-limit:] if len(matched) > limit else matched

    def _subscriptions_dir(self) -> Path:
        sub_dir = self._root / ".subscriptions"
        sub_dir.mkdir(parents=True, exist_ok=True)
        return sub_dir

    def subscribe(
        self,
        agent_name: str,
        topics: list[str],
        *,
        ttl_hours: int | None = None,
    ) -> HivemindSubscription:
        """Merge topics into an agent's subscription file."""
        now = datetime.now(timezone.utc).isoformat()
        existing = self.get_subscriptions(agent_name)
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
                agent_name=agent_name,
                topics=normalized_topics,
                created_at=now,
                updated_at=now,
                ttl_hours=normalized_ttl,
            )
        sub_path = self._subscriptions_dir() / f"{agent_name}.json"
        sub_path.write_text(json.dumps(sub.to_dict(), indent=2), encoding="utf-8")
        try:
            from .history import log_hivemind_event
            log_hivemind_event(
                "subscribe",
                agent_name,
                metadata={"topics": sub.topics, "ttl_hours": sub.ttl_hours},
                context_root=self._context_path,
            )
        except Exception:
            pass
        return sub

    def unsubscribe(self, agent_name: str, topics: list[str]) -> HivemindSubscription:
        """Remove topics from an agent's subscription."""
        now = datetime.now(timezone.utc).isoformat()
        existing = self.get_subscriptions(agent_name)
        if existing:
            existing.topics = [t for t in existing.topics if t not in set(topics)]
            existing.updated_at = now
            sub = existing
        else:
            sub = HivemindSubscription(
                agent_name=agent_name,
                topics=[],
                created_at=now,
                updated_at=now,
            )
        sub_path = self._subscriptions_dir() / f"{agent_name}.json"
        sub_path.write_text(json.dumps(sub.to_dict(), indent=2), encoding="utf-8")
        try:
            from .history import log_hivemind_event
            log_hivemind_event(
                "unsubscribe",
                agent_name,
                metadata={"topics": list(topics), "remaining_topics": sub.topics},
                context_root=self._context_path,
            )
        except Exception:
            pass
        return sub

    def get_subscriptions(self, agent_name: str) -> HivemindSubscription | None:
        """Read an agent's subscription file."""
        sub_path = self._subscriptions_dir() / f"{agent_name}.json"
        if not sub_path.exists():
            return None
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
            if m.to == agent_name:
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
        if not self._root.exists():
            return {
                "removed_count": 0,
                "remaining_count": 0,
                "expired_count": 0,
                "aged_out_count": 0,
                "oldest_remaining": None,
                "per_agent": {},
            }
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        now = datetime.now(timezone.utc)
        removed_count = 0
        remaining_count = 0
        expired_count = 0
        aged_out_count = 0
        oldest_remaining: float | None = None
        per_agent: dict[str, dict[str, int]] = {}
        for agent_dir in self._root.iterdir():
            if not agent_dir.is_dir() or agent_dir.name.startswith("."):
                continue
            agent_name = agent_dir.name
            agent_stats = {"removed": 0, "remaining": 0, "expired": 0, "aged_out": 0}
            for msg_file in agent_dir.glob("*.json"):
                try:
                    mtime = msg_file.stat().st_mtime
                except OSError:
                    continue
                expired = False
                try:
                    msg = HivemindMessage.from_dict(
                        json.loads(msg_file.read_text(encoding="utf-8"))
                    )
                    expired = _message_is_expired(msg, now)
                except (json.JSONDecodeError, OSError):
                    pass
                aged_out = mtime < cutoff
                if expired or aged_out:
                    if not dry_run:
                        try:
                            msg_file.unlink()
                        except OSError:
                            continue
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
