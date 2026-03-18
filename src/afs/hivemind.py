"""Hivemind message bus for inter-agent communication."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_scope import assert_mount_allowed
from .models import MountType


@dataclass
class HivemindMessage:
    id: str
    from_agent: str
    to: str | None
    msg_type: str  # "finding", "request", "status"
    payload: dict[str, Any]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from": self.from_agent,
            "to": self.to,
            "type": self.msg_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HivemindMessage:
        return cls(
            id=str(data.get("id", "")),
            from_agent=str(data.get("from", "")),
            to=data.get("to"),
            msg_type=str(data.get("type", "status")),
            payload=data.get("payload") or {},
            timestamp=str(data.get("timestamp", "")),
        )


class HivemindBus:
    """File-based message bus using the hivemind mount directory."""

    def __init__(self, context_path: Path) -> None:
        assert_mount_allowed(MountType.HIVEMIND, operation="access")
        self._root = context_path / "hivemind"

    def send(
        self,
        from_agent: str,
        msg_type: str,
        payload: dict[str, Any] | None = None,
        *,
        to: str | None = None,
    ) -> HivemindMessage:
        now = datetime.now(timezone.utc)
        msg_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        message = HivemindMessage(
            id=msg_id,
            from_agent=from_agent,
            to=to,
            msg_type=msg_type,
            payload=payload or {},
            timestamp=now.isoformat(),
        )

        agent_dir = self._root / from_agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        msg_path = agent_dir / f"{msg_id}.json"
        msg_path.write_text(
            json.dumps(message.to_dict(), indent=2), encoding="utf-8"
        )
        return message

    def read(
        self,
        *,
        agent_name: str | None = None,
        since: datetime | None = None,
        msg_type: str | None = None,
        limit: int = 50,
    ) -> list[HivemindMessage]:
        messages: list[HivemindMessage] = []

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

                if since and msg.timestamp < since.isoformat():
                    continue
                if msg_type and msg.msg_type != msg_type:
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

    def cleanup(self, *, max_age_hours: int = 24) -> int:
        """Remove messages older than max_age_hours. Returns count removed."""
        if not self._root.exists():
            return 0
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        removed = 0
        for agent_dir in self._root.iterdir():
            if not agent_dir.is_dir() or agent_dir.name.startswith("."):
                continue
            for msg_file in agent_dir.glob("*.json"):
                try:
                    if msg_file.stat().st_mtime < cutoff:
                        msg_file.unlink()
                        removed += 1
                except OSError:
                    continue
            # Remove empty agent dirs
            if agent_dir.exists() and not any(agent_dir.iterdir()):
                try:
                    agent_dir.rmdir()
                except OSError:
                    pass
        return removed
