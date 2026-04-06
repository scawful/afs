"""Structured conversation handoff protocol for AFS."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context_paths import resolve_mount_root
from .models import MountType

HANDOFF_SCHEMA_VERSION = "2"


@dataclass
class HandoffPacket:
    session_id: str
    agent_name: str
    timestamp: str
    accomplished: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    open_tasks: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    target_agent: str | None = None
    priority: str = "normal"
    schema_version: str = HANDOFF_SCHEMA_VERSION
    acknowledged_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
            "accomplished": list(self.accomplished),
            "blocked": list(self.blocked),
            "next_steps": list(self.next_steps),
            "context_snapshot": dict(self.context_snapshot),
            "open_tasks": list(self.open_tasks),
            "metadata": dict(self.metadata),
            "priority": self.priority,
            "schema_version": self.schema_version,
            "acknowledged_by": list(self.acknowledged_by),
        }
        if self.target_agent is not None:
            result["target_agent"] = self.target_agent
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandoffPacket:
        target_agent_raw = data.get("target_agent")
        target_agent = str(target_agent_raw) if target_agent_raw is not None else None
        acknowledged_raw = data.get("acknowledged_by")
        acknowledged_by = (
            [str(item) for item in acknowledged_raw if isinstance(item, str)]
            if isinstance(acknowledged_raw, list)
            else []
        )
        return cls(
            session_id=str(data.get("session_id", "")),
            agent_name=str(data.get("agent_name", "")),
            timestamp=str(data.get("timestamp", "")),
            accomplished=list(data.get("accomplished", [])),
            blocked=list(data.get("blocked", [])),
            next_steps=list(data.get("next_steps", [])),
            context_snapshot=data.get("context_snapshot") or {},
            open_tasks=list(data.get("open_tasks", [])),
            metadata=data.get("metadata") or {},
            target_agent=target_agent,
            priority=str(data.get("priority", "normal")).strip() or "normal",
            schema_version=str(data.get("schema_version", "1")).strip() or "1",
            acknowledged_by=acknowledged_by,
        )


class HandoffStore:
    """File-based handoff packet store under scratchpad/handoffs/."""

    def __init__(self, context_path: Path, *, config: Any = None) -> None:
        self._context_path = context_path.expanduser().resolve()
        self._root = resolve_mount_root(
            self._context_path, MountType.SCRATCHPAD, config=config
        ) / "handoffs"
        self._root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._root / "_manifest.json"

    def create(
        self,
        *,
        session_id: str | None = None,
        agent_name: str,
        accomplished: list[str] | None = None,
        blocked: list[str] | None = None,
        next_steps: list[str] | None = None,
        context_snapshot: dict[str, Any] | None = None,
        open_tasks: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        target_agent: str | None = None,
        priority: str = "normal",
    ) -> HandoffPacket:
        sid = session_id or uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        packet = HandoffPacket(
            session_id=sid,
            agent_name=agent_name,
            timestamp=now,
            accomplished=accomplished or [],
            blocked=blocked or [],
            next_steps=next_steps or [],
            context_snapshot=context_snapshot or {},
            open_tasks=open_tasks or [],
            metadata=metadata or {},
            target_agent=target_agent,
            priority=priority,
            schema_version=HANDOFF_SCHEMA_VERSION,
        )
        packet_path = self._root / f"{sid}.json"
        packet_path.write_text(
            json.dumps(packet.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        self._update_manifest(sid)
        try:
            from .history import log_session_event
            log_session_event(
                "handoff",
                session_id=sid,
                metadata={
                    "agent_name": agent_name,
                    "accomplished_count": len(packet.accomplished),
                    "blocked_count": len(packet.blocked),
                    "target_agent": target_agent or "",
                    "priority": priority,
                },
                context_root=self._context_path,
            )
        except Exception:
            pass
        return packet

    def read(self, *, session_id: str | None = None) -> HandoffPacket | None:
        if session_id:
            packet_path = self._root / f"{session_id}.json"
        else:
            manifest = self._load_manifest()
            if not manifest:
                return None
            latest_id = manifest[-1]
            packet_path = self._root / f"{latest_id}.json"
        if not packet_path.exists():
            return None
        try:
            data = json.loads(packet_path.read_text(encoding="utf-8"))
            return HandoffPacket.from_dict(data)
        except (json.JSONDecodeError, OSError):
            return None

    def list(self, *, limit: int = 10) -> list[HandoffPacket]:
        manifest = self._load_manifest()
        packets: list[HandoffPacket] = []
        for sid in reversed(manifest):
            if len(packets) >= limit:
                break
            packet = self.read(session_id=sid)
            if packet:
                packets.append(packet)
        return packets

    def pending_for_agent(self, name: str) -> list[HandoffPacket]:
        """Return unacknowledged handoff packets targeted at *name*.

        A packet is pending if its ``target_agent`` matches *name* and
        *name* is not yet listed in ``acknowledged_by``.  Results are
        ordered newest-first and high-priority packets sort before
        normal ones at the same position.
        """
        manifest = self._load_manifest()
        pending: list[HandoffPacket] = []
        for sid in reversed(manifest):
            packet = self.read(session_id=sid)
            if packet is None:
                continue
            if packet.target_agent != name:
                continue
            if name in packet.acknowledged_by:
                continue
            pending.append(packet)

        priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        pending.sort(key=lambda p: priority_order.get(p.priority, 2))
        return pending

    def acknowledge(self, session_id: str, name: str) -> bool:
        """Mark handoff *session_id* as acknowledged by agent *name*.

        Returns ``True`` if the packet was found and updated, ``False``
        otherwise.  Acknowledging is idempotent — calling it again for
        the same name is a no-op that still returns ``True``.
        """
        packet_path = self._root / f"{session_id}.json"
        if not packet_path.exists():
            return False
        try:
            data = json.loads(packet_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        packet = HandoffPacket.from_dict(data)
        if name not in packet.acknowledged_by:
            packet.acknowledged_by.append(name)
            packet_path.write_text(
                json.dumps(packet.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
        return True

    def _load_manifest(self) -> list[str]:
        if not self._manifest_path.exists():
            return []
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(item) for item in data]
        except (json.JSONDecodeError, OSError):
            pass
        return []

    def _update_manifest(self, session_id: str) -> None:
        manifest = self._load_manifest()
        if session_id not in manifest:
            manifest.append(session_id)
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
