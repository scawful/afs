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

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
            "accomplished": list(self.accomplished),
            "blocked": list(self.blocked),
            "next_steps": list(self.next_steps),
            "context_snapshot": dict(self.context_snapshot),
            "open_tasks": list(self.open_tasks),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandoffPacket:
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
