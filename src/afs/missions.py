"""Durable background-mission records for AFS.

A *mission* is a long-running unit of background/incident work that outlives a
single session or subagent — the state that otherwise gets lost or duplicated when
one agent hands off to another. Missions are stored as JSON records under the
``items`` mount (with a manifest index), mirroring the handoff store, so a resumed
session can see what is already in flight.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context_paths import resolve_mount_root
from .models import MountType

MISSION_SCHEMA_VERSION = "1"
VALID_MISSION_STATUSES = ("active", "blocked", "done", "abandoned")
# States that represent work still in flight (surfaced into session context).
OPEN_MISSION_STATUSES = ("active", "blocked")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Mission:
    mission_id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    summary: str = ""
    owner: str = ""
    next_steps: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    linked_sessions: list[str] = field(default_factory=list)
    linked_handoffs: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = MISSION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
            "owner": self.owner,
            "next_steps": list(self.next_steps),
            "blockers": list(self.blockers),
            "linked_sessions": list(self.linked_sessions),
            "linked_handoffs": list(self.linked_handoffs),
            "tags": list(self.tags),
            "log": list(self.log),
            "metadata": dict(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Mission:
        def _str_list(value: Any) -> list[str]:
            if not isinstance(value, list):
                return []
            return [str(item) for item in value if str(item).strip()]

        return cls(
            mission_id=str(data.get("mission_id", "")),
            title=str(data.get("title", "")),
            status=str(data.get("status", "active")).strip() or "active",
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            summary=str(data.get("summary", "")),
            owner=str(data.get("owner", "")),
            next_steps=_str_list(data.get("next_steps")),
            blockers=_str_list(data.get("blockers")),
            linked_sessions=_str_list(data.get("linked_sessions")),
            linked_handoffs=_str_list(data.get("linked_handoffs")),
            tags=_str_list(data.get("tags")),
            log=[entry for entry in data.get("log", []) if isinstance(entry, dict)],
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
            schema_version=str(data.get("schema_version", "1")).strip() or "1",
        )


class MissionNotFoundError(KeyError):
    """Raised when a mission id does not resolve to a stored record."""


class MissionStore:
    """File-based mission store under ``items/missions/``."""

    def __init__(self, context_path: Path, *, config: Any = None) -> None:
        self._context_path = context_path.expanduser().resolve()
        # Construction is READ-ONLY: never create the mission directory here. Session
        # bootstrap constructs a store just to read active missions, and that read path
        # must not dirty a repo or an external mount. The directory is created lazily on
        # the first write (see :meth:`_ensure_root`).
        self._root = resolve_mount_root(
            self._context_path, MountType.ITEMS, config=config
        ) / "missions"
        self._manifest_path = self._root / "_manifest.json"

    # -- persistence helpers -------------------------------------------------
    def _ensure_root(self) -> None:
        """Create the mission directory. Called only from write paths."""
        self._root.mkdir(parents=True, exist_ok=True)

    def _mission_path(self, mission_id: str) -> Path:
        return self._root / f"{mission_id}.json"

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        """Write via a temp file + atomic rename so a concurrent reader/writer never
        observes a partially written file."""
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    def _load_manifest(self) -> list[str]:
        if not self._manifest_path.exists():
            return []
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return [str(item) for item in data] if isinstance(data, list) else []

    def _disk_mission_ids(self) -> list[str]:
        """Mission ids present on disk, independent of the manifest."""
        if not self._root.exists():
            return []
        return sorted(path.stem for path in self._root.glob("mission_*.json"))

    def _reconciled_ids(self) -> list[str]:
        """Manifest order plus any on-disk mission the manifest is missing.

        ``_append_manifest`` is a lock-free read-modify-write, so two concurrent
        ``create`` calls can race and the later manifest write can drop the earlier id.
        Reconciling reads against the actual ``mission_*.json`` files means a mission is
        never invisible to ``list``/``active``/session bootstrap just because it lost the
        race — the durable per-mission file is the source of truth, the manifest only
        supplies ordering.
        """
        manifest = self._load_manifest()
        seen = set(manifest)
        extras = [mid for mid in self._disk_mission_ids() if mid not in seen]
        return manifest + extras

    def _append_manifest(self, mission_id: str) -> None:
        self._ensure_root()
        manifest = self._load_manifest()
        if mission_id not in manifest:
            manifest.append(mission_id)
            self._atomic_write(self._manifest_path, json.dumps(manifest, indent=2) + "\n")

    def _write(self, mission: Mission) -> None:
        self._ensure_root()
        self._atomic_write(
            self._mission_path(mission.mission_id),
            json.dumps(mission.to_dict(), indent=2) + "\n",
        )

    # -- public API ----------------------------------------------------------
    def create(
        self,
        *,
        title: str,
        summary: str = "",
        owner: str = "",
        next_steps: list[str] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Mission:
        if not title.strip():
            raise ValueError("mission title is required")
        now = _now()
        mission = Mission(
            mission_id=f"mission_{uuid.uuid4().hex[:12]}",
            title=title.strip(),
            status="active",
            created_at=now,
            updated_at=now,
            summary=summary.strip(),
            owner=owner.strip(),
            next_steps=list(next_steps or []),
            tags=list(tags or []),
            metadata=dict(metadata or {}),
        )
        self._write(mission)
        self._append_manifest(mission.mission_id)
        self._log_event("mission_created", mission)
        return mission

    def get(self, mission_id: str) -> Mission | None:
        path = self._mission_path(mission_id)
        if not path.exists():
            return None
        try:
            return Mission.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return None

    def list(
        self, *, status: str | None = None, limit: int = 50
    ) -> list[Mission]:
        missions: list[Mission] = []
        for mission_id in reversed(self._reconciled_ids()):
            mission = self.get(mission_id)
            if mission is None:
                continue
            if status and mission.status != status:
                continue
            missions.append(mission)
            if len(missions) >= max(1, limit):
                break
        return missions

    def active(self, *, limit: int = 20) -> list[Mission]:
        """Return in-flight missions (active or blocked), newest first."""
        missions = [
            mission
            for mission in self.list(limit=1000)
            if mission.status in OPEN_MISSION_STATUSES
        ]
        return missions[: max(1, limit)]

    def update(
        self,
        mission_id: str,
        *,
        status: str | None = None,
        summary: str | None = None,
        owner: str | None = None,
        next_steps: list[str] | None = None,
        blockers: list[str] | None = None,
        link_session: str | None = None,
        link_handoff: str | None = None,
        add_tags: list[str] | None = None,
        note: str | None = None,
        actor: str = "",
    ) -> Mission:
        mission = self.get(mission_id)
        if mission is None:
            raise MissionNotFoundError(mission_id)

        if status is not None:
            if status not in VALID_MISSION_STATUSES:
                raise ValueError(
                    f"invalid mission status {status!r}; valid: "
                    + ", ".join(VALID_MISSION_STATUSES)
                )
            mission.status = status
        if summary is not None:
            mission.summary = summary.strip()
        if owner is not None:
            mission.owner = owner.strip()
        if next_steps is not None:
            mission.next_steps = list(next_steps)
        if blockers is not None:
            mission.blockers = list(blockers)
        if link_session:
            if link_session not in mission.linked_sessions:
                mission.linked_sessions.append(link_session)
        if link_handoff:
            if link_handoff not in mission.linked_handoffs:
                mission.linked_handoffs.append(link_handoff)
        if add_tags:
            for tag in add_tags:
                if tag and tag not in mission.tags:
                    mission.tags.append(tag)
        if note and note.strip():
            mission.log.append(
                {"timestamp": _now(), "actor": actor, "note": note.strip()}
            )

        mission.updated_at = _now()
        self._write(mission)
        self._log_event("mission_updated", mission)
        return mission

    # -- history -------------------------------------------------------------
    def _log_event(self, event_type: str, mission: Mission) -> None:
        try:
            from .history import log_session_event

            log_session_event(
                event_type,
                session_id=mission.mission_id,
                metadata={
                    "title": mission.title,
                    "status": mission.status,
                    "owner": mission.owner,
                },
                context_root=self._context_path,
            )
        except Exception:
            pass
