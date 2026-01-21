"""Core AFS data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class MountType(str, Enum):
    """Supported AFS directory roles."""

    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    TOOLS = "tools"
    SCRATCHPAD = "scratchpad"
    HISTORY = "history"
    HIVEMIND = "hivemind"
    GLOBAL = "global"
    ITEMS = "items"


@dataclass(frozen=True)
class MountPoint:
    """A mounted resource inside an AFS directory."""

    name: str
    source: Path
    mount_type: MountType
    is_symlink: bool = True


@dataclass
class ProjectMetadata:
    """Metadata for an AFS context root."""

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    agents: list[str] = field(default_factory=list)
    directories: dict[str, str] = field(default_factory=dict)
    manual_only: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ProjectMetadata:
        data = data or {}
        created_at = data.get("created_at")
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()
        if not isinstance(created_at, str):
            created_at = datetime.now().isoformat()
        description = data.get("description") if isinstance(data.get("description"), str) else ""
        agents = [agent for agent in data.get("agents", []) if isinstance(agent, str)]
        manual_only = [p for p in data.get("manual_only", []) if isinstance(p, str)]
        directories: dict[str, str] = {}
        raw_dirs = data.get("directories")
        if isinstance(raw_dirs, dict):
            for key, value in raw_dirs.items():
                directories[str(key)] = str(value)
        return cls(
            created_at=created_at,
            description=description,
            agents=agents,
            directories=directories,
            manual_only=manual_only,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "description": self.description,
            "agents": list(self.agents),
            "directories": dict(self.directories),
            "manual_only": list(self.manual_only),
        }


@dataclass
class ContextRoot:
    """An AFS .context directory."""

    path: Path
    project_name: str
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    mounts: dict[MountType, list[MountPoint]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        required = [mount_type.value for mount_type in MountType]
        directory_map = self.metadata.directories if self.metadata else {}
        return all(
            (self.path / directory_map.get(role, role)).exists() for role in required
        )

    @property
    def total_mounts(self) -> int:
        return sum(len(mounts) for mounts in self.mounts.values())

    def get_mounts(self, mount_type: MountType) -> list[MountPoint]:
        return self.mounts.get(mount_type, [])
