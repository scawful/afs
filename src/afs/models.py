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
    MONOREPO = "monorepo"


# Mount types that are not required for context validation.
OPTIONAL_MOUNT_TYPES: frozenset[MountType] = frozenset({MountType.MONOREPO})


@dataclass(frozen=True)
class MountProvenance:
    """Persisted metadata describing why a mount exists."""

    alias: str
    mount_type: MountType
    source: Path
    managed_by: str = "manual"
    profile_name: str | None = None
    remapped_from: Path | None = None
    updated_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MountProvenance | None:
        alias = data.get("alias")
        mount_type_raw = data.get("mount_type")
        source = data.get("source")
        if not isinstance(alias, str) or not alias.strip():
            return None
        if not isinstance(mount_type_raw, str):
            return None
        if not isinstance(source, str) or not source.strip():
            return None
        try:
            mount_type = MountType(mount_type_raw)
        except ValueError:
            return None
        profile_name = data.get("profile_name")
        remapped_from = data.get("remapped_from")
        updated_at = data.get("updated_at")
        return cls(
            alias=alias.strip(),
            mount_type=mount_type,
            source=Path(source).expanduser().resolve(),
            managed_by=str(data.get("managed_by", "manual")).strip() or "manual",
            profile_name=profile_name.strip() if isinstance(profile_name, str) and profile_name.strip() else None,
            remapped_from=Path(remapped_from).expanduser().resolve()
            if isinstance(remapped_from, str) and remapped_from.strip()
            else None,
            updated_at=updated_at if isinstance(updated_at, str) and updated_at.strip() else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "mount_type": self.mount_type.value,
            "source": str(self.source),
            "managed_by": self.managed_by,
            "profile_name": self.profile_name,
            "remapped_from": str(self.remapped_from) if self.remapped_from else None,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class MountPoint:
    """A mounted resource inside an AFS directory."""

    name: str
    source: Path
    mount_type: MountType
    is_symlink: bool = True
    provenance: MountProvenance | None = None


@dataclass
class ProjectMetadata:
    """Metadata for an AFS context root."""

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    agents: list[str] = field(default_factory=list)
    directories: dict[str, str] = field(default_factory=dict)
    manual_only: list[str] = field(default_factory=list)
    mount_provenance: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

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
        mount_provenance: dict[str, dict[str, dict[str, Any]]] = {}
        raw_mounts = data.get("mount_provenance")
        if isinstance(raw_mounts, dict):
            for mount_type, entries in raw_mounts.items():
                if not isinstance(entries, dict):
                    continue
                normalized: dict[str, dict[str, Any]] = {}
                for alias, payload in entries.items():
                    if not isinstance(alias, str) or not alias.strip():
                        continue
                    if not isinstance(payload, dict):
                        continue
                    normalized[alias] = dict(payload)
                if normalized:
                    mount_provenance[str(mount_type)] = normalized
        return cls(
            created_at=created_at,
            description=description,
            agents=agents,
            directories=directories,
            manual_only=manual_only,
            mount_provenance=mount_provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "description": self.description,
            "agents": list(self.agents),
            "directories": dict(self.directories),
            "manual_only": list(self.manual_only),
            "mount_provenance": {
                mount_type: {
                    alias: dict(payload)
                    for alias, payload in sorted(entries.items())
                }
                for mount_type, entries in sorted(self.mount_provenance.items())
            },
        }

    def get_mount_provenance(
        self,
        mount_type: MountType,
        alias: str,
    ) -> MountProvenance | None:
        payload = self.mount_provenance.get(mount_type.value, {}).get(alias)
        if not isinstance(payload, dict):
            return None
        return MountProvenance.from_dict(payload)

    def set_mount_provenance(self, provenance: MountProvenance) -> None:
        by_mount = self.mount_provenance.setdefault(provenance.mount_type.value, {})
        by_mount[provenance.alias] = provenance.to_dict()

    def remove_mount_provenance(self, mount_type: MountType, alias: str) -> None:
        by_mount = self.mount_provenance.get(mount_type.value)
        if not by_mount:
            return
        by_mount.pop(alias, None)
        if not by_mount:
            self.mount_provenance.pop(mount_type.value, None)

    def iter_mount_provenance(self) -> list[MountProvenance]:
        entries: list[MountProvenance] = []
        for mount_type in self.mount_provenance.values():
            for payload in mount_type.values():
                if not isinstance(payload, dict):
                    continue
                provenance = MountProvenance.from_dict(payload)
                if provenance is not None:
                    entries.append(provenance)
        return entries


@dataclass
class ContextRoot:
    """An AFS .context directory."""

    path: Path
    project_name: str
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    mounts: dict[MountType, list[MountPoint]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        required = [
            mt.value for mt in MountType if mt not in OPTIONAL_MOUNT_TYPES
        ]
        directory_map = self.metadata.directories if self.metadata else {}
        return all(
            (self.path / directory_map.get(role, role)).exists() for role in required
        )

    @property
    def total_mounts(self) -> int:
        return sum(len(mounts) for mounts in self.mounts.values())

    def get_mounts(self, mount_type: MountType) -> list[MountPoint]:
        return self.mounts.get(mount_type, [])
