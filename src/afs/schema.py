"""Minimal configuration schema for AFS."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value).expanduser().resolve()


@dataclass
class WorkspaceDirectory:
    path: Path
    description: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkspaceDirectory":
        path = _as_path(data.get("path", ""))
        description = data.get("description")
        return cls(path=path, description=description)


@dataclass
class GeneralConfig:
    context_root: Path = field(default_factory=lambda: Path.home() / ".context")
    agent_workspaces_dir: Path = field(
        default_factory=lambda: Path.home() / ".context" / "workspaces"
    )
    python_executable: Path | None = None
    workspace_directories: list[WorkspaceDirectory] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneralConfig":
        context_root = data.get("context_root")
        agent_workspaces_dir = data.get("agent_workspaces_dir")
        python_executable = data.get("python_executable")
        workspace_directories = [
            WorkspaceDirectory.from_dict(item)
            for item in data.get("workspace_directories", [])
            if isinstance(item, dict)
        ]
        return cls(
            context_root=_as_path(context_root)
            if context_root
            else cls().context_root,
            agent_workspaces_dir=_as_path(agent_workspaces_dir)
            if agent_workspaces_dir
            else cls().agent_workspaces_dir,
            python_executable=_as_path(python_executable)
            if python_executable
            else None,
            workspace_directories=workspace_directories,
        )


@dataclass
class PluginsConfig:
    enabled_plugins: list[str] = field(default_factory=list)
    plugin_dirs: list[Path] = field(default_factory=list)
    auto_discover: bool = True
    auto_discover_prefixes: list[str] = field(
        default_factory=lambda: ["afs_plugin", "afs_scawful"]
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginsConfig":
        enabled_plugins = [
            item for item in data.get("enabled_plugins", []) if isinstance(item, str)
        ]
        plugin_dirs = [
            _as_path(item)
            for item in data.get("plugin_dirs", [])
            if isinstance(item, (str, Path))
        ]
        auto_discover = data.get("auto_discover", True)
        prefixes = data.get("auto_discover_prefixes")
        if prefixes and isinstance(prefixes, list):
            auto_discover_prefixes = [p for p in prefixes if isinstance(p, str)]
        else:
            auto_discover_prefixes = cls().auto_discover_prefixes
        return cls(
            enabled_plugins=enabled_plugins,
            plugin_dirs=plugin_dirs,
            auto_discover=bool(auto_discover),
            auto_discover_prefixes=auto_discover_prefixes,
        )


@dataclass
class CognitiveConfig:
    enabled: bool = False
    record_emotions: bool = False
    record_metacognition: bool = False
    record_goals: bool = False
    record_epistemic: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CognitiveConfig":
        return cls(
            enabled=bool(data.get("enabled", False)),
            record_emotions=bool(data.get("record_emotions", False)),
            record_metacognition=bool(data.get("record_metacognition", False)),
            record_goals=bool(data.get("record_goals", False)),
            record_epistemic=bool(data.get("record_epistemic", False)),
        )


@dataclass
class AFSConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AFSConfig":
        data = data or {}
        general = GeneralConfig.from_dict(data.get("general", {}))
        plugins = PluginsConfig.from_dict(data.get("plugins", {}))
        cognitive = CognitiveConfig.from_dict(data.get("cognitive", {}))
        return cls(general=general, plugins=plugins, cognitive=cognitive)
