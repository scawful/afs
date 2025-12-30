"""Minimal configuration schema for AFS."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .models import MountType


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value).expanduser().resolve()


def default_discovery_ignore() -> list[str]:
    return ["legacy", "archive", "archives"]


class PolicyType(str, Enum):
    READ_ONLY = "read_only"
    WRITABLE = "writable"
    EXECUTABLE = "executable"


@dataclass
class DirectoryConfig:
    name: str
    policy: PolicyType
    description: str = ""
    role: MountType | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DirectoryConfig":
        name = str(data.get("name", "")).strip()
        role_raw = data.get("role")
        role = None
        if isinstance(role_raw, str):
            try:
                role = MountType(role_raw)
            except ValueError:
                role = None
        if not name and role:
            name = role.value
        policy_raw = data.get("policy", PolicyType.READ_ONLY.value)
        try:
            policy = PolicyType(policy_raw)
        except ValueError:
            policy = PolicyType.READ_ONLY
        description = data.get("description") if isinstance(data.get("description"), str) else ""
        return cls(name=name, policy=policy, description=description, role=role)


def default_directory_configs() -> list[DirectoryConfig]:
    return [
        DirectoryConfig(
            name="memory",
            policy=PolicyType.READ_ONLY,
            role=MountType.MEMORY,
        ),
        DirectoryConfig(
            name="knowledge",
            policy=PolicyType.READ_ONLY,
            role=MountType.KNOWLEDGE,
        ),
        DirectoryConfig(
            name="tools",
            policy=PolicyType.EXECUTABLE,
            role=MountType.TOOLS,
        ),
        DirectoryConfig(
            name="scratchpad",
            policy=PolicyType.WRITABLE,
            role=MountType.SCRATCHPAD,
        ),
        DirectoryConfig(
            name="history",
            policy=PolicyType.READ_ONLY,
            role=MountType.HISTORY,
        ),
        DirectoryConfig(
            name="hivemind",
            policy=PolicyType.WRITABLE,
            role=MountType.HIVEMIND,
        ),
        DirectoryConfig(
            name="global",
            policy=PolicyType.WRITABLE,
            role=MountType.GLOBAL,
        ),
        DirectoryConfig(
            name="items",
            policy=PolicyType.WRITABLE,
            role=MountType.ITEMS,
        ),
    ]


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
    discovery_ignore: list[str] = field(default_factory=default_discovery_ignore)

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
        raw_ignore = data.get("discovery_ignore")
        if isinstance(raw_ignore, list):
            discovery_ignore = [item for item in raw_ignore if isinstance(item, str)]
        else:
            discovery_ignore = default_discovery_ignore()
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
            discovery_ignore=discovery_ignore,
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
class AgentConfig:
    name: str
    role: str = "general"
    backend: str = "local"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    auto_start: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        tags = data.get("tags", [])
        if isinstance(tags, list):
            tags = [tag for tag in tags if isinstance(tag, str)]
        else:
            tags = []
        return cls(
            name=str(data.get("name", "")).strip(),
            role=str(data.get("role", "general")).strip() or "general",
            backend=str(data.get("backend", "local")).strip() or "local",
            description=str(data.get("description", "")).strip(),
            tags=tags,
            auto_start=bool(data.get("auto_start", False)),
        )


@dataclass
class OrchestratorConfig:
    enabled: bool = False
    max_agents: int = 5
    default_agents: list[AgentConfig] = field(default_factory=list)
    auto_routing: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrchestratorConfig":
        agents_raw = data.get("default_agents", [])
        agents = [
            AgentConfig.from_dict(item)
            for item in agents_raw
            if isinstance(item, dict)
        ]
        max_agents = data.get("max_agents", cls().max_agents)
        return cls(
            enabled=bool(data.get("enabled", False)),
            max_agents=int(max_agents) if isinstance(max_agents, int) else cls().max_agents,
            default_agents=agents,
            auto_routing=bool(data.get("auto_routing", True)),
        )


@dataclass
class ServiceConfig:
    name: str
    enabled: bool = True
    auto_start: bool = False
    command: list[str] = field(default_factory=list)
    working_directory: Path | None = None
    environment: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServiceConfig":
        command = data.get("command", [])
        if isinstance(command, list):
            command = [str(item) for item in command]
        else:
            command = []
        env = data.get("environment", {})
        if isinstance(env, dict):
            environment = {str(key): str(value) for key, value in env.items()}
        else:
            environment = {}
        working_directory = data.get("working_directory")
        return cls(
            name=str(data.get("name", "")).strip(),
            enabled=bool(data.get("enabled", True)),
            auto_start=bool(data.get("auto_start", False)),
            command=command,
            working_directory=_as_path(working_directory)
            if working_directory
            else None,
            environment=environment,
        )


@dataclass
class ServicesConfig:
    enabled: bool = False
    services: dict[str, ServiceConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServicesConfig":
        enabled = bool(data.get("enabled", False))
        raw_services = data.get("services", {})
        parsed: dict[str, ServiceConfig] = {}
        if isinstance(raw_services, dict):
            for name, payload in raw_services.items():
                if not isinstance(payload, dict):
                    continue
                payload = dict(payload)
                payload.setdefault("name", name)
                parsed[name] = ServiceConfig.from_dict(payload)
        return cls(enabled=enabled, services=parsed)


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
    directories: list[DirectoryConfig] = field(default_factory=default_directory_configs)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    services: ServicesConfig = field(default_factory=ServicesConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AFSConfig":
        data = data or {}
        general = GeneralConfig.from_dict(data.get("general", {}))
        plugins = PluginsConfig.from_dict(data.get("plugins", {}))
        directories = _parse_directory_config(data)
        cognitive = CognitiveConfig.from_dict(data.get("cognitive", {}))
        orchestrator = OrchestratorConfig.from_dict(data.get("orchestrator", {}))
        services = ServicesConfig.from_dict(data.get("services", {}))
        return cls(
            general=general,
            plugins=plugins,
            directories=directories,
            cognitive=cognitive,
            orchestrator=orchestrator,
            services=services,
        )


def _parse_directory_config(data: dict[str, Any]) -> list[DirectoryConfig]:
    raw = data.get("directories")
    if raw is None:
        raw = data.get("afs_directories")
    if raw is None:
        return default_directory_configs()
    if not isinstance(raw, list):
        return default_directory_configs()
    return [DirectoryConfig.from_dict(item) for item in raw if isinstance(item, dict)]
