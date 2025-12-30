"""Minimal service manager for AFS background tasks."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Iterable

from ..config import load_config_model
from ..schema import AFSConfig, ServiceConfig
from .adapters.base import ServiceAdapter
from .adapters.launchd import LaunchdAdapter
from .adapters.systemd import SystemdAdapter
from .models import ServiceDefinition, ServiceState, ServiceStatus, ServiceType


class ServiceManager:
    """Build and render service definitions without mutating the system."""

    def __init__(
        self,
        config: AFSConfig | None = None,
        *,
        service_root: Path | None = None,
        platform_name: str | None = None,
    ) -> None:
        self.config = config or load_config_model()
        self.service_root = service_root or Path.home() / ".config" / "afs" / "services"
        self.platform_name = platform_name or platform.system().lower()
        self._adapter = self._build_adapter(self.platform_name)

    def list_definitions(self) -> list[ServiceDefinition]:
        definitions = self._builtin_definitions()
        merged = self._merge_config(definitions)
        return sorted(merged.values(), key=lambda item: item.name)

    def get_definition(self, name: str) -> ServiceDefinition | None:
        merged = self._merge_config(self._builtin_definitions())
        return merged.get(name)

    def render_unit(self, name: str) -> str:
        definition = self.get_definition(name)
        if not definition:
            raise KeyError(f"Unknown service: {name}")
        return self._adapter.render(definition)

    def status(self, name: str) -> ServiceStatus:
        definition = self.get_definition(name)
        if not definition:
            return ServiceStatus(name=name, state=ServiceState.UNKNOWN, enabled=False)
        return ServiceStatus(name=definition.name, state=ServiceState.UNKNOWN, enabled=False)

    def _merge_config(
        self, definitions: dict[str, ServiceDefinition]
    ) -> dict[str, ServiceDefinition]:
        merged = dict(definitions)
        for name, config in self.config.services.services.items():
            if not config.enabled:
                merged.pop(name, None)
                continue
            base = merged.get(name)
            if base:
                merged[name] = _merge_definition(base, config)
            elif config.command:
                merged[name] = ServiceDefinition(
                    name=config.name,
                    label=config.name,
                    command=list(config.command),
                    working_directory=config.working_directory,
                    environment=dict(config.environment),
                    service_type=ServiceType.DAEMON,
                    keep_alive=True,
                    run_at_load=config.auto_start,
                )
        return merged

    def _builtin_definitions(self) -> dict[str, ServiceDefinition]:
        python = self._resolve_python_executable()
        repo_root = self._find_repo_root()
        environment = self._service_environment()

        return {
            "orchestrator": ServiceDefinition(
                name="orchestrator",
                label="AFS Orchestrator",
                description="Routing and coordination for local agents",
                command=[python, "-m", "afs.orchestration", "--daemon"],
                working_directory=repo_root,
                environment=environment,
                service_type=ServiceType.DAEMON,
                keep_alive=True,
                run_at_load=False,
            ),
            "context-discovery": ServiceDefinition(
                name="context-discovery",
                label="AFS Context Discovery",
                description="Discover and index AFS contexts",
                command=[python, "-m", "afs", "context", "discover"],
                working_directory=repo_root,
                environment=environment,
                service_type=ServiceType.ONESHOT,
                keep_alive=False,
                run_at_load=False,
            ),
            "context-graph-export": ServiceDefinition(
                name="context-graph-export",
                label="AFS Context Graph Export",
                description="Export AFS context graph JSON",
                command=[python, "-m", "afs", "graph", "export"],
                working_directory=repo_root,
                environment=environment,
                service_type=ServiceType.ONESHOT,
                keep_alive=False,
                run_at_load=False,
            ),
        }

    def _build_adapter(self, platform_name: str) -> ServiceAdapter:
        platform_name = platform_name.lower()
        if platform_name.startswith("darwin") or platform_name.startswith("mac"):
            return LaunchdAdapter()
        if platform_name.startswith("linux"):
            return SystemdAdapter()
        return SystemdAdapter()

    def _resolve_python_executable(self) -> str:
        if self.config.general.python_executable:
            return str(self.config.general.python_executable)
        return sys.executable

    def _find_repo_root(self) -> Path | None:
        for parent in Path(__file__).resolve().parents:
            if (parent / "pyproject.toml").exists():
                return parent
        return None

    def _service_environment(self) -> dict[str, str]:
        env: dict[str, str] = {}
        repo_root = self._find_repo_root()
        if repo_root and (repo_root / "src").exists():
            env["PYTHONPATH"] = str(repo_root / "src")
        user_config = Path.home() / ".config" / "afs" / "config.toml"
        if user_config.exists():
            env["AFS_CONFIG_PATH"] = str(user_config)
            env["AFS_PREFER_USER_CONFIG"] = "1"
        return env


def _merge_definition(
    base: ServiceDefinition, override: ServiceConfig
) -> ServiceDefinition:
    command = list(override.command) if override.command else list(base.command)
    environment = dict(base.environment)
    environment.update(override.environment)
    return ServiceDefinition(
        name=base.name,
        label=base.label,
        description=base.description,
        command=command,
        working_directory=override.working_directory or base.working_directory,
        environment=environment,
        service_type=base.service_type,
        keep_alive=base.keep_alive,
        run_at_load=override.auto_start,
    )

