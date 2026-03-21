"""Data models for AFS service management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ServiceState(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    FAILED = "failed"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


class ServiceType(str, Enum):
    DAEMON = "daemon"
    ONESHOT = "oneshot"


@dataclass
class ServiceDefinition:
    name: str
    label: str
    description: str = ""
    command: list[str] = field(default_factory=list)
    working_directory: Path | None = None
    environment: dict[str, str] = field(default_factory=dict)
    service_type: ServiceType = ServiceType.DAEMON
    keep_alive: bool = True
    run_at_load: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "command": list(self.command),
            "working_directory": str(self.working_directory)
            if self.working_directory
            else None,
            "environment": dict(self.environment),
            "service_type": self.service_type.value,
            "keep_alive": self.keep_alive,
            "run_at_load": self.run_at_load,
        }


@dataclass
class ServiceStatus:
    name: str
    state: ServiceState
    pid: int | None = None
    enabled: bool = False
    installed: bool = False
    unit_path: Path | None = None
    stdout_log: Path | None = None
    stderr_log: Path | None = None
    last_started: datetime | None = None
    last_stopped: datetime | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "pid": self.pid,
            "enabled": self.enabled,
            "installed": self.installed,
            "unit_path": str(self.unit_path) if self.unit_path else None,
            "stdout_log": str(self.stdout_log) if self.stdout_log else None,
            "stderr_log": str(self.stderr_log) if self.stderr_log else None,
            "last_started": self.last_started.isoformat() if self.last_started else None,
            "last_stopped": self.last_stopped.isoformat() if self.last_stopped else None,
            "error_message": self.error_message,
        }
