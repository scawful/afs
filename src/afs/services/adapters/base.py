"""Base classes for AFS service adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess

from ..models import ServiceDefinition


@dataclass
class ServiceAdapter:
    platform_name: str

    def render(
        self,
        definition: ServiceDefinition,
        *,
        stdout_log: Path | None = None,
        stderr_log: Path | None = None,
    ) -> str:
        raise NotImplementedError

    def file_extension(self) -> str:
        return ""

    def unit_path(self, service_root: Path, definition: ServiceDefinition) -> Path:
        return service_root / self.platform_name / f"afs.{definition.name}{self.file_extension()}"

    def install(
        self,
        service_root: Path,
        definition: ServiceDefinition,
        *,
        stdout_log: Path | None = None,
        stderr_log: Path | None = None,
    ) -> Path:
        unit_path = self.unit_path(service_root, definition)
        unit_path.parent.mkdir(parents=True, exist_ok=True)
        unit_path.write_text(
            self.render(
                definition,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
            )
            + "\n",
            encoding="utf-8",
        )
        return unit_path

    def uninstall(self, service_root: Path, definition: ServiceDefinition) -> bool:
        unit_path = self.unit_path(service_root, definition)
        if not unit_path.exists():
            return False
        unit_path.unlink()
        return True

    def enable(self, _unit_path: Path) -> CompletedProcess[str] | None:
        return None

    def disable(self, _unit_path: Path) -> CompletedProcess[str] | None:
        return None

    def system_status(self, _definition: ServiceDefinition, unit_path: Path) -> dict[str, object]:
        return {
            "installed": unit_path.exists(),
            "enabled": False,
            "active": False,
            "detail": "",
        }
