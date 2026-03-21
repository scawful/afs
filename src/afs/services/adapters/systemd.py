"""Systemd adapter (render-only) for AFS services."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..models import ServiceDefinition, ServiceType
from .base import ServiceAdapter


class SystemdAdapter(ServiceAdapter):
    def __init__(self) -> None:
        super().__init__(platform_name="linux")

    def render(
        self,
        definition: ServiceDefinition,
        *,
        stdout_log: Path | None = None,
        stderr_log: Path | None = None,
    ) -> str:
        lines = [
            "[Unit]",
            f"Description={definition.label}",
            "",
            "[Service]",
            f"ExecStart={' '.join(definition.command)}",
        ]
        if definition.working_directory:
            lines.append(f"WorkingDirectory={definition.working_directory}")
        if definition.environment:
            for key, value in definition.environment.items():
                lines.append(f"Environment={key}={value}")
        if stdout_log is not None:
            lines.append(f"StandardOutput=append:{stdout_log}")
        if stderr_log is not None:
            lines.append(f"StandardError=append:{stderr_log}")
        if definition.service_type == ServiceType.ONESHOT:
            lines.append("Type=oneshot")
        if definition.keep_alive:
            lines.append("Restart=on-failure")
        lines.extend([
            "",
            "[Install]",
            "WantedBy=default.target",
        ])
        return "\n".join(lines)

    def file_extension(self) -> str:
        return ".service"

    def enable(self, unit_path: Path) -> subprocess.CompletedProcess[str] | None:
        return subprocess.run(
            ["systemctl", "--user", "enable", "--now", str(unit_path)],
            check=False,
            capture_output=True,
            text=True,
        )

    def disable(self, unit_path: Path) -> subprocess.CompletedProcess[str] | None:
        return subprocess.run(
            ["systemctl", "--user", "disable", "--now", str(unit_path)],
            check=False,
            capture_output=True,
            text=True,
        )

    def system_status(self, definition: ServiceDefinition, unit_path: Path) -> dict[str, object]:
        payload = {
            "installed": unit_path.exists(),
            "enabled": False,
            "active": False,
            "detail": "",
        }
        if not unit_path.exists():
            return payload
        result = subprocess.run(
            ["systemctl", "--user", "is-active", f"afs.{definition.name}.service"],
            check=False,
            capture_output=True,
            text=True,
        )
        payload["active"] = result.returncode == 0 and result.stdout.strip() == "active"
        enabled = subprocess.run(
            ["systemctl", "--user", "is-enabled", f"afs.{definition.name}.service"],
            check=False,
            capture_output=True,
            text=True,
        )
        payload["enabled"] = enabled.returncode == 0 and enabled.stdout.strip() == "enabled"
        payload["detail"] = enabled.stdout.strip() or result.stdout.strip() or enabled.stderr.strip() or result.stderr.strip()
        return payload
