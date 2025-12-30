"""Systemd adapter (render-only) for AFS services."""

from __future__ import annotations

from .base import ServiceAdapter
from ..models import ServiceDefinition, ServiceType


class SystemdAdapter(ServiceAdapter):
    def __init__(self) -> None:
        super().__init__(platform_name="linux")

    def render(self, definition: ServiceDefinition) -> str:
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
