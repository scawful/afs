"""Launchd adapter (render-only) for AFS services."""

from __future__ import annotations

import json

from ..models import ServiceDefinition
from .base import ServiceAdapter


class LaunchdAdapter(ServiceAdapter):
    def __init__(self) -> None:
        super().__init__(platform_name="darwin")

    def render(self, definition: ServiceDefinition) -> str:
        payload: dict[str, object] = {
            "Label": f"afs.{definition.name}",
            "ProgramArguments": list(definition.command),
            "RunAtLoad": bool(definition.run_at_load),
            "KeepAlive": bool(definition.keep_alive),
        }
        if definition.working_directory:
            payload["WorkingDirectory"] = str(definition.working_directory)
        if definition.environment:
            payload["EnvironmentVariables"] = dict(definition.environment)
        return json.dumps(payload, indent=2)

    def file_extension(self) -> str:
        return ".plist"
