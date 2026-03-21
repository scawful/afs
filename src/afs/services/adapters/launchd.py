"""Launchd adapter (render-only) for AFS services."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ..models import ServiceDefinition
from .base import ServiceAdapter


class LaunchdAdapter(ServiceAdapter):
    def __init__(self) -> None:
        super().__init__(platform_name="darwin")

    def render(
        self,
        definition: ServiceDefinition,
        *,
        stdout_log: Path | None = None,
        stderr_log: Path | None = None,
    ) -> str:
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
        if stdout_log is not None:
            payload["StandardOutPath"] = str(stdout_log)
        if stderr_log is not None:
            payload["StandardErrorPath"] = str(stderr_log)
        return json.dumps(payload, indent=2)

    def file_extension(self) -> str:
        return ".plist"

    def enable(self, unit_path: Path) -> subprocess.CompletedProcess[str] | None:
        uid = os.getuid()
        return subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(unit_path)],
            check=False,
            capture_output=True,
            text=True,
        )

    def disable(self, unit_path: Path) -> subprocess.CompletedProcess[str] | None:
        uid = os.getuid()
        return subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}", str(unit_path)],
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
        uid = os.getuid()
        result = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/afs.{definition.name}"],
            check=False,
            capture_output=True,
            text=True,
        )
        payload["enabled"] = result.returncode == 0
        payload["active"] = result.returncode == 0
        payload["detail"] = result.stdout.strip() or result.stderr.strip()
        return payload
