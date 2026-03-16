"""Dynamic agent lifecycle supervisor."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..schema import AgentConfig


@dataclass
class RunningAgent:
    name: str
    pid: int | None = None
    state: str = "stopped"  # stopped, running, failed
    started_at: str = ""
    module: str = ""


class AgentSupervisor:
    STATE_DIR = Path.home() / ".config" / "afs" / "agents" / "state"

    def __init__(self, state_dir: Path | None = None) -> None:
        self._state_dir = state_dir or self.STATE_DIR
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def _state_path(self, name: str) -> Path:
        return self._state_dir / f"{name}.json"

    def _write_state(self, agent: RunningAgent) -> None:
        self._state_path(agent.name).write_text(
            json.dumps({
                "name": agent.name,
                "pid": agent.pid,
                "state": agent.state,
                "started_at": agent.started_at,
                "module": agent.module,
            }),
            encoding="utf-8",
        )

    def _read_state(self, name: str) -> RunningAgent | None:
        path = self._state_path(name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return RunningAgent(
            name=data.get("name", name),
            pid=data.get("pid"),
            state=data.get("state", "stopped"),
            started_at=data.get("started_at", ""),
            module=data.get("module", ""),
        )

    def _pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def spawn(
        self,
        name: str,
        module: str,
        args: list[str] | None = None,
    ) -> RunningAgent:
        existing = self.status(name)
        if existing and existing.state == "running":
            return existing

        cmd = [sys.executable, "-m", module] + (args or [])
        try:
            proc = subprocess.Popen(
                cmd,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            agent = RunningAgent(
                name=name,
                state="failed",
                module=module,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            self._write_state(agent)
            raise RuntimeError(f"Failed to spawn agent {name}: {exc}") from exc

        agent = RunningAgent(
            name=name,
            pid=proc.pid,
            state="running",
            started_at=datetime.now(timezone.utc).isoformat(),
            module=module,
        )
        self._write_state(agent)
        return agent

    def stop(self, name: str) -> bool:
        agent = self._read_state(name)
        if not agent or not agent.pid:
            self._state_path(name).unlink(missing_ok=True)
            return False

        if self._pid_alive(agent.pid):
            try:
                os.kill(agent.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass

        agent.state = "stopped"
        agent.pid = None
        self._state_path(name).unlink(missing_ok=True)
        return True

    def list_running(self) -> list[RunningAgent]:
        agents: list[RunningAgent] = []
        if not self._state_dir.exists():
            return agents
        for path in sorted(self._state_dir.glob("*.json")):
            agent = self._read_state(path.stem)
            if agent is None:
                continue
            if agent.pid and not self._pid_alive(agent.pid):
                agent.state = "failed"
                self._write_state(agent)
            agents.append(agent)
        return agents

    def status(self, name: str) -> RunningAgent | None:
        agent = self._read_state(name)
        if agent is None:
            return None
        if agent.pid and not self._pid_alive(agent.pid):
            agent.state = "failed"
            self._write_state(agent)
        return agent

    def auto_start(self, agent_configs: list[AgentConfig]) -> list[RunningAgent]:
        started: list[RunningAgent] = []
        for config in agent_configs:
            if not config.auto_start:
                continue
            if not config.module:
                continue
            try:
                agent = self.spawn(config.name, config.module)
                started.append(agent)
            except RuntimeError:
                continue
        return started

    def evaluate_triggers(self, event: str) -> list[AgentConfig]:
        """Return agent configs whose triggers match the given event.

        This is a static helper — call it with the configs from a resolved
        profile, not with running state.  Kept here for API cohesion.
        """
        return []  # Placeholder; callers pass their own config lists.

    def evaluate_triggers_from(
        self, event: str, agent_configs: list[AgentConfig]
    ) -> list[AgentConfig]:
        matched: list[AgentConfig] = []
        for config in agent_configs:
            if event in config.triggers:
                matched.append(config)
        return matched
