"""Dynamic agent lifecycle supervisor."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import load_config_model
from ..context_paths import resolve_agent_output_root
from ..profiles import resolve_active_profile
from ..schema import AFSConfig, AgentConfig
from .base import AgentResult, build_base_parser, configure_logging, emit_result, now_iso

DEFAULT_INTERVAL_SECONDS = 60

AGENT_NAME = "agent-supervisor"
AGENT_DESCRIPTION = (
    "Reconcile configured background agents using auto-start, interval schedules, "
    "and watch-path triggers."
)


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _path_contains(base: Path, candidate: Path) -> bool:
    try:
        return candidate == base or candidate.is_relative_to(base)
    except ValueError:
        return False


def _parse_schedule_interval(schedule: str) -> float | None:
    value = schedule.strip().lower()
    if not value:
        return None
    if value in {"hourly", "@hourly"}:
        return 3600.0
    if value in {"daily", "@daily"}:
        return 86400.0
    if value in {"weekly", "@weekly"}:
        return 7 * 86400.0

    multiplier = 1.0
    if value[-1:] in {"s", "m", "h", "d"}:
        suffix = value[-1]
        value = value[:-1]
        multiplier = {
            "s": 1.0,
            "m": 60.0,
            "h": 3600.0,
            "d": 86400.0,
        }[suffix]
    try:
        amount = float(value)
    except ValueError:
        return None
    return amount * multiplier if amount > 0 else None


def _watch_signature(path: Path) -> tuple[bool, int, int, int]:
    if not path.exists():
        return (False, 0, 0, 0)
    try:
        stat = path.stat()
    except OSError:
        return (False, 0, 0, 0)

    if path.is_file():
        return (True, stat.st_mtime_ns, stat.st_size, 1)

    newest_mtime = stat.st_mtime_ns
    total_size = 0
    file_count = 0
    for child in path.rglob("*"):
        try:
            child_stat = child.stat()
        except OSError:
            continue
        newest_mtime = max(newest_mtime, child_stat.st_mtime_ns)
        if child.is_file():
            total_size += child_stat.st_size
            file_count += 1
    return (True, newest_mtime, total_size, file_count)


@dataclass
class RunningAgent:
    name: str
    pid: int | None = None
    state: str = "stopped"  # stopped, running, failed, awaiting_review
    started_at: str = ""
    module: str = ""
    args: list[str] = field(default_factory=list)
    last_error: str = ""
    last_event: str = ""
    last_seen_at: str = ""
    stopped_at: str = ""
    launch_count: int = 0
    manually_stopped: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "pid": self.pid,
            "state": self.state,
            "started_at": self.started_at,
            "module": self.module,
            "args": list(self.args),
            "last_error": self.last_error,
            "last_event": self.last_event,
            "last_seen_at": self.last_seen_at,
            "stopped_at": self.stopped_at,
            "launch_count": self.launch_count,
            "manually_stopped": self.manually_stopped,
        }


class AgentSupervisor:
    STATE_DIR = Path.home() / ".config" / "afs" / "agents" / "state"

    def __init__(
        self,
        state_dir: Path | None = None,
        *,
        config: AFSConfig | None = None,
    ) -> None:
        self._config = config
        self._state_dir = self._resolve_state_dir(state_dir, config)
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_state_dir(
        self,
        state_dir: Path | None,
        config: AFSConfig | None,
    ) -> Path:
        if state_dir is not None:
            return state_dir.expanduser().resolve()
        env_value = os.environ.get("AFS_AGENT_STATE_DIR", "").strip()
        if env_value:
            return Path(env_value).expanduser().resolve()
        resolved_config = config or load_config_model(merge_user=True)
        return (
            resolve_agent_output_root(
                resolved_config.general.context_root,
                config=resolved_config,
            )
            / "supervisor"
        ).expanduser().resolve()

    def _python_executable(self) -> str:
        if self._config and self._config.general.python_executable:
            return str(self._config.general.python_executable)
        return sys.executable

    def _state_path(self, name: str) -> Path:
        return self._state_dir / f"{name}.json"

    def _write_state(self, agent: RunningAgent) -> None:
        self._state_path(agent.name).write_text(
            json.dumps(agent.to_dict()),
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
        args = data.get("args")
        if not isinstance(args, list):
            args = []
        return RunningAgent(
            name=data.get("name", name),
            pid=data.get("pid"),
            state=data.get("state", "stopped"),
            started_at=data.get("started_at", ""),
            module=data.get("module", ""),
            args=[str(item) for item in args if isinstance(item, str)],
            last_error=data.get("last_error", ""),
            last_event=data.get("last_event", ""),
            last_seen_at=data.get("last_seen_at", ""),
            stopped_at=data.get("stopped_at", ""),
            launch_count=int(data.get("launch_count", 0) or 0),
            manually_stopped=bool(data.get("manually_stopped", False)),
        )

    def _pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _refresh_state(self, agent: RunningAgent) -> RunningAgent:
        if agent.pid and not self._pid_alive(agent.pid):
            agent.state = "failed"
            agent.pid = None
            agent.last_seen_at = now_iso()
            if not agent.last_error:
                agent.last_error = "process exited"
            self._write_state(agent)
        return agent

    def _build_agent_env(
        self, name: str, agent_config: AgentConfig | None,
    ) -> dict[str, str] | None:
        """Build environment dict with sandbox vars if agent_config is set."""
        if agent_config is None:
            return None
        if (
            not agent_config.allowed_mounts
            and not agent_config.allowed_tools
            and not agent_config.workspace_isolated
        ):
            return None
        env = dict(os.environ)
        env["AFS_AGENT_NAME"] = name
        if agent_config.allowed_mounts:
            env["AFS_ALLOWED_MOUNTS"] = ",".join(agent_config.allowed_mounts)
        if agent_config.allowed_tools:
            env["AFS_ALLOWED_TOOLS"] = ",".join(agent_config.allowed_tools)
        if agent_config.workspace_isolated:
            env["AFS_WORKSPACE_ISOLATED"] = "1"
            env["AFS_PREFER_REPO_CONFIG"] = "1"
            env["AFS_PREFER_USER_CONFIG"] = "0"
        return env

    def _terminate_pid(self, pid: int, *, grace_seconds: float = 1.0) -> bool:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        except OSError:
            return False

        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if not self._pid_alive(pid):
                return True
            time.sleep(0.05)

        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except OSError:
            return False

        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if not self._pid_alive(pid):
                return True
            time.sleep(0.05)
        return not self._pid_alive(pid)

    def _quiesce_agent(self, agent: RunningAgent, *, reason: str) -> bool:
        if agent.pid is None:
            return True
        if not self._pid_alive(agent.pid):
            agent.pid = None
            return True
        if self._terminate_pid(agent.pid):
            agent.pid = None
            return True
        agent.last_error = f"failed to stop pid {agent.pid} for {reason}"
        agent.last_seen_at = now_iso()
        self._write_state(agent)
        return False

    def spawn(
        self,
        name: str,
        module: str,
        args: list[str] | None = None,
        *,
        reason: str = "",
        agent_config: AgentConfig | None = None,
    ) -> RunningAgent:
        existing = self.status(name)
        if existing and existing.state == "running":
            return existing

        cmd = [self._python_executable(), "-m", module] + (args or [])
        launch_count = (existing.launch_count if existing else 0) + 1
        started_at = _now_utc().isoformat()
        agent_env = self._build_agent_env(name, agent_config)
        try:
            proc = subprocess.Popen(
                cmd,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=agent_env,
            )
        except Exception as exc:
            agent = RunningAgent(
                name=name,
                state="failed",
                module=module,
                args=list(args or []),
                started_at=started_at,
                last_error=str(exc),
                last_event=reason,
                last_seen_at=started_at,
                launch_count=launch_count,
            )
            self._write_state(agent)
            raise RuntimeError(f"Failed to spawn agent {name}: {exc}") from exc

        agent = RunningAgent(
            name=name,
            pid=proc.pid,
            state="running",
            started_at=started_at,
            module=module,
            args=list(args or []),
            last_event=reason,
            last_seen_at=started_at,
            launch_count=launch_count,
        )
        self._write_state(agent)
        return agent

    def stop(self, name: str) -> bool:
        agent = self._read_state(name)
        if not agent:
            return False

        if not self._quiesce_agent(agent, reason="stop"):
            return False

        agent.state = "stopped"
        agent.pid = None
        agent.stopped_at = now_iso()
        agent.last_seen_at = agent.stopped_at
        agent.manually_stopped = True
        self._write_state(agent)
        return True

    def list_agents(self) -> list[RunningAgent]:
        agents: list[RunningAgent] = []
        if not self._state_dir.exists():
            return agents
        for path in sorted(self._state_dir.glob("*.json")):
            agent = self._read_state(path.stem)
            if agent is None:
                continue
            agents.append(self._refresh_state(agent))
        return agents

    def list_running(self) -> list[RunningAgent]:
        return [agent for agent in self.list_agents() if agent.state == "running"]

    def status(self, name: str) -> RunningAgent | None:
        agent = self._read_state(name)
        if agent is None:
            return None
        return self._refresh_state(agent)

    def set_awaiting_review(self, name: str) -> bool:
        """Transition agent to awaiting_review state."""
        agent = self._read_state(name)
        if agent is None:
            return False
        if not self._quiesce_agent(agent, reason="review"):
            return False
        agent.state = "awaiting_review"
        agent.last_event = "review_requested"
        agent.last_seen_at = now_iso()
        self._write_state(agent)
        return True

    def approve_review(self, name: str) -> bool:
        """Approve review, transition agent back to stopped."""
        agent = self._read_state(name)
        if agent is None or agent.state != "awaiting_review":
            return False
        if not self._quiesce_agent(agent, reason="review approval"):
            return False
        agent.state = "stopped"
        agent.pid = None
        agent.stopped_at = now_iso()
        agent.last_event = "review_approved"
        agent.last_seen_at = agent.stopped_at
        self._write_state(agent)
        return True

    def reject_review(self, name: str) -> bool:
        """Reject review, transition agent to failed."""
        agent = self._read_state(name)
        if agent is None or agent.state != "awaiting_review":
            return False
        if not self._quiesce_agent(agent, reason="review rejection"):
            return False
        agent.state = "failed"
        agent.pid = None
        agent.last_error = "review_rejected"
        agent.last_event = "review_rejected"
        agent.last_seen_at = now_iso()
        self._write_state(agent)
        return True

    def auto_start(self, agent_configs: list[AgentConfig]) -> list[RunningAgent]:
        started: list[RunningAgent] = []
        for config in agent_configs:
            if not config.auto_start or not config.module:
                continue
            existing = self.status(config.name)
            if existing and (existing.state in ("running", "awaiting_review") or existing.manually_stopped):
                continue
            try:
                started.append(
                    self.spawn(
                        config.name,
                        config.module,
                        reason="auto_start",
                        agent_config=config,
                    )
                )
            except RuntimeError:
                continue
        return started

    def evaluate_triggers(
        self,
        event: str,
        agent_configs: list[AgentConfig] | None = None,
    ) -> list[AgentConfig]:
        if not agent_configs:
            return []
        return self.evaluate_triggers_from(event, agent_configs)

    def evaluate_triggers_from(
        self,
        event: str,
        agent_configs: list[AgentConfig],
    ) -> list[AgentConfig]:
        matched: list[AgentConfig] = []
        for config in agent_configs:
            if event in config.triggers:
                matched.append(config)
        return matched

    def evaluate_watch_paths(
        self,
        changed_paths: list[Path],
        agent_configs: list[AgentConfig],
    ) -> list[AgentConfig]:
        matched: list[AgentConfig] = []
        resolved_changes = [path.expanduser().resolve() for path in changed_paths]
        for config in agent_configs:
            watch_paths = [path.expanduser().resolve() for path in config.watch_paths]
            if not watch_paths:
                continue
            for watch_path in watch_paths:
                if any(
                    _path_contains(watch_path, changed_path)
                    or _path_contains(changed_path, watch_path)
                    for changed_path in resolved_changes
                ):
                    matched.append(config)
                    break
        return matched

    def due_schedules(
        self,
        agent_configs: list[AgentConfig],
        *,
        now: datetime | None = None,
    ) -> list[AgentConfig]:
        due: list[AgentConfig] = []
        current = now or _now_utc()
        for config in agent_configs:
            interval_seconds = _parse_schedule_interval(config.schedule)
            if interval_seconds is None or not config.module:
                continue
            existing = self.status(config.name)
            if existing and (existing.state in ("running", "awaiting_review") or existing.manually_stopped):
                continue
            if existing is None:
                due.append(config)
                continue
            last_seen = (
                _parse_timestamp(existing.started_at)
                or _parse_timestamp(existing.stopped_at)
                or _parse_timestamp(existing.last_seen_at)
            )
            if last_seen is None:
                due.append(config)
                continue
            elapsed = (current - last_seen).total_seconds()
            if elapsed >= interval_seconds:
                due.append(config)
        return due

    def reconcile(
        self,
        agent_configs: list[AgentConfig],
        *,
        event: str | None = None,
        changed_paths: list[Path] | None = None,
        now: datetime | None = None,
    ) -> list[RunningAgent]:
        candidates: dict[str, tuple[AgentConfig, str]] = {}
        for config in agent_configs:
            if config.auto_start:
                candidates.setdefault(config.name, (config, "auto_start"))
        if event:
            for config in self.evaluate_triggers_from(event, agent_configs):
                candidates.setdefault(config.name, (config, event))
        if changed_paths:
            for config in self.evaluate_watch_paths(changed_paths, agent_configs):
                candidates.setdefault(config.name, (config, "file_watch"))
        for config in self.due_schedules(agent_configs, now=now):
            candidates.setdefault(config.name, (config, f"schedule:{config.schedule}"))

        started: list[RunningAgent] = []
        for config, reason in candidates.values():
            if not config.module:
                continue
            existing = self.status(config.name)
            if existing and (existing.state in ("running", "awaiting_review") or existing.manually_stopped):
                continue
            try:
                started.append(
                    self.spawn(
                        config.name,
                        config.module,
                        reason=reason,
                        agent_config=config,
                    )
                )
            except RuntimeError:
                continue
        return started

    def audit(self) -> dict[str, object]:
        agents = self.list_agents()
        counts = {
            "running": 0,
            "failed": 0,
            "stopped": 0,
            "manual_stop": 0,
            "configured": len(agents),
        }
        stale_pid_files: list[str] = []
        for agent in agents:
            counts.setdefault(agent.state, 0)
            if agent.state == "running":
                counts["running"] += 1
            elif agent.state == "failed":
                counts["failed"] += 1
                stale_pid_files.append(agent.name)
            else:
                counts["stopped"] += 1
            if agent.manually_stopped:
                counts["manual_stop"] += 1
        return {
            "state_dir": str(self._state_dir),
            "counts": counts,
            "stale_pid_files": stale_pid_files,
            "agents": [agent.to_dict() for agent in agents],
        }


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser("Reconcile configured background agents.")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help=f"Seconds between reconciliation runs (default: {DEFAULT_INTERVAL_SECONDS}).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Maximum runs when interval > 0 (0 = unlimited).",
    )
    parser.add_argument(
        "--sleep-first",
        action="store_true",
        help="Sleep for the interval before the first reconciliation run.",
    )
    return parser


def _snapshot_watch_paths(agent_configs: list[AgentConfig]) -> dict[Path, tuple[bool, int, int, int]]:
    paths: dict[Path, tuple[bool, int, int, int]] = {}
    for config in agent_configs:
        for watch_path in config.watch_paths:
            resolved = watch_path.expanduser().resolve()
            if resolved in paths:
                continue
            paths[resolved] = _watch_signature(resolved)
    return paths


def _diff_watch_paths(
    previous: dict[Path, tuple[bool, int, int, int]],
    current: dict[Path, tuple[bool, int, int, int]],
) -> list[Path]:
    changed: list[Path] = []
    keys = set(previous) | set(current)
    for path in sorted(keys):
        if previous.get(path) != current.get(path):
            changed.append(path)
    return changed


def _run_once(
    args: argparse.Namespace,
    previous_watch_state: dict[Path, tuple[bool, int, int, int]],
    *,
    first_run: bool,
) -> tuple[AgentResult, dict[Path, tuple[bool, int, int, int]]]:
    started_at = now_iso()
    start = time.time()
    config = load_config_model(
        config_path=Path(args.config).expanduser() if args.config else None,
        merge_user=True,
    )
    profile = resolve_active_profile(config)
    supervisor = AgentSupervisor(config=config)
    current_watch_state = _snapshot_watch_paths(profile.agent_configs)
    changed_paths = (
        _diff_watch_paths(previous_watch_state, current_watch_state)
        if previous_watch_state
        else []
    )
    started_agents = supervisor.reconcile(
        profile.agent_configs,
        event="on_boot" if first_run else None,
        changed_paths=changed_paths,
        now=_now_utc(),
    )
    audit = supervisor.audit()
    notes: list[str] = []
    if changed_paths:
        notes.append("watch-path changes detected")
    if first_run:
        notes.append("boot reconciliation complete")
    result = AgentResult(
        name=AGENT_NAME,
        status="ok",
        started_at=started_at,
        finished_at=now_iso(),
        duration_seconds=time.time() - start,
        metrics={
            "configured_agents": len(profile.agent_configs),
            "started_agents": len(started_agents),
            "changed_watch_paths": len(changed_paths),
            "running_agents": int(audit["counts"]["running"]),
            "failed_agents": int(audit["counts"]["failed"]),
        },
        notes=notes,
        payload={
            "profile": profile.name,
            "started_agents": [agent.name for agent in started_agents],
            "changed_watch_paths": [str(path) for path in changed_paths],
            "audit": audit,
        },
    )
    return result, current_watch_state


def run(args: argparse.Namespace) -> int:
    configure_logging(args.quiet)
    watch_state: dict[Path, tuple[bool, int, int, int]] = {}
    runs = 0

    while True:
        if runs == 0 and args.sleep_first and args.interval > 0:
            time.sleep(args.interval)

        result, watch_state = _run_once(args, watch_state, first_run=runs == 0)
        emit_result(
            result,
            output_path=Path(args.output).expanduser() if args.output else None,
            force_stdout=bool(args.stdout),
            pretty=bool(args.pretty),
        )
        runs += 1

        if args.interval <= 0:
            break
        if args.max_runs and runs >= args.max_runs:
            break
        time.sleep(args.interval)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
