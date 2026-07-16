"""Dynamic agent lifecycle supervisor."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..agent_registry import (
    AGENT_CONTEXT_ROOT_ENV,
    AGENT_EXPECTED_RESULT_NAME_ENV,
    AGENT_NAME_ENV,
    AGENT_RUN_ID_ENV,
    AGENT_SUPERVISED_ENV,
)
from ..config import load_config_model
from ..context_index import INDEX_SCAN_SKIP_NAMES
from ..context_paths import resolve_agent_output_root
from ..profiles import resolve_active_profile
from ..schema import AFSConfig, AgentConfig
from .base import AgentResult, build_base_parser, configure_logging, emit_result, now_iso
from .event_reactor import (
    DEFAULT_EVENT_DEBOUNCE_SECONDS,
    VALID_EVENT_ACTIONS,
    ReactorBatch,
    ReactorBusyError,
    ReactorEvent,
    match_event_rules,
    open_event_batch,
)

_log = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 60
WATCH_SIGNATURE_MAX_ENTRIES = 50_000
WATCH_SIGNATURE_TIMEOUT_SECONDS = 5.0
WatchSignature = tuple[bool, int, int, int, bool]

# Keep handles process-wide because ``_run_once`` constructs a fresh supervisor
# while the daemon loop remains in the same Python process. Polling these handles
# both detects completion and reaps short-lived one-shot children.
_OWNED_PROCESSES: dict[int, subprocess.Popen[Any]] = {}
_INCOMPLETE_WATCH_WARNED: set[Path] = set()

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


def _watch_signature(
    path: Path,
    *,
    max_entries: int = WATCH_SIGNATURE_MAX_ENTRIES,
    timeout_seconds: float = WATCH_SIGNATURE_TIMEOUT_SECONDS,
) -> WatchSignature:
    if not path.exists():
        return (False, 0, 0, 0, False)
    try:
        stat = path.stat()
    except OSError:
        return (False, 0, 0, 0, True)

    if path.is_file():
        return (True, stat.st_mtime_ns, stat.st_size, 1, False)

    newest_mtime = stat.st_mtime_ns
    total_size = 0
    file_count = 0
    entries_seen = 0
    incomplete = False
    pending = [path]
    visited_directories: set[tuple[int, int]] = set()
    deadline = time.monotonic() + max(timeout_seconds, 0.0)

    # AFS mounts are commonly directory symlinks. Path.rglob() does not walk
    # those targets, so use an inode-guarded traversal that follows them while
    # avoiding cycles and duplicate targets.
    while pending:
        if time.monotonic() >= deadline:
            incomplete = True
            break
        directory = pending.pop()
        try:
            directory_stat = directory.stat()
        except OSError:
            incomplete = True
            continue
        identity = (directory_stat.st_dev, directory_stat.st_ino)
        if identity in visited_directories:
            continue
        visited_directories.add(identity)
        newest_mtime = max(newest_mtime, directory_stat.st_mtime_ns)
        children: list[Path] = []
        child_iterator = None
        try:
            child_iterator = directory.iterdir()
            for child in child_iterator:
                if time.monotonic() >= deadline:
                    incomplete = True
                    break
                if child.name in INDEX_SCAN_SKIP_NAMES:
                    continue
                if entries_seen >= max(max_entries, 0):
                    incomplete = True
                    break
                entries_seen += 1
                children.append(child)
        except OSError:
            incomplete = True
        finally:
            close_iterator = getattr(child_iterator, "close", None)
            if callable(close_iterator):
                close_iterator()

        # Only the bounded, observed prefix is sorted. This keeps traversal
        # stable without eagerly materializing an arbitrarily large directory.
        children.sort(key=lambda child: child.name)

        child_directories: list[Path] = []
        for child in children:
            if time.monotonic() >= deadline:
                incomplete = True
                break
            try:
                child_stat = child.stat()
                is_directory = child.is_dir()
            except OSError:
                incomplete = True
                continue
            newest_mtime = max(newest_mtime, child_stat.st_mtime_ns)
            if is_directory:
                child_directories.append(child)
            else:
                total_size += child_stat.st_size
                file_count += 1
        if incomplete:
            break
        # Reverse before stack insertion so traversal remains lexical and the
        # capped prefix is deterministic across runs.
        pending.extend(reversed(child_directories))
    return (True, newest_mtime, total_size, file_count, incomplete)


@dataclass
class RunningAgent:
    name: str
    pid: int | None = None
    state: str = "stopped"  # stopped, running, failed, awaiting_review, circuit_open
    started_at: str = ""
    module: str = ""
    args: list[str] = field(default_factory=list)
    session_id: str = ""
    run_id: str = ""
    launch_reason: str = ""
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
            "session_id": self.session_id,
            "run_id": self.run_id,
            "launch_reason": self.launch_reason,
            "last_error": self.last_error,
            "last_event": self.last_event,
            "last_seen_at": self.last_seen_at,
            "stopped_at": self.stopped_at,
            "launch_count": self.launch_count,
            "manually_stopped": self.manually_stopped,
        }


class AgentSupervisor:
    STATE_DIR = Path.home() / ".config" / "afs" / "agents" / "state"

    # Restart backoff parameters
    RESTART_BASE_DELAY: float = 30.0  # seconds
    RESTART_MAX_DELAY: float = 300.0  # seconds
    CIRCUIT_COOLDOWN: float = 3600.0  # 1 hour

    def __init__(
        self,
        state_dir: Path | None = None,
        *,
        config: AFSConfig | None = None,
        config_path: Path | None = None,
    ) -> None:
        self._config = config
        self._config_path = (
            config_path.expanduser().resolve() if config_path is not None else None
        )
        self._state_dir = self._resolve_state_dir(state_dir, config)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        # Per-agent failure tracking for restart-with-backoff
        self._failure_counts: dict[str, int] = {}
        self._last_failure_at: dict[str, float] = {}
        # Circuit breaker: agents that have exhausted max_restarts
        self._circuit_opened_at: dict[str, float] = {}

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

    def _context_root_str(self) -> str:
        resolved_config = self._config or load_config_model(merge_user=True)
        return str(resolved_config.general.context_root.expanduser().resolve())

    def _context_root_path(self) -> Path:
        resolved_config = self._config or load_config_model(merge_user=True)
        return resolved_config.general.context_root.expanduser().resolve()

    def _log_lifecycle(
        self,
        agent_name: str,
        op: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            from ..history import log_agent_lifecycle

            log_agent_lifecycle(
                agent_name,
                op,
                metadata=metadata or {},
                context_root=self._context_root_path(),
            )
        except Exception:
            pass

    def _resolve_task(self, name: str, module: str, agent_config: AgentConfig | None) -> str:
        from ..agent_registry import resolve_agent_task

        description = agent_config.description if agent_config is not None else ""
        return resolve_agent_task(name, module=module, description=description)

    def _update_registry(
        self,
        agent: RunningAgent,
        *,
        task: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        from ..agent_registry import ACTIVE_AGENT_STATES, AgentRegistry

        AgentRegistry().update(
            name=agent.name,
            status=agent.state,
            task=task,
            module=agent.module,
            context_root=self._context_root_str(),
            started_at=agent.started_at,
            finished_at=agent.stopped_at if agent.state not in ACTIVE_AGENT_STATES else "",
            last_output_at=agent.last_seen_at,
            last_error=agent.last_error,
            pid=agent.pid,
            run_id=agent.run_id,
            metadata=metadata,
        )

    def _registry_completion(self, agent: RunningAgent) -> dict[str, Any] | None:
        from ..agent_registry import ACTIVE_AGENT_STATES, AgentRegistry

        registry = AgentRegistry()
        entry = registry.get(agent.name, context_root=self._context_root_str())
        if entry is None and not agent.run_id:
            # Legacy/unsupervised agents did not publish a context or run ID.
            # Keep reading those records while requiring exact scoping for all
            # newly supervised launches.
            entry = registry.get(agent.name)
        if entry is None:
            return None
        status = str(entry.get("status", ""))
        if status in ACTIVE_AGENT_STATES:
            return None
        entry_run_id = str(entry.get("run_id", "") or "")
        if agent.run_id and entry_run_id != agent.run_id:
            return None
        entry_output = _parse_timestamp(str(entry.get("last_output_at", "") or ""))
        agent_started = _parse_timestamp(agent.started_at)
        if entry_output is None:
            return None
        if agent_started is not None and entry_output < agent_started:
            return None
        return entry

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
            session_id=str(data.get("session_id", "") or ""),
            run_id=str(data.get("run_id", "") or ""),
            launch_reason=str(data.get("launch_reason", "") or ""),
            last_error=data.get("last_error", ""),
            last_event=data.get("last_event", ""),
            last_seen_at=data.get("last_seen_at", ""),
            stopped_at=data.get("stopped_at", ""),
            launch_count=int(data.get("launch_count", 0) or 0),
            manually_stopped=bool(data.get("manually_stopped", False)),
        )

    def _pid_alive(self, pid: int) -> bool:
        owned = _OWNED_PROCESSES.get(pid)
        if owned is not None:
            try:
                return_code = owned.poll()
            except OSError:
                _OWNED_PROCESSES.pop(pid, None)
            else:
                if return_code is None:
                    return True
                _OWNED_PROCESSES.pop(pid, None)
                return False
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _get_agent_config(self, name: str) -> AgentConfig | None:
        """Look up the AgentConfig for *name* from the active profile."""
        try:
            config = self._config or load_config_model(merge_user=True)
            profile = resolve_active_profile(config)
            for ac in profile.agent_configs:
                if ac.name == name:
                    return ac
        except Exception:
            pass
        return None

    def _record_failure(self, name: str) -> int:
        """Increment failure counter for *name* and return the new count."""
        self._failure_counts[name] = self._failure_counts.get(name, 0) + 1
        self._last_failure_at[name] = time.monotonic()
        return self._failure_counts[name]

    def _reset_failure(self, name: str) -> None:
        """Reset failure tracking for *name* (e.g. after successful completion)."""
        self._failure_counts.pop(name, None)
        self._last_failure_at.pop(name, None)
        self._circuit_opened_at.pop(name, None)

    def _backoff_delay(self, failure_count: int) -> float:
        """Calculate exponential backoff: min(base * 2^(n-1), max_delay)."""
        return min(
            self.RESTART_BASE_DELAY * (2 ** (failure_count - 1)),
            self.RESTART_MAX_DELAY,
        )

    def _should_restart(self, name: str, agent_config: AgentConfig) -> bool:
        """Decide whether a failed agent should be automatically restarted.

        Checks ``restart_on_failure``, failure budget, circuit breaker cooldown,
        and backoff timing.  Returns True only when all conditions are met.
        """
        if not agent_config.restart_on_failure:
            return False

        now = time.monotonic()
        failure_count = self._failure_counts.get(name, 0)

        # Circuit breaker: already exhausted restarts?
        if name in self._circuit_opened_at:
            opened = self._circuit_opened_at[name]
            if now - opened < self.CIRCUIT_COOLDOWN:
                return False
            # Cooldown elapsed -- reset and allow a fresh round of retries.
            _log.warning(
                "Agent %s: circuit breaker cooldown elapsed, resetting failure state",
                name,
            )
            self._reset_failure(name)
            return True

        # Budget check
        if failure_count >= agent_config.max_restarts:
            _log.warning(
                "Agent %s: max restarts (%d) exhausted, opening circuit breaker",
                name,
                agent_config.max_restarts,
            )
            self._circuit_opened_at[name] = now
            return False

        # Backoff: enough time since last failure?
        last_failure = self._last_failure_at.get(name)
        if last_failure is not None:
            required_delay = self._backoff_delay(failure_count)
            elapsed = now - last_failure
            if elapsed < required_delay:
                return False

        return True

    def _attempt_restart(self, agent: RunningAgent) -> RunningAgent | None:
        """Try to restart a failed agent with backoff.

        Returns the new RunningAgent if restarted, None otherwise.
        """
        agent_config = self._get_agent_config(agent.name)
        if agent_config is None or not agent_config.module:
            return None

        if not self._should_restart(agent.name, agent_config):
            return None

        failure_count = self._failure_counts.get(agent.name, 0)
        delay = self._backoff_delay(failure_count) if failure_count > 0 else 0
        _log.warning(
            "Agent %s: automatic restart attempt %d/%d (backoff=%.0fs)",
            agent.name,
            failure_count + 1,
            agent_config.max_restarts,
            delay,
        )

        try:
            # Clear manually_stopped so spawn proceeds.
            agent.manually_stopped = False
            self._write_state(agent)
            restarted = self.spawn(
                agent.name,
                agent_config.module,
                args=agent.args if agent.args else None,
                reason=f"auto_restart (attempt {failure_count + 1})",
                agent_config=agent_config,
            )
            return restarted
        except RuntimeError:
            _log.warning(
                "Agent %s: restart spawn failed",
                agent.name,
                exc_info=True,
            )
            return None

    def _validate_agent_result(self, result: AgentResult) -> None:
        """Run validation on an AgentResult and log warnings for any issues."""
        errors = result.validate()
        if errors:
            _log.warning(
                "Agent result validation warnings for %s: %s",
                result.name,
                "; ".join(errors),
            )

    def _refresh_state(self, agent: RunningAgent) -> RunningAgent:
        if agent.pid and not self._pid_alive(agent.pid):
            previous_state = agent.state
            completion = self._registry_completion(agent)
            if completion is not None:
                completion_status = str(completion.get("status", ""))
                agent.state = "failed" if completion_status in {"error", "failed"} else "stopped"
                agent.pid = None
                agent.stopped_at = str(completion.get("finished_at", "") or now_iso())
                agent.last_seen_at = str(
                    completion.get("last_output_at", "")
                    or completion.get("updated_at", "")
                    or agent.stopped_at
                )
                agent.last_error = str(completion.get("last_error", "") or "")
                self._write_state(agent)
                self._update_registry(
                    agent,
                    task=str(completion.get("task", "") or ""),
                    metadata=completion.get("metadata")
                    if isinstance(completion.get("metadata"), dict)
                    else None,
                )
                if previous_state == "running":
                    if agent.state == "failed":
                        self._record_failure(agent.name)
                        self._log_lifecycle(
                            agent.name,
                            "failed",
                            metadata={
                                "session_id": agent.session_id,
                                "launch_reason": agent.launch_reason,
                                "state": agent.state,
                                "error": agent.last_error,
                                "failure_count": self._failure_counts.get(agent.name, 0),
                            },
                        )
                        # Attempt automatic restart
                        restarted = self._attempt_restart(agent)
                        if restarted is not None:
                            return restarted
                        # If restart was not attempted or failed, check circuit breaker
                        ac = self._get_agent_config(agent.name)
                        if (
                            ac is not None
                            and ac.restart_on_failure
                            and agent.name in self._circuit_opened_at
                        ):
                            agent.state = "circuit_open"
                            self._write_state(agent)
                            self._update_registry(agent)
                    else:
                        # Successful completion -- reset failure tracking
                        self._reset_failure(agent.name)
                        self._log_lifecycle(
                            agent.name,
                            "completed",
                            metadata={
                                "session_id": agent.session_id,
                                "launch_reason": agent.launch_reason,
                                "state": agent.state,
                                "error": agent.last_error,
                            },
                        )
                return agent

            # A clean one-shot must emit an AgentResult completion record.
            # Without one, fail closed: the process may have crashed before
            # reporting and must not be postponed until its next schedule.
            agent.state = "failed"
            agent.pid = None
            agent.last_seen_at = now_iso()
            if not agent.last_error:
                agent.last_error = "process exited without a completion record"
            self._write_state(agent)
            self._update_registry(agent)
            if previous_state == "running":
                if agent.state == "failed":
                    self._record_failure(agent.name)
                    self._log_lifecycle(
                        agent.name,
                        "failed",
                        metadata={
                            "session_id": agent.session_id,
                            "launch_reason": agent.launch_reason,
                            "state": agent.state,
                            "error": agent.last_error,
                            "failure_count": self._failure_counts.get(agent.name, 0),
                        },
                    )
                    # Attempt automatic restart
                    restarted = self._attempt_restart(agent)
                    if restarted is not None:
                        return restarted
                    # If restart was not attempted or failed, check circuit breaker
                    ac = self._get_agent_config(agent.name)
                    if (
                        ac is not None
                        and ac.restart_on_failure
                        and agent.name in self._circuit_opened_at
                    ):
                        agent.state = "circuit_open"
                        self._write_state(agent)
                        self._update_registry(agent)
                else:
                    self._reset_failure(agent.name)
                    self._log_lifecycle(
                        agent.name,
                        "completed",
                        metadata={
                            "session_id": agent.session_id,
                            "launch_reason": agent.launch_reason,
                            "state": agent.state,
                            "error": agent.last_error,
                        },
                    )

        return agent

    def _check_dependencies(
        self,
        agent_name: str,
        config: AgentConfig,
        all_configs: list[AgentConfig],
    ) -> tuple[bool, str]:
        """Check whether *agent_name* is ready to run.

        Verifies two opt-in constraints:

        1. **depends_on** -- every named agent must have completed
           successfully (state == "stopped" with no error).
        2. **mutex_group** -- no other agent in the same group may be
           currently running.

        Returns ``(True, "")`` when all constraints are satisfied, or
        ``(False, reason)`` with a human-readable explanation otherwise.
        """
        # --- depends_on ---
        for dep_name in config.depends_on:
            dep_status = self.status(dep_name)
            if dep_status is None:
                return False, f"dependency '{dep_name}' has never run"
            if dep_status.state == "running":
                return False, f"dependency '{dep_name}' is still running"
            if dep_status.state in ("failed", "circuit_open"):
                return False, f"dependency '{dep_name}' is in state '{dep_status.state}'"
            if dep_status.state == "awaiting_review":
                return False, f"dependency '{dep_name}' is awaiting review"
            # "stopped" is the successful terminal state

        # --- mutex_group ---
        if config.mutex_group:
            group_members = [
                c for c in all_configs
                if c.mutex_group == config.mutex_group and c.name != agent_name
            ]
            for member in group_members:
                member_status = self.status(member.name)
                if member_status is not None and member_status.state == "running":
                    return (
                        False,
                        f"mutex group '{config.mutex_group}': "
                        f"agent '{member.name}' is already running",
                    )

        return True, ""

    def _process_handoff_targets(
        self,
        completed_agent_name: str,
        agent_configs: list[AgentConfig],
    ) -> list[RunningAgent]:
        """After an agent completes, check for targeted handoffs and spawn targets.

        Scans the handoff store for packets whose ``target_agent`` matches a
        configured agent.  If found, acknowledges the handoff and spawns the
        target (subject to normal dependency / mutex checks).
        """
        started: list[RunningAgent] = []
        try:
            from ..handoff import HandoffStore

            context_root = self._context_root_path()
            store = HandoffStore(context_root, config=self._config)
        except Exception:
            return started

        for config in agent_configs:
            pending = store.pending_for_agent(config.name)
            if not pending:
                continue
            # Only act on handoffs created by the agent that just completed
            relevant = [p for p in pending if p.agent_name == completed_agent_name]
            if not relevant:
                continue

            existing = self.status(config.name)
            if existing and (
                existing.state in ("running", "awaiting_review")
                or existing.manually_stopped
            ):
                continue

            if not config.module:
                continue

            ready, reason = self._check_dependencies(
                config.name, config, agent_configs,
            )
            if not ready:
                _log.info(
                    "Handoff target '%s' not ready: %s", config.name, reason,
                )
                continue

            # Acknowledge all relevant handoffs for this target
            for packet in relevant:
                store.acknowledge(packet.session_id, config.name)

            try:
                agent = self.spawn(
                    config.name,
                    config.module,
                    reason=f"handoff from {completed_agent_name}",
                    agent_config=config,
                )
                started.append(agent)
            except RuntimeError:
                continue

        return started

    def _build_agent_env(
        self,
        name: str,
        agent_config: AgentConfig | None,
        *,
        expected_result_name: str = "",
        run_id: str = "",
    ) -> dict[str, str] | None:
        """Build environment dict with sandbox vars and context snapshot."""
        # Always build env so we can inject context snapshot
        env = dict(os.environ)
        env[AGENT_NAME_ENV] = name
        env[AGENT_EXPECTED_RESULT_NAME_ENV] = expected_result_name or name
        env[AGENT_CONTEXT_ROOT_ENV] = self._context_root_str()
        if run_id:
            env[AGENT_RUN_ID_ENV] = run_id
            env[AGENT_SUPERVISED_ENV] = "1"
        else:
            # Private callers predating run IDs remain unsupervised and must
            # not accidentally inherit a parent agent's run scope.
            env.pop(AGENT_RUN_ID_ENV, None)
            env.pop(AGENT_SUPERVISED_ENV, None)
        if self._config_path is not None:
            env["AFS_CONFIG_PATH"] = str(self._config_path)

        # Inject context snapshot so the agent starts with index/memory/event awareness
        try:
            from ..agent_context import (
                AGENT_CONTEXT_ENV,
                build_agent_context_snapshot,
                write_agent_context_snapshot,
            )

            context_root = self._context_root_path()
            snapshot = build_agent_context_snapshot(
                name, context_root, config=self._config,
            )
            snapshot_path = write_agent_context_snapshot(
                snapshot, self._state_dir / "context_snapshots",
            )
            env[AGENT_CONTEXT_ENV] = str(snapshot_path)
        except Exception:
            _log.debug("Failed to build context snapshot for %s", name, exc_info=True)

        if agent_config is None:
            return env

        agent_spec = None
        try:
            from . import get_agent
            agent_spec = get_agent(name)
            if agent_spec and agent_spec.capabilities:
                pass
        except Exception:
            pass
        if agent_config.allowed_mounts:
            env["AFS_ALLOWED_MOUNTS"] = ",".join(agent_config.allowed_mounts)
        if agent_config.allowed_tools:
            env["AFS_ALLOWED_TOOLS"] = ",".join(agent_config.allowed_tools)
        if agent_config.workspace_isolated:
            env["AFS_WORKSPACE_ISOLATED"] = "1"
            env["AFS_PREFER_REPO_CONFIG"] = "1"
            env["AFS_PREFER_USER_CONFIG"] = "0"
        if agent_spec and agent_spec.capabilities:
            if "AFS_ALLOWED_MOUNTS" not in env and agent_spec.capabilities.mount_types:
                env["AFS_ALLOWED_MOUNTS"] = ",".join(agent_spec.capabilities.mount_types)
            if "AFS_ALLOWED_TOOLS" not in env and agent_spec.capabilities.tools:
                env["AFS_ALLOWED_TOOLS"] = ",".join(agent_spec.capabilities.tools)
        return env

    def _expected_result_name(self, name: str, module: str) -> str:
        """Resolve the canonical result name while preserving profile aliases."""
        try:
            loaded = importlib.import_module(module)
        except Exception:
            return name
        result_name = getattr(loaded, "AGENT_NAME", "")
        if isinstance(result_name, str) and result_name.strip():
            return result_name.strip()
        return name

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
        session_id = os.environ.get("AFS_SESSION_ID", "").strip()
        run_id = uuid4().hex
        agent_env = self._build_agent_env(
            name,
            agent_config,
            expected_result_name=self._expected_result_name(name, module),
            run_id=run_id,
        )
        agent = RunningAgent(
            name=name,
            state="running",
            module=module,
            args=list(args or []),
            session_id=session_id,
            run_id=run_id,
            launch_reason=reason,
            started_at=started_at,
            last_event=reason,
            last_seen_at=started_at,
            launch_count=launch_count,
        )
        metadata = {"launch_reason": reason, "run_id": run_id}
        if session_id:
            metadata["session_id"] = session_id
        # Establish the current context+run row before the child can emit a
        # terminal result. A later PID update for this same run is forbidden
        # from regressing a terminal status back to running.
        self._update_registry(
            agent,
            task=self._resolve_task(name, module, agent_config),
            metadata=metadata,
        )
        try:
            proc = subprocess.Popen(
                cmd,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=agent_env,
            )
        except Exception as exc:
            agent.state = "failed"
            agent.last_error = str(exc)
            self._write_state(agent)
            self._update_registry(
                agent,
                task=self._resolve_task(name, module, agent_config),
                metadata=metadata,
            )
            self._log_lifecycle(
                name,
                "spawn_failed",
                metadata={
                    "session_id": session_id,
                    "run_id": run_id,
                    "launch_reason": reason,
                    "module": module,
                    "error": str(exc),
                },
            )
            raise RuntimeError(f"Failed to spawn agent {name}: {exc}") from exc

        agent.pid = proc.pid
        if callable(getattr(proc, "poll", None)):
            _OWNED_PROCESSES[proc.pid] = proc
        self._write_state(agent)
        self._update_registry(
            agent,
            task=self._resolve_task(name, module, agent_config),
            metadata=metadata,
        )
        self._log_lifecycle(
            name,
            "spawned",
            metadata={
                "session_id": session_id,
                "run_id": run_id,
                "launch_reason": reason,
                "module": module,
                "pid": proc.pid,
            },
        )
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
        self._update_registry(agent)
        self._log_lifecycle(
            agent.name,
            "stopped",
            metadata={
                "session_id": agent.session_id,
                "launch_reason": agent.launch_reason,
                "pid": agent.pid,
            },
        )
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
        self._update_registry(agent)
        self._log_lifecycle(
            agent.name,
            "awaiting_review",
            metadata={
                "session_id": agent.session_id,
                "launch_reason": agent.launch_reason,
            },
        )
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
        self._update_registry(agent)
        self._log_lifecycle(
            agent.name,
            "review_approved",
            metadata={
                "session_id": agent.session_id,
                "launch_reason": agent.launch_reason,
            },
        )
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
        self._update_registry(agent)
        self._log_lifecycle(
            agent.name,
            "review_rejected",
            metadata={
                "session_id": agent.session_id,
                "launch_reason": agent.launch_reason,
            },
        )
        return True

    def auto_start(self, agent_configs: list[AgentConfig]) -> list[RunningAgent]:
        started: list[RunningAgent] = []
        for config in agent_configs:
            if not config.auto_start or not config.module:
                continue
            existing = self.status(config.name)
            if existing and (
                existing.state in ("running", "awaiting_review", "circuit_open")
                or existing.manually_stopped
            ):
                continue
            ready, reason = self._check_dependencies(
                config.name, config, agent_configs,
            )
            if not ready:
                _log.info(
                    "Skipping auto_start for '%s': %s", config.name, reason,
                )
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
            if existing and (
                existing.state in ("running", "awaiting_review", "circuit_open")
                or existing.manually_stopped
            ):
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

    def _event_debounce_seconds(self, config: AgentConfig) -> float:
        if config.event_debounce:
            parsed = _parse_schedule_interval(config.event_debounce)
            if parsed is not None:
                return parsed
        return DEFAULT_EVENT_DEBOUNCE_SECONDS

    def evaluate_event_records(
        self,
        events: list[ReactorEvent],
        agent_configs: list[AgentConfig],
        *,
        now: datetime | None = None,
        batch: ReactorBatch | None = None,
    ) -> list[tuple[AgentConfig, str]]:
        """Match reactor events against on_event rules with per-agent debounce.

        Configs with an unknown ``on_event_action`` are skipped entirely (fail
        closed): a typo must never fall through to a spawn. Debounce checks
        the persisted per-agent dispatch time from the reactor state first —
        job actions never start an agent, so the agent's own start time cannot
        debounce them — and the agent's last start as a fallback.
        """
        current = now or _now_utc()
        results: list[tuple[AgentConfig, str]] = []
        for config, reason in match_event_rules(events, agent_configs):
            if config.on_event_action not in VALID_EVENT_ACTIONS:
                _log.warning(
                    "Agent '%s' has invalid on_event_action %r "
                    "(valid: %s); skipping its event trigger",
                    config.name,
                    config.on_event_action,
                    ", ".join(VALID_EVENT_ACTIONS),
                )
                continue
            debounce = self._event_debounce_seconds(config)
            dispatched = batch.last_dispatch(config.name) if batch else None
            if dispatched and (current - dispatched).total_seconds() < debounce:
                continue
            existing = self.status(config.name)
            if existing:
                last_start = _parse_timestamp(existing.started_at)
                if (
                    last_start
                    and (current - last_start).total_seconds() < debounce
                ):
                    continue
            results.append((config, reason))
        return results

    def enqueue_event_jobs(
        self,
        batch: ReactorBatch,
        agent_configs: list[AgentConfig],
        *,
        now: datetime | None = None,
    ) -> list[str]:
        """Enqueue agent-jobs for on_event matches with action \"job\".

        Job enqueues pass the same authorization gates as event spawns —
        circuit breaker / manual stop / dependency checks — so ``job`` is a
        delivery mode, not a bypass. The whole read->enqueue window runs
        inside the reactor batch lock, so the queued/running dedupe check
        cannot race a concurrent supervisor. Prompts are built from operator
        config plus the sanitized event label only; event payload text never
        reaches a job prompt.
        """
        matches = [
            (config, reason)
            for config, reason in self.evaluate_event_records(
                batch.events, agent_configs, now=now, batch=batch
            )
            if config.on_event_action == "job"
        ]
        if not matches:
            return []
        from ..agent_jobs import AgentJobQueue

        resolved_config = self._config or load_config_model(merge_user=True)
        queue = AgentJobQueue(resolved_config.general.context_root)
        queue.ensure()
        active_keys = {
            job.dedupe_key
            for status in ("queue", "running")
            for job in queue.list(status=status)
            if job.dedupe_key
        }
        created: list[str] = []
        for config, reason in matches:
            existing = self.status(config.name)
            if existing and (
                existing.state in ("running", "awaiting_review", "circuit_open")
                or existing.manually_stopped
            ):
                continue
            ready, dep_reason = self._check_dependencies(
                config.name, config, agent_configs,
            )
            if not ready:
                _log.info(
                    "Skipping event job for '%s': %s", config.name, dep_reason
                )
                continue
            dedupe_key = f"on_event:{config.name}"
            if dedupe_key in active_keys:
                continue
            prompt = (
                f"Agent config '{config.name}' matched {reason}. "
                + (
                    f"Run `python -m {config.module}` and review its result."
                    if config.module
                    else config.description
                )
            ).strip()
            job = queue.create(
                title=f"{config.name}: react to {reason}",
                prompt=prompt,
                created_by=AGENT_NAME,
                dedupe_key=dedupe_key,
            )
            active_keys.add(dedupe_key)
            batch.mark_dispatched(config.name)
            created.append(job.id)
        return created

    def reconcile(
        self,
        agent_configs: list[AgentConfig],
        *,
        event: str | None = None,
        changed_paths: list[Path] | None = None,
        event_batch: ReactorBatch | None = None,
        now: datetime | None = None,
    ) -> list[RunningAgent]:
        candidates: dict[str, tuple[AgentConfig, str]] = {}
        event_candidates: set[str] = set()
        for config in agent_configs:
            if config.auto_start:
                candidates.setdefault(config.name, (config, "auto_start"))
        if event:
            for config in self.evaluate_triggers_from(event, agent_configs):
                candidates.setdefault(config.name, (config, event))
        if changed_paths:
            for config in self.evaluate_watch_paths(changed_paths, agent_configs):
                candidates.setdefault(config.name, (config, "file_watch"))
        if event_batch and event_batch.events:
            for config, reason in self.evaluate_event_records(
                event_batch.events, agent_configs, now=now, batch=event_batch
            ):
                if config.on_event_action == "job":
                    continue  # enqueue_event_jobs owns the job action
                if config.name not in candidates:
                    candidates[config.name] = (config, reason)
                    event_candidates.add(config.name)
        for config in self.due_schedules(agent_configs, now=now):
            candidates.setdefault(config.name, (config, f"schedule:{config.schedule}"))

        started: list[RunningAgent] = []
        for config, reason in candidates.values():
            if not config.module:
                continue
            existing = self.status(config.name)
            if existing and (
                existing.state in ("running", "awaiting_review", "circuit_open")
                or existing.manually_stopped
            ):
                continue
            ready, dep_reason = self._check_dependencies(
                config.name, config, agent_configs,
            )
            if not ready:
                _log.info(
                    "Skipping '%s' during reconcile: %s",
                    config.name,
                    dep_reason,
                )
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
            if event_batch and config.name in event_candidates:
                # Debounce keys off actual dispatches, so only a spawn that
                # really happened advances the agent's dispatch time.
                event_batch.mark_dispatched(config.name)

        # After spawning, check for targeted handoff-driven agents.
        # We scan for any agents that completed during this reconcile cycle
        # (their state transitioned from running to stopped) and process
        # their handoff targets.
        for agent in self.list_agents():
            if agent.state == "stopped" and not agent.manually_stopped:
                handoff_started = self._process_handoff_targets(
                    agent.name, agent_configs,
                )
                started.extend(handoff_started)

        return started

    def audit(self) -> dict[str, object]:
        agents = self.list_agents()
        counts: dict[str, int] = {
            "running": 0,
            "failed": 0,
            "stopped": 0,
            "manual_stop": 0,
            "circuit_open": 0,
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
            elif agent.state == "circuit_open":
                counts["circuit_open"] += 1
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


def _snapshot_watch_paths(agent_configs: list[AgentConfig]) -> dict[Path, WatchSignature]:
    paths: dict[Path, WatchSignature] = {}
    for config in agent_configs:
        for watch_path in config.watch_paths:
            resolved = watch_path.expanduser().resolve()
            if resolved in paths:
                continue
            signature = _watch_signature(resolved)
            paths[resolved] = signature
            if signature[-1]:
                if resolved not in _INCOMPLETE_WATCH_WARNED:
                    _log.warning(
                        "Watch signature for %s was bounded or incomplete; "
                        "changes outside the scanned prefix may require a manual index refresh",
                        resolved,
                    )
                    _INCOMPLETE_WATCH_WARNED.add(resolved)
            else:
                _INCOMPLETE_WATCH_WARNED.discard(resolved)
    return paths


def _diff_watch_paths(
    previous: dict[Path, WatchSignature],
    current: dict[Path, WatchSignature],
) -> list[Path]:
    changed: list[Path] = []
    keys = set(previous) | set(current)
    for path in sorted(keys):
        if previous.get(path) != current.get(path):
            changed.append(path)
    return changed


def _run_once(
    args: argparse.Namespace,
    previous_watch_state: dict[Path, WatchSignature],
    *,
    first_run: bool,
) -> tuple[AgentResult, dict[Path, WatchSignature]]:
    started_at = now_iso()
    start = time.time()
    config = load_config_model(
        config_path=Path(args.config).expanduser() if args.config else None,
        merge_user=True,
    )
    profile = resolve_active_profile(config)
    supervisor = AgentSupervisor(
        config=config,
        config_path=Path(args.config) if args.config else None,
    )
    current_watch_state = _snapshot_watch_paths(profile.agent_configs)
    changed_paths = (
        _diff_watch_paths(previous_watch_state, current_watch_state)
        if previous_watch_state
        else []
    )
    # Event handling is transactional: dispatch happens inside the batch and
    # the cursor only advances on ack afterwards, so a crash mid-dispatch
    # redelivers next cycle instead of losing events. A contended lock means
    # another supervisor owns this cycle's events — skip them, never double
    # deliver.
    reactor_event_count = 0
    reactor_skipped = 0
    reactor_truncated = False
    reactor_busy = False
    try:
        with open_event_batch(
            config.general.context_root,
            supervisor._state_dir,
            config=config,
            now=_now_utc(),
        ) as batch:
            reactor_event_count = len(batch.events)
            reactor_skipped = batch.skipped_malformed
            reactor_truncated = batch.truncated
            started_agents = supervisor.reconcile(
                profile.agent_configs,
                event="on_boot" if first_run else None,
                changed_paths=changed_paths,
                event_batch=batch,
                now=_now_utc(),
            )
            event_jobs = supervisor.enqueue_event_jobs(
                batch,
                profile.agent_configs,
                now=_now_utc(),
            )
            batch.ack()
    except ReactorBusyError:
        reactor_busy = True
        started_agents = supervisor.reconcile(
            profile.agent_configs,
            event="on_boot" if first_run else None,
            changed_paths=changed_paths,
            now=_now_utc(),
        )
        event_jobs = []
    audit = supervisor.audit()
    notes: list[str] = []
    if changed_paths:
        notes.append("watch-path changes detected")
    if reactor_event_count:
        notes.append(f"{reactor_event_count} new reactor events")
    if reactor_truncated:
        notes.append("reactor backlog truncated; remainder next cycle")
    if reactor_skipped:
        notes.append(f"{reactor_skipped} malformed reactor records skipped")
    if reactor_busy:
        notes.append("reactor lock contended; events deferred one cycle")
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
            "reactor_events": reactor_event_count,
            "reactor_skipped_malformed": reactor_skipped,
            "reactor_truncated": int(reactor_truncated),
            "reactor_busy": int(reactor_busy),
            "event_jobs": len(event_jobs),
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
    # Validate the supervisor's own result
    supervisor._validate_agent_result(result)
    return result, current_watch_state


def run(args: argparse.Namespace) -> int:
    configure_logging(args.quiet)
    watch_state: dict[Path, WatchSignature] = {}
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
