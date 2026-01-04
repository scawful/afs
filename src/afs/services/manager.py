"""Minimal service manager for AFS background tasks."""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ..config import load_config_model
from ..schema import AFSConfig, ServiceConfig
from .adapters.base import ServiceAdapter
from .adapters.launchd import LaunchdAdapter
from .adapters.systemd import SystemdAdapter
from .models import ServiceDefinition, ServiceState, ServiceStatus, ServiceType

# State directory for tracking running services
STATE_DIR = Path.home() / ".config" / "afs" / "services" / "state"


def _resolve_interval_env(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


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
        """Get status of a service."""
        definition = self.get_definition(name)
        if not definition:
            return ServiceStatus(name=name, state=ServiceState.UNKNOWN, enabled=False)

        state_file = STATE_DIR / f"{name}.json"
        pid = None
        last_started = None

        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                pid = data.get("pid")
                if data.get("started_at"):
                    last_started = datetime.fromisoformat(data["started_at"])
            except (json.JSONDecodeError, ValueError):
                pass

        # Check if process is running
        state = ServiceState.STOPPED
        if pid:
            try:
                os.kill(pid, 0)  # Check if process exists
                state = ServiceState.RUNNING
            except (ProcessLookupError, PermissionError):
                pid = None
                state = ServiceState.STOPPED

        # Special handling for docker-based services
        if name == "openwebui":
            state = self._check_docker_service("afs-chat-simple") or self._check_docker_service("afs-chat")

        return ServiceStatus(
            name=definition.name,
            state=state,
            pid=pid,
            enabled=True,
            last_started=last_started,
        )

    def _check_docker_service(self, container_name: str) -> ServiceState:
        """Check if a docker container is running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip() == "true":
                return ServiceState.RUNNING
        except FileNotFoundError:
            pass
        return ServiceState.STOPPED

    def start(self, name: str, foreground: bool = False) -> bool:
        """Start a service."""
        definition = self.get_definition(name)
        if not definition:
            raise KeyError(f"Unknown service: {name}")

        # Check if already running
        current = self.status(name)
        if current.state == ServiceState.RUNNING:
            return True

        STATE_DIR.mkdir(parents=True, exist_ok=True)

        # Build environment
        env = os.environ.copy()
        env.update(definition.environment)

        # Handle Open WebUI via chat-service to ensure secrets sync
        if name == "openwebui":
            return self._start_openwebui_service(definition, env)

        # Start the process
        if foreground:
            subprocess.run(
                definition.command,
                cwd=definition.working_directory,
                env=env,
            )
            return True

        # Background daemon
        process = subprocess.Popen(
            definition.command,
            cwd=definition.working_directory,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Save state
        state_file = STATE_DIR / f"{name}.json"
        state_file.write_text(json.dumps({
            "pid": process.pid,
            "started_at": datetime.now().isoformat(),
            "command": definition.command,
        }))

        return True

    def _find_docker_compose(self) -> list[str]:
        """Find docker compose command (handles both old and new syntax)."""
        import shutil

        # First try docker-compose (old standalone binary) - more reliable
        compose_paths = [
            "/usr/local/bin/docker-compose",
            "/opt/homebrew/bin/docker-compose",
            shutil.which("docker-compose"),
        ]
        for p in compose_paths:
            if p and Path(p).exists():
                return [p]

        # Check docker locations for new "docker compose" syntax
        docker_paths = [
            "/usr/local/bin/docker",
            "/opt/homebrew/bin/docker",
            "/Applications/Docker.app/Contents/Resources/bin/docker",
            shutil.which("docker"),
        ]
        docker_bin = None
        for p in docker_paths:
            if p and Path(p).exists():
                docker_bin = p
                break

        if not docker_bin:
            raise FileNotFoundError("Docker not found. Install docker-compose: brew install docker-compose")

        # Try new syntax: docker compose
        result = subprocess.run(
            [docker_bin, "compose", "version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return [docker_bin, "compose"]

        raise FileNotFoundError(
            "docker-compose not found. Install it with: brew install docker-compose"
        )

    def _start_docker_service(self, definition: ServiceDefinition) -> bool:
        """Start a docker-based service."""
        try:
            cwd = str(definition.working_directory) if definition.working_directory else None
            compose_cmd = self._find_docker_compose()

            # Build command: [docker, compose] or [docker-compose] + [-f, file, up, -d]
            # Extract args after "docker compose" from definition
            args = []
            for c in definition.command:
                if c in ("docker", "compose"):
                    continue
                args.append(c)

            cmd = compose_cmd + args
            print(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"Error: {result.stderr.strip()}")
            return result.returncode == 0
        except FileNotFoundError as e:
            print(f"Docker not found: {e}")
            return False
        except Exception as e:
            print(f"Exception: {e}")
            return False

    def _openwebui_script_path(self, definition: ServiceDefinition) -> Path:
        if definition.command:
            return Path(definition.command[0])
        repo_root = self._find_repo_root() or Path.home() / "src" / "lab" / "afs"
        return repo_root / "scripts" / "chat-service.sh"

    def _openwebui_compose_path(self) -> Path:
        repo_root = self._find_repo_root() or Path.home() / "src" / "lab" / "afs"
        return repo_root / "docker" / "docker-compose.simple.yml"

    def _start_openwebui_service(self, definition: ServiceDefinition, env: dict[str, str]) -> bool:
        """Start Open WebUI using the chat-service helper script."""
        try:
            if definition.command:
                cmd = list(definition.command)
            else:
                script = self._openwebui_script_path(definition)
                cmd = [str(script), "start", "simple"]
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=definition.working_directory,
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 and result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
            return result.returncode == 0
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return False
        except Exception as e:
            print(f"Exception: {e}")
            return False

    def stop(self, name: str) -> bool:
        """Stop a service."""
        definition = self.get_definition(name)
        if not definition:
            raise KeyError(f"Unknown service: {name}")

        # Handle Open WebUI via chat-service to mirror startup path
        if name == "openwebui":
            return self._stop_openwebui_service(definition)

        state_file = STATE_DIR / f"{name}.json"
        if not state_file.exists():
            return True

        try:
            data = json.loads(state_file.read_text())
            pid = data.get("pid")
            if pid:
                os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, json.JSONDecodeError):
            pass

        state_file.unlink(missing_ok=True)
        return True

    def _stop_docker_service(self, definition: ServiceDefinition) -> bool:
        """Stop a docker-based service."""
        cwd = str(definition.working_directory) if definition.working_directory else None
        compose_cmd = self._find_docker_compose()
        compose_file = None

        # Find compose file from command
        for i, arg in enumerate(definition.command):
            if arg == "-f" and i + 1 < len(definition.command):
                compose_file = definition.command[i + 1]
                break

        if compose_file:
            try:
                cmd = compose_cmd + ["-f", compose_file, "down"]
                result = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                )
                return result.returncode == 0
            except Exception:
                pass
        return False

    def _stop_openwebui_service(self, definition: ServiceDefinition) -> bool:
        """Stop Open WebUI using docker compose for the simple stack."""
        try:
            compose_cmd = self._find_docker_compose()
            compose_file = self._openwebui_compose_path()
            cmd = compose_cmd + ["-f", str(compose_file), "down"]
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=str(compose_file.parent),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 and result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
            return result.returncode == 0
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return False
        except Exception as e:
            print(f"Exception: {e}")
            return False

    def restart(self, name: str) -> bool:
        """Restart a service."""
        self.stop(name)
        return self.start(name)

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
        afs_root = repo_root or Path.home() / "src" / "lab" / "afs"
        environment = self._service_environment()
        context_root = self.config.general.context_root
        agent_output_dir = context_root / "scratchpad" / "afs_agents"
        memory_cfg = self.config.memory_export
        memory_report = memory_cfg.report_output or (agent_output_dir / "memory_export.json")
        memory_dataset = memory_cfg.dataset_output
        memory_interval = memory_cfg.interval_seconds if memory_cfg.interval_seconds > 0 else 3600
        context_warm_report = agent_output_dir / "context_warm.json"
        context_warm_interval = _resolve_interval_env("AFS_CONTEXT_WARM_INTERVAL", default=3600)
        docker_dir = afs_root / "docker"
        chat_service = afs_root / "scripts" / "chat-service.sh"

        memory_command = [
            python,
            "-m",
            "afs.agents.memory_export",
            "--output",
            str(memory_report),
            "--dataset-output",
            str(memory_dataset),
            "--interval",
            str(memory_interval),
        ]
        if memory_cfg.allow_raw:
            memory_command.append("--allow-raw")
        if memory_cfg.default_instruction:
            memory_command.extend(["--default-instruction", memory_cfg.default_instruction])
        if memory_cfg.limit and memory_cfg.limit > 0:
            memory_command.extend(["--limit", str(memory_cfg.limit)])

        context_warm_command = [
            python,
            "-m",
            "afs.agents.context_warm",
            "--output",
            str(context_warm_report),
            "--interval",
            str(context_warm_interval),
        ]

        return {
            # Chat infrastructure services
            "openwebui": ServiceDefinition(
                name="openwebui",
                label="Open WebUI",
                description="Web chat interface for testing Zelda models",
                command=[str(chat_service), "start", "simple"],
                working_directory=afs_root,
                environment={},
                service_type=ServiceType.ONESHOT,
                keep_alive=False,
                run_at_load=False,
            ),
            "gateway": ServiceDefinition(
                name="gateway",
                label="AFS Gateway",
                description="OpenAI-compatible API for MoE routing",
                command=[python, "-m", "uvicorn", "afs.gateway.server:app", "--host", "0.0.0.0", "--port", "8000"],
                working_directory=repo_root,
                environment=environment,
                service_type=ServiceType.DAEMON,
                keep_alive=True,
                run_at_load=False,
            ),
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
            "context-audit": ServiceDefinition(
                name="context-audit",
                label="AFS Context Audit",
                description="Audit AFS contexts for missing directories",
                command=[
                    python,
                    "-m",
                    "afs.agents.context_audit",
                    "--output",
                    str(agent_output_dir / "context_audit.json"),
                ],
                working_directory=repo_root,
                environment=environment,
                service_type=ServiceType.ONESHOT,
                keep_alive=False,
                run_at_load=False,
            ),
            "context-inventory": ServiceDefinition(
                name="context-inventory",
                label="AFS Context Inventory",
                description="Summarize AFS contexts and mount counts",
                command=[
                    python,
                    "-m",
                    "afs.agents.context_inventory",
                    "--output",
                    str(agent_output_dir / "context_inventory.json"),
                ],
                working_directory=repo_root,
                environment=environment,
                service_type=ServiceType.ONESHOT,
                keep_alive=False,
                run_at_load=False,
            ),
            "memory-export": ServiceDefinition(
                name="memory-export",
                label="AFS Memory Export",
                description="Export memory entries into training JSONL on an interval",
                command=memory_command,
                working_directory=repo_root,
                environment=environment,
                service_type=ServiceType.DAEMON,
                keep_alive=True,
                run_at_load=bool(memory_cfg.auto_start),
            ),
            "context-warm": ServiceDefinition(
                name="context-warm",
                label="AFS Context Warm",
                description="Sync workspace paths, discover contexts, and refresh embeddings on an interval",
                command=context_warm_command,
                working_directory=repo_root,
                environment=environment,
                service_type=ServiceType.DAEMON,
                keep_alive=True,
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
