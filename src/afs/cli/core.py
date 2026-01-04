"""Core CLI commands: init, plugins, status, services, agents, orchestrator, studio."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ._utils import (
    AFS_DIRS,
    build_config,
    ensure_context_root,
    load_manager,
    resolve_studio_root,
    run_command,
    studio_binary_name,
    studio_binary_path,
    studio_build,
    studio_build_dir,
    write_config,
)


def init_command(args: argparse.Namespace) -> int:
    """Initialize AFS context/root."""
    from ..config import load_config_model
    from ..schema import GeneralConfig

    config_path = Path(args.config) if args.config else Path.cwd() / "afs.toml"
    if args.no_config:
        config_path = None

    workspace_path = None
    if args.workspace_path or args.workspace_name:
        workspace_path = Path(args.workspace_path) if args.workspace_path else Path.cwd()

    existing_config = None
    if config_path and config_path.exists() and not args.force:
        existing_config = load_config_model(config_path=config_path, merge_user=False)

    if args.context_root:
        context_root = Path(args.context_root).expanduser().resolve()
    elif existing_config:
        context_root = existing_config.general.context_root
    else:
        context_root = GeneralConfig().context_root

    ensure_context_root(context_root)

    if args.link_context:
        link_path = Path.cwd() / ".context"
        if not link_path.exists():
            link_path.symlink_to(context_root)

    if config_path:
        if config_path.exists() and not args.force:
            print(f"Config exists, not modified: {config_path}")
        else:
            config = build_config(context_root, workspace_path, args.workspace_name)
            write_config(config_path, config)
            print(f"Wrote config: {config_path}")

    return 0


def plugins_command(args: argparse.Namespace) -> int:
    """List or load plugins."""
    from ..config import load_config_model
    from ..plugins import discover_plugins, load_plugins

    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    plugin_names = discover_plugins(config)
    if args.load:
        loaded = load_plugins(plugin_names, config.plugins.plugin_dirs)
        for name in plugin_names:
            status = "ok" if name in loaded else "failed"
            print(f"{name}\t{status}")
    else:
        for name in plugin_names:
            print(name)
    return 0


def services_list_command(args: argparse.Namespace) -> int:
    """List service definitions."""
    from ..services import ServiceManager

    manager = ServiceManager()
    for definition in manager.list_definitions():
        print(f"{definition.name}\t{definition.label}")
    return 0


def services_render_command(args: argparse.Namespace) -> int:
    """Render service unit."""
    from ..services import ServiceManager

    manager = ServiceManager()
    print(manager.render_unit(args.name))
    return 0


def services_start_command(args: argparse.Namespace) -> int:
    """Start a service."""
    from ..services import ServiceManager

    manager = ServiceManager()
    try:
        if manager.start(args.name, foreground=args.foreground):
            print(f"Started: {args.name}")
            return 0
        print(f"Failed to start: {args.name}")
        return 1
    except KeyError as e:
        print(str(e))
        return 1


def services_stop_command(args: argparse.Namespace) -> int:
    """Stop a service."""
    from ..services import ServiceManager

    manager = ServiceManager()
    try:
        if manager.stop(args.name):
            print(f"Stopped: {args.name}")
            return 0
        print(f"Failed to stop: {args.name}")
        return 1
    except KeyError as e:
        print(str(e))
        return 1


def services_status_command(args: argparse.Namespace) -> int:
    """Get service status."""
    from ..services import ServiceManager

    manager = ServiceManager()

    if args.name:
        status = manager.status(args.name)
        symbol = "●" if status.state.value == "running" else "○"
        color = "\033[32m" if status.state.value == "running" else "\033[31m"
        reset = "\033[0m"
        print(f"{color}{symbol}{reset} {status.name}: {status.state.value}")
        if status.pid:
            print(f"  PID: {status.pid}")
        if status.last_started:
            print(f"  Started: {status.last_started.isoformat()}")
    else:
        # Show all services
        for definition in manager.list_definitions():
            status = manager.status(definition.name)
            symbol = "●" if status.state.value == "running" else "○"
            color = "\033[32m" if status.state.value == "running" else "\033[31m"
            reset = "\033[0m"
            print(f"{color}{symbol}{reset} {definition.name}: {status.state.value}")

    return 0


def services_restart_command(args: argparse.Namespace) -> int:
    """Restart a service."""
    from ..services import ServiceManager

    manager = ServiceManager()
    try:
        if manager.restart(args.name):
            print(f"Restarted: {args.name}")
            return 0
        print(f"Failed to restart: {args.name}")
        return 1
    except KeyError as e:
        print(str(e))
        return 1


def agents_list_command(args: argparse.Namespace) -> int:
    """List available agents."""
    from ..agents import list_agents

    for agent in list_agents():
        if agent.description:
            print(f"{agent.name}\t{agent.description}")
        else:
            print(agent.name)
    return 0


def agents_run_command(args: argparse.Namespace) -> int:
    """Run a built-in agent."""
    from ..agents import get_agent

    agent = get_agent(args.name)
    if not agent:
        print(f"unknown agent: {args.name}")
        return 1
    agent_args = list(args.agent_args or [])
    if agent_args and agent_args[0] == "--":
        agent_args = agent_args[1:]
    return agent.entrypoint(agent_args)


def orchestrator_list_command(args: argparse.Namespace) -> int:
    """List orchestrator agents."""
    from ..orchestration import Orchestrator

    orchestrator = Orchestrator()
    for agent in orchestrator.list_agents():
        tags = ",".join(agent.tags) if agent.tags else "-"
        print(f"{agent.name}\t{agent.role}\t{agent.backend}\t{tags}")
    return 0


def orchestrator_plan_command(args: argparse.Namespace) -> int:
    """Plan task execution."""
    from ..orchestration import Orchestrator, TaskRequest

    orchestrator = Orchestrator()
    request = TaskRequest(summary=args.summary, tags=args.tag or [], role=args.role)
    plan = orchestrator.plan(request)
    if plan.notes:
        for note in plan.notes:
            print(f"note: {note}")
    for agent in plan.agents:
        tags = ",".join(agent.tags) if agent.tags else "-"
        print(f"{agent.name}\t{agent.role}\t{agent.backend}\t{tags}")
    return 0


def studio_build_command(args: argparse.Namespace) -> int:
    """Build AFS studio."""
    try:
        root = resolve_studio_root()
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    build_dir = studio_build_dir(root, args.build_dir)
    status = studio_build(root, build_dir, args.build_type, args.config)
    if status == 0:
        print(f"build_dir: {build_dir}")
    return status


def studio_run_command(args: argparse.Namespace) -> int:
    """Run AFS studio."""
    try:
        root = resolve_studio_root()
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    build_dir = studio_build_dir(root, args.build_dir)
    binary = studio_binary_path(build_dir, args.config)
    if not binary.exists() and args.build:
        status = studio_build(root, build_dir, args.build_type, args.config)
        if status != 0:
            return status
        binary = studio_binary_path(build_dir, args.config)
    if not binary.exists():
        print(f"binary not found: {binary}")
        return 1
    cmd = [str(binary)]
    if args.args:
        cmd.extend(args.args)
    return run_command(cmd)


def studio_install_command(args: argparse.Namespace) -> int:
    """Install AFS studio."""
    try:
        root = resolve_studio_root()
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    build_dir = studio_build_dir(root, args.build_dir)
    if not build_dir.exists():
        print(f"build dir missing: {build_dir}")
        return 1
    prefix = (
        Path(args.prefix).expanduser().resolve()
        if args.prefix
        else Path.home() / ".local"
    )
    cmd = ["cmake", "--install", str(build_dir), "--prefix", str(prefix)]
    if args.config:
        cmd.extend(["--config", args.config])
    status = run_command(cmd)
    if status == 0:
        print(f"installed: {prefix / 'bin' / studio_binary_name()}")
    return status


def studio_path_command(args: argparse.Namespace) -> int:
    """Show studio binary path."""
    try:
        root = resolve_studio_root()
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    build_dir = studio_build_dir(root, args.build_dir)
    binary = studio_binary_path(build_dir, args.config)
    print(binary)
    return 0


def studio_alias_command(args: argparse.Namespace) -> int:
    """Print shell aliases for studio."""
    try:
        studio_root = resolve_studio_root()
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    afs_root = Path(os.getenv("AFS_ROOT") or Path(__file__).resolve().parents[3]).expanduser().resolve()
    print(f"export AFS_ROOT=\"{afs_root}\"")
    if studio_root.resolve() != (afs_root / "apps" / "studio").resolve():
        print(f"export AFS_STUDIO_ROOT=\"{studio_root}\"")
    print("alias afs-studio='PYTHONPATH=\"$AFS_ROOT/src\" python -m afs studio run --build'")
    print("alias afs-studio-build='PYTHONPATH=\"$AFS_ROOT/src\" python -m afs studio build'")
    return 0


def status_command(args: argparse.Namespace) -> int:
    """Show AFS status."""
    from ..config import load_config_model
    from ..core import find_root, resolve_context_root

    start_dir = Path(args.start_dir).expanduser().resolve() if args.start_dir else None
    root = find_root(start_dir)
    config = load_config_model()
    context_root = resolve_context_root(config, root)

    missing = []
    for name in AFS_DIRS:
        if not (context_root / name).exists():
            missing.append(name)

    if args.json:
        payload = {
            "context_root": str(context_root),
            "linked_root": str(root) if root else None,
            "missing_dirs": missing,
            "valid": not missing,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"context_root: {context_root}")
    print(f"linked_root: {root if root else '(none)'}")
    if missing:
        print("missing_dirs: " + ", ".join(missing))
    else:
        print("missing_dirs: (none)")

    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register core command parsers."""
    # init
    init_parser = subparsers.add_parser("init", help="Initialize AFS context/root.")
    init_parser.add_argument("--context-root", help="Context root path.")
    init_parser.add_argument("--config", help="Path to write afs.toml.")
    init_parser.add_argument("--no-config", action="store_true", help="Do not write config.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite config if it exists.")
    init_parser.add_argument("--workspace-path", help="Workspace path to register.")
    init_parser.add_argument("--workspace-name", help="Workspace label/description.")
    init_parser.add_argument("--link-context", action="store_true", help="Symlink .context to context root.")
    init_parser.set_defaults(func=init_command)

    # plugins
    plugins_parser = subparsers.add_parser("plugins", help="List or load plugins.")
    plugins_parser.add_argument("--config", help="Config path for plugin discovery.")
    plugins_parser.add_argument("--load", action="store_true", help="Attempt to import plugins.")
    plugins_parser.set_defaults(func=plugins_command)

    # status
    status_parser = subparsers.add_parser("status", help="Show AFS status.")
    status_parser.add_argument("--start-dir", help="Starting directory.")
    status_parser.add_argument("--json", action="store_true", help="Output JSON.")
    status_parser.set_defaults(func=status_command)

    # services
    services_parser = subparsers.add_parser("services", help="Service definitions.")
    services_sub = services_parser.add_subparsers(dest="services_command")

    services_list = services_sub.add_parser("list", help="List service definitions.")
    services_list.set_defaults(func=services_list_command)

    services_render = services_sub.add_parser("render", help="Render service unit.")
    services_render.add_argument("name", help="Service name.")
    services_render.set_defaults(func=services_render_command)

    services_start = services_sub.add_parser("start", help="Start a service.")
    services_start.add_argument("name", help="Service name.")
    services_start.add_argument("--foreground", "-f", action="store_true", help="Run in foreground.")
    services_start.set_defaults(func=services_start_command)

    services_stop = services_sub.add_parser("stop", help="Stop a service.")
    services_stop.add_argument("name", help="Service name.")
    services_stop.set_defaults(func=services_stop_command)

    services_status = services_sub.add_parser("status", help="Get service status.")
    services_status.add_argument("name", nargs="?", help="Service name (optional, shows all if omitted).")
    services_status.set_defaults(func=services_status_command)

    services_restart = services_sub.add_parser("restart", help="Restart a service.")
    services_restart.add_argument("name", help="Service name.")
    services_restart.set_defaults(func=services_restart_command)

    # agents
    agents_parser = subparsers.add_parser("agents", help="Run built-in agents.")
    agents_sub = agents_parser.add_subparsers(dest="agents_command")

    agents_list = agents_sub.add_parser("list", help="List available agents.")
    agents_list.set_defaults(func=agents_list_command)

    agents_run = agents_sub.add_parser("run", help="Run a built-in agent.")
    agents_run.add_argument("name", help="Agent name.")
    agents_run.add_argument(
        "agent_args",
        nargs=argparse.REMAINDER,
        help="Arguments for the agent (prefix with -- to pass through).",
    )
    agents_run.set_defaults(func=agents_run_command)

    # orchestrator
    orch_parser = subparsers.add_parser("orchestrator", help="Orchestrator helpers.")
    orch_sub = orch_parser.add_subparsers(dest="orchestrator_command")

    orch_list = orch_sub.add_parser("list", help="List configured agents.")
    orch_list.set_defaults(func=orchestrator_list_command)

    orch_plan = orch_sub.add_parser("plan", help="Generate a task plan.")
    orch_plan.add_argument("summary", help="Task summary.")
    orch_plan.add_argument("--tag", action="append", help="Task tags.")
    orch_plan.add_argument("--role", help="Required role.")
    orch_plan.set_defaults(func=orchestrator_plan_command)

    # studio
    studio_parser = subparsers.add_parser("studio", help="AFS Studio GUI.")
    studio_sub = studio_parser.add_subparsers(dest="studio_command")

    studio_build_p = studio_sub.add_parser("build", help="Build AFS Studio.")
    studio_build_p.add_argument("--build-dir", help="Custom build directory.")
    studio_build_p.add_argument("--build-type", help="CMake build type.")
    studio_build_p.add_argument("--config", help="Build configuration (multi-config).")
    studio_build_p.set_defaults(func=studio_build_command)

    studio_run_p = studio_sub.add_parser("run", help="Run AFS Studio.")
    studio_run_p.add_argument("--build-dir", help="Custom build directory.")
    studio_run_p.add_argument("--config", help="Build configuration.")
    studio_run_p.add_argument("--build", action="store_true", help="Build before run.")
    studio_run_p.add_argument("--build-type", help="CMake build type.")
    studio_run_p.add_argument("args", nargs="*", help="Arguments to pass.")
    studio_run_p.set_defaults(func=studio_run_command)

    studio_install_p = studio_sub.add_parser("install", help="Install AFS Studio.")
    studio_install_p.add_argument("--build-dir", help="Custom build directory.")
    studio_install_p.add_argument("--prefix", help="Installation prefix.")
    studio_install_p.add_argument("--config", help="Build configuration.")
    studio_install_p.set_defaults(func=studio_install_command)

    studio_path_p = studio_sub.add_parser("path", help="Print binary path.")
    studio_path_p.add_argument("--build-dir", help="Custom build directory.")
    studio_path_p.add_argument("--config", help="Build configuration.")
    studio_path_p.set_defaults(func=studio_path_command)

    studio_alias_p = studio_sub.add_parser("alias", help="Print shell aliases.")
    studio_alias_p.set_defaults(func=studio_alias_command)
