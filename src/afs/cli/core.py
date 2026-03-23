"""Core CLI commands: init, plugins, status, services, agents, orchestrator, studio."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import load_runtime_config_model
from ..context_paths import resolve_mount_root
from ..event_log import read_agent_events
from ..memory_consolidation import consolidate_history_to_memory
from ..models import MountType
from ._utils import (
    build_config,
    ensure_context_root,
    load_manager,
    resolve_context_paths,
    resolve_studio_root,
    run_command,
    studio_binary_name,
    studio_binary_path,
    studio_build,
    studio_build_dir,
    write_config,
)


def _hint(text: str) -> str:
    """Format a contextual hint, dimmed when color is supported."""
    if sys.stdout.isatty() and os.getenv("NO_COLOR") is None:
        return f"  \033[2m{text}\033[0m"
    return f"  {text}"


def _load_service_manager(args: argparse.Namespace):
    from ..services import ServiceManager

    explicit_config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    config, config_path = load_runtime_config_model(
        config_path=explicit_config_path,
        merge_user=True,
        start_dir=Path.cwd(),
    )
    return ServiceManager(config=config, config_path=config_path)


def _resolve_command_context(args: argparse.Namespace) -> Path:
    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    if not hasattr(args, "path"):
        args.path = None
    if not hasattr(args, "context_root"):
        args.context_root = None
    if not hasattr(args, "context_dir"):
        args.context_dir = None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    return context_path


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

    print(f"Context root: {context_root}")
    print()
    print(_hint("Next steps:"))
    print(_hint("  afs status                      # verify context health"))
    print(_hint("  afs context discover --path .    # index this project"))
    print(_hint("  afs profile current              # check active profile"))
    print(_hint("  afs skills list                  # see available skills"))

    return 0


def plugins_command(args: argparse.Namespace) -> int:
    """List or load plugins."""
    from ..config import load_config_model
    from ..plugins import (
        discover_extension_manifests,
        discover_plugins,
        load_enabled_extensions,
        load_plugins,
        resolve_extensions_config,
        resolve_plugins_config,
    )

    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    plugins_config = resolve_plugins_config(config)
    plugin_names = discover_plugins(plugins_config)
    extensions_config = resolve_extensions_config(config)
    extension_manifests = discover_extension_manifests(extensions_config)
    loaded_extensions = load_enabled_extensions(config=config)
    loaded = {}
    if args.load:
        loaded = load_plugins(plugin_names, plugins_config.plugin_dirs)

    if args.json:
        payload = {
            "plugin_dirs": [str(path) for path in plugins_config.plugin_dirs],
            "enabled_plugins": list(plugins_config.enabled_plugins),
            "auto_discover": plugins_config.auto_discover,
            "auto_discover_prefixes": list(plugins_config.auto_discover_prefixes),
            "plugins": [
                {
                    "name": name,
                    "status": "ok" if name in loaded else "failed" if args.load else "unknown",
                }
                for name in plugin_names
            ],
            "extension_dirs": [str(path) for path in extensions_config.extension_dirs],
            "enabled_extensions": list(extensions_config.enabled_extensions),
            "extensions": [
                {
                    "name": name,
                    "manifest": str(path),
                    "status": "ok" if name in loaded_extensions else "discovered",
                }
                for name, path in extension_manifests.items()
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.details:
        plugin_dirs = (
            ", ".join(str(path) for path in plugins_config.plugin_dirs)
            if plugins_config.plugin_dirs
            else "(none)"
        )
        enabled = (
            ", ".join(plugins_config.enabled_plugins)
            if plugins_config.enabled_plugins
            else "(none)"
        )
        prefixes = (
            ", ".join(plugins_config.auto_discover_prefixes)
            if plugins_config.auto_discover_prefixes
            else "(none)"
        )
        print(f"plugin_dirs: {plugin_dirs}")
        print(f"enabled_plugins: {enabled}")
        print(f"auto_discover: {str(plugins_config.auto_discover).lower()}")
        print(f"auto_discover_prefixes: {prefixes}")
        extension_dirs = (
            ", ".join(str(path) for path in extensions_config.extension_dirs)
            if extensions_config.extension_dirs
            else "(none)"
        )
        enabled_extensions = (
            ", ".join(extensions_config.enabled_extensions)
            if extensions_config.enabled_extensions
            else "(none)"
        )
        print(f"extension_dirs: {extension_dirs}")
        print(f"enabled_extensions: {enabled_extensions}")

    if args.load:
        for name in plugin_names:
            status = "ok" if name in loaded else "failed"
            print(f"{name}\t{status}")
    else:
        for name in plugin_names:
            print(name)
    return 0


def services_list_command(args: argparse.Namespace) -> int:
    """List service definitions."""
    manager = _load_service_manager(args)
    for definition in manager.list_definitions():
        print(f"{definition.name}\t{definition.label}")
    return 0


def services_render_command(args: argparse.Namespace) -> int:
    """Render service unit."""
    manager = _load_service_manager(args)
    print(manager.render_unit(args.name))
    return 0


def services_start_command(args: argparse.Namespace) -> int:
    """Start a service."""
    manager = _load_service_manager(args)
    try:
        if manager.start(args.name, foreground=args.foreground):
            print(f"Started: {args.name}")
            return 0
        print(f"Failed to start: {args.name}")
        return 1
    except KeyError as e:
        print(str(e))
        return 1


def services_install_command(args: argparse.Namespace) -> int:
    """Install a managed service unit."""
    manager = _load_service_manager(args)
    try:
        unit_path = manager.install(args.name, enable=args.enable)
    except (KeyError, RuntimeError) as exc:
        print(str(exc))
        return 1
    print(f"Installed: {args.name}")
    print(f"  unit: {unit_path}")
    return 0


def services_uninstall_command(args: argparse.Namespace) -> int:
    """Uninstall a managed service unit."""
    manager = _load_service_manager(args)
    try:
        removed = manager.uninstall(args.name, disable=not args.keep_enabled)
    except (KeyError, RuntimeError) as exc:
        print(str(exc))
        return 1
    print(f"{'Removed' if removed else 'Not installed'}: {args.name}")
    return 0


def services_enable_command(args: argparse.Namespace) -> int:
    """Enable a managed service unit."""
    manager = _load_service_manager(args)
    try:
        manager.enable(args.name)
    except (KeyError, RuntimeError) as exc:
        print(str(exc))
        return 1
    print(f"Enabled: {args.name}")
    return 0


def services_disable_command(args: argparse.Namespace) -> int:
    """Disable a managed service unit."""
    manager = _load_service_manager(args)
    try:
        manager.disable(args.name)
    except (KeyError, RuntimeError) as exc:
        print(str(exc))
        return 1
    print(f"Disabled: {args.name}")
    return 0


def services_stop_command(args: argparse.Namespace) -> int:
    """Stop a service."""
    manager = _load_service_manager(args)
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
    manager = _load_service_manager(args)

    if args.name:
        if args.system:
            try:
                payload = manager.system_status(args.name)
            except KeyError as exc:
                print(str(exc))
                return 1
            if args.json:
                print(json.dumps(payload, indent=2))
                return 0
            print(f"{payload['name']}: installed={str(payload['installed']).lower()} enabled={str(payload['enabled']).lower()} active={str(payload['active']).lower()}")
            print(f"  unit: {payload['unit_path']}")
            print(f"  stdout_log: {payload['stdout_log']}")
            print(f"  stderr_log: {payload['stderr_log']}")
            if payload.get("detail"):
                print(f"  detail: {payload['detail']}")
            return 0
        status = manager.status(args.name)
        if args.json:
            print(json.dumps(status.to_dict(), indent=2))
            return 0
        symbol = "●" if status.state.value == "running" else "○"
        color = "\033[32m" if status.state.value == "running" else "\033[31m"
        reset = "\033[0m"
        print(f"{color}{symbol}{reset} {status.name}: {status.state.value}")
        if status.pid:
            print(f"  PID: {status.pid}")
        if status.last_started:
            print(f"  Started: {status.last_started.isoformat()}")
        print(f"  Installed: {'yes' if status.installed else 'no'}")
        if status.unit_path:
            print(f"  Unit: {status.unit_path}")
        if status.stdout_log:
            print(f"  Stdout: {status.stdout_log}")
        if status.stderr_log:
            print(f"  Stderr: {status.stderr_log}")
    else:
        if args.system:
            payload = []
            for definition in manager.list_definitions():
                payload.append(manager.system_status(definition.name))
            if args.json:
                print(json.dumps(payload, indent=2))
                return 0
            for item in payload:
                print(
                    f"{item['name']}: installed={str(item['installed']).lower()} "
                    f"enabled={str(item['enabled']).lower()} active={str(item['active']).lower()}"
                )
            return 0
        # Show all services
        if args.json:
            payload = [manager.status(definition.name).to_dict() for definition in manager.list_definitions()]
            print(json.dumps(payload, indent=2))
            return 0
        for definition in manager.list_definitions():
            status = manager.status(definition.name)
            symbol = "●" if status.state.value == "running" else "○"
            color = "\033[32m" if status.state.value == "running" else "\033[31m"
            reset = "\033[0m"
            print(f"{color}{symbol}{reset} {definition.name}: {status.state.value}")

    return 0


def services_restart_command(args: argparse.Namespace) -> int:
    """Restart a service."""
    manager = _load_service_manager(args)
    try:
        if manager.restart(args.name):
            print(f"Restarted: {args.name}")
            return 0
        print(f"Failed to restart: {args.name}")
        return 1
    except KeyError as e:
        print(str(e))
        return 1


def services_logs_command(args: argparse.Namespace) -> int:
    """Show captured stdout/stderr logs for a service."""
    manager = _load_service_manager(args)
    try:
        payload = manager.logs(args.name, lines=args.lines)
    except KeyError as exc:
        print(str(exc))
        return 1
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"stdout_log: {payload['stdout_log']}")
    for line in payload["stdout"]:
        print(line)
    print()
    print(f"stderr_log: {payload['stderr_log']}")
    for line in payload["stderr"]:
        print(line)
    return 0


def agents_list_command(args: argparse.Namespace) -> int:
    """List available agents."""
    from ..agents import list_agents

    agents = list_agents()
    if not agents:
        print("no agents registered")
        print(_hint("agents are defined in config or extensions"))
        return 0
    for agent in agents:
        if agent.description:
            print(f"{agent.name}\t{agent.description}")
        else:
            print(agent.name)
    print(_hint(f"{len(agents)} agents available  |  afs agents run <name>  |  afs agents ps"))
    return 0


def agents_ps_command(args: argparse.Namespace) -> int:
    """List running background agents."""
    from ..agents.supervisor import AgentSupervisor

    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    supervisor = AgentSupervisor(config=manager.config)
    agents = supervisor.list_agents()
    if not args.all:
        agents = [agent for agent in agents if agent.state == "running"]
    if not agents:
        print("no matching agents")
        print(_hint("spawn agents with: afs agents run <name>"))
        print(_hint("or via MCP: agent.spawn / agent.ps / agent.stop"))
        return 0
    if args.json:
        payload = [
            {
                "name": agent.name,
                "state": agent.state,
                "pid": agent.pid,
                "started_at": agent.started_at,
                "module": agent.module,
                "last_event": agent.last_event,
                "last_error": agent.last_error,
                "manually_stopped": agent.manually_stopped,
            }
            for agent in agents
        ]
        print(json.dumps(payload, indent=2))
        return 0
    for agent in agents:
        pid = agent.pid or "-"
        extra = []
        if agent.last_event:
            extra.append(f"event={agent.last_event}")
        if agent.manually_stopped:
            extra.append("manual-stop")
        if agent.last_error:
            extra.append(f"error={agent.last_error}")
        suffix = f"\t{' '.join(extra)}" if extra else ""
        print(f"{agent.name}\t{agent.state}\tpid={pid}\t{agent.started_at}{suffix}")
    return 0


def agents_watch_command(args: argparse.Namespace) -> int:
    """Show recent progress events for an agent."""
    agent_name = args.name
    limit = args.limit
    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    history_dir = resolve_mount_root(context_path, MountType.HISTORY)
    if not history_dir.exists():
        print("no history directory")
        return 1
    events = read_agent_events(
        context_path,
        agent_name=agent_name,
        limit=limit,
        config=manager.config,
    )
    if not events:
        print(f"no events for {agent_name}")
        return 0
    for event in events:
        ts = event.get("timestamp", "")[:19]
        op = event.get("op", "")
        detail = (event.get("metadata") or {}).get("detail", "")
        print(f"{ts}  {op}  {detail}")
    return 0


def tasks_list_command(args: argparse.Namespace) -> int:
    """List tasks from the items queue."""
    from ..tasks import TaskQueue

    context_path = _resolve_command_context(args)
    queue = TaskQueue(context_path)
    tasks = queue.list(status=args.status if hasattr(args, "status") else None)
    if not tasks:
        print("no tasks")
        print(_hint("create tasks via MCP: task.create / task.list / task.claim"))
        print(_hint("or shell: afs-task 'Fix lint errors'"))
        return 0
    if hasattr(args, "json") and args.json:
        print(json.dumps([t.to_dict() for t in tasks], indent=2))
        return 0
    for task in tasks:
        assigned = f" -> {task.assigned_to}" if task.assigned_to else ""
        print(f"{task.id}\t[{task.status}]\tp{task.priority}\t{task.title}{assigned}")
    return 0


def hivemind_list_command(args: argparse.Namespace) -> int:
    """List recent hivemind messages."""
    from ..hivemind import HivemindBus

    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    context_path = _resolve_command_context(args)
    bus = HivemindBus(context_path, config=manager.config)
    topic = getattr(args, "topic", None)
    messages = bus.read(limit=args.limit if hasattr(args, "limit") else 20, topic=topic)
    if not messages:
        print("no messages")
        print(_hint("agents communicate via: hivemind.send / hivemind.read"))
        print(_hint("or shell: afs-say my-agent finding key=value"))
        return 0
    for msg in messages:
        to_part = f" -> {msg.to}" if msg.to else ""
        topic_part = f" #{msg.topic}" if msg.topic else ""
        print(f"{msg.timestamp[:19]}  [{msg.msg_type}]  {msg.from_agent}{to_part}{topic_part}")
        if msg.payload:
            for key, value in msg.payload.items():
                print(f"  {key}: {value}")
    return 0


def hivemind_subscribe_command(args: argparse.Namespace) -> int:
    """Subscribe an agent to topics."""
    from ..hivemind import HivemindBus

    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    context_path = _resolve_command_context(args)
    bus = HivemindBus(context_path, config=manager.config)
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    if not topics:
        print("no topics specified")
        return 1
    sub = bus.subscribe(args.agent, topics, ttl_hours=getattr(args, "ttl_hours", None))
    print(f"subscribed {args.agent} to {', '.join(sub.topics)}")
    return 0


def hivemind_unsubscribe_command(args: argparse.Namespace) -> int:
    """Unsubscribe an agent from topics."""
    from ..hivemind import HivemindBus

    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    context_path = _resolve_command_context(args)
    bus = HivemindBus(context_path, config=manager.config)
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    if not topics:
        print("no topics specified")
        return 1
    sub = bus.unsubscribe(args.agent, topics)
    remaining = ", ".join(sub.topics) if sub.topics else "(none)"
    print(f"unsubscribed {args.agent}; remaining topics: {remaining}")
    return 0


def hivemind_reap_command(args: argparse.Namespace) -> int:
    """Reap expired or stale hivemind messages."""
    from ..hivemind import HivemindBus

    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    context_path = _resolve_command_context(args)
    bus = HivemindBus(context_path, config=manager.config)
    summary = bus.reap(
        max_age_hours=getattr(args, "max_age_hours", None),
        dry_run=bool(getattr(args, "dry_run", False)),
    )

    if getattr(args, "json", False):
        print(json.dumps(summary, indent=2))
        return 0

    print(f"removed: {summary['removed_count']}")
    print(f"remaining: {summary['remaining_count']}")
    if summary.get("expired_count") is not None:
        print(f"expired: {summary['expired_count']}")
    if summary.get("aged_out_count") is not None:
        print(f"aged_out: {summary['aged_out_count']}")
    return 0


def memory_consolidate_command(args: argparse.Namespace) -> int:
    """Consolidate recent history into durable memory entries."""
    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    context_path = _resolve_command_context(args)
    result = consolidate_history_to_memory(
        context_path,
        config=manager.config,
        max_events_per_run=args.max_events,
        max_events_per_entry=args.max_events_per_entry,
        include_event_types=args.event_types,
        write_markdown=not args.no_markdown,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    age = "n/a"
    if result.last_timestamp:
        try:
            normalized = result.last_timestamp.replace("Z", "+00:00")
            last_seen = datetime.fromisoformat(normalized)
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=timezone.utc)
            age = str(int((datetime.now(timezone.utc) - last_seen).total_seconds()))
        except ValueError:
            age = "n/a"

    print(f"context_root: {result.context_root}")
    print(f"history_root: {result.history_root}")
    print(f"memory_root: {result.memory_root}")
    print(f"entries_path: {result.entries_path}")
    print(f"checkpoint: {result.checkpoint_path}")
    print()
    print(
        "consolidation: "
        f"scanned={result.scanned_events} "
        f"consolidated={result.consolidated_events} "
        f"entries={result.entries_written} "
        f"markdown={result.markdown_written} "
        f"last_age={age}s"
    )
    if result.notes:
        print("notes:")
        for note in result.notes:
            print(f"  - {note}")
    return 0


def memory_status_command(args: argparse.Namespace) -> int:
    """Show memory pipeline status."""
    from ..memory_consolidation import memory_status

    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    context_path = _resolve_command_context(args)
    status = memory_status(context_path, config=manager.config)
    if args.json:
        print(json.dumps(status, indent=2))
        return 0
    print(f"entries: {status['entries_count']}")
    print(f"cursor: {status['cursor_timestamp'] or '(none)'}")
    age = status["cursor_age_seconds"]
    print(f"cursor_age: {int(age)}s" if age is not None else "cursor_age: n/a")
    print(f"stale: {status['stale']}")
    if status["latest_summary_path"]:
        print(f"latest_summary: {status['latest_summary_path']}")
    return 0


def memory_search_command(args: argparse.Namespace) -> int:
    """Search memory entries."""
    from ..memory_consolidation import search_memory

    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    context_path = _resolve_command_context(args)
    results = search_memory(context_path, args.query, config=manager.config, limit=args.limit)
    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    if not results:
        print("(no matches)")
        return 0
    for entry in results:
        entry_id = entry.get("id", "?")
        instruction = str(entry.get("instruction", ""))[:80]
        print(f"  {entry_id}: {instruction}")
    return 0


def session_bootstrap_command(args: argparse.Namespace) -> int:
    """Build a proactive session bootstrap packet for the active context."""
    from ..session_bootstrap import (
        build_session_bootstrap,
        render_session_bootstrap,
        write_session_bootstrap_artifacts,
    )

    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    context_path = _resolve_command_context(args)
    summary = build_session_bootstrap(
        manager,
        context_path,
        task_limit=args.task_limit,
        message_limit=args.message_limit,
        agent_name=getattr(args, "agent_name", "cli") or "cli",
    )
    if not args.no_write_artifacts:
        summary["artifact_paths"] = write_session_bootstrap_artifacts(
            manager,
            context_path,
            summary,
        )

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(render_session_bootstrap(summary))
    return 0


def session_pack_command(args: argparse.Namespace) -> int:
    """Build a token-budgeted context pack for a target model."""
    from ..context_pack import (
        build_context_pack,
        render_context_pack,
        write_context_pack_artifacts,
    )

    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    context_path = _resolve_command_context(args)
    pack = build_context_pack(
        manager,
        context_path,
        query=args.query or "",
        task=args.task or "",
        model=args.model,
        workflow=args.workflow,
        tool_profile=args.tool_profile,
        token_budget=args.token_budget,
        include_content=args.include_content,
        max_query_results=args.max_query_results,
        max_embedding_results=args.max_embedding_results,
    )
    if not args.no_write_artifacts and not bool((pack.get("cache") or {}).get("hit")):
        pack["artifact_paths"] = write_context_pack_artifacts(
            manager,
            context_path,
            pack,
        )

    if args.json:
        print(json.dumps(pack, indent=2))
        return 0

    print(render_context_pack(pack))
    return 0


def session_handoff_command(args: argparse.Namespace) -> int:
    """Create a handoff packet."""
    from ..handoff import HandoffStore

    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    context_path = _resolve_command_context(args)
    store = HandoffStore(context_path, config=manager.config)

    accomplished = [s.strip() for s in (args.accomplished or "").split(";") if s.strip()]
    blocked = [s.strip() for s in (args.blocked or "").split(";") if s.strip()]
    next_steps = [s.strip() for s in (getattr(args, "next", None) or "").split(";") if s.strip()]

    packet = store.create(
        agent_name=getattr(args, "agent_name", None) or "cli",
        accomplished=accomplished,
        blocked=blocked,
        next_steps=next_steps,
    )

    if getattr(args, "json", False):
        print(json.dumps(packet.to_dict(), indent=2))
    else:
        print(f"handoff created: {packet.session_id}")
    return 0


def session_handoff_list_command(args: argparse.Namespace) -> int:
    """List handoff packets."""
    from ..handoff import HandoffStore

    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    context_path = _resolve_command_context(args)
    store = HandoffStore(context_path, config=manager.config)
    packets = store.list(limit=args.limit)

    if getattr(args, "json", False):
        print(json.dumps([p.to_dict() for p in packets], indent=2))
        return 0

    for p in packets:
        print(f"{p.timestamp[:19]}  {p.session_id}  {p.agent_name}")
    return 0


def session_handoff_read_command(args: argparse.Namespace) -> int:
    """Read a handoff packet."""
    from ..handoff import HandoffStore

    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    context_path = _resolve_command_context(args)
    store = HandoffStore(context_path, config=manager.config)
    packet = store.read(session_id=getattr(args, "session_id", None))

    if packet is None:
        print("no handoff packet found")
        return 1

    if getattr(args, "json", False):
        print(json.dumps(packet.to_dict(), indent=2))
    else:
        print(f"session_id: {packet.session_id}")
        print(f"agent: {packet.agent_name}")
        print(f"timestamp: {packet.timestamp}")
        if packet.accomplished:
            print("accomplished:")
            for item in packet.accomplished:
                print(f"  - {item}")
        if packet.blocked:
            print("blocked:")
            for item in packet.blocked:
                print(f"  - {item}")
        if packet.next_steps:
            print("next_steps:")
            for item in packet.next_steps:
                print(f"  - {item}")
    return 0


def session_replay_command(args: argparse.Namespace) -> int:
    """Replay session timeline."""
    from ..event_log import build_session_timeline

    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    context_path = _resolve_command_context(args)
    timeline = build_session_timeline(
        context_path,
        session_id=args.session_id,
        since=args.since,
        limit=args.limit,
        config=manager.config,
    )
    if args.json:
        print(json.dumps(timeline, indent=2))
        return 0
    if not timeline["timeline"]:
        print("(no events)")
        return 0
    for event in timeline["timeline"]:
        ts = event["timestamp"][:19] if event["timestamp"] else "?"
        print(f"  {ts} [{event['type']}] {event['summary']}")
    return 0


def session_replay_list_command(args: argparse.Namespace) -> int:
    """List available sessions."""
    from ..event_log import list_sessions

    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    context_path = _resolve_command_context(args)
    sessions = list_sessions(context_path, limit=args.limit, config=manager.config)
    if args.json:
        print(json.dumps(sessions, indent=2))
        return 0
    if not sessions:
        print("(no sessions)")
        return 0
    for session in sessions:
        print(f"  {session['session_id']}: {session['event_count']} events ({', '.join(session['event_types'][:3])})")
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


def agents_capabilities_command(args: argparse.Namespace) -> int:
    """List agent capabilities."""
    from ..agents import list_agents

    agents = list_agents()
    results = []
    for spec in agents:
        entry: dict[str, Any] = {"name": spec.name, "description": spec.description}
        if spec.capabilities:
            entry["capabilities"] = {
                "tools": spec.capabilities.tools,
                "topics": spec.capabilities.topics,
                "mount_types": spec.capabilities.mount_types,
                "description": spec.capabilities.description,
            }
        else:
            entry["capabilities"] = None
        if args.agent and spec.name != args.agent:
            continue
        results.append(entry)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    for entry in results:
        caps = entry["capabilities"]
        if caps:
            print(f"{entry['name']}:")
            print(f"  tools: {', '.join(caps['tools']) or '(none)'}")
            print(f"  topics: {', '.join(caps['topics']) or '(none)'}")
            print(f"  mounts: {', '.join(caps['mount_types']) or '(none)'}")
        else:
            print(f"{entry['name']}: (no capabilities declared)")
    return 0


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
    print("alias afs-studio=\"$AFS_ROOT/scripts/afs-studio\"")
    print("alias afs-studio-build=\"$AFS_ROOT/scripts/afs-studio-build\"")
    return 0


def _count_mount_files(mount_dir: Path) -> int:
    """Count files in a mount directory, including symlinked directories."""
    from ..context_index import count_mount_files

    return count_mount_files(mount_dir)


def _human_size(size_bytes: int) -> str:
    """Format bytes as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def status_command(args: argparse.Namespace) -> int:
    """Show AFS status."""
    from ..core import find_root, resolve_context_root
    from ..health.afs_status import _maintenance_health
    from ..manager import AFSManager
    from ..models import MountType

    start_dir = Path(args.start_dir).expanduser().resolve() if args.start_dir else None
    root = find_root(start_dir)
    config, _resolved_config_path = load_runtime_config_model(
        merge_user=True,
        start_dir=start_dir or Path.cwd(),
    )
    context_root = resolve_context_root(config, root)
    manager = AFSManager(config=config)

    mount_health = manager.context_health(context_root)
    maintenance = _maintenance_health(config, context_root)

    missing = list(mount_health["missing_dirs"])

    # Gather mount counts
    mount_counts: dict[str, int] = {}
    total_files = 0
    for mount_type in MountType:
        mount_dir = manager.resolve_mount_root(context_root, mount_type)
        count = _count_mount_files(mount_dir)
        if count > 0:
            mount_counts[mount_type.value] = count
            total_files += count

    # Gather index stats
    index_stats: dict[str, Any] = {"available": False}
    db_path = manager.resolve_mount_root(context_root, MountType.GLOBAL) / config.context_index.db_filename
    if config.context_index.enabled and db_path.exists():
        try:
            from ..context_index import ContextSQLiteIndex
            index = ContextSQLiteIndex(manager, context_root)
            has_entries = index.has_entries()
            index_stats = {
                "available": True,
                "db_path": str(db_path),
                "db_size": db_path.stat().st_size,
                "has_entries": has_entries,
                "stale": index.needs_refresh() if has_entries else False,
            }
            index_stats["total_entries"] = index.total_entries
        except Exception:
            pass

    # Active profile
    active_profile = config.profiles.active_profile

    if args.json:
        payload: dict[str, Any] = {
            "context_root": str(context_root),
            "linked_root": str(root) if root else None,
            "missing_dirs": missing,
            "valid": not missing,
            "active_profile": active_profile,
            "mount_counts": mount_counts,
            "total_files": total_files,
            "mount_health": mount_health,
            "index": index_stats,
            "maintenance": maintenance,
        }
        print(json.dumps(payload, indent=2))
        return 0

    # Pretty-print
    print(f"  context_root: {context_root}")
    print(f"  linked_root:  {root if root else '(none)'}")
    print(f"  profile:      {active_profile}")
    print(f"  valid:        {'yes' if not missing else 'no — missing: ' + ', '.join(missing)}")
    print()

    # Mount summary
    if mount_counts:
        print("  mounts:")
        for mount_name, count in sorted(mount_counts.items()):
            print(f"    {mount_name:12s}  {count:>5} files")
        print(f"    {'total':12s}  {total_files:>5} files")
    else:
        print("  mounts: (empty)")
    print()

    mount_issues = []
    if mount_health["broken_mounts"]:
        mount_issues.append(f"broken={len(mount_health['broken_mounts'])}")
    if mount_health["duplicate_mount_sources"]:
        mount_issues.append(f"duplicates={len(mount_health['duplicate_mount_sources'])}")
    profile_info = mount_health["profile"]
    if profile_info["missing_mounts"]:
        mount_issues.append(f"profile_missing={len(profile_info['missing_mounts'])}")
    if profile_info["missing_sources"]:
        mount_issues.append(f"profile_sources_missing={len(profile_info['missing_sources'])}")
    if profile_info["mismatched_mounts"]:
        mount_issues.append(f"profile_mismatched={len(profile_info['mismatched_mounts'])}")
    if mount_issues:
        print("  mount_health: " + ", ".join(mount_issues))
        if mount_health["suggested_actions"]:
            print("  actions:     " + "; ".join(mount_health["suggested_actions"]))
        print()

    # Index summary
    if index_stats.get("available"):
        stale_label = "stale" if index_stats.get("stale") else "fresh"
        db_size = _human_size(index_stats.get("db_size", 0))
        entries = index_stats.get("total_entries", "?")
        print(f"  index: {entries} entries, {db_size}, {stale_label}")
    elif config.context_index.enabled:
        print("  index: enabled but not yet built")
    else:
        print("  index: disabled")

    # Running agents
    supervisor_audit = maintenance.get("supervisor", {})
    counts = supervisor_audit.get("counts", {})
    print()
    print(
        "  agents: "
        f"running={counts.get('running', 0)} "
        f"failed={counts.get('failed', 0)} "
        f"stopped={counts.get('stopped', 0)} "
        f"manual_stop={counts.get('manual_stop', 0)}"
    )
    if supervisor_audit.get("stale_pid_files"):
        print("  agent_issues: " + ", ".join(supervisor_audit["stale_pid_files"]))

    warm = maintenance["reports"]["context_warm"]
    watch = maintenance["reports"]["context_watch"]
    agent_supervisor = maintenance["reports"]["agent_supervisor"]
    history_memory = maintenance["reports"]["history_memory"]
    print()
    print(
        "  maintenance: "
        f"context_warm={warm['status'] or 'unknown'} "
        f"context_watch={watch['status'] or 'unknown'} "
        f"agent_supervisor={agent_supervisor['status'] or 'unknown'} "
        f"history_memory={history_memory['status'] or 'unknown'} "
        f"degraded_contexts={maintenance['degraded_contexts']} "
        f"remapped_mounts={maintenance['remapped_mounts']}"
    )
    brief = maintenance["reports"]["gemini_workspace_brief"]
    if brief["available"]:
        print(
            "  gemini_brief: "
            f"{brief['status'] or 'unknown'} "
            f"age={brief['age_seconds'] if brief['age_seconds'] is not None else 'n/a'}s"
        )

    # Contextual hints based on state
    hints: list[str] = []
    if missing:
        hints.append("afs init --link-context  # fix missing mount dirs")
    if not mount_counts:
        hints.append("afs context discover --path .  # index this project")
    if not index_stats.get("available"):
        hints.append("afs context ensure-all --path .  # build context index")
    elif index_stats.get("stale"):
        hints.append("afs context ensure-all --path .  # refresh stale index")
    if counts.get("failed", 0) > 0:
        hints.append("afs agents ps --all  # check failed agents")
    if hints:
        print()
        for hint in hints:
            print(_hint(hint))

    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register core command parsers."""
    def add_context_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--config", help="Config path.")
        parser.add_argument("--path", help="Project path.")
        parser.add_argument("--context-root", help="Context root override.")
        parser.add_argument("--context-dir", help="Context directory name.")

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
    plugins_parser.add_argument(
        "--details", action="store_true", help="Show resolved plugin configuration."
    )
    plugins_parser.add_argument("--json", action="store_true", help="Output JSON.")
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
    services_list.add_argument("--config", help="Config path.")
    services_list.set_defaults(func=services_list_command)

    services_render = services_sub.add_parser("render", help="Render service unit.")
    services_render.add_argument("--config", help="Config path.")
    services_render.add_argument("name", help="Service name.")
    services_render.set_defaults(func=services_render_command)

    services_install = services_sub.add_parser("install", help="Install a managed service unit.")
    services_install.add_argument("--config", help="Config path.")
    services_install.add_argument("--enable", action="store_true", help="Enable/load the service after installing.")
    services_install.add_argument("name", help="Service name.")
    services_install.set_defaults(func=services_install_command)

    services_uninstall = services_sub.add_parser("uninstall", help="Remove a managed service unit.")
    services_uninstall.add_argument("--config", help="Config path.")
    services_uninstall.add_argument("--keep-enabled", action="store_true", help="Skip disable/unload before removing the unit file.")
    services_uninstall.add_argument("name", help="Service name.")
    services_uninstall.set_defaults(func=services_uninstall_command)

    services_enable = services_sub.add_parser("enable", help="Enable/load a managed service unit.")
    services_enable.add_argument("--config", help="Config path.")
    services_enable.add_argument("name", help="Service name.")
    services_enable.set_defaults(func=services_enable_command)

    services_disable = services_sub.add_parser("disable", help="Disable/unload a managed service unit.")
    services_disable.add_argument("--config", help="Config path.")
    services_disable.add_argument("name", help="Service name.")
    services_disable.set_defaults(func=services_disable_command)

    services_start = services_sub.add_parser("start", help="Start a service.")
    services_start.add_argument("--config", help="Config path.")
    services_start.add_argument("name", help="Service name.")
    services_start.add_argument("--foreground", "-f", action="store_true", help="Run in foreground.")
    services_start.set_defaults(func=services_start_command)

    services_stop = services_sub.add_parser("stop", help="Stop a service.")
    services_stop.add_argument("--config", help="Config path.")
    services_stop.add_argument("name", help="Service name.")
    services_stop.set_defaults(func=services_stop_command)

    services_status = services_sub.add_parser("status", help="Get service status.")
    services_status.add_argument("--config", help="Config path.")
    services_status.add_argument("--json", action="store_true", help="Output JSON.")
    services_status.add_argument("--system", action="store_true", help="Inspect installed OS-level service state.")
    services_status.add_argument("name", nargs="?", help="Service name (optional, shows all if omitted).")
    services_status.set_defaults(func=services_status_command)

    services_restart = services_sub.add_parser("restart", help="Restart a service.")
    services_restart.add_argument("--config", help="Config path.")
    services_restart.add_argument("name", help="Service name.")
    services_restart.set_defaults(func=services_restart_command)

    services_logs = services_sub.add_parser("logs", help="Show captured service logs.")
    services_logs.add_argument("--config", help="Config path.")
    services_logs.add_argument("--json", action="store_true", help="Output JSON.")
    services_logs.add_argument("--lines", type=int, default=50, help="Number of lines per log stream.")
    services_logs.add_argument("name", help="Service name.")
    services_logs.set_defaults(func=services_logs_command)

    # agents
    agents_parser = subparsers.add_parser("agents", help="Run built-in agents.")
    agents_sub = agents_parser.add_subparsers(dest="agents_command")

    agents_list = agents_sub.add_parser("list", help="List available agents.")
    agents_list.set_defaults(func=agents_list_command)

    agents_ps = agents_sub.add_parser("ps", help="List background agent processes.")
    agents_ps.add_argument("--config", help="Config path.")
    agents_ps.add_argument("--all", action="store_true", help="Include failed/stopped agent state.")
    agents_ps.add_argument("--json", action="store_true", help="Output JSON.")
    agents_ps.set_defaults(func=agents_ps_command)

    agents_watch = agents_sub.add_parser("watch", help="Show recent progress events for an agent.")
    add_context_args(agents_watch)
    agents_watch.add_argument("name", help="Agent name.")
    agents_watch.add_argument("--limit", type=int, default=20, help="Max events to show.")
    agents_watch.set_defaults(func=agents_watch_command)

    agents_run = agents_sub.add_parser("run", help="Run a built-in agent.")
    agents_run.add_argument("name", help="Agent name.")
    agents_run.add_argument(
        "agent_args",
        nargs=argparse.REMAINDER,
        help="Arguments for the agent (prefix with -- to pass through).",
    )
    agents_run.set_defaults(func=agents_run_command)

    agents_caps = agents_sub.add_parser("capabilities", help="List agent capabilities.")
    agents_caps.add_argument("--agent", help="Filter by agent name.")
    agents_caps.add_argument("--json", action="store_true", help="Output JSON.")
    agents_caps.set_defaults(func=agents_capabilities_command)

    # tasks
    tasks_parser = subparsers.add_parser("tasks", help="Task queue operations.")
    tasks_sub = tasks_parser.add_subparsers(dest="tasks_command")
    tasks_ls = tasks_sub.add_parser("list", help="List tasks.")
    add_context_args(tasks_ls)
    tasks_ls.add_argument("--status", help="Filter by status.")
    tasks_ls.add_argument("--json", action="store_true", help="Output JSON.")
    tasks_ls.set_defaults(func=tasks_list_command)

    # memory
    memory_parser = subparsers.add_parser("memory", help="Durable memory maintenance.")
    memory_sub = memory_parser.add_subparsers(dest="memory_command")
    memory_consolidate = memory_sub.add_parser(
        "consolidate",
        help="Consolidate recent history into durable memory entries.",
    )
    add_context_args(memory_consolidate)
    memory_consolidate.add_argument(
        "--max-events",
        type=int,
        help="Maximum history events to consolidate.",
    )
    memory_consolidate.add_argument(
        "--max-events-per-entry",
        type=int,
        help="Maximum history events per memory entry.",
    )
    memory_consolidate.add_argument(
        "--event-type",
        action="append",
        dest="event_types",
        help="Limit consolidation to specific history event types.",
    )
    memory_consolidate.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip writing markdown summaries.",
    )
    memory_consolidate.add_argument("--json", action="store_true", help="Output JSON.")
    memory_consolidate.set_defaults(func=memory_consolidate_command)

    memory_status_p = memory_sub.add_parser("status", help="Show memory pipeline status.")
    add_context_args(memory_status_p)
    memory_status_p.add_argument("--json", action="store_true", help="Output JSON.")
    memory_status_p.set_defaults(func=memory_status_command)

    memory_search_p = memory_sub.add_parser("search", help="Search memory entries.")
    add_context_args(memory_search_p)
    memory_search_p.add_argument("query", help="Search query.")
    memory_search_p.add_argument("--limit", type=int, default=10, help="Max results.")
    memory_search_p.add_argument("--json", action="store_true", help="Output JSON.")
    memory_search_p.set_defaults(func=memory_search_command)

    # session
    session_parser = subparsers.add_parser(
        "session",
        help="Session bootstrap and handoff helpers.",
    )
    session_sub = session_parser.add_subparsers(dest="session_command")
    session_bootstrap = session_sub.add_parser(
        "bootstrap",
        help="Build a startup packet from context health, scratchpad, tasks, hivemind, and memory.",
    )
    add_context_args(session_bootstrap)
    session_bootstrap.add_argument(
        "--task-limit",
        type=int,
        default=10,
        help="Maximum queued tasks to include.",
    )
    session_bootstrap.add_argument(
        "--message-limit",
        type=int,
        default=10,
        help="Maximum hivemind messages to include.",
    )
    session_bootstrap.add_argument(
        "--agent-name",
        help="Agent name for registration (default: cli).",
    )
    session_bootstrap.add_argument(
        "--no-write-artifacts",
        action="store_true",
        help="Do not update scratchpad/afs_agents/session_bootstrap.{json,md}.",
    )
    session_bootstrap.add_argument("--json", action="store_true", help="Output JSON.")
    session_bootstrap.set_defaults(func=session_bootstrap_command)

    session_pack = session_sub.add_parser(
        "pack",
        help="Build a token-budgeted context pack for Gemini, Claude, Codex, or generic clients.",
    )
    add_context_args(session_pack)
    session_pack.add_argument("query", nargs="?", help="Optional retrieval query.")
    session_pack.add_argument(
        "--task",
        help="Explicit task statement to place at the end of the rendered pack.",
    )
    session_pack.add_argument(
        "--model",
        default="generic",
        choices=["generic", "gemini", "claude", "codex"],
        help="Target model profile (default: generic).",
    )
    session_pack.add_argument(
        "--workflow",
        default="general",
        choices=[
            "general",
            "scan_fast",
            "edit_fast",
            "review_deep",
            "root_cause_deep",
        ],
        help="Execution workflow profile to encode into the pack.",
    )
    session_pack.add_argument(
        "--tool-profile",
        default="default",
        choices=[
            "default",
            "context_readonly",
            "context_repair",
            "edit_and_verify",
            "handoff_only",
        ],
        help="Preferred AFS surface mix to encode into the pack.",
    )
    session_pack.add_argument(
        "--token-budget",
        type=int,
        help="Approximate token budget (defaults depend on model).",
    )
    session_pack.add_argument(
        "--include-content",
        action="store_true",
        help="Include indexed file content instead of excerpts when available.",
    )
    session_pack.add_argument(
        "--max-query-results",
        type=int,
        default=6,
        help="Maximum indexed hits to include.",
    )
    session_pack.add_argument(
        "--max-embedding-results",
        type=int,
        default=4,
        help="Maximum embedding hits to include.",
    )
    session_pack.add_argument(
        "--no-write-artifacts",
        action="store_true",
        help="Do not update scratchpad/afs_agents/session_pack_<model>.{json,md}.",
    )
    session_pack.add_argument("--json", action="store_true", help="Output JSON.")
    session_pack.set_defaults(func=session_pack_command)

    session_handoff = session_sub.add_parser(
        "handoff", help="Create or read handoff packets."
    )
    session_handoff_sub = session_handoff.add_subparsers(dest="handoff_command")

    handoff_create = session_handoff_sub.add_parser("create", help="Create a handoff packet.")
    add_context_args(handoff_create)
    handoff_create.add_argument("--accomplished", help="Semicolon-separated accomplished items.")
    handoff_create.add_argument("--blocked", help="Semicolon-separated blocked items.")
    handoff_create.add_argument("--next", help="Semicolon-separated next steps.")
    handoff_create.add_argument("--agent-name", help="Agent name (default: cli).")
    handoff_create.add_argument("--json", action="store_true", help="Output JSON.")
    handoff_create.set_defaults(func=session_handoff_command)

    handoff_list = session_handoff_sub.add_parser("list", help="List handoff packets.")
    add_context_args(handoff_list)
    handoff_list.add_argument("--limit", type=int, default=10, help="Max packets.")
    handoff_list.add_argument("--json", action="store_true", help="Output JSON.")
    handoff_list.set_defaults(func=session_handoff_list_command)

    handoff_read = session_handoff_sub.add_parser("read", help="Read a handoff packet.")
    add_context_args(handoff_read)
    handoff_read.add_argument("--session-id", help="Session ID (latest if omitted).")
    handoff_read.add_argument("--json", action="store_true", help="Output JSON.")
    handoff_read.set_defaults(func=session_handoff_read_command)

    session_replay = session_sub.add_parser("replay", help="Replay session timeline.")
    add_context_args(session_replay)
    session_replay.add_argument("--session-id", help="Filter by session ID (date).")
    session_replay.add_argument("--since", help="Filter events after this datetime.")
    session_replay.add_argument("--limit", type=int, default=100, help="Max events.")
    session_replay.add_argument("--json", action="store_true", help="Output JSON.")
    session_replay.set_defaults(func=session_replay_command)

    session_replay_list = session_sub.add_parser("replay-list", help="List available sessions.")
    add_context_args(session_replay_list)
    session_replay_list.add_argument("--limit", type=int, default=20, help="Max sessions.")
    session_replay_list.add_argument("--json", action="store_true", help="Output JSON.")
    session_replay_list.set_defaults(func=session_replay_list_command)

    # hivemind
    hivemind_parser = subparsers.add_parser("hivemind", help="Inter-agent message bus.")
    hivemind_sub = hivemind_parser.add_subparsers(dest="hivemind_command")
    hivemind_ls = hivemind_sub.add_parser("list", help="List recent messages.")
    add_context_args(hivemind_ls)
    hivemind_ls.add_argument("--limit", type=int, default=20, help="Max messages.")
    hivemind_ls.add_argument("--topic", help="Filter by topic.")
    hivemind_ls.set_defaults(func=hivemind_list_command)

    hivemind_sub_cmd = hivemind_sub.add_parser("subscribe", help="Subscribe agent to topics.")
    add_context_args(hivemind_sub_cmd)
    hivemind_sub_cmd.add_argument("--agent", required=True, help="Agent name.")
    hivemind_sub_cmd.add_argument("--topics", required=True, help="Comma-separated topics.")
    hivemind_sub_cmd.add_argument("--ttl-hours", type=int, help="Optional subscription TTL window.")
    hivemind_sub_cmd.set_defaults(func=hivemind_subscribe_command)

    hivemind_unsub_cmd = hivemind_sub.add_parser("unsubscribe", help="Unsubscribe agent from topics.")
    add_context_args(hivemind_unsub_cmd)
    hivemind_unsub_cmd.add_argument("--agent", required=True, help="Agent name.")
    hivemind_unsub_cmd.add_argument("--topics", required=True, help="Comma-separated topics.")
    hivemind_unsub_cmd.set_defaults(func=hivemind_unsubscribe_command)

    hivemind_reap_cmd = hivemind_sub.add_parser(
        "reap",
        help="Remove expired or stale hivemind messages.",
    )
    add_context_args(hivemind_reap_cmd)
    hivemind_reap_cmd.add_argument("--max-age-hours", type=int, help="Override retention window.")
    hivemind_reap_cmd.add_argument("--dry-run", action="store_true", help="Report removals without deleting.")
    hivemind_reap_cmd.add_argument("--json", action="store_true", help="Output JSON.")
    hivemind_reap_cmd.set_defaults(func=hivemind_reap_command)

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
    studio_parser = subparsers.add_parser(
        "studio",
        help="AFS Studio tooling (build/install/path/alias).",
    )
    studio_sub = studio_parser.add_subparsers(dest="studio_command")

    studio_build_p = studio_sub.add_parser("build", help="Build AFS Studio.")
    studio_build_p.add_argument("--build-dir", help="Custom build directory.")
    studio_build_p.add_argument("--build-type", help="CMake build type.")
    studio_build_p.add_argument("--config", help="Build configuration (multi-config).")
    studio_build_p.set_defaults(func=studio_build_command)

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
