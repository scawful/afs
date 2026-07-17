"""Plain-language entry points over stable AFS APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..context_layout import LAYOUT_VERSION, detect_layout_version
from ..messages import MessageBus
from ..project_registry import COMMON_SCOPE_ID, ProjectRecord, ProjectRegistry
from ._utils import load_manager, resolve_context_paths


def _add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Config path.")
    parser.add_argument("--path", help="Project or working path.")
    parser.add_argument("--context-root", help="Context root override.")
    parser.add_argument("--context-dir", help="Context directory name.")


def _manager_context(args: argparse.Namespace):
    config = Path(args.config).expanduser().resolve() if getattr(args, "config", None) else None
    manager = load_manager(config)
    project, context, _override, _directory = resolve_context_paths(args, manager)
    return manager, project, context


def _scope_for(
    context: Path,
    project: Path,
) -> tuple[str, ProjectRecord | None]:
    if detect_layout_version(context) != LAYOUT_VERSION:
        return COMMON_SCOPE_ID, None
    record = ProjectRegistry(context).resolve(project)
    if record is None:
        raise PermissionError(f"project is not registered in central context: {project}")
    return record.scope_id, record


def start_command(args: argparse.Namespace) -> int:
    """Build the normal session-start packet without exposing nested jargon."""
    from .core import session_bootstrap_command

    return session_bootstrap_command(args)


def projects_current_command(args: argparse.Namespace) -> int:
    manager, project, context = _manager_context(args)
    scope_id, record = _scope_for(context, project)
    payload: dict[str, Any] = {
        "context_root": str(context),
        "layout_version": detect_layout_version(context),
        "project_path": str(project),
        "registered": record is not None,
        "scope_id": scope_id,
        "project": record.to_dict() if record else None,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"context_root: {payload['context_root']}")
        print(f"layout_version: {payload['layout_version']}")
        print(f"project_path: {payload['project_path']}")
        print(f"scope_id: {payload['scope_id']}")
        print(f"registered: {str(payload['registered']).lower()}")
    return 0


def _registry_from_args(args: argparse.Namespace) -> ProjectRegistry:
    config = Path(args.config).expanduser().resolve() if getattr(args, "config", None) else None
    manager = load_manager(config)
    root = (
        Path(args.context_root).expanduser().resolve()
        if getattr(args, "context_root", None)
        else manager.config.general.context_root.expanduser().resolve()
    )
    return ProjectRegistry(root)


def projects_list_command(args: argparse.Namespace) -> int:
    registry = _registry_from_args(args)
    records = registry.all_records()
    payload = [record.to_dict() for record in records]
    if args.json:
        print(json.dumps(payload, indent=2))
    elif not records:
        print("no registered projects")
    else:
        for record in records:
            print(f"{record.project_id}\t{record.name}\t{record.path}")
    return 0


def projects_register_command(args: argparse.Namespace) -> int:
    registry = _registry_from_args(args)
    project = Path(args.project_path).expanduser().resolve()
    record = registry.register(project, name=args.name)
    if args.json:
        print(json.dumps(record.to_dict(), indent=2))
    else:
        print(f"registered: {record.project_id}")
        print(f"scope_id: {record.scope_id}")
        print(f"path: {record.path}")
    return 0


def _message_bus(args: argparse.Namespace) -> MessageBus:
    manager, project, context = _manager_context(args)
    scope_id, _record = _scope_for(context, project)
    return MessageBus(
        context,
        scope_id=scope_id,
        config=manager.config,
        all_projects=bool(getattr(args, "all_projects", False)),
        include_legacy=bool(getattr(args, "include_legacy", False)),
    )


def messages_list_command(args: argparse.Namespace) -> int:
    bus = _message_bus(args)
    messages = bus.read(
        agent_name=args.agent,
        msg_type=args.type,
        topic=args.topic,
        limit=args.limit,
    )
    payload = [message.to_dict() for message in messages]
    if args.json:
        print(json.dumps(payload, indent=2))
    elif not messages:
        print("no messages")
    else:
        for message in messages:
            target = f" -> {message.to}" if message.to else ""
            topic = f" #{message.topic}" if message.topic else ""
            print(
                f"{message.timestamp[:19]}  [{message.msg_type}]  "
                f"{message.from_agent}{target}{topic}  ({message.scope_id or 'legacy'})"
            )
    return 0


def _payload(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--payload must be a JSON object")
    return parsed


def messages_send_command(args: argparse.Namespace) -> int:
    bus = _message_bus(args)
    message = bus.send(
        args.from_agent,
        args.type,
        _payload(args.payload),
        to=args.to,
        topic=args.topic,
        ttl_hours=args.ttl_hours,
        scope_id=args.scope,
    )
    if args.json:
        print(json.dumps(message.to_dict(), indent=2))
    else:
        print(f"sent: {message.id}")
        print(f"scope_id: {message.scope_id}")
    return 0


def messages_subscribe_command(args: argparse.Namespace) -> int:
    bus = _message_bus(args)
    subscription = bus.subscribe(
        args.agent,
        args.topic,
        ttl_hours=args.ttl_hours,
    )
    if args.json:
        print(json.dumps(subscription.to_dict(), indent=2))
    else:
        print(f"subscribed {subscription.agent_name}: {', '.join(subscription.topics)}")
    return 0


def messages_unsubscribe_command(args: argparse.Namespace) -> int:
    bus = _message_bus(args)
    subscription = bus.unsubscribe(args.agent, args.topic)
    if args.json:
        print(json.dumps(subscription.to_dict(), indent=2))
    else:
        remaining = ", ".join(subscription.topics) or "(none)"
        print(f"subscribed topics for {subscription.agent_name}: {remaining}")
    return 0


def messages_clean_command(args: argparse.Namespace) -> int:
    if not args.all_projects:
        raise PermissionError(
            "message cleanup is queue-wide; pass --all-projects to authorize it"
        )
    bus = _message_bus(args)
    result = bus.reap(
        max_age_hours=args.max_age_hours,
        dry_run=not args.apply,
    )
    result["applied"] = bool(args.apply)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        mode = "applied" if args.apply else "dry-run"
        print(f"mode: {mode}")
        print(f"would_remove: {result['removed_count']}")
        print(f"remaining: {result['remaining_count']}")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    start = subparsers.add_parser(
        "start",
        help="Start or resume work with the current scoped context.",
    )
    _add_context_args(start)
    start.add_argument("--task-limit", type=int, default=10)
    start.add_argument("--message-limit", type=int, default=10)
    start.add_argument("--agent-name", default="cli")
    start.add_argument("--skills-prompt", default="")
    start.add_argument("--skills-top-k", type=int, default=5)
    start.add_argument("--no-write-artifacts", action="store_true")
    start.add_argument("--engage", action="store_true")
    start.add_argument("--json", action="store_true")
    start.set_defaults(func=start_command)

    projects = subparsers.add_parser(
        "projects",
        help="Inspect projects registered with the central context.",
    )
    project_commands = projects.add_subparsers(dest="projects_command")
    current = project_commands.add_parser("current", help="Show the current project and scope.")
    _add_context_args(current)
    current.add_argument("--json", action="store_true")
    current.set_defaults(func=projects_current_command)
    listing = project_commands.add_parser("list", help="List project registry metadata.")
    listing.add_argument("--config")
    listing.add_argument("--context-root")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=projects_list_command)
    register = project_commands.add_parser("register", help="Register a project path.")
    register.add_argument("project_path")
    register.add_argument("--name")
    register.add_argument("--config")
    register.add_argument("--context-root")
    register.add_argument("--json", action="store_true")
    register.set_defaults(func=projects_register_command)

    messages = subparsers.add_parser(
        "messages",
        help="Send and read scoped inter-agent messages.",
    )
    message_commands = messages.add_subparsers(dest="messages_command")

    def add_message_context(parser: argparse.ArgumentParser) -> None:
        _add_context_args(parser)
        parser.add_argument(
            "--all-projects",
            action="store_true",
            help="Explicitly authorize all registered project scopes.",
        )
        parser.add_argument(
            "--include-legacy",
            action="store_true",
            help="Include unscoped compatibility records.",
        )

    listing = message_commands.add_parser("list", help="List visible messages.")
    add_message_context(listing)
    listing.add_argument("--agent")
    listing.add_argument("--type")
    listing.add_argument("--topic")
    listing.add_argument("--limit", type=int, default=50)
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=messages_list_command)

    send = message_commands.add_parser("send", help="Send a scoped message.")
    add_message_context(send)
    send.add_argument("--from", dest="from_agent", required=True)
    send.add_argument("--type", default="status")
    send.add_argument("--payload", help="JSON object payload.")
    send.add_argument("--to")
    send.add_argument("--topic")
    send.add_argument("--ttl-hours", type=int)
    send.add_argument("--scope", help="Destination scope (default: current project).")
    send.add_argument("--json", action="store_true")
    send.set_defaults(func=messages_send_command)

    subscribe = message_commands.add_parser("subscribe", help="Subscribe to topics.")
    add_message_context(subscribe)
    subscribe.add_argument("--agent", required=True)
    subscribe.add_argument("--topic", action="append", required=True)
    subscribe.add_argument("--ttl-hours", type=int)
    subscribe.add_argument("--json", action="store_true")
    subscribe.set_defaults(func=messages_subscribe_command)

    unsubscribe = message_commands.add_parser("unsubscribe", help="Unsubscribe from topics.")
    add_message_context(unsubscribe)
    unsubscribe.add_argument("--agent", required=True)
    unsubscribe.add_argument("--topic", action="append", required=True)
    unsubscribe.add_argument("--json", action="store_true")
    unsubscribe.set_defaults(func=messages_unsubscribe_command)

    clean = message_commands.add_parser("clean", help="Preview or apply retention cleanup.")
    add_message_context(clean)
    clean.add_argument("--max-age-hours", type=int)
    clean.add_argument("--apply", action="store_true", help="Apply removals; default is dry-run.")
    clean.add_argument("--json", action="store_true")
    clean.set_defaults(func=messages_clean_command)
