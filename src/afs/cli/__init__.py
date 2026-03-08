"""AFS command-line interface package.

Core `afs` exposes filesystem/context/runtime primitives.
Legacy model-training, benchmark, and gateway surfaces are extension-owned and
should be reintroduced by extension manifests such as `afs-scawful`.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from ..health import cli as health_cli
from ..history import log_cli_invocation
from . import (
    claude,
    context,
    core,
    embeddings,
    fs,
    mcp,
    profile,
    review,
    skills,
)
from ._help import render_default_help, render_topic_help


@contextmanager
def _extension_import_path(extension_root: Path) -> Iterable[None]:
    candidates = [
        str(extension_root),
        str(extension_root.parent),
    ]
    original = list(sys.path)
    sys.path = [entry for entry in candidates if Path(entry).exists()] + original
    try:
        yield
    finally:
        sys.path = original


def build_parser() -> argparse.ArgumentParser:
    """Build the main argument parser."""
    parser = argparse.ArgumentParser(prog="afs")
    subparsers = parser.add_subparsers(dest="command")

    # Register core commands (init, plugins, status, services, agents, orchestrator, studio)
    core.register_parsers(subparsers)

    # Register context commands (context, graph, workspace)
    context.register_parsers(subparsers)

    # Register filesystem commands
    fs.register_parsers(subparsers)

    # Register embedding commands
    embeddings.register_parsers(subparsers)

    # Register MCP server commands
    mcp.register_parsers(subparsers)

    # Register profile switching commands
    profile.register_parsers(subparsers)

    # Register review commands
    review.register_parsers(subparsers)

    # Register skill metadata commands
    skills.register_parsers(subparsers)

    # Register Claude log analysis commands
    claude.register_parsers(subparsers)

    # Register health commands
    health_cli.register_parsers(subparsers)

    # help
    help_parser = subparsers.add_parser("help", help="Show help for commands.")
    help_parser.add_argument(
        "topic",
        nargs=argparse.REMAINDER,
        help="Command path to show help for (e.g., context init).",
    )
    help_parser.set_defaults(
        func=lambda args, root_parser=parser: render_topic_help(
            root_parser, args.topic
        )
    )

    # Allow plugins to extend the CLI surface.
    try:
        from ..plugins import (
            call_plugin_hook,
            load_enabled_extensions,
            load_enabled_plugins,
        )

        plugins = load_enabled_plugins()
        call_plugin_hook("register_cli", subparsers, plugins=plugins.values())
        call_plugin_hook("register_parsers", subparsers, plugins=plugins.values())

        extensions = load_enabled_extensions()
        for extension in extensions.values():
            for module_name in extension.cli_modules:
                try:
                    with _extension_import_path(extension.root):
                        module = importlib.import_module(module_name)
                except Exception:
                    continue
                register = getattr(module, "register_parsers", None)
                if callable(register):
                    register(subparsers)
    except Exception:
        pass

    return parser


def _get_subparser_action(
    parser: argparse.ArgumentParser,
) -> argparse._SubParsersAction | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _get_command_parser(
    parser: argparse.ArgumentParser, command: str
) -> argparse.ArgumentParser | None:
    sub_action = _get_subparser_action(parser)
    if not sub_action:
        return None
    return sub_action.choices.get(command)


def _requires_subcommand(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> bool:
    if getattr(args, "_allow_missing_subcommand", False):
        return False
    command = getattr(args, "command", None)
    if not command:
        return False
    command_parser = _get_command_parser(parser, command)
    if not command_parser:
        return False
    sub_action = _get_subparser_action(command_parser)
    if not sub_action:
        return False
    return not getattr(args, sub_action.dest, None)


def main(argv: Iterable[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        render_default_help(parser)
        return 1

    if _requires_subcommand(parser, args):
        command_parser = _get_command_parser(parser, args.command)
        if command_parser:
            command_parser.print_help()
        else:
            parser.print_help()
        return 1

    exit_code = args.func(args)
    try:
        log_cli_invocation(argv or sys.argv[1:], exit_code)
    except Exception:
        pass
    return exit_code


__all__ = ["build_parser", "main"]
