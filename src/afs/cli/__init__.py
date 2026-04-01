"""AFS command-line interface package.

Core ``afs`` exposes context/runtime primitives plus the built-in training
workflow surfaces that now depend on those primitives. Legacy benchmark,
gateway, and other domain-specific extensions remain extension-owned.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from ..config import load_runtime_config_model
from ..health import cli as health_cli
from ..history import log_cli_invocation
from ..profiles import resolve_active_profile
from . import (
    approvals,
    briefing,
    bundle,
    claude,
    context,
    core,
    doctor,
    embeddings,
    events,
    fs,
    gemini,
    gws_cli,
    mcp,
    profile,
    review,
    skills,
    training,
    watch,
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


def _path_within_roots(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.expanduser().resolve()
    for root in roots:
        candidate = root.expanduser().resolve()
        if resolved == candidate or resolved.is_relative_to(candidate):
            return True
    return False


def _purge_import_cache(module_name: str, search_roots: Iterable[Path]) -> None:
    roots = list(search_roots)
    related_names = {
        ".".join(parts)
        for parts in (
            module_name.split(".")[:index]
            for index in range(1, len(module_name.split(".")) + 1)
        )
    }
    prefixes = tuple(f"{name}." for name in related_names)
    for loaded_name, loaded_module in list(sys.modules.items()):
        if loaded_name not in related_names and not loaded_name.startswith(prefixes):
            continue
        module_file = getattr(loaded_module, "__file__", None)
        if isinstance(module_file, str) and _path_within_roots(Path(module_file), roots):
            continue
        sys.modules.pop(loaded_name, None)


def _register_cli_module(
    subparsers: argparse._SubParsersAction,
    module_name: str,
    *,
    search_roots: Iterable[Path] = (),
) -> None:
    roots = [root for root in search_roots if root.exists()]
    with _extension_import_path(roots[0]) if len(roots) == 1 else _multi_extension_import_path(roots):
        _purge_import_cache(module_name, roots)
        module = importlib.import_module(module_name)
    register = getattr(module, "register_parsers", None)
    if callable(register):
        register(subparsers)


@contextmanager
def _multi_extension_import_path(search_roots: Iterable[Path]) -> Iterable[None]:
    candidates: list[str] = []
    for root in search_roots:
        candidates.extend([str(root), str(root.parent)])
    original = list(sys.path)
    sys.path = [entry for entry in candidates if Path(entry).exists()] + original
    try:
        yield
    finally:
        sys.path = original


def _extract_config_argument(argv: Iterable[str] | None) -> Path | None:
    if argv is None:
        return None
    tokens = list(argv)
    for index, token in enumerate(tokens):
        if token == "--config" and index + 1 < len(tokens):
            return Path(tokens[index + 1]).expanduser().resolve()
        if token.startswith("--config="):
            _, _, value = token.partition("=")
            if value:
                return Path(value).expanduser().resolve()
    return None


def build_parser(argv: Iterable[str] | None = None) -> argparse.ArgumentParser:
    """Build the main argument parser."""
    parser = argparse.ArgumentParser(prog="afs")
    subparsers = parser.add_subparsers(dest="command")

    # Register core commands (init, plugins, status, services, agents, orchestrator, studio)
    core.register_parsers(subparsers)

    # Register approvals commands
    approvals.register_parsers(subparsers)

    # Register context commands (context, graph, workspace)
    context.register_parsers(subparsers)

    # Register filesystem commands
    fs.register_parsers(subparsers)

    # Register embedding commands
    embeddings.register_parsers(subparsers)

    # Register Gemini integration commands
    gemini.register_parsers(subparsers)

    # Register MCP server commands
    mcp.register_parsers(subparsers)

    # Register profile switching commands
    profile.register_parsers(subparsers)

    # Register briefing command
    briefing.register_parsers(subparsers)

    # Register Google Workspace commands
    gws_cli.register_parsers(subparsers)

    # Register review commands
    review.register_parsers(subparsers)

    # Register skill metadata commands
    skills.register_parsers(subparsers)

    # Register bundle commands
    bundle.register_parsers(subparsers)

    # Register health commands
    health_cli.register_parsers(subparsers)

    # Register events commands
    events.register_parsers(subparsers)

    # Register claude commands
    claude.register_parsers(subparsers)

    # Register doctor command
    doctor.register_parsers(subparsers)

    # Register training pipeline commands
    training.register_parsers(subparsers)

    # Register watch command
    watch.register_parsers(subparsers)

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

        config_path = _extract_config_argument(argv)
        config, _resolved_config_path = load_runtime_config_model(
            config_path=config_path,
            merge_user=True,
            start_dir=Path.cwd(),
        )
        plugins = load_enabled_plugins(config=config)
        call_plugin_hook("register_cli", subparsers, plugins=plugins.values())
        call_plugin_hook("register_parsers", subparsers, plugins=plugins.values())

        extensions = load_enabled_extensions(config=config)
        resolved_profile = resolve_active_profile(config)
        extension_roots: dict[str, list[Path]] = {}
        for extension in extensions.values():
            for module_name in extension.cli_modules:
                extension_roots.setdefault(module_name, []).append(extension.root)

        for module_name in resolved_profile.cli_modules:
            try:
                _register_cli_module(
                    subparsers,
                    module_name,
                    search_roots=extension_roots.get(module_name, []),
                )
            except Exception:
                continue
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
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser(argv_list)
    args = parser.parse_args(argv_list)

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

    try:
        exit_code = args.func(args)
    except ImportError as exc:
        print(f"afs: missing dependency: {exc}", file=sys.stderr)
        print("  hint: run `afs doctor` to diagnose", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"afs: file not found: {exc}", file=sys.stderr)
        print("  hint: run `afs doctor --fix` to repair", file=sys.stderr)
        return 1
    except PermissionError as exc:
        print(f"afs: permission denied: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"afs: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  hint: run `afs doctor` to diagnose", file=sys.stderr)
        return 1

    try:
        log_cli_invocation(argv_list, exit_code)
    except Exception:
        pass
    return exit_code


__all__ = ["build_parser", "main"]
