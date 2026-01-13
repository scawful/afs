"""Custom CLI help rendering."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from ..config import load_config_model
from ..schema import AFSConfig, GeneralConfig


_TOP_LEVEL_ORDER = [
    "fs",
    "context",
    "workspace",
    "graph",
    "status",
    "init",
    "plugins",
    "services",
    "agents",
    "orchestrator",
    "gateway",
    "vastai",
    "embeddings",
    "training",
    "discriminator",
    "tokenizer",
    "encoder",
    "entity",
    "generators",
    "generator",
    "active-learning",
    "scoring",
    "pipeline",
    "evaluation",
    "benchmark",
    "distill",
    "studio",
    "review",
    "help",
]

_TOP_LEVEL_COLORS = {
    "fs": "1;32",
    "context": "1;32",
    "workspace": "1;32",
    "graph": "1;32",
    "status": "1;34",
    "init": "1;34",
    "plugins": "1;34",
    "services": "1;34",
    "agents": "1;34",
    "orchestrator": "1;34",
    "gateway": "1;33",
    "vastai": "1;33",
    "embeddings": "1;35",
    "training": "1;35",
    "discriminator": "1;35",
    "tokenizer": "1;35",
    "encoder": "1;35",
    "entity": "1;35",
    "generators": "1;35",
    "generator": "1;35",
    "active-learning": "1;35",
    "scoring": "1;35",
    "pipeline": "1;35",
    "evaluation": "1;35",
    "benchmark": "1;35",
    "distill": "1;35",
    "studio": "1;36",
    "review": "1;36",
    "help": "1;36",
}


def render_default_help(parser: argparse.ArgumentParser, config: AFSConfig | None = None) -> None:
    """Render the default (no-args) help screen."""
    config, config_error = _safe_load_config(config)
    general = config.general if config else GeneralConfig()

    context_root = general.context_root
    workspaces = list(general.workspace_directories or [])
    workspace_root = _guess_workspace_root(workspaces)
    workspace_label = workspace_root.name or "workspace"

    env_config = os.getenv("AFS_CONFIG_PATH")
    user_config_path = Path.home() / ".config" / "afs" / "config.toml"
    local_config_path = Path.cwd() / "afs.toml"

    lines: list[str] = []
    lines.append(_section("AFS CLI"))
    lines.append(f"Usage: {_cmd('afs')} <command> [options]")
    lines.append(
        _dim("Context roots, filesystem mounts, workspaces, services, training, gateway.")
    )
    lines.append("")

    lines.append(_section("Scope"))
    lines.extend(
        _format_list(
            [
                f"{_tag('fs', '32')} context roots, mounts, workspaces, graph",
                f"{_tag('ops', '34')} init/status/plugins/services/agents/orchestrator",
                f"{_tag('train', '35')} training/pipeline/evaluation/generators",
                f"{_tag('gateway', '33')} gateway/vastai backends",
                f"{_tag('studio', '36')} studio build/install/path/alias",
            ]
        )
    )
    lines.append("")

    lines.append(_section("Filesystem First"))
    lines.extend(
        _format_list(
            [
                f"{_cmd('afs fs list')} <mount> --relative <path>",
                f"{_cmd('afs fs read')} <mount> <path>",
                f"{_cmd('afs fs write')} <mount> <path> --content ...",
                f"{_cmd('afs context discover')} --path <root>",
                f"{_cmd('afs context ensure-all')} --path <root>",
                f"{_cmd('afs workspace sync')} --root <root>",
            ]
        )
    )
    lines.append("")

    lines.append(_section("Defaults"))
    lines.extend(
        _format_kv(
            [
                ("context_root", _format_path(context_root), _path_status(context_root)),
                (
                    "agent_workspaces_dir",
                    _format_path(general.agent_workspaces_dir),
                    _path_status(general.agent_workspaces_dir),
                ),
                ("workspace_root", _format_path(workspace_root), _path_status(workspace_root)),
                ("workspaces", str(len(workspaces)), ""),
                ("discovery_ignore", ", ".join(general.discovery_ignore), ""),
            ]
        )
    )
    lines.append("")

    lines.append(_section("Config Sources"))
    if env_config:
        lines.append(f"  env:   AFS_CONFIG_PATH={env_config}")
    lines.append(
        f"  user:  {_format_path(user_config_path)} {_path_status(user_config_path)}"
    )
    lines.append(
        f"  local: {_format_path(local_config_path)} {_path_status(local_config_path)}"
    )
    if config_error:
        lines.append(f"  note:  config load failed ({config_error})")
    lines.append("")

    lines.append(_section("Quickstart"))
    lines.extend(
        _format_list(
            [
                _cmd("afs status"),
                _cmd(
                    f"afs init --context-root {_format_path(context_root)} --workspace-name {workspace_label}"
                ),
                _cmd(f"afs context discover --path {_format_path(workspace_root)}"),
                _cmd(f"afs context ensure-all --path {_format_path(workspace_root)}"),
                _cmd(f"afs fs list memory --path {_format_path(workspace_root)}"),
                _cmd(f"afs workspace sync --root {_format_path(workspace_root)}"),
            ]
        )
    )
    lines.append("")

    lines.append(_section("Command Tree"))
    tree = _build_command_tree(parser)
    lines.extend(_render_tree(tree))
    lines.append("")

    lines.append(_section("App Launchers"))
    lines.extend(
        _format_list(
            [
                f"{_cmd('afs-studio')}  # build + run",
                f"{_cmd('afs-studio-build')}  # build only",
                f"Use {_cmd('afs studio alias')} to print shell aliases.",
            ]
        )
    )
    lines.append("")

    lines.append(_section("Tips"))
    lines.extend(
        _format_list(
            [
                f"{_cmd('afs help <command>')}            # or: {_cmd('afs <command> --help')}",
                f"{_cmd('afs context discover --json')}   # agent-friendly output",
                _cmd("afs status --json"),
            ]
        )
    )

    sys.stdout.write("\n".join(lines) + "\n")


def render_topic_help(parser: argparse.ArgumentParser, topic: list[str] | None) -> int:
    """Render help for a specific command path."""
    if not topic:
        render_default_help(parser)
        return 0

    tokens = [t for t in topic if t != "--"]
    current = parser
    for token in tokens:
        sub_action = _get_subparser_action(current)
        if not sub_action or token not in sub_action.choices:
            print(f"Unknown command: {' '.join(tokens)}")
            print("Use 'afs help' to list commands.")
            return 1
        current = sub_action.choices[token]

    current.print_help()
    return 0


def _safe_load_config(config: AFSConfig | None) -> tuple[AFSConfig | None, str | None]:
    if config is not None:
        return config, None
    try:
        return load_config_model(merge_user=True), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, str(exc)


def _guess_workspace_root(workspaces: list[Any]) -> Path:
    if workspaces:
        return Path(workspaces[0].path)
    candidate = Path.home() / "src"
    if candidate.exists():
        return candidate
    return Path.cwd()


def _format_path(path: Path | str) -> str:
    raw = str(path)
    home = str(Path.home())
    if raw.startswith(home):
        return "~" + raw[len(home):]
    return raw


def _path_status(path: Path | str | None) -> str:
    if not path:
        return _dim("(unknown)")
    try:
        exists = Path(path).exists()
    except OSError:
        exists = False
    if exists:
        return _ok("(ok)")
    return _warn("(missing)")


def _format_list(items: list[str]) -> list[str]:
    return [f"  - {item}" for item in items]


def _format_kv(rows: list[tuple[str, str, str]]) -> list[str]:
    key_width = max(len(key) for key, _val, _status in rows) if rows else 0
    lines = []
    for key, val, status in rows:
        pad = key.ljust(key_width)
        if status:
            lines.append(f"  {pad}  {val} {status}")
        else:
            lines.append(f"  {pad}  {val}")
    return lines


def _build_command_tree(parser: argparse.ArgumentParser) -> dict[str, dict[str, Any]]:
    sub_action = _get_subparser_action(parser)
    if not sub_action:
        return {}

    help_map = {
        action.dest: action.help
        for action in getattr(sub_action, "_choices_actions", [])
        if action.dest
    }

    tree: dict[str, dict[str, Any]] = {}
    for name, subparser in sub_action.choices.items():
        tree[name] = {
            "help": help_map.get(name, ""),
            "children": _build_command_tree(subparser),
        }

    return tree


def _get_subparser_action(
    parser: argparse.ArgumentParser,
) -> argparse._SubParsersAction | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _render_tree(tree: dict[str, dict[str, Any]], indent: str = "") -> list[str]:
    if not tree:
        return ["  (no commands registered)"]

    lines: list[str] = []
    keys = _sort_keys(list(tree.keys()), _TOP_LEVEL_ORDER if not indent else None)
    width = max(len(key) for key in keys) if keys else 0

    for key in keys:
        node = tree[key]
        description = node.get("help") or ""
        label = key.ljust(width)
        if not indent:
            color = _TOP_LEVEL_COLORS.get(key)
            if color:
                label = _style(label, color)
        if description:
            lines.append(f"{indent}{label}  {description}")
        else:
            lines.append(f"{indent}{label}")
        children = node.get("children") or {}
        if children:
            lines.extend(_render_tree(children, indent + "  "))

    return ["  " + line for line in lines]


def _sort_keys(keys: list[str], preferred: list[str] | None) -> list[str]:
    if not preferred:
        return sorted(keys)
    preferred_set = [key for key in preferred if key in keys]
    remaining = sorted(key for key in keys if key not in preferred_set)
    return preferred_set + remaining


def _section(title: str) -> str:
    return _style(title, "1;36")


def _ok(text: str) -> str:
    return _style(text, "32")


def _warn(text: str) -> str:
    return _style(text, "33")


def _dim(text: str) -> str:
    return _style(text, "2")


def _cmd(text: str) -> str:
    return _style(text, "1;34")


def _tag(text: str, color: str) -> str:
    return _style(f"[{text}]", color)


def _style(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def _supports_color() -> bool:
    if not sys.stdout.isatty():
        return False
    if os.getenv("NO_COLOR") is not None:
        return False
    return True
