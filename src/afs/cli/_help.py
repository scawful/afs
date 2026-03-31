"""Custom CLI help rendering."""

from __future__ import annotations

import argparse
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..config import load_runtime_config_model, resolve_runtime_config_path
from ..schema import AFSConfig, GeneralConfig

_TOP_LEVEL_ORDER = [
    "fs",
    "context",
    "session",
    "workspace",
    "graph",
    "status",
    "init",
    "plugins",
    "services",
    "agents",
    "tasks",
    "hivemind",
    "orchestrator",
    "embeddings",
    "mcp",
    "profile",
    "skills",
    "bundle",
    "claude",
    "doctor",
    "health",
    "studio",
    "review",
    "help",
]

_TOP_LEVEL_COLORS = {
    "fs": "1;32",
    "context": "1;32",
    "session": "1;32",
    "workspace": "1;32",
    "graph": "1;32",
    "status": "1;34",
    "init": "1;34",
    "plugins": "1;34",
    "services": "1;34",
    "agents": "1;34",
    "tasks": "1;34",
    "hivemind": "1;34",
    "orchestrator": "1;34",
    "embeddings": "1;35",
    "mcp": "1;35",
    "profile": "1;35",
    "skills": "1;35",
    "bundle": "1;35",
    "claude": "1;35",
    "doctor": "1;35",
    "health": "1;35",
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
    local_config_path = resolve_runtime_config_path(start_dir=Path.cwd())

    lines: list[str] = []
    lines.append(_section("AFS CLI"))
    lines.append(f"Usage: {_cmd('afs')} <command> [options]")
    lines.append(
        _dim("Context roots, filesystem mounts, workspaces, profiles, hooks, and agent operations.")
    )
    lines.append("")

    lines.append(_section("Scope"))
    lines.extend(
        _format_list(
            [
                f"{_tag('fs', '32')} context roots, mounts, workspaces, graph",
                f"{_tag('ops', '34')} init/status/plugins/services/agents/tasks/hivemind",
                f"{_tag('agent', '35')} embeddings/mcp/profile/skills/bundle/doctor/health",
                f"{_tag('studio', '36')} studio build/install/path/alias",
            ]
        )
    )
    lines.append("")

    lines.append(_section("Extensions"))
    lines.extend(
        _format_list(
            [
                "Legacy personal/model-training, benchmark, gateway, and Claude commands are extension-owned.",
                "Enable an extension such as afs-scawful to restore personal/domain command groups.",
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
                _cmd("afs session bootstrap"),
                _cmd("afs doctor"),
                _cmd(
                    f"afs init --context-root {_format_path(context_root)} --workspace-name {workspace_label}"
                ),
                _cmd(f"afs context discover --path {_format_path(workspace_root)}"),
                _cmd(f"afs context ensure-all --path {_format_path(workspace_root)}"),
                _cmd("afs profile current"),
                _cmd("afs health"),
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
                _cmd("afs session bootstrap --json"),
                _cmd("afs doctor --fix"),
                _cmd("afs status --json"),
            ]
        )
    )
    lines.append("")

    lines.append(_section("Shell Integration"))
    lines.extend(
        _format_list(
            [
                f"{_cmd('source scripts/afs-shell-init.sh')}  # aliases, completions, helpers",
                f"{_dim('a=afs  as=status  ab=session-bootstrap  ap=agents-ps  tl=tasks  hm=hivemind  sk=skills')}",
                f"{_dim('afs-here  afs-bootstrap  afs-find  afs-watch  afs-spawn  afs-task  afs-say')}",
                f"{_dim('afs-gemini  afs-claude  afs-codex  # client launchers with AFS bootstrap')}",
                f"{_dim('afs-check  # lint + tests helper')}",
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
            suggestions = _suggest_command_paths(parser, tokens)
            if suggestions:
                print("Closest matches:")
                width = max(len(" ".join(match["path"])) for match in suggestions)
                for match in suggestions:
                    label = " ".join(match["path"]).ljust(width)
                    description = match.get("help") or ""
                    if description:
                        print(f"  - {label}  {description}")
                    else:
                        print(f"  - {label}")
            print("Use 'afs help' to list commands.")
            return 1
        current = sub_action.choices[token]

    current.print_help()
    return 0


def _safe_load_config(config: AFSConfig | None) -> tuple[AFSConfig | None, str | None]:
    if config is not None:
        return config, None
    try:
        loaded, _resolved_path = load_runtime_config_model(
            merge_user=True,
            start_dir=Path.cwd(),
        )
        return loaded, None
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


def _flatten_command_tree(
    tree: dict[str, dict[str, Any]],
    prefix: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    keys = _sort_keys(list(tree.keys()), _TOP_LEVEL_ORDER if not prefix else None)
    for key in keys:
        node = tree[key]
        path = prefix + (key,)
        entries.append({"path": path, "help": node.get("help") or ""})
        children = node.get("children") or {}
        if children:
            entries.extend(_flatten_command_tree(children, path))
    return entries


def _suggest_command_paths(
    parser: argparse.ArgumentParser,
    tokens: list[str],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    query_tokens = [token.lower() for token in tokens if token]
    if not query_tokens:
        return []

    query = " ".join(query_tokens)
    entries = _flatten_command_tree(_build_command_tree(parser))
    scored: list[tuple[tuple[int, int, float, int, str], dict[str, Any]]] = []

    for entry in entries:
        path = tuple(entry["path"])
        path_text = " ".join(path)
        path_key = path_text.lower()
        leaf_key = path[-1].lower()
        help_key = str(entry.get("help") or "").lower()
        path_parts = [part.lower() for part in path]

        full_prefix = path_key.startswith(query)
        leaf_prefix = leaf_key.startswith(query_tokens[-1])
        token_prefix = all(
            any(part.startswith(token) for part in path_parts)
            for token in query_tokens
        )
        token_contains = all(
            token in path_key or token in help_key for token in query_tokens
        )
        similarity = max(
            SequenceMatcher(None, query, path_key).ratio(),
            SequenceMatcher(None, query_tokens[-1], leaf_key).ratio(),
        )

        if not (full_prefix or leaf_prefix or token_prefix or token_contains or similarity >= 0.55):
            continue

        if full_prefix:
            tier = 0
        elif leaf_prefix:
            tier = 1
        elif token_prefix:
            tier = 2
        elif token_contains:
            tier = 3
        else:
            tier = 4

        score = (
            tier,
            -sum(1 for part, token in zip(path_parts, query_tokens, strict=False) if part == token),
            -similarity,
            len(path),
            path_text,
        )
        scored.append((score, entry))

    scored.sort(key=lambda item: item[0])

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for _score, entry in scored:
        path = tuple(entry["path"])
        if path in seen:
            continue
        seen.add(path)
        unique.append(entry)
        if len(unique) >= limit:
            break
    return unique


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
