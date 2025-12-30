"""AFS command-line entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from .config import load_config_model
from .core import find_root, resolve_context_root
from .plugins import discover_plugins, load_plugins
from .schema import AFSConfig, GeneralConfig, WorkspaceDirectory


AFS_DIRS = [
    "memory",
    "knowledge",
    "history",
    "scratchpad",
    "tools",
    "hivemind",
    "global",
    "items",
]


def _ensure_context_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in AFS_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "workspaces").mkdir(parents=True, exist_ok=True)


def _write_config(path: Path, config: AFSConfig) -> None:
    general = config.general
    lines: list[str] = [
        "[general]",
        f"context_root = \"{general.context_root}\"",
        f"agent_workspaces_dir = \"{general.agent_workspaces_dir}\"",
    ]

    if general.workspace_directories:
        for ws in general.workspace_directories:
            lines.append("")
            lines.append("[[general.workspace_directories]]")
            lines.append(f"path = \"{ws.path}\"")
            if ws.description:
                lines.append(f"description = \"{ws.description}\"")

    lines.append("")
    lines.append("[cognitive]")
    lines.append(f"enabled = {str(config.cognitive.enabled).lower()}")
    lines.append(f"record_emotions = {str(config.cognitive.record_emotions).lower()}")
    lines.append(
        f"record_metacognition = {str(config.cognitive.record_metacognition).lower()}"
    )
    lines.append(f"record_goals = {str(config.cognitive.record_goals).lower()}")
    lines.append(f"record_epistemic = {str(config.cognitive.record_epistemic).lower()}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_config(
    context_root: Path,
    workspace_path: Path | None,
    workspace_name: str | None,
) -> AFSConfig:
    general = GeneralConfig()
    general.context_root = context_root
    general.agent_workspaces_dir = context_root / "workspaces"
    if workspace_path:
        general.workspace_directories = [
            WorkspaceDirectory(path=workspace_path, description=workspace_name)
        ]
    return AFSConfig(general=general)


def _init_command(args: argparse.Namespace) -> int:
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

    _ensure_context_root(context_root)

    if args.link_context:
        link_path = Path.cwd() / ".context"
        if not link_path.exists():
            link_path.symlink_to(context_root)

    if config_path:
        if config_path.exists() and not args.force:
            print(f"Config exists, not modified: {config_path}")
        else:
            config = _build_config(context_root, workspace_path, args.workspace_name)
            _write_config(config_path, config)
            print(f"Wrote config: {config_path}")

    return 0


def _plugins_command(args: argparse.Namespace) -> int:
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


def _status_command(args: argparse.Namespace) -> int:
    start_dir = Path(args.start_dir).expanduser().resolve() if args.start_dir else None
    root = find_root(start_dir)
    config = load_config_model()
    context_root = resolve_context_root(config, root)

    print(f"context_root: {context_root}")
    print(f"linked_root: {root if root else '(none)'}")

    missing = []
    for name in AFS_DIRS:
        if not (context_root / name).exists():
            missing.append(name)
    if missing:
        print("missing_dirs: " + ", ".join(missing))
    else:
        print("missing_dirs: (none)")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afs")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Initialize AFS context/root.")
    init_parser.add_argument("--context-root", help="Context root path.")
    init_parser.add_argument("--config", help="Path to write afs.toml.")
    init_parser.add_argument("--no-config", action="store_true", help="Do not write config.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite config if it exists.")
    init_parser.add_argument("--workspace-path", help="Workspace path to register.")
    init_parser.add_argument("--workspace-name", help="Workspace label/description.")
    init_parser.add_argument("--link-context", action="store_true", help="Symlink .context to context root.")
    init_parser.set_defaults(func=_init_command)

    plugins_parser = subparsers.add_parser("plugins", help="List or load plugins.")
    plugins_parser.add_argument("--config", help="Config path for plugin discovery.")
    plugins_parser.add_argument("--load", action="store_true", help="Attempt to import plugins.")
    plugins_parser.set_defaults(func=_plugins_command)

    status_parser = subparsers.add_parser("status", help="Show context root status.")
    status_parser.add_argument("--start-dir", help="Directory to search from.")
    status_parser.set_defaults(func=_status_command)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
