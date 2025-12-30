"""AFS command-line entry points."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Iterable

from .config import load_config, load_config_model
from .core import find_root, resolve_context_root
from .discovery import discover_contexts, get_project_stats
from .graph import build_graph, default_graph_path, write_graph
from .manager import AFSManager
from .models import MountType
from .orchestration import Orchestrator, TaskRequest
from .plugins import discover_plugins, load_plugins
from .services import ServiceManager
from .schema import AFSConfig, GeneralConfig, WorkspaceDirectory
from .validator import AFSValidator


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


def _parse_mount_type(value: str) -> MountType:
    try:
        return MountType(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Unknown mount type: {value}") from exc


def _resolve_studio_root() -> Path:
    candidates: list[Path] = []
    env_root = os.getenv("AFS_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    trunk_root = os.getenv("TRUNK_ROOT")
    if trunk_root:
        candidates.append(Path(trunk_root).expanduser().resolve() / "lab" / "afs")
    candidates.append(Path(__file__).resolve().parents[2])

    for candidate in candidates:
        if (candidate / "apps" / "studio" / "CMakeLists.txt").exists():
            return candidate

    raise FileNotFoundError(
        "AFS studio source not found. Set AFS_ROOT to the repo root."
    )


def _studio_binary_name() -> str:
    return "afs_studio.exe" if os.name == "nt" else "afs_studio"


def _studio_build_dir(root: Path, override: str | None) -> Path:
    return (
        Path(override).expanduser().resolve()
        if override
        else root / "build" / "studio"
    )


def _studio_binary_path(build_dir: Path, config: str | None) -> Path:
    if config:
        candidate = build_dir / config / _studio_binary_name()
        if candidate.exists():
            return candidate
    return build_dir / _studio_binary_name()


def _run_command(cmd: list[str]) -> int:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(f"command not found: {cmd[0]}")
        return 1
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


def _studio_build(
    root: Path,
    build_dir: Path,
    build_type: str | None,
    config: str | None,
) -> int:
    src_dir = root / "apps" / "studio"
    cmake_cmd = ["cmake", "-S", str(src_dir), "-B", str(build_dir)]
    if build_type:
        cmake_cmd.append(f"-DCMAKE_BUILD_TYPE={build_type}")
    status = _run_command(cmake_cmd)
    if status != 0:
        return status
    build_cmd = ["cmake", "--build", str(build_dir), "--target", "afs_studio"]
    if config:
        build_cmd.extend(["--config", config])
    return _run_command(build_cmd)


def _load_manager(config_path: Path | None) -> AFSManager:
    config = load_config_model(config_path=config_path, merge_user=True)
    return AFSManager(config=config)


def _resolve_context_paths(
    args: argparse.Namespace, manager: AFSManager
) -> tuple[Path, Path, Path | None, str | None]:
    project_path = Path(args.path).expanduser().resolve() if args.path else Path.cwd()
    context_root = (
        Path(args.context_root).expanduser().resolve() if args.context_root else None
    )
    context_dir = args.context_dir if args.context_dir else None
    context_path = manager.resolve_context_path(
        project_path,
        context_root=context_root,
        context_dir=context_dir,
    )
    return project_path, context_path, context_root, context_dir


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


def _services_list_command(args: argparse.Namespace) -> int:
    manager = ServiceManager()
    for definition in manager.list_definitions():
        print(f"{definition.name}\t{definition.label}")
    return 0


def _services_render_command(args: argparse.Namespace) -> int:
    manager = ServiceManager()
    print(manager.render_unit(args.name))
    return 0


def _orchestrator_list_command(args: argparse.Namespace) -> int:
    orchestrator = Orchestrator()
    for agent in orchestrator.list_agents():
        tags = ",".join(agent.tags) if agent.tags else "-"
        print(f"{agent.name}\t{agent.role}\t{agent.backend}\t{tags}")
    return 0


def _orchestrator_plan_command(args: argparse.Namespace) -> int:
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


def _studio_build_command(args: argparse.Namespace) -> int:
    try:
        root = _resolve_studio_root()
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    build_dir = _studio_build_dir(root, args.build_dir)
    status = _studio_build(root, build_dir, args.build_type, args.config)
    if status == 0:
        print(f"build_dir: {build_dir}")
    return status


def _studio_run_command(args: argparse.Namespace) -> int:
    try:
        root = _resolve_studio_root()
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    build_dir = _studio_build_dir(root, args.build_dir)
    binary = _studio_binary_path(build_dir, args.config)
    if not binary.exists() and args.build:
        status = _studio_build(root, build_dir, args.build_type, args.config)
        if status != 0:
            return status
        binary = _studio_binary_path(build_dir, args.config)
    if not binary.exists():
        print(f"binary not found: {binary}")
        return 1
    cmd = [str(binary)]
    if args.args:
        cmd.extend(args.args)
    return _run_command(cmd)


def _studio_install_command(args: argparse.Namespace) -> int:
    try:
        root = _resolve_studio_root()
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    build_dir = _studio_build_dir(root, args.build_dir)
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
    status = _run_command(cmd)
    if status == 0:
        print(f"installed: {prefix / 'bin' / _studio_binary_name()}")
    return status


def _studio_path_command(args: argparse.Namespace) -> int:
    try:
        root = _resolve_studio_root()
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    build_dir = _studio_build_dir(root, args.build_dir)
    binary = _studio_binary_path(build_dir, args.config)
    print(binary)
    return 0


def _studio_alias_command(args: argparse.Namespace) -> int:
    try:
        root = _resolve_studio_root()
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    root_value = os.getenv("AFS_ROOT") or str(root)
    print(f"export AFS_ROOT=\"{root_value}\"")
    print("alias afs-studio='PYTHONPATH=\"$AFS_ROOT/src\" python -m afs studio run --build'")
    print("alias afs-studio-build='PYTHONPATH=\"$AFS_ROOT/src\" python -m afs studio build'")
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


def _context_init_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else None
    manager = _load_manager(config_path)
    project_path, _context_path, context_root, context_dir = _resolve_context_paths(
        args, manager
    )
    context = manager.init(
        path=project_path,
        context_root=context_root,
        context_dir=context_dir,
        link_context=args.link_context,
        force=args.force,
    )
    print(f"context_path: {context.path}")
    print(f"project: {context.project_name}")
    return 0


def _context_ensure_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else None
    manager = _load_manager(config_path)
    project_path, _context_path, context_root, context_dir = _resolve_context_paths(
        args, manager
    )
    context = manager.ensure(
        path=project_path,
        context_root=context_root,
        context_dir=context_dir,
        link_context=args.link_context,
    )
    print(f"context_path: {context.path}")
    print(f"project: {context.project_name}")
    return 0


def _context_list_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else None
    manager = _load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = _resolve_context_paths(
        args, manager
    )
    context = manager.list_context(context_path=context_path)
    print(f"context_path: {context.path}")
    print(f"project: {context.project_name}")
    if not context.mounts:
        print("mounts: (none)")
        return 0
    for mount_type in MountType:
        mounts = context.mounts.get(mount_type, [])
        if not mounts:
            continue
        print(f"{mount_type.value}:")
        for mount in mounts:
            suffix = " (link)" if mount.is_symlink else ""
            print(f"- {mount.name} -> {mount.source}{suffix}")
    return 0


def _context_mount_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else None
    manager = _load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = _resolve_context_paths(
        args, manager
    )
    mount_type = _parse_mount_type(args.mount_type)
    source = Path(args.source).expanduser().resolve()
    mount = manager.mount(
        source=source,
        mount_type=mount_type,
        alias=args.alias,
        context_path=context_path,
    )
    print(f"mounted {mount.name} in {mount.mount_type.value}: {mount.source}")
    return 0


def _context_unmount_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else None
    manager = _load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = _resolve_context_paths(
        args, manager
    )
    mount_type = _parse_mount_type(args.mount_type)
    removed = manager.unmount(
        alias=args.alias,
        mount_type=mount_type,
        context_path=context_path,
    )
    if not removed:
        print(f"mount not found: {args.alias}")
        return 1
    print(f"unmounted {args.alias} from {mount_type.value}")
    return 0


def _context_validate_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else None
    manager = _load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = _resolve_context_paths(
        args, manager
    )
    validator = AFSValidator(context_path, afs_directories=manager.config.directories)
    status = validator.check_integrity()
    missing = ", ".join(status.get("missing", [])) or "(none)"
    errors = status.get("errors", [])
    print(f"valid: {status.get('valid', False)}")
    print(f"missing: {missing}")
    if errors:
        print(f"errors: {', '.join(errors)}")
    return 0 if status.get("valid", False) else 1


def _context_discover_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    search_paths = None
    if args.path:
        search_paths = [Path(path).expanduser() for path in args.path]
    ignore_names = args.ignore if args.ignore else None
    projects = discover_contexts(
        search_paths=search_paths,
        max_depth=args.max_depth,
        ignore_names=ignore_names,
        config=config,
    )
    if not projects:
        print("(no contexts)")
        return 0
    for project in projects:
        label = project.project_name
        print(f"{label}\t{project.path}")
    if args.stats:
        stats = get_project_stats(projects)
        pairs = [f"{key}={value}" for key, value in stats.items()]
        print("stats: " + ", ".join(pairs))
    return 0


def _context_ensure_all_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    search_paths = None
    if args.path:
        search_paths = [Path(path).expanduser() for path in args.path]
    ignore_names = args.ignore if args.ignore else None
    projects = discover_contexts(
        search_paths=search_paths,
        max_depth=args.max_depth,
        ignore_names=ignore_names,
        config=config,
    )
    if not projects:
        print("(no contexts)")
        return 0

    manager = AFSManager(config=config)
    for project in projects:
        if args.dry_run:
            print(f"would ensure: {project.project_name}\t{project.path}")
            continue
        context = manager.ensure(
            path=project.path.parent,
            context_root=project.path,
        )
        print(f"ensured: {context.project_name}\t{context.path}")
    return 0


def _graph_export_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    search_paths = None
    if args.path:
        search_paths = [Path(path).expanduser() for path in args.path]
    ignore_names = args.ignore if args.ignore else None
    graph = build_graph(
        search_paths=search_paths,
        max_depth=args.max_depth,
        ignore_names=ignore_names,
        config=config,
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_graph_path(config)
    )
    write_graph(graph, output_path)
    print(f"graph: {output_path}")
    return 0


def _workspace_registry_path() -> Path:
    config = load_config_model()
    return config.general.context_root / "workspaces.toml"


def _load_workspaces_from_registry(path: Path) -> list[WorkspaceDirectory]:
    if not path.exists():
        return []
    data = load_config(config_path=path, merge_user=False)
    entries = data.get("workspaces", [])
    workspaces: list[WorkspaceDirectory] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ws_path = entry.get("path")
        if not ws_path:
            continue
        label = entry.get("description") or entry.get("name")
        workspaces.append(
            WorkspaceDirectory(
                path=Path(ws_path).expanduser().resolve(),
                description=label,
            )
        )
    return workspaces


def _write_workspace_registry(path: Path, workspaces: list[WorkspaceDirectory]) -> None:
    lines = [
        "# AFS workspace registry",
        "# Auto-generated; safe to edit.",
        "",
    ]
    for ws in sorted(workspaces, key=lambda item: str(item.path).lower()):
        lines.append("[[workspaces]]")
        lines.append(f"path = \"{ws.path}\"")
        if ws.description:
            lines.append(f"description = \"{ws.description}\"")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_config_for_workspace(config_path: Path) -> AFSConfig:
    if config_path.exists():
        return load_config_model(config_path=config_path, merge_user=False)
    return AFSConfig()


def _write_workspace_config(config_path: Path, config: AFSConfig) -> None:
    _write_config(config_path, config)


def _workspace_add_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else None
    workspace_path = Path(args.path).expanduser().resolve() if args.path else Path.cwd()
    if config_path:
        config = _load_config_for_workspace(config_path)
        updated = list(config.general.workspace_directories)
    else:
        registry_path = _workspace_registry_path()
        updated = _load_workspaces_from_registry(registry_path)

    replaced = False
    for ws in updated:
        if ws.path == workspace_path:
            if args.force:
                updated.append(
                    WorkspaceDirectory(path=workspace_path, description=args.name)
                )
                replaced = True
            else:
                updated.append(ws)
        else:
            updated.append(ws)

    if not any(ws.path == workspace_path for ws in updated):
        updated.append(WorkspaceDirectory(path=workspace_path, description=args.name))

    if config_path:
        config.general.workspace_directories = updated
        _write_workspace_config(config_path, config)
    else:
        _write_workspace_registry(registry_path, updated)

    action = "updated" if replaced else "added"
    print(f"{action} workspace: {workspace_path}")
    return 0


def _workspace_list_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else None
    if config_path:
        config = _load_config_for_workspace(config_path)
        workspaces = config.general.workspace_directories
    else:
        registry_path = _workspace_registry_path()
        workspaces = _load_workspaces_from_registry(registry_path)
    if not workspaces:
        print("(no workspaces)")
        return 0
    for ws in workspaces:
        label = f" ({ws.description})" if ws.description else ""
        print(f"{ws.path}{label}")
    return 0


def _workspace_remove_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else None
    workspace_path = Path(args.path).expanduser().resolve()
    if config_path:
        config = _load_config_for_workspace(config_path)
        original = list(config.general.workspace_directories)
        updated = [ws for ws in original if ws.path != workspace_path]
        if len(updated) == len(original):
            print(f"workspace not found: {workspace_path}")
            return 1
        config.general.workspace_directories = updated
        _write_workspace_config(config_path, config)
    else:
        registry_path = _workspace_registry_path()
        original = _load_workspaces_from_registry(registry_path)
        updated = [ws for ws in original if ws.path != workspace_path]
        if len(updated) == len(original):
            print(f"workspace not found: {workspace_path}")
            return 1
        _write_workspace_registry(registry_path, updated)
    print(f"removed workspace: {workspace_path}")
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

    services_parser = subparsers.add_parser("services", help="Service definitions.")
    services_sub = services_parser.add_subparsers(dest="services_command")

    services_list = services_sub.add_parser("list", help="List service definitions.")
    services_list.set_defaults(func=_services_list_command)

    services_render = services_sub.add_parser("render", help="Render service unit.")
    services_render.add_argument("name", help="Service name.")
    services_render.set_defaults(func=_services_render_command)

    orch_parser = subparsers.add_parser("orchestrator", help="Orchestrator helpers.")
    orch_sub = orch_parser.add_subparsers(dest="orchestrator_command")

    orch_list = orch_sub.add_parser("list", help="List configured agents.")
    orch_list.set_defaults(func=_orchestrator_list_command)

    orch_plan = orch_sub.add_parser("plan", help="Plan agent routing.")
    orch_plan.add_argument("summary", help="Task summary.")
    orch_plan.add_argument("--tag", action="append", help="Tag to match.")
    orch_plan.add_argument("--role", help="Role to match.")
    orch_plan.set_defaults(func=_orchestrator_plan_command)

    studio_parser = subparsers.add_parser("studio", help="AFS Studio helpers.")
    studio_sub = studio_parser.add_subparsers(dest="studio_command")

    studio_build = studio_sub.add_parser("build", help="Build AFS Studio.")
    studio_build.add_argument("--build-dir", help="Build directory override.")
    studio_build.add_argument(
        "--build-type",
        default="RelWithDebInfo",
        help="CMake build type (default: RelWithDebInfo).",
    )
    studio_build.add_argument("--config", help="Multi-config build name.")
    studio_build.set_defaults(func=_studio_build_command)

    studio_run = studio_sub.add_parser("run", help="Run AFS Studio.")
    studio_run.add_argument("--build", action="store_true", help="Build if missing.")
    studio_run.add_argument("--build-dir", help="Build directory override.")
    studio_run.add_argument(
        "--build-type",
        default="RelWithDebInfo",
        help="CMake build type (default: RelWithDebInfo).",
    )
    studio_run.add_argument("--config", help="Multi-config build name.")
    studio_run.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for afs_studio.")
    studio_run.set_defaults(func=_studio_run_command)

    studio_install = studio_sub.add_parser("install", help="Install AFS Studio.")
    studio_install.add_argument("--prefix", help="Install prefix (default: ~/.local).")
    studio_install.add_argument("--build-dir", help="Build directory override.")
    studio_install.add_argument("--config", help="Multi-config build name.")
    studio_install.set_defaults(func=_studio_install_command)

    studio_path = studio_sub.add_parser("path", help="Print studio binary path.")
    studio_path.add_argument("--build-dir", help="Build directory override.")
    studio_path.add_argument("--config", help="Multi-config build name.")
    studio_path.set_defaults(func=_studio_path_command)

    studio_alias = studio_sub.add_parser("alias", help="Print alias suggestions.")
    studio_alias.set_defaults(func=_studio_alias_command)

    status_parser = subparsers.add_parser("status", help="Show context root status.")
    status_parser.add_argument("--start-dir", help="Directory to search from.")
    status_parser.set_defaults(func=_status_command)

    context_parser = subparsers.add_parser(
        "context", help="Manage per-project .context directories."
    )
    context_sub = context_parser.add_subparsers(dest="context_command")

    ctx_init = context_sub.add_parser("init", help="Initialize a project context.")
    ctx_init.add_argument("--path", help="Project path (default: cwd).")
    ctx_init.add_argument("--context-root", help="Context root path override.")
    ctx_init.add_argument("--context-dir", help="Context directory name.")
    ctx_init.add_argument(
        "--link-context",
        action="store_true",
        help="Link project context to the specified context root.",
    )
    ctx_init.add_argument("--force", action="store_true", help="Overwrite existing context.")
    ctx_init.add_argument("--config", help="Config path for directory policies.")
    ctx_init.set_defaults(func=_context_init_command)

    ctx_ensure = context_sub.add_parser("ensure", help="Ensure a project context exists.")
    ctx_ensure.add_argument("--path", help="Project path (default: cwd).")
    ctx_ensure.add_argument("--context-root", help="Context root path override.")
    ctx_ensure.add_argument("--context-dir", help="Context directory name.")
    ctx_ensure.add_argument(
        "--link-context",
        action="store_true",
        help="Link project context to the specified context root.",
    )
    ctx_ensure.add_argument("--config", help="Config path for directory policies.")
    ctx_ensure.set_defaults(func=_context_ensure_command)

    ctx_list = context_sub.add_parser("list", help="List mounts for a project context.")
    ctx_list.add_argument("--path", help="Project path (default: cwd).")
    ctx_list.add_argument("--context-root", help="Context root path override.")
    ctx_list.add_argument("--context-dir", help="Context directory name.")
    ctx_list.add_argument("--config", help="Config path for directory policies.")
    ctx_list.set_defaults(func=_context_list_command)

    ctx_mount = context_sub.add_parser("mount", help="Mount a resource into a context.")
    ctx_mount.add_argument("source", help="Source path to mount.")
    ctx_mount.add_argument(
        "--mount-type",
        required=True,
        choices=[m.value for m in MountType],
        help="Target mount type.",
    )
    ctx_mount.add_argument("--alias", help="Alias for the mount point.")
    ctx_mount.add_argument("--path", help="Project path (default: cwd).")
    ctx_mount.add_argument("--context-root", help="Context root path override.")
    ctx_mount.add_argument("--context-dir", help="Context directory name.")
    ctx_mount.add_argument("--config", help="Config path for directory policies.")
    ctx_mount.set_defaults(func=_context_mount_command)

    ctx_unmount = context_sub.add_parser("unmount", help="Remove a mounted resource.")
    ctx_unmount.add_argument("alias", help="Alias of the mount point to remove.")
    ctx_unmount.add_argument(
        "--mount-type",
        required=True,
        choices=[m.value for m in MountType],
        help="Mount type containing the alias.",
    )
    ctx_unmount.add_argument("--path", help="Project path (default: cwd).")
    ctx_unmount.add_argument("--context-root", help="Context root path override.")
    ctx_unmount.add_argument("--context-dir", help="Context directory name.")
    ctx_unmount.add_argument("--config", help="Config path for directory policies.")
    ctx_unmount.set_defaults(func=_context_unmount_command)

    ctx_validate = context_sub.add_parser("validate", help="Validate context structure.")
    ctx_validate.add_argument("--path", help="Project path (default: cwd).")
    ctx_validate.add_argument("--context-root", help="Context root path override.")
    ctx_validate.add_argument("--context-dir", help="Context directory name.")
    ctx_validate.add_argument("--config", help="Config path for directory policies.")
    ctx_validate.set_defaults(func=_context_validate_command)

    ctx_discover = context_sub.add_parser(
        "discover", help="Discover .context directories."
    )
    ctx_discover.add_argument(
        "--path",
        action="append",
        help="Search root path (repeatable). Defaults to workspace directories.",
    )
    ctx_discover.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum directory depth to scan.",
    )
    ctx_discover.add_argument(
        "--ignore",
        action="append",
        help="Directory name to ignore (repeatable).",
    )
    ctx_discover.add_argument("--stats", action="store_true", help="Print summary stats.")
    ctx_discover.add_argument("--config", help="Config path for directory policies.")
    ctx_discover.set_defaults(func=_context_discover_command)

    ctx_ensure_all = context_sub.add_parser(
        "ensure-all", help="Ensure all discovered contexts exist."
    )
    ctx_ensure_all.add_argument(
        "--path",
        action="append",
        help="Search root path (repeatable). Defaults to workspace directories.",
    )
    ctx_ensure_all.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum directory depth to scan.",
    )
    ctx_ensure_all.add_argument(
        "--ignore",
        action="append",
        help="Directory name to ignore (repeatable).",
    )
    ctx_ensure_all.add_argument(
        "--dry-run",
        action="store_true",
        help="List contexts without writing.",
    )
    ctx_ensure_all.add_argument("--config", help="Config path for directory policies.")
    ctx_ensure_all.set_defaults(func=_context_ensure_all_command)

    graph_parser = subparsers.add_parser("graph", help="Export AFS graph data.")
    graph_sub = graph_parser.add_subparsers(dest="graph_command")

    graph_export = graph_sub.add_parser("export", help="Export graph JSON.")
    graph_export.add_argument(
        "--path",
        action="append",
        help="Search root path (repeatable). Defaults to workspace directories.",
    )
    graph_export.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum directory depth to scan.",
    )
    graph_export.add_argument(
        "--ignore",
        action="append",
        help="Directory name to ignore (repeatable).",
    )
    graph_export.add_argument(
        "--output",
        help="Output path for graph JSON (default: context_root/index/afs_graph.json).",
    )
    graph_export.add_argument("--config", help="Config path for directory policies.")
    graph_export.set_defaults(func=_graph_export_command)

    workspace_parser = subparsers.add_parser("workspace", help="Manage workspace links.")
    workspace_sub = workspace_parser.add_subparsers(dest="workspace_command")

    ws_add = workspace_sub.add_parser("add", help="Add a workspace to registry or afs.toml.")
    ws_add.add_argument("--path", help="Workspace path (default: cwd).")
    ws_add.add_argument("--name", help="Workspace label/description.")
    ws_add.add_argument("--config", help="Config path to update (default: registry).")
    ws_add.add_argument("--force", action="store_true", help="Overwrite existing entry.")
    ws_add.set_defaults(func=_workspace_add_command)

    ws_list = workspace_sub.add_parser("list", help="List configured workspaces.")
    ws_list.add_argument("--config", help="Config path to read (default: registry).")
    ws_list.set_defaults(func=_workspace_list_command)

    ws_remove = workspace_sub.add_parser("remove", help="Remove a workspace by path.")
    ws_remove.add_argument("--path", required=True, help="Workspace path to remove.")
    ws_remove.add_argument("--config", help="Config path to update (default: registry).")
    ws_remove.set_defaults(func=_workspace_remove_command)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    if args.command == "workspace" and not getattr(args, "workspace_command", None):
        parser.print_help()
        return 1
    if args.command == "context" and not getattr(args, "context_command", None):
        parser.print_help()
        return 1
    if args.command == "graph" and not getattr(args, "graph_command", None):
        parser.print_help()
        return 1
    if args.command == "services" and not getattr(args, "services_command", None):
        parser.print_help()
        return 1
    if args.command == "orchestrator" and not getattr(args, "orchestrator_command", None):
        parser.print_help()
        return 1
    if args.command == "studio" and not getattr(args, "studio_command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
