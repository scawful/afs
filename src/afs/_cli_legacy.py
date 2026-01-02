"""AFS command-line entry points."""

from __future__ import annotations

import argparse
import json
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


def _agents_list_command(args: argparse.Namespace) -> int:
    from .agents import list_agents

    for agent in list_agents():
        if agent.description:
            print(f"{agent.name}\t{agent.description}")
        else:
            print(agent.name)
    return 0


def _agents_run_command(args: argparse.Namespace) -> int:
    from .agents import get_agent

    agent = get_agent(args.name)
    if not agent:
        print(f"unknown agent: {args.name}")
        return 1
    agent_args = list(args.agent_args or [])
    if agent_args and agent_args[0] == "--":
        agent_args = agent_args[1:]
    return agent.entrypoint(agent_args)


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


def _generators_asm_augment_command(args: argparse.Namespace) -> int:
    from .generators import AsmAugmentConfig, AsmAugmentGenerator, write_jsonl

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    config = AsmAugmentConfig(
        paraphrase_count=args.paraphrase_count,
        include_original=args.include_original,
        shuffle_output=not args.no_shuffle,
        min_instruction_len=args.min_len,
        random_seed=args.seed,
    )

    generator = AsmAugmentGenerator(input_path=input_path, config=config)
    result = generator.generate()

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        # Default: same directory with _augmented suffix
        output_path = input_path.parent / f"{input_path.stem}_augmented.jsonl"

    count = write_jsonl(result.samples, output_path)

    print(f"Source samples: {result.source_count}")
    print(f"Generated samples: {result.total}")
    print(f"Skipped: {result.skipped}")
    if result.errors:
        print(f"Errors: {len(result.errors)}")
        for error in result.errors[:5]:
            print(f"  - {error}")
    print(f"Output: {output_path}")
    print(f"Wrote {count} samples")
    return 0


def _generators_cot_command(args: argparse.Namespace) -> int:
    from .generators.cot import CotConfig, CotGenerator, CotFormat
    from .generators import write_jsonl

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    try:
        cot_format = CotFormat(args.format)
    except ValueError:
        print(f"Invalid format: {args.format}")
        print(f"Valid formats: {', '.join(f.value for f in CotFormat)}")
        return 1

    config = CotConfig(
        api_provider=args.provider,
        model_name=args.model,
        cot_format=cot_format,
        requests_per_minute=args.rpm,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )

    generator = CotGenerator(input_path=input_path, config=config)

    # Apply limit if specified
    if args.limit:
        print(f"Limiting to {args.limit} samples")

    result = generator.generate()

    if args.limit and len(result.samples) > args.limit:
        result.samples = result.samples[: args.limit]

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.parent / f"{input_path.stem}_cot.jsonl"

    count = write_jsonl(result.samples, output_path)

    print(f"\nResults:")
    print(f"  Source samples: {result.source_count}")
    print(f"  Generated CoT: {result.total}")
    print(f"  Skipped: {result.skipped}")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
        for error in result.errors[:5]:
            print(f"    - {error}")
    print(f"  Output: {output_path}")
    print(f"  Wrote {count} samples")
    return 0


def _generators_clean_command(args: argparse.Namespace) -> int:
    """Clean training data by fixing malformed samples."""
    from .generators.data_cleaner import clean_dataset

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    output_path = None
    if args.output:
        output_path = Path(args.output).expanduser().resolve()

    regen_output_path = None
    if args.regen_output:
        regen_output_path = Path(args.regen_output).expanduser().resolve()

    stats = clean_dataset(
        input_path=input_path,
        output_path=output_path,
        regen_output_path=regen_output_path,
        min_output_length=args.min_output_length,
    )

    print("\nCleaning Results:")
    print("-" * 40)
    print(stats.summary())

    actual_output = output_path or input_path.parent / f"{input_path.stem}_cleaned.jsonl"
    print(f"\nOutput: {actual_output}")

    if regen_output_path and stats.marked_for_regen > 0:
        print(f"Samples for regeneration: {regen_output_path}")

    if stats.errors:
        print("\nErrors:")
        for error in stats.errors[:5]:
            print(f"  - {error}")
        if len(stats.errors) > 5:
            print(f"  ... and {len(stats.errors) - 5} more")

    return 0


def _generators_validate_command(args: argparse.Namespace) -> int:
    """Validate assembly code in training samples using asar."""
    from .generators.asar_validator import (
        AsarValidatorConfig,
        check_asar_available,
        validate_training_data,
    )

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    if not check_asar_available():
        print("Error: asar not found. Install asar SNES assembler and ensure it's in PATH.")
        print("  - macOS: brew install asar (if available) or build from source")
        print("  - Linux: build from source at https://github.com/RPGHacker/asar")
        print("  - You can also specify --asar-path to point to the executable")
        return 1

    # Determine output paths
    if args.output_pass:
        output_pass_path = Path(args.output_pass).expanduser().resolve()
    else:
        output_pass_path = input_path.parent / f"{input_path.stem}_valid.jsonl"

    if args.output_fail:
        output_fail_path = Path(args.output_fail).expanduser().resolve()
    else:
        output_fail_path = input_path.parent / f"{input_path.stem}_invalid.jsonl"

    # Build config
    config = AsarValidatorConfig(
        asar_path=args.asar_path,
        include_alttp_context=not args.no_alttp_context,
        min_output_length=args.min_length,
        keep_temp_files=args.keep_temp,
    )

    if args.include_path:
        config.include_paths = [Path(p).expanduser().resolve() for p in args.include_path]

    if args.skip_domain:
        config.skip_domains = list(args.skip_domain)

    print(f"Input: {input_path}")
    print(f"Pass output: {output_pass_path}")
    print(f"Fail output: {output_fail_path}")
    print()

    stats = validate_training_data(
        input_path=input_path,
        output_pass_path=output_pass_path,
        output_fail_path=output_fail_path,
        config=config,
        verbose=True,
    )

    print()
    print(f"Wrote {stats.passed} samples to: {output_pass_path}")
    print(f"Wrote {stats.failed + stats.skipped} samples to: {output_fail_path}")

    if stats.errors:
        print("\nErrors encountered:")
        for error in stats.errors[:10]:
            print(f"  - {error}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more")

    return 0


def _training_prepare_command(args: argparse.Namespace) -> int:
    """Split dataset into train/val/test sets."""
    from .training import split_dataset

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    output_dir = Path(args.output).expanduser().resolve()

    result = split_dataset(
        input_path=input_path,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        stratify_by=args.stratify_by if not args.no_stratify else None,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )

    print(result.summary())
    print(f"\nOutput directory: {output_dir}")
    return 0


def _training_convert_command(args: argparse.Namespace) -> int:
    """Convert training data to framework format."""
    from .training import get_converter

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.parent / f"{input_path.stem}_{args.format}.jsonl"

    converter = get_converter(
        format_name=args.format,
        include_cot=not args.no_cot,
        cot_mode=args.cot_mode,
    )

    count = converter.convert_file(input_path, output_path)
    print(f"Converted {count} samples to {args.format} format")
    print(f"Output: {output_path}")
    return 0


def _training_registry_list_command(args: argparse.Namespace) -> int:
    """List experiments in registry."""
    from .training import ModelRegistry

    registry = ModelRegistry()
    experiments = registry.list(
        status=args.status,
        ab_group=args.ab_group,
        framework=args.framework,
    )

    if not experiments:
        print("No experiments found.")
        return 0

    print(f"Found {len(experiments)} experiments:\n")
    for exp in experiments:
        loss_str = f"loss={exp.metrics.final_loss:.4f}" if exp.metrics.final_loss else ""
        print(f"  {exp.experiment_id}: {exp.run_name}")
        print(f"    Model: {exp.base_model}")
        print(f"    Status: {exp.status} {loss_str}")
        if exp.ab_group:
            print(f"    A/B Group: {exp.ab_group} ({exp.ab_variant or 'unassigned'})")
        print()

    return 0


def _training_registry_create_command(args: argparse.Namespace) -> int:
    """Create a new experiment."""
    from .training import ModelRegistry

    registry = ModelRegistry()
    exp = registry.create_experiment(
        run_name=args.name,
        base_model=args.model,
        framework=args.framework,
        dataset_path=args.dataset,
        ab_group=args.ab_group,
        ab_variant=args.ab_variant,
        tags=args.tag or [],
        notes=args.notes or "",
    )

    print(f"Created experiment: {exp.experiment_id}")
    print(f"  Run name: {exp.run_name}")
    print(f"  Base model: {exp.base_model}")
    print(f"  Framework: {exp.framework}")
    return 0


def _discriminator_data_command(args: argparse.Namespace) -> int:
    """Create ELECTRA training data from assembly sources."""
    from .discriminator import create_training_data

    sources = [Path(s).expanduser() for s in args.sources]
    output = Path(args.output).expanduser()

    print(f"Creating ELECTRA training data...")
    print(f"  Sources: {len(sources)} paths")
    print(f"  Fake ratio: {args.fake_ratio}")

    dataset = create_training_data(
        real_sources=sources,
        fake_ratio=args.fake_ratio,
        min_lines=args.min_lines,
        max_lines=args.max_lines,
    )

    dataset.to_jsonl(output)
    stats = dataset.stats()

    print(f"\nResults:")
    print(f"  Total: {stats['total']}")
    print(f"  Real: {stats['real']}")
    print(f"  Fake: {stats['fake']}")
    print(f"  Output: {output}")

    return 0


def _discriminator_train_command(args: argparse.Namespace) -> int:
    """Train ASM-ELECTRA discriminator."""
    from .discriminator import ASMElectra, ElectraConfig, ElectraDataset

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    val_path = Path(args.val).expanduser() if args.val else None

    config = ElectraConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=output_dir,
    )

    print(f"Training ASM-ELECTRA...")
    print(f"  Input: {input_path}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")

    # Load data
    train_dataset = ElectraDataset.from_jsonl(input_path)
    train_data = train_dataset.to_hf_format()

    val_data = None
    if val_path:
        val_dataset = ElectraDataset.from_jsonl(val_path)
        val_data = val_dataset.to_hf_format()

    # Train
    electra = ASMElectra(config=config)
    metrics = electra.train(train_data, val_data)

    print(f"\nTraining complete:")
    print(f"  Loss: {metrics['train_loss']:.4f}")
    print(f"  Steps: {metrics['steps']}")
    print(f"  Model saved: {output_dir / 'final'}")

    return 0


def _discriminator_filter_command(args: argparse.Namespace) -> int:
    """Filter training data using trained discriminator."""
    from .discriminator import SampleFilter, FilterConfig

    model_path = Path(args.model).expanduser()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    rejected_path = Path(args.rejected).expanduser() if args.rejected else None

    config = FilterConfig(min_score=args.min_score)

    print(f"Filtering training data...")
    print(f"  Model: {model_path}")
    print(f"  Min score: {config.min_score}")

    filter = SampleFilter(model_path=model_path, config=config)
    result = filter.filter_jsonl(input_path, output_path, rejected_path)

    print(f"\n{result}")
    print(f"\nScore distribution:")
    for bucket, count in result.score_distribution.items():
        print(f"  {bucket}: {count}")

    return 0


def _discriminator_score_command(args: argparse.Namespace) -> int:
    """Score assembly code quality."""
    from .discriminator import ASMElectra

    model_path = Path(args.model).expanduser()

    if args.file:
        text = Path(args.file).expanduser().read_text()
    elif args.text:
        text = args.text
    else:
        print("Error: must provide --text or --file")
        return 1

    electra = ASMElectra(model_path=model_path)
    score = electra.score(text)
    prediction, confidence = electra.predict(text)

    label = "REAL" if prediction == 0 else "FAKE"
    print(f"Score: {score:.4f}")
    print(f"Prediction: {label} (confidence: {confidence:.2%})")

    return 0


def _training_registry_compare_command(args: argparse.Namespace) -> int:
    """Compare experiments."""
    from .training import ModelRegistry

    registry = ModelRegistry()
    results = registry.compare(args.experiments)

    if not results:
        print("No experiments found for comparison.")
        return 1

    # Print comparison table
    headers = list(results[0].keys())
    print(" | ".join(f"{h:15}" for h in headers))
    print("-" * (17 * len(headers)))

    for row in results:
        values = []
        for h in headers:
            v = row.get(h)
            if isinstance(v, float):
                values.append(f"{v:.4f}")
            elif v is None:
                values.append("-")
            else:
                values.append(str(v)[:15])
        print(" | ".join(f"{v:15}" for v in values))

    return 0


# =============================================================================
# Tokenizer Commands
# =============================================================================


def _tokenizer_create_command(args: argparse.Namespace) -> int:
    """Create a new ASM tokenizer."""
    from .tokenizer import ASMTokenizer

    tokenizer = ASMTokenizer(
        split_addresses=args.split_addresses,
        max_length=args.max_length,
    )

    output = Path(args.output)
    tokenizer.save(output)
    print(f"Created tokenizer with {len(tokenizer)} tokens")
    print(f"Saved to {output}")
    return 0


def _tokenizer_train_command(args: argparse.Namespace) -> int:
    """Train tokenizer on corpus to expand vocabulary."""
    from .tokenizer import ASMTokenizer
    import json

    # Load existing tokenizer or create new
    if args.tokenizer:
        tokenizer = ASMTokenizer.load(args.tokenizer)
        print(f"Loaded tokenizer with {len(tokenizer)} tokens")
    else:
        tokenizer = ASMTokenizer()
        print(f"Created new tokenizer with {len(tokenizer)} base tokens")

    # Load training texts
    texts = []
    input_path = Path(args.input)
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try JSONL
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    # Support various field names
                    for field in ["output", "text", "code", "asm"]:
                        if field in data:
                            texts.append(data[field])
                            break
                    continue
                except json.JSONDecodeError:
                    pass
            texts.append(line)

    print(f"Training on {len(texts)} samples...")
    added = tokenizer.train_on_corpus(
        texts,
        min_frequency=args.min_frequency,
        max_vocab_size=args.max_vocab_size,
    )

    print(f"Added {added} tokens. New vocab size: {len(tokenizer)}")

    # Save
    output = Path(args.output)
    tokenizer.save(output)
    print(f"Saved to {output}")

    # Show unknown tokens if requested
    if args.show_unknowns:
        unknowns = tokenizer.get_unknown_tokens()
        print(f"\nTop unknown tokens (not added):")
        for token, count in list(unknowns.items())[:20]:
            print(f"  {token}: {count}")

    return 0


def _tokenizer_analyze_command(args: argparse.Namespace) -> int:
    """Analyze text with tokenizer."""
    from .tokenizer import ASMTokenizer

    tokenizer = ASMTokenizer.load(args.tokenizer)

    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text()
    else:
        print("Error: --text or --file required")
        return 1

    # Tokenize
    tokens = tokenizer.tokenize(text)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]

    # Count unknowns
    unk_id = tokenizer.unk_token_id
    unk_count = sum(1 for i in input_ids if i == unk_id)
    unk_ratio = unk_count / len(input_ids) if input_ids else 0

    print(f"Tokens: {len(tokens)}")
    print(f"Unknown: {unk_count} ({100*unk_ratio:.1f}%)")
    print(f"\nTokens: {tokens[:50]}{'...' if len(tokens) > 50 else ''}")

    if args.verbose:
        print(f"\nIDs: {input_ids[:50]}{'...' if len(input_ids) > 50 else ''}")

        # Decode back
        decoded = tokenizer.decode(input_ids)
        print(f"\nDecoded: {decoded[:200]}{'...' if len(decoded) > 200 else ''}")

    return 0


def _tokenizer_info_command(args: argparse.Namespace) -> int:
    """Show tokenizer info."""
    from .tokenizer import ASMTokenizer

    tokenizer = ASMTokenizer.load(args.tokenizer)

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Max length: {tokenizer.max_length}")
    print(f"Split addresses: {tokenizer.split_addresses}")
    print(f"\nSpecial tokens:")
    for name, token in tokenizer.special_tokens_map.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {name}: {token} (id={token_id})")

    return 0


# =============================================================================
# Encoder Data Commands (for improving pretraining)
# =============================================================================


def _encoder_analyze_command(args: argparse.Namespace) -> int:
    """Analyze training data quality using encoder."""
    from .tokenizer import ASMTokenizer
    from .training import EncoderDataProcessor, EncoderConfig
    import json

    # Load tokenizer
    if args.tokenizer:
        tokenizer = ASMTokenizer.load(args.tokenizer)
    else:
        tokenizer = ASMTokenizer()

    config = EncoderConfig(
        min_instruction_tokens=args.min_instruction_tokens,
        min_output_tokens=args.min_output_tokens,
        max_unk_ratio=args.max_unk_ratio,
    )
    processor = EncoderDataProcessor(config=config, tokenizer=tokenizer)

    # Load samples
    input_path = Path(args.input)
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Analyzing {len(samples)} samples...")

    # Analyze
    issues_count: dict[str, int] = {}
    valid_count = 0
    unk_ratios = []

    for sample in samples:
        analysis = processor.analyze_sample(sample)
        if analysis["is_valid"]:
            valid_count += 1
        unk_ratios.append(analysis["output_unk_ratio"])
        for issue in analysis["issues"]:
            issues_count[issue] = issues_count.get(issue, 0) + 1

    # Report
    print(f"\nResults:")
    print(f"  Valid samples: {valid_count}/{len(samples)} ({100*valid_count/len(samples):.1f}%)")
    print(f"  Mean UNK ratio: {sum(unk_ratios)/len(unk_ratios):.3f}")
    print(f"\nIssues:")
    for issue, count in sorted(issues_count.items(), key=lambda x: -x[1]):
        print(f"  {issue}: {count} ({100*count/len(samples):.1f}%)")

    return 0


def _encoder_filter_command(args: argparse.Namespace) -> int:
    """Filter training data by quality."""
    from .tokenizer import ASMTokenizer
    from .training import EncoderDataProcessor, EncoderConfig
    import json

    # Load tokenizer
    if args.tokenizer:
        tokenizer = ASMTokenizer.load(args.tokenizer)
    else:
        tokenizer = ASMTokenizer()

    config = EncoderConfig(
        min_instruction_tokens=args.min_instruction_tokens,
        min_output_tokens=args.min_output_tokens,
        max_unk_ratio=args.max_unk_ratio,
    )
    processor = EncoderDataProcessor(config=config, tokenizer=tokenizer)

    # Load samples
    input_path = Path(args.input)
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Filtering {len(samples)} samples...")

    # Filter
    passed, failed = processor.filter_by_quality(samples, verbose=True)

    # Save passed
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for sample in passed:
            f.write(json.dumps(sample) + "\n")
    print(f"Passed: {len(passed)} -> {output_path}")

    # Save failed if requested
    if args.rejected:
        rejected_path = Path(args.rejected)
        with open(rejected_path, "w") as f:
            for sample in failed:
                f.write(json.dumps(sample) + "\n")
        print(f"Rejected: {len(failed)} -> {rejected_path}")

    return 0


def _encoder_dedupe_command(args: argparse.Namespace) -> int:
    """Deduplicate training data using semantic similarity."""
    from .tokenizer import ASMTokenizer
    from .training import EncoderDataProcessor, EncoderConfig
    import json

    # Load tokenizer
    if args.tokenizer:
        tokenizer = ASMTokenizer.load(args.tokenizer)
    else:
        tokenizer = ASMTokenizer()

    config = EncoderConfig(similarity_threshold=args.threshold)
    processor = EncoderDataProcessor(config=config, tokenizer=tokenizer)

    # Load samples
    input_path = Path(args.input)
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Deduplicating {len(samples)} samples (threshold={args.threshold})...")

    # Deduplicate
    deduped = processor.deduplicate(samples, field=args.field, keep=args.keep)

    # Save
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for sample in deduped:
            f.write(json.dumps(sample) + "\n")

    removed = len(samples) - len(deduped)
    print(f"Kept: {len(deduped)}, Removed: {removed} ({100*removed/len(samples):.1f}% duplicates)")
    print(f"Saved to {output_path}")

    return 0


def _encoder_sample_command(args: argparse.Namespace) -> int:
    """Sample diverse subset from training data."""
    from .tokenizer import ASMTokenizer
    from .training import EncoderDataProcessor, EncoderConfig
    import json

    # Load tokenizer
    if args.tokenizer:
        tokenizer = ASMTokenizer.load(args.tokenizer)
    else:
        tokenizer = ASMTokenizer()

    config = EncoderConfig(num_clusters=args.clusters)
    processor = EncoderDataProcessor(config=config, tokenizer=tokenizer)

    # Load samples
    input_path = Path(args.input)
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Sampling {args.n} diverse samples from {len(samples)}...")

    # Sample
    diverse = processor.sample_diverse(samples, n_samples=args.n, field=args.field)

    # Save
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for sample in diverse:
            f.write(json.dumps(sample) + "\n")

    print(f"Sampled {len(diverse)} samples -> {output_path}")

    return 0


def _encoder_pipeline_command(args: argparse.Namespace) -> int:
    """Run full preprocessing pipeline: expand vocab, filter, dedupe."""
    from .tokenizer import ASMTokenizer
    from .training import EncoderDataProcessor, EncoderConfig
    import json

    print("=" * 60)
    print("AFS Pretraining Data Pipeline")
    print("=" * 60)

    # Step 1: Load or create tokenizer
    print("\n[1/4] Loading tokenizer...")
    if args.tokenizer and Path(args.tokenizer).exists():
        tokenizer = ASMTokenizer.load(args.tokenizer)
        print(f"  Loaded tokenizer with {len(tokenizer)} tokens")
    else:
        tokenizer = ASMTokenizer()
        print(f"  Created new tokenizer with {len(tokenizer)} base tokens")

    # Load samples
    input_path = Path(args.input)
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"  Loaded {len(samples)} samples from {input_path}")

    # Step 2: Expand vocabulary
    if not args.skip_vocab_expansion:
        print("\n[2/4] Expanding vocabulary...")
        outputs = [s.get("output", "") for s in samples]
        added = tokenizer.train_on_corpus(outputs, min_frequency=args.min_frequency)
        print(f"  Added {added} tokens. New vocab: {len(tokenizer)}")

        # Save expanded tokenizer
        tokenizer_out = Path(args.output_dir) / "tokenizer"
        tokenizer.save(tokenizer_out)
        print(f"  Saved tokenizer to {tokenizer_out}")
    else:
        print("\n[2/4] Skipping vocabulary expansion")

    # Step 3: Filter by quality
    print("\n[3/4] Filtering by quality...")
    config = EncoderConfig(
        max_unk_ratio=args.max_unk_ratio,
    )
    processor = EncoderDataProcessor(config=config, tokenizer=tokenizer)

    passed, failed = processor.filter_by_quality(samples)
    print(f"  Passed: {len(passed)}, Failed: {len(failed)}")

    # Step 4: Deduplicate
    if not args.skip_dedupe:
        print("\n[4/4] Deduplicating...")
        config = EncoderConfig(similarity_threshold=args.dedupe_threshold)
        processor = EncoderDataProcessor(config=config, tokenizer=tokenizer)
        final = processor.deduplicate(passed, keep="longest")
        removed = len(passed) - len(final)
        print(f"  Removed {removed} duplicates. Final: {len(final)}")
    else:
        print("\n[4/4] Skipping deduplication")
        final = passed

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save cleaned data
    cleaned_path = output_dir / "train_cleaned.jsonl"
    with open(cleaned_path, "w") as f:
        for sample in final:
            f.write(json.dumps(sample) + "\n")

    # Save rejected data
    rejected_path = output_dir / "train_rejected.jsonl"
    with open(rejected_path, "w") as f:
        for sample in failed:
            f.write(json.dumps(sample) + "\n")

    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"  Input:    {len(samples)} samples")
    print(f"  Output:   {len(final)} samples ({100*len(final)/len(samples):.1f}% retained)")
    print(f"  Cleaned:  {cleaned_path}")
    print(f"  Rejected: {rejected_path}")
    print(f"  Tokenizer: {output_dir / 'tokenizer'}")

    return 0


# =============================================================================
# Encoder Train Commands
# =============================================================================


def _encoder_train_command(args: argparse.Namespace) -> int:
    """Train ASM encoder model."""
    from pathlib import Path

    from .training import ASMTrainer, ASMTrainerConfig
    from .tokenizer import ASMTokenizer

    tokenizer_path = Path(args.tokenizer)
    output_path = Path(args.output)
    train_path = Path(args.train)

    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = ASMTokenizer.load(tokenizer_path)

    # Load training texts
    print(f"Loading training data from {train_path}")
    train_texts = []
    with open(train_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if "output" in data:
                    train_texts.append(data["output"])
                elif "text" in data:
                    train_texts.append(data["text"])

    print(f"  Loaded {len(train_texts)} samples")

    # Load validation texts if provided
    val_texts = None
    if args.val:
        val_path = Path(args.val)
        print(f"Loading validation data from {val_path}")
        val_texts = []
        with open(val_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if "output" in data:
                        val_texts.append(data["output"])
                    elif "text" in data:
                        val_texts.append(data["text"])
        print(f"  Loaded {len(val_texts)} validation samples")

    # Configure training
    config = ASMTrainerConfig(
        output_dir=output_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )

    # Train
    print(f"\nTraining encoder with config:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")

    trainer = ASMTrainer(tokenizer=tokenizer, config=config)
    metrics = trainer.train(train_texts, val_texts)

    print(f"\nTraining complete!")
    print(f"  Final loss: {metrics.get('train_loss', 'N/A')}")
    print(f"  Model saved to: {output_path}")

    return 0


# =============================================================================
# Entity Extraction Commands
# =============================================================================


def _entity_extract_command(args: argparse.Namespace) -> int:
    """Extract entities from training data."""
    from pathlib import Path

    from .knowledge import EntityExtractor
    from .generators.base import TrainingSample

    input_path = Path(args.input)
    output_path = Path(args.output)

    extractor = EntityExtractor(include_hardware=not args.no_hardware)

    # Load and process samples
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(TrainingSample.from_dict(json.loads(line)))

    print(f"Processing {len(samples)} samples...")

    total_entities = 0
    known_entities = 0

    for sample in samples:
        sample.populate_kg_entities(extractor, validate=args.validate)
        result = extractor.extract(sample.output)
        total_entities += len(result.entities)
        known_entities += result.known_count

    # Write output
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")

    print(f"\nEntity Extraction Complete")
    print(f"  Samples processed: {len(samples)}")
    print(f"  Total entities found: {total_entities}")
    print(f"  Known entities: {known_entities}")
    print(f"  Coverage: {100 * known_entities / total_entities:.1f}%" if total_entities else "  Coverage: N/A")
    print(f"  Output: {output_path}")

    return 0


def _entity_list_command(args: argparse.Namespace) -> int:
    """List known entities."""
    from .knowledge import (
        ALTTP_ADDRESSES,
        AddressCategory,
        get_addresses_by_category,
    )

    if args.category:
        try:
            category = AddressCategory(args.category.lower())
        except ValueError:
            print(f"Unknown category: {args.category}")
            print(f"Valid categories: {[c.value for c in AddressCategory]}")
            return 1

        addresses = get_addresses_by_category(category)
        print(f"\n{category.value.upper()} Addresses ({len(addresses)}):")
        print("-" * 60)
    else:
        addresses = ALTTP_ADDRESSES
        print(f"\nAll Known Addresses ({len(addresses)}):")
        print("-" * 60)

    for name, info in sorted(addresses.items()):
        print(f"  {name:30} {info.full_address:12} {info.description[:40]}")

    return 0


def _entity_search_command(args: argparse.Namespace) -> int:
    """Search for entity by address."""
    from .knowledge import lookup_by_address

    query = args.address.upper().lstrip("$")

    # Parse address
    if len(query) == 6:
        bank = query[:2]
        offset = int(query[2:], 16)
    elif len(query) == 4:
        bank = "7E"
        offset = int(query, 16)
    elif len(query) == 2:
        bank = "7E"
        offset = int(query, 16)
    else:
        print(f"Invalid address format: {args.address}")
        print("Expected: $XX, $XXXX, or $XXXXXX")
        return 1

    matches = lookup_by_address(offset, bank)

    if not matches:
        print(f"No matches found for ${bank}{offset:04X}")
        return 0

    print(f"\nMatches for ${bank}{offset:04X}:")
    print("-" * 60)
    for name, info in matches:
        print(f"  Name: {name}")
        print(f"  Address: {info.full_address}")
        print(f"  Category: {info.category.value}")
        print(f"  Description: {info.description}")
        if info.notes:
            print(f"  Notes: {info.notes}")
        print()

    return 0


# =============================================================================
# Scoring Commands
# =============================================================================


def _scoring_score_command(args: argparse.Namespace) -> int:
    """Score training samples."""
    from pathlib import Path

    from .training.scoring import score_jsonl, ScoringWeights

    input_path = Path(args.input)
    output_path = Path(args.output)
    electra_path = Path(args.electra) if args.electra else None

    weights = ScoringWeights(
        electra=args.weight_electra,
        asar=args.weight_asar,
        entity=args.weight_entity,
        length=args.weight_length,
    )

    print(f"Scoring samples from {input_path}")
    if electra_path:
        print(f"  Using ELECTRA model: {electra_path}")
    print(f"  Weights: electra={weights.electra}, asar={weights.asar}, entity={weights.entity}, length={weights.length}")

    stats = score_jsonl(
        input_path=input_path,
        output_path=output_path,
        electra_path=electra_path,
        weights=weights,
        min_score=args.min_score,
    )

    print(f"\nScoring Complete")
    print(f"  Input: {stats['input_count']} samples")
    print(f"  Output: {stats['output_count']} samples")
    if args.min_score:
        print(f"  Filtered: {stats['filtered_count']} samples (below {args.min_score})")
    print(f"  Mean score: {stats['mean_score']:.3f}")
    print(f"  Score range: {stats['min_score']:.3f} - {stats['max_score']:.3f}")
    print(f"  Output: {output_path}")

    return 0


def _scoring_analyze_command(args: argparse.Namespace) -> int:
    """Analyze score distribution."""
    from pathlib import Path

    from .generators.base import TrainingSample
    from .training.scoring import QualityScorer, ScoringConfig, analyze_scores

    input_path = Path(args.input)
    electra_path = Path(args.electra) if args.electra else None

    # Load samples
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(TrainingSample.from_dict(json.loads(line)))

    print(f"Analyzing {len(samples)} samples...")

    config = ScoringConfig(electra_model_path=electra_path)
    scorer = QualityScorer(config=config)
    scores = scorer.score_batch(samples, update_samples=False)

    analysis = analyze_scores(scores)

    print(f"\nScore Analysis")
    print("=" * 60)
    print(f"  Total samples: {analysis['count']}")
    print(f"\nOverall Score:")
    print(f"  Mean: {analysis['overall']['mean']:.3f}")
    print(f"  Min:  {analysis['overall']['min']:.3f}")
    print(f"  Max:  {analysis['overall']['max']:.3f}")

    if args.histogram:
        print(f"\n  Distribution:")
        for bucket, count in sorted(analysis['overall']['histogram'].items()):
            bar = "#" * (count * 40 // len(samples)) if samples else ""
            print(f"    {bucket}: {count:4} {bar}")

    print(f"\nComponent Scores:")
    print(f"  ELECTRA mean: {analysis['electra']['mean']:.3f}")
    print(f"  Entity coverage mean: {analysis['entity_coverage']['mean']:.3f}")
    print(f"  Asar pass rate: {100 * analysis['asar_pass_rate']:.1f}%")

    print(f"\nEntity Stats:")
    print(f"  Total entities: {analysis['entity_stats']['total_entities']}")
    print(f"  Known entities: {analysis['entity_stats']['known_entities']}")

    return 0


# =============================================================================
# Pipeline Commands
# =============================================================================


def _pipeline_run_command(args: argparse.Namespace) -> int:
    """Run the full data pipeline."""
    from pathlib import Path

    from .training.pipeline import DataPipeline, PipelineConfig
    from .training.scoring import ScoringWeights

    # Parse input paths
    input_paths = [Path(p) for p in args.input]

    # Build config
    config = PipelineConfig(
        input_paths=input_paths,
        output_dir=Path(args.output),
        expand_vocab=not args.skip_vocab,
        extract_entities=not args.skip_entities,
        score_quality=not args.skip_scoring,
        min_quality_score=args.min_score,
        apply_phase1_augment=not args.skip_augment,
        phase1_paraphrase_count=args.paraphrase_count,
        apply_phase2_augment=not args.skip_augment and not args.skip_phase2,
        deduplicate=not args.skip_dedupe,
        dedupe_threshold=args.dedupe_threshold,
        split_data=not args.skip_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        verbose=not args.quiet,
    )

    if args.tokenizer:
        config.tokenizer_path = Path(args.tokenizer)

    if args.electra:
        config.electra_model_path = Path(args.electra)

    # Run pipeline
    pipeline = DataPipeline(config)
    result = pipeline.run()

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")
        return 1

    return 0


def _pipeline_status_command(args: argparse.Namespace) -> int:
    """Show status of a pipeline run."""
    from pathlib import Path

    output_dir = Path(args.dir)
    result_path = output_dir / "pipeline_result.json"

    if not result_path.exists():
        print(f"No pipeline result found at {result_path}")
        return 1

    with open(result_path) as f:
        result = json.load(f)

    print("\nPipeline Result")
    print("=" * 60)
    print(f"  Input samples: {result['input_count']}")
    print(f"  Output samples: {result['output_count']}")
    print(f"  Filtered: {result['filtered_count']}")
    print(f"  Augmented: {result['augmented_count']}")
    print(f"  Deduped: {result['dedupe_removed']}")
    print(f"\nQuality:")
    print(f"  Mean score: {result['mean_quality_score']:.3f}")
    print(f"  Range: {result['min_quality_score']:.3f} - {result['max_quality_score']:.3f}")
    print(f"\nEntities:")
    print(f"  Total: {result['total_entities']}")
    print(f"  Known: {result['known_entities']}")
    print(f"  Coverage: {100 * result['entity_coverage']:.1f}%")
    print(f"\nDuration: {result['duration_seconds']:.1f} seconds")
    print(f"\nOutput files:")
    for name, path in result['output_paths'].items():
        print(f"  {name}: {path}")

    return 0


# =============================================================================
# Evaluation Commands
# =============================================================================


def _evaluation_run_command(args: argparse.Namespace) -> int:
    """Run evaluation on training samples."""
    from pathlib import Path

    from .generators.base import TrainingSample
    from .evaluation import EvaluationHarness, evaluate_samples
    from .training.scoring import QualityScorer, ScoringConfig

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(TrainingSample.from_dict(json.loads(line)))

    print(f"Loaded {len(samples)} samples from {args.input}")

    # Create scorer
    config = ScoringConfig()
    if args.electra:
        config.electra_model_path = Path(args.electra)
    scorer = QualityScorer(config=config)

    # Run evaluation
    harness = EvaluationHarness(scorer=scorer)
    result = harness.evaluate(samples)

    # Output
    print()
    print(result.summary())

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


def _evaluation_compare_command(args: argparse.Namespace) -> int:
    """Compare two datasets."""
    from pathlib import Path

    from .generators.base import TrainingSample
    from .evaluation import EvaluationHarness
    from .training.scoring import QualityScorer, ScoringConfig

    def load_samples(path: str) -> list[TrainingSample]:
        samples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    samples.append(TrainingSample.from_dict(json.loads(line)))
        return samples

    baseline = load_samples(args.baseline)
    candidate = load_samples(args.candidate)

    print(f"Baseline: {len(baseline)} samples from {args.baseline}")
    print(f"Candidate: {len(candidate)} samples from {args.candidate}")

    # Create scorer
    config = ScoringConfig()
    if args.electra:
        config.electra_model_path = Path(args.electra)
    scorer = QualityScorer(config=config)

    # Compare
    harness = EvaluationHarness(scorer=scorer)
    result = harness.compare(baseline, candidate)

    print()
    print(result.summary())

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


def _evaluation_human_create_command(args: argparse.Namespace) -> int:
    """Create human evaluation batch."""
    from pathlib import Path

    from .generators.base import TrainingSample
    from .evaluation.human import HumanEvaluationManager, SamplingStrategy
    from .training.scoring import QualityScorer, ScoringConfig

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(TrainingSample.from_dict(json.loads(line)))

    print(f"Loaded {len(samples)} samples")

    # Score samples if needed
    config = ScoringConfig()
    if args.electra:
        config.electra_model_path = Path(args.electra)
    scorer = QualityScorer(config=config)
    scorer.score_batch(samples, update_samples=True)

    # Create batch
    manager = HumanEvaluationManager()
    batch = manager.create_batch(
        samples,
        name=args.name or "",
        n=args.n,
        strategy=SamplingStrategy(args.strategy),
    )

    # Save
    output_path = Path(args.output)
    manager.save_batch(batch, output_path)
    print(f"Created batch with {len(batch.tasks)} tasks")
    print(f"Saved to {output_path}")

    # Export CSV if requested
    if args.csv:
        csv_path = output_path.with_suffix(".csv")
        manager.export_csv(batch, csv_path)
        print(f"Exported CSV to {csv_path}")

    return 0


def _evaluation_human_import_command(args: argparse.Namespace) -> int:
    """Import human evaluation results."""
    from pathlib import Path

    from .evaluation.human import HumanEvaluationManager

    manager = HumanEvaluationManager()

    # Load batch
    batch = manager.load_batch(Path(args.batch))
    print(f"Loaded batch {batch.batch_id} with {len(batch.tasks)} tasks")

    # Import results
    results_path = Path(args.results)
    if results_path.suffix == ".csv":
        updated = manager.import_csv(batch, results_path)
    else:
        updated = manager.import_results(batch, results_path)

    print(f"Updated {updated} tasks")

    # Save updated batch
    output_path = Path(args.output) if args.output else Path(args.batch)
    manager.save_batch(batch, output_path)
    print(f"Saved updated batch to {output_path}")

    # Show summary
    summary = manager.get_batch_summary(batch)
    print(f"\nBatch Summary:")
    print(f"  Completed: {summary['completed']}/{summary['total_tasks']}")
    if summary['ratings']['count'] > 0:
        print(f"  Mean rating: {summary['ratings']['mean']:.2f}")

    return 0


# =============================================================================
# Active Learning Commands
# =============================================================================


def _active_learning_sample_command(args: argparse.Namespace) -> int:
    """Sample using uncertainty strategy."""
    from pathlib import Path

    from .generators.base import TrainingSample
    from .active_learning import UncertaintySampler, sample_by_uncertainty
    from .training.scoring import QualityScorer, ScoringConfig

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(TrainingSample.from_dict(json.loads(line)))

    print(f"Loaded {len(samples)} samples")

    # Score if needed
    config = ScoringConfig()
    if args.electra:
        config.electra_model_path = Path(args.electra)
    scorer = QualityScorer(config=config)
    scorer.score_batch(samples, update_samples=True)

    # Sample
    sampler = UncertaintySampler()
    selected = sampler.sample(samples, args.n, scorer=None)  # Already scored

    print(f"Selected {len(selected)} samples by uncertainty")

    # Show distribution
    dist = sampler.get_uncertainty_distribution(samples)
    print(f"\nUncertainty distribution (all samples):")
    for level, count in dist.items():
        print(f"  {level}: {count}")

    # Save if output specified
    if args.output:
        with open(args.output, "w") as f:
            for sample in selected:
                f.write(json.dumps(sample.to_dict()) + "\n")
        print(f"\nSaved to {args.output}")

    return 0


def _active_learning_curriculum_command(args: argparse.Namespace) -> int:
    """Get samples for curriculum stage."""
    from pathlib import Path

    from .generators.base import TrainingSample
    from .active_learning import CurriculumManager, CurriculumStage
    from .training.scoring import QualityScorer, ScoringConfig

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(TrainingSample.from_dict(json.loads(line)))

    print(f"Loaded {len(samples)} samples")

    # Populate kg_entities if needed
    from .knowledge import EntityExtractor
    extractor = EntityExtractor()
    for sample in samples:
        if not sample.kg_entities:
            sample.populate_kg_entities(extractor)

    # Get curriculum distribution
    manager = CurriculumManager()

    if args.plan:
        # Show curriculum plan
        plan = manager.get_curriculum_plan(samples)
        print(f"\nCurriculum Plan ({plan['total_samples']} total samples):")
        print(f"{'Stage':<12} {'Count':>8} {'%':>8} {'Cumulative':>12}")
        print("-" * 45)
        for stage in plan['stages']:
            print(f"{stage['stage']:<12} {stage['sample_count']:>8} {stage['percentage']:>7.1f}% {stage['cumulative_count']:>12}")
        return 0

    # Get samples for specific stage
    stage = CurriculumStage(args.stage)
    stage_samples = manager.get_samples_for_stage(samples, stage)

    print(f"\nStage '{stage.value}': {len(stage_samples)} samples")

    if args.output:
        with open(args.output, "w") as f:
            for sample in stage_samples:
                f.write(json.dumps(sample.to_dict()) + "\n")
        print(f"Saved to {args.output}")

    return 0


def _active_learning_queue_add_command(args: argparse.Namespace) -> int:
    """Add samples to priority queue."""
    from pathlib import Path

    from .generators.base import TrainingSample
    from .active_learning import PriorityQueue
    from .training.scoring import QualityScorer, ScoringConfig

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(TrainingSample.from_dict(json.loads(line)))

    print(f"Loaded {len(samples)} samples")

    # Score if needed
    config = ScoringConfig()
    if args.electra:
        config.electra_model_path = Path(args.electra)
    scorer = QualityScorer(config=config)
    scorer.score_batch(samples, update_samples=True)

    # Add to queue
    queue_path = Path(args.queue)
    queue = PriorityQueue(storage_path=queue_path)
    added = queue.add(samples, scorer=None)  # Already scored

    print(f"Added {added} samples to queue")

    stats = queue.get_stats()
    print(f"Queue now has {stats['total_items']} items")
    print(f"  Pending: {stats['by_status']['pending']}")
    print(f"  In progress: {stats['by_status']['in_progress']}")
    print(f"  Reviewed: {stats['by_status']['reviewed']}")

    return 0


def _active_learning_queue_get_command(args: argparse.Namespace) -> int:
    """Get next batch from priority queue."""
    from pathlib import Path

    from .active_learning import PriorityQueue

    queue_path = Path(args.queue)
    if not queue_path.exists():
        print(f"Queue not found: {queue_path}")
        return 1

    queue = PriorityQueue(storage_path=queue_path)
    items = queue.get_batch(args.n)

    print(f"Retrieved {len(items)} items from queue")

    if args.output:
        output_data = [item.to_dict() for item in items]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved to {args.output}")
    else:
        # Print items
        for i, item in enumerate(items):
            print(f"\n[{i+1}] {item.item_id}")
            print(f"    Domain: {item.domain}")
            print(f"    Priority: {item.priority:.3f}")
            print(f"    Uncertainty: {item.uncertainty:.3f}")
            print(f"    Instruction: {item.instruction[:60]}...")

    return 0


def _active_learning_queue_status_command(args: argparse.Namespace) -> int:
    """Show priority queue status."""
    from pathlib import Path

    from .active_learning import PriorityQueue

    queue_path = Path(args.queue)
    if not queue_path.exists():
        print(f"Queue not found: {queue_path}")
        return 1

    queue = PriorityQueue(storage_path=queue_path)
    stats = queue.get_stats()

    print(f"Priority Queue Status")
    print("=" * 40)
    print(f"Total items: {stats['total_items']}")
    print(f"\nBy status:")
    for status, count in stats['by_status'].items():
        print(f"  {status}: {count}")

    print(f"\nBy domain:")
    for domain, count in stats['by_domain'].items():
        print(f"  {domain}: {count}")

    if stats['ratings']['count'] > 0:
        print(f"\nRatings:")
        print(f"  Count: {stats['ratings']['count']}")
        print(f"  Mean: {stats['ratings']['mean']:.2f}")
        print(f"  Range: {stats['ratings']['min']:.2f} - {stats['ratings']['max']:.2f}")

    return 0


# Generator Commands


def _generator_model_command(args: argparse.Namespace) -> int:
    """Generate assembly code using a trained model."""
    from pathlib import Path

    from .generators.model_generator import (
        ModelGenerator,
        ModelGeneratorConfig,
        ModelType,
        create_generator,
    )

    # Determine model type
    model_type = args.type or "api"

    # Create generator
    kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    if model_type == "api":
        kwargs["api_provider"] = args.provider or "gemini"
        if args.api_key:
            kwargs["api_key"] = args.api_key
        if args.model_name:
            kwargs["model_name"] = args.model_name
    else:
        if not args.model:
            print(f"--model required for type '{model_type}'")
            return 1
        kwargs["model_path"] = Path(args.model)

    generator = create_generator(model_type=model_type, **kwargs)

    # Generate
    instruction = args.instruction
    context = args.context or ""

    sample = generator.generate_one(instruction, context=context)

    if sample is None:
        print("Generation failed (quality check did not pass)")
        return 1

    print(f"Instruction: {instruction}")
    print("=" * 60)
    print(sample.output)
    print("=" * 60)
    print(f"Quality: {sample.quality_score:.2f}")
    if "generation_attempt" in sample._metadata:
        print(f"Attempts: {sample._metadata['generation_attempt']}")

    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(sample.output)
        print(f"\nSaved to: {output_path}")

    return 0


def _generator_knowledge_command(args: argparse.Namespace) -> int:
    """Generate assembly code with ALTTP knowledge context."""
    from pathlib import Path

    from .generators.knowledge_generator import (
        KnowledgeAwareGenerator,
        KnowledgeGeneratorConfig,
        create_knowledge_generator,
    )

    # Parse entities
    required_entities = []
    if args.entities:
        required_entities = [e.strip() for e in args.entities.split(",")]

    # Create generator
    kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "include_entity_context": True,
        "extract_entities_from_instruction": True,
    }

    if args.provider:
        kwargs["api_provider"] = args.provider
    if args.api_key:
        kwargs["api_key"] = args.api_key

    generator = create_knowledge_generator(**kwargs)

    # Suggest entities first if requested
    if args.suggest:
        suggestions = generator.suggest_entities(args.instruction)
        print(f"Suggested entities for: {args.instruction}")
        for entity in suggestions:
            print(f"  - {entity}")
        return 0

    # Generate
    sample = generator.generate_with_context(
        args.instruction,
        required_entities=required_entities,
    )

    if sample is None:
        print("Generation failed (quality check did not pass)")
        return 1

    print(f"Instruction: {args.instruction}")
    if required_entities:
        print(f"Required entities: {', '.join(required_entities)}")
    print("=" * 60)
    print(sample.output)
    print("=" * 60)
    print(f"Quality: {sample.quality_score:.2f}")

    # Show knowledge context info
    if "knowledge_context" in sample._metadata:
        kc = sample._metadata["knowledge_context"]
        if kc.get("entity_hints"):
            print(f"Extracted hints: {', '.join(kc['entity_hints'])}")
        if kc.get("context_entities"):
            print(f"Context entities: {', '.join(kc['context_entities'])}")

    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(sample.output)
        print(f"\nSaved to: {output_path}")

    return 0


def _generator_batch_command(args: argparse.Namespace) -> int:
    """Generate assembly code for multiple instructions."""
    import json
    from pathlib import Path

    from .generators.knowledge_generator import create_knowledge_generator

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 1

    # Load instructions
    instructions = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                # Support JSON lines or plain text
                if line.startswith("{"):
                    data = json.loads(line)
                    instructions.append(data.get("instruction", data.get("text", line)))
                else:
                    instructions.append(line)

    if not instructions:
        print("No instructions found in input file")
        return 1

    print(f"Loaded {len(instructions)} instructions")

    # Create generator
    kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.provider:
        kwargs["api_provider"] = args.provider
    if args.api_key:
        kwargs["api_key"] = args.api_key

    generator = create_knowledge_generator(**kwargs)

    # Generate batch
    samples = generator.generate_batch(instructions)

    print(f"Generated {len(samples)}/{len(instructions)} samples")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in samples:
            record = {
                "sample_id": sample.sample_id,
                "instruction": sample.instruction,
                "output": sample.output,
                "quality_score": sample.quality_score,
                "domain": sample.domain,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved to: {output_path}")
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

    agents_parser = subparsers.add_parser("agents", help="Run built-in agents.")
    agents_sub = agents_parser.add_subparsers(dest="agents_command")

    agents_list = agents_sub.add_parser("list", help="List available agents.")
    agents_list.set_defaults(func=_agents_list_command)

    agents_run = agents_sub.add_parser("run", help="Run a built-in agent.")
    agents_run.add_argument("name", help="Agent name.")
    agents_run.add_argument(
        "agent_args",
        nargs=argparse.REMAINDER,
        help="Arguments for the agent (prefix with -- to pass through).",
    )
    agents_run.set_defaults(func=_agents_run_command)

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

    # Generators
    generators_parser = subparsers.add_parser(
        "generators", help="Training data generators."
    )
    generators_sub = generators_parser.add_subparsers(dest="generators_command")

    gen_asm_augment = generators_sub.add_parser(
        "asm-augment", help="Augment ASM training samples via paraphrasing."
    )
    gen_asm_augment.add_argument(
        "--input", required=True, help="Source JSONL file with training samples."
    )
    gen_asm_augment.add_argument(
        "--output", help="Output JSONL path (default: input_augmented.jsonl)."
    )
    gen_asm_augment.add_argument(
        "--paraphrase-count",
        type=int,
        default=5,
        help="Number of paraphrases per sample (default: 5).",
    )
    gen_asm_augment.add_argument(
        "--no-original",
        action="store_false",
        dest="include_original",
        help="Exclude original samples from output.",
    )
    gen_asm_augment.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle output samples.",
    )
    gen_asm_augment.add_argument(
        "--min-len",
        type=int,
        default=10,
        help="Minimum instruction length to augment (default: 10).",
    )
    gen_asm_augment.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    gen_asm_augment.set_defaults(func=_generators_asm_augment_command)

    # Chain of Thought generator
    gen_cot = generators_sub.add_parser(
        "cot", help="Generate Chain of Thought reasoning for samples."
    )
    gen_cot.add_argument(
        "--input", required=True, help="Source JSONL file with training samples."
    )
    gen_cot.add_argument(
        "--output", help="Output JSONL path (default: input_cot.jsonl)."
    )
    gen_cot.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "claude", "openai"],
        help="LLM provider for CoT generation (default: gemini).",
    )
    gen_cot.add_argument(
        "--model",
        default="gemini-2.0-flash-exp",
        help="Model name (default: gemini-2.0-flash-exp).",
    )
    gen_cot.add_argument(
        "--format",
        default="separate",
        choices=["separate", "embedded", "special_tokens"],
        help="CoT output format (default: separate).",
    )
    gen_cot.add_argument(
        "--rpm",
        type=int,
        default=60,
        help="Requests per minute rate limit (default: 60).",
    )
    gen_cot.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10).",
    )
    gen_cot.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7).",
    )
    gen_cot.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing).",
    )
    gen_cot.set_defaults(func=_generators_cot_command)

    # Data cleaner
    gen_clean = generators_sub.add_parser(
        "clean", help="Clean training data by fixing malformed samples."
    )
    gen_clean.add_argument(
        "--input", required=True, help="Source JSONL file with training samples."
    )
    gen_clean.add_argument(
        "--output", help="Output JSONL path (default: input_cleaned.jsonl)."
    )
    gen_clean.add_argument(
        "--regen-output",
        help="Output file for samples needing regeneration (optional).",
    )
    gen_clean.add_argument(
        "--min-output-length",
        type=int,
        default=100,
        help="Minimum output length to retain sample (default: 100).",
    )
    gen_clean.set_defaults(func=_generators_clean_command)

    # Asar validation
    gen_validate = generators_sub.add_parser(
        "validate", help="Validate assembly code samples using asar SNES assembler."
    )
    gen_validate.add_argument(
        "--input", required=True, help="Source JSONL file with training samples."
    )
    gen_validate.add_argument(
        "--output-pass",
        help="Output JSONL for passing samples (default: input_valid.jsonl).",
    )
    gen_validate.add_argument(
        "--output-fail",
        help="Output JSONL for failing samples (default: input_invalid.jsonl).",
    )
    gen_validate.add_argument(
        "--asar-path",
        help="Path to asar executable (default: search PATH).",
    )
    gen_validate.add_argument(
        "--include-path",
        action="append",
        help="Additional include path for asar (repeatable).",
    )
    gen_validate.add_argument(
        "--no-alttp-context",
        action="store_true",
        help="Don't include ALTTP-specific defines and context.",
    )
    gen_validate.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum output length to validate (default: 10).",
    )
    gen_validate.add_argument(
        "--skip-domain",
        action="append",
        help="Domain to skip (repeatable). Default: text, docs.",
    )
    gen_validate.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary .asm files for debugging.",
    )
    gen_validate.set_defaults(func=_generators_validate_command)

    # Training
    training_parser = subparsers.add_parser(
        "training", help="Training data preparation and experiment tracking."
    )
    training_sub = training_parser.add_subparsers(dest="training_command")

    # training prepare - split dataset
    train_prepare = training_sub.add_parser(
        "prepare", help="Split dataset into train/val/test sets."
    )
    train_prepare.add_argument(
        "--input", required=True, help="Source JSONL file."
    )
    train_prepare.add_argument(
        "--output", required=True, help="Output directory for split files."
    )
    train_prepare.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8).",
    )
    train_prepare.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1).",
    )
    train_prepare.add_argument(
        "--stratify-by",
        default="domain",
        help="Field to stratify by (default: domain).",
    )
    train_prepare.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratification.",
    )
    train_prepare.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle samples.",
    )
    train_prepare.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    train_prepare.set_defaults(func=_training_prepare_command)

    # training convert - format conversion
    train_convert = training_sub.add_parser(
        "convert", help="Convert data to framework format."
    )
    train_convert.add_argument(
        "--input", required=True, help="Source JSONL file."
    )
    train_convert.add_argument(
        "--output", help="Output path (default: input_<format>.jsonl)."
    )
    train_convert.add_argument(
        "--format",
        required=True,
        choices=["mlx", "alpaca", "chatml", "sharegpt", "llama_cpp", "gguf"],
        help="Target format.",
    )
    train_convert.add_argument(
        "--no-cot",
        action="store_true",
        help="Exclude chain of thought from output.",
    )
    train_convert.add_argument(
        "--cot-mode",
        default="separate",
        choices=["none", "separate", "embedded", "special_tokens"],
        help="How to include CoT (default: separate).",
    )
    train_convert.set_defaults(func=_training_convert_command)

    # training registry list
    train_reg_list = training_sub.add_parser(
        "list", help="List experiments in registry."
    )
    train_reg_list.add_argument(
        "--status",
        choices=["pending", "running", "completed", "failed"],
        help="Filter by status.",
    )
    train_reg_list.add_argument(
        "--ab-group", help="Filter by A/B test group."
    )
    train_reg_list.add_argument(
        "--framework",
        choices=["mlx", "unsloth", "llama_cpp"],
        help="Filter by framework.",
    )
    train_reg_list.set_defaults(func=_training_registry_list_command)

    # training registry create
    train_reg_create = training_sub.add_parser(
        "create", help="Create a new experiment."
    )
    train_reg_create.add_argument(
        "--name", required=True, help="Experiment run name."
    )
    train_reg_create.add_argument(
        "--model", required=True, help="Base model identifier."
    )
    train_reg_create.add_argument(
        "--framework",
        required=True,
        choices=["mlx", "unsloth", "llama_cpp"],
        help="Training framework.",
    )
    train_reg_create.add_argument(
        "--dataset", help="Path to training dataset."
    )
    train_reg_create.add_argument(
        "--ab-group", help="A/B test group name."
    )
    train_reg_create.add_argument(
        "--ab-variant", help="A/B test variant (A, B, control)."
    )
    train_reg_create.add_argument(
        "--tag", action="append", help="Tag for categorization (repeatable)."
    )
    train_reg_create.add_argument(
        "--notes", help="Free-form notes."
    )
    train_reg_create.set_defaults(func=_training_registry_create_command)

    # training registry compare
    train_reg_compare = training_sub.add_parser(
        "compare", help="Compare experiments."
    )
    train_reg_compare.add_argument(
        "experiments", nargs="+", help="Experiment IDs to compare."
    )
    train_reg_compare.set_defaults(func=_training_registry_compare_command)

    # Discriminator
    disc_parser = subparsers.add_parser(
        "discriminator", help="ASM-ELECTRA discriminator for quality filtering."
    )
    disc_sub = disc_parser.add_subparsers(dest="discriminator_command")

    # discriminator data - create training data
    disc_data = disc_sub.add_parser(
        "data", help="Create ELECTRA training data from assembly sources."
    )
    disc_data.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Paths to files/directories containing real assembly.",
    )
    disc_data.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    disc_data.add_argument(
        "--fake-ratio",
        type=float,
        default=0.5,
        help="Ratio of fake samples (default: 0.5).",
    )
    disc_data.add_argument(
        "--min-lines",
        type=int,
        default=3,
        help="Minimum lines per sample (default: 3).",
    )
    disc_data.add_argument(
        "--max-lines",
        type=int,
        default=50,
        help="Maximum lines per sample (default: 50).",
    )
    disc_data.set_defaults(func=_discriminator_data_command)

    # discriminator train - train ELECTRA
    disc_train = disc_sub.add_parser(
        "train", help="Train ASM-ELECTRA discriminator."
    )
    disc_train.add_argument(
        "--input",
        required=True,
        help="Training JSONL from 'discriminator data'.",
    )
    disc_train.add_argument(
        "--output",
        required=True,
        help="Output directory for trained model.",
    )
    disc_train.add_argument(
        "--val",
        help="Optional validation JSONL.",
    )
    disc_train.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs (default: 3).",
    )
    disc_train.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    disc_train.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5).",
    )
    disc_train.set_defaults(func=_discriminator_train_command)

    # discriminator filter - filter training samples
    disc_filter = disc_sub.add_parser(
        "filter", help="Filter training data using trained discriminator."
    )
    disc_filter.add_argument(
        "--model",
        required=True,
        help="Path to trained ASM-ELECTRA model.",
    )
    disc_filter.add_argument(
        "--input",
        required=True,
        help="Input training JSONL to filter.",
    )
    disc_filter.add_argument(
        "--output",
        required=True,
        help="Output filtered JSONL.",
    )
    disc_filter.add_argument(
        "--min-score",
        type=float,
        default=0.7,
        help="Minimum score to accept (default: 0.7).",
    )
    disc_filter.add_argument(
        "--rejected",
        help="Optional output for rejected samples.",
    )
    disc_filter.set_defaults(func=_discriminator_filter_command)

    # discriminator score - score a sample
    disc_score = disc_sub.add_parser(
        "score", help="Score assembly code quality."
    )
    disc_score.add_argument(
        "--model",
        required=True,
        help="Path to trained ASM-ELECTRA model.",
    )
    disc_score.add_argument(
        "--text",
        help="Assembly code to score (or use --file).",
    )
    disc_score.add_argument(
        "--file",
        help="File containing assembly code to score.",
    )
    disc_score.set_defaults(func=_discriminator_score_command)

    # Tokenizer
    tokenizer_parser = subparsers.add_parser(
        "tokenizer", help="ASM tokenizer utilities for 65816 assembly."
    )
    tokenizer_sub = tokenizer_parser.add_subparsers(dest="tokenizer_command")

    # tokenizer create
    tok_create = tokenizer_sub.add_parser(
        "create", help="Create a new ASM tokenizer."
    )
    tok_create.add_argument(
        "--output", required=True, help="Output directory for tokenizer."
    )
    tok_create.add_argument(
        "--split-addresses",
        action="store_true",
        help="Split addresses into components (default: keep whole).",
    )
    tok_create.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512).",
    )
    tok_create.set_defaults(func=_tokenizer_create_command)

    # tokenizer train
    tok_train = tokenizer_sub.add_parser(
        "train", help="Train tokenizer on corpus to expand vocabulary."
    )
    tok_train.add_argument(
        "--input", required=True, help="Input JSONL or text file."
    )
    tok_train.add_argument(
        "--output", required=True, help="Output directory for tokenizer."
    )
    tok_train.add_argument(
        "--tokenizer", help="Existing tokenizer to expand (creates new if not specified)."
    )
    tok_train.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency to add token (default: 2).",
    )
    tok_train.add_argument(
        "--max-vocab-size",
        type=int,
        default=50000,
        help="Maximum vocabulary size (default: 50000).",
    )
    tok_train.add_argument(
        "--show-unknowns",
        action="store_true",
        help="Show top unknown tokens after training.",
    )
    tok_train.set_defaults(func=_tokenizer_train_command)

    # tokenizer analyze
    tok_analyze = tokenizer_sub.add_parser(
        "analyze", help="Analyze text with tokenizer."
    )
    tok_analyze.add_argument(
        "--tokenizer", required=True, help="Path to tokenizer."
    )
    tok_analyze.add_argument(
        "--text", help="Text to analyze."
    )
    tok_analyze.add_argument(
        "--file", help="File to analyze."
    )
    tok_analyze.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output."
    )
    tok_analyze.set_defaults(func=_tokenizer_analyze_command)

    # tokenizer info
    tok_info = tokenizer_sub.add_parser(
        "info", help="Show tokenizer info."
    )
    tok_info.add_argument(
        "--tokenizer", required=True, help="Path to tokenizer."
    )
    tok_info.set_defaults(func=_tokenizer_info_command)

    # Encoder data utilities
    encoder_parser = subparsers.add_parser(
        "encoder", help="Encoder-based data preprocessing for pretraining."
    )
    encoder_sub = encoder_parser.add_subparsers(dest="encoder_command")

    # encoder analyze
    enc_analyze = encoder_sub.add_parser(
        "analyze", help="Analyze training data quality."
    )
    enc_analyze.add_argument(
        "--input", required=True, help="Input JSONL file."
    )
    enc_analyze.add_argument(
        "--tokenizer", help="Path to tokenizer (uses default if not specified)."
    )
    enc_analyze.add_argument(
        "--min-instruction-tokens",
        type=int,
        default=5,
        help="Minimum instruction words (default: 5).",
    )
    enc_analyze.add_argument(
        "--min-output-tokens",
        type=int,
        default=10,
        help="Minimum output tokens (default: 10).",
    )
    enc_analyze.add_argument(
        "--max-unk-ratio",
        type=float,
        default=0.1,
        help="Maximum unknown token ratio (default: 0.1).",
    )
    enc_analyze.set_defaults(func=_encoder_analyze_command)

    # encoder filter
    enc_filter = encoder_sub.add_parser(
        "filter", help="Filter training data by quality."
    )
    enc_filter.add_argument(
        "--input", required=True, help="Input JSONL file."
    )
    enc_filter.add_argument(
        "--output", required=True, help="Output JSONL for passed samples."
    )
    enc_filter.add_argument(
        "--rejected", help="Output JSONL for rejected samples."
    )
    enc_filter.add_argument(
        "--tokenizer", help="Path to tokenizer."
    )
    enc_filter.add_argument(
        "--min-instruction-tokens",
        type=int,
        default=5,
        help="Minimum instruction words (default: 5).",
    )
    enc_filter.add_argument(
        "--min-output-tokens",
        type=int,
        default=10,
        help="Minimum output tokens (default: 10).",
    )
    enc_filter.add_argument(
        "--max-unk-ratio",
        type=float,
        default=0.1,
        help="Maximum unknown token ratio (default: 0.1).",
    )
    enc_filter.set_defaults(func=_encoder_filter_command)

    # encoder dedupe
    enc_dedupe = encoder_sub.add_parser(
        "dedupe", help="Deduplicate training data using semantic similarity."
    )
    enc_dedupe.add_argument(
        "--input", required=True, help="Input JSONL file."
    )
    enc_dedupe.add_argument(
        "--output", required=True, help="Output JSONL file."
    )
    enc_dedupe.add_argument(
        "--tokenizer", help="Path to tokenizer."
    )
    enc_dedupe.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Similarity threshold (default: 0.95).",
    )
    enc_dedupe.add_argument(
        "--field",
        default="output",
        choices=["output", "instruction"],
        help="Field to compare (default: output).",
    )
    enc_dedupe.add_argument(
        "--keep",
        default="longest",
        choices=["first", "longest", "shortest"],
        help="Which duplicate to keep (default: longest).",
    )
    enc_dedupe.set_defaults(func=_encoder_dedupe_command)

    # encoder sample
    enc_sample = encoder_sub.add_parser(
        "sample", help="Sample diverse subset from training data."
    )
    enc_sample.add_argument(
        "--input", required=True, help="Input JSONL file."
    )
    enc_sample.add_argument(
        "--output", required=True, help="Output JSONL file."
    )
    enc_sample.add_argument(
        "--n", type=int, required=True, help="Number of samples to select."
    )
    enc_sample.add_argument(
        "--tokenizer", help="Path to tokenizer."
    )
    enc_sample.add_argument(
        "--clusters",
        type=int,
        default=100,
        help="Number of clusters for diversity sampling (default: 100).",
    )
    enc_sample.add_argument(
        "--field",
        default="output",
        choices=["output", "instruction"],
        help="Field to embed (default: output).",
    )
    enc_sample.set_defaults(func=_encoder_sample_command)

    # encoder pipeline
    enc_pipeline = encoder_sub.add_parser(
        "pipeline", help="Run full preprocessing pipeline: expand vocab, filter, dedupe."
    )
    enc_pipeline.add_argument(
        "--input", required=True, help="Input JSONL file."
    )
    enc_pipeline.add_argument(
        "--output-dir", required=True, help="Output directory for results."
    )
    enc_pipeline.add_argument(
        "--tokenizer", help="Existing tokenizer path (creates new if not specified)."
    )
    enc_pipeline.add_argument(
        "--skip-vocab-expansion",
        action="store_true",
        help="Skip vocabulary expansion step.",
    )
    enc_pipeline.add_argument(
        "--skip-dedupe",
        action="store_true",
        help="Skip deduplication step.",
    )
    enc_pipeline.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for vocab expansion (default: 2).",
    )
    enc_pipeline.add_argument(
        "--max-unk-ratio",
        type=float,
        default=0.1,
        help="Maximum unknown token ratio (default: 0.1).",
    )
    enc_pipeline.add_argument(
        "--dedupe-threshold",
        type=float,
        default=0.95,
        help="Deduplication similarity threshold (default: 0.95).",
    )
    enc_pipeline.set_defaults(func=_encoder_pipeline_command)

    # encoder train
    enc_train = encoder_sub.add_parser(
        "train", help="Train ASM encoder model."
    )
    enc_train.add_argument(
        "--tokenizer", required=True, help="Path to tokenizer."
    )
    enc_train.add_argument(
        "--train", required=True, help="Training data JSONL file."
    )
    enc_train.add_argument(
        "--output", required=True, help="Output directory for model."
    )
    enc_train.add_argument(
        "--val", help="Validation data JSONL file."
    )
    enc_train.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs (default: 10)."
    )
    enc_train.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: 16)."
    )
    enc_train.add_argument(
        "--learning-rate", type=float, default=5e-4, help="Learning rate (default: 5e-4)."
    )
    enc_train.add_argument(
        "--hidden-size", type=int, default=256, help="Hidden size (default: 256)."
    )
    enc_train.add_argument(
        "--num-layers", type=int, default=4, help="Number of layers (default: 4)."
    )
    enc_train.add_argument(
        "--num-heads", type=int, default=4, help="Number of attention heads (default: 4)."
    )
    enc_train.set_defaults(func=_encoder_train_command)

    # ==========================================================================
    # Entity Commands
    # ==========================================================================
    entity_parser = subparsers.add_parser(
        "entity", help="Entity extraction and knowledge base utilities."
    )
    entity_sub = entity_parser.add_subparsers(dest="entity_command")

    # entity extract
    ent_extract = entity_sub.add_parser(
        "extract", help="Extract entities from training data and populate kg_entities."
    )
    ent_extract.add_argument(
        "--input", required=True, help="Input JSONL file."
    )
    ent_extract.add_argument(
        "--output", required=True, help="Output JSONL with populated entities."
    )
    ent_extract.add_argument(
        "--validate", action="store_true", help="Validate entity usage in context."
    )
    ent_extract.add_argument(
        "--no-hardware", action="store_true", help="Exclude hardware register matches."
    )
    ent_extract.set_defaults(func=_entity_extract_command)

    # entity list
    ent_list = entity_sub.add_parser(
        "list", help="List known entities from knowledge base."
    )
    ent_list.add_argument(
        "--category", help="Filter by category (link_state, inventory, sprite, etc.)."
    )
    ent_list.set_defaults(func=_entity_list_command)

    # entity search
    ent_search = entity_sub.add_parser(
        "search", help="Search for entity by address."
    )
    ent_search.add_argument(
        "address", help="Address to search (e.g., $7EF36C, $0022, $22)."
    )
    ent_search.set_defaults(func=_entity_search_command)

    # ==========================================================================
    # Scoring Commands
    # ==========================================================================
    scoring_parser = subparsers.add_parser(
        "scoring", help="Quality scoring for training samples."
    )
    scoring_sub = scoring_parser.add_subparsers(dest="scoring_command")

    # scoring score
    scr_score = scoring_sub.add_parser(
        "score", help="Score training samples and populate quality_score field."
    )
    scr_score.add_argument(
        "--input", required=True, help="Input JSONL file."
    )
    scr_score.add_argument(
        "--output", required=True, help="Output JSONL with scores."
    )
    scr_score.add_argument(
        "--electra", help="Path to ELECTRA model (optional)."
    )
    scr_score.add_argument(
        "--min-score", type=float, help="Filter samples below this score."
    )
    scr_score.add_argument(
        "--weight-electra", type=float, default=0.4, help="ELECTRA weight (default: 0.4)."
    )
    scr_score.add_argument(
        "--weight-asar", type=float, default=0.3, help="Asar validation weight (default: 0.3)."
    )
    scr_score.add_argument(
        "--weight-entity", type=float, default=0.2, help="Entity coverage weight (default: 0.2)."
    )
    scr_score.add_argument(
        "--weight-length", type=float, default=0.1, help="Length score weight (default: 0.1)."
    )
    scr_score.set_defaults(func=_scoring_score_command)

    # scoring analyze
    scr_analyze = scoring_sub.add_parser(
        "analyze", help="Analyze score distribution of samples."
    )
    scr_analyze.add_argument(
        "--input", required=True, help="Input JSONL file."
    )
    scr_analyze.add_argument(
        "--electra", help="Path to ELECTRA model (optional)."
    )
    scr_analyze.add_argument(
        "--histogram", action="store_true", help="Show score histogram."
    )
    scr_analyze.set_defaults(func=_scoring_analyze_command)

    # ==========================================================================
    # Pipeline Commands
    # ==========================================================================
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Full data processing pipeline."
    )
    pipeline_sub = pipeline_parser.add_subparsers(dest="pipeline_command")

    # pipeline run
    pipe_run = pipeline_sub.add_parser(
        "run", help="Run the full preprocessing pipeline."
    )
    pipe_run.add_argument(
        "--input", nargs="+", required=True, help="Input JSONL files."
    )
    pipe_run.add_argument(
        "--output", required=True, help="Output directory."
    )
    pipe_run.add_argument(
        "--tokenizer", help="Path to existing tokenizer (optional)."
    )
    pipe_run.add_argument(
        "--electra", help="Path to ELECTRA model for scoring (optional)."
    )
    pipe_run.add_argument(
        "--min-score", type=float, default=0.5, help="Minimum quality score threshold (default: 0.5)."
    )
    pipe_run.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training set ratio (default: 0.8)."
    )
    pipe_run.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation set ratio (default: 0.1)."
    )
    pipe_run.add_argument(
        "--paraphrase-count", type=int, default=3, help="Phase 1 paraphrase count (default: 3)."
    )
    pipe_run.add_argument(
        "--dedupe-threshold", type=float, default=0.95, help="Deduplication similarity threshold (default: 0.95)."
    )
    pipe_run.add_argument(
        "--skip-vocab", action="store_true", help="Skip vocabulary expansion."
    )
    pipe_run.add_argument(
        "--skip-entities", action="store_true", help="Skip entity extraction."
    )
    pipe_run.add_argument(
        "--skip-scoring", action="store_true", help="Skip quality scoring."
    )
    pipe_run.add_argument(
        "--skip-augment", action="store_true", help="Skip all augmentation."
    )
    pipe_run.add_argument(
        "--skip-phase2", action="store_true", help="Skip Phase 2 augmentation only."
    )
    pipe_run.add_argument(
        "--skip-dedupe", action="store_true", help="Skip deduplication."
    )
    pipe_run.add_argument(
        "--skip-split", action="store_true", help="Skip train/val/test split."
    )
    pipe_run.add_argument(
        "--quiet", action="store_true", help="Suppress progress output."
    )
    pipe_run.set_defaults(func=_pipeline_run_command)

    # pipeline status
    pipe_status = pipeline_sub.add_parser(
        "status", help="Show status of a pipeline run."
    )
    pipe_status.add_argument(
        "--dir", required=True, help="Pipeline output directory."
    )
    pipe_status.set_defaults(func=_pipeline_status_command)

    # ==========================================================================
    # Evaluation Commands
    # ==========================================================================
    eval_parser = subparsers.add_parser(
        "evaluation", help="Evaluation harness and human evaluation."
    )
    eval_sub = eval_parser.add_subparsers(dest="evaluation_command")

    # evaluation run
    eval_run = eval_sub.add_parser(
        "run", help="Run evaluation on training samples."
    )
    eval_run.add_argument(
        "--input", required=True, help="Input JSONL file."
    )
    eval_run.add_argument(
        "--output", help="Output JSON file for results."
    )
    eval_run.add_argument(
        "--electra", help="Path to ELECTRA model (optional)."
    )
    eval_run.set_defaults(func=_evaluation_run_command)

    # evaluation compare
    eval_compare = eval_sub.add_parser(
        "compare", help="Compare two datasets."
    )
    eval_compare.add_argument(
        "--baseline", required=True, help="Baseline JSONL file."
    )
    eval_compare.add_argument(
        "--candidate", required=True, help="Candidate JSONL file."
    )
    eval_compare.add_argument(
        "--output", help="Output JSON file for results."
    )
    eval_compare.add_argument(
        "--electra", help="Path to ELECTRA model (optional)."
    )
    eval_compare.set_defaults(func=_evaluation_compare_command)

    # evaluation human create
    eval_human_create = eval_sub.add_parser(
        "human-create", help="Create human evaluation batch."
    )
    eval_human_create.add_argument(
        "--input", required=True, help="Input JSONL file."
    )
    eval_human_create.add_argument(
        "--output", required=True, help="Output batch JSON file."
    )
    eval_human_create.add_argument(
        "--n", type=int, default=50, help="Number of samples (default: 50)."
    )
    eval_human_create.add_argument(
        "--strategy", default="uncertainty",
        choices=["random", "uncertainty", "low_quality", "high_quality", "stratified"],
        help="Sampling strategy (default: uncertainty)."
    )
    eval_human_create.add_argument(
        "--name", help="Batch name."
    )
    eval_human_create.add_argument(
        "--electra", help="Path to ELECTRA model (optional)."
    )
    eval_human_create.add_argument(
        "--csv", action="store_true", help="Also export as CSV."
    )
    eval_human_create.set_defaults(func=_evaluation_human_create_command)

    # evaluation human import
    eval_human_import = eval_sub.add_parser(
        "human-import", help="Import human evaluation results."
    )
    eval_human_import.add_argument(
        "--batch", required=True, help="Batch JSON file."
    )
    eval_human_import.add_argument(
        "--results", required=True, help="Results file (JSON or CSV)."
    )
    eval_human_import.add_argument(
        "--output", help="Output updated batch JSON (default: overwrite input)."
    )
    eval_human_import.set_defaults(func=_evaluation_human_import_command)

    # ==========================================================================
    # Active Learning Commands
    # ==========================================================================
    al_parser = subparsers.add_parser(
        "active-learning", help="Active learning utilities."
    )
    al_sub = al_parser.add_subparsers(dest="active_learning_command")

    # active-learning sample
    al_sample = al_sub.add_parser(
        "sample", help="Sample by uncertainty."
    )
    al_sample.add_argument(
        "--input", required=True, help="Input JSONL file."
    )
    al_sample.add_argument(
        "--n", type=int, default=100, help="Number of samples (default: 100)."
    )
    al_sample.add_argument(
        "--output", help="Output JSONL file."
    )
    al_sample.add_argument(
        "--electra", help="Path to ELECTRA model (optional)."
    )
    al_sample.set_defaults(func=_active_learning_sample_command)

    # active-learning curriculum
    al_curriculum = al_sub.add_parser(
        "curriculum", help="Get samples for curriculum stage."
    )
    al_curriculum.add_argument(
        "--input", required=True, help="Input JSONL file."
    )
    al_curriculum.add_argument(
        "--stage", default="simple",
        choices=["simple", "moderate", "complex", "advanced"],
        help="Curriculum stage (default: simple)."
    )
    al_curriculum.add_argument(
        "--plan", action="store_true", help="Show curriculum plan instead of filtering."
    )
    al_curriculum.add_argument(
        "--output", help="Output JSONL file."
    )
    al_curriculum.set_defaults(func=_active_learning_curriculum_command)

    # active-learning queue add
    al_queue_add = al_sub.add_parser(
        "queue-add", help="Add samples to priority queue."
    )
    al_queue_add.add_argument(
        "--input", required=True, help="Input JSONL file."
    )
    al_queue_add.add_argument(
        "--queue", required=True, help="Queue storage file."
    )
    al_queue_add.add_argument(
        "--electra", help="Path to ELECTRA model (optional)."
    )
    al_queue_add.set_defaults(func=_active_learning_queue_add_command)

    # active-learning queue get
    al_queue_get = al_sub.add_parser(
        "queue-get", help="Get next batch from priority queue."
    )
    al_queue_get.add_argument(
        "--queue", required=True, help="Queue storage file."
    )
    al_queue_get.add_argument(
        "--n", type=int, default=10, help="Number of items (default: 10)."
    )
    al_queue_get.add_argument(
        "--output", help="Output JSON file."
    )
    al_queue_get.set_defaults(func=_active_learning_queue_get_command)

    # active-learning queue status
    al_queue_status = al_sub.add_parser(
        "queue-status", help="Show priority queue status."
    )
    al_queue_status.add_argument(
        "--queue", required=True, help="Queue storage file."
    )
    al_queue_status.set_defaults(func=_active_learning_queue_status_command)

    # Generator Commands
    gen_parser = subparsers.add_parser(
        "generator", help="Generate assembly code using models."
    )
    gen_sub = gen_parser.add_subparsers(dest="generator_command")

    # generator model
    gen_model = gen_sub.add_parser(
        "model", help="Generate using trained model or API."
    )
    gen_model.add_argument(
        "--instruction", "-i", required=True, help="Natural language instruction."
    )
    gen_model.add_argument(
        "--model", "-m", help="Path to local model (for mlx/huggingface/llama_cpp)."
    )
    gen_model.add_argument(
        "--type", "-t",
        choices=["mlx", "huggingface", "llama_cpp", "api"],
        default="api",
        help="Model type (default: api)."
    )
    gen_model.add_argument(
        "--provider", "-p",
        choices=["gemini", "claude", "openai"],
        default="gemini",
        help="API provider (default: gemini)."
    )
    gen_model.add_argument(
        "--model-name", help="Model name for API or HuggingFace hub."
    )
    gen_model.add_argument(
        "--api-key", help="API key (or use environment variable)."
    )
    gen_model.add_argument(
        "--context", "-c", help="Optional context to include."
    )
    gen_model.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature."
    )
    gen_model.add_argument(
        "--max-tokens", type=int, default=1024, help="Max tokens to generate."
    )
    gen_model.add_argument(
        "--output", "-o", help="Output file path."
    )
    gen_model.set_defaults(func=_generator_model_command)

    # generator knowledge
    gen_knowledge = gen_sub.add_parser(
        "knowledge", help="Generate with ALTTP knowledge context."
    )
    gen_knowledge.add_argument(
        "--instruction", "-i", required=True, help="Natural language instruction."
    )
    gen_knowledge.add_argument(
        "--entities", "-e", help="Comma-separated list of required entities."
    )
    gen_knowledge.add_argument(
        "--suggest", action="store_true", help="Only suggest entities, don't generate."
    )
    gen_knowledge.add_argument(
        "--provider", "-p",
        choices=["gemini", "claude", "openai"],
        default="gemini",
        help="API provider (default: gemini)."
    )
    gen_knowledge.add_argument(
        "--api-key", help="API key (or use environment variable)."
    )
    gen_knowledge.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature."
    )
    gen_knowledge.add_argument(
        "--max-tokens", type=int, default=1024, help="Max tokens to generate."
    )
    gen_knowledge.add_argument(
        "--output", "-o", help="Output file path."
    )
    gen_knowledge.set_defaults(func=_generator_knowledge_command)

    # generator batch
    gen_batch = gen_sub.add_parser(
        "batch", help="Generate from multiple instructions."
    )
    gen_batch.add_argument(
        "--input", required=True, help="Input file with instructions (one per line or JSONL)."
    )
    gen_batch.add_argument(
        "--output", "-o", required=True, help="Output JSONL file."
    )
    gen_batch.add_argument(
        "--provider", "-p",
        choices=["gemini", "claude", "openai"],
        default="gemini",
        help="API provider (default: gemini)."
    )
    gen_batch.add_argument(
        "--api-key", help="API key (or use environment variable)."
    )
    gen_batch.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature."
    )
    gen_batch.add_argument(
        "--max-tokens", type=int, default=1024, help="Max tokens to generate."
    )
    gen_batch.set_defaults(func=_generator_batch_command)

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
    if args.command == "agents" and not getattr(args, "agents_command", None):
        parser.print_help()
        return 1
    if args.command == "orchestrator" and not getattr(args, "orchestrator_command", None):
        parser.print_help()
        return 1
    if args.command == "studio" and not getattr(args, "studio_command", None):
        parser.print_help()
        return 1
    if args.command == "generators" and not getattr(args, "generators_command", None):
        parser.print_help()
        return 1
    if args.command == "training" and not getattr(args, "training_command", None):
        parser.print_help()
        return 1
    if args.command == "discriminator" and not getattr(args, "discriminator_command", None):
        parser.print_help()
        return 1
    if args.command == "tokenizer" and not getattr(args, "tokenizer_command", None):
        parser.print_help()
        return 1
    if args.command == "encoder" and not getattr(args, "encoder_command", None):
        parser.print_help()
        return 1
    if args.command == "entity" and not getattr(args, "entity_command", None):
        parser.print_help()
        return 1
    if args.command == "scoring" and not getattr(args, "scoring_command", None):
        parser.print_help()
        return 1
    if args.command == "pipeline" and not getattr(args, "pipeline_command", None):
        parser.print_help()
        return 1
    if args.command == "evaluation" and not getattr(args, "evaluation_command", None):
        parser.print_help()
        return 1
    if args.command == "active-learning" and not getattr(args, "active_learning_command", None):
        parser.print_help()
        return 1
    if args.command == "generator" and not getattr(args, "generator_command", None):
        parser.print_help()
        return 1
    return args.func(args)


def register_remaining_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register parsers for commands not yet migrated to cli/ modules.

    This function is called by cli/__init__.py to add commands that
    haven't been migrated to the new modular structure yet.

    All commands have been migrated to cli/ modules:
    - init, plugins, status, services, agents, orchestrator, studio (core.py)
    - context, graph, workspace (context.py)
    - training, discriminator (training.py)
    - generators (generators.py)
    - tokenizer (tokenizer.py)
    - encoder (encoder.py)
    - entity (entity.py)
    - scoring, pipeline, evaluation (pipeline.py)
    - active-learning (active_learning.py)
    - generator (generator.py)

    This function is now empty but kept for backwards compatibility.
    """
    pass


if __name__ == "__main__":
    raise SystemExit(main())
