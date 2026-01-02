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
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
