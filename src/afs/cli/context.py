"""Context and workspace CLI commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._utils import load_manager, parse_mount_type, resolve_context_paths


def _mount_to_dict(mount) -> dict:
    return {
        "name": mount.name,
        "source": str(mount.source),
        "mount_type": mount.mount_type.value,
        "is_symlink": mount.is_symlink,
    }


def _context_to_dict(context) -> dict:
    mounts: dict[str, list[dict]] = {}
    for mount_type, mount_list in context.mounts.items():
        mounts[mount_type.value] = [_mount_to_dict(mount) for mount in mount_list]
    metadata = context.metadata.to_dict() if context.metadata else {}
    return {
        "path": str(context.path),
        "project_name": context.project_name,
        "is_valid": context.is_valid,
        "total_mounts": context.total_mounts,
        "metadata": metadata,
        "mounts": mounts,
    }


def context_init_command(args: argparse.Namespace) -> int:
    """Initialize context for a project."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    project_path, _context_path, context_root, context_dir = resolve_context_paths(
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


def context_ensure_command(args: argparse.Namespace) -> int:
    """Ensure context exists for a project."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    project_path, _context_path, context_root, context_dir = resolve_context_paths(
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


def context_list_command(args: argparse.Namespace) -> int:
    """List context mounts."""
    from ..models import MountType

    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    context = manager.list_context(context_path=context_path)
    if args.json:
        print(json.dumps(_context_to_dict(context), indent=2))
        return 0

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


def context_mount_command(args: argparse.Namespace) -> int:
    """Mount a resource to context."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    mount_type = parse_mount_type(args.mount_type)
    source = Path(args.source).expanduser().resolve()
    mount = manager.mount(
        source=source,
        mount_type=mount_type,
        alias=args.alias,
        context_path=context_path,
    )
    print(f"mounted {mount.name} in {mount.mount_type.value}: {mount.source}")
    return 0


def context_unmount_command(args: argparse.Namespace) -> int:
    """Unmount a resource from context."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    mount_type = parse_mount_type(args.mount_type)
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


def context_validate_command(args: argparse.Namespace) -> int:
    """Validate context integrity."""
    from ..validator import AFSValidator

    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    validator = AFSValidator(context_path, afs_directories=manager.config.directories)
    status = validator.check_integrity()
    missing = ", ".join(status.get("missing", [])) or "(none)"
    errors = status.get("errors", [])
    if args.json:
        payload = {
            "valid": status.get("valid", False),
            "missing": status.get("missing", []),
            "errors": errors,
        }
        print(json.dumps(payload, indent=2))
        return 0 if status.get("valid", False) else 1

    print(f"valid: {status.get('valid', False)}")
    print(f"missing: {missing}")
    if errors:
        print(f"errors: {', '.join(errors)}")
    return 0 if status.get("valid", False) else 1


def context_discover_command(args: argparse.Namespace) -> int:
    """Discover project contexts."""
    from ..config import load_config_model
    from ..discovery import discover_contexts, get_project_stats

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
        if args.json:
            payload = {"contexts": [], "stats": {"total_projects": 0, "total_mounts": 0}}
            print(json.dumps(payload, indent=2))
        else:
            print("(no contexts)")
        return 0
    if args.json:
        payload = {"contexts": [_context_to_dict(project) for project in projects]}
        if args.stats:
            payload["stats"] = get_project_stats(projects)
        print(json.dumps(payload, indent=2))
        return 0
    for project in projects:
        label = project.project_name
        print(f"{label}\t{project.path}")
    if args.stats:
        stats = get_project_stats(projects)
        pairs = [f"{key}={value}" for key, value in stats.items()]
        print("stats: " + ", ".join(pairs))
    return 0


def context_report_command(args: argparse.Namespace) -> int:
    """Generate a summary report of all discovered contexts."""
    from ..config import load_config_model
    from ..discovery import discover_contexts, get_project_stats

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

    stats = get_project_stats(projects) if projects else {
        "total_projects": 0,
        "total_mounts": 0,
    }
    stats["invalid_projects"] = sum(1 for project in projects if not project.is_valid)

    payload = {
        "context_root": str(config.general.context_root)
        if config.general.context_root
        else None,
        "stats": stats,
        "contexts": [_context_to_dict(project) for project in projects],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"context_root: {payload['context_root']}")
    print(f"contexts: {stats['total_projects']}")
    print(f"invalid: {stats['invalid_projects']}")
    print(f"total_mounts: {stats['total_mounts']}")
    return 0


def context_protect_command(args: argparse.Namespace) -> int:
    """Protect a path (manual only)."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    metadata = manager.protect(args.path_to_protect, context_path=context_path)
    print(f"protected: {args.path_to_protect}")
    print(f"manual_only: {', '.join(metadata.manual_only)}")
    return 0


def context_unprotect_command(args: argparse.Namespace) -> int:
    """Unprotect a path."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    metadata = manager.unprotect(args.path_to_unprotect, context_path=context_path)
    print(f"unprotected: {args.path_to_unprotect}")
    print(f"manual_only: {', '.join(metadata.manual_only)}")
    return 0


def context_ensure_all_command(args: argparse.Namespace) -> int:
    """Ensure contexts for all discovered projects."""
    from ..config import load_config_model
    from ..discovery import discover_contexts
    from ..manager import AFSManager

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


def graph_export_command(args: argparse.Namespace) -> int:
    """Export project graph."""
    from ..config import load_config_model
    from ..discovery import discover_contexts
    from ..graph import build_graph, default_graph_path, write_graph

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
    print(f"exported {len(projects)} contexts to {output_path}")
    return 0


def workspace_add_command(args: argparse.Namespace) -> int:
    """Add a workspace directory."""
    from ..config import load_config_model
    from ..schema import WorkspaceDirectory

    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    workspace_path = Path(args.path).expanduser().resolve()

    if not workspace_path.is_dir():
        print(f"path not a directory: {workspace_path}")
        return 1

    existing = [
        ws for ws in config.general.workspace_directories or []
        if Path(ws.path).resolve() == workspace_path
    ]
    if existing:
        print(f"workspace already registered: {workspace_path}")
        return 1

    if config.general.workspace_directories is None:
        config.general.workspace_directories = []
    config.general.workspace_directories.append(
        WorkspaceDirectory(path=workspace_path, description=args.description)
    )

    from ._utils import write_config
    from ..workspace_sync import resolve_config_output

    output = resolve_config_output(Path(args.config) if args.config else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_config(output, config)
    print(f"added workspace: {workspace_path}")
    return 0


def workspace_list_command(args: argparse.Namespace) -> int:
    """List workspace directories."""
    from ..config import load_config_model

    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    workspaces = config.general.workspace_directories or []
    if not workspaces:
        print("(no workspaces)")
        return 0
    for ws in workspaces:
        desc = ws.description or ""
        print(f"{ws.path}\t{desc}")
    return 0


def workspace_remove_command(args: argparse.Namespace) -> int:
    """Remove a workspace directory."""
    from ..config import load_config_model

    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    workspace_path = Path(args.path).expanduser().resolve()

    workspaces = config.general.workspace_directories or []
    new_workspaces = [
        ws for ws in workspaces
        if Path(ws.path).resolve() != workspace_path
    ]
    if len(new_workspaces) == len(workspaces):
        print(f"workspace not found: {workspace_path}")
        return 1

    config.general.workspace_directories = new_workspaces

    from ._utils import write_config
    from ..workspace_sync import resolve_config_output

    output = resolve_config_output(Path(args.config) if args.config else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_config(output, config)
    print(f"removed workspace: {workspace_path}")
    return 0


def workspace_sync_command(args: argparse.Namespace) -> int:
    """Sync workspace directories from WORKSPACE.toml."""
    from ..config import load_config_model
    from ..workspace_sync import (
        load_workspace_entries,
        resolve_config_output,
        sync_workspace_config,
    )
    from ._utils import write_config

    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)

    root = Path(args.root).expanduser().resolve()
    try:
        entries = load_workspace_entries(
            root,
            include_sections=not args.no_sections,
            include_items=not args.no_items,
            include_local=not args.no_local,
        )
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    sync_workspace_config(config, entries, merge=args.merge)

    if args.dry_run:
        for entry in config.general.workspace_directories:
            desc = entry.description or ""
            print(f"{entry.path}\t{desc}")
        return 0

    output = resolve_config_output(config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_config(output, config)
    print(f"synced workspaces: {len(config.general.workspace_directories)}")
    return 0




def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register context and workspace command parsers."""
    from ..models import MountType

    # context
    context_parser = subparsers.add_parser("context", help="Manage project contexts.")
    context_sub = context_parser.add_subparsers(dest="context_command")

    # Common context arguments
    def add_context_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--config", help="Config path.")
        parser.add_argument("--path", help="Project path.")
        parser.add_argument("--context-root", help="Context root override.")
        parser.add_argument("--context-dir", help="Context directory name.")

    # context init
    ctx_init = context_sub.add_parser("init", help="Initialize context.")
    add_context_args(ctx_init)
    ctx_init.add_argument("--link-context", action="store_true", help="Create .context symlink.")
    ctx_init.add_argument("--force", action="store_true", help="Overwrite existing.")
    ctx_init.set_defaults(func=context_init_command)

    # context ensure
    ctx_ensure = context_sub.add_parser("ensure", help="Ensure context exists.")
    add_context_args(ctx_ensure)
    ctx_ensure.add_argument("--link-context", action="store_true", help="Create .context symlink.")
    ctx_ensure.set_defaults(func=context_ensure_command)

    # context list
    ctx_list = context_sub.add_parser("list", help="List context mounts.")
    add_context_args(ctx_list)
    ctx_list.add_argument("--json", action="store_true", help="Output JSON.")
    ctx_list.set_defaults(func=context_list_command)

    # context mount
    ctx_mount = context_sub.add_parser("mount", help="Mount resource to context.")
    add_context_args(ctx_mount)
    ctx_mount.add_argument("mount_type", choices=[t.value for t in MountType], help="Mount type.")
    ctx_mount.add_argument("source", help="Source path.")
    ctx_mount.add_argument("--alias", help="Mount alias.")
    ctx_mount.set_defaults(func=context_mount_command)

    # context unmount
    ctx_unmount = context_sub.add_parser("unmount", help="Unmount resource.")
    add_context_args(ctx_unmount)
    ctx_unmount.add_argument("mount_type", choices=[t.value for t in MountType], help="Mount type.")
    ctx_unmount.add_argument("alias", help="Mount alias.")
    ctx_unmount.set_defaults(func=context_unmount_command)

    # context validate
    ctx_validate = context_sub.add_parser("validate", help="Validate context.")
    add_context_args(ctx_validate)
    ctx_validate.add_argument("--json", action="store_true", help="Output JSON.")
    ctx_validate.set_defaults(func=context_validate_command)

    # context discover
    ctx_discover = context_sub.add_parser("discover", help="Discover contexts.")
    ctx_discover.add_argument("--config", help="Config path.")
    ctx_discover.add_argument("--path", action="append", help="Search paths.")
    ctx_discover.add_argument("--max-depth", type=int, default=3, help="Max search depth.")
    ctx_discover.add_argument("--ignore", action="append", help="Directories to ignore.")
    ctx_discover.add_argument("--stats", action="store_true", help="Show statistics.")
    ctx_discover.add_argument("--json", action="store_true", help="Output JSON.")
    ctx_discover.set_defaults(func=context_discover_command)

    # context report
    ctx_report = context_sub.add_parser("report", help="Summarize discovered contexts.")
    ctx_report.add_argument("--config", help="Config path.")
    ctx_report.add_argument("--path", action="append", help="Search paths.")
    ctx_report.add_argument("--max-depth", type=int, default=3, help="Max search depth.")
    ctx_report.add_argument("--ignore", action="append", help="Directories to ignore.")
    ctx_report.add_argument("--json", action="store_true", help="Output JSON.")
    ctx_report.set_defaults(func=context_report_command)

    # context ensure-all
    ctx_ensure_all = context_sub.add_parser("ensure-all", help="Ensure all discovered contexts.")
    ctx_ensure_all.add_argument("--config", help="Config path.")
    ctx_ensure_all.add_argument("--path", action="append", help="Search paths.")
    ctx_ensure_all.add_argument("--max-depth", type=int, default=3, help="Max search depth.")
    ctx_ensure_all.add_argument("--ignore", action="append", help="Directories to ignore.")
    ctx_ensure_all.add_argument("--dry-run", action="store_true", help="Show what would be done.")
    ctx_ensure_all.set_defaults(func=context_ensure_all_command)

    # context protect
    ctx_protect = context_sub.add_parser("protect", help="Protect a path (manual only).")
    add_context_args(ctx_protect)
    ctx_protect.add_argument("path_to_protect", help="Path to protect.")
    ctx_protect.set_defaults(func=context_protect_command)

    # context unprotect
    ctx_unprotect = context_sub.add_parser("unprotect", help="Unprotect a path.")
    add_context_args(ctx_unprotect)
    ctx_unprotect.add_argument("path_to_unprotect", help="Path to unprotect.")
    ctx_unprotect.set_defaults(func=context_unprotect_command)

    # graph
    graph_parser = subparsers.add_parser("graph", help="Project graph operations.")
    graph_sub = graph_parser.add_subparsers(dest="graph_command")

    graph_export = graph_sub.add_parser("export", help="Export project graph.")
    graph_export.add_argument("--config", help="Config path.")
    graph_export.add_argument("--path", action="append", help="Search paths.")
    graph_export.add_argument("--max-depth", type=int, default=3, help="Max search depth.")
    graph_export.add_argument("--ignore", action="append", help="Directories to ignore.")
    graph_export.add_argument("--output", "-o", help="Output path.")
    graph_export.set_defaults(func=graph_export_command)

    # workspace
    ws_parser = subparsers.add_parser("workspace", help="Manage workspaces.")
    ws_sub = ws_parser.add_subparsers(dest="workspace_command")

    ws_add = ws_sub.add_parser("add", help="Add workspace.")
    ws_add.add_argument("path", help="Workspace path.")
    ws_add.add_argument("--description", help="Workspace description.")
    ws_add.add_argument("--config", help="Config path.")
    ws_add.set_defaults(func=workspace_add_command)

    ws_list = ws_sub.add_parser("list", help="List workspaces.")
    ws_list.add_argument("--config", help="Config path.")
    ws_list.set_defaults(func=workspace_list_command)

    ws_remove = ws_sub.add_parser("remove", help="Remove workspace.")
    ws_remove.add_argument("path", help="Workspace path.")
    ws_remove.add_argument("--config", help="Config path.")
    ws_remove.set_defaults(func=workspace_remove_command)

    ws_sync = ws_sub.add_parser("sync", help="Sync workspaces from WORKSPACE.toml.")
    ws_sync.add_argument(
        "--root",
        default=str(Path.home() / "src"),
        help="Workspace root (default: ~/src).",
    )
    ws_sync.add_argument("--config", help="Config path.")
    ws_sync.add_argument("--no-sections", action="store_true", help="Ignore sections.")
    ws_sync.add_argument("--no-items", action="store_true", help="Ignore items.")
    ws_sync.add_argument("--no-local", action="store_true", help="Ignore local overrides.")
    ws_sync.add_argument("--dry-run", action="store_true", help="Show planned entries.")
    replace_group = ws_sync.add_mutually_exclusive_group()
    replace_group.add_argument("--merge", action="store_true", help="Merge with existing.")
    replace_group.add_argument(
        "--replace", action="store_false", dest="merge", help="Replace existing."
    )
    ws_sync.set_defaults(func=workspace_sync_command, merge=True)
