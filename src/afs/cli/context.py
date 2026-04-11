"""Context and workspace CLI commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..context_index import ContextSQLiteIndex
from ._utils import load_manager, parse_mount_type, resolve_context_paths


def _mount_to_dict(mount) -> dict:
    payload = {
        "name": mount.name,
        "source": str(mount.source),
        "mount_type": mount.mount_type.value,
        "is_symlink": mount.is_symlink,
    }
    if mount.provenance is not None:
        payload["provenance"] = mount.provenance.to_dict()
    return payload


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
        args, manager, prefer_existing=False
    )
    context = manager.init(
        path=project_path,
        context_root=context_root,
        context_dir=context_dir,
        link_context=args.link_context,
        force=args.force,
        profile=args.profile,
    )
    print(f"context_path: {context.path}")
    print(f"project: {context.project_name}")
    return 0


def context_ensure_command(args: argparse.Namespace) -> int:
    """Ensure context exists for a project."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    project_path, _context_path, context_root, context_dir = resolve_context_paths(
        args, manager, prefer_existing=False
    )
    context = manager.ensure(
        path=project_path,
        context_root=context_root,
        context_dir=context_dir,
        link_context=args.link_context,
        profile=args.profile,
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


def context_repair_command(args: argparse.Namespace) -> int:
    """Repair context mounts, provenance, and optionally the index."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    payload = manager.repair_context(
        context_path=context_path,
        profile_name=args.profile,
        dry_run=args.dry_run,
        reapply_profile=not args.no_profile_reapply,
        remap_missing_sources=not args.no_remap,
        rebuild_index=args.rebuild_index,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"context_path: {payload['context_path']}")
    print(f"dry_run: {str(payload['dry_run']).lower()}")
    print(f"changed: {str(payload['changed']).lower()}")
    actions = payload["actions"] or ["(none)"]
    print("actions:")
    for action in actions:
        print(f"- {action}")
    applied = payload["applied_actions"] or ["(none)"]
    print("applied:")
    for action in applied:
        print(f"- {action}")
    remapped = payload["remapped_mounts"]
    if remapped:
        print("remapped_mounts:")
        for entry in remapped:
            print(
                f"- {entry['mount_type']}/{entry['alias']}: "
                f"{entry['previous_source']} -> {entry['new_source']}"
            )
    print(f"healthy_before: {str(payload['health_before']['healthy']).lower()}")
    print(f"healthy_after: {str(payload['health_after']['healthy']).lower()}")
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
        include_nested=getattr(args, "include_nested", False),
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
        include_nested=getattr(args, "include_nested", False),
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
        include_nested=getattr(args, "include_nested", False),
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
            profile=args.profile,
        )
        print(f"ensured: {context.project_name}\t{context.path}")
    return 0


def context_profile_show_command(args: argparse.Namespace) -> int:
    """Show resolved profile details."""
    from ..config import load_config_model
    from ..profiles import resolve_active_profile

    config_path = Path(args.config) if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    profile = resolve_active_profile(config, profile_name=args.profile)

    payload = {
        "profile": profile.name,
        "extensions": profile.enabled_extensions,
        "policies": profile.policies,
        "memory_mounts": [str(path) for path in profile.memory_mounts],
        "knowledge_mounts": [str(path) for path in profile.knowledge_mounts],
        "skill_roots": [str(path) for path in profile.skill_roots],
        "model_registries": [str(path) for path in profile.model_registries],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    extensions = ", ".join(payload["extensions"]) if payload["extensions"] else "(none)"
    policies = ", ".join(payload["policies"]) if payload["policies"] else "(none)"
    print(f"profile: {payload['profile']}")
    print(f"extensions: {extensions}")
    print(f"policies: {policies}")

    print("memory_mounts:")
    for entry in payload["memory_mounts"]:
        print(f"- {entry}")
    if not payload["memory_mounts"]:
        print("- (none)")

    print("knowledge_mounts:")
    for entry in payload["knowledge_mounts"]:
        print(f"- {entry}")
    if not payload["knowledge_mounts"]:
        print("- (none)")

    print("skill_roots:")
    for entry in payload["skill_roots"]:
        print(f"- {entry}")
    if not payload["skill_roots"]:
        print("- (none)")

    print("model_registries:")
    for entry in payload["model_registries"]:
        print(f"- {entry}")
    if not payload["model_registries"]:
        print("- (none)")
    return 0


def context_profile_apply_command(args: argparse.Namespace) -> int:
    """Apply profile mounts to an existing context."""
    from ..profiles import apply_profile_mounts, resolve_active_profile

    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )

    profile = resolve_active_profile(manager.config, profile_name=args.profile)
    result = apply_profile_mounts(manager, context_path, profile)

    payload = {
        "profile": result.profile_name,
        "mounted": result.mounted,
        "missing": result.skipped_missing,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"profile: {result.profile_name}")
    print(
        "mounted: "
        f"memory={result.mounted.get('memory', 0)} "
        f"knowledge={result.mounted.get('knowledge', 0)} "
        f"skills={result.mounted.get('skills', 0)} "
        f"model_registries={result.mounted.get('model_registries', 0)}"
    )
    if result.skipped_missing:
        print("missing:")
        for path in result.skipped_missing:
            print(f"- {path}")
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

    from ..workspace_sync import resolve_config_output
    from ._utils import write_config

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

    from ..workspace_sync import resolve_config_output
    from ._utils import write_config

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




def context_freshness_command(args: argparse.Namespace) -> int:
    """Show per-file freshness scores."""
    from ..context_index import ContextSQLiteIndex
    from ..models import MountType

    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    index = ContextSQLiteIndex(manager, context_path)

    mount_types = None
    if args.mount:
        try:
            mount_types = [MountType(args.mount)]
        except ValueError:
            print(f"unknown mount type: {args.mount}")
            return 1

    decay_hours = args.decay_hours or manager.config.context_index.decay_hours
    result = index.freshness_scores(
        mount_types=mount_types,
        decay_hours=decay_hours,
        threshold=args.threshold,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    for mount_key, avg_score in result["mount_scores"].items():
        print(f"{mount_key}: avg_score={avg_score:.4f}")
        for f in result["files"].get(mount_key, [])[:10]:
            print(f"  {f['score']:.4f} [{f['status']}] {f['relative_path']}")
    return 0


def _parse_mount_filters(
    raw_mounts: list[str] | None,
) -> list["MountType"] | None:
    from ..models import MountType

    if not raw_mounts:
        return None
    mount_types: list[MountType] = []
    for raw_mount in raw_mounts:
        try:
            mount_types.append(MountType(raw_mount))
        except ValueError as exc:
            raise ValueError(f"unknown mount type: {raw_mount}") from exc
    return mount_types


def _maybe_refresh_context_index(
    *,
    index: ContextSQLiteIndex,
    manager,
    mount_types: list["MountType"] | None,
    auto_index: bool,
    auto_refresh: bool,
) -> dict | None:
    should_rebuild = (
        manager.config.context_index.enabled
        and auto_index
        and (
            not index.has_entries()
            or (auto_refresh and index.needs_refresh(mount_types=mount_types))
        )
    )
    if not should_rebuild:
        return None

    summary = index.rebuild(
        mount_types=mount_types,
        include_content=manager.config.context_index.include_content,
        max_file_size_bytes=manager.config.context_index.max_file_size_bytes,
        max_content_chars=manager.config.context_index.max_content_chars,
    )
    return summary.to_dict()


def context_query_command(args: argparse.Namespace) -> int:
    """Query the SQLite-backed context index."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )

    if not args.query and not args.prefix:
        print("Provide a query string or --prefix for context query.")
        return 1

    try:
        mount_types = _parse_mount_filters(args.mount)
    except ValueError as exc:
        print(str(exc))
        return 1

    index = ContextSQLiteIndex(manager, context_path)
    rebuild_summary = _maybe_refresh_context_index(
        index=index,
        manager=manager,
        mount_types=mount_types,
        auto_index=not args.no_auto_index,
        auto_refresh=not args.no_auto_refresh,
    )
    entries = index.query(
        query=args.query,
        mount_types=mount_types,
        relative_prefix=args.prefix,
        limit=args.limit,
        include_content=args.include_content,
    )

    payload = {
        "context_path": str(context_path),
        "query": args.query or "",
        "relative_prefix": args.prefix or "",
        "count": len(entries),
        "entries": entries,
    }
    if rebuild_summary:
        payload["index_rebuild"] = rebuild_summary

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    if rebuild_summary:
        print("index refreshed before query")
        print()

    if not entries:
        print("(no results)")
        return 0

    for entry in entries:
        line = (
            f"{entry['mount_type']}\t{entry['relative_path']}\t"
            f"{entry['size_bytes']} bytes"
        )
        print(line)
        excerpt = entry.get("content_excerpt")
        if isinstance(excerpt, str) and excerpt.strip():
            print(f"  {excerpt.strip()}")
    return 0


def context_index_rebuild_command(args: argparse.Namespace) -> int:
    """Rebuild the SQLite-backed context index."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )

    try:
        mount_types = _parse_mount_filters(args.mount)
    except ValueError as exc:
        print(str(exc))
        return 1

    include_content = manager.config.context_index.include_content
    if args.include_content:
        include_content = True
    if args.no_include_content:
        include_content = False

    index = ContextSQLiteIndex(manager, context_path)
    summary = index.rebuild(
        mount_types=mount_types,
        include_content=include_content,
        max_file_size_bytes=args.max_file_size_bytes
        or manager.config.context_index.max_file_size_bytes,
        max_content_chars=args.max_content_chars
        or manager.config.context_index.max_content_chars,
    )
    payload = summary.to_dict()

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"context_path: {payload['context_path']}")
    print(f"db_path: {payload['db_path']}")
    print(f"indexed_at: {payload['indexed_at']}")
    print(f"rows_written: {payload['rows_written']}")
    print(f"rows_deleted: {payload['rows_deleted']}")
    if payload["by_mount_type"]:
        print("by_mount_type:")
        for mount_key, count in sorted(payload["by_mount_type"].items()):
            print(f"- {mount_key}: {count}")
    if payload["errors"]:
        print("errors:")
        for error in payload["errors"]:
            print(f"- {error}")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register context and workspace command parsers."""
    from ..models import MountType

    query_epilog = (
        "Examples:\n"
        "  afs context query \"startup guidance\" --path .\n"
        "  afs context query sqlite --path . --mount scratchpad --mount knowledge\n"
        "  afs context query sqlite --path . --prefix docs/sqlite --limit 10 --include-content --json\n"
        "  afs query sqlite --path . --mount knowledge --prefix public/\n"
        "\n"
        "Output fields:\n"
        "  count          number of indexed matches returned\n"
        "  entries[]      mount/path metadata plus content_excerpt (or content when requested)\n"
        "  index_rebuild  present when the command auto-built or auto-refreshed the SQLite index\n"
    )

    # context
    context_parser = subparsers.add_parser("context", help="Manage project contexts.")
    context_sub = context_parser.add_subparsers(dest="context_command")

    # Common context arguments
    def add_context_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--config", help="Config path.")
        parser.add_argument("--path", help="Project path.")
        parser.add_argument("--context-root", help="Context root override.")
        parser.add_argument("--context-dir", help="Context directory name.")
        parser.add_argument("--profile", help="Profile name override.")

    def add_query_args(parser: argparse.ArgumentParser) -> None:
        add_context_args(parser)
        parser.add_argument("query", nargs="?", help="Search string for indexed paths/content.")
        parser.add_argument("--mount", action="append", help="Restrict to a mount type (repeatable).")
        parser.add_argument("--prefix", help="Restrict results to a relative path prefix.")
        parser.add_argument("--limit", type=int, default=25, help="Maximum indexed hits to return.")
        parser.add_argument(
            "--include-content",
            action="store_true",
            help="Include indexed content instead of only excerpts.",
        )
        parser.add_argument(
            "--no-auto-index",
            action="store_true",
            help="Skip automatic index creation when the index is missing.",
        )
        parser.add_argument(
            "--no-auto-refresh",
            action="store_true",
            help="Skip automatic refresh when the index is stale.",
        )
        parser.add_argument("--json", action="store_true", help="Output JSON.")

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

    # context repair
    ctx_repair = context_sub.add_parser("repair", help="Repair context mounts and provenance.")
    add_context_args(ctx_repair)
    ctx_repair.add_argument("--dry-run", action="store_true", help="Show planned repairs only.")
    ctx_repair.add_argument(
        "--no-profile-reapply",
        action="store_true",
        help="Skip reapplying profile-managed mounts.",
    )
    ctx_repair.add_argument(
        "--no-remap",
        action="store_true",
        help="Skip conservative workspace remapping for missing sources.",
    )
    ctx_repair.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the context index if it is stale or empty.",
    )
    ctx_repair.add_argument("--json", action="store_true", help="Output JSON.")
    ctx_repair.set_defaults(func=context_repair_command)

    # context discover
    ctx_discover = context_sub.add_parser("discover", help="Discover contexts.")
    ctx_discover.add_argument("--config", help="Config path.")
    ctx_discover.add_argument("--path", action="append", help="Search paths.")
    ctx_discover.add_argument("--max-depth", type=int, default=3, help="Max search depth.")
    ctx_discover.add_argument("--ignore", action="append", help="Directories to ignore.")
    ctx_discover.add_argument(
        "--include-nested",
        action="store_true",
        help="Continue scanning inside directories that already contain a .context root.",
    )
    ctx_discover.add_argument("--stats", action="store_true", help="Show statistics.")
    ctx_discover.add_argument("--json", action="store_true", help="Output JSON.")
    ctx_discover.set_defaults(func=context_discover_command)

    # context report
    ctx_report = context_sub.add_parser("report", help="Summarize discovered contexts.")
    ctx_report.add_argument("--config", help="Config path.")
    ctx_report.add_argument("--path", action="append", help="Search paths.")
    ctx_report.add_argument("--max-depth", type=int, default=3, help="Max search depth.")
    ctx_report.add_argument("--ignore", action="append", help="Directories to ignore.")
    ctx_report.add_argument(
        "--include-nested",
        action="store_true",
        help="Continue scanning inside directories that already contain a .context root.",
    )
    ctx_report.add_argument("--json", action="store_true", help="Output JSON.")
    ctx_report.set_defaults(func=context_report_command)

    # context ensure-all
    ctx_ensure_all = context_sub.add_parser("ensure-all", help="Ensure all discovered contexts.")
    ctx_ensure_all.add_argument("--config", help="Config path.")
    ctx_ensure_all.add_argument("--path", action="append", help="Search paths.")
    ctx_ensure_all.add_argument("--max-depth", type=int, default=3, help="Max search depth.")
    ctx_ensure_all.add_argument("--ignore", action="append", help="Directories to ignore.")
    ctx_ensure_all.add_argument(
        "--include-nested",
        action="store_true",
        help="Continue scanning inside directories that already contain a .context root.",
    )
    ctx_ensure_all.add_argument("--dry-run", action="store_true", help="Show what would be done.")
    ctx_ensure_all.add_argument("--profile", help="Profile name override.")
    ctx_ensure_all.set_defaults(func=context_ensure_all_command)

    # context profile-show
    ctx_profile_show = context_sub.add_parser("profile-show", help="Show resolved profile.")
    ctx_profile_show.add_argument("--config", help="Config path.")
    ctx_profile_show.add_argument("--profile", help="Profile name override.")
    ctx_profile_show.add_argument("--json", action="store_true", help="Output JSON.")
    ctx_profile_show.set_defaults(func=context_profile_show_command)

    # context profile-apply
    ctx_profile_apply = context_sub.add_parser("profile-apply", help="Apply resolved profile mounts.")
    add_context_args(ctx_profile_apply)
    ctx_profile_apply.add_argument("--json", action="store_true", help="Output JSON.")
    ctx_profile_apply.set_defaults(func=context_profile_apply_command)

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

    # context freshness
    ctx_freshness = context_sub.add_parser("freshness", help="Show per-file freshness scores.")
    add_context_args(ctx_freshness)
    ctx_freshness.add_argument("--mount", help="Filter by mount type.")
    ctx_freshness.add_argument("--threshold", type=float, default=0.0, help="Minimum score threshold.")
    ctx_freshness.add_argument("--decay-hours", type=float, help="Decay window in hours.")
    ctx_freshness.add_argument("--json", action="store_true", help="Output JSON.")
    ctx_freshness.set_defaults(func=context_freshness_command)

    # context query
    ctx_query = context_sub.add_parser(
        "query",
        help="Query indexed context files.",
        description="Query the SQLite-backed context index for path/content matches.",
        epilog=query_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_query_args(ctx_query)
    ctx_query.set_defaults(func=context_query_command)

    # top-level query shortcut
    query_parser = subparsers.add_parser(
        "query",
        help="Shortcut for `afs context query`.",
        description="Shortcut for `afs context query` against the active workspace context.",
        epilog=query_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_query_args(query_parser)
    query_parser.set_defaults(func=context_query_command)

    # top-level index compatibility aliases
    index_parser = subparsers.add_parser(
        "index",
        help="Indexed context search and rebuild helpers.",
    )
    index_sub = index_parser.add_subparsers(dest="index_command")

    idx_query = index_sub.add_parser(
        "query",
        help="Compatibility alias for `afs context query`.",
        description="Compatibility alias for `afs context query`.",
        epilog=query_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_query_args(idx_query)
    idx_query.set_defaults(func=context_query_command)

    idx_rebuild = index_sub.add_parser(
        "rebuild",
        help="Rebuild the context SQLite index.",
    )
    add_context_args(idx_rebuild)
    idx_rebuild.add_argument("--mount", action="append", help="Restrict rebuild to a mount type (repeatable).")
    idx_rebuild.add_argument(
        "--include-content",
        action="store_true",
        help="Force content indexing on for this rebuild.",
    )
    idx_rebuild.add_argument(
        "--no-include-content",
        action="store_true",
        help="Force content indexing off for this rebuild.",
    )
    idx_rebuild.add_argument(
        "--max-file-size-bytes",
        type=int,
        help="Maximum file size to index when content indexing is enabled.",
    )
    idx_rebuild.add_argument(
        "--max-content-chars",
        type=int,
        help="Maximum content chars to retain per indexed file.",
    )
    idx_rebuild.add_argument("--json", action="store_true", help="Output JSON.")
    idx_rebuild.set_defaults(func=context_index_rebuild_command)

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
