"""Filesystem CLI commands for agentic context operations."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ..context_fs import ContextFileSystem
from ..context_index import ContextSQLiteIndex
from ._utils import load_manager, parse_mount_type, resolve_context_paths


def fs_read_command(args: argparse.Namespace) -> int:
    """Read a file from a context mount."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    fs = ContextFileSystem(manager, context_path)
    mount_type = parse_mount_type(args.mount_type)
    try:
        content = fs.read_text(
            mount_type,
            args.relative_path,
            encoding=args.encoding,
            errors=args.errors,
        )
        entry = fs.stat_entry(mount_type, args.relative_path)
    except (OSError, ValueError, PermissionError) as exc:
        print(str(exc))
        return 1

    if args.json:
        payload = entry.to_dict()
        payload.update(
            {
                "mount_type": mount_type.value,
                "content": content,
            }
        )
        print(json.dumps(payload, indent=2))
        return 0

    sys.stdout.write(content)
    return 0


def fs_write_command(args: argparse.Namespace) -> int:
    """Write a file into a context mount."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    fs = ContextFileSystem(manager, context_path)
    mount_type = parse_mount_type(args.mount_type)

    if args.content is not None and args.input:
        print("Provide only one of --content or --input.")
        return 1

    if args.input:
        try:
            content = Path(args.input).read_text(
                encoding=args.encoding, errors=args.errors
            )
        except OSError as exc:
            print(str(exc))
            return 1
    elif args.content is not None:
        content = args.content
    else:
        content = sys.stdin.read()

    try:
        target = fs.write_text(
            mount_type,
            args.relative_path,
            content,
            encoding=args.encoding,
            append=args.append,
            mkdirs=args.mkdirs,
        )
    except (OSError, ValueError, PermissionError) as exc:
        print(str(exc))
        return 1

    print(f"wrote: {target}")
    return 0


def fs_list_command(args: argparse.Namespace) -> int:
    """List files in a context mount."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    fs = ContextFileSystem(manager, context_path)
    mount_type = parse_mount_type(args.mount_type)

    include_files = not args.dirs_only
    include_dirs = not args.files_only
    try:
        entries = fs.list_entries(
            mount_type,
            relative_path=args.relative,
            max_depth=args.max_depth,
            glob_patterns=args.glob,
            include_files=include_files,
            include_dirs=include_dirs,
        )
    except (OSError, ValueError, PermissionError) as exc:
        print(str(exc))
        return 1

    if args.json:
        payload = {
            "mount_type": mount_type.value,
            "context_path": str(fs.context_path),
            "entries": [entry.to_dict() for entry in entries],
        }
        print(json.dumps(payload, indent=2))
        return 0

    for entry in entries:
        kind = "dir" if entry.is_dir else "file"
        print(f"{kind}\t{entry.relative_path}")
    return 0


def fs_info_command(args: argparse.Namespace) -> int:
    """Show metadata for a context path."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    fs = ContextFileSystem(manager, context_path)
    mount_type = parse_mount_type(args.mount_type)

    try:
        entry = fs.stat_entry(mount_type, args.relative_path)
    except (OSError, ValueError, PermissionError) as exc:
        print(str(exc))
        return 1

    if args.json:
        payload = entry.to_dict()
        payload["mount_type"] = mount_type.value
        print(json.dumps(payload, indent=2))
        return 0

    print(f"path: {entry.relative_path}")
    print(f"type: {'dir' if entry.is_dir else 'file'}")
    print(f"size: {entry.size_bytes}")
    print(f"modified: {entry.modified_at}")
    return 0


def fs_delete_command(args: argparse.Namespace) -> int:
    """Delete a file or directory from a context mount."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    fs = ContextFileSystem(manager, context_path)
    mount_type = parse_mount_type(args.mount_type)

    try:
        target, root = fs.resolve_path(mount_type, args.relative_path)
    except (OSError, ValueError, PermissionError) as exc:
        print(str(exc))
        return 1

    if target == root:
        print("Refusing to delete an entire mount root.")
        return 1
    if not target.exists() and not target.is_symlink():
        print(f"Path not found: {target}")
        return 1

    is_symlink = target.is_symlink()
    is_dir = target.is_dir() and not is_symlink
    relative_path = target.relative_to(root).as_posix()

    try:
        if is_symlink or target.is_file():
            target.unlink()
        elif is_dir:
            if args.recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()
        else:
            raise OSError(f"Unsupported path type: {target}")
    except (OSError, ValueError, PermissionError) as exc:
        print(str(exc))
        return 1

    index = ContextSQLiteIndex(manager, context_path)
    if is_dir:
        rows_deleted = index.delete_relative_prefix(mount_type, relative_path)
    else:
        rows_deleted = index.delete_relative_path(mount_type, relative_path)

    if args.json:
        print(
            json.dumps(
                {
                    "mount_type": mount_type.value,
                    "context_path": str(fs.context_path),
                    "path": str(target),
                    "relative_path": relative_path,
                    "deleted": True,
                    "recursive": bool(args.recursive),
                    "rows_deleted": rows_deleted,
                },
                indent=2,
            )
        )
        return 0

    print(f"deleted: {target}")
    return 0


def fs_move_command(args: argparse.Namespace) -> int:
    """Move or rename a path within the context filesystem."""
    config_path = Path(args.config) if args.config else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    fs = ContextFileSystem(manager, context_path)
    source_mount_type = parse_mount_type(args.source_mount_type)
    destination_mount_type = parse_mount_type(args.destination_mount_type)

    try:
        source, _source_root = fs.resolve_path(source_mount_type, args.source_relative_path)
        destination, destination_root = fs.resolve_path(
            destination_mount_type, args.destination_relative_path
        )
    except (OSError, ValueError, PermissionError) as exc:
        print(str(exc))
        return 1

    if not source.exists() and not source.is_symlink():
        print(f"Path not found: {source}")
        return 1
    if destination.exists() or destination.is_symlink():
        print(f"Destination already exists: {destination}")
        return 1
    if source.is_dir() and not source.is_symlink():
        try:
            destination.relative_to(source)
        except ValueError:
            pass
        else:
            print("Destination cannot be inside source directory.")
            return 1
    if not destination.parent.exists():
        if not args.mkdirs:
            print(f"Parent directory missing: {destination.parent}")
            return 1
        destination.parent.mkdir(parents=True, exist_ok=True)
    if destination == destination_root:
        print("Destination must include a relative path inside the destination mount.")
        return 1

    try:
        shutil.move(str(source), str(destination))
    except (OSError, ValueError, PermissionError) as exc:
        print(str(exc))
        return 1

    index = ContextSQLiteIndex(manager, context_path)
    mounts_to_rebuild = list({source_mount_type, destination_mount_type})
    summary = index.rebuild(
        mount_types=mounts_to_rebuild,
        include_content=manager.config.context_index.include_content,
        max_file_size_bytes=manager.config.context_index.max_file_size_bytes,
        max_content_chars=manager.config.context_index.max_content_chars,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "context_path": str(fs.context_path),
                    "source_mount_type": source_mount_type.value,
                    "destination_mount_type": destination_mount_type.value,
                    "source": str(source),
                    "destination": str(destination),
                    "index_rebuild": summary.to_dict(),
                },
                indent=2,
            )
        )
        return 0

    print(f"moved: {source} -> {destination}")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register filesystem command parsers."""
    from ..models import MountType

    fs_parser = subparsers.add_parser("fs", help="Agentic filesystem operations.")
    fs_sub = fs_parser.add_subparsers(dest="fs_command")
    mount_choices = [mount.value for mount in MountType]

    def add_context_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--config", help="Config path.")
        parser.add_argument("--path", help="Project path.")
        parser.add_argument("--context-root", help="Context root override.")
        parser.add_argument("--context-dir", help="Context directory name.")

    def add_encoding_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--encoding", default="utf-8", help="Text encoding.")
        parser.add_argument("--errors", default="replace", help="Decode error handling.")

    fs_read = fs_sub.add_parser("read", help="Read a file from context.")
    add_context_args(fs_read)
    fs_read.add_argument("mount_type", choices=mount_choices, help="Mount type.")
    fs_read.add_argument("relative_path", help="Path relative to mount root.")
    add_encoding_args(fs_read)
    fs_read.add_argument("--json", action="store_true", help="Output JSON.")
    fs_read.set_defaults(func=fs_read_command)

    fs_write = fs_sub.add_parser("write", help="Write a file into context.")
    add_context_args(fs_write)
    fs_write.add_argument("mount_type", choices=mount_choices, help="Mount type.")
    fs_write.add_argument("relative_path", help="Path relative to mount root.")
    add_encoding_args(fs_write)
    fs_write.add_argument("--content", help="Inline content to write.")
    fs_write.add_argument("--input", help="Read content from file.")
    fs_write.add_argument("--append", action="store_true", help="Append to file.")
    fs_write.add_argument("--mkdirs", action="store_true", help="Create parent dirs.")
    fs_write.set_defaults(func=fs_write_command)

    fs_list = fs_sub.add_parser("list", help="List files in context.")
    add_context_args(fs_list)
    fs_list.add_argument("mount_type", choices=mount_choices, help="Mount type.")
    fs_list.add_argument("--relative", help="Subpath within mount.")
    fs_list.add_argument("--glob", action="append", help="Glob filter.")
    fs_list.add_argument("--max-depth", type=int, default=1, help="Max depth.")
    fs_list.add_argument("--json", action="store_true", help="Output JSON.")
    list_group = fs_list.add_mutually_exclusive_group()
    list_group.add_argument("--files-only", action="store_true", help="Only files.")
    list_group.add_argument("--dirs-only", action="store_true", help="Only dirs.")
    fs_list.set_defaults(func=fs_list_command)

    fs_info = fs_sub.add_parser("info", help="Show path metadata.")
    add_context_args(fs_info)
    fs_info.add_argument("mount_type", choices=mount_choices, help="Mount type.")
    fs_info.add_argument("relative_path", help="Path relative to mount root.")
    fs_info.add_argument("--json", action="store_true", help="Output JSON.")
    fs_info.set_defaults(func=fs_info_command)

    fs_delete = fs_sub.add_parser("delete", help="Delete a path from context.")
    add_context_args(fs_delete)
    fs_delete.add_argument("mount_type", choices=mount_choices, help="Mount type.")
    fs_delete.add_argument("relative_path", help="Path relative to mount root.")
    fs_delete.add_argument(
        "--recursive", action="store_true", help="Delete directories recursively."
    )
    fs_delete.add_argument("--json", action="store_true", help="Output JSON.")
    fs_delete.set_defaults(func=fs_delete_command)

    fs_move = fs_sub.add_parser("move", help="Move or rename a path in context.")
    add_context_args(fs_move)
    fs_move.add_argument(
        "source_mount_type", choices=mount_choices, help="Source mount type."
    )
    fs_move.add_argument("source_relative_path", help="Source path relative to mount root.")
    fs_move.add_argument(
        "destination_mount_type", choices=mount_choices, help="Destination mount type."
    )
    fs_move.add_argument(
        "destination_relative_path", help="Destination path relative to mount root."
    )
    fs_move.add_argument("--mkdirs", action="store_true", help="Create parent dirs.")
    fs_move.add_argument("--json", action="store_true", help="Output JSON.")
    fs_move.set_defaults(func=fs_move_command)
