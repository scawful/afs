"""Lightweight MCP server exposing AFS context operations over stdio.

NOTE: Registry types, transport functions, and protocol constants have
been extracted to ``afs.mcp``.  This module re-imports them so that
existing callers (``from afs.mcp_server import MCPToolRegistry``) keep
working.  New code should import directly from ``afs.mcp``.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import shutil
import sys
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .agent_scope import is_tool_allowed
from .config import load_config_model
from .context_index import (
    DEFAULT_MAX_CONTENT_CHARS,
    DEFAULT_MAX_FILE_SIZE_BYTES,
    ContextSQLiteIndex,
)
from .context_pack import build_context_pack, render_context_pack
from .context_paths import resolve_mount_root
from .discovery import discover_contexts
from .event_log import read_agent_events
from .manager import AFSManager
from .mcp.registry import (
    CORE_PROMPT_NAMES as _CORE_PROMPT_NAMES_NEW,
)
from .mcp.registry import (
    CORE_RESOURCE_PREFIXES as _CORE_RESOURCE_PREFIXES_BASE,
)
from .mcp.registry import (
    ExtensionMCPStatus,
    MCPExtensionContribution,
    MCPPromptDefinition,
    MCPResourceDefinition,
    MCPToolDefinition,
    MCPToolRegistry,
)
from .mcp.transport import (
    PROTOCOL_VERSION,
    SERVER_NAME,
    SERVER_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
)
from .mcp.transport import (
    error_response as _error_response_fn,
)
from .mcp.transport import (
    read_message as _read_message_fn,
)
from .mcp.transport import (
    success_response as _success_response_fn,
)
from .mcp.transport import (
    write_message as _write_message_fn,
)
from .models import MountType
from .operator_digests import KIND_CHOICES, digest_operator_output
from .plugins import load_enabled_extensions
from .profiles import resolve_active_profile
from .response_schemas import (
    SCHEMA_MIME_TYPE,
    SCHEMA_URI_PREFIX,
    get_response_schema,
    list_response_schema_names,
    list_response_schema_specs,
)
from .schema import ContextIndexConfig
from .session_bootstrap import (
    build_session_bootstrap,
    collect_context_diff,
    collect_context_status,
    render_session_bootstrap,
)

# Extend core prefixes with schema URI prefix from response_schemas
_CORE_RESOURCE_PREFIXES = (*_CORE_RESOURCE_PREFIXES_BASE, SCHEMA_URI_PREFIX)
_CORE_PROMPT_NAMES = _CORE_PROMPT_NAMES_NEW


# --- Backward-compat aliases for classes/functions now in afs.mcp ---
# MCPToolDefinition, MCPResourceDefinition, MCPPromptDefinition,
# MCPExtensionContribution, ExtensionMCPStatus, MCPToolRegistry
# are imported above from .mcp.registry.

_read_message = _read_message_fn
_write_message = _write_message_fn
_error_response = _error_response_fn
_success_response = _success_response_fn


def _allowed_roots(manager: AFSManager) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    def _add(root: Path | None) -> None:
        if root is None:
            return
        resolved = root.expanduser().resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        roots.append(resolved)

    _add(Path.home() / ".context")
    _add(manager.config.general.context_root)
    for workspace in manager.config.general.workspace_directories:
        _add(workspace.path)
    for root in manager.config.general.mcp_allowed_roots:
        _add(root)
    raw_env_roots = os.environ.get("AFS_MCP_ALLOWED_ROOTS", "").strip()
    if raw_env_roots:
        for item in raw_env_roots.split(os.pathsep):
            if item.strip():
                _add(Path(item.strip()))
    local_context = Path.cwd() / ".context"
    if local_context.exists():
        _add(local_context)
    return roots


def _is_allowed_path(path: Path, manager: AFSManager) -> bool:
    resolved = path.expanduser().resolve()
    for root in _allowed_roots(manager):
        if resolved == root or resolved.is_relative_to(root):
            return True
    return False


def _assert_allowed(path: Path, manager: AFSManager) -> Path:
    resolved = path.expanduser().resolve()
    if _is_allowed_path(resolved, manager):
        return resolved
    raise PermissionError(f"Path outside allowed roots: {resolved}")


def _resolve_context_path(arguments: dict[str, Any], manager: AFSManager) -> Path:
    raw = arguments.get("context_path")
    if isinstance(raw, str) and raw.strip():
        return _assert_allowed(Path(raw), manager)
    default = Path.cwd() / ".context"
    return _assert_allowed(default, manager)


def _resolve_project_path(arguments: dict[str, Any]) -> Path:
    raw = arguments.get("project_path", arguments.get("path"))
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_explicit_allowed_context_path(raw: Any, manager: AFSManager) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("context_path must be a non-empty string")
    return _assert_allowed(Path(raw), manager)


def _validate_context_init_scope(
    project_path: Path,
    *,
    context_root: Path | None,
    manager: AFSManager,
) -> None:
    if context_root is not None:
        _assert_allowed(context_root, manager)
        return

    cwd = Path.cwd().resolve()
    resolved_project = project_path.expanduser().resolve()
    if resolved_project == cwd or resolved_project.is_relative_to(cwd):
        return
    for root in _allowed_roots(manager):
        if resolved_project == root or resolved_project.is_relative_to(root):
            return

    raise PermissionError(
        "context.init requires project_path under the current working directory "
        "or under an allowed workspace root, or an explicit context_root under an allowed root"
    )


def _parse_mount_types(raw: Any) -> list[MountType] | None:
    if raw is None:
        return None
    values = raw
    if isinstance(raw, str):
        values = [raw]
    if not isinstance(values, list):
        raise ValueError("mount_types must be a string or list of strings")

    parsed: list[MountType] = []
    seen: set[MountType] = set()
    for item in values:
        if not isinstance(item, str):
            raise ValueError("mount_types must contain only strings")
        mount_type = MountType(item)
        if mount_type in seen:
            continue
        seen.add(mount_type)
        parsed.append(mount_type)
    return parsed


def _coerce_int(
    value: Any,
    *,
    default: int,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int):
        return default
    if value < minimum:
        return default
    if maximum is not None and value > maximum:
        return maximum
    return value


def _context_index_settings(manager: AFSManager) -> ContextIndexConfig:
    return manager.config.context_index


def _resolve_prompt_context_path(arguments: dict[str, Any], manager: AFSManager) -> Path:
    raw_path = arguments.get("context_path")
    if isinstance(raw_path, str) and raw_path.strip():
        return _resolve_explicit_allowed_context_path(raw_path, manager)
    return manager.config.general.context_root


def _discover_allowed_contexts(manager: AFSManager) -> list[Any]:
    contexts: list[Any] = []
    try:
        contexts = discover_contexts(config=manager.config)
    except Exception:
        contexts = []

    default_context = manager.config.general.context_root
    if default_context.exists():
        try:
            contexts.append(manager.list_context(context_path=default_context))
        except Exception:
            pass

    allowed: list[Any] = []
    seen: set[Path] = set()
    for ctx in contexts:
        resolved = ctx.path.expanduser().resolve()
        if resolved in seen or not _is_allowed_path(resolved, manager):
            continue
        seen.add(resolved)
        allowed.append(ctx)
    return allowed


def _context_candidates_for_path(path: Path, manager: AFSManager) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def _add(candidate: Path) -> None:
        resolved = candidate.expanduser().resolve()
        if resolved in seen or not resolved.exists():
            return
        seen.add(resolved)
        candidates.append(resolved)

    for parent in [path.parent, *path.parents]:
        metadata_path = parent / AFSManager.METADATA_FILE
        if metadata_path.exists():
            _add(parent)
        elif parent.name == AFSManager.CONTEXT_DIR_DEFAULT:
            _add(parent)

    for root in _allowed_roots(manager):
        if path == root or path.is_relative_to(root):
            _add(root)

    return candidates


def _sync_context_index_for_path(path: Path, manager: AFSManager) -> bool:
    settings = _context_index_settings(manager)
    if not settings.enabled:
        return False

    for context_path in _context_candidates_for_path(path, manager):
        try:
            index = ContextSQLiteIndex(manager, context_path)
            if not index.sync_absolute_path(
                path,
                include_content=settings.include_content,
                max_file_size_bytes=settings.max_file_size_bytes,
                max_content_chars=settings.max_content_chars,
            ):
                continue
            return True
        except Exception:
            continue
    return False


def _query_context_index(
    *,
    context_path: Path,
    manager: AFSManager,
    query: str,
    mount_types: list[MountType] | None,
    relative_prefix: str | None = None,
    limit: int = 25,
    include_content: bool = False,
    auto_index: bool | None = None,
    auto_refresh: bool | None = None,
    refresh: bool = False,
    max_file_size_bytes: int | None = None,
    max_content_chars: int | None = None,
) -> dict[str, Any]:
    settings = _context_index_settings(manager)
    limit_value = max(1, min(limit, 500))
    effective_auto_index = settings.auto_index if auto_index is None else auto_index
    effective_auto_refresh = settings.auto_refresh if auto_refresh is None else auto_refresh
    effective_max_file_size_bytes = settings.max_file_size_bytes
    if isinstance(max_file_size_bytes, int) and max_file_size_bytes >= 1024:
        effective_max_file_size_bytes = max_file_size_bytes
    effective_max_content_chars = settings.max_content_chars
    if isinstance(max_content_chars, int) and max_content_chars >= 0:
        effective_max_content_chars = max_content_chars

    index = ContextSQLiteIndex(manager, context_path)
    rebuild_summary: dict[str, Any] | None = None
    should_auto_refresh = settings.enabled and effective_auto_index and (
        not index.has_entries(mount_types=mount_types)
        or (effective_auto_refresh and index.needs_refresh(mount_types=mount_types))
    )
    if refresh or should_auto_refresh:
        summary = index.rebuild(
            mount_types=mount_types,
            include_content=settings.include_content,
            max_file_size_bytes=effective_max_file_size_bytes,
            max_content_chars=effective_max_content_chars,
        )
        rebuild_summary = summary.to_dict()

    entries = index.query(
        query=query,
        mount_types=mount_types,
        relative_prefix=relative_prefix,
        limit=limit_value,
        include_content=include_content,
    )
    payload: dict[str, Any] = {
        "context_path": str(context_path),
        "db_path": str(index.db_path),
        "query": query,
        "relative_prefix": relative_prefix or "",
        "mount_types": [mount.value for mount in (mount_types or list(MountType))],
        "count": len(entries),
        "limit": limit_value,
        "entries": entries,
    }
    if rebuild_summary:
        payload["index_rebuild"] = rebuild_summary
    return payload


def _as_text_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=True)}],
        "structuredContent": payload,
    }


def _tool_fs_read(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    path_value = arguments.get("path")
    if not isinstance(path_value, str):
        raise ValueError("path must be a string")
    path = _assert_allowed(Path(path_value), manager)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Path is a directory: {path}")
    return {
        "path": str(path),
        "content": path.read_text(encoding="utf-8", errors="replace"),
    }


def _tool_fs_write(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    path_value = arguments.get("path")
    content = arguments.get("content")
    append = bool(arguments.get("append", False))
    mkdirs = bool(arguments.get("mkdirs", False))
    if not isinstance(path_value, str) or not isinstance(content, str):
        raise ValueError("path and content must be strings")

    path = _assert_allowed(Path(path_value), manager)
    if not path.parent.exists():
        if not mkdirs:
            raise FileNotFoundError(f"Parent directory missing: {path.parent}")
        path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(content)
    index_updated = _sync_context_index_for_path(path, manager)
    return {
        "path": str(path),
        "bytes": len(content.encode("utf-8")),
        "append": append,
        "index_updated": index_updated,
    }


def _tool_fs_delete(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    path_value = arguments.get("path")
    recursive = bool(arguments.get("recursive", False))
    if not isinstance(path_value, str):
        raise ValueError("path must be a string")

    path = _assert_allowed(Path(path_value), manager)
    if not path.exists() and not path.is_symlink():
        raise FileNotFoundError(f"Path not found: {path}")

    for root in _allowed_roots(manager):
        if path == root:
            raise PermissionError(f"Refusing to delete allowed root: {path}")

    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        if recursive:
            shutil.rmtree(path)
        else:
            path.rmdir()
    else:
        raise OSError(f"Unsupported path type: {path}")

    index_updated = _sync_context_index_for_path(path, manager)
    return {
        "path": str(path),
        "deleted": True,
        "recursive": recursive,
        "index_updated": index_updated,
    }


def _tool_fs_move(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    source_value = arguments.get("source")
    destination_value = arguments.get("destination")
    mkdirs = bool(arguments.get("mkdirs", False))

    if not isinstance(source_value, str) or not isinstance(destination_value, str):
        raise ValueError("source and destination must be strings")

    source = _assert_allowed(Path(source_value), manager)
    destination = _assert_allowed(Path(destination_value), manager)

    if not source.exists() and not source.is_symlink():
        raise FileNotFoundError(f"Source path not found: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination already exists: {destination}")
    if source.is_dir() and not source.is_symlink():
        try:
            destination.relative_to(source)
        except ValueError:
            pass
        else:
            raise ValueError("Destination cannot be inside source directory")
    if not destination.parent.exists():
        if not mkdirs:
            raise FileNotFoundError(f"Parent directory missing: {destination.parent}")
        destination.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(source), str(destination))
    source_synced = _sync_context_index_for_path(source, manager)
    destination_synced = _sync_context_index_for_path(destination, manager)

    return {
        "source": str(source),
        "destination": str(destination),
        "index_updated": bool(source_synced or destination_synced),
        "source_index_updated": source_synced,
        "destination_index_updated": destination_synced,
    }


def _tool_fs_list(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    path_value = arguments.get("path")
    max_depth = arguments.get("max_depth", 1)
    if not isinstance(path_value, str):
        raise ValueError("path must be a string")
    if not isinstance(max_depth, int):
        max_depth = 1

    root = _assert_allowed(Path(path_value), manager)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    entries: list[dict[str, Any]] = []
    if root.is_file():
        entries.append({"path": str(root), "is_dir": False})
    else:
        for candidate in root.rglob("*"):
            try:
                depth = len(candidate.relative_to(root).parts)
            except Exception:
                continue
            if max_depth >= 0 and depth > max_depth:
                continue
            entries.append({"path": str(candidate), "is_dir": candidate.is_dir()})
    return {"path": str(root), "entries": entries}


def _tool_context_discover(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    search_paths_value = arguments.get("search_paths", [])
    max_depth = arguments.get("max_depth", 3)

    search_paths: list[Path] | None = None
    if isinstance(search_paths_value, list):
        values: list[Path] = []
        for item in search_paths_value:
            if isinstance(item, str):
                values.append(Path(item).expanduser())
        if values:
            search_paths = values

    if not isinstance(max_depth, int):
        max_depth = 3

    contexts = discover_contexts(search_paths=search_paths, max_depth=max_depth, config=manager.config)
    return {
        "contexts": [
            {
                "project": context.project_name,
                "path": str(context.path),
                "valid": context.is_valid,
                "mounts": context.total_mounts,
            }
            for context in contexts
        ]
    }


def _tool_context_mount(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    context_path = _resolve_context_path(arguments, manager)
    source_value = arguments.get("source")
    mount_type_value = arguments.get("mount_type")
    alias_value = arguments.get("alias")

    if not isinstance(source_value, str):
        raise ValueError("source must be a string")
    if not isinstance(mount_type_value, str):
        raise ValueError("mount_type must be a string")

    source = Path(source_value).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    mount_type = MountType(mount_type_value)
    alias = alias_value if isinstance(alias_value, str) else None
    mount = manager.mount(source, mount_type, alias=alias, context_path=context_path)
    return {
        "context_path": str(context_path),
        "mount": {
            "name": mount.name,
            "mount_type": mount.mount_type.value,
            "source": str(mount.source),
            "is_symlink": mount.is_symlink,
        },
    }


def _tool_context_init(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    project_path = _resolve_project_path(arguments)

    context_root_value = arguments.get("context_root", arguments.get("context_path"))
    context_root = None
    if isinstance(context_root_value, str) and context_root_value.strip():
        context_root = Path(context_root_value).expanduser().resolve()

    context_dir_value = arguments.get("context_dir")
    context_dir = context_dir_value.strip() if isinstance(context_dir_value, str) else None

    profile_value = arguments.get("profile")
    profile = profile_value.strip() if isinstance(profile_value, str) else None

    link_context = bool(arguments.get("link_context", False))
    force = bool(arguments.get("force", False))
    _validate_context_init_scope(project_path, context_root=context_root, manager=manager)

    context = manager.init(
        path=project_path,
        context_root=context_root,
        context_dir=context_dir,
        link_context=link_context,
        force=force,
        profile=profile,
    )
    return {
        "context_path": str(context.path),
        "project": context.project_name,
        "valid": context.is_valid,
        "mounts": context.total_mounts,
    }


def _tool_context_unmount(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    context_path = _resolve_context_path(arguments, manager)
    alias_value = arguments.get("alias")
    mount_type_value = arguments.get("mount_type")
    if not isinstance(alias_value, str) or not alias_value.strip():
        raise ValueError("alias must be a non-empty string")
    if not isinstance(mount_type_value, str):
        raise ValueError("mount_type must be a string")

    mount_type = MountType(mount_type_value)
    removed = manager.unmount(
        alias=alias_value.strip(),
        mount_type=mount_type,
        context_path=context_path,
    )
    return {
        "context_path": str(context_path),
        "alias": alias_value.strip(),
        "mount_type": mount_type.value,
        "removed": bool(removed),
    }


def _tool_context_index_rebuild(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    context_path = _resolve_context_path(arguments, manager)
    mount_types = _parse_mount_types(arguments.get("mount_types"))
    settings = _context_index_settings(manager)
    include_content = bool(arguments.get("include_content", settings.include_content))
    max_file_size_bytes = _coerce_int(
        arguments.get("max_file_size_bytes"),
        default=settings.max_file_size_bytes,
        minimum=1024,
    )
    max_content_chars = _coerce_int(
        arguments.get("max_content_chars"),
        default=settings.max_content_chars,
        minimum=0,
    )
    index = ContextSQLiteIndex(manager, context_path)
    summary = index.rebuild(
        mount_types=mount_types,
        include_content=include_content,
        max_file_size_bytes=max_file_size_bytes,
        max_content_chars=max_content_chars,
    )
    payload = summary.to_dict()
    payload["mount_types"] = [mount.value for mount in (mount_types or list(MountType))]
    return payload


def _tool_context_query(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    context_path = _resolve_context_path(arguments, manager)
    mount_types = _parse_mount_types(arguments.get("mount_types"))
    query_value = arguments.get("query", "")
    if query_value is None:
        query_value = ""
    if not isinstance(query_value, str):
        raise ValueError("query must be a string")

    relative_prefix = arguments.get("relative_prefix")
    if relative_prefix is not None and not isinstance(relative_prefix, str):
        raise ValueError("relative_prefix must be a string")

    include_content = bool(arguments.get("include_content", False))
    auto_index = arguments.get("auto_index")
    if auto_index is not None:
        auto_index = bool(auto_index)
    auto_refresh = arguments.get("auto_refresh")
    if auto_refresh is not None:
        auto_refresh = bool(auto_refresh)
    refresh = bool(arguments.get("refresh", False))
    return _query_context_index(
        context_path=context_path,
        manager=manager,
        query=query_value,
        mount_types=mount_types,
        relative_prefix=relative_prefix,
        limit=_coerce_int(arguments.get("limit"), default=25, minimum=1, maximum=500),
        include_content=include_content,
        auto_index=auto_index,
        auto_refresh=auto_refresh,
        refresh=refresh,
        max_file_size_bytes=arguments.get("max_file_size_bytes"),
        max_content_chars=arguments.get("max_content_chars"),
    )


def _tool_context_diff(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    """Show changes between filesystem and index."""
    context_path = _resolve_context_path(arguments, manager)
    mount_types = _parse_mount_types(arguments.get("mount_types"))
    return collect_context_diff(manager, context_path, mount_types=mount_types)


def _tool_context_status(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    """Return a summary of the context: mounts, index health, profile."""
    context_path = _resolve_context_path(arguments, manager)
    return collect_context_status(manager, context_path)


def _tool_session_pack(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    """Build a token-budgeted context pack for a target model."""
    context_path = _resolve_context_path(arguments, manager)
    query_value = arguments.get("query", "")
    if query_value is None:
        query_value = ""
    if not isinstance(query_value, str):
        raise ValueError("query must be a string")
    model = arguments.get("model", "generic")
    if not isinstance(model, str):
        raise ValueError("model must be a string")
    return build_context_pack(
        manager,
        context_path,
        query=query_value,
        task=str(arguments.get("task", "") or ""),
        model=model,
        workflow=str(arguments.get("workflow", "general") or "general"),
        tool_profile=str(arguments.get("tool_profile", "default") or "default"),
        pack_mode=str(arguments.get("pack_mode", "focused") or "focused"),
        token_budget=_coerce_int(
            arguments.get("token_budget"),
            default=0,
            minimum=0,
            maximum=200000,
        )
        or None,
        include_content=bool(arguments.get("include_content", False)),
        max_query_results=_coerce_int(
            arguments.get("max_query_results"),
            default=6,
            minimum=1,
            maximum=50,
        ),
        max_embedding_results=_coerce_int(
            arguments.get("max_embedding_results"),
            default=4,
            minimum=1,
            maximum=50,
        ),
    )


def _tool_context_repair(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    """Repair provenance, broken mounts, and stale indexes for a context."""
    context_path = _resolve_context_path(arguments, manager)
    dry_run = bool(arguments.get("dry_run", False))
    reapply_profile = bool(arguments.get("reapply_profile", True))
    remap_missing_sources = bool(arguments.get("remap_missing_sources", True))
    rebuild_index = bool(arguments.get("rebuild_index", False))
    profile_name = arguments.get("profile_name")
    if profile_name is not None and not isinstance(profile_name, str):
        raise ValueError("profile_name must be a string")
    return manager.repair_context(
        context_path=context_path,
        profile_name=profile_name,
        dry_run=dry_run,
        reapply_profile=reapply_profile,
        remap_missing_sources=remap_missing_sources,
        rebuild_index=rebuild_index,
    )


_PARAMETER_ROLE_ALIASES: dict[str, set[str]] = {
    "arguments": {"arguments", "args", "input", "params", "payload"},
    "manager": {"manager", "afs_manager", "context_manager", "mgr"},
    "uri": {"uri", "resource_uri", "resource", "path"},
}


def _parameter_role(name: str) -> str | None:
    normalized = name.strip().lstrip("_").lower()
    for role, aliases in _PARAMETER_ROLE_ALIASES.items():
        if normalized in aliases:
            return role
    return None


def _invoke_extension_callable(
    handler: Callable[..., Any],
    *,
    manager: AFSManager,
    fallback_roles: list[str],
    arguments: dict[str, Any] | None = None,
    uri: str | None = None,
) -> Any:
    named_values = {
        "arguments": arguments,
        "manager": manager,
        "uri": uri,
    }
    signature = inspect.signature(handler)
    params = list(signature.parameters.values())
    if not params:
        return handler()

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
        kwargs = {
            role: value
            for role, value in named_values.items()
            if value is not None
        }
        return handler(**kwargs)

    fallback = [
        role for role in fallback_roles if named_values.get(role) is not None
    ]
    fallback_index = 0
    positional_args: list[Any] = []
    keyword_args: dict[str, Any] = {}

    for param in params:
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            while fallback_index < len(fallback):
                positional_args.append(named_values[fallback[fallback_index]])
                fallback_index += 1
            continue

        role = _parameter_role(param.name)
        value = named_values.get(role) if role else None
        if value is None:
            while fallback_index < len(fallback):
                next_role = fallback[fallback_index]
                fallback_index += 1
                candidate = named_values.get(next_role)
                if candidate is not None:
                    value = candidate
                    break
        if value is None and param.default is not inspect.Parameter.empty:
            continue

        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            keyword_args[param.name] = value
        else:
            positional_args.append(value)

    return handler(*positional_args, **keyword_args)


def _agent_supervisor(manager: AFSManager):
    from .agents.supervisor import AgentSupervisor

    return AgentSupervisor(config=manager.config)


def _tool_agent_spawn(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    name_value = arguments.get("name", "")
    module_value = arguments.get("module", "")
    args = arguments.get("args")
    if not isinstance(name_value, str) or not name_value.strip():
        raise ValueError("name is required")
    if not isinstance(module_value, str) or not module_value.strip():
        raise ValueError("module is required")
    if args is None:
        args = []
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ValueError("args must be a list of strings")

    name = name_value.strip()
    module = module_value.strip()
    supervisor = _agent_supervisor(manager)
    try:
        agent = supervisor.spawn(name, module, args)
        return {"name": agent.name, "pid": agent.pid, "state": agent.state}
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc


def _tool_agent_ps(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    supervisor = _agent_supervisor(manager)
    agents = supervisor.list_running()
    return {
        "agents": [
            {
                "name": a.name,
                "state": a.state,
                "pid": a.pid,
                "started_at": a.started_at,
            }
            for a in agents
        ]
    }


def _tool_agent_stop(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    name_value = arguments.get("name", "")
    if not isinstance(name_value, str) or not name_value.strip():
        raise ValueError("name is required")
    name = name_value.strip()
    supervisor = _agent_supervisor(manager)
    stopped = supervisor.stop(name)
    return {"name": name, "stopped": stopped}


def _tool_hivemind_send(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .hivemind import HivemindBus

    from_agent = arguments.get("from", "")
    if not isinstance(from_agent, str) or not from_agent.strip():
        raise ValueError("from is required")
    msg_type = str(arguments.get("type", "status")).strip()
    payload = arguments.get("payload") or {}
    to = arguments.get("to")
    if isinstance(to, str):
        to = to.strip() or None
    topic = arguments.get("topic")
    if isinstance(topic, str):
        topic = topic.strip() or None
    ttl_hours = arguments.get("ttl_hours")
    if ttl_hours is not None:
        ttl_hours = _coerce_int(ttl_hours, default=24, minimum=1, maximum=24 * 30)

    context_path = _resolve_context_path(arguments, manager)
    bus = HivemindBus(context_path, config=manager.config)
    msg = bus.send(
        from_agent.strip(),
        msg_type,
        payload,
        to=to,
        topic=topic,
        ttl_hours=ttl_hours,
    )
    return msg.to_dict()


def _tool_hivemind_read(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .hivemind import HivemindBus

    agent_name = arguments.get("agent")
    if isinstance(agent_name, str):
        agent_name = agent_name.strip() or None
    msg_type = arguments.get("type")
    if isinstance(msg_type, str):
        msg_type = msg_type.strip() or None
    topic = arguments.get("topic")
    if isinstance(topic, str):
        topic = topic.strip() or None
    limit = _coerce_int(arguments.get("limit"), default=50, minimum=1, maximum=500)

    context_path = _resolve_context_path(arguments, manager)
    bus = HivemindBus(context_path, config=manager.config)
    messages = bus.read(agent_name=agent_name, msg_type=msg_type, topic=topic, limit=limit)
    return {"messages": [m.to_dict() for m in messages]}


def _tool_task_create(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .tasks import TaskQueue

    title = arguments.get("title", "")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title is required")
    context_path = _resolve_context_path(arguments, manager)
    priority = _coerce_int(arguments.get("priority"), default=5, minimum=1, maximum=10)
    queue = TaskQueue(context_path)
    task = queue.create(
        title.strip(),
        created_by=str(arguments.get("created_by", "")).strip(),
        priority=priority,
        context=arguments.get("context") or {},
    )
    return task.to_dict()


def _tool_task_list(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .tasks import TaskQueue

    status = arguments.get("status")
    if isinstance(status, str):
        status = status.strip() or None
    context_path = _resolve_context_path(arguments, manager)
    queue = TaskQueue(context_path)
    tasks = queue.list(status=status)
    return {"tasks": [t.to_dict() for t in tasks]}


def _tool_task_claim(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .tasks import TaskQueue

    task_id = arguments.get("task_id", "")
    agent_name = arguments.get("agent_name", "")
    if not task_id or not agent_name:
        raise ValueError("task_id and agent_name are required")
    context_path = _resolve_context_path(arguments, manager)
    queue = TaskQueue(context_path)
    task = queue.claim(str(task_id).strip(), str(agent_name).strip())
    return task.to_dict()


def _tool_task_complete(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .tasks import TaskQueue

    task_id = arguments.get("task_id", "")
    if not task_id:
        raise ValueError("task_id is required")
    result = arguments.get("result")
    context_path = _resolve_context_path(arguments, manager)
    queue = TaskQueue(context_path)
    task = queue.complete(str(task_id).strip(), result=result)
    return task.to_dict()


def _tool_agent_logs(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    agent_name = arguments.get("name", "")
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise ValueError("name is required")
    limit = _coerce_int(arguments.get("limit"), default=20, minimum=1, maximum=500)
    context_path = _resolve_context_path(arguments, manager)
    events = read_agent_events(
        context_path,
        agent_name=agent_name.strip(),
        limit=limit,
        config=manager.config,
    )
    return {"events": events}


def _tool_review_list(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    supervisor = _agent_supervisor(manager)
    agents = supervisor.list_agents()
    awaiting = [a for a in agents if a.state == "awaiting_review"]
    return {
        "agents": [
            {"name": a.name, "state": a.state, "last_event": a.last_event}
            for a in awaiting
        ]
    }


def _tool_review_approve(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    name = arguments.get("name", "")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name is required")
    supervisor = _agent_supervisor(manager)
    approved = supervisor.approve_review(name.strip())
    return {"name": name.strip(), "approved": approved}


def _tool_review_reject(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    name = arguments.get("name", "")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name is required")
    supervisor = _agent_supervisor(manager)
    rejected = supervisor.reject_review(name.strip())
    return {"name": name.strip(), "rejected": rejected}


def _tool_briefing(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    """Run morning briefing and return structured data."""
    from .cli.briefing import _build_briefing
    days = arguments.get("days", 7)
    return _build_briefing(days=days)


def _tool_events_query(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .context_paths import resolve_mount_root
    from .history import query_events

    context_path = _resolve_context_path(arguments, manager)
    history_root = resolve_mount_root(context_path, MountType.HISTORY, config=manager.config)
    event_type = arguments.get("event_type")
    event_types = {event_type} if isinstance(event_type, str) and event_type.strip() else None
    since = arguments.get("since")
    limit = _coerce_int(arguments.get("limit"), default=50, minimum=1, maximum=500)
    source = arguments.get("source")
    if isinstance(source, str):
        source = source.strip() or None
    session_id = arguments.get("session_id")
    if isinstance(session_id, str):
        session_id = session_id.strip() or None
    events = query_events(
        history_root,
        event_types=event_types,
        since=since,
        limit=limit,
        source=source,
        session_id=session_id,
    )
    return {"events": events, "count": len(events)}


def _tool_events_tail(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .context_paths import resolve_mount_root
    from .history import query_events

    context_path = _resolve_context_path(arguments, manager)
    history_root = resolve_mount_root(context_path, MountType.HISTORY, config=manager.config)
    limit = _coerce_int(arguments.get("limit"), default=20, minimum=1, maximum=500)
    events = query_events(history_root, limit=limit)
    return {"events": events, "count": len(events)}


def _tool_events_analytics(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .event_log import summarize_event_analytics

    context_path = _resolve_context_path(arguments, manager)
    lookback_hours = _coerce_int(arguments.get("hours"), default=24, minimum=1, maximum=24 * 30)
    event_type = arguments.get("event_type")
    event_types = [event_type] if isinstance(event_type, str) and event_type.strip() else None
    return summarize_event_analytics(
        context_path,
        lookback_hours=lookback_hours,
        event_types=event_types,
        config=manager.config,
    )


def _tool_events_replay(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .event_log import build_session_replay

    session_id = arguments.get("session_id", "")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id is required")
    context_path = _resolve_context_path(arguments, manager)
    limit = _coerce_int(arguments.get("limit"), default=200, minimum=0, maximum=2000)
    include_payloads = bool(arguments.get("include_payloads", False))
    return build_session_replay(
        context_path,
        session_id=session_id.strip(),
        limit=limit,
        include_payloads=include_payloads,
        config=manager.config,
    )


def _tool_hivemind_subscribe(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .hivemind import HivemindBus

    agent_name = arguments.get("agent_name", "")
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise ValueError("agent_name is required")
    topics = arguments.get("topics", [])
    if not isinstance(topics, list) or not topics:
        raise ValueError("topics must be a non-empty list")

    context_path = _resolve_context_path(arguments, manager)
    bus = HivemindBus(context_path, config=manager.config)
    ttl_hours = arguments.get("ttl_hours")
    if ttl_hours is not None:
        ttl_hours = _coerce_int(ttl_hours, default=24, minimum=1, maximum=24 * 30)
    sub = bus.subscribe(
        agent_name.strip(),
        [str(t).strip() for t in topics if str(t).strip()],
        ttl_hours=ttl_hours,
    )
    return sub.to_dict()


def _tool_hivemind_unsubscribe(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .hivemind import HivemindBus

    agent_name = arguments.get("agent_name", "")
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise ValueError("agent_name is required")
    topics = arguments.get("topics", [])
    if not isinstance(topics, list) or not topics:
        raise ValueError("topics must be a non-empty list")

    context_path = _resolve_context_path(arguments, manager)
    bus = HivemindBus(context_path, config=manager.config)
    sub = bus.unsubscribe(agent_name.strip(), [str(t).strip() for t in topics if str(t).strip()])
    return sub.to_dict()


def _tool_hivemind_reap(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .hivemind import HivemindBus

    context_path = _resolve_context_path(arguments, manager)
    bus = HivemindBus(context_path, config=manager.config)
    max_age_hours = arguments.get("max_age_hours")
    if max_age_hours is not None:
        max_age_hours = _coerce_int(max_age_hours, default=24, minimum=1, maximum=24 * 30)
    return bus.reap(
        max_age_hours=max_age_hours,
        dry_run=bool(arguments.get("dry_run", False)),
    )


def _tool_embeddings_index(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .embeddings import build_embedding_index

    sources_raw = arguments.get("sources", [])
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ValueError("sources must be a non-empty list of paths")
    sources = [Path(s).expanduser().resolve() for s in sources_raw]
    output_dir = Path(str(arguments.get("output_dir", ""))).expanduser().resolve()
    include_patterns = arguments.get("include_patterns")
    exclude_patterns = arguments.get("exclude_patterns")
    incremental = bool(arguments.get("incremental", False))

    result = build_embedding_index(
        sources,
        output_dir,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        incremental=incremental,
    )
    return {
        "summary": result.summary(),
        "total_files": result.total_files,
        "indexed": result.indexed,
        "skipped": result.skipped,
        "reused": getattr(result, "reused", 0),
        "removed": getattr(result, "removed", 0),
        "errors": result.errors[:10],
    }


def _tool_handoff_create(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .handoff import HandoffStore

    agent_name = arguments.get("agent_name", "")
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise ValueError("agent_name is required")

    context_path = _resolve_context_path(arguments, manager)
    store = HandoffStore(context_path, config=manager.config)
    packet = store.create(
        session_id=arguments.get("session_id"),
        agent_name=agent_name.strip(),
        accomplished=arguments.get("accomplished", []),
        blocked=arguments.get("blocked", []),
        next_steps=arguments.get("next_steps", []),
        context_snapshot=arguments.get("context_snapshot", {}),
        open_tasks=arguments.get("open_tasks", []),
        metadata=arguments.get("metadata", {}),
    )
    return packet.to_dict()


def _tool_handoff_read(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .handoff import HandoffStore

    context_path = _resolve_context_path(arguments, manager)
    store = HandoffStore(context_path, config=manager.config)
    session_id = arguments.get("session_id")
    packet = store.read(session_id=session_id)
    if packet is None:
        return {"error": "no handoff packet found"}
    return packet.to_dict()


def _tool_handoff_list(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .handoff import HandoffStore

    context_path = _resolve_context_path(arguments, manager)
    store = HandoffStore(context_path, config=manager.config)
    limit = _coerce_int(arguments.get("limit"), default=10, minimum=1, maximum=100)
    packets = store.list(limit=limit)
    return {"packets": [p.to_dict() for p in packets], "count": len(packets)}


def _tool_hivemind_cleanup(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .hivemind import HivemindBus

    context_path = _resolve_context_path(arguments, manager)
    max_age_hours = _coerce_int(arguments.get("max_age_hours"), default=manager.config.hivemind.default_ttl_hours, minimum=1, maximum=8760)
    dry_run = bool(arguments.get("dry_run", False))
    bus = HivemindBus(context_path)
    return bus.cleanup_stats(max_age_hours=max_age_hours, dry_run=dry_run)


def _tool_memory_status(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .memory_consolidation import memory_status

    context_path = _resolve_context_path(arguments, manager)
    return memory_status(context_path, config=manager.config)


def _tool_memory_search(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .memory_consolidation import search_memory

    context_path = _resolve_context_path(arguments, manager)
    query = str(arguments.get("query", "")).strip()
    if not query:
        return {"error": "query is required", "results": []}
    limit = _coerce_int(arguments.get("limit"), default=10, minimum=1, maximum=100)
    results = search_memory(context_path, query, config=manager.config, limit=limit)
    return {"query": query, "results": results, "count": len(results)}


def _tool_agent_capabilities(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agents import list_agents

    agent_name = arguments.get("agent_name")
    agents = list_agents()
    results = []
    for spec in agents:
        if agent_name and spec.name != agent_name:
            continue
        entry: dict[str, Any] = {"name": spec.name, "description": spec.description}
        if spec.capabilities:
            entry["capabilities"] = {
                "tools": spec.capabilities.tools,
                "topics": spec.capabilities.topics,
                "mount_types": spec.capabilities.mount_types,
                "description": spec.capabilities.description,
            }
        else:
            entry["capabilities"] = None
        results.append(entry)
    return {"agents": results, "count": len(results)}


def _tool_context_freshness(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    context_path = _resolve_context_path(arguments, manager)
    index = ContextSQLiteIndex(manager, context_path)
    mount_type_str = arguments.get("mount_type")
    mount_types = None
    if mount_type_str:
        try:
            mount_types = [MountType(mount_type_str)]
        except ValueError:
            return {"error": f"unknown mount type: {mount_type_str}"}
    decay_hours = float(arguments.get("decay_hours", manager.config.context_index.decay_hours))
    threshold = float(arguments.get("threshold", 0.0))
    return index.freshness_scores(mount_types=mount_types, decay_hours=decay_hours, threshold=threshold)


def _tool_training_antigravity_status(
    arguments: dict[str, Any],
    manager: AFSManager,
) -> dict[str, Any]:
    from .antigravity_status import antigravity_status

    _ = manager
    db_path = arguments.get("db_path")
    state_keys = arguments.get("state_keys")
    resolved_db = (
        Path(db_path).expanduser().resolve()
        if isinstance(db_path, str) and db_path.strip()
        else None
    )
    parsed_state_keys = (
        [str(item).strip() for item in state_keys if str(item).strip()]
        if isinstance(state_keys, list)
        else None
    )
    return antigravity_status(db_path=resolved_db, state_keys=parsed_state_keys)


def _tool_operator_digest(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    _ = manager
    text = arguments.get("text")
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    kind = arguments.get("kind", "auto")
    if not isinstance(kind, str):
        raise ValueError("kind must be a string")

    payload = digest_operator_output(
        text,
        kind=kind,
        max_items=_coerce_int(arguments.get("max_items"), default=5, minimum=1, maximum=20),
    )
    label = arguments.get("label")
    if isinstance(label, str) and label.strip():
        payload["label"] = label.strip()
    return payload


def _tool_session_replay(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .event_log import build_session_timeline

    context_path = _resolve_context_path(arguments, manager)
    session_id = arguments.get("session_id")
    since = arguments.get("since")
    limit = _coerce_int(arguments.get("limit"), default=100, minimum=1, maximum=1000)
    return build_session_timeline(context_path, session_id=session_id, since=since, limit=limit, config=manager.config)


def _tool_alias_definition(
    tool: MCPToolDefinition,
    *,
    name: str,
    description: str | None = None,
) -> MCPToolDefinition:
    return MCPToolDefinition(
        name=name,
        description=description or tool.description,
        input_schema=tool.input_schema,
        handler=tool.handler,
        source=tool.source,
    )


def _builtin_tool_definitions() -> list[MCPToolDefinition]:
    fs_read = MCPToolDefinition(
        name="fs.read",
        description="Legacy compatibility alias for `context.read`. Read UTF-8 text from a context-scoped file after bootstrap or `context.status` anchors the active context.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=_tool_fs_read,
    )
    fs_write = MCPToolDefinition(
        name="fs.write",
        description="Legacy compatibility alias for `context.write`. Write UTF-8 text to a context-scoped file, preferring scratchpad paths for working notes unless the user explicitly wants durable memory or knowledge updates.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "append": {"type": "boolean", "default": False},
                "mkdirs": {"type": "boolean", "default": False},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        handler=_tool_fs_write,
    )
    fs_delete = MCPToolDefinition(
        name="fs.delete",
        description="Legacy compatibility alias for `context.delete`. Delete a file or directory under allowed context roots.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean", "default": False},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=_tool_fs_delete,
    )
    fs_move = MCPToolDefinition(
        name="fs.move",
        description="Legacy compatibility alias for `context.move`. Move or rename a file or directory under allowed context roots.",
        input_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "destination": {"type": "string"},
                "mkdirs": {"type": "boolean", "default": False},
            },
            "required": ["source", "destination"],
            "additionalProperties": False,
        },
        handler=_tool_fs_move,
    )
    fs_list = MCPToolDefinition(
        name="fs.list",
        description="Legacy compatibility alias for `context.list`. List files under a context-scoped path.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_depth": {"type": "integer", "default": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=_tool_fs_list,
    )

    return [
        MCPToolDefinition(
            name="briefing",
            description="Morning briefing — git velocity across all projects, stale project alerts, carry-over items, open tasks, and active agents. Use at the start of a session to understand current state.",
            input_schema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 7, "description": "Lookback window in days."},
                },
                "additionalProperties": False,
            },
            handler=_tool_briefing,
        ),
        fs_read,
        _tool_alias_definition(
            fs_read,
            name="context.read",
            description="Preferred agent-facing file read. Read UTF-8 text from a context-scoped file.",
        ),
        fs_write,
        _tool_alias_definition(
            fs_write,
            name="context.write",
            description="Preferred agent-facing file write. Write UTF-8 text to a context-scoped file.",
        ),
        fs_delete,
        _tool_alias_definition(
            fs_delete,
            name="context.delete",
            description="Preferred agent-facing file delete. Delete a file or directory under allowed context roots.",
        ),
        fs_move,
        _tool_alias_definition(
            fs_move,
            name="context.move",
            description="Preferred agent-facing file move. Move or rename a file or directory under allowed context roots.",
        ),
        fs_list,
        _tool_alias_definition(
            fs_list,
            name="context.list",
            description="Preferred agent-facing file listing. List files under a context-scoped path.",
        ),
        MCPToolDefinition(
            name="context.discover",
            description="Discover project .context roots.",
            input_schema={
                "type": "object",
                "properties": {
                    "search_paths": {"type": "array", "items": {"type": "string"}},
                    "max_depth": {"type": "integer", "default": 3},
                },
                "additionalProperties": False,
            },
            handler=_tool_context_discover,
        ),
        MCPToolDefinition(
            name="context.init",
            description="Initialize a context root for a project path.",
            input_schema={
                "type": "object",
                "properties": {
                    "project_path": {"type": "string"},
                    "context_root": {"type": "string"},
                    "context_dir": {"type": "string"},
                    "profile": {"type": "string"},
                    "link_context": {"type": "boolean", "default": False},
                    "force": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            handler=_tool_context_init,
        ),
        MCPToolDefinition(
            name="context.mount",
            description="Mount a source path into a context mount type.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "source": {"type": "string"},
                    "mount_type": {"type": "string", "enum": [mount.value for mount in MountType]},
                    "alias": {"type": "string"},
                },
                "required": ["source", "mount_type"],
                "additionalProperties": False,
            },
            handler=_tool_context_mount,
        ),
        MCPToolDefinition(
            name="context.unmount",
            description="Unmount a source alias from a context mount type.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "mount_type": {"type": "string", "enum": [mount.value for mount in MountType]},
                    "alias": {"type": "string"},
                },
                "required": ["mount_type", "alias"],
                "additionalProperties": False,
            },
            handler=_tool_context_unmount,
        ),
        MCPToolDefinition(
            name="context.index.rebuild",
            description="Rebuild the SQLite context index. Use this when bootstrap/status reports the index is missing or stale.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "mount_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": [mount.value for mount in MountType]},
                    },
                    "include_content": {"type": "boolean", "default": True},
                    "max_file_size_bytes": {
                        "type": "integer",
                        "default": DEFAULT_MAX_FILE_SIZE_BYTES,
                    },
                    "max_content_chars": {
                        "type": "integer",
                        "default": DEFAULT_MAX_CONTENT_CHARS,
                    },
                },
                "additionalProperties": False,
            },
            handler=_tool_context_index_rebuild,
        ),
        MCPToolDefinition(
            name="context.query",
            description="Search the SQLite context index by path/content. Use this before asking the user for context that may already be in memory, knowledge, or scratchpad.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "query": {"type": "string"},
                    "relative_prefix": {"type": "string"},
                    "mount_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": [mount.value for mount in MountType]},
                    },
                    "limit": {"type": "integer", "default": 25},
                    "include_content": {"type": "boolean", "default": False},
                    "auto_index": {"type": "boolean", "default": True},
                    "auto_refresh": {"type": "boolean", "default": True},
                    "refresh": {"type": "boolean", "default": False},
                    "max_file_size_bytes": {
                        "type": "integer",
                        "default": DEFAULT_MAX_FILE_SIZE_BYTES,
                    },
                    "max_content_chars": {
                        "type": "integer",
                        "default": DEFAULT_MAX_CONTENT_CHARS,
                    },
                },
                "additionalProperties": False,
            },
            handler=_tool_context_query,
        ),
        MCPToolDefinition(
            name="context.diff",
            description="Show new, modified, and deleted files since the last index build. Call this before editing to understand drift since the previous session.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "mount_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": [mount.value for mount in MountType]},
                    },
                },
                "additionalProperties": False,
            },
            handler=_tool_context_diff,
        ),
        MCPToolDefinition(
            name="context.status",
            description="Summary of context health: mount counts, index stats, active profile. Call this first in a new session if `afs.session.bootstrap` is not available.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                },
                "additionalProperties": False,
            },
            handler=_tool_context_status,
        ),
        MCPToolDefinition(
            name="session.pack",
            description="Build a token-budgeted context pack for Gemini, Claude, Codex, or generic clients. Use this after bootstrap when the agent needs a compact cited working set; repeated calls can reuse the last artifact-backed pack when the bootstrap snapshot and pack inputs have not changed.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "query": {"type": "string", "description": "Optional retrieval query."},
                    "task": {
                        "type": "string",
                        "description": "Explicit task statement to render at the end of the pack.",
                    },
                    "model": {
                        "type": "string",
                        "enum": ["generic", "gemini", "claude", "codex"],
                        "default": "generic",
                    },
                    "workflow": {
                        "type": "string",
                        "enum": [
                            "general",
                            "scan_fast",
                            "edit_fast",
                            "review_deep",
                            "root_cause_deep",
                        ],
                        "default": "general",
                    },
                    "tool_profile": {
                        "type": "string",
                        "enum": [
                            "default",
                            "context_readonly",
                            "context_repair",
                            "edit_and_verify",
                            "handoff_only",
                        ],
                        "default": "default",
                    },
                    "pack_mode": {
                        "type": "string",
                        "enum": ["focused", "retrieval", "full_slice"],
                        "default": "focused",
                    },
                    "token_budget": {"type": "integer", "description": "Approximate token budget."},
                    "include_content": {"type": "boolean", "default": False},
                    "max_query_results": {"type": "integer", "default": 6},
                    "max_embedding_results": {"type": "integer", "default": 4},
                },
                "additionalProperties": False,
            },
            handler=_tool_session_pack,
        ),
        MCPToolDefinition(
            name="operator.digest",
            description="Compress noisy command, test, traceback, grep, or diffstat output into a small digest before sending it back into model context.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Raw command or tool output to summarize.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": list(KIND_CHOICES),
                        "default": "auto",
                        "description": "Digest strategy hint. Use auto unless the caller already knows the output type.",
                    },
                    "max_items": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum highlighted entries to keep in the digest.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional human label such as `pytest -q` or `git diff --stat`.",
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=_tool_operator_digest,
        ),
        MCPToolDefinition(
            name="context.repair",
            description="Repair mount provenance, broken mounts, and stale indexes for a context.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "profile_name": {"type": "string"},
                    "dry_run": {"type": "boolean", "default": False},
                    "reapply_profile": {"type": "boolean", "default": True},
                    "remap_missing_sources": {"type": "boolean", "default": True},
                    "rebuild_index": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            handler=_tool_context_repair,
        ),
        MCPToolDefinition(
            name="agent.spawn",
            description="Spawn a background agent process.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Agent name."},
                    "module": {"type": "string", "description": "Python module to run."},
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extra CLI arguments.",
                    },
                },
                "required": ["name", "module"],
                "additionalProperties": False,
            },
            handler=_tool_agent_spawn,
        ),
        MCPToolDefinition(
            name="agent.ps",
            description="List running background agents.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=_tool_agent_ps,
        ),
        MCPToolDefinition(
            name="agent.stop",
            description="Stop a running background agent.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Agent name to stop."},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=_tool_agent_stop,
        ),
        MCPToolDefinition(
            name="agent.logs",
            description="Read recent progress events for a background agent.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "name": {"type": "string", "description": "Agent name."},
                    "limit": {"type": "integer", "description": "Max events to return.", "default": 20},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=_tool_agent_logs,
        ),
        MCPToolDefinition(
            name="hivemind.send",
            description="Send a message to the hivemind inter-agent bus.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "from": {"type": "string", "description": "Sender agent name."},
                    "type": {"type": "string", "description": "Message type: finding, request, or status."},
                    "payload": {"type": "object", "description": "Message payload."},
                    "to": {"type": "string", "description": "Optional recipient agent name."},
                    "topic": {"type": "string", "description": "Optional topic for pub/sub routing (e.g. context:repair, agent:lifecycle)."},
                    "ttl_hours": {"type": "integer", "description": "Optional per-message retention window in hours."},
                },
                "required": ["from", "type"],
                "additionalProperties": False,
            },
            handler=_tool_hivemind_send,
        ),
        MCPToolDefinition(
            name="hivemind.read",
            description="Read messages from the hivemind inter-agent bus.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "agent": {"type": "string", "description": "Filter by sender agent name."},
                    "type": {"type": "string", "description": "Filter by message type."},
                    "topic": {"type": "string", "description": "Filter by topic."},
                    "limit": {"type": "integer", "description": "Max messages to return.", "default": 50},
                },
                "additionalProperties": False,
            },
            handler=_tool_hivemind_read,
        ),
        MCPToolDefinition(
            name="task.create",
            description="Create a task in the items queue.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "title": {"type": "string", "description": "Task title."},
                    "created_by": {"type": "string", "description": "Creator agent name."},
                    "priority": {"type": "integer", "description": "Priority (1=highest, 10=lowest).", "default": 5},
                    "context": {"type": "object", "description": "Task context (files, issue, etc)."},
                },
                "required": ["title"],
                "additionalProperties": False,
            },
            handler=_tool_task_create,
        ),
        MCPToolDefinition(
            name="task.list",
            description="List tasks from the items queue.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "status": {"type": "string", "description": "Filter by status: pending, claimed, in_progress, done, failed."},
                },
                "additionalProperties": False,
            },
            handler=_tool_task_list,
        ),
        MCPToolDefinition(
            name="task.claim",
            description="Claim a pending task for an agent.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "task_id": {"type": "string", "description": "Task ID to claim."},
                    "agent_name": {"type": "string", "description": "Agent claiming the task."},
                },
                "required": ["task_id", "agent_name"],
                "additionalProperties": False,
            },
            handler=_tool_task_claim,
        ),
        MCPToolDefinition(
            name="task.complete",
            description="Mark a task as completed.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "task_id": {"type": "string", "description": "Task ID to complete."},
                    "result": {"type": "object", "description": "Optional result data."},
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
            handler=_tool_task_complete,
        ),
        MCPToolDefinition(
            name="review.list",
            description="List agents awaiting review.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=_tool_review_list,
        ),
        MCPToolDefinition(
            name="review.approve",
            description="Approve an agent's review and transition it back to stopped.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Agent name to approve."},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=_tool_review_approve,
        ),
        MCPToolDefinition(
            name="review.reject",
            description="Reject an agent's review and mark it as failed.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Agent name to reject."},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=_tool_review_reject,
        ),
        MCPToolDefinition(
            name="events.query",
            description="Query the AFS event log with optional type, source, session, since, and limit filters.",
            input_schema={
                "type": "object",
                "properties": {
                    "event_type": {"type": "string", "description": "Filter by event type (mcp_tool, hivemind, embedding, agent_lifecycle, session)."},
                    "since": {"type": "string", "description": "ISO 8601 datetime cutoff."},
                    "limit": {"type": "integer", "default": 50, "description": "Max events to return."},
                    "source": {"type": "string", "description": "Filter by event source."},
                    "session_id": {"type": "string", "description": "Filter by recorded AFS session ID."},
                },
                "additionalProperties": False,
            },
            handler=_tool_events_query,
        ),
        MCPToolDefinition(
            name="events.tail",
            description="Show the most recent AFS events.",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20, "description": "Max events to return."},
                },
                "additionalProperties": False,
            },
            handler=_tool_events_tail,
        ),
        MCPToolDefinition(
            name="events.analytics",
            description="Summarize recent AFS event volume, MCP tool usage, durations, and error rates.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "hours": {"type": "integer", "default": 24, "description": "Lookback window in hours."},
                    "event_type": {"type": "string", "description": "Optional single event type filter."},
                },
                "additionalProperties": False,
            },
            handler=_tool_events_analytics,
        ),
        MCPToolDefinition(
            name="events.replay",
            description="Replay a recorded AFS session timeline by session ID.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "session_id": {"type": "string", "description": "Recorded AFS session ID."},
                    "limit": {"type": "integer", "default": 200, "description": "Max events to return (0 for all)."},
                    "include_payloads": {"type": "boolean", "default": False, "description": "Include event payloads when available."},
                },
                "required": ["session_id"],
                "additionalProperties": False,
            },
            handler=_tool_events_replay,
        ),
        MCPToolDefinition(
            name="hivemind.subscribe",
            description="Subscribe an agent to one or more hivemind topics.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "agent_name": {"type": "string", "description": "Agent name."},
                    "topics": {"type": "array", "items": {"type": "string"}, "description": "Topics to subscribe to."},
                    "ttl_hours": {"type": "integer", "description": "Optional subscription TTL window in hours."},
                },
                "required": ["agent_name", "topics"],
                "additionalProperties": False,
            },
            handler=_tool_hivemind_subscribe,
        ),
        MCPToolDefinition(
            name="hivemind.unsubscribe",
            description="Unsubscribe an agent from one or more hivemind topics.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "agent_name": {"type": "string", "description": "Agent name."},
                    "topics": {"type": "array", "items": {"type": "string"}, "description": "Topics to unsubscribe from."},
                },
                "required": ["agent_name", "topics"],
                "additionalProperties": False,
            },
            handler=_tool_hivemind_unsubscribe,
        ),
        MCPToolDefinition(
            name="hivemind.reap",
            description="Remove expired or stale hivemind messages and return cleanup statistics.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "max_age_hours": {"type": "integer", "description": "Override retention window in hours."},
                    "dry_run": {"type": "boolean", "default": False, "description": "Report removals without deleting files."},
                },
                "additionalProperties": False,
            },
            handler=_tool_hivemind_reap,
        ),
        MCPToolDefinition(
            name="embeddings.index",
            description="Build an embedding index for source paths. Supports incremental mode to skip unchanged files.",
            input_schema={
                "type": "object",
                "properties": {
                    "sources": {"type": "array", "items": {"type": "string"}, "description": "Source paths to index."},
                    "output_dir": {"type": "string", "description": "Output directory for the index."},
                    "include_patterns": {"type": "array", "items": {"type": "string"}, "description": "Glob patterns to include."},
                    "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "Glob patterns to exclude."},
                    "incremental": {"type": "boolean", "default": False, "description": "Skip unchanged files using size+mtime comparison."},
                },
                "required": ["sources", "output_dir"],
                "additionalProperties": False,
            },
            handler=_tool_embeddings_index,
        ),
        MCPToolDefinition(
            name="handoff.create",
            description="Create a conversation handoff packet for the next session.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "agent_name": {"type": "string", "description": "Agent creating the handoff."},
                    "accomplished": {"type": "array", "items": {"type": "string"}, "description": "What got done."},
                    "blocked": {"type": "array", "items": {"type": "string"}, "description": "What's stuck."},
                    "next_steps": {"type": "array", "items": {"type": "string"}, "description": "Recommended actions."},
                    "context_snapshot": {"type": "object", "description": "Scratchpad state, open files, etc."},
                    "open_tasks": {"type": "array", "items": {"type": "object"}, "description": "Open task dicts."},
                    "metadata": {"type": "object", "description": "Freeform metadata."},
                    "session_id": {"type": "string", "description": "Optional session ID (auto-generated if omitted)."},
                },
                "required": ["agent_name"],
                "additionalProperties": False,
            },
            handler=_tool_handoff_create,
        ),
        MCPToolDefinition(
            name="handoff.read",
            description="Read a handoff packet. Returns the latest if no session_id is given.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "session_id": {"type": "string", "description": "Session ID to read (latest if omitted)."},
                },
                "additionalProperties": False,
            },
            handler=_tool_handoff_read,
        ),
        MCPToolDefinition(
            name="handoff.list",
            description="List recent handoff packets.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "limit": {"type": "integer", "default": 10, "description": "Max packets to return."},
                },
                "additionalProperties": False,
            },
            handler=_tool_handoff_list,
        ),
        MCPToolDefinition(
            name="hivemind.cleanup",
            description="Clean up old hivemind messages. Returns per-agent removal stats.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string", "description": "Context path."},
                    "max_age_hours": {"type": "integer", "default": 24, "description": "Max message age in hours."},
                    "dry_run": {"type": "boolean", "default": False, "description": "Preview without deleting."},
                },
                "additionalProperties": False,
            },
            handler=_tool_hivemind_cleanup,
        ),
        MCPToolDefinition(
            name="memory.status",
            description="Show memory pipeline health: entry count, cursor position, staleness.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string", "description": "Context path."},
                },
                "additionalProperties": False,
            },
            handler=_tool_memory_status,
        ),
        MCPToolDefinition(
            name="memory.search",
            description="Search durable memory entries by keyword.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string", "description": "Context path."},
                    "query": {"type": "string", "description": "Search query."},
                    "limit": {"type": "integer", "default": 10, "description": "Max results."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=_tool_memory_search,
        ),
        MCPToolDefinition(
            name="agent.capabilities",
            description="List agent capabilities: declared tools, topics, and mount types.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string", "description": "Filter by agent name."},
                },
                "additionalProperties": False,
            },
            handler=_tool_agent_capabilities,
        ),
        MCPToolDefinition(
            name="context.freshness",
            description="Compute per-file freshness scores showing which context files are stale.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string", "description": "Context path."},
                    "mount_type": {"type": "string", "description": "Filter by mount type."},
                    "decay_hours": {"type": "number", "default": 168.0, "description": "Decay window in hours."},
                    "threshold": {"type": "number", "default": 0.0, "description": "Min score threshold."},
                },
                "additionalProperties": False,
            },
            handler=_tool_context_freshness,
        ),
        MCPToolDefinition(
            name="training.antigravity.status",
            description="Show Antigravity capture database status for editor integrations and export readiness checks.",
            input_schema={
                "type": "object",
                "properties": {
                    "db_path": {
                        "type": "string",
                        "description": "Optional Antigravity state.vscdb override.",
                    },
                    "state_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional Antigravity state key override list.",
                    },
                },
                "additionalProperties": False,
            },
            handler=_tool_training_antigravity_status,
        ),
        MCPToolDefinition(
            name="session.replay",
            description="Replay a session timeline showing chronological events.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string", "description": "Context path."},
                    "session_id": {"type": "string", "description": "Session ID (date) to filter."},
                    "since": {"type": "string", "description": "Filter events after this datetime."},
                    "limit": {"type": "integer", "default": 100, "description": "Max events."},
                },
                "additionalProperties": False,
            },
            handler=_tool_session_replay,
        ),
    ]


@contextmanager
def _prepend_paths(paths: Iterable[Path]):
    values = [str(path) for path in paths if path and path.exists()]
    if not values:
        yield
        return
    original = list(sys.path)
    sys.path = values + original
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


def _purge_extension_module_cache(module_name: str, search_roots: list[Path]) -> None:
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
        if isinstance(module_file, str) and _path_within_roots(Path(module_file), search_roots):
            continue
        sys.modules.pop(loaded_name, None)


def _invoke_tool_handler(
    handler: Callable[..., dict[str, Any]],
    arguments: dict[str, Any],
    manager: AFSManager,
) -> dict[str, Any]:
    return _invoke_extension_callable(
        handler,
        manager=manager,
        arguments=arguments,
        fallback_roles=["arguments", "manager"],
    )


def _invoke_resource_handler(
    handler: Callable[..., dict[str, Any]],
    uri: str,
    manager: AFSManager,
) -> dict[str, Any]:
    return _invoke_extension_callable(
        handler,
        manager=manager,
        uri=uri,
        fallback_roles=["uri", "manager"],
    )


def _invoke_prompt_handler(
    handler: Callable[..., list[dict[str, Any]]],
    arguments: dict[str, Any],
    manager: AFSManager,
) -> list[dict[str, Any]]:
    return _invoke_extension_callable(
        handler,
        manager=manager,
        arguments=arguments,
        fallback_roles=["arguments", "manager"],
    )


def _normalize_extension_tools(
    extension_name: str,
    definitions: Any,
    *,
    source: str,
) -> list[MCPToolDefinition]:
    if definitions is None:
        return []
    if isinstance(definitions, MCPToolDefinition):
        payloads = [definitions]
    elif isinstance(definitions, dict):
        payloads = [definitions]
    elif isinstance(definitions, (list, tuple)):
        payloads = list(definitions)
    else:
        raise TypeError(
            f"Extension {extension_name} must return list[dict] from mcp tool factory"
        )

    tools: list[MCPToolDefinition] = []
    for payload in payloads:
        if isinstance(payload, MCPToolDefinition):
            handler = payload.handler

            def _wrapped(
                arguments: dict[str, Any],
                manager: AFSManager,
                _handler=handler,
            ) -> dict[str, Any]:
                return _invoke_tool_handler(_handler, arguments, manager)

            tools.append(
                MCPToolDefinition(
                    name=payload.name.strip(),
                    description=payload.description.strip(),
                    input_schema=payload.input_schema,
                    handler=_wrapped,
                    source=source,
                )
            )
            continue

        if not isinstance(payload, dict):
            raise TypeError(
                f"Extension {extension_name} returned non-dict MCP tool payload"
            )
        name = payload.get("name")
        description = payload.get("description")
        input_schema = payload.get("inputSchema", payload.get("input_schema"))
        handler = payload.get("handler")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Extension {extension_name} returned tool without valid name")
        if not isinstance(description, str):
            description = ""
        if not isinstance(input_schema, dict):
            input_schema = {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            }
        if not callable(handler):
            raise ValueError(
                f"Extension {extension_name} tool '{name}' missing callable handler"
            )

        def _wrapped(
            arguments: dict[str, Any],
            manager: AFSManager,
            _handler=handler,
        ) -> dict[str, Any]:
            return _invoke_tool_handler(_handler, arguments, manager)

        tools.append(
            MCPToolDefinition(
                name=name.strip(),
                description=description.strip(),
                input_schema=input_schema,
                handler=_wrapped,
                source=source,
            )
        )
    return tools


def _normalize_extension_resources(
    extension_name: str,
    definitions: Any,
    *,
    source: str,
) -> list[MCPResourceDefinition]:
    if definitions is None:
        return []
    if isinstance(definitions, MCPResourceDefinition):
        payloads = [definitions]
    elif isinstance(definitions, dict):
        payloads = [definitions]
    elif isinstance(definitions, (list, tuple)):
        payloads = list(definitions)
    else:
        raise TypeError(
            f"Extension {extension_name} must return list[dict] from mcp resource factory"
        )

    resources: list[MCPResourceDefinition] = []
    for payload in payloads:
        if isinstance(payload, MCPResourceDefinition):
            handler = payload.handler
            mime_type = payload.mime_type
            uri = payload.uri.strip()

            def _wrapped(
                resource_uri: str,
                manager: AFSManager,
                _handler=handler,
                _mime_type=mime_type,
                _uri=uri,
            ) -> dict[str, Any]:
                result = _invoke_resource_handler(_handler, resource_uri, manager)
                return _coerce_resource_result(result, uri=_uri, mime_type=_mime_type)

            resources.append(
                MCPResourceDefinition(
                    uri=uri,
                    name=payload.name.strip(),
                    description=payload.description.strip(),
                    mime_type=mime_type,
                    handler=_wrapped,
                    source=source,
                )
            )
            continue

        if not isinstance(payload, dict):
            raise TypeError(
                f"Extension {extension_name} returned non-dict MCP resource payload"
            )

        uri = payload.get("uri")
        name = payload.get("name")
        description = payload.get("description")
        mime_type = payload.get("mimeType", payload.get("mime_type"))
        handler = payload.get("handler", payload.get("reader"))

        if not isinstance(uri, str) or not uri.strip():
            raise ValueError(f"Extension {extension_name} returned resource without valid uri")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Extension {extension_name} returned resource '{uri}' without valid name"
            )
        if not isinstance(description, str):
            description = ""
        if not isinstance(mime_type, str) or not mime_type.strip():
            mime_type = "application/json"
        if not callable(handler):
            raise ValueError(
                f"Extension {extension_name} resource '{uri}' missing callable handler"
            )

        def _wrapped(
            resource_uri: str,
            manager: AFSManager,
            _handler=handler,
            _mime_type=mime_type,
            _uri=uri.strip(),
        ) -> dict[str, Any]:
            result = _invoke_resource_handler(_handler, resource_uri, manager)
            return _coerce_resource_result(result, uri=_uri, mime_type=_mime_type)

        resources.append(
            MCPResourceDefinition(
                uri=uri.strip(),
                name=name.strip(),
                description=description.strip(),
                mime_type=mime_type.strip(),
                handler=_wrapped,
                source=source,
            )
        )
    return resources


def _normalize_prompt_arguments(arguments: Any) -> list[dict[str, Any]]:
    if not isinstance(arguments, list):
        return []
    return [dict(item) for item in arguments if isinstance(item, dict)]


def _coerce_resource_result(
    result: Any,
    *,
    uri: str,
    mime_type: str,
) -> dict[str, Any]:
    if isinstance(result, dict):
        payload = dict(result)
        payload.setdefault("uri", uri)
        payload.setdefault("mimeType", mime_type)
        if "text" not in payload:
            payload["text"] = json.dumps(payload.get("data", payload), ensure_ascii=True)
        return payload
    if isinstance(result, str):
        return {"uri": uri, "mimeType": mime_type, "text": result}
    return {
        "uri": uri,
        "mimeType": mime_type,
        "text": json.dumps(result, ensure_ascii=True),
    }


def _coerce_prompt_result(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, str):
        return [{"role": "user", "content": {"type": "text", "text": result}}]
    if isinstance(result, dict):
        return [dict(result)]
    if isinstance(result, (list, tuple)):
        messages = [dict(item) for item in result if isinstance(item, dict)]
        if messages:
            return messages
    raise TypeError("prompt handler must return string, dict, or list[dict]")


def _normalize_extension_prompts(
    extension_name: str,
    definitions: Any,
    *,
    source: str,
) -> list[MCPPromptDefinition]:
    if definitions is None:
        return []
    if isinstance(definitions, MCPPromptDefinition):
        payloads = [definitions]
    elif isinstance(definitions, dict):
        payloads = [definitions]
    elif isinstance(definitions, (list, tuple)):
        payloads = list(definitions)
    else:
        raise TypeError(
            f"Extension {extension_name} must return list[dict] from mcp prompt factory"
        )

    prompts: list[MCPPromptDefinition] = []
    for payload in payloads:
        if isinstance(payload, MCPPromptDefinition):
            handler = payload.handler

            def _wrapped(
                arguments: dict[str, Any],
                manager: AFSManager,
                _handler=handler,
            ) -> list[dict[str, Any]]:
                result = _invoke_prompt_handler(_handler, arguments, manager)
                return _coerce_prompt_result(result)

            prompts.append(
                MCPPromptDefinition(
                    name=payload.name.strip(),
                    description=payload.description.strip(),
                    arguments=list(payload.arguments),
                    handler=_wrapped,
                    source=source,
                )
            )
            continue

        if not isinstance(payload, dict):
            raise TypeError(
                f"Extension {extension_name} returned non-dict MCP prompt payload"
            )

        name = payload.get("name")
        description = payload.get("description")
        arguments = payload.get("arguments")
        handler = payload.get("handler", payload.get("get_messages"))

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Extension {extension_name} returned prompt without valid name")
        if not isinstance(description, str):
            description = ""
        if not callable(handler):
            raise ValueError(
                f"Extension {extension_name} prompt '{name}' missing callable handler"
            )

        def _wrapped(
            prompt_arguments: dict[str, Any],
            manager: AFSManager,
            _handler=handler,
        ) -> list[dict[str, Any]]:
            result = _invoke_prompt_handler(_handler, prompt_arguments, manager)
            return _coerce_prompt_result(result)

        prompts.append(
            MCPPromptDefinition(
                name=name.strip(),
                description=description.strip(),
                arguments=_normalize_prompt_arguments(arguments),
                handler=_wrapped,
                source=source,
            )
        )
    return prompts


def _normalize_extension_contribution(
    extension_name: str,
    definitions: Any,
    *,
    source: str,
) -> MCPExtensionContribution:
    if definitions is None:
        return MCPExtensionContribution()

    if isinstance(definitions, MCPExtensionContribution):
        return MCPExtensionContribution(
            tools=_normalize_extension_tools(
                extension_name,
                definitions.tools,
                source=source,
            ),
            resources=_normalize_extension_resources(
                extension_name,
                definitions.resources,
                source=source,
            ),
            prompts=_normalize_extension_prompts(
                extension_name,
                definitions.prompts,
                source=source,
            ),
        )

    if isinstance(definitions, dict) and any(
        key in definitions for key in ("tools", "resources", "prompts")
    ):
        return MCPExtensionContribution(
            tools=_normalize_extension_tools(
                extension_name,
                definitions.get("tools"),
                source=source,
            ),
            resources=_normalize_extension_resources(
                extension_name,
                definitions.get("resources"),
                source=source,
            ),
            prompts=_normalize_extension_prompts(
                extension_name,
                definitions.get("prompts"),
                source=source,
            ),
        )

    return MCPExtensionContribution(
        tools=_normalize_extension_tools(extension_name, definitions, source=source)
    )


def _load_extension_surface(
    manager: AFSManager,
    *,
    extension_name: str,
    extension_root: Path,
    surface: str,
    module_name: str,
    factory_name: str,
) -> tuple[MCPExtensionContribution, ExtensionMCPStatus]:
    source = f"extension:{extension_name}"
    search_roots = [extension_root, extension_root.parent]
    status = ExtensionMCPStatus(
        extension=extension_name,
        surface=surface,
        module=module_name,
        factory=factory_name,
    )

    with _prepend_paths(search_roots):
        _purge_extension_module_cache(module_name, search_roots)
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            status.error = f"import failed: {exc}"
            return MCPExtensionContribution(), status

        factory = getattr(module, factory_name, None)
        if not callable(factory):
            status.error = f"factory not callable: {factory_name}"
            return MCPExtensionContribution(), status

        try:
            definitions = _invoke_extension_callable(
                factory,
                manager=manager,
                fallback_roles=["manager"],
            )
            contribution = _normalize_extension_contribution(
                extension_name,
                definitions,
                source=source,
            )
        except Exception as exc:
            status.error = str(exc)
            return MCPExtensionContribution(), status

    status.loaded_tools = [tool.name for tool in contribution.tools]
    status.loaded_resources = [resource.uri for resource in contribution.resources]
    status.loaded_prompts = [prompt.name for prompt in contribution.prompts]
    return contribution, status


def _load_extension_mcp_definitions(
    manager: AFSManager,
) -> tuple[MCPExtensionContribution, list[ExtensionMCPStatus]]:
    merged = MCPExtensionContribution()
    statuses: list[ExtensionMCPStatus] = []

    extensions = load_enabled_extensions(config=manager.config)
    for extension_name, manifest in sorted(extensions.items()):
        surfaces = [
            (
                "mcp_tools",
                manifest.mcp_tools_module.strip(),
                manifest.mcp_tools_factory.strip() or "register_mcp_tools",
            ),
            (
                "mcp_server",
                manifest.mcp_server_module.strip(),
                manifest.mcp_server_factory.strip() or "register_mcp_server",
            ),
        ]
        for surface, module_name, factory_name in surfaces:
            if not module_name:
                continue
            contribution, status = _load_extension_surface(
                manager,
                extension_name=extension_name,
                extension_root=manifest.root,
                surface=surface,
                module_name=module_name,
                factory_name=factory_name,
            )
            statuses.append(status)
            merged.tools.extend(contribution.tools)
            merged.resources.extend(contribution.resources)
            merged.prompts.extend(contribution.prompts)

    return merged, statuses


def _load_profile_mcp_definitions(
    manager: AFSManager,
) -> tuple[MCPExtensionContribution, list[ExtensionMCPStatus]]:
    merged = MCPExtensionContribution()
    statuses: list[ExtensionMCPStatus] = []

    resolved_profile = resolve_active_profile(manager.config)
    if not resolved_profile.mcp_tools:
        return merged, statuses

    extensions = load_enabled_extensions(config=manager.config)
    extension_modules = {
        module_name
        for manifest in extensions.values()
        for module_name in (
            manifest.mcp_tools_module.strip(),
            manifest.mcp_server_module.strip(),
        )
        if module_name
    }

    for module_name in resolved_profile.mcp_tools:
        normalized = module_name.strip()
        if not normalized or normalized in extension_modules:
            continue

        status = ExtensionMCPStatus(
            extension=f"profile:{resolved_profile.name}",
            surface="profile_mcp",
            module=normalized,
            factory="auto",
        )
        statuses.append(status)

        _purge_extension_module_cache(normalized, [])
        try:
            module = importlib.import_module(normalized)
        except Exception as exc:
            status.error = f"import failed: {exc}"
            continue

        factory_name = ""
        factory = getattr(module, "register_mcp_server", None)
        if callable(factory):
            factory_name = "register_mcp_server"
        else:
            factory = getattr(module, "register_mcp_tools", None)
            if callable(factory):
                factory_name = "register_mcp_tools"
        if not callable(factory):
            status.error = "factory not callable: register_mcp_server/register_mcp_tools"
            continue

        status.factory = factory_name
        try:
            definitions = _invoke_extension_callable(
                factory,
                manager=manager,
                fallback_roles=["manager"],
            )
            contribution = _normalize_extension_contribution(
                f"profile:{resolved_profile.name}",
                definitions,
                source=f"profile:{resolved_profile.name}",
            )
        except Exception as exc:
            status.error = str(exc)
            continue

        status.loaded_tools = [tool.name for tool in contribution.tools]
        status.loaded_resources = [resource.uri for resource in contribution.resources]
        status.loaded_prompts = [prompt.name for prompt in contribution.prompts]
        merged.tools.extend(contribution.tools)
        merged.resources.extend(contribution.resources)
        merged.prompts.extend(contribution.prompts)

    return merged, statuses


def build_mcp_registry(manager: AFSManager) -> MCPToolRegistry:
    """Build MCP registry including extension tools, resources, and prompts."""
    from .mcp.hooks import SENSITIVITY_TOOL_NAMES, sensitivity_pre_hook

    registry = MCPToolRegistry()

    for tool in _builtin_tool_definitions():
        # Apply sensitivity pre-hook to filesystem/context tools
        if tool.name in SENSITIVITY_TOOL_NAMES:
            tool = MCPToolDefinition(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
                handler=tool.handler,
                source=tool.source,
                pre_hook=sensitivity_pre_hook,
            )
        registry.add_tool(tool)

    extension_contribution, statuses = _load_extension_mcp_definitions(manager)
    profile_contribution, profile_statuses = _load_profile_mcp_definitions(manager)
    registry.extension_status = statuses + profile_statuses
    for status in statuses:
        if status.error:
            registry.load_errors[f"{status.extension}:{status.surface}"] = status.error
    for status in profile_statuses:
        if status.error:
            registry.load_errors[f"{status.extension}:{status.surface}"] = status.error

    for tool in extension_contribution.tools:
        try:
            registry.add_tool(tool)
        except ValueError as exc:
            registry.load_errors[f"{tool.source}:{tool.name}"] = str(exc)
    for tool in profile_contribution.tools:
        try:
            registry.add_tool(tool)
        except ValueError as exc:
            registry.load_errors[f"{tool.source}:{tool.name}"] = str(exc)

    for resource in extension_contribution.resources:
        try:
            registry.add_resource(resource)
        except ValueError as exc:
            registry.load_errors[f"{resource.source}:{resource.uri}"] = str(exc)
    for resource in profile_contribution.resources:
        try:
            registry.add_resource(resource)
        except ValueError as exc:
            registry.load_errors[f"{resource.source}:{resource.uri}"] = str(exc)

    for prompt in extension_contribution.prompts:
        try:
            registry.add_prompt(prompt)
        except ValueError as exc:
            registry.load_errors[f"{prompt.source}:{prompt.name}"] = str(exc)
    for prompt in profile_contribution.prompts:
        try:
            registry.add_prompt(prompt)
        except ValueError as exc:
            registry.load_errors[f"{prompt.source}:{prompt.name}"] = str(exc)

    return registry


def _tool_specs(registry: MCPToolRegistry | None = None) -> list[dict[str, Any]]:
    if registry is None:
        specs = [tool.to_spec() for tool in _builtin_tool_definitions()]
    else:
        specs = registry.specs()
    return [spec for spec in specs if is_tool_allowed(spec["name"])]


def get_mcp_status(config_path: Path | None = None) -> dict[str, Any]:
    """Return MCP registry and extension status for diagnostics."""
    config = load_config_model(config_path=config_path, merge_user=True)
    manager = AFSManager(config=config)
    registry = build_mcp_registry(manager)
    return {
        "tools": sorted(registry.tools.keys()),
        "resources": sorted(registry.resources.keys()),
        "prompts": sorted(registry.prompts.keys()),
        "extension_status": [status.to_dict() for status in registry.extension_status],
        "load_errors": dict(registry.load_errors),
    }


def _list_resources(
    manager: AFSManager,
    registry: MCPToolRegistry | None = None,
) -> list[dict[str, Any]]:
    """Return MCP resource descriptors for discovered contexts."""
    resources: list[dict[str, Any]] = [
        {
            "uri": "afs://contexts",
            "name": "AFS Contexts",
            "description": "All discovered .context roots",
            "mimeType": "application/json",
        },
        {
            "uri": "afs://claude/bootstrap",
            "name": "Claude Bootstrap",
            "description": "Session bootstrap as Claude-optimized markdown",
            "mimeType": "text/markdown",
        },
    ]
    resources.extend(list_response_schema_specs())
    for ctx in _discover_allowed_contexts(manager):
        ctx_uri = f"afs://context/{ctx.path}"
        resources.append({
            "uri": f"{ctx_uri}/bootstrap",
            "name": f"{ctx.project_name} bootstrap",
            "description": f"Session bootstrap packet for {ctx.project_name}",
            "mimeType": "application/json",
        })
        resources.append({
            "uri": f"{ctx_uri}/metadata",
            "name": f"{ctx.project_name} metadata",
            "description": f"Project metadata for {ctx.project_name}",
            "mimeType": "application/json",
        })
        resources.append({
            "uri": f"{ctx_uri}/mounts",
            "name": f"{ctx.project_name} mounts",
            "description": f"Mount listing for {ctx.project_name}",
            "mimeType": "application/json",
        })
        resources.append({
            "uri": f"{ctx_uri}/index",
            "name": f"{ctx.project_name} index",
            "description": f"Index summary for {ctx.project_name}",
            "mimeType": "application/json",
        })
    if registry is not None:
        resources.extend(registry.resource_specs())
    return resources


def _read_resource(
    uri: str,
    manager: AFSManager,
    registry: MCPToolRegistry | None = None,
) -> dict[str, Any]:
    """Read a single MCP resource by URI."""
    if uri == "afs://contexts":
        data = [
            {
                "project": ctx.project_name,
                "path": str(ctx.path),
                "valid": ctx.is_valid,
                "mounts": ctx.total_mounts,
            }
            for ctx in _discover_allowed_contexts(manager)
        ]
        return {"uri": uri, "mimeType": "application/json", "text": json.dumps(data)}

    if uri == "afs://claude/bootstrap":
        context_path = _resolve_context_path({}, manager)
        payload = build_session_bootstrap(manager, context_path)
        text = render_session_bootstrap(payload)
        return {"uri": uri, "mimeType": "text/markdown", "text": text}

    if uri.startswith(SCHEMA_URI_PREFIX):
        name = uri[len(SCHEMA_URI_PREFIX) :].strip()
        try:
            schema = get_response_schema(name)
        except KeyError as exc:
            raise ValueError(f"Unknown resource URI: {uri}") from exc
        return {
            "uri": uri,
            "mimeType": SCHEMA_MIME_TYPE,
            "text": json.dumps(schema),
        }

    if registry is not None and uri in registry.resources:
        return registry.read_resource(uri, manager)

    prefix = "afs://context/"
    if not uri.startswith(prefix):
        raise ValueError(f"Unknown resource URI: {uri}")

    remainder = uri[len(prefix):]
    if remainder.endswith("/metadata"):
        context_path_str = remainder[: -len("/metadata")]
        context_path = _resolve_explicit_allowed_context_path(context_path_str, manager)
        metadata_file = context_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"metadata.json not found: {metadata_file}")
        text = metadata_file.read_text(encoding="utf-8", errors="replace")
        return {"uri": uri, "mimeType": "application/json", "text": text}

    if remainder.endswith("/bootstrap"):
        context_path_str = remainder[: -len("/bootstrap")]
        context_path = _resolve_explicit_allowed_context_path(context_path_str, manager)
        payload = build_session_bootstrap(manager, context_path)
        return {"uri": uri, "mimeType": "application/json", "text": json.dumps(payload)}

    if remainder.endswith("/mounts"):
        context_path_str = remainder[: -len("/mounts")]
        context_path = _resolve_explicit_allowed_context_path(context_path_str, manager)
        ctx_root = manager.list_context(context_path=context_path)
        data: dict[str, Any] = {}
        for mount_type, mount_list in ctx_root.mounts.items():
            data[mount_type.value] = [
                {"name": m.name, "source": str(m.source), "is_symlink": m.is_symlink}
                for m in mount_list
            ]
        return {"uri": uri, "mimeType": "application/json", "text": json.dumps(data)}

    if remainder.endswith("/index"):
        context_path_str = remainder[: -len("/index")]
        context_path = _resolve_explicit_allowed_context_path(context_path_str, manager)
        index = ContextSQLiteIndex(manager, context_path)
        has = index.has_entries()
        stale = index.needs_refresh() if has else False
        data_dict: dict[str, Any] = {
            "context_path": str(context_path),
            "db_path": str(index.db_path),
            "has_entries": has,
            "needs_refresh": stale,
        }
        return {"uri": uri, "mimeType": "application/json", "text": json.dumps(data_dict)}

    raise ValueError(f"Unknown resource URI: {uri}")


def _list_prompts(registry: MCPToolRegistry | None = None) -> list[dict[str, Any]]:
    """Return MCP prompt descriptors."""
    prompts = [
        {
            "name": "afs.session.bootstrap",
            "description": "Build a session-start bootstrap packet with health, scratchpad, tasks, hivemind, and durable memory. Call this first in a new session.",
            "arguments": [
                {
                    "name": "context_path",
                    "description": "Path to .context root (uses configured default if omitted)",
                    "required": False,
                },
                {
                    "name": "task_limit",
                    "description": "Maximum queued tasks to include (default 10)",
                    "required": False,
                },
                {
                    "name": "message_limit",
                    "description": "Maximum hivemind messages to include (default 10)",
                    "required": False,
                },
            ],
        },
        {
            "name": "afs.session.pack",
            "description": "Build a token-budgeted context pack with cited working context for Gemini, Claude, Codex, or generic clients.",
            "arguments": [
                {
                    "name": "context_path",
                    "description": "Path to .context root (uses configured default if omitted)",
                    "required": False,
                },
                {
                    "name": "query",
                    "description": "Optional retrieval query to bias the pack.",
                    "required": False,
                },
                {
                    "name": "model",
                    "description": "Target model profile: generic, gemini, claude, codex.",
                    "required": False,
                },
                {
                    "name": "task",
                    "description": "Explicit task statement rendered after the context sections.",
                    "required": False,
                },
                {
                    "name": "workflow",
                    "description": "Execution workflow profile: general, scan_fast, edit_fast, review_deep, root_cause_deep.",
                    "required": False,
                },
                {
                    "name": "tool_profile",
                    "description": "Preferred AFS surface mix: default, context_readonly, context_repair, edit_and_verify, handoff_only.",
                    "required": False,
                },
                {
                    "name": "pack_mode",
                    "description": "Context shaping mode: focused, retrieval, or full_slice.",
                    "required": False,
                },
                {
                    "name": "token_budget",
                    "description": "Approximate token budget override.",
                    "required": False,
                },
            ],
        },
        {
            "name": "afs.context.overview",
            "description": "Describe the AFS context structure and available mounts",
            "arguments": [
                {
                    "name": "context_path",
                    "description": "Path to .context root (uses default if omitted)",
                    "required": False,
                },
            ],
        },
        {
            "name": "afs.workflow.structured",
            "description": "Build a structured workflow prompt by combining a session pack with one of the built-in AFS response schemas.",
            "arguments": [
                {
                    "name": "context_path",
                    "description": "Path to .context root (uses configured default if omitted)",
                    "required": False,
                },
                {
                    "name": "schema_name",
                    "description": "Built-in response schema name. Supported: "
                    + ", ".join(list_response_schema_names()),
                    "required": False,
                },
                {
                    "name": "task",
                    "description": "Explicit task statement for the structured response",
                    "required": True,
                },
                {
                    "name": "query",
                    "description": "Optional retrieval query to narrow the context pack",
                    "required": False,
                },
                {
                    "name": "model",
                    "description": "Prompt-shaping target: generic, gemini, claude, or codex.",
                    "required": False,
                },
                {
                    "name": "workflow",
                    "description": "Workflow profile: general, scan_fast, edit_fast, review_deep, or root_cause_deep.",
                    "required": False,
                },
                {
                    "name": "tool_profile",
                    "description": "Optional tool profile hint carried into the session pack.",
                    "required": False,
                },
                {
                    "name": "pack_mode",
                    "description": "Context shaping mode: focused, retrieval, or full_slice.",
                    "required": False,
                },
                {
                    "name": "token_budget",
                    "description": "Optional pack token budget.",
                    "required": False,
                },
            ],
        },
        {
            "name": "afs.query.search",
            "description": "Search the context index with a query",
            "arguments": [
                {
                    "name": "context_path",
                    "description": "Path to .context root (uses configured default if omitted)",
                    "required": False,
                },
                {
                    "name": "query",
                    "description": "Full-text search query",
                    "required": True,
                },
                {
                    "name": "mount_types",
                    "description": "Comma-separated mount types to search (e.g. scratchpad,knowledge)",
                    "required": False,
                },
                {
                    "name": "relative_prefix",
                    "description": "Optional relative path prefix filter",
                    "required": False,
                },
                {
                    "name": "limit",
                    "description": "Optional result limit (default 25)",
                    "required": False,
                },
            ],
        },
        {
            "name": "afs.scratchpad.review",
            "description": "Review the scratchpad for current agent state and deferred tasks",
            "arguments": [
                {
                    "name": "context_path",
                    "description": "Path to .context root (uses default if omitted)",
                    "required": False,
                },
            ],
        },
    ]
    if registry is not None:
        prompts.extend(registry.prompt_specs())
    return prompts


def _get_prompt(
    name: str,
    arguments: dict[str, Any],
    manager: AFSManager,
    registry: MCPToolRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build MCP prompt messages for a named prompt."""
    if name == "afs.session.bootstrap":
        context_path = _resolve_prompt_context_path(arguments, manager)
        payload = build_session_bootstrap(
            manager,
            context_path,
            task_limit=_coerce_int(arguments.get("task_limit"), default=10, minimum=1, maximum=100),
            message_limit=_coerce_int(arguments.get("message_limit"), default=10, minimum=1, maximum=100),
        )
        text = render_session_bootstrap(payload)
        return [{"role": "user", "content": {"type": "text", "text": text}}]

    if name == "afs.session.pack":
        context_path = _resolve_prompt_context_path(arguments, manager)
        payload = build_context_pack(
            manager,
            context_path,
            query=str(arguments.get("query", "") or ""),
            task=str(arguments.get("task", "") or ""),
            model=str(arguments.get("model", "generic") or "generic"),
            workflow=str(arguments.get("workflow", "general") or "general"),
            tool_profile=str(arguments.get("tool_profile", "default") or "default"),
            pack_mode=str(arguments.get("pack_mode", "focused") or "focused"),
            token_budget=_coerce_int(
                arguments.get("token_budget"),
                default=0,
                minimum=0,
                maximum=200000,
            )
            or None,
        )
        text = render_context_pack(payload)
        return [{"role": "user", "content": {"type": "text", "text": text}}]

    if name == "afs.context.overview":
        context_path = _resolve_prompt_context_path(arguments, manager)
        ctx_root = manager.list_context(context_path=context_path)
        lines = [
            f"# AFS Context: {ctx_root.project_name}",
            f"Path: {ctx_root.path}",
            f"Valid: {ctx_root.is_valid}",
            f"Total mounts: {ctx_root.total_mounts}",
            "",
            "## Mounts",
        ]
        for mount_type, mount_list in ctx_root.mounts.items():
            lines.append(f"### {mount_type.value}")
            if mount_list:
                for m in mount_list:
                    lines.append(f"  - {m.name} → {m.source} (symlink={m.is_symlink})")
            else:
                lines.append("  (empty)")
        if ctx_root.metadata:
            lines.append("")
            lines.append("## Metadata")
            lines.append(f"Description: {ctx_root.metadata.description or '(none)'}")
            lines.append(f"Agents: {', '.join(ctx_root.metadata.agents) or '(none)'}")
            if ctx_root.metadata.manual_only:
                lines.append(f"Protected paths: {', '.join(ctx_root.metadata.manual_only)}")
        return [{"role": "user", "content": {"type": "text", "text": "\n".join(lines)}}]

    if name == "afs.workflow.structured":
        schema_names = set(list_response_schema_names())
        schema_name = str(arguments.get("schema_name", "plan") or "plan").strip()
        if schema_name not in schema_names:
            raise ValueError(
                "schema_name must be one of: " + ", ".join(sorted(schema_names))
            )
        task = arguments.get("task", "")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task argument is required")
        context_path = _resolve_prompt_context_path(arguments, manager)
        payload = build_context_pack(
            manager,
            context_path,
            query=str(arguments.get("query", "") or ""),
            task=task,
            model=str(arguments.get("model", "generic") or "generic"),
            workflow=str(arguments.get("workflow", "general") or "general"),
            tool_profile=str(arguments.get("tool_profile", "default") or "default"),
            pack_mode=str(arguments.get("pack_mode", "focused") or "focused"),
            token_budget=_coerce_int(
                arguments.get("token_budget"),
                default=0,
                minimum=0,
                maximum=200000,
            )
            or None,
        )
        schema = get_response_schema(schema_name)
        lines = [
            "# AFS Structured Workflow Prompt",
            "",
            "Return only JSON matching the response schema below.",
            "Do not wrap the response in markdown fences.",
            "Keep every field grounded in the supplied context. If evidence is missing, keep the answer minimal and note the uncertainty in fields that allow it.",
            "",
            f"Schema resource: {SCHEMA_URI_PREFIX}{schema_name}",
            "",
            "## Response Schema",
            "```json",
            json.dumps(schema, indent=2),
            "```",
            "",
            "## Working Context",
            render_context_pack(payload),
        ]
        return [{"role": "user", "content": {"type": "text", "text": "\n".join(lines)}}]

    if name == "afs.query.search":
        query = arguments.get("query", "")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query argument is required")
        mount_types_raw = arguments.get("mount_types")
        mount_types: list[MountType] | None = None
        if isinstance(mount_types_raw, str) and mount_types_raw.strip():
            mount_types = [MountType(mt.strip()) for mt in mount_types_raw.split(",") if mt.strip()]
        relative_prefix = arguments.get("relative_prefix")
        if relative_prefix is not None and not isinstance(relative_prefix, str):
            raise ValueError("relative_prefix must be a string")
        context_path = _resolve_prompt_context_path(arguments, manager)
        payload = _query_context_index(
            context_path=context_path,
            manager=manager,
            query=query,
            mount_types=mount_types,
            relative_prefix=relative_prefix,
            limit=_coerce_int(arguments.get("limit"), default=25, minimum=1, maximum=500),
        )
        entries = payload["entries"]
        if entries:
            lines = [f"# Search results for: {query}", ""]
            if payload.get("index_rebuild"):
                lines.append("_Index refreshed before search._")
                lines.append("")
            for entry in entries:
                excerpt = entry.get("content_excerpt")
                line = (
                    f"- **{entry['relative_path']}** ({entry['mount_type']}, "
                    f"{entry['size_bytes']} bytes)"
                )
                if isinstance(excerpt, str) and excerpt.strip():
                    line += f": {excerpt}"
                lines.append(line)
        else:
            lines = [f"No results found for: {query}"]
        return [{"role": "user", "content": {"type": "text", "text": "\n".join(lines)}}]

    if name == "afs.scratchpad.review":
        context_path = _resolve_prompt_context_path(arguments, manager)

        scratchpad_dir = resolve_mount_root(
            context_path,
            MountType.SCRATCHPAD,
            config=manager.config,
        )
        lines = [f"# Scratchpad Review: {context_path}", ""]

        state_file = scratchpad_dir / "state.md"
        if state_file.exists():
            lines.append("## State")
            lines.append(state_file.read_text(encoding="utf-8", errors="replace").strip())
            lines.append("")

        deferred_file = scratchpad_dir / "deferred.md"
        if deferred_file.exists():
            lines.append("## Deferred")
            lines.append(deferred_file.read_text(encoding="utf-8", errors="replace").strip())
            lines.append("")

        if scratchpad_dir.exists():
            other_files = [
                f.name
                for f in scratchpad_dir.iterdir()
                if f.is_file() and f.name not in ("state.md", "deferred.md")
            ]
            if other_files:
                lines.append("## Other files")
                for name_str in sorted(other_files):
                    lines.append(f"- {name_str}")

        return [{"role": "user", "content": {"type": "text", "text": "\n".join(lines)}}]

    if registry is not None and name in registry.prompts:
        return registry.get_prompt(name, arguments, manager)

    raise ValueError(f"Unknown prompt: {name}")


def _handle_request(
    request: dict[str, Any],
    manager: AFSManager,
    registry: MCPToolRegistry | None = None,
) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    active_registry = registry or build_mcp_registry(manager)

    if method == "initialize":
        params = request.get("params", {})
        requested_protocol = (
            params.get("protocolVersion")
            if isinstance(params, dict)
            else None
        )
        negotiated_protocol = (
            requested_protocol
            if requested_protocol in SUPPORTED_PROTOCOL_VERSIONS
            else PROTOCOL_VERSION
        )
        return _success_response(
            request_id,
            {
                "protocolVersion": negotiated_protocol,
                "capabilities": {
                    "experimental": {},
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )

    if method == "notifications/initialized":
        return None

    if method == "ping":
        return _success_response(request_id, {})

    if method == "tools/list":
        return _success_response(request_id, {"tools": _tool_specs(active_registry)})

    if method == "tools/call":
        params = request.get("params", {})
        if not isinstance(params, dict):
            return _error_response(request_id, -32602, "Invalid params")

        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str):
            return _error_response(request_id, -32602, "Missing tool name")
        if not isinstance(arguments, dict):
            return _error_response(request_id, -32602, "arguments must be object")

        try:
            payload = active_registry.call(name, arguments, manager)
        except Exception as exc:
            print(f"[afs-mcp] tool error: {name}: {exc}", file=sys.stderr)
            error_msg = _annotate_error(exc)
            return _error_response(request_id, -32000, error_msg)

        return _success_response(request_id, _as_text_result(payload))

    if method == "resources/list":
        return _success_response(
            request_id, {"resources": _list_resources(manager, registry=active_registry)}
        )

    if method == "resources/read":
        params = request.get("params", {})
        if not isinstance(params, dict):
            return _error_response(request_id, -32602, "Invalid params")
        uri = params.get("uri", "")
        if not isinstance(uri, str):
            return _error_response(request_id, -32602, "uri must be a string")
        try:
            content = _read_resource(uri, manager, registry=active_registry)
        except Exception as exc:
            print(f"[afs-mcp] resource error: {uri}: {exc}", file=sys.stderr)
            return _error_response(request_id, -32000, _annotate_error(exc))
        return _success_response(request_id, {"contents": [content]})

    if method == "prompts/list":
        return _success_response(
            request_id, {"prompts": _list_prompts(active_registry)}
        )

    if method == "prompts/get":
        params = request.get("params", {})
        if not isinstance(params, dict):
            return _error_response(request_id, -32602, "Invalid params")
        prompt_name = params.get("name", "")
        prompt_args = params.get("arguments", {})
        if not isinstance(prompt_name, str):
            return _error_response(request_id, -32602, "name must be a string")
        if not isinstance(prompt_args, dict):
            return _error_response(request_id, -32602, "arguments must be an object")
        try:
            messages = _get_prompt(
                prompt_name,
                prompt_args,
                manager,
                registry=active_registry,
            )
        except Exception as exc:
            print(f"[afs-mcp] prompt error: {prompt_name}: {exc}", file=sys.stderr)
            return _error_response(request_id, -32000, _annotate_error(exc))
        return _success_response(request_id, {"messages": messages})

    if request_id is not None:
        return _error_response(request_id, -32601, f"Method not found: {method}")
    return None


def _annotate_error(exc: Exception) -> str:
    """Add recovery hints to common exception types."""
    msg = str(exc)
    if isinstance(exc, FileNotFoundError):
        return f"{msg} (hint: run `afs doctor --fix` to repair missing paths)"
    if isinstance(exc, PermissionError) and "outside allowed roots" in msg:
        return f"{msg} (hint: check AFS_MCP_ALLOWED_ROOTS or mcp_allowed_roots in config)"
    if isinstance(exc, ImportError):
        return f"{msg} (hint: run `afs doctor` to check dependencies)"
    if isinstance(exc, ValueError) and "mount" in msg.lower():
        return f"{msg} (hint: run `afs context repair` to fix mount issues)"
    return msg


def _setup_demo_context(manager: AFSManager) -> None:
    """Configure demo-mode profile with bundled agents and MCP tools."""
    from .agents.supervisor import AgentSupervisor
    from .profiles import resolve_active_profile
    from .schema import AgentConfig

    config = manager.config
    resolved = resolve_active_profile(config)
    demo_agents = resolved.agent_configs or [
        AgentConfig(
            name="demo-watcher",
            role="observer",
            description="Demo context watcher agent",
            auto_start=True,
            module="afs.agents.context_warm",
            triggers=["on_mount", "on_profile_switch"],
        ),
    ]

    # Print demo banner
    print(
        f"[demo] profile={resolved.name} "
        f"extensions={len(resolved.enabled_extensions)} "
        f"mcp_tools={len(resolved.mcp_tools)} "
        f"agent_configs={len(demo_agents)}",
        file=sys.stderr,
    )

    # Auto-start demo agents
    supervisor = AgentSupervisor()
    started = supervisor.auto_start(demo_agents)
    for agent in started:
        print(f"[demo] auto-started agent: {agent.name} pid={agent.pid}", file=sys.stderr)


def _startup_diagnostics(config_path: Path | None = None) -> None:
    """Run lightweight diagnostics on server startup, logging to stderr."""
    try:
        from .diagnostics import run_startup_checks

        results = run_startup_checks(config_path=config_path)
        for result in results:
            if result.status == "error":
                print(f"[afs-mcp] ERROR: {result.name}: {result.message}", file=sys.stderr)
            elif result.status == "warn":
                print(f"[afs-mcp] WARN: {result.name}: {result.message}", file=sys.stderr)
    except Exception as exc:
        print(f"[afs-mcp] startup diagnostics failed: {exc}", file=sys.stderr)


def serve(config_path: Path | None = None, *, demo: bool = False) -> int:
    config = load_config_model(config_path=config_path, merge_user=True)
    manager = AFSManager(config=config)
    registry = build_mcp_registry(manager)
    response_mode = "content-length"

    # Run diagnostics in background thread so the MCP message loop starts
    # immediately — Claude Desktop times out if initialize takes >60s.
    import threading
    threading.Thread(
        target=_startup_diagnostics,
        args=(config_path,),
        daemon=True,
    ).start()

    if demo:
        _setup_demo_context(manager)

    while True:
        try:
            message, detected_mode = _read_message(sys.stdin.buffer)
            if message is None:
                break
            if detected_mode is not None:
                response_mode = detected_mode
            response = _handle_request(message, manager, registry=registry)
            if response is not None:
                _write_message(sys.stdout.buffer, response, mode=response_mode)
        except json.JSONDecodeError as exc:
            print(f"[afs-mcp] malformed message: {exc}", file=sys.stderr)
            continue
        except BrokenPipeError:
            break
        except Exception as exc:
            print(f"[afs-mcp] message loop error: {exc}", file=sys.stderr)
            continue

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AFS MCP server")
    parser.add_argument("--config", help="Config path override.")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode with sample agents.")
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    return serve(config_path=config_path, demo=args.demo)


if __name__ == "__main__":
    raise SystemExit(main())
