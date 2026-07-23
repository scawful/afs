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
import re
import shutil
import stat
import sys
import unicodedata
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn, cast

from .agent_scope import allowed_tools, is_tool_allowed
from .codebase_explorer import (
    build_codebase_summary,
    build_scoped_codebase_summary,
    render_codebase_summary,
)
from .config import load_config_model
from .context_index import (
    DEFAULT_MAX_CONTENT_CHARS,
    DEFAULT_MAX_FILE_SIZE_BYTES,
    ContextSQLiteIndex,
)
from .context_layout import LAYOUT_VERSION, detect_layout_version
from .context_pack import build_context_pack, render_context_pack
from .context_paths import load_context_metadata
from .core import find_existing_root
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
from .models import ContextCategory, MountType
from .operator_digests import KIND_CHOICES, digest_operator_output
from .path_safety import (
    assert_no_linklike_components,
    is_linklike,
    lexical_absolute,
)
from .plugins import load_enabled_extensions
from .profiles import resolve_active_profile
from .project_registry import COMMON_SCOPE_ID, ProjectRegistry
from .repo_policy import evaluate_repo_policy, load_repo_policy
from .response_schemas import (
    SCHEMA_MIME_TYPE,
    SCHEMA_URI_PREFIX,
    get_response_schema,
    list_response_schema_names,
    list_response_schema_specs,
)
from .schema import ContextIndexConfig
from .scopes import ResolvedScope, resolve_scope, visible_scope_prefixes
from .sensitivity import SensitivityRuleSet
from .session_bootstrap import (
    build_session_bootstrap,
    collect_context_diff,
    collect_context_status,
    render_session_bootstrap,
)
from .skills import (
    MAX_SKILL_BODIES_CHARS,
    MAX_SKILL_BODY_CHARS,
    MAX_SKILL_BODY_MATCHES,
    MAX_SKILL_MATCHES,
    MAX_SKILL_METADATA_ITEM_CHARS,
    MAX_SKILL_NAME_CHARS,
    bounded_skill_diagnostics,
    bounded_skill_metadata,
    build_skill_matches_with_diagnostics,
    discover_skills_with_diagnostics,
    escape_skill_diagnostic_text,
    read_skill_body,
    resolve_skill_roots,
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

MCP_TOOL_CATALOG_ENV = "AFS_MCP_TOOL_CATALOG"
MCP_TOOL_NAME_STYLE_ENV = "AFS_MCP_TOOL_NAME_STYLE"
FULL_MCP_TOOL_CATALOG_VALUES = frozenset({"all", "full", "legacy"})
SAFE_MCP_TOOL_NAME_STYLE_VALUES = frozenset({"anthropic", "claude", "safe", "underscore"})
MCP_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MCP_TOOL_NAME_UNSAFE_CHARS = re.compile(r"[^a-zA-Z0-9_-]+")
DEFAULT_MCP_TOOL_CATALOG = frozenset(
    {
        "context.status",
        "context.query",
        "context.read",
        "context.list",
        "context.write",
    }
)
MAX_SKILL_MATCH_PROMPT_CHARS = 8_000
MAX_SKILL_KNOWN_PREVIEW_CHARS = 1_024
MAX_MCP_SKILL_DIAGNOSTICS = 20
MAX_SKILL_DIAGNOSTIC_ERROR_CHARS = 512
SKILL_MATCH_ARGUMENT_NAMES = frozenset({"prompt", "top_k", "include_bodies"})
SKILL_READ_ARGUMENT_NAMES = frozenset({"name"})
INDEX_SYNC_WARNING = (
    "filesystem mutation succeeded, but the context index was not synchronized; "
    "run context.index.rebuild"
)
_WARNING_OPERATION_MAX_CHARS = 256
_WARNING_DETAIL_MAX_CHARS = 1_024
_LOG_ESCAPE_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})


def _sanitize_log_field(value: object, *, max_chars: int) -> str:
    """Escape terminal controls and bound one untrusted log field."""

    try:
        raw = str(value)
    except Exception:  # noqa: BLE001 - logging must survive hostile __str__ methods
        raw = f"<unprintable {type(value).__name__}>"

    pieces: list[str] = []
    used = 0
    for char in raw:
        codepoint = ord(char)
        if unicodedata.category(char) in _LOG_ESCAPE_CATEGORIES:
            piece = (
                f"\\u{codepoint:04x}"
                if codepoint <= 0xFFFF
                else f"\\U{codepoint:08x}"
            )
        else:
            piece = char
        if used + len(piece) > max_chars:
            while pieces and used + 3 > max_chars:
                used -= len(pieces.pop())
            return "".join([*pieces, "..."])
        pieces.append(piece)
        used += len(piece)
    return "".join(pieces)


def _warn_fallback(operation: str, exc: BaseException) -> None:
    """Keep best-effort MCP fallbacks observable without polluting stdout."""

    safe_operation = _sanitize_log_field(
        operation,
        max_chars=_WARNING_OPERATION_MAX_CHARS,
    )
    safe_detail = _sanitize_log_field(
        exc,
        max_chars=_WARNING_DETAIL_MAX_CHARS,
    )
    print(f"[afs-mcp] warning: {safe_operation}: {safe_detail}", file=sys.stderr)


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
    project_raw = arguments.get("project_path", arguments.get("path"))
    project_path = (
        Path(project_raw).expanduser().resolve()
        if isinstance(project_raw, str) and project_raw.strip()
        else Path.cwd().resolve()
    )
    existing_context = find_existing_root(project_path)
    if existing_context is not None:
        return _assert_allowed(existing_context, manager)
    raise FileNotFoundError(
        f"No existing AFS context found for {project_path}. "
        "Use context.init/context.ensure before context.query."
    )


def _resolve_project_path(arguments: dict[str, Any]) -> Path:
    raw = arguments.get("project_path", arguments.get("path"))
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_mcp_scope(
    arguments: dict[str, Any],
    manager: AFSManager,
) -> tuple[Path, ResolvedScope]:
    """Resolve the current/common scope without treating context_path as access."""

    context_path: Path
    raw_context = arguments.get("context_path")
    raw_project = arguments.get("project_path")
    requested_scope = arguments.get("scope_id")
    implicit_scope = not any(
        isinstance(value, str) and value.strip()
        for value in (raw_context, raw_project, requested_scope)
    )
    raw_file_path = next(
        (
            arguments.get(name)
            for name in ("path", "source", "destination")
            if isinstance(arguments.get(name), str) and str(arguments.get(name)).strip()
        ),
        None,
    )
    configured_root = manager.config.general.context_root.expanduser().resolve()
    explicit_project = (
        Path(raw_project).expanduser().resolve()
        if isinstance(raw_project, str) and raw_project.strip()
        else None
    )
    registered_configured_project = (
        explicit_project is not None
        and detect_layout_version(configured_root) == LAYOUT_VERSION
        and ProjectRegistry(configured_root).resolve(explicit_project) is not None
    )
    inferred_project: Path | None = None
    if implicit_scope:
        cwd = Path.cwd().resolve()
        if (
            detect_layout_version(configured_root) == LAYOUT_VERSION
            and ProjectRegistry(configured_root).resolve(cwd) is not None
        ):
            context_path = _assert_allowed(configured_root, manager)
            inferred_project = cwd
        else:
            context_path = _resolve_context_path(arguments, manager)
    elif registered_configured_project:
        context_path = _assert_allowed(configured_root, manager)
    elif (
        not (isinstance(raw_context, str) and raw_context.strip())
        and not (
            isinstance(arguments.get("project_path"), str) and arguments["project_path"].strip()
        )
        and isinstance(raw_file_path, str)
        and raw_file_path.strip()
        and Path(raw_file_path).expanduser().is_absolute()
        and Path(raw_file_path).expanduser().resolve(strict=False).is_relative_to(configured_root)
    ):
        context_path = _assert_allowed(configured_root, manager)
    else:
        context_path = _resolve_context_path(arguments, manager)
    project_path = explicit_project or inferred_project
    if requested_scope is not None and not isinstance(requested_scope, str):
        raise ValueError("scope_id must be a string")
    common = isinstance(requested_scope, str) and requested_scope.strip() == COMMON_SCOPE_ID
    resolved = resolve_scope(
        context_path,
        requester_path=project_path,
        common=common,
    )
    if (
        detect_layout_version(context_path) == LAYOUT_VERSION
        and isinstance(requested_scope, str)
        and requested_scope.strip()
        and requested_scope.strip() != resolved.scope_id
    ):
        if project_path is None:
            raise PermissionError("project_path is required to authorize a project scope")
        registry = ProjectRegistry(context_path)
        registry.assert_scope_authorized(
            requested_scope.strip(),
            requester_path=project_path,
        )
        record = registry.resolve(project_path)
        resolved = ResolvedScope(
            context_root=context_path,
            requester_path=project_path,
            layout_version=LAYOUT_VERSION,
            scope_id=requested_scope.strip(),
            project_id=(
                requested_scope.strip().removeprefix("project:")
                if requested_scope.strip().startswith("project:")
                else ""
            ),
            project_name=record.name if record is not None else project_path.name,
        )
    return context_path, resolved


_CATEGORY_NAMES = {category.value: category for category in ContextCategory}


def _mcp_scope_properties(*, include_all_projects: bool = False) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "context_path": {
            "type": "string",
            "description": "Central AFS context root; does not grant project access by itself.",
        },
        "project_path": {
            "type": "string",
            "description": "Current project path used to authorize its registered scope.",
        },
        "scope_id": {
            "type": "string",
            "description": "Optional authorized scope: common or project:<id>.",
        },
    }
    if include_all_projects:
        properties["all_projects"] = {
            "type": "boolean",
            "default": False,
            "description": "Explicitly search or operate across every registered project.",
        }
    return properties


def _resolve_context_file_path(
    arguments: dict[str, Any],
    manager: AFSManager,
    value: Any,
    *,
    operation: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a non-empty string")
    raw_path = Path(value.strip()).expanduser()
    configured_root = manager.config.general.context_root.expanduser().resolve()
    has_scope_anchor = any(
        isinstance(arguments.get(name), str) and str(arguments.get(name)).strip()
        for name in ("context_path", "project_path", "scope_id")
    )
    if detect_layout_version(configured_root) != LAYOUT_VERSION and not has_scope_anchor:
        return _assert_allowed(raw_path, manager)

    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    if detect_layout_version(context_path) != LAYOUT_VERSION:
        candidate = raw_path if raw_path.is_absolute() else context_path / raw_path
        return _assert_allowed(candidate, manager)

    registry = ProjectRegistry(context_path)
    if raw_path.is_absolute():
        candidate = lexical_absolute(raw_path)
        allowed_roots: list[Path] = []
        if scoped.scope_id == COMMON_SCOPE_ID:
            for category in ContextCategory:
                _scope, root = registry.resolve_scope_root(
                    category,
                    requester_path=scoped.requester_path or context_path,
                    scope_id=COMMON_SCOPE_ID,
                )
                allowed_roots.append(root)
        elif scoped.requester_path is not None:
            record = registry.resolve(scoped.requester_path)
            if record is not None:
                allowed_roots.extend(lexical_absolute(root) for root in record.roots())
            for category in ContextCategory:
                _scope, root = registry.resolve_scope_root(
                    category,
                    requester_path=scoped.requester_path,
                    scope_id=scoped.scope_id,
                )
                allowed_roots.append(root)
        matching_root = next(
            (
                lexical_absolute(root)
                for root in allowed_roots
                if candidate == lexical_absolute(root)
                or candidate.is_relative_to(lexical_absolute(root))
            ),
            None,
        )
        if matching_root is None:
            raise PermissionError(
                f"absolute path is outside the authorized {scoped.scope_id} scope "
                f"for {operation}: {candidate}"
            )
        return assert_no_linklike_components(
            candidate,
            boundary=matching_root,
        )

    parts = list(raw_path.parts)
    if parts and parts[0] == ".context":
        parts = parts[1:]
    matched_category = _CATEGORY_NAMES.get(parts[0]) if parts else None
    if matched_category is not None:
        category = matched_category
        parts = parts[1:]
    else:
        category = ContextCategory.SCRATCHPAD
    requester = scoped.requester_path or context_path
    if not parts:
        _scope, root = registry.resolve_scope_root(
            category,
            requester_path=requester,
            scope_id=scoped.scope_id,
        )
        return root
    return registry.resolve_scoped_path(
        category,
        Path(*parts),
        requester_path=requester,
        scope_id=scoped.scope_id,
    )


def _scoped_relative_prefixes(
    scoped: ResolvedScope,
    relative_prefix: str | None,
) -> list[str]:
    """Build contained index prefixes for the current project and common data."""

    suffix = ""
    if relative_prefix:
        raw = Path(relative_prefix.strip())
        if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
            raise ValueError("relative_prefix must be a contained relative path")
        suffix = raw.as_posix().strip("/")

    scope_roots = ["common"]
    if scoped.project_id:
        scope_roots.insert(0, f"projects/{scoped.project_id}")
    return [f"{root}/{suffix}" if suffix else f"{root}/" for root in scope_roots]


def _merge_scoped_query_payloads(
    payloads: list[dict[str, Any]],
    *,
    scoped: ResolvedScope,
    limit: int,
) -> dict[str, Any]:
    """Merge independently scoped query legs without admitting another scope."""

    if not payloads:
        raise ValueError("at least one query payload is required")
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for payload in payloads:
        for entry in payload.get("entries", []):
            key = (str(entry.get("mount_type", "")), str(entry.get("relative_path", "")))
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)

    def _rank(entry: dict[str, Any]) -> tuple[int, float, str, str]:
        score = entry.get("relevance_score")
        return (
            0 if isinstance(score, (int, float)) else 1,
            float(score) if isinstance(score, (int, float)) else 0.0,
            str(entry.get("mount_type", "")),
            str(entry.get("relative_path", "")),
        )

    entries.sort(key=_rank)
    merged = dict(payloads[0])
    merged["scope_id"] = scoped.scope_id
    merged["project_id"] = scoped.project_id
    merged["relative_prefix"] = ""
    merged["scope_prefixes"] = [str(payload.get("relative_prefix", "")) for payload in payloads]
    merged["entries"] = entries[:limit]
    merged["count"] = len(merged["entries"])
    merged["limit"] = limit
    rebuilds = [payload["index_rebuild"] for payload in payloads if "index_rebuild" in payload]
    if rebuilds:
        merged["index_rebuild"] = rebuilds[0]
    return merged


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
    if isinstance(value, bool):
        return default
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            return default
    if not isinstance(value, int):
        return default
    if value < minimum:
        return default
    if maximum is not None and value > maximum:
        return maximum
    return value


def _coerce_bool(value: Any) -> bool:
    """Require a literal JSON boolean for an explicit MCP consent boundary."""

    return value is True


def _context_index_settings(manager: AFSManager) -> ContextIndexConfig:
    return manager.config.context_index


def _resolve_prompt_context_path(arguments: dict[str, Any], manager: AFSManager) -> Path:
    raw_path = arguments.get("context_path")
    if isinstance(raw_path, str) and raw_path.strip():
        return _resolve_explicit_allowed_context_path(raw_path, manager)
    return manager.config.general.context_root


def _resolve_prompt_project_path(arguments: dict[str, Any]) -> Path | None:
    raw_path = arguments.get("project_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    return Path(raw_path).expanduser().resolve()


def _discover_allowed_contexts(manager: AFSManager) -> list[Any]:
    contexts: list[Any] = []
    try:
        contexts = discover_contexts(config=manager.config)
    except Exception as exc:  # noqa: BLE001 - discovery is an optional integration boundary
        _warn_fallback("context discovery failed", exc)
        contexts = []

    default_context = manager.config.general.context_root
    if default_context.exists():
        try:
            contexts.append(manager.list_context(context_path=default_context))
        except Exception as exc:  # noqa: BLE001 - retain partial discovery results
            _warn_fallback(f"default context inspection failed for {default_context}", exc)

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


def _sensitivity_export_rules(manager: AFSManager) -> SensitivityRuleSet:
    sensitivity = manager.config.sensitivity
    return SensitivityRuleSet.from_patterns([*sensitivity.never_index, *sensitivity.never_export])


def _sensitivity_relative_paths(path: Path, manager: AFSManager) -> list[str]:
    """Return context-relative path variants used by sensitivity globs."""
    resolved = path.expanduser().resolve()
    values: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        cleaned = value.replace("\\", "/").strip().lstrip("./")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            values.append(cleaned)

    for context_path in _context_candidates_for_path(resolved, manager):
        for mount_type in MountType:
            try:
                mount_root = manager.resolve_mount_root(context_path, mount_type)
            except (OSError, RuntimeError, ValueError) as exc:
                raise PermissionError(
                    "Unable to resolve sensitivity paths safely; access was denied"
                ) from exc
            try:
                rel = resolved.relative_to(mount_root.expanduser().resolve()).as_posix()
            except ValueError:
                continue
            _add(rel)
            _add(f"{mount_type.value}/{rel}")
    return values


def _sensitivity_block_match(path: Path, manager: AFSManager) -> tuple[str, str] | None:
    rules = _sensitivity_export_rules(manager)
    if not rules.enabled:
        return None
    resolved = path.expanduser().resolve()
    relative_values = _sensitivity_relative_paths(resolved, manager)
    if not relative_values:
        relative_values = [resolved.name]
    match = rules.match(resolved, relative_paths=relative_values)
    return (match.relative_path, match.pattern) if match is not None else None


def _assert_sensitivity_allowed(path: Path, manager: AFSManager, *, operation: str) -> None:
    match = _sensitivity_block_match(path, manager)
    if match is None:
        return
    relative_path, pattern = match
    raise PermissionError(
        f"Blocked by sensitivity rule during {operation}: "
        f"path '{relative_path}' matches pattern '{pattern}'"
    )


def _entry_blocked_by_sensitivity(entry: dict[str, Any], manager: AFSManager) -> bool:
    rules = _sensitivity_export_rules(manager)
    if not rules.enabled:
        return False
    absolute_path = Path(str(entry.get("absolute_path", "")))
    relative_path = f"{entry.get('mount_type', '')}/{entry.get('relative_path', '')}".replace(
        "\\", "/"
    ).strip("/")
    return rules.blocked(absolute_path, relative_paths=(relative_path,))


def _sync_context_index_for_path(
    path: Path,
    manager: AFSManager,
    *,
    scoped: ResolvedScope | None = None,
) -> bool:
    settings = _context_index_settings(manager)
    if not settings.enabled:
        return False

    for context_path in _context_candidates_for_path(path, manager):
        try:
            index = ContextSQLiteIndex(manager, context_path)
            if not index.sync_absolute_path(
                path,
                scoped=(
                    scoped
                    if scoped is not None and scoped.context_root == context_path
                    else None
                ),
                include_content=settings.include_content,
                max_file_size_bytes=settings.max_file_size_bytes,
                max_content_chars=settings.max_content_chars,
            ):
                continue
            return True
        except Exception as exc:  # noqa: BLE001 - best-effort sync after committed mutation
            _warn_fallback(f"context index sync failed for {path}", exc)
            continue
    return False


def _add_index_sync_warning(payload: dict[str, Any], *, manager: AFSManager) -> None:
    detailed_sync_keys = ("source_index_updated", "destination_index_updated")
    detailed_sync_values = [
        bool(payload[key]) for key in detailed_sync_keys if key in payload
    ]
    sync_complete = (
        all(detailed_sync_values)
        if detailed_sync_values
        else bool(payload.get("index_updated"))
    )
    if not sync_complete and _context_index_settings(manager).enabled:
        payload["warnings"] = [INDEX_SYNC_WARNING]


def _resolve_optional_v2_index_scope(
    arguments: dict[str, Any],
    manager: AFSManager,
) -> ResolvedScope | None:
    """Resolve v2 sync scope without changing legacy absolute-path behavior."""

    configured_root = manager.config.general.context_root.expanduser().resolve()
    if detect_layout_version(configured_root) != LAYOUT_VERSION:
        return None
    try:
        context_path, scoped = _resolve_mcp_scope(arguments, manager)
    except FileNotFoundError:
        return None
    return scoped if detect_layout_version(context_path) == LAYOUT_VERSION else None


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
    scoped: ResolvedScope | None = None,
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
    has_entries = (
        index.has_entries_scoped(scoped, mount_types=mount_types)
        if scoped is not None and scoped.layout_version == LAYOUT_VERSION
        else index.has_entries(mount_types=mount_types)
    )
    should_auto_refresh = False
    if settings.enabled and effective_auto_index:
        needs_refresh = False
        if has_entries and effective_auto_refresh:
            needs_refresh = (
                index.needs_refresh_scoped(scoped, mount_types=mount_types)
                if scoped is not None and scoped.layout_version == LAYOUT_VERSION
                else index.needs_refresh(mount_types=mount_types)
            )
        should_auto_refresh = not has_entries or (
            effective_auto_refresh and needs_refresh
        )
    if refresh or should_auto_refresh:
        summary = (
            index.rebuild_scoped(
                scoped,
                mount_types=mount_types,
                include_content=settings.include_content,
                max_file_size_bytes=effective_max_file_size_bytes,
                max_content_chars=effective_max_content_chars,
            )
            if scoped is not None and scoped.layout_version == LAYOUT_VERSION
            else index.rebuild(
                mount_types=mount_types,
                include_content=settings.include_content,
                max_file_size_bytes=effective_max_file_size_bytes,
                max_content_chars=effective_max_content_chars,
            )
        )
        rebuild_summary = summary.to_dict()

    query_limit = 500 if _sensitivity_export_rules(manager).enabled else limit_value
    entries = index.query(
        query=query,
        mount_types=mount_types,
        relative_prefix=relative_prefix,
        limit=query_limit,
        include_content=include_content,
    )
    entries = [entry for entry in entries if not _entry_blocked_by_sensitivity(entry, manager)][
        :limit_value
    ]
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


def _work_store(
    arguments: dict[str, Any],
    manager: AFSManager,
) -> tuple[Any, Path]:
    from .work_assistant import WorkAssistantStore

    context_path = _resolve_context_path(arguments, manager)
    return WorkAssistantStore(context_path, config=manager.config), context_path


def _tool_work_communication_list(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    store, context_path = _work_store(arguments, manager)
    limit = _coerce_int(arguments.get("limit"), default=20, minimum=1, maximum=100)
    person_id = arguments.get("person_id")
    purpose = arguments.get("purpose")
    samples = store.list_communication_samples(
        person_id=person_id if isinstance(person_id, str) and person_id.strip() else None,
        purpose=purpose if isinstance(purpose, str) and purpose.strip() else None,
        limit=limit,
    )
    return {"context_path": str(context_path), "samples": samples, "count": len(samples)}


def _tool_work_communication_add(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    store, context_path = _work_store(arguments, manager)
    text = arguments.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("work.communication.add requires non-empty text")
    style_notes = arguments.get("style_notes")
    provenance = arguments.get("provenance")
    sample_id = store.record_communication_sample(
        text=text,
        person_id=str(arguments.get("person_id") or ""),
        source_system=str(arguments.get("source_system") or ""),
        source_id=str(arguments.get("source_id") or ""),
        channel=str(arguments.get("channel") or ""),
        purpose=str(arguments.get("purpose") or "work_communication"),
        style_notes=style_notes if isinstance(style_notes, list) else [],
        provenance=provenance
        if isinstance(provenance, list)
        else ([provenance] if provenance else []),
        confidence=float(arguments.get("confidence") or 0.5),
        dedupe_key=str(arguments.get("dedupe_key") or "") or None,
    )
    return {"context_path": str(context_path), "sample_id": sample_id}


def _tool_work_communication_guide(
    arguments: dict[str, Any], manager: AFSManager
) -> dict[str, Any]:
    store, context_path = _work_store(arguments, manager)
    limit = _coerce_int(arguments.get("limit"), default=20, minimum=1, maximum=100)
    person_id = arguments.get("person_id")
    purpose = arguments.get("purpose")
    summary = dict(
        store.communication_style_summary(
            person_id=person_id if isinstance(person_id, str) and person_id.strip() else None,
            purpose=purpose if isinstance(purpose, str) and purpose.strip() else None,
            limit=limit,
        )
    )
    summary["context_path"] = str(context_path)
    return summary


def _load_personal_context_for_work(arguments: dict[str, Any]) -> Any | None:
    personal_mode = arguments.get("personal_mode")
    if not isinstance(personal_mode, str) or not personal_mode.strip():
        return None
    from .personal_context import load_personal_context

    raw_root = arguments.get("personal_context_root")
    return load_personal_context(
        personal_mode.strip(),
        context_root=Path(raw_root).expanduser()
        if isinstance(raw_root, str) and raw_root.strip()
        else None,
    )


def _tool_work_communication_preflight(
    arguments: dict[str, Any], manager: AFSManager
) -> dict[str, Any]:
    store, context_path = _work_store(arguments, manager)
    limit = _coerce_int(arguments.get("limit"), default=20, minimum=1, maximum=100)
    approval_limit = _coerce_int(
        arguments.get("approval_limit"), default=10, minimum=1, maximum=100
    )
    person_id = arguments.get("person_id")
    purpose = arguments.get("purpose")
    return dict(
        store.communication_preflight(
            person_id=person_id if isinstance(person_id, str) and person_id.strip() else None,
            purpose=purpose if isinstance(purpose, str) and purpose.strip() else None,
            limit=limit,
            approval_limit=approval_limit,
            personal_context=_load_personal_context_for_work(arguments),
            context_path=context_path,
        )
    )


def _tool_work_approvals_list(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    store, context_path = _work_store(arguments, manager)
    limit = _coerce_int(arguments.get("limit"), default=50, minimum=1, maximum=100)
    status_arg = arguments.get("status")
    status: str | None = (
        status_arg if isinstance(status_arg, str) and status_arg.strip() else "pending"
    )
    if bool(arguments.get("all", False)):
        status = None
    approvals = store.list_approvals(status=status, limit=limit)
    return {"context_path": str(context_path), "approvals": approvals, "count": len(approvals)}


def _tool_work_approvals_show(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    store, context_path = _work_store(arguments, manager)
    approval_id = arguments.get("approval_id")
    if not isinstance(approval_id, str) or not approval_id.strip():
        raise ValueError("work.approvals.show requires approval_id")
    approval = store.get_approval(approval_id)
    if approval is None:
        raise FileNotFoundError(f"No approval found: {approval_id}")
    return {"context_path": str(context_path), "approval": approval}


def _tool_work_approvals_request(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    store, context_path = _work_store(arguments, manager)
    required = ("target_system", "target_id", "action", "summary")
    missing = [name for name in required if not str(arguments.get(name) or "").strip()]
    if missing:
        raise ValueError(f"work.approvals.request missing required fields: {', '.join(missing)}")
    preview = arguments.get("preview")
    approval_id = store.create_approval(
        target_system=str(arguments.get("target_system") or ""),
        target_id=str(arguments.get("target_id") or ""),
        action=str(arguments.get("action") or ""),
        summary=str(arguments.get("summary") or ""),
        preview=preview if preview is not None else {},
        affected_people=arguments.get("affected_people")
        if isinstance(arguments.get("affected_people"), list)
        else [],
        risk_level=str(arguments.get("risk_level") or "medium"),
        permission_required=str(arguments.get("permission_required") or "human approval"),
        requested_by=str(arguments.get("requested_by") or "agent"),
        expires_at=str(arguments.get("expires_at") or "") or None,
        dedupe_key=str(arguments.get("dedupe_key") or "") or None,
    )
    return {"context_path": str(context_path), "approval_id": approval_id, "status": "pending"}


def _tool_fs_read(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    path_value = arguments.get("path")
    path = _resolve_context_file_path(
        arguments,
        manager,
        path_value,
        operation="read",
    )
    _assert_sensitivity_allowed(path, manager, operation="read")
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

    scoped = _resolve_optional_v2_index_scope(arguments, manager)
    path = _resolve_context_file_path(
        arguments,
        manager,
        path_value,
        operation="write",
    )
    _assert_sensitivity_allowed(path, manager, operation="write")
    if not path.parent.exists():
        if not mkdirs:
            raise FileNotFoundError(f"Parent directory missing: {path.parent}")
        path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(content)
    index_updated = _sync_context_index_for_path(path, manager, scoped=scoped)
    payload: dict[str, Any] = {
        "path": str(path),
        "bytes": len(content.encode("utf-8")),
        "append": append,
        "index_updated": index_updated,
    }
    _add_index_sync_warning(payload, manager=manager)
    return payload


def _tool_fs_delete(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    path_value = arguments.get("path")
    recursive = bool(arguments.get("recursive", False))
    scoped = _resolve_optional_v2_index_scope(arguments, manager)
    path = _resolve_context_file_path(
        arguments,
        manager,
        path_value,
        operation="delete",
    )
    _assert_sensitivity_allowed(path, manager, operation="delete")
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

    index_updated = _sync_context_index_for_path(path, manager, scoped=scoped)
    payload: dict[str, Any] = {
        "path": str(path),
        "deleted": True,
        "recursive": recursive,
        "index_updated": index_updated,
    }
    _add_index_sync_warning(payload, manager=manager)
    return payload


def _tool_fs_move(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    source_value = arguments.get("source")
    destination_value = arguments.get("destination")
    mkdirs = bool(arguments.get("mkdirs", False))

    if not isinstance(source_value, str) or not isinstance(destination_value, str):
        raise ValueError("source and destination must be strings")

    scoped = _resolve_optional_v2_index_scope(arguments, manager)
    source = _resolve_context_file_path(
        arguments,
        manager,
        source_value,
        operation="move",
    )
    destination = _resolve_context_file_path(
        arguments,
        manager,
        destination_value,
        operation="move",
    )
    _assert_sensitivity_allowed(source, manager, operation="move")
    _assert_sensitivity_allowed(destination, manager, operation="move")

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
    source_synced = _sync_context_index_for_path(source, manager, scoped=scoped)
    destination_synced = _sync_context_index_for_path(
        destination,
        manager,
        scoped=scoped,
    )

    payload: dict[str, Any] = {
        "source": str(source),
        "destination": str(destination),
        "index_updated": bool(source_synced or destination_synced),
        "source_index_updated": source_synced,
        "destination_index_updated": destination_synced,
    }
    _add_index_sync_warning(payload, manager=manager)
    return payload


def _tool_fs_list(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    path_value = arguments.get("path")
    max_depth = arguments.get("max_depth", 1)
    if not isinstance(max_depth, int):
        max_depth = 1

    configured_context = manager.config.general.context_root.expanduser().resolve()
    listing_context = configured_context
    if any(
        isinstance(arguments.get(name), str) and str(arguments.get(name)).strip()
        for name in ("context_path", "project_path", "scope_id")
    ):
        listing_context, _listing_scope = _resolve_mcp_scope(arguments, manager)
    listing_is_v2 = detect_layout_version(listing_context) == LAYOUT_VERSION

    root = _resolve_context_file_path(
        arguments,
        manager,
        path_value,
        operation="list",
    )
    _assert_sensitivity_allowed(root, manager, operation="list")
    if not root.exists():
        if listing_is_v2:
            return {"path": str(root), "entries": []}
        raise FileNotFoundError(f"Path not found: {root}")

    entries: list[dict[str, Any]] = []
    if root.is_file():
        entries.append({"path": str(root), "is_dir": False})
    elif listing_is_v2:
        pending: list[tuple[Path, int]] = [(root, 0)]
        while pending:
            base, depth = pending.pop()
            try:
                with os.scandir(base) as scan:
                    children = sorted(scan, key=lambda child: child.name)
            except OSError:
                continue
            child_directories: list[Path] = []
            for child in children:
                try:
                    child_stat = child.stat(follow_symlinks=False)
                except OSError:
                    continue
                if is_linklike(child_stat):
                    continue
                candidate = Path(child.path)
                candidate_depth = depth + 1
                if max_depth >= 0 and candidate_depth > max_depth:
                    continue
                if _sensitivity_block_match(candidate, manager) is not None:
                    continue
                child_is_dir = stat.S_ISDIR(child_stat.st_mode)
                entries.append(
                    {"path": str(candidate), "is_dir": child_is_dir}
                )
                if child_is_dir and (max_depth < 0 or candidate_depth < max_depth):
                    child_directories.append(candidate)
            pending.extend(
                (path, depth + 1) for path in reversed(child_directories)
            )
    else:
        for candidate in root.rglob("*"):
            try:
                depth = len(candidate.relative_to(root).parts)
            except ValueError:
                continue
            if max_depth >= 0 and depth > max_depth:
                continue
            if _sensitivity_block_match(candidate, manager) is not None:
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

    contexts = discover_contexts(
        search_paths=search_paths, max_depth=max_depth, config=manager.config
    )
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
    context_path, scoped = _resolve_mcp_scope(arguments, manager)
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
    allow_all_projects = arguments.get("all_projects") is True
    summary = (
        index.rebuild(
            mount_types=mount_types,
            include_content=include_content,
            max_file_size_bytes=max_file_size_bytes,
            max_content_chars=max_content_chars,
        )
        if scoped.layout_version != LAYOUT_VERSION or allow_all_projects
        else index.rebuild_scoped(
            scoped,
            mount_types=mount_types,
            include_content=include_content,
            max_file_size_bytes=max_file_size_bytes,
            max_content_chars=max_content_chars,
        )
    )
    payload = summary.to_dict()
    payload["mount_types"] = list(summary.by_mount_type)
    payload["scope_id"] = "all-projects" if allow_all_projects else scoped.scope_id
    payload["project_id"] = scoped.project_id
    return payload


def _tool_context_query(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    mount_types = _parse_mount_types(arguments.get("mount_types"))
    allow_all_projects = arguments.get("all_projects") is True
    if scoped.layout_version == LAYOUT_VERSION and not allow_all_projects:
        mount_types = [
            mount_type
            for mount_type in (mount_types or list(MountType))
            if ContextCategory.from_mount_type(mount_type) is not None
        ]
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
    limit = _coerce_int(arguments.get("limit"), default=25, minimum=1, maximum=500)
    common_arguments = {
        "context_path": context_path,
        "manager": manager,
        "query": query_value,
        "mount_types": mount_types,
        "limit": limit,
        "include_content": include_content,
        "auto_index": auto_index,
        "auto_refresh": auto_refresh,
        "max_file_size_bytes": arguments.get("max_file_size_bytes"),
        "max_content_chars": arguments.get("max_content_chars"),
    }
    if detect_layout_version(context_path) != LAYOUT_VERSION or bool(
        allow_all_projects
    ):
        payload = _query_context_index(
            **common_arguments,
            relative_prefix=relative_prefix,
            refresh=refresh,
        )
        payload["scope_id"] = "all-projects" if allow_all_projects else scoped.scope_id
        payload["project_id"] = scoped.project_id
        return payload

    payloads = [
        _query_context_index(
            **common_arguments,
            relative_prefix=prefix,
            refresh=refresh if index == 0 else False,
            scoped=scoped,
        )
        for index, prefix in enumerate(_scoped_relative_prefixes(scoped, relative_prefix))
    ]
    return _merge_scoped_query_payloads(payloads, scoped=scoped, limit=limit)


def _tool_context_search(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    """Search the immutable v2 hybrid index after applying scope authorization."""
    from .context_layout import resolve_system_path
    from .hybrid_search import HybridSearchEngine, hybrid_hit_blocked

    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query is required")
    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    if detect_layout_version(context_path) != LAYOUT_VERSION:
        raise ValueError("context.search requires a v2 context; use context.query for v1")
    engine = HybridSearchEngine(resolve_system_path(context_path, "search"))
    mode = str(arguments.get("mode", "text") or "text").strip().lower()
    allow_semantic = _coerce_bool(arguments.get("semantic", False))
    allow_all_projects = arguments.get("all_projects") is True
    if allow_semantic:
        mode = "hybrid"
    response = engine.search(
        query,
        scope_ids=[scoped.scope_id] if scoped.scope_id != COMMON_SCOPE_ID else [],
        include_common=True,
        all_projects=allow_all_projects,
        mode=mode,
        top_k=_coerce_int(arguments.get("limit"), default=10, minimum=1, maximum=100),
        recreate_query_embedder=allow_semantic,
        required_scope_ids=(
            [record.scope_id for record in ProjectRegistry(context_path).all_records()]
            if allow_all_projects
            else [scoped.scope_id]
        ),
    )
    response.results = [
        hit
        for hit in response.results
        if not hybrid_hit_blocked(
            hit,
            context_root=context_path,
            patterns=[
                *manager.config.sensitivity.never_index,
                *manager.config.sensitivity.never_export,
            ],
        )
    ]
    payload = response.to_dict()
    payload["context_path"] = str(context_path)
    payload["scope_id"] = "all-projects" if allow_all_projects else scoped.scope_id
    payload["project_id"] = scoped.project_id
    return payload


def _tool_context_diff(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    """Show changes between filesystem and index."""
    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    mount_types = _parse_mount_types(arguments.get("mount_types"))
    return collect_context_diff(
        manager,
        context_path,
        mount_types=mount_types,
        scoped=scoped,
    )


def _tool_context_status(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    """Return a summary of the context: mounts, index health, profile."""
    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    payload = collect_context_status(manager, context_path, scoped=scoped)
    payload["scope_id"] = scoped.scope_id
    payload["project_id"] = scoped.project_id
    payload["project_name"] = scoped.project_name
    return payload


def _tool_session_pack(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    """Build a token-budgeted context pack for a target model."""
    context_path, scoped = _resolve_mcp_scope(arguments, manager)
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
        project_path=scoped.requester_path,
        scope_id=scoped.scope_id,
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
        semantic=_coerce_bool(arguments.get("semantic", False)),
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
        kwargs = {role: value for role, value in named_values.items() if value is not None}
        return handler(**kwargs)

    fallback = [role for role in fallback_roles if named_values.get(role) is not None]
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
    from .messages import MessageBus

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

    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    bus = MessageBus(
        context_path,
        scope_id=scoped.scope_id,
        config=manager.config,
        all_projects=arguments.get("all_projects") is True,
        include_legacy=scoped.layout_version != LAYOUT_VERSION,
    )
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
    from .messages import MessageBus

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

    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    allow_all_projects = arguments.get("all_projects") is True
    bus = MessageBus(
        context_path,
        scope_id=scoped.scope_id,
        config=manager.config,
        all_projects=allow_all_projects,
        include_legacy=arguments.get("include_legacy") is True
        or scoped.layout_version != LAYOUT_VERSION,
    )
    messages = bus.read(agent_name=agent_name, msg_type=msg_type, topic=topic, limit=limit)
    return {
        "scope_id": "all-projects" if allow_all_projects else scoped.scope_id,
        "messages": [m.to_dict() for m in messages],
    }


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


def _tool_agent_manifest_show(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_manifest import (
        default_manifest_path,
        export_for_harness,
        load_manifest,
        summarize_manifest,
        validate_manifest,
    )

    raw_path = arguments.get("file")
    path = Path(str(raw_path)).expanduser() if raw_path else default_manifest_path()
    data = load_manifest(path)
    harness = str(arguments.get("harness", "") or "").strip()
    payload = export_for_harness(data, harness) if harness else summarize_manifest(data)
    if bool(arguments.get("validate", False)):
        payload["issues"] = [
            issue.to_dict()
            for issue in validate_manifest(
                data,
                check_paths=bool(arguments.get("check_paths", False)),
            )
        ]
    payload["path"] = str(path)
    return payload


def _tool_agent_run_start(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_runs import AgentRunStore

    task = str(arguments.get("task", "") or "").strip()
    if not task:
        raise ValueError("task is required")
    context_path = _resolve_context_path(arguments, manager)
    run = AgentRunStore(context_path).start(
        task,
        harness=str(arguments.get("harness", "") or "").strip(),
        workspace=str(arguments.get("workspace", "") or "").strip(),
        prompt=str(arguments.get("prompt", "") or ""),
    )
    return run.to_dict()


def _tool_agent_run_list(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_runs import AgentRunStore

    context_path = _resolve_context_path(arguments, manager)
    status = arguments.get("status")
    if isinstance(status, str):
        status = status.strip() or None
    limit = _coerce_int(arguments.get("limit"), default=20, minimum=1, maximum=100)
    runs = AgentRunStore(context_path).list(status=status, limit=limit)
    return {"runs": [run.to_dict() for run in runs]}


def _tool_agent_run_show(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_runs import AgentRunStore

    run_id = str(arguments.get("run_id", "") or "").strip()
    if not run_id:
        raise ValueError("run_id is required")
    context_path = _resolve_context_path(arguments, manager)
    run = AgentRunStore(context_path).get(run_id)
    if run is None:
        raise FileNotFoundError(f"Agent run not found: {run_id}")
    return run.to_dict()


def _tool_agent_run_event(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_runs import AgentRunStore

    run_id = str(arguments.get("run_id", "") or "").strip()
    event_type = str(arguments.get("event_type", "") or "").strip()
    if not run_id or not event_type:
        raise ValueError("run_id and event_type are required")
    context_path = _resolve_context_path(arguments, manager)
    run = AgentRunStore(context_path).record_event(
        run_id,
        event_type,
        summary=str(arguments.get("summary", "") or ""),
        data=arguments.get("data") if isinstance(arguments.get("data"), dict) else {},
    )
    return run.to_dict()


def _tool_agent_run_finish(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_runs import AgentRunStore

    run_id = str(arguments.get("run_id", "") or "").strip()
    if not run_id:
        raise ValueError("run_id is required")
    context_path = _resolve_context_path(arguments, manager)
    files_changed = arguments.get("files_changed")
    commands = arguments.get("commands")
    verification = arguments.get("verification")
    run = AgentRunStore(context_path).finish(
        run_id,
        status=str(arguments.get("status", "") or "done"),
        summary=str(arguments.get("summary", "") or ""),
        files_changed=[str(item) for item in files_changed]
        if isinstance(files_changed, list)
        else [],
        commands=[str(item) for item in commands] if isinstance(commands, list) else [],
        verification=[item for item in verification if isinstance(item, dict)]
        if isinstance(verification, list)
        else [],
        handoff_path=str(arguments.get("handoff_path", "") or ""),
    )
    return run.to_dict()


def _tool_agent_job_create(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_jobs import AgentJobQueue

    title = str(arguments.get("title", "") or "").strip()
    if not title:
        raise ValueError("title is required")
    prompt = str(arguments.get("prompt", "") or title)
    context_path = _resolve_context_path(arguments, manager)
    priority = _coerce_int(arguments.get("priority"), default=5, minimum=1, maximum=10)
    job = AgentJobQueue(context_path).create(
        title,
        prompt,
        priority=priority,
        created_by=str(arguments.get("created_by", "") or "").strip(),
        scope=str(arguments.get("scope", "") or "").strip(),
        expected_output=str(arguments.get("expected_output", "") or "").strip(),
        allow_destructive=bool(arguments.get("allow_destructive", False)),
    )
    return job.to_dict()


def _tool_agent_job_status(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_job_status import build_agent_job_status

    context_path = _resolve_context_path(arguments, manager)
    stale_after_raw = arguments.get("stale_after_seconds", 3600.0)
    stale_after = float(stale_after_raw) if isinstance(stale_after_raw, (int, float)) else 3600.0
    recent_runs = _coerce_int(arguments.get("recent_runs"), default=5, minimum=0, maximum=50)
    label = str(arguments.get("label", "") or "").strip() or "com.afs.agent-jobs"
    return build_agent_job_status(
        context_path,
        label=label,
        stale_after_seconds=stale_after,
        recent_runs_limit=recent_runs,
    )


def _tool_agent_job_inbox(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_job_inbox import build_agent_job_inbox

    context_path = _resolve_context_path(arguments, manager)
    stale_after_raw = arguments.get("stale_after_seconds", 3600.0)
    stale_after = float(stale_after_raw) if isinstance(stale_after_raw, (int, float)) else 3600.0
    limit = _coerce_int(arguments.get("limit"), default=20, minimum=1, maximum=100)
    return build_agent_job_inbox(
        context_path,
        stale_after_seconds=stale_after,
        limit=limit,
    )


def _tool_agent_job_seed(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_job_seeds import seed_agent_jobs

    context_path = _resolve_context_path(arguments, manager)
    profile = str(arguments.get("profile", "") or "repo-maintenance").strip()
    cadence = str(arguments.get("cadence", "") or "daily").strip()
    created_by = str(arguments.get("created_by", "") or "agent.job.seed").strip()
    return seed_agent_jobs(
        context_path,
        profile=profile,
        cadence=cadence,
        created_by=created_by,
        dry_run=bool(arguments.get("dry_run", False)),
        force=bool(arguments.get("force", False)),
    )


def _tool_agent_job_list(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_jobs import AgentJobQueue

    status = arguments.get("status")
    if isinstance(status, str):
        status = status.strip() or None
    context_path = _resolve_context_path(arguments, manager)
    jobs = AgentJobQueue(context_path).list(status=status)
    return {"jobs": [job.to_dict() for job in jobs]}


def _tool_agent_job_show(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_jobs import AgentJobQueue

    job_id = str(arguments.get("job_id", "") or "").strip()
    if not job_id:
        raise ValueError("job_id is required")
    context_path = _resolve_context_path(arguments, manager)
    job = AgentJobQueue(context_path).get(job_id)
    if job is None:
        raise FileNotFoundError(f"Agent job not found: {job_id}")
    return job.to_dict()


def _tool_agent_job_review(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_job_inbox import review_agent_job

    job_id = str(arguments.get("job_id", "") or "").strip()
    if not job_id:
        raise ValueError("job_id is required")
    context_path = _resolve_context_path(arguments, manager)
    return review_agent_job(context_path, job_id)


def _tool_agent_job_archive(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_job_inbox import archive_agent_job

    job_id = str(arguments.get("job_id", "") or "").strip()
    if not job_id:
        raise ValueError("job_id is required")
    context_path = _resolve_context_path(arguments, manager)
    return archive_agent_job(context_path, job_id).to_dict()


def _tool_agent_job_promote(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_job_inbox import archive_agent_job, promote_agent_job_to_handoff

    job_id = str(arguments.get("job_id", "") or "").strip()
    if not job_id:
        raise ValueError("job_id is required")
    if not bool(arguments.get("to_handoff", True)):
        raise ValueError("agent.job.promote currently supports --to-handoff only")
    context_path = _resolve_context_path(arguments, manager)
    payload = promote_agent_job_to_handoff(
        context_path,
        job_id,
        handoff_name=str(arguments.get("handoff_name", "") or ""),
    )
    if bool(arguments.get("archive", False)):
        payload["archived"] = archive_agent_job(context_path, job_id).to_dict()
    return payload


def _tool_agent_job_claim(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_jobs import AgentJobQueue

    job_id = str(arguments.get("job_id", "") or "").strip()
    agent_name = str(arguments.get("agent_name", "") or "").strip()
    if not job_id or not agent_name:
        raise ValueError("job_id and agent_name are required")
    context_path = _resolve_context_path(arguments, manager)
    job = AgentJobQueue(context_path).claim(job_id, agent_name)
    return job.to_dict()


def _tool_agent_job_move(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .agent_jobs import AgentJobQueue

    job_id = str(arguments.get("job_id", "") or "").strip()
    status = str(arguments.get("status", "") or "").strip()
    if not job_id or not status:
        raise ValueError("job_id and status are required")
    context_path = _resolve_context_path(arguments, manager)
    job = AgentJobQueue(context_path).move(
        job_id,
        status,
        result=str(arguments.get("result", "") or ""),
    )
    return job.to_dict()


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
        "agents": [{"name": a.name, "state": a.state, "last_event": a.last_event} for a in awaiting]
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
    from .messages import MessageBus

    agent_name = arguments.get("agent_name", "")
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise ValueError("agent_name is required")
    topics = arguments.get("topics", [])
    if not isinstance(topics, list) or not topics:
        raise ValueError("topics must be a non-empty list")

    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    bus = MessageBus(context_path, scope_id=scoped.scope_id, config=manager.config)
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
    from .messages import MessageBus

    agent_name = arguments.get("agent_name", "")
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise ValueError("agent_name is required")
    topics = arguments.get("topics", [])
    if not isinstance(topics, list) or not topics:
        raise ValueError("topics must be a non-empty list")

    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    bus = MessageBus(context_path, scope_id=scoped.scope_id, config=manager.config)
    sub = bus.unsubscribe(agent_name.strip(), [str(t).strip() for t in topics if str(t).strip()])
    return sub.to_dict()


def _tool_hivemind_reap(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .messages import MessageBus

    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    if (
        scoped.layout_version == LAYOUT_VERSION
        and arguments.get("all_projects") is not True
    ):
        raise PermissionError(
            "v2 hivemind cleanup is queue-wide; set all_projects=true"
        )
    bus = MessageBus(
        context_path,
        scope_id=scoped.scope_id,
        config=manager.config,
        all_projects=True,
        include_legacy=scoped.layout_version != LAYOUT_VERSION,
    )
    max_age_hours = arguments.get("max_age_hours")
    if max_age_hours is not None:
        max_age_hours = _coerce_int(max_age_hours, default=24, minimum=1, maximum=24 * 30)
    raw_dry_run = arguments.get("dry_run")
    dry_run = raw_dry_run if isinstance(raw_dry_run, bool) else raw_dry_run is not None
    return bus.reap(
        max_age_hours=max_age_hours,
        dry_run=dry_run,
    )


def _tool_messages_clean(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    if arguments.get("all_projects") is not True:
        raise PermissionError("message cleanup is queue-wide; set all_projects=true")
    apply = arguments.get("apply") is True
    forwarded = dict(arguments)
    forwarded["dry_run"] = not apply
    result = _tool_hivemind_reap(forwarded, manager)
    result["applied"] = apply
    return result


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
    chunk_size = _coerce_int(arguments.get("chunk_size"), default=0, minimum=0, maximum=100000)
    chunk_overlap = _coerce_int(
        arguments.get("chunk_overlap"), default=200, minimum=0, maximum=100000
    )

    result = build_embedding_index(
        sources,
        output_dir,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        incremental=incremental,
        chunk_size=chunk_size or None,
        chunk_overlap=chunk_overlap,
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

    title = arguments.get("title")
    if not isinstance(title, str) or not title.strip():
        title = f"Handoff from {agent_name.strip()}"
    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    store = HandoffStore(
        context_path,
        scope_id=scoped.scope_id,
        config=manager.config,
    )
    packet = store.create_revision(
        title=title,
        revision_id=arguments.get("session_id"),
        agent_name=agent_name.strip(),
        accomplished=arguments.get("accomplished", []),
        blocked=arguments.get("blocked", []),
        next_steps=arguments.get("next_steps", []),
        context_snapshot=arguments.get("context_snapshot", {}),
        open_tasks=arguments.get("open_tasks", []),
        metadata=arguments.get("metadata", {}),
        target_agent=arguments.get("target_agent"),
        priority=str(arguments.get("priority", "normal") or "normal"),
        project_id=scoped.project_id,
    )
    payload = packet.to_dict()
    payload["scope_id"] = scoped.scope_id
    return payload


def _tool_handoff_read(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .handoff import HandoffStore

    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    store = HandoffStore(context_path, scope_id=scoped.scope_id, config=manager.config)
    session_id = arguments.get("session_id")
    packet = store.read(session_id=session_id)
    if packet is None:
        return {"error": "no handoff packet found"}
    payload = packet.to_dict()
    payload["scope_id"] = scoped.scope_id
    return payload


def _tool_handoff_list(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .handoff import HandoffStore

    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    store = HandoffStore(context_path, scope_id=scoped.scope_id, config=manager.config)
    limit = _coerce_int(arguments.get("limit"), default=10, minimum=1, maximum=100)
    packets = store.list(limit=limit)
    return {
        "scope_id": scoped.scope_id,
        "packets": [p.to_dict() for p in packets],
        "count": len(packets),
    }


def _tool_handoff_revise(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .handoff import HandoffStore

    revision_id = str(arguments.get("revision_id", "")).strip()
    title = str(arguments.get("title", "")).strip()
    agent_name = str(arguments.get("agent_name", "")).strip()
    if not revision_id or not title or not agent_name:
        raise ValueError("revision_id, title, and agent_name are required")
    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    store = HandoffStore(context_path, scope_id=scoped.scope_id, config=manager.config)
    parent = store.read(session_id=revision_id)
    if parent is None:
        raise FileNotFoundError(f"handoff revision not found: {revision_id}")
    packet = store.create_revision(
        title=title,
        agent_name=agent_name,
        stream_id=parent.stream_id,
        supersedes=parent.revision_id,
        accomplished=arguments.get("accomplished", []),
        blocked=arguments.get("blocked", []),
        next_steps=arguments.get("next_steps", []),
        context_snapshot=arguments.get("context_snapshot", {}),
        open_tasks=arguments.get("open_tasks", []),
        metadata=arguments.get("metadata", {}),
        target_agent=arguments.get("target_agent"),
        priority=str(arguments.get("priority", "normal") or "normal"),
        project_id=scoped.project_id,
    )
    payload = packet.to_dict()
    payload["scope_id"] = scoped.scope_id
    return payload


def _tool_handoff_threads(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .handoff import HandoffStore, HandoffStream

    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    store = HandoffStore(context_path, scope_id=scoped.scope_id, config=manager.config)
    limit = _coerce_int(arguments.get("limit"), default=20, minimum=1, maximum=100)
    streams: list[HandoffStream] = store.list_streams(limit=limit)
    return {"threads": [stream.to_dict() for stream in streams], "count": len(streams)}


def _tool_handoff_ack(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .handoff import HandoffStore

    revision_id = str(arguments.get("revision_id", "")).strip()
    actor = str(arguments.get("by", "")).strip()
    if not revision_id or not actor:
        raise ValueError("revision_id and by are required")
    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    store = HandoffStore(context_path, scope_id=scoped.scope_id, config=manager.config)
    if not store.acknowledge(revision_id, actor):
        raise FileNotFoundError(f"handoff revision not found: {revision_id}")
    return {"revision_id": revision_id, "acknowledged_by": actor}


def _tool_handoff_close(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .handoff import HandoffStore

    identifier = str(arguments.get("identifier", "")).strip()
    actor = str(arguments.get("by", "")).strip()
    if not identifier or not actor:
        raise ValueError("identifier and by are required")
    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    store = HandoffStore(context_path, scope_id=scoped.scope_id, config=manager.config)
    if not store.close(identifier, actor=actor, reason=str(arguments.get("reason", "") or "")):
        raise FileNotFoundError(f"handoff thread or revision not found: {identifier}")
    return {"identifier": identifier, "closed_by": actor}


def _tool_note_create(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .artifacts import NoteStore

    title = arguments.get("title")
    body = arguments.get("body")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title is required")
    if not isinstance(body, str):
        raise ValueError("body must be a string")
    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    note = NoteStore(context_path, scope_id=scoped.scope_id, config=manager.config).create(
        title=title,
        body=body,
        project_id=scoped.project_id,
        task_id=str(arguments.get("task_id", "") or ""),
        agent_name=str(arguments.get("agent_name", "") or ""),
        author_kind=str(arguments.get("author_kind", "agent") or "agent"),
        sensitivity=str(arguments.get("sensitivity", "internal") or "internal"),
    )
    return note.to_dict()


def _tool_note_read(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .artifacts import NoteStore

    identifier = str(arguments.get("identifier", "")).strip()
    if not identifier:
        raise ValueError("identifier is required")
    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    note = NoteStore(context_path, scope_id=scoped.scope_id, config=manager.config).read(identifier)
    if note is None:
        raise FileNotFoundError(f"note not found: {identifier}")
    return note.to_dict()


def _tool_note_list(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .artifacts import NoteStore

    context_path, scoped = _resolve_mcp_scope(arguments, manager)
    limit = _coerce_int(arguments.get("limit"), default=20, minimum=1, maximum=100)
    notes = NoteStore(context_path, scope_id=scoped.scope_id, config=manager.config).list(
        limit=limit
    )
    return {"notes": [note.to_dict() for note in notes], "count": len(notes)}


def _tool_hivemind_cleanup(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    from .hivemind import HivemindBus

    context_path = _resolve_context_path(arguments, manager)
    max_age_hours = _coerce_int(
        arguments.get("max_age_hours"),
        default=manager.config.hivemind.default_ttl_hours,
        minimum=1,
        maximum=8760,
    )
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
    context_path, scoped = _resolve_mcp_scope(arguments, manager)
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
    payload = index.freshness_scores(
        mount_types=mount_types,
        decay_hours=decay_hours,
        threshold=threshold,
        relative_prefixes=(
            visible_scope_prefixes(scoped)
            if scoped.layout_version == LAYOUT_VERSION
            else None
        ),
        scoped=scoped if scoped.layout_version == LAYOUT_VERSION else None,
    )
    payload["scope_id"] = scoped.scope_id
    payload["project_id"] = scoped.project_id
    return payload


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
    return build_session_timeline(
        context_path, session_id=session_id, since=since, limit=limit, config=manager.config
    )


def _resolve_codebase_index_dir(context_path: Path, manager: AFSManager) -> Path:
    """Return the codebase symbol index directory for *context_path*."""
    from .codebase_index import codebase_index_dir

    return codebase_index_dir(context_path)


def _resolve_embedding_index_dir(context_path: Path, manager: AFSManager) -> Path | None:
    """Return the embedding index dir if it exists, else None."""
    from .context_paths import resolve_mount_root
    from .models import MountType

    try:
        knowledge = resolve_mount_root(context_path, MountType.KNOWLEDGE)
        candidate = knowledge / "embeddings"
        if (candidate / "embedding_index.json").exists():
            return candidate
    except Exception as exc:  # noqa: BLE001 - optional embedding-index boundary
        _warn_fallback(f"knowledge embedding index discovery failed for {context_path}", exc)
    # Fallback: look in scratchpad
    try:
        scratchpad = resolve_mount_root(context_path, MountType.SCRATCHPAD)
        candidate = scratchpad / "embeddings"
        if (candidate / "embedding_index.json").exists():
            return candidate
    except Exception as exc:  # noqa: BLE001 - optional embedding-index boundary
        _warn_fallback(f"scratchpad embedding index discovery failed for {context_path}", exc)
    return None


def _reject_legacy_retrieval_tool_in_v2(context_path: Path, *, tool_name: str) -> None:
    """Fail closed before an unscoped legacy index can read a v2 namespace."""
    if detect_layout_version(context_path) == LAYOUT_VERSION:
        raise ValueError(
            f"{tool_name} is unavailable for context v2 because its indexes are unscoped; "
            "use context.search with an authorized project_path or scope_id"
        )


def _fuse_search_results(
    fts: list[dict[str, Any]],
    emb: list[dict[str, Any]],
    sym: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Normalize, merge, deduplicate by source_path, and re-rank results."""
    # Normalize FTS relevance_score to [0,1]: BM25 from SQLite is a positive
    # float where higher is better (the index uses -bm25() internally).
    fts_scores = [float(e.get("relevance_score") or 0) for e in fts]
    fts_max = max(fts_scores, default=1.0) or 1.0
    # Normalize embedding scores (already [0,1] cosine or keyword [0,1])
    emb_scores = [float(e.get("score", 0)) for e in emb]
    emb_max = max(emb_scores, default=1.0) or 1.0
    sym_scores = [float(e.get("score", 1)) for e in sym]
    sym_max = max(sym_scores, default=1.0) or 1.0

    # key: absolute source_path, value: best merged entry
    merged: dict[str, dict[str, Any]] = {}

    def _update(entry: dict[str, Any], norm_score: float, source_tag: str) -> None:
        path = str(entry.get("source_path") or entry.get("absolute_path", ""))
        if not path:
            return
        if path not in merged:
            merged[path] = {
                "source_path": path,
                "relative_path": str(entry.get("relative_path") or ""),
                "fused_score": norm_score,
                "sources": [source_tag],
                "mount_type": entry.get("mount_type", ""),
                "size_bytes": entry.get("size_bytes", 0),
                "content_excerpt": entry.get("content_excerpt") or entry.get("text_preview", ""),
                "symbol_name": entry.get("symbol_name"),
                "symbol_kind": entry.get("kind"),
                "line_start": entry.get("line_start"),
                "chunk_index": entry.get("chunk_index"),
                "chunk_line_start": entry.get("line_start")
                if entry.get("chunk_index") is not None
                else None,
                "chunk_line_end": entry.get("line_end")
                if entry.get("chunk_index") is not None
                else None,
            }
        else:
            existing = merged[path]
            existing["fused_score"] = max(existing["fused_score"], norm_score)
            if source_tag not in existing["sources"]:
                existing["sources"].append(source_tag)
            if not existing.get("symbol_name") and entry.get("symbol_name"):
                existing["symbol_name"] = entry["symbol_name"]
                existing["symbol_kind"] = entry.get("kind")
                existing["line_start"] = entry.get("line_start")

    for i, entry in enumerate(fts):
        raw = fts_scores[i]
        norm = raw / fts_max if fts_max else (1.0 / (i + 1))
        _update(entry, norm * 0.85, "fts")  # slight weight discount vs embedding

    for i, entry in enumerate(emb):
        raw = emb_scores[i]
        norm = raw / emb_max if emb_max else (1.0 / (i + 1))
        _update(
            {
                "source_path": entry.get("source_path", ""),
                "text_preview": entry.get("text_preview", ""),
                "score": raw,
                "chunk_index": entry.get("chunk_index"),
                "line_start": entry.get("line_start"),
                "line_end": entry.get("line_end"),
            },
            norm,
            "embedding",
        )

    for i, entry in enumerate(sym):
        raw = sym_scores[i]
        norm = raw / sym_max if sym_max else (1.0 / (i + 1))
        _update(
            {
                "source_path": entry.get("source_path", ""),
                "symbol_name": entry.get("symbol_name"),
                "kind": entry.get("kind"),
                "line_start": entry.get("line_start"),
            },
            norm * 0.9,
            "symbol",
        )

    ranked = sorted(merged.values(), key=lambda r: r["fused_score"], reverse=True)
    return ranked[:limit]


def _tool_afs_search(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    """Unified search: FTS + embedding + symbol index, fused and ranked."""
    query = str(arguments.get("query", "")).strip()
    if not query:
        raise ValueError("query is required")

    try:
        context_path = _resolve_context_path(arguments, manager)
    except FileNotFoundError as exc:
        return {"error": str(exc), "results": [], "sources_used": [], "query": query}
    _reject_legacy_retrieval_tool_in_v2(context_path, tool_name="afs.search")

    limit = _coerce_int(arguments.get("limit"), default=20, minimum=1, maximum=100)
    include_fts = bool(arguments.get("include_fts", True))
    include_embeddings = bool(arguments.get("include_embeddings", True))
    include_symbols = bool(arguments.get("include_symbols", True))
    mount_types = _parse_mount_types(arguments.get("mount_types"))

    fts_entries: list[dict[str, Any]] = []
    emb_entries: list[dict[str, Any]] = []
    semantic_status = "not_requested"
    semantic_reason: str | None = None
    sym_entries: list[dict[str, Any]] = []
    sources_used: list[str] = []

    # FTS leg
    if include_fts:
        try:
            payload = _query_context_index(
                context_path=context_path,
                manager=manager,
                query=query,
                mount_types=mount_types,
                limit=min(limit * 2, 40),
                include_content=False,
            )
            fts_entries = payload.get("entries", [])
            if fts_entries:
                sources_used.append("fts")
        except Exception as exc:  # noqa: BLE001 - optional search-provider boundary
            _warn_fallback("FTS search unavailable", exc)

    # Embedding leg
    if include_embeddings:
        emb_dir = _resolve_embedding_index_dir(context_path, manager)
        if emb_dir is not None:
            provider = arguments.get("provider")
            model = arguments.get("model")
            try:
                from .embeddings import create_embed_fn, search_embedding_index_detailed

                embed_fn = None
                if isinstance(provider, str) and provider.strip():
                    kwargs: dict[str, Any] = {}
                    if isinstance(model, str) and model.strip():
                        kwargs["model"] = model.strip()
                    embed_fn = create_embed_fn(provider.strip(), **kwargs)
                emb_response = search_embedding_index_detailed(
                    emb_dir,
                    query,
                    embed_fn=embed_fn,
                    recreate_query_embedder=embed_fn is None,
                    top_k=min(limit, 15),
                    min_score=0.2,
                )
                semantic_status = emb_response.semantic_status
                semantic_reason = emb_response.semantic_reason
                emb_results = emb_response.results
                emb_entries = [r.to_dict() for r in emb_results]
                if emb_entries:
                    sources_used.append(
                        "semantic" if semantic_status == "ready" else "indexed_text"
                    )
            except Exception as exc:  # noqa: BLE001 - reported through semantic status
                semantic_status = "fallback"
                semantic_reason = f"embedding search unavailable: {exc}"
        else:
            semantic_status = "unavailable"
            semantic_reason = "no embedding index was discovered"

    # Symbol leg
    if include_symbols:
        idx_dir = _resolve_codebase_index_dir(context_path, manager)
        if (idx_dir / "index.json").exists():
            try:
                from .codebase_index import search_codebase_index

                sym_results = search_codebase_index(idx_dir, query, limit=min(limit, 15))
                sym_entries = [r.to_dict() for r in sym_results]
                if sym_entries:
                    sources_used.append("symbol")
            except Exception as exc:  # noqa: BLE001 - optional search-provider boundary
                _warn_fallback("symbol search unavailable", exc)

    results = _fuse_search_results(fts_entries, emb_entries, sym_entries, limit)

    return {
        "query": query,
        "context_path": str(context_path),
        "sources_used": sources_used,
        "semantic_status": semantic_status,
        "semantic_reason": semantic_reason,
        "total": len(results),
        "results": results,
    }


def _tool_afs_codebase_symbols(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    """Look up symbol definitions from the AST codebase index."""
    from .codebase_index import search_codebase_index

    query = str(arguments.get("query", "")).strip()
    if not query:
        raise ValueError("query is required")

    try:
        context_path = _resolve_context_path(arguments, manager)
    except FileNotFoundError as exc:
        return {"error": str(exc), "symbols": [], "query": query}
    _reject_legacy_retrieval_tool_in_v2(
        context_path,
        tool_name="afs.codebase.symbols",
    )

    idx_dir = _resolve_codebase_index_dir(context_path, manager)
    if not (idx_dir / "index.json").exists():
        return {
            "query": query,
            "symbols": [],
            "total": 0,
            "error": f"No codebase index found at {idx_dir}. Run afs.codebase.index first.",
        }

    kind = arguments.get("kind")
    language = arguments.get("language")
    limit = _coerce_int(arguments.get("limit"), default=20, minimum=1, maximum=200)
    exact = bool(arguments.get("exact", False))

    results = search_codebase_index(
        idx_dir,
        query,
        kind=kind if isinstance(kind, str) else None,
        language=language if isinstance(language, str) else None,
        limit=limit,
        exact=exact,
    )

    return {
        "query": query,
        "kind": kind,
        "language": language,
        "total": len(results),
        "symbols": [r.to_dict() for r in results],
    }


def _tool_afs_codebase_index(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    """Build or update the AST codebase symbol index for a project."""
    from .codebase_index import build_codebase_index, codebase_index_dir

    try:
        context_path = _resolve_context_path(arguments, manager)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    _reject_legacy_retrieval_tool_in_v2(
        context_path,
        tool_name="afs.codebase.index",
    )

    project_root_raw = arguments.get("project_path") or arguments.get("path")
    if isinstance(project_root_raw, str) and project_root_raw.strip():
        project_root = Path(project_root_raw).expanduser().resolve()
    else:
        from .codebase_explorer import infer_project_root as _infer

        project_root = _infer(context_path)

    output_dir = codebase_index_dir(context_path)
    max_files = _coerce_int(arguments.get("max_files"), default=5000, minimum=1, maximum=50000)
    incremental = bool(arguments.get("incremental", True))
    languages_raw = arguments.get("languages")
    languages = (
        [str(lang) for lang in languages_raw if isinstance(lang, str)]
        if isinstance(languages_raw, list)
        else None
    )

    result = build_codebase_index(
        project_root,
        output_dir,
        max_files=max_files,
        incremental=incremental,
        languages=languages,
    )

    return {
        "summary": result.summary(),
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "total_files": result.total_files,
        "indexed": result.indexed,
        "skipped": result.skipped,
        "reused": result.reused,
        "removed": result.removed,
        "errors": result.errors[:10],
        "mode": result.mode,
    }


def _profile_skill_roots(manager: AFSManager) -> tuple[str, list[Path]]:
    profile = resolve_active_profile(manager.config)
    roots = resolve_skill_roots(
        list(profile.skill_roots),
        afs_root=os.getenv("AFS_ROOT", "").strip() or None,
    )
    return profile.name, roots


def _reject_skill_arguments(
    arguments: dict[str, Any],
    message: str,
) -> NoReturn:
    # Registry failures are recorded after the handler returns. Drop rejected
    # payloads so invalid or unknown values cannot be persisted in history.
    arguments.clear()
    raise ValueError(message)


def _require_known_skill_arguments(
    arguments: dict[str, Any],
    *,
    allowed: frozenset[str],
    tool_name: str,
) -> None:
    if any(key not in allowed for key in arguments):
        _reject_skill_arguments(
            arguments,
            f"{tool_name} received unsupported arguments",
        )


def _has_ascii_or_c1_controls(value: str) -> bool:
    return any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in value)


def _safe_skill_error_label(value: str, *, max_chars: int) -> str:
    """Render one bounded label without terminal control characters."""
    return escape_skill_diagnostic_text(
        value,
        max_chars=max_chars,
    )


def _skill_diagnostic_error_suffix(
    diagnostic_payload: dict[str, Any],
) -> str:
    count = diagnostic_payload["diagnostic_count"]
    if not isinstance(count, int) or count <= 0:
        return ""
    diagnostics = diagnostic_payload["diagnostics"]
    examples: list[str] = []
    if isinstance(diagnostics, list):
        for item in diagnostics[:3]:
            if not isinstance(item, dict):
                continue
            code = _safe_skill_error_label(
                str(item.get("code") or "skill_warning"),
                max_chars=80,
            )
            path = _safe_skill_error_label(
                str(item.get("path") or item.get("root") or ""),
                max_chars=120,
            )
            examples.append(f"{code}: {path}")
    detail = f": {'; '.join(examples)}" if examples else ""
    return " " + _safe_skill_error_label(
        f"Skill discovery warnings: {count}{detail}.",
        max_chars=MAX_SKILL_DIAGNOSTIC_ERROR_CHARS,
    )


def _skill_match_top_k(raw: Any) -> int:
    if raw is None:
        return 5
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError("top_k must be an integer from 1 to 10")
    if not 1 <= raw <= MAX_SKILL_MATCHES:
        raise ValueError("top_k must be an integer from 1 to 10")
    return int(raw)


def _skill_match_include_bodies(raw: Any) -> bool:
    if raw is None:
        return False
    if not isinstance(raw, bool):
        raise ValueError("include_bodies must be a boolean")
    return raw


def _tool_skill_match(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    _require_known_skill_arguments(
        arguments,
        allowed=SKILL_MATCH_ARGUMENT_NAMES,
        tool_name="skill.match",
    )
    raw_prompt = arguments.get("prompt")
    if not isinstance(raw_prompt, str):
        _reject_skill_arguments(arguments, "prompt must be a non-empty string")
    if len(raw_prompt) > MAX_SKILL_MATCH_PROMPT_CHARS:
        _reject_skill_arguments(
            arguments, f"prompt must be at most {MAX_SKILL_MATCH_PROMPT_CHARS} characters"
        )
    prompt = raw_prompt.strip()
    if not prompt:
        _reject_skill_arguments(arguments, "prompt must be a non-empty string")

    try:
        top_k = _skill_match_top_k(arguments.get("top_k"))
        include_bodies = _skill_match_include_bodies(arguments.get("include_bodies"))
    except ValueError as exc:
        _reject_skill_arguments(arguments, str(exc))
    profile_name, roots = _profile_skill_roots(manager)
    match_result = build_skill_matches_with_diagnostics(
        prompt,
        roots,
        profile=profile_name,
        top_k=top_k,
        max_body_chars=MAX_SKILL_BODY_CHARS if include_bodies else 0,
        max_total_body_chars=MAX_SKILL_BODIES_CHARS if include_bodies else 0,
        max_body_matches=MAX_SKILL_BODY_MATCHES if include_bodies else 0,
    )
    matches = match_result.matches
    if not include_bodies:
        for match in matches:
            match["body_omitted"] = "not_requested"
    diagnostic_payload = bounded_skill_diagnostics(
        match_result.diagnostics,
        diagnostic_count=match_result.diagnostic_count,
        limit=MAX_MCP_SKILL_DIAGNOSTICS,
    )
    return {
        "profile": profile_name,
        "prompt": prompt,
        "include_bodies": include_bodies,
        "matches": matches,
        **diagnostic_payload,
    }


def _skill_path_within_roots(path: Path, roots: list[Path]) -> tuple[Path, Path]:
    resolved = path.expanduser().resolve()
    for root in roots:
        resolved_root = root.expanduser().resolve()
        try:
            resolved.relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        return resolved, resolved_root
    raise PermissionError(f"Skill path outside configured roots: {resolved}")


def _tool_skill_read(arguments: dict[str, Any], manager: AFSManager) -> dict[str, Any]:
    _require_known_skill_arguments(
        arguments,
        allowed=SKILL_READ_ARGUMENT_NAMES,
        tool_name="skill.read",
    )
    raw_name = arguments.get("name")
    if not isinstance(raw_name, str):
        _reject_skill_arguments(arguments, "name must be a non-empty string")
    if len(raw_name) > MAX_SKILL_NAME_CHARS:
        _reject_skill_arguments(
            arguments,
            f"name must be at most {MAX_SKILL_NAME_CHARS} characters",
        )
    if _has_ascii_or_c1_controls(raw_name):
        _reject_skill_arguments(
            arguments,
            "name must not contain ASCII or C1 control characters",
        )
    name = raw_name.strip()
    if not name:
        _reject_skill_arguments(arguments, "name must be a non-empty string")
    wanted = name.casefold()
    profile_name, roots = _profile_skill_roots(manager)
    discovery = discover_skills_with_diagnostics(roots, profile=profile_name)
    skills = discovery.skills
    diagnostic_payload = bounded_skill_diagnostics(
        discovery.diagnostics,
        diagnostic_count=discovery.diagnostic_count,
        limit=MAX_MCP_SKILL_DIAGNOSTICS,
    )
    for skill in skills:
        if skill.name.casefold() != wanted:
            continue
        skill_path, trusted_root = _skill_path_within_roots(skill.path, roots)
        body, body_truncated = read_skill_body(
            skill_path,
            max_chars=MAX_SKILL_BODY_CHARS,
            trusted_root=trusted_root,
        )
        return {
            "profile": profile_name,
            **bounded_skill_metadata(replace(skill, path=skill_path)),
            "body": body,
            "body_truncated": body_truncated,
            "body_chars": len(body),
            **diagnostic_payload,
        }

    known = sorted({skill.name for skill in skills}, key=str.casefold)
    preview_items = [
        _safe_skill_error_label(
            item,
            max_chars=MAX_SKILL_METADATA_ITEM_CHARS,
        )
        for item in known[:MAX_SKILL_MATCHES]
    ]
    preview = ", ".join(preview_items) or "none discovered"
    if len(preview) > MAX_SKILL_KNOWN_PREVIEW_CHARS:
        preview = preview[: MAX_SKILL_KNOWN_PREVIEW_CHARS - 3].rstrip() + "..."
    suffix = (
        f" (and {len(known) - MAX_SKILL_MATCHES} more)" if len(known) > MAX_SKILL_MATCHES else ""
    )
    diagnostic_suffix = _skill_diagnostic_error_suffix(diagnostic_payload)
    raise ValueError(
        f"Unknown skill '{name}'. Known skills: {preview}{suffix}{diagnostic_suffix}"
    )


def _tool_alias_definition(
    tool: MCPToolDefinition,
    *,
    name: str,
    description: str | None = None,
) -> MCPToolDefinition:
    return replace(
        tool,
        name=name,
        description=description or tool.description,
    )


def _builtin_tool_definitions() -> list[MCPToolDefinition]:
    fs_read = MCPToolDefinition(
        name="fs.read",
        description="Legacy compatibility alias for `context.read`. Read UTF-8 text from a context-scoped file after bootstrap or `context.status` anchors the active context.",
        input_schema={
            "type": "object",
            "properties": {
                **_mcp_scope_properties(),
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
                **_mcp_scope_properties(),
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
                **_mcp_scope_properties(),
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
                **_mcp_scope_properties(),
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
                **_mcp_scope_properties(),
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
            name="messages.send",
            description="Send a scoped inter-agent message to the current project or common scope.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(include_all_projects=True),
                    "from": {"type": "string", "description": "Sending agent name."},
                    "type": {"type": "string", "default": "status"},
                    "payload": {"type": "object"},
                    "to": {"type": "string"},
                    "topic": {"type": "string"},
                    "ttl_hours": {"type": "integer"},
                },
                "required": ["from"],
                "additionalProperties": False,
            },
            handler=_tool_hivemind_send,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="messages.read",
            description="Read messages visible to the current project plus common messages.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(include_all_projects=True),
                    "agent": {"type": "string"},
                    "type": {"type": "string"},
                    "topic": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                    "include_legacy": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            handler=_tool_hivemind_read,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="messages.subscribe",
            description="Subscribe an agent to message topics.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "agent_name": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}},
                    "ttl_hours": {"type": "integer"},
                },
                "required": ["agent_name", "topics"],
                "additionalProperties": False,
            },
            handler=_tool_hivemind_subscribe,
        ),
        MCPToolDefinition(
            name="messages.unsubscribe",
            description="Unsubscribe an agent from message topics.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "agent_name": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["agent_name", "topics"],
                "additionalProperties": False,
            },
            handler=_tool_hivemind_unsubscribe,
        ),
        MCPToolDefinition(
            name="messages.clean",
            description="Preview or apply queue-wide expired-message cleanup; all_projects is required.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(include_all_projects=True),
                    "max_age_hours": {"type": "integer"},
                    "apply": {"type": "boolean", "default": False},
                },
                "required": ["all_projects"],
                "additionalProperties": False,
            },
            handler=_tool_messages_clean,
        ),
        MCPToolDefinition(
            name="note.create",
            description="Create an immutable readable Markdown note in the current project scope.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "task_id": {"type": "string"},
                    "agent_name": {"type": "string"},
                    "author_kind": {
                        "type": "string",
                        "enum": ["agent", "human", "import", "system"],
                        "default": "agent",
                    },
                    "sensitivity": {
                        "type": "string",
                        "enum": ["internal", "private", "public", "restricted"],
                        "default": "internal",
                    },
                },
                "required": ["title", "body"],
                "additionalProperties": False,
            },
            handler=_tool_note_create,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="note.read",
            description="Read one immutable note from the authorized scope.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "identifier": {"type": "string"},
                },
                "required": ["identifier"],
                "additionalProperties": False,
            },
            handler=_tool_note_read,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="note.list",
            description="List recent immutable notes from the authorized scope.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "limit": {"type": "integer", "default": 20},
                },
                "additionalProperties": False,
            },
            handler=_tool_note_list,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="skill.match",
            description=(
                "Rank configured AFS skills against a bounded task description. "
                "Use when the task shifts mid-session; request bodies only when "
                "you need inline instructions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_SKILL_MATCH_PROMPT_CHARS,
                        "description": "Task or intent text to match against skill triggers.",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_SKILL_MATCHES,
                        "default": 5,
                    },
                    "include_bodies": {"type": "boolean", "default": False},
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
            handler=_tool_skill_match,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="skill.read",
            description=(
                "Read one configured AFS skill by name. Returns metadata and at "
                f"most {MAX_SKILL_BODY_CHARS} characters of instructions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 256,
                        "description": "Skill name reported by skill.match.",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=_tool_skill_read,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="briefing",
            description="Morning briefing — calendar, gmail, open tasks, and active agents. Use at the start of a session to understand current state.",
            input_schema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "default": 7,
                        "description": "Lookback window in days.",
                    },
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
                    **_mcp_scope_properties(include_all_projects=True),
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
                    **_mcp_scope_properties(include_all_projects=True),
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
            name="context.search",
            description=(
                "Search the v2 scoped hybrid index. Defaults to local text retrieval; "
                "semantic=true explicitly permits the index's embedding provider for the query."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(include_all_projects=True),
                    "query": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["text", "symbol"],
                        "default": "text",
                    },
                    "semantic": {"type": "boolean", "default": False},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=_tool_context_search,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="context.diff",
            description="Show new, modified, and deleted files since the last index build. Call this before editing to understand drift since the previous session.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
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
                    **_mcp_scope_properties(),
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
                    **_mcp_scope_properties(),
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
                    "semantic": {
                        "type": "boolean",
                        "default": False,
                        "description": "Explicitly permit remote query embeddings.",
                    },
                    "max_query_results": {"type": "integer", "default": 6},
                    "max_embedding_results": {"type": "integer", "default": 4},
                },
                "additionalProperties": False,
            },
            handler=_tool_session_pack,
        ),
        MCPToolDefinition(
            name="work.communication.list",
            description="List captured work communication samples before drafting docs, design docs, requirements, or replies in the user's work style.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "person_id": {"type": "string"},
                    "purpose": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
                "additionalProperties": False,
            },
            handler=_tool_work_communication_list,
        ),
        MCPToolDefinition(
            name="work.communication.add",
            description="Capture a work communication sample for future tone/style grounding. Store only deliberate work samples, not arbitrary private text.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "text": {"type": "string"},
                    "person_id": {"type": "string"},
                    "source_system": {"type": "string"},
                    "source_id": {"type": "string"},
                    "channel": {"type": "string"},
                    "purpose": {"type": "string"},
                    "style_notes": {"type": "array", "items": {"type": "string"}},
                    "provenance": {"type": "array", "items": {"type": "object"}},
                    "confidence": {"type": "number", "default": 0.5},
                    "dedupe_key": {"type": "string"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=_tool_work_communication_add,
        ),
        MCPToolDefinition(
            name="work.communication.guide",
            description="Summarize available work communication style evidence and mandatory approval guardrails for work-context writing.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "person_id": {"type": "string"},
                    "purpose": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
                "additionalProperties": False,
            },
            handler=_tool_work_communication_guide,
        ),
        MCPToolDefinition(
            name="work.communication.preflight",
            description=(
                "Run the mandatory work-writing preflight: style evidence, optional opt-in "
                "personal context, pending approvals, and explicit approval guardrails. "
                "This never executes an external write."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "person_id": {"type": "string"},
                    "purpose": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                    "approval_limit": {"type": "integer", "default": 10},
                    "personal_mode": {"type": "string"},
                    "personal_context_root": {"type": "string"},
                },
                "additionalProperties": False,
            },
            handler=_tool_work_communication_preflight,
        ),
        MCPToolDefinition(
            name="work.approvals.list",
            description="List local AFS work approval requests. Review this before using connector write tools or claiming permission to post externally.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "status": {"type": "string", "default": "pending"},
                    "all": {"type": "boolean", "default": False},
                    "limit": {"type": "integer", "default": 50},
                },
                "additionalProperties": False,
            },
            handler=_tool_work_approvals_list,
        ),
        MCPToolDefinition(
            name="work.approvals.show",
            description="Show one local AFS work approval request and its preview/result details.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "approval_id": {"type": "string"},
                },
                "required": ["approval_id"],
                "additionalProperties": False,
            },
            handler=_tool_work_approvals_show,
        ),
        MCPToolDefinition(
            name="work.approvals.request",
            description="Create a local approval request for posting, sending, submitting, or editing an external work system. This asks for permission; it does not execute the write.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "target_system": {"type": "string"},
                    "target_id": {"type": "string"},
                    "action": {"type": "string"},
                    "summary": {"type": "string"},
                    "preview": {"type": "object"},
                    "affected_people": {"type": "array", "items": {}},
                    "risk_level": {"type": "string", "default": "medium"},
                    "permission_required": {"type": "string", "default": "human approval"},
                    "requested_by": {"type": "string", "default": "agent"},
                    "expires_at": {"type": "string"},
                    "dedupe_key": {"type": "string"},
                },
                "required": ["target_system", "target_id", "action", "summary"],
                "additionalProperties": False,
            },
            handler=_tool_work_approvals_request,
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
                    "limit": {
                        "type": "integer",
                        "description": "Max events to return.",
                        "default": 20,
                    },
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
                    "type": {
                        "type": "string",
                        "description": "Message type: finding, request, or status.",
                    },
                    "payload": {"type": "object", "description": "Message payload."},
                    "to": {"type": "string", "description": "Optional recipient agent name."},
                    "topic": {
                        "type": "string",
                        "description": "Optional topic for pub/sub routing (e.g. context:repair, agent:lifecycle).",
                    },
                    "ttl_hours": {
                        "type": "integer",
                        "description": "Optional per-message retention window in hours.",
                    },
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
                    "limit": {
                        "type": "integer",
                        "description": "Max messages to return.",
                        "default": 50,
                    },
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
                    "priority": {
                        "type": "integer",
                        "description": "Priority (1=highest, 10=lowest).",
                        "default": 5,
                    },
                    "context": {
                        "type": "object",
                        "description": "Task context (files, issue, etc).",
                    },
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
                    "status": {
                        "type": "string",
                        "description": "Filter by status: pending, claimed, in_progress, done, failed.",
                    },
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
            name="agent.manifest.show",
            description="Show or export the repo-owned agent harness manifest.",
            input_schema={
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Optional manifest TOML path."},
                    "harness": {
                        "type": "string",
                        "description": "Optional harness name to export.",
                    },
                    "validate": {"type": "boolean", "default": False},
                    "check_paths": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            handler=_tool_agent_manifest_show,
        ),
        MCPToolDefinition(
            name="agent.run.start",
            description="Start a replayable shared agent-run record.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "task": {"type": "string"},
                    "harness": {"type": "string"},
                    "workspace": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["task"],
                "additionalProperties": False,
            },
            handler=_tool_agent_run_start,
        ),
        MCPToolDefinition(
            name="agent.run.list",
            description="List replayable shared agent-run records.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "status": {"type": "string", "description": "Optional status filter."},
                    "limit": {"type": "integer", "default": 20},
                },
                "additionalProperties": False,
            },
            handler=_tool_agent_run_list,
        ),
        MCPToolDefinition(
            name="agent.run.show",
            description="Show one replayable agent run record.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "run_id": {"type": "string"},
                },
                "required": ["run_id"],
                "additionalProperties": False,
            },
            handler=_tool_agent_run_show,
        ),
        MCPToolDefinition(
            name="agent.run.event",
            description="Append an event to a replayable agent run record.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "run_id": {"type": "string"},
                    "event_type": {"type": "string"},
                    "summary": {"type": "string"},
                    "data": {"type": "object"},
                },
                "required": ["run_id", "event_type"],
                "additionalProperties": False,
            },
            handler=_tool_agent_run_event,
        ),
        MCPToolDefinition(
            name="agent.run.finish",
            description="Finish a replayable agent run record with summary, commands, and verification.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "run_id": {"type": "string"},
                    "status": {"type": "string", "default": "done"},
                    "summary": {"type": "string"},
                    "files_changed": {"type": "array", "items": {"type": "string"}},
                    "commands": {"type": "array", "items": {"type": "string"}},
                    "verification": {"type": "array", "items": {"type": "object"}},
                    "handoff_path": {"type": "string"},
                },
                "required": ["run_id"],
                "additionalProperties": False,
            },
            handler=_tool_agent_run_finish,
        ),
        MCPToolDefinition(
            name="agent.job.create",
            description="Create a markdown background agent job in items/agent_jobs/queue.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "title": {"type": "string"},
                    "prompt": {"type": "string"},
                    "priority": {"type": "integer", "default": 5},
                    "created_by": {"type": "string"},
                    "scope": {"type": "string"},
                    "expected_output": {"type": "string"},
                    "allow_destructive": {"type": "boolean", "default": False},
                },
                "required": ["title"],
                "additionalProperties": False,
            },
            handler=_tool_agent_job_create,
        ),
        MCPToolDefinition(
            name="agent.job.status",
            description="Show background job queue, worker, recent run, and watchdog status.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "label": {"type": "string", "default": "com.afs.agent-jobs"},
                    "stale_after_seconds": {"type": "number", "default": 3600},
                    "recent_runs": {"type": "integer", "default": 5},
                },
                "additionalProperties": False,
            },
            handler=_tool_agent_job_status,
        ),
        MCPToolDefinition(
            name="agent.job.inbox",
            description="Show completed, failed, stale, or blocked background jobs needing human review.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "stale_after_seconds": {"type": "number", "default": 3600},
                    "limit": {"type": "integer", "default": 20},
                },
                "additionalProperties": False,
            },
            handler=_tool_agent_job_inbox,
        ),
        MCPToolDefinition(
            name="agent.job.seed",
            description="Idempotently queue safe report-only background maintenance jobs.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "profile": {"type": "string", "default": "repo-maintenance"},
                    "cadence": {"type": "string", "default": "daily"},
                    "created_by": {"type": "string"},
                    "dry_run": {"type": "boolean", "default": False},
                    "force": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
            handler=_tool_agent_job_seed,
        ),
        MCPToolDefinition(
            name="agent.job.list",
            description="List markdown background agent jobs.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "status": {
                        "type": "string",
                        "description": "queue, running, done, failed, or archived.",
                    },
                },
                "additionalProperties": False,
            },
            handler=_tool_agent_job_list,
        ),
        MCPToolDefinition(
            name="agent.job.show",
            description="Show one markdown background agent job.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "job_id": {"type": "string"},
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
            handler=_tool_agent_job_show,
        ),
        MCPToolDefinition(
            name="agent.job.review",
            description="Review one background job and its linked run record.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "job_id": {"type": "string"},
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
            handler=_tool_agent_job_review,
        ),
        MCPToolDefinition(
            name="agent.job.archive",
            description="Archive one background job without deleting its markdown record.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "job_id": {"type": "string"},
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
            handler=_tool_agent_job_archive,
        ),
        MCPToolDefinition(
            name="agent.job.promote",
            description="Promote one background job review into a durable handoff.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "job_id": {"type": "string"},
                    "to_handoff": {"type": "boolean", "default": True},
                    "handoff_name": {"type": "string"},
                    "archive": {"type": "boolean", "default": False},
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
            handler=_tool_agent_job_promote,
        ),
        MCPToolDefinition(
            name="agent.job.claim",
            description="Claim a queued markdown background agent job.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "job_id": {"type": "string"},
                    "agent_name": {"type": "string"},
                },
                "required": ["job_id", "agent_name"],
                "additionalProperties": False,
            },
            handler=_tool_agent_job_claim,
        ),
        MCPToolDefinition(
            name="agent.job.move",
            description="Move a markdown background agent job to queue, running, done, failed, or archived.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string"},
                    "job_id": {"type": "string"},
                    "status": {"type": "string"},
                    "result": {"type": "string"},
                },
                "required": ["job_id", "status"],
                "additionalProperties": False,
            },
            handler=_tool_agent_job_move,
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
                    "event_type": {
                        "type": "string",
                        "description": "Filter by event type (mcp_tool, hivemind, embedding, agent_lifecycle, session).",
                    },
                    "since": {"type": "string", "description": "ISO 8601 datetime cutoff."},
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "description": "Max events to return.",
                    },
                    "source": {"type": "string", "description": "Filter by event source."},
                    "session_id": {
                        "type": "string",
                        "description": "Filter by recorded AFS session ID.",
                    },
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
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Max events to return.",
                    },
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
                    "hours": {
                        "type": "integer",
                        "default": 24,
                        "description": "Lookback window in hours.",
                    },
                    "event_type": {
                        "type": "string",
                        "description": "Optional single event type filter.",
                    },
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
                    "limit": {
                        "type": "integer",
                        "default": 200,
                        "description": "Max events to return (0 for all).",
                    },
                    "include_payloads": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include event payloads when available.",
                    },
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
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topics to subscribe to.",
                    },
                    "ttl_hours": {
                        "type": "integer",
                        "description": "Optional subscription TTL window in hours.",
                    },
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
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topics to unsubscribe from.",
                    },
                },
                "required": ["agent_name", "topics"],
                "additionalProperties": False,
            },
            handler=_tool_hivemind_unsubscribe,
        ),
        MCPToolDefinition(
            name="hivemind.reap",
            description=(
                "Remove expired or stale hivemind messages and return cleanup statistics; "
                "v2 queues require explicit all_projects=true."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(include_all_projects=True),
                    "max_age_hours": {
                        "type": "integer",
                        "description": "Override retention window in hours.",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Report removals without deleting files.",
                    },
                },
                "required": ["all_projects"],
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
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Source paths to index.",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for the index.",
                    },
                    "include_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns to include.",
                    },
                    "exclude_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns to exclude.",
                    },
                    "incremental": {
                        "type": "boolean",
                        "default": False,
                        "description": "Skip unchanged files using size+mtime comparison.",
                    },
                },
                "required": ["sources", "output_dir"],
                "additionalProperties": False,
            },
            handler=_tool_embeddings_index,
        ),
        MCPToolDefinition(
            name="afs.search",
            description=(
                "Unified search across the FTS context index, embedding index, and AST symbol index. "
                "Returns fused, ranked results from whichever sources are available. "
                "Use this before grep/glob when looking for code, docs, or context files."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "context_path": {
                        "type": "string",
                        "description": "Context path (uses default if omitted).",
                    },
                    "mount_types": {
                        "type": "string",
                        "description": "Comma-separated mount types for the FTS leg (e.g. 'scratchpad,knowledge').",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Embedding provider (ollama, openai, gemini, hf). Omit to skip embedding leg.",
                    },
                    "model": {"type": "string", "description": "Embedding model name."},
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Max results to return.",
                    },
                    "include_symbols": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include AST symbol index results.",
                    },
                    "include_embeddings": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include embedding index results.",
                    },
                    "include_fts": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include FTS context index results.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=_tool_afs_search,
        ),
        MCPToolDefinition(
            name="afs.codebase.symbols",
            description=(
                "Look up function, class, and import definitions from the AST codebase index. "
                "Faster and more precise than grep for finding where something is defined."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Symbol name or substring to search.",
                    },
                    "context_path": {
                        "type": "string",
                        "description": "Context path (uses default if omitted).",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["functions", "classes", "imports", "exports"],
                        "description": "Filter by symbol kind.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Filter by language (python, typescript, javascript, rust, go).",
                    },
                    "limit": {"type": "integer", "default": 20},
                    "exact": {
                        "type": "boolean",
                        "default": False,
                        "description": "Require exact (case-insensitive) name match.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=_tool_afs_codebase_symbols,
        ),
        MCPToolDefinition(
            name="afs.codebase.index",
            description=(
                "Build or update the AST symbol index for a project. "
                "Run this once after checkout or when files change. "
                "Results are used by afs.codebase.symbols and the symbol leg of afs.search."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {
                        "type": "string",
                        "description": "Context path (uses default if omitted).",
                    },
                    "path": {
                        "type": "string",
                        "description": "Project root to index (inferred from context if omitted).",
                    },
                    "max_files": {
                        "type": "integer",
                        "default": 5000,
                        "description": "Max files to index.",
                    },
                    "incremental": {
                        "type": "boolean",
                        "default": True,
                        "description": "Skip unchanged files.",
                    },
                    "languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Limit to specific languages (python, typescript, javascript, rust, go). All supported languages if omitted.",
                    },
                },
                "additionalProperties": False,
            },
            handler=_tool_afs_codebase_index,
        ),
        MCPToolDefinition(
            name="handoff.create",
            description="Create an immutable, readable handoff revision in the current scope.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "title": {"type": "string", "description": "Readable handoff title."},
                    "agent_name": {"type": "string", "description": "Agent creating the handoff."},
                    "accomplished": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What got done.",
                    },
                    "blocked": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What's stuck.",
                    },
                    "next_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Recommended actions.",
                    },
                    "context_snapshot": {
                        "type": "object",
                        "description": "Scratchpad state, open files, etc.",
                    },
                    "open_tasks": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Open task dicts.",
                    },
                    "metadata": {"type": "object", "description": "Freeform metadata."},
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID (auto-generated if omitted).",
                    },
                    "target_agent": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "critical"],
                        "default": "normal",
                    },
                },
                "required": ["agent_name"],
                "additionalProperties": False,
            },
            handler=_tool_handoff_create,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="handoff.revise",
            description="Create a new immutable revision that supersedes an earlier handoff revision.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "revision_id": {"type": "string"},
                    "title": {"type": "string"},
                    "agent_name": {"type": "string"},
                    "accomplished": {"type": "array", "items": {"type": "string"}},
                    "blocked": {"type": "array", "items": {"type": "string"}},
                    "next_steps": {"type": "array", "items": {"type": "string"}},
                    "context_snapshot": {"type": "object"},
                    "open_tasks": {"type": "array", "items": {"type": "object"}},
                    "metadata": {"type": "object"},
                    "target_agent": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high", "critical"],
                        "default": "normal",
                    },
                },
                "required": ["revision_id", "title", "agent_name"],
                "additionalProperties": False,
            },
            handler=_tool_handoff_revise,
        ),
        MCPToolDefinition(
            name="handoff.read",
            description="Read a handoff packet. Returns the latest if no session_id is given.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to read (latest if omitted).",
                    },
                },
                "additionalProperties": False,
            },
            handler=_tool_handoff_read,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="handoff.list",
            description="List recent handoff packets.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Max packets to return.",
                    },
                },
                "additionalProperties": False,
            },
            handler=_tool_handoff_list,
            catalog="slim",
        ),
        MCPToolDefinition(
            name="handoff.threads",
            description="List readable handoff threads and their latest revision state.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "limit": {"type": "integer", "default": 20},
                },
                "additionalProperties": False,
            },
            handler=_tool_handoff_threads,
        ),
        MCPToolDefinition(
            name="handoff.ack",
            description="Acknowledge a handoff revision without modifying its immutable content.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "revision_id": {"type": "string"},
                    "by": {"type": "string"},
                },
                "required": ["revision_id", "by"],
                "additionalProperties": False,
            },
            handler=_tool_handoff_ack,
        ),
        MCPToolDefinition(
            name="handoff.close",
            description="Close a handoff thread via separate lifecycle state.",
            input_schema={
                "type": "object",
                "properties": {
                    **_mcp_scope_properties(),
                    "identifier": {"type": "string"},
                    "by": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["identifier", "by"],
                "additionalProperties": False,
            },
            handler=_tool_handoff_close,
        ),
        MCPToolDefinition(
            name="hivemind.cleanup",
            description="Clean up old hivemind messages. Returns per-agent removal stats.",
            input_schema={
                "type": "object",
                "properties": {
                    "context_path": {"type": "string", "description": "Context path."},
                    "max_age_hours": {
                        "type": "integer",
                        "default": 24,
                        "description": "Max message age in hours.",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Preview without deleting.",
                    },
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
                    **_mcp_scope_properties(),
                    "mount_type": {"type": "string", "description": "Filter by mount type."},
                    "decay_hours": {
                        "type": "number",
                        "default": 168.0,
                        "description": "Decay window in hours.",
                    },
                    "threshold": {
                        "type": "number",
                        "default": 0.0,
                        "description": "Min score threshold.",
                    },
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
                    "since": {
                        "type": "string",
                        "description": "Filter events after this datetime.",
                    },
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
            module_name.split(".")[:index] for index in range(1, len(module_name.split(".")) + 1)
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
    return cast(
        dict[str, Any],
        _invoke_extension_callable(
            handler,
            manager=manager,
            arguments=arguments,
            fallback_roles=["arguments", "manager"],
        ),
    )


def _invoke_resource_handler(
    handler: Callable[..., dict[str, Any]],
    uri: str,
    manager: AFSManager,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        _invoke_extension_callable(
            handler,
            manager=manager,
            uri=uri,
            fallback_roles=["uri", "manager"],
        ),
    )


def _invoke_prompt_handler(
    handler: Callable[..., list[dict[str, Any]]],
    arguments: dict[str, Any],
    manager: AFSManager,
) -> list[dict[str, Any]]:
    return cast(
        list[dict[str, Any]],
        _invoke_extension_callable(
            handler,
            manager=manager,
            arguments=arguments,
            fallback_roles=["arguments", "manager"],
        ),
    )


def _normalize_tool_catalog(
    value: Any,
    *,
    default: str = "full",
    allow_inherit: bool = True,
) -> str:
    if default not in {"full", "slim"}:
        raise ValueError("default MCP tool catalog must be 'full' or 'slim'")
    if value is None or value == "":
        if allow_inherit:
            return default
        raise ValueError("MCP tool catalog must be 'full' or 'slim'")
    if not isinstance(value, str):
        raise ValueError("MCP tool catalog must be 'full' or 'slim'")
    normalized = value.strip().lower()
    if normalized not in {"full", "slim"}:
        raise ValueError("MCP tool catalog must be 'full' or 'slim'")
    return normalized


def _normalize_extension_tools(
    extension_name: str,
    definitions: Any,
    *,
    source: str,
    default_catalog: str = "full",
) -> list[MCPToolDefinition]:
    default_catalog = _normalize_tool_catalog(default_catalog, allow_inherit=False)
    payloads: list[Any]
    if definitions is None:
        return []
    if isinstance(definitions, MCPToolDefinition):
        payloads = [definitions]
    elif isinstance(definitions, dict):
        payloads = [definitions]
    elif isinstance(definitions, (list, tuple)):
        payloads = list(definitions)
    else:
        raise TypeError(f"Extension {extension_name} must return list[dict] from mcp tool factory")

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
                replace(
                    payload,
                    name=payload.name.strip(),
                    description=payload.description.strip(),
                    handler=_wrapped,
                    source=source,
                    catalog=_normalize_tool_catalog(
                        payload.catalog,
                        default=default_catalog,
                    ),
                )
            )
            continue

        if not isinstance(payload, dict):
            raise TypeError(f"Extension {extension_name} returned non-dict MCP tool payload")
        name = payload.get("name")
        description = payload.get("description")
        input_schema = payload.get("inputSchema", payload.get("input_schema"))
        raw_handler = payload.get("handler")
        catalog = (
            _normalize_tool_catalog(
                payload["catalog"],
                default=default_catalog,
                allow_inherit=False,
            )
            if "catalog" in payload
            else default_catalog
        )

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
        if not callable(raw_handler):
            raise ValueError(f"Extension {extension_name} tool '{name}' missing callable handler")

        def _wrapped(
            arguments: dict[str, Any],
            manager: AFSManager,
            _handler=raw_handler,
        ) -> dict[str, Any]:
            return _invoke_tool_handler(_handler, arguments, manager)

        tools.append(
            MCPToolDefinition(
                name=name.strip(),
                description=description.strip(),
                input_schema=input_schema,
                handler=_wrapped,
                source=source,
                catalog=catalog,
            )
        )
    return tools


def _normalize_extension_resources(
    extension_name: str,
    definitions: Any,
    *,
    source: str,
) -> list[MCPResourceDefinition]:
    payloads: list[Any]
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
            raise TypeError(f"Extension {extension_name} returned non-dict MCP resource payload")

        raw_uri = payload.get("uri")
        name = payload.get("name")
        description = payload.get("description")
        mime_type = payload.get("mimeType", payload.get("mime_type"))
        raw_handler = payload.get("handler", payload.get("reader"))

        if not isinstance(raw_uri, str) or not raw_uri.strip():
            raise ValueError(f"Extension {extension_name} returned resource without valid uri")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Extension {extension_name} returned resource '{raw_uri}' without valid name"
            )
        if not isinstance(description, str):
            description = ""
        if not isinstance(mime_type, str) or not mime_type.strip():
            mime_type = "application/json"
        if not callable(raw_handler):
            raise ValueError(
                f"Extension {extension_name} resource '{raw_uri}' missing callable handler"
            )
        normalized_uri = raw_uri.strip()

        def _wrapped(
            resource_uri: str,
            manager: AFSManager,
            _handler=raw_handler,
            _mime_type=mime_type,
            _uri=normalized_uri,
        ) -> dict[str, Any]:
            result = _invoke_resource_handler(_handler, resource_uri, manager)
            return _coerce_resource_result(result, uri=_uri, mime_type=_mime_type)

        resources.append(
            MCPResourceDefinition(
                uri=normalized_uri,
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
    payloads: list[Any]
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

            def _wrapped_prompt_definition(
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
                    handler=_wrapped_prompt_definition,
                    source=source,
                )
            )
            continue

        if not isinstance(payload, dict):
            raise TypeError(f"Extension {extension_name} returned non-dict MCP prompt payload")

        name = payload.get("name")
        description = payload.get("description")
        arguments = payload.get("arguments")
        raw_handler = payload.get("handler", payload.get("get_messages"))

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Extension {extension_name} returned prompt without valid name")
        if not isinstance(description, str):
            description = ""
        if not callable(raw_handler):
            raise ValueError(f"Extension {extension_name} prompt '{name}' missing callable handler")

        def _wrapped_prompt_payload(
            prompt_arguments: dict[str, Any],
            manager: AFSManager,
            _handler=raw_handler,
        ) -> list[dict[str, Any]]:
            result = _invoke_prompt_handler(_handler, prompt_arguments, manager)
            return _coerce_prompt_result(result)

        prompts.append(
            MCPPromptDefinition(
                name=name.strip(),
                description=description.strip(),
                arguments=_normalize_prompt_arguments(arguments),
                handler=_wrapped_prompt_payload,
                source=source,
            )
        )
    return prompts


def _normalize_extension_contribution(
    extension_name: str,
    definitions: Any,
    *,
    source: str,
    default_catalog: str = "full",
) -> MCPExtensionContribution:
    if definitions is None:
        return MCPExtensionContribution()

    if isinstance(definitions, MCPExtensionContribution):
        return MCPExtensionContribution(
            tools=_normalize_extension_tools(
                extension_name,
                definitions.tools,
                source=source,
                default_catalog=default_catalog,
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
                default_catalog=default_catalog,
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
        tools=_normalize_extension_tools(
            extension_name,
            definitions,
            source=source,
            default_catalog=default_catalog,
        )
    )


def _load_extension_surface(
    manager: AFSManager,
    *,
    extension_name: str,
    extension_roots: list[Path],
    surface: str,
    module_name: str,
    factory_name: str,
    default_catalog: str = "full",
) -> tuple[MCPExtensionContribution, ExtensionMCPStatus]:
    source = f"extension:{extension_name}"
    search_roots: list[Path] = []
    for root in extension_roots:
        search_roots.extend([root, root.parent])
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
        except Exception as exc:  # noqa: BLE001 - extension import boundary
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
                default_catalog=default_catalog,
            )
        except Exception as exc:  # noqa: BLE001 - extension factory boundary
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
                manifest.mcp_tools_catalog,
            ),
            (
                "mcp_server",
                manifest.mcp_server_module.strip(),
                manifest.mcp_server_factory.strip() or "register_mcp_server",
                "full",
            ),
        ]
        for surface, module_name, factory_name, default_catalog in surfaces:
            if not module_name:
                continue
            contribution, status = _load_extension_surface(
                manager,
                extension_name=extension_name,
                extension_roots=manifest.import_roots,
                surface=surface,
                module_name=module_name,
                factory_name=factory_name,
                default_catalog=default_catalog,
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
        except Exception as exc:  # noqa: BLE001 - profile plugin import boundary
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
        except Exception as exc:  # noqa: BLE001 - profile plugin factory boundary
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
            tool = replace(
                tool,
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
        tools = _builtin_tool_definitions()
        specs = [tool.to_spec() for tool in tools]
        tool_names = [tool.name for tool in tools]
        core_names = frozenset(tool_names)
        slim_names = frozenset(tool.name for tool in tools if tool.catalog == "slim")
    else:
        specs = registry.specs()
        tool_names = list(registry.tools)
        core_names = frozenset(
            name for name, tool in registry.tools.items() if tool.source == "core"
        )
        slim_names = frozenset(
            name for name, tool in registry.tools.items() if tool.catalog == "slim"
        )
    filtered = [
        spec for spec in specs if _should_list_tool(str(spec["name"]), slim_names=slim_names)
    ]
    return _client_tool_specs(filtered, tool_names, core_names=core_names)


def _safe_mcp_tool_names_enabled() -> bool:
    style = os.environ.get(MCP_TOOL_NAME_STYLE_ENV, "").strip().lower()
    return style in SAFE_MCP_TOOL_NAME_STYLE_VALUES


def _safe_mcp_tool_name(name: str) -> str:
    if MCP_TOOL_NAME_PATTERN.fullmatch(name):
        return name
    safe_name = MCP_TOOL_NAME_UNSAFE_CHARS.sub("_", name).strip("_")
    if not safe_name:
        safe_name = "tool"
    if len(safe_name) > 64:
        safe_name = safe_name[:64].rstrip("_") or "tool"
    return safe_name


def _tool_name_alias_maps(
    tool_names: Iterable[str],
    *,
    core_names: frozenset[str] = frozenset(),
) -> tuple[dict[str, str], dict[str, str]]:
    original_to_client: dict[str, str] = {}
    client_to_original: dict[str, str] = {}
    used: set[str] = set()

    # Reserve stable aliases for core tools before an extension can claim the
    # sanitized spelling. Within each source tier, preserve already-valid names.
    ordered = sorted(
        tool_names,
        key=lambda name: (
            name not in core_names,
            not MCP_TOOL_NAME_PATTERN.fullmatch(name),
            name,
        ),
    )
    for original in ordered:
        alias = _safe_mcp_tool_name(original)
        candidate = alias
        suffix_index = 2
        while candidate in used:
            suffix = f"_{suffix_index}"
            base = alias[: 64 - len(suffix)].rstrip("_") or "tool"
            candidate = f"{base}{suffix}"
            suffix_index += 1
        used.add(candidate)
        original_to_client[original] = candidate
        client_to_original[candidate] = original

    return original_to_client, client_to_original


def _client_tool_specs(
    specs: list[dict[str, Any]],
    tool_names: Iterable[str],
    *,
    core_names: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    if not _safe_mcp_tool_names_enabled():
        return specs

    original_to_client, _ = _tool_name_alias_maps(
        tool_names,
        core_names=core_names,
    )
    client_specs: list[dict[str, Any]] = []
    for spec in specs:
        original_name = str(spec["name"])
        client_name = original_to_client.get(original_name, _safe_mcp_tool_name(original_name))
        client_spec = dict(spec)
        client_spec["name"] = client_name
        client_specs.append(client_spec)
    return client_specs


def _server_tool_name(client_name: str, registry: MCPToolRegistry) -> str:
    if not _safe_mcp_tool_names_enabled():
        return client_name
    core_names = frozenset(name for name, tool in registry.tools.items() if tool.source == "core")
    _, client_to_original = _tool_name_alias_maps(
        registry.tools,
        core_names=core_names,
    )
    return client_to_original.get(client_name, client_name)


def _should_list_tool(
    tool_name: str,
    *,
    slim_names: frozenset[str] = frozenset(),
) -> bool:
    """Return whether *tool_name* belongs in MCP ``tools/list``.

    AFS has more operational tools than a model should see by default.  Keep
    call-time permissions separate from discoverability: explicit
    ``AFS_ALLOWED_TOOLS`` / ``AFS_TOOL_PROFILE`` still constrain both listing
    and calls, while the default catalog only narrows ``tools/list``. Tool
    definitions can opt into that default through *slim_names*.
    """
    if not is_tool_allowed(tool_name):
        return False

    if allowed_tools() is not None:
        return True

    catalog = os.environ.get(MCP_TOOL_CATALOG_ENV, "slim").strip().lower() or "slim"
    if catalog in FULL_MCP_TOOL_CATALOG_VALUES:
        return True
    return tool_name in DEFAULT_MCP_TOOL_CATALOG or tool_name in slim_names


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
        resources.append(
            {
                "uri": f"{ctx_uri}/bootstrap",
                "name": f"{ctx.project_name} bootstrap",
                "description": f"Session bootstrap packet for {ctx.project_name}",
                "mimeType": "application/json",
            }
        )
        resources.append(
            {
                "uri": f"{ctx_uri}/metadata",
                "name": f"{ctx.project_name} metadata",
                "description": f"Project metadata for {ctx.project_name}",
                "mimeType": "application/json",
            }
        )
        resources.append(
            {
                "uri": f"{ctx_uri}/mounts",
                "name": f"{ctx.project_name} mounts",
                "description": f"Mount listing for {ctx.project_name}",
                "mimeType": "application/json",
            }
        )
        resources.append(
            {
                "uri": f"{ctx_uri}/index",
                "name": f"{ctx.project_name} index",
                "description": f"Index summary for {ctx.project_name}",
                "mimeType": "application/json",
            }
        )
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
        contexts_data = [
            {
                "project": ctx.project_name,
                "path": str(ctx.path),
                "valid": ctx.is_valid,
                "mounts": ctx.total_mounts,
            }
            for ctx in _discover_allowed_contexts(manager)
        ]
        return {
            "uri": uri,
            "mimeType": "application/json",
            "text": json.dumps(contexts_data),
        }

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

    remainder = uri[len(prefix) :]
    if remainder.endswith("/metadata"):
        context_path_str = remainder[: -len("/metadata")]
        context_path = _resolve_explicit_allowed_context_path(context_path_str, manager)
        if detect_layout_version(context_path) == LAYOUT_VERSION:
            metadata = load_context_metadata(context_path)
            if metadata is None:  # pragma: no cover - v2 loader supplies defaults
                raise FileNotFoundError(f"metadata not found: {context_path}")
            return {
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(metadata.to_dict()),
            }
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
        stale = index.needs_health_refresh() if has else False
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
            "description": "Build a scoped session-start packet with health, notes, tasks, messages, handoffs, and durable memory. Call this first in a new session.",
            "arguments": [
                {
                    "name": "context_path",
                    "description": "Path to .context root (uses configured default if omitted)",
                    "required": False,
                },
                {
                    "name": "project_path",
                    "description": "Registered current project path used to authorize its v2 scope.",
                    "required": False,
                },
                {
                    "name": "task_limit",
                    "description": "Maximum queued tasks to include (default 10)",
                    "required": False,
                },
                {
                    "name": "message_limit",
                    "description": "Maximum scoped messages to include (default 10)",
                    "required": False,
                },
                {
                    "name": "skills_prompt",
                    "description": "Optional task prompt used to match bounded skill bodies.",
                    "required": False,
                },
                {
                    "name": "skills_top_k",
                    "description": "Maximum skill matches to retain, capped at 10 (default 5).",
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
                    "name": "project_path",
                    "description": "Registered project path used to select the current project scope.",
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
                {
                    "name": "semantic",
                    "description": "Explicitly permit remote query embeddings.",
                    "required": False,
                },
            ],
        },
        {
            "name": "afs.context.overview",
            "description": "Describe the AFS context structure plus a cheap project codebase summary",
            "arguments": [
                {
                    "name": "context_path",
                    "description": "Path to .context root (uses default if omitted)",
                    "required": False,
                },
                {
                    "name": "path",
                    "description": "Project path to summarize when no .context exists yet.",
                    "required": False,
                },
                {
                    "name": "project_path",
                    "description": "Registered project path to summarize from a central v2 context.",
                    "required": False,
                },
                {
                    "name": "scope_id",
                    "description": "Optional authorized scope: common or project:<id>.",
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
                    "name": "project_path",
                    "description": "Registered project path used to authorize its current scope.",
                    "required": False,
                },
                {
                    "name": "scope_id",
                    "description": "Optional authorized scope: common or project:<id>.",
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
                {
                    "name": "semantic",
                    "description": "Explicitly permit remote query embeddings.",
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
                    "name": "project_path",
                    "description": "Registered project path used to authorize its current scope.",
                    "required": False,
                },
                {
                    "name": "scope_id",
                    "description": "Optional authorized scope: common or project:<id>.",
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
                {
                    "name": "project_path",
                    "description": "Registered project path used to authorize its v2 scope.",
                    "required": False,
                },
            ],
        },
        {
            "name": "afs.personal.load",
            "description": (
                "Load personal context for a personalized conversation. Opt-in: "
                "requires an explicit mode declared in the manifest.toml of the "
                "personal context root. The default root is "
                "$AFS_PERSONAL_CONTEXT_ROOT or ~/.config/afs/personal."
            ),
            "arguments": [
                {
                    "name": "mode",
                    "description": (
                        "Conversation mode declared in manifest.toml (e.g. "
                        "claudia, advice, checkin)."
                    ),
                    "required": True,
                },
                {
                    "name": "context_root",
                    "description": (
                        "Override personal context root. Defaults to "
                        "$AFS_PERSONAL_CONTEXT_ROOT or ~/.config/afs/personal."
                    ),
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
        context_path, scoped = _resolve_mcp_scope(arguments, manager)
        payload = build_session_bootstrap(
            manager,
            context_path,
            project_path=scoped.requester_path,
            task_limit=_coerce_int(arguments.get("task_limit"), default=10, minimum=1, maximum=100),
            message_limit=_coerce_int(
                arguments.get("message_limit"), default=10, minimum=1, maximum=100
            ),
            skills_prompt=str(arguments.get("skills_prompt", "") or ""),
            skills_top_k=_coerce_int(
                arguments.get("skills_top_k"),
                default=5,
                minimum=0,
                maximum=10,
            ),
        )
        text = render_session_bootstrap(payload)
        return [{"role": "user", "content": {"type": "text", "text": text}}]

    if name == "afs.session.pack":
        context_path, scoped = _resolve_mcp_scope(arguments, manager)
        payload = build_context_pack(
            manager,
            context_path,
            project_path=scoped.requester_path,
            scope_id=scoped.scope_id,
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
            semantic=_coerce_bool(arguments.get("semantic", False)),
        )
        text = render_context_pack(payload)
        return [{"role": "user", "content": {"type": "text", "text": text}}]

    if name == "afs.context.overview":
        explicit_context = (
            isinstance(arguments.get("context_path"), str)
            and str(arguments.get("context_path")).strip()
        )
        explicit_project = any(
            isinstance(arguments.get(key), str) and str(arguments.get(key)).strip()
            for key in ("path", "project_path")
        )
        context_available = True
        ctx_root = None
        overview_context_path: Path | None = None
        overview_project_path: Path | None
        common_v2 = False
        scoped_project_name = ""
        overview_scope: ResolvedScope | None = None
        if explicit_context:
            overview_context_path = _resolve_prompt_context_path(arguments, manager)
            ctx_root = manager.list_context(context_path=overview_context_path)
            overview_project_path = overview_context_path.parent
        elif explicit_project:
            resolved_project_path = _resolve_project_path(arguments)
            if resolved_project_path is None:  # pragma: no cover - guarded above
                raise ValueError("project path is required")
            overview_project_path = _assert_allowed(resolved_project_path, manager)
            try:
                overview_context_path = _resolve_context_path(arguments, manager)
                ctx_root = manager.list_context(context_path=overview_context_path)
            except FileNotFoundError:
                context_available = False
                ctx_root = None
                overview_context_path = None
        else:
            overview_context_path = _resolve_prompt_context_path(arguments, manager)
            ctx_root = manager.list_context(context_path=overview_context_path)
            overview_project_path = overview_context_path.parent
        if (
            ctx_root is not None
            and overview_context_path is not None
            and detect_layout_version(overview_context_path) == LAYOUT_VERSION
        ):
            scope_arguments = dict(arguments)
            scope_arguments["context_path"] = str(overview_context_path)
            if not scope_arguments.get("project_path") and explicit_project:
                scope_arguments["project_path"] = str(overview_project_path)
            _scope_root, overview_scope = _resolve_mcp_scope(scope_arguments, manager)
            if overview_scope.requester_path is not None:
                overview_project_path = overview_scope.requester_path
                scoped_project_name = overview_scope.project_name
            else:
                common_v2 = True
                overview_project_path = None

        if common_v2:
            codebase = {
                "project_root": "(common scope; no project selected)",
                "scan": {
                    "files_scanned": 0,
                    "max_scan_depth": 0,
                    "truncated": False,
                },
            }
            project_name = COMMON_SCOPE_ID
        else:
            if overview_scope is not None and overview_scope.requester_path is not None:
                codebase_target = overview_scope.requester_path
                codebase = build_scoped_codebase_summary(
                    overview_scope.context_root,
                    codebase_target,
                    project_id=overview_scope.project_id,
                )
            else:
                fallback_codebase_target = (
                    overview_project_path
                    if explicit_project or ctx_root is None
                    else overview_context_path
                )
                if fallback_codebase_target is None:  # pragma: no cover - guarded by scope state
                    raise FileNotFoundError("No project path is available for context overview")
                codebase = build_codebase_summary(fallback_codebase_target)
            fallback_project_name = (
                overview_project_path.name
                if overview_project_path is not None
                else COMMON_SCOPE_ID
            )
            project_name = scoped_project_name or (
                fallback_project_name
                if explicit_project or ctx_root is None
                else ctx_root.project_name
            )
        lines = [
            f"# AFS Context: {project_name}",
            f"Context available: {'yes' if context_available else 'no'}",
            "Project path: "
            f"{overview_project_path or '(common scope; no project selected)'}",
        ]
        if ctx_root is not None:
            lines.extend(
                [
                    f"Path: {ctx_root.path}",
                    f"Valid: {ctx_root.is_valid}",
                    f"Total mounts: {ctx_root.total_mounts}",
                    "",
                    "## Mounts",
                ]
            )
            if not common_v2 and project_name != ctx_root.project_name:
                lines.append(f"Nearest context project: {ctx_root.project_name}")
            for mount_type, mount_list in ctx_root.mounts.items():
                lines.append(f"### {mount_type.value}")
                if mount_list:
                    for m in mount_list:
                        lines.append(f"  - {m.name} → {m.source} (symlink={m.is_symlink})")
                else:
                    lines.append("  (empty)")
        else:
            lines.append("Valid: false")
            lines.append("Total mounts: 0")
            lines.append("")
            lines.append("## Mounts")
            lines.append("  (no context yet)")
        if ctx_root is not None and ctx_root.metadata:
            lines.append("")
            lines.append("## Metadata")
            lines.append(f"Description: {ctx_root.metadata.description or '(none)'}")
            lines.append(f"Agents: {', '.join(ctx_root.metadata.agents) or '(none)'}")
            if ctx_root.metadata.manual_only:
                lines.append(f"Protected paths: {', '.join(ctx_root.metadata.manual_only)}")
        lines.append("")
        lines.append("## Codebase")
        lines.extend(render_codebase_summary(codebase).splitlines())
        return [{"role": "user", "content": {"type": "text", "text": "\n".join(lines)}}]

    if name == "afs.workflow.structured":
        schema_names = set(list_response_schema_names())
        schema_name = str(arguments.get("schema_name", "plan") or "plan").strip()
        if schema_name not in schema_names:
            raise ValueError("schema_name must be one of: " + ", ".join(sorted(schema_names)))
        task = arguments.get("task", "")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task argument is required")
        context_path, scoped = _resolve_mcp_scope(arguments, manager)
        payload = build_context_pack(
            manager,
            context_path,
            project_path=scoped.requester_path,
            scope_id=scoped.scope_id,
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
            semantic=_coerce_bool(arguments.get("semantic", False)),
        )
        schema = get_response_schema(schema_name)
        if detect_layout_version(context_path) == LAYOUT_VERSION:
            policy_root = scoped.requester_path or context_path
        else:
            policy_root = Path(context_path).parent
        policy = load_repo_policy(start_dir=policy_root)
        policy_summary = evaluate_repo_policy(
            policy,
            repo_root=policy_root,
            changed_paths=[],
        )
        lines = [
            "# AFS Structured Workflow Prompt",
            "",
            "Return only JSON matching the response schema below.",
            "Do not wrap the response in markdown fences.",
            "Keep every field grounded in the supplied context. If evidence is missing, keep the answer minimal and note the uncertainty in fields that allow it.",
            (
                "Before returning, self-check your JSON: it is validated with "
                f"`afs schema validate --schema {schema_name}` (exit 1 lists the exact "
                "violations). Fix every reported violation before finalizing."
            ),
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
        if policy_summary.get("available"):
            review_focus = (
                policy_summary.get("review_focus")
                if isinstance(policy_summary.get("review_focus"), list)
                else []
            )
            design_constraints = (
                policy_summary.get("design_constraints")
                if isinstance(policy_summary.get("design_constraints"), list)
                else []
            )
            planning_principles = (
                policy_summary.get("planning_principles")
                if isinstance(policy_summary.get("planning_principles"), list)
                else []
            )
            if review_focus or design_constraints or planning_principles:
                policy_lines = ["", "## Repo Policy"]
                if review_focus:
                    policy_lines.append("Review focus:")
                    policy_lines.extend(f"- {item}" for item in review_focus[:6])
                if design_constraints:
                    policy_lines.append("Design constraints:")
                    policy_lines.extend(f"- {item}" for item in design_constraints[:6])
                if planning_principles:
                    policy_lines.append("Planning principles:")
                    policy_lines.extend(f"- {item}" for item in planning_principles[:6])
                lines.extend(policy_lines)
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
        query_arguments = dict(arguments)
        if not any(
            isinstance(query_arguments.get(name), str)
            and str(query_arguments.get(name)).strip()
            for name in ("context_path", "project_path")
        ):
            query_arguments["context_path"] = str(manager.config.general.context_root)
        query_arguments["mount_types"] = (
            [mount_type.value for mount_type in mount_types]
            if mount_types is not None
            else None
        )
        payload = _tool_context_query(query_arguments, manager)
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
        from .session_bootstrap import _collect_scratchpad

        context_path, scoped = _resolve_mcp_scope(arguments, manager)
        scratchpad = _collect_scratchpad(
            manager,
            context_path,
            scoped=scoped,
        )
        lines = [f"# Scratchpad Review: {context_path}", ""]

        state_text = str(scratchpad.get("state_text", "")).strip()
        if state_text:
            lines.append("## State")
            lines.append(state_text)
            lines.append("")

        deferred_text = str(scratchpad.get("deferred_text", "")).strip()
        if deferred_text:
            lines.append("## Deferred")
            lines.append(deferred_text)
            lines.append("")

        other_files = scratchpad.get("other_files") or []
        if other_files:
            lines.append("## Other files")
            for name_str in sorted(str(item) for item in other_files):
                lines.append(f"- {name_str}")

        return [{"role": "user", "content": {"type": "text", "text": "\n".join(lines)}}]

    if name == "afs.personal.load":
        from .personal_context import render_personal_context

        mode_arg = arguments.get("mode", "")
        if not isinstance(mode_arg, str) or not mode_arg.strip():
            raise ValueError("mode argument is required")
        root_arg = arguments.get("context_root")
        context_root = (
            Path(str(root_arg)).expanduser()
            if isinstance(root_arg, str) and root_arg.strip()
            else None
        )
        text = render_personal_context(mode_arg.strip(), context_root=context_root)
        return [{"role": "user", "content": {"type": "text", "text": text}}]

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
        requested_protocol = params.get("protocolVersion") if isinstance(params, dict) else None
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
        registry_name = _server_tool_name(name, active_registry)

        try:
            payload = active_registry.call(registry_name, arguments, manager)
        except Exception as exc:  # noqa: BLE001 - MCP tool transport boundary
            print(f"[afs-mcp] tool error: {registry_name}: {exc}", file=sys.stderr)
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
        except Exception as exc:  # noqa: BLE001 - MCP resource transport boundary
            print(f"[afs-mcp] resource error: {uri}: {exc}", file=sys.stderr)
            return _error_response(request_id, -32000, _annotate_error(exc))
        return _success_response(request_id, {"contents": [content]})

    if method == "prompts/list":
        return _success_response(request_id, {"prompts": _list_prompts(active_registry)})

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
        except Exception as exc:  # noqa: BLE001 - MCP prompt transport boundary
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
    except Exception as exc:  # noqa: BLE001 - startup diagnostics must not block serving
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
        except Exception as exc:  # noqa: BLE001 - per-message transport boundary
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
