"""AFS discovery helpers for locating .context roots."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from .config import load_config_model
from .manager import AFSManager
from .models import ContextRoot
from .schema import AFSConfig


def discover_contexts(
    search_paths: Iterable[Path] | None = None,
    *,
    max_depth: int = 3,
    ignore_names: Iterable[str] | None = None,
    config: AFSConfig | None = None,
) -> list[ContextRoot]:
    config = config or load_config_model()
    manager = AFSManager(config=config)
    roots = _resolve_search_paths(search_paths, config)
    ignore_set = _normalize_ignore_names(ignore_names, config)

    contexts: list[ContextRoot] = []
    seen: set[Path] = set()

    for root in roots:
        if root.name.lower() in ignore_set:
            continue
        for context_path in _find_context_dirs(root, max_depth, ignore_set):
            resolved = context_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                contexts.append(manager.list_context(context_path=resolved))
            except Exception:
                continue

    contexts.sort(key=lambda item: item.project_name.lower())
    return contexts


def get_project_stats(projects: list[ContextRoot]) -> dict[str, int]:
    total_mounts = 0
    mounts_by_type: dict[str, int] = {}

    for project in projects:
        for mount_type, mount_list in project.mounts.items():
            total_mounts += len(mount_list)
            mounts_by_type[mount_type.value] = (
                mounts_by_type.get(mount_type.value, 0) + len(mount_list)
            )

    return {
        "total_projects": len(projects),
        "total_mounts": total_mounts,
        **mounts_by_type,
    }


def _resolve_search_paths(
    search_paths: Iterable[Path] | None,
    config: AFSConfig,
) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()

    def _add_path(path: Path) -> None:
        try:
            resolved_path = path.expanduser().resolve()
        except OSError:
            return
        if resolved_path in seen or not resolved_path.exists():
            return
        seen.add(resolved_path)
        resolved.append(resolved_path)

    if search_paths:
        for entry in search_paths:
            _add_path(entry)
        return resolved

    for workspace in config.general.workspace_directories:
        _add_path(workspace.path)

    if config.general.agent_workspaces_dir:
        _add_path(config.general.agent_workspaces_dir)

    return resolved


def _find_context_dirs(
    root: Path,
    max_depth: int,
    ignore_names: set[str],
    current_depth: int = 0,
) -> Iterator[Path]:
    if current_depth > max_depth:
        return

    try:
        for entry in root.iterdir():
            if entry.name.lower() in ignore_names:
                continue
            if entry.name == ".context" and entry.is_dir():
                yield entry
            elif entry.is_dir() and not entry.name.startswith("."):
                yield from _find_context_dirs(
                    entry, max_depth, ignore_names, current_depth + 1
                )
    except OSError:
        return


def _normalize_ignore_names(
    ignore_names: Iterable[str] | None, config: AFSConfig
) -> set[str]:
    names: list[str] = []
    if config.general.discovery_ignore:
        names.extend(config.general.discovery_ignore)
    if ignore_names:
        names.extend(ignore_names)
    return {name.strip().lower() for name in names if name and name.strip()}
