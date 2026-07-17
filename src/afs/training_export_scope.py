"""Authorization and staging helpers for memory training exports."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .context_layout import LAYOUT_VERSION, detect_layout_version
from .context_paths import resolve_mount_root
from .models import MountType
from .path_safety import iter_regular_files_no_links
from .schema import AFSConfig
from .scopes import resolve_scope, visible_mount_roots
from .sensitivity import filtered_tree_copy, matches_path_rules


@dataclass(frozen=True)
class MemoryExportTree:
    """One authorized input tree prepared for the generic memory exporter."""

    export_root: Path
    memory_root: Path
    source_roots: tuple[Path, ...]
    scope_id: str
    project_id: str
    explicit_override: bool = False


@contextmanager
def resolve_memory_export_tree(
    context_root: Path,
    *,
    config: AFSConfig,
    requester_path: Path,
    memory_root_override: Path | None = None,
    sensitivity_patterns: Iterable[str] = (),
) -> Iterator[MemoryExportTree]:
    """Yield an export tree without widening a v2 requester's memory scope.

    An explicit memory-root is an administrative escape hatch and therefore
    retains the legacy direct-tree behavior. Version 1 contexts also remain
    unscoped. For version 2, the requester must resolve to a registered
    project and only that project's memory plus common memory is staged.
    """

    patterns = tuple(str(pattern) for pattern in sensitivity_patterns if str(pattern).strip())
    if memory_root_override is not None:
        memory_root = memory_root_override.expanduser().resolve()
        with filtered_tree_copy(memory_root, patterns) as export_root:
            yield MemoryExportTree(
                export_root=export_root,
                memory_root=memory_root,
                source_roots=(memory_root,),
                scope_id="admin-override",
                project_id="",
                explicit_override=True,
            )
        return

    root = context_root.expanduser().resolve()
    memory_root = resolve_mount_root(root, MountType.MEMORY, config=config)
    if detect_layout_version(root) != LAYOUT_VERSION:
        with filtered_tree_copy(memory_root, patterns) as export_root:
            yield MemoryExportTree(
                export_root=export_root,
                memory_root=memory_root,
                source_roots=(memory_root,),
                scope_id="common",
                project_id="",
            )
        return

    scoped = resolve_scope(root, requester_path=requester_path)
    source_roots = visible_mount_roots(
        memory_root,
        mount_type=MountType.MEMORY,
        scoped=scoped,
    )
    with tempfile.TemporaryDirectory(prefix="afs-memory-export-") as temp_dir:
        export_root = Path(temp_dir) / "memory"
        export_root.mkdir(parents=True)
        for source_root in source_roots:
            if not source_root.is_dir():
                continue
            for candidate in iter_regular_files_no_links(source_root):
                relative = candidate.relative_to(memory_root)
                if matches_path_rules(
                    candidate,
                    relative_path=relative.as_posix(),
                    patterns=patterns,
                ):
                    continue
                destination = export_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, destination)
        yield MemoryExportTree(
            export_root=export_root,
            memory_root=memory_root,
            source_roots=source_roots,
            scope_id=scoped.scope_id,
            project_id=scoped.project_id,
        )


__all__ = ["MemoryExportTree", "resolve_memory_export_tree"]
