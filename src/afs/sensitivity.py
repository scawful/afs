"""Sensitivity rules for indexing, embedding, and export surfaces."""

from __future__ import annotations

import fnmatch
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path


def matches_path_rules(
    absolute_path: Path,
    *,
    relative_path: str = "",
    patterns: Iterable[str],
) -> bool:
    """Return True when any configured glob pattern matches the path."""
    try:
        absolute = str(absolute_path.expanduser().resolve())
    except OSError:
        absolute = str(absolute_path.expanduser())
    relative_value = relative_path.strip().replace("\\", "/").lstrip("./")
    basename = absolute_path.name

    for raw_pattern in patterns:
        pattern = str(raw_pattern).strip()
        if not pattern:
            continue
        expanded = os.path.expanduser(pattern)
        if os.path.isabs(expanded) or pattern.startswith("~"):
            if fnmatch.fnmatch(absolute, str(Path(expanded))):
                return True
            continue
        if relative_value and fnmatch.fnmatch(relative_value, pattern):
            return True
        if basename and fnmatch.fnmatch(basename, pattern):
            return True
    return False


@contextmanager
def filtered_tree_copy(
    source_root: Path,
    patterns: Iterable[str],
) -> Iterator[Path]:
    """Yield a filtered copy of a tree with matching files omitted."""
    patterns_list = [str(pattern) for pattern in patterns if str(pattern).strip()]
    if not patterns_list:
        yield source_root
        return

    with tempfile.TemporaryDirectory(prefix="afs-filtered-tree-") as temp_dir:
        filtered_root = Path(temp_dir) / source_root.name
        filtered_root.mkdir(parents=True, exist_ok=True)
        for candidate in source_root.rglob("*"):
            if candidate.is_dir():
                continue
            relative = candidate.relative_to(source_root).as_posix()
            if matches_path_rules(
                candidate,
                relative_path=relative,
                patterns=patterns_list,
            ):
                continue
            destination = filtered_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
        yield filtered_root
