"""Sensitivity rules for indexing, embedding, and export surfaces."""

from __future__ import annotations

import fnmatch
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# Generated or personal trees that no indexer should ever read, regardless of
# configured never_index rules: embedding payloads, backups, AFS runtime
# state, and the user's media library.
DEFAULT_INDEX_EXCLUDED_DIRS = frozenset({"embeddings", ".afs"})
DEFAULT_INDEX_EXCLUDED_GLOBS = ("*.bak*", "embedding_index.json")
DEFAULT_INDEX_EXCLUDED_ROOTS = ("~/Music",)


def is_default_index_excluded(absolute_path: Path, *, relative_path: str = "") -> bool:
    """Return True when a path falls under the built-in indexing exclusions.

    Directory names are matched against components of ``relative_path`` only,
    so a context root that itself lives under a directory named ``embeddings``
    is not excluded wholesale.
    """
    # Path().parts drops "." segments; _normalize_relative_path's lstrip("./")
    # would also eat the dot in ".afs".
    relative_parts = Path(str(relative_path).replace("\\", "/").strip()).parts
    if any(part in DEFAULT_INDEX_EXCLUDED_DIRS for part in relative_parts):
        return True
    name = absolute_path.name
    if name in DEFAULT_INDEX_EXCLUDED_DIRS:
        return True
    if any(fnmatch.fnmatch(name, pattern) for pattern in DEFAULT_INDEX_EXCLUDED_GLOBS):
        return True
    try:
        resolved = absolute_path.expanduser().resolve()
    except OSError:
        resolved = absolute_path.expanduser()
    for root in DEFAULT_INDEX_EXCLUDED_ROOTS:
        excluded_root = Path(os.path.expanduser(root)).resolve()
        if resolved == excluded_root or excluded_root in resolved.parents:
            return True
    return False


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


@dataclass(frozen=True, slots=True)
class SensitivityMatch:
    """A single sensitivity-rule match with the path spelling that matched."""

    relative_path: str
    pattern: str


@dataclass(frozen=True, slots=True)
class SensitivityRuleSet:
    """Compiled sensitivity patterns for export/read surfaces.

    Keep this as a small value object: callers provide the relevant relative
    path spellings for their surface, and the rule set owns pattern cleanup,
    absolute-path fallback, and deterministic match reporting.
    """

    patterns: tuple[str, ...] = ()

    @classmethod
    def from_patterns(cls, patterns: Iterable[str]) -> SensitivityRuleSet:
        return cls(tuple(str(pattern).strip() for pattern in patterns if str(pattern).strip()))

    @property
    def enabled(self) -> bool:
        return bool(self.patterns)

    def match(
        self,
        absolute_path: Path,
        *,
        relative_paths: Iterable[str] = (),
        include_absolute_fallback: bool = True,
    ) -> SensitivityMatch | None:
        if not self.patterns:
            return None

        resolved = absolute_path.expanduser()
        relative_values = tuple(_normalize_relative_path(value) for value in relative_paths)
        relative_values = tuple(value for value in relative_values if value)

        for pattern in self.patterns:
            for relative_path in relative_values:
                if matches_path_rules(resolved, relative_path=relative_path, patterns=(pattern,)):
                    return SensitivityMatch(relative_path=relative_path, pattern=pattern)
            if include_absolute_fallback and matches_path_rules(
                resolved,
                relative_path="",
                patterns=(pattern,),
            ):
                return SensitivityMatch(relative_path=str(resolved), pattern=pattern)
        return None

    def blocked(
        self,
        absolute_path: Path,
        *,
        relative_paths: Iterable[str] = (),
        include_absolute_fallback: bool = True,
    ) -> bool:
        return self.match(
            absolute_path,
            relative_paths=relative_paths,
            include_absolute_fallback=include_absolute_fallback,
        ) is not None


def _normalize_relative_path(value: str) -> str:
    return str(value).replace("\\", "/").strip().lstrip("./")


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
