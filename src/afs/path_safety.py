"""Filesystem path validation helpers for trusted AFS namespaces."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from pathlib import Path


def lexical_absolute(path: Path) -> Path:
    """Return an absolute path without following filesystem links."""

    return Path(os.path.abspath(path.expanduser()))


def is_linklike(path_stat: os.stat_result) -> bool:
    """Return whether an lstat result is a symlink or Windows reparse point."""

    if stat.S_ISLNK(path_stat.st_mode):
        return True
    attributes = getattr(path_stat, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def assert_no_linklike_components(
    path: Path,
    *,
    boundary: Path | None = None,
    allow_missing: bool = True,
) -> Path:
    """Validate a lexical path without following symlinks or reparse points.

    When *boundary* is supplied, only components below that already-trusted
    directory are inspected and lexical containment is required. Missing
    components are allowed by default so callers can validate a destination
    before creating it.
    """

    candidate = lexical_absolute(path)
    if boundary is None:
        trusted = Path(candidate.anchor)
        parts = candidate.parts[1:]
    else:
        trusted = lexical_absolute(boundary)
        try:
            relative = candidate.relative_to(trusted)
        except ValueError as exc:
            raise ValueError("path escapes its trusted boundary") from exc
        parts = () if relative == Path(".") else relative.parts

    current = trusted
    for part in parts:
        current /= part
        try:
            path_stat = os.lstat(current)
        except FileNotFoundError:
            if allow_missing:
                continue
            raise
        if is_linklike(path_stat):
            raise ValueError(f"path contains a symbolic link or reparse point: {current}")
    return candidate


def iter_regular_files_no_links(
    root: Path,
    *,
    skip_names: frozenset[str] = frozenset(),
) -> Iterator[Path]:
    """Yield regular descendants without following link-like components."""

    try:
        children = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError:
        return
    for child in children:
        if child.name in skip_names:
            continue
        try:
            child_stat = os.lstat(child)
        except OSError:
            continue
        if is_linklike(child_stat):
            continue
        if stat.S_ISREG(child_stat.st_mode):
            yield child
        elif stat.S_ISDIR(child_stat.st_mode):
            yield from iter_regular_files_no_links(
                child,
                skip_names=skip_names,
            )
