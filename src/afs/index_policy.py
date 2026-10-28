"""Ignore-file matching and per-context-root index path policy.

``.afsignore`` lines are shared with hybrid search: blank lines and ``#``
comments are skipped, ``!`` negates, a leading ``/`` is dropped, a trailing
``/`` marks a directory, and the last matching line wins.  Patterns use
fnmatch, so ``*`` also crosses ``/``; ``**/`` also matches zero directories.
A pattern without ``/`` matches any single path component.

The context index reads two optional files at the context root.  When
neither exists, indexing behaves exactly as before.

- ``.afsignore`` is matched against ``<mount_type>/<relative_path>``, the
  same virtual path ``sensitivity.never_index`` sees.  An excluded
  directory excludes everything below it; a ``!`` line cannot re-include
  a path whose parent directory is excluded.
- ``.afsindex.toml`` holds per-mount allowlists::

      [allowlist.scratchpad]
      suffixes = [".md", ".json"]
      max_depth = 2
      max_bytes = 65536
      receipt_globs = ["RESULT*", "README*"]
      receipt_max_bytes = 262144
      skip_hash_dirs = true

  Under an allowlisted mount a file is indexed only when its suffix is
  listed and it is either a receipt (basename glob, case-insensitive) no
  larger than ``receipt_max_bytes``, or it sits at most ``max_depth``
  directories below the mount (or v2 scope) root and is no larger than
  ``max_bytes``.  Directories are kept up to ``max_depth``.
  ``skip_hash_dirs`` drops directories named with 12 or more lowercase hex
  characters and everything below them.  Omitted keys do not filter.
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import MountType
from .toml_compat import tomllib

IGNORE_FILENAME = ".afsignore"
POLICY_FILENAME = ".afsindex.toml"

IgnorePattern = tuple[str, bool]
PolicySignature = tuple[tuple[int, int] | None, tuple[int, int] | None]

_HASH_DIR_RE = re.compile(r"^[0-9a-f]{12,}$")
_ALLOWLIST_KEYS = frozenset(
    {
        "suffixes",
        "max_depth",
        "max_bytes",
        "receipt_globs",
        "receipt_max_bytes",
        "skip_hash_dirs",
    }
)
_MOUNT_TYPE_VALUES = frozenset(mount_type.value for mount_type in MountType)
_DIR_CACHE_LIMIT = 200_000


def load_ignore_file(path: Path) -> list[IgnorePattern]:
    """Parse an ignore file into ``(pattern, negated)`` pairs."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    patterns: list[IgnorePattern] = []
    for line in lines:
        pattern = line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        negated = pattern.startswith("!")
        if negated:
            pattern = pattern[1:]
        patterns.append((pattern.lstrip("/"), negated))
    return patterns


def ignore_matches(relative_path: str, patterns: Sequence[IgnorePattern]) -> bool:
    """Return whether the last pattern matching ``relative_path`` ignores it."""
    if not patterns:
        return False
    stripped = relative_path.rstrip("/")
    rel = os.path.normcase(stripped)
    parts = [os.path.normcase(part) for part in stripped.split("/") if part]
    ignored = False
    for rule in _compile_patterns(tuple(patterns)):
        if rule.matches(rel, parts):
            ignored = not rule.negated
    return ignored


@dataclass(frozen=True)
class _IgnoreRule:
    negated: bool
    part_regex: re.Pattern[str] | None
    path_regexes: tuple[re.Pattern[str], ...]
    dir_prefixes: tuple[str, ...]

    def matches(self, rel: str, parts: Sequence[str]) -> bool:
        if self.part_regex is not None:
            if any(self.part_regex.match(part) for part in parts):
                return True
        elif any(regex.match(rel) for regex in self.path_regexes):
            return True
        return any(rel.startswith(prefix) for prefix in self.dir_prefixes)


@lru_cache(maxsize=64)
def _compile_patterns(patterns: tuple[IgnorePattern, ...]) -> tuple[_IgnoreRule, ...]:
    return tuple(_compile_pattern(pattern, negated) for pattern, negated in patterns)


def _compile_pattern(pattern: str, negated: bool) -> _IgnoreRule:
    directory = pattern.endswith("/")
    candidate = pattern.rstrip("/")
    if "/" not in candidate:
        return _IgnoreRule(
            negated=negated,
            part_regex=_glob(candidate),
            path_regexes=(),
            dir_prefixes=(os.path.normcase(candidate + "/"),) if directory else (),
        )
    variants = _double_star_variants(candidate)
    return _IgnoreRule(
        negated=negated,
        part_regex=None,
        path_regexes=tuple(
            _glob(variant + suffix) for variant in variants for suffix in ("", "/**")
        ),
        dir_prefixes=(
            tuple(os.path.normcase(variant + "/") for variant in variants)
            if directory
            else ()
        ),
    )


def _glob(pattern: str) -> re.Pattern[str]:
    return re.compile(fnmatch.translate(os.path.normcase(pattern)))


def _double_star_variants(pattern: str) -> tuple[str, ...]:
    """Expand each ``**/`` into itself and its zero-directory form."""
    variants = {pattern}
    pending = [pattern]
    while pending:
        variant = pending.pop()
        collapsed: list[str] = []
        if variant.startswith("**/"):
            collapsed.append(variant[3:])
        index = variant.find("/**/")
        while index != -1:
            collapsed.append(variant[:index] + variant[index + 3 :])
            index = variant.find("/**/", index + 1)
        for candidate in collapsed:
            if candidate not in variants:
                variants.add(candidate)
                pending.append(candidate)
    return tuple(sorted(variants))


@dataclass(frozen=True)
class MountAllowlist:
    """Allowlist rules for one mount type; ``None`` means no limit."""

    suffixes: frozenset[str] | None = None
    max_depth: int | None = None
    max_bytes: int | None = None
    receipt_globs: tuple[str, ...] = ()
    receipt_max_bytes: int | None = None
    skip_hash_dirs: bool = False

    @classmethod
    def from_mapping(cls, data: Any) -> MountAllowlist:
        if not isinstance(data, dict):
            raise ValueError("must be a table")
        unknown = sorted(set(data) - _ALLOWLIST_KEYS)
        if unknown:
            raise ValueError(f"unknown keys: {', '.join(unknown)}")
        suffixes = data.get("suffixes")
        if suffixes is not None:
            suffixes = frozenset(
                _normalize_suffix(item) for item in _string_list(suffixes, "suffixes")
            )
        receipt_globs = tuple(
            item.lower() for item in _string_list(data.get("receipt_globs", []), "receipt_globs")
        )
        skip_hash_dirs = data.get("skip_hash_dirs", False)
        if not isinstance(skip_hash_dirs, bool):
            raise ValueError("skip_hash_dirs must be true or false")
        return cls(
            suffixes=suffixes,
            max_depth=_optional_count(data, "max_depth"),
            max_bytes=_optional_count(data, "max_bytes"),
            receipt_globs=receipt_globs,
            receipt_max_bytes=_optional_count(data, "receipt_max_bytes"),
            skip_hash_dirs=skip_hash_dirs,
        )

    def is_receipt(self, name: str) -> bool:
        lowered = name.lower()
        return any(fnmatch.fnmatchcase(lowered, glob) for glob in self.receipt_globs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suffixes": sorted(self.suffixes) if self.suffixes is not None else None,
            "max_depth": self.max_depth,
            "max_bytes": self.max_bytes,
            "receipt_globs": list(self.receipt_globs),
            "receipt_max_bytes": self.receipt_max_bytes,
            "skip_hash_dirs": self.skip_hash_dirs,
        }


def _string_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return value


def _normalize_suffix(value: str) -> str:
    suffix = value.strip().lower()
    return suffix if suffix.startswith(".") else f".{suffix}"


def _optional_count(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    raise ValueError(f"{key} must be a non-negative integer")


def index_policy_signature(context_path: Path) -> PolicySignature:
    """Return a cheap change signature for the two policy files."""
    return (
        _file_signature(context_path / IGNORE_FILENAME),
        _file_signature(context_path / POLICY_FILENAME),
    )


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


@dataclass
class IndexPathPolicy:
    """Optional ``.afsignore`` + ``.afsindex.toml`` rules for one context root."""

    context_path: Path
    ignore_patterns: tuple[IgnorePattern, ...] = ()
    allowlists: dict[str, MountAllowlist] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    signature: PolicySignature = (None, None)
    _ignored_dirs: dict[str, bool] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, context_path: Path) -> IndexPathPolicy:
        signature = index_policy_signature(context_path)
        ignore_patterns: tuple[IgnorePattern, ...] = ()
        if signature[0] is not None:
            ignore_patterns = tuple(load_ignore_file(context_path / IGNORE_FILENAME))
        allowlists: dict[str, MountAllowlist] = {}
        errors: list[str] = []
        if signature[1] is not None:
            allowlists = _load_allowlists(context_path / POLICY_FILENAME, errors)
        return cls(
            context_path=context_path,
            ignore_patterns=ignore_patterns,
            allowlists=allowlists,
            errors=tuple(errors),
            signature=signature,
        )

    @property
    def active(self) -> bool:
        return bool(self.ignore_patterns or self.allowlists)

    def describe(self) -> dict[str, Any]:
        return {
            "ignore_file": (
                str(self.context_path / IGNORE_FILENAME)
                if self.signature[0] is not None
                else None
            ),
            "ignore_patterns": len(self.ignore_patterns),
            "config_file": (
                str(self.context_path / POLICY_FILENAME)
                if self.signature[1] is not None
                else None
            ),
            "allowlists": {
                mount: allowlist.to_dict()
                for mount, allowlist in sorted(self.allowlists.items())
            },
        }

    def prunes_mount(self, mount_type: str) -> bool:
        """Return whether walks under ``mount_type`` can skip whole subtrees."""
        allowlist = self.allowlists.get(mount_type)
        return bool(self.ignore_patterns) or bool(allowlist and allowlist.skip_hash_dirs)

    def prunes(
        self,
        mount_type: str,
        relative_path: str,
        *,
        scope_relative_path: str,
    ) -> bool:
        """Return whether the directory at ``relative_path`` is excluded with its subtree."""
        if self.ignore_patterns and self._ignored_dir(_virtual_path(mount_type, relative_path)):
            return True
        allowlist = self.allowlists.get(mount_type)
        if allowlist is None or not allowlist.skip_hash_dirs:
            return False
        return any(_HASH_DIR_RE.match(part) for part in _parts(scope_relative_path))

    def excludes(
        self,
        mount_type: str,
        relative_path: str,
        *,
        scope_relative_path: str,
        path: Path,
    ) -> bool:
        """Return whether one entry stays out of the index under this policy."""
        if self.ignore_patterns:
            virtual = _virtual_path(mount_type, relative_path)
            parent = virtual.rpartition("/")[0]
            if parent and self._ignored_dir(parent):
                return True
            if ignore_matches(virtual, self.ignore_patterns):
                return True

        allowlist = self.allowlists.get(mount_type)
        parts = _parts(scope_relative_path)
        if allowlist is None or not parts:
            return False
        try:
            is_dir = path.is_dir()
        except OSError:
            return False
        dir_parts = parts if is_dir else parts[:-1]
        if allowlist.skip_hash_dirs and any(_HASH_DIR_RE.match(part) for part in dir_parts):
            return True
        depth = len(parts) - 1
        too_deep = allowlist.max_depth is not None and depth > allowlist.max_depth
        if is_dir:
            return too_deep

        name = parts[-1]
        if allowlist.suffixes is not None and Path(name).suffix.lower() not in allowlist.suffixes:
            return True
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if allowlist.is_receipt(name):
            limit = allowlist.receipt_max_bytes
            if limit is None:
                limit = allowlist.max_bytes
            return limit is not None and size > limit
        if too_deep:
            return True
        return allowlist.max_bytes is not None and size > allowlist.max_bytes

    def _ignored_dir(self, virtual_dir: str) -> bool:
        cached = self._ignored_dirs.get(virtual_dir)
        if cached is not None:
            return cached
        parent = virtual_dir.rpartition("/")[0]
        ignored = bool(parent) and self._ignored_dir(parent)
        if not ignored:
            ignored = ignore_matches(virtual_dir + "/", self.ignore_patterns)
        if len(self._ignored_dirs) >= _DIR_CACHE_LIMIT:
            self._ignored_dirs.clear()
        self._ignored_dirs[virtual_dir] = ignored
        return ignored


def _load_allowlists(path: Path, errors: list[str]) -> dict[str, MountAllowlist]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        errors.append(f"{path.name}: unreadable, allowlists ignored: {exc}")
        return {}
    section = data.get("allowlist", {})
    if not isinstance(section, dict):
        errors.append(f"{path.name}: [allowlist] must be a table, ignored")
        return {}
    allowlists: dict[str, MountAllowlist] = {}
    for mount, table in section.items():
        if mount not in _MOUNT_TYPE_VALUES:
            errors.append(f"{path.name}: [allowlist.{mount}] is not a mount type, ignored")
            continue
        try:
            allowlists[mount] = MountAllowlist.from_mapping(table)
        except ValueError as exc:
            errors.append(f"{path.name}: [allowlist.{mount}] {exc}, ignored")
    return allowlists


def _virtual_path(mount_type: str, relative_path: str) -> str:
    return f"{mount_type}/{relative_path}".strip("/")


def _parts(relative_path: str) -> list[str]:
    return [part for part in relative_path.split("/") if part]
