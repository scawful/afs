"""Read-only size/age report for ``handoffs/<topic>_<date>/`` folders.

Handoff folders accumulate ROM copies, build trees, logs and screenshots with
no retention policy. This report only measures; it never moves or deletes.
A folder is an archive candidate when its newest file is older than the cutoff
and its name does not appear in the live inbox or board.
"""

from __future__ import annotations

import os
import stat
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .handoff_prompt import read_supersede_marker

KIND_SUFFIXES = {
    "build": {".o", ".a", ".dylib", ".so", ".pch", ".gch", ".obj", ".pack", ".idx"},
    "rom": {".sfc", ".smc"},
    "debug": {".cdl", ".sym", ".map", ".mlb", ".mss"},
    "logs": {".log"},
    "images": {".png", ".jpg", ".jpeg", ".gif", ".webp"},
    "fonts": {".ttf", ".otf"},
}


def _kind(name: str, mode: int) -> str:
    suffix = Path(name).suffix.lower()
    for kind, suffixes in KIND_SUFFIXES.items():
        if suffix in suffixes:
            return kind
    if not suffix and mode & stat.S_IXUSR:
        return "build"
    return "other"


@dataclass
class FolderStats:
    name: str
    path: Path
    newest_mtime: float = 0.0
    size_bytes: int = 0
    file_count: int = 0
    by_kind: Counter[str] = field(default_factory=Counter)
    superseded_by: str = ""
    referenced: bool = False

    def age_days(self, now: float) -> float:
        return max(0.0, (now - self.newest_mtime) / 86400) if self.newest_mtime else 0.0

    def to_dict(self, now: float) -> dict[str, Any]:
        heavy = {kind: size for kind, size in self.by_kind.most_common(3) if size > 0}
        return {
            "name": self.name,
            "path": str(self.path),
            "age_days": round(self.age_days(now), 1),
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "largest_kinds": heavy,
            "superseded_by": self.superseded_by,
            "referenced": self.referenced,
        }


def _walk(stats: FolderStats) -> None:
    stack = [stats.path]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            stats.file_count += 1
            stats.size_bytes += info.st_size
            stats.newest_mtime = max(stats.newest_mtime, info.st_mtime)
            stats.by_kind[_kind(entry.name, info.st_mode)] += info.st_size


def folder_report(
    root: Path,
    *,
    older_than_days: float = 14.0,
    live_texts: tuple[str, ...] = (),
    now: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    folders: list[FolderStats] = []
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            stats = FolderStats(name=entry.name, path=entry)
            _walk(stats)
            marker = read_supersede_marker(entry)
            stats.superseded_by = marker.superseded_by if marker else ""
            stats.referenced = any(entry.name in text for text in live_texts)
            folders.append(stats)
    candidates = [
        stats
        for stats in folders
        if stats.age_days(now) >= older_than_days and not stats.referenced
    ]
    by_kind: Counter[str] = Counter()
    for stats in folders:
        by_kind.update(stats.by_kind)
    return {
        "root": str(root),
        "older_than_days": older_than_days,
        "folder_count": len(folders),
        "total_bytes": sum(stats.size_bytes for stats in folders),
        "bytes_by_kind": dict(by_kind.most_common()),
        "superseded_count": sum(1 for stats in folders if stats.superseded_by),
        "candidate_count": len(candidates),
        "candidate_bytes": sum(stats.size_bytes for stats in candidates),
        "largest": [
            stats.to_dict(now)
            for stats in sorted(folders, key=lambda item: item.size_bytes, reverse=True)
        ],
        "candidates": [
            stats.to_dict(now)
            for stats in sorted(candidates, key=lambda item: item.size_bytes, reverse=True)
        ],
    }
