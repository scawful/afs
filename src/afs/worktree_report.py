"""Read-only expiry report for a repository's linked git worktrees.

Agents create worktrees per task and rarely remove them. This report lists
each linked worktree with idle age, dirtiness and merge state, and flags
removal candidates; it never runs ``git worktree remove`` or ``prune``.
A branch counts as merged when its commit is in the default branch or its
upstream was deleted (the usual result of a squash-merged pull request).
"""

from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GIT_TIMEOUT_SECONDS = 20


@dataclass
class WorktreeInfo:
    path: str
    head: str = ""
    branch: str = ""
    prunable: bool = False
    locked: bool = False
    exists: bool = True
    last_commit: float = 0.0
    dirty: bool | None = None
    merged: bool = False
    upstream_gone: bool = False

    def idle_days(self, now: float) -> float:
        return max(0.0, (now - self.last_commit) / 86400) if self.last_commit else 0.0

    def candidate_reason(self, now: float, older_than_days: float) -> str:
        if self.locked:
            return ""
        if self.prunable or not self.exists:
            return "missing directory (git worktree prune)"
        if self.dirty is not False or self.idle_days(now) < older_than_days:
            return ""
        if self.merged:
            return "clean, merged into default branch"
        if self.upstream_gone:
            return "clean, upstream branch deleted"
        return ""

    def to_dict(self, now: float, older_than_days: float) -> dict[str, Any]:
        return {
            "path": self.path,
            "branch": self.branch,
            "head": self.head[:12],
            "idle_days": round(self.idle_days(now), 1),
            "dirty": self.dirty,
            "merged": self.merged,
            "upstream_gone": self.upstream_gone,
            "locked": self.locked,
            "prunable": self.prunable,
            "candidate": self.candidate_reason(now, older_than_days),
        }


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
        check=False,
    )


def parse_worktree_porcelain(text: str) -> list[WorktreeInfo]:
    items: list[WorktreeInfo] = []
    current: WorktreeInfo | None = None
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if key == "worktree":
            current = WorktreeInfo(path=value)
            items.append(current)
        elif current is None:
            continue
        elif key == "HEAD":
            current.head = value
        elif key == "branch":
            current.branch = value.removeprefix("refs/heads/")
        elif key == "prunable":
            current.prunable = True
        elif key == "locked":
            current.locked = True
    return items


def default_branch(repo: Path) -> str:
    result = _git(repo, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    for name in ("origin/main", "origin/master", "main", "master"):
        if _git(repo, "rev-parse", "--verify", "--quiet", name).returncode == 0:
            return name
    return "HEAD"


def _gone_branches(repo: Path) -> set[str]:
    result = _git(
        repo, "for-each-ref", "--format=%(refname:short)\t%(upstream:track)", "refs/heads"
    )
    gone: set[str] = set()
    for line in result.stdout.splitlines():
        name, _, track = line.partition("\t")
        if track.strip() == "[gone]":
            gone.add(name)
    return gone


def _inspect(repo: Path, info: WorktreeInfo, base: str, gone: set[str]) -> WorktreeInfo:
    path = Path(info.path)
    info.exists = path.is_dir()
    info.upstream_gone = info.branch in gone
    if info.head:
        log = _git(repo, "log", "-1", "--format=%ct", info.head)
        if log.returncode == 0 and log.stdout.strip().isdigit():
            info.last_commit = float(log.stdout.strip())
        info.merged = _git(repo, "merge-base", "--is-ancestor", info.head, base).returncode == 0
    if info.exists and not info.prunable:
        status = _git(path, "status", "--porcelain", "--untracked-files=no")
        info.dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return info


def worktree_report(
    repo: Path, *, older_than_days: float = 7.0, now: float | None = None, workers: int = 8
) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    now = time.time() if now is None else now
    listing = _git(repo, "worktree", "list", "--porcelain")
    if listing.returncode != 0:
        raise ValueError(listing.stderr.strip() or f"not a git repository: {repo}")
    entries = parse_worktree_porcelain(listing.stdout)
    linked = entries[1:]
    base = default_branch(repo)
    gone = _gone_branches(repo)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        inspected = list(pool.map(lambda item: _inspect(repo, item, base, gone), linked))
    rows = [item.to_dict(now, older_than_days) for item in inspected]
    rows.sort(key=lambda row: row["idle_days"], reverse=True)
    candidates = [row for row in rows if row["candidate"]]
    return {
        "repo": str(repo),
        "default_branch": base,
        "older_than_days": older_than_days,
        "linked_count": len(rows),
        "dirty_count": sum(1 for row in rows if row["dirty"]),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "worktrees": rows,
    }
