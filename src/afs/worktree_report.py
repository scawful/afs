"""Non-destructive worktree exit planning using existing mission metadata.

Git observations are distinct from supplied lifecycle claims. Nothing here merges,
removes, prunes, approves, or infers abandonment from an old commit timestamp.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GIT_TIMEOUT_SECONDS = 20
EXIT_CHOICES = {"merge-ready", "retained", "quarantined", "cleanup-ready"}


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
    untracked: int = 0
    ignored: int = 0
    status_sha256: str = ""
    ahead: int | None = None
    behind: int | None = None
    changed_paths: list[str] = field(default_factory=list)
    visibility: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def idle_days(self, now: float) -> float:
        # Commit age is visibility only. It is not last agent activity.
        return max(0.0, (now - self.last_commit) / 86400) if self.last_commit else 0.0

    def to_dict(self, now: float, older_than_days: float) -> dict[str, Any]:
        return {"path": self.path, "branch": self.branch, "head": self.head,
                "idle_days": round(self.idle_days(now), 1), "dirty": self.dirty,
                "merged": self.merged, "upstream_gone": self.upstream_gone,
                "locked": self.locked, "prunable": self.prunable, "exists": self.exists,
                "untracked_count": self.untracked, "ignored_count": self.ignored,
                "status_sha256": self.status_sha256, "ahead": self.ahead, "behind": self.behind,
                "changed_paths": self.changed_paths, "inspection_errors": self.errors,
                "index_visibility": self.visibility,
                "candidate": ""}


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(repo), *args],
        capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=False,
    )


def parse_worktree_porcelain(text: str) -> list[WorktreeInfo]:
    items: list[WorktreeInfo] = []
    current: WorktreeInfo | None = None
    # -z preserves paths containing newlines, tabs or quotes. Retain legacy
    # parser input support for callers supplying ordinary porcelain.
    for line in text.split("\0") if "\0" in text else text.splitlines():
        key, _, value = line.partition(" ")
        if key == "worktree":
            current = WorktreeInfo(path=value)
            items.append(current)
        elif current is not None:
            if key == "HEAD":
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
    result = _git(repo, "for-each-ref", "--format=%(refname:short)\t%(upstream:track)", "refs/heads")
    if result.returncode:
        raise ValueError("Cannot inspect upstream branches")
    return {line.partition("\t")[0] for line in result.stdout.splitlines()
            if line.partition("\t")[2].strip() == "[gone]"}


def _status(path: Path) -> str:
    status = _git(path, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored=matching",
                  "--ignore-submodules=none")
    if status.returncode:
        raise ValueError("Cannot inspect worktree status")
    return status.stdout


def _index_visibility(path: Path) -> dict[str, Any]:
    """Refuse preservation claims where Git can hide tracked/nested content.

    Do not clear index bits or initialize submodules to make a report pass.
    Reobserve this fingerprint alongside status before recommending review.
    """
    listing = _git(path, "ls-files", "--stage", "-v", "-z")
    if listing.returncode:
        raise ValueError("Cannot inspect index visibility")
    counts = {"skip_worktree": 0, "assume_unchanged": 0, "submodules": 0, "other_index_flags": 0}
    for entry in listing.stdout.split("\0"):
        if not entry:
            continue
        header, separator, _name = entry.partition("\t")
        fields = header.split()
        if not separator or len(fields) != 4:
            raise ValueError("Unrecognized index visibility record")
        flag, mode, _object, stage = fields
        counts["skip_worktree"] += flag.upper() == "S"
        counts["assume_unchanged"] += flag.islower()
        counts["submodules"] += mode == "160000"
        counts["other_index_flags"] += flag.upper() not in {"H", "S"} or stage != "0" or mode == "040000"
    settings = {}
    for name in ("core.sparseCheckout", "index.sparse", "core.ignoreStat"):
        result = _git(path, "config", "--bool", "--get", name)
        if result.returncode not in (0, 1):
            raise ValueError("Cannot inspect sparse/index visibility settings")
        settings[name] = result.returncode == 0 and result.stdout.strip() == "true"
    blockers = []
    if counts["skip_worktree"] or counts["assume_unchanged"] or counts["other_index_flags"]:
        blockers.append("Index visibility flags can hide tracked content; preservation unverified")
    if any(settings.values()):
        blockers.append("Sparse/ignoreStat index state requires manual preservation verification")
    if counts["submodules"]:
        blockers.append("Submodule content is not verified; preserve nested tracked/untracked/ignored work")
    snapshot = {"entries": listing.stdout, "settings": settings}
    return {**counts, "settings": settings, "blockers": blockers,
            "sha256": hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()}


def _inspect(repo: Path, info: WorktreeInfo, base: str, gone: set[str]) -> WorktreeInfo:
    path = Path(info.path)
    info.exists = path.is_dir()
    info.upstream_gone = info.branch in gone
    try:
        if info.head:
            log = _git(repo, "log", "-1", "--format=%ct", info.head)
            if log.returncode == 0 and log.stdout.strip().isdigit():
                info.last_commit = float(log.stdout.strip())
            ancestor = _git(repo, "merge-base", "--is-ancestor", info.head, base)
            if ancestor.returncode not in (0, 1):
                raise ValueError("Cannot inspect ancestry")
            info.merged = ancestor.returncode == 0
            counts = _git(repo, "rev-list", "--left-right", "--count", f"{base}...{info.head}")
            if counts.returncode:
                raise ValueError("Cannot inspect divergence")
            info.behind, info.ahead = map(int, counts.stdout.split())
            common = _git(repo, "merge-base", base, info.head)
            if common.returncode:
                raise ValueError("Cannot resolve merge base")
            changed = _git(repo, "diff", "--no-ext-diff", "--no-renames", "--name-only", "-z", common.stdout.strip(), info.head)
            if changed.returncode:
                raise ValueError("Cannot inspect changed paths")
            info.changed_paths = [p for p in changed.stdout.split("\0") if p]
        if info.exists and not info.prunable:
            status = _status(path)
            entries = [p for p in status.split("\0") if p]
            info.untracked = sum(p.startswith("?? ") for p in entries)
            info.ignored = sum(p.startswith("!! ") for p in entries)
            info.dirty = any(not p.startswith("!! ") for p in entries)
            info.status_sha256 = hashlib.sha256(status.encode()).hexdigest()
            info.visibility = _index_visibility(path)
            if info.visibility["blockers"]:
                info.dirty = None
            head = _git(path, "rev-parse", "HEAD")
            branch = _git(path, "symbolic-ref", "--quiet", "--short", "HEAD")
            if head.returncode or head.stdout.strip() != info.head or branch.stdout.strip() != info.branch:
                raise ValueError("Worktree identity changed during inspection")
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        info.errors.append(str(exc))
    return info


def _artifact(item: Any, worktree: Path) -> bool:
    """Check retention outside the worktree without exporting artifact content."""
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        return False
    path = Path(item["path"])
    if not path.is_absolute() or path.is_symlink() or path.resolve().is_relative_to(worktree.resolve()):
        return False
    try:
        if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
            return False
        return hashlib.sha256(path.read_bytes()).hexdigest() == item.get("sha256")
    except OSError:
        return False


def load_lifecycle_records(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    """Read existing AFS mission records; no new lifecycle store is created."""
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = path.read_bytes()
        mission = json.loads(data)
        if not isinstance(mission, dict) or not all(isinstance(mission.get(k), str) for k in ("mission_id", "owner", "status")):
            raise ValueError("Expected an AFS mission object with owner, ID and status")
        metadata = mission.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Mission metadata must be an object")
        exits = metadata.get("worktree_exit", [])
        if not isinstance(exits, list):
            raise ValueError("mission metadata.worktree_exit must be a list")
        for entry in exits:
            if not isinstance(entry, dict) or not isinstance(entry.get("worktree"), str):
                raise ValueError("worktree exit requires a worktree path")
            if not Path(entry["worktree"]).is_absolute() or not isinstance(entry.get("intended_exit", "retained"), str):
                raise ValueError("Worktree path must be absolute and intended exit must be text")
            key = str(Path(entry["worktree"]).expanduser().resolve())
            if key in records:
                raise ValueError(f"Conflicting lifecycle records for {key}")
            records[key] = {**entry, "owner": mission.get("owner", ""),
                            "task": mission.get("mission_id", ""), "task_status": mission.get("status", ""),
                            "record_path": str(path), "record_sha256": hashlib.sha256(data).hexdigest()}
    return records


def _owner_inactive(pid: Any) -> bool:
    """Only an observed absent positive PID qualifies; uncertainty retains."""
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (OSError, OverflowError):
        return False
    return False


def _exit_plan(info: WorktreeInfo, base: str, record: dict[str, Any] | None) -> dict[str, Any]:
    blockers = list(info.errors) + info.visibility.get("blockers", [])
    plan: dict[str, Any] = {"intended_exit": "retained", "ready_for_review": False,
                           "execution_authorized": False, "blockers": blockers,
                           "claims_trust": "supplied mission metadata; not human approval"}
    if record is None:
        blockers.append("No owner/task/exit contract; retain worktree")
        return plan
    intent = record.get("intended_exit", "retained")
    plan.update(intended_exit=intent, owner=record.get("owner"), task=record.get("task"),
                review_status=record.get("review_status", "unknown"),
                adoption_status=record.get("adoption_status", "unknown"),
                retention_reason=record.get("retention_reason", ""),
                record_path=record.get("record_path"), record_sha256=record.get("record_sha256"))
    if intent not in EXIT_CHOICES:
        blockers.append("Unknown intended exit")
    if not record.get("owner") or not record.get("task"):
        blockers.append("Owner/task missing")
    if not isinstance(record.get("repo"), str) or record.get("repo") != str(Path(record["repo"]).resolve()):
        blockers.append("Repository must be an absolute canonical path")
    if record.get("base_commit") != base or record.get("head_commit") != info.head:
        blockers.append("Contract does not bind the current exact base/head")
    scope = record.get("allowed_scope")
    if not isinstance(scope, list) or not scope or not all(isinstance(s, str) and s for s in scope):
        blockers.append("Allowed scope missing")
    elif any(not any(fnmatch.fnmatchcase(p, s) for s in scope) for p in info.changed_paths):
        blockers.append("Committed changes outside declared scope")
    if record.get("branch") != info.branch:
        blockers.append("Contract branch does not match")
    if intent in {"retained", "quarantined"}:
        if not record.get("retention_reason"):
            blockers.append("Retention reason missing")
        return plan
    if info.locked or info.prunable or not info.exists:
        blockers.append("Locked, missing or prunable worktree requires manual reconciliation")
    if info.dirty is not False or info.untracked or info.ignored:
        blockers.append("Dirty, untracked, ignored or unknown content must be preserved")
    if record.get("owner_state") != "inactive" or record.get("task_status") in {"active", "blocked"}:
        blockers.append("Owner/task is active or inactivity is unconfirmed")
    plan["owner_pid_inactive"] = _owner_inactive(record.get("owner_pid"))
    if not plan["owner_pid_inactive"]:
        blockers.append("Recorded owner PID is live, absent from contract or unverifiable")
    # Inspect existing AFS leases; metadata is never authority to release one.
    lease_path = record.get("lease_path")
    if not isinstance(lease_path, str) or not Path(lease_path).is_absolute():
        blockers.append("Explicit AFS lease observation missing")
    else:
        from .leases import inspect
        try:
            state = inspect(Path(lease_path))
            plan["lease_status"] = state["status"]
            if state["status"] != "free":
                blockers.append("Lease is not free; retain and reconcile owner/workers")
        except OSError:
            blockers.append("Lease inspection unavailable")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts or not all(_artifact(a, Path(info.path)) for a in artifacts):
        blockers.append("Hash-bound external handoff/artifact retention missing")
    if intent == "cleanup-ready":
        if not info.merged:
            blockers.append("Head is not reachable from the exact target; preserve commits")
    else:
        verification = record.get("verification")
        if not isinstance(verification, dict) or not _artifact(verification, Path(info.path)):
            blockers.append("Verification receipt missing or hash mismatch")
        else:
            try:
                evidence = json.loads(Path(verification["path"]).read_text())
                if not isinstance(evidence, dict):
                    raise ValueError("Verification receipt must be an object")
                if evidence.get("base_commit") != base or evidence.get("head_commit") != info.head:
                    blockers.append("Verification receipt is stale for base/head")
                if evidence.get("passed") is not True or not evidence.get("checks"):
                    blockers.append("Relevant checks have not passed")
                if evidence.get("conflicts") != "none":
                    blockers.append("Exact-pair conflict check missing or failed")
            except (OSError, ValueError):
                blockers.append("Verification receipt unreadable or invalid")
    plan["ready_for_review"] = not blockers
    return plan


def worktree_report(repo: Path, *, older_than_days: float = 7.0, now: float | None = None,
                    workers: int = 8, mission_records: tuple[Path, ...] = (),
                    base_ref: str | None = None) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    now = time.time() if now is None else now
    listing = _git(repo, "worktree", "list", "--porcelain", "-z")
    if listing.returncode:
        raise ValueError("Cannot list repository worktrees")
    all_entries = parse_worktree_porcelain(listing.stdout)
    entries = all_entries[1:]
    base = base_ref or default_branch(repo)
    resolved = _git(repo, "rev-parse", "--verify", f"{base}^{{commit}}")
    if resolved.returncode:
        raise ValueError("Cannot resolve target commit")
    base_commit = resolved.stdout.strip()
    contracts = load_lifecycle_records(mission_records)
    gone = _gone_branches(repo)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        inspected = list(pool.map(lambda item: _inspect(repo, item, base_commit, gone), entries))
    rows = []
    for info in inspected:
        row = info.to_dict(now, older_than_days)
        record = contracts.get(str(Path(info.path).resolve()))
        if record is not None and record.get("repo") != str(repo):
            info.errors.append("Contract repository does not match inspected repository")
        row["exit_plan"] = _exit_plan(info, base_commit, record)
        # Reobserve the snapshot before issuing a review recommendation.
        try:
            if info.exists and (hashlib.sha256(_status(Path(info.path)).encode()).hexdigest() != info.status_sha256
                                or _index_visibility(Path(info.path))["sha256"] != info.visibility.get("sha256")
                                or _git(Path(info.path), "rev-parse", "HEAD").stdout.strip() != info.head
                                or _git(Path(info.path), "symbolic-ref", "--quiet", "--short", "HEAD").stdout.strip() != info.branch):
                row["exit_plan"]["blockers"].append("Worktree changed during report; rerun")
                row["exit_plan"]["ready_for_review"] = False
        except (OSError, ValueError, subprocess.TimeoutExpired):
            row["exit_plan"]["blockers"].append("Snapshot recheck unavailable")
            row["exit_plan"]["ready_for_review"] = False
        if row["exit_plan"]["ready_for_review"] and row["exit_plan"]["intended_exit"] == "cleanup-ready":
            row["candidate"] = "eligible for cleanup review; no removal authorized"
        rows.append(row)
    branches = _git(repo, "for-each-ref", "--format=%(refname:short)%00%(objectname)%00%(upstream:track)", "refs/heads")
    merged_branches = _git(repo, "for-each-ref", "--format=%(refname:short)", f"--merged={base_commit}", "refs/heads")
    if branches.returncode or merged_branches.returncode:
        raise ValueError("Cannot inspect unattached branches")
    attached = {entry.branch for entry in all_entries}
    merged_names = set(merged_branches.stdout.splitlines())
    unattached = []
    for line in branches.stdout.splitlines():
        name, head, upstream = line.split("\0")
        if name not in attached:
            unattached.append({"branch": name, "head": head, "merged": name in merged_names,
                               "upstream_gone": upstream.strip() == "[gone]", "intended_exit": "retained",
                               "blockers": ["No checked-out owner/task contract; no branch deletion authorized"]})
    final_base = _git(repo, "rev-parse", "--verify", f"{base}^{{commit}}")
    if final_base.returncode or final_base.stdout.strip() != base_commit:
        for row in rows:
            row["exit_plan"]["blockers"].append("Target changed during report; rerun")
            row["exit_plan"]["ready_for_review"] = False
            row["candidate"] = ""
    rows.sort(key=lambda row: row["idle_days"], reverse=True)
    candidates = [row for row in rows if row["candidate"]]
    return {"schema": "afs.worktree-exit-report.v1", "repo": str(repo), "default_branch": base,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "base_commit": base_commit, "observed_at": now, "dry_run": True,
            "older_than_days": older_than_days, "age_semantics": "last commit age; not owner activity",
            "linked_count": len(rows), "dirty_count": sum(1 for row in rows if row["dirty"]),
            "candidate_count": len(candidates), "candidates": candidates, "worktrees": rows,
            "unattached_branches": unattached}
