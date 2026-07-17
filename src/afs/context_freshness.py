"""Per-mount freshness scoring and session-aware context diff tracking."""

from __future__ import annotations

import json
import logging
import os
import stat
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context_index import INDEX_SCAN_SKIP_NAMES, _iter_mount_entries
from .context_layout import LAYOUT_VERSION, _atomic_write_text, detect_layout_version
from .context_paths import resolve_agent_output_root, resolve_mount_root
from .models import MountType
from .path_safety import (
    assert_no_linklike_components,
    is_linklike,
    iter_regular_files_no_links,
)
from .scopes import ResolvedScope, resolve_scope, visible_mount_roots

logger = logging.getLogger(__name__)

# --- Part 1: Per-Mount Freshness ---

_DEFAULT_DECAY_HOURS = 168.0  # 1 week
_STALE_THRESHOLD = 0.3


def _iter_mount_files(mount_root: Path, *, no_links: bool = False):
    """Yield mount files, optionally refusing every link-like descendant."""
    if no_links:
        for entry in iter_regular_files_no_links(
            mount_root,
            skip_names=frozenset(INDEX_SCAN_SKIP_NAMES),
        ):
            yield entry, entry.relative_to(mount_root).as_posix()
        return
    for entry, relative_path in _iter_mount_entries(mount_root):
        try:
            if not entry.is_file():
                continue
        except OSError:
            continue
        yield entry, relative_path


@dataclass
class MountFreshness:
    """Freshness summary for a single context mount."""

    mount_path: str
    mount_type: str
    file_count: int
    newest_mtime: float | None
    freshness_score: float
    stale: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["newest_mtime"] is not None:
            d["newest_mtime_iso"] = datetime.fromtimestamp(
                d["newest_mtime"], tz=timezone.utc
            ).isoformat()
        return d


def mount_freshness(
    context_path: Path,
    *,
    config: Any | None = None,
    decay_hours: float = _DEFAULT_DECAY_HOURS,
    stale_threshold: float = _STALE_THRESHOLD,
    scoped: ResolvedScope | None = None,
) -> dict[str, MountFreshness]:
    """Compute per-mount freshness for all active mounts under a context path.

    Walks each mount directory on the filesystem (no DB required), finds the
    newest file mtime, and computes an exponential-decay freshness score.

    Returns a dict keyed by mount_type value (e.g. "memory", "knowledge").
    """
    context_path = context_path.expanduser().resolve()
    scope = scoped or resolve_scope(context_path)
    decay_seconds = decay_hours * 3600.0
    now = time.time()
    result: dict[str, MountFreshness] = {}

    for mount_type in MountType:
        try:
            mount_root = resolve_mount_root(
                context_path, mount_type, config=config
            )
        except Exception:
            continue
        if not mount_root.exists():
            continue
        roots = visible_mount_roots(mount_root, mount_type=mount_type, scoped=scope)
        if not roots:
            continue

        file_count = 0
        newest_mtime: float | None = None
        contributing_root: Path | None = None

        for visible_root in roots:
            if not visible_root.exists():
                continue
            root_file_count = 0
            for entry, _relative_path in _iter_mount_files(
                visible_root,
                no_links=scope.layout_version == LAYOUT_VERSION,
            ):
                file_count += 1
                root_file_count += 1
                try:
                    mtime = entry.stat().st_mtime
                    if newest_mtime is None or mtime > newest_mtime:
                        newest_mtime = mtime
                except OSError:
                    continue
            if root_file_count and contributing_root is None:
                contributing_root = visible_root

        if file_count == 0:
            continue

        score = _decay_score(now, newest_mtime, decay_seconds)

        result[mount_type.value] = MountFreshness(
            mount_path=str(contributing_root or roots[0]),
            mount_type=mount_type.value,
            file_count=file_count,
            newest_mtime=newest_mtime,
            freshness_score=round(score, 4),
            stale=score < stale_threshold,
        )

    return result


def _decay_score(
    now: float,
    mtime: float | None,
    decay_seconds: float,
) -> float:
    """Compute freshness score: 1.0 = just modified, 0.0 = fully decayed."""
    if mtime is None:
        return 0.0
    if decay_seconds <= 0:
        return 1.0
    age = max(0.0, now - mtime)
    return max(0.0, 1.0 - age / decay_seconds)


# --- Part 2: Session-Aware Diff Tracking ---

_SNAPSHOT_DIR_NAME = "context_snapshots"
_MAX_SNAPSHOTS = 10


@dataclass
class FileSnapshot:
    """Lightweight record of a single file's state."""

    path: str
    mtime: float
    size: int


@dataclass
class ContextDiffReport:
    """Comparison between a saved snapshot and the current filesystem state."""

    session_id: str | None
    snapshot_timestamp: str | None
    added: list[dict[str, Any]] = field(default_factory=list)
    modified: list[dict[str, Any]] = field(default_factory=list)
    deleted: list[dict[str, Any]] = field(default_factory=list)
    per_mount_summary: dict[str, dict[str, int]] = field(default_factory=dict)
    change_summary: str = ""

    @property
    def total_changes(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "snapshot_timestamp": self.snapshot_timestamp,
            "added": self.added,
            "modified": self.modified,
            "deleted": self.deleted,
            "total_changes": self.total_changes,
            "per_mount_summary": self.per_mount_summary,
            "change_summary": self.change_summary,
        }


def _snapshot_dir(
    context_path: Path,
    config: Any | None = None,
    *,
    scope_id: str = "common",
) -> Path:
    """Resolve the directory where context snapshots are stored."""
    output_root = resolve_agent_output_root(context_path, config=config, scope_id=scope_id)
    snapshot_root = output_root / _SNAPSHOT_DIR_NAME
    if detect_layout_version(context_path) == LAYOUT_VERSION:
        return assert_no_linklike_components(snapshot_root, boundary=output_root)
    return snapshot_root


def _snapshot_filename(session_id: str) -> str:
    # Sanitize session_id for safe filename usage
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return f"snapshot_{safe}.json"


def save_context_snapshot(
    context_path: Path,
    session_id: str,
    *,
    config: Any | None = None,
    scoped: ResolvedScope | None = None,
) -> Path:
    """Save a lightweight (path, mtime, size) snapshot of all mount files.

    Keeps at most _MAX_SNAPSHOTS per context, pruning oldest first.
    Returns the path to the saved snapshot file.
    """
    context_path = context_path.expanduser().resolve()
    scope = scoped or resolve_scope(context_path)
    snap_dir = _snapshot_dir(context_path, config=config, scope_id=scope.scope_id)
    snap_dir.mkdir(parents=True, exist_ok=True)
    if scope.layout_version == LAYOUT_VERSION:
        snap_dir = assert_no_linklike_components(
            snap_dir,
            boundary=resolve_agent_output_root(
                context_path,
                config=config,
                scope_id=scope.scope_id,
            ),
            allow_missing=False,
        )

    entries: list[dict[str, Any]] = []
    for mount_type in MountType:
        try:
            mount_root = resolve_mount_root(
                context_path, mount_type, config=config
            )
        except Exception:
            continue
        if not mount_root.exists():
            continue
        for visible_root in visible_mount_roots(mount_root, mount_type=mount_type, scoped=scope):
            if not visible_root.exists():
                continue
            prefix = visible_root.relative_to(mount_root).as_posix()
            for entry, relative_path in _iter_mount_files(
                visible_root,
                no_links=scope.layout_version == LAYOUT_VERSION,
            ):
                try:
                    st = entry.stat()
                    scoped_relative = f"{prefix}/{relative_path}" if prefix != "." else relative_path
                    rel = f"{mount_type.value}/{scoped_relative}"
                    entries.append({
                        "path": rel,
                        "mount_type": mount_type.value,
                        "mtime": st.st_mtime,
                        "size": st.st_size,
                    })
                except OSError:
                    continue

    payload = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "context_path": str(context_path),
        "file_count": len(entries),
        "entries": entries,
    }

    snap_path = snap_dir / _snapshot_filename(session_id)
    if scope.layout_version == LAYOUT_VERSION:
        snap_path = assert_no_linklike_components(snap_path, boundary=snap_dir)
    _atomic_write_text(snap_path, json.dumps(payload, indent=1) + "\n")

    # Prune old snapshots, keeping newest _MAX_SNAPSHOTS
    _prune_snapshots(
        snap_dir,
        harden=scope.layout_version == LAYOUT_VERSION,
    )

    return snap_path


def _snapshot_candidates(
    snap_dir: Path,
    *,
    harden: bool,
) -> list[tuple[Path, float]]:
    """Return snapshot leaves and mtimes without following v2 links."""

    candidates: list[tuple[Path, float]] = []
    for candidate in snap_dir.iterdir():
        if not candidate.name.startswith("snapshot_"):
            continue
        if harden:
            try:
                candidate = assert_no_linklike_components(
                    candidate,
                    boundary=snap_dir,
                    allow_missing=False,
                )
                candidate_stat = os.lstat(candidate)
            except (OSError, ValueError):
                continue
            if is_linklike(candidate_stat) or not stat.S_ISREG(candidate_stat.st_mode):
                continue
        else:
            try:
                if not candidate.is_file():
                    continue
                candidate_stat = candidate.stat()
            except OSError:
                continue
        candidates.append((candidate, candidate_stat.st_mtime))
    return candidates


def _prune_snapshots(snap_dir: Path, *, harden: bool = False) -> None:
    """Remove oldest snapshots beyond _MAX_SNAPSHOTS."""
    try:
        snapshots = sorted(
            _snapshot_candidates(snap_dir, harden=harden),
            key=lambda item: item[1],
            reverse=True,
        )
    except OSError:
        return

    for old, _mtime in snapshots[_MAX_SNAPSHOTS:]:
        try:
            if harden:
                old = assert_no_linklike_components(
                    old,
                    boundary=snap_dir,
                    allow_missing=False,
                )
            old.unlink()
        except (OSError, ValueError):
            pass


def _load_latest_snapshot(
    context_path: Path,
    session_id: str | None = None,
    *,
    config: Any | None = None,
    scope_id: str = "common",
) -> dict[str, Any] | None:
    """Load a snapshot by session_id, or the most recent one if session_id is None."""
    snap_dir = _snapshot_dir(context_path, config=config, scope_id=scope_id)
    if not snap_dir.exists():
        return None

    if session_id is not None:
        target = snap_dir / _snapshot_filename(session_id)
        if detect_layout_version(context_path) == LAYOUT_VERSION:
            try:
                target = assert_no_linklike_components(
                    target,
                    boundary=snap_dir,
                    allow_missing=False,
                )
            except (OSError, ValueError):
                return None
        if not target.exists():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    # Find the most recent snapshot
    try:
        harden = detect_layout_version(context_path) == LAYOUT_VERSION
        snapshots = sorted(
            _snapshot_candidates(snap_dir, harden=harden),
            key=lambda item: item[1],
            reverse=True,
        )
    except OSError:
        return None

    if not snapshots:
        return None

    try:
        target = snapshots[0][0]
        if harden:
            target = assert_no_linklike_components(
                target,
                boundary=snap_dir,
                allow_missing=False,
            )
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def context_diff_since_session(
    context_path: Path,
    session_id: str | None = None,
    *,
    config: Any | None = None,
    scoped: ResolvedScope | None = None,
) -> ContextDiffReport | None:
    """Compare the current filesystem state against a saved snapshot.

    If session_id is provided, loads that specific snapshot.
    If None, loads the most recent snapshot.
    Returns None if no snapshot is found.
    """
    context_path = context_path.expanduser().resolve()
    scope = scoped or resolve_scope(context_path)
    snapshot = _load_latest_snapshot(
        context_path,
        session_id,
        config=config,
        scope_id=scope.scope_id,
    )
    if snapshot is None:
        return None

    # Build lookup from snapshot: keyed by (mount_type, relative_path)
    old_entries: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in snapshot.get("entries", []):
        key = (entry["mount_type"], entry["path"])
        old_entries[key] = entry

    # Build current filesystem state
    current_entries: dict[tuple[str, str], dict[str, Any]] = {}
    for mount_type in MountType:
        try:
            mount_root = resolve_mount_root(
                context_path, mount_type, config=config
            )
        except Exception:
            continue
        if not mount_root.exists():
            continue
        for visible_root in visible_mount_roots(mount_root, mount_type=mount_type, scoped=scope):
            if not visible_root.exists():
                continue
            prefix = visible_root.relative_to(mount_root).as_posix()
            for fentry, relative_path in _iter_mount_files(
                visible_root,
                no_links=scope.layout_version == LAYOUT_VERSION,
            ):
                try:
                    st = fentry.stat()
                    scoped_relative = f"{prefix}/{relative_path}" if prefix != "." else relative_path
                    rel = f"{mount_type.value}/{scoped_relative}"
                    current_entries[(mount_type.value, rel)] = {
                        "path": rel,
                        "mount_type": mount_type.value,
                        "mtime": st.st_mtime,
                        "size": st.st_size,
                    }
                except OSError:
                    continue

    # Compare
    added: list[dict[str, Any]] = []
    modified: list[dict[str, Any]] = []
    deleted: list[dict[str, Any]] = []
    mount_changes: dict[str, dict[str, int]] = {}

    def _bump(mt: str, change_type: str) -> None:
        if mt not in mount_changes:
            mount_changes[mt] = {"added": 0, "modified": 0, "deleted": 0}
        mount_changes[mt][change_type] += 1

    for key, cur in current_entries.items():
        mount_type_val, rel_path = key
        if key not in old_entries:
            added.append({
                "mount_type": mount_type_val,
                "relative_path": rel_path,
                "size": cur["size"],
            })
            _bump(mount_type_val, "added")
        else:
            old = old_entries[key]
            # Compare mtime with 1s tolerance, and size
            if abs(cur["mtime"] - old["mtime"]) > 1.0 or cur["size"] != old["size"]:
                modified.append({
                    "mount_type": mount_type_val,
                    "relative_path": rel_path,
                    "size": cur["size"],
                    "old_size": old["size"],
                })
                _bump(mount_type_val, "modified")

    for key, _old in old_entries.items():
        if key not in current_entries:
            mount_type_val, rel_path = key
            deleted.append({
                "mount_type": mount_type_val,
                "relative_path": rel_path,
            })
            _bump(mount_type_val, "deleted")

    total = len(added) + len(modified) + len(deleted)
    snap_session_id = snapshot.get("session_id")
    snap_timestamp = snapshot.get("timestamp")

    # Build human-readable summary
    parts: list[str] = []
    if added:
        parts.append(f"{len(added)} added")
    if modified:
        parts.append(f"{len(modified)} modified")
    if deleted:
        parts.append(f"{len(deleted)} deleted")
    change_summary = (
        f"{total} changes since session {snap_session_id or 'unknown'}: "
        + ", ".join(parts)
        if parts
        else f"No changes since session {snap_session_id or 'unknown'}"
    )

    return ContextDiffReport(
        session_id=snap_session_id,
        snapshot_timestamp=snap_timestamp,
        added=added,
        modified=modified,
        deleted=deleted,
        per_mount_summary=mount_changes,
        change_summary=change_summary,
    )
