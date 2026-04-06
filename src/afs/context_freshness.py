"""Per-mount freshness scoring and session-aware context diff tracking."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context_paths import resolve_agent_output_root, resolve_mount_root
from .models import MountType

logger = logging.getLogger(__name__)

# --- Part 1: Per-Mount Freshness ---

_DEFAULT_DECAY_HOURS = 168.0  # 1 week
_STALE_THRESHOLD = 0.3


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
) -> dict[str, MountFreshness]:
    """Compute per-mount freshness for all active mounts under a context path.

    Walks each mount directory on the filesystem (no DB required), finds the
    newest file mtime, and computes an exponential-decay freshness score.

    Returns a dict keyed by mount_type value (e.g. "memory", "knowledge").
    """
    context_path = context_path.expanduser().resolve()
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

        file_count = 0
        newest_mtime: float | None = None

        try:
            for entry in mount_root.rglob("*"):
                if not entry.is_file():
                    continue
                if entry.name == ".keep":
                    continue
                file_count += 1
                try:
                    mtime = entry.stat().st_mtime
                    if newest_mtime is None or mtime > newest_mtime:
                        newest_mtime = mtime
                except OSError:
                    continue
        except OSError:
            continue

        if file_count == 0:
            continue

        score = _decay_score(now, newest_mtime, decay_seconds)

        result[mount_type.value] = MountFreshness(
            mount_path=str(mount_root),
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


def _snapshot_dir(context_path: Path, config: Any | None = None) -> Path:
    """Resolve the directory where context snapshots are stored."""
    output_root = resolve_agent_output_root(context_path, config=config)
    return output_root / _SNAPSHOT_DIR_NAME


def _snapshot_filename(session_id: str) -> str:
    # Sanitize session_id for safe filename usage
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return f"snapshot_{safe}.json"


def save_context_snapshot(
    context_path: Path,
    session_id: str,
    *,
    config: Any | None = None,
) -> Path:
    """Save a lightweight (path, mtime, size) snapshot of all mount files.

    Keeps at most _MAX_SNAPSHOTS per context, pruning oldest first.
    Returns the path to the saved snapshot file.
    """
    context_path = context_path.expanduser().resolve()
    snap_dir = _snapshot_dir(context_path, config=config)
    snap_dir.mkdir(parents=True, exist_ok=True)

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

        try:
            for entry in mount_root.rglob("*"):
                if not entry.is_file():
                    continue
                if entry.name == ".keep":
                    continue
                try:
                    st = entry.stat()
                    rel = str(entry.relative_to(context_path))
                    entries.append({
                        "path": rel,
                        "mount_type": mount_type.value,
                        "mtime": st.st_mtime,
                        "size": st.st_size,
                    })
                except OSError:
                    continue
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
    snap_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    # Prune old snapshots, keeping newest _MAX_SNAPSHOTS
    _prune_snapshots(snap_dir)

    return snap_path


def _prune_snapshots(snap_dir: Path) -> None:
    """Remove oldest snapshots beyond _MAX_SNAPSHOTS."""
    try:
        snapshots = sorted(
            (f for f in snap_dir.iterdir() if f.is_file() and f.name.startswith("snapshot_")),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return

    for old in snapshots[_MAX_SNAPSHOTS:]:
        try:
            old.unlink()
        except OSError:
            pass


def _load_latest_snapshot(
    context_path: Path,
    session_id: str | None = None,
    *,
    config: Any | None = None,
) -> dict[str, Any] | None:
    """Load a snapshot by session_id, or the most recent one if session_id is None."""
    snap_dir = _snapshot_dir(context_path, config=config)
    if not snap_dir.exists():
        return None

    if session_id is not None:
        target = snap_dir / _snapshot_filename(session_id)
        if not target.exists():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    # Find the most recent snapshot
    try:
        snapshots = sorted(
            (f for f in snap_dir.iterdir() if f.is_file() and f.name.startswith("snapshot_")),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None

    if not snapshots:
        return None

    try:
        return json.loads(snapshots[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def context_diff_since_session(
    context_path: Path,
    session_id: str | None = None,
    *,
    config: Any | None = None,
) -> ContextDiffReport | None:
    """Compare the current filesystem state against a saved snapshot.

    If session_id is provided, loads that specific snapshot.
    If None, loads the most recent snapshot.
    Returns None if no snapshot is found.
    """
    context_path = context_path.expanduser().resolve()
    snapshot = _load_latest_snapshot(context_path, session_id, config=config)
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

        try:
            for fentry in mount_root.rglob("*"):
                if not fentry.is_file():
                    continue
                if fentry.name == ".keep":
                    continue
                try:
                    st = fentry.stat()
                    rel = str(fentry.relative_to(context_path))
                    current_entries[(mount_type.value, rel)] = {
                        "path": rel,
                        "mount_type": mount_type.value,
                        "mtime": st.st_mtime,
                        "size": st.st_size,
                    }
                except OSError:
                    continue
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
