"""Read-only storage auditing and human-gated cleanup transactions.

The cleanup surface is intentionally narrow.  Plans may only reference
rebuildable files discovered beneath fixed per-user roots; large model,
archive, context, application, Trash, and APFS snapshot footprints are
informational and never become deletion candidates.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import unicodedata
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .atomic_io import atomic_create_text, secure_mkdir
from .human_provenance import (
    HumanAuthorization,
    consume_human_authorization,
    decision_scope_parts,
)
from .path_safety import assert_no_linklike_components, is_linklike, lexical_absolute

GIB = 1024**3
PLAN_SCHEMA = "afs.storage.plan.v1"
RECEIPT_SCHEMA = "afs.storage.receipt.v1"
MAX_PLAN_BYTES = 8 * 1024 * 1024
MAX_PLAN_CANDIDATES = 50_000
MAX_RATIONALE_CHARS = 4096
MAX_INT64 = 2**63 - 1
DEFAULT_STALE_DAYS = 30
PLAN_TTL_SECONDS = 24 * 60 * 60
MAX_LSOF_BATCH_CANDIDATES = 100
MAX_LSOF_PAYLOAD_BYTES = 64 * 1024
LSOF_EXEC_HEADROOM_BYTES = 16 * 1024
LSOF_COMMAND_TIMEOUT_SECONDS = 10.0
LSOF_OPERATION_DEADLINE_SECONDS = 15.0
GC_PID_MAX_BYTES = 64
MAX_PROCESS_ID = 2**31 - 1
MIN_TIMESTAMP_NS = 946_684_800 * 1_000_000_000  # 2000-01-01
MAX_TIMESTAMP_NS = 4_102_444_800 * 1_000_000_000  # 2100-01-01
_SNAPSHOT_NAME = re.compile(r"^[0-9a-f]{40}$")
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MOUNT_ESCAPE = re.compile(r"\\([0-7]{3})")
_RENAME_NOREPLACE = 0x00000001
_RENAME_EXCL = 0x00000004
ALLOWED_CATEGORIES = frozenset(
    {"yaze_build", "opencode_temp", "lmstudio_log", "lmstudio_broken_link"}
)
PROTECTED_CATEGORIES = (
    "models",
    "caches",
    "archives",
    "trash",
    "afs_context",
    "applications",
    "apfs_snapshots",
)

CandidateKind = Literal["file", "directory", "broken_symlink"]
OpenStatus = Literal["clear", "open", "unknown"]
OpenChecker = Callable[[Path, CandidateKind], tuple[OpenStatus, str]]
SnapshotReader = Callable[[], tuple[str, ...]]
MountReader = Callable[[], frozenset[Path] | None]
MountIdentity = tuple[int, int]
MountIdentityReader = Callable[[Path], MountIdentity]
RemovalStatus = Literal["deleted", "deleted_durability_uncertain"]
MonotonicClock = Callable[[], float]


class StorageSafetyError(ValueError):
    """Raised when a storage plan or candidate fails closed."""


class StorageApplyError(RuntimeError):
    """Raised after an apply claim exists but cleanup did not complete."""

    def __init__(self, message: str, receipt: dict[str, Any]) -> None:
        super().__init__(message)
        self.receipt = receipt


@dataclass(frozen=True)
class _OperationDeadline:
    expires_at: float
    clock: MonotonicClock

    def remaining(self) -> float:
        remaining = self.expires_at - self.clock()
        if not math.isfinite(remaining) or remaining <= 0:
            return 0.0
        return remaining

    def command_timeout(self) -> float | None:
        remaining = self.remaining()
        if remaining <= 0:
            return None
        return min(LSOF_COMMAND_TIMEOUT_SECONDS, remaining)


def _new_lsof_deadline(
    *,
    clock: MonotonicClock | None = None,
) -> _OperationDeadline:
    active_clock = time.monotonic if clock is None else clock
    return _OperationDeadline(
        expires_at=active_clock() + LSOF_OPERATION_DEADLINE_SECONDS,
        clock=active_clock,
    )


@dataclass(frozen=True)
class RemovalOutcome:
    """Observed namespace result from removing one exact candidate."""

    status: RemovalStatus
    durability_error: str | None = None


@dataclass(frozen=True)
class TreeMeasurement:
    logical_bytes: int
    allocated_bytes: int
    estimated_reclaim_bytes: int
    entry_count: int
    tree_digest: str
    crosses_device: bool
    crosses_mount: bool


@dataclass(frozen=True)
class _PinnedTreeMeasurement:
    """One descriptor-relative tree measurement plus a rename-stable digest."""

    measurement: TreeMeasurement
    stable_digest: str


@dataclass(frozen=True)
class StorageCandidate:
    candidate_id: str
    category: str
    path: str
    kind: CandidateKind
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    logical_bytes: int
    allocated_bytes: int
    estimated_reclaim_bytes: int
    entry_count: int
    tree_digest: str
    open_status: OpenStatus
    blocked_reasons: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return not self.blocked_reasons

    def plan_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.pop("blocked_reasons")
        return record


RemoveCandidate = Callable[[StorageCandidate], RemovalOutcome | None]


@dataclass(frozen=True)
class StorageFootprint:
    name: str
    path: str
    logical_bytes: int
    allocated_bytes: int
    entry_count: int
    note: str
    issue: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_from_ns(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, timezone.utc).isoformat()


def _validate_timestamp_ns(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise StorageSafetyError(f"{field} must be an integer")
    if not MIN_TIMESTAMP_NS <= value <= MAX_TIMESTAMP_NS:
        raise StorageSafetyError(f"{field} is outside the supported 2000-2100 range")
    return value


def _contains_unsafe_path_text(value: str) -> bool:
    return any(
        character == "\0" or unicodedata.category(character).startswith("C") for character in value
    )


def _trusted_home(path: Path) -> tuple[Path, dict[str, int]]:
    home = lexical_absolute(path)
    if _contains_unsafe_path_text(os.fspath(home)):
        raise StorageSafetyError("home contains control or formatting characters")
    assert_no_linklike_components(home, boundary=None, allow_missing=False)
    home_stat = os.lstat(home)
    if not stat.S_ISDIR(home_stat.st_mode) or is_linklike(home_stat):
        raise StorageSafetyError(f"home must be a real directory: {home}")
    return home, {
        "device": home_stat.st_dev,
        "inode": home_stat.st_ino,
        "mode": home_stat.st_mode,
    }


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _candidate_id(category: str, home: Path, path: Path) -> str:
    relative = path.relative_to(home)
    digest = hashlib.sha256(f"{category}\0{relative.as_posix()}".encode()).hexdigest()
    return f"{category}_{digest[:24]}"


def _decode_mount_path(value: str) -> Path:
    decoded = _MOUNT_ESCAPE.sub(
        lambda match: chr(int(match.group(1), 8)),
        value,
    )
    if not decoded.startswith(os.sep):
        raise StorageSafetyError("mount inventory contained a relative path")
    return Path(os.path.normpath(decoded))


def _parse_linux_mountinfo(payload: str) -> frozenset[Path]:
    mounts: set[Path] = set()
    for line in payload.splitlines():
        if not line:
            continue
        fields = line.split()
        if len(fields) < 6 or "-" not in fields:
            raise StorageSafetyError("Linux mount inventory was malformed")
        mounts.add(_decode_mount_path(fields[4]))
    if not mounts:
        raise StorageSafetyError("Linux mount inventory was empty")
    return frozenset(mounts)


def _parse_darwin_mount_output(payload: str) -> frozenset[Path]:
    mounts: set[Path] = set()
    for line in payload.splitlines():
        if not line:
            continue
        match = re.match(r"^.+ on (.+) \([^)]+\)$", line)
        if match is None:
            raise StorageSafetyError("Darwin mount inventory was malformed")
        mounts.add(_decode_mount_path(match.group(1)))
    if not mounts:
        raise StorageSafetyError("Darwin mount inventory was empty")
    return frozenset(mounts)


def default_mount_reader() -> frozenset[Path] | None:
    """Return current mount points, or ``None`` when they cannot be proven."""

    try:
        if sys.platform.startswith("linux"):
            return _parse_linux_mountinfo(
                Path("/proc/self/mountinfo").read_text(
                    encoding="utf-8",
                    errors="strict",
                )
            )
        if sys.platform == "darwin":
            mount = shutil.which("mount") or "/sbin/mount"
            result = subprocess.run(
                [mount],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            return _parse_darwin_mount_output(result.stdout)
    except (OSError, StorageSafetyError, subprocess.TimeoutExpired, UnicodeError):
        return None
    return None


def _mount_strictly_between(
    home: Path,
    candidate: Path,
    mount_points: frozenset[Path],
) -> Path | None:
    for mount_point in mount_points:
        if mount_point == home or mount_point == candidate:
            continue
        if mount_point.is_relative_to(home) and candidate.is_relative_to(mount_point):
            return mount_point
    return None


def _iter_tree_no_links(
    root: Path,
    *,
    mount_points: frozenset[Path] = frozenset(),
) -> Iterator[tuple[Path, os.stat_result, str | None]]:
    """Yield one tree without following link-like descendants."""

    root = lexical_absolute(root)
    root_device = os.lstat(root).st_dev
    pending = [root]
    while pending:
        path = pending.pop()
        path_stat = os.lstat(path)
        target = os.readlink(path) if is_linklike(path_stat) else None
        yield path, path_stat, target
        if path != root and path_stat.st_dev != root_device:
            continue
        if path != root and path in mount_points:
            continue
        if not stat.S_ISDIR(path_stat.st_mode) or is_linklike(path_stat):
            continue
        children = sorted(
            (Path(entry.path) for entry in os.scandir(path)),
            key=os.fspath,
            reverse=True,
        )
        pending.extend(children)


def measure_tree(
    root: Path,
    *,
    mount_points: frozenset[Path] = frozenset(),
) -> TreeMeasurement:
    """Measure logical and allocated bytes without following filesystem links."""

    root = lexical_absolute(root)
    digest = hashlib.sha256()
    logical = 0
    allocated = 0
    estimated_reclaim = 0
    entries = 0
    crosses_device = False
    crosses_mount = root in mount_points
    seen: set[tuple[int, int]] = set()
    root_device = os.lstat(root).st_dev
    for path, path_stat, target in _iter_tree_no_links(
        root,
        mount_points=mount_points,
    ):
        relative = "." if path == root else path.relative_to(root).as_posix()
        digest.update(
            (
                f"{relative}\0{path_stat.st_dev}\0{path_stat.st_ino}\0"
                f"{path_stat.st_mode}\0{path_stat.st_size}\0"
                f"{path_stat.st_mtime_ns}\0{path_stat.st_ctime_ns}\0"
                f"{target or ''}\n"
            ).encode("utf-8", errors="surrogateescape")
        )
        entries += 1
        if path_stat.st_dev != root_device:
            crosses_device = True
        if path != root and path in mount_points:
            crosses_mount = True
        identity = (path_stat.st_dev, path_stat.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        blocks = getattr(path_stat, "st_blocks", 0) * 512
        allocated += blocks
        if stat.S_ISREG(path_stat.st_mode) or is_linklike(path_stat):
            logical += path_stat.st_size
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink == 1:
            estimated_reclaim += blocks
    return TreeMeasurement(
        logical_bytes=logical,
        allocated_bytes=allocated,
        estimated_reclaim_bytes=estimated_reclaim,
        entry_count=entries,
        tree_digest=digest.hexdigest(),
        crosses_device=crosses_device,
        crosses_mount=crosses_mount,
    )


def _age_block(path_stat: os.stat_result, *, stale_days: int, now_ns: int) -> str | None:
    cutoff_ns = now_ns - stale_days * 24 * 60 * 60 * 1_000_000_000
    if path_stat.st_mtime_ns > cutoff_ns:
        return f"newer_than_{stale_days}_days"
    return None


def _pid_marker_block(path: Path) -> str | None:
    if path.name != "gc.pid":
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        payload = os.read(descriptor, GC_PID_MAX_BYTES + 1)
        if len(payload) > GC_PID_MAX_BYTES:
            return "gc_pid_marker_truncated"
        try:
            raw = payload.decode("ascii", errors="strict")
        except UnicodeDecodeError:
            return "gc_pid_marker_non_ascii"
        if not raw:
            return "gc_pid_marker_empty"
        if re.fullmatch(r"[1-9][0-9]* [!-~]+\n?", raw) is None:
            return "gc_pid_marker_malformed"
    finally:
        os.close(descriptor)
    pid = int(raw.split(maxsplit=1)[0])
    if pid > MAX_PROCESS_ID:
        return "gc_pid_marker_out_of_range"
    try:
        os.kill(pid, 0)
    except OverflowError:
        return "gc_pid_marker_out_of_range"
    except ProcessLookupError:
        return None
    except PermissionError:
        return "gc_pid_status_unknown"
    except OSError:
        return "gc_pid_status_unknown"
    return f"gc_pid_{pid}_is_running"


def default_open_checker(
    path: Path,
    kind: CandidateKind,
    *,
    deadline: _OperationDeadline | None = None,
) -> tuple[OpenStatus, str]:
    """Return whether ``lsof`` observes a candidate in use; never stop it."""

    path = lexical_absolute(path)
    if _contains_unsafe_path_text(os.fspath(path)):
        return "unknown", "path_contains_control_or_formatting_characters"
    lsof = shutil.which("lsof")
    if lsof is None:
        return "unknown", "lsof_unavailable"
    command = [lsof, "-Fn"]
    if kind == "directory":
        command.extend(["+D", os.fspath(path)])
    else:
        command.extend(["--", os.fspath(path)])
    if len(os.fsencode(path)) + 1 > _lsof_payload_budget(tuple(command[:-1])):
        return "unknown", "lsof_argument_too_long"
    active_deadline = deadline or _new_lsof_deadline()
    command_timeout = active_deadline.command_timeout()
    if command_timeout is None:
        return "unknown", "lsof_operation_deadline_exceeded"
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=command_timeout,
        )
    except subprocess.TimeoutExpired:
        if active_deadline.remaining() <= 0:
            return "unknown", "lsof_operation_deadline_exceeded"
        return "unknown", "lsof_timed_out"
    except (OSError, UnicodeError) as exc:
        return "unknown", f"lsof_failed:{type(exc).__name__}"
    if active_deadline.remaining() <= 0:
        return "unknown", "lsof_operation_deadline_exceeded"
    if result.stderr.strip():
        return "unknown", "lsof_reported_errors"
    if result.returncode == 1:
        return "clear", ""
    if result.returncode != 0:
        return "unknown", f"lsof_exit_{result.returncode}"
    names = {
        line[1:]
        for line in result.stdout.splitlines()
        if line.startswith("n") and line[1:].startswith(os.sep)
    }
    if not names:
        return "unknown", "lsof_output_missing_names"
    candidate = os.fspath(path)
    prefix = f"{candidate}{os.sep}"
    if any(
        name == candidate or (kind == "directory" and name.startswith(prefix)) for name in names
    ):
        return "open", "open_files_detected"
    return "unknown", "lsof_output_missing_candidate"


def _lsof_payload_budget(command_prefix: tuple[str, ...]) -> int:
    """Bound selected path arguments below the platform exec limit."""

    try:
        argument_max = int(os.sysconf("SC_ARG_MAX"))
    except (AttributeError, OSError, TypeError, ValueError):
        argument_max = 128 * 1024
    environment_bytes = sum(
        len(os.fsencode(key)) + len(os.fsencode(value)) + 2 for key, value in os.environ.items()
    )
    prefix_bytes = sum(len(os.fsencode(value)) + 1 for value in command_prefix)
    pointer_bytes = (len(os.environ) + len(command_prefix) + MAX_LSOF_BATCH_CANDIDATES + 2) * 8
    available = (
        argument_max - environment_bytes - prefix_bytes - pointer_bytes - LSOF_EXEC_HEADROOM_BYTES
    )
    return max(0, min(MAX_LSOF_PAYLOAD_BYTES, available))


def _lsof_file_batches(
    paths: tuple[Path, ...],
    *,
    payload_budget: int,
) -> tuple[tuple[tuple[Path, ...], ...], tuple[Path, ...]]:
    batches: list[tuple[Path, ...]] = []
    oversized: list[Path] = []
    pending: list[Path] = []
    pending_bytes = 0
    for path in paths:
        path_bytes = len(os.fsencode(path)) + 1
        if path_bytes > payload_budget:
            oversized.append(path)
            continue
        if pending and (
            len(pending) >= MAX_LSOF_BATCH_CANDIDATES or pending_bytes + path_bytes > payload_budget
        ):
            batches.append(tuple(pending))
            pending = []
            pending_bytes = 0
        pending.append(path)
        pending_bytes += path_bytes
    if pending:
        batches.append(tuple(pending))
    return tuple(batches), tuple(oversized)


def _batched_open_checker(
    specs: tuple[tuple[str, Path, CandidateKind], ...],
    *,
    deadline: _OperationDeadline | None = None,
) -> OpenChecker:
    """Check exact files in bounded batches and directories one at a time."""

    active_deadline = deadline or _new_lsof_deadline()
    checked: dict[Path, tuple[OpenStatus, str]] = {}
    paths_by_kind = {
        kind: tuple(
            sorted(
                {
                    lexical_absolute(path)
                    for _category, path, candidate_kind in specs
                    if candidate_kind == kind
                },
                key=os.fspath,
            )
        )
        for kind in ("file", "directory")
    }
    for path in (*paths_by_kind["file"], *paths_by_kind["directory"]):
        if _contains_unsafe_path_text(os.fspath(path)):
            checked[path] = (
                "unknown",
                "path_contains_control_or_formatting_characters",
            )

    lsof = shutil.which("lsof")
    if lsof is None:
        return lambda path, _kind: checked.get(
            lexical_absolute(path),
            ("unknown", "lsof_unavailable"),
        )

    file_paths = tuple(path for path in paths_by_kind["file"] if path not in checked)
    prefix = (lsof, "-Fn", "--")
    batches, oversized = _lsof_file_batches(
        file_paths,
        payload_budget=_lsof_payload_budget(prefix),
    )
    for path in oversized:
        checked[path] = ("unknown", "lsof_argument_too_long")

    def run(
        command: list[str],
        affected: tuple[Path, ...],
        *,
        directory: bool = False,
    ) -> None:
        command_timeout = active_deadline.command_timeout()
        if command_timeout is None:
            for path in affected:
                checked[path] = (
                    "unknown",
                    "lsof_operation_deadline_exceeded",
                )
            return
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=command_timeout,
            )
        except subprocess.TimeoutExpired:
            detail = (
                "lsof_operation_deadline_exceeded"
                if active_deadline.remaining() <= 0
                else "lsof_timed_out"
            )
            for path in affected:
                checked[path] = ("unknown", detail)
            return
        except (OSError, UnicodeError) as exc:
            detail = f"lsof_failed:{type(exc).__name__}"
            for path in affected:
                checked[path] = ("unknown", detail)
            return
        if active_deadline.remaining() <= 0:
            for path in affected:
                checked[path] = (
                    "unknown",
                    "lsof_operation_deadline_exceeded",
                )
            return
        if result.stderr.strip():
            for path in affected:
                checked[path] = ("unknown", "lsof_reported_errors")
            return
        if result.returncode == 1:
            for path in affected:
                checked[path] = ("clear", "")
            return
        if result.returncode != 0:
            detail = f"lsof_exit_{result.returncode}"
            for path in affected:
                checked[path] = ("unknown", detail)
            return
        names = {
            line[1:]
            for line in result.stdout.splitlines()
            if line.startswith("n") and line[1:].startswith(os.sep)
        }
        if not names:
            for path in affected:
                checked[path] = ("unknown", "lsof_output_missing_names")
            return
        for path in affected:
            if directory or os.fspath(path) in names:
                checked[path] = ("open", "open_files_detected")
            else:
                checked[path] = ("clear", "")

    for batch in batches:
        run([*prefix, *(os.fspath(path) for path in batch)], batch)
    directory_budget = _lsof_payload_budget((lsof, "-Fn", "+D"))
    for path in paths_by_kind["directory"]:
        if path in checked:
            continue
        if len(os.fsencode(path)) + 1 > directory_budget:
            checked[path] = ("unknown", "lsof_argument_too_long")
            continue
        run([lsof, "-Fn", "+D", os.fspath(path)], (path,), directory=True)

    def check(path: Path, _kind: CandidateKind) -> tuple[OpenStatus, str]:
        return checked.get(
            lexical_absolute(path),
            ("unknown", "lsof_candidate_not_checked"),
        )

    return check


def _capture_candidate(
    *,
    home: Path,
    category: str,
    path: Path,
    kind: CandidateKind,
    stale_days: int,
    now_ns: int,
    open_checker: OpenChecker,
    mount_points: frozenset[Path] | None = None,
) -> StorageCandidate:
    path = lexical_absolute(path)
    try:
        path.relative_to(home)
    except ValueError as exc:
        raise StorageSafetyError("candidate escapes the audited home") from exc

    path_stat = os.lstat(path)
    if kind == "directory":
        if not stat.S_ISDIR(path_stat.st_mode) or is_linklike(path_stat):
            raise StorageSafetyError(f"candidate is no longer a real directory: {path}")
    elif kind == "file":
        if not stat.S_ISREG(path_stat.st_mode) or is_linklike(path_stat):
            raise StorageSafetyError(f"candidate is no longer a regular file: {path}")
    elif not is_linklike(path_stat) or path.exists():
        raise StorageSafetyError(f"candidate is no longer a broken link: {path}")

    measurement = measure_tree(
        path,
        mount_points=mount_points or frozenset(),
    )
    blocked: list[str] = []
    if _contains_unsafe_path_text(os.fspath(path)):
        blocked.append("path_contains_control_or_formatting_characters")
    age_reason = _age_block(path_stat, stale_days=stale_days, now_ns=now_ns)
    if age_reason:
        blocked.append(age_reason)
    if measurement.crosses_device:
        blocked.append("contains_cross_device_entry")
    if mount_points is None:
        blocked.append("mount_inventory_unavailable")
    else:
        if _mount_strictly_between(home, path, mount_points) is not None:
            blocked.append("mounted_ancestor_between_home_and_candidate")
        if measurement.crosses_mount:
            blocked.append("contains_mount_point")
    pid_reason = _pid_marker_block(path) if kind == "file" else None
    if pid_reason:
        blocked.append(pid_reason)
    if kind == "broken_symlink":
        open_status: OpenStatus = "clear"
        open_detail = ""
    else:
        open_status, open_detail = open_checker(path, kind)
    if open_status != "clear":
        blocked.append(open_detail or f"open_status_{open_status}")

    return StorageCandidate(
        candidate_id=_candidate_id(category, home, path),
        category=category,
        path=os.fspath(path),
        kind=kind,
        device=path_stat.st_dev,
        inode=path_stat.st_ino,
        mode=path_stat.st_mode,
        size=path_stat.st_size,
        mtime_ns=path_stat.st_mtime_ns,
        ctime_ns=path_stat.st_ctime_ns,
        logical_bytes=measurement.logical_bytes,
        allocated_bytes=measurement.allocated_bytes,
        estimated_reclaim_bytes=measurement.estimated_reclaim_bytes,
        entry_count=measurement.entry_count,
        tree_digest=measurement.tree_digest,
        open_status=open_status,
        blocked_reasons=tuple(sorted(set(blocked))),
    )


def _walk_directories(root: Path, *, max_depth: int) -> Iterator[Path]:
    if not root.exists():
        return
    pending: list[tuple[Path, int]] = [(root, 0)]
    while pending:
        directory, depth = pending.pop()
        try:
            directory_stat = os.lstat(directory)
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(directory_stat.st_mode) or is_linklike(directory_stat):
            continue
        yield directory
        if depth >= max_depth:
            continue
        children: list[Path] = []
        for entry in os.scandir(directory):
            entry_path = Path(entry.path)
            try:
                entry_stat = os.lstat(entry_path)
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(entry_stat.st_mode) and not is_linklike(entry_stat):
                children.append(entry_path)
        pending.extend((path, depth + 1) for path in sorted(children, reverse=True))


def _is_real_directory(path: Path) -> bool:
    try:
        path_stat = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(path_stat.st_mode) and not is_linklike(path_stat)


def _is_trusted_directory(home: Path, path: Path) -> bool:
    try:
        assert_no_linklike_components(path, boundary=home, allow_missing=False)
    except (FileNotFoundError, ValueError):
        return False
    return _is_real_directory(path)


def _candidate_specs(home: Path) -> Iterator[tuple[str, Path, CandidateKind]]:
    yaze_root = home / ".yaze" / "nightly"
    if _is_trusted_directory(home, yaze_root):
        for directory in _walk_directories(yaze_root, max_depth=5):
            if directory != yaze_root and directory.name.startswith("build-nightly"):
                yield "yaze_build", directory, "directory"

    snapshots = home / ".local" / "share" / "opencode" / "snapshot"
    if _is_trusted_directory(home, snapshots):
        for snapshot in sorted(snapshots.iterdir()):
            if not _SNAPSHOT_NAME.fullmatch(snapshot.name):
                continue
            if not _is_trusted_directory(home, snapshot):
                continue
            pack_dir = snapshot / "objects" / "pack"
            if not _is_trusted_directory(home, pack_dir):
                continue
            for entry in sorted(pack_dir.iterdir()):
                if entry.name != "gc.pid" and not entry.name.startswith("tmp_pack_"):
                    continue
                try:
                    entry_stat = os.lstat(entry)
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(entry_stat.st_mode) and not is_linklike(entry_stat):
                    yield "opencode_temp", entry, "file"

    server_logs = home / ".lmstudio" / "server-logs"
    if _is_trusted_directory(home, server_logs):
        for directory in _walk_directories(server_logs, max_depth=3):
            for entry in sorted(directory.iterdir()):
                try:
                    entry_stat = os.lstat(entry)
                except FileNotFoundError:
                    continue
                if (
                    stat.S_ISREG(entry_stat.st_mode)
                    and not is_linklike(entry_stat)
                    and entry.name.endswith(".log")
                ):
                    yield "lmstudio_log", entry, "file"

    desktop_logs = home / "Library" / "Logs" / "LM Studio"
    if _is_trusted_directory(home, desktop_logs):
        for entry in sorted(desktop_logs.glob("*.old.log")):
            try:
                entry_stat = os.lstat(entry)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(entry_stat.st_mode) and not is_linklike(entry_stat):
                yield "lmstudio_log", entry, "file"

    models = home / ".lmstudio" / "models"
    if _is_trusted_directory(home, models):
        for directory in _walk_directories(models, max_depth=4):
            for entry in sorted(directory.iterdir()):
                try:
                    entry_stat = os.lstat(entry)
                except FileNotFoundError:
                    continue
                if is_linklike(entry_stat) and not entry.exists():
                    yield "lmstudio_broken_link", entry, "broken_symlink"


def discover_candidates(
    home: Path,
    *,
    stale_days: int = DEFAULT_STALE_DAYS,
    now_ns: int | None = None,
    open_checker: OpenChecker = default_open_checker,
    mount_reader: MountReader = default_mount_reader,
    lsof_deadline: _OperationDeadline | None = None,
) -> tuple[StorageCandidate, ...]:
    """Discover only bounded, rebuildable cleanup candidates."""

    if stale_days < 1 or stale_days > 3650:
        raise ValueError("stale_days must be between 1 and 3650")
    trusted_home, _home_identity = _trusted_home(home)
    timestamp = time.time_ns() if now_ns is None else now_ns
    mount_points = mount_reader()
    specs = tuple(_candidate_specs(trusted_home))
    candidate_open_checker = (
        _batched_open_checker(specs, deadline=lsof_deadline)
        if open_checker is default_open_checker
        and any(kind != "broken_symlink" for _category, _path, kind in specs)
        else open_checker
    )
    candidates: list[StorageCandidate] = []
    seen_paths: set[Path] = set()
    covered_directories: list[Path] = []
    for category, path, kind in specs:
        candidate_path = lexical_absolute(path)
        if candidate_path in seen_paths:
            continue
        if any(candidate_path.is_relative_to(parent) for parent in covered_directories):
            continue
        seen_paths.add(candidate_path)
        unsafe_path = _contains_unsafe_path_text(os.fspath(candidate_path))
        assert_no_linklike_components(
            candidate_path.parent,
            boundary=trusted_home,
            allow_missing=False,
        )
        try:
            candidate = _capture_candidate(
                home=trusted_home,
                category=category,
                path=candidate_path,
                kind=kind,
                stale_days=stale_days,
                now_ns=timestamp,
                open_checker=(
                    (
                        lambda _path, _kind: (
                            "unknown",
                            "path_contains_control_or_formatting_characters",
                        )
                    )
                    if unsafe_path
                    else candidate_open_checker
                ),
                mount_points=mount_points,
            )
        except FileNotFoundError:
            continue
        candidates.append(candidate)
        if kind == "directory":
            covered_directories.append(candidate_path)
    return tuple(sorted(candidates, key=lambda item: (item.category, item.path)))


def _footprint(
    name: str,
    path: Path,
    note: str,
    *,
    mount_points: frozenset[Path] | None,
) -> StorageFootprint:
    if not os.path.lexists(path):
        return StorageFootprint(name, os.fspath(path), 0, 0, 0, note)
    if mount_points is None:
        return StorageFootprint(
            name,
            os.fspath(path),
            0,
            0,
            0,
            note,
            "mount inventory unavailable; footprint was not traversed",
        )
    try:
        measurement = measure_tree(path, mount_points=mount_points)
    except OSError as exc:
        return StorageFootprint(
            name,
            os.fspath(path),
            0,
            0,
            0,
            note,
            f"{type(exc).__name__}: {exc}",
        )
    return StorageFootprint(
        name=name,
        path=os.fspath(path),
        logical_bytes=measurement.logical_bytes,
        allocated_bytes=measurement.allocated_bytes,
        entry_count=measurement.entry_count,
        note=note,
    )


def default_snapshot_reader() -> tuple[str, ...]:
    """List local APFS/Time Machine snapshots without mutating them."""

    if sys.platform != "darwin":
        return ()
    tmutil = shutil.which("tmutil")
    if tmutil is None:
        return ()
    try:
        result = subprocess.run(
            [tmutil, "listlocalsnapshots", "/"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if result.returncode != 0:
        return ()
    return tuple(
        snapshot
        for line in result.stdout.splitlines()
        if (snapshot := line.strip()).startswith("com.apple.")
    )


def _pressure(free_bytes: int) -> str:
    free_gib = free_bytes / GIB
    if free_gib >= 450:
        return "healthy"
    if free_gib >= 400:
        return "watch"
    if free_gib >= 300:
        return "warning"
    return "critical"


def _huggingface_footprints(
    home: Path,
    *,
    mount_points: frozenset[Path] | None,
) -> tuple[StorageFootprint, ...]:
    default_root = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache")) / "huggingface"
    hf_home = lexical_absolute(
        Path(os.environ.get("HF_HOME", os.fspath(default_root))).expanduser()
    )
    cache_note = "informational; cache cleanup requires a separate reviewed policy"
    roots: list[tuple[str, Path, str]] = [
        ("huggingface_cache", hf_home, cache_note),
    ]
    for name, source in (
        ("huggingface_hub_cache", "HF_HUB_CACHE"),
        ("huggingface_legacy_hub_cache", "HUGGINGFACE_HUB_CACHE"),
    ):
        value = os.environ.get(source)
        if not value:
            continue
        cache = lexical_absolute(Path(value).expanduser())
        if any(cache == root or cache.is_relative_to(root) for _name, root, _note in roots):
            continue
        roots = [record for record in roots if not record[1].is_relative_to(cache)]
        roots.append(
            (
                name,
                cache,
                f"informational; external {source} cache requires reviewed cleanup",
            )
        )
    return tuple(
        _footprint(name, path, note, mount_points=mount_points) for name, path, note in roots
    )


def audit_storage(
    home: Path,
    *,
    stale_days: int = DEFAULT_STALE_DAYS,
    now_ns: int | None = None,
    open_checker: OpenChecker = default_open_checker,
    snapshot_reader: SnapshotReader = default_snapshot_reader,
    mount_reader: MountReader = default_mount_reader,
) -> dict[str, Any]:
    """Return one read-only storage audit."""

    trusted_home, _home_identity = _trusted_home(home)
    timestamp = time.time_ns() if now_ns is None else now_ns
    _validate_timestamp_ns(timestamp, field="audit timestamp")
    usage = shutil.disk_usage(trusted_home)
    mount_points = mount_reader()
    candidates = discover_candidates(
        trusted_home,
        stale_days=stale_days,
        now_ns=timestamp,
        open_checker=open_checker,
        mount_reader=lambda: mount_points,
    )
    ollama_models = os.environ.get("OLLAMA_MODELS")
    if not ollama_models:
        ollama_models = os.environ.get(
            "OLLAMA_MODELS_DIR",
            os.fspath(trusted_home / ".ollama" / "models"),
        )
    footprints = (
        _footprint(
            "local_models",
            trusted_home / "models",
            "informational; never auto-delete",
            mount_points=mount_points,
        ),
        _footprint(
            "ollama_models",
            Path(ollama_models),
            "informational; remove only through an explicit model lifecycle decision",
            mount_points=mount_points,
        ),
        *_huggingface_footprints(trusted_home, mount_points=mount_points),
        _footprint(
            "lmstudio_models",
            trusted_home / ".lmstudio" / "models",
            "informational; only broken links can enter a cleanup plan",
            mount_points=mount_points,
        ),
        _footprint(
            "archives",
            trusted_home / "Archives",
            "informational; never auto-delete",
            mount_points=mount_points,
        ),
        _footprint(
            "trash",
            trusted_home / ".Trash",
            "informational; never auto-delete",
            mount_points=mount_points,
        ),
        _footprint(
            "afs_context",
            trusted_home / ".context",
            "informational; never auto-delete",
            mount_points=mount_points,
        ),
    )
    candidate_records = [
        asdict(candidate) | {"eligible": candidate.eligible} for candidate in candidates
    ]
    reclaim = sum(
        candidate.estimated_reclaim_bytes for candidate in candidates if candidate.eligible
    )
    return {
        "schema": "afs.storage.audit.v1",
        "generated_at": _iso_from_ns(timestamp),
        "home": os.fspath(trusted_home),
        "stale_days": stale_days,
        "disk": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "pressure": _pressure(usage.free),
            "thresholds_gib": {
                "healthy": 450,
                "watch": 400,
                "warning": 300,
            },
        },
        "candidates": candidate_records,
        "eligible_count": sum(candidate.eligible for candidate in candidates),
        "estimated_reclaim_bytes": reclaim,
        "footprints": [asdict(item) for item in footprints],
        "local_snapshots": list(snapshot_reader()),
        "safety": {
            "online": True,
            "processes_stopped": False,
            "protected_categories": list(PROTECTED_CATEGORIES),
            "note": (
                "Allocated bytes are filesystem accounting, not a guaranteed reclaim "
                "estimate; APFS clones and snapshots may share blocks."
            ),
        },
    }


def _candidate_from_audit_record(record: Any) -> StorageCandidate:
    if not isinstance(record, dict):
        raise StorageSafetyError("storage audit candidate must be an object")
    eligible = record.get("eligible")
    if not isinstance(eligible, bool):
        raise StorageSafetyError("storage audit candidate eligibility is invalid")
    blocked = record.get("blocked_reasons")
    if not isinstance(blocked, (list, tuple)) or any(
        not isinstance(reason, str) for reason in blocked
    ):
        raise StorageSafetyError("storage audit candidate blockers are invalid")
    reasons = tuple(reason for reason in blocked if isinstance(reason, str))
    plan_record = {
        key: value for key, value in record.items() if key not in {"blocked_reasons", "eligible"}
    }
    candidate = replace(
        _candidate_from_record(plan_record),
        blocked_reasons=reasons,
    )
    if candidate.eligible != eligible:
        raise StorageSafetyError("storage audit candidate eligibility is inconsistent")
    return candidate


def build_storage_plan(
    home: Path,
    *,
    stale_days: int = DEFAULT_STALE_DAYS,
    now_ns: int | None = None,
    open_checker: OpenChecker = default_open_checker,
    snapshot_reader: SnapshotReader = default_snapshot_reader,
    mount_reader: MountReader = default_mount_reader,
    mount_identity_reader: MountIdentityReader | None = None,
) -> dict[str, Any]:
    """Build a canonical, expiring plan from currently eligible candidates."""

    timestamp = time.time_ns() if now_ns is None else now_ns
    _validate_timestamp_ns(timestamp, field="plan timestamp")
    expires_at_ns = timestamp + PLAN_TTL_SECONDS * 1_000_000_000
    _validate_timestamp_ns(expires_at_ns, field="plan expiry")
    trusted_home, home_identity = _trusted_home(home)
    identity_reader = mount_identity_reader or default_mount_identity_reader
    home_mount_identity = identity_reader(trusted_home)
    audit = audit_storage(
        trusted_home,
        stale_days=stale_days,
        now_ns=timestamp,
        open_checker=open_checker,
        snapshot_reader=snapshot_reader,
        mount_reader=mount_reader,
    )
    if identity_reader(trusted_home) != home_mount_identity:
        raise StorageSafetyError("home mount identity changed while building the plan")
    audited_candidates = tuple(
        _candidate_from_audit_record(record) for record in audit["candidates"]
    )
    eligible = [candidate for candidate in audited_candidates if candidate.eligible]
    if len(eligible) > MAX_PLAN_CANDIDATES:
        raise StorageSafetyError(
            f"storage plan exceeds the {MAX_PLAN_CANDIDATES:,} candidate limit"
        )
    base: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "created_at": _iso_from_ns(timestamp),
        "created_at_ns": timestamp,
        "expires_at_ns": expires_at_ns,
        "home": audit["home"],
        "home_identity": home_identity,
        "home_mount_identity": _mount_identity_record(home_mount_identity),
        "stale_days": stale_days,
        "disk": audit["disk"],
        "candidates": [candidate.plan_record() for candidate in eligible],
        "estimated_reclaim_bytes": sum(candidate.estimated_reclaim_bytes for candidate in eligible),
        "protected_categories": audit["safety"]["protected_categories"],
        "snapshots_are_informational_only": True,
    }
    digest = hashlib.sha256(_canonical_json(base)).hexdigest()
    plan = base | {
        "plan_sha256": digest,
        "transaction": f"storage_{digest[:32]}",
    }
    if len(json.dumps(plan, indent=2, sort_keys=True).encode("utf-8")) > MAX_PLAN_BYTES:
        raise StorageSafetyError("storage plan exceeds the 8 MiB limit")
    return plan


def write_storage_plan(path: Path, plan: dict[str, Any]) -> Path:
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > MAX_PLAN_CANDIDATES:
        raise StorageSafetyError("storage plan candidates are invalid or excessive")
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if len(rendered.encode("utf-8")) > MAX_PLAN_BYTES:
        raise StorageSafetyError("storage plan exceeds the 8 MiB limit")
    output = lexical_absolute(path)
    secure_mkdir(output.parent, mode=0o700)
    atomic_create_text(
        output,
        rendered,
        mode=0o600,
        durable=True,
    )
    return output


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StorageSafetyError(f"plan contains duplicate field {key!r}")
        result[key] = value
    return result


def load_storage_plan(path: Path) -> dict[str, Any]:
    """Load and hash-check one regular, no-follow plan."""

    plan_path = lexical_absolute(path)
    path_stat = os.lstat(plan_path)
    if not stat.S_ISREG(path_stat.st_mode) or is_linklike(path_stat):
        raise StorageSafetyError("storage plan must be a regular file")
    if path_stat.st_size > MAX_PLAN_BYTES:
        raise StorageSafetyError("storage plan exceeds the 8 MiB limit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(plan_path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            path_stat.st_dev,
            path_stat.st_ino,
            path_stat.st_size,
            path_stat.st_mtime_ns,
            path_stat.st_ctime_ns,
        ):
            raise StorageSafetyError("storage plan changed while opening")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 64 * 1024):
            total += len(chunk)
            if total > MAX_PLAN_BYTES:
                raise StorageSafetyError("storage plan exceeds the 8 MiB limit")
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        final_path_stat = os.lstat(plan_path)
    except FileNotFoundError as exc:
        raise StorageSafetyError("storage plan changed while reading") from exc
    signatures = {
        (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        for item in (path_stat, opened, after, final_path_stat)
    }
    if len(raw) > MAX_PLAN_BYTES or len(signatures) != 1:
        raise StorageSafetyError("storage plan changed while reading")

    def reject_nonfinite(value: str) -> None:
        raise StorageSafetyError(f"storage plan contains non-finite number {value}")

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StorageSafetyError(f"invalid storage plan JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise StorageSafetyError("storage plan must be a JSON object")
    expected = {
        "schema",
        "created_at",
        "created_at_ns",
        "expires_at_ns",
        "home",
        "home_identity",
        "home_mount_identity",
        "stale_days",
        "disk",
        "candidates",
        "estimated_reclaim_bytes",
        "protected_categories",
        "snapshots_are_informational_only",
        "plan_sha256",
        "transaction",
    }
    if set(payload) != expected:
        raise StorageSafetyError("storage plan fields do not match the v1 schema")
    if payload.get("schema") != PLAN_SCHEMA:
        raise StorageSafetyError("unsupported storage plan schema")
    created_at_ns = _validate_timestamp_ns(
        payload.get("created_at_ns"),
        field="storage plan created_at_ns",
    )
    expires_at_ns = _validate_timestamp_ns(
        payload.get("expires_at_ns"),
        field="storage plan expires_at_ns",
    )
    if (
        expires_at_ns <= created_at_ns
        or expires_at_ns - created_at_ns > PLAN_TTL_SECONDS * 1_000_000_000
    ):
        raise StorageSafetyError("storage plan expiry exceeds the allowed lifetime")
    if type(payload.get("stale_days")) is not int or not 1 <= payload["stale_days"] <= 3650:
        raise StorageSafetyError("storage plan stale_days is invalid")
    home = payload.get("home")
    if (
        not isinstance(home, str)
        or not Path(home).is_absolute()
        or _contains_unsafe_path_text(home)
    ):
        raise StorageSafetyError("storage plan home must be an absolute path")
    home_identity = payload.get("home_identity")
    if (
        not isinstance(home_identity, dict)
        or set(home_identity) != {"device", "inode", "mode"}
        or any(type(value) is not int for value in home_identity.values())
        or any(not 0 <= value <= MAX_INT64 for value in home_identity.values())
    ):
        raise StorageSafetyError("storage plan home identity is invalid")
    _mount_identity_from_record(payload.get("home_mount_identity"))
    if not isinstance(payload.get("created_at"), str):
        raise StorageSafetyError("storage plan created_at must be a string")
    if payload["created_at"] != _iso_from_ns(created_at_ns):
        raise StorageSafetyError("storage plan created_at does not match created_at_ns")
    reclaim_total = payload.get("estimated_reclaim_bytes")
    if type(reclaim_total) is not int or not 0 <= reclaim_total <= MAX_INT64:
        raise StorageSafetyError("storage plan reclaim total must be a non-negative 64-bit integer")
    if payload.get("snapshots_are_informational_only") is not True:
        raise StorageSafetyError("storage plans cannot authorize snapshot deletion")
    if payload.get("protected_categories") != list(PROTECTED_CATEGORIES):
        raise StorageSafetyError("storage plan protected categories are invalid")
    if not isinstance(payload.get("disk"), dict):
        raise StorageSafetyError("storage plan disk snapshot is invalid")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > MAX_PLAN_CANDIDATES:
        raise StorageSafetyError("storage plan candidates are invalid or excessive")
    base = {
        key: value for key, value in payload.items() if key not in {"plan_sha256", "transaction"}
    }
    digest = hashlib.sha256(_canonical_json(base)).hexdigest()
    if payload.get("plan_sha256") != digest:
        raise StorageSafetyError("storage plan hash does not match its contents")
    if payload.get("transaction") != f"storage_{digest[:32]}":
        raise StorageSafetyError("storage transaction does not match the plan hash")
    return payload


def _validate_rationale(value: Any) -> str:
    if not isinstance(value, str):
        raise StorageSafetyError("apply requires a string --because rationale")
    rationale = value.strip()
    if not rationale:
        raise StorageSafetyError("apply requires a non-empty --because rationale")
    if rationale != value:
        raise StorageSafetyError("rationale must not have surrounding whitespace")
    if len(rationale) > MAX_RATIONALE_CHARS:
        raise StorageSafetyError("rationale exceeds 4096 characters")
    if any(unicodedata.category(character).startswith("C") for character in rationale):
        raise StorageSafetyError("rationale contains control or formatting characters")
    return rationale


def _candidate_from_record(record: Any) -> StorageCandidate:
    if not isinstance(record, dict):
        raise StorageSafetyError("storage plan candidate must be an object")
    expected = {
        field for field in StorageCandidate.__dataclass_fields__ if field != "blocked_reasons"
    }
    if set(record) != expected:
        raise StorageSafetyError("storage plan candidate fields are invalid")
    string_fields = {"candidate_id", "category", "path", "kind", "tree_digest", "open_status"}
    integer_fields = expected - string_fields
    if any(not isinstance(record[field], str) for field in string_fields):
        raise StorageSafetyError("storage plan candidate string fields are invalid")
    if any(type(record[field]) is not int for field in integer_fields):
        raise StorageSafetyError("storage plan candidate integer fields are invalid")
    if record["kind"] not in {"file", "directory", "broken_symlink"}:
        raise StorageSafetyError("storage plan candidate kind is invalid")
    if record["category"] not in ALLOWED_CATEGORIES:
        raise StorageSafetyError("storage plan candidate category is invalid")
    if record["open_status"] != "clear":
        raise StorageSafetyError("storage plan candidate was not clear when planned")
    if not Path(record["path"]).is_absolute() or _contains_unsafe_path_text(record["path"]):
        raise StorageSafetyError("storage plan candidate path must be absolute")
    if any(not 0 <= record[field] <= MAX_INT64 for field in integer_fields):
        raise StorageSafetyError(
            "storage plan candidate integer fields must be non-negative 64-bit values"
        )
    if not _HEX_DIGEST.fullmatch(record["tree_digest"]):
        raise StorageSafetyError("storage plan candidate tree digest is invalid")
    try:
        return StorageCandidate(
            candidate_id=record["candidate_id"],
            category=record["category"],
            path=record["path"],
            kind=record["kind"],
            device=record["device"],
            inode=record["inode"],
            mode=record["mode"],
            size=record["size"],
            mtime_ns=record["mtime_ns"],
            ctime_ns=record["ctime_ns"],
            logical_bytes=record["logical_bytes"],
            allocated_bytes=record["allocated_bytes"],
            estimated_reclaim_bytes=record["estimated_reclaim_bytes"],
            entry_count=record["entry_count"],
            tree_digest=record["tree_digest"],
            open_status=record["open_status"],
            blocked_reasons=(),
        )
    except (TypeError, ValueError) as exc:
        raise StorageSafetyError(f"storage plan candidate is invalid: {exc}") from exc


def _identity_record(candidate: StorageCandidate) -> dict[str, Any]:
    return candidate.plan_record()


def _assert_current_candidate(
    planned: StorageCandidate,
    current: StorageCandidate,
) -> None:
    if _identity_record(planned) != _identity_record(current):
        raise StorageSafetyError(f"candidate drifted since planning: {planned.candidate_id}")
    if not current.eligible:
        raise StorageSafetyError(f"candidate is no longer eligible: {planned.candidate_id}")


def _mount_identity_for_fd(descriptor: int) -> tuple[int, int]:
    path_stat = os.fstat(descriptor)
    if sys.platform.startswith("linux"):
        try:
            payload = Path(f"/proc/self/fdinfo/{descriptor}").read_text(
                encoding="ascii",
                errors="strict",
            )
        except (OSError, UnicodeError) as exc:
            raise StorageSafetyError("cannot prove the directory mount identity") from exc
        match = re.search(r"^mnt_id:\s*(\d+)$", payload, flags=re.MULTILINE)
        if match is None:
            raise StorageSafetyError("cannot prove the directory mount identity")
        return path_stat.st_dev, int(match.group(1))
    if sys.platform == "darwin":
        # Darwin has no procfs mount ID.  st_dev plus a fresh system mount-point
        # inventory is the strongest no-privilege identity available here.
        return path_stat.st_dev, 0
    raise StorageSafetyError("this platform cannot prove directory mount boundaries")


def default_mount_identity_reader(path: Path) -> MountIdentity:
    """Read one no-follow directory mount identity."""

    descriptor = os.open(path, _directory_open_flags())
    try:
        return _mount_identity_for_fd(descriptor)
    finally:
        os.close(descriptor)


def _mount_identity_record(identity: MountIdentity) -> dict[str, int]:
    device, mount_id = identity
    if not 0 <= device <= MAX_INT64 or not 0 <= mount_id <= MAX_INT64:
        raise StorageSafetyError("home mount identity is outside the supported range")
    return {"device": device, "mount_id": mount_id}


def _mount_identity_from_record(record: Any) -> MountIdentity:
    if (
        not isinstance(record, dict)
        or set(record) != {"device", "mount_id"}
        or any(type(value) is not int for value in record.values())
        or any(not 0 <= value <= MAX_INT64 for value in record.values())
    ):
        raise StorageSafetyError("storage plan home mount identity is invalid")
    return record["device"], record["mount_id"]


def _mount_at_or_below(path: Path, mount_points: frozenset[Path]) -> Path | None:
    for mount_point in mount_points:
        if mount_point == path or mount_point.is_relative_to(path):
            return mount_point
    return None


def _read_mounts_or_raise(mount_reader: MountReader) -> frozenset[Path]:
    mount_points = mount_reader()
    if mount_points is None:
        raise StorageSafetyError("mount inventory is unavailable")
    return frozenset(lexical_absolute(path) for path in mount_points)


def _assert_candidate_parent_on_home_mount(
    candidate: StorageCandidate,
    *,
    home: Path,
    home_mount_identity: MountIdentity,
) -> None:
    path = lexical_absolute(Path(candidate.path))
    try:
        path.relative_to(home)
    except ValueError as exc:
        raise StorageSafetyError("candidate escapes its pinned home") from exc
    assert_no_linklike_components(
        path.parent,
        boundary=home,
        allow_missing=False,
    )
    descriptor = os.open(path.parent, _directory_open_flags())
    try:
        if _mount_identity_for_fd(descriptor) != home_mount_identity:
            raise StorageSafetyError("candidate parent is not on the plan's pinned home mount")
    finally:
        os.close(descriptor)


def _directory_open_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise StorageSafetyError("this platform cannot open directories without following links")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_directory_beneath(
    home: Path,
    directory: Path,
    *,
    home_mount_identity: MountIdentity,
) -> int:
    """Open one directory by walking beneath the trusted home without links."""

    trusted_home = lexical_absolute(home)
    target = lexical_absolute(directory)
    try:
        relative = target.relative_to(trusted_home)
    except ValueError as exc:
        raise StorageSafetyError("directory escapes its pinned home") from exc

    flags = _directory_open_flags()
    descriptor = os.open(trusted_home, flags)
    try:
        home_stat = os.fstat(descriptor)
        current_home_stat = os.stat(trusted_home, follow_symlinks=False)
        if (
            not stat.S_ISDIR(home_stat.st_mode)
            or is_linklike(current_home_stat)
            or (home_stat.st_dev, home_stat.st_ino)
            != (current_home_stat.st_dev, current_home_stat.st_ino)
        ):
            raise StorageSafetyError("pinned home changed while opening a directory")
        if _mount_identity_for_fd(descriptor) != home_mount_identity:
            raise StorageSafetyError("pinned home mount changed while opening a directory")

        parts = () if relative == Path(".") else relative.parts
        for part in parts:
            following = os.open(part, flags, dir_fd=descriptor)
            try:
                following_stat = os.fstat(following)
                if not stat.S_ISDIR(following_stat.st_mode):
                    raise StorageSafetyError("opened path component is not a directory")
                if _mount_identity_for_fd(following) != home_mount_identity:
                    raise StorageSafetyError("opened path component crosses the pinned home mount")
            except BaseException:
                os.close(following)
                raise
            os.close(descriptor)
            descriptor = following
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _assert_directory_fd_matches_path(
    descriptor: int,
    path: Path,
    *,
    label: str,
) -> os.stat_result:
    """Prove that a reporting path still names one already-opened directory."""

    opened = os.fstat(descriptor)
    current = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or is_linklike(current)
        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
    ):
        raise StorageSafetyError(f"{label} directory changed after it was opened")
    return opened


def _renameat_noreplace(
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
) -> None:
    """Atomically rename basenames between pinned directory descriptors."""

    for name in (source_name, destination_name):
        if (
            not name
            or name in {".", ".."}
            or Path(name).name != name
            or _contains_unsafe_path_text(name)
        ):
            raise StorageSafetyError("descriptor-relative rename requires a safe basename")

    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and getattr(library, "renameatx_np", None) is not None:
        renameatx_np = library.renameatx_np
        renameatx_np.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            source_descriptor,
            os.fsencode(source_name),
            destination_descriptor,
            os.fsencode(destination_name),
            _RENAME_EXCL,
        )
    elif sys.platform.startswith("linux") and getattr(library, "renameat2", None) is not None:
        renameat2 = library.renameat2
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            source_descriptor,
            os.fsencode(source_name),
            destination_descriptor,
            os.fsencode(destination_name),
            _RENAME_NOREPLACE,
        )
    else:
        raise OSError(errno.ENOTSUP, "descriptor-relative no-replace rename is unavailable")
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination_name)


def _strict_fsync_directory_fd(descriptor: int) -> None:
    """Durably sync one already-pinned directory descriptor."""

    directory_stat = os.fstat(descriptor)
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise NotADirectoryError("pinned descriptor is not a directory")
    if directory_stat.st_nlink < 1:
        raise OSError("pinned directory has no durable link")
    os.fsync(descriptor)


def _update_tree_digests(
    exact: Any,
    stable: Any,
    *,
    relative: str,
    metadata: os.stat_result,
    target: str | None,
    root: bool,
) -> None:
    def line(ctime_ns: int) -> bytes:
        return (
            f"{relative}\0{metadata.st_dev}\0{metadata.st_ino}\0"
            f"{metadata.st_mode}\0{metadata.st_size}\0"
            f"{metadata.st_mtime_ns}\0{ctime_ns}\0{target or ''}\n"
        ).encode("utf-8", errors="surrogateescape")

    exact.update(line(metadata.st_ctime_ns))
    stable.update(line(0 if root else metadata.st_ctime_ns))


def _measure_tree_from_fd(descriptor: int) -> _PinnedTreeMeasurement:
    """Measure a directory tree without resolving its external path."""

    exact_digest = hashlib.sha256()
    stable_digest = hashlib.sha256()
    logical = 0
    allocated = 0
    estimated_reclaim = 0
    entries = 0
    crosses_device = False
    crosses_mount = False
    seen: set[tuple[int, int]] = set()
    root_stat = os.fstat(descriptor)
    root_device = root_stat.st_dev
    root_mount_identity = _mount_identity_for_fd(descriptor)

    def record(
        metadata: os.stat_result,
        *,
        relative: str,
        target: str | None,
        root: bool,
    ) -> None:
        nonlocal logical, allocated, estimated_reclaim, entries, crosses_device
        _update_tree_digests(
            exact_digest,
            stable_digest,
            relative=relative,
            metadata=metadata,
            target=target,
            root=root,
        )
        entries += 1
        if metadata.st_dev != root_device:
            crosses_device = True
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in seen:
            return
        seen.add(identity)
        blocks = getattr(metadata, "st_blocks", 0) * 512
        allocated += blocks
        if stat.S_ISREG(metadata.st_mode) or is_linklike(metadata):
            logical += metadata.st_size
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink == 1:
            estimated_reclaim += blocks

    def walk(
        directory_descriptor: int,
        *,
        relative: str,
        metadata: os.stat_result,
        root: bool,
    ) -> None:
        nonlocal crosses_mount
        record(metadata, relative=relative, target=None, root=root)
        for name in sorted(os.listdir(directory_descriptor)):
            child_relative = name if relative == "." else f"{relative}/{name}"
            child_stat = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if is_linklike(child_stat):
                target = os.readlink(name, dir_fd=directory_descriptor)
                record(
                    child_stat,
                    relative=child_relative,
                    target=target,
                    root=False,
                )
                continue
            if not stat.S_ISDIR(child_stat.st_mode):
                record(
                    child_stat,
                    relative=child_relative,
                    target=None,
                    root=False,
                )
                continue

            child_descriptor = os.open(
                name,
                _directory_open_flags(),
                dir_fd=directory_descriptor,
            )
            try:
                opened_stat = os.fstat(child_descriptor)
                if (opened_stat.st_dev, opened_stat.st_ino) != (
                    child_stat.st_dev,
                    child_stat.st_ino,
                ):
                    raise StorageSafetyError(
                        "directory entry changed while measuring quarantined candidate"
                    )
                child_mount_identity = _mount_identity_for_fd(child_descriptor)
                if child_mount_identity != root_mount_identity:
                    crosses_mount = True
                    record(
                        opened_stat,
                        relative=child_relative,
                        target=None,
                        root=False,
                    )
                    continue
                walk(
                    child_descriptor,
                    relative=child_relative,
                    metadata=opened_stat,
                    root=False,
                )
            finally:
                os.close(child_descriptor)

    if not stat.S_ISDIR(root_stat.st_mode):
        raise StorageSafetyError("quarantined directory candidate changed type")
    walk(descriptor, relative=".", metadata=root_stat, root=True)
    measurement = TreeMeasurement(
        logical_bytes=logical,
        allocated_bytes=allocated,
        estimated_reclaim_bytes=estimated_reclaim,
        entry_count=entries,
        tree_digest=exact_digest.hexdigest(),
        crosses_device=crosses_device,
        crosses_mount=crosses_mount,
    )
    return _PinnedTreeMeasurement(
        measurement=measurement,
        stable_digest=stable_digest.hexdigest(),
    )


def _measure_regular_file_from_fd(
    descriptor: int,
) -> tuple[_PinnedTreeMeasurement, os.stat_result]:
    """Measure one pinned regular file without resolving its external path."""

    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise StorageSafetyError("quarantined file candidate changed type")
    exact_digest = hashlib.sha256()
    stable_digest = hashlib.sha256()
    _update_tree_digests(
        exact_digest,
        stable_digest,
        relative=".",
        metadata=metadata,
        target=None,
        root=True,
    )
    allocated = getattr(metadata, "st_blocks", 0) * 512
    measurement = TreeMeasurement(
        logical_bytes=metadata.st_size,
        allocated_bytes=allocated,
        estimated_reclaim_bytes=allocated if metadata.st_nlink == 1 else 0,
        entry_count=1,
        tree_digest=exact_digest.hexdigest(),
        crosses_device=False,
        crosses_mount=False,
    )
    return (
        _PinnedTreeMeasurement(
            measurement=measurement,
            stable_digest=stable_digest.hexdigest(),
        ),
        metadata,
    )


def _measure_link_at(
    directory_descriptor: int,
    name: str,
) -> tuple[_PinnedTreeMeasurement, os.stat_result]:
    """Measure one link-like entry relative to a pinned parent directory."""

    metadata = os.stat(
        name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
    if not is_linklike(metadata):
        raise StorageSafetyError("quarantined link candidate changed type")
    target = os.readlink(name, dir_fd=directory_descriptor)
    exact_digest = hashlib.sha256()
    stable_digest = hashlib.sha256()
    _update_tree_digests(
        exact_digest,
        stable_digest,
        relative=".",
        metadata=metadata,
        target=target,
        root=True,
    )
    allocated = getattr(metadata, "st_blocks", 0) * 512
    measurement = TreeMeasurement(
        logical_bytes=metadata.st_size,
        allocated_bytes=allocated,
        estimated_reclaim_bytes=allocated,
        entry_count=1,
        tree_digest=exact_digest.hexdigest(),
        crosses_device=False,
        crosses_mount=False,
    )
    return (
        _PinnedTreeMeasurement(
            measurement=measurement,
            stable_digest=stable_digest.hexdigest(),
        ),
        metadata,
    )


def _assert_planned_file_measurement(
    candidate: StorageCandidate,
    pinned: _PinnedTreeMeasurement,
    metadata: os.stat_result,
) -> None:
    expected = (
        candidate.device,
        candidate.inode,
        candidate.mode,
        candidate.size,
        candidate.mtime_ns,
        candidate.ctime_ns,
        candidate.logical_bytes,
        candidate.allocated_bytes,
        candidate.estimated_reclaim_bytes,
        candidate.entry_count,
        candidate.tree_digest,
        False,
        False,
    )
    measured = pinned.measurement
    actual = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        measured.logical_bytes,
        measured.allocated_bytes,
        measured.estimated_reclaim_bytes,
        measured.entry_count,
        measured.tree_digest,
        measured.crosses_device,
        measured.crosses_mount,
    )
    if actual != expected:
        raise StorageSafetyError("file candidate changed immediately before removal")


def _assert_planned_link_measurement(
    candidate: StorageCandidate,
    pinned: _PinnedTreeMeasurement,
    metadata: os.stat_result,
) -> None:
    expected = (
        candidate.device,
        candidate.inode,
        candidate.mode,
        candidate.size,
        candidate.mtime_ns,
        candidate.ctime_ns,
        candidate.logical_bytes,
        candidate.allocated_bytes,
        candidate.estimated_reclaim_bytes,
        candidate.entry_count,
        candidate.tree_digest,
        False,
        False,
    )
    measured = pinned.measurement
    actual = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        measured.logical_bytes,
        measured.allocated_bytes,
        measured.estimated_reclaim_bytes,
        measured.entry_count,
        measured.tree_digest,
        measured.crosses_device,
        measured.crosses_mount,
    )
    if actual != expected:
        raise StorageSafetyError("link candidate changed immediately before removal")


def _assert_planned_tree_measurement(
    candidate: StorageCandidate,
    measured: TreeMeasurement,
) -> None:
    expected = (
        candidate.logical_bytes,
        candidate.allocated_bytes,
        candidate.estimated_reclaim_bytes,
        candidate.entry_count,
        candidate.tree_digest,
        False,
        False,
    )
    actual = (
        measured.logical_bytes,
        measured.allocated_bytes,
        measured.estimated_reclaim_bytes,
        measured.entry_count,
        measured.tree_digest,
        measured.crosses_device,
        measured.crosses_mount,
    )
    if actual != expected:
        raise StorageSafetyError("directory candidate tree changed immediately before removal")


def _remove_directory_contents(
    descriptor: int,
    *,
    path: Path,
    mount_identity: tuple[int, int],
    mount_points: frozenset[Path],
) -> None:
    for name in sorted(os.listdir(descriptor)):
        child_path = path / name
        child_stat = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        mounted = _mount_at_or_below(child_path, mount_points)
        if mounted is not None or os.path.ismount(child_path):
            raise StorageSafetyError(
                f"directory candidate contains a mount point: {mounted or child_path}"
            )
        if stat.S_ISDIR(child_stat.st_mode) and not is_linklike(child_stat):
            child_descriptor = os.open(
                name,
                _directory_open_flags(),
                dir_fd=descriptor,
            )
            try:
                opened_stat = os.fstat(child_descriptor)
                if (
                    opened_stat.st_dev,
                    opened_stat.st_ino,
                ) != (
                    child_stat.st_dev,
                    child_stat.st_ino,
                ):
                    raise StorageSafetyError("directory entry changed while it was being opened")
                if _mount_identity_for_fd(child_descriptor) != mount_identity:
                    raise StorageSafetyError(
                        f"directory candidate crosses a mount boundary: {child_path}"
                    )
                _remove_directory_contents(
                    child_descriptor,
                    path=child_path,
                    mount_identity=mount_identity,
                    mount_points=mount_points,
                )
                current_stat = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if (
                    current_stat.st_dev,
                    current_stat.st_ino,
                ) != (
                    opened_stat.st_dev,
                    opened_stat.st_ino,
                ):
                    raise StorageSafetyError("directory entry changed during removal")
            finally:
                os.close(child_descriptor)
            os.rmdir(name, dir_fd=descriptor)
            continue
        if child_stat.st_dev != mount_identity[0]:
            raise StorageSafetyError(
                f"directory candidate crosses a filesystem boundary: {child_path}"
            )
        os.unlink(name, dir_fd=descriptor)


def _remove_via_quarantine(
    candidate: StorageCandidate,
    *,
    quarantine_dir: Path,
    home: Path,
    home_mount_identity: MountIdentity,
    mount_reader: MountReader,
) -> RemovalOutcome:
    """Atomically isolate one exact candidate before deleting it."""

    source = lexical_absolute(Path(candidate.path))
    quarantine = lexical_absolute(quarantine_dir)
    trusted_home = lexical_absolute(home)
    assert_no_linklike_components(
        source.parent,
        boundary=trusted_home,
        allow_missing=False,
    )
    assert_no_linklike_components(
        quarantine,
        boundary=trusted_home,
        allow_missing=False,
    )
    source_parent_descriptor = _open_directory_beneath(
        trusted_home,
        source.parent,
        home_mount_identity=home_mount_identity,
    )
    quarantine_descriptor = -1
    candidate_descriptor = -1
    moved = False
    destination_name = f"{candidate.candidate_id}.{candidate.kind}"
    destination = quarantine / destination_name
    try:
        quarantine_descriptor = _open_directory_beneath(
            trusted_home,
            quarantine,
            home_mount_identity=home_mount_identity,
        )
        source_parent_stat = _assert_directory_fd_matches_path(
            source_parent_descriptor,
            source.parent,
            label="candidate parent",
        )
        quarantine_stat = _assert_directory_fd_matches_path(
            quarantine_descriptor,
            quarantine,
            label="candidate quarantine",
        )
        if (
            source_parent_stat.st_dev != candidate.device
            or quarantine_stat.st_dev != candidate.device
        ):
            raise StorageSafetyError(
                "candidate and transaction quarantine must be on the same filesystem"
            )

        source_stat = os.stat(
            source.name,
            dir_fd=source_parent_descriptor,
            follow_symlinks=False,
        )
        if (source_stat.st_dev, source_stat.st_ino) != (
            candidate.device,
            candidate.inode,
        ):
            raise StorageSafetyError("candidate identity changed immediately before removal")
        if candidate.kind == "directory":
            type_matches = stat.S_ISDIR(source_stat.st_mode) and not is_linklike(source_stat)
        elif candidate.kind == "file":
            type_matches = stat.S_ISREG(source_stat.st_mode) and not is_linklike(source_stat)
        else:
            type_matches = is_linklike(source_stat)
        if not type_matches:
            raise StorageSafetyError(f"{candidate.kind} candidate changed type")

        before_tree: _PinnedTreeMeasurement | None = None
        mount_points: frozenset[Path] = frozenset()
        if candidate.kind == "directory":
            mount_points = _read_mounts_or_raise(mount_reader)
            mounted_ancestor = _mount_strictly_between(trusted_home, source, mount_points)
            if mounted_ancestor is not None:
                raise StorageSafetyError(
                    f"directory candidate has a mounted ancestor: {mounted_ancestor}"
                )
            mounted = _mount_at_or_below(source, mount_points)
            if mounted is not None:
                raise StorageSafetyError(f"directory candidate contains a mount point: {mounted}")
            candidate_descriptor = os.open(
                source.name,
                _directory_open_flags(),
                dir_fd=source_parent_descriptor,
            )
            opened_stat = os.fstat(candidate_descriptor)
            if (opened_stat.st_dev, opened_stat.st_ino) != (
                candidate.device,
                candidate.inode,
            ):
                raise StorageSafetyError("directory candidate changed while it was being opened")
            if _mount_identity_for_fd(candidate_descriptor) != home_mount_identity:
                raise StorageSafetyError("directory candidate is itself a mount point")
            before_tree = _measure_tree_from_fd(candidate_descriptor)
            _assert_planned_tree_measurement(candidate, before_tree.measurement)
        elif candidate.kind == "file":
            candidate_descriptor = os.open(
                source.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                dir_fd=source_parent_descriptor,
            )
            before_tree, opened_stat = _measure_regular_file_from_fd(candidate_descriptor)
            _assert_planned_file_measurement(
                candidate,
                before_tree,
                opened_stat,
            )
        else:
            before_tree, link_stat = _measure_link_at(
                source_parent_descriptor,
                source.name,
            )
            _assert_planned_link_measurement(
                candidate,
                before_tree,
                link_stat,
            )

        # These path checks make pre-existing swaps fail closed. A swap after
        # this point still cannot redirect the operation because rename uses
        # only the already-pinned descriptors and basenames.
        _assert_directory_fd_matches_path(
            source_parent_descriptor,
            source.parent,
            label="candidate parent",
        )
        _assert_directory_fd_matches_path(
            quarantine_descriptor,
            quarantine,
            label="candidate quarantine",
        )
        current_source_stat = os.stat(
            source.name,
            dir_fd=source_parent_descriptor,
            follow_symlinks=False,
        )
        if (current_source_stat.st_dev, current_source_stat.st_ino) != (
            candidate.device,
            candidate.inode,
        ):
            raise StorageSafetyError("candidate changed before quarantine rename")
        if candidate.kind == "file":
            assert before_tree is not None
            before_tree, immediate_stat = _measure_regular_file_from_fd(candidate_descriptor)
            _assert_planned_file_measurement(
                candidate,
                before_tree,
                immediate_stat,
            )
        elif candidate.kind == "broken_symlink":
            before_tree, immediate_stat = _measure_link_at(
                source_parent_descriptor,
                source.name,
            )
            _assert_planned_link_measurement(
                candidate,
                before_tree,
                immediate_stat,
            )

        _renameat_noreplace(
            source_parent_descriptor,
            source.name,
            quarantine_descriptor,
            destination_name,
        )
        moved = True
        _strict_fsync_directory_fd(source_parent_descriptor)
        _strict_fsync_directory_fd(quarantine_descriptor)

        moved_stat = os.stat(
            destination_name,
            dir_fd=quarantine_descriptor,
            follow_symlinks=False,
        )
        if (moved_stat.st_dev, moved_stat.st_ino) != (
            candidate.device,
            candidate.inode,
        ):
            raise StorageSafetyError(
                f"quarantined candidate identity changed; recover it at {destination}"
            )
        if candidate.kind == "directory":
            moved_type_matches = stat.S_ISDIR(moved_stat.st_mode) and not is_linklike(moved_stat)
        elif candidate.kind == "file":
            moved_type_matches = stat.S_ISREG(moved_stat.st_mode) and not is_linklike(moved_stat)
        else:
            moved_type_matches = is_linklike(moved_stat)
        if not moved_type_matches:
            raise StorageSafetyError(
                f"quarantined candidate type changed; recover it at {destination}"
            )

        if candidate.kind == "directory":
            assert before_tree is not None
            after_tree = _measure_tree_from_fd(candidate_descriptor)
            before = before_tree.measurement
            after = after_tree.measurement
            if after_tree.stable_digest != before_tree.stable_digest or (
                after.logical_bytes,
                after.allocated_bytes,
                after.estimated_reclaim_bytes,
                after.entry_count,
                after.crosses_device,
                after.crosses_mount,
            ) != (
                before.logical_bytes,
                before.allocated_bytes,
                before.estimated_reclaim_bytes,
                before.entry_count,
                False,
                False,
            ):
                raise StorageSafetyError(
                    f"quarantined directory tree changed; recover it at {destination}"
                )
            _remove_directory_contents(
                candidate_descriptor,
                path=source,
                mount_identity=home_mount_identity,
                mount_points=mount_points,
            )
            current_moved_stat = os.stat(
                destination_name,
                dir_fd=quarantine_descriptor,
                follow_symlinks=False,
            )
            if (current_moved_stat.st_dev, current_moved_stat.st_ino) != (
                candidate.device,
                candidate.inode,
            ):
                raise StorageSafetyError(
                    f"quarantined directory root changed; recover it at {destination}"
                )
            os.rmdir(destination_name, dir_fd=quarantine_descriptor)
        elif candidate.kind == "file":
            assert before_tree is not None
            after_tree, _after_stat = _measure_regular_file_from_fd(candidate_descriptor)
            before = before_tree.measurement
            after = after_tree.measurement
            if after_tree.stable_digest != before_tree.stable_digest or (
                after.logical_bytes,
                after.allocated_bytes,
                after.estimated_reclaim_bytes,
                after.entry_count,
                after.crosses_device,
                after.crosses_mount,
            ) != (
                before.logical_bytes,
                before.allocated_bytes,
                before.estimated_reclaim_bytes,
                before.entry_count,
                False,
                False,
            ):
                raise StorageSafetyError(f"quarantined file changed; recover it at {destination}")
            os.unlink(destination_name, dir_fd=quarantine_descriptor)
        else:
            assert before_tree is not None
            after_tree, _after_stat = _measure_link_at(
                quarantine_descriptor,
                destination_name,
            )
            before = before_tree.measurement
            after = after_tree.measurement
            if after_tree.stable_digest != before_tree.stable_digest or (
                after.logical_bytes,
                after.allocated_bytes,
                after.estimated_reclaim_bytes,
                after.entry_count,
                after.crosses_device,
                after.crosses_mount,
            ) != (
                before.logical_bytes,
                before.allocated_bytes,
                before.estimated_reclaim_bytes,
                before.entry_count,
                False,
                False,
            ):
                raise StorageSafetyError(f"quarantined link changed; recover it at {destination}")
            os.unlink(destination_name, dir_fd=quarantine_descriptor)

        try:
            _strict_fsync_directory_fd(quarantine_descriptor)
        except OSError as exc:
            return RemovalOutcome(
                status="deleted_durability_uncertain",
                durability_error=f"{type(exc).__name__}: {exc}",
            )
        return RemovalOutcome(status="deleted")
    except (OSError, StorageSafetyError) as exc:
        if moved:
            raise StorageSafetyError(
                f"candidate was quarantined but cleanup failed; recover it at {destination}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        raise
    finally:
        if candidate_descriptor >= 0:
            os.close(candidate_descriptor)
        if quarantine_descriptor >= 0:
            os.close(quarantine_descriptor)
        os.close(source_parent_descriptor)


def default_remove_candidate(
    candidate: StorageCandidate,
    *,
    mount_reader: MountReader = default_mount_reader,
    quarantine_dir: Path | None = None,
    trusted_home: Path | None = None,
    home_mount_identity: MountIdentity | None = None,
) -> RemovalOutcome:
    """Remove only the exact no-follow candidate type recorded in the plan."""

    path = Path(candidate.path)
    path_stat = os.lstat(path)
    if (path_stat.st_dev, path_stat.st_ino) != (candidate.device, candidate.inode):
        raise StorageSafetyError("candidate identity changed immediately before removal")
    if trusted_home is None or home_mount_identity is None:
        raise StorageSafetyError("candidate removal requires a pinned home mount")
    _assert_candidate_parent_on_home_mount(
        candidate,
        home=trusted_home,
        home_mount_identity=home_mount_identity,
    )
    if quarantine_dir is None:
        raise StorageSafetyError(
            f"{candidate.kind} candidate removal requires a transaction quarantine"
        )
    if candidate.kind == "directory":
        if not stat.S_ISDIR(path_stat.st_mode) or is_linklike(path_stat):
            raise StorageSafetyError("directory candidate changed type")
    elif candidate.kind == "file":
        if not stat.S_ISREG(path_stat.st_mode) or is_linklike(path_stat):
            raise StorageSafetyError("file candidate changed type")
    else:
        if not is_linklike(path_stat):
            raise StorageSafetyError("broken-link candidate changed type")
    return _remove_via_quarantine(
        candidate,
        quarantine_dir=quarantine_dir,
        home=trusted_home,
        home_mount_identity=home_mount_identity,
        mount_reader=mount_reader,
    )


def _transaction_dir(home: Path, transaction: str) -> Path:
    state_root = home / ".afs" / "storage" / "transactions"
    assert_no_linklike_components(state_root, boundary=home)
    secure_mkdir(state_root, mode=0o700, durable=True)
    assert_no_linklike_components(state_root, boundary=home, allow_missing=False)
    transaction_dir = state_root / transaction
    secure_mkdir(transaction_dir, mode=0o700, durable=True)
    assert_no_linklike_components(transaction_dir, boundary=home, allow_missing=False)
    return transaction_dir


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    atomic_create_text(
        path,
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        mode=0o600,
        durable=True,
    )


def storage_authorization_scope(
    plan_sha256: str,
    transaction: str,
    rationale: str,
) -> str:
    """Bind terminal confirmation to one exact plan and rationale."""

    return decision_scope_parts(
        "storage",
        "apply",
        plan_sha256,
        transaction,
        rationale,
    )


def _write_transaction_record(path: Path, payload: dict[str, Any]) -> None:
    atomic_create_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        mode=0o600,
        durable=True,
    )


def apply_storage_plan(
    plan_path: Path,
    *,
    confirm: str,
    because: str,
    now_ns: int | None = None,
    open_checker: OpenChecker = default_open_checker,
    remove_candidate: RemoveCandidate = default_remove_candidate,
    authorization: HumanAuthorization | None = None,
    mount_reader: MountReader = default_mount_reader,
    mount_identity_reader: MountIdentityReader | None = None,
) -> dict[str, Any]:
    """Apply an exact, fresh plan without stopping any process."""

    plan = load_storage_plan(plan_path)
    if confirm != plan["transaction"]:
        raise StorageSafetyError("confirmation must exactly match the transaction")
    rationale = _validate_rationale(because)
    timestamp = time.time_ns() if now_ns is None else now_ns
    _validate_timestamp_ns(timestamp, field="apply timestamp")
    if timestamp > plan["expires_at_ns"]:
        raise StorageSafetyError("storage plan expired; build a fresh plan")
    if timestamp < plan["created_at_ns"]:
        raise StorageSafetyError("current time predates the storage plan")
    planned = tuple(_candidate_from_record(record) for record in plan["candidates"])
    if not planned:
        raise StorageSafetyError("storage plan has no eligible candidates")
    planned_ids = [candidate.candidate_id for candidate in planned]
    if len(planned_ids) != len(set(planned_ids)):
        raise StorageSafetyError("storage plan candidate IDs are not unique")

    home, home_identity = _trusted_home(Path(plan["home"]))
    if home_identity != plan["home_identity"]:
        raise StorageSafetyError("storage plan home identity changed")
    identity_reader = mount_identity_reader or default_mount_identity_reader
    pinned_home_mount_identity = _mount_identity_from_record(plan["home_mount_identity"])
    if identity_reader(home) != pinned_home_mount_identity:
        raise StorageSafetyError("storage plan home mount identity changed")
    for candidate in planned:
        candidate_path = lexical_absolute(Path(candidate.path))
        try:
            candidate_path.relative_to(home)
        except ValueError as exc:
            raise StorageSafetyError("planned candidate escapes its home") from exc
        if candidate.candidate_id != _candidate_id(
            candidate.category,
            home,
            candidate_path,
        ):
            raise StorageSafetyError("planned candidate ID does not match its path")
    if plan["estimated_reclaim_bytes"] != sum(
        candidate.estimated_reclaim_bytes for candidate in planned
    ):
        raise StorageSafetyError("storage plan reclaim total does not match candidates")
    pending_transaction = home / ".afs" / "storage" / "transactions" / plan["transaction"]
    if os.path.lexists(pending_transaction):
        raise StorageSafetyError("storage transaction was already claimed or initialized")
    lsof_deadline = _new_lsof_deadline() if open_checker is default_open_checker else None
    fresh_candidates = discover_candidates(
        home,
        stale_days=plan["stale_days"],
        now_ns=timestamp,
        open_checker=open_checker,
        mount_reader=mount_reader,
        lsof_deadline=lsof_deadline,
    )
    fresh_ids = [candidate.candidate_id for candidate in fresh_candidates]
    if len(fresh_ids) != len(set(fresh_ids)):
        raise StorageSafetyError("fresh candidate IDs are not unique")
    discovered = {candidate.candidate_id: candidate for candidate in fresh_candidates}
    for candidate in planned:
        current = discovered.get(candidate.candidate_id)
        if current is None:
            raise StorageSafetyError(
                f"planned candidate is missing or outside bounded discovery: "
                f"{candidate.candidate_id}"
            )
        _assert_current_candidate(candidate, current)
        _assert_candidate_parent_on_home_mount(
            current,
            home=home,
            home_mount_identity=pinned_home_mount_identity,
        )

    scope = storage_authorization_scope(
        plan["plan_sha256"],
        plan["transaction"],
        rationale,
    )
    if not consume_human_authorization(authorization, scope=scope):
        raise StorageSafetyError("storage apply requires controlling-terminal human authorization")
    assert authorization is not None

    if identity_reader(home) != pinned_home_mount_identity:
        raise StorageSafetyError("home mount identity changed before transaction creation")
    transaction_dir = _transaction_dir(home, plan["transaction"])
    if identity_reader(transaction_dir) != pinned_home_mount_identity:
        raise StorageSafetyError("transaction state is not on the pinned home mount")
    manifest_path = transaction_dir / "plan.json"
    claim_path = transaction_dir / "claim.json"
    receipt_path = transaction_dir / "receipt.json"
    journal_path = transaction_dir / "journal"
    quarantine_path = transaction_dir / "quarantine"
    _write_transaction_record(manifest_path, plan)
    secure_mkdir(journal_path, mode=0o700, durable=True)
    assert_no_linklike_components(journal_path, boundary=home, allow_missing=False)
    secure_mkdir(quarantine_path, mode=0o700, durable=True)
    assert_no_linklike_components(quarantine_path, boundary=home, allow_missing=False)
    if (
        identity_reader(journal_path) != pinned_home_mount_identity
        or identity_reader(quarantine_path) != pinned_home_mount_identity
    ):
        raise StorageSafetyError(
            "transaction journal or quarantine is not on the pinned home mount"
        )
    claim = {
        "schema": "afs.storage.claim.v1",
        "transaction": plan["transaction"],
        "plan_sha256": plan["plan_sha256"],
        "plan_path": os.fspath(lexical_absolute(plan_path)),
        "manifest_path": os.fspath(manifest_path),
        "journal_path": os.fspath(journal_path),
        "quarantine_path": os.fspath(quarantine_path),
        "because": rationale,
        "claimed_at": _utc_now(),
        "authorized_via": authorization.confirmed_via,
        "reviewer": authorization.identity.reviewer,
        "reviewer_subject": authorization.identity.subject,
    }
    _write_transaction_record(claim_path, claim)

    deleted: list[dict[str, Any]] = []
    durability_uncertain: list[dict[str, Any]] = []
    failure: dict[str, str] | None = None

    def immediate_open_checker(
        path: Path,
        kind: CandidateKind,
    ) -> tuple[OpenStatus, str]:
        if open_checker is not default_open_checker:
            return open_checker(path, kind)
        assert lsof_deadline is not None
        return default_open_checker(path, kind, deadline=lsof_deadline)

    for index, candidate in enumerate(planned):
        stem = f"{index:05d}-{candidate.candidate_id}"
        intent_path = journal_path / f"{stem}.intent.json"
        deleted_path = journal_path / f"{stem}.deleted.json"
        failed_path = journal_path / f"{stem}.failed.json"
        try:
            mount_points = mount_reader()
            current = _capture_candidate(
                home=home,
                category=candidate.category,
                path=Path(candidate.path),
                kind=candidate.kind,
                stale_days=plan["stale_days"],
                now_ns=timestamp,
                open_checker=immediate_open_checker,
                mount_points=mount_points,
            )
            _assert_current_candidate(candidate, current)
            _assert_candidate_parent_on_home_mount(
                current,
                home=home,
                home_mount_identity=pinned_home_mount_identity,
            )
            intent = {
                "schema": "afs.storage.intent.v1",
                "transaction": plan["transaction"],
                "plan_sha256": plan["plan_sha256"],
                "sequence": index,
                "candidate": current.plan_record(),
                "recorded_at": _utc_now(),
            }
            _write_transaction_record(intent_path, intent)
            if remove_candidate is default_remove_candidate:
                removal_outcome = default_remove_candidate(
                    current,
                    mount_reader=mount_reader,
                    quarantine_dir=quarantine_path,
                    trusted_home=home,
                    home_mount_identity=pinned_home_mount_identity,
                )
            else:
                removal_outcome = remove_candidate(current) or RemovalOutcome(status="deleted")
            if not isinstance(removal_outcome, RemovalOutcome):
                raise StorageSafetyError("candidate remover returned an invalid outcome")
            deleted_record = {
                "candidate_id": current.candidate_id,
                "category": current.category,
                "path": current.path,
                "estimated_reclaim_bytes": current.estimated_reclaim_bytes,
                "outcome": removal_outcome.status,
            }
            if removal_outcome.durability_error is not None:
                deleted_record["durability_error"] = removal_outcome.durability_error
            deleted.append(deleted_record)
            if removal_outcome.status == "deleted_durability_uncertain":
                durability_uncertain.append(deleted_record)
            outcome = {
                "schema": "afs.storage.outcome.v1",
                "transaction": plan["transaction"],
                "plan_sha256": plan["plan_sha256"],
                "sequence": index,
                "status": removal_outcome.status,
                "candidate": deleted_record,
                "recorded_at": _utc_now(),
            }
            _write_transaction_record(deleted_path, outcome)
            if removal_outcome.status == "deleted_durability_uncertain":
                break
        except (OSError, StorageSafetyError) as exc:
            failure = {
                "candidate_id": candidate.candidate_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failed = {
                "schema": "afs.storage.outcome.v1",
                "transaction": plan["transaction"],
                "plan_sha256": plan["plan_sha256"],
                "sequence": index,
                "status": "failed",
                "candidate_id": candidate.candidate_id,
                "error": failure["error"],
                "recorded_at": _utc_now(),
            }
            try:
                _write_transaction_record(failed_path, failed)
            except OSError as journal_exc:
                failure["journal_error"] = f"{type(journal_exc).__name__}: {journal_exc}"
            break

    if durability_uncertain:
        status = "durability_uncertain"
    elif failure is None and len(deleted) == len(planned):
        status = "applied"
    else:
        status = "partial_failure"
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "transaction": plan["transaction"],
        "plan_sha256": plan["plan_sha256"],
        "status": status,
        "because": rationale,
        "completed_at": _utc_now(),
        "processes_stopped": False,
        "deleted": deleted,
        "durability_uncertain": durability_uncertain,
        "failure": failure,
        "manifest_path": os.fspath(manifest_path),
        "claim_path": os.fspath(claim_path),
        "journal_path": os.fspath(journal_path),
        "quarantine_path": os.fspath(quarantine_path),
        "receipt_path": os.fspath(receipt_path),
        "receipt_written": True,
    }
    try:
        _write_receipt(receipt_path, receipt)
    except OSError as exc:
        receipt["status"] = "receipt_failure"
        receipt["receipt_written"] = False
        receipt["receipt_error"] = f"{type(exc).__name__}: {exc}"
        raise StorageApplyError(
            "cleanup outcome could not be recorded in a final receipt; "
            "the durable claim and per-candidate journal block replay",
            receipt,
        ) from exc
    if status != "applied":
        if status == "durability_uncertain":
            raise StorageApplyError(
                "storage cleanup stopped because deletion durability is uncertain",
                receipt,
            )
        raise StorageApplyError("storage cleanup stopped after a partial failure", receipt)
    return receipt
