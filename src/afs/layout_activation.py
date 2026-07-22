"""Human-gated, reversible activation of a verified context-layout candidate.

Activation keeps the configured context path stable.  It atomically exchanges
the reviewed v1 source and v2 candidate directories, leaving v1 intact at the
old candidate path.  Rollback performs the same atomic exchange in reverse.
There is deliberately no sequential-rename, symlink, copy, or delete fallback.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .atomic_io import atomic_create_text, atomic_write_text, strict_fsync_directory
from .human_provenance import (
    HumanAuthorization,
    consume_human_authorization,
    decision_scope_parts,
)
from .layout_migration import (
    MigrationPreflightError,
    VerifiedMigrationCandidate,
    candidate_tree_sha256,
    tree_fingerprint,
    verify_completed_candidate,
    verify_relocated_candidate,
)
from .path_safety import assert_no_linklike_components, is_linklike, lexical_absolute

__all__ = [
    "ActivationApplyError",
    "ActivationPreflight",
    "ActivationPreflightError",
    "ActivationResult",
    "LayoutActivationError",
    "RollbackApplyError",
    "RollbackPreflight",
    "RollbackPreflightError",
    "RollbackResult",
    "activate_layout",
    "activation_confirmation_token",
    "layout_activation_authorization_scope",
    "layout_rollback_authorization_scope",
    "preflight_activation",
    "preflight_rollback",
    "rollback_confirmation_token",
    "rollback_layout",
]

_LOCK_TIMEOUT_SECONDS = 30.0
_MAX_RECORD_BYTES = 1024 * 1024
_MAX_RATIONALE_CHARACTERS = 4096
_RENAME_SWAP = 0x00000002
_RENAME_EXCHANGE = 0x00000002
_AT_FDCWD = -100
_JOURNAL_NAME = "activation-journal.json"
_ACTIVATION_RECEIPT_NAME = "activation-receipt.json"
_ROLLBACK_RECEIPT_NAME = "rollback-receipt.json"
_LOCK_NAME = ".activation.lock"


class LayoutActivationError(RuntimeError):
    """Base class for activation and rollback failures."""


class ActivationPreflightError(LayoutActivationError):
    """Raised when activation evidence or topology is unsafe."""


class ActivationApplyError(LayoutActivationError):
    """Raised after an authorized activation cannot complete."""


class RollbackPreflightError(LayoutActivationError):
    """Raised when rollback evidence or topology is unsafe."""


class RollbackApplyError(LayoutActivationError):
    """Raised after an authorized rollback cannot complete."""


@dataclass(frozen=True)
class ActivationPreflight:
    """Read-only evidence for an activation or receipt finalization."""

    status: Literal["ready", "receipt_pending", "already_active", "already_rolled_back"]
    plan: Any
    evidence: VerifiedMigrationCandidate
    state_dir: Path
    active_root: Path
    inactive_root: Path
    activation_id: str
    journal: dict[str, Any] | None = None
    activation_receipt: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = self.evidence.result
        return {
            "status": self.status,
            "ready": self.status in {"ready", "receipt_pending"},
            "activation_id": self.activation_id,
            "transaction_id": result.transaction_id,
            "plan_sha256": result.plan_hash,
            "migration_receipt_sha256": self.evidence.receipt_sha256,
            "candidate_sha256": self.evidence.candidate_sha256,
            "active_root": str(self.active_root),
            "inactive_root": str(self.inactive_root),
            "state_dir": str(self.state_dir),
            "journal_path": str(self.state_dir / _JOURNAL_NAME),
            "activation_receipt": str(self.state_dir / _ACTIVATION_RECEIPT_NAME),
            "source_device": self.evidence.source_device,
            "source_inode": self.evidence.source_inode,
            "candidate_device": self.evidence.candidate_device,
            "candidate_inode": self.evidence.candidate_inode,
            "retained_source_count": len(result.retained_sources),
            "retained_path_count": len(result.retained_paths),
            "atomic_exchange": _atomic_exchange_backend(),
            "rollback_preserves_v2_at": str(self.inactive_root),
        }


@dataclass(frozen=True)
class ActivationResult:
    """Outcome of activation or recognition of an active transaction."""

    status: Literal["activated", "already_active"]
    activation_id: str
    active_root: Path
    inactive_root: Path
    state_dir: Path
    activation_receipt: Path

    @property
    def journal_path(self) -> Path:
        """Return the external durable journal path."""

        return self.state_dir / _JOURNAL_NAME

    @property
    def receipt_path(self) -> Path:
        """Return the immutable activation receipt path."""

        return self.activation_receipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "activation_id": self.activation_id,
            "active_root": str(self.active_root),
            "inactive_root": str(self.inactive_root),
            "state_dir": str(self.state_dir),
            "activation_receipt": str(self.activation_receipt),
            "journal_path": str(self.journal_path),
            "v1_preserved": True,
            "v1_preserved_at": str(self.inactive_root),
        }


@dataclass(frozen=True)
class RollbackPreflight:
    """Read-only evidence for rollback or rollback-receipt finalization."""

    status: Literal["ready", "receipt_pending", "already_rolled_back"]
    state_dir: Path
    active_root: Path
    inactive_root: Path
    activation_id: str
    journal: dict[str, Any]
    activation_receipt: dict[str, Any]
    current_v2_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ready": self.status in {"ready", "receipt_pending"},
            "activation_id": self.activation_id,
            "active_root": str(self.active_root),
            "inactive_root": str(self.inactive_root),
            "state_dir": str(self.state_dir),
            "journal_path": str(self.state_dir / _JOURNAL_NAME),
            "rollback_receipt": str(self.state_dir / _ROLLBACK_RECEIPT_NAME),
            "current_v2_sha256": self.current_v2_sha256,
            "v2_writes_will_be_preserved_at": str(self.inactive_root),
            "atomic_exchange": _atomic_exchange_backend(),
        }


@dataclass(frozen=True)
class RollbackResult:
    """Outcome of rollback or recognition of a rolled-back transaction."""

    status: Literal["rolled_back", "already_rolled_back"]
    activation_id: str
    active_root: Path
    inactive_root: Path
    state_dir: Path
    rollback_receipt: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "activation_id": self.activation_id,
            "active_root": str(self.active_root),
            "inactive_root": str(self.inactive_root),
            "state_dir": str(self.state_dir),
            "rollback_receipt": str(self.rollback_receipt),
            "v1_restored": True,
            "v2_preserved_at": str(self.inactive_root),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_rationale(rationale: str) -> str:
    normalized = rationale.strip()
    if not normalized:
        raise ValueError("a human-authored rationale is required")
    if len(normalized) > _MAX_RATIONALE_CHARACTERS:
        raise ValueError(f"rationale exceeds {_MAX_RATIONALE_CHARACTERS} characters")
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        raise ValueError("rationale contains control or formatting characters")
    return normalized


def _record_hash(payload: dict[str, Any], hash_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(hash_field, None)
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _activation_id(evidence: VerifiedMigrationCandidate, state_dir: Path) -> str:
    material = "\0".join(
        (
            evidence.result.plan_hash,
            evidence.result.transaction_id,
            evidence.receipt_sha256,
            evidence.candidate_sha256,
            str(evidence.result.source_root),
            str(evidence.result.destination_root),
            str(state_dir),
        )
    )
    return f"activation_{hashlib.sha256(material.encode()).hexdigest()[:32]}"


def _scope_values(preflight: ActivationPreflight) -> tuple[str, ...]:
    evidence = preflight.evidence
    return (
        preflight.activation_id,
        evidence.result.plan_hash,
        evidence.result.transaction_id,
        evidence.receipt_sha256,
        evidence.candidate_sha256,
        str(preflight.active_root),
        str(preflight.inactive_root),
        str(preflight.state_dir),
        str(evidence.source_device),
        str(evidence.source_inode),
        str(evidence.candidate_device),
        str(evidence.candidate_inode),
    )


def layout_activation_authorization_scope(
    preflight: ActivationPreflight,
    rationale: str,
) -> str:
    """Return the exact broker scope required to activate or finalize."""

    return decision_scope_parts(
        "layout-activation",
        "activate",
        *_scope_values(preflight),
        _normalize_rationale(rationale),
    )


def activation_confirmation_token(preflight: ActivationPreflight, rationale: str) -> str:
    """Return the exact human confirmation token for activation."""

    scope = layout_activation_authorization_scope(preflight, rationale)
    return f"activate_{hashlib.sha256(scope.encode()).hexdigest()[:32]}"


def layout_rollback_authorization_scope(
    preflight: RollbackPreflight,
    rationale: str,
) -> str:
    """Return the exact broker scope required to rollback or finalize."""

    activation_receipt_sha256 = str(preflight.activation_receipt["receipt_sha256"])
    return decision_scope_parts(
        "layout-activation",
        "rollback",
        preflight.activation_id,
        activation_receipt_sha256,
        preflight.current_v2_sha256,
        str(preflight.active_root),
        str(preflight.inactive_root),
        str(preflight.state_dir),
        _normalize_rationale(rationale),
    )


def rollback_confirmation_token(preflight: RollbackPreflight, rationale: str) -> str:
    """Return the exact human confirmation token for rollback."""

    scope = layout_rollback_authorization_scope(preflight, rationale)
    return f"rollback_{hashlib.sha256(scope.encode()).hexdigest()[:32]}"


def _atomic_exchange_backend() -> str:
    if sys.platform == "darwin":
        library = ctypes.CDLL(None)
        if getattr(library, "renamex_np", None) is not None:
            return "darwin-renamex_np"
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None)
        if getattr(library, "renameat2", None) is not None:
            return "linux-renameat2"
    return "unsupported"


def _atomic_exchange(left: Path, right: Path) -> None:
    """Atomically exchange two directory names or fail without fallback."""

    left_bytes = os.fsencode(left)
    right_bytes = os.fsencode(right)
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and getattr(library, "renamex_np", None) is not None:
        renamex_np = library.renamex_np
        renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex_np.restype = ctypes.c_int
        result = renamex_np(left_bytes, right_bytes, _RENAME_SWAP)
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
        result = renameat2(_AT_FDCWD, left_bytes, _AT_FDCWD, right_bytes, _RENAME_EXCHANGE)
    else:
        raise OSError(errno.ENOTSUP, "atomic directory exchange is unavailable")
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), f"{left} <-> {right}")


def _path_stat(path: Path, *, label: str) -> os.stat_result:
    try:
        path_stat = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"cannot inspect {label} {path}: {exc}") from exc
    if is_linklike(path_stat) or not stat.S_ISDIR(path_stat.st_mode):
        raise ValueError(f"{label} must be a real directory: {path}")
    return path_stat


def _assert_private_state_dir(
    state_dir: Path,
    roots: tuple[Path, Path],
    *,
    allow_missing: bool = False,
) -> Path:
    state = lexical_absolute(state_dir)
    try:
        assert_no_linklike_components(
            state,
            boundary=Path(state.anchor),
            allow_missing=allow_missing,
        )
        if allow_missing and not os.path.lexists(state):
            parent_stat = _path_stat(state.parent, label="activation state parent")
            if hasattr(os, "geteuid") and parent_stat.st_uid != os.geteuid():
                raise ValueError("activation state parent must be owned by the current user")
            for root in roots:
                if state == root or state.is_relative_to(root):
                    raise ValueError(
                        "activation state directory must be outside both exchange roots"
                    )
            return state
        state_stat = _path_stat(state, label="activation state directory")
    except (OSError, ValueError) as exc:
        raise ValueError(f"unsafe activation state directory: {exc}") from exc
    if hasattr(os, "geteuid") and state_stat.st_uid != os.geteuid():
        raise ValueError("activation state directory must be owned by the current user")
    if stat.S_IMODE(state_stat.st_mode) & 0o077:
        raise ValueError("activation state directory must have mode 0700 or stricter")
    for root in roots:
        if state == root or state.is_relative_to(root):
            raise ValueError("activation state directory must be outside both exchange roots")
    return state


def _create_private_state_dir(state_dir: Path, roots: tuple[Path, Path]) -> Path:
    state = _assert_private_state_dir(state_dir, roots, allow_missing=True)
    if not os.path.lexists(state):
        os.mkdir(state, 0o700)
        os.chmod(state, 0o700)
        strict_fsync_directory(state.parent)
    return _assert_private_state_dir(state, roots)


def _configured_root(path: Path, active_root: Path) -> None:
    configured = lexical_absolute(path)
    if configured != active_root:
        raise ValueError(
            f"configured context root {configured} does not match activation path {active_root}"
        )
    env_value = os.environ.get("AFS_CONTEXT_ROOT")
    if env_value and lexical_absolute(Path(env_value).expanduser()) != active_root:
        raise ValueError("AFS_CONTEXT_ROOT conflicts with the activation path")


def _assert_exchange_topology(
    active: Path, inactive: Path
) -> tuple[os.stat_result, os.stat_result]:
    if active.parent != inactive.parent:
        raise ValueError("active and inactive roots must be siblings")
    try:
        assert_no_linklike_components(active, boundary=Path(active.anchor), allow_missing=False)
        assert_no_linklike_components(inactive, boundary=Path(inactive.anchor), allow_missing=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"unsafe exchange root: {exc}") from exc
    parent_stat = _path_stat(active.parent, label="exchange parent")
    if stat.S_IMODE(parent_stat.st_mode) & 0o022:
        raise ValueError("exchange parent must not be group- or world-writable")
    active_stat = _path_stat(active, label="active root")
    inactive_stat = _path_stat(inactive, label="inactive root")
    if len({parent_stat.st_dev, active_stat.st_dev, inactive_stat.st_dev}) != 1:
        raise ValueError("exchange roots must be on the same filesystem as their parent")
    if (active_stat.st_dev, active_stat.st_ino) == (inactive_stat.st_dev, inactive_stat.st_ino):
        raise ValueError("exchange roots resolve to the same directory")
    if os.path.ismount(active) or os.path.ismount(inactive):
        raise ValueError("mount-point roots cannot be atomically activated")
    if _atomic_exchange_backend() == "unsupported":
        raise ValueError("atomic directory exchange is unavailable on this platform")
    return active_stat, inactive_stat


def _assert_private_candidate(path: Path) -> None:
    candidate_stat = _path_stat(path, label="v2 candidate")
    if stat.S_IMODE(candidate_stat.st_mode) & 0o077:
        raise ValueError("v2 candidate must have mode 0700 or stricter")


def _open_processes(root: Path) -> tuple[str, ...]:
    command = ["/usr/sbin/lsof" if sys.platform == "darwin" else "lsof", "-Fpcfn", "+D", str(root)]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed executable and arguments
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"cannot prove writer quiescence for {root}: {exc}") from exc
    if completed.returncode == 1 and not completed.stdout and not completed.stderr:
        return ()
    if completed.returncode == 1 and completed.stderr:
        detail = " ".join(completed.stderr.split())[:512]
        raise ValueError(f"lsof quiescence check was incomplete for {root}: {detail}")
    if completed.returncode not in {0, 1}:
        detail = " ".join(completed.stderr.split())[:512]
        raise ValueError(f"lsof quiescence check failed for {root}: {detail}")
    processes: dict[str, str] = {}
    current_pid = "unknown"
    current_command = "unknown"
    for line in completed.stdout.splitlines():
        if line.startswith("p"):
            current_pid = line[1:]
            processes.setdefault(current_pid, "unknown")
        elif line.startswith("c"):
            current_command = line[1:]
            processes[current_pid] = current_command
    return tuple(f"{pid}:{processes[pid]}" for pid in sorted(processes))


def _assert_quiescent(*roots: Path) -> None:
    open_processes: list[str] = []
    for root in roots:
        open_processes.extend(f"{root} ({process})" for process in _open_processes(root))
    if open_processes:
        examples = ", ".join(open_processes[:10])
        raise ValueError(f"context writers or readers are still open: {examples}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"record contains duplicate field {key!r}")
        payload[key] = value
    return payload


def _read_record(path: Path, *, state_dir: Path, hash_field: str) -> dict[str, Any]:
    assert_no_linklike_components(path, boundary=state_dir, allow_missing=False)
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o077
        or before.st_size > _MAX_RECORD_BYTES
        or (hasattr(os, "geteuid") and before.st_uid != os.geteuid())
    ):
        raise ValueError(f"activation record is not a small private unique file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 64 * 1024):
            total += len(chunk)
            if total > _MAX_RECORD_BYTES:
                raise ValueError("activation record exceeds its size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = os.lstat(path)
    signatures = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        for item in (before, opened, after, final)
    }
    if len(signatures) != 1:
        raise ValueError("activation record changed while reading")
    try:
        payload = json.loads(
            b"".join(chunks).decode(),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid activation record JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get(hash_field) != _record_hash(
        payload, hash_field
    ):
        raise ValueError("activation record hash is invalid")
    return payload


def _record_text(payload: dict[str, Any], hash_field: str) -> str:
    document = dict(payload)
    document[hash_field] = _record_hash(document, hash_field)
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_new_record(path: Path, payload: dict[str, Any], hash_field: str) -> None:
    atomic_create_text(path, _record_text(payload, hash_field), mode=0o600, durable=True)


def _write_journal(state_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    document = dict(payload)
    document["journal_sha256"] = _record_hash(document, "journal_sha256")
    path = state_dir / _JOURNAL_NAME
    atomic_write_text(
        path,
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        mode=0o600,
        durable=True,
    )
    strict_fsync_directory(state_dir)
    persisted = _read_record(path, state_dir=state_dir, hash_field="journal_sha256")
    if persisted != document:
        raise ValueError("durable activation journal verification failed")
    return document


def _journal_from_evidence(
    preflight: ActivationPreflight,
    *,
    rationale: str,
    authorization: HumanAuthorization,
) -> dict[str, Any]:
    evidence = preflight.evidence
    now = _utc_now()
    return {
        "schema_version": 1,
        "status": "activate_prepared",
        "activation_id": preflight.activation_id,
        "plan_sha256": evidence.result.plan_hash,
        "migration_transaction_id": evidence.result.transaction_id,
        "migration_receipt_sha256": evidence.receipt_sha256,
        "candidate_sha256": evidence.candidate_sha256,
        "active_root": str(preflight.active_root),
        "inactive_root": str(preflight.inactive_root),
        "state_dir": str(preflight.state_dir),
        "source_device": evidence.source_device,
        "source_inode": evidence.source_inode,
        "candidate_device": evidence.candidate_device,
        "candidate_inode": evidence.candidate_inode,
        "source_fingerprint": evidence.result.source_fingerprint,
        "source_file_count": evidence.result.source_file_count,
        "source_bytes": evidence.result.source_bytes,
        "activation_rationale": _normalize_rationale(rationale),
        "activation_scope": layout_activation_authorization_scope(preflight, rationale),
        "activation_authorized_by": authorization.identity.reviewer,
        "activation_reviewer_subject": authorization.identity.subject,
        "activation_authorized_via": authorization.confirmed_via,
        "prepared_at": now,
        "updated_at": now,
    }


def _journal_evidence(journal: dict[str, Any]) -> VerifiedMigrationCandidate:
    from .layout_migration import MigrationResult

    result = MigrationResult(
        status="already_applied",
        plan_hash=str(journal["plan_sha256"]),
        transaction_id=str(journal["migration_transaction_id"]),
        source_root=Path(str(journal["active_root"])),
        destination_root=Path(str(journal["inactive_root"])),
        source_fingerprint=str(journal["source_fingerprint"]),
        source_file_count=int(journal["source_file_count"]),
        source_bytes=int(journal["source_bytes"]),
        copy_file_count=0,
        copy_bytes=0,
        operation_count=0,
        receipt_path=Path(str(journal["inactive_root"])),
    )
    return VerifiedMigrationCandidate(
        result=result,
        candidate_sha256=str(journal["candidate_sha256"]),
        receipt_sha256=str(journal["migration_receipt_sha256"]),
        source_device=int(journal["source_device"]),
        source_inode=int(journal["source_inode"]),
        candidate_device=int(journal["candidate_device"]),
        candidate_inode=int(journal["candidate_inode"]),
    )


def _validate_journal(journal: dict[str, Any], state_dir: Path) -> None:
    required = {
        "schema_version",
        "status",
        "activation_id",
        "plan_sha256",
        "migration_transaction_id",
        "migration_receipt_sha256",
        "candidate_sha256",
        "active_root",
        "inactive_root",
        "state_dir",
        "source_device",
        "source_inode",
        "candidate_device",
        "candidate_inode",
        "source_fingerprint",
        "source_file_count",
        "source_bytes",
        "activation_rationale",
        "activation_scope",
        "activation_authorized_by",
        "activation_reviewer_subject",
        "activation_authorized_via",
        "prepared_at",
        "updated_at",
        "journal_sha256",
    }
    if set(journal) != required or journal.get("schema_version") != 1:
        raise ValueError("activation journal schema is invalid")
    if journal.get("status") not in {
        "activate_prepared",
        "active_receipt_pending",
        "active",
        "rollback_prepared",
        "rollback_receipt_pending",
        "rolled_back",
    }:
        raise ValueError("activation journal status is invalid")
    if journal.get("state_dir") != str(state_dir):
        raise ValueError("activation journal state directory does not match")
    if journal.get("activation_authorized_via") != "controlling_terminal":
        raise ValueError("activation journal lacks controlling-terminal provenance")
    string_fields = required - {
        "schema_version",
        "source_device",
        "source_inode",
        "candidate_device",
        "candidate_inode",
        "source_file_count",
        "source_bytes",
    }
    if any(type(journal.get(field)) is not str or not journal[field] for field in string_fields):
        raise ValueError("activation journal string fields are invalid")
    integer_fields = {
        "source_device",
        "source_inode",
        "candidate_device",
        "candidate_inode",
        "source_file_count",
        "source_bytes",
    }
    if any(type(journal.get(field)) is not int or journal[field] < 0 for field in integer_fields):
        raise ValueError("activation journal identity fields are invalid")
    hash_fields = {"plan_sha256", "migration_receipt_sha256", "candidate_sha256"}
    if any(
        type(journal.get(field)) is not str or re.fullmatch(r"[a-f0-9]{64}", journal[field]) is None
        for field in hash_fields
    ):
        raise ValueError("activation journal digest fields are invalid")
    if re.fullmatch(r"activation_[a-f0-9]{32}", str(journal.get("activation_id", ""))) is None:
        raise ValueError("activation journal id is invalid")
    if (
        re.fullmatch(r"layout_[a-f0-9]{32}", str(journal.get("migration_transaction_id", "")))
        is None
    ):
        raise ValueError("activation journal migration transaction is invalid")
    expected_scope = decision_scope_parts(
        "layout-activation",
        "activate",
        str(journal["activation_id"]),
        str(journal["plan_sha256"]),
        str(journal["migration_transaction_id"]),
        str(journal["migration_receipt_sha256"]),
        str(journal["candidate_sha256"]),
        str(journal["active_root"]),
        str(journal["inactive_root"]),
        str(journal["state_dir"]),
        str(journal["source_device"]),
        str(journal["source_inode"]),
        str(journal["candidate_device"]),
        str(journal["candidate_inode"]),
        _normalize_rationale(str(journal["activation_rationale"])),
    )
    if journal.get("activation_scope") != expected_scope:
        raise ValueError("activation journal authorization scope is invalid")


def _identity(path: Path) -> tuple[int, int]:
    path_stat = _path_stat(path, label="exchange root")
    return int(path_stat.st_dev), int(path_stat.st_ino)


def _topology(journal: dict[str, Any], active: Path, inactive: Path) -> str:
    active_identity = _identity(active)
    inactive_identity = _identity(inactive)
    source = int(journal["source_device"]), int(journal["source_inode"])
    candidate = int(journal["candidate_device"]), int(journal["candidate_inode"])
    if (active_identity, inactive_identity) == (source, candidate):
        return "initial"
    if (active_identity, inactive_identity) == (candidate, source):
        return "active"
    raise ValueError("exchange-root inode topology conflicts with activation evidence")


def _receipt_path(state_dir: Path, name: str) -> Path:
    return state_dir / name


def _activation_receipt_payload(journal: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "active",
        "activation_id": journal["activation_id"],
        "plan_sha256": journal["plan_sha256"],
        "migration_transaction_id": journal["migration_transaction_id"],
        "migration_receipt_sha256": journal["migration_receipt_sha256"],
        "candidate_sha256": journal["candidate_sha256"],
        "active_root": journal["active_root"],
        "inactive_root": journal["inactive_root"],
        "state_dir": journal["state_dir"],
        "source_device": journal["source_device"],
        "source_inode": journal["source_inode"],
        "candidate_device": journal["candidate_device"],
        "candidate_inode": journal["candidate_inode"],
        "source_fingerprint": journal["source_fingerprint"],
        "source_file_count": journal["source_file_count"],
        "source_bytes": journal["source_bytes"],
        "rationale": journal["activation_rationale"],
        "authorization_scope": journal["activation_scope"],
        "authorized_by": journal["activation_authorized_by"],
        "reviewer_subject": journal["activation_reviewer_subject"],
        "authorized_via": journal["activation_authorized_via"],
        "journal_sha256": journal["journal_sha256"],
        "activated_at": _utc_now(),
    }


def _validate_activation_receipt(receipt: dict[str, Any], journal: dict[str, Any]) -> None:
    required = set(_activation_receipt_payload(journal)) | {"receipt_sha256"}
    if set(receipt) != required or receipt.get("schema_version") != 1:
        raise ValueError("activation receipt schema is invalid")
    immutable_fields = required - {"receipt_sha256", "journal_sha256", "activated_at"}
    expected = _activation_receipt_payload(journal)
    if any(receipt.get(field) != expected.get(field) for field in immutable_fields):
        raise ValueError("activation receipt does not match its journal evidence")
    if receipt.get("authorized_via") != "controlling_terminal":
        raise ValueError("activation receipt lacks controlling-terminal provenance")
    if type(receipt.get("activated_at")) is not str or not receipt["activated_at"]:
        raise ValueError("activation receipt timestamp is invalid")
    expected_scope = decision_scope_parts(
        "layout-activation",
        "activate",
        str(receipt["activation_id"]),
        str(receipt["plan_sha256"]),
        str(receipt["migration_transaction_id"]),
        str(receipt["migration_receipt_sha256"]),
        str(receipt["candidate_sha256"]),
        str(receipt["active_root"]),
        str(receipt["inactive_root"]),
        str(receipt["state_dir"]),
        str(receipt["source_device"]),
        str(receipt["source_inode"]),
        str(receipt["candidate_device"]),
        str(receipt["candidate_inode"]),
        _normalize_rationale(str(receipt["rationale"])),
    )
    if receipt.get("authorization_scope") != expected_scope:
        raise ValueError("activation receipt authorization scope is invalid")


@contextmanager
def _activation_lock(state_dir: Path, *, timeout: float) -> Iterator[None]:
    path = state_dir / _LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.chmod(path, 0o600)
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise OSError("activation lock is not a private unique regular file")
        import fcntl

        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for activation lock: {path}") from exc
                time.sleep(0.05)
        yield
    finally:
        with contextlib.suppress(OSError):
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _load_state(
    state_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    journal_path = state_dir / _JOURNAL_NAME
    activation_path = state_dir / _ACTIVATION_RECEIPT_NAME
    rollback_path = state_dir / _ROLLBACK_RECEIPT_NAME
    journal = (
        _read_record(journal_path, state_dir=state_dir, hash_field="journal_sha256")
        if os.path.lexists(journal_path)
        else None
    )
    activation = (
        _read_record(activation_path, state_dir=state_dir, hash_field="receipt_sha256")
        if os.path.lexists(activation_path)
        else None
    )
    rollback = (
        _read_record(rollback_path, state_dir=state_dir, hash_field="receipt_sha256")
        if os.path.lexists(rollback_path)
        else None
    )
    if journal is not None:
        _validate_journal(journal, state_dir)
    if activation is not None:
        if journal is None:
            raise ValueError("activation receipt exists without its journal")
        _validate_activation_receipt(activation, journal)
    if rollback is not None:
        if activation is None:
            raise ValueError("rollback receipt exists without an activation receipt")
        _validate_rollback_receipt(rollback, activation)
    return journal, activation, rollback


def _preflight_existing_activation(
    plan: Any,
    state_dir: Path,
    configured_context_root: Path,
    journal: dict[str, Any],
    activation_receipt: dict[str, Any] | None,
    rollback_receipt: dict[str, Any] | None,
) -> ActivationPreflight:
    active = lexical_absolute(Path(str(journal["active_root"])))
    inactive = lexical_absolute(Path(str(journal["inactive_root"])))
    _configured_root(configured_context_root, active)
    _assert_exchange_topology(active, inactive)
    evidence = _journal_evidence(journal)
    if str(getattr(plan, "plan_sha256", "")) != evidence.result.plan_hash:
        raise ValueError("activation journal does not match the requested plan")
    topology = _topology(journal, active, inactive)
    _assert_private_candidate(active if topology == "active" else inactive)
    if (
        journal["status"]
        in {
            "rollback_prepared",
            "rollback_receipt_pending",
            "rolled_back",
        }
        and rollback_receipt is None
    ):
        raise ValueError("rollback evidence is pending; use layout rollback to finalize it")
    if topology == "initial":
        fresh = verify_completed_candidate(plan)
        expected_evidence = (
            evidence.result.plan_hash,
            evidence.result.transaction_id,
            evidence.receipt_sha256,
            evidence.candidate_sha256,
            evidence.source_device,
            evidence.source_inode,
            evidence.candidate_device,
            evidence.candidate_inode,
            evidence.result.source_fingerprint,
            evidence.result.source_file_count,
            evidence.result.source_bytes,
        )
        current_evidence = (
            fresh.result.plan_hash,
            fresh.result.transaction_id,
            fresh.receipt_sha256,
            fresh.candidate_sha256,
            fresh.source_device,
            fresh.source_inode,
            fresh.candidate_device,
            fresh.candidate_inode,
            fresh.result.source_fingerprint,
            fresh.result.source_file_count,
            fresh.result.source_bytes,
        )
        if current_evidence != expected_evidence:
            raise ValueError("prepared activation evidence changed before retry")
        evidence = fresh
    status: Literal["ready", "receipt_pending", "already_active", "already_rolled_back"]
    if rollback_receipt is not None:
        if topology != "initial":
            raise ValueError("rollback receipt conflicts with the current root topology")
        status = "already_rolled_back"
    elif activation_receipt is not None:
        if topology != "active":
            raise ValueError("activation receipt conflicts with the current root topology")
        status = "already_active" if journal["status"] == "active" else "receipt_pending"
    elif topology == "active":
        status = "receipt_pending"
    else:
        status = "ready"
    return ActivationPreflight(
        status=status,
        plan=plan,
        evidence=evidence,
        state_dir=state_dir,
        active_root=active,
        inactive_root=inactive,
        activation_id=str(journal["activation_id"]),
        journal=journal,
        activation_receipt=activation_receipt,
    )


def preflight_activation(
    plan: Any,
    state_dir: Path,
    configured_context_root: Path,
) -> ActivationPreflight:
    """Verify a fresh candidate, stable routing, quiescence, and topology."""

    if os.name == "nt":
        raise ActivationPreflightError("layout activation is unavailable on Windows")
    try:
        active = lexical_absolute(Path(str(getattr(plan, "source_root", ""))))
        inactive = lexical_absolute(Path(str(getattr(plan, "destination_root", ""))))
        state = _assert_private_state_dir(
            state_dir,
            (active, inactive),
            allow_missing=True,
        )
        if not os.path.lexists(state):
            journal = activation_receipt = rollback_receipt = None
        else:
            journal, activation_receipt, rollback_receipt = _load_state(state)
        if journal is not None:
            return _preflight_existing_activation(
                plan,
                state,
                configured_context_root,
                journal,
                activation_receipt,
                rollback_receipt,
            )
        evidence = verify_completed_candidate(plan)
        if evidence.result.retained_sources or evidence.result.retained_paths:
            raise ValueError(
                "activation is blocked while the migration plan retains source-only data; "
                "import, recreate, or explicitly archive every retained path in a new plan"
            )
        _configured_root(configured_context_root, active)
        active_stat, inactive_stat = _assert_exchange_topology(active, inactive)
        _assert_private_candidate(inactive)
        if (active_stat.st_dev, active_stat.st_ino) != (
            evidence.source_device,
            evidence.source_inode,
        ) or (inactive_stat.st_dev, inactive_stat.st_ino) != (
            evidence.candidate_device,
            evidence.candidate_inode,
        ):
            raise ValueError("exchange-root identities no longer match migration evidence")
        _assert_quiescent(active, inactive)
        activation_id = _activation_id(evidence, state)
        return ActivationPreflight(
            status="ready",
            plan=plan,
            evidence=evidence,
            state_dir=state,
            active_root=active,
            inactive_root=inactive,
            activation_id=activation_id,
        )
    except (OSError, ValueError, MigrationPreflightError) as exc:
        raise ActivationPreflightError(str(exc)) from exc


def _verify_active_candidate(preflight: ActivationPreflight) -> None:
    journal = preflight.journal
    if journal is None:
        raise ValueError("activation journal is missing")
    if _topology(journal, preflight.active_root, preflight.inactive_root) != "active":
        raise ValueError("candidate is not active after atomic exchange")
    verify_relocated_candidate(
        preflight.plan,
        preflight.active_root,
        preflight.evidence,
    )
    source_evidence = tree_fingerprint(preflight.inactive_root)
    expected_source = (
        preflight.evidence.result.source_fingerprint,
        preflight.evidence.result.source_file_count,
        preflight.evidence.result.source_bytes,
    )
    if source_evidence != expected_source:
        raise ValueError("inactive v1 source changed during exchange")


def _finalize_activation(
    preflight: ActivationPreflight, journal: dict[str, Any]
) -> ActivationResult:
    # A crash immediately after exchange may leave the topology committed but
    # its parent sync incomplete. Retry the namespace durability barrier before
    # any receipt can authorize the active tree.
    strict_fsync_directory(preflight.active_root.parent)
    pending = dict(journal)
    pending["status"] = "active_receipt_pending"
    pending["updated_at"] = _utc_now()
    pending = _write_journal(preflight.state_dir, pending)
    current = ActivationPreflight(
        status="receipt_pending",
        plan=preflight.plan,
        evidence=preflight.evidence,
        state_dir=preflight.state_dir,
        active_root=preflight.active_root,
        inactive_root=preflight.inactive_root,
        activation_id=preflight.activation_id,
        journal=pending,
    )
    receipt_path = _receipt_path(preflight.state_dir, _ACTIVATION_RECEIPT_NAME)
    if not os.path.lexists(receipt_path):
        try:
            _verify_active_candidate(current)
        except (OSError, ValueError, MigrationPreflightError) as verification_error:
            try:
                _atomic_exchange(preflight.active_root, preflight.inactive_root)
                strict_fsync_directory(preflight.active_root.parent)
                reverted = dict(pending)
                reverted["status"] = "activate_prepared"
                reverted["updated_at"] = _utc_now()
                _write_journal(preflight.state_dir, reverted)
            except (OSError, ValueError) as compensation_error:
                raise ValueError(
                    "post-activation verification failed and compensation also failed; "
                    f"roots remain receipt-pending: {compensation_error}"
                ) from verification_error
            raise ValueError(
                f"post-activation verification failed; original topology restored: "
                f"{verification_error}"
            ) from verification_error
        receipt_payload = _activation_receipt_payload(pending)
        _write_new_record(receipt_path, receipt_payload, "receipt_sha256")
    strict_fsync_directory(preflight.state_dir)
    receipt = _read_record(
        receipt_path,
        state_dir=preflight.state_dir,
        hash_field="receipt_sha256",
    )
    _validate_activation_receipt(receipt, pending)
    completed = dict(pending)
    completed["status"] = "active"
    completed["updated_at"] = _utc_now()
    with contextlib.suppress(OSError, ValueError):
        _write_journal(preflight.state_dir, completed)
    return ActivationResult(
        status="activated",
        activation_id=preflight.activation_id,
        active_root=preflight.active_root,
        inactive_root=preflight.inactive_root,
        state_dir=preflight.state_dir,
        activation_receipt=receipt_path,
    )


def activate_layout(
    plan: Any,
    state_dir: Path,
    configured_context_root: Path,
    *,
    rationale: str,
    authorization: HumanAuthorization,
    lock_timeout: float = _LOCK_TIMEOUT_SECONDS,
) -> ActivationResult:
    """Atomically activate a verified candidate after fresh human approval."""

    try:
        roots = (
            lexical_absolute(Path(str(getattr(plan, "source_root", "")))),
            lexical_absolute(Path(str(getattr(plan, "destination_root", "")))),
        )
        state = _create_private_state_dir(
            state_dir,
            roots,
        )
        with _activation_lock(state, timeout=lock_timeout):
            preflight = preflight_activation(plan, state, configured_context_root)
            if preflight.status == "already_active":
                return ActivationResult(
                    status="already_active",
                    activation_id=preflight.activation_id,
                    active_root=preflight.active_root,
                    inactive_root=preflight.inactive_root,
                    state_dir=preflight.state_dir,
                    activation_receipt=_receipt_path(state, _ACTIVATION_RECEIPT_NAME),
                )
            if preflight.status == "already_rolled_back":
                raise ActivationPreflightError(
                    "this activation transaction was rolled back and cannot be replayed"
                )
            scope = layout_activation_authorization_scope(preflight, rationale)
            if not consume_human_authorization(authorization, scope=scope):
                raise ActivationApplyError(
                    "a fresh HumanDecisionBroker activation authorization is required"
                )
            _assert_quiescent(preflight.active_root, preflight.inactive_root)
            if preflight.journal is None:
                journal = _journal_from_evidence(
                    preflight,
                    rationale=rationale,
                    authorization=authorization,
                )
                journal = _write_journal(state, journal)
                prepared = ActivationPreflight(
                    **{
                        **preflight.__dict__,
                        "journal": journal,
                    }
                )
                _atomic_exchange(prepared.active_root, prepared.inactive_root)
                return _finalize_activation(prepared, journal)
            if preflight.status == "ready":
                # A prior invocation wrote the prepared journal but did not
                # exchange. Reauthorization permits exactly one exchange.
                journal = dict(preflight.journal)
                journal.update(
                    {
                        "activation_rationale": _normalize_rationale(rationale),
                        "activation_scope": scope,
                        "activation_authorized_by": authorization.identity.reviewer,
                        "activation_reviewer_subject": authorization.identity.subject,
                        "activation_authorized_via": authorization.confirmed_via,
                        "updated_at": _utc_now(),
                    }
                )
                journal = _write_journal(state, journal)
                _atomic_exchange(preflight.active_root, preflight.inactive_root)
                return _finalize_activation(preflight, journal)
            pending_journal = preflight.journal
            receipt_path = _receipt_path(state, _ACTIVATION_RECEIPT_NAME)
            if not os.path.lexists(receipt_path):
                pending_journal = dict(pending_journal)
                pending_journal.update(
                    {
                        "activation_rationale": _normalize_rationale(rationale),
                        "activation_scope": scope,
                        "activation_authorized_by": authorization.identity.reviewer,
                        "activation_reviewer_subject": authorization.identity.subject,
                        "activation_authorized_via": authorization.confirmed_via,
                        "updated_at": _utc_now(),
                    }
                )
                pending_journal = _write_journal(state, pending_journal)
            return _finalize_activation(preflight, pending_journal)
    except ActivationPreflightError:
        raise
    except ActivationApplyError:
        raise
    except BaseException as exc:  # noqa: BLE001 - preserve phase after interrupts and bugs
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ActivationApplyError(f"layout activation failed: {exc}") from exc


def _validate_rollback_receipt(
    receipt: dict[str, Any],
    activation_receipt: dict[str, Any],
) -> None:
    required = {
        "schema_version",
        "status",
        "activation_id",
        "activation_receipt_sha256",
        "active_root",
        "inactive_root",
        "state_dir",
        "preserved_v2_sha256",
        "rationale",
        "authorization_scope",
        "authorized_by",
        "reviewer_subject",
        "authorized_via",
        "rolled_back_at",
        "receipt_sha256",
    }
    if set(receipt) != required or receipt.get("schema_version") != 1:
        raise ValueError("rollback receipt schema is invalid")
    if (
        receipt.get("status") != "rolled_back"
        or receipt.get("activation_id") != activation_receipt.get("activation_id")
        or receipt.get("activation_receipt_sha256") != activation_receipt.get("receipt_sha256")
        or receipt.get("authorized_via") != "controlling_terminal"
        or receipt.get("active_root") != activation_receipt.get("active_root")
        or receipt.get("inactive_root") != activation_receipt.get("inactive_root")
        or receipt.get("state_dir") != activation_receipt.get("state_dir")
    ):
        raise ValueError("rollback receipt does not match activation evidence")
    string_fields = required - {"schema_version"}
    if any(type(receipt.get(field)) is not str or not receipt[field] for field in string_fields):
        raise ValueError("rollback receipt string fields are invalid")
    if re.fullmatch(r"[a-f0-9]{64}", str(receipt["preserved_v2_sha256"])) is None:
        raise ValueError("rollback receipt candidate digest is invalid")
    expected_scope = decision_scope_parts(
        "layout-activation",
        "rollback",
        str(receipt["activation_id"]),
        str(receipt["activation_receipt_sha256"]),
        str(receipt["preserved_v2_sha256"]),
        str(receipt["active_root"]),
        str(receipt["inactive_root"]),
        str(receipt["state_dir"]),
        _normalize_rationale(str(receipt["rationale"])),
    )
    if receipt.get("authorization_scope") != expected_scope:
        raise ValueError("rollback receipt authorization scope is invalid")


def preflight_rollback(
    state_dir: Path,
    configured_context_root: Path,
) -> RollbackPreflight:
    """Verify an active transaction and its preserved inactive v1 source."""

    if os.name == "nt":
        raise RollbackPreflightError("layout rollback is unavailable on Windows")
    try:
        # Roots are learned only from the validated private journal.
        preliminary = lexical_absolute(state_dir)
        state = _assert_private_state_dir(preliminary, (Path("/__unused1"), Path("/__unused2")))
        journal, activation_receipt, rollback_receipt = _load_state(state)
        if journal is None or activation_receipt is None:
            raise ValueError("a verified activation journal and receipt are required")
        active = lexical_absolute(Path(str(journal["active_root"])))
        inactive = lexical_absolute(Path(str(journal["inactive_root"])))
        if (
            state == active
            or state.is_relative_to(active)
            or state == inactive
            or state.is_relative_to(inactive)
        ):
            raise ValueError("activation state directory must be outside both exchange roots")
        _configured_root(configured_context_root, active)
        _assert_exchange_topology(active, inactive)
        topology = _topology(journal, active, inactive)
        _assert_private_candidate(active if topology == "active" else inactive)
        status: Literal["ready", "receipt_pending", "already_rolled_back"]
        if rollback_receipt is not None:
            _validate_rollback_receipt(rollback_receipt, activation_receipt)
            if topology != "initial":
                raise ValueError("rollback receipt conflicts with current root topology")
            status = (
                "already_rolled_back" if journal["status"] == "rolled_back" else "receipt_pending"
            )
            current_v2_sha256 = str(rollback_receipt["preserved_v2_sha256"])
            if _current_candidate_digest(inactive, journal) != current_v2_sha256:
                raise ValueError("preserved v2 tree changed after rollback")
        else:
            if topology == "initial":
                if journal.get("status") not in {
                    "rollback_prepared",
                    "rollback_receipt_pending",
                }:
                    raise ValueError("roots are rolled back without pending rollback evidence")
                status = "receipt_pending"
                current_v2_sha256 = _current_candidate_digest(inactive, journal)
            else:
                status = "ready"
                current_v2_sha256 = _current_candidate_digest(active, journal)
        if topology == "active":
            source_evidence = tree_fingerprint(inactive)
            expected = (
                str(journal["source_fingerprint"]),
                int(journal["source_file_count"]),
                int(journal["source_bytes"]),
            )
            if source_evidence != expected:
                raise ValueError("preserved inactive v1 source changed after activation")
            _assert_quiescent(active, inactive)
        return RollbackPreflight(
            status=status,
            state_dir=state,
            active_root=active,
            inactive_root=inactive,
            activation_id=str(journal["activation_id"]),
            journal=journal,
            activation_receipt=activation_receipt,
            current_v2_sha256=current_v2_sha256,
        )
    except (OSError, ValueError, MigrationPreflightError) as exc:
        raise RollbackPreflightError(str(exc)) from exc


def _current_candidate_digest(root: Path, journal: dict[str, Any]) -> str:
    return candidate_tree_sha256(root, str(journal["migration_transaction_id"]))


def _rollback_receipt_payload(
    preflight: RollbackPreflight,
    *,
    rationale: str,
    authorization: HumanAuthorization,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "rolled_back",
        "activation_id": preflight.activation_id,
        "activation_receipt_sha256": preflight.activation_receipt["receipt_sha256"],
        "active_root": str(preflight.active_root),
        "inactive_root": str(preflight.inactive_root),
        "state_dir": str(preflight.state_dir),
        "preserved_v2_sha256": preflight.current_v2_sha256,
        "rationale": _normalize_rationale(rationale),
        "authorization_scope": layout_rollback_authorization_scope(preflight, rationale),
        "authorized_by": authorization.identity.reviewer,
        "reviewer_subject": authorization.identity.subject,
        "authorized_via": authorization.confirmed_via,
        "rolled_back_at": _utc_now(),
    }


def _finalize_rollback(
    preflight: RollbackPreflight,
    *,
    rationale: str,
    authorization: HumanAuthorization,
) -> RollbackResult:
    strict_fsync_directory(preflight.active_root.parent)
    if _topology(preflight.journal, preflight.active_root, preflight.inactive_root) != "initial":
        raise ValueError("v1 is not active after atomic rollback exchange")
    receipt_path = _receipt_path(preflight.state_dir, _ROLLBACK_RECEIPT_NAME)
    if not os.path.lexists(receipt_path):
        active_fingerprint = tree_fingerprint(preflight.active_root)
        expected = (
            str(preflight.journal["source_fingerprint"]),
            int(preflight.journal["source_file_count"]),
            int(preflight.journal["source_bytes"]),
        )
        if active_fingerprint != expected:
            raise ValueError("restored v1 tree does not match activation evidence")
        receipt_payload = _rollback_receipt_payload(
            preflight,
            rationale=rationale,
            authorization=authorization,
        )
        _write_new_record(receipt_path, receipt_payload, "receipt_sha256")
    strict_fsync_directory(preflight.state_dir)
    receipt = _read_record(
        receipt_path,
        state_dir=preflight.state_dir,
        hash_field="receipt_sha256",
    )
    _validate_rollback_receipt(receipt, preflight.activation_receipt)
    journal = dict(preflight.journal)
    journal["status"] = "rolled_back"
    journal["updated_at"] = _utc_now()
    with contextlib.suppress(OSError, ValueError):
        _write_journal(preflight.state_dir, journal)
    return RollbackResult(
        status="rolled_back",
        activation_id=preflight.activation_id,
        active_root=preflight.active_root,
        inactive_root=preflight.inactive_root,
        state_dir=preflight.state_dir,
        rollback_receipt=receipt_path,
    )


def rollback_layout(
    state_dir: Path,
    configured_context_root: Path,
    *,
    rationale: str,
    authorization: HumanAuthorization,
    lock_timeout: float = _LOCK_TIMEOUT_SECONDS,
) -> RollbackResult:
    """Atomically restore v1 while preserving all v2-era writes."""

    try:
        state = _assert_private_state_dir(
            state_dir,
            (Path("/__unused1"), Path("/__unused2")),
        )
        with _activation_lock(state, timeout=lock_timeout):
            preflight = preflight_rollback(state, configured_context_root)
            if preflight.status == "already_rolled_back":
                return RollbackResult(
                    status="already_rolled_back",
                    activation_id=preflight.activation_id,
                    active_root=preflight.active_root,
                    inactive_root=preflight.inactive_root,
                    state_dir=state,
                    rollback_receipt=_receipt_path(state, _ROLLBACK_RECEIPT_NAME),
                )
            scope = layout_rollback_authorization_scope(preflight, rationale)
            if not consume_human_authorization(authorization, scope=scope):
                raise RollbackApplyError(
                    "a fresh HumanDecisionBroker rollback authorization is required"
                )
            _assert_quiescent(preflight.active_root, preflight.inactive_root)
            journal = dict(preflight.journal)
            journal.update(
                {
                    "status": "rollback_prepared",
                    "updated_at": _utc_now(),
                }
            )
            journal = _write_journal(state, journal)
            prepared = RollbackPreflight(
                **{
                    **preflight.__dict__,
                    "journal": journal,
                }
            )
            if preflight.status == "ready":
                _atomic_exchange(prepared.active_root, prepared.inactive_root)
            pending_journal = dict(journal)
            pending_journal["status"] = "rollback_receipt_pending"
            pending_journal["updated_at"] = _utc_now()
            pending_journal = _write_journal(state, pending_journal)
            pending = RollbackPreflight(
                **{
                    **prepared.__dict__,
                    "journal": pending_journal,
                }
            )
            return _finalize_rollback(
                pending,
                rationale=rationale,
                authorization=authorization,
            )
    except RollbackPreflightError:
        raise
    except RollbackApplyError:
        raise
    except BaseException as exc:  # noqa: BLE001 - preserve phase after interrupts and bugs
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise RollbackApplyError(f"layout rollback failed: {exc}") from exc
