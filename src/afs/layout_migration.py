"""Fail-closed execution for separate-destination context layout migrations.

Planning remains in :mod:`afs.context_layout`.  This module deliberately has
no in-place, activation, source-deletion, or rollback path: a successful run
only creates a new, independently verifiable v2 tree next to an untouched v1
source.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import time
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

from . import context_layout
from .atomic_io import atomic_write_text, strict_fsync_directory
from .human_provenance import (
    HumanAuthorization,
    consume_human_authorization,
    decision_scope_parts,
)
from .models import ContextCategory
from .path_safety import assert_no_linklike_components, is_linklike, lexical_absolute

__all__ = [
    "LayoutMigrationError",
    "MigrationApplyError",
    "MigrationPreflight",
    "MigrationPreflightError",
    "MigrationResult",
    "VerifiedMigrationCandidate",
    "apply_migration",
    "candidate_tree_sha256",
    "layout_migration_authorization_scope",
    "preflight_migration",
    "source_tree_fingerprint",
    "tree_fingerprint",
    "verify_completed_candidate",
    "verify_relocated_candidate",
]

_COPY_CHUNK_SIZE = 1024 * 1024
_LOCK_TIMEOUT_SECONDS = 30.0
_SPACE_FLOOR_BYTES = 16 * 1024 * 1024
_MAX_RATIONALE_CHARACTERS = 4096
_RECEIPT_DIRECTORY = Path(context_layout.AFS_STATE_DIR) / "migrations"
_MAX_RECEIPT_BYTES = 1024 * 1024
_MAX_MARKER_BYTES = 64 * 1024
_MAX_README_BYTES = 64 * 1024
_WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}.union(
        (f"com{number}" for number in range(1, 10)),
        (f"lpt{number}" for number in range(1, 10)),
    )
)


class LayoutMigrationError(RuntimeError):
    """Base class for migration execution failures."""


class MigrationPreflightError(LayoutMigrationError):
    """Raised when a plan or either filesystem tree is unsafe to execute."""


class MigrationApplyError(LayoutMigrationError):
    """Raised after an authorized apply fails.

    ``failed_destination`` identifies the quarantined partial tree when AFS
    managed to move it out of the requested destination name.
    """

    def __init__(self, message: str, *, failed_destination: Path | None = None) -> None:
        super().__init__(message)
        self.failed_destination = failed_destination


@dataclass(frozen=True)
class _Retention:
    source_relative: Path
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source_relative.as_posix(),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MigrationPreflight:
    """Read-only evidence that a migration plan is presently executable."""

    plan: Any
    plan_hash: str
    source_root: Path
    destination_root: Path
    required_bytes: int
    available_bytes: int
    source_file_count: int
    source_bytes: int
    copy_file_count: int
    copy_bytes: int
    mapping_count: int
    retained_sources: tuple[_Retention, ...] = ()
    retained_paths: tuple[_Retention, ...] = ()
    status: Literal["ready", "already_applied"] = "ready"

    def to_dict(self) -> dict[str, Any]:
        operations = [
            {
                "source": str(self.source_root / operation.source_relative),
                "destination": str(self.destination_root / operation.destination_relative),
            }
            for operation in _operations(self.plan, self.source_root, self.destination_root)
        ]
        payload: dict[str, Any] = {
            "status": self.status,
            "transaction_id": str(self.plan.transaction_id),
            "plan_sha256": self.plan_hash,
            "plan_hash": self.plan_hash,
            "source_root": str(self.source_root),
            "destination_root": str(self.destination_root),
            "source_fingerprint": str(self.plan.source_fingerprint),
            "source_files": self.source_file_count,
            "source_bytes": self.source_bytes,
            "operations": operations,
            "operation_count": self.mapping_count,
            "ready": True,
            "blockers": [],
            "required_bytes": self.required_bytes,
            "available_bytes": self.available_bytes,
        }
        if _plan_schema_version(self.plan) == 3:
            payload.update(
                {
                    "copy_files": self.copy_file_count,
                    "copy_bytes": self.copy_bytes,
                    "retained_source_count": len(self.retained_sources),
                    "retained_path_count": len(self.retained_paths),
                    "retained_sources": [item.to_dict() for item in self.retained_sources],
                    "retained_paths": [item.to_dict() for item in self.retained_paths],
                }
            )
        return payload


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of applying, or recognizing, one completed transaction."""

    status: Literal["applied", "already_applied"]
    plan_hash: str
    transaction_id: str
    source_root: Path
    destination_root: Path
    source_fingerprint: str
    source_file_count: int
    source_bytes: int
    copy_file_count: int
    copy_bytes: int
    operation_count: int
    receipt_path: Path
    plan_schema_version: int = 2
    retained_sources: tuple[_Retention, ...] = ()
    retained_paths: tuple[_Retention, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "plan_sha256": self.plan_hash,
            "plan_hash": self.plan_hash,
            "transaction_id": self.transaction_id,
            "source_root": str(self.source_root),
            "destination_root": str(self.destination_root),
            "source_fingerprint": self.source_fingerprint,
            "source_files": self.source_file_count,
            "source_bytes": self.source_bytes,
            "source_unchanged": True,
            "operations": self.operation_count,
            "receipt_path": str(self.receipt_path),
        }
        if self.plan_schema_version == 3:
            payload.update(
                {
                    "copy_files": self.copy_file_count,
                    "copy_bytes": self.copy_bytes,
                    "retained_source_count": len(self.retained_sources),
                    "retained_path_count": len(self.retained_paths),
                    "retained_sources": [item.to_dict() for item in self.retained_sources],
                    "retained_paths": [item.to_dict() for item in self.retained_paths],
                }
            )
        return payload


@dataclass(frozen=True)
class VerifiedMigrationCandidate:
    """Path-independent evidence for one completed migration candidate.

    Activation deliberately moves the candidate away from the destination
    recorded in its immutable migration receipt.  Callers must therefore
    capture this evidence while the original source and destination paths
    still match the reviewed plan.
    """

    result: MigrationResult
    candidate_sha256: str
    receipt_sha256: str
    source_device: int
    source_inode: int
    candidate_device: int
    candidate_inode: int

    def to_dict(self) -> dict[str, Any]:
        payload = self.result.to_dict()
        payload.update(
            {
                "candidate_sha256": self.candidate_sha256,
                "migration_receipt_sha256": self.receipt_sha256,
                "source_device": self.source_device,
                "source_inode": self.source_inode,
                "candidate_device": self.candidate_device,
                "candidate_inode": self.candidate_inode,
            }
        )
        return payload


@dataclass(frozen=True)
class _Mapping:
    source_relative: Path
    destination_relative: Path


@dataclass(frozen=True)
class _Operation:
    source_relative: Path
    destination_relative: Path


@dataclass(frozen=True)
class _Entry:
    relative: Path
    kind: Literal["file", "directory"]
    size: int


def layout_migration_authorization_scope(
    plan_sha256: str,
    transaction_id: str,
    rationale: str,
) -> str:
    """Return the exact broker scope required by :func:`apply_migration`."""

    normalized_hash = plan_sha256.strip().lower()
    normalized_transaction = transaction_id.strip()
    normalized_rationale = rationale.strip()
    if not re.fullmatch(r"[a-f0-9]{64}", normalized_hash):
        raise ValueError("migration plan SHA-256 must be 64 lowercase hexadecimal characters")
    if not normalized_rationale:
        raise ValueError("a human-authored migration rationale is required")
    if len(normalized_rationale) > _MAX_RATIONALE_CHARACTERS:
        raise ValueError(f"migration rationale exceeds {_MAX_RATIONALE_CHARACTERS} characters")
    if any(unicodedata.category(character).startswith("C") for character in normalized_rationale):
        raise ValueError("migration rationale contains control or formatting characters")
    if not re.fullmatch(r"layout_[a-f0-9]{32}", normalized_transaction):
        raise ValueError("migration transaction id is invalid")
    return decision_scope_parts(
        "layout-migration",
        "apply",
        normalized_hash,
        normalized_transaction,
        normalized_rationale,
    )


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
    elif is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)  # type: ignore[arg-type]
    else:
        raise MigrationPreflightError("migration plan is not a supported schema object")
    if not isinstance(payload, dict):
        raise MigrationPreflightError("migration plan payload must be an object")
    return payload


def _load_plan(value: Any) -> Any:
    if isinstance(value, (str, os.PathLike)):
        loader = getattr(context_layout, "load_migration_plan", None)
        if loader is None:
            raise MigrationPreflightError("migration plan loading is unavailable")
        try:
            return loader(Path(value))
        except (OSError, TypeError, ValueError) as exc:
            raise MigrationPreflightError(f"invalid migration plan: {exc}") from exc
    return value


def _canonical_plan_hash(plan: Any) -> str:
    payload = _payload(plan)
    stored = payload.get("plan_sha256", getattr(plan, "plan_sha256", ""))
    if not isinstance(stored, str) or not re.fullmatch(r"[a-f0-9]{64}", stored):
        raise MigrationPreflightError("migration plan is missing its canonical plan hash")

    hasher = getattr(plan, "canonical_sha256", None)
    if callable(hasher):
        try:
            computed = hasher()
        except (TypeError, ValueError) as exc:
            raise MigrationPreflightError(f"cannot hash migration plan: {exc}") from exc
    else:
        unhashed = dict(payload)
        unhashed.pop("plan_sha256", None)
        computed = hashlib.sha256(
            json.dumps(
                unhashed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    if not isinstance(computed, str) or not computed:
        raise MigrationPreflightError("migration plan hash function returned an invalid value")
    if not secrets_compare(stored, computed):
        raise MigrationPreflightError("migration plan hash does not match its contents")
    return stored


def secrets_compare(left: str, right: str) -> bool:
    """Constant-time text equality without accepting Unicode hash lookalikes."""

    try:
        return hmac.compare_digest(left.encode("ascii"), right.encode("ascii"))
    except UnicodeEncodeError:
        return False


def _relative_path(value: Any, *, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MigrationPreflightError(f"mapping {field_name} must be a non-empty string")
    candidate = Path(value)
    if candidate.is_absolute() or candidate == Path(".") or ".." in candidate.parts:
        raise MigrationPreflightError(f"mapping {field_name} must be a safe relative path")
    for part in candidate.parts:
        _validate_name(part)
    return candidate


def _plan_schema_version(plan: Any) -> int:
    value = getattr(plan, "schema_version", None)
    if type(value) is not int or value not in {2, 3}:
        raise MigrationPreflightError("migration execution requires a schema-v2 or schema-v3 plan")
    return value


def _retained_relative_path(value: Any, *, field_name: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise MigrationPreflightError(f"{field_name} must be a safe relative POSIX path")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or value in {".", ".."}
        or ".." in candidate.parts
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise MigrationPreflightError(f"{field_name} must be a normalized relative POSIX path")
    return Path(*candidate.parts)


def _retentions(plan: Any, field_name: str) -> tuple[_Retention, ...]:
    schema_version = _plan_schema_version(plan)
    raw = getattr(plan, field_name, None)
    if raw is None:
        raw = _payload(plan).get(field_name, ())
    if not isinstance(raw, (list, tuple)):
        raise MigrationPreflightError(f"migration plan {field_name} must be a sequence")
    result: list[_Retention] = []
    for item in raw:
        data = _payload(item) if not isinstance(item, dict) else item
        if set(data) != {"source", "reason"}:
            raise MigrationPreflightError(
                f"migration plan {field_name} entries require only source and reason"
            )
        source = _retained_relative_path(
            data.get("source"),
            field_name=f"migration plan {field_name} source",
        )
        reason = data.get("reason")
        if (
            not isinstance(reason, str)
            or not reason
            or reason != reason.strip()
            or len(reason) > 1024
            or any(unicodedata.category(character).startswith("C") for character in reason)
        ):
            raise MigrationPreflightError(
                f"migration plan {field_name} reason must be reviewed non-empty text"
            )
        if field_name == "retained_sources" and len(source.parts) != 1:
            raise MigrationPreflightError(
                "migration plan retained_sources entries must be top-level paths"
            )
        if field_name == "retained_paths" and len(source.parts) < 2:
            raise MigrationPreflightError(
                "migration plan retained_paths entries must be nested paths"
            )
        result.append(_Retention(source, reason))
    ordered = tuple(sorted(result, key=lambda item: item.source_relative.as_posix()))
    if tuple(result) != ordered:
        raise MigrationPreflightError(f"migration plan {field_name} must be sorted")
    paths = [item.source_relative.parts for item in ordered]
    if len(set(paths)) != len(paths):
        raise MigrationPreflightError(f"migration plan {field_name} paths must be unique")
    for index, first in enumerate(paths):
        for second in paths[index + 1 :]:
            if first[: len(second)] == second or second[: len(first)] == first:
                raise MigrationPreflightError(f"migration plan {field_name} paths must not overlap")
    if schema_version == 2 and ordered:
        raise MigrationPreflightError("schema-v2 migration plans cannot retain source paths")
    return ordered


def _all_retained_paths(plan: Any) -> tuple[str, ...]:
    return tuple(
        item.source_relative.as_posix()
        for item in (*_retentions(plan, "retained_sources"), *_retentions(plan, "retained_paths"))
    )


def _validate_retention_spellings(plan: Any, source_root: Path) -> None:
    for item in (*_retentions(plan, "retained_sources"), *_retentions(plan, "retained_paths")):
        try:
            context_layout._resolve_exact_retention_path(
                source_root,
                item.source_relative.as_posix(),
                field_name="retained source",
            )
        except (OSError, ValueError) as exc:
            raise MigrationPreflightError(f"invalid source-only exclusion: {exc}") from exc


def _operation_retained_paths(plan: Any, operation: _Operation) -> tuple[Path, ...]:
    root = operation.source_relative.parts[0]
    return tuple(
        Path(*item.source_relative.parts[1:])
        for item in _retentions(plan, "retained_paths")
        if item.source_relative.parts[0] == root
    )


def _is_retained_path(relative: Path, retained_paths: tuple[Path, ...]) -> bool:
    if relative == Path("."):
        return False
    return any(
        relative.parts[: len(retained.parts)] == retained.parts for retained in retained_paths
    )


def _mappings(plan: Any) -> tuple[_Mapping, ...]:
    raw = getattr(plan, "explicit_mappings", None)
    if raw is None:
        raw = _payload(plan).get("explicit_mappings")
    if not isinstance(raw, (list, tuple)):
        raise MigrationPreflightError("migration plan explicit_mappings must be a sequence")
    result: list[_Mapping] = []
    for item in raw:
        data = _payload(item) if not isinstance(item, dict) else item
        source = data.get("source")
        destination = data.get("destination")
        result.append(
            _Mapping(
                _relative_path(source, field_name="source_relative"),
                _relative_path(destination, field_name="destination_relative"),
            )
        )
    return tuple(result)


def _operations(plan: Any, source_root: Path, destination_root: Path) -> tuple[_Operation, ...]:
    raw = getattr(plan, "operations", None)
    if raw is None:
        raw = _payload(plan).get("operations")
    if not isinstance(raw, (list, tuple)):
        raise MigrationPreflightError("migration plan operations must be a sequence")
    result: list[_Operation] = []
    for item in raw:
        data = _payload(item) if not isinstance(item, dict) else item
        if data.get("operation") != "copy_verify" or data.get("verify") != "sha256":
            raise MigrationPreflightError("migration executor only accepts copy_verify/sha256")
        raw_source = data.get("source")
        raw_destination = data.get("destination")
        if not isinstance(raw_source, str) or not isinstance(raw_destination, str):
            raise MigrationPreflightError("migration operation paths must be strings")
        operation_source = lexical_absolute(Path(raw_source))
        operation_destination = lexical_absolute(Path(raw_destination))
        try:
            source_relative = operation_source.relative_to(source_root)
            destination_relative = operation_destination.relative_to(destination_root)
        except ValueError as exc:
            raise MigrationPreflightError("migration operation escapes a declared root") from exc
        if len(source_relative.parts) != 1:
            raise MigrationPreflightError(
                "migration operation source must be one direct child of the source root"
            )
        if destination_relative == Path("."):
            raise MigrationPreflightError("migration operation cannot replace the destination root")
        for part in source_relative.parts + destination_relative.parts:
            _validate_name(part)
        if str(operation_source) != raw_source or str(operation_destination) != raw_destination:
            raise MigrationPreflightError("migration operation paths must use canonical spelling")
        result.append(_Operation(source_relative, destination_relative))
    if len({_collision_key(item.source_relative) for item in result}) != len(result):
        raise MigrationPreflightError("migration operation sources collide")
    return tuple(result)


def _validate_name(name: str) -> None:
    if name in {"", ".", ".."}:
        raise MigrationPreflightError(f"unsafe path name: {name!r}")
    if name[-1:] in {" ", "."}:
        raise MigrationPreflightError(f"non-portable trailing character in name: {name!r}")
    if any(unicodedata.category(character).startswith("C") for character in name):
        raise MigrationPreflightError(f"control or formatting character in path name: {name!r}")
    invalid = '<>:"/\\|?*'
    if any(character in invalid for character in name):
        raise MigrationPreflightError(f"non-portable character in path name: {name!r}")
    stem = name.split(".", 1)[0].casefold()
    if stem in _WINDOWS_RESERVED_NAMES:
        raise MigrationPreflightError(f"reserved path name: {name!r}")


def _collision_key(path: Path) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFKC", part).casefold() for part in path.parts)


def _lstat(path: Path, *, label: str) -> os.stat_result:
    try:
        value = os.lstat(path)
    except OSError as exc:
        raise MigrationPreflightError(f"cannot inspect {label} {path}: {exc}") from exc
    if is_linklike(value):
        raise MigrationPreflightError(f"{label} is a symbolic link or reparse point: {path}")
    return value


def _walk_source(
    path: Path,
    *,
    relative: Path = Path("."),
    retained_paths: tuple[Path, ...] = (),
) -> Iterator[_Entry]:
    if _is_retained_path(relative, retained_paths):
        return
    path_stat = _lstat(path, label="migration source")
    if stat.S_ISREG(path_stat.st_mode):
        if path_stat.st_nlink != 1:
            raise MigrationPreflightError(f"hard-linked source file is not allowed: {path}")
        yield _Entry(relative, "file", path_stat.st_size)
        return
    if not stat.S_ISDIR(path_stat.st_mode):
        raise MigrationPreflightError(f"special source file is not allowed: {path}")
    if relative != Path("."):
        yield _Entry(relative, "directory", 0)
    try:
        children = sorted(os.scandir(path), key=lambda entry: entry.name)
    except OSError as exc:
        raise MigrationPreflightError(f"cannot enumerate migration source {path}: {exc}") from exc
    for child in children:
        child_relative = Path(child.name) if relative == Path(".") else relative / child.name
        if _is_retained_path(child_relative, retained_paths):
            continue
        _validate_name(child.name)
        yield from _walk_source(
            Path(child.path),
            relative=child_relative,
            retained_paths=retained_paths,
        )


def _scaffold_entries(transaction_id: str) -> dict[tuple[str, ...], str]:
    entries: dict[tuple[str, ...], str] = {}

    def add(path: Path, kind: str) -> None:
        key = _collision_key(path)
        current = entries.get(key)
        if current is not None and current != kind:
            raise AssertionError(f"internal scaffold collision at {path}")
        entries[key] = kind

    add(Path(context_layout.AFS_STATE_DIR), "directory")
    for category in ContextCategory:
        add(Path(category.value), "directory")
    for relative in context_layout.V2_COMPAT_MOUNT_PATHS.values():
        path = Path(relative)
        for parent in reversed(path.parents):
            if parent != Path("."):
                add(parent, "directory")
        add(path, "directory")
    for relative in context_layout.V2_SYSTEM_PATHS.values():
        path = Path(relative)
        for parent in reversed(path.parents):
            if parent != Path("."):
                add(parent, "directory")
        add(path, "directory")
    add(Path("README.md"), "file")
    add(Path(context_layout.AFS_STATE_DIR) / context_layout.LAYOUT_FILE, "file")
    receipt = _receipt_path(Path("."), transaction_id)
    for parent in reversed(receipt.parents):
        if parent != Path("."):
            add(parent, "directory")
    add(receipt, "file")
    return entries


def _validate_operation_trees(
    source: Path,
    destination: Path,
    operations: tuple[_Operation, ...],
    transaction_id: str,
    retained_paths: tuple[_Retention, ...] = (),
) -> tuple[int, int]:
    seen_sources: set[tuple[str, ...]] = set()
    targets = _scaffold_entries(transaction_id)
    target_owners: dict[tuple[str, ...], int | None] = dict.fromkeys(targets)
    file_count = 0
    total_bytes = 0
    for operation_index, operation in enumerate(operations):
        source_key = _collision_key(operation.source_relative)
        if source_key in seen_sources:
            raise MigrationPreflightError(
                f"duplicate or case-folded source operation: {operation.source_relative}"
            )
        seen_sources.add(source_key)
        source_path = source / operation.source_relative
        source_stat = _lstat(source_path, label="mapped source")
        target_root = destination / operation.destination_relative
        try:
            target_root.relative_to(destination)
        except ValueError as exc:
            raise MigrationPreflightError(
                "mapping destination escapes the destination root"
            ) from exc
        root_kind = "directory" if stat.S_ISDIR(source_stat.st_mode) else "file"
        root_key = _collision_key(operation.destination_relative)
        for part_count in range(1, len(root_key)):
            if targets.get(root_key[:part_count]) == "file":
                raise MigrationPreflightError(
                    f"file/directory collision at {operation.destination_relative}"
                )
        if root_kind == "file" and any(
            key[: len(root_key)] == root_key and key != root_key for key in targets
        ):
            raise MigrationPreflightError(
                f"file/directory collision at {operation.destination_relative}"
            )
        existing_root_kind = targets.get(root_key)
        if existing_root_kind is not None and existing_root_kind != root_kind:
            raise MigrationPreflightError(
                f"file/directory collision at {operation.destination_relative}"
            )
        existing_owner = target_owners.get(root_key)
        if existing_owner is not None:
            raise MigrationPreflightError(
                f"destination operation collision at {operation.destination_relative}"
            )
        targets[root_key] = root_kind
        target_owners[root_key] = operation_index
        operation_retained = tuple(
            Path(*item.source_relative.parts[1:])
            for item in retained_paths
            if item.source_relative.parts[0] == operation.source_relative.parts[0]
        )
        for entry in _walk_source(source_path, retained_paths=operation_retained):
            target = (
                operation.destination_relative
                if entry.relative == Path(".")
                else operation.destination_relative / entry.relative
            )
            key = _collision_key(target)
            current = targets.get(key)
            owner = target_owners.get(key)
            if current is not None:
                same_operation_root = (
                    entry.relative == Path(".")
                    and owner == operation_index
                    and current == entry.kind
                )
                scaffold_directory = owner is None and current == entry.kind == "directory"
                if not (same_operation_root or scaffold_directory):
                    raise MigrationPreflightError(f"destination name collision at {target}")
            for part_count in range(1, len(key)):
                if targets.get(key[:part_count]) == "file":
                    raise MigrationPreflightError(f"file/directory collision at {target}")
            if entry.kind == "file" and any(
                existing[: len(key)] == key and existing != key for existing in targets
            ):
                raise MigrationPreflightError(f"file/directory collision at {target}")
            targets[key] = entry.kind
            target_owners[key] = operation_index
            if entry.kind == "file":
                file_count += 1
                total_bytes += entry.size
    return file_count, total_bytes


def _tree_fingerprint(
    root: Path,
    *,
    retained_paths: tuple[str, ...] = (),
) -> tuple[str, int, int]:
    try:
        if retained_paths:
            value = context_layout._tree_fingerprint(root, retained_paths=retained_paths)
        else:
            value = context_layout._tree_fingerprint(root)
    except (OSError, ValueError) as exc:
        raise MigrationPreflightError(f"cannot inventory migration source: {exc}") from exc
    if isinstance(value, tuple) and len(value) == 3:
        return str(value[0]), int(value[1]), int(value[2])
    digest = getattr(value, "digest", getattr(value, "fingerprint", ""))
    count = getattr(value, "file_count", 0)
    size = getattr(value, "total_bytes", getattr(value, "source_bytes", 0))
    if not isinstance(digest, str) or not digest:
        raise MigrationPreflightError("source fingerprint function returned invalid evidence")
    return digest, int(count), int(size)


def _semantic_plan(plan: Any) -> tuple[Any, ...]:
    source = lexical_absolute(Path(str(getattr(plan, "source_root", ""))))
    destination = lexical_absolute(Path(str(plan.destination_root)))
    return (
        _plan_schema_version(plan),
        str(getattr(plan, "source_root", "")),
        str(getattr(plan, "destination_root", "")),
        str(getattr(plan, "source_fingerprint", "")),
        int(getattr(plan, "source_file_count", -1)),
        int(getattr(plan, "source_bytes", -1)),
        int(getattr(plan, "source_device", -1)),
        int(getattr(plan, "source_inode", -1)),
        bool(getattr(plan, "ready", False)),
        tuple(getattr(plan, "blocking_entries", ())),
        tuple(
            (item.source_relative.as_posix(), item.destination_relative.as_posix())
            for item in _mappings(plan)
        ),
        tuple(
            (item.source_relative.as_posix(), item.reason)
            for item in _retentions(plan, "retained_sources")
        ),
        tuple(
            (item.source_relative.as_posix(), item.reason)
            for item in _retentions(plan, "retained_paths")
        ),
        int(getattr(plan, "copy_file_count", -1)),
        int(getattr(plan, "copy_bytes", -1)),
        tuple(
            (item.source_relative.as_posix(), item.destination_relative.as_posix())
            for item in _operations(plan, source, destination)
        ),
    )


def _recompute_plan(plan: Any, source: Path, destination: Path) -> Any:
    explicit_mappings = {
        mapping.source_relative.as_posix(): mapping.destination_relative.as_posix()
        for mapping in _mappings(plan)
    }
    retained_sources = {
        item.source_relative.as_posix(): item.reason
        for item in _retentions(plan, "retained_sources")
    }
    retained_paths = {
        item.source_relative.as_posix(): item.reason for item in _retentions(plan, "retained_paths")
    }
    try:
        recomputed = context_layout.build_migration_plan(
            source,
            destination,
            explicit_mappings=explicit_mappings,
            retained_sources=retained_sources,
            retained_paths=retained_paths,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise MigrationPreflightError(f"cannot recompute migration plan: {exc}") from exc
    if _semantic_plan(recomputed) != _semantic_plan(plan):
        raise MigrationPreflightError("migration plan is stale or semantically tampered")
    return recomputed


def _path_relation_error(source: Path, destination: Path) -> str | None:
    if source == destination:
        return "in-place migration is not supported"
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        return "source and destination roots must not contain one another"
    try:
        real_source = source.resolve(strict=True)
        real_parent = destination.parent.resolve(strict=True)
        real_destination = real_parent / destination.name
    except OSError as exc:
        return f"cannot resolve migration roots: {exc}"
    if real_source == real_destination:
        return "source and destination resolve to the same root"
    if real_destination.is_relative_to(real_source) or real_source.is_relative_to(real_destination):
        return "resolved source and destination roots must not contain one another"
    return None


def _destination_exists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _has_group_or_world_permissions(path_stat: os.stat_result) -> bool:
    """Return POSIX public-mode exposure; Windows needs ACL-aware review."""

    return os.name != "nt" and bool(stat.S_IMODE(path_stat.st_mode) & 0o077)


def _supports_private_destination_permissions() -> bool:
    """Return whether this executor can create and verify private trees."""

    return os.name != "nt"


def _require_private_destination_permissions() -> None:
    if not _supports_private_destination_permissions():
        raise MigrationPreflightError(
            "layout migration apply is unavailable on Windows until AFS can create "
            "and verify a private destination DACL"
        )


def preflight_migration(plan_or_path: Any) -> MigrationPreflight:
    """Validate a migration without creating, locking, or writing anything."""

    _require_private_destination_permissions()
    plan = _load_plan(plan_or_path)
    schema_version = _plan_schema_version(plan)
    if not bool(getattr(plan, "ready", False)):
        raise MigrationPreflightError("migration plan is not ready")
    if tuple(getattr(plan, "blocking_entries", ())):
        raise MigrationPreflightError("migration plan contains blocking entries")
    plan_hash = _canonical_plan_hash(plan)
    retained_sources = _retentions(plan, "retained_sources")
    retained_paths = _retentions(plan, "retained_paths")
    source = lexical_absolute(Path(str(getattr(plan, "source_root", ""))))
    destination = lexical_absolute(Path(str(getattr(plan, "destination_root", ""))))
    if not str(getattr(plan, "source_root", "")) or not str(getattr(plan, "destination_root", "")):
        raise MigrationPreflightError("migration plan roots are required")
    try:
        assert_no_linklike_components(source, boundary=Path(source.anchor), allow_missing=False)
        assert_no_linklike_components(
            destination.parent,
            boundary=Path(destination.anchor),
            allow_missing=False,
        )
    except (OSError, ValueError) as exc:
        raise MigrationPreflightError(f"unsafe migration root: {exc}") from exc
    _validate_retention_spellings(plan, source)
    completed = _completed_result(plan)
    if completed is not None:
        return MigrationPreflight(
            plan=plan,
            plan_hash=plan_hash,
            source_root=source,
            destination_root=destination,
            required_bytes=0,
            available_bytes=shutil.disk_usage(destination.parent).free,
            source_file_count=plan.source_file_count,
            source_bytes=plan.source_bytes,
            copy_file_count=int(getattr(plan, "copy_file_count", plan.source_file_count)),
            copy_bytes=int(getattr(plan, "copy_bytes", plan.source_bytes)),
            mapping_count=len(plan.operations),
            retained_sources=retained_sources,
            retained_paths=retained_paths,
            status="already_applied",
        )
    source_stat = _lstat(source, label="source root")
    if not stat.S_ISDIR(source_stat.st_mode):
        raise MigrationPreflightError(f"source root is not a directory: {source}")
    if not destination.parent.is_dir():
        raise MigrationPreflightError(
            f"destination parent is not a directory: {destination.parent}"
        )
    if _destination_exists(destination):
        raise MigrationPreflightError(
            f"destination already exists and is not a verified completed candidate: {destination}"
        )
    relation_error = _path_relation_error(source, destination)
    if relation_error:
        raise MigrationPreflightError(relation_error)
    if int(getattr(plan, "source_device", -1)) != int(source_stat.st_dev) or int(
        getattr(plan, "source_inode", -1)
    ) != int(source_stat.st_ino):
        raise MigrationPreflightError("migration source identity changed after planning")

    operations = _operations(plan, source, destination)
    transaction_id = str(getattr(plan, "transaction_id", ""))
    if not transaction_id or any(
        unicodedata.category(character).startswith("C") for character in transaction_id
    ):
        raise MigrationPreflightError("migration transaction id is unsafe")
    file_count, total_bytes = _validate_operation_trees(
        source,
        destination,
        operations,
        transaction_id,
        retained_paths,
    )
    fingerprint, fingerprint_count, fingerprint_bytes = _tree_fingerprint(
        source,
        retained_paths=_all_retained_paths(plan),
    )
    expected_fingerprint = str(getattr(plan, "source_fingerprint", ""))
    if (
        fingerprint != expected_fingerprint
        or fingerprint_count != int(getattr(plan, "source_file_count", -1))
        or fingerprint_bytes != int(getattr(plan, "source_bytes", -1))
    ):
        raise MigrationPreflightError("migration source changed after planning")
    expected_copy_count = int(getattr(plan, "copy_file_count", -1))
    expected_copy_bytes = int(getattr(plan, "copy_bytes", -1))
    if file_count != expected_copy_count or total_bytes != expected_copy_bytes:
        raise MigrationPreflightError(
            "migration operations do not match the reviewed candidate copy set"
        )
    if schema_version == 2 and (
        file_count != fingerprint_count or total_bytes != fingerprint_bytes
    ):
        raise MigrationPreflightError("explicit mappings do not cover the complete source plan")

    _recompute_plan(plan, source, destination)
    required = total_bytes + max(_SPACE_FLOOR_BYTES, total_bytes // 20)
    try:
        available = shutil.disk_usage(destination.parent).free
    except OSError as exc:
        raise MigrationPreflightError(f"cannot determine destination capacity: {exc}") from exc
    if available < required:
        raise MigrationPreflightError(
            f"insufficient destination space: require {required} bytes, have {available}"
        )
    return MigrationPreflight(
        plan=plan,
        plan_hash=plan_hash,
        source_root=source,
        destination_root=destination,
        required_bytes=required,
        available_bytes=available,
        source_file_count=fingerprint_count,
        source_bytes=fingerprint_bytes,
        copy_file_count=file_count,
        copy_bytes=total_bytes,
        mapping_count=len(operations),
        retained_sources=retained_sources,
        retained_paths=retained_paths,
    )


def _receipt_path(destination: Path, transaction_id: str) -> Path:
    return destination / _RECEIPT_DIRECTORY / transaction_id / "receipt.json"


def _receipt_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"migration receipt contains duplicate field {key!r}")
        result[key] = value
    return result


def _read_stable_regular(
    path: Path,
    *,
    boundary: Path,
    maximum_bytes: int,
    label: str,
) -> tuple[bytes, os.stat_result]:
    assert_no_linklike_components(path, boundary=boundary, allow_missing=False)
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{label} must be a unique regular file")
    if before.st_size > maximum_bytes:
        raise ValueError(f"{label} exceeds its {maximum_bytes}-byte limit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise ValueError(f"{label} changed while opening")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 64 * 1024):
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"{label} exceeds its {maximum_bytes}-byte limit")
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
        raise ValueError(f"{label} changed while reading")
    return b"".join(chunks), final


def _read_receipt(path: Path, *, boundary: Path) -> tuple[dict[str, Any], os.stat_result]:
    document, path_stat = _read_stable_regular(
        path,
        boundary=boundary,
        maximum_bytes=_MAX_RECEIPT_BYTES,
        label="migration receipt",
    )
    payload = json.loads(
        document.decode("utf-8"),
        object_pairs_hook=_receipt_object,
    )
    if not isinstance(payload, dict):
        raise ValueError("migration receipt must be a JSON object")
    return payload, path_stat


def _completed_result(plan: Any) -> MigrationResult | None:
    destination = lexical_absolute(Path(str(getattr(plan, "destination_root", ""))))
    source = lexical_absolute(Path(str(getattr(plan, "source_root", ""))))
    try:
        assert_no_linklike_components(
            source,
            boundary=Path(source.anchor),
            allow_missing=False,
        )
        assert_no_linklike_components(
            destination,
            boundary=Path(destination.anchor),
            allow_missing=False,
        )
    except (OSError, ValueError):
        return None
    if not _destination_exists(destination):
        return None
    try:
        destination_stat = os.lstat(destination)
    except OSError:
        return None
    if is_linklike(destination_stat) or not stat.S_ISDIR(destination_stat.st_mode):
        return None
    transaction_id = str(getattr(plan, "transaction_id", ""))
    try:
        plan_schema_version = _plan_schema_version(plan)
        retained_sources = _retentions(plan, "retained_sources")
        retained_paths = _retentions(plan, "retained_paths")
        _validate_retention_spellings(plan, source)
    except MigrationPreflightError:
        return None
    receipt_path = _receipt_path(destination, transaction_id)
    try:
        receipt, receipt_stat = _read_receipt(receipt_path, boundary=destination)
        marker_path = destination / context_layout.AFS_STATE_DIR / context_layout.LAYOUT_FILE
        marker_document, marker_stat = _read_stable_regular(
            marker_path,
            boundary=destination,
            maximum_bytes=_MAX_MARKER_BYTES,
            label="layout marker",
        )
        readme_path = destination / "README.md"
        readme_document, readme_stat = _read_stable_regular(
            readme_path,
            boundary=destination,
            maximum_bytes=_MAX_README_BYTES,
            label="layout README",
        )
        marker_text = marker_document.decode("utf-8")
        readme = readme_document.decode("utf-8")
        metadata = context_layout.LayoutMetadata.load(destination)
        plan_hash = _canonical_plan_hash(plan)
        source_stat = _lstat(source, label="source root")
        source_fingerprint = _tree_fingerprint(
            source,
            retained_paths=_all_retained_paths(plan),
        )
        operations_sha256 = _verified_operations_digest(plan, source, destination)
        candidate_sha256 = _candidate_tree_digest(destination, transaction_id)
        audit = context_layout.audit_layout(destination)
        marker_sha256 = hashlib.sha256(marker_text.encode("utf-8")).hexdigest()
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        LayoutMigrationError,
    ):
        return None
    if (
        metadata is None
        or not isinstance(receipt, dict)
        or not _valid_receipt_schema(receipt, plan)
    ):
        return None
    expected_categories = tuple(category.value for category in ContextCategory)
    if (
        metadata.namespace != "central"
        or metadata.categories != expected_categories
        or receipt.get("status") != "applied"
        or receipt.get("transaction_id") != transaction_id
        or receipt.get("plan_hash") != plan_hash
        or receipt.get("plan_created_at") != plan.created_at
        or receipt.get("source_root") != str(source)
        or receipt.get("destination_root") != str(destination)
        or receipt.get("source_fingerprint") != plan.source_fingerprint
        or receipt.get("source_file_count") != plan.source_file_count
        or receipt.get("source_bytes") != plan.source_bytes
        or receipt.get("source_device") != plan.source_device
        or receipt.get("source_inode") != plan.source_inode
        or receipt.get("candidate_sha256") != candidate_sha256
        or receipt.get("operations_sha256") != operations_sha256
        or receipt.get("marker_sha256") != marker_sha256
        or source_fingerprint
        != (plan.source_fingerprint, plan.source_file_count, plan.source_bytes)
        or (source_stat.st_dev, source_stat.st_ino) != (plan.source_device, plan.source_inode)
        or not audit.valid
        or _has_group_or_world_permissions(receipt_stat)
        or _has_group_or_world_permissions(marker_stat)
        or _has_group_or_world_permissions(readme_stat)
        or readme != context_layout.LAYOUT_README
    ):
        return None
    return MigrationResult(
        status="already_applied",
        plan_hash=plan_hash,
        transaction_id=transaction_id,
        source_root=source,
        destination_root=destination,
        source_fingerprint=str(plan.source_fingerprint),
        source_file_count=int(plan.source_file_count),
        source_bytes=int(plan.source_bytes),
        copy_file_count=int(getattr(plan, "copy_file_count", plan.source_file_count)),
        copy_bytes=int(getattr(plan, "copy_bytes", plan.source_bytes)),
        operation_count=len(tuple(plan.operations)),
        receipt_path=receipt_path,
        plan_schema_version=plan_schema_version,
        retained_sources=retained_sources,
        retained_paths=retained_paths,
    )


def verify_completed_candidate(plan_or_path: Any) -> VerifiedMigrationCandidate:
    """Return durable evidence for a fresh, fully verified candidate.

    Unlike :func:`preflight_migration`, this function never treats an absent
    destination as ready work.  It is the activation boundary: both the
    original source and the separate destination must still match the plan
    and the migration receipt byte-for-byte.
    """

    plan = _load_plan(plan_or_path)
    result = _completed_result(plan)
    if result is None:
        raise MigrationPreflightError(
            "migration candidate is absent, stale, or no longer fully verified"
        )
    source_stat = _lstat(result.source_root, label="source root")
    candidate_stat = _lstat(result.destination_root, label="candidate root")
    try:
        receipt_document, _receipt_stat = _read_stable_regular(
            result.receipt_path,
            boundary=result.destination_root,
            maximum_bytes=_MAX_RECEIPT_BYTES,
            label="migration receipt",
        )
        candidate_sha256 = _candidate_tree_digest(
            result.destination_root,
            result.transaction_id,
        )
    except (OSError, ValueError, LayoutMigrationError) as exc:
        raise MigrationPreflightError(f"cannot verify completed candidate: {exc}") from exc
    return VerifiedMigrationCandidate(
        result=result,
        candidate_sha256=candidate_sha256,
        receipt_sha256=hashlib.sha256(receipt_document).hexdigest(),
        source_device=int(source_stat.st_dev),
        source_inode=int(source_stat.st_ino),
        candidate_device=int(candidate_stat.st_dev),
        candidate_inode=int(candidate_stat.st_ino),
    )


def source_tree_fingerprint(plan_or_path: Any, source_root: Path) -> tuple[str, int, int]:
    """Fingerprint a source-shaped tree using a plan's reviewed exclusions."""

    plan = _load_plan(plan_or_path)
    root = lexical_absolute(source_root)
    try:
        assert_no_linklike_components(root, boundary=Path(root.anchor), allow_missing=False)
        _validate_retention_spellings(plan, root)
        return _tree_fingerprint(root, retained_paths=_all_retained_paths(plan))
    except (OSError, ValueError, LayoutMigrationError) as exc:
        raise MigrationPreflightError(f"cannot fingerprint source tree: {exc}") from exc


def tree_fingerprint(root: Path) -> tuple[str, int, int]:
    """Fingerprint a complete regular-file tree without exclusions."""

    source = lexical_absolute(root)
    try:
        assert_no_linklike_components(source, boundary=Path(source.anchor), allow_missing=False)
        return _tree_fingerprint(source)
    except (OSError, ValueError, LayoutMigrationError) as exc:
        raise MigrationPreflightError(f"cannot fingerprint tree: {exc}") from exc


def candidate_tree_sha256(root: Path, transaction_id: str) -> str:
    """Hash a candidate tree using the migration receipt exclusion contract."""

    candidate = lexical_absolute(root)
    try:
        assert_no_linklike_components(
            candidate,
            boundary=Path(candidate.anchor),
            allow_missing=False,
        )
        return _candidate_tree_digest(candidate, transaction_id)
    except (OSError, ValueError, LayoutMigrationError) as exc:
        raise MigrationPreflightError(f"cannot fingerprint candidate tree: {exc}") from exc


def verify_relocated_candidate(
    plan_or_path: Any,
    candidate_root: Path,
    evidence: VerifiedMigrationCandidate,
) -> None:
    """Verify immutable candidate evidence after an atomic path relocation."""

    plan = _load_plan(plan_or_path)
    candidate = lexical_absolute(candidate_root)
    transaction_id = str(getattr(plan, "transaction_id", ""))
    try:
        assert_no_linklike_components(
            candidate,
            boundary=Path(candidate.anchor),
            allow_missing=False,
        )
        receipt_path = _receipt_path(candidate, transaction_id)
        receipt_document, receipt_stat = _read_stable_regular(
            receipt_path,
            boundary=candidate,
            maximum_bytes=_MAX_RECEIPT_BYTES,
            label="relocated migration receipt",
        )
        receipt = json.loads(receipt_document.decode("utf-8"), object_pairs_hook=_receipt_object)
        marker_path = candidate / context_layout.AFS_STATE_DIR / context_layout.LAYOUT_FILE
        marker_document, marker_stat = _read_stable_regular(
            marker_path,
            boundary=candidate,
            maximum_bytes=_MAX_MARKER_BYTES,
            label="relocated layout marker",
        )
        readme_document, readme_stat = _read_stable_regular(
            candidate / "README.md",
            boundary=candidate,
            maximum_bytes=_MAX_README_BYTES,
            label="relocated layout README",
        )
        candidate_sha256 = _candidate_tree_digest(candidate, transaction_id)
        metadata = context_layout.LayoutMetadata.load(candidate)
        audit = context_layout.audit_layout(candidate)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, LayoutMigrationError) as exc:
        raise MigrationPreflightError(f"cannot verify relocated candidate: {exc}") from exc
    if not isinstance(receipt, dict) or not _valid_receipt_schema(receipt, plan):
        raise MigrationPreflightError("relocated migration receipt schema is invalid")
    plan_hash = _canonical_plan_hash(plan)
    marker_sha256 = hashlib.sha256(marker_document).hexdigest()
    expected_categories = tuple(category.value for category in ContextCategory)
    if (
        hashlib.sha256(receipt_document).hexdigest() != evidence.receipt_sha256
        or candidate_sha256 != evidence.candidate_sha256
        or receipt.get("candidate_sha256") != evidence.candidate_sha256
        or receipt.get("marker_sha256") != marker_sha256
        or receipt.get("transaction_id") != transaction_id
        or receipt.get("plan_hash") != plan_hash
        or receipt.get("source_root") != str(evidence.result.source_root)
        or receipt.get("destination_root") != str(evidence.result.destination_root)
        or metadata is None
        or metadata.namespace != "central"
        or metadata.categories != expected_categories
        or not audit.valid
        or audit.layout_version != 2
        or readme_document.decode("utf-8") != context_layout.LAYOUT_README
        or _has_group_or_world_permissions(receipt_stat)
        or _has_group_or_world_permissions(marker_stat)
        or _has_group_or_world_permissions(readme_stat)
    ):
        raise MigrationPreflightError("relocated candidate evidence does not match migration")


@contextmanager
def _sibling_lock(destination: Path, *, timeout: float) -> Iterator[None]:
    lock_path = destination.parent / f".{destination.name}.migration.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise MigrationApplyError(f"cannot open migration lock {lock_path}: {exc}") from exc
    try:
        os.chmod(lock_path, 0o600)
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise MigrationApplyError(f"migration lock is not a private regular file: {lock_path}")
        deadline = time.monotonic() + timeout
        while True:
            try:
                _lock_descriptor(descriptor)
                break
            except OSError as exc:
                contention = isinstance(exc, BlockingIOError) or exc.errno in {
                    errno.EACCES,
                    errno.EAGAIN,
                    errno.EDEADLK,
                }
                if not contention:
                    raise MigrationApplyError(
                        f"cannot acquire migration lock {lock_path}: {exc}"
                    ) from exc
                if time.monotonic() >= deadline:
                    raise MigrationApplyError(
                        f"timed out waiting for migration lock: {lock_path}"
                    ) from exc
                time.sleep(0.05)
        yield
    finally:
        with contextlib.suppress(OSError):
            _unlock_descriptor(descriptor)
        os.close(descriptor)


def _lock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        locking = msvcrt.__dict__["locking"]
        locking(descriptor, msvcrt.__dict__["LK_NBLCK"], 1)
        return
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        locking = msvcrt.__dict__["locking"]
        locking(descriptor, msvcrt.__dict__["LK_UNLCK"], 1)
        return
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _mkdir_private(path: Path) -> None:
    os.mkdir(path, 0o700)
    os.chmod(path, 0o700)


def _ensure_private_directory(path: Path, *, root: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        try:
            current_stat = os.lstat(current)
        except FileNotFoundError:
            _mkdir_private(current)
            strict_fsync_directory(current.parent)
            continue
        if is_linklike(current_stat) or not stat.S_ISDIR(current_stat.st_mode):
            raise MigrationApplyError(f"unsafe scaffold path: {current}")


def _build_unmarked_scaffold(destination: Path) -> None:
    for category in ContextCategory:
        _ensure_private_directory(destination / category.value, root=destination)
    for relative in context_layout.V2_COMPAT_MOUNT_PATHS.values():
        if relative.startswith(f"{context_layout.AFS_STATE_DIR}/"):
            _ensure_private_directory(destination / relative, root=destination)
    for relative in context_layout.V2_SYSTEM_PATHS.values():
        _ensure_private_directory(destination / relative, root=destination)


def _copy_regular_file(source: Path, destination: Path) -> tuple[str, int]:
    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    source_fd = os.open(source, source_flags)
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise MigrationApplyError(f"source is no longer a unique regular file: {source}")
        target_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            target_flags |= os.O_NOFOLLOW
        mode = 0o600 | (stat.S_IMODE(before.st_mode) & 0o100)
        target_fd = os.open(destination, target_flags, mode)
        digest = hashlib.sha256()
        size = 0
        try:
            while True:
                chunk = os.read(source_fd, _COPY_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(target_fd, view)
                    view = view[written:]
            fchmod = os.__dict__.get("fchmod")
            if fchmod is not None:
                fchmod(target_fd, mode)
            else:
                # The path was exclusively created below a private root and
                # O_NOFOLLOW was used where available.
                os.chmod(destination, mode)
            os.fsync(target_fd)
        finally:
            os.close(target_fd)
        after = os.fstat(source_fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise MigrationApplyError(f"source changed while it was copied: {source}")
        return digest.hexdigest(), size
    finally:
        os.close(source_fd)


def _hash_regular_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        path_stat = os.fstat(descriptor)
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
            raise MigrationApplyError(f"copied target is not a unique regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, _COPY_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def _mapped_content_fingerprint(
    source: Path,
    destination: Path,
    *,
    retained_paths: tuple[Path, ...] = (),
) -> str:
    """Verify and hash only entries owned by one source operation.

    A destination directory may contain additional entries owned by another
    independently validated operation (for example ``memory`` plus the
    legacy top-level ``missions`` import). Exact source paths must still have
    identical kinds and file bytes.
    """

    digest = hashlib.sha256()

    def visit(source_path: Path, destination_path: Path, relative: str) -> None:
        relative_path = Path(relative)
        if _is_retained_path(relative_path, retained_paths):
            return
        source_stat = _lstat(source_path, label="operation source verification path")
        destination_stat = _lstat(
            destination_path,
            label="operation destination verification path",
        )
        encoded = os.fsencode(relative)
        if stat.S_ISDIR(source_stat.st_mode):
            if not stat.S_ISDIR(destination_stat.st_mode):
                raise MigrationApplyError(f"operation kind mismatch at {relative}")
            digest.update(b"D\0" + encoded + b"\0")
            try:
                children = sorted(
                    os.scandir(source_path),
                    key=lambda entry: os.fsencode(entry.name),
                )
            except OSError as exc:
                raise MigrationApplyError(
                    f"cannot verify operation directory {source_path}: {exc}"
                ) from exc
            for child in children:
                child_relative = child.name if relative == "." else f"{relative}/{child.name}"
                if _is_retained_path(Path(child_relative), retained_paths):
                    continue
                _validate_name(child.name)
                visit(
                    Path(child.path),
                    destination_path / child.name,
                    child_relative,
                )
            return
        if (
            not stat.S_ISREG(source_stat.st_mode)
            or source_stat.st_nlink != 1
            or not stat.S_ISREG(destination_stat.st_mode)
            or destination_stat.st_nlink != 1
        ):
            raise MigrationApplyError(f"operation file is not a unique regular file: {relative}")
        source_hash, source_size = _hash_regular_file(source_path)
        destination_hash, destination_size = _hash_regular_file(destination_path)
        if (source_hash, source_size) != (destination_hash, destination_size):
            raise MigrationApplyError(f"operation content mismatch at {relative}")
        digest.update(
            b"F\0"
            + encoded
            + b"\0"
            + str(source_size).encode("ascii")
            + b"\0"
            + source_hash.encode("ascii")
            + b"\0"
        )

    visit(source, destination, ".")
    return digest.hexdigest()


def _verified_operations_digest(
    plan: Any,
    source_root: Path,
    destination_root: Path,
) -> str:
    digest = hashlib.sha256()
    for operation in _operations(plan, source_root, destination_root):
        source = source_root / operation.source_relative
        destination = destination_root / operation.destination_relative
        destination_digest = _mapped_content_fingerprint(
            source,
            destination,
            retained_paths=_operation_retained_paths(plan, operation),
        )
        for value in (
            operation.source_relative.as_posix(),
            operation.destination_relative.as_posix(),
            destination_digest,
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _candidate_tree_digest(destination: Path, transaction_id: str) -> str:
    """Hash the complete candidate, excluding the hash-bearing receipt/marker."""

    digest = hashlib.sha256()
    receipt_relative = _receipt_path(Path("."), transaction_id)
    marker_relative = Path(context_layout.AFS_STATE_DIR) / context_layout.LAYOUT_FILE
    excluded = {receipt_relative.as_posix(), marker_relative.as_posix()}

    def visit(path: Path, relative: str) -> None:
        if relative in excluded:
            return
        path_stat = _lstat(path, label="candidate integrity path")
        encoded = os.fsencode(relative)
        mode = str(stat.S_IMODE(path_stat.st_mode)).encode("ascii")
        if stat.S_ISDIR(path_stat.st_mode):
            digest.update(b"D\0" + encoded + b"\0" + mode + b"\0")
            try:
                children = sorted(os.scandir(path), key=lambda entry: os.fsencode(entry.name))
            except OSError as exc:
                raise MigrationApplyError(f"cannot hash candidate directory {path}: {exc}") from exc
            for child in children:
                _validate_name(child.name)
                child_relative = child.name if relative == "." else f"{relative}/{child.name}"
                visit(Path(child.path), child_relative)
            return
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
            raise MigrationApplyError(
                f"candidate integrity path is not a unique regular file: {path}"
            )
        content_hash, size = _hash_regular_file(path)
        digest.update(
            b"F\0"
            + encoded
            + b"\0"
            + mode
            + b"\0"
            + str(size).encode("ascii")
            + b"\0"
            + content_hash.encode("ascii")
            + b"\0"
        )

    visit(destination, ".")
    return digest.hexdigest()


def _copy_tree(
    source: Path,
    destination: Path,
    *,
    boundary: Path,
    relative: Path = Path("."),
    retained_paths: tuple[Path, ...] = (),
) -> tuple[int, int]:
    if _is_retained_path(relative, retained_paths):
        return 0, 0
    source_stat = _lstat(source, label="copy source")
    if stat.S_ISREG(source_stat.st_mode):
        _ensure_private_directory(destination.parent, root=boundary)
        source_digest, source_size = _copy_regular_file(source, destination)
        target_digest, target_size = _hash_regular_file(destination)
        if (source_digest, source_size) != (target_digest, target_size):
            raise MigrationApplyError(f"copy verification failed: {source}")
        # fsync(file) persists content; fsync(parent) persists the exclusive
        # directory entry before the layout marker can authorize the tree.
        strict_fsync_directory(destination.parent)
        return 1, source_size
    if not stat.S_ISDIR(source_stat.st_mode):
        raise MigrationApplyError(f"copy source is not a regular file or directory: {source}")
    _ensure_private_directory(destination, root=boundary)
    count = 0
    total = 0
    try:
        children = sorted(os.scandir(source), key=lambda entry: entry.name)
    except OSError as exc:
        raise MigrationApplyError(f"cannot enumerate copy source {source}: {exc}") from exc
    for child in children:
        child_relative = Path(child.name) if relative == Path(".") else relative / child.name
        if _is_retained_path(child_relative, retained_paths):
            continue
        _validate_name(child.name)
        child_count, child_bytes = _copy_tree(
            Path(child.path),
            destination / child.name,
            boundary=boundary,
            relative=child_relative,
            retained_paths=retained_paths,
        )
        count += child_count
        total += child_bytes
    return count, total


def _copy_operations(preflight: MigrationPreflight) -> tuple[int, int]:
    count = 0
    total = 0
    for operation in _operations(
        preflight.plan,
        preflight.source_root,
        preflight.destination_root,
    ):
        source = preflight.source_root / operation.source_relative
        destination = preflight.destination_root / operation.destination_relative
        retained_paths = _operation_retained_paths(preflight.plan, operation)
        copied_count, copied_bytes = _copy_tree(
            source,
            destination,
            boundary=preflight.destination_root,
            retained_paths=retained_paths,
        )
        _mapped_content_fingerprint(
            source,
            destination,
            retained_paths=retained_paths,
        )
        count += copied_count
        total += copied_bytes
    return count, total


def _receipt_payload(
    preflight: MigrationPreflight,
    *,
    rationale: str,
    authorization: HumanAuthorization,
    candidate_sha256: str,
    operations_sha256: str,
    marker_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "applied",
        "transaction_id": str(preflight.plan.transaction_id),
        "plan_hash": preflight.plan_hash,
        "plan_created_at": str(preflight.plan.created_at),
        "source_root": str(preflight.source_root),
        "destination_root": str(preflight.destination_root),
        "source_fingerprint": str(preflight.plan.source_fingerprint),
        "source_file_count": preflight.source_file_count,
        "source_bytes": preflight.source_bytes,
        "source_device": int(preflight.plan.source_device),
        "source_inode": int(preflight.plan.source_inode),
        "source_unchanged": True,
        "candidate_sha256": candidate_sha256,
        "operations_sha256": operations_sha256,
        "marker_sha256": marker_sha256,
        "rationale": rationale.strip(),
        "authorization_scope": layout_migration_authorization_scope(
            preflight.plan_hash,
            preflight.plan.transaction_id,
            rationale,
        ),
        "authorized_by": authorization.identity.reviewer,
        "reviewer_subject": authorization.identity.subject,
        "authorized_via": authorization.confirmed_via,
        "applied_at": context_layout._utc_now(),
    }
    if _plan_schema_version(preflight.plan) == 3:
        payload.update(
            {
                "schema_version": 2,
                "plan_schema_version": 3,
                "copy_file_count": preflight.copy_file_count,
                "copy_bytes": preflight.copy_bytes,
                "retained_sources": [item.to_dict() for item in preflight.retained_sources],
                "retained_paths": [item.to_dict() for item in preflight.retained_paths],
            }
        )
    payload["receipt_sha256"] = _canonical_receipt_sha256(payload)
    return payload


def _canonical_receipt_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("receipt_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _valid_receipt_schema(receipt: dict[str, Any], plan: Any) -> bool:
    plan_schema_version = _plan_schema_version(plan)
    expected_fields = {
        "schema_version",
        "status",
        "transaction_id",
        "plan_hash",
        "plan_created_at",
        "source_root",
        "destination_root",
        "source_fingerprint",
        "source_file_count",
        "source_bytes",
        "source_device",
        "source_inode",
        "source_unchanged",
        "candidate_sha256",
        "operations_sha256",
        "marker_sha256",
        "rationale",
        "authorization_scope",
        "authorized_by",
        "reviewer_subject",
        "authorized_via",
        "applied_at",
        "receipt_sha256",
    }
    if plan_schema_version == 3:
        expected_fields.update(
            {
                "plan_schema_version",
                "copy_file_count",
                "copy_bytes",
                "retained_sources",
                "retained_paths",
            }
        )
    if set(receipt) != expected_fields:
        return False
    string_fields = expected_fields - {
        "schema_version",
        "plan_schema_version",
        "source_file_count",
        "source_bytes",
        "copy_file_count",
        "copy_bytes",
        "source_device",
        "source_inode",
        "source_unchanged",
        "retained_sources",
        "retained_paths",
    }
    if any(type(receipt[field]) is not str or not receipt[field] for field in string_fields):
        return False
    integer_fields = {
        "schema_version",
        "plan_schema_version",
        "source_file_count",
        "source_bytes",
        "copy_file_count",
        "copy_bytes",
        "source_device",
        "source_inode",
    }
    integer_fields.intersection_update(expected_fields)
    if any(type(receipt[field]) is not int or receipt[field] < 0 for field in integer_fields):
        return False
    expected_receipt_schema = 2 if plan_schema_version == 3 else 1
    if (
        receipt["schema_version"] != expected_receipt_schema
        or receipt["source_unchanged"] is not True
    ):
        return False
    if plan_schema_version == 3:
        expected_sources = [item.to_dict() for item in _retentions(plan, "retained_sources")]
        expected_paths = [item.to_dict() for item in _retentions(plan, "retained_paths")]
        if (
            receipt.get("plan_schema_version") != 3
            or receipt.get("copy_file_count") != getattr(plan, "copy_file_count", None)
            or receipt.get("copy_bytes") != getattr(plan, "copy_bytes", None)
            or receipt.get("retained_sources") != expected_sources
            or receipt.get("retained_paths") != expected_paths
        ):
            return False
    if receipt["authorized_via"] != "controlling_terminal":
        return False
    if receipt["receipt_sha256"] != _canonical_receipt_sha256(receipt):
        return False
    try:
        expected_scope = layout_migration_authorization_scope(
            plan.plan_sha256,
            plan.transaction_id,
            receipt["rationale"],
        )
    except ValueError:
        return False
    return bool(receipt["authorization_scope"] == expected_scope)


def _quarantine_destination(
    destination: Path,
    transaction_id: str,
) -> tuple[Path | None, bool]:
    if not _destination_exists(destination):
        return None, False
    base = destination.with_name(f"{destination.name}.failed-{transaction_id}")
    candidate = base
    while _destination_exists(candidate):
        candidate = destination.with_name(f"{base.name}-{uuid4().hex[:8]}")
    try:
        os.replace(destination, candidate)
    except OSError:
        return None, False
    try:
        strict_fsync_directory(destination.parent)
    except OSError:
        return candidate, False
    return candidate, True


def apply_migration(
    plan_or_path: Any,
    *,
    rationale: str,
    authorization: HumanAuthorization,
    lock_timeout: float = _LOCK_TIMEOUT_SECONDS,
) -> MigrationResult:
    """Copy one authorized schema-v2/v3 plan to a separate destination.

    The source is never written, renamed, or deleted.  The destination's
    layout marker is the final durable write, so every tree visible as v2 is
    complete. Partial pre-marker trees are quarantined for diagnosis.
    """

    _require_private_destination_permissions()
    plan = _load_plan(plan_or_path)
    completed = _completed_result(plan)
    if completed is not None:
        return completed
    destination = lexical_absolute(Path(str(getattr(plan, "destination_root", ""))))
    try:
        assert_no_linklike_components(
            destination.parent,
            boundary=Path(destination.anchor),
            allow_missing=False,
        )
    except (OSError, ValueError) as exc:
        raise MigrationPreflightError(f"unsafe migration destination parent: {exc}") from exc
    if not destination.parent.is_dir():
        raise MigrationPreflightError(
            f"destination parent is not a directory: {destination.parent}"
        )
    with _sibling_lock(destination, timeout=lock_timeout):
        completed = _completed_result(plan)
        if completed is not None:
            return completed
        current = preflight_migration(plan)
        scope = layout_migration_authorization_scope(
            current.plan_hash,
            str(plan.transaction_id),
            rationale,
        )
        if not consume_human_authorization(authorization, scope=scope):
            raise MigrationApplyError("a fresh HumanDecisionBroker authorization is required")

        created = False
        marker = destination / context_layout.AFS_STATE_DIR / context_layout.LAYOUT_FILE
        failed: Path | None = None
        quarantine_durable = False
        try:
            _mkdir_private(destination)
            created = True
            strict_fsync_directory(destination.parent)
            _build_unmarked_scaffold(destination)
            copied_count, copied_bytes = _copy_operations(current)
            if (copied_count, copied_bytes) != (
                current.copy_file_count,
                current.copy_bytes,
            ):
                raise MigrationApplyError("copied file totals do not match preflight evidence")
            final_fingerprint = _tree_fingerprint(
                current.source_root,
                retained_paths=_all_retained_paths(plan),
            )
            expected = (
                str(plan.source_fingerprint),
                int(plan.source_file_count),
                int(plan.source_bytes),
            )
            if final_fingerprint != expected:
                raise MigrationApplyError("source changed before migration publication")

            receipt_path = _receipt_path(destination, str(plan.transaction_id))
            _ensure_private_directory(receipt_path.parent, root=destination)
            atomic_write_text(
                destination / "README.md",
                context_layout.LAYOUT_README,
                mode=0o600,
                durable=True,
            )
            operations_sha256 = _verified_operations_digest(
                plan,
                current.source_root,
                destination,
            )
            candidate_sha256 = _candidate_tree_digest(destination, plan.transaction_id)
            metadata = context_layout.LayoutMetadata()
            marker_text = metadata.render()
            marker_sha256 = hashlib.sha256(marker_text.encode("utf-8")).hexdigest()
            receipt = _receipt_payload(
                current,
                rationale=rationale,
                authorization=authorization,
                candidate_sha256=candidate_sha256,
                operations_sha256=operations_sha256,
                marker_sha256=marker_sha256,
            )
            atomic_write_text(
                receipt_path,
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                mode=0o600,
                durable=True,
            )
            # Re-read every self-referential/private artifact before the
            # commit point. No fallible verification follows marker publish.
            persisted_receipt, receipt_stat = _read_receipt(
                receipt_path,
                boundary=destination,
            )
            if persisted_receipt != receipt or _has_group_or_world_permissions(receipt_stat):
                raise MigrationApplyError("durable migration receipt verification failed")
            readme_document, readme_stat = _read_stable_regular(
                destination / "README.md",
                boundary=destination,
                maximum_bytes=_MAX_README_BYTES,
                label="layout README",
            )
            if readme_document.decode(
                "utf-8"
            ) != context_layout.LAYOUT_README or _has_group_or_world_permissions(readme_stat):
                raise MigrationApplyError("durable layout README verification failed")
            if _candidate_tree_digest(destination, plan.transaction_id) != candidate_sha256:
                raise MigrationApplyError("candidate changed before marker publication")
            if (
                _verified_operations_digest(plan, current.source_root, destination)
                != operations_sha256
            ):
                raise MigrationApplyError("operation evidence changed before marker publication")
            if (
                _tree_fingerprint(
                    current.source_root,
                    retained_paths=_all_retained_paths(plan),
                )
                != expected
            ):
                raise MigrationApplyError("source changed before final marker publication")
            final_result = MigrationResult(
                status="applied",
                plan_hash=current.plan_hash,
                transaction_id=str(plan.transaction_id),
                source_root=current.source_root,
                destination_root=destination,
                source_fingerprint=str(plan.source_fingerprint),
                source_file_count=current.source_file_count,
                source_bytes=current.source_bytes,
                copy_file_count=current.copy_file_count,
                copy_bytes=current.copy_bytes,
                operation_count=current.mapping_count,
                receipt_path=receipt_path,
                plan_schema_version=_plan_schema_version(plan),
                retained_sources=current.retained_sources,
                retained_paths=current.retained_paths,
            )
            atomic_write_text(
                marker,
                marker_text,
                mode=0o600,
                durable=True,
            )
            return final_result
        except BaseException as exc:  # noqa: BLE001 - cleanup must cover interrupts and bugs
            if _destination_exists(marker):
                try:
                    marker.unlink()
                    strict_fsync_directory(marker.parent)
                except OSError as revoke_error:
                    raise MigrationApplyError(
                        "migration failed after layout marker publication and marker "
                        f"revocation failed; marked candidate remains at {destination}: "
                        f"{revoke_error}",
                        failed_destination=destination,
                    ) from exc
            if created:
                failed, quarantine_durable = _quarantine_destination(
                    destination,
                    str(plan.transaction_id),
                )
            if failed is not None:
                location = f"; partial tree quarantined at {failed}"
                if not quarantine_durable:
                    location += (
                        "; quarantine rename completed but parent-directory durability sync failed"
                    )
                reported_destination = failed
            elif created and _destination_exists(destination):
                location = f"; quarantine rename failed and partial tree remains at {destination}"
                reported_destination = destination
            else:
                location = ""
                reported_destination = None
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise MigrationApplyError(
                f"migration apply failed: {exc}{location}",
                failed_destination=reported_destination,
            ) from exc
