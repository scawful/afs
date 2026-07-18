"""Versioned central context layout primitives.

This module intentionally separates inspection/planning from execution.  It
can scaffold a *new* v2 namespace and write plan manifests, but it never moves
or deletes an existing context tree.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import tempfile
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from .models import ContextCategory, MountType
from .path_safety import assert_no_linklike_components, is_linklike, lexical_absolute
from .toml_compat import tomllib

LAYOUT_VERSION = 2
AFS_STATE_DIR = ".afs"
LAYOUT_FILE = "layout.toml"
LAYOUT_README = """# AFS Context

This is a version 2 central AFS context namespace.

- `history/` records immutable chronology and provenance.
- `memory/` contains durable learned context and handoffs.
- `scratchpad/` contains temporary task and project work.
- `knowledge/` contains reference material and mounted project sources.
- `tools/` contains trusted skills and executable resources.
- `human/` contains human-authored intent, decisions, and approvals.
- `.afs/` contains indexes, registries, queues, health, and runtime state.

Use `afs layout audit --context-root PATH` before making layout changes.
"""

V2_SYSTEM_PATHS: dict[str, str] = {
    "messages": ".afs/queue/messages",
    "projects": ".afs/projects",
    "search": ".afs/search",
}

# Runtime state is created lazily by the subsystem that owns it.  Keep these
# paths separate from ``V2_SYSTEM_PATHS`` so adding a new optional subsystem
# does not make an already-valid v2 scaffold fail layout audit.
V2_RUNTIME_PATHS: dict[str, str] = {
    "graph": ".afs/graph",
    "health": ".afs/health",
    "logs": ".afs/logs",
    "training": ".afs/training",
}


class LayoutStateError(ValueError):
    """Raised when a root looks like v2 but its authorization marker is invalid."""


def _v2_state_sentinels(context_root: Path) -> tuple[Path, ...]:
    """Return paths that cannot be treated as an unmarked v1 namespace.

    A v2 root is authorized by ``.afs/layout.toml``.  The project registry is a
    secondary sentinel so deleting the marker cannot silently downgrade an
    existing central namespace to v1. Category ``projects`` directories are
    not definitive because legacy scoped stores may also create them.
    """

    root = context_root.expanduser().resolve()
    sentinels = [
        root / AFS_STATE_DIR / LAYOUT_FILE,
        root / V2_SYSTEM_PATHS["projects"],
    ]
    readme = root / "README.md"
    try:
        safe_readme = assert_no_linklike_components(readme, boundary=root)
        if safe_readme.is_file() and safe_readme.read_text(encoding="utf-8") == LAYOUT_README:
            sentinels.append(readme)
    except (OSError, UnicodeError, ValueError):
        pass
    if _has_v2_structural_signature(root):
        # ``human`` is v2-only in the standard layout. Requiring every
        # category alongside it makes this a conservative damaged-v2 signal
        # rather than treating one user-created directory as authoritative.
        sentinels.append(root / ContextCategory.HUMAN.value)
    return tuple(sentinels)


def _has_v2_structural_signature(context_root: Path) -> bool:
    """Recognize a scaffolded v2 tree even when mutable markers are lost."""

    root = context_root.expanduser().resolve()
    for category in ContextCategory:
        candidate = root / category.value
        try:
            candidate_stat = os.lstat(candidate)
        except OSError:
            return False
        if not (is_linklike(candidate_stat) or candidate.is_dir()):
            return False
    return True

V2_COMPAT_MOUNT_PATHS: dict[MountType, str] = {
    MountType.MEMORY: "memory",
    MountType.KNOWLEDGE: "knowledge",
    MountType.TOOLS: "tools",
    MountType.SCRATCHPAD: "scratchpad",
    MountType.HISTORY: "history",
    MountType.HIVEMIND: V2_SYSTEM_PATHS["messages"],
    MountType.GLOBAL: ".afs/compat/global",
    MountType.ITEMS: ".afs/compat/items",
}

_V1_MIGRATION_TARGETS: dict[str, str] = {
    # Legacy category roots are imported into the shared compatibility scope.
    # Writing them directly to the v2 category root would make them look
    # project-neutral while bypassing the explicit ``common`` boundary.
    "history": "history/common",
    "memory": "memory/common",
    "scratchpad": "scratchpad/common",
    "knowledge": "knowledge/common",
    "tools": "tools/common",
    "hivemind": V2_SYSTEM_PATHS["messages"],
    "global": ".afs/compat/global",
    "items": ".afs/compat/items",
    "metadata.json": ".afs/compat/metadata.json",
    "health": ".afs/health",
    "index": ".afs/search/legacy-index",
    "metrics": ".afs/metrics",
    "monitoring": ".afs/monitoring",
    "missions": "memory/common/missions",
    "reports": "memory/common/reports",
    "review": "human/common/review",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class LayoutMetadata:
    """Metadata controlling a central context namespace."""

    layout_version: int = LAYOUT_VERSION
    namespace: str = "central"
    created_at: str = field(default_factory=_utc_now)
    categories: tuple[str, ...] = field(
        default_factory=lambda: tuple(category.value for category in ContextCategory)
    )

    @classmethod
    def load(cls, context_root: Path) -> LayoutMetadata | None:
        root = context_root.expanduser().resolve()
        path = root / AFS_STATE_DIR / LAYOUT_FILE
        try:
            path = assert_no_linklike_components(path, boundary=root)
        except ValueError as exc:
            raise LayoutStateError(str(exc)) from exc
        if not path.is_file():
            return None
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return None
        version = payload.get("layout_version")
        namespace = payload.get("namespace")
        created_at = payload.get("created_at")
        categories = payload.get("categories")
        if version != LAYOUT_VERSION or not isinstance(namespace, str):
            return None
        if not isinstance(created_at, str) or not created_at:
            return None
        if not isinstance(categories, list) or not all(isinstance(item, str) for item in categories):
            return None
        return cls(
            layout_version=version,
            namespace=namespace,
            created_at=created_at,
            categories=tuple(categories),
        )

    def render(self) -> str:
        categories = ", ".join(json.dumps(item) for item in self.categories)
        lines = [
            f"layout_version = {self.layout_version}",
            f"namespace = {json.dumps(self.namespace)}",
            f"created_at = {json.dumps(self.created_at)}",
            f"categories = [{categories}]",
            "",
            "[compatibility.mounts]",
        ]
        for mount_type, relative_path in V2_COMPAT_MOUNT_PATHS.items():
            lines.append(f"{mount_type.value} = {json.dumps(relative_path)}")
        return "\n".join(lines) + "\n"

    def write(self, context_root: Path) -> Path:
        path = context_root.expanduser().resolve() / AFS_STATE_DIR / LAYOUT_FILE
        _atomic_write_text(path, self.render())
        return path


def detect_layout_version(context_root: Path) -> int:
    """Return the authorized layout version, failing closed for damaged v2.

    Genuine unmarked v1 roots remain supported.  A root that contains the v2
    marker, runtime sentinels, or the complete v2 category scaffold must not
    be reinterpreted as v1 when the marker is missing or malformed because
    doing so would bypass project scope authorization.
    """

    root = context_root.expanduser().resolve()
    try:
        assert_no_linklike_components(root / AFS_STATE_DIR, boundary=root)
        assert_no_linklike_components(
            root / AFS_STATE_DIR / LAYOUT_FILE,
            boundary=root,
        )
    except ValueError as exc:
        raise LayoutStateError(str(exc)) from exc
    if LayoutMetadata.load(root) is not None:
        return LAYOUT_VERSION
    sentinels = [path for path in _v2_state_sentinels(root) if path.exists()]
    if sentinels:
        marker = root / AFS_STATE_DIR / LAYOUT_FILE
        found = ", ".join(path.relative_to(root).as_posix() for path in sentinels)
        raise LayoutStateError(
            f"v2 context marker is missing or invalid: {marker}; "
            f"found v2 state ({found}). Run `afs layout audit --context-root {root}` "
            "and repair the marker before accessing context data"
        )
    return 1


def v2_directory_map() -> dict[str, str]:
    """Return the legacy role mapping used by compatibility readers."""

    return {mount_type.value: path for mount_type, path in V2_COMPAT_MOUNT_PATHS.items()}


def resolve_system_path(context_root: Path, name: str) -> Path:
    """Resolve a named v2 system path without granting category access."""

    try:
        relative_path = V2_SYSTEM_PATHS[name]
    except KeyError as exc:
        known = ", ".join(sorted(V2_SYSTEM_PATHS))
        raise ValueError(f"unknown v2 system path {name!r}; expected one of: {known}") from exc
    root = context_root.expanduser().resolve()
    return assert_no_linklike_components(root / relative_path, boundary=root)


def resolve_runtime_root(
    context_root: Path,
    name: str,
    *,
    legacy_relative: str,
    create: bool = False,
) -> Path:
    """Resolve one layout-aware runtime-state directory.

    Version 1 keeps its historical top-level directory.  Version 2 routes
    optional runtime state below ``.afs`` and rejects symbolic-link/reparse
    components before and after lazy directory creation.
    """

    root = context_root.expanduser().resolve()
    if detect_layout_version(root) != LAYOUT_VERSION:
        directory = root / legacy_relative
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    try:
        relative_path = V2_RUNTIME_PATHS[name]
    except KeyError as exc:
        known = ", ".join(sorted(V2_RUNTIME_PATHS))
        raise ValueError(
            f"unknown v2 runtime path {name!r}; expected one of: {known}"
        ) from exc
    directory = assert_no_linklike_components(
        root / relative_path,
        boundary=root,
    )
    if create:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory = assert_no_linklike_components(
            directory,
            boundary=root,
            allow_missing=False,
        )
    return directory


def scaffold_v2(context_root: Path) -> LayoutMetadata:
    """Create a fresh central v2 namespace without touching existing data."""

    root = context_root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        existing = LayoutMetadata.load(root)
        if existing is not None:
            return existing
        # Produce the same actionable fail-closed error used by readers when
        # this is a damaged v2 namespace, rather than calling it generic data.
        detect_layout_version(root)
        raise FileExistsError(f"refusing to scaffold v2 over non-empty context root: {root}")

    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    for category in ContextCategory:
        directory = root / category.value
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
    for relative_path in V2_COMPAT_MOUNT_PATHS.values():
        if relative_path.startswith(f"{AFS_STATE_DIR}/"):
            directory = root / relative_path
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(directory, 0o700)
    for relative_path in V2_SYSTEM_PATHS.values():
        directory = root / relative_path
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
    projects = resolve_system_path(root, "projects")
    projects.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(projects, 0o700)
    metadata = LayoutMetadata()
    metadata.write(root)
    _atomic_write_text(root / "README.md", LAYOUT_README)
    return metadata


@dataclass(frozen=True)
class LayoutIssue:
    code: str
    message: str
    path: str | None = None
    blocking: bool = False


@dataclass(frozen=True)
class LayoutAudit:
    context_root: str
    layout_version: int
    valid: bool
    migration_ready: bool
    categories: tuple[str, ...]
    unknown_entries: tuple[str, ...]
    issues: tuple[LayoutIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_layout(context_root: Path) -> LayoutAudit:
    """Inspect a context root without creating or changing any files."""

    root = context_root.expanduser().resolve()
    issues: list[LayoutIssue] = []
    if not root.is_dir():
        issues.append(LayoutIssue("missing_root", "context root does not exist", str(root), True))
        return LayoutAudit(str(root), 1, False, False, (), (), tuple(issues))

    try:
        version = detect_layout_version(root)
    except LayoutStateError as exc:
        version = LAYOUT_VERSION
        issues.append(
            LayoutIssue(
                "invalid_layout_marker",
                str(exc),
                f"{AFS_STATE_DIR}/{LAYOUT_FILE}",
                True,
            )
        )
    safe_directories: set[str] = set()
    required_directories = list(dict.fromkeys([
        *(category.value for category in ContextCategory),
        AFS_STATE_DIR,
        *V2_SYSTEM_PATHS.values(),
        *(
            path
            for path in V2_COMPAT_MOUNT_PATHS.values()
            if path.startswith(f"{AFS_STATE_DIR}/")
        ),
    ]))
    if version == LAYOUT_VERSION:
        for relative in required_directories:
            path = root / relative
            try:
                safe = assert_no_linklike_components(path, boundary=root)
            except ValueError as exc:
                issues.append(
                    LayoutIssue(
                        "linklike_required_directory",
                        str(exc),
                        relative,
                        True,
                    )
                )
                continue
            if safe.is_dir():
                safe_directories.add(relative)
            else:
                issues.append(
                    LayoutIssue(
                        "missing_required_directory",
                        f"required directory is missing: {relative}",
                        relative,
                        True,
                    )
                )
    categories = tuple(
        category.value
        for category in ContextCategory
        if (
            category.value in safe_directories
            if version == LAYOUT_VERSION
            else (root / category.value).is_dir()
        )
    )
    known = set(_V1_MIGRATION_TARGETS)
    if version == LAYOUT_VERSION:
        known = {category.value for category in ContextCategory} | {AFS_STATE_DIR, "README.md"}
    else:
        issues.append(
            LayoutIssue(
                "legacy_layout",
                "v1 context root requires a reviewed migration plan",
                blocking=False,
            )
        )
        for entry in root.iterdir():
            if entry.name not in known:
                continue
            try:
                entry_stat = os.lstat(entry)
            except OSError:
                continue
            if is_linklike(entry_stat):
                issues.append(
                    LayoutIssue(
                        "linklike_migration_source",
                        "top-level migration sources must not be symbolic links "
                        "or reparse points",
                        entry.name,
                        True,
                    )
                )

    unknown = tuple(sorted(entry.name for entry in root.iterdir() if entry.name not in known))
    for name in unknown:
        issues.append(
            LayoutIssue(
                "unknown_entry",
                f"entry has no deterministic v2 destination: {name}",
                name,
                True,
            )
        )
    valid = version == LAYOUT_VERSION and not any(issue.blocking for issue in issues)
    migration_ready = version == 1 and not any(issue.blocking for issue in issues)
    return LayoutAudit(
        str(root),
        version,
        valid,
        migration_ready,
        categories,
        unknown,
        tuple(issues),
    )


@dataclass(frozen=True)
class MigrationOperation:
    operation: str
    source: str
    destination: str
    verify: str = "sha256"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: object) -> MigrationOperation:
        payload = _strict_object(
            data,
            expected={"operation", "source", "destination", "verify"},
            label="migration operation",
        )
        operation = _strict_string(payload, "operation", label="migration operation")
        source = _strict_string(payload, "source", label="migration operation")
        destination = _strict_string(payload, "destination", label="migration operation")
        verify = _strict_string(payload, "verify", label="migration operation")
        if operation != "copy_verify":
            raise ValueError(f"unsupported migration operation: {operation!r}")
        if verify != "sha256":
            raise ValueError(f"unsupported migration verification method: {verify!r}")
        _validate_absolute_manifest_path(source, field_name="operation source")
        _validate_absolute_manifest_path(destination, field_name="operation destination")
        return cls(operation, source, destination, verify)


@dataclass(frozen=True, order=True)
class MigrationMapping:
    """A reviewed mapping for one otherwise-unknown v1 top-level entry."""

    source: str
    destination: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: object) -> MigrationMapping:
        payload = _strict_object(
            data,
            expected={"source", "destination"},
            label="migration mapping",
        )
        source = _strict_string(payload, "source", label="migration mapping")
        destination = _strict_string(payload, "destination", label="migration mapping")
        _validate_top_level_entry(source, field_name="mapping source")
        _validate_explicit_destination(destination)
        return cls(source, destination)


@dataclass(frozen=True)
class SourceInventory:
    """Stable, no-follow inventory metadata for one source tree."""

    fingerprint: str
    file_count: int
    total_bytes: int
    device: int
    inode: int


@dataclass(frozen=True)
class MigrationPlan:
    schema_version: int
    transaction_id: str
    created_at: str
    source_root: str
    destination_root: str
    source_fingerprint: str
    source_file_count: int
    source_bytes: int
    source_device: int
    source_inode: int
    ready: bool
    blocking_entries: tuple[str, ...]
    explicit_mappings: tuple[MigrationMapping, ...]
    operations: tuple[MigrationOperation, ...]
    plan_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "transaction_id": self.transaction_id,
            "created_at": self.created_at,
            "source_root": self.source_root,
            "destination_root": self.destination_root,
            "source_fingerprint": self.source_fingerprint,
            "source_file_count": self.source_file_count,
            "source_bytes": self.source_bytes,
            "source_device": self.source_device,
            "source_inode": self.source_inode,
            "ready": self.ready,
            "blocking_entries": list(self.blocking_entries),
            "explicit_mappings": [mapping.to_dict() for mapping in self.explicit_mappings],
            "operations": [operation.to_dict() for operation in self.operations],
            "plan_sha256": self.plan_sha256,
        }

    def canonical_sha256(self) -> str:
        """Hash the canonical plan payload, excluding the hash field itself."""

        payload = self.to_dict()
        payload.pop("plan_sha256")
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def with_canonical_hash(self) -> MigrationPlan:
        return replace(self, plan_sha256=self.canonical_sha256())

    @classmethod
    def from_dict(cls, data: object) -> MigrationPlan:
        expected = {
            "schema_version",
            "transaction_id",
            "created_at",
            "source_root",
            "destination_root",
            "source_fingerprint",
            "source_file_count",
            "source_bytes",
            "source_device",
            "source_inode",
            "ready",
            "blocking_entries",
            "explicit_mappings",
            "operations",
            "plan_sha256",
        }
        payload = _strict_object(data, expected=expected, label="migration plan")
        schema_version = _strict_integer(payload, "schema_version", label="migration plan")
        if schema_version != 2:
            raise ValueError(
                f"unsupported migration plan schema_version {schema_version!r}; "
                "regenerate the plan with this AFS version"
            )
        transaction_id = _strict_string(payload, "transaction_id", label="migration plan")
        if not re.fullmatch(r"layout_[a-f0-9]{32}", transaction_id):
            raise ValueError("migration plan transaction_id is invalid")
        created_at = _strict_string(payload, "created_at", label="migration plan")
        source_root = _strict_string(payload, "source_root", label="migration plan")
        destination_root = _strict_string(payload, "destination_root", label="migration plan")
        _validate_absolute_manifest_path(source_root, field_name="source_root")
        _validate_absolute_manifest_path(destination_root, field_name="destination_root")
        source_fingerprint = _strict_string(
            payload, "source_fingerprint", label="migration plan"
        )
        _validate_sha256(source_fingerprint, field_name="source_fingerprint")
        source_file_count = _strict_nonnegative_integer(
            payload, "source_file_count", label="migration plan"
        )
        source_bytes = _strict_nonnegative_integer(
            payload, "source_bytes", label="migration plan"
        )
        source_device = _strict_nonnegative_integer(
            payload, "source_device", label="migration plan"
        )
        source_inode = _strict_nonnegative_integer(
            payload, "source_inode", label="migration plan"
        )
        ready = payload["ready"]
        if type(ready) is not bool:
            raise ValueError("migration plan field 'ready' must be a boolean")
        blocking_entries = _strict_string_tuple(
            payload, "blocking_entries", label="migration plan"
        )
        if blocking_entries != tuple(sorted(set(blocking_entries))):
            raise ValueError("migration plan blocking_entries must be sorted and unique")
        for entry in blocking_entries:
            _validate_top_level_entry(entry, field_name="blocking entry")

        mappings_raw = payload["explicit_mappings"]
        if not isinstance(mappings_raw, list):
            raise ValueError("migration plan field 'explicit_mappings' must be a list")
        explicit_mappings = tuple(MigrationMapping.from_dict(item) for item in mappings_raw)
        if explicit_mappings != tuple(sorted(explicit_mappings)):
            raise ValueError("migration plan explicit_mappings must be sorted")
        if len({mapping.source for mapping in explicit_mappings}) != len(explicit_mappings):
            raise ValueError("migration plan mapping sources must be unique")
        _validate_mapping_destination_collisions(explicit_mappings)

        operations_raw = payload["operations"]
        if not isinstance(operations_raw, list):
            raise ValueError("migration plan field 'operations' must be a list")
        operations = tuple(MigrationOperation.from_dict(item) for item in operations_raw)
        if operations != tuple(sorted(operations, key=lambda item: item.source)):
            raise ValueError("migration plan operations must be sorted by source")
        if len({operation.source for operation in operations}) != len(operations):
            raise ValueError("migration plan operation sources must be unique")

        plan_sha256 = _strict_string(payload, "plan_sha256", label="migration plan")
        _validate_sha256(plan_sha256, field_name="plan_sha256")
        plan = cls(
            schema_version=schema_version,
            transaction_id=transaction_id,
            created_at=created_at,
            source_root=source_root,
            destination_root=destination_root,
            source_fingerprint=source_fingerprint,
            source_file_count=source_file_count,
            source_bytes=source_bytes,
            source_device=source_device,
            source_inode=source_inode,
            ready=ready,
            blocking_entries=blocking_entries,
            explicit_mappings=explicit_mappings,
            operations=operations,
            plan_sha256=plan_sha256,
        )
        if not hmac.compare_digest(plan.plan_sha256, plan.canonical_sha256()):
            raise ValueError("migration plan canonical SHA-256 does not match its payload")
        _validate_plan_semantics(plan)
        return plan


@dataclass(frozen=True)
class RollbackManifest:
    schema_version: int
    transaction_id: str
    created_at: str
    source_fingerprint: str
    source_root: str
    destination_root: str
    source_unchanged: bool
    action: str
    operations: tuple[MigrationOperation, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strict_object(
    value: object,
    *,
    expected: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    keys = set(value)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(unknown)}")
    return value


def _strict_string(payload: dict[str, object], field_name: str, *, label: str) -> str:
    value = payload[field_name]
    if type(value) is not str or not value:
        raise ValueError(f"{label} field {field_name!r} must be a non-empty string")
    _reject_control_characters(value, field_name=f"{label} field {field_name!r}")
    return value


def _strict_integer(payload: dict[str, object], field_name: str, *, label: str) -> int:
    value = payload[field_name]
    if type(value) is not int:
        raise ValueError(f"{label} field {field_name!r} must be an integer")
    return value


def _strict_nonnegative_integer(
    payload: dict[str, object], field_name: str, *, label: str
) -> int:
    value = _strict_integer(payload, field_name, label=label)
    if value < 0:
        raise ValueError(f"{label} field {field_name!r} must be non-negative")
    return value


def _strict_string_tuple(
    payload: dict[str, object], field_name: str, *, label: str
) -> tuple[str, ...]:
    value = payload[field_name]
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise ValueError(f"{label} field {field_name!r} must be a list of strings")
    return tuple(value)


def _validate_sha256(value: str, *, field_name: str) -> None:
    if not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError(f"{field_name} must be 64 lowercase hexadecimal characters")


def _validate_absolute_manifest_path(value: str, *, field_name: str) -> None:
    _reject_control_characters(value, field_name=field_name)
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"migration plan {field_name} must be an absolute normalized path")
    if str(path) != value:
        raise ValueError(f"migration plan {field_name} must use canonical path spelling")


def _validate_top_level_entry(value: str, *, field_name: str) -> None:
    _reject_control_characters(value, field_name=field_name)
    if value in {"", ".", ".."} or Path(value).name != value:
        raise ValueError(f"{field_name} must be one top-level entry name")
    if "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{field_name} must be one top-level entry name")


def _validate_explicit_destination(value: str) -> str:
    _reject_control_characters(value, field_name="explicit migration destination")
    if "\\" in value or "\x00" in value:
        raise ValueError("explicit migration destination must use a safe relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or value != relative.as_posix() or ".." in relative.parts:
        raise ValueError("explicit migration destination must be a normalized relative path")
    parts = relative.parts
    categories = {category.value for category in ContextCategory}
    category_destination = len(parts) >= 3 and parts[0] in categories and parts[1] == "common"
    imported_destination = (
        len(parts) >= 4
        and parts[0] == AFS_STATE_DIR
        and parts[1] == "compat"
        and parts[2] == "imported"
    )
    if not (category_destination or imported_destination):
        raise ValueError(
            "explicit migration destination must be below '<category>/common/' "
            "or '.afs/compat/imported/'"
        )
    return relative.as_posix()


def _reject_control_characters(value: str, *, field_name: str) -> None:
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{field_name} must not contain control or format characters")


def _validate_mapping_destination_collisions(
    mappings: tuple[MigrationMapping, ...],
) -> None:
    destinations: list[tuple[tuple[str, ...], str, str]] = []
    for mapping in mappings:
        normalized = _validate_explicit_destination(mapping.destination)
        parts = tuple(part.casefold() for part in PurePosixPath(normalized).parts)
        for existing_parts, existing_destination, existing_source in destinations:
            if parts == existing_parts:
                raise ValueError(
                    "explicit migration destinations collide: "
                    f"{existing_source!r}->{existing_destination!r} and "
                    f"{mapping.source!r}->{mapping.destination!r}"
                )
            if parts[: len(existing_parts)] == existing_parts or existing_parts[: len(parts)] == parts:
                raise ValueError(
                    "explicit migration destinations have a prefix collision: "
                    f"{existing_destination!r} and {mapping.destination!r}"
                )
        destinations.append((parts, mapping.destination, mapping.source))


def _paths_collide_or_overlap(first: str, second: str) -> bool:
    first_parts = tuple(part.casefold() for part in PurePosixPath(first).parts)
    second_parts = tuple(part.casefold() for part in PurePosixPath(second).parts)
    return (
        first_parts == second_parts
        or first_parts[: len(second_parts)] == second_parts
        or second_parts[: len(first_parts)] == first_parts
    )


def _validate_explicit_builtin_collisions(
    mappings: tuple[MigrationMapping, ...],
    *,
    present_builtin_sources: set[str],
) -> None:
    for mapping in mappings:
        for source in sorted(present_builtin_sources):
            target = _V1_MIGRATION_TARGETS[source]
            if _paths_collide_or_overlap(mapping.destination, target):
                raise ValueError(
                    "explicit migration destination collides with a built-in mapping: "
                    f"{mapping.source!r}->{mapping.destination!r} overlaps "
                    f"{source!r}->{target!r}"
                )


def _validate_plan_semantics(plan: MigrationPlan) -> None:
    source_root = Path(plan.source_root)
    destination_root = Path(plan.destination_root)
    if (
        source_root == destination_root
        or destination_root.is_relative_to(source_root)
        or source_root.is_relative_to(destination_root)
    ):
        raise ValueError("migration plan source and destination roots must be separate")
    if plan.ready != (not plan.blocking_entries):
        raise ValueError("migration plan ready flag is inconsistent with blocking_entries")

    mapping_targets = {mapping.source: mapping.destination for mapping in plan.explicit_mappings}
    known_sources = set(_V1_MIGRATION_TARGETS)
    if known_sources.intersection(mapping_targets):
        raise ValueError("explicit mappings may only name unknown top-level entries")
    if set(plan.blocking_entries).intersection(mapping_targets):
        raise ValueError("mapped entries must not remain in blocking_entries")

    operation_sources: set[str] = set()
    destination_keys: set[str] = set()
    for operation in plan.operations:
        operation_source = Path(operation.source)
        try:
            relative_source = operation_source.relative_to(source_root)
        except ValueError as exc:
            raise ValueError("migration operation source escapes source_root") from exc
        if len(relative_source.parts) != 1:
            raise ValueError("migration operation source must be one top-level source entry")
        source_entry = relative_source.parts[0]
        _validate_top_level_entry(source_entry, field_name="operation source entry")
        if source_entry in operation_sources:
            raise ValueError("migration operation sources must be unique")
        if source_entry in plan.blocking_entries:
            raise ValueError("blocking entries must not have migration operations")
        target = _V1_MIGRATION_TARGETS.get(source_entry) or mapping_targets.get(source_entry)
        if target is None:
            raise ValueError(
                f"migration operation source has no built-in or explicit mapping: {source_entry}"
            )
        expected_destination = destination_root / target
        if Path(operation.destination) != expected_destination:
            raise ValueError(
                f"migration operation destination does not match its approved mapping: {source_entry}"
            )
        destination_key = operation.destination.casefold()
        if destination_key in destination_keys:
            raise ValueError("migration operation destinations must be unique")
        destination_keys.add(destination_key)
        operation_sources.add(source_entry)

    missing_mapped_operations = set(mapping_targets) - operation_sources
    if missing_mapped_operations:
        raise ValueError(
            "explicit mappings are missing migration operations: "
            + ", ".join(sorted(missing_mapped_operations))
        )
    _validate_explicit_builtin_collisions(
        plan.explicit_mappings,
        present_builtin_sources=operation_sources.intersection(known_sources),
    )


def _stat_signature(path_stat: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        path_stat.st_dev,
        path_stat.st_ino,
        path_stat.st_mode,
        path_stat.st_size,
        path_stat.st_mtime_ns,
        path_stat.st_ctime_ns,
    )


def _update_inventory_digest(
    digest: hashlib._Hash,
    *fields: bytes,
) -> None:
    for value in fields:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)


def inventory_source_tree(root: Path) -> SourceInventory:
    """Hash a source tree without following links and reject unstable reads."""

    source = lexical_absolute(root)
    try:
        root_before = os.lstat(source)
    except OSError:
        raise
    if is_linklike(root_before) or not stat.S_ISDIR(root_before.st_mode):
        raise ValueError(f"migration source must be a regular directory: {source}")

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0

    def record(path: Path, relative: str) -> None:
        nonlocal file_count, total_bytes
        try:
            before = os.lstat(path)
        except FileNotFoundError as exc:
            raise ValueError(f"source changed while inventorying: {relative}") from exc
        mode = str(stat.S_IMODE(before.st_mode)).encode("ascii")
        relative_bytes = os.fsencode(relative)
        if is_linklike(before):
            raise ValueError(
                f"symbolic links and reparse points are not executable migration sources: {relative}"
            )
        if stat.S_ISDIR(before.st_mode):
            _update_inventory_digest(digest, b"D", relative_bytes, mode)
            try:
                children = sorted(path.iterdir(), key=lambda child: os.fsencode(child.name))
            except FileNotFoundError as exc:
                raise ValueError(f"source changed while inventorying: {relative}") from exc
            for child in children:
                child_relative = child.relative_to(source).as_posix()
                record(child, child_relative)
            try:
                after = os.lstat(path)
            except FileNotFoundError as exc:
                raise ValueError(f"source changed while inventorying: {relative}") from exc
            if _stat_signature(before) != _stat_signature(after):
                raise ValueError(f"source changed while inventorying: {relative}")
            return
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"unsupported special file in migration source: {relative}")
        if before.st_nlink > 1:
            raise ValueError(f"hard-linked files are not executable migration sources: {relative}")

        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError as exc:
            raise ValueError(f"source changed while inventorying: {relative}") from exc
        content_digest = hashlib.sha256()
        bytes_read = 0
        try:
            opened = os.fstat(descriptor)
            if _stat_signature(before) != _stat_signature(opened):
                raise ValueError(f"source changed while inventorying: {relative}")
            while chunk := os.read(descriptor, 1024 * 1024):
                content_digest.update(chunk)
                bytes_read += len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        try:
            final_path_stat = os.lstat(path)
        except FileNotFoundError as exc:
            raise ValueError(f"source changed while inventorying: {relative}") from exc
        if (
            _stat_signature(before) != _stat_signature(after)
            or _stat_signature(after) != _stat_signature(final_path_stat)
            or bytes_read != before.st_size
        ):
            raise ValueError(f"source changed while inventorying: {relative}")
        file_count += 1
        total_bytes += bytes_read
        _update_inventory_digest(
            digest,
            b"F",
            relative_bytes,
            mode,
            str(bytes_read).encode("ascii"),
            content_digest.digest(),
        )

    record(source, ".")
    root_after = os.lstat(source)
    if _stat_signature(root_before) != _stat_signature(root_after):
        raise ValueError("source changed while inventorying: .")
    return SourceInventory(
        fingerprint=digest.hexdigest(),
        file_count=file_count,
        total_bytes=total_bytes,
        device=root_after.st_dev,
        inode=root_after.st_ino,
    )


def _tree_fingerprint(root: Path) -> tuple[str, int, int]:
    inventory = inventory_source_tree(root)
    return inventory.fingerprint, inventory.file_count, inventory.total_bytes


def _normalize_explicit_mappings(
    explicit_mappings: Mapping[str, str] | None,
    *,
    unknown_entries: tuple[str, ...],
) -> tuple[MigrationMapping, ...]:
    if explicit_mappings is None:
        return ()
    if not isinstance(explicit_mappings, Mapping):
        raise ValueError("explicit_mappings must be a mapping of source names to destinations")
    unknown = set(unknown_entries)
    normalized: list[MigrationMapping] = []
    for source, destination in explicit_mappings.items():
        if type(source) is not str or type(destination) is not str:
            raise ValueError("explicit migration mapping keys and values must be strings")
        _validate_top_level_entry(source, field_name="mapping source")
        if source not in unknown:
            raise ValueError(
                f"explicit migration mapping source is not an unknown top-level entry: {source}"
            )
        normalized.append(
            MigrationMapping(source, _validate_explicit_destination(destination))
        )
    mappings = tuple(sorted(normalized))
    _validate_mapping_destination_collisions(mappings)
    return mappings


def build_migration_plan(
    context_root: Path,
    destination_root: Path,
    *,
    explicit_mappings: Mapping[str, str] | None = None,
) -> MigrationPlan:
    """Build a hash-bound v1-to-v2 plan; never execute its operations."""

    source = lexical_absolute(context_root)
    assert_no_linklike_components(source, allow_missing=False)
    if not source.is_dir():
        raise FileNotFoundError(f"context root does not exist: {source}")
    source_stat = os.lstat(source)
    if is_linklike(source_stat):
        raise ValueError("migration source root must not be a symbolic link or reparse point")
    source = source.resolve()
    if detect_layout_version(source) != 1:
        raise ValueError("migration planning requires an unmarked v1 context root")
    destination_lexical = lexical_absolute(destination_root)
    assert_no_linklike_components(destination_lexical)
    destination = destination_lexical.resolve(strict=False)
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(
        destination
    ):
        raise ValueError(
            "migration destination must be separate from, and must not contain or be "
            "contained by, the v1 source root"
        )
    try:
        destination_stat = os.lstat(destination_lexical)
    except FileNotFoundError:
        destination_stat = None
    if destination_stat is not None:
        raise FileExistsError(f"migration destination already exists: {destination_lexical}")
    _validate_absolute_manifest_path(str(source), field_name="source_root")
    _validate_absolute_manifest_path(str(destination), field_name="destination_root")
    audit = audit_layout(source)
    mappings = _normalize_explicit_mappings(
        explicit_mappings,
        unknown_entries=audit.unknown_entries,
    )
    present_builtin_sources = {
        entry.name for entry in source.iterdir() if entry.name in _V1_MIGRATION_TARGETS
    }
    _validate_explicit_builtin_collisions(
        mappings,
        present_builtin_sources=present_builtin_sources,
    )
    mapping_targets = {mapping.source: mapping.destination for mapping in mappings}
    blocking_entries = tuple(
        sorted(
            {
                issue.path
                for issue in audit.issues
                if issue.blocking and issue.path
                and not (
                    issue.code == "unknown_entry" and issue.path in mapping_targets
                )
            }
        )
    )
    inventory = inventory_source_tree(source)
    operations: list[MigrationOperation] = []
    for entry in sorted(source.iterdir(), key=lambda item: item.name):
        target = _V1_MIGRATION_TARGETS.get(entry.name) or mapping_targets.get(entry.name)
        if target is None or entry.name in blocking_entries:
            continue
        destination_path = destination / target
        if entry.resolve(strict=False) == destination_path.resolve(strict=False):
            continue
        operations.append(
            MigrationOperation(
                operation="copy_verify",
                source=str(entry),
                destination=str(destination_path),
            )
        )
    plan = MigrationPlan(
        schema_version=2,
        transaction_id=f"layout_{uuid4().hex}",
        created_at=_utc_now(),
        source_root=str(source),
        destination_root=str(destination),
        source_fingerprint=inventory.fingerprint,
        source_file_count=inventory.file_count,
        source_bytes=inventory.total_bytes,
        source_device=inventory.device,
        source_inode=inventory.inode,
        ready=not blocking_entries,
        blocking_entries=blocking_entries,
        explicit_mappings=mappings,
        operations=tuple(operations),
        plan_sha256="",
    )
    return plan.with_canonical_hash()


def build_rollback_manifest(plan: MigrationPlan) -> RollbackManifest:
    """Describe non-overwriting rollback for a copy-only migration plan."""

    return RollbackManifest(
        schema_version=2,
        transaction_id=plan.transaction_id,
        created_at=_utc_now(),
        source_fingerprint=plan.source_fingerprint,
        source_root=plan.source_root,
        destination_root=plan.destination_root,
        source_unchanged=True,
        action="retain_source_and_deactivate_destination",
        operations=(),
    )


def load_migration_plan(path: Path) -> MigrationPlan:
    """Load a private regular-file plan and verify its canonical hash."""

    plan_path = path.expanduser()
    try:
        before = os.lstat(plan_path)
    except FileNotFoundError:
        raise
    if is_linklike(before):
        raise ValueError(f"migration plan must not be a symbolic link or reparse point: {plan_path}")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"migration plan must be a regular file: {plan_path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(plan_path, flags)
    try:
        opened = os.fstat(descriptor)
        if _stat_signature(before) != _stat_signature(opened):
            raise ValueError("migration plan changed while opening")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            total += len(chunk)
            if total > 16 * 1024 * 1024:
                raise ValueError("migration plan exceeds the 16 MiB size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final_path_stat = os.lstat(plan_path)
    if (
        _stat_signature(before) != _stat_signature(after)
        or _stat_signature(after) != _stat_signature(final_path_stat)
    ):
        raise ValueError("migration plan changed while reading")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"migration plan JSON contains duplicate key: {key!r}")
            payload[key] = value
        return payload

    try:
        payload = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid migration plan JSON: {exc}") from exc
    plan = MigrationPlan.from_dict(payload)
    source_root = Path(plan.source_root)
    source_stat = os.lstat(source_root)
    if is_linklike(source_stat) or not stat.S_ISDIR(source_stat.st_mode):
        raise ValueError("migration plan source_root is no longer a regular directory")
    if source_stat.st_dev != plan.source_device or source_stat.st_ino != plan.source_inode:
        raise ValueError("migration plan source device/inode no longer matches")
    audit = audit_layout(source_root)
    actual_unknown = set(audit.unknown_entries)
    mapped_sources = {mapping.source for mapping in plan.explicit_mappings}
    for mapping in plan.explicit_mappings:
        if mapping.source not in actual_unknown:
            raise ValueError(
                "explicit migration mapping source is not currently an unknown "
                f"top-level entry: {mapping.source}"
            )
    expected_blocking = tuple(
        sorted(
            {
                issue.path
                for issue in audit.issues
                if issue.blocking
                and issue.path
                and not (issue.code == "unknown_entry" and issue.path in mapped_sources)
            }
        )
    )
    if plan.blocking_entries != expected_blocking:
        raise ValueError("migration plan blocking_entries no longer match the source root")
    expected_operation_sources = {
        str(entry)
        for entry in source_root.iterdir()
        if entry.name not in expected_blocking
        and (entry.name in _V1_MIGRATION_TARGETS or entry.name in mapped_sources)
    }
    actual_operation_sources = {operation.source for operation in plan.operations}
    if actual_operation_sources != expected_operation_sources:
        raise ValueError("migration plan operation set no longer matches the source root")
    return plan


def write_manifest(path: Path, manifest: MigrationPlan | RollbackManifest) -> Path:
    """Atomically persist a plan or rollback manifest as private JSON."""

    destination = lexical_absolute(path)
    assert_no_linklike_components(destination)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert_no_linklike_components(destination.parent, allow_missing=False)
    # Re-check the leaf after creating parents. A concurrently planted leaf
    # symlink is replaced as a directory entry by the atomic writer rather than
    # followed, but rejecting it keeps the reviewed output contract explicit.
    assert_no_linklike_components(destination)
    _atomic_write_text(destination, json.dumps(manifest.to_dict(), indent=2) + "\n")
    return destination
