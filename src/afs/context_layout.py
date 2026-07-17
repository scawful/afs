"""Versioned central context layout primitives.

This module intentionally separates inspection/planning from execution.  It
can scaffold a *new* v2 namespace and write plan manifests, but it never moves
or deletes an existing context tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import ContextCategory, MountType
from .path_safety import assert_no_linklike_components, is_linklike
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
    ready: bool
    blocking_entries: tuple[str, ...]
    operations: tuple[MigrationOperation, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RollbackManifest:
    schema_version: int
    transaction_id: str
    created_at: str
    source_fingerprint: str
    operations: tuple[MigrationOperation, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tree_fingerprint(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            digest.update(b"L\0" + relative.encode() + b"\0" + os.readlink(path).encode() + b"\0")
            continue
        if not path.is_file():
            continue
        file_count += 1
        digest.update(b"F\0" + relative.encode() + b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                total_bytes += len(chunk)
        digest.update(b"\0")
    return digest.hexdigest(), file_count, total_bytes


def build_migration_plan(context_root: Path, destination_root: Path | None = None) -> MigrationPlan:
    """Build a hash-bound v1-to-v2 plan; never execute its operations."""

    source = context_root.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"context root does not exist: {source}")
    if detect_layout_version(source) != 1:
        raise ValueError("migration planning requires an unmarked v1 context root")
    destination = (destination_root or source).expanduser().resolve()
    if destination != source and (
        destination.is_relative_to(source) or source.is_relative_to(destination)
    ):
        raise ValueError(
            "a separate migration destination must not contain, or be contained by, "
            "the v1 source root"
        )
    audit = audit_layout(source)
    blocking_entries = tuple(
        sorted(
            {
                issue.path
                for issue in audit.issues
                if issue.blocking and issue.path
            }
        )
    )
    fingerprint, file_count, total_bytes = _tree_fingerprint(source)
    operations: list[MigrationOperation] = []
    for entry in sorted(source.iterdir(), key=lambda item: item.name):
        target = _V1_MIGRATION_TARGETS.get(entry.name)
        if target is None or entry.name in blocking_entries:
            continue
        destination_path = destination / target
        if entry.resolve(strict=False) == destination_path.resolve(strict=False):
            continue
        operations.append(
            MigrationOperation(
                # Category imports target a child of their legacy source for
                # an in-place plan.  A future executor must stage that copy to
                # avoid recursive traversal; AFS intentionally provides no
                # executor today.
                operation=(
                    "copy_verify_staged"
                    if destination == source
                    and destination_path.is_relative_to(entry)
                    else "copy_verify"
                ),
                source=str(entry),
                destination=str(destination_path),
            )
        )
    return MigrationPlan(
        schema_version=1,
        transaction_id=f"layout_{uuid4().hex}",
        created_at=_utc_now(),
        source_root=str(source),
        destination_root=str(destination),
        source_fingerprint=fingerprint,
        source_file_count=file_count,
        source_bytes=total_bytes,
        ready=audit.migration_ready,
        blocking_entries=blocking_entries,
        operations=tuple(operations),
    )


def build_rollback_manifest(plan: MigrationPlan) -> RollbackManifest:
    """Describe how a not-yet-executed plan would be reversed."""

    reverse = tuple(
        MigrationOperation(
            operation="restore_verified_source",
            source=operation.destination,
            destination=operation.source,
            verify=operation.verify,
        )
        for operation in reversed(plan.operations)
    )
    return RollbackManifest(
        schema_version=1,
        transaction_id=plan.transaction_id,
        created_at=_utc_now(),
        source_fingerprint=plan.source_fingerprint,
        operations=reverse,
    )


def write_manifest(path: Path, manifest: MigrationPlan | RollbackManifest) -> Path:
    """Atomically persist a plan or rollback manifest as private JSON."""

    destination = path.expanduser().resolve()
    _atomic_write_text(destination, json.dumps(manifest.to_dict(), indent=2) + "\n")
    return destination
