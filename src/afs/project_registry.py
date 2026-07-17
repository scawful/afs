"""Project identity and scope authorization for a central context root."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .context_layout import LAYOUT_VERSION, LayoutMetadata, _atomic_write_text
from .index_storage import index_file_lock
from .models import ContextCategory
from .path_safety import assert_no_linklike_components
from .toml_compat import tomllib

COMMON_SCOPE_ID = "common"


def _validate_project_id(project_id: str) -> str:
    normalized = str(project_id).strip()
    if (
        not normalized.startswith("prj_")
        or not normalized[4:]
        or not normalized[4:].isalnum()
    ):
        raise ValueError("invalid project id")
    return normalized


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(path: Path) -> Path:
    return path.expanduser().resolve()


def _contains(root: Path, candidate: Path) -> bool:
    return candidate == root or candidate.is_relative_to(root)


class ScopeAuthorizationError(PermissionError):
    """Raised when a project-scoped lookup crosses its authorization boundary."""


@dataclass(frozen=True)
class ProjectRecord:
    """Persistent project identity independent of checkout path."""

    project_id: str
    scope_id: str
    name: str
    path: str
    aliases: tuple[str, ...] = ()
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectRecord:
        project_id = payload.get("project_id")
        scope_id = payload.get("scope_id")
        name = payload.get("name")
        path = payload.get("path")
        aliases = payload.get("aliases", [])
        created_at = payload.get("created_at")
        updated_at = payload.get("updated_at")
        if (
            not isinstance(project_id, str)
            or not project_id
            or not isinstance(scope_id, str)
            or not scope_id
            or not isinstance(name, str)
            or not name
            or not isinstance(path, str)
            or not path
        ):
            raise ValueError(
                "project record requires non-empty project_id, scope_id, name, and path"
            )
        project_id = _validate_project_id(project_id)
        if scope_id != f"project:{project_id}":
            raise ValueError("project scope_id must be derived from project_id")
        if not isinstance(aliases, list) or not all(isinstance(item, str) for item in aliases):
            raise ValueError("project aliases must be a list of paths")
        if not isinstance(created_at, str) or not isinstance(updated_at, str):
            raise ValueError("project timestamps must be strings")
        return cls(
            project_id=project_id,
            scope_id=scope_id,
            name=name,
            path=str(_canonical(Path(path))),
            aliases=tuple(str(_canonical(Path(item))) for item in aliases),
            created_at=created_at,
            updated_at=updated_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def roots(self) -> tuple[Path, ...]:
        return (Path(self.path), *(Path(alias) for alias in self.aliases))

    def render(self) -> str:
        aliases = ", ".join(json.dumps(item) for item in self.aliases)
        return "\n".join(
            [
                f"project_id = {json.dumps(self.project_id)}",
                f"scope_id = {json.dumps(self.scope_id)}",
                f"name = {json.dumps(self.name)}",
                f"path = {json.dumps(self.path)}",
                f"aliases = [{aliases}]",
                f"created_at = {json.dumps(self.created_at)}",
                f"updated_at = {json.dumps(self.updated_at)}",
                "",
            ]
        )


class ProjectRegistry:
    """Scoped project registry stored below ``.afs/projects``."""

    def __init__(self, context_root: Path) -> None:
        self.context_root = _canonical(context_root)
        metadata = LayoutMetadata.load(self.context_root)
        if metadata is None or metadata.layout_version != LAYOUT_VERSION:
            raise ValueError(f"project registry requires a v2 context root: {self.context_root}")
        self.root = assert_no_linklike_components(
            self.context_root / ".afs" / "projects",
            boundary=self.context_root,
        )

    def _safe_root(self, *, allow_missing: bool = True) -> Path:
        return assert_no_linklike_components(
            self.root,
            boundary=self.context_root,
            allow_missing=allow_missing,
        )

    def _record_path(self, project_id: str) -> Path:
        project_id = _validate_project_id(project_id)
        return assert_no_linklike_components(
            self._safe_root() / f"{project_id}.toml",
            boundary=self.context_root,
        )

    def _assert_external_project_path(self, path: Path, *, field_name: str) -> None:
        if path == self.context_root or path.is_relative_to(self.context_root):
            raise ValueError(
                f"{field_name} must be outside the central context root: {path}"
            )

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        """Serialize registry identity and alias read-modify-write operations."""

        root = self._safe_root()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = assert_no_linklike_components(
            root / ".registry.lock",
            boundary=self.context_root,
        )
        with index_file_lock(lock_path):
            self._safe_root(allow_missing=False)
            yield

    def _load_path(self, path: Path) -> ProjectRecord:
        path = assert_no_linklike_components(
            path,
            boundary=self.context_root,
            allow_missing=False,
        )
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"invalid project record: {path}") from exc
        expected_project_id = _validate_project_id(path.stem)
        record = ProjectRecord.from_dict(payload)
        if record.project_id != expected_project_id:
            raise ValueError(
                f"project record id does not match filename: {path.name}"
            )
        self._assert_external_project_path(Path(record.path), field_name="project path")
        for alias in record.aliases:
            self._assert_external_project_path(Path(alias), field_name="project alias")
        return record

    def all_records(self) -> tuple[ProjectRecord, ...]:
        """Return registry metadata; callers must not use this for artifact reads."""

        root = self._safe_root()
        if not root.is_dir():
            return ()
        return tuple(self._load_path(path) for path in sorted(root.glob("prj_*.toml")))

    def register(self, project_path: Path, *, name: str | None = None) -> ProjectRecord:
        """Register a path once and preserve its stable project/scope identity."""

        path = _canonical(project_path)
        if not path.is_dir():
            raise FileNotFoundError(f"project path does not exist: {path}")
        self._assert_external_project_path(path, field_name="project path")
        with self._write_lock():
            existing = self.resolve(path)
            if existing is not None and path in existing.roots():
                return existing

            now = _utc_now()
            project_id = f"prj_{uuid4().hex}"
            record = ProjectRecord(
                project_id=project_id,
                scope_id=f"project:{project_id}",
                name=(name or path.name).strip() or path.name,
                path=str(path),
                created_at=now,
                updated_at=now,
            )
            _atomic_write_text(self._record_path(project_id), record.render())
            return record

    def add_alias(
        self, project_id: str, alias_path: Path, *, requester_path: Path
    ) -> ProjectRecord:
        """Add a checkout/worktree alias after authorizing the current project."""

        alias = _canonical(alias_path)
        if not alias.is_dir():
            raise FileNotFoundError(f"project alias path does not exist: {alias}")
        self._assert_external_project_path(alias, field_name="project alias")
        with self._write_lock():
            record = self.get(project_id, requester_path=requester_path)
            claimed = next(
                (
                    candidate
                    for candidate in self.all_records()
                    if alias in candidate.roots()
                ),
                None,
            )
            if claimed is not None and claimed.project_id != record.project_id:
                raise ValueError(
                    f"project alias is already registered to {claimed.project_id}: {alias}"
                )
            aliases = tuple(dict.fromkeys((*record.aliases, str(alias))))
            updated = ProjectRecord(
                project_id=record.project_id,
                scope_id=record.scope_id,
                name=record.name,
                path=record.path,
                aliases=aliases,
                created_at=record.created_at,
                updated_at=_utc_now(),
            )
            _atomic_write_text(self._record_path(project_id), updated.render())
            return updated

    def resolve(self, cwd: Path) -> ProjectRecord | None:
        """Resolve cwd to the most-specific registered project root."""

        candidate = _canonical(cwd)
        matches: list[tuple[int, ProjectRecord]] = []
        for record in self.all_records():
            for root in record.roots():
                if _contains(root, candidate):
                    matches.append((len(root.parts), record))
        if not matches:
            return None
        matches.sort(key=lambda item: (-item[0], item[1].project_id))
        return matches[0][1]

    def authorized_scope_ids(
        self,
        requester_path: Path,
        *,
        allow_all_projects: bool = False,
    ) -> frozenset[str]:
        """Return the only scopes a lookup may use for this requester."""

        allowed = {COMMON_SCOPE_ID}
        if allow_all_projects:
            allowed.update(record.scope_id for record in self.all_records())
            return frozenset(allowed)
        current = self.resolve(requester_path)
        if current is not None:
            allowed.add(current.scope_id)
        return frozenset(allowed)

    def codebase_excluded_roots(
        self,
        requester_path: Path,
        *,
        project_id: str,
    ) -> tuple[Path, ...]:
        """Return managed/foreign descendants a project crawl must not enter."""

        source = self.resolve_codebase_root(
            requester_path,
            project_id=project_id,
        )
        return self._codebase_excluded_roots(source, project_id=project_id)

    def resolve_codebase_root(
        self,
        requester_path: Path,
        *,
        project_id: str,
    ) -> Path:
        """Return the exact registered checkout root that owns a requester."""

        candidate = _canonical(requester_path)
        _validate_project_id(project_id)
        self._assert_external_project_path(candidate, field_name="project requester")
        owner = self.resolve(candidate)
        if owner is None or owner.project_id != project_id:
            raise ScopeAuthorizationError(
                f"project requester {candidate} is not owned by {project_id}"
            )
        matches = [
            _canonical(root)
            for root in owner.roots()
            if _contains(_canonical(root), candidate)
        ]
        if not matches:
            raise ScopeAuthorizationError(
                f"project requester {candidate} has no registered root for {project_id}"
            )
        return max(matches, key=lambda root: (len(root.parts), str(root)))

    def resolve_codebase_boundaries(
        self,
        requester_path: Path,
        *,
        project_id: str,
    ) -> tuple[Path, tuple[Path, ...]]:
        """Return the authorized crawl root and descendants it must exclude."""

        source = self.resolve_codebase_root(requester_path, project_id=project_id)
        return source, self._codebase_excluded_roots(source, project_id=project_id)

    def _codebase_excluded_roots(
        self,
        source: Path,
        *,
        project_id: str,
    ) -> tuple[Path, ...]:
        self._assert_external_project_path(source, field_name="project source")
        excluded: set[Path] = set()
        if self.context_root.is_relative_to(source):
            excluded.add(self.context_root)
        for record in self.all_records():
            if record.project_id == project_id:
                continue
            for root in record.roots():
                canonical_root = _canonical(root)
                if canonical_root != source and canonical_root.is_relative_to(source):
                    excluded.add(canonical_root)
        return tuple(sorted(excluded, key=str))

    def assert_scope_authorized(
        self,
        scope_id: str,
        *,
        requester_path: Path,
        allow_all_projects: bool = False,
    ) -> None:
        allowed = self.authorized_scope_ids(
            requester_path,
            allow_all_projects=allow_all_projects,
        )
        if scope_id not in allowed:
            raise ScopeAuthorizationError(
                f"scope {scope_id!r} is not authorized for {Path(requester_path).expanduser()}"
            )

    def get(
        self,
        project_id: str,
        *,
        requester_path: Path,
        allow_all_projects: bool = False,
    ) -> ProjectRecord:
        """Read one project record only after authorizing its scope."""

        record = self._load_path(self._record_path(project_id))
        self.assert_scope_authorized(
            record.scope_id,
            requester_path=requester_path,
            allow_all_projects=allow_all_projects,
        )
        return record

    def resolve_scoped_path(
        self,
        category: ContextCategory,
        relative_path: str | Path,
        *,
        requester_path: Path,
        scope_id: str | None = None,
        allow_all_projects: bool = False,
    ) -> Path:
        """Resolve an artifact path with project/common authorization enforced."""

        _requested_scope, expected_root = self.resolve_scope_root(
            category,
            requester_path=requester_path,
            scope_id=scope_id,
            allow_all_projects=allow_all_projects,
        )
        relative = Path(relative_path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("artifact path must be a contained relative path")
        return assert_no_linklike_components(
            expected_root / relative,
            boundary=expected_root,
        )

    def resolve_scope_root(
        self,
        category: ContextCategory,
        *,
        requester_path: Path,
        scope_id: str | None = None,
        allow_all_projects: bool = False,
    ) -> tuple[str, Path]:
        """Resolve the authorized root of one category scope."""

        requested_scope = scope_id
        if requested_scope is None:
            current = self.resolve(requester_path)
            requested_scope = current.scope_id if current is not None else COMMON_SCOPE_ID
        self.assert_scope_authorized(
            requested_scope,
            requester_path=requester_path,
            allow_all_projects=allow_all_projects,
        )
        scope_directory = (
            Path("common")
            if requested_scope == COMMON_SCOPE_ID
            else Path("projects") / requested_scope.removeprefix("project:")
        )
        root = assert_no_linklike_components(
            self.context_root / category.value / scope_directory,
            boundary=self.context_root,
        )
        return requested_scope, root
