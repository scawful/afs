"""Project identity and scope authorization for a central context root."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .context_layout import LAYOUT_VERSION, LayoutMetadata, _atomic_write_text
from .models import ContextCategory
from .toml_compat import tomllib

COMMON_SCOPE_ID = "common"


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
        if not all(isinstance(item, str) and item for item in (project_id, scope_id, name, path)):
            raise ValueError("project record requires non-empty project_id, scope_id, name, and path")
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
        self.root = self.context_root / ".afs" / "projects"

    def _record_path(self, project_id: str) -> Path:
        if not project_id.startswith("prj_") or not project_id[4:].isalnum():
            raise ValueError("invalid project id")
        return self.root / f"{project_id}.toml"

    def _load_path(self, path: Path) -> ProjectRecord:
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"invalid project record: {path}") from exc
        return ProjectRecord.from_dict(payload)

    def all_records(self) -> tuple[ProjectRecord, ...]:
        """Return registry metadata; callers must not use this for artifact reads."""

        if not self.root.is_dir():
            return ()
        return tuple(self._load_path(path) for path in sorted(self.root.glob("prj_*.toml")))

    def register(self, project_path: Path, *, name: str | None = None) -> ProjectRecord:
        """Register a path once and preserve its stable project/scope identity."""

        path = _canonical(project_path)
        if not path.is_dir():
            raise FileNotFoundError(f"project path does not exist: {path}")
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
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _atomic_write_text(self._record_path(project_id), record.render())
        return record

    def add_alias(self, project_id: str, alias_path: Path, *, requester_path: Path) -> ProjectRecord:
        """Add a checkout/worktree alias after authorizing the current project."""

        record = self.get(project_id, requester_path=requester_path)
        alias = _canonical(alias_path)
        if not alias.is_dir():
            raise FileNotFoundError(f"project alias path does not exist: {alias}")
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

        requested_scope = scope_id
        if requested_scope is None:
            current = self.resolve(requester_path)
            requested_scope = current.scope_id if current is not None else COMMON_SCOPE_ID
        self.assert_scope_authorized(
            requested_scope,
            requester_path=requester_path,
            allow_all_projects=allow_all_projects,
        )
        relative = Path(relative_path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("artifact path must be a contained relative path")
        scope_directory = (
            Path("common")
            if requested_scope == COMMON_SCOPE_ID
            else Path("projects") / requested_scope.removeprefix("project:")
        )
        category_root = self.context_root / category.value
        target = (category_root / scope_directory / relative).resolve(strict=False)
        expected_root = (category_root / scope_directory).resolve(strict=False)
        if not _contains(expected_root, target):
            raise ValueError("artifact path escapes its authorized scope")
        return target

