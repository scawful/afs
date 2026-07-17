from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from afs.cli._utils import resolve_context_paths
from afs.context_layout import scaffold_v2
from afs.core import find_existing_root, find_root
from afs.manager import AFSManager
from afs.models import ContextCategory
from afs.project_registry import (
    COMMON_SCOPE_ID,
    ProjectRecord,
    ProjectRegistry,
    ScopeAuthorizationError,
)
from afs.schema import AFSConfig, GeneralConfig


def _registry(tmp_path: Path) -> tuple[ProjectRegistry, Path]:
    root = tmp_path / ".context"
    scaffold_v2(root)
    return ProjectRegistry(root), root


def test_registry_resolves_most_specific_project_and_keeps_stable_ids(tmp_path: Path) -> None:
    registry, _root = _registry(tmp_path)
    workspace = tmp_path / "src"
    project = workspace / "project"
    nested = project / "packages" / "app"
    nested.mkdir(parents=True)

    parent_record = registry.register(workspace, name="workspace")
    project_record = registry.register(project)

    assert registry.register(project) == project_record
    assert registry.resolve(nested) == project_record
    assert registry.resolve(tmp_path / "elsewhere") is None
    assert parent_record.project_id != project_record.project_id


@pytest.mark.parametrize("relative", [Path(), Path("knowledge") / "common"])
def test_registry_rejects_project_inside_central_context(
    tmp_path: Path,
    relative: Path,
) -> None:
    registry, root = _registry(tmp_path)
    project = root / relative
    project.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="outside the central context root"):
        registry.register(project)


def test_search_rejects_existing_context_root_project_record(tmp_path: Path) -> None:
    from afs.cli.friendly import _search_sources

    registry, root = _registry(tmp_path)
    project_id = "prj_manual"
    record = ProjectRecord(
        project_id=project_id,
        scope_id=f"project:{project_id}",
        name="invalid-context-project",
        path=str(root),
    )
    (root / ".afs" / "projects" / f"{project_id}.toml").write_text(
        record.render(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside the central context root"):
        _search_sources(
            root,
            root,
            semantic=False,
            all_projects=False,
        )


def test_concurrent_registration_allocates_one_project_identity(tmp_path: Path) -> None:
    registry, root = _registry(tmp_path)
    project = tmp_path / "project"
    project.mkdir()

    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(lambda _: registry.register(project), range(16)))

    assert len({record.project_id for record in records}) == 1
    assert len(ProjectRegistry(root).all_records()) == 1


def test_alias_cannot_claim_path_registered_to_another_project(tmp_path: Path) -> None:
    registry, _root = _registry(tmp_path)
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)

    with pytest.raises(ValueError, match=beta_record.project_id):
        registry.add_alias(
            alpha_record.project_id,
            beta,
            requester_path=alpha,
        )


def test_nested_project_alias_is_not_claimed_by_registered_parent(tmp_path: Path) -> None:
    registry, _root = _registry(tmp_path)
    workspace = tmp_path / "workspace"
    nested = workspace / "nested"
    worktree = workspace / "nested-worktree"
    nested.mkdir(parents=True)
    worktree.mkdir()
    registry.register(workspace)
    nested_record = registry.register(nested)

    updated = registry.add_alias(
        nested_record.project_id,
        worktree,
        requester_path=nested,
    )

    assert str(worktree.resolve()) in updated.aliases
    assert registry.resolve(worktree) == updated


def test_registry_rejects_linked_system_root_before_read_or_write(tmp_path: Path) -> None:
    registry, root = _registry(tmp_path)
    outside = tmp_path / "outside"
    project = tmp_path / "project"
    outside.mkdir()
    project.mkdir()
    shutil.rmtree(root / ".afs" / "projects")
    (root / ".afs" / "projects").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        registry.register(project)
    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        ProjectRegistry(root)
    assert not any(outside.iterdir())


def test_registry_rejects_linked_record_leaf(tmp_path: Path) -> None:
    registry, root = _registry(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    record = registry.register(project)
    record_path = root / ".afs" / "projects" / f"{record.project_id}.toml"
    outside_record = tmp_path / "crafted.toml"
    outside_record.write_text(record.render(), encoding="utf-8")
    record_path.unlink()
    record_path.symlink_to(outside_record)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        registry.all_records()


def test_project_record_rejects_unsafe_project_identity(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(ValueError, match="invalid project id"):
        ProjectRecord.from_dict(
            {
                "project_id": "prj_../escape",
                "scope_id": "project:prj_../escape",
                "name": "escape",
                "path": str(project),
                "aliases": [],
                "created_at": "2026-07-16T00:00:00Z",
                "updated_at": "2026-07-16T00:00:00Z",
            }
        )


def test_registry_rejects_filename_and_record_identity_mismatch(
    tmp_path: Path,
) -> None:
    registry, root = _registry(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    record = registry.register(project)
    original_path = root / ".afs" / "projects" / f"{record.project_id}.toml"
    mismatched = ProjectRecord(
        project_id="prj_other",
        scope_id="project:prj_other",
        name=record.name,
        path=record.path,
        aliases=record.aliases,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
    original_path.write_text(mismatched.render(), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match filename"):
        registry.all_records()


def test_scope_authorization_defaults_to_current_project_and_common(tmp_path: Path) -> None:
    registry, root = _registry(tmp_path)
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)

    assert registry.authorized_scope_ids(alpha) == {COMMON_SCOPE_ID, alpha_record.scope_id}
    assert registry.resolve_scoped_path(
        ContextCategory.MEMORY,
        "notes/decision.md",
        requester_path=alpha,
    ) == root / "memory" / "projects" / alpha_record.project_id / "notes" / "decision.md"
    assert registry.resolve_scoped_path(
        ContextCategory.KNOWLEDGE,
        "shared.md",
        requester_path=alpha,
        scope_id=COMMON_SCOPE_ID,
    ) == root / "knowledge" / "common" / "shared.md"

    with pytest.raises(ScopeAuthorizationError):
        registry.get(beta_record.project_id, requester_path=alpha)
    with pytest.raises(ScopeAuthorizationError):
        registry.resolve_scoped_path(
            ContextCategory.MEMORY,
            "private.md",
            requester_path=alpha,
            scope_id=beta_record.scope_id,
        )

    assert registry.get(
        beta_record.project_id,
        requester_path=alpha,
        allow_all_projects=True,
    ) == beta_record


def test_scoped_path_rejects_traversal(tmp_path: Path) -> None:
    registry, _root = _registry(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    registry.register(project)

    with pytest.raises(ValueError, match="contained relative path"):
        registry.resolve_scoped_path(
            ContextCategory.SCRATCHPAD,
            "../other/secret.md",
            requester_path=project,
        )


@pytest.mark.parametrize("target_kind", ["other-project", "outside"])
def test_scope_root_rejects_project_directory_link_escape(
    tmp_path: Path,
    target_kind: str,
) -> None:
    registry, root = _registry(tmp_path)
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    target = (
        root / "memory" / "projects" / beta_record.project_id
        if target_kind == "other-project"
        else tmp_path / "outside"
    )
    target.mkdir(parents=True)
    alpha_scope = root / "memory" / "projects" / alpha_record.project_id
    alpha_scope.parent.mkdir(parents=True, exist_ok=True)
    try:
        alpha_scope.symlink_to(target, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        registry.resolve_scope_root(ContextCategory.MEMORY, requester_path=alpha)


def test_scoped_path_rejects_link_inside_authorized_root(tmp_path: Path) -> None:
    registry, root = _registry(tmp_path)
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    alpha_root = root / "memory" / "projects" / alpha_record.project_id
    beta_notes = root / "memory" / "projects" / beta_record.project_id / "notes"
    alpha_root.mkdir(parents=True)
    beta_notes.mkdir(parents=True)
    try:
        (alpha_root / "notes").symlink_to(beta_notes, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        registry.resolve_scoped_path(
            ContextCategory.MEMORY,
            "notes/decision.md",
            requester_path=alpha,
        )


def test_resolve_context_paths_uses_registered_central_v2_without_local_link(tmp_path: Path) -> None:
    root = tmp_path / ".context"
    project = tmp_path / "project"
    nested = project / "src"
    nested.mkdir(parents=True)
    scaffold_v2(root)
    ProjectRegistry(root).register(project)
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=root)))

    class Args:
        path = str(nested)
        context_root = None
        context_dir = None

    project_path, context_path, context_override, context_dir = resolve_context_paths(Args(), manager)

    assert project_path == nested.resolve()
    assert context_path == root.resolve()
    assert context_override == root.resolve()
    assert context_dir is None


def test_unregistered_project_does_not_inherit_central_context(tmp_path: Path) -> None:
    root = tmp_path / ".context"
    project = tmp_path / "project"
    project.mkdir()
    scaffold_v2(root)
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=root)))

    class Args:
        path = str(project)
        context_root = None
        context_dir = None

    with pytest.raises(FileNotFoundError):
        resolve_context_paths(Args(), manager)
    assert find_root(project) is None
    assert find_existing_root(project) is None

    record = ProjectRegistry(root).register(project)
    assert record.scope_id.startswith("project:prj_")
    assert find_root(project) == root
    assert find_existing_root(project) == root
