from __future__ import annotations

from pathlib import Path

import pytest

from afs.cli._utils import resolve_context_paths
from afs.context_layout import scaffold_v2
from afs.core import find_existing_root, find_root
from afs.manager import AFSManager
from afs.models import ContextCategory
from afs.project_registry import COMMON_SCOPE_ID, ProjectRegistry, ScopeAuthorizationError
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
