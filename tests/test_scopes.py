from pathlib import Path

import pytest

from afs.context_layout import scaffold_v2
from afs.project_registry import ProjectRegistry
from afs.scopes import resolve_scope


def test_v2_scope_requires_registered_requester(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    project = tmp_path / "project"
    unknown = tmp_path / "unknown"
    project.mkdir()
    unknown.mkdir()
    scaffold_v2(context)
    record = ProjectRegistry(context).register(project)

    resolved = resolve_scope(context, requester_path=project)
    assert resolved.scope_id == record.scope_id
    assert resolved.project_name == record.name

    with pytest.raises(PermissionError, match="not registered"):
        resolve_scope(context, requester_path=unknown)


def test_missing_requester_and_v1_are_conservative_common(tmp_path: Path) -> None:
    v2 = tmp_path / "v2"
    scaffold_v2(v2)
    assert resolve_scope(v2).scope_id == "common"

    v1 = tmp_path / "project" / ".context"
    v1.mkdir(parents=True)
    assert resolve_scope(v1, requester_path=v1.parent).scope_id == "common"
