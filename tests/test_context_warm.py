from __future__ import annotations

from pathlib import Path

from afs.agents import context_warm as agent
from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import AFSConfig, GeneralConfig, ProfileConfig, ProfilesConfig


def _build_config(tmp_path: Path) -> AFSConfig:
    context_root = tmp_path / "context"
    return AFSConfig(
        general=GeneralConfig(
            context_root=context_root,
            agent_workspaces_dir=context_root / "workspaces",
        ),
        profiles=ProfilesConfig(
            active_profile="work",
            auto_apply=True,
            profiles={"work": ProfileConfig(knowledge_mounts=[tmp_path / "knowledge-src"])},
        ),
    )


def _clear_profile_env(monkeypatch) -> None:  # noqa: ANN001
    for name in (
        "AFS_PROFILE",
        "AFS_ENABLED_EXTENSIONS",
        "AFS_KNOWLEDGE_MOUNTS",
        "AFS_SKILL_ROOTS",
        "AFS_MODEL_REGISTRIES",
        "AFS_POLICIES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_audit_contexts_repairs_profile_mounts(tmp_path: Path, monkeypatch) -> None:
    _clear_profile_env(monkeypatch)
    knowledge_src = tmp_path / "knowledge-src"
    knowledge_src.mkdir()
    config = _build_config(tmp_path)
    manager = AFSManager(config=config)

    project_path = tmp_path / "project"
    project_path.mkdir()
    context = manager.ensure(path=project_path, context_root=config.general.context_root, profile="work")
    managed_mount = context.get_mounts(MountType.KNOWLEDGE)[0]
    manager.unmount(managed_mount.name, MountType.KNOWLEDGE, context_path=context.path)

    audits, metrics, _notes = agent._audit_contexts(
        config,
        [context.path],
        repair_profile_mounts=True,
        rebuild_stale_indexes=False,
    )

    assert metrics["contexts_audited"] == 1
    assert metrics["profile_repairs"] == 1
    assert audits[0]["repair"]["profile_reapply"]["profile"] == "work"
    assert audits[0]["mount_health"]["healthy"] is True


def test_audit_contexts_rebuilds_stale_indexes(tmp_path: Path, monkeypatch) -> None:
    _clear_profile_env(monkeypatch)
    knowledge_src = tmp_path / "knowledge-src"
    knowledge_src.mkdir()
    config = _build_config(tmp_path)
    manager = AFSManager(config=config)

    project_path = tmp_path / "project"
    project_path.mkdir()
    context = manager.ensure(path=project_path, context_root=config.general.context_root, profile="work")
    (context.path / "scratchpad" / "notes.md").write_text("warm audit", encoding="utf-8")

    audits, metrics, _notes = agent._audit_contexts(
        config,
        [context.path],
        rebuild_stale_indexes=True,
    )

    assert metrics["contexts_audited"] == 1
    assert metrics["indexes_rebuilt"] == 1
    assert audits[0]["index"]["has_entries"] is True
    assert audits[0]["index"]["stale"] is False


def test_resolve_audit_context_paths_filters_and_limits(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    selected = agent._resolve_audit_context_paths(
        config,
        [
            {"path": str(tmp_path / "alpha" / ".context")},
            {"path": str(tmp_path / "beta" / ".context")},
            {"path": str(tmp_path / "gamma" / ".context")},
        ],
        filters=["beta"],
        max_contexts=1,
    )

    assert selected == []

    for name in ("alpha", "beta", "gamma"):
        (tmp_path / name / ".context").mkdir(parents=True, exist_ok=True)

    selected = agent._resolve_audit_context_paths(
        config,
        [
            {"path": str(tmp_path / "alpha" / ".context")},
            {"path": str(tmp_path / "beta" / ".context")},
            {"path": str(tmp_path / "gamma" / ".context")},
        ],
        filters=["beta", "gamma"],
        max_contexts=1,
    )

    assert len(selected) == 1
    assert "beta" in str(selected[0])


def test_context_watch_roots_include_context_and_mount_sources(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    (tmp_path / "knowledge-src").mkdir()
    manager = AFSManager(config=config)
    project_path = tmp_path / "project"
    project_path.mkdir()
    context = manager.ensure(path=project_path, context_root=config.general.context_root, profile="work")

    watch_roots = agent._context_watch_roots(config, [context.path])

    roots = watch_roots[context.path]
    assert context.path in roots
    assert any(root.name == "knowledge-src" for root in roots)
