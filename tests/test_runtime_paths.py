from __future__ import annotations

from pathlib import Path

from afs.config import load_config_model
from afs.runtime_paths import (
    default_config_root,
    default_context_root,
    default_training_dataset,
    default_workspace_root,
    default_worktrees_root,
)


def test_workspace_root_prefers_override(monkeypatch, tmp_path: Path) -> None:
    configured = tmp_path / "company" / "tools"
    monkeypatch.setenv("AFS_WORKSPACE_ROOT", str(configured))

    assert default_workspace_root(tmp_path) == configured.resolve()


def test_workspace_root_discovers_catalog(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AFS_WORKSPACE_ROOT", raising=False)
    workspace = tmp_path / "workspace"
    nested = workspace / "team" / "project"
    nested.mkdir(parents=True)
    (workspace / "WORKSPACE.toml").write_text("[workspace]\n", encoding="utf-8")

    assert default_workspace_root(nested) == workspace.resolve()


def test_runtime_roots_honor_environment(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config"
    context = tmp_path / "context"
    training = tmp_path / "training"
    monkeypatch.setenv("AFS_CONFIG_HOME", str(config))
    monkeypatch.setenv("AFS_CONTEXT_ROOT", str(context))
    monkeypatch.setenv("AFS_TRAINING_ROOT", str(training))

    assert default_config_root() == config.resolve()
    assert default_context_root() == context.resolve()
    assert (
        default_training_dataset("samples.jsonl")
        == training.resolve() / "datasets" / "samples.jsonl"
    )


def test_worktree_root_is_repo_namespaced(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "source" / "afs"
    configured = tmp_path / "company-worktrees"
    monkeypatch.setenv("AFS_WORKTREES_ROOT", str(configured))

    assert default_worktrees_root(repo) == configured.resolve() / "afs"


def test_config_home_is_used_by_normal_config_loading(monkeypatch, tmp_path: Path) -> None:
    config_root = tmp_path / "company-config" / "afs"
    context_root = tmp_path / "company-context"
    config_root.mkdir(parents=True)
    (config_root / "config.toml").write_text(
        f'[general]\ncontext_root = "{context_root}"\n', encoding="utf-8"
    )
    empty_project = tmp_path / "project"
    empty_project.mkdir()
    monkeypatch.setenv("AFS_CONFIG_HOME", str(config_root))
    monkeypatch.delenv("AFS_CONFIG_PATH", raising=False)

    config = load_config_model(start_dir=empty_project, prefer_local=False)

    assert config.general.context_root == context_root.resolve()
