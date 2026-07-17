"""Tests for the agent context bridge (bootstrap, indexing, mixin)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from afs.agent_context import (
    AGENT_CONTEXT_ENV,
    AgentContextSnapshot,
    ContextAwareAgent,
    build_agent_context_snapshot,
    index_agent_output,
    load_agent_context_snapshot,
    write_agent_context_snapshot,
)
from afs.context_index import ContextSQLiteIndex
from afs.context_layout import scaffold_v2
from afs.manager import AFSManager
from afs.models import MountType
from afs.project_registry import ProjectRegistry
from afs.schema import AFSConfig, GeneralConfig
from afs.scopes import resolve_scope

# ---------------------------------------------------------------------------
# Snapshot serialization
# ---------------------------------------------------------------------------

def test_snapshot_roundtrip():
    snap = AgentContextSnapshot(
        agent_name="test-agent",
        built_at="2026-04-04T00:00:00",
        index_summary={"knowledge": 42, "scratchpad": 10},
        index_total=52,
        memory_topics=["topic-a", "topic-b"],
        memory_entry_count=2,
        recent_events=[{"type": "agent_progress", "source": "x"}],
        active_agents=["workspace-analyst"],
        mount_freshness={"knowledge": {"file_count": 42, "freshness_score": 0.9, "stale": False}},
        codebase_summary={"project_root": "/tmp/test", "source_roots": ["src"]},
        context_root="/tmp/test",
    )
    data = snap.to_dict()
    restored = AgentContextSnapshot.from_dict(data)
    assert restored.agent_name == "test-agent"
    assert restored.index_total == 52
    assert restored.memory_topics == ["topic-a", "topic-b"]
    assert restored.active_agents == ["workspace-analyst"]
    assert restored.mount_freshness["knowledge"]["file_count"] == 42
    assert restored.codebase_summary["source_roots"] == ["src"]


def test_snapshot_write_and_load(tmp_path: Path, monkeypatch):
    snap = AgentContextSnapshot(agent_name="test", built_at="now", context_root=str(tmp_path))
    path = write_agent_context_snapshot(snap, tmp_path / "snapshots")
    assert path.exists()

    monkeypatch.setenv(AGENT_CONTEXT_ENV, str(path))
    loaded = load_agent_context_snapshot()
    assert loaded is not None
    assert loaded.agent_name == "test"


@pytest.mark.parametrize("linked_component", ["root", "leaf"])
def test_managed_snapshot_writer_rejects_linked_output(
    tmp_path: Path,
    linked_component: str,
) -> None:
    state_root = tmp_path / "supervisor"
    state_root.mkdir()
    output_dir = state_root / "context_snapshots"
    outside = tmp_path / "outside"
    outside.mkdir()
    poison = outside / "agent_context_test.json"
    poison.write_text("do-not-overwrite", encoding="utf-8")
    try:
        if linked_component == "root":
            output_dir.symlink_to(outside, target_is_directory=True)
        else:
            output_dir.mkdir()
            (output_dir / "agent_context_test.json").symlink_to(poison)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        write_agent_context_snapshot(
            AgentContextSnapshot(agent_name="test"),
            output_dir,
            trusted_root=state_root,
        )

    assert poison.read_text(encoding="utf-8") == "do-not-overwrite"


def test_load_snapshot_returns_none_when_no_env(monkeypatch):
    monkeypatch.delenv(AGENT_CONTEXT_ENV, raising=False)
    assert load_agent_context_snapshot() is None


def test_load_snapshot_returns_none_for_missing_file(monkeypatch):
    monkeypatch.setenv(AGENT_CONTEXT_ENV, "/nonexistent/path.json")
    assert load_agent_context_snapshot() is None


# ---------------------------------------------------------------------------
# Build snapshot with real context
# ---------------------------------------------------------------------------

def test_build_snapshot_with_minimal_context(tmp_path: Path):
    context = tmp_path / ".context"
    context.mkdir()
    (context / "knowledge").mkdir()
    (context / "scratchpad").mkdir()
    (context / "memory").mkdir()
    # Write a memory entry
    (context / "memory" / "test-topic.md").write_text("# Test\nSome memory.", encoding="utf-8")

    snap = build_agent_context_snapshot("test-agent", context)
    assert snap.agent_name == "test-agent"
    assert snap.context_root == str(context)
    assert snap.memory_entry_count >= 1
    assert "test-topic" in snap.memory_topics


def test_build_snapshot_includes_codebase_summary(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    context = project / ".context"
    context.mkdir()
    (context / "memory").mkdir()
    (context / "memory" / "test-topic.md").write_text("# Test\nSome memory.", encoding="utf-8")
    (project / "README.md").write_text("# Demo\n", encoding="utf-8")
    (project / "AGENTS.md").write_text("# Agents\n", encoding="utf-8")
    (project / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (project / "src").mkdir()
    (project / "src" / "demo.py").write_text("def demo() -> int:\n    return 1\n", encoding="utf-8")
    (project / "tests").mkdir()
    (project / "tests" / "test_demo.py").write_text("def test_demo():\n    assert True\n", encoding="utf-8")
    (project / "docs").mkdir()
    (project / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")

    snap = build_agent_context_snapshot("test-agent", context)

    assert snap.codebase_summary["project_root"] == str(project)
    assert "pyproject.toml" in snap.codebase_summary["manifests"]
    assert "src" in snap.codebase_summary["source_roots"]
    assert "tests" in snap.codebase_summary["test_roots"]
    assert "docs" in snap.codebase_summary["docs_roots"]
    assert snap.codebase_summary["language_hints"]["python"] >= 2


def test_build_snapshot_supports_training_repo_layout(tmp_path: Path):
    project = tmp_path / "scawfulbot"
    project.mkdir()
    context = project / ".context"
    context.mkdir()
    (context / "memory").mkdir()
    (project / "config").mkdir()
    (project / "config" / "registry.json").write_text('{"models": []}\n', encoding="utf-8")
    (project / "config" / "system_prompt.md").write_text("# Prompt\n", encoding="utf-8")
    (project / "data").mkdir()
    (project / "eval").mkdir()
    (project / "eval" / "smoke.py").write_text("def smoke() -> bool:\n    return True\n", encoding="utf-8")
    (project / "models").mkdir()
    (project / "scripts").mkdir()
    (project / "scripts" / "train.py").write_text("def train() -> None:\n    pass\n", encoding="utf-8")
    (project / "training").mkdir()
    (project / "training" / "dataset.py").write_text("def load_dataset() -> list[str]:\n    return []\n", encoding="utf-8")

    snap = build_agent_context_snapshot("test-agent", context)

    assert "config" in snap.codebase_summary["workflow_roots"]
    assert "eval" in snap.codebase_summary["workflow_roots"]
    assert "models" in snap.codebase_summary["workflow_roots"]
    assert "training" in snap.codebase_summary["workflow_roots"]
    assert "scripts" in snap.codebase_summary["script_roots"]
    assert snap.codebase_summary["language_hints"]["python"] >= 2


def test_build_snapshot_handles_missing_context(tmp_path: Path):
    snap = build_agent_context_snapshot("test", tmp_path / "nonexistent")
    assert snap.agent_name == "test"
    assert snap.index_total == 0


# ---------------------------------------------------------------------------
# Auto-indexing agent output
# ---------------------------------------------------------------------------

def test_index_agent_output_does_not_crash(tmp_path: Path):
    """index_agent_output should never raise, regardless of context state."""
    result_path = tmp_path / "result.json"
    result_path.write_text('{"status": "ok"}', encoding="utf-8")
    # Should return bool without crashing
    result = index_agent_output(result_path, "test-agent", {"status": "ok"})
    assert isinstance(result, bool)


def test_index_agent_output_uses_v2_mount_relative_scope_and_visibility(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context)
    registry = ProjectRegistry(context)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    result_path = (
        context
        / "scratchpad"
        / "projects"
        / alpha_record.project_id
        / "afs_agents"
        / "result.json"
    )
    result_path.parent.mkdir(parents=True)
    result_path.write_text('{"status": "ok"}\n', encoding="utf-8")
    config = AFSConfig(general=GeneralConfig(context_root=context))
    manager = AFSManager(config=config)
    knowledge_path = (
        context
        / "knowledge"
        / "projects"
        / alpha_record.project_id
        / "afs_agents"
        / "result.json"
    )
    knowledge_path.parent.mkdir(parents=True)
    knowledge_path.write_text("knowledge-cross-mount-canary\n", encoding="utf-8")
    ContextSQLiteIndex(manager, context).rebuild(
        mount_types=[MountType.KNOWLEDGE],
    )
    monkeypatch.setenv("AFS_CONTEXT_ROOT", str(context))
    monkeypatch.setattr("afs.config.load_config_model", lambda **_kwargs: config)

    assert index_agent_output(
        result_path,
        "test-agent",
        {"status": "ok", "notes": ["alpha-private-canary"]},
    ) is True

    index = ContextSQLiteIndex(manager, context)
    alpha_hits = index.query(
        query="alpha-private-canary",
        mount_types=[MountType.SCRATCHPAD],
        relative_prefix=f"projects/{alpha_record.project_id}",
        include_content=True,
    )
    beta_hits = index.query(
        query="alpha-private-canary",
        mount_types=[MountType.SCRATCHPAD],
        relative_prefix=f"projects/{beta_record.project_id}",
        include_content=True,
    )
    assert [hit["relative_path"] for hit in alpha_hits] == [
        f"projects/{alpha_record.project_id}/afs_agents/result.json"
    ]
    assert beta_hits == []
    assert len(
        index.query(
            query="knowledge-cross-mount-canary",
            mount_types=[MountType.KNOWLEDGE],
            relative_prefix=f"projects/{alpha_record.project_id}",
        )
    ) == 1


def test_index_agent_output_rejects_v2_external_raw_and_linked_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = tmp_path / ".context"
    outside = tmp_path / "outside.json"
    scaffold_v2(context)
    config = AFSConfig(general=GeneralConfig(context_root=context))
    monkeypatch.setenv("AFS_CONTEXT_ROOT", str(context))
    monkeypatch.setattr("afs.config.load_config_model", lambda **_kwargs: config)
    outside.write_text("outside\n", encoding="utf-8")
    raw = context / "scratchpad" / "afs_agents" / "raw.json"
    raw.parent.mkdir()
    raw.write_text("raw\n", encoding="utf-8")
    common = context / "scratchpad" / "common" / "afs_agents"
    common.mkdir(parents=True)
    linked = common / "linked.json"
    linked.symlink_to(outside)

    for path in (outside, raw, linked):
        assert index_agent_output(path, "test-agent", {"status": "ok"}) is False

    index = ContextSQLiteIndex(AFSManager(config=config), context)
    assert index.query(mount_types=[MountType.SCRATCHPAD]) == []


# ---------------------------------------------------------------------------
# ContextAwareAgent mixin
# ---------------------------------------------------------------------------

class TestMixin(ContextAwareAgent):
    pass


def test_mixin_load_context_without_env(monkeypatch):
    monkeypatch.delenv(AGENT_CONTEXT_ENV, raising=False)
    monkeypatch.delenv("AFS_CONTEXT_ROOT", raising=False)
    agent = TestMixin()
    result = agent.load_context()
    assert result is None


def test_mixin_load_context_with_snapshot(tmp_path: Path, monkeypatch):
    snap = AgentContextSnapshot(
        agent_name="test",
        context_root=str(tmp_path),
        index_total=100,
        memory_topics=["topic-a"],
    )
    path = write_agent_context_snapshot(snap, tmp_path / "snapshots")
    monkeypatch.setenv(AGENT_CONTEXT_ENV, str(path))

    agent = TestMixin()
    loaded = agent.load_context()
    assert loaded is not None
    assert loaded.index_total == 100
    assert agent._context_path == tmp_path


def test_mixin_search_memory(tmp_path: Path, monkeypatch):
    # Set up memory directory
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "deployment-notes.md").write_text("# Deployment\nRan deploy at 3pm.", encoding="utf-8")
    (memory_dir / "training-log.md").write_text("# Training\nTrained model v2.", encoding="utf-8")

    snap = AgentContextSnapshot(agent_name="test", context_root=str(tmp_path))
    path = write_agent_context_snapshot(snap, tmp_path / "snapshots")
    monkeypatch.setenv(AGENT_CONTEXT_ENV, str(path))

    agent = TestMixin()
    agent.load_context()

    results = agent.search_memory("deploy")
    assert len(results) == 1
    assert "deployment-notes" in results[0]["name"]


def test_mixin_search_memory_empty(tmp_path: Path, monkeypatch):
    snap = AgentContextSnapshot(agent_name="test", context_root=str(tmp_path))
    path = write_agent_context_snapshot(snap, tmp_path / "snapshots")
    monkeypatch.setenv(AGENT_CONTEXT_ENV, str(path))

    agent = TestMixin()
    agent.load_context()

    results = agent.search_memory("nonexistent")
    assert results == []


def test_mixin_get_mount_freshness_from_snapshot(tmp_path: Path, monkeypatch):
    snap = AgentContextSnapshot(
        agent_name="test",
        context_root=str(tmp_path),
        mount_freshness={
            "knowledge": {"file_count": 50, "freshness_score": 0.85, "stale": False},
            "scratchpad": {"file_count": 10, "freshness_score": 0.3, "stale": True},
        },
    )
    path = write_agent_context_snapshot(snap, tmp_path / "snapshots")
    monkeypatch.setenv(AGENT_CONTEXT_ENV, str(path))

    agent = TestMixin()
    agent.load_context()

    freshness = agent.get_mount_freshness()
    assert freshness["knowledge"]["stale"] is False
    assert freshness["scratchpad"]["stale"] is True


def test_mixin_get_codebase_summary_from_snapshot(tmp_path: Path, monkeypatch):
    snap = AgentContextSnapshot(
        agent_name="test",
        context_root=str(tmp_path),
        codebase_summary={"project_root": str(tmp_path), "source_roots": ["src"]},
    )
    path = write_agent_context_snapshot(snap, tmp_path / "snapshots")
    monkeypatch.setenv(AGENT_CONTEXT_ENV, str(path))

    agent = TestMixin()
    agent.load_context()

    summary = agent.get_codebase_summary()
    assert summary["source_roots"] == ["src"]


def test_mixin_query_context_returns_empty_without_context():
    agent = TestMixin()
    results = agent.query_context("anything")
    assert results == []


def test_v2_agent_context_query_and_snapshot_are_scope_bounded(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    (alpha / "README.md").write_text("# Alpha\n", encoding="utf-8")
    scaffold_v2(context)
    registry = ProjectRegistry(context)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    files = {
        "alpha": context
        / "knowledge"
        / "projects"
        / alpha_record.project_id
        / "alpha.md",
        "beta": context
        / "knowledge"
        / "projects"
        / beta_record.project_id
        / "beta.md",
        "common": context / "knowledge" / "common" / "common.md",
    }
    for name, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"scope-token {name}-marker", encoding="utf-8")

    config = AFSConfig(general=GeneralConfig(context_root=context))
    index = ContextSQLiteIndex(AFSManager(config=config), context)
    index.rebuild(mount_types=[MountType.KNOWLEDGE], include_content=True)
    alpha_scope = resolve_scope(context, requester_path=alpha)
    snapshot = build_agent_context_snapshot(
        "alpha-agent",
        context,
        config=config,
        requester_path=alpha,
    )
    assert snapshot.scope_id == alpha_record.scope_id
    assert snapshot.project_id == alpha_record.project_id
    assert snapshot.codebase_summary["project_root"] == str(alpha)
    assert snapshot.index_total == index.count_entries_scoped(alpha_scope)

    agent = TestMixin()
    agent.context_snapshot = snapshot
    agent._context_path = context
    agent._config = config
    results = agent.query_context("scope-token", include_content=True)
    rendered = json.dumps(results)
    assert "alpha-marker" in rendered
    assert "common-marker" in rendered
    assert "beta-marker" not in rendered

    unregistered = TestMixin()
    unregistered.context_snapshot = AgentContextSnapshot(
        context_root=str(context),
        scope_id="project:not-real",
        requester_path=str(tmp_path / "unregistered"),
    )
    unregistered._context_path = context
    unregistered._config = config
    common_results = unregistered.query_context("scope-token", include_content=True)
    common_rendered = json.dumps(common_results)
    assert "common-marker" in common_rendered
    assert "alpha-marker" not in common_rendered
    assert "beta-marker" not in common_rendered
    assert unregistered.get_codebase_summary() == {}

    common = TestMixin()
    common.context_snapshot = AgentContextSnapshot(
        context_root=str(context),
        scope_id="common",
    )
    common._context_path = context
    common._config = config
    assert common.get_codebase_summary() == {}


def test_v2_agent_snapshot_prunes_nested_project_and_visible_context(
    tmp_path: Path,
) -> None:
    context = tmp_path / "workspace" / "central-context"
    alpha = context.parent
    beta = alpha / "nested-beta"
    alpha.mkdir(parents=True)
    beta.mkdir()
    scaffold_v2(context)
    registry = ProjectRegistry(context)
    registry.register(alpha)
    registry.register(beta)
    (alpha / "alpha_safe.py").write_text("ALPHA_SAFE = True\n", encoding="utf-8")
    (beta / "beta_confidential_canary.py").write_text(
        "BETA_PRIVATE = True\n",
        encoding="utf-8",
    )
    config = AFSConfig(general=GeneralConfig(context_root=context))

    snapshot = build_agent_context_snapshot(
        "alpha-agent",
        context,
        config=config,
        requester_path=alpha,
    )
    rendered = json.dumps(snapshot.codebase_summary)

    assert "alpha_safe.py" in rendered
    assert "nested-beta" not in rendered
    assert "beta_confidential_canary" not in rendered
    assert "central-context" not in rendered

    file_snapshot = build_agent_context_snapshot(
        "alpha-file-agent",
        context,
        config=config,
        requester_path=alpha / "alpha_safe.py",
    )
    file_rendered = json.dumps(file_snapshot.codebase_summary)
    assert file_snapshot.codebase_summary["project_root"] == str(alpha)
    assert "alpha_safe.py" in file_rendered
    assert "nested-beta" not in file_rendered
    assert "beta_confidential_canary" not in file_rendered
    assert "central-context" not in file_rendered


def test_v2_agent_snapshot_keeps_registered_dot_context_as_exact_project_root(
    tmp_path: Path,
) -> None:
    central = tmp_path / "central" / ".context"
    project_parent = tmp_path / "owner"
    project = project_parent / ".context"
    project.mkdir(parents=True)
    scaffold_v2(central)
    record = ProjectRegistry(central).register(project)
    (project / "inside_project.py").write_text("INSIDE = True\n", encoding="utf-8")
    (project_parent / "outside_confidential_canary.py").write_text(
        "OUTSIDE = True\n",
        encoding="utf-8",
    )
    config = AFSConfig(general=GeneralConfig(context_root=central))

    snapshot = build_agent_context_snapshot(
        "dot-context-agent",
        central,
        config=config,
        requester_path=project,
    )
    rendered = json.dumps(snapshot.codebase_summary)

    assert snapshot.project_id == record.project_id
    assert snapshot.codebase_summary["project_root"] == str(project)
    assert "inside_project.py" in rendered
    assert "outside_confidential_canary" not in rendered
