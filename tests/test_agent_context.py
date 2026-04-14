"""Tests for the agent context bridge (bootstrap, indexing, mixin)."""

from __future__ import annotations

import json
import os
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
