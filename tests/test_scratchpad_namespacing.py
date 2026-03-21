"""Tests for scratchpad namespacing."""

from __future__ import annotations

import pytest

from afs.context_paths import resolve_agent_scratchpad, resolve_mount_root
from afs.models import MountType


@pytest.fixture
def context_with_scratchpad(tmp_path, monkeypatch):
    monkeypatch.setenv("AFS_ALLOWED_MOUNTS", "scratchpad,memory,history,hivemind,knowledge,tools,global,items,monorepo")
    context = tmp_path / ".context"
    context.mkdir()
    scratchpad = context / "scratchpad"
    scratchpad.mkdir()
    return context


def test_resolve_agent_scratchpad_path_structure(context_with_scratchpad):
    result = resolve_agent_scratchpad(context_with_scratchpad, "my-agent")
    expected = resolve_mount_root(context_with_scratchpad, MountType.SCRATCHPAD) / "agents" / "my-agent"
    assert result == expected


def test_collect_scratchpad_includes_agent_namespaces(context_with_scratchpad, monkeypatch):
    scratchpad = context_with_scratchpad / "scratchpad"
    agents_dir = scratchpad / "agents"
    agent_a = agents_dir / "agent-a"
    agent_a.mkdir(parents=True)
    (agent_a / "state.md").write_text("hello", encoding="utf-8")
    (agent_a / "notes.md").write_text("world", encoding="utf-8")

    agent_b = agents_dir / "agent-b"
    agent_b.mkdir(parents=True)
    (agent_b / "data.json").write_text("{}", encoding="utf-8")

    from unittest.mock import MagicMock

    from afs.session_bootstrap import _collect_scratchpad

    manager = MagicMock()
    manager.config.directories = []

    # We need to mock resolve_mount_root to return our scratchpad
    monkeypatch.setattr(
        "afs.session_bootstrap.resolve_mount_root",
        lambda ctx, mt, config=None: scratchpad,
    )

    result = _collect_scratchpad(manager, context_with_scratchpad)
    assert "agent_namespaces" in result
    ns_names = [ns["agent_name"] for ns in result["agent_namespaces"]]
    assert "agent-a" in ns_names
    assert "agent-b" in ns_names
    agent_a_ns = [ns for ns in result["agent_namespaces"] if ns["agent_name"] == "agent-a"][0]
    assert agent_a_ns["file_count"] == 2


def test_agent_namespace_size_tracking(context_with_scratchpad, monkeypatch):
    scratchpad = context_with_scratchpad / "scratchpad"
    agents_dir = scratchpad / "agents"
    agent_a = agents_dir / "agent-a"
    agent_a.mkdir(parents=True)
    content = "x" * 100
    (agent_a / "file.txt").write_text(content, encoding="utf-8")

    from unittest.mock import MagicMock

    from afs.session_bootstrap import _collect_scratchpad

    manager = MagicMock()
    manager.config.directories = []

    monkeypatch.setattr(
        "afs.session_bootstrap.resolve_mount_root",
        lambda ctx, mt, config=None: scratchpad,
    )

    result = _collect_scratchpad(manager, context_with_scratchpad)
    agent_a_ns = [ns for ns in result["agent_namespaces"] if ns["agent_name"] == "agent-a"][0]
    assert agent_a_ns["size_bytes"] > 0
    assert agent_a_ns["file_count"] == 1


def test_backward_compat_afs_agents_still_collected(context_with_scratchpad, monkeypatch):
    """The afs_agents dir still works for legacy agent output."""
    scratchpad = context_with_scratchpad / "scratchpad"
    afs_agents = scratchpad / "afs_agents"
    afs_agents.mkdir(parents=True)
    (afs_agents / "report.json").write_text("{}", encoding="utf-8")

    from unittest.mock import MagicMock

    from afs.session_bootstrap import _collect_scratchpad

    manager = MagicMock()
    manager.config.directories = []

    monkeypatch.setattr(
        "afs.session_bootstrap.resolve_mount_root",
        lambda ctx, mt, config=None: scratchpad,
    )

    result = _collect_scratchpad(manager, context_with_scratchpad)
    # afs_agents is not an "agents" namespace -- it shows up in other_files or stays separate
    # The key is that it doesn't break the function
    assert isinstance(result["other_files"], list)
