"""Tests for AgentSupervisor lifecycle management."""

from __future__ import annotations

import json
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from afs.agent_registry import (
    AGENT_CONTEXT_ROOT_ENV,
    AGENT_EXPECTED_RESULT_NAME_ENV,
    AGENT_RUN_ID_ENV,
    AGENT_SUPERVISED_ENV,
    AgentRegistry,
)
from afs.agents.base import now_iso
from afs.agents.supervisor import (
    DEFAULT_FAILURE_HISTORY_SECONDS,
    AgentSupervisor,
    RunningAgent,
    _watch_signature,
)
from afs.context_layout import scaffold_v2
from afs.event_log import read_agent_events
from afs.project_registry import ProjectRegistry
from afs.schema import (
    AFSConfig,
    AgentConfig,
    DirectoryConfig,
    GeneralConfig,
    ProfileConfig,
    ProfilesConfig,
    default_directory_configs,
)


@pytest.fixture(autouse=True)
def _isolated_agent_registry(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AFS_AGENT_REGISTRY_PATH", str(tmp_path / "agent_registry.json"))


def _remap_directories(**overrides: str) -> list[DirectoryConfig]:
    directories: list[DirectoryConfig] = []
    for directory in default_directory_configs():
        name = (
            overrides.get(directory.role.value, directory.name)
            if directory.role
            else directory.name
        )
        directories.append(
            DirectoryConfig(
                name=name,
                policy=directory.policy,
                description=directory.description,
                role=directory.role,
            )
        )
    return directories


def test_agent_timestamps_are_timezone_aware_utc() -> None:
    timestamp = datetime.fromisoformat(now_iso())

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)


def test_registry_completion_accepts_builtin_agent_timestamp(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    supervisor = AgentSupervisor(
        state_dir=tmp_path / "state",
        config=AFSConfig(general=GeneralConfig(context_root=context_root)),
    )
    started_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    agent = RunningAgent(name="timestamped", started_at=started_at)
    finished_at = now_iso()
    AgentRegistry().mark_result(
        name=agent.name,
        status="ok",
        started_at=started_at,
        finished_at=finished_at,
    )

    completion = supervisor._registry_completion(agent)

    assert completion is not None
    assert completion["finished_at"] == finished_at


def test_supervisor_list_empty(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    assert supervisor.list_running() == []


def test_supervisor_status_unknown(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    assert supervisor.status("nonexistent") is None


def test_supervisor_stop_unknown(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    assert supervisor.stop("nonexistent") is False


def test_supervisor_spawn_and_status(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AFS_SESSION_ID", "sess-123")
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(
        state_dir=state_dir,
        config=AFSConfig(general=GeneralConfig(context_root=tmp_path / "context")),
    )

    # Mock Popen to avoid actually launching a process
    mock_proc = type("MockProc", (), {"pid": 99999})()
    with patch("subprocess.Popen", return_value=mock_proc):
        agent = supervisor.spawn("test-agent", "afs.agents.context_warm")

    assert agent.name == "test-agent"
    assert agent.pid == 99999
    assert agent.state == "running"
    assert agent.module == "afs.agents.context_warm"
    assert agent.session_id == "sess-123"
    assert agent.launch_reason == ""

    # State file should exist
    assert (state_dir / "test-agent.json").exists()

    # Status should reflect running (mock pid_alive)
    with patch.object(supervisor, "_pid_alive", return_value=True):
        status = supervisor.status("test-agent")
    assert status is not None
    assert status.state == "running"
    registry_path = tmp_path / "agent_registry.json"
    assert registry_path.exists()
    entries = json.loads(registry_path.read_text(encoding="utf-8"))
    assert entries[0]["name"] == "test-agent"
    assert entries[0]["status"] == "running"
    assert entries[0]["task"].startswith("Sync workspace paths")
    assert entries[0]["metadata"]["session_id"] == "sess-123"


def test_supervisor_injects_explicit_config_path_into_child_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("AFS_CONFIG_PATH", raising=False)
    config_path = tmp_path / "explicit-afs.toml"
    config_path.write_text("[extensions]\nauto_discover = false\n", encoding="utf-8")
    supervisor = AgentSupervisor(
        state_dir=tmp_path / "state",
        config=AFSConfig(general=GeneralConfig(context_root=tmp_path / "context")),
        config_path=config_path,
    )
    mock_proc = type("MockProc", (), {"pid": 99999})()

    with patch("subprocess.Popen", return_value=mock_proc) as popen:
        agent = supervisor.spawn("configured-agent", "pkg.agent")

    child_env = popen.call_args.kwargs["env"]
    assert child_env["AFS_CONFIG_PATH"] == str(config_path.resolve())
    assert child_env[AGENT_CONTEXT_ROOT_ENV] == str((tmp_path / "context").resolve())
    assert child_env[AGENT_EXPECTED_RESULT_NAME_ENV] == "configured-agent"
    assert child_env[AGENT_RUN_ID_ENV] == agent.run_id
    assert child_env[AGENT_SUPERVISED_ENV] == "1"
    assert agent.run_id


def test_supervisor_reaps_short_lived_agent_and_records_completion(
    tmp_path: Path,
) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir()
    config_path = tmp_path / "afs.toml"
    config_path.write_text(
        f'[general]\ncontext_root = "{context_root}"\n',
        encoding="utf-8",
    )
    config = AFSConfig(general=GeneralConfig(context_root=context_root))
    supervisor = AgentSupervisor(
        state_dir=tmp_path / "state",
        config=config,
        config_path=config_path,
    )

    spawned = supervisor.spawn(
        "custom-briefing",
        "afs.agents.briefing_agent",
        agent_config=AgentConfig(
            name="custom-briefing",
            module="afs.agents.briefing_agent",
        ),
    )
    deadline = time.monotonic() + 5.0
    completed = supervisor.status(spawned.name)
    while (
        completed is not None
        and completed.state == "running"
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
        completed = supervisor.status(spawned.name)

    assert completed is not None
    assert completed.state == "stopped"
    assert completed.pid is None
    assert completed.run_id == spawned.run_id
    registry_entry = AgentRegistry().get(
        spawned.name,
        context_root=str(context_root.resolve()),
    )
    assert registry_entry is not None
    assert registry_entry["status"] == "stopped"
    assert registry_entry["run_id"] == spawned.run_id
    assert registry_entry["metadata"]["reported_name"] == "morning-briefing"


def test_supervisor_logs_lifecycle_events(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AFS_SESSION_ID", "sess-456")
    context_root = tmp_path / "context"
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(
        state_dir=state_dir,
        config=AFSConfig(general=GeneralConfig(context_root=context_root)),
    )

    mock_proc = type("MockProc", (), {"pid": 12345})()
    with patch("subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("lifecycle-agent", "afs.agents.context_warm", reason="shell_helper")

    events = read_agent_events(context_root, agent_name="lifecycle-agent")
    assert any(event["op"] == "spawned" for event in events)
    spawned = next(event for event in events if event["op"] == "spawned")
    assert spawned["metadata"]["session_id"] == "sess-456"
    assert spawned["metadata"]["scope_attribution"] == "unregistered"


def test_supervisor_lifecycle_uses_only_explicit_configured_scope(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / ".context"
    project = tmp_path / "project"
    unknown = tmp_path / "unknown"
    project.mkdir()
    unknown.mkdir()
    scaffold_v2(context_root)
    project_record = ProjectRegistry(context_root).register(project)
    project_agent = AgentConfig(
        name="project-reflect",
        module="afs.agents.insights",
        extra={"project_path": str(project)},
    )
    common_agent = AgentConfig(
        name="common-reflect",
        module="afs.agents.insights",
        extra={"common": True},
    )
    unknown_agent = AgentConfig(
        name="unknown-reflect",
        module="afs.agents.insights",
        extra={"project_path": str(unknown)},
    )
    unscoped_agent = AgentConfig(
        name="unscoped",
        module="afs.agents.context_warm",
    )
    config = AFSConfig(
        general=GeneralConfig(context_root=context_root),
        profiles=ProfilesConfig(
            active_profile="work",
            profiles={
                "work": ProfileConfig(
                    agent_configs=[
                        project_agent,
                        common_agent,
                        unknown_agent,
                        unscoped_agent,
                    ]
                )
            },
        ),
    )
    supervisor = AgentSupervisor(
        state_dir=tmp_path / "state",
        config=config,
    )
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(
        "afs.history.log_agent_lifecycle",
        lambda _name, _op, **kwargs: captured.append(
            dict(kwargs.get("metadata") or {})
        ),
    )
    for agent in (
        project_agent,
        common_agent,
        unknown_agent,
        unscoped_agent,
    ):
        supervisor._log_lifecycle(
            agent.name,
            "failed",
            metadata={
                "status": "failed",
                "scope_id": "forged",
                "scope_attribution": "registry",
            },
        )

    assert captured[0]["scope_attribution"] == "registry"
    assert captured[0]["scope_id"] == project_record.scope_id
    assert captured[0]["project_id"] == project_record.project_id
    assert captured[1]["scope_attribution"] == "common"
    assert captured[1]["scope_id"] == "common"
    assert captured[2]["scope_attribution"] == "unregistered"
    assert "scope_id" not in captured[2]
    assert captured[3]["scope_attribution"] == "unregistered"
    assert "scope_id" not in captured[3]


def test_supervisor_spawn_and_stop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(
        state_dir=state_dir,
        config=AFSConfig(general=GeneralConfig(context_root=tmp_path / "context")),
    )

    mock_proc = type("MockProc", (), {"pid": 88888})()
    with patch("subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("stop-test", "some.module")

    def kill_side_effect(pid, sig):
        if sig == 0 and kill_side_effect.alive:
            return None  # pid_alive: process exists
        if sig == signal.SIGTERM:
            kill_side_effect.alive = False
            return None  # SIGTERM sent
        raise ProcessLookupError("no such process")

    kill_side_effect.alive = True
    with patch("os.kill", side_effect=kill_side_effect):
        stopped = supervisor.stop("stop-test")

    assert stopped is True
    status = supervisor.status("stop-test")
    assert status is not None
    assert status.state == "stopped"
    assert status.manually_stopped is True
    entries = json.loads((tmp_path / "agent_registry.json").read_text(encoding="utf-8"))
    assert entries[0]["status"] == "stopped"


def test_supervisor_list_detects_dead_pid(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)

    mock_proc = type("MockProc", (), {"pid": 77777})()
    with patch("subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("dead-agent", "some.module")

    # Simulate dead PID — list_agents refreshes state, marking it failed
    with patch.object(supervisor, "_pid_alive", return_value=False):
        agents = supervisor.list_agents()

    failed = [a for a in agents if a.state == "failed"]
    assert len(failed) == 1
    assert failed[0].name == "dead-agent"
    # list_running should be empty since dead agents aren't "running"
    with patch.object(supervisor, "_pid_alive", return_value=False):
        running = supervisor.list_running()
    assert len(running) == 0


def test_scheduled_process_without_completion_record_is_failed(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(
        state_dir=tmp_path / "state",
        config=AFSConfig(general=GeneralConfig(context_root=tmp_path / "context")),
    )
    scheduled = RunningAgent(
        name="weekly-agent",
        pid=12345,
        state="running",
        module="pkg.weekly",
        launch_reason="schedule:weekly",
    )

    with (
        patch.object(supervisor, "_pid_alive", return_value=False),
        patch.object(supervisor, "_registry_completion", return_value=None),
        patch.object(supervisor, "_attempt_restart", return_value=None),
        patch.object(supervisor, "_get_agent_config", return_value=None),
    ):
        refreshed = supervisor._refresh_state(scheduled)

    assert refreshed.state == "failed"
    assert refreshed.pid is None
    assert "completion record" in refreshed.last_error


def test_supervisor_uses_registry_completion_for_clean_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    context_root = tmp_path / "context"
    supervisor = AgentSupervisor(
        state_dir=tmp_path / "state",
        config=AFSConfig(general=GeneralConfig(context_root=context_root)),
    )

    mock_proc = type("MockProc", (), {"pid": 77777})()
    with patch("subprocess.Popen", return_value=mock_proc):
        spawned = supervisor.spawn("clean-exit", "afs.agents.context_warm")

    AgentRegistry().mark_result(
        name="clean-exit",
        status="ok",
        task="Sync workspace paths, discover contexts, and refresh embeddings.",
        context_root=str(context_root.resolve()),
        run_id=spawned.run_id,
        started_at=spawned.started_at,
        finished_at=(datetime.fromisoformat(spawned.started_at) + timedelta(seconds=10)).isoformat(),
    )

    with patch.object(supervisor, "_pid_alive", return_value=False):
        status = supervisor.status("clean-exit")

    assert status is not None
    assert status.state == "stopped"


def test_supervisor_auto_start(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)

    configs = [
        AgentConfig(name="auto1", auto_start=True, module="afs.agents.context_warm"),
        AgentConfig(name="skip1", auto_start=False, module="afs.agents.context_warm"),
        AgentConfig(name="no-module", auto_start=True, module=""),
    ]

    mock_proc = type("MockProc", (), {"pid": 11111})()
    with patch("subprocess.Popen", return_value=mock_proc):
        started = supervisor.auto_start(configs)

    assert len(started) == 1
    assert started[0].name == "auto1"


def test_supervisor_evaluate_triggers(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    configs = [
        AgentConfig(name="a1", triggers=["on_mount", "on_profile_switch"]),
        AgentConfig(name="a2", triggers=["file_watch"]),
        AgentConfig(name="a3", triggers=[]),
    ]
    matched = supervisor.evaluate_triggers_from("on_mount", configs)
    assert len(matched) == 1
    assert matched[0].name == "a1"


def test_supervisor_state_persistence(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"

    # Create supervisor and spawn
    sup1 = AgentSupervisor(state_dir=state_dir)
    mock_proc = type("MockProc", (), {"pid": 55555})()
    with patch("subprocess.Popen", return_value=mock_proc):
        sup1.spawn("persist-test", "some.module")

    # New supervisor instance reads state from disk
    sup2 = AgentSupervisor(state_dir=state_dir)
    with patch.object(sup2, "_pid_alive", return_value=True):
        status = sup2.status("persist-test")
    assert status is not None
    assert status.pid == 55555
    assert status.state == "running"


def test_supervisor_uses_context_scoped_state_dir(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    supervisor = AgentSupervisor(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    assert supervisor._state_dir == context_root / "scratchpad" / "afs_agents" / "supervisor"


@pytest.mark.parametrize("linked_component", ["root", "leaf"])
def test_v2_supervisor_rejects_linked_default_state_path(
    tmp_path: Path,
    monkeypatch,
    linked_component: str,
) -> None:
    from afs.context_paths import resolve_agent_output_root

    monkeypatch.delenv("AFS_AGENT_STATE_DIR", raising=False)
    context_root = tmp_path / "context"
    project = tmp_path / "project"
    project.mkdir()
    config = AFSConfig(general=GeneralConfig(context_root=context_root))
    from afs.manager import AFSManager

    AFSManager(config=config).ensure(
        path=project,
        context_root=context_root,
        layout_version=2,
    )
    output_root = resolve_agent_output_root(context_root, config=config, scope_id="common")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir()

    try:
        if linked_component == "root":
            output_root.symlink_to(outside, target_is_directory=True)
        else:
            output_root.mkdir()
            (output_root / "supervisor").symlink_to(
                outside,
                target_is_directory=True,
            )
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        AgentSupervisor(config=config)


@pytest.mark.parametrize("linked_component", ["root", "leaf"])
def test_v2_supervisor_state_io_rejects_links_added_after_initialization(
    tmp_path: Path,
    monkeypatch,
    linked_component: str,
) -> None:
    from afs.manager import AFSManager

    monkeypatch.delenv("AFS_AGENT_STATE_DIR", raising=False)
    context_root = tmp_path / "context"
    project = tmp_path / "project"
    project.mkdir()
    config = AFSConfig(general=GeneralConfig(context_root=context_root))
    AFSManager(config=config).ensure(
        path=project,
        context_root=context_root,
        layout_version=2,
    )
    supervisor = AgentSupervisor(config=config)
    outside = tmp_path / "outside"
    outside.mkdir()
    poison = outside / "poison.json"
    poison.write_text(
        json.dumps(RunningAgent(name="poison", state="running").to_dict()),
        encoding="utf-8",
    )

    try:
        if linked_component == "root":
            supervisor._state_dir.rmdir()
            supervisor._state_dir.symlink_to(outside, target_is_directory=True)
        else:
            (supervisor._state_dir / "poison.json").symlink_to(poison)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    assert supervisor.list_agents() == []
    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        supervisor.status("poison")
    with pytest.raises(ValueError, match="symbolic link|reparse point"):
        supervisor._write_state(RunningAgent(name="poison", state="stopped"))

    persisted = json.loads(poison.read_text(encoding="utf-8"))
    assert persisted["state"] == "running"


def test_v2_supervisor_state_io_round_trip(tmp_path: Path, monkeypatch) -> None:
    from afs.manager import AFSManager

    monkeypatch.delenv("AFS_AGENT_STATE_DIR", raising=False)
    context_root = tmp_path / "context"
    project = tmp_path / "project"
    project.mkdir()
    config = AFSConfig(general=GeneralConfig(context_root=context_root))
    AFSManager(config=config).ensure(
        path=project,
        context_root=context_root,
        layout_version=2,
    )
    supervisor = AgentSupervisor(config=config)
    expected = RunningAgent(name="safe-agent", state="stopped", launch_count=2)

    supervisor._write_state(expected)
    loaded = supervisor._read_state("safe-agent")

    assert loaded is not None
    assert loaded.name == "safe-agent"
    assert loaded.launch_count == 2
    assert not (supervisor._state_dir / "safe-agent.json").is_symlink()


def test_supervisor_rejects_agent_name_path_traversal(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    outside = tmp_path / "outside.json"
    outside.write_text("do-not-read", encoding="utf-8")

    with pytest.raises(ValueError, match="one safe filesystem segment"):
        supervisor.status("../../outside")
    with pytest.raises(ValueError, match="one safe filesystem segment"):
        supervisor.stop("../../outside")

    assert outside.read_text(encoding="utf-8") == "do-not-read"


def test_supervisor_uses_remapped_scratchpad_state_dir(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    supervisor = AgentSupervisor(
        config=AFSConfig(
            general=GeneralConfig(context_root=context_root),
            directories=_remap_directories(scratchpad="notes"),
        )
    )
    assert supervisor._state_dir == context_root / "notes" / "afs_agents" / "supervisor"


def test_supervisor_due_schedules(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)
    configs = [
        AgentConfig(name="scheduled", module="pkg.agent", schedule="5m"),
        AgentConfig(name="disabled", module="pkg.agent", schedule=""),
    ]
    matched = supervisor.due_schedules(configs)
    assert [config.name for config in matched] == ["scheduled"]


def test_supervisor_watch_path_matching(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    watch_path = tmp_path / "workspace"
    watch_path.mkdir()
    configs = [
        AgentConfig(name="watcher", module="pkg.agent", watch_paths=[watch_path]),
        AgentConfig(name="other", module="pkg.agent", watch_paths=[tmp_path / "other"]),
    ]
    matched = supervisor.evaluate_watch_paths([watch_path / "changed.txt"], configs)
    assert [config.name for config in matched] == ["watcher"]


def test_watch_signature_detects_edit_through_directory_symlink(tmp_path: Path) -> None:
    watch_root = tmp_path / "knowledge"
    target = tmp_path / "mounted-notes"
    watch_root.mkdir()
    target.mkdir()
    note = target / "note.md"
    note.write_text("before", encoding="utf-8")
    try:
        (watch_root / "notes").symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    before = _watch_signature(watch_root)
    note.write_text("after with a different size", encoding="utf-8")
    after = _watch_signature(watch_root)

    assert after != before


def test_watch_signature_ignores_index_excluded_directories(tmp_path: Path) -> None:
    watch_root = tmp_path / "knowledge"
    target = tmp_path / "repo"
    git_dir = target / ".git"
    node_modules = target / "node_modules"
    watch_root.mkdir()
    git_dir.mkdir(parents=True)
    node_modules.mkdir()
    git_head = git_dir / "HEAD"
    dependency = node_modules / "package.js"
    git_head.write_text("before", encoding="utf-8")
    dependency.write_text("before", encoding="utf-8")
    try:
        (watch_root / "repo").symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    before = _watch_signature(watch_root)
    git_head.write_text("after with different size", encoding="utf-8")
    dependency.write_text("after with different size", encoding="utf-8")

    assert _watch_signature(watch_root) == before


def test_watch_signature_handles_directory_symlink_cycle(tmp_path: Path) -> None:
    watch_root = tmp_path / "knowledge"
    target = tmp_path / "mounted-notes"
    watch_root.mkdir()
    target.mkdir()
    (target / "note.md").write_text("note", encoding="utf-8")
    try:
        (watch_root / "notes").symlink_to(target, target_is_directory=True)
        (target / "cycle").symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    signature = _watch_signature(watch_root)

    assert signature[3] == 1
    assert signature[4] is False


def test_watch_signature_cap_is_deterministic(tmp_path: Path) -> None:
    watch_root = tmp_path / "knowledge"
    watch_root.mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (watch_root / name).write_text(name, encoding="utf-8")

    first = _watch_signature(watch_root, max_entries=2)
    second = _watch_signature(watch_root, max_entries=2)

    assert first == second
    assert first[3] == 2
    assert first[4] is True


def test_watch_signature_cap_does_not_materialize_entire_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    watch_root = tmp_path / "knowledge"
    watch_root.mkdir()
    for name in ("a.md", "b.md"):
        (watch_root / name).write_text(name, encoding="utf-8")
    consumed = 0

    def guarded_iterdir(path: Path):
        nonlocal consumed
        assert path == watch_root
        for name in ("a.md", "b.md", "unread.md", "must-not-be-read.md"):
            consumed += 1
            if consumed > 3:
                raise AssertionError("watch scan read past its entry cap")
            yield watch_root / name

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)

    signature = _watch_signature(watch_root, max_entries=2)

    assert consumed == 3
    assert signature[3] == 2
    assert signature[4] is True


def test_watch_signature_stops_at_monotonic_deadline(tmp_path: Path) -> None:
    watch_root = tmp_path / "knowledge"
    watch_root.mkdir()
    (watch_root / "note.md").write_text("note", encoding="utf-8")

    with patch(
        "afs.agents.supervisor.time.monotonic",
        side_effect=(100.0, 100.0, 101.0),
    ):
        signature = _watch_signature(watch_root, timeout_seconds=0.5)

    assert signature[3] == 0
    assert signature[4] is True


def test_supervisor_audit_reports_failed_and_manual_stop(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)

    failed = type("MockProc", (), {"pid": 12345})()
    stopped = type("MockProc", (), {"pid": 23456})()
    with patch("subprocess.Popen", side_effect=[failed, stopped]):
        supervisor.spawn("failed-agent", "pkg.failed")
        supervisor.spawn("stopped-agent", "pkg.stopped")

    with patch.object(supervisor, "_pid_alive", return_value=False):
        supervisor.status("failed-agent")
    with patch.object(supervisor, "_pid_alive", return_value=True):
        supervisor.stop("stopped-agent")
    audit = supervisor.audit()

    assert audit["counts"]["failed"] == 1
    assert audit["counts"]["recent_failed"] == 1
    assert audit["counts"]["historical_failed"] == 0
    assert audit["counts"]["manual_stop"] == 1
    assert "failed-agent" in audit["stale_pid_files"]
    assert "failed-agent" in audit["active_issues"]


def test_supervisor_audit_separates_historical_failures(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)
    supervisor._write_state(
        RunningAgent(
            name="old-failure",
            state="failed",
            started_at="2020-01-01T00:00:00+00:00",
            last_error="process exited",
        )
    )

    audit = supervisor.audit()

    assert audit["counts"]["failed"] == 1
    assert audit["counts"]["recent_failed"] == 0
    assert audit["counts"]["historical_failed"] == 1
    assert audit["active_issues"] == []
    assert audit["historical_failures"] == ["old-failure"]


def test_supervisor_audit_uses_failure_time_for_long_running_agent(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)
    supervisor._write_state(
        RunningAgent(
            name="fresh-failure",
            state="failed",
            started_at="2020-01-01T00:00:00+00:00",
            last_seen_at=datetime.now().astimezone().isoformat(),
            last_error="process exited",
        )
    )

    audit = supervisor.audit()

    assert audit["counts"]["recent_failed"] == 1
    assert audit["counts"]["historical_failed"] == 0
    assert audit["active_issues"] == ["fresh-failure"]


@pytest.mark.parametrize("value", ["invalid", "nan", "inf", "-inf"])
def test_supervisor_audit_rejects_invalid_failure_history_window(
    tmp_path: Path,
    monkeypatch,
    value: str,
) -> None:
    monkeypatch.setenv("AFS_AGENT_FAILURE_HISTORY_SECONDS", value)

    audit = AgentSupervisor(state_dir=tmp_path / "state").audit()

    assert audit["failure_history_seconds"] == float(DEFAULT_FAILURE_HISTORY_SECONDS)


# --- Dependency management tests ---


def test_check_dependencies_no_deps(tmp_path: Path) -> None:
    """Agents with no depends_on or mutex_group always pass."""
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    config = AgentConfig(name="a", module="pkg.a")
    ready, reason = supervisor._check_dependencies("a", config, [config])
    assert ready is True
    assert reason == ""


def test_check_dependencies_depends_on_never_run(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    config = AgentConfig(name="b", module="pkg.b", depends_on=["a"])
    ready, reason = supervisor._check_dependencies("b", config, [config])
    assert ready is False
    assert "has never run" in reason


def test_check_dependencies_depends_on_still_running(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    mock_proc = type("MockProc", (), {"pid": 11111})()
    with patch("subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("dep-agent", "pkg.dep")

    config = AgentConfig(name="child", module="pkg.child", depends_on=["dep-agent"])
    with patch.object(supervisor, "_pid_alive", return_value=True):
        ready, reason = supervisor._check_dependencies("child", config, [config])
    assert ready is False
    assert "still running" in reason


def test_check_dependencies_depends_on_completed(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    mock_proc = type("MockProc", (), {"pid": 22222})()
    with patch("subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("dep-agent", "pkg.dep")

    # Simulate clean stop
    with patch.object(supervisor, "_pid_alive", return_value=True):
        supervisor.stop("dep-agent")

    # Mark as not manually stopped so state is clean "stopped"
    agent = supervisor._read_state("dep-agent")
    agent.manually_stopped = False
    supervisor._write_state(agent)

    config = AgentConfig(name="child", module="pkg.child", depends_on=["dep-agent"])
    ready, reason = supervisor._check_dependencies("child", config, [config])
    assert ready is True
    assert reason == ""


def test_check_dependencies_depends_on_failed(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    mock_proc = type("MockProc", (), {"pid": 33333})()
    with patch("subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("dep-agent", "pkg.dep")

    # Simulate failure
    with patch.object(supervisor, "_pid_alive", return_value=False):
        supervisor.status("dep-agent")

    config = AgentConfig(name="child", module="pkg.child", depends_on=["dep-agent"])
    ready, reason = supervisor._check_dependencies("child", config, [config])
    assert ready is False
    assert "failed" in reason


def test_check_dependencies_mutex_group_blocks(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    mock_proc = type("MockProc", (), {"pid": 44444})()
    with patch("subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("agent-a", "pkg.a")

    configs = [
        AgentConfig(name="agent-a", module="pkg.a", mutex_group="exclusive"),
        AgentConfig(name="agent-b", module="pkg.b", mutex_group="exclusive"),
    ]
    with patch.object(supervisor, "_pid_alive", return_value=True):
        ready, reason = supervisor._check_dependencies("agent-b", configs[1], configs)
    assert ready is False
    assert "mutex group" in reason
    assert "agent-a" in reason


def test_check_dependencies_mutex_group_allows_when_stopped(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    mock_proc = type("MockProc", (), {"pid": 55555})()
    with patch("subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("agent-a", "pkg.a")

    with patch.object(supervisor, "_pid_alive", return_value=True):
        supervisor.stop("agent-a")

    configs = [
        AgentConfig(name="agent-a", module="pkg.a", mutex_group="exclusive"),
        AgentConfig(name="agent-b", module="pkg.b", mutex_group="exclusive"),
    ]
    ready, reason = supervisor._check_dependencies("agent-b", configs[1], configs)
    assert ready is True


def test_reconcile_skips_unmet_dependencies(tmp_path: Path) -> None:
    """reconcile() should not spawn an agent whose dependencies aren't met."""
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    configs = [
        AgentConfig(name="dep", module="pkg.dep", auto_start=True),
        AgentConfig(name="child", module="pkg.child", auto_start=True, depends_on=["dep"]),
    ]
    mock_proc = type("MockProc", (), {"pid": 66666})()
    with patch("subprocess.Popen", return_value=mock_proc):
        started = supervisor.reconcile(configs)

    # Only "dep" should start, not "child" (because dep hasn't completed yet)
    names = [a.name for a in started]
    assert "dep" in names
    assert "child" not in names


def test_auto_start_skips_unmet_dependencies(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    configs = [
        AgentConfig(name="dep", module="pkg.dep", auto_start=True),
        AgentConfig(name="child", module="pkg.child", auto_start=True, depends_on=["dep"]),
    ]
    mock_proc = type("MockProc", (), {"pid": 77777})()
    with patch("subprocess.Popen", return_value=mock_proc):
        started = supervisor.auto_start(configs)

    names = [a.name for a in started]
    assert "dep" in names
    assert "child" not in names
