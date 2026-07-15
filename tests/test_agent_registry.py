from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from afs.agent_registry import (
    AGENT_CONTEXT_ROOT_ENV,
    AGENT_EXPECTED_RESULT_NAME_ENV,
    AGENT_NAME_ENV,
    AGENT_RUN_ID_ENV,
    AGENT_SUPERVISED_ENV,
    AgentRegistry,
    agent_registry_path,
    resolve_agent_task,
)
from afs.agents.base import AgentResult, emit_result


def test_resolve_agent_task_uses_core_agent_description() -> None:
    task = resolve_agent_task("context-warm")
    assert task.startswith("Sync workspace paths")


def test_agent_registry_updates_and_prunes_old_entries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    registry = AgentRegistry()

    stale_time = (datetime.now() - timedelta(days=3)).isoformat()
    registry.path.parent.mkdir(parents=True, exist_ok=True)
    registry.path.write_text(
        json.dumps(
            [
                {
                    "name": "stale-agent",
                    "task": "old",
                    "status": "success",
                    "last_output_at": stale_time,
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    registry.mark_started(
        name="context-warm",
        module="afs.agents.context_warm",
        task="Warm contexts",
        pid=4321,
    )
    registry.mark_result(
        name="context-warm",
        status="ok",
        task="Warm contexts",
        output_path=str(tmp_path / "context_warm.json"),
    )

    entries = registry.entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "context-warm"
    assert entry["status"] == "ok"
    assert entry["task"] == "Warm contexts"
    assert entry["output_path"].endswith("context_warm.json")
    assert "pid" not in entry


def test_emit_result_updates_global_agent_registry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(AGENT_CONTEXT_ROOT_ENV, "/inherited/context")
    monkeypatch.setenv(AGENT_RUN_ID_ENV, "inherited-run")
    monkeypatch.delenv(AGENT_SUPERVISED_ENV, raising=False)
    output_path = tmp_path / "reports" / "context_audit.json"

    now = datetime.now()
    result = AgentResult(
        name="context-audit",
        status="ok",
        started_at=(now - timedelta(seconds=5)).isoformat(),
        finished_at=now.isoformat(),
        duration_seconds=5.0,
    )

    emit_result(result, output_path=output_path, force_stdout=False, pretty=False)

    registry_data = json.loads(agent_registry_path().read_text(encoding="utf-8"))
    assert isinstance(registry_data, list)
    assert registry_data
    entry = registry_data[0]
    assert entry["name"] == "context-audit"
    assert entry["status"] == "ok"
    assert entry["task"].startswith("Audit AFS contexts")
    assert entry["output_path"] == str(output_path.resolve())
    assert "context_root" not in entry
    assert "run_id" not in entry


def test_supervised_results_are_scoped_by_context_and_run(tmp_path: Path) -> None:
    registry = AgentRegistry(tmp_path / "registry.json")
    registry.mark_started(
        name="morning-briefing",
        context_root="/ctx/A",
        run_id="run-a",
    )
    registry.mark_started(
        name="morning-briefing",
        context_root="/ctx/B",
        run_id="run-b",
    )

    registry.mark_result(
        name="morning-briefing",
        status="ok",
        context_root="/ctx/A",
        run_id="run-a",
    )

    assert registry.get("morning-briefing", context_root="/ctx/A")["status"] == "ok"
    assert registry.get("morning-briefing", context_root="/ctx/B")["status"] == "running"


def test_matching_terminal_result_cannot_regress_to_running(tmp_path: Path) -> None:
    registry = AgentRegistry(tmp_path / "registry.json")
    registry.mark_started(
        name="fast-agent",
        context_root="/ctx",
        run_id="run-1",
    )
    registry.mark_result(
        name="fast-agent",
        status="ok",
        context_root="/ctx",
        run_id="run-1",
    )

    registry.update(
        name="fast-agent",
        status="running",
        context_root="/ctx",
        run_id="run-1",
        pid=12345,
    )

    entry = registry.get("fast-agent", context_root="/ctx")
    assert entry["status"] == "ok"
    assert "pid" not in entry


def test_older_active_run_cannot_replace_newer_context_run(tmp_path: Path) -> None:
    registry = AgentRegistry(tmp_path / "registry.json")
    older_started = "2026-07-15T12:00:00+00:00"
    newer_started = "2026-07-15T12:00:01+00:00"
    registry.mark_started(
        name="concurrent-agent",
        context_root="/ctx",
        run_id="newer-run",
        started_at=newer_started,
        pid=200,
    )

    registry.update(
        name="concurrent-agent",
        status="running",
        context_root="/ctx",
        run_id="older-run",
        started_at=older_started,
        pid=100,
    )

    entry = registry.get("concurrent-agent", context_root="/ctx")
    assert entry["run_id"] == "newer-run"
    assert entry["started_at"] == newer_started
    assert entry["pid"] == 200


def test_unsupervised_result_does_not_overwrite_scoped_run(tmp_path: Path) -> None:
    registry = AgentRegistry(tmp_path / "registry.json")
    registry.mark_started(
        name="shared-agent",
        context_root="/ctx",
        run_id="supervised-run",
    )

    registry.mark_result(name="shared-agent", status="ok")

    scoped = registry.get("shared-agent", context_root="/ctx")
    assert scoped["status"] == "running"
    unscoped = [
        entry
        for entry in registry.entries()
        if entry["name"] == "shared-agent" and not entry.get("context_root")
    ]
    assert len(unscoped) == 1
    assert unscoped[0]["status"] == "ok"


def test_inherited_supervisor_scope_rejects_unexpected_result_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(AGENT_SUPERVISED_ENV, "1")
    monkeypatch.setenv(AGENT_NAME_ENV, "profile-alias")
    monkeypatch.setenv(AGENT_EXPECTED_RESULT_NAME_ENV, "canonical-agent")
    monkeypatch.setenv(AGENT_CONTEXT_ROOT_ENV, "/ctx")
    monkeypatch.setenv(AGENT_RUN_ID_ENV, "parent-run")
    registry = AgentRegistry()
    registry.mark_started(
        name="profile-alias",
        context_root="/ctx",
        run_id="parent-run",
    )
    now = datetime.now()

    emit_result(
        AgentResult(
            name="nested-agent",
            status="ok",
            started_at=now.isoformat(),
            finished_at=(now + timedelta(seconds=1)).isoformat(),
            duration_seconds=1.0,
        ),
        output_path=None,
        force_stdout=False,
        pretty=False,
    )

    parent = registry.get("profile-alias", context_root="/ctx")
    assert parent["status"] == "running"
    nested = registry.get("nested-agent")
    assert nested is not None
    assert nested["status"] == "ok"
    assert "context_root" not in nested
    assert "run_id" not in nested
