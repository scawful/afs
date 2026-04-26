from __future__ import annotations

from pathlib import Path

from afs.agent_runs import AgentRunStore


def test_agent_run_start_event_finish(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    (ctx / "scratchpad").mkdir(parents=True)
    store = AgentRunStore(ctx)

    run = store.start("Patch harnesses", harness="codex", workspace="/tmp/repo", prompt="do it")
    assert run.status == "running"
    assert (ctx / "scratchpad" / "agent_runs" / f"{run.id}.json").exists()

    updated = store.record_event(run.id, "verification", summary="pytest passed")
    assert updated.events[-1]["type"] == "verification"

    done = store.finish(
        run.id,
        summary="finished",
        files_changed=["src/afs/example.py"],
        commands=["pytest tests/test_agent_runs.py"],
        verification=[{"command": "pytest", "status": "passed"}],
        handoff_path="scratchpad/handoffs/run.md",
    )
    assert done.status == "done"
    assert done.files_changed == ["src/afs/example.py"]
    assert done.handoff_path == "scratchpad/handoffs/run.md"
    assert store.list()[0].id == run.id
