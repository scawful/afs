from __future__ import annotations

import json
from pathlib import Path

import pytest

from afs.agent_runs import AgentRun, AgentRunStore
from afs.context_layout import scaffold_v2


def _write_run(path: Path, *, run_id: str, task: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            AgentRun(
                id=run_id,
                task=task,
                started_at="2026-07-17T00:00:00+00:00",
                updated_at="2026-07-17T00:00:00+00:00",
            ).to_dict()
        ),
        encoding="utf-8",
    )


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"symlinks unavailable: {exc}")


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


def test_v1_agent_runs_preserve_external_remapped_scratchpad(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    context.mkdir()
    external = tmp_path / "external-scratchpad"
    external.mkdir()
    (context / "metadata.json").write_text(
        json.dumps({"directories": {"scratchpad": str(external)}}),
        encoding="utf-8",
    )

    store = AgentRunStore(context)
    run = store.start("External v1 run")

    assert (external / "agent_runs" / f"{run.id}.json").is_file()
    restored = store.get(run.id)
    assert restored is not None and restored.task == "External v1 run"
    assert [item.id for item in store.list()] == [run.id]


def test_v2_agent_runs_use_common_control_plane_scope(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)

    run = AgentRunStore(context).start("Scoped control-plane record")

    record = context / "scratchpad" / "common" / "agent_runs" / f"{run.id}.json"
    assert record.is_file()
    assert not (context / "scratchpad" / "agent_runs" / f"{run.id}.json").exists()
    assert json.loads(record.read_text(encoding="utf-8"))["id"] == run.id


def test_v2_agent_runs_read_migrated_and_old_legacy_records_without_init_writes(
    tmp_path: Path,
) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    common_root = context / "scratchpad" / "common" / "agent_runs"
    old_root = context / "scratchpad" / "agent_runs"

    store = AgentRunStore(context)
    assert not common_root.exists()
    assert not old_root.exists()
    assert store.list() == []
    assert not common_root.exists()
    assert not old_root.exists()

    _write_run(common_root / "migrated.json", run_id="migrated", task="migrated run")
    _write_run(old_root / "old.json", run_id="old", task="old fallback run")

    migrated = store.get("migrated")
    old = store.get("old")
    assert migrated is not None and migrated.task == "migrated run"
    assert old is not None and old.task == "old fallback run"
    assert {run.id for run in store.list()} == {"migrated", "old"}


def test_agent_run_store_rejects_linked_primary_root(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    outside = tmp_path / "outside-runs"
    outside.mkdir()
    root = context / "scratchpad" / "common" / "agent_runs"
    _symlink_or_skip(root, outside, directory=True)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        AgentRunStore(context)
    assert list(outside.iterdir()) == []


def test_agent_run_store_rejects_linked_legacy_root_on_read(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    outside = tmp_path / "outside-runs"
    outside.mkdir()
    _write_run(outside / "leak.json", run_id="leak", task="outside secret")
    legacy_root = context / "scratchpad" / "agent_runs"
    _symlink_or_skip(legacy_root, outside, directory=True)

    store = AgentRunStore(context)
    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        store.get("leak")


@pytest.mark.parametrize("root_kind", ["primary", "legacy"])
def test_agent_run_store_rejects_linked_record_leaf_on_read(
    tmp_path: Path,
    root_kind: str,
) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    root = (
        context / "scratchpad" / "common" / "agent_runs"
        if root_kind == "primary"
        else context / "scratchpad" / "agent_runs"
    )
    root.mkdir(parents=True)
    outside = tmp_path / f"{root_kind}-outside.json"
    _write_run(outside, run_id="leak", task="outside secret")
    _symlink_or_skip(root / "leak.json", outside)

    store = AgentRunStore(context)
    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        store.get("leak")
    assert json.loads(outside.read_text(encoding="utf-8"))["task"] == "outside secret"


def test_agent_run_store_rejects_linked_record_leaf_on_write(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    root = context / "scratchpad" / "common" / "agent_runs"
    root.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    _write_run(outside, run_id="fixed", task="outside secret")
    _symlink_or_skip(root / "fixed.json", outside)

    store = AgentRunStore(context)
    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        store._write(AgentRun(id="fixed", task="must not escape"))
    assert json.loads(outside.read_text(encoding="utf-8"))["task"] == "outside secret"
    assert not list(root.glob(".fixed.*.tmp"))


def test_agent_run_store_rejects_path_like_record_ids(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    (context / "scratchpad").mkdir(parents=True)
    store = AgentRunStore(context)

    with pytest.raises(ValueError, match="safe identifier"):
        store.get("../outside")
