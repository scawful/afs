from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from afs.agents import tooling_watch as watch
from afs.messages import MessageBus
from afs.tasks import TaskQueue


@pytest.fixture
def delivery(tmp_path: Path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("AFS_"):
            monkeypatch.delenv(key)
    root = tmp_path.resolve()
    context = root / ".context"
    for name in ("items", "scratchpad", "hivemind"):
        (context / name).mkdir(parents=True)
    source = root / "tool.py"
    source.write_text("print('fixture; never execute')\n")
    metadata = {
        "state": "ready",
        "source_path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "consumer": "consumer-agent",
        "verification": "focused fixture passed",
        "evidence": "scratchpad/handoff.md",
        "next_action": "adopt the tested tool",
    }
    queue = TaskQueue(context)
    task = queue.create("Deliver tool", context={"kind": "tooling", "delivery": metadata})
    task = queue.claim(task.id, "producer")
    state = context / "scratchpad/watch/state.json"
    return context, queue, task, state, source


def _messages(context: Path):
    return MessageBus(context, scope_id="common").read(limit=0)


def _update(queue, task, **values):
    task.context["delivery"].update(values)
    queue._write(task)


def test_ready_alert_reaches_consumer_and_codex_once(delivery):
    context, _queue, _task, state, _source = delivery
    assert watch.run_watch(context, state)["alerts"] == 2
    assert {item.to for item in _messages(context)} == {"codex", "consumer-agent"}
    for item in _messages(context):
        assert item.payload["summary"].startswith("Deliver tool: ready; adoption pending. Next: adopt")
        assert item.payload["next"] == "adopt the tested tool"
        assert item.payload["evidence"] == "scratchpad/handoff.md"
    before = state.read_bytes()
    assert watch.run_watch(context, state)["alerts"] == 0
    assert state.read_bytes() == before
    assert len(_messages(context)) == 2


def test_explicit_adoption_requires_matching_installed_artifact(delivery):
    context, queue, task, state, source = delivery
    installed = source.with_name("installed.py")
    installed.write_bytes(source.read_bytes())
    _update(queue, task, state="adopted", installed_path=str(installed))
    assert watch.run_watch(context, state)["alerts"] == 0
    installed.write_text("different installed version\n")
    result = watch.run_watch(context, state)
    assert result["observations"][task.id]["issues"] == ["installation changed"]
    assert result["alerts"] == 2
    installed.write_bytes(source.read_bytes())
    assert watch.run_watch(context, state)["alerts"] == 0


@pytest.mark.parametrize("missing", [False, True])
def test_source_drift_or_absence_is_a_deduplicated_finding(delivery, missing):
    context, queue, task, state, source = delivery
    _update(queue, task, state="adopted")
    assert watch.run_watch(context, state)["alerts"] == 0
    source.unlink() if missing else source.write_text("changed\n")
    result = watch.run_watch(context, state)
    expected = "source missing" if missing else "source changed"
    assert result["observations"][task.id]["issues"] == [expected]
    assert result["alerts"] == 2
    assert watch.run_watch(context, state)["alerts"] == 0


def test_explicit_deferral_and_unproduced_opportunity_are_quiet(delivery):
    context, queue, task, state, source = delivery
    _update(queue, task, state="deferred")
    source.unlink()
    queue.create("Tooling opportunity", context={"kind": "tooling", "next_action": "review audit"})
    result = watch.run_watch(context, state)
    assert result["alerts"] == 0
    assert result["observations"] == {task.id: {"state": "deferred", "issues": []}}


def test_read_failure_preserves_last_good_state_and_alerts_once_then_recovers(delivery, monkeypatch):
    context, _queue, _task, state, _source = delivery
    watch.run_watch(context, state)
    good = state.read_bytes()
    original = watch._collect

    def denied(*_args):
        raise PermissionError("fixture read denied")

    monkeypatch.setattr(watch, "_collect", denied)
    assert watch.run_watch(context, state)["status"] == "error"
    assert state.read_bytes() == good
    assert watch.run_watch(context, state)["alerts"] == 0
    summaries = [item.payload.get("summary") for item in _messages(context)]
    assert summaries.count("Tooling scan failed") == 1
    monkeypatch.setattr(watch, "_collect", original)
    assert watch.run_watch(context, state)["alerts"] == 1
    assert watch.run_watch(context, state)["alerts"] == 0
    summaries = [item.payload.get("summary") for item in _messages(context)]
    assert summaries.count("Tooling scan recovered") == 1


def test_malformed_queue_record_does_not_look_resolved(delivery):
    context, queue, task, state, _source = delivery
    watch.run_watch(context, state)
    good = state.read_bytes()
    queue._task_path(task.id).write_text("{partial")
    result = watch.run_watch(context, state)
    assert result["status"] == "error"
    assert state.read_bytes() == good
    assert "JSONDecodeError" in result["error"]


def test_malformed_last_good_state_is_preserved_and_reported_once(delivery):
    context, _queue, _task, state, _source = delivery
    state.parent.mkdir(parents=True)
    state.write_text("{incomplete")
    assert watch.run_watch(context, state)["status"] == "error"
    assert state.read_text() == "{incomplete"
    assert watch.run_watch(context, state)["alerts"] == 0
    assert len(_messages(context)) == 1


def test_removed_delivery_is_not_silently_resolved(delivery):
    context, queue, task, state, _source = delivery
    watch.run_watch(context, state)
    good = state.read_bytes()
    queue._task_path(task.id).unlink()
    result = watch.run_watch(context, state)
    assert result["status"] == "error"
    assert "previously watched tooling records missing" in result["error"]
    assert state.read_bytes() == good


def test_failed_message_send_does_not_acknowledge_findings(delivery, monkeypatch):
    context, _queue, _task, state, _source = delivery

    def failed(*_args, **_kwargs):
        raise OSError("fixture send failed")

    monkeypatch.setattr(MessageBus, "send", failed)
    with pytest.raises(OSError, match="fixture send failed"):
        watch.run_watch(context, state)
    assert not state.exists()


def test_failed_atomic_replace_keeps_last_good_state(delivery, monkeypatch):
    context, queue, task, state, _source = delivery
    watch.run_watch(context, state)
    good = state.read_bytes()
    _update(queue, task, next_action="new adoption action")
    original = os.replace

    def fail_state(source, destination, *args, **kwargs):
        if Path(destination) == state:
            raise OSError("fixture replace failed")
        return original(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_state)
    assert watch.run_watch(context, state)["status"] == "error"
    assert state.read_bytes() == good


def test_staleness_needs_both_explicit_fields_and_respects_timezone(delivery):
    context, queue, task, state, _source = delivery
    old = (datetime.now(timezone(timedelta(hours=-4))) - timedelta(hours=3)).isoformat()
    _update(queue, task, updated_at=old)
    assert "progress stale" not in watch.run_watch(context, state)["observations"][task.id]["issues"]
    _update(queue, task, stale_after_hours=2)
    assert "progress stale" in watch.run_watch(context, state)["observations"][task.id]["issues"]


@pytest.mark.parametrize("updated_at", [1700000000, "2026-01-01T00:00:00"])
def test_invalid_progress_timestamp_is_error_not_crash(delivery, updated_at):
    context, queue, task, state, _source = delivery
    _update(queue, task, updated_at=updated_at, stale_after_hours=2)
    result = watch.run_watch(context, state)
    assert result["status"] == "error"
    assert "invalid explicit progress expiry" in result["error"]
    assert not state.exists()


@pytest.mark.parametrize("key,value", [("source_path", "relative.py"), ("sha256", "bad"),
                                      ("installed_path", 123), ("state", "unknown")])
def test_invalid_delivery_metadata_is_error(delivery, key, value):
    context, queue, task, state, _source = delivery
    _update(queue, task, **{key: value})
    result = watch.run_watch(context, state)
    assert result["status"] == "error"
    assert not state.exists()


def test_symlink_artifact_and_escaping_state_are_rejected(delivery):
    context, queue, task, state, source = delivery
    link = source.with_name("link.py")
    link.symlink_to(source)
    _update(queue, task, source_path=str(link))
    assert watch.run_watch(context, state)["status"] == "error"
    assert not state.exists()
    with pytest.raises(ValueError, match="trusted boundary"):
        watch.run_watch(context, source.with_name("outside-state.json"))
    assert source.read_text().startswith("print")


def test_symlink_state_and_lock_are_rejected_before_open(delivery):
    context, _queue, _task, state, source = delivery
    state.parent.mkdir(parents=True)
    state.symlink_to(source)
    with pytest.raises(ValueError, match="symbolic link"):
        watch.run_watch(context, state)
    state.unlink()
    state.with_suffix(".json.lock").symlink_to(source)
    with pytest.raises(ValueError, match="symbolic link"):
        watch.run_watch(context, state)
    assert source.read_text().startswith("print")


def test_inventory_limit_is_error_instead_of_partial_success(delivery, monkeypatch):
    context, _queue, _task, state, _source = delivery
    monkeypatch.setattr(watch, "MAX_TASKS", 0)
    result = watch.run_watch(context, state)
    assert result["status"] == "error"
    assert "scan not complete" in result["error"]
    assert not state.exists()


def test_dry_run_writes_no_state_lock_or_messages(delivery):
    context, _queue, task, state, _source = delivery
    result = watch.run_watch(context, state, dry_run=True)
    assert result["observations"][task.id]["issues"] == ["ready; adoption pending"]
    assert not state.parent.exists()
    assert _messages(context) == []


def test_scheduled_entry_reads_its_named_profile_and_emits_result(delivery, monkeypatch):
    context, _queue, task, state, _source = delivery
    from afs.schema import AFSConfig, AgentConfig, ProfileConfig

    config = AFSConfig()
    config.profiles.active_profile = "test"
    config.profiles.profiles["test"] = ProfileConfig(agent_configs=[AgentConfig(
        name="oracle-tooling-watch", module="afs.agents.tooling_watch",
        extra={"context_root": str(context), "state_path": str(state)},
    )])
    monkeypatch.setenv("AFS_AGENT_NAME", "oracle-tooling-watch")
    monkeypatch.setattr(watch, "load_config_model", lambda **_kwargs: config)
    results = []
    monkeypatch.setattr(watch, "emit_result", lambda result, **_kwargs: results.append(result))
    assert watch.main([]) == 0
    assert results[0].name == "tooling-watch"
    assert task.id in results[0].payload["observations"]
    assert state.exists()
