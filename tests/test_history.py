from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import afs.history as history_module
from afs.context_fs import ContextFileSystem
from afs.context_layout import LayoutMetadata, scaffold_v2
from afs.history import (
    append_history_event,
    iter_history_events,
    log_cli_invocation,
    log_event,
    log_session_event,
    prepare_history_reactor_root,
)
from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import AFSConfig, GeneralConfig, HistoryConfig
from afs.work_assistant import WorkAssistantStore


def _write_config(path: Path, context_root: Path, *, include_payloads: bool = True) -> None:
    path.write_text(
        "[general]\n"
        f"context_root = \"{context_root}\"\n"
        "\n[history]\n"
        "enabled = true\n"
        f"include_payloads = {'true' if include_payloads else 'false'}\n"
        "max_inline_chars = 10\n",
        encoding="utf-8",
    )


def test_log_event_writes_payload_file(tmp_path, monkeypatch) -> None:
    context_root = tmp_path / "context"
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, context_root)

    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("AFS_CONTEXT_ROOT", str(context_root))
    monkeypatch.delenv("AFS_HISTORY_DISABLED", raising=False)

    event_id = log_event(
        "model",
        "test",
        payload={"prompt": "x" * 50, "response": "y"},
    )
    assert event_id

    history_dir = context_root / "history"
    log_files = list(history_dir.glob("events_*.jsonl"))
    assert log_files

    event = json.loads(log_files[0].read_text(encoding="utf-8").splitlines()[-1])
    assert event.get("payload_ref")

    payload_path = history_dir / event["payload_ref"]
    assert payload_path.exists()


def test_history_config_defaults_disable_payloads() -> None:
    assert HistoryConfig().include_payloads is False


def test_log_cli_invocation_redacts(tmp_path, monkeypatch) -> None:
    context_root = tmp_path / "context"
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, context_root)

    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("AFS_CONTEXT_ROOT", str(context_root))
    monkeypatch.delenv("AFS_HISTORY_DISABLED", raising=False)

    log_cli_invocation(["--token", "secret", "--api-key=abcd"], 0)

    history_dir = context_root / "history"
    log_files = list(history_dir.glob("events_*.jsonl"))
    assert log_files

    event = json.loads(log_files[0].read_text(encoding="utf-8").splitlines()[-1])
    argv = event["metadata"]["argv"]
    assert argv == ["--token", "[redacted]", "--api-key=[redacted]"]


def test_log_event_uses_remapped_history_dir(tmp_path, monkeypatch) -> None:
    context_root = tmp_path / "context"
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, context_root)
    context_root.mkdir(parents=True)
    (context_root / "metadata.json").write_text(
        json.dumps({"directories": {"history": "ledger"}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("AFS_CONTEXT_ROOT", str(context_root))
    monkeypatch.delenv("AFS_HISTORY_DISABLED", raising=False)

    event_id = log_event("model", "test", payload={"prompt": "hello"})
    assert event_id

    assert list((context_root / "history").glob("events_*.jsonl")) == []
    log_files = list((context_root / "ledger").glob("events_*.jsonl"))
    assert log_files


def test_v1_log_event_preserves_external_remapped_history_dir(tmp_path, monkeypatch) -> None:
    context_root = tmp_path / "context"
    external_history = tmp_path / "external-ledger"
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, context_root)
    context_root.mkdir(parents=True)
    (context_root / "metadata.json").write_text(
        json.dumps({"directories": {"history": str(external_history)}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("AFS_CONTEXT_ROOT", str(context_root))
    monkeypatch.delenv("AFS_HISTORY_DISABLED", raising=False)

    assert log_event("model", "test", context_root=context_root)

    assert list(external_history.glob("events_*.jsonl"))
    assert not (context_root / "history").exists()


def test_log_event_omits_payloads_by_default(tmp_path, monkeypatch) -> None:
    context_root = tmp_path / "context"
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, context_root, include_payloads=False)

    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("AFS_CONTEXT_ROOT", str(context_root))
    monkeypatch.delenv("AFS_HISTORY_DISABLED", raising=False)

    event_id = log_event("model", "test", payload={"prompt": "secret", "response": "y"})
    assert event_id

    history_dir = context_root / "history"
    log_files = list(history_dir.glob("events_*.jsonl"))
    assert log_files

    event = json.loads(log_files[0].read_text(encoding="utf-8").splitlines()[-1])
    assert "payload" not in event
    assert "payload_ref" not in event


def test_log_event_can_explicitly_include_payloads(tmp_path, monkeypatch) -> None:
    context_root = tmp_path / "context"
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, context_root, include_payloads=False)

    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("AFS_CONTEXT_ROOT", str(context_root))
    monkeypatch.delenv("AFS_HISTORY_DISABLED", raising=False)

    event_id = log_event(
        "model",
        "test",
        payload={"prompt": "x" * 50, "response": "y"},
        include_payloads=True,
    )
    assert event_id

    history_dir = context_root / "history"
    log_files = list(history_dir.glob("events_*.jsonl"))
    assert log_files

    event = json.loads(log_files[0].read_text(encoding="utf-8").splitlines()[-1])
    assert event.get("payload_ref")


def test_log_session_event_respects_explicit_context_root(tmp_path, monkeypatch) -> None:
    context_root = tmp_path / "context"
    config = AFSConfig(
        general=GeneralConfig(
            context_root=tmp_path / "other-context",
        )
    )
    context_root.mkdir(parents=True, exist_ok=True)
    (context_root / "history").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(history_module, "load_config_model", lambda *args, **kwargs: config)
    monkeypatch.delenv("AFS_HISTORY_DISABLED", raising=False)

    event_id = log_session_event("bootstrap", context_root=context_root)

    assert event_id
    log_files = list((context_root / "history").glob("events_*.jsonl"))
    assert log_files


def test_log_event_enriches_work_assistant_state(tmp_path, monkeypatch) -> None:
    context_root = tmp_path / "context"
    config_path = tmp_path / "afs.toml"
    _write_config(config_path, context_root, include_payloads=True)

    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("AFS_CONTEXT_ROOT", str(context_root))
    monkeypatch.delenv("AFS_HISTORY_DISABLED", raising=False)
    monkeypatch.delenv("AFS_WORK_ASSISTANT_ENRICH_DISABLED", raising=False)

    event_id = log_event(
        "context",
        "test.connector",
        op="snapshot",
        metadata={
            "target_system": "google-docs",
            "target_type": "docs",
            "target_id": "doc-1",
            "owner": {"display_name": "Doc Owner", "email": "owner@example.com"},
            "requires_approval": True,
            "action": "edit_doc",
            "summary": "Apply doc edit",
        },
        context_root=context_root,
    )

    assert event_id
    store = WorkAssistantStore(context_root)
    assert store.list_people()[0]["display_name"] == "Doc Owner"
    assert store.list_approvals()[0]["summary"] == "Apply doc edit"
    assert store.list_activity()[0]["event_id"] == event_id


def test_context_fs_history_records_metadata_not_file_contents(tmp_path, monkeypatch) -> None:
    context_root = tmp_path / ".context"
    config = AFSConfig(
        general=GeneralConfig(
            context_root=context_root,
        ),
        history=HistoryConfig(include_payloads=True),
    )
    manager = AFSManager(config=config)
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    monkeypatch.setattr(history_module, "load_config_model", lambda *args, **kwargs: config)
    monkeypatch.delenv("AFS_HISTORY_DISABLED", raising=False)

    fs = ContextFileSystem(manager, context_root)
    fs.write_text(MountType.SCRATCHPAD, "note.md", "top secret", mkdirs=True)
    assert fs.read_text(MountType.SCRATCHPAD, "note.md") == "top secret"

    history_dir = context_root / "history"
    events = [
        json.loads(line)
        for line in next(history_dir.glob("events_*.jsonl")).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fs_events = [event for event in events if event.get("type") == "fs"]

    assert len(fs_events) == 2
    for event in fs_events:
        metadata = event["metadata"]
        assert metadata["content_chars"] == len("top secret")
        assert metadata["content_sha256"]
        assert "payload" not in event
        assert "payload_ref" not in event


def _v2_history_config(context_root: Path) -> AFSConfig:
    return AFSConfig(
        general=GeneralConfig(context_root=context_root),
        history=HistoryConfig(include_payloads=True, max_inline_chars=10),
    )


def test_v2_log_event_writes_canonical_common_ledger(tmp_path, monkeypatch) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    config = _v2_history_config(context_root)
    monkeypatch.setattr(history_module, "load_config_model", lambda *args, **kwargs: config)

    assert log_event("session", "test", context_root=context_root)
    append_history_event(
        context_root / "history" / "common",
        "session",
        "test",
    )

    assert not list((context_root / "history").glob("events_*.jsonl"))
    assert len(list((context_root / "history" / "common").glob("events_*.jsonl"))) == 1
    assert not (context_root / "history" / "common" / "common").exists()


def test_v2_log_event_rejects_linked_common_root(tmp_path, monkeypatch) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    common = context_root / "history" / "common"
    try:
        common.symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"directory symlinks unavailable: {exc}")
    config = _v2_history_config(context_root)
    monkeypatch.setattr(history_module, "load_config_model", lambda *args, **kwargs: config)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        log_event("session", "test", context_root=context_root)

    assert list(outside.iterdir()) == []


def test_v2_log_event_rejects_replaced_history_root(tmp_path, monkeypatch) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    original_history = context_root / "history"
    original_history.rename(context_root / "history-original")
    outside = tmp_path / "outside-history"
    outside.mkdir()
    try:
        original_history.symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"directory symlinks unavailable: {exc}")
    config = _v2_history_config(context_root)
    monkeypatch.setattr(history_module, "load_config_model", lambda *args, **kwargs: config)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        log_event("session", "test", context_root=context_root)

    assert list(outside.iterdir()) == []


def test_v2_payload_write_rejects_linked_directory_and_leaf(tmp_path, monkeypatch) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    common = context_root / "history" / "common"
    common.mkdir()
    outside_dir = tmp_path / "outside-payloads"
    outside_dir.mkdir()
    payload_dir = common / "payloads"
    try:
        payload_dir.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"directory symlinks unavailable: {exc}")
    config = _v2_history_config(context_root)
    monkeypatch.setattr(history_module, "load_config_model", lambda *args, **kwargs: config)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        log_event(
            "session",
            "test",
            payload={"prompt": "x" * 50},
            context_root=context_root,
        )
    assert list(outside_dir.iterdir()) == []

    payload_dir.unlink()
    payload_dir.mkdir()
    outside_file = tmp_path / "outside.json"
    outside_file.write_text("do not overwrite", encoding="utf-8")
    event_id = "event-safe"
    timestamp = "2026-07-17T12:00:00+00:00"
    expected = payload_dir / "20260717T120000000000Z_event-safe.json"
    expected.symlink_to(outside_file)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        append_history_event(
            context_root / "history",
            "session",
            "test",
            payload={"prompt": "x" * 50},
            include_payloads=True,
            max_inline_chars=10,
            event_id=event_id,
            timestamp=timestamp,
        )
    assert outside_file.read_text(encoding="utf-8") == "do not overwrite"


@pytest.mark.parametrize(
    ("event_id", "payload_dir_name"),
    [
        ("../../../scratchpad/common/pwn", "payloads"),
        ("event-safe", "../scratchpad"),
    ],
)
def test_v2_payload_write_rejects_path_traversal(
    tmp_path,
    event_id,
    payload_dir_name,
) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)

    with pytest.raises(ValueError, match="safe filesystem segment|unsafe filesystem"):
        append_history_event(
            context_root / "history",
            "session",
            "test",
            payload={"prompt": "x" * 50},
            include_payloads=True,
            max_inline_chars=10,
            event_id=event_id,
            payload_dir_name=payload_dir_name,
        )
    assert not (context_root / "scratchpad" / "common" / "pwn.json").exists()


def test_v2_payload_filename_does_not_use_raw_timestamp(tmp_path) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)

    append_history_event(
        context_root / "history",
        "session",
        "test",
        payload={"prompt": "x" * 50},
        include_payloads=True,
        max_inline_chars=10,
        event_id="event-safe",
        timestamp="../../outside",
    )

    payloads = list((context_root / "history" / "common" / "payloads").iterdir())
    assert len(payloads) == 1
    assert payloads[0].name.endswith("_event-safe.json")
    assert ".." not in payloads[0].name


def test_v2_iterator_skips_linked_log_and_payload(tmp_path) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    common = context_root / "history" / "common"
    common.mkdir()
    outside_log = tmp_path / "events_20260717.jsonl"
    outside_log.write_text(
        json.dumps({"id": "outside", "type": "secret", "payload": {"canary": "outside"}})
        + "\n",
        encoding="utf-8",
    )
    linked_log = common / "events_20260717.jsonl"
    try:
        linked_log.symlink_to(outside_log)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"file symlinks unavailable: {exc}")

    assert list(iter_history_events(context_root / "history")) == []

    linked_log.unlink()
    outside_payload = tmp_path / "outside-payload.json"
    outside_payload.write_text('{"canary": "outside-payload"}', encoding="utf-8")
    payload_dir = common / "payloads"
    payload_dir.mkdir()
    (payload_dir / "linked.json").symlink_to(outside_payload)
    linked_log.write_text(
        json.dumps(
            {
                "id": "safe-event",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "session",
                "source": "test",
                "metadata": {},
                "payload_ref": "payloads/linked.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    events = list(iter_history_events(context_root / "history"))
    assert len(events) == 1
    assert "payload" not in events[0]


def test_v2_iterator_reads_legacy_root_and_common_once(tmp_path) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    common = context_root / "history" / "common"
    common.mkdir()
    legacy_event = {
        "id": "migrated-event",
        "timestamp": "2026-07-17T12:00:00+00:00",
        "type": "session",
        "source": "legacy",
        "metadata": {"version": "stale"},
    }
    canonical_event = {
        **legacy_event,
        "source": "canonical",
        "metadata": {"version": "current"},
    }
    legacy_unique = {
        **legacy_event,
        "id": "legacy-only",
        "timestamp": "2026-07-17T11:00:00+00:00",
    }
    common_new = {
        **canonical_event,
        "id": "common-new",
        "timestamp": "2026-07-17T13:00:00+00:00",
    }
    (context_root / "history" / "events_20260717.jsonl").write_text(
        json.dumps(legacy_unique) + "\n" + json.dumps(legacy_event) + "\n",
        encoding="utf-8",
    )
    (common / "events_20260717.jsonl").write_text(
        json.dumps(canonical_event) + "\n" + json.dumps(common_new) + "\n",
        encoding="utf-8",
    )

    events = list(iter_history_events(common))

    assert [event["id"] for event in events] == [
        "legacy-only",
        "migrated-event",
        "common-new",
    ]
    assert events[1]["source"] == "canonical"
    assert events[1]["metadata"] == {"version": "current"}


def test_seeded_v2_legacy_event_reads_only_hash_matching_legacy_payload(
    tmp_path: Path,
) -> None:
    context_root = tmp_path / ".context"
    legacy_history = context_root / "history"
    append_history_event(
        legacy_history,
        "session",
        "legacy",
        payload={"canary": "legacy-payload" * 20},
        include_payloads=True,
        max_inline_chars=10,
        timestamp="2026-07-17T12:00:00+00:00",
        event_id="legacy-payload-event",
    )
    LayoutMetadata().write(context_root)
    prepare_history_reactor_root(context_root)

    events = list(iter_history_events(context_root / "history"))

    assert len(events) == 1
    assert events[0]["payload"]["canary"].startswith("legacy-payload")

    payload_path = next((legacy_history / "payloads").glob("*.json"))
    payload_path.write_text('{"canary":"tampered"}', encoding="utf-8")
    tampered = list(iter_history_events(context_root / "history"))
    assert len(tampered) == 1
    assert "payload" not in tampered[0]
