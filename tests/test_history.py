from __future__ import annotations

import json
from pathlib import Path

import afs.history as history_module
from afs.context_fs import ContextFileSystem
from afs.history import log_cli_invocation, log_event, log_session_event
from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import AFSConfig, GeneralConfig, HistoryConfig


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
            agent_workspaces_dir=(tmp_path / "other-context" / "workspaces"),
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


def test_context_fs_history_records_metadata_not_file_contents(tmp_path, monkeypatch) -> None:
    context_root = tmp_path / ".context"
    config = AFSConfig(
        general=GeneralConfig(
            context_root=context_root,
            agent_workspaces_dir=context_root / "workspaces",
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
