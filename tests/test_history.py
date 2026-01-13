from __future__ import annotations

import json
from pathlib import Path

from afs.history import log_cli_invocation, log_event


def _write_config(path: Path, context_root: Path) -> None:
    path.write_text(
        "[general]\n"
        f"context_root = \"{context_root}\"\n"
        "\n[history]\n"
        "enabled = true\n"
        "include_payloads = true\n"
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
