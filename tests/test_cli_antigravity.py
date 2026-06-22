from __future__ import annotations

import argparse
import json
from pathlib import Path

from afs.cli import antigravity


def _args(**overrides) -> argparse.Namespace:
    payload = {
        "settings_path": None,
        "scope": "user",
        "project_path": None,
        "binary": "agy",
        "python_module": False,
        "force": False,
        "apply": False,
        "json": False,
        "path": None,
        "db_path": None,
        "skip_version": True,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_antigravity_setup_is_dry_run_by_default(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    exit_code = antigravity.antigravity_setup_command(_args())

    assert exit_code == 0
    assert "Dry run only" in capsys.readouterr().out
    assert not (tmp_path / ".gemini" / "antigravity-cli" / "settings.json").exists()


def test_antigravity_setup_apply_writes_mcp_entry(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    exit_code = antigravity.antigravity_setup_command(_args(apply=True))
    capsys.readouterr()

    assert exit_code == 0
    settings = tmp_path / ".gemini" / "antigravity-cli" / "settings.json"
    payload = json.loads(settings.read_text(encoding="utf-8"))
    entry = payload["mcpServers"]["afs"]
    assert entry["args"] == ["mcp", "serve"]
    assert entry["env"]["AFS_PREFER_REPO_CONFIG"] == "1"


def test_antigravity_status_handles_missing_binary_and_db(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(antigravity.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        antigravity,
        "find_afs_mcp_registrations",
        lambda **_kwargs: {"antigravity": []},
    )

    exit_code = antigravity.antigravity_status_command(
        _args(json=True, db_path=str(tmp_path / "missing.vscdb"))
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["binary"]["available"] is False
    assert payload["capture"]["db_exists"] is False
    assert payload["gemini_cli_cutoff"] == "2026-06-18"
