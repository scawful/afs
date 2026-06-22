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
        "timeout": 10.0,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_antigravity_setup_is_dry_run_by_default(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    exit_code = antigravity.antigravity_setup_command(_args())

    assert exit_code == 0
    assert "Dry run only" in capsys.readouterr().out
    assert not (tmp_path / ".gemini" / "config" / "mcp_config.json").exists()


def test_antigravity_setup_apply_writes_mcp_entry(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    exit_code = antigravity.antigravity_setup_command(_args(apply=True))
    capsys.readouterr()

    assert exit_code == 0
    settings = tmp_path / ".gemini" / "config" / "mcp_config.json"
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


def test_antigravity_status_detects_migrated_mcp_config(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(antigravity.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        antigravity,
        "find_afs_mcp_registrations",
        lambda **_kwargs: {"antigravity": [str(tmp_path / ".gemini" / "config" / "mcp_config.json")]},
    )
    settings = tmp_path / ".gemini" / "config" / "mcp_config.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"mcpServers": {"afs": {"command": "afs", "args": ["mcp", "serve"]}}}))

    exit_code = antigravity.antigravity_status_command(
        _args(json=True, db_path=str(tmp_path / "missing.vscdb"))
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["settings"]["has_mcp_servers"] is True
    assert payload["settings"]["has_afs_mcp"] is True
    assert payload["settings"]["afs_mcp_paths"] == [str(settings)]


def test_antigravity_models_json_parses_labels(monkeypatch, capsys) -> None:
    monkeypatch.setattr(antigravity.shutil, "which", lambda _name: "/usr/local/bin/agy")

    def fake_run(*_args, **_kwargs):
        return antigravity.subprocess.CompletedProcess(
            args=["agy", "models"],
            returncode=0,
            stdout="Gemini 3.5 Flash (Medium)\nClaude Opus 4.6 (Thinking)\n",
            stderr="",
        )

    monkeypatch.setattr(antigravity.subprocess, "run", fake_run)

    exit_code = antigravity.antigravity_models_command(_args(json=True))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"][0] == {
        "raw": "Gemini 3.5 Flash (Medium)",
        "name": "Gemini 3.5 Flash",
        "label": "Medium",
    }
    assert payload["models"][1]["name"] == "Claude Opus 4.6"
