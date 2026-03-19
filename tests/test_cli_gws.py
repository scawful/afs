from __future__ import annotations

import argparse
import json

from afs.cli import build_parser, gws_cli


def test_gws_status_returns_json_for_available_client(monkeypatch, capsys) -> None:
    class _FakeClient:
        available = True

        def auth_status(self) -> dict[str, str]:
            return {"auth_method": "oauth"}

    monkeypatch.setattr(gws_cli, "get_client", lambda: _FakeClient())

    exit_code = gws_cli._gws_status(argparse.Namespace())

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"auth_method": "oauth"}


def test_gws_agenda_returns_error_when_binary_missing(monkeypatch, capsys) -> None:
    class _FakeClient:
        available = False

    monkeypatch.setattr(gws_cli, "get_client", lambda: _FakeClient())

    exit_code = gws_cli._gws_agenda(argparse.Namespace())

    assert exit_code == 1
    assert "not installed" in capsys.readouterr().out


def test_parser_registers_gws_command(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "afs.toml"
    config_path.write_text("[extensions]\nauto_discover = false\n", encoding="utf-8")
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("AFS_EXTENSION_DIRS", raising=False)
    monkeypatch.delenv("AFS_ENABLED_EXTENSIONS", raising=False)

    parser = build_parser()
    command_action = next(
        action for action in parser._actions if getattr(action, "choices", None)
    )

    assert "gws" in command_action.choices
