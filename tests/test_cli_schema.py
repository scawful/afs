from __future__ import annotations

import io
import json
from argparse import Namespace

from afs.cli.schema import (
    schema_list_command,
    schema_show_command,
    schema_validate_command,
)

_VALID_EDIT = {"summary": "x", "paths": ["a.py"], "checks": ["pytest"]}


def _args(**kwargs) -> Namespace:
    values = {
        "schema": None,
        "workflow": None,
        "text": None,
        "file": None,
        "json": False,
        "name": None,
    }
    values.update(kwargs)
    return Namespace(**values)


def test_schema_list_json(capsys) -> None:
    assert schema_list_command(_args(json=True)) == 0
    names = json.loads(capsys.readouterr().out)
    assert "implementation-plan" in names
    assert "handoff-summary" in names


def test_schema_show_unknown_returns_2(capsys) -> None:
    assert schema_show_command(_args(name="nope")) == 2


def test_validate_valid_via_text_exit_0(capsys) -> None:
    rc = schema_validate_command(
        _args(schema="edit-intent", text=json.dumps(_VALID_EDIT))
    )
    assert rc == 0
    assert "valid: response matches `edit-intent`" in capsys.readouterr().out


def test_validate_invalid_prints_correction_exit_1(capsys) -> None:
    rc = schema_validate_command(_args(schema="handoff-summary", text='{"accomplished":["x"]}'))
    assert rc == 1
    out = capsys.readouterr().out
    assert "did not match the required `handoff-summary`" in out
    assert "next_steps" in out


def test_validate_by_workflow_resolves_schema(capsys) -> None:
    # edit_fast -> edit-intent
    rc = schema_validate_command(
        _args(workflow="edit_fast", text=json.dumps(_VALID_EDIT))
    )
    assert rc == 0
    assert "edit-intent" in capsys.readouterr().out


def test_validate_json_output_includes_correction(capsys) -> None:
    rc = schema_validate_command(
        _args(schema="handoff-summary", text="not json", json=True)
    )
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert payload["schema"] == "handoff-summary"
    assert "invalid JSON" in payload["parse_error"]
    assert "handoff-summary" in payload["correction"]


def test_validate_reads_from_stdin(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_VALID_EDIT)))
    rc = schema_validate_command(_args(schema="edit-intent"))
    assert rc == 0
    assert "valid" in capsys.readouterr().out


def test_validate_unknown_schema_returns_2(capsys) -> None:
    assert schema_validate_command(_args(schema="nope", text="{}")) == 2
