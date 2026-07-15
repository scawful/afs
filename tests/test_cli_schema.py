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


def test_validate_rejects_non_utf8_stdin(monkeypatch, capsys) -> None:
    stdin = io.TextIOWrapper(io.BytesIO(b'{"summary":"caf\xe9"}'), encoding="latin-1")
    monkeypatch.setattr("sys.stdin", stdin)

    rc = schema_validate_command(_args(schema="edit-intent"))

    assert rc == 2
    assert "not valid UTF-8" in capsys.readouterr().err


def test_validate_unknown_schema_returns_2(capsys) -> None:
    assert schema_validate_command(_args(schema="nope", text="{}")) == 2


def test_validate_skeleton_flags_modified_intent(tmp_path, capsys) -> None:
    skeleton = {
        "human_intent": {"goal": "human goal"},
        "summary": "seed",
    }
    skeleton_path = tmp_path / "skeleton.json"
    skeleton_path.write_text(json.dumps(skeleton), encoding="utf-8")

    plan = {
        "human_intent": {"goal": "agent reworded goal"},
        "summary": "expanded",
        "steps": ["do it"],
        "verification": ["pytest"],
        "risks": ["none"],
    }
    rc = schema_validate_command(
        _args(
            schema="implementation-plan",
            text=json.dumps(plan),
            skeleton=str(skeleton_path),
        )
    )
    assert rc == 1
    assert "modified" in capsys.readouterr().out


def test_validate_skeleton_passes_when_intent_preserved(tmp_path, capsys) -> None:
    intent = {"goal": "human goal", "done_when": ["tests pass"]}
    skeleton_path = tmp_path / "skeleton.json"
    skeleton_path.write_text(json.dumps({"human_intent": intent}), encoding="utf-8")

    plan = {
        "human_intent": intent,
        "summary": "expanded",
        "steps": ["do it"],
        "verification": ["pytest"],
        "risks": ["none"],
    }
    rc = schema_validate_command(
        _args(
            schema="implementation-plan",
            text=json.dumps(plan),
            skeleton=str(skeleton_path),
            json=True,
        )
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["human_intent_violations"] == []


def test_validate_skeleton_is_parsed_strictly(tmp_path, capsys) -> None:
    """The skeleton is a trust anchor: fenced/markdown-wrapped JSON that the
    lenient response coercion would accept must be rejected here."""
    skeleton_path = tmp_path / "skeleton.json"
    skeleton_path.write_text(
        '```json\n{"human_intent": {"goal": "g"}}\n```', encoding="utf-8"
    )

    plan = {
        "summary": "expanded",
        "steps": ["do it"],
        "verification": ["pytest"],
        "risks": ["none"],
    }
    rc = schema_validate_command(
        _args(
            schema="implementation-plan",
            text=json.dumps(plan),
            skeleton=str(skeleton_path),
        )
    )
    assert rc == 2
    assert "invalid skeleton" in capsys.readouterr().err


def test_validate_skeleton_must_be_object(tmp_path, capsys) -> None:
    skeleton_path = tmp_path / "skeleton.json"
    skeleton_path.write_text('["not", "an", "object"]', encoding="utf-8")

    plan = {
        "summary": "expanded",
        "steps": ["do it"],
        "verification": ["pytest"],
        "risks": ["none"],
    }
    rc = schema_validate_command(
        _args(
            schema="implementation-plan",
            text=json.dumps(plan),
            skeleton=str(skeleton_path),
        )
    )
    assert rc == 2
    assert "must be a JSON object" in capsys.readouterr().err
