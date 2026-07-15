from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from afs.cli import build_parser


def _payload(tmp_path: Path, *, isolation: str = "process") -> dict:
    return {
        "schema_version": "1.0",
        "command": {
            "kind": "argv",
            "argv": [sys.executable, "-c", "raise RuntimeError('must not execute')"],
        },
        "caller": "pytest",
        "purpose": "read-only CLI inspection",
        "cwd": str(tmp_path),
        "isolation": isolation,
        "network": "inherit",
    }


def _invoke(argv: list[str]) -> int:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    return args.func(args)


def test_execution_inspect_allows_without_running(tmp_path: Path, capsys) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_payload(tmp_path)), encoding="utf-8")

    rc = _invoke(
        [
            "execution",
            "inspect",
            "--request",
            str(request_path),
            "--allowed-root",
            str(tmp_path),
            "--allowed-executable",
            sys.executable,
            "--json",
        ]
    )

    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["allowed"] is True
    assert output["resolved_executable"] == str(Path(sys.executable).resolve())


def test_execution_inspect_returns_three_for_policy_block(tmp_path: Path, capsys) -> None:
    rc = _invoke(
        [
            "execution",
            "inspect",
            "--request",
            json.dumps(_payload(tmp_path, isolation="sandbox")),
            "--allowed-root",
            str(tmp_path),
            "--allowed-executable",
            sys.executable,
            "--json",
        ]
    )

    assert rc == 3
    output = json.loads(capsys.readouterr().out)
    assert output["allowed"] is False
    assert "unsupported_isolation" in output["reason_codes"]


def test_execution_inspect_returns_two_for_invalid_input(tmp_path: Path, capsys) -> None:
    payload = _payload(tmp_path)
    payload["unknown"] = True

    rc = _invoke(
        [
            "execution",
            "inspect",
            "--request",
            json.dumps(payload),
            "--allowed-root",
            str(tmp_path),
            "--allowed-executable",
            sys.executable,
            "--json",
        ]
    )

    assert rc == 2
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is False
    assert "unknown fields" in output["error"]


def test_execution_has_no_generic_run_subcommand() -> None:
    parser = build_parser(["execution", "run"])
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["execution", "run"])
    assert exc_info.value.code == 2
