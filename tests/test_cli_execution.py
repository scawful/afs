from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pytest

from afs.cli import build_parser
from afs.cli.execution import _MAX_REQUEST_BYTES


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


def test_execution_inspect_rejects_oversized_regular_file(
    tmp_path: Path,
    capsys,
) -> None:
    request_path = tmp_path / "oversized.json"
    request_path.write_bytes(b"{" + b" " * _MAX_REQUEST_BYTES + b"}")

    rc = _invoke(
        [
            "execution",
            "inspect",
            "--request",
            str(request_path),
            "--allowed-root",
            str(tmp_path),
            "--json",
        ]
    )

    assert rc == 2
    assert "exceeds 1048576 bytes" in json.loads(capsys.readouterr().out)["error"]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO support is unavailable")
def test_execution_inspect_rejects_fifo_without_blocking(
    tmp_path: Path,
    capsys,
) -> None:
    request_path = tmp_path / "request.pipe"
    os.mkfifo(request_path)

    rc = _invoke(
        [
            "execution",
            "inspect",
            "--request",
            str(request_path),
            "--allowed-root",
            str(tmp_path),
            "--json",
        ]
    )

    assert rc == 2
    assert "regular file" in json.loads(capsys.readouterr().out)["error"]


def test_execution_inspect_returns_two_for_non_utf8_request_text(
    tmp_path: Path,
    capsys,
) -> None:
    payload = _payload(tmp_path)
    payload["command"]["argv"][2] = "\ud800"

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
    assert "UTF-8-encodable" in output["error"]


def test_execution_inspect_stdin_requires_utf8_bytes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    payload = json.dumps(_payload(tmp_path), ensure_ascii=False).replace(
        "read-only CLI inspection",
        "caf\u00e9",
    )
    stdin = io.TextIOWrapper(io.BytesIO(payload.encode("latin-1")), encoding="latin-1")
    monkeypatch.setattr(sys, "stdin", stdin)

    rc = _invoke(
        [
            "execution",
            "inspect",
            "--request",
            "-",
            "--allowed-root",
            str(tmp_path),
            "--allowed-executable",
            sys.executable,
            "--json",
        ]
    )

    assert rc == 2
    output = json.loads(capsys.readouterr().out)
    assert "not valid UTF-8" in output["error"]


def test_execution_inspect_rejects_duplicate_json_members(
    tmp_path: Path,
    capsys,
) -> None:
    encoded = json.dumps(_payload(tmp_path))
    duplicate_payloads = [
        encoded[:-1] + ', "caller": "duplicate"}',
        encoded.replace(
            '"kind": "argv"',
            '"kind": "argv", "kind": "legacy_shell"',
            1,
        ),
        encoded[:-1] + ', "set_env": {"TOKEN": "a", "TOKEN": "b"}}',
    ]

    for raw_request in duplicate_payloads:
        rc = _invoke(
            [
                "execution",
                "inspect",
                "--request",
                raw_request,
                "--allowed-root",
                str(tmp_path),
                "--allowed-executable",
                sys.executable,
                "--json",
            ]
        )

        assert rc == 2
        output = json.loads(capsys.readouterr().out)
        assert "duplicate JSON object member" in output["error"]


def test_execution_has_no_generic_run_subcommand() -> None:
    parser = build_parser(["execution", "run"])
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["execution", "run"])
    assert exc_info.value.code == 2
