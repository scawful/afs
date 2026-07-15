from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from afs.cli import build_parser
from afs.cli.optimize import _MAX_INPUT_BYTES

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = ROOT / "examples" / "optimization_gate"


def _run(parser, argv: list[str]) -> int:
    args = parser.parse_args(argv)
    return args.func(args)


def test_optimize_decide_json_is_byte_stable(capsys) -> None:
    parser = build_parser()
    argv = [
        "optimize",
        "decide",
        "--baseline",
        str(EXAMPLE_ROOT / "baseline.json"),
        "--candidate",
        str(EXAMPLE_ROOT / "candidate.json"),
        "--policy",
        str(EXAMPLE_ROOT / "policy.json"),
        "--json",
    ]

    assert _run(parser, argv) == 0
    first = capsys.readouterr().out
    assert _run(parser, argv) == 0
    second = capsys.readouterr().out

    assert second == first
    assert json.loads(first)["decision"] == "eligible_for_human_review"


def test_optimize_decide_rejects_nonstandard_json_number(tmp_path: Path, capsys) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"value": NaN}\n', encoding="utf-8")
    parser = build_parser()

    exit_code = _run(
        parser,
        [
            "optimize",
            "decide",
            "--baseline",
            str(EXAMPLE_ROOT / "baseline.json"),
            "--candidate",
            str(candidate),
            "--policy",
            str(EXAMPLE_ROOT / "policy.json"),
            "--json",
        ],
    )

    assert exit_code == 2
    assert "non-standard JSON number" in capsys.readouterr().err


def test_optimize_decide_rejects_oversized_regular_file(tmp_path: Path, capsys) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_bytes(b"{" + b" " * _MAX_INPUT_BYTES + b"}")
    parser = build_parser()

    exit_code = _run(
        parser,
        [
            "optimize",
            "decide",
            "--baseline",
            str(EXAMPLE_ROOT / "baseline.json"),
            "--candidate",
            str(candidate),
            "--policy",
            str(EXAMPLE_ROOT / "policy.json"),
        ],
    )

    assert exit_code == 2
    assert "input limit" in capsys.readouterr().err


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO support is unavailable")
def test_optimize_decide_rejects_non_regular_file_without_blocking(tmp_path: Path, capsys) -> None:
    candidate = tmp_path / "candidate.pipe"
    os.mkfifo(candidate)
    parser = build_parser()

    exit_code = _run(
        parser,
        [
            "optimize",
            "decide",
            "--baseline",
            str(EXAMPLE_ROOT / "baseline.json"),
            "--candidate",
            str(candidate),
            "--policy",
            str(EXAMPLE_ROOT / "policy.json"),
        ],
    )

    assert exit_code == 2
    assert "not a regular file" in capsys.readouterr().err


def test_optimize_decide_internal_error_does_not_reuse_verdict_exit_codes(
    monkeypatch, capsys
) -> None:
    def _explode(*_args, **_kwargs):
        raise RuntimeError("simulated gate defect")

    monkeypatch.setattr("afs.cli.optimize.decide_optimization_step", _explode)
    parser = build_parser()

    exit_code = _run(
        parser,
        [
            "optimize",
            "decide",
            "--baseline",
            str(EXAMPLE_ROOT / "baseline.json"),
            "--candidate",
            str(EXAMPLE_ROOT / "candidate.json"),
            "--policy",
            str(EXAMPLE_ROOT / "policy.json"),
            "--json",
        ],
    )

    assert exit_code == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "internal error" in captured.err
    assert "simulated gate defect" in captured.err


def test_optimize_decide_reports_large_integer_as_invalid_input(tmp_path: Path, capsys) -> None:
    candidate_payload = json.loads((EXAMPLE_ROOT / "candidate.json").read_text(encoding="utf-8"))
    candidate_payload["metrics"][0]["value"] = 10**400
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(candidate_payload), encoding="utf-8")
    parser = build_parser()

    exit_code = _run(
        parser,
        [
            "optimize",
            "decide",
            "--baseline",
            str(EXAMPLE_ROOT / "baseline.json"),
            "--candidate",
            str(candidate),
            "--policy",
            str(EXAMPLE_ROOT / "policy.json"),
        ],
    )

    assert exit_code == 2
    assert "supported finite range" in capsys.readouterr().err


def test_optimize_decide_rejects_lone_unicode_surrogate(
    tmp_path: Path,
    capsys,
) -> None:
    candidate_payload = json.loads((EXAMPLE_ROOT / "candidate.json").read_text(encoding="utf-8"))
    candidate_payload["artifact_ref"] = "invalid-\ud800"
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(candidate_payload), encoding="utf-8")

    exit_code = _run(
        build_parser(),
        [
            "optimize",
            "decide",
            "--baseline",
            str(EXAMPLE_ROOT / "baseline.json"),
            "--candidate",
            str(candidate),
            "--policy",
            str(EXAMPLE_ROOT / "policy.json"),
            "--json",
        ],
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "UTF-8-encodable Unicode" in captured.err


@pytest.mark.parametrize("nested", [False, True])
def test_optimize_decide_rejects_duplicate_json_members(
    tmp_path: Path,
    capsys,
    nested: bool,
) -> None:
    candidate_text = (EXAMPLE_ROOT / "candidate.json").read_text(encoding="utf-8")
    if nested:
        candidate_text = candidate_text.replace(
            '"seed": 42,',
            '"seed": 42, "seed": 43,',
            1,
        )
    else:
        candidate_text = candidate_text.rstrip()[:-1] + ', "candidate_id": "duplicate"}\n'
    candidate = tmp_path / "candidate.json"
    candidate.write_text(candidate_text, encoding="utf-8")

    exit_code = _run(
        build_parser(),
        [
            "optimize",
            "decide",
            "--baseline",
            str(EXAMPLE_ROOT / "baseline.json"),
            "--candidate",
            str(candidate),
            "--policy",
            str(EXAMPLE_ROOT / "policy.json"),
            "--json",
        ],
    )

    assert exit_code == 2
    assert "duplicate JSON object member" in capsys.readouterr().err
