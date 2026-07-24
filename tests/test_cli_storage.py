from __future__ import annotations

import json
import os
from pathlib import Path

from afs.cli import build_parser, main
from afs.cli import storage as storage_cli
from afs.storage_doctor import StorageApplyError


def test_storage_commands_are_registered() -> None:
    parser = build_parser([])
    audit = parser.parse_args(["storage", "audit", "--json"])
    plan = parser.parse_args(["storage", "plan", "--output", "/tmp/afs-storage-plan.json"])
    apply = parser.parse_args(
        [
            "storage",
            "apply",
            "--plan",
            "/tmp/afs-storage-plan.json",
            "--confirm",
            "storage_example",
            "--because",
            "Reviewed cleanup.",
        ]
    )
    assert audit.storage_command == "audit"
    assert plan.storage_command == "plan"
    assert apply.storage_command == "apply"


def test_storage_audit_and_empty_plan_are_readable(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    assert main(["storage", "audit", "--home", str(tmp_path), "--json"]) == 0
    audit = json.loads(capsys.readouterr().out)
    assert audit["schema"] == "afs.storage.audit.v1"
    assert audit["eligible_count"] == 0
    assert audit["safety"]["processes_stopped"] is False

    plan_path = tmp_path / "plan.json"
    assert (
        main(
            [
                "storage",
                "plan",
                "--home",
                str(tmp_path),
                "--output",
                str(plan_path),
                "--json",
            ]
        )
        == 0
    )
    response = json.loads(capsys.readouterr().out)
    assert response["candidate_count"] == 0
    assert response["plan"] == str(plan_path)
    assert plan_path.is_file()

    monkeypatch.setattr(
        storage_cli,
        "_TTY_READER",
        lambda _prompt: response["transaction"],
    )
    assert (
        main(
            [
                "storage",
                "apply",
                "--plan",
                str(plan_path),
                "--confirm",
                response["transaction"],
                "--because",
                "Empty plans must be refused.",
                "--json",
            ]
        )
        == 2
    )
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["status"] == "blocked"
    assert "no eligible candidates" in refusal["error"]


def test_storage_audit_escapes_control_characters_in_terminal_output(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    unsafe = "old\x1b[31m\nspoof.log"
    payload = {
        "home": f"{tmp_path}\nspoof-home",
        "disk": {"free_bytes": 1024, "pressure": "healthy"},
        "eligible_count": 0,
        "estimated_reclaim_bytes": 0,
        "candidates": [
            {
                "eligible": False,
                "category": "logs\nfake",
                "path": str(tmp_path / unsafe),
                "blocked_reasons": ["unsafe\x1b[0m\nreason"],
                "estimated_reclaim_bytes": 0,
            }
        ],
        "footprints": [
            {
                "name": "cache\nfake",
                "path": f"{tmp_path}\x1b[2J",
                "allocated_bytes": 0,
                "issue": "unreadable\nspoof",
            }
        ],
        "local_snapshots": ["snapshot\nspoof"],
        "safety": {"note": "safe\nspoof"},
    }
    monkeypatch.setattr(storage_cli, "audit_storage", lambda *_args, **_kwargs: payload)

    assert main(["storage", "audit", "--home", str(tmp_path)]) == 0
    output = capsys.readouterr().out

    assert "\x1b" not in output
    assert "old�[31m�spoof.log" in output
    assert "unsafe�[0m�reason" in output
    assert "snapshot�spoof" in output


def test_storage_apply_refuses_without_controlling_terminal(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    logs = tmp_path / ".lmstudio" / "server-logs"
    logs.mkdir(parents=True)
    old_log = logs / "old.log"
    old_log.write_text("old\n", encoding="utf-8")
    os.utime(old_log, (0, 0))
    plan_path = tmp_path / "plan.json"
    assert (
        main(
            [
                "storage",
                "plan",
                "--home",
                str(tmp_path),
                "--output",
                str(plan_path),
                "--json",
            ]
        )
        == 0
    )
    response = json.loads(capsys.readouterr().out)
    monkeypatch.setattr(storage_cli, "_TTY_READER", lambda _prompt: None)

    assert (
        main(
            [
                "storage",
                "apply",
                "--plan",
                str(plan_path),
                "--confirm",
                response["transaction"],
                "--because",
                "Reviewed the bounded cleanup plan.",
                "--json",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    refusal = json.loads(captured.out)
    assert refusal["status"] == "blocked"
    assert "interactive human confirmation" in refusal["error"]
    assert captured.err == ""


def test_storage_apply_receipt_failure_reports_only_durable_claim(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    plan_path = tmp_path / "plan.json"
    transaction = "storage_" + "a" * 32
    plan = {
        "transaction": transaction,
        "plan_sha256": "b" * 64,
        "home": str(tmp_path),
        "candidates": [{}],
        "estimated_reclaim_bytes": 1,
    }
    monkeypatch.setattr(storage_cli, "load_storage_plan", lambda _path: plan)
    monkeypatch.setattr(
        storage_cli,
        "_confirm_storage_apply",
        lambda *_args, **_kwargs: object(),
    )

    def fail_apply(*_args: object, **_kwargs: object) -> None:
        raise StorageApplyError(
            "receipt failed",
            {
                "status": "receipt_failure",
                "receipt_written": False,
                "claim_path": str(tmp_path / "claim.json"),
                "receipt_path": str(tmp_path / "receipt.json"),
            },
        )

    monkeypatch.setattr(storage_cli, "apply_storage_plan", fail_apply)
    assert (
        main(
            [
                "storage",
                "apply",
                "--plan",
                str(plan_path),
                "--confirm",
                transaction,
                "--because",
                "Reviewed.",
            ]
        )
        == 3
    )
    output = capsys.readouterr().out
    assert f"Durable claim: {tmp_path / 'claim.json'}" in output
    assert "Receipt:" not in output
    assert "No final receipt was persisted" in output
