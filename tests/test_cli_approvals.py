"""Tests for the approvals CLI subcommand."""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from afs.agents.guardrails import ApprovalGate, ApprovalRequest
from afs.cli.approvals import (
    approvals_approve_command,
    approvals_clear_command,
    approvals_history_command,
    approvals_list_command,
    approvals_reject_command,
    register_parsers,
)


def _make_gate(tmp_path: Path, requests: list[ApprovalRequest] | None = None) -> Path:
    """Create an approvals file and return its path."""
    approvals_file = tmp_path / "approvals.json"
    if requests:
        data = [r.to_dict() for r in requests]
        approvals_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return approvals_file


def _ns(approvals_file: Path, **kwargs) -> Namespace:
    """Build a Namespace with the approvals_file set."""
    return Namespace(approvals_file=str(approvals_file), **kwargs)


def _sample_request(
    agent: str = "scout",
    action: str = "git_push",
    detail: str = "push to origin/main",
    status: str = "pending",
) -> ApprovalRequest:
    return ApprovalRequest(
        agent=agent,
        action=action,
        detail=detail,
        timestamp=datetime.now(timezone.utc).isoformat(),
        status=status,
    )


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_no_approvals(tmp_path: Path, capsys) -> None:
    approvals_file = _make_gate(tmp_path)
    exit_code = approvals_list_command(_ns(approvals_file))
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No pending" in out


def test_list_with_pending(tmp_path: Path, capsys) -> None:
    requests = [
        _sample_request(agent="scout", action="git_push", detail="push changes"),
        _sample_request(agent="janitor", action="file_delete", detail="remove temp"),
    ]
    approvals_file = _make_gate(tmp_path, requests)
    exit_code = approvals_list_command(_ns(approvals_file))
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "scout" in out
    assert "janitor" in out
    assert "git_push" in out
    assert "file_delete" in out
    assert "2 pending" in out


def test_list_json(tmp_path: Path, capsys) -> None:
    requests = [_sample_request()]
    approvals_file = _make_gate(tmp_path, requests)
    exit_code = approvals_list_command(_ns(approvals_file, json=True))
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["agent"] == "scout"


def test_list_excludes_completed(tmp_path: Path, capsys) -> None:
    requests = [
        _sample_request(status="pending"),
        _sample_request(agent="bot", action="deploy", status="approved"),
    ]
    approvals_file = _make_gate(tmp_path, requests)
    exit_code = approvals_list_command(_ns(approvals_file))
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "1 pending" in out
    # The approved request should not appear in the list output
    assert "deploy" not in out


# ---------------------------------------------------------------------------
# approve
# ---------------------------------------------------------------------------


def test_approve_flow(tmp_path: Path, capsys) -> None:
    requests = [_sample_request()]
    approvals_file = _make_gate(tmp_path, requests)

    exit_code = approvals_approve_command(
        _ns(approvals_file, agent="scout", action="git_push")
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Approved" in out

    # Verify it's now approved in the file
    gate = ApprovalGate(path=approvals_file)
    assert len(gate.pending_requests()) == 0
    assert gate._pending[0].status == "approved"
    assert gate._pending[0].reviewed_by == "cli"


def test_approve_not_found(tmp_path: Path, capsys) -> None:
    approvals_file = _make_gate(tmp_path)
    exit_code = approvals_approve_command(
        _ns(approvals_file, agent="ghost", action="nope")
    )
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "No pending request" in out


# ---------------------------------------------------------------------------
# reject
# ---------------------------------------------------------------------------


def test_reject_flow(tmp_path: Path, capsys) -> None:
    requests = [_sample_request()]
    approvals_file = _make_gate(tmp_path, requests)

    exit_code = approvals_reject_command(
        _ns(approvals_file, agent="scout", action="git_push")
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Rejected" in out

    # Verify it's now rejected in the file
    gate = ApprovalGate(path=approvals_file)
    assert len(gate.pending_requests()) == 0
    assert gate._pending[0].status == "rejected"
    assert gate._pending[0].reviewed_by == "cli"


def test_reject_not_found(tmp_path: Path, capsys) -> None:
    approvals_file = _make_gate(tmp_path)
    exit_code = approvals_reject_command(
        _ns(approvals_file, agent="ghost", action="nope")
    )
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "No pending request" in out


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_removes_completed(tmp_path: Path, capsys) -> None:
    requests = [
        _sample_request(agent="scout", action="git_push", status="approved"),
        _sample_request(agent="scout", action="deploy", status="rejected"),
        _sample_request(agent="janitor", action="file_delete", status="pending"),
    ]
    approvals_file = _make_gate(tmp_path, requests)

    exit_code = approvals_clear_command(_ns(approvals_file))
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Cleared 2" in out
    assert "1 pending" in out

    # Verify only the pending request remains
    gate = ApprovalGate(path=approvals_file)
    assert len(gate._pending) == 1
    assert gate._pending[0].agent == "janitor"
    assert gate._pending[0].status == "pending"


def test_clear_nothing_to_clear(tmp_path: Path, capsys) -> None:
    requests = [_sample_request(status="pending")]
    approvals_file = _make_gate(tmp_path, requests)

    exit_code = approvals_clear_command(_ns(approvals_file))
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Cleared 0" in out


def test_clear_json(tmp_path: Path, capsys) -> None:
    requests = [
        _sample_request(status="approved"),
        _sample_request(agent="x", action="y", status="pending"),
    ]
    approvals_file = _make_gate(tmp_path, requests)

    exit_code = approvals_clear_command(_ns(approvals_file, json=True))
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cleared"] == 1
    assert payload["remaining"] == 1


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


def test_history_shows_all(tmp_path: Path, capsys) -> None:
    requests = [
        _sample_request(agent="scout", action="git_push", status="approved"),
        _sample_request(agent="janitor", action="file_delete", status="pending"),
        _sample_request(agent="bot", action="deploy", status="rejected"),
    ]
    approvals_file = _make_gate(tmp_path, requests)

    exit_code = approvals_history_command(_ns(approvals_file))
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "scout" in out
    assert "janitor" in out
    assert "bot" in out
    assert "approved" in out
    assert "pending" in out
    assert "rejected" in out
    assert "3 total" in out
    assert "1 pending" in out


def test_history_empty(tmp_path: Path, capsys) -> None:
    approvals_file = _make_gate(tmp_path)
    exit_code = approvals_history_command(_ns(approvals_file))
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No approval requests" in out


def test_history_json(tmp_path: Path, capsys) -> None:
    requests = [
        _sample_request(status="approved"),
        _sample_request(agent="x", action="y", status="pending"),
    ]
    approvals_file = _make_gate(tmp_path, requests)

    exit_code = approvals_history_command(_ns(approvals_file, json=True))
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2


# ---------------------------------------------------------------------------
# parser registration
# ---------------------------------------------------------------------------


def test_register_parsers_creates_subcommands() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parsers(subparsers)

    args = parser.parse_args(["approvals", "list"])
    assert args.command == "approvals"
    assert args.approvals_command == "list"

    args = parser.parse_args(["approvals", "approve", "myagent", "git_push"])
    assert args.agent == "myagent"
    assert args.action == "git_push"

    args = parser.parse_args(["approvals", "reject", "myagent", "deploy"])
    assert args.agent == "myagent"
    assert args.action == "deploy"
