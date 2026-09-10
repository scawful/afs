"""Tests for the approvals CLI subcommand."""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from afs.agents.guardrails import ApprovalGate, ApprovalRequest
from afs.cli import approvals as cli_approvals
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
    detail: str = "push to origin/development",
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


def test_approve_flow(tmp_path: Path, capsys, monkeypatch) -> None:
    requests = [_sample_request()]
    approvals_file = _make_gate(tmp_path, requests)
    monkeypatch.setattr(cli_approvals, "_TTY_READER", lambda prompt: "scout:git_push")

    exit_code = approvals_approve_command(
        _ns(approvals_file, agent="scout", action="git_push", because="push is release-gated")
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Approved" in out

    # Verify it's now approved in the file, with the rationale and provenance
    gate = ApprovalGate(path=approvals_file)
    assert len(gate.pending_requests()) == 0
    assert gate._pending[0].status == "approved"
    assert gate._pending[0].reviewed_by  # OS user, never a claimable flag
    assert gate._pending[0].reviewed_via == "controlling_terminal"
    assert gate._pending[0].human_confirmed is True
    assert gate._pending[0].rationale == "push is release-gated"
    assert gate._pending[0].request_id.startswith("gate_")


def test_approve_refused_headless(tmp_path: Path, capsys, monkeypatch) -> None:
    """An agent without a controlling terminal cannot self-approve."""
    requests = [_sample_request()]
    approvals_file = _make_gate(tmp_path, requests)
    monkeypatch.setattr(cli_approvals, "_TTY_READER", lambda prompt: None)

    exit_code = approvals_approve_command(
        _ns(approvals_file, agent="scout", action="git_push", because="looks fine")
    )
    assert exit_code == 2
    assert "interactive human confirmation" in capsys.readouterr().err
    gate = ApprovalGate(path=approvals_file)
    assert len(gate.pending_requests()) == 1


def test_approve_refused_on_wrong_token(tmp_path: Path, capsys, monkeypatch) -> None:
    """Typing anything but the agent:action pair aborts the approval."""
    requests = [_sample_request()]
    approvals_file = _make_gate(tmp_path, requests)
    monkeypatch.setattr(cli_approvals, "_TTY_READER", lambda prompt: "yes")

    exit_code = approvals_approve_command(
        _ns(approvals_file, agent="scout", action="git_push", because="looks fine")
    )
    assert exit_code == 2
    gate = ApprovalGate(path=approvals_file)
    assert len(gate.pending_requests()) == 1


def test_approve_not_found(tmp_path: Path, capsys) -> None:
    approvals_file = _make_gate(tmp_path)
    exit_code = approvals_approve_command(
        _ns(approvals_file, agent="ghost", action="nope", because="checked the diff")
    )
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "No pending request" in out


def test_approve_requires_rationale(tmp_path: Path, capsys) -> None:
    requests = [_sample_request()]
    approvals_file = _make_gate(tmp_path, requests)

    for because in (None, "", "   "):
        exit_code = approvals_approve_command(
            _ns(approvals_file, agent="scout", action="git_push", because=because)
        )
        assert exit_code == 2
        assert "--because" in capsys.readouterr().err

    # The request must remain pending — no rubber-stamp path.
    gate = ApprovalGate(path=approvals_file)
    assert len(gate.pending_requests()) == 1


# ---------------------------------------------------------------------------
# reject
# ---------------------------------------------------------------------------


def test_reject_flow(tmp_path: Path, capsys, monkeypatch) -> None:
    requests = [_sample_request()]
    approvals_file = _make_gate(tmp_path, requests)
    monkeypatch.setattr(cli_approvals, "_TTY_READER", lambda prompt: "scout:git_push")

    exit_code = approvals_reject_command(
        _ns(approvals_file, agent="scout", action="git_push", because="branch is frozen")
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Rejected" in out

    # Verify it's now rejected in the file, with the rationale stored.
    # CLI judgments carry broker-derived human provenance.
    gate = ApprovalGate(path=approvals_file)
    assert len(gate.pending_requests()) == 0
    assert gate._pending[0].status == "rejected"
    assert gate._pending[0].reviewed_by
    assert gate._pending[0].reviewed_via == "controlling_terminal"
    assert gate._pending[0].human_confirmed is True
    assert gate._pending[0].rationale == "branch is frozen"


def test_reject_not_found(tmp_path: Path, capsys) -> None:
    approvals_file = _make_gate(tmp_path)
    exit_code = approvals_reject_command(
        _ns(approvals_file, agent="ghost", action="nope", because="not a known agent")
    )
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "No pending request" in out


def test_reject_requires_rationale(tmp_path: Path, capsys) -> None:
    requests = [_sample_request()]
    approvals_file = _make_gate(tmp_path, requests)

    exit_code = approvals_reject_command(
        _ns(approvals_file, agent="scout", action="git_push", because="  ")
    )
    assert exit_code == 2
    assert "--because" in capsys.readouterr().err

    gate = ApprovalGate(path=approvals_file)
    assert len(gate.pending_requests()) == 1


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
    assert "Archived 2" in out
    assert "1 pending" in out

    # Verify only the pending request remains
    gate = ApprovalGate(path=approvals_file)
    assert len(gate._pending) == 1
    assert gate._pending[0].agent == "janitor"
    assert gate._pending[0].status == "pending"
    archived = [request for request in gate.all_requests() if request.status != "pending"]
    assert len(archived) == 2


def test_clear_nothing_to_clear(tmp_path: Path, capsys) -> None:
    requests = [_sample_request(status="pending")]
    approvals_file = _make_gate(tmp_path, requests)

    exit_code = approvals_clear_command(_ns(approvals_file))
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Archived 0" in out


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


def test_history_shows_rationale(tmp_path: Path, capsys, monkeypatch) -> None:
    requests = [_sample_request()]
    approvals_file = _make_gate(tmp_path, requests)
    monkeypatch.setattr(cli_approvals, "_TTY_READER", lambda prompt: "scout:git_push")
    approvals_approve_command(
        _ns(approvals_file, agent="scout", action="git_push", because="verified locally")
    )
    capsys.readouterr()

    exit_code = approvals_history_command(_ns(approvals_file))
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "because: verified locally" in out
    assert "(controlling_terminal)" in out  # reviewer provenance is visible
    assert "ref: gate_" in out  # calibration ref is visible


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


def test_approvals_file_routes_before_or_after_subcommand(tmp_path: Path) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parsers(subparsers)
    custom = str(tmp_path / "custom-approvals.json")

    before = parser.parse_args(
        ["approvals", "--approvals-file", custom, "list"]
    )
    after = parser.parse_args(
        ["approvals", "list", "--approvals-file", custom]
    )

    assert before.approvals_file == custom
    assert after.approvals_file == custom


# ---------------------------------------------------------------------------
# --request-id: pick one row when several share an agent/action pair
# ---------------------------------------------------------------------------


def _two_same_pair(tmp_path: Path) -> tuple[Path, list[ApprovalRequest]]:
    """Two pending requests with the same agent/action but different detail —
    the mission-runner shape (one `mission_phase_decide` request per mission)."""
    first = _sample_request(
        agent="mission-runner", action="mission_phase_act", detail="Mission 'git-hygiene'"
    )
    first.request_id = "gate_first000000"
    second = _sample_request(
        agent="mission-runner", action="mission_phase_act", detail="Mission 'pr-review-prep'"
    )
    second.request_id = "gate_second00000"
    return _make_gate(tmp_path, [first, second]), [first, second]


def test_list_shows_request_id(tmp_path: Path, capsys) -> None:
    approvals_file, _ = _two_same_pair(tmp_path)
    assert approvals_list_command(_ns(approvals_file)) == 0
    out = capsys.readouterr().out
    assert "REQUEST" in out
    assert "gate_first000000" in out
    assert "gate_second00000" in out


def test_approve_by_request_id_picks_that_row(tmp_path: Path, capsys, monkeypatch) -> None:
    approvals_file, _ = _two_same_pair(tmp_path)
    monkeypatch.setattr(
        cli_approvals, "_TTY_READER", lambda prompt: "mission-runner:mission_phase_act"
    )

    exit_code = approvals_approve_command(
        _ns(
            approvals_file,
            agent=None,
            action=None,
            request_id="gate_second00000",
            because="pr drafts are fine to act on",
        )
    )
    assert exit_code == 0
    assert "gate_second00000" in capsys.readouterr().out

    gate = ApprovalGate(path=approvals_file)
    by_id = {r.request_id: r for r in gate._pending}
    assert by_id["gate_second00000"].status == "approved"
    assert by_id["gate_second00000"].human_confirmed is True
    assert by_id["gate_first000000"].status == "pending"


def test_reject_by_request_id_picks_that_row(tmp_path: Path, capsys, monkeypatch) -> None:
    approvals_file, _ = _two_same_pair(tmp_path)
    monkeypatch.setattr(
        cli_approvals, "_TTY_READER", lambda prompt: "mission-runner:mission_phase_act"
    )

    exit_code = approvals_reject_command(
        _ns(
            approvals_file,
            agent=None,
            action=None,
            request_id="gate_first000000",
            because="stale since July",
        )
    )
    assert exit_code == 0

    gate = ApprovalGate(path=approvals_file)
    by_id = {r.request_id: r for r in gate._pending}
    assert by_id["gate_first000000"].status == "rejected"
    assert by_id["gate_second00000"].status == "pending"


def test_approve_unknown_request_id(tmp_path: Path, capsys) -> None:
    approvals_file, _ = _two_same_pair(tmp_path)
    exit_code = approvals_approve_command(
        _ns(approvals_file, agent=None, action=None, request_id="gate_nope", because="x")
    )
    assert exit_code == 1
    assert "request_id=gate_nope" in capsys.readouterr().out


def test_approve_needs_id_or_pair(tmp_path: Path, capsys) -> None:
    approvals_file, _ = _two_same_pair(tmp_path)
    exit_code = approvals_approve_command(
        _ns(approvals_file, agent="mission-runner", action=None, request_id=None, because="x")
    )
    assert exit_code == 2
    assert "--request-id" in capsys.readouterr().err


def test_parser_accepts_request_id_without_pair() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    register_parsers(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        ["approvals", "approve", "--request-id", "gate_abc", "--because", "ok"]
    )
    assert args.agent is None and args.action is None
    assert args.request_id == "gate_abc"
    args = parser.parse_args(["approvals", "reject", "myagent", "deploy", "--because", "no"])
    assert args.request_id is None
