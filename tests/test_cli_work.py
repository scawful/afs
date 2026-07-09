from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

from afs.cli.work import (
    approvals_approve_command,
    approvals_execute_command,
    approvals_list_command,
    approvals_request_command,
    approvals_show_command,
    communication_add_command,
    communication_guide_command,
    communication_list_command,
    communication_preflight_command,
    people_list_command,
    register_parsers,
    reviewers_command,
    work_summary_command,
)
from afs.work_assistant import WorkAssistantStore


def _args(context_root: Path, **kwargs) -> Namespace:
    values = {
        "config": None,
        "path": None,
        "context_root": str(context_root),
        "context_dir": None,
    }
    values.update(kwargs)
    return Namespace(**values)


def test_work_summary_and_people_json(tmp_path: Path, capsys) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    store.upsert_person({"display_name": "Reviewer", "email": "reviewer@example.com"})

    assert work_summary_command(_args(context_root, json=True)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["people"] == 1
    assert summary["context_path"] == str(context_root)

    assert people_list_command(_args(context_root, json=True, limit=10)) == 0
    people = json.loads(capsys.readouterr().out)
    assert people[0]["display_name"] == "Reviewer"


def test_reviewers_command_lists_native_routes(tmp_path: Path, capsys) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    person_id = store.upsert_person({"display_name": "Code Reviewer", "email": "reviewer@example.com"})
    store.upsert_review_route(
        person_id=person_id,
        scope_type="repo",
        scope_id="afs",
        target_type="code",
        reason="owns the package",
    )

    assert reviewers_command(
        _args(
            context_root,
            target_type="code",
            scope_type="repo",
            scope_id="afs",
            limit=10,
            json=True,
        )
    ) == 0
    reviewers = json.loads(capsys.readouterr().out)
    assert reviewers[0]["display_name"] == "Code Reviewer"
    assert reviewers[0]["reason"] == "owns the package"


def test_communication_list_command(tmp_path: Path, capsys) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    person_id = store.upsert_person({"display_name": "Doc Author", "email": "author@example.com"})
    store.record_communication_sample(
        person_id=person_id,
        source_system="google-docs",
        source_id="doc-1",
        channel="design_doc",
        purpose="design_feedback",
        text="Prefer a concise findings-first response with exact evidence.",
        style_notes=["findings-first"],
    )

    assert communication_list_command(
        _args(
            context_root,
            person_id=None,
            purpose="design_feedback",
            limit=10,
            json=True,
        )
    ) == 0
    samples = json.loads(capsys.readouterr().out)
    assert samples[0]["display_name"] == "Doc Author"
    assert samples[0]["style_notes"] == ["findings-first"]


def test_communication_add_and_guide_commands(tmp_path: Path, capsys) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()

    assert communication_add_command(
        _args(
            context_root,
            text="Short direct comment with exact file evidence.",
            text_file=None,
            person_id="",
            source_system="github",
            source_id="comment-1",
            channel="pr_review",
            purpose="responding_to_comments",
            style_note=["direct", "evidence-backed"],
            provenance_json='{"url":"https://example.test/comment-1"}',
            confidence=0.8,
            dedupe_key=None,
            json=True,
        )
    ) == 0
    sample_id = json.loads(capsys.readouterr().out)["sample_id"]
    assert sample_id

    assert communication_guide_command(
        _args(
            context_root,
            person_id=None,
            purpose="responding_to_comments",
            limit=10,
            json=True,
        )
    ) == 0
    guide = json.loads(capsys.readouterr().out)
    assert guide["sample_count"] == 1
    assert guide["style_notes"] == ["direct", "evidence-backed"]
    assert any("explicit approval" in line for line in guide["guidance"])


def test_communication_preflight_command_loads_personal_context(tmp_path: Path, capsys) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    personal_root = tmp_path / "personal"
    personal_root.mkdir()
    (personal_root / "profile.toml").write_text('name = "Test User"\n', encoding="utf-8")
    (personal_root / "style.md").write_text(
        "Use direct notes with exact evidence.",
        encoding="utf-8",
    )
    (personal_root / "manifest.toml").write_text(
        """
[modes.work]
tone = "direct"
work_context = true
load = ["style.md"]
style_instructions = ["findings first"]
communication_sources = ["approved review replies"]
posting_policy = "Ask before posting."
""".strip()
        + "\n",
        encoding="utf-8",
    )

    store = WorkAssistantStore(context_root)
    store.record_communication_sample(
        text="Short direct comment with exact file evidence.",
        source_system="github",
        source_id="comment-1",
        purpose="responding_to_comments",
        style_notes=["direct"],
    )

    assert communication_preflight_command(
        _args(
            context_root,
            person_id=None,
            purpose="responding_to_comments",
            limit=10,
            approval_limit=10,
            personal_mode="work",
            personal_context_root=str(personal_root),
            json=True,
        )
    ) == 0
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["style"]["sample_count"] == 1
    assert preflight["personal_context"]["mode"] == "work"
    assert preflight["personal_context"]["style_instructions"] == ["findings first"]
    assert preflight["approval_guardrail"]["requires_explicit_approval"] is True


def test_work_approval_request_and_approve(tmp_path: Path, capsys, monkeypatch) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()

    assert approvals_request_command(
        _args(
            context_root,
            target_system="zendesk",
            target_id="ticket-1",
            action="post_ticket_comment",
            summary="Send drafted support reply",
            preview="Thanks for the report.",
            preview_json=None,
            affected_person=["requester@example.com"],
            risk_level="medium",
            permission_required="ticket comment approval",
            requested_by="agent",
            json=True,
        )
    ) == 0
    created = json.loads(capsys.readouterr().out)
    approval_id = created["approval_id"]

    assert approvals_list_command(_args(context_root, status="pending", all=False, limit=10, json=True)) == 0
    approvals = json.loads(capsys.readouterr().out)
    assert approvals[0]["action"] == "post_ticket_comment"

    # post_ticket_comment is an external write, so approve now requires interactive
    # human confirmation. Simulate the operator typing the approval id at the terminal.
    monkeypatch.setattr(
        "afs.work_execution._default_tty_reader",
        lambda tty_path: (lambda prompt: approval_id),
    )
    assert approvals_approve_command(_args(context_root, approval_id=approval_id, by="human")) == 0
    assert "Approved" in capsys.readouterr().out

    assert approvals_show_command(_args(context_root, approval_id=approval_id, json=True)) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["status"] == "approved"

    assert approvals_request_command(
        _args(
            context_root,
            target_system="gmail",
            target_id="message",
            action="send_email",
            summary="Send approved email",
            preview=None,
            preview_json='{"to":"person@example.com","subject":"Hi","body":"Approved."}',
            affected_person=[],
            risk_level="medium",
            permission_required="email approval",
            requested_by="agent",
            json=True,
        )
    ) == 0
    structured_id = json.loads(capsys.readouterr().out)["approval_id"]
    structured = WorkAssistantStore(context_root).get_approval(structured_id)
    assert structured is not None
    assert structured["preview"]["subject"] == "Hi"


def test_work_approval_external_write_refused_without_tty(tmp_path: Path, capsys, monkeypatch) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    approval_id = store.create_approval(
        target_system="gmail",
        target_id="message",
        action="send_email",
        summary="Send approved email",
    )
    # No controlling terminal (headless/agent context): the reader returns None.
    monkeypatch.setattr(
        "afs.work_execution._default_tty_reader",
        lambda tty_path: (lambda prompt: None),
    )
    assert approvals_approve_command(_args(context_root, approval_id=approval_id, by="human")) == 2
    assert "interactive human confirmation" in capsys.readouterr().out
    # The approval must remain pending — the agent could not self-approve.
    assert store.get_approval(approval_id)["status"] == "pending"  # type: ignore[index]


def test_work_approval_internal_action_skips_confirmation(tmp_path: Path, capsys) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    approval_id = store.create_approval(
        target_system="local",
        target_id="note-1",
        action="internal_note",  # not an external/communication write
        summary="Record an internal note",
    )
    # No tty monkeypatch: a non-external action must approve without any prompt.
    assert approvals_approve_command(_args(context_root, approval_id=approval_id, by="human")) == 0
    assert store.get_approval(approval_id)["status"] == "approved"  # type: ignore[index]


def test_work_approval_execute_command(tmp_path: Path, capsys, monkeypatch) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    approval_id = store.create_approval(
        target_system="google-docs",
        target_id="doc-1",
        action="edit_doc",
        summary="Apply approved edit",
    )
    assert store.approve(approval_id, approved_by="human")

    # edit_doc is an external write, so execution (where the connector actually fires)
    # now also requires a terminal confirmation. Simulate the operator at the terminal.
    monkeypatch.setattr(
        "afs.work_execution._default_tty_reader",
        lambda tty_path: (lambda prompt: approval_id),
    )
    assert approvals_execute_command(
        _args(
            context_root,
            approval_id=approval_id,
            actor="test-agent",
            timeout=10,
            dry_run=False,
            json=True,
            executor=(
                f"{sys.executable} -c "
                "'import json,sys; "
                "payload=json.load(open(sys.argv[-1])); "
                "print(json.dumps({\"approval\": payload[\"approval\"][\"approval_id\"]}))'"
            ),
        )
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "applied"
    assert result["output"]["approval"] == approval_id


def test_work_approval_execute_refused_without_tty(tmp_path: Path, capsys, monkeypatch) -> None:
    # Closes the "already-approved row" hole: even a row that is already `approved`
    # cannot execute an external write from a headless (no-terminal) context.
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    approval_id = store.create_approval(
        target_system="gmail",
        target_id="message",
        action="send_email",
        summary="Send approved email",
    )
    assert store.approve(approval_id, approved_by="human")
    monkeypatch.setattr(
        "afs.work_execution._default_tty_reader",
        lambda tty_path: (lambda prompt: None),  # no controlling terminal
    )
    assert approvals_execute_command(
        _args(
            context_root,
            approval_id=approval_id,
            actor="test-agent",
            timeout=10,
            dry_run=False,
            json=True,
            executor=f"{sys.executable} -c 'print(\"should not run\")'",
        )
    ) == 1
    assert "interactive human confirmation" in capsys.readouterr().out
    # The outward action never ran; the row stays approved (retryable).
    assert store.get_approval(approval_id)["status"] == "approved"  # type: ignore[index]


def test_register_parsers_includes_work_subcommands() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parsers(subparsers)

    args = parser.parse_args(["work", "reviewers", "--target-type", "docs"])
    assert args.command == "work"
    assert args.work_command == "reviewers"
    assert args.target_type == "docs"

    args = parser.parse_args(
        [
            "work",
            "approvals",
            "request",
            "--target-system",
            "docs",
            "--target-id",
            "doc-1",
            "--action",
            "edit_doc",
            "--summary",
            "Apply draft",
        ]
    )
    assert args.action == "edit_doc"

    args = parser.parse_args(
        [
            "work",
            "approvals",
            "execute",
            "approval_1",
            "--dry-run",
            "--executor",
            "python3 connector.py",
        ]
    )
    assert args.work_approvals_command == "execute"
    assert args.approval_id == "approval_1"
    assert args.executor == "python3 connector.py"

    args = parser.parse_args(["work", "communication", "list", "--purpose", "design_feedback"])
    assert args.work_command == "communication"
    assert args.communication_command == "list"
    assert args.purpose == "design_feedback"

    args = parser.parse_args(
        [
            "work",
            "communication",
            "add",
            "--text",
            "Use concrete evidence.",
            "--style-note",
            "direct",
        ]
    )
    assert args.communication_command == "add"
    assert args.text == "Use concrete evidence."
    assert args.style_note == ["direct"]

    args = parser.parse_args(["work", "communication", "guide", "--purpose", "docs"])
    assert args.communication_command == "guide"
    assert args.purpose == "docs"

    args = parser.parse_args(
        [
            "work",
            "communication",
            "preflight",
            "--purpose",
            "reply",
            "--personal-mode",
            "work",
        ]
    )
    assert args.communication_command == "preflight"
    assert args.purpose == "reply"
    assert args.personal_mode == "work"
