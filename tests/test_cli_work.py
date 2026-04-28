from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from afs.cli.work import (
    approvals_approve_command,
    approvals_list_command,
    approvals_request_command,
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


def test_work_approval_request_and_approve(tmp_path: Path, capsys) -> None:
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

    assert approvals_approve_command(_args(context_root, approval_id=approval_id, by="human")) == 0
    assert "Approved" in capsys.readouterr().out


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
