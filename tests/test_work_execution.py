from __future__ import annotations

import sys
from pathlib import Path

import pytest

from afs.work_assistant import WorkAssistantStore
from afs.work_execution import (
    HumanApprovalRequiredError,
    WorkApprovalExecutionError,
    action_requires_human_ack,
    approval_requires_human_ack,
    confirm_human_approval,
    execute_approved_action,
)


def _approved_action(
    store: WorkAssistantStore,
    *,
    action: str = "edit_doc",
    target_system: str = "google-docs",
    preview: dict[str, str] | None = None,
) -> str:
    approval_id = store.create_approval(
        target_system=target_system,
        target_id="doc-1",
        action=action,
        summary="Apply approved edit",
        preview=preview or {"diff": "-old\n+new"},
        permission_required="doc edit approval",
    )
    assert store.approve(approval_id, approved_by="human")
    return approval_id


def test_action_requires_human_ack_classification() -> None:
    assert action_requires_human_ack("send_email") is True
    assert action_requires_human_ack("post_pr_comment") is True
    assert action_requires_human_ack("internal_note") is False
    assert action_requires_human_ack("") is False


def test_action_requires_human_ack_covers_generic_and_novel_outward_actions() -> None:
    # The generic sentinel stamped on gated approvals with no specific verb used to
    # slip past the gate; it must now require confirmation.
    assert action_requires_human_ack("external_write") is True
    assert action_requires_human_ack("external-write") is True
    # A novel outward action from a future connector, not on the enumerated list.
    assert action_requires_human_ack("escalate_incident") is True
    assert action_requires_human_ack("page_oncall") is True
    assert action_requires_human_ack("delete_ticket") is True
    assert action_requires_human_ack("update_crm_record") is True
    assert action_requires_human_ack("archive_ticket") is True
    assert action_requires_human_ack("remove_user") is True
    # Token-matched, so a benign name whose substring contains an outward stem
    # ("preview" ⊃ "review") is NOT misclassified.
    assert action_requires_human_ack("preview_doc") is False
    assert action_requires_human_ack("read_ticket") is False


def test_approval_requires_human_ack_uses_external_target_backstop() -> None:
    assert approval_requires_human_ack(
        {"action": "internal_note", "target_system": "google-docs"}
    ) is True
    assert approval_requires_human_ack(
        {"action": "internal_note", "target_system": "local"}
    ) is False


def test_confirm_human_approval_noop_for_internal_action() -> None:
    def _reader(_prompt: str) -> str | None:
        raise AssertionError("reader must not be consulted for a non-external action")

    # Should return without raising and without touching the reader.
    confirm_human_approval(
        {"action": "internal_note", "approval_id": "a1", "target_system": "local"},
        reader=_reader,
    )


def test_confirm_human_approval_accepts_matching_id() -> None:
    approval = {"action": "send_email", "approval_id": "approval_abc"}
    confirm_human_approval(approval, reader=lambda _prompt: "approval_abc")


def test_confirm_human_approval_rejects_mismatch() -> None:
    approval = {"action": "send_email", "approval_id": "approval_abc"}
    with pytest.raises(HumanApprovalRequiredError, match="did not match"):
        confirm_human_approval(approval, reader=lambda _prompt: "nope")


def test_confirm_human_approval_refuses_without_terminal() -> None:
    approval = {"action": "send_email", "approval_id": "approval_abc"}
    # reader returns None → no controlling terminal available (agent context).
    with pytest.raises(HumanApprovalRequiredError, match="no terminal"):
        confirm_human_approval(approval, reader=lambda _prompt: None)


def test_execute_approved_action_marks_success_applied(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    approval_id = _approved_action(store)

    result = execute_approved_action(
        store,
        context_root=context_root,
        approval_id=approval_id,
        executor_command=[
            sys.executable,
            "-c",
            (
                "import json,sys; "
                "payload=json.load(open(sys.argv[-1])); "
                "print(json.dumps({'seen': payload['approval']['approval_id']}))"
            ),
        ],
        actor="test-agent",
        confirm_reader=lambda _prompt: approval_id,
    )

    assert result["status"] == "applied"
    assert result["output"]["seen"] == approval_id
    approval = store.get_approval(approval_id)
    assert approval is not None
    assert approval["status"] == "applied"
    assert approval["result"]["output"]["seen"] == approval_id
    assert store.list_activity()[0]["activity_type"] == "approval_applied"
    assert store.list_communication_samples() == []


def test_execute_approved_comment_records_style_sample(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    approval_id = _approved_action(
        store,
        action="post_pr_comment",
        preview={"body": "Thanks — I tightened the guardrail and added focused tests."},
    )

    result = execute_approved_action(
        store,
        context_root=context_root,
        approval_id=approval_id,
        executor_command=[
            sys.executable,
            "-c",
            "import json; print(json.dumps({'ok': True}))",
        ],
        actor="test-agent",
        confirm_reader=lambda _prompt: approval_id,
    )

    assert result["status"] == "applied"
    samples = store.list_communication_samples(purpose="post_pr_comment")
    assert len(samples) == 1
    assert "tightened the guardrail" in samples[0]["text_excerpt"]
    assert samples[0]["style_notes"] == ["approved external write"]
    assert samples[0]["provenance"][0]["approval_id"] == approval_id
    assert store.list_activity()[0]["metadata"]["communication_sample_id"] == samples[0]["sample_id"]


def test_execute_approved_action_dry_run_returns_payload(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    approval_id = _approved_action(store)

    result = execute_approved_action(
        store,
        context_root=context_root,
        approval_id=approval_id,
        executor_command=[],
        actor="test-agent",
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["payload"]["approval"]["approval_id"] == approval_id
    assert store.get_approval(approval_id)["status"] == "approved"  # type: ignore[index]


def test_execute_external_write_refused_without_terminal(tmp_path: Path) -> None:
    # A row can reach `approved` by any path (here store.approve directly, mimicking a
    # pre-existing row or a non-CLI approve). Execution must still demand a human ack,
    # and a headless agent (reader returns None) cannot satisfy it.
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    approval_id = _approved_action(store, action="post_pr_comment")

    with pytest.raises(HumanApprovalRequiredError, match="no terminal"):
        execute_approved_action(
            store,
            context_root=context_root,
            approval_id=approval_id,
            executor_command=[sys.executable, "-c", "print('should not run')"],
            confirm_reader=lambda _prompt: None,
        )
    # The outward action never ran; the approval stays approved (retryable), not applied.
    approval = store.get_approval(approval_id)
    assert approval is not None
    assert approval["status"] == "approved"


def test_execute_generic_external_write_sentinel_is_gated(tmp_path: Path) -> None:
    # The generic sentinel used to slip past the classifier; execution must gate it.
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    approval_id = _approved_action(store, action="external_write")

    with pytest.raises(HumanApprovalRequiredError):
        execute_approved_action(
            store,
            context_root=context_root,
            approval_id=approval_id,
            executor_command=[sys.executable, "-c", "print('nope')"],
            confirm_reader=lambda _prompt: None,
        )


def test_execute_internal_action_needs_no_terminal(tmp_path: Path) -> None:
    # A non-outward action executes without a terminal — the gate is scoped to
    # external writes so it never blocks legitimate internal automation.
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    approval_id = _approved_action(store, action="internal_note", target_system="local")

    result = execute_approved_action(
        store,
        context_root=context_root,
        approval_id=approval_id,
        executor_command=[sys.executable, "-c", "import json; print(json.dumps({'ok': True}))"],
        confirm_reader=lambda _prompt: None,  # would refuse if consulted
    )
    assert result["status"] == "applied"


def test_execute_external_target_backstop_refuses_unclassified_action(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    approval_id = store.create_approval(
        target_system="zendesk",
        target_id="ticket-1",
        action="internal_note",
        summary="Misclassified external update",
    )
    assert store.approve(approval_id, approved_by="human")

    with pytest.raises(HumanApprovalRequiredError, match="no terminal"):
        execute_approved_action(
            store,
            context_root=context_root,
            approval_id=approval_id,
            executor_command=[sys.executable, "-c", "print('should not run')"],
            confirm_reader=lambda _prompt: None,
        )


def test_execute_requires_approved_status(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    approval_id = store.create_approval(
        target_system="zendesk",
        target_id="ticket-1",
        action="post_ticket_comment",
        summary="Post reply",
    )

    with pytest.raises(WorkApprovalExecutionError, match="must be approved"):
        execute_approved_action(
            store,
            context_root=context_root,
            approval_id=approval_id,
            executor_command=[sys.executable, "-c", "print('nope')"],
        )


def test_execute_failure_leaves_approval_retryable(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    approval_id = _approved_action(store)

    result = execute_approved_action(
        store,
        context_root=context_root,
        approval_id=approval_id,
        executor_command=[sys.executable, "-c", "import sys; print('failed'); sys.exit(7)"],
        confirm_reader=lambda _prompt: approval_id,
    )

    assert result["status"] == "failed"
    assert result["returncode"] == 7
    approval = store.get_approval(approval_id)
    assert approval is not None
    assert approval["status"] == "approved"
    assert approval["result"]["returncode"] == 7
    assert store.list_activity()[0]["activity_type"] == "approval_execution_failed"
