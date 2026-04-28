from __future__ import annotations

import sys
from pathlib import Path

import pytest

from afs.work_assistant import WorkAssistantStore
from afs.work_execution import WorkApprovalExecutionError, execute_approved_action


def _approved_action(store: WorkAssistantStore) -> str:
    approval_id = store.create_approval(
        target_system="google-docs",
        target_id="doc-1",
        action="edit_doc",
        summary="Apply approved edit",
        preview={"diff": "-old\n+new"},
        permission_required="doc edit approval",
    )
    assert store.approve(approval_id, approved_by="human")
    return approval_id


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
    )

    assert result["status"] == "applied"
    assert result["output"]["seen"] == approval_id
    approval = store.get_approval(approval_id)
    assert approval is not None
    assert approval["status"] == "applied"
    assert approval["result"]["output"]["seen"] == approval_id
    assert store.list_activity()[0]["activity_type"] == "approval_applied"


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
    )

    assert result["status"] == "failed"
    assert result["returncode"] == 7
    approval = store.get_approval(approval_id)
    assert approval is not None
    assert approval["status"] == "approved"
    assert approval["result"]["returncode"] == 7
    assert store.list_activity()[0]["activity_type"] == "approval_execution_failed"
