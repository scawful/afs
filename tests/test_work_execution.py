from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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
    from afs.human_provenance import _broker_for_reader

    authorization = _broker_for_reader(lambda _prompt: approval_id).confirm_token(
        approval_id,
        "prompt",
        scope=store.human_authorization_scope("approve", approval_id, "reviewed"),
    )
    assert authorization is not None
    assert store.approve_human(
        approval_id, rationale="reviewed", authorization=authorization
    )
    return approval_id


def test_action_requires_human_ack_classification() -> None:
    assert action_requires_human_ack("send_email") is True
    assert action_requires_human_ack("post_pr_comment") is True
    assert action_requires_human_ack("internal_note") is False
    assert action_requires_human_ack("") is False


@pytest.mark.parametrize("change", ["text", "target", "split", "attachment"])
def test_content_approval_rejects_changed_dispatch(tmp_path: Path, change: str) -> None:
    from afs.approval_content import validate_approved_content

    store = WorkAssistantStore(tmp_path / "context")
    approval_id = _approved_action(store, preview={"text": "hello\nworld", "attachment": "abc"})
    approval = store.get_approval(approval_id)
    assert approval is not None
    args = {
        "target_system": approval["target_system"], "target_id": approval["target_id"],
        "action": approval["action"], "preview": dict(approval["preview"]),
    }
    assert validate_approved_content(approval, **args) == approval["content_sha256"]
    if change == "target":
        args["target_id"] = "different-destination"
    elif change == "split":
        args["preview"]["text"] = ["hello", "world"]
    else:
        args["preview"][change] = "changed"
    with pytest.raises(PermissionError, match="differs"):
        validate_approved_content(approval, **args)


def test_executor_rejects_tampered_stored_content(tmp_path: Path, monkeypatch) -> None:
    store = WorkAssistantStore(tmp_path / "context")
    approval_id = _approved_action(store)
    with store._connect() as connection:
        connection.execute(
            "UPDATE approvals SET preview_json = ? WHERE approval_id = ?",
            ('{"text":"unapproved"}', approval_id),
        )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("executor must not run")

    monkeypatch.setattr("afs.work_execution.subprocess.run", forbidden)
    with pytest.raises(PermissionError, match="differs"):
        execute_approved_action(
            store, context_root=tmp_path / "context", approval_id=approval_id,
            executor_command=[sys.executable], require_human_ack=False,
        )


def test_dedupe_key_cannot_reuse_different_content(tmp_path: Path) -> None:
    store = WorkAssistantStore(tmp_path / "context")
    args = {"target_system": "chat", "target_id": "thread", "action": "send", "summary": "send", "dedupe_key": "key"}
    store.create_approval(**args, preview={"text": "one"})
    assert store.create_approval(**args, preview={"text": "one"}) == "key"
    with pytest.raises(ValueError, match="different content"):
        store.create_approval(**args, preview={"text": "two"})


def test_claimed_payload_is_revalidated_and_claim_released(tmp_path: Path, monkeypatch) -> None:
    store = WorkAssistantStore(tmp_path / "context")
    approval_id = _approved_action(store)
    claim = store.claim_approval_execution

    def changed_claim(key):
        row = claim(key)
        assert row is not None
        row["preview"] = {"text": "changed between read and claim"}
        return row

    monkeypatch.setattr(store, "claim_approval_execution", changed_claim)
    with pytest.raises(PermissionError, match="differs"):
        execute_approved_action(
            store, context_root=tmp_path / "context", approval_id=approval_id,
            executor_command=[sys.executable], require_human_ack=False,
        )
    assert store.get_approval(approval_id)["status"] == "approved"


def test_legacy_approved_content_requires_a_new_human_decision(tmp_path: Path) -> None:
    from afs.human_provenance import _broker_for_reader

    root = tmp_path / "context"
    store = WorkAssistantStore(root)
    approval_id = _approved_action(store)
    with store._connect() as connection:
        connection.execute("UPDATE approvals SET content_sha256 = '' WHERE approval_id = ?", (approval_id,))
    migrated = WorkAssistantStore(root)
    assert migrated.get_approval(approval_id)["status"] == "pending"
    authorization = _broker_for_reader(lambda _prompt: approval_id).confirm_token(
        approval_id, "prompt",
        scope=migrated.human_authorization_scope("approve", approval_id, "reviewed content"),
    )
    assert migrated.approve_human(
        approval_id, rationale="reviewed content", authorization=authorization
    )
    assert len(migrated.get_approval(approval_id)["content_sha256"]) == 64


def test_changed_pending_content_can_still_be_rejected(tmp_path: Path) -> None:
    store = WorkAssistantStore(tmp_path / "context")
    approval_id = store.create_approval(
        target_system="chat", target_id="thread", action="send", summary="send",
        preview={"text": "original"},
    )
    with store._connect() as connection:
        connection.execute(
            "UPDATE approvals SET preview_json = ? WHERE approval_id = ?",
            ('{"text":"changed"}', approval_id),
        )
    assert store.reject(approval_id, rationale="content changed") is True
    assert store.get_approval(approval_id)["status"] == "rejected"


def test_empty_preview_values_remain_distinct_content(tmp_path: Path) -> None:
    store = WorkAssistantStore(tmp_path / "context")
    hashes = []
    for preview in ("", [], {}, False, 0):
        approval_id = store.create_approval(
            target_system="local", target_id="target", action="write", summary="write",
            preview=preview,
        )
        approval = store.get_approval(approval_id)
        assert approval is not None
        assert type(approval["preview"]) is type(preview)
        hashes.append(approval["content_sha256"])
    assert len(set(hashes)) == len(hashes)


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
    # Even a human-confirmed row demands a fresh execution-time ack for an
    # outward action; a headless agent cannot satisfy it.
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


def test_execute_claim_allows_only_one_concurrent_connector_run(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    approval_id = _approved_action(
        store, action="internal_note", target_system="local"
    )
    marker = tmp_path / "executions.txt"
    start = threading.Barrier(2)

    def invoke() -> tuple[str, object]:
        start.wait(timeout=5)
        try:
            result = execute_approved_action(
                store,
                context_root=context_root,
                approval_id=approval_id,
                executor_command=[
                    sys.executable,
                    "-c",
                    (
                        "import pathlib,sys,time; time.sleep(0.25); "
                        "pathlib.Path(sys.argv[1]).open('a').write('ran\\n')"
                    ),
                    str(marker),
                ],
                require_human_ack=False,
            )
        except WorkApprovalExecutionError as exc:
            return "blocked", str(exc)
        return "applied", result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=10) for future in [pool.submit(invoke), pool.submit(invoke)]]

    assert sorted(kind for kind, _value in results) == ["applied", "blocked"]
    assert marker.read_text(encoding="utf-8").splitlines() == ["ran"]
    assert store.get_approval(approval_id)["status"] == "applied"  # type: ignore[index]


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
    from afs.human_provenance import _broker_for_reader

    authorization = _broker_for_reader(lambda _prompt: approval_id).confirm_token(
        approval_id,
        "prompt",
        scope=store.human_authorization_scope("approve", approval_id, "reviewed"),
    )
    assert authorization is not None
    assert store.approve_human(
        approval_id, rationale="reviewed", authorization=authorization
    )

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


def test_programmatic_approval_never_authorizes_execution(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = WorkAssistantStore(context_root)
    approval_id = store.create_approval(
        target_system="local",
        target_id="note",
        action="internal_note",
        summary="Claimed approval",
    )
    assert store.approve(
        approval_id, approved_by="human", rationale="claimed terminal review"
    )

    with pytest.raises(HumanApprovalRequiredError, match="programmatically"):
        execute_approved_action(
            store,
            context_root=context_root,
            approval_id=approval_id,
            executor_command=[sys.executable, "-c", "print('must not run')"],
            require_human_ack=False,
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
