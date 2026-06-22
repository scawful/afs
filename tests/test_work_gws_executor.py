from __future__ import annotations

import json
from pathlib import Path

import pytest

from afs.work_gws_executor import GWSApprovalExecutorError, execute_gws_approval_payload


class _FakeGWSClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def gmail_send(self, *, to: str, subject: str, body: str) -> dict:
        self.calls.append(("gmail_send", {"to": to, "subject": subject, "body": body}))
        return {"id": "message-1"}

    def sheets_append(self, *, spreadsheet_id: str, range: str, values: list[list[str]]) -> dict:
        self.calls.append(
            (
                "sheets_append",
                {"spreadsheet_id": spreadsheet_id, "range": range, "values": values},
            )
        )
        return {"updates": {"updatedRows": len(values)}}

    def calendar_event_create(self, *, summary: str, start: str, end: str, **kwargs) -> dict:
        self.calls.append(
            (
                "calendar_event_create",
                {"summary": summary, "start": start, "end": end, **kwargs},
            )
        )
        return {"id": "event-1"}


def _payload(action: str, *, target_system: str = "google-workspace", preview: dict) -> dict:
    return {
        "version": 1,
        "context_root": "/tmp/context",
        "actor": "test-agent",
        "approval": {
            "approval_id": "approval_1",
            "status": "approved",
            "target_system": target_system,
            "target_id": "target-1",
            "action": action,
            "summary": "Approved action",
            "preview": preview,
        },
    }


def test_execute_gws_send_email_from_approved_payload() -> None:
    client = _FakeGWSClient()

    result = execute_gws_approval_payload(
        _payload(
            "send_email",
            target_system="gmail",
            preview={"to": "person@example.com", "subject": "Hello", "body": "Approved body."},
        ),
        client=client,  # type: ignore[arg-type]
    )

    assert result["ok"] is True
    assert result["gws_result"]["id"] == "message-1"
    assert client.calls == [
        (
            "gmail_send",
            {"to": "person@example.com", "subject": "Hello", "body": "Approved body."},
        )
    ]


def test_execute_gws_append_sheet_rows_from_approved_payload() -> None:
    client = _FakeGWSClient()

    result = execute_gws_approval_payload(
        _payload(
            "append_sheet_rows",
            target_system="google-sheets",
            preview={"range": "Sheet1!A:B", "values": [["name", "status"], ["Dana", "ready"]]},
        ),
        client=client,  # type: ignore[arg-type]
    )

    assert result["gws_result"]["updates"]["updatedRows"] == 2
    assert client.calls[0][0] == "sheets_append"
    assert client.calls[0][1]["spreadsheet_id"] == "target-1"


def test_execute_gws_create_calendar_event_from_approved_payload() -> None:
    client = _FakeGWSClient()

    result = execute_gws_approval_payload(
        _payload(
            "create_calendar_event",
            target_system="google-calendar",
            preview={
                "summary": "Planning review",
                "start": "2026-04-28T13:00:00-04:00",
                "end": "2026-04-28T13:30:00-04:00",
                "attendees": [{"email": "person@example.com"}],
            },
        ),
        client=client,  # type: ignore[arg-type]
    )

    assert result["gws_result"]["id"] == "event-1"
    assert client.calls[0][1]["attendees"] == [{"email": "person@example.com"}]


def test_execute_gws_rejects_unapproved_payload() -> None:
    payload = _payload("send_email", preview={"to": "a@b.com", "subject": "s", "body": "b"})
    payload["approval"]["status"] = "pending"

    with pytest.raises(GWSApprovalExecutorError, match="must be approved"):
        execute_gws_approval_payload(payload, client=_FakeGWSClient())  # type: ignore[arg-type]


def test_execute_gws_rejects_unsupported_action() -> None:
    with pytest.raises(GWSApprovalExecutorError, match="unsupported gws action"):
        execute_gws_approval_payload(
            _payload("delete_drive_file", preview={}),
            client=_FakeGWSClient(),  # type: ignore[arg-type]
        )


def test_gws_executor_script_reports_errors(tmp_path: Path, capsys) -> None:
    from afs.work_gws_executor import main

    payload_path = tmp_path / "approval.json"
    payload_path.write_text(
        json.dumps(_payload("send_email", preview={"to": "a@b.com"})),
        encoding="utf-8",
    )

    assert main([str(payload_path)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
