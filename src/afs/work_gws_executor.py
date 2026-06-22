"""Google Workspace executor for approved AFS actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .gws import GWSClient, GWSConfirmationRequired, GWSPolicyError, _load_policy_from_config

SUPPORTED_ACTIONS = frozenset(
    {
        "append_sheet_rows",
        "create_calendar_event",
        "send_email",
    }
)
SUPPORTED_TARGET_SYSTEMS = frozenset(
    {
        "calendar",
        "gmail",
        "google-calendar",
        "google-gmail",
        "google-sheets",
        "google-workspace",
        "gws",
        "sheets",
    }
)


class GWSApprovalExecutorError(RuntimeError):
    """Raised when a Google Workspace approval payload cannot be executed."""


def execute_gws_approval_payload(
    payload: dict[str, Any],
    *,
    client: GWSClient | None = None,
) -> dict[str, Any]:
    """Execute one approved Google Workspace action from an AFS approval payload."""
    approval = _approval(payload)
    _validate_approval(approval)
    action = str(approval.get("action") or "").strip()
    details = _approval_details(approval)
    client = client or GWSClient(
        policy_config=_load_policy_from_config(),
        confirm_callback=_approved_confirmation,
    )

    if action == "send_email":
        result = _send_email(client, approval, details)
    elif action == "append_sheet_rows":
        result = _append_sheet_rows(client, approval, details)
    elif action == "create_calendar_event":
        result = _create_calendar_event(client, approval, details)
    else:
        raise GWSApprovalExecutorError(f"unsupported gws action: {action}")

    if result is None:
        raise GWSApprovalExecutorError("gws command failed or returned no JSON")
    return {
        "ok": True,
        "approval_id": approval["approval_id"],
        "target_system": approval["target_system"],
        "target_id": approval["target_id"],
        "action": action,
        "gws_result": result,
    }


def _approval(payload: dict[str, Any]) -> dict[str, Any]:
    approval = payload.get("approval")
    if not isinstance(approval, dict):
        raise GWSApprovalExecutorError("payload missing approval object")
    return approval


def _validate_approval(approval: dict[str, Any]) -> None:
    status = str(approval.get("status") or "").strip()
    if status != "approved":
        raise GWSApprovalExecutorError(f"approval must be approved, got: {status or 'unknown'}")
    target_system = str(approval.get("target_system") or "").strip()
    if target_system not in SUPPORTED_TARGET_SYSTEMS:
        raise GWSApprovalExecutorError(f"unsupported gws target_system: {target_system}")
    action = str(approval.get("action") or "").strip()
    if action not in SUPPORTED_ACTIONS:
        raise GWSApprovalExecutorError(f"unsupported gws action: {action}")


def _approval_details(approval: dict[str, Any]) -> dict[str, Any]:
    preview = approval.get("preview")
    if isinstance(preview, dict):
        details = dict(preview)
        nested = details.get("gws")
        if isinstance(nested, dict):
            details.update(nested)
        return details
    return {}


def _send_email(
    client: GWSClient,
    approval: dict[str, Any],
    details: dict[str, Any],
) -> Any:
    to = _required_text(details, "to", aliases=("recipient", "email"))
    subject = _required_text(details, "subject")
    body = _required_text(details, "body", aliases=("text", "message"))
    return client.gmail_send(to=to, subject=subject, body=body)


def _append_sheet_rows(
    client: GWSClient,
    approval: dict[str, Any],
    details: dict[str, Any],
) -> Any:
    spreadsheet_id = str(details.get("spreadsheet_id") or approval.get("target_id") or "").strip()
    if not spreadsheet_id:
        raise GWSApprovalExecutorError("append_sheet_rows requires target_id or preview.spreadsheet_id")
    range_name = _required_text(details, "range")
    values = details.get("values", details.get("rows"))
    if not isinstance(values, list) or not values or not all(isinstance(row, list) for row in values):
        raise GWSApprovalExecutorError("append_sheet_rows requires preview.values as a non-empty list of rows")
    return client.sheets_append(spreadsheet_id=spreadsheet_id, range=range_name, values=values)


def _create_calendar_event(
    client: GWSClient,
    approval: dict[str, Any],
    details: dict[str, Any],
) -> Any:
    summary = str(details.get("summary") or approval.get("summary") or "").strip()
    if not summary:
        raise GWSApprovalExecutorError("create_calendar_event requires preview.summary or approval summary")
    start = _required_text(details, "start", aliases=("start_time", "start_datetime"))
    end = _required_text(details, "end", aliases=("end_time", "end_datetime"))
    extras: dict[str, Any] = {}
    for key in ("description", "location", "attendees", "conferenceData", "reminders"):
        if key in details:
            extras[key] = details[key]
    return client.calendar_event_create(summary=summary, start=start, end=end, **extras)


def _required_text(
    details: dict[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
) -> str:
    for candidate in (key, *aliases):
        value = str(details.get(candidate) or "").strip()
        if value:
            return value
    choices = ", ".join((key, *aliases))
    raise GWSApprovalExecutorError(f"missing required preview field: {choices}")


def _approved_confirmation(_decision: Any) -> bool:
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute one approved AFS Google Workspace action.")
    parser.add_argument("approval_json", help="AFS work approval JSON payload file.")
    args = parser.parse_args(argv)

    payload_path = Path(args.approval_json).expanduser()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        result = execute_gws_approval_payload(payload)
    except (GWSApprovalExecutorError, GWSPolicyError, GWSConfirmationRequired, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1

    print(json.dumps(result, sort_keys=True))
    return 0
