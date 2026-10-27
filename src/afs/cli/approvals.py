"""CLI commands for managing agent approval requests."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..agents.approvals import SYSTEM_RETENTION_DAYS
from ..agents.guardrails import ApprovalGate, ApprovalRequest
from ..human_provenance import (
    HumanAuthorization,
    _broker_for_reader,
)

# Test seam: production uses the platform controlling-terminal backend.
_TTY_READER = None

_PAYLOAD_PREVIEW_CHARS = 400


def _load_gate(args: argparse.Namespace) -> ApprovalGate:
    """Load ApprovalGate, optionally from a custom path."""
    path = Path(args.approvals_file) if getattr(args, "approvals_file", None) else None
    return ApprovalGate(path=path)


def _expires_in(expires_at: str) -> str:
    """Render an expiry as time left (``71h``, ``45m``) for the list table."""
    try:
        expires = datetime.fromisoformat(expires_at)
    except ValueError:
        return expires_at or "-"
    seconds = (expires - datetime.now(timezone.utc)).total_seconds()
    if seconds <= 0:
        return "now"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.0f}h"


def _payload_preview(payload: str) -> list[str]:
    if not payload:
        return []
    preview = payload[:_PAYLOAD_PREVIEW_CHARS]
    if len(payload) > _PAYLOAD_PREVIEW_CHARS:
        preview += " ...(see `afs approvals show`)"
    return [f"  payload: {preview}"]


def _confirm_gate_decision(
    gate: ApprovalGate,
    request: ApprovalRequest,
    decision: str,
    rationale: str,
) -> HumanAuthorization | None:
    """Interactive human confirmation for an agent-gate decision.

    Every request the gate queues is for a dangerous or unknown action, so
    a decision requires a person at a terminal: the operator must re-type
    the agent:action pair on the controlling tty, which piped stdin cannot
    satisfy. Returns a decision-scoped broker authorization, or ``None`` to
    refuse.
    """
    token = f"{request.agent}:{request.action}"
    prompt = "\n".join(
        [
            "",
            "=== HUMAN CONFIRMATION REQUIRED (agent approval) ===",
            f"  agent:   {request.agent}",
            f"  action:  {request.action}",
            f"  detail:  {request.detail}",
            f"  source:  {request.source or '-'}",
            f"  scope:   {request.scope}",
            f"  expires: {request.expires_at or '-'}",
            *_payload_preview(request.payload),
            f"  request: {request.request_id}",
            f"  because: {rationale}",
            f"Decision: {decision} this guarded action.",
            f"Type '{token}' to confirm, anything else aborts: ",
        ]
    )
    scope = gate.human_authorization_scope(
        decision, request.request_id, rationale
    )
    return _broker_for_reader(_TTY_READER).confirm_token(token, prompt, scope=scope)


def _require_rationale(args: argparse.Namespace, decision: str) -> str | None:
    """Return the stripped --because rationale, or None if missing/empty."""
    rationale = (getattr(args, "because", None) or "").strip()
    if rationale:
        return rationale
    print(
        f"A rationale is required to {decision}: pass --because "
        '"<why this is the right call>".\n'
        "It is stored in approvals history and resurfaced during calibration review.",
        file=sys.stderr,
    )
    return None


def _locate_request(
    gate: ApprovalGate, args: argparse.Namespace
) -> tuple[ApprovalRequest | None, int]:
    """Resolve the request a decision names: ``--request-id`` or ``agent action``.

    Returns ``(request, 0)`` on success, else ``(None, exit_code)`` after
    printing why. The id form is exact; the pair form takes the first pending
    match and is ambiguous when several requests share it.
    """
    request_id = (getattr(args, "request_id", None) or "").strip()
    agent = getattr(args, "agent", None) or ""
    action = getattr(args, "action", None) or ""
    if not request_id and not (agent and action):
        print(
            "Name the request: pass --request-id <gate_...> "
            "or both <agent> <action>.",
            file=sys.stderr,
        )
        return None, 2
    request = gate.find_pending(agent, action, request_id=request_id or None)
    if request is None:
        existing = gate.get(request_id) if request_id else None
        if existing is not None:
            print(f"Request {request_id} is {existing.status}, not pending.")
            return None, 1
        what = (
            f"request_id={request_id}" if request_id else f"agent={agent} action={action}"
        )
        print(f"No pending request found for {what}")
        return None, 1
    return request, 0


def _warn_legacy_skips(gate: ApprovalGate) -> None:
    skipped = gate.health()["legacy_skipped"]
    if skipped:
        print(
            f"warning: {skipped} malformed legacy approval record(s) were skipped "
            f"while importing beside {gate.path}",
            file=sys.stderr,
        )


def approvals_list_command(args: argparse.Namespace) -> int:
    """List pending approval requests."""
    gate = _load_gate(args)
    pending = gate.pending_requests()
    _warn_legacy_skips(gate)

    if getattr(args, "json", False):
        print(json.dumps([r.to_dict() for r in pending], indent=2))
        return 0

    if not pending:
        print("No pending approval requests.")
        return 0

    # Calculate column widths for alignment
    agent_w = max(len("AGENT"), *(len(r.agent) for r in pending))
    action_w = max(len("ACTION"), *(len(r.action) for r in pending))
    detail_w = max(len("DETAIL"), *(len(r.detail) for r in pending))
    source_w = max(len("SOURCE"), *(len(r.source or "-") for r in pending))

    header = (
        f"{'AGENT':<{agent_w}}  {'ACTION':<{action_w}}  {'DETAIL':<{detail_w}}  "
        f"{'SOURCE':<{source_w}}  {'EXPIRES':<7}  REQUEST"
    )
    print(header)
    print("-" * len(header))

    for req in pending:
        print(
            f"{req.agent:<{agent_w}}  {req.action:<{action_w}}  {req.detail:<{detail_w}}  "
            f"{req.source or '-':<{source_w}}  {_expires_in(req.expires_at):<7}  "
            f"{req.request_id}"
        )

    print()
    print(f"{len(pending)} pending request(s)")
    return 0


def approvals_show_command(args: argparse.Namespace) -> int:
    """Show one request in full, including its payload."""
    gate = _load_gate(args)
    request = gate.get(args.request_id)
    if request is None:
        print(f"No request found for request_id={args.request_id}")
        return 1
    if getattr(args, "json", False):
        print(json.dumps(request.to_dict(), indent=2))
        return 0
    rows = [
        ("request", request.request_id),
        ("agent", request.agent),
        ("action", request.action),
        ("detail", request.detail),
        ("source", request.source or "-"),
        ("run", request.run_id or "-"),
        ("scope", request.scope),
        ("status", request.status),
        ("created", request.timestamp),
        ("expires", request.expires_at or "-"),
    ]
    if request.reviewed_by:
        rows.append(
            ("reviewed", f"{request.reviewed_by} via {request.reviewed_via} at {request.reviewed_at}")
        )
    if request.rationale:
        rows.append(("because", request.rationale))
    if request.consumed_at:
        rows.append(("consumed", f"{request.consumed_at} by run {request.consumed_by_run or '-'}"))
    if request.revoked_at:
        rows.append(("revoked", f"{request.revoked_at}: {request.revoked_reason}"))
    if request.superseded_by:
        rows.append(("superseded by", request.superseded_by))
    for label, value in rows:
        print(f"{label + ':':<14} {value}")
    if request.payload:
        print("payload:")
        try:
            rendered = json.dumps(json.loads(request.payload), indent=2)
        except json.JSONDecodeError:
            rendered = request.payload
        for line in rendered.splitlines():
            print(f"  {line}")
    return 0


def approvals_approve_command(args: argparse.Namespace) -> int:
    """Approve a pending request. Requires an interactive human confirmation."""
    rationale = _require_rationale(args, "approve")
    if rationale is None:
        return 2
    gate = _load_gate(args)
    request, code = _locate_request(gate, args)
    if request is None:
        return code
    authorization = _confirm_gate_decision(
        gate, request, "approve", rationale
    )
    if authorization is None:
        print(
            "approve requires an interactive human confirmation on a terminal; "
            "refusing in a non-interactive context. Re-run `afs approvals approve` "
            "from an interactive terminal.",
            file=sys.stderr,
        )
        return 2
    ok = gate.approve_human(
        request.agent,
        request.action,
        rationale=rationale,
        authorization=authorization,
        request_id=request.request_id,
    )
    if ok:
        print(
            f"Approved: agent={request.agent} action={request.action} "
            f"({request.request_id}) by {authorization.identity.reviewer}"
        )
        return 0
    print(f"No pending request found for request_id={request.request_id}")
    return 1


def _reject_stale(gate: ApprovalGate, args: argparse.Namespace, rationale: str) -> int:
    """Reject every pending request older than ``--older-than`` hours at once.

    One terminal confirmation covers the batch; the token names the count and
    the broker scope binds the exact id set, so the ack cannot be replayed
    against requests queued after the prompt was shown.
    """
    hours = args.older_than
    stale = gate.stale_pending(hours)
    if not stale:
        print(f"No pending requests older than {hours:g}h.")
        return 0
    token = f"reject {len(stale)}"
    prompt = "\n".join(
        [
            "",
            "=== HUMAN CONFIRMATION REQUIRED (batch reject) ===",
            *(
                f"  {r.request_id}  {r.agent} {r.action}  {r.detail}  ({r.timestamp[:19]})"
                for r in stale
            ),
            f"  because: {rationale}",
            f"Decision: reject these {len(stale)} request(s) pending over {hours:g}h.",
            f"Type '{token}' to confirm, anything else aborts: ",
        ]
    )
    ids = [r.request_id for r in stale]
    authorization = _broker_for_reader(_TTY_READER).confirm_token(
        token, prompt, scope=gate.batch_authorization_scope("reject", ids, rationale)
    )
    if authorization is None:
        print(
            "reject --stale requires an interactive human confirmation on a terminal; "
            "refusing to label a programmatic denial as human-reviewed.",
            file=sys.stderr,
        )
        return 2
    rejected = gate.reject_human_batch(ids, rationale=rationale, authorization=authorization)
    print(f"Rejected {len(rejected)} stale request(s) by {authorization.identity.reviewer}")
    if len(rejected) < len(ids):
        print(f"{len(ids) - len(rejected)} stopped being pending before the decision landed.")
    return 0


def approvals_reject_command(args: argparse.Namespace) -> int:
    """Reject a pending request with human provenance.

    Programmatic callers may still deny through ``ApprovalGate.reject`` but
    that record is explicitly unauthenticated and is not a human calibration
    judgment. The CLI's judgment-bearing path requires the broker.
    """
    stale = getattr(args, "stale", False)
    if stale and (
        getattr(args, "request_id", None)
        or getattr(args, "agent", None)
        or getattr(args, "action", None)
    ):
        print("--stale rejects by age; do not also name a request.", file=sys.stderr)
        return 2
    rationale = _require_rationale(args, "reject")
    if rationale is None:
        return 2
    gate = _load_gate(args)
    if stale:
        return _reject_stale(gate, args, rationale)
    request, code = _locate_request(gate, args)
    if request is None:
        return code
    authorization = _confirm_gate_decision(
        gate, request, "reject", rationale
    )
    if authorization is None:
        print(
            "reject requires an interactive human confirmation on a terminal; "
            "refusing to label a programmatic denial as human-reviewed.",
            file=sys.stderr,
        )
        return 2
    ok = gate.reject_human(
        request.agent,
        request.action,
        rationale=rationale,
        authorization=authorization,
        request_id=request.request_id,
    )
    if ok:
        print(f"Rejected: agent={request.agent} action={request.action} ({request.request_id})")
        return 0
    print(f"No pending request found for request_id={request.request_id}")
    return 1


def approvals_revoke_command(args: argparse.Namespace) -> int:
    """Stop an approval from authorizing further checks.

    Revoking only removes permission, so it needs no terminal confirmation.
    """
    reason = _require_rationale(args, "revoke")
    if reason is None:
        return 2
    gate = _load_gate(args)
    if gate.revoke(args.request_id, reason=reason):
        print(f"Revoked {args.request_id}: it no longer authorizes any check.")
        return 0
    request = gate.get(args.request_id)
    if request is None:
        print(f"No request found for request_id={args.request_id}")
    elif request.revoked_at:
        print(f"Request {args.request_id} was already revoked at {request.revoked_at}.")
    else:
        print(f"Request {args.request_id} is {request.status}; only approvals can be revoked.")
    return 1


def approvals_clear_command(args: argparse.Namespace) -> int:
    """Prune old system-closed requests; human decisions are kept."""
    gate = _load_gate(args)
    days = getattr(args, "older_than_days", SYSTEM_RETENTION_DAYS)
    pruned, remaining = gate.clear_completed(older_than_days=days)

    if getattr(args, "json", False):
        print(json.dumps({"pruned": pruned, "cleared": pruned, "remaining": remaining}))
        return 0

    print(
        f"Pruned {pruned} expired/superseded/withdrawn request(s) older than "
        f"{days:g} day(s); {remaining} pending remain. Human decisions are kept."
    )
    return 0


def approvals_history_command(args: argparse.Namespace) -> int:
    """Show all requests including completed ones."""
    gate = _load_gate(args)
    all_requests = gate.all_requests()

    if getattr(args, "json", False):
        print(json.dumps([r.to_dict() for r in all_requests], indent=2))
        return 0

    if not all_requests:
        print("No approval requests.")
        return 0

    # Calculate column widths for alignment
    agent_w = max(len(r.agent) for r in all_requests)
    action_w = max(len(r.action) for r in all_requests)
    status_w = max(len(r.status) for r in all_requests)
    detail_w = max(len(r.detail) for r in all_requests)

    agent_w = max(agent_w, len("AGENT"))
    action_w = max(action_w, len("ACTION"))
    status_w = max(status_w, len("STATUS"))
    detail_w = max(detail_w, len("DETAIL"))

    header = (
        f"{'AGENT':<{agent_w}}  {'ACTION':<{action_w}}  "
        f"{'STATUS':<{status_w}}  {'DETAIL':<{detail_w}}  TIMESTAMP"
    )
    print(header)
    print("-" * len(header))

    for req in all_requests:
        ts = req.timestamp[:19] if len(req.timestamp) >= 19 else req.timestamp
        print(
            f"{req.agent:<{agent_w}}  {req.action:<{action_w}}  "
            f"{req.status:<{status_w}}  {req.detail:<{detail_w}}  {ts}"
        )
        if req.reviewed_by:
            via = req.reviewed_via or "unconfirmed"
            print(f"  by: {req.reviewed_by} ({via})  ref: {req.request_id}")
        if req.rationale:
            print(f"  because: {req.rationale}")
        if req.revoked_at:
            print(f"  revoked: {req.revoked_reason}")

    pending_count = sum(1 for r in all_requests if r.status == "pending")
    print()
    print(f"{len(all_requests)} total, {pending_count} pending")
    return 0


_STORE_HELP = (
    "Path to the approvals store (.sqlite3). A legacy approvals.json path "
    "maps to the .sqlite3 file beside it."
)
_REQUEST_ID_HELP = (
    "Exact request to decide (gate_...), from `afs approvals list`. "
    "Use it when several pending requests share an agent/action pair."
)


def _add_store_arg(parser: argparse.ArgumentParser) -> None:
    # SUPPRESS keeps a value given before the subcommand from being reset.
    parser.add_argument("--approvals-file", default=argparse.SUPPRESS, help=_STORE_HELP)


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register approvals command parsers."""
    approvals_parser = subparsers.add_parser(
        "approvals", help="Manage agent approval requests."
    )
    approvals_parser.add_argument("--approvals-file", help=_STORE_HELP)
    approvals_sub = approvals_parser.add_subparsers(dest="approvals_command")

    # list
    list_parser = approvals_sub.add_parser("list", help="List pending approval requests.")
    _add_store_arg(list_parser)
    list_parser.add_argument("--json", action="store_true", help="Output JSON.")
    list_parser.set_defaults(func=approvals_list_command)

    # show
    show_parser = approvals_sub.add_parser(
        "show", help="Show one request in full, including its payload."
    )
    _add_store_arg(show_parser)
    show_parser.add_argument("--request-id", required=True, help="Request to show (gate_...).")
    show_parser.add_argument("--json", action="store_true", help="Output JSON.")
    show_parser.set_defaults(func=approvals_show_command)

    # approve
    approve_parser = approvals_sub.add_parser("approve", help="Approve a pending request.")
    _add_store_arg(approve_parser)
    approve_parser.add_argument("agent", nargs="?", help="Agent name (with action).")
    approve_parser.add_argument("action", nargs="?", help="Action name (with agent).")
    approve_parser.add_argument("--request-id", help=_REQUEST_ID_HELP)
    approve_parser.add_argument(
        "--because",
        help="Required rationale for the decision; stored in immutable approvals history.",
    )
    approve_parser.set_defaults(func=approvals_approve_command)

    # reject
    reject_parser = approvals_sub.add_parser("reject", help="Reject a pending request.")
    _add_store_arg(reject_parser)
    reject_parser.add_argument("agent", nargs="?", help="Agent name (with action).")
    reject_parser.add_argument("action", nargs="?", help="Action name (with agent).")
    reject_parser.add_argument("--request-id", help=_REQUEST_ID_HELP)
    reject_parser.add_argument(
        "--stale",
        action="store_true",
        help="Reject every pending request older than --older-than hours "
        "with one terminal confirmation.",
    )
    reject_parser.add_argument(
        "--older-than",
        type=float,
        default=24.0,
        help="With --stale: minimum request age in hours (default 24).",
    )
    reject_parser.add_argument(
        "--because",
        help="Required rationale for the decision; stored in immutable approvals history.",
    )
    reject_parser.set_defaults(func=approvals_reject_command)

    # revoke
    revoke_parser = approvals_sub.add_parser(
        "revoke", help="Stop an approval from authorizing further checks."
    )
    _add_store_arg(revoke_parser)
    revoke_parser.add_argument("--request-id", required=True, help="Approval to revoke.")
    revoke_parser.add_argument("--because", help="Required reason; kept with the approval.")
    revoke_parser.set_defaults(func=approvals_revoke_command)

    # clear
    clear_parser = approvals_sub.add_parser(
        "clear",
        help="Prune old expired/superseded/withdrawn requests; human decisions are kept.",
    )
    _add_store_arg(clear_parser)
    clear_parser.add_argument(
        "--older-than-days",
        type=float,
        default=SYSTEM_RETENTION_DAYS,
        help=f"Minimum age of a closed request to prune (default {SYSTEM_RETENTION_DAYS:g}).",
    )
    clear_parser.add_argument("--json", action="store_true", help="Output JSON.")
    clear_parser.set_defaults(func=approvals_clear_command)

    # history
    history_parser = approvals_sub.add_parser(
        "history", help="Show all requests including completed ones."
    )
    _add_store_arg(history_parser)
    history_parser.add_argument("--json", action="store_true", help="Output JSON.")
    history_parser.set_defaults(func=approvals_history_command)
