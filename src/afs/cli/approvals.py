"""CLI commands for managing agent approval requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..agents.guardrails import ApprovalGate, ApprovalRequest
from ..human_provenance import (
    HumanAuthorization,
    _broker_for_reader,
)

# Test seam: production uses the platform controlling-terminal backend.
_TTY_READER = None


def _load_gate(args: argparse.Namespace) -> ApprovalGate:
    """Load ApprovalGate, optionally from a custom path."""
    path = Path(args.approvals_file) if getattr(args, "approvals_file", None) else None
    return ApprovalGate(path=path)


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


def approvals_list_command(args: argparse.Namespace) -> int:
    """List pending approval requests."""
    gate = _load_gate(args)
    pending = gate.pending_requests()

    if getattr(args, "json", False):
        print(json.dumps([r.to_dict() for r in pending], indent=2))
        return 0

    if not pending:
        print("No pending approval requests.")
        return 0

    # Calculate column widths for alignment
    agent_w = max(len(r.agent) for r in pending)
    action_w = max(len(r.action) for r in pending)
    detail_w = max(len(r.detail) for r in pending)

    agent_w = max(agent_w, len("AGENT"))
    action_w = max(action_w, len("ACTION"))
    detail_w = max(detail_w, len("DETAIL"))

    header = f"{'AGENT':<{agent_w}}  {'ACTION':<{action_w}}  {'DETAIL':<{detail_w}}  TIMESTAMP"
    print(header)
    print("-" * len(header))

    for req in pending:
        ts = req.timestamp[:19] if len(req.timestamp) >= 19 else req.timestamp
        print(f"{req.agent:<{agent_w}}  {req.action:<{action_w}}  {req.detail:<{detail_w}}  {ts}")

    print()
    print(f"{len(pending)} pending request(s)")
    return 0


def approvals_approve_command(args: argparse.Namespace) -> int:
    """Approve a pending request. Requires an interactive human confirmation."""
    rationale = _require_rationale(args, "approve")
    if rationale is None:
        return 2
    gate = _load_gate(args)
    request = gate.find_pending(args.agent, args.action)
    if request is None:
        print(f"No pending request found for agent={args.agent} action={args.action}")
        return 1
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
        args.agent,
        args.action,
        rationale=rationale,
        authorization=authorization,
    )
    if ok:
        print(
            f"Approved: agent={args.agent} action={args.action} "
            f"by {authorization.identity.reviewer}"
        )
        return 0
    print(f"No pending request found for agent={args.agent} action={args.action}")
    return 1


def approvals_reject_command(args: argparse.Namespace) -> int:
    """Reject a pending request with human provenance.

    Programmatic callers may still deny through ``ApprovalGate.reject`` but
    that record is explicitly unauthenticated and is not a human calibration
    judgment. The CLI's judgment-bearing path requires the broker.
    """
    rationale = _require_rationale(args, "reject")
    if rationale is None:
        return 2
    gate = _load_gate(args)
    request = gate.find_pending(args.agent, args.action)
    if request is None:
        print(f"No pending request found for agent={args.agent} action={args.action}")
        return 1
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
        args.agent,
        args.action,
        rationale=rationale,
        authorization=authorization,
    )
    if ok:
        print(f"Rejected: agent={args.agent} action={args.action}")
        return 0
    print(f"No pending request found for agent={args.agent} action={args.action}")
    return 1


def approvals_clear_command(args: argparse.Namespace) -> int:
    """Archive completed requests and compact active state."""
    gate = _load_gate(args)
    removed, remaining = gate.clear_completed()

    if getattr(args, "json", False):
        print(json.dumps({"archived": removed, "cleared": removed, "remaining": remaining}))
        return 0

    print(f"Archived {removed} completed request(s), {remaining} pending remain.")
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

    pending_count = sum(1 for r in all_requests if r.status == "pending")
    print()
    print(f"{len(all_requests)} total, {pending_count} pending")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register approvals command parsers."""
    approvals_parser = subparsers.add_parser(
        "approvals", help="Manage agent approval requests."
    )
    approvals_parser.add_argument(
        "--approvals-file", help="Path to approvals JSON file."
    )
    approvals_sub = approvals_parser.add_subparsers(dest="approvals_command")

    # list
    list_parser = approvals_sub.add_parser("list", help="List pending approval requests.")
    list_parser.add_argument(
        "--approvals-file",
        default=argparse.SUPPRESS,
        help="Path to approvals JSON file.",
    )
    list_parser.add_argument("--json", action="store_true", help="Output JSON.")
    list_parser.set_defaults(func=approvals_list_command)

    # approve
    approve_parser = approvals_sub.add_parser("approve", help="Approve a pending request.")
    approve_parser.add_argument(
        "--approvals-file",
        default=argparse.SUPPRESS,
        help="Path to approvals JSON file.",
    )
    approve_parser.add_argument("agent", help="Agent name.")
    approve_parser.add_argument("action", help="Action name.")
    approve_parser.add_argument(
        "--because",
        help="Required rationale for the decision; stored in immutable approvals history.",
    )
    approve_parser.set_defaults(func=approvals_approve_command)

    # reject
    reject_parser = approvals_sub.add_parser("reject", help="Reject a pending request.")
    reject_parser.add_argument(
        "--approvals-file",
        default=argparse.SUPPRESS,
        help="Path to approvals JSON file.",
    )
    reject_parser.add_argument("agent", help="Agent name.")
    reject_parser.add_argument("action", help="Action name.")
    reject_parser.add_argument(
        "--because",
        help="Required rationale for the decision; stored in immutable approvals history.",
    )
    reject_parser.set_defaults(func=approvals_reject_command)

    # clear
    clear_parser = approvals_sub.add_parser(
        "clear", help="Archive completed requests and compact active state."
    )
    clear_parser.add_argument(
        "--approvals-file",
        default=argparse.SUPPRESS,
        help="Path to approvals JSON file.",
    )
    clear_parser.add_argument("--json", action="store_true", help="Output JSON.")
    clear_parser.set_defaults(func=approvals_clear_command)

    # history
    history_parser = approvals_sub.add_parser(
        "history", help="Show all requests including completed ones."
    )
    history_parser.add_argument(
        "--approvals-file",
        default=argparse.SUPPRESS,
        help="Path to approvals JSON file.",
    )
    history_parser.add_argument("--json", action="store_true", help="Output JSON.")
    history_parser.set_defaults(func=approvals_history_command)
