"""CLI commands for managing agent approval requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..agents.guardrails import ApprovalGate


def _load_gate(args: argparse.Namespace) -> ApprovalGate:
    """Load ApprovalGate, optionally from a custom path."""
    path = Path(args.approvals_file) if getattr(args, "approvals_file", None) else None
    return ApprovalGate(path=path)


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
    """Approve a pending request."""
    gate = _load_gate(args)
    ok = gate.approve(args.agent, args.action, reviewer="cli")
    if ok:
        print(f"Approved: agent={args.agent} action={args.action}")
        return 0
    print(f"No pending request found for agent={args.agent} action={args.action}")
    return 1


def approvals_reject_command(args: argparse.Namespace) -> int:
    """Reject a pending request."""
    gate = _load_gate(args)
    ok = gate.reject(args.agent, args.action, reviewer="cli")
    if ok:
        print(f"Rejected: agent={args.agent} action={args.action}")
        return 0
    print(f"No pending request found for agent={args.agent} action={args.action}")
    return 1


def approvals_clear_command(args: argparse.Namespace) -> int:
    """Clear all completed (approved/rejected) requests."""
    gate = _load_gate(args)
    before = len(gate._pending)
    gate._pending = [r for r in gate._pending if r.status == "pending"]
    removed = before - len(gate._pending)
    gate._save()

    if getattr(args, "json", False):
        print(json.dumps({"cleared": removed, "remaining": len(gate._pending)}))
        return 0

    print(f"Cleared {removed} completed request(s), {len(gate._pending)} pending remain.")
    return 0


def approvals_history_command(args: argparse.Namespace) -> int:
    """Show all requests including completed ones."""
    gate = _load_gate(args)
    all_requests = gate._pending

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
    list_parser.add_argument("--approvals-file", help="Path to approvals JSON file.")
    list_parser.add_argument("--json", action="store_true", help="Output JSON.")
    list_parser.set_defaults(func=approvals_list_command)

    # approve
    approve_parser = approvals_sub.add_parser("approve", help="Approve a pending request.")
    approve_parser.add_argument("--approvals-file", help="Path to approvals JSON file.")
    approve_parser.add_argument("agent", help="Agent name.")
    approve_parser.add_argument("action", help="Action name.")
    approve_parser.set_defaults(func=approvals_approve_command)

    # reject
    reject_parser = approvals_sub.add_parser("reject", help="Reject a pending request.")
    reject_parser.add_argument("--approvals-file", help="Path to approvals JSON file.")
    reject_parser.add_argument("agent", help="Agent name.")
    reject_parser.add_argument("action", help="Action name.")
    reject_parser.set_defaults(func=approvals_reject_command)

    # clear
    clear_parser = approvals_sub.add_parser(
        "clear", help="Clear completed (approved/rejected) requests."
    )
    clear_parser.add_argument("--approvals-file", help="Path to approvals JSON file.")
    clear_parser.add_argument("--json", action="store_true", help="Output JSON.")
    clear_parser.set_defaults(func=approvals_clear_command)

    # history
    history_parser = approvals_sub.add_parser(
        "history", help="Show all requests including completed ones."
    )
    history_parser.add_argument("--approvals-file", help="Path to approvals JSON file.")
    history_parser.add_argument("--json", action="store_true", help="Output JSON.")
    history_parser.set_defaults(func=approvals_history_command)
