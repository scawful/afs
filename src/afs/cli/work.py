"""CLI commands for AFS work-assistant state."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from ..work_assistant import WorkAssistantStore
from ..work_execution import WorkApprovalExecutionError, execute_approved_action
from ._utils import load_manager, resolve_context_paths


def _add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Config path.")
    parser.add_argument("--path", help="Project path.")
    parser.add_argument("--context-root", help="Context root override.")
    parser.add_argument("--context-dir", help="Context directory name.")


def _store_from_args(args: argparse.Namespace) -> tuple[WorkAssistantStore, Path]:
    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args,
        manager,
    )
    return WorkAssistantStore(context_path, config=manager.config), context_path


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _print_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    if not rows:
        print("No records.")
        return
    widths = {
        key: max(len(label), *(len(str(row.get(key, ""))) for row in rows))
        for key, label in columns
    }
    header = "  ".join(label.ljust(widths[key]) for key, label in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _label in columns))


def work_summary_command(args: argparse.Namespace) -> int:
    store, context_path = _store_from_args(args)
    summary = store.summary()
    summary["context_path"] = str(context_path)
    if args.json:
        _print_json(summary)
        return 0
    print(f"context: {context_path}")
    print(f"database: {summary['db_path']}")
    print(f"people: {summary['people']}")
    print(f"relationships: {summary['relationships']}")
    print(f"review_routes: {summary['review_routes']}")
    print(f"approvals: {summary['approvals']} ({summary['pending_approvals']} pending)")
    print(f"activity: {summary['activity']}")
    return 0


def people_list_command(args: argparse.Namespace) -> int:
    store, _context_path = _store_from_args(args)
    rows = store.list_people(limit=args.limit)
    if args.json:
        _print_json(rows)
        return 0
    _print_table(
        rows,
        [
            ("display_name", "NAME"),
            ("person_id", "PERSON ID"),
            ("organization", "ORG"),
            ("team", "TEAM"),
        ],
    )
    return 0


def relationships_list_command(args: argparse.Namespace) -> int:
    store, _context_path = _store_from_args(args)
    rows = store.list_relationships(limit=args.limit)
    if args.json:
        _print_json(rows)
        return 0
    _print_table(
        rows,
        [
            ("display_name", "PERSON"),
            ("relationship_type", "ROLE"),
            ("scope_type", "SCOPE"),
            ("scope_id", "SCOPE ID"),
            ("permission_class", "PERMISSION"),
        ],
    )
    return 0


def reviewers_command(args: argparse.Namespace) -> int:
    store, _context_path = _store_from_args(args)
    rows = store.suggest_reviewers(
        target_type=args.target_type,
        scope_type=args.scope_type,
        scope_id=args.scope_id,
        limit=args.limit,
    )
    if args.json:
        _print_json(rows)
        return 0
    _print_table(
        rows,
        [
            ("display_name", "REVIEWER"),
            ("target_type", "TARGET"),
            ("scope_type", "SCOPE"),
            ("scope_id", "SCOPE ID"),
            ("reason", "REASON"),
        ],
    )
    return 0


def approvals_list_command(args: argparse.Namespace) -> int:
    store, _context_path = _store_from_args(args)
    status = None if args.all else args.status
    rows = store.list_approvals(status=status, limit=args.limit)
    if args.json:
        _print_json(rows)
        return 0
    _print_table(
        rows,
        [
            ("approval_id", "APPROVAL ID"),
            ("status", "STATUS"),
            ("target_system", "SYSTEM"),
            ("action", "ACTION"),
            ("summary", "SUMMARY"),
        ],
    )
    return 0


def approvals_show_command(args: argparse.Namespace) -> int:
    store, _context_path = _store_from_args(args)
    approval = store.get_approval(args.approval_id)
    if approval is None:
        print(f"No approval found: {args.approval_id}")
        return 1
    if args.json:
        _print_json(approval)
        return 0
    _print_json(approval)
    return 0


def approvals_request_command(args: argparse.Namespace) -> int:
    store, _context_path = _store_from_args(args)
    approval_id = store.create_approval(
        target_system=args.target_system,
        target_id=args.target_id,
        action=args.action,
        summary=args.summary,
        preview={"text": args.preview} if args.preview else {},
        affected_people=args.affected_person or [],
        risk_level=args.risk_level,
        permission_required=args.permission_required,
        requested_by=args.requested_by,
    )
    if args.json:
        _print_json({"approval_id": approval_id, "status": "pending"})
        return 0
    print(f"Created approval request: {approval_id}")
    print(f"Review with: afs work approvals list --path {args.path or '.'}")
    return 0


def approvals_approve_command(args: argparse.Namespace) -> int:
    store, _context_path = _store_from_args(args)
    if store.approve(args.approval_id, approved_by=args.by):
        print(f"Approved: {args.approval_id}")
        return 0
    print(f"No pending approval found: {args.approval_id}")
    return 1


def approvals_reject_command(args: argparse.Namespace) -> int:
    store, _context_path = _store_from_args(args)
    if store.reject(args.approval_id, rejected_by=args.by):
        print(f"Rejected: {args.approval_id}")
        return 0
    print(f"No pending approval found: {args.approval_id}")
    return 1


def approvals_execute_command(args: argparse.Namespace) -> int:
    store, context_path = _store_from_args(args)
    executor_command = _split_executor(args.executor or "")
    try:
        result = execute_approved_action(
            store,
            context_root=context_path,
            approval_id=args.approval_id,
            executor_command=executor_command,
            actor=args.actor,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
    except WorkApprovalExecutionError as exc:
        if args.json:
            _print_json({"error": str(exc)})
        else:
            print(str(exc))
        return 1

    if args.json or args.dry_run:
        _print_json(result)
        return 0 if result.get("status") != "failed" else 1
    if result.get("status") == "failed":
        print(f"Executor failed for {args.approval_id}: {result.get('stderr') or result.get('stdout')}")
        return 1
    print(f"Applied approved action: {args.approval_id}")
    return 0


def activity_list_command(args: argparse.Namespace) -> int:
    store, _context_path = _store_from_args(args)
    rows = store.list_activity(limit=args.limit)
    if args.json:
        _print_json(rows)
        return 0
    _print_table(
        rows,
        [
            ("timestamp", "TIMESTAMP"),
            ("activity_type", "TYPE"),
            ("actor", "ACTOR"),
            ("summary", "SUMMARY"),
        ],
    )
    return 0


def _split_executor(value: str) -> list[str]:
    if not value.strip():
        return []
    return shlex.split(value)


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register work-assistant command parsers."""
    work_parser = subparsers.add_parser(
        "work",
        help="Manage work-assistant people, review routes, approvals, and activity.",
    )
    _add_context_args(work_parser)
    work_parser.add_argument("--json", action="store_true", help="Output JSON.")
    work_parser.set_defaults(func=work_summary_command, _allow_missing_subcommand=True)
    work_sub = work_parser.add_subparsers(dest="work_command")

    people_parser = work_sub.add_parser("people", help="People records.")
    people_sub = people_parser.add_subparsers(dest="people_command")
    people_list = people_sub.add_parser("list", help="List known people.")
    _add_context_args(people_list)
    people_list.add_argument("--limit", type=int, default=50, help="Max records.")
    people_list.add_argument("--json", action="store_true", help="Output JSON.")
    people_list.set_defaults(func=people_list_command)

    rel_parser = work_sub.add_parser("relationships", help="Project relationship records.")
    rel_sub = rel_parser.add_subparsers(dest="relationships_command")
    rel_list = rel_sub.add_parser("list", help="List relationships.")
    _add_context_args(rel_list)
    rel_list.add_argument("--limit", type=int, default=50, help="Max records.")
    rel_list.add_argument("--json", action="store_true", help="Output JSON.")
    rel_list.set_defaults(func=relationships_list_command)

    reviewers = work_sub.add_parser("reviewers", help="Suggest reviewers for a target.")
    _add_context_args(reviewers)
    reviewers.add_argument("--target-type", required=True, help="docs, code, sheets, tickets, plans, etc.")
    reviewers.add_argument("--scope-type", help="Scope type such as project, repo, doc, or ticket_queue.")
    reviewers.add_argument("--scope-id", help="Scope identifier.")
    reviewers.add_argument("--limit", type=int, default=10, help="Max reviewers.")
    reviewers.add_argument("--json", action="store_true", help="Output JSON.")
    reviewers.set_defaults(func=reviewers_command)

    approvals = work_sub.add_parser("approvals", help="Approval requests for external writes.")
    approvals_sub = approvals.add_subparsers(dest="work_approvals_command")

    approvals_list = approvals_sub.add_parser("list", help="List approval requests.")
    _add_context_args(approvals_list)
    approvals_list.add_argument("--status", default="pending", help="Status filter.")
    approvals_list.add_argument("--all", action="store_true", help="Show all statuses.")
    approvals_list.add_argument("--limit", type=int, default=50, help="Max records.")
    approvals_list.add_argument("--json", action="store_true", help="Output JSON.")
    approvals_list.set_defaults(func=approvals_list_command)

    approvals_show = approvals_sub.add_parser("show", help="Show one approval request.")
    _add_context_args(approvals_show)
    approvals_show.add_argument("approval_id", help="Approval id.")
    approvals_show.add_argument("--json", action="store_true", help="Output JSON.")
    approvals_show.set_defaults(func=approvals_show_command)

    approvals_request = approvals_sub.add_parser("request", help="Create an approval request.")
    _add_context_args(approvals_request)
    approvals_request.add_argument("--target-system", required=True, help="External system name.")
    approvals_request.add_argument("--target-id", required=True, help="External object id.")
    approvals_request.add_argument("--action", required=True, help="Action to approve.")
    approvals_request.add_argument("--summary", required=True, help="Plain-language summary.")
    approvals_request.add_argument("--preview", help="Short preview or diff text.")
    approvals_request.add_argument(
        "--affected-person",
        action="append",
        help="Person id or handle affected by this action; repeatable.",
    )
    approvals_request.add_argument("--risk-level", default="medium", help="Risk level.")
    approvals_request.add_argument("--permission-required", default="human approval", help="Permission needed.")
    approvals_request.add_argument("--requested-by", default="agent", help="Requester identity.")
    approvals_request.add_argument("--json", action="store_true", help="Output JSON.")
    approvals_request.set_defaults(func=approvals_request_command)

    approvals_approve = approvals_sub.add_parser("approve", help="Approve one request.")
    _add_context_args(approvals_approve)
    approvals_approve.add_argument("approval_id", help="Approval id.")
    approvals_approve.add_argument("--by", default="human", help="Approver identity.")
    approvals_approve.set_defaults(func=approvals_approve_command)

    approvals_reject = approvals_sub.add_parser("reject", help="Reject one request.")
    _add_context_args(approvals_reject)
    approvals_reject.add_argument("approval_id", help="Approval id.")
    approvals_reject.add_argument("--by", default="human", help="Reviewer identity.")
    approvals_reject.set_defaults(func=approvals_reject_command)

    approvals_execute = approvals_sub.add_parser(
        "execute",
        help="Execute one approved request with an explicit connector command.",
    )
    _add_context_args(approvals_execute)
    approvals_execute.add_argument("approval_id", help="Approval id.")
    approvals_execute.add_argument("--actor", default="agent", help="Executor identity.")
    approvals_execute.add_argument("--timeout", type=int, default=60, help="Executor timeout seconds.")
    approvals_execute.add_argument("--dry-run", action="store_true", help="Print payload without executing.")
    approvals_execute.add_argument("--json", action="store_true", help="Output JSON.")
    approvals_execute.add_argument(
        "--executor",
        default="",
        help="Executor command. AFS parses it without a shell and appends the approval JSON path.",
    )
    approvals_execute.set_defaults(func=approvals_execute_command)

    activity = work_sub.add_parser("activity", help="Work-assistant activity.")
    activity_sub = activity.add_subparsers(dest="activity_command")
    activity_list = activity_sub.add_parser("list", help="List recent activity.")
    _add_context_args(activity_list)
    activity_list.add_argument("--limit", type=int, default=50, help="Max records.")
    activity_list.add_argument("--json", action="store_true", help="Output JSON.")
    activity_list.set_defaults(func=activity_list_command)
