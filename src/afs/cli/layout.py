"""Read-only layout inspection and migration planning commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..context_layout import (
    audit_layout,
    build_migration_plan,
    build_rollback_manifest,
    write_manifest,
)


def _context_root(args: argparse.Namespace) -> Path:
    return Path(args.context_root or Path.home() / ".context").expanduser().resolve()


def layout_audit_command(args: argparse.Namespace) -> int:
    audit = audit_layout(_context_root(args))
    payload = audit.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"context_root: {audit.context_root}")
        print(f"layout_version: {audit.layout_version}")
        print(f"valid: {str(audit.valid).lower()}")
        print(f"migration_ready: {str(audit.migration_ready).lower()}")
        print("issues:")
        for issue in audit.issues:
            suffix = " (blocking)" if issue.blocking else ""
            print(f"- {issue.code}: {issue.message}{suffix}")
        if not audit.issues:
            print("- (none)")
    return 0 if audit.valid else 1


def layout_plan_command(args: argparse.Namespace) -> int:
    destination = Path(args.destination_root).expanduser() if args.destination_root else None
    plan = build_migration_plan(_context_root(args), destination)
    rollback = build_rollback_manifest(plan)
    if args.output:
        write_manifest(Path(args.output), plan)
    if args.rollback_output:
        write_manifest(Path(args.rollback_output), rollback)
    payload = {
        "plan": plan.to_dict(),
        "rollback": rollback.to_dict(),
        "plan_path": str(Path(args.output).expanduser().resolve()) if args.output else None,
        "rollback_path": str(Path(args.rollback_output).expanduser().resolve())
        if args.rollback_output
        else None,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"transaction_id: {plan.transaction_id}")
        print(f"ready: {str(plan.ready).lower()}")
        print(f"source_fingerprint: {plan.source_fingerprint}")
        print(f"source_files: {plan.source_file_count}")
        print(f"source_bytes: {plan.source_bytes}")
        print(f"operations: {len(plan.operations)}")
        if plan.blocking_entries:
            print("blocking_entries:")
            for entry in plan.blocking_entries:
                print(f"- {entry}")
        if args.output:
            print(f"plan_path: {payload['plan_path']}")
        if args.rollback_output:
            print(f"rollback_path: {payload['rollback_path']}")
    return 0 if plan.ready else 2


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    layout = subparsers.add_parser(
        "layout",
        help="Audit or plan changes to the versioned context layout.",
    )
    commands = layout.add_subparsers(dest="layout_command")

    audit = commands.add_parser("audit", help="Inspect layout health without changing files.")
    audit.add_argument("--context-root", help="Context root (default: ~/.context).")
    audit.add_argument("--json", action="store_true", help="Output JSON.")
    audit.set_defaults(func=layout_audit_command)

    plan = commands.add_parser(
        "plan",
        help="Build a hash-bound v1-to-v2 plan without executing it.",
    )
    plan.add_argument("--context-root", help="Source context root (default: ~/.context).")
    plan.add_argument("--destination-root", help="Optional separate v2 destination root.")
    plan.add_argument("--output", help="Write the private JSON migration plan atomically.")
    plan.add_argument("--rollback-output", help="Write the paired rollback manifest atomically.")
    plan.add_argument("--json", action="store_true", help="Output JSON.")
    plan.set_defaults(func=layout_plan_command)

