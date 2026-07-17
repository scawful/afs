"""AFS doctor — diagnose and auto-fix common AFS issues.

Usage:
    afs doctor              # run all checks
    afs doctor --fix        # auto-apply available fixes
    afs doctor --json       # machine-readable output
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..diagnostics import (
    format_results_json,
    format_results_text,
    run_all_checks,
)


def _doctor_command(args: argparse.Namespace) -> int:
    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    auto_fix = getattr(args, "fix", False)
    results = run_all_checks(config_path=config_path, auto_fix=auto_fix)

    if getattr(args, "json", False):
        print(format_results_json(results))
    else:
        print(format_results_text(results))

    has_errors = any(r.status == "error" for r in results)
    return 1 if has_errors else 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register the doctor command."""
    parser = subparsers.add_parser(
        "doctor",
        aliases=["repair"],
        help="Diagnose and repair common AFS issues (friendly alias: repair).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-apply available fixes.",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="JSON output.",
    )
    parser.add_argument(
        "--config",
        help="Config path override.",
    )
    parser.set_defaults(func=_doctor_command)
