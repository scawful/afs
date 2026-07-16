"""CLI verbs for the calibration trail: review past decisions, score outcomes.

``afs calibration review`` resurfaces the week's approval decisions (with the
rationale given at decision time), closed missions (next to their
human-authored acceptance), and predict-before-reveal entries, then prompts
for outcome scoring. ``--markdown`` emits a digest section that a weekly
review document can pull in verbatim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..calibration import (
    VALID_OUTCOMES,
    collect_decisions,
    record_outcome,
)
from ..human_provenance import confirm_typed_token
from ._utils import load_manager, resolve_context_paths

# Test seam: tests inject a fake terminal reader; production uses /dev/tty.
_TTY_READER = None


def _context_path(args: argparse.Namespace):
    config_path = Path(args.config) if getattr(args, "config", None) else None
    manager = load_manager(config_path)
    _project_path, context_path, _root, _dir = resolve_context_paths(args, manager)
    return context_path, manager.config


def _outcome_line(entry: dict, scored: dict) -> str:
    score = scored.get(entry.get("ref"))
    if score:
        note = f" — {score['note']}" if score.get("note") else ""
        return f"scored: {score.get('outcome')}{note}"
    return f"unscored — afs calibration score {entry.get('ref')} --outcome hit|miss|unclear"


def calibration_review_command(args: argparse.Namespace) -> int:
    context_path, config = _context_path(args)
    report = collect_decisions(context_path, days=args.days, config=config)
    scored = report["scored"]

    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, default=str))
        return 0

    if getattr(args, "markdown", False):
        return _print_markdown(report)

    total = len(report["approvals"]) + len(report["missions"]) + len(report["predictions"])
    print(f"Calibration review — last {report['window_days']} day(s), {total} decision(s)")

    if report["approvals"]:
        print("\nApproval decisions:")
        for entry in report["approvals"]:
            rationale = entry.get("rationale") or "(no rationale recorded)"
            print(f"  {entry['status']}: {entry.get('action')}  [{entry.get('ref')}]")
            print(f"    because: {rationale}")
            print(f"    {_outcome_line(entry, scored)}")

    if report["missions"]:
        print("\nClosed missions:")
        for entry in report["missions"]:
            acceptance = entry.get("acceptance") or "(no acceptance was set)"
            print(f"  {entry['status']}: {entry.get('title')}  [{entry.get('ref')}]")
            print(f"    acceptance was: {acceptance}")
            print(f"    {_outcome_line(entry, scored)}")

    if report["predictions"]:
        print("\nPredictions:")
        for entry in report["predictions"]:
            marker = {True: "matched", False: "missed", None: "unresolved"}[entry.get("match")]
            print(
                f"  {marker}: predicted {entry.get('predicted')!r} vs actual "
                f"{entry.get('actual')!r}  [{entry.get('id')}]"
            )

    if total == 0:
        print("No decisions recorded in the window.")
    return 0


def _print_markdown(report: dict) -> int:
    scored = report["scored"]
    print(f"## Calibration — decisions from the last {report['window_days']} day(s)")
    if report["approvals"]:
        print("\n### Approvals")
        for entry in report["approvals"]:
            rationale = entry.get("rationale") or "_no rationale recorded_"
            score = scored.get(entry.get("ref"))
            outcome = score.get("outcome") if score else "unscored"
            print(
                f"- **{entry['status']}** `{entry.get('action')}` — {rationale} "
                f"(outcome: {outcome})"
            )
    if report["missions"]:
        print("\n### Closed missions")
        for entry in report["missions"]:
            acceptance = entry.get("acceptance") or "_no acceptance was set_"
            score = scored.get(entry.get("ref"))
            outcome = score.get("outcome") if score else "unscored"
            print(
                f"- **{entry['status']}** {entry.get('title')} — acceptance was: "
                f"{acceptance} (outcome: {outcome})"
            )
    if report["predictions"]:
        matches = sum(1 for entry in report["predictions"] if entry.get("match") is True)
        print(
            f"\n### Predictions\n- {matches}/{len(report['predictions'])} "
            "top-priority predictions matched the actual queue."
        )
    return 0


def _confirm_outcome_score(ref: str, outcome: str) -> str | None:
    """Interactive human confirmation for recording an outcome score.

    The outcome score is the judgment the whole calibration trail exists to
    capture — an agent recording one headlessly would be grading its own
    work, the exact offload the trail is meant to counter. The operator must
    re-type the outcome on the controlling tty, which piped stdin cannot
    satisfy. Returns the OS-level reviewer identity, or ``None`` to refuse.
    """
    prompt = "\n".join(
        [
            "",
            "=== HUMAN CONFIRMATION REQUIRED (calibration score) ===",
            f"  ref:     {ref}",
            f"  outcome: {outcome}",
            "Outcome scores are the human half of the calibration trail.",
            f"Type '{outcome}' to confirm, anything else aborts: ",
        ]
    )
    return confirm_typed_token(outcome, prompt, reader=_TTY_READER)


def calibration_score_command(args: argparse.Namespace) -> int:
    context_path, config = _context_path(args)
    reviewer = _confirm_outcome_score(args.ref, args.outcome)
    if reviewer is None:
        print(
            "score requires an interactive human confirmation on a terminal; "
            "refusing in a non-interactive context. Re-run `afs calibration "
            "score` from an interactive terminal.",
            file=sys.stderr,
        )
        return 2
    try:
        entry = record_outcome(
            context_path,
            ref=args.ref,
            outcome=args.outcome,
            note=getattr(args, "note", "") or "",
            scored_by=reviewer,
            scored_via="tty",
            config=config,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if getattr(args, "json", False):
        print(json.dumps(entry, indent=2))
    else:
        print(f"scored {entry['kind']} {entry['ref']}: {entry['outcome']}")
    return 0


def _add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Config path.")
    parser.add_argument("--path", help="Project path.")
    parser.add_argument("--context-root", help="Context root override.")
    parser.add_argument("--context-dir", help="Context directory name.")


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    calibration_parser = subparsers.add_parser(
        "calibration",
        help="Review past decisions with their rationales and score outcomes.",
    )
    calibration_sub = calibration_parser.add_subparsers(dest="calibration_command")

    review_parser = calibration_sub.add_parser(
        "review", help="Resurface recent decisions for outcome scoring."
    )
    _add_context_args(review_parser)
    review_parser.add_argument(
        "--days", type=int, default=7, help="Window in days (default 7)."
    )
    review_parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit a markdown digest section for a weekly review document.",
    )
    review_parser.add_argument("--json", action="store_true", help="Output JSON.")
    review_parser.set_defaults(func=calibration_review_command)

    score_parser = calibration_sub.add_parser(
        "score", help="Record the outcome of a past decision."
    )
    _add_context_args(score_parser)
    score_parser.add_argument("ref", help="Approval id, mission id, or prediction id.")
    score_parser.add_argument(
        "--outcome", required=True, choices=VALID_OUTCOMES, help="Outcome verdict."
    )
    score_parser.add_argument("--note", help="Optional note on what happened.")
    score_parser.add_argument("--json", action="store_true", help="Output JSON.")
    score_parser.set_defaults(func=calibration_score_command)
