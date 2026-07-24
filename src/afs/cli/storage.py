"""Read-only storage diagnostics and exact-plan cleanup commands."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

from ..human_provenance import HumanAuthorization, _broker_for_reader
from ..storage_doctor import (
    DEFAULT_STALE_DAYS,
    StorageApplyError,
    StorageSafetyError,
    apply_storage_plan,
    audit_storage,
    build_storage_plan,
    load_storage_plan,
    storage_authorization_scope,
    write_storage_plan,
)

# Test seam: production uses the platform controlling-terminal backend.
_TTY_READER = None


def _home(args: argparse.Namespace) -> Path:
    return Path(args.home or Path.home()).expanduser()


def _gib(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _terminal_text(value: object, *, limit: int = 4096) -> str:
    """Render untrusted plan text without terminal control characters."""

    text = str(value)
    rendered = "".join(
        character
        if not unicodedata.category(character).startswith("C")
        else "\N{REPLACEMENT CHARACTER}"
        for character in text
    )
    return rendered[:limit]


def _confirm_storage_apply(
    plan: dict[str, Any],
    *,
    confirm: str,
    rationale: str,
) -> HumanAuthorization | None:
    """Collect a decision-scoped confirmation from the controlling terminal."""

    prompt = "\n".join(
        [
            "",
            "=== HUMAN CONFIRMATION REQUIRED (storage cleanup) ===",
            f"  transaction: {_terminal_text(plan['transaction'])}",
            f"  plan sha256: {_terminal_text(plan['plan_sha256'])}",
            f"  home:        {_terminal_text(plan['home'])}",
            f"  candidates:  {len(plan['candidates'])}",
            f"  estimated:   {_gib(plan['estimated_reclaim_bytes'])}",
            f"  because:     {_terminal_text(rationale)}",
            "  no process will be stopped; every candidate is revalidated.",
            f"Type '{_terminal_text(confirm)}' to confirm, anything else aborts: ",
        ]
    )
    scope = storage_authorization_scope(
        plan["plan_sha256"],
        plan["transaction"],
        rationale,
    )
    return _broker_for_reader(_TTY_READER).confirm_token(
        confirm,
        prompt,
        scope=scope,
    )


def _render_audit(payload: dict[str, Any]) -> None:
    disk = payload["disk"]
    print("AFS storage audit (read-only, online)")
    print(f"Home: {_terminal_text(payload['home'])}")
    print(f"Disk free: {_gib(disk['free_bytes'])} ({disk['pressure']})")
    print(
        f"Eligible cleanup: {payload['eligible_count']} item(s), "
        f"up to {_gib(payload['estimated_reclaim_bytes'])}"
    )
    print()
    print("Bounded cleanup candidates:")
    if not payload["candidates"]:
        print("  none")
    for candidate in payload["candidates"]:
        state = "eligible" if candidate["eligible"] else "blocked"
        reasons = ", ".join(_terminal_text(reason) for reason in candidate["blocked_reasons"])
        suffix = f" ({reasons})" if reasons else ""
        print(
            f"  [{state}] {_terminal_text(candidate['category'])}: "
            f"{_terminal_text(candidate['path'])} "
            f"[{_gib(candidate['estimated_reclaim_bytes'])}]{suffix}"
        )
    print()
    print("Protected/informational footprints:")
    for footprint in payload["footprints"]:
        issue = f" ({_terminal_text(footprint['issue'])})" if footprint["issue"] else ""
        print(
            f"  {_terminal_text(footprint['name'])}: "
            f"{_gib(footprint['allocated_bytes'])} "
            f"at {_terminal_text(footprint['path'])}{issue}"
        )
    snapshots = payload["local_snapshots"]
    print()
    print(f"Local snapshots: {len(snapshots)} (informational only; never auto-deleted)")
    for snapshot in snapshots:
        print(f"  {_terminal_text(snapshot)}")
    print()
    print("No process was stopped and no file was changed.")
    print(_terminal_text(payload["safety"]["note"]))


def storage_audit_command(args: argparse.Namespace) -> int:
    try:
        payload = audit_storage(
            _home(args),
            stale_days=args.stale_days,
        )
    except (OSError, ValueError) as exc:
        error = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
        if args.json:
            _print_json(error)
        else:
            print(f"Storage audit blocked: {_terminal_text(error['error'])}")
        return 2
    if args.json:
        _print_json(payload)
    else:
        _render_audit(payload)
    return 0


def storage_plan_command(args: argparse.Namespace) -> int:
    try:
        plan = build_storage_plan(
            _home(args),
            stale_days=args.stale_days,
        )
        output = write_storage_plan(Path(args.output).expanduser(), plan)
    except (OSError, ValueError) as exc:
        error = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
        if args.json:
            _print_json(error)
        else:
            print(f"Storage plan blocked: {_terminal_text(error['error'])}")
        return 2
    response = {
        "status": "planned",
        "plan": str(output),
        "transaction": plan["transaction"],
        "candidate_count": len(plan["candidates"]),
        "estimated_reclaim_bytes": plan["estimated_reclaim_bytes"],
        "expires_at_ns": plan["expires_at_ns"],
    }
    if args.json:
        _print_json(response)
    else:
        rendered_output = _terminal_text(output)
        rendered_transaction = _terminal_text(response["transaction"])
        print(f"Storage plan: {rendered_output}")
        print(f"Candidates: {response['candidate_count']}")
        print(f"Estimated reclaim: {_gib(response['estimated_reclaim_bytes'])}")
        print(f"Transaction: {rendered_transaction}")
        if response["candidate_count"]:
            print("Review the JSON plan, then apply it with the exact transaction and a rationale:")
            print(
                f"  afs storage apply --plan {rendered_output} "
                f'--confirm {rendered_transaction} --because "..."'
            )
        else:
            print("Nothing is eligible; apply will refuse an empty plan.")
    return 0


def storage_apply_command(args: argparse.Namespace) -> int:
    try:
        plan_path = Path(args.plan).expanduser()
        plan = load_storage_plan(plan_path)
        if args.confirm != plan["transaction"]:
            raise StorageSafetyError("confirmation must exactly match the storage plan transaction")
        if not plan["candidates"]:
            raise StorageSafetyError("storage plan has no eligible candidates")
        authorization = _confirm_storage_apply(
            plan,
            confirm=args.confirm,
            rationale=args.because,
        )
        if authorization is None:
            message = (
                "storage apply requires an interactive human confirmation on "
                "the controlling terminal; no cleanup was started"
            )
            if args.json:
                _print_json({"status": "blocked", "error": message})
            else:
                print(message, file=sys.stderr)
            return 2
        receipt = apply_storage_plan(
            plan_path,
            confirm=args.confirm,
            because=args.because,
            authorization=authorization,
        )
    except StorageApplyError as exc:
        if args.json:
            _print_json(exc.receipt)
        else:
            print(f"Storage apply stopped: {_terminal_text(exc)}")
            if exc.receipt.get("receipt_written"):
                print(f"Receipt: {_terminal_text(exc.receipt['receipt_path'])}")
            else:
                print(f"Durable claim: {_terminal_text(exc.receipt['claim_path'])}")
                print(
                    "No final receipt was persisted; inspect the transaction "
                    "journal before taking further action."
                )
        return 3
    except (OSError, StorageSafetyError, ValueError) as exc:
        error = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
        if args.json:
            _print_json(error)
        else:
            print(f"Storage apply blocked: {_terminal_text(error['error'])}")
        return 2
    if args.json:
        _print_json(receipt)
    else:
        print(f"Storage cleanup applied: {len(receipt['deleted'])} item(s).")
        print(f"Receipt: {_terminal_text(receipt['receipt_path'])}")
        print("No process was stopped.")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register the storage audit/plan/apply command group."""

    parser = subparsers.add_parser(
        "storage",
        help="Audit storage and apply narrow, human-confirmed cleanup plans.",
    )
    commands = parser.add_subparsers(dest="storage_command")

    audit = commands.add_parser(
        "audit",
        help="Report disk pressure, bounded cleanup candidates, and protected footprints.",
    )
    audit.add_argument("--home", help="Home root override (default: current home).")
    audit.add_argument(
        "--stale-days",
        type=int,
        default=DEFAULT_STALE_DAYS,
        help=f"Minimum candidate age (default: {DEFAULT_STALE_DAYS} days).",
    )
    audit.add_argument("--json", action="store_true", help="Output JSON.")
    audit.set_defaults(func=storage_audit_command, _skip_cli_history=True)

    plan = commands.add_parser(
        "plan",
        help="Write an exact, expiring JSON plan of currently eligible candidates.",
    )
    plan.add_argument("--home", help="Home root override (default: current home).")
    plan.add_argument(
        "--stale-days",
        type=int,
        default=DEFAULT_STALE_DAYS,
        help=f"Minimum candidate age (default: {DEFAULT_STALE_DAYS} days).",
    )
    plan.add_argument("--output", required=True, help="New JSON plan path.")
    plan.add_argument("--json", action="store_true", help="Output JSON.")
    plan.set_defaults(func=storage_plan_command, _skip_cli_history=True)

    apply = commands.add_parser(
        "apply",
        help="Apply one exact plan after revalidation; never stops processes.",
    )
    apply.add_argument("--plan", required=True, help="Plan JSON path.")
    apply.add_argument(
        "--confirm",
        required=True,
        help="Exact transaction printed by storage plan.",
    )
    apply.add_argument(
        "--because",
        required=True,
        help="Non-empty human rationale recorded in the receipt.",
    )
    apply.add_argument("--json", action="store_true", help="Output JSON.")
    apply.set_defaults(func=storage_apply_command, _skip_cli_history=True)
