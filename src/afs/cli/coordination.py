"""``afs inbox``, ``afs board`` and ``afs lease``: shared coordination surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .. import leases
from ..coordination import (
    BOARD_FIELDS,
    BOARD_FILE,
    INBOX_FILE,
    append_inbox_line,
    format_inbox_line,
    mark_inbox_done,
    read_board,
    read_inbox,
    resolve_coordination_file,
    set_board_section,
)
from ..models import MountType
from ..scopes import resolve_scope, visible_mount_roots
from ._utils import load_manager, resolve_context_paths


def _add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Config path.")
    parser.add_argument("--path", help="Project or working path.")
    parser.add_argument("--context-root", help="Context root override.")
    parser.add_argument("--context-dir", help="Context directory name.")
    parser.add_argument("--file", help="Explicit coordination file path (skips scratchpad lookup).")


def _coordination_file(args: argparse.Namespace, name: str) -> Path:
    explicit = getattr(args, "file", None)
    if explicit:
        return Path(explicit).expanduser().resolve()
    config = Path(args.config).expanduser().resolve() if getattr(args, "config", None) else None
    manager = load_manager(config)
    project, context, _override, _directory = resolve_context_paths(args, manager)
    scope = resolve_scope(context, requester_path=project)
    scratchpad = manager.resolve_mount_root(context, MountType.SCRATCHPAD)
    roots = visible_mount_roots(scratchpad, mount_type=MountType.SCRATCHPAD, scoped=scope)
    return resolve_coordination_file(roots, name)


def inbox_list_command(args: argparse.Namespace) -> int:
    path = _coordination_file(args, INBOX_FILE)
    inbox, _digest = read_inbox(path)
    if args.to:
        entries = inbox.open_for(args.to)
    else:
        entries = [entry for entry in inbox.entries if args.all or not entry.done]
    entries = entries[-args.limit :] if args.limit else entries
    if args.json:
        print(
            json.dumps(
                {
                    "path": str(path),
                    "goal": inbox.goal,
                    "unparsed_lines": inbox.unparsed_lines,
                    "entries": [entry.to_dict() for entry in entries],
                },
                indent=2,
            )
        )
        return 0
    print(f"inbox: {path}")
    if inbox.goal:
        print(f"goal: {inbox.goal}")
    if not entries:
        print("no open lines" + (f" for {args.to}" if args.to else ""))
    for entry in entries:
        marker = "DONE " if entry.done else ""
        print(f"{entry.line_no:>5}  {marker}{entry.raw.strip()}")
    return 0


def inbox_add_command(args: argparse.Namespace) -> int:
    path = _coordination_file(args, INBOX_FILE)
    line = format_inbox_line(args.from_agent, args.to, args.ask, path=args.ref or "")
    append_inbox_line(path, line)
    if args.json:
        print(json.dumps({"path": str(path), "line": line}, indent=2))
    else:
        print(line)
    return 0


def inbox_done_command(args: argparse.Namespace) -> int:
    if args.line is None and not args.match:
        raise ValueError("pass --line N or --match TEXT to choose the inbox line")
    path = _coordination_file(args, INBOX_FILE)
    try:
        entry = mark_inbox_done(path, line_no=args.line, match=args.match or "")
    except LookupError as exc:
        print(str(exc))
        return 1
    if args.json:
        print(json.dumps({"path": str(path), "done": entry.to_dict()}, indent=2))
    else:
        print(f"DONE line {entry.line_no}: {entry.raw.strip()}")
    return 0


def board_show_command(args: argparse.Namespace) -> int:
    path = _coordination_file(args, BOARD_FILE)
    board, _digest = read_board(path)
    sections = list(board.sections)
    if args.owner:
        section = board.section(args.owner)
        sections = [section] if section else []
    if args.json:
        print(
            json.dumps(
                {
                    "path": str(path),
                    "pinned": board.pinned(),
                    "sections": [section.to_dict() for section in sections],
                },
                indent=2,
            )
        )
        return 0
    print(f"board: {path}")
    for label, value in board.pinned().items():
        print(f"{label}: {value}")
    for section in sections:
        print(f"\n## {section.owner} — {section.stamp}")
        for label in BOARD_FIELDS:
            if label in section.fields:
                print(f"{label}: {section.fields[label]}")
    return 0


def board_set_command(args: argparse.Namespace) -> int:
    path = _coordination_file(args, BOARD_FILE)
    fields = {
        "Result": args.result,
        "Ran": args.ran,
        "Next": args.next_step,
        "Holds": args.holds,
        "Handoff": args.handoff,
    }
    lines = set_board_section(path, args.owner, fields)
    if args.json:
        print(json.dumps({"path": str(path), "section": lines}, indent=2))
    else:
        print("\n".join(lines))
    return 0


def _print_lease(payload: dict[str, Any], as_json: bool, summary: str) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(summary)


def _report_wait(state: dict[str, Any]) -> None:
    print(f"waiting: {leases.describe(state)}", file=sys.stderr)


def lease_status_command(args: argparse.Namespace) -> int:
    state = leases.inspect(leases.lease_path(args.name))
    _print_lease(state, args.json, leases.describe(state))
    return 0


def lease_acquire_command(args: argparse.Namespace) -> int:
    path = leases.lease_path(args.name)
    try:
        metadata = leases.acquire(
            path,
            args.owner,
            pid=args.pid,
            task=args.task or "",
            source=args.source or "",
            ttl_minutes=args.ttl_minutes,
            wait_seconds=args.wait,
            on_wait=None if args.json else _report_wait,
        )
    except leases.LeaseBusy as exc:
        _print_lease({"acquired": False, "state": exc.state}, args.json, str(exc))
        return 2
    _print_lease(
        {"acquired": True, "path": str(path), "metadata": metadata},
        args.json,
        f"acquired {path} as {args.owner}; release with: afs lease release {args.name} "
        f"--token {metadata['token']}",
    )
    return 0


def lease_release_command(args: argparse.Namespace) -> int:
    result = leases.release(
        leases.lease_path(args.name), token=args.token or "", owner=args.owner or ""
    )
    summary = "released" if result["released"] else f"not released: {result['reason']}"
    _print_lease(result, args.json, summary)
    return 0 if result["released"] else 1


def lease_reclaim_command(args: argparse.Namespace) -> int:
    result = leases.reclaim(leases.lease_path(args.name), apply=args.apply)
    state = result["state"]
    if result.get("reason"):
        summary = f"not reclaimable: {result['reason']} ({leases.describe(state)})"
    elif args.apply:
        summary = f"reclaimed stale lease from {state.get('owner')} (pid {state.get('pid')})"
    else:
        summary = f"stale and reclaimable (dry run): {leases.describe(state)}; pass --apply"
    _print_lease(result, args.json, summary)
    return 0 if not result.get("reason") else 1


def lease_run_command(args: argparse.Namespace) -> int:
    command = list(args.run_argv)
    if not command:
        raise ValueError(
            "pass the command after --, e.g. afs lease run oos-heavy --owner me -- make"
        )
    try:
        return leases.run(
            leases.lease_path(args.name),
            args.owner,
            command,
            task=args.task or "",
            source=args.source or "",
            wait_seconds=args.wait,
            on_wait=_report_wait,
        )
    except leases.LeaseBusy as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _register_lease(subparsers: argparse._SubParsersAction) -> None:
    lease = subparsers.add_parser(
        "lease",
        help="Machine-wide resource leases (directory locks such as /tmp/oos-heavy.lock).",
    )
    commands = lease.add_subparsers(dest="lease_command")

    def add_name(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("name", help="Lease name (oos-heavy -> /tmp/oos-heavy.lock) or path.")
        parser.add_argument("--json", action="store_true")

    status = commands.add_parser(
        "status",
        help="Show owner, liveness and leftover workers (free/held/owner-exited/stale/legacy).",
    )
    add_name(status)
    status.set_defaults(func=lease_status_command)

    run = commands.add_parser(
        "run",
        help="Wait for the lease, run a command in its own process group, release after it drains.",
    )
    add_name(run)
    run.add_argument("--owner", required=True)
    run.add_argument("--task")
    run.add_argument("--source", help="Work tree; leftover processes under it keep the lease busy.")
    run.add_argument("--wait", type=float, default=leases.DEFAULT_WAIT_SECONDS)
    run.add_argument("run_argv", nargs="*", metavar="command", help="-- command [args...]")
    run.set_defaults(func=lease_run_command)

    acquire = commands.add_parser(
        "acquire", help="Take the lease across several steps (prefer `run` for one command)."
    )
    add_name(acquire)
    acquire.add_argument("--owner", required=True)
    acquire.add_argument("--task")
    acquire.add_argument("--source")
    acquire.add_argument("--pid", type=int, help="Long-lived holder PID (default: this process).")
    acquire.add_argument(
        "--ttl-minutes",
        type=float,
        default=leases.DEFAULT_ACQUIRE_TTL_MINUTES,
        help="The lease stays held this long after the holder PID exits.",
    )
    acquire.add_argument("--wait", type=float, default=leases.DEFAULT_WAIT_SECONDS)
    acquire.set_defaults(func=lease_acquire_command)

    release = commands.add_parser("release", help="Release your lease (token or owner must match).")
    add_name(release)
    release.add_argument("--token")
    release.add_argument("--owner")
    release.set_defaults(func=lease_release_command)

    reclaim = commands.add_parser(
        "reclaim",
        help="Remove a stale lease: owner exited, no leftover workers, TTL passed. Dry run by default.",
    )
    add_name(reclaim)
    reclaim.add_argument("--apply", action="store_true")
    reclaim.set_defaults(func=lease_reclaim_command)


def worktrees_report_command(args: argparse.Namespace) -> int:
    from ..worktree_report import worktree_report

    report = worktree_report(Path(args.repo), older_than_days=args.older_than_days)
    rows = report["worktrees"] if args.all else report["candidates"]
    if args.json:
        print(json.dumps({**report, "worktrees": report["worktrees"][: args.limit]}, indent=2))
        return 0
    print(f"worktrees: {report['repo']} (report only; nothing is removed)")
    print(
        f"  {report['linked_count']} linked, {report['dirty_count']} dirty, "
        f"{report['candidate_count']} removal candidates "
        f"(>= {args.older_than_days:g} days idle vs {report['default_branch']})"
    )
    for row in rows[: args.limit]:
        state = "dirty" if row["dirty"] else "clean" if row["dirty"] is False else "?"
        reason = f"  <- {row['candidate']}" if row["candidate"] else ""
        print(
            f"  - {row['idle_days']:>5g}d  {state:5}  {row['branch'] or row['head']}  "
            f"{row['path']}{reason}"
        )
    if report["candidates"]:
        print("  remove one with: git worktree remove <path>  (prune missing: git worktree prune)")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    _register_lease(subparsers)
    inbox = subparsers.add_parser(
        "inbox",
        help="Read or append the scratchpad inbox.md (one line per ask, claim or release).",
    )
    inbox_commands = inbox.add_subparsers(dest="inbox_command")

    listing = inbox_commands.add_parser("list", help="Show open inbox lines.")
    _add_context_args(listing)
    listing.add_argument("--to", help="Only open lines addressed to this agent (or its family).")
    listing.add_argument("--all", action="store_true", help="Include DONE lines.")
    listing.add_argument("--limit", type=int, default=30)
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=inbox_list_command)

    add = inbox_commands.add_parser("add", help="Append one ask/claim/release line (UTC-stamped).")
    _add_context_args(add)
    add.add_argument("--from", dest="from_agent", required=True)
    add.add_argument("--to", required=True, help="Recipient(s), e.g. codex/cursor or all.")
    add.add_argument("--ref", help="Handoff path for detail, appended as (path).")
    add.add_argument("ask", help="One line; the whole inbox line must stay within 240 chars.")
    add.add_argument("--json", action="store_true")
    add.set_defaults(func=inbox_add_command)

    done = inbox_commands.add_parser("done", help="Prefix DONE on one open line.")
    _add_context_args(done)
    done.add_argument("--line", type=int, help="Line number from `afs inbox list`.")
    done.add_argument("--match", help="Substring that identifies exactly one open line.")
    done.add_argument("--json", action="store_true")
    done.set_defaults(func=inbox_done_command)

    board = subparsers.add_parser(
        "board",
        help="Read or update the scratchpad agent_board.md (one section per owner).",
    )
    board_commands = board.add_subparsers(dest="board_command")

    show = board_commands.add_parser("show", help="Show pinned lines and owner sections.")
    _add_context_args(show)
    show.add_argument("--owner")
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=board_show_command)

    set_section = board_commands.add_parser(
        "set", help="Overwrite one owner's section (other sections are untouched)."
    )
    _add_context_args(set_section)
    set_section.add_argument(
        "--owner", required=True, help="Section owner, e.g. Cursor or Codex-Lyra."
    )
    set_section.add_argument("--result", default="")
    set_section.add_argument("--ran", default="")
    set_section.add_argument("--next", dest="next_step", default="")
    set_section.add_argument(
        "--holds", default="", help="Locks, worktrees, file or ROM-range claims."
    )
    set_section.add_argument("--handoff", default="")
    set_section.add_argument("--json", action="store_true")
    set_section.set_defaults(func=board_set_command)

    worktrees = subparsers.add_parser(
        "worktrees", help="Report linked git worktrees and removal candidates (read-only)."
    )
    worktree_commands = worktrees.add_subparsers(dest="worktrees_command")
    report = worktree_commands.add_parser("report", help="List idle, merged or missing worktrees.")
    report.add_argument("--repo", default=".", help="Repository path (default: current directory).")
    report.add_argument("--older-than-days", type=float, default=7.0)
    report.add_argument("--all", action="store_true", help="List every linked worktree.")
    report.add_argument("--limit", type=int, default=25)
    report.add_argument("--json", action="store_true")
    report.set_defaults(func=worktrees_report_command)
