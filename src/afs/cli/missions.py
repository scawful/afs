"""CLI verbs for durable background missions.

Missions track long-running/background work as first-class state so it survives
across sessions and subagents. Active missions are also surfaced into the session
context block, so a resumed session sees what is already in flight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..human_provenance import (
    HumanAuthorization,
    _broker_for_reader,
)
from ..missions import MissionNotFoundError, MissionStore
from ._utils import load_manager, resolve_context_paths

# Test seam: production uses the platform controlling-terminal backend.
_TTY_READER = None


def _store(args: argparse.Namespace) -> MissionStore:
    config_path = Path(args.config) if getattr(args, "config", None) else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args, manager
    )
    return MissionStore(context_path, config=manager.config)


def _print_mission_line(mission) -> None:
    steps = f"; next: {mission.next_steps[0]}" if mission.next_steps else ""
    owner = f" [{mission.owner}]" if mission.owner else ""
    print(f"{mission.mission_id}  ({mission.status}){owner}  {mission.title}{steps}")


def _confirm_flag_acceptance(
    text: str, *, scope: str
) -> HumanAuthorization | None:
    """Confirm a ``--acceptance`` flag value on the controlling terminal.

    Acceptance is the human-authored definition of done that later outcome
    scoring calibrates against, so setting, changing, or clearing it requires
    typing ``human`` on the tty — piped stdin cannot satisfy it, and headless
    callers (agents) are refused. Returns a decision-scoped broker
    authorization, or ``None`` to refuse.
    """
    shown = text if text else "(clear the existing acceptance)"
    authorization = _broker_for_reader(_TTY_READER).confirm_token(
        "human",
        "\n".join(
            [
                "",
                "Acceptance is the human-authored definition of done:",
                f"  {shown}",
                "Type 'human' to confirm you (not an agent) authored this change: ",
            ]
        ),
        scope=scope,
    )
    if authorization is None:
        print(
            "acceptance is human-authored; setting or clearing it requires an "
            "interactive terminal confirmation. Agents must leave it unset and "
            "surface the nudge instead.",
            file=sys.stderr,
        )
    return authorization


def _prompt_for_acceptance(
    args: argparse.Namespace, *, title: str, store: MissionStore
) -> tuple[str, HumanAuthorization | None]:
    """Ask the human to author acceptance when creating a mission.

    The prompt is written to and read from the controlling terminal, so it
    never contaminates stdout and piped stdin cannot answer it. Headless
    callers are never blocked: without a terminal the prompt is skipped and
    the follow-up nudge is printed instead. Never prompts in ``--json`` mode.
    Returns ``(acceptance, authorization)``.
    """
    if getattr(args, "json", False):
        return "", None
    result = _broker_for_reader(_TTY_READER).read_line(
        "Acceptance — what does done look like? (enter to skip): ",
        scope=lambda response: store.human_acceptance_scope(
            "create", title.strip(), response.strip()
        ),
    )
    if result is None:
        return "", None
    line, authorization = result
    acceptance = line.strip()
    return acceptance, (authorization if acceptance else None)


def mission_create_command(args: argparse.Namespace) -> int:
    store = _store(args)
    flag = str(getattr(args, "acceptance", None) or "").strip()
    if flag:
        scope = store.human_acceptance_scope(
            "create", args.title.strip(), flag
        )
        authorization = _confirm_flag_acceptance(flag, scope=scope)
        if authorization is None:
            return 2
        acceptance = flag
    else:
        acceptance, authorization = _prompt_for_acceptance(
            args, title=args.title, store=store
        )
    mission = store.create(
        title=args.title,
        summary=getattr(args, "summary", "") or "",
        owner=getattr(args, "owner", "") or "",
        acceptance=acceptance,
        acceptance_authorization=authorization,
        next_steps=list(getattr(args, "next_step", None) or []),
        tags=list(getattr(args, "tag", None) or []),
    )
    if getattr(args, "json", False):
        print(json.dumps(mission.to_dict(), indent=2))
    else:
        print(f"created: {mission.mission_id}")
        _print_mission_line(mission)
        if not mission.acceptance:
            print(
                "acceptance not set; add one with: "
                f"afs mission update {mission.mission_id} --acceptance '<what done looks like>'"
            )
    return 0


def mission_list_command(args: argparse.Namespace) -> int:
    store = _store(args)
    status = getattr(args, "status", None)
    missions = store.list(status=status, limit=getattr(args, "limit", 50))
    if getattr(args, "json", False):
        print(json.dumps([m.to_dict() for m in missions], indent=2))
        return 0
    if not missions:
        print("no missions")
        return 0
    for mission in missions:
        _print_mission_line(mission)
    return 0


def mission_show_command(args: argparse.Namespace) -> int:
    store = _store(args)
    mission = store.get(args.mission_id)
    if mission is None:
        print(f"mission not found: {args.mission_id}")
        return 1
    print(json.dumps(mission.to_dict(), indent=2))
    return 0


def mission_update_command(args: argparse.Namespace) -> int:
    store = _store(args)
    acceptance = getattr(args, "acceptance", None)
    acceptance_authorization = None
    if acceptance is not None:
        acceptance = str(acceptance).strip()
        scope = store.human_acceptance_scope(
            "update", args.mission_id, acceptance
        )
        acceptance_authorization = _confirm_flag_acceptance(
            acceptance, scope=scope
        )
        if acceptance_authorization is None:
            return 2
    try:
        mission = store.update(
            args.mission_id,
            status=getattr(args, "status", None),
            summary=getattr(args, "summary", None),
            owner=getattr(args, "owner", None),
            acceptance=acceptance,
            acceptance_authorization=acceptance_authorization,
            next_steps=list(args.next_step) if getattr(args, "next_step", None) else None,
            blockers=list(args.blocker) if getattr(args, "blocker", None) else None,
            link_session=getattr(args, "link_session", None),
            link_handoff=getattr(args, "link_handoff", None),
            add_tags=list(getattr(args, "tag", None) or []),
            note=getattr(args, "note", None),
            actor=getattr(args, "actor", "") or "",
        )
    except MissionNotFoundError:
        print(f"mission not found: {args.mission_id}")
        return 1
    except ValueError as exc:
        print(str(exc))
        return 2
    if getattr(args, "json", False):
        print(json.dumps(mission.to_dict(), indent=2))
    else:
        print(f"updated: {mission.mission_id}")
        _print_mission_line(mission)
    return 0


def _add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Config path.")
    parser.add_argument("--path", help="Project path.")
    parser.add_argument("--context-root", help="Context root override.")
    parser.add_argument("--context-dir", help="Context directory name.")


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    mission_parser = subparsers.add_parser(
        "mission", help="Track durable background missions across sessions."
    )
    mission_sub = mission_parser.add_subparsers(dest="mission_command")

    create_parser = mission_sub.add_parser("create", help="Create a mission.")
    _add_context_args(create_parser)
    create_parser.add_argument("--title", required=True, help="Mission title.")
    create_parser.add_argument("--summary", help="One-line mission summary.")
    create_parser.add_argument("--owner", help="Owning agent or session.")
    create_parser.add_argument(
        "--acceptance",
        help="Human-authored definition of done; prompted for on a tty when omitted.",
    )
    create_parser.add_argument(
        "--next-step", action="append", help="A next step (repeatable)."
    )
    create_parser.add_argument("--tag", action="append", help="A tag (repeatable).")
    create_parser.add_argument("--json", action="store_true", help="Output JSON.")
    create_parser.set_defaults(func=mission_create_command)

    list_parser = mission_sub.add_parser("list", help="List missions.")
    _add_context_args(list_parser)
    list_parser.add_argument(
        "--status", help="Filter by status (active, blocked, done, abandoned)."
    )
    list_parser.add_argument("--limit", type=int, default=50, help="Max missions.")
    list_parser.add_argument("--json", action="store_true", help="Output JSON.")
    list_parser.set_defaults(func=mission_list_command)

    show_parser = mission_sub.add_parser("show", help="Show a mission as JSON.")
    _add_context_args(show_parser)
    show_parser.add_argument("mission_id", help="Mission id.")
    show_parser.set_defaults(func=mission_show_command)

    update_parser = mission_sub.add_parser("update", help="Update a mission.")
    _add_context_args(update_parser)
    update_parser.add_argument("mission_id", help="Mission id.")
    update_parser.add_argument(
        "--status", help="New status (active, blocked, done, abandoned)."
    )
    update_parser.add_argument("--summary", help="Replace the summary.")
    update_parser.add_argument("--owner", help="Replace the owner.")
    update_parser.add_argument(
        "--acceptance", help="Replace the human-authored definition of done."
    )
    update_parser.add_argument(
        "--next-step", action="append", help="Replace next steps (repeatable)."
    )
    update_parser.add_argument(
        "--blocker", action="append", help="Replace blockers (repeatable)."
    )
    update_parser.add_argument("--link-session", help="Link a working session id.")
    update_parser.add_argument("--link-handoff", help="Link a handoff session id.")
    update_parser.add_argument("--tag", action="append", help="Add a tag (repeatable).")
    update_parser.add_argument("--note", help="Append a progress log note.")
    update_parser.add_argument("--actor", help="Actor recorded on the log note.")
    update_parser.add_argument("--json", action="store_true", help="Output JSON.")
    update_parser.set_defaults(func=mission_update_command)
