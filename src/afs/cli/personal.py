"""``afs personal`` — load personal context for personalized agents.

Opt-in only. Loads a personal-context directory containing ``profile.toml``
and ``manifest.toml`` (declaring named conversation modes). The default
location is ``~/.config/afs/personal``; override with the
``AFS_PERSONAL_CONTEXT_ROOT`` env var or ``--context-root``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..personal_context import (
    default_context_root,
    list_modes,
    load_personal_context,
    render_personal_context,
)


def personal_load_command(args: argparse.Namespace) -> int:
    context_root = Path(args.context_root).expanduser() if args.context_root else None
    try:
        if args.json:
            payload = load_personal_context(args.mode, context_root=context_root)
            data = {
                "mode": payload.mode,
                "tone": payload.tone,
                "bias_warning": payload.bias_warning,
                "profile_text": payload.profile_text,
                "files": [
                    {"path": rel, "content": content}
                    for rel, content in payload.files
                ],
                "missing": payload.missing,
            }
            print(json.dumps(data, indent=2))
        else:
            print(render_personal_context(args.mode, context_root=context_root))
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"afs personal: {exc}", file=sys.stderr)
        return 1


def personal_modes_command(args: argparse.Namespace) -> int:
    context_root = Path(args.context_root).expanduser() if args.context_root else None
    try:
        modes = list_modes(context_root=context_root)
    except Exception as exc:
        print(f"afs personal: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"modes": modes}, indent=2))
    else:
        for mode in modes:
            print(mode)
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "personal",
        help="Load personal context (opt-in, requires explicit mode).",
    )
    sub = parser.add_subparsers(dest="personal_command")

    load_parser = sub.add_parser(
        "load",
        help="Render personal context for a mode as markdown (or JSON).",
    )
    load_parser.add_argument(
        "mode",
        help=(
            "Conversation mode declared in the manifest.toml of the personal "
            "context root (e.g. claudia, advice, checkin)."
        ),
    )
    load_parser.add_argument(
        "--context-root",
        help=(
            "Override personal context root. Defaults to "
            "$AFS_PERSONAL_CONTEXT_ROOT or ~/.config/afs/personal."
        ),
    )
    load_parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    load_parser.set_defaults(func=personal_load_command)

    modes_parser = sub.add_parser("modes", help="List available conversation modes.")
    modes_parser.add_argument(
        "--context-root",
        help=(
            "Override personal context root. Defaults to "
            "$AFS_PERSONAL_CONTEXT_ROOT or ~/.config/afs/personal."
        ),
    )
    modes_parser.add_argument("--json", action="store_true", help="Emit JSON.")
    modes_parser.set_defaults(func=personal_modes_command)
