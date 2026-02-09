"""Claude Code log analysis CLI commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..claude.session_report import build_session_report, render_session_report_markdown
from ..context_fs import ContextFileSystem
from ..models import MountType
from ._utils import load_manager, resolve_context_paths


def claude_session_report_command(args: argparse.Namespace) -> int:
    claude_root = Path(args.claude_root).expanduser().resolve() if args.claude_root else None
    report = build_session_report(
        args.session,
        claude_root=claude_root,
        include_subagents=not args.no_subagents,
        max_subagent_chars=args.max_subagent_chars,
    )

    if args.json:
        payload = {
            "session_id": report.paths.session_id,
            "transcript_path": str(report.paths.transcript_path) if report.paths.transcript_path else None,
            "artifacts_dir": str(report.paths.artifacts_dir) if report.paths.artifacts_dir else None,
            "debug_log_path": str(report.paths.debug_log_path) if report.paths.debug_log_path else None,
            "project_slug": report.paths.project_slug,
            "cwd": report.cwd,
            "git_branch": report.git_branch,
            "version": report.version,
            "slug": report.slug,
            "start_timestamp": report.start_timestamp,
            "end_timestamp": report.end_timestamp,
            "models": list(report.models),
            "tool_calls": dict(report.tool_calls),
            "subagents": [
                {
                    "path": str(sub.path),
                    "agent_id": sub.agent_id,
                    "model": sub.model,
                    "cwd": sub.cwd,
                    "git_branch": sub.git_branch,
                    "slug": sub.slug,
                    "tool_calls": dict(sub.tool_calls),
                }
                for sub in report.subagents
            ],
            "warnings": list(report.paths.warnings),
        }
        print(json.dumps(payload, indent=2))
        return 0

    markdown = render_session_report_markdown(report)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        if out_path.exists() and not args.force:
            print(f"refusing to overwrite existing file: {out_path}")
            return 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"wrote: {out_path}")
        return 0

    if args.write_scratchpad:
        config_path = Path(args.config) if args.config else None
        manager = load_manager(config_path)
        _project_path, context_path, _context_root, _context_dir = resolve_context_paths(args, manager)
        fs = ContextFileSystem(manager, context_path)

        relative = args.scratchpad_path
        if not relative:
            prefix = report.paths.session_id[:8]
            relative = f"claude-session-{prefix}.md"

        target, _root = fs.resolve_path(MountType.SCRATCHPAD, relative)
        if target.exists() and not args.force:
            print(f"refusing to overwrite existing file: {target}")
            return 1

        fs.write_text(
            MountType.SCRATCHPAD,
            relative,
            markdown,
            mkdirs=True,
        )
        print(f"wrote: {target}")
        return 0

    print(markdown, end="")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    claude_parser = subparsers.add_parser("claude", help="Claude Code log analysis.")
    claude_sub = claude_parser.add_subparsers(dest="claude_command")

    report_parser = claude_sub.add_parser(
        "session-report", help="Generate a session report from Claude logs."
    )
    report_parser.add_argument("--config", help="Config path.")
    report_parser.add_argument("--path", help="Project path for context writes.")
    report_parser.add_argument("--context-root", help="Context root override.")
    report_parser.add_argument("--context-dir", help="Context directory name.")
    report_parser.add_argument("--claude-root", help="Claude root (default: ~/.claude).")
    report_parser.add_argument("--session", required=True, help="Session UUID or prefix.")
    report_parser.add_argument(
        "--max-subagent-chars",
        type=int,
        default=4000,
        help="Max chars captured from last assistant message per subagent.",
    )
    report_parser.add_argument(
        "--no-subagents",
        action="store_true",
        help="Do not scan per-session subagent artifacts.",
    )
    report_parser.add_argument("--output", help="Write markdown report to this path.")
    report_parser.add_argument(
        "--write-scratchpad",
        action="store_true",
        help="Write report into the project's scratchpad mount.",
    )
    report_parser.add_argument(
        "--scratchpad-path",
        help="Scratchpad relative path override (used with --write-scratchpad).",
    )
    report_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    report_parser.add_argument("--json", action="store_true", help="Output JSON.")
    report_parser.set_defaults(func=claude_session_report_command)

