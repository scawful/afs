"""Claude Code log analysis CLI commands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..claude.doctor import inspect_claude_sessions, reap_claude_sessions
from ..claude.session_report import build_session_report, render_session_report_markdown
from ..context_fs import ContextFileSystem
from ..context_layout import LAYOUT_VERSION, detect_layout_version
from ..models import MountType
from ..scopes import resolve_scope
from ..scratchpad import ScratchpadStore
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
        project_path, context_path, _context_root, _context_dir = resolve_context_paths(
            args,
            manager,
        )
        if detect_layout_version(context_path) == LAYOUT_VERSION:
            if args.scratchpad_path:
                print(
                    "--scratchpad-path is not supported for v2 immutable drafts; "
                    "omit it and AFS will allocate a readable unique name"
                )
                return 2
            scoped = resolve_scope(
                context_path,
                requester_path=project_path,
                common=bool(getattr(args, "common", False)),
            )
            artifact = ScratchpadStore(
                context_path,
                scope_id=scoped.scope_id,
                config=manager.config,
            ).create(
                title=f"Claude session {report.paths.session_id[:8]}",
                body=markdown,
                project_id=scoped.project_id,
                agent_name="claude",
                author_kind="agent",
                provenance={
                    "source": "afs.claude.session-report",
                    "session_id": report.paths.session_id,
                },
            )
            print(f"wrote: {artifact.path}")
            return 0

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


def claude_setup_command(args: argparse.Namespace) -> int:
    """Configure Claude Code to use AFS MCP server."""
    from ..claude_integration import (
        default_claude_user_settings_path,
        generate_claude_md,
        generate_claude_settings,
        merge_claude_settings,
    )

    config_path = Path(args.config) if getattr(args, "config", None) else None
    manager = load_manager(config_path)
    project_path, context_path, _context_root, _context_dir = resolve_context_paths(args, manager)
    scope = getattr(args, "scope", "project")

    # Generate and merge settings
    afs_settings = generate_claude_settings(
        project_path,
        config=manager.config,
        config_path=config_path,
        include_project_context=scope == "project",
    )
    settings_path_arg = getattr(args, "settings_path", None)
    if settings_path_arg:
        settings_path = Path(settings_path_arg).expanduser().resolve()
    elif scope == "user":
        settings_path = default_claude_user_settings_path().resolve()
    else:
        settings_path = project_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if settings_path.exists():
        try:
            existing = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}

    merged = merge_claude_settings(existing, afs_settings)
    settings_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {settings_path}")

    if scope != "project":
        return 0

    # Generate CLAUDE.md
    claude_md_path = project_path / "CLAUDE.md"
    if not claude_md_path.exists() or getattr(args, "force", False):
        project_name = project_path.name
        content = generate_claude_md(project_name, str(context_path))
        claude_md_path.write_text(content, encoding="utf-8")
        print(f"wrote: {claude_md_path}")
    else:
        print(f"skipped: {claude_md_path} (exists, use --force to overwrite)")

    return 0


def claude_context_command(args: argparse.Namespace) -> int:
    """Output Claude-optimized context block."""
    from ..session_bootstrap import build_session_bootstrap, render_session_bootstrap

    config_path = Path(args.config) if getattr(args, "config", None) else None
    manager = load_manager(config_path)
    project_path, context_path, _context_root, _context_dir = resolve_context_paths(args, manager)

    summary = build_session_bootstrap(
        manager,
        context_path,
        project_path=project_path,
    )
    print(render_session_bootstrap(summary))
    return 0


def claude_hook_command(args: argparse.Namespace) -> int:
    """Emit AFS grounding as a Claude Code hook ``additionalContext`` payload.

    Wire this as a ``SessionStart`` / ``UserPromptSubmit`` hook so AFS context is
    pushed into the session automatically instead of waiting for an explicit prompt.
    Reads the hook's stdin JSON (event name, cwd, prompt) as Claude Code provides it,
    and degrades to a silent no-op (exit 0, no output) whenever context can't be
    resolved — a lifecycle hook must never break or spam the host session.
    """
    import sys

    from ..model_prompts import build_hook_injection
    from ..session_bootstrap import build_session_bootstrap

    stdin_payload: dict = {}
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
            if raw.strip():
                stdin_payload = json.loads(raw)
    except (OSError, ValueError):
        stdin_payload = {}
    if not isinstance(stdin_payload, dict):
        stdin_payload = {}

    event = (
        str(getattr(args, "event", "") or "").strip()
        or str(stdin_payload.get("hook_event_name") or "").strip()
        or "SessionStart"
    )
    prompt = str(stdin_payload.get("prompt") or "")

    # Resolve the workspace: explicit --path, else the hook's cwd, else current dir.
    if not getattr(args, "path", None):
        cwd = str(stdin_payload.get("cwd") or "").strip()
        if cwd:
            args.path = cwd

    injection = ""
    try:
        config_path = Path(args.config) if getattr(args, "config", None) else None
        manager = load_manager(config_path)
        project_path, context_path, _context_root, _context_dir = resolve_context_paths(
            args, manager
        )
        session_state = None
        if event != "UserPromptSubmit":
            skills_enabled = os.getenv("AFS_SESSION_SKILLS_MATCH_ENABLED", "1") != "0"
            skills_prompt = (
                os.getenv("AFS_SESSION_SKILLS_PROMPT", "").strip()[:8192]
                if skills_enabled
                else ""
            )
            session_state = build_session_bootstrap(
                manager,
                context_path,
                project_path=project_path,
                token_budget=0,
                record_event=False,
                skills_prompt=skills_prompt,
                include_skills=skills_enabled,
            )
        injection = build_hook_injection(
            event=event,
            context_path=context_path,
            session_state=session_state,
            prompt=prompt,
        )
    except Exception:
        injection = ""

    if not injection.strip():
        return 0

    if getattr(args, "raw", False):
        # Host-agnostic mode: emit the injection text only, for callers that wrap it
        # themselves (e.g. the hcode system-transform seam pushing into system[]).
        print(injection)
        return 0

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": event,
                    "additionalContext": injection,
                }
            }
        )
    )
    return 0


def claude_doctor_command(args: argparse.Namespace) -> int:
    """Inspect Claude session accumulation, bridge protection, and debug signals."""
    claude_root = Path(args.claude_root).expanduser().resolve() if args.claude_root else None
    report = inspect_claude_sessions(
        claude_root=claude_root,
        active_hours=args.active_hours,
        reap_after_hours=args.reap_after_hours,
        recent_debug_logs=args.recent_debug_logs,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    print(f"claude_root: {report.claude_root}")
    print(f"projects_root: {report.projects_root}")
    print(f"archive_root: {report.archive_root}")
    print(f"projects: {report.project_count}")
    print(f"sessions: {report.session_count}")
    counts = ", ".join(
        f"{name}={report.status_counts.get(name, 0)}"
        for name in ("active", "stale", "zombie", "protected")
    )
    print(f"status_counts: {counts}")
    candidates = sum(1 for session in report.sessions if session.reap_candidate)
    print(f"reap_candidates: {candidates}")

    if report.bridge_pointers:
        print("bridge_pointers:")
        for pointer in report.bridge_pointers[: args.limit]:
            env = f" env={pointer.environment_id}" if pointer.environment_id else ""
            session = f" session={pointer.session_id}" if pointer.session_id else ""
            source = f" source={pointer.source}" if pointer.source else ""
            print(f"  - {pointer.project_slug}:{session}{env}{source}")

    signals = report.debug_signals
    print(
        "recent_debug: "
        f"logs={signals.logs_scanned} "
        f"rate_limits={signals.rate_limit_errors} "
        f"permission_blocks={signals.permission_blocks} "
        f"timeouts={signals.timeout_errors} "
        f"missing_tools={signals.missing_tool_errors}"
    )
    if signals.mcp_servers:
        print("mcp_connect_ms:")
        for server in signals.mcp_servers[: args.limit]:
            print(
                f"  - {server.server}: avg={server.average_ms:.1f} max={server.max_ms} count={server.count}"
            )

    if report.sessions:
        print("sessions:")
        for session in report.sessions[: args.limit]:
            age_text = f"{session.age_hours:.1f}h" if session.age_hours is not None else "n/a"
            project = session.project_slug or "_orphans"
            reason_text = ", ".join(session.reasons) if session.reasons else "none"
            candidate_flag = " reap" if session.reap_candidate else ""
            print(
                f"  - {session.status}{candidate_flag} {age_text} {session.session_id} "
                f"project={project} last={session.last_activity_source or 'n/a'} reasons={reason_text}"
            )
    return 0


def claude_reap_command(args: argparse.Namespace) -> int:
    """Archive stale/zombie Claude sessions. Dry-run by default."""
    claude_root = Path(args.claude_root).expanduser().resolve() if args.claude_root else None
    archive_root = Path(args.archive_root).expanduser().resolve() if args.archive_root else None
    summary = reap_claude_sessions(
        claude_root=claude_root,
        active_hours=args.active_hours,
        reap_after_hours=args.reap_after_hours,
        recent_debug_logs=args.recent_debug_logs,
        apply=bool(args.apply),
        archive_root=archive_root,
        limit=args.limit,
    )

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    mode = "apply" if summary.apply else "dry-run"
    print(f"mode: {mode}")
    print(f"claude_root: {summary.claude_root}")
    print(f"archive_root: {summary.archive_root}")
    print(f"candidate_count: {summary.candidate_count}")
    print(f"moved_count: {summary.moved_count}")
    print(f"skipped_count: {summary.skipped_count}")
    if summary.manifest_path:
        print(f"manifest_path: {summary.manifest_path}")
    if summary.sessions:
        print("sessions:")
        for session in summary.sessions:
            age_text = f"{session.age_hours:.1f}h" if session.age_hours is not None else "n/a"
            project = session.project_slug or "_orphans"
            reason_text = ", ".join(session.reasons) if session.reasons else "none"
            print(
                f"  - {session.status} {age_text} {session.session_id} "
                f"project={project} reasons={reason_text}"
            )
    else:
        print("sessions: none")
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
        "--common",
        action="store_true",
        help="Write a v2 scratchpad report to shared common scope instead of the current project.",
    )
    report_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    report_parser.add_argument("--json", action="store_true", help="Output JSON.")
    report_parser.set_defaults(func=claude_session_report_command)

    # setup
    setup_parser = claude_sub.add_parser(
        "setup", help="Configure Claude Code to use AFS MCP server."
    )
    setup_parser.add_argument("--config", help="Config path.")
    setup_parser.add_argument("--path", help="Project path.")
    setup_parser.add_argument("--context-root", help="Context root override.")
    setup_parser.add_argument("--context-dir", help="Context directory name.")
    setup_parser.add_argument(
        "--scope",
        choices=("project", "user"),
        default="project",
        help="Write project-local settings (default) or user-level ~/.claude/settings.json.",
    )
    setup_parser.add_argument(
        "--settings-path",
        help="Explicit Claude settings.json target override.",
    )
    setup_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing CLAUDE.md."
    )
    setup_parser.set_defaults(func=claude_setup_command)

    # context
    context_parser = claude_sub.add_parser(
        "context", help="Output Claude-optimized context block."
    )
    context_parser.add_argument("--config", help="Config path.")
    context_parser.add_argument("--path", help="Project path.")
    context_parser.add_argument("--context-root", help="Context root override.")
    context_parser.add_argument("--context-dir", help="Context directory name.")
    context_parser.set_defaults(func=claude_context_command)

    # hook: runtime injector for SessionStart / UserPromptSubmit
    hook_parser = claude_sub.add_parser(
        "hook",
        help="Emit AFS grounding as a Claude Code hook additionalContext payload.",
    )
    hook_parser.add_argument("--config", help="Config path.")
    hook_parser.add_argument("--path", help="Project path (defaults to the hook cwd).")
    hook_parser.add_argument("--context-root", help="Context root override.")
    hook_parser.add_argument("--context-dir", help="Context directory name.")
    hook_parser.add_argument(
        "--event",
        help="Hook event name (SessionStart or UserPromptSubmit). "
        "Falls back to the stdin hook payload, then SessionStart.",
    )
    hook_parser.add_argument(
        "--raw",
        action="store_true",
        help="Emit the injection text only (no Claude hook JSON wrapper), "
        "for host-agnostic callers such as the hcode system-transform seam.",
    )
    hook_parser.set_defaults(func=claude_hook_command)

    doctor_parser = claude_sub.add_parser(
        "doctor",
        help="Inspect Claude session accumulation, debug signals, and reap candidates.",
    )
    doctor_parser.add_argument("--claude-root", help="Claude root (default: ~/.claude).")
    doctor_parser.add_argument(
        "--active-hours",
        type=int,
        default=6,
        help="Sessions newer than this are considered active (default: 6).",
    )
    doctor_parser.add_argument(
        "--reap-after-hours",
        type=int,
        default=72,
        help="Only sessions older than this become reap candidates (default: 72).",
    )
    doctor_parser.add_argument(
        "--recent-debug-logs",
        type=int,
        default=20,
        help="How many recent debug logs to scan for MCP/permission signals.",
    )
    doctor_parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Maximum bridge pointers, servers, and sessions to print (default: 15).",
    )
    doctor_parser.add_argument("--json", action="store_true", help="Output JSON.")
    doctor_parser.set_defaults(func=claude_doctor_command)

    reap_parser = claude_sub.add_parser(
        "reap",
        help="Archive stale/zombie Claude sessions. Dry-run unless --apply is set.",
    )
    reap_parser.add_argument("--claude-root", help="Claude root (default: ~/.claude).")
    reap_parser.add_argument(
        "--archive-root",
        help="Explicit archive root override. Defaults under ~/.claude/archive/.",
    )
    reap_parser.add_argument(
        "--active-hours",
        type=int,
        default=6,
        help="Sessions newer than this are considered active (default: 6).",
    )
    reap_parser.add_argument(
        "--reap-after-hours",
        type=int,
        default=72,
        help="Only sessions older than this become reap candidates (default: 72).",
    )
    reap_parser.add_argument(
        "--recent-debug-logs",
        type=int,
        default=20,
        help="How many recent debug logs to scan while building the candidate list.",
    )
    reap_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum candidate sessions to archive/report (default: 50).",
    )
    reap_parser.add_argument(
        "--apply",
        action="store_true",
        help="Move candidate sessions into the archive root. Default is dry-run.",
    )
    reap_parser.add_argument("--json", action="store_true", help="Output JSON.")
    reap_parser.set_defaults(func=claude_reap_command)
