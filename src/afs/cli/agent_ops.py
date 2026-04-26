"""Agent operations CLI: manifest, run recorder, and background job queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..agent_jobs import AgentJobQueue, JOB_STATES
from ..agent_manifest import (
    default_manifest_path,
    export_for_harness,
    load_manifest,
    summarize_manifest,
    validate_manifest,
)
from ..agent_runs import AgentRunStore
from ._utils import load_manager, resolve_context_paths


def _resolve_context(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).expanduser().resolve() if getattr(args, "config", None) else None
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(args, manager)
    return context_path


def _add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Config path.")
    parser.add_argument("--path", help="Project path.")
    parser.add_argument("--context-root", help="Context root override.")
    parser.add_argument("--context-dir", help="Context directory name.")


def _read_text_arg(value: str | None, file_value: str | None) -> str:
    if value and file_value:
        raise ValueError("provide only one inline value or file value")
    if file_value:
        return Path(file_value).expanduser().read_text(encoding="utf-8")
    return value or ""


def _print_json_or_text(payload: Any, *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        if isinstance(payload, dict):
            for key, value in payload.items():
                print(f"{key}: {value}")
        else:
            print(payload)
    return 0


def manifest_show_command(args: argparse.Namespace) -> int:
    path = Path(args.file).expanduser() if args.file else default_manifest_path()
    data = load_manifest(path)
    payload = data if args.full else summarize_manifest(data)
    payload = {"path": str(path), **payload}
    return _print_json_or_text(payload, as_json=args.json)


def manifest_validate_command(args: argparse.Namespace) -> int:
    path = Path(args.file).expanduser() if args.file else default_manifest_path()
    data = load_manifest(path)
    issues = validate_manifest(data, check_paths=args.check_paths)
    payload = {
        "path": str(path),
        "ok": not any(issue.level == "error" for issue in issues),
        "issues": [issue.to_dict() for issue in issues],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"path: {payload['path']}")
        print(f"ok: {payload['ok']}")
        for issue in issues:
            print(f"{issue.level}: {issue.message}")
    return 0 if payload["ok"] else 1


def manifest_export_command(args: argparse.Namespace) -> int:
    path = Path(args.file).expanduser() if args.file else default_manifest_path()
    data = load_manifest(path)
    payload = export_for_harness(data, args.harness)
    print(json.dumps(payload, indent=2))
    return 0


def runs_start_command(args: argparse.Namespace) -> int:
    context_path = _resolve_context(args)
    prompt = _read_text_arg(args.prompt, args.prompt_file)
    store = AgentRunStore(context_path)
    run = store.start(
        args.task,
        harness=args.harness or "",
        workspace=args.workspace or str(Path.cwd()),
        prompt=prompt,
    )
    payload = run.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(run.id)
    return 0


def runs_list_command(args: argparse.Namespace) -> int:
    context_path = _resolve_context(args)
    runs = AgentRunStore(context_path).list(status=args.status, limit=args.limit)
    if args.json:
        print(json.dumps([run.to_dict() for run in runs], indent=2))
    else:
        for run in runs:
            print(f"{run.id}\t[{run.status}]\t{run.harness or '-'}\t{run.task}")
    return 0


def runs_show_command(args: argparse.Namespace) -> int:
    context_path = _resolve_context(args)
    run = AgentRunStore(context_path).get(args.run_id)
    if run is None:
        print(f"run not found: {args.run_id}", file=sys.stderr)
        return 1
    print(json.dumps(run.to_dict(), indent=2))
    return 0


def runs_event_command(args: argparse.Namespace) -> int:
    context_path = _resolve_context(args)
    data = json.loads(args.data) if args.data else {}
    run = AgentRunStore(context_path).record_event(
        args.run_id,
        args.event_type,
        summary=args.summary or "",
        data=data,
    )
    print(json.dumps(run.to_dict(), indent=2) if args.json else f"updated: {run.id}")
    return 0


def runs_finish_command(args: argparse.Namespace) -> int:
    context_path = _resolve_context(args)
    verification: list[dict[str, Any]] = []
    for item in args.verify or []:
        command, _, status = item.partition("=")
        verification.append({"command": command, "status": status or "recorded"})
    run = AgentRunStore(context_path).finish(
        args.run_id,
        status=args.status,
        summary=args.summary or "",
        files_changed=args.changed or [],
        commands=args.ran_command or [],
        verification=verification,
        handoff_path=args.handoff or "",
    )
    print(json.dumps(run.to_dict(), indent=2) if args.json else f"finished: {run.id}")
    return 0


def jobs_create_command(args: argparse.Namespace) -> int:
    context_path = _resolve_context(args)
    prompt = _read_text_arg(args.prompt, args.prompt_file) or args.title
    job = AgentJobQueue(context_path).create(
        args.title,
        prompt,
        priority=args.priority,
        created_by=args.created_by or "",
        scope=args.scope or "",
        expected_output=args.expected_output or "",
    )
    print(json.dumps(job.to_dict(), indent=2) if args.json else job.id)
    return 0


def jobs_list_command(args: argparse.Namespace) -> int:
    context_path = _resolve_context(args)
    jobs = AgentJobQueue(context_path).list(status=args.status)
    if args.json:
        print(json.dumps([job.to_dict() for job in jobs], indent=2))
    else:
        for job in jobs:
            assigned = f" -> {job.assigned_to}" if job.assigned_to else ""
            print(f"{job.id}\t[{job.status}]\tp{job.priority}\t{job.title}{assigned}")
    return 0


def jobs_show_command(args: argparse.Namespace) -> int:
    context_path = _resolve_context(args)
    job = AgentJobQueue(context_path).get(args.job_id)
    if job is None:
        print(f"job not found: {args.job_id}", file=sys.stderr)
        return 1
    print(json.dumps(job.to_dict(), indent=2))
    return 0


def jobs_claim_command(args: argparse.Namespace) -> int:
    context_path = _resolve_context(args)
    job = AgentJobQueue(context_path).claim(args.job_id, args.agent)
    print(json.dumps(job.to_dict(), indent=2) if args.json else f"claimed: {job.id}")
    return 0


def jobs_move_command(args: argparse.Namespace) -> int:
    context_path = _resolve_context(args)
    result = _read_text_arg(args.result, args.result_file)
    job = AgentJobQueue(context_path).move(args.job_id, args.status, result=result)
    print(json.dumps(job.to_dict(), indent=2) if args.json else f"moved: {job.id} -> {job.status}")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    manifest = subparsers.add_parser("agent-manifest", help="Inspect the agent harness manifest.")
    manifest_sub = manifest.add_subparsers(dest="agent_manifest_command")

    show = manifest_sub.add_parser("show", help="Show manifest summary.")
    show.add_argument("--file", help="Manifest TOML path.")
    show.add_argument("--full", action="store_true", help="Show full manifest.")
    show.add_argument("--json", action="store_true", help="Output JSON.")
    show.set_defaults(func=manifest_show_command)

    validate = manifest_sub.add_parser("validate", help="Validate manifest structure.")
    validate.add_argument("--file", help="Manifest TOML path.")
    validate.add_argument("--check-paths", action="store_true", help="Warn on missing local paths.")
    validate.add_argument("--json", action="store_true", help="Output JSON.")
    validate.set_defaults(func=manifest_validate_command)

    export = manifest_sub.add_parser("export", help="Export harness-specific manifest data.")
    export.add_argument("harness", help="Harness name, e.g. codex, claude, gemini.")
    export.add_argument("--file", help="Manifest TOML path.")
    export.set_defaults(func=manifest_export_command)

    runs = subparsers.add_parser("agent-runs", help="Record and inspect agent run records.")
    runs_sub = runs.add_subparsers(dest="agent_runs_command")

    start = runs_sub.add_parser("start", help="Start an agent run record.")
    _add_context_args(start)
    start.add_argument("task", help="Task statement.")
    start.add_argument("--harness", help="Harness name.")
    start.add_argument("--workspace", help="Workspace path.")
    start.add_argument("--prompt", help="Prompt text.")
    start.add_argument("--prompt-file", help="Read prompt text from file.")
    start.add_argument("--json", action="store_true", help="Output JSON.")
    start.set_defaults(func=runs_start_command)

    list_runs = runs_sub.add_parser("list", help="List agent run records.")
    _add_context_args(list_runs)
    list_runs.add_argument("--status", help="Filter by status.")
    list_runs.add_argument("--limit", type=int, default=20)
    list_runs.add_argument("--json", action="store_true")
    list_runs.set_defaults(func=runs_list_command)

    show_run = runs_sub.add_parser("show", help="Show one run record.")
    _add_context_args(show_run)
    show_run.add_argument("run_id")
    show_run.set_defaults(func=runs_show_command)

    event = runs_sub.add_parser("event", help="Append an event to a run record.")
    _add_context_args(event)
    event.add_argument("run_id")
    event.add_argument("event_type")
    event.add_argument("--summary")
    event.add_argument("--data", help="JSON object payload.")
    event.add_argument("--json", action="store_true")
    event.set_defaults(func=runs_event_command)

    finish = runs_sub.add_parser("finish", help="Finish a run record.")
    _add_context_args(finish)
    finish.add_argument("run_id")
    finish.add_argument("--status", choices=["done", "failed", "abandoned"], default="done")
    finish.add_argument("--summary")
    finish.add_argument("--changed", action="append")
    finish.add_argument("--command", dest="ran_command", action="append")
    finish.add_argument("--verify", action="append", help="command=status; repeatable.")
    finish.add_argument("--handoff")
    finish.add_argument("--json", action="store_true")
    finish.set_defaults(func=runs_finish_command)

    jobs = subparsers.add_parser("agent-jobs", help="Manage markdown background agent jobs.")
    jobs_sub = jobs.add_subparsers(dest="agent_jobs_command")

    create = jobs_sub.add_parser("create", help="Create a queued markdown job.")
    _add_context_args(create)
    create.add_argument("title")
    create.add_argument("--prompt")
    create.add_argument("--prompt-file")
    create.add_argument("--scope")
    create.add_argument("--expected-output")
    create.add_argument("--created-by")
    create.add_argument("--priority", type=int, default=5)
    create.add_argument("--json", action="store_true")
    create.set_defaults(func=jobs_create_command)

    list_jobs = jobs_sub.add_parser("list", help="List jobs.")
    _add_context_args(list_jobs)
    list_jobs.add_argument("--status", choices=JOB_STATES)
    list_jobs.add_argument("--json", action="store_true")
    list_jobs.set_defaults(func=jobs_list_command)

    show_job = jobs_sub.add_parser("show", help="Show one job.")
    _add_context_args(show_job)
    show_job.add_argument("job_id")
    show_job.set_defaults(func=jobs_show_command)

    claim = jobs_sub.add_parser("claim", help="Move a queued job to running.")
    _add_context_args(claim)
    claim.add_argument("job_id")
    claim.add_argument("--agent", required=True)
    claim.add_argument("--json", action="store_true")
    claim.set_defaults(func=jobs_claim_command)

    move = jobs_sub.add_parser("move", help="Move a job to queue, running, done, or failed.")
    _add_context_args(move)
    move.add_argument("job_id")
    move.add_argument("status", choices=JOB_STATES)
    move.add_argument("--result")
    move.add_argument("--result-file")
    move.add_argument("--json", action="store_true")
    move.set_defaults(func=jobs_move_command)
