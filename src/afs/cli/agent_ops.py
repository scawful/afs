"""Agent operations CLI: manifest, run recorder, and background job queue."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from ..agent_hooks import (
    DEFAULT_WORKER_LABEL,
    default_shell_profile,
    install_shell_profile_hooks,
    install_worker_launchd,
    render_launchd_plist,
    render_shell_profile_block,
    shell_hooks_installed,
    worker_launchd_installed,
)
from ..agent_job_worker import run_agent_job_worker
from ..agent_job_status import build_agent_job_status, format_agent_job_status
from ..agent_jobs import AgentJobQueue, JOB_STATES
from ..agent_manifest import (
    default_manifest_path,
    export_for_harness,
    load_manifest,
    summarize_manifest,
    validate_manifest,
)
from ..agent_manifest_sync import sync_manifest
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


def _default_afs_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def manifest_sync_command(args: argparse.Namespace) -> int:
    path = Path(args.file).expanduser() if args.file else default_manifest_path()
    data = load_manifest(path)
    selected = set(args.harness or []) or None
    actions = sync_manifest(
        data,
        apply=args.apply,
        harnesses=selected,
        sync_skills=not args.no_skills,
        sync_exports=not args.no_exports,
    )
    payload = {
        "path": str(path),
        "applied": bool(args.apply),
        "actions": [action.to_dict() for action in actions],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        mode = "apply" if args.apply else "dry-run"
        print(f"mode: {mode}")
        for action in actions:
            print(
                f"{action.status}\t{action.action}\t{action.harness}\t{action.target}"
            )
    return 1 if any(action.status == "error" for action in actions) else 0


def hooks_show_command(args: argparse.Namespace) -> int:
    afs_root = Path(args.afs_root).expanduser() if args.afs_root else _default_afs_root()
    context_path = Path(args.path).expanduser() if args.path else afs_root
    if args.kind == "launchd":
        payload = render_launchd_plist(
            afs_root=afs_root,
            context_path=context_path,
            agent_name=args.agent,
            command=args.job_command,
            poll_seconds=args.poll_seconds,
            label=args.label,
        )
        print(payload.decode("utf-8"), end="")
    else:
        print(render_shell_profile_block(afs_root), end="")
    return 0


def hooks_install_shell_command(args: argparse.Namespace) -> int:
    afs_root = Path(args.afs_root).expanduser() if args.afs_root else _default_afs_root()
    profile = Path(args.profile).expanduser() if args.profile else default_shell_profile()
    result = install_shell_profile_hooks(
        afs_root=afs_root,
        profile_path=profile,
        apply=args.apply,
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        mode = "applied" if result.applied else "planned"
        print(f"{mode}: {result.target}")
        print(f"changed: {str(result.changed).lower()}")
        print(result.message)
    return 0


def hooks_install_worker_command(args: argparse.Namespace) -> int:
    afs_root = Path(args.afs_root).expanduser() if args.afs_root else _default_afs_root()
    context_path = Path(args.path).expanduser() if args.path else afs_root
    result = install_worker_launchd(
        afs_root=afs_root,
        context_path=context_path,
        agent_name=args.agent,
        command=args.job_command,
        poll_seconds=args.poll_seconds,
        label=args.label,
        apply=args.apply,
        load=args.load,
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        mode = "applied" if result.applied or result.loaded else "planned"
        print(f"{mode}: {result.target}")
        print(f"changed: {str(result.changed).lower()}")
        print(f"loaded: {str(result.loaded).lower()}")
        print(result.message)
    return 0


def hooks_status_command(args: argparse.Namespace) -> int:
    profile = Path(args.profile).expanduser() if args.profile else default_shell_profile()
    payload = {
        "shell_profile": str(profile),
        "shell_hooks_installed": shell_hooks_installed(profile),
        "launchd_label": args.label,
        "worker_launchd_installed": worker_launchd_installed(args.label),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"shell_profile: {payload['shell_profile']}")
        print(f"shell_hooks_installed: {str(payload['shell_hooks_installed']).lower()}")
        print(f"launchd_label: {payload['launchd_label']}")
        print(f"worker_launchd_installed: {str(payload['worker_launchd_installed']).lower()}")
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
        allow_destructive=args.allow_destructive,
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


def jobs_status_command(args: argparse.Namespace) -> int:
    context_path = _resolve_context(args)
    payload = build_agent_job_status(
        context_path,
        label=args.label,
        stale_after_seconds=args.stale_after,
        recent_runs_limit=args.recent_runs,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_agent_job_status(payload))
    if args.strict and not payload["watchdog"]["healthy"]:
        return 1
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


def jobs_work_command(args: argparse.Namespace) -> int:
    context_path = _resolve_context(args)
    command = args.job_command or ""
    if not command:
        command = os.getenv(args.command_env or "AFS_AGENT_JOB_COMMAND", "")
    if not command and not args.dry_run:
        raise ValueError("provide --command, set AFS_AGENT_JOB_COMMAND, or use --dry-run")

    all_results: list[dict[str, object]] = []
    while True:
        results = run_agent_job_worker(
            context_path,
            agent_name=args.agent,
            command=command,
            workspace=Path(args.workspace).expanduser() if args.workspace else Path.cwd(),
            limit=args.limit,
            timeout=args.timeout,
            dry_run=args.dry_run,
            allow_destructive=args.allow_destructive,
        )
        all_results.extend(result.to_dict() for result in results)
        if args.once or args.dry_run or not args.loop:
            break
        if not results or all(result.status == "skipped_destructive" for result in results):
            time.sleep(args.poll_seconds)
            continue

    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        for result in all_results:
            print(
                f"{result['status']}\t{result['job_id']}\t{result['title']}"
            )
        if not all_results:
            print("no queued jobs")
    return 1 if any(result["status"] == "failed" for result in all_results) else 0


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

    sync = manifest_sub.add_parser("sync", help="Copy shared skills and write harness manifest exports.")
    sync.add_argument("--file", help="Manifest TOML path.")
    sync.add_argument("--harness", action="append", help="Limit sync to one harness; repeatable.")
    sync.add_argument("--apply", action="store_true", help="Apply changes. Default is dry-run.")
    sync.add_argument("--no-skills", action="store_true", help="Skip skill directory copies.")
    sync.add_argument("--no-exports", action="store_true", help="Skip per-harness manifest exports.")
    sync.add_argument("--json", action="store_true", help="Output JSON.")
    sync.set_defaults(func=manifest_sync_command)

    hooks = subparsers.add_parser("agent-hooks", help="Install shell and background hooks for agent harnesses.")
    hooks_sub = hooks.add_subparsers(dest="agent_hooks_command")

    hooks_show = hooks_sub.add_parser("show", help="Render hook content without installing.")
    hooks_show.add_argument("--kind", choices=["shell", "launchd"], default="shell")
    hooks_show.add_argument("--afs-root", help="AFS repo root.")
    hooks_show.add_argument("--path", help="Context/project path for the worker.")
    hooks_show.add_argument("--agent", default="local-worker", help="Worker agent name.")
    hooks_show.add_argument("--command", dest="job_command", help="Worker shell command.")
    hooks_show.add_argument("--poll-seconds", type=float, default=30.0)
    hooks_show.add_argument("--label", default=DEFAULT_WORKER_LABEL)
    hooks_show.set_defaults(func=hooks_show_command)

    install_shell = hooks_sub.add_parser("install-shell", help="Install shell profile hooks.")
    install_shell.add_argument("--afs-root", help="AFS repo root.")
    install_shell.add_argument("--profile", help="Shell profile path. Defaults to ~/.zshrc.")
    install_shell.add_argument("--apply", action="store_true", help="Write the profile block.")
    install_shell.add_argument("--json", action="store_true")
    install_shell.set_defaults(func=hooks_install_shell_command)

    install_worker = hooks_sub.add_parser("install-worker", help="Install a launchd agent-jobs worker.")
    install_worker.add_argument("--afs-root", help="AFS repo root.")
    install_worker.add_argument("--path", help="Context/project path for queued jobs.")
    install_worker.add_argument("--agent", default="local-worker", help="Worker agent name.")
    install_worker.add_argument("--command", dest="job_command", help="Worker shell command.")
    install_worker.add_argument("--poll-seconds", type=float, default=30.0)
    install_worker.add_argument("--label", default=DEFAULT_WORKER_LABEL)
    install_worker.add_argument("--apply", action="store_true", help="Write the LaunchAgent plist.")
    install_worker.add_argument("--load", action="store_true", help="Load the LaunchAgent with launchctl.")
    install_worker.add_argument("--json", action="store_true")
    install_worker.set_defaults(func=hooks_install_worker_command)

    hooks_status = hooks_sub.add_parser("status", help="Show installed hook status.")
    hooks_status.add_argument("--profile", help="Shell profile path. Defaults to ~/.zshrc.")
    hooks_status.add_argument("--label", default=DEFAULT_WORKER_LABEL)
    hooks_status.add_argument("--json", action="store_true")
    hooks_status.set_defaults(func=hooks_status_command)

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
    create.add_argument(
        "--allow-destructive",
        action="store_true",
        help="Allow the background worker to run obvious destructive operations for this job.",
    )
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

    status = jobs_sub.add_parser("status", help="Show queue, worker, run, and watchdog status.")
    _add_context_args(status)
    status.add_argument("--label", default=DEFAULT_WORKER_LABEL, help="LaunchAgent label to inspect.")
    status.add_argument(
        "--stale-after",
        type=float,
        default=3600.0,
        help="Seconds before a running job is reported as stale.",
    )
    status.add_argument("--recent-runs", type=int, default=5, help="Recent run records to include.")
    status.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when watchdog checks need attention.",
    )
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=jobs_status_command)

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

    work = jobs_sub.add_parser("work", help="Claim queued jobs and run a local command.")
    _add_context_args(work)
    work.add_argument("--agent", required=True, help="Worker or harness name.")
    work.add_argument("--command", dest="job_command", help="Shell command to run for each job.")
    work.add_argument(
        "--command-env",
        default="AFS_AGENT_JOB_COMMAND",
        help="Environment variable used when --command is omitted.",
    )
    work.add_argument("--workspace", help="Working directory for the command.")
    work.add_argument("--limit", type=int, default=1, help="Maximum jobs to process per pass.")
    work.add_argument("--timeout", type=int, help="Per-job timeout in seconds.")
    work.add_argument(
        "--allow-destructive",
        action="store_true",
        help="Let this worker run jobs that look explicitly destructive.",
    )
    work.add_argument("--dry-run", action="store_true", help="Show queued work without claiming jobs.")
    work.add_argument("--once", action="store_true", help="Run one pass and exit.")
    work.add_argument("--loop", action="store_true", help="Poll for jobs until interrupted.")
    work.add_argument("--poll-seconds", type=float, default=5.0)
    work.add_argument("--json", action="store_true")
    work.set_defaults(func=jobs_work_command)
