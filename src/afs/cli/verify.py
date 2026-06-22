"""Verification planning and execution commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..repo_policy import evaluate_repo_policy, load_repo_policy
from ..verification import (
    build_structured_guidance,
    build_verification_plan,
    run_verification_command,
)
from ._utils import load_manager, load_runtime_config_from_args


def _load_payload(path: str | None) -> tuple[dict[str, Any], Path | None]:
    if not path:
        return {}, None
    payload_path = Path(path).expanduser().resolve()
    return json.loads(payload_path.read_text(encoding="utf-8")), payload_path


def _resolve_cwd(args: argparse.Namespace, payload: dict[str, Any]) -> Path:
    explicit = str(getattr(args, "cwd", "") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    payload_cwd = str(payload.get("cwd", "") or "").strip()
    if payload_cwd:
        return Path(payload_cwd).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve_workflow(args: argparse.Namespace, payload: dict[str, Any]) -> str:
    explicit = str(getattr(args, "workflow", "") or "").strip()
    if explicit:
        return explicit
    return str(((payload.get("pack") or {}) if isinstance(payload.get("pack"), dict) else {}).get("workflow", "general") or "general").strip()


def _resolve_tool_profile(args: argparse.Namespace, payload: dict[str, Any]) -> str:
    explicit = str(getattr(args, "tool_profile", "") or "").strip()
    if explicit:
        return explicit
    return str(((payload.get("pack") or {}) if isinstance(payload.get("pack"), dict) else {}).get("tool_profile", "default") or "default").strip()


def _resolve_model(args: argparse.Namespace, payload: dict[str, Any]) -> str:
    explicit = str(getattr(args, "model", "") or "").strip()
    if explicit:
        return explicit
    prompt = payload.get("prompt") if isinstance(payload.get("prompt"), dict) else {}
    return str(prompt.get("model_family", payload.get("client", "generic")) or "generic").strip()


def _resolved_changed_paths(args: argparse.Namespace, payload: dict[str, Any]) -> list[str] | None:
    explicit = list(getattr(args, "changed_path", []) or [])
    if explicit:
        return explicit
    verification_plan = payload.get("verification_plan") if isinstance(payload.get("verification_plan"), dict) else {}
    cached = verification_plan.get("changed_paths")
    if isinstance(cached, list):
        return [str(item) for item in cached if str(item).strip()]
    return None


def _build_plan_bundle(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path | None]:
    payload, payload_path = _load_payload(getattr(args, "payload_file", None))
    cwd = _resolve_cwd(args, payload)
    config, config_path = load_runtime_config_from_args(args, start_dir=cwd)
    workflow = _resolve_workflow(args, payload)
    tool_profile = _resolve_tool_profile(args, payload)
    model = _resolve_model(args, payload)
    matched_skills = (
        ((payload.get("skills") or {}) if isinstance(payload.get("skills"), dict) else {}).get("matches")
        or list(getattr(args, "skill", []) or [])
    )
    changed_paths = _resolved_changed_paths(args, payload)
    policy = load_repo_policy(getattr(args, "repo_policy_file", None), start_dir=cwd)

    bootstrap = payload.get("bootstrap") if isinstance(payload.get("bootstrap"), dict) else {}
    context_path = str(payload.get("context_path") or bootstrap.get("context_path") or "").strip()
    repo_root = cwd
    if context_path:
        repo_root = Path(context_path).expanduser().resolve().parent

    plan = build_verification_plan(
        config=config,
        cwd=cwd,
        workflow=workflow,
        tool_profile=tool_profile,
        matched_skills=matched_skills,
        changed_paths=changed_paths,
        verification_profile=str(getattr(args, "verification_profile", "") or "").strip(),
        policy_summary=None,
    )
    repo_root = Path(str(plan.get("repo_root") or repo_root)).expanduser().resolve()
    policy_summary = evaluate_repo_policy(
        policy,
        repo_root=repo_root,
        changed_paths=list(plan.get("changed_paths") or []),
    )
    plan = build_verification_plan(
        config=config,
        cwd=repo_root,
        workflow=workflow,
        tool_profile=tool_profile,
        matched_skills=matched_skills,
        changed_paths=list(plan.get("changed_paths") or []),
        verification_profile=str(getattr(args, "verification_profile", "") or "").strip(),
        policy_summary=policy_summary,
    )
    structured = build_structured_guidance(
        model=model or "generic",
        workflow=workflow,
        policy_summary=policy_summary,
    )
    return {
        "config_path": str(config_path) if config_path else "",
        "context_path": context_path,
        "client": str(payload.get("client", "")).strip(),
        "session_id": str(payload.get("session_id", "")).strip(),
        "cwd": str(cwd),
    }, {
        "verification_plan": plan,
        "repo_policy": policy_summary,
        "structured_guidance": structured,
    }, payload_path


def verify_plan_command(args: argparse.Namespace) -> int:
    meta, bundle, _payload_path = _build_plan_bundle(args)
    output = dict(bundle)
    output.update(meta)
    if args.json:
        print(json.dumps(output, indent=2))
        return 0

    plan = bundle["verification_plan"]
    policy = bundle["repo_policy"]
    structured = bundle["structured_guidance"]
    print(f"repo_root: {plan['repo_root']}")
    print(f"profile: {plan.get('profile') or '(none)'}")
    changed_paths = list(plan.get("changed_paths") or [])
    print(f"changed_paths: {len(changed_paths)}")
    for path in changed_paths[:12]:
        print(f"  - {path}")
    checks = list(plan.get("selected_checks") or [])
    if checks:
        print("checks:")
        for check in checks:
            name = str(check.get("name", "")).strip()
            matched_by = ", ".join(check.get("matched_by") or []) or "profile"
            print(f"  - {name} [{matched_by}]")
            for command in check.get("commands") or []:
                print(f"    {command}")
    else:
        print("checks: (none)")
    if policy.get("matched_risks") or policy.get("anti_pattern_hits"):
        print("policy:")
        for item in policy.get("matched_risks") or []:
            print(f"  - risk {item['name']}: {', '.join(item.get('paths') or [])}")
        for item in policy.get("anti_pattern_hits") or []:
            print(f"  - anti-pattern {item['name']}: {item['path']}")
    print(f"recommended_schema: {structured['recommended_schema']}")
    print(f"followup_schema: {structured['followup_schema']}")
    return 0


def _filter_selected_checks(bundle: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    checks = list((bundle.get("verification_plan") or {}).get("selected_checks") or [])
    if not names:
        return checks
    selected = {name.strip() for name in names if name.strip()}
    return [check for check in checks if str(check.get("name", "")).strip() in selected]


def _record_result(
    *,
    args: argparse.Namespace,
    meta: dict[str, Any],
    payload_path: Path | None,
    result: dict[str, Any],
) -> None:
    if payload_path is None:
        return
    context_path = str(meta.get("context_path", "")).strip()
    client = str(meta.get("client", "")).strip()
    session_id = str(meta.get("session_id", "")).strip()
    if not context_path or not client or not session_id:
        return

    from .core import _emit_session_event

    config_raw = str(meta.get("config_path", "")).strip()
    config_path = Path(config_raw).expanduser().resolve() if config_raw else None
    manager = load_manager(config_path)
    _emit_session_event(
        manager=manager,
        context_path=Path(context_path).expanduser().resolve(),
        config_path=config_path,
        event_name="verification_recorded",
        client=client,
        session_id=session_id,
        cwd=Path(str(meta.get("cwd", Path.cwd()))).expanduser().resolve(),
        payload_file=payload_path,
        summary=str(result.get("summary", "")),
        verification_status=str(result.get("status", "")),
        verification_command=str(result.get("command", "")),
        seed_payload={
            "verification_check": str(result.get("check_name", "")),
            "verification_digest": result.get("digest") or {},
            "verification_returncode": result.get("returncode"),
        },
    )


def verify_run_command(args: argparse.Namespace) -> int:
    meta, bundle, payload_path = _build_plan_bundle(args)
    checks = _filter_selected_checks(bundle, list(getattr(args, "check", []) or []))
    runnable: list[tuple[str, str]] = []
    for check in checks:
        for command in check.get("commands") or []:
            runnable.append((str(check.get("name", "")).strip(), str(command).strip()))

    if not runnable:
        output = {
            **meta,
            **bundle,
            "results": [],
            "outcome": "blocked" if args.require_checks else "passed",
            "message": "No runnable verification commands matched the current scope.",
        }
        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print(output["message"])
        return 2 if args.require_checks else 0

    results: list[dict[str, Any]] = []
    failed = False
    repo_root = Path(bundle["verification_plan"]["repo_root"]).expanduser().resolve()
    for check_name, command in runnable:
        result = run_verification_command(
            repo_root=repo_root,
            check_name=check_name,
            command=command,
            max_digest_items=args.max_digest_items,
        )
        results.append(result)
        _record_result(args=args, meta=meta, payload_path=payload_path, result=result)
        if result["status"] != "passed":
            failed = True
            if not args.continue_on_fail:
                break

    output = {
        **meta,
        **bundle,
        "results": results,
        "outcome": "failed" if failed else "passed",
    }
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        for result in results:
            print(f"[{result['status']}] {result['command']}")
            print(f"  {result['digest']['summary']}")
        print(f"outcome: {output['outcome']}")
    return 1 if failed else 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    verify_parser = subparsers.add_parser(
        "verify",
        help="Build and run repo-aware verification plans.",
    )
    verify_sub = verify_parser.add_subparsers(dest="verify_command")

    common_arguments = {
        "config": {"help": "Config file path."},
        "cwd": {"help": "Working directory used for repo and git detection."},
        "payload_file": {"help": "Session payload JSON from `afs session prepare-client`."},
        "workflow": {"help": "Workflow override for plan selection."},
        "tool_profile": {"help": "Tool profile override for plan selection."},
        "model": {"help": "Model family override for structured guidance."},
        "verification_profile": {"help": "Verification profile name from afs.toml."},
        "repo_policy_file": {"help": "Explicit `.afs/policy.toml` override."},
    }

    plan_parser = verify_sub.add_parser("plan", help="Compute a verification plan for the current repo state.")
    for name, kwargs in common_arguments.items():
        plan_parser.add_argument(f"--{name.replace('_', '-')}", dest=name, **kwargs)
    plan_parser.add_argument(
        "--changed-path",
        action="append",
        default=[],
        help="Explicit changed path relative to repo root. Repeat to override git auto-detection.",
    )
    plan_parser.add_argument(
        "--skill",
        action="append",
        default=[],
        help="Matched skill name to feed into verification planning. Repeat as needed.",
    )
    plan_parser.add_argument("--json", action="store_true", help="Output JSON.")
    plan_parser.set_defaults(func=verify_plan_command)

    run_parser = verify_sub.add_parser("run", help="Execute the runnable commands from the current verification plan.")
    for name, kwargs in common_arguments.items():
        run_parser.add_argument(f"--{name.replace('_', '-')}", dest=name, **kwargs)
    run_parser.add_argument(
        "--changed-path",
        action="append",
        default=[],
        help="Explicit changed path relative to repo root. Repeat to override git auto-detection.",
    )
    run_parser.add_argument(
        "--skill",
        action="append",
        default=[],
        help="Matched skill name to feed into verification planning. Repeat as needed.",
    )
    run_parser.add_argument(
        "--check",
        action="append",
        default=[],
        help="Run only the named verification check. Repeat as needed.",
    )
    run_parser.add_argument(
        "--max-digest-items",
        type=int,
        default=5,
        help="Maximum digest highlights to keep per command.",
    )
    run_parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Continue running later commands after a failure.",
    )
    run_parser.add_argument(
        "--require-checks",
        action="store_true",
        help="Return a non-zero exit code when no runnable checks match.",
    )
    run_parser.add_argument("--json", action="store_true", help="Output JSON.")
    run_parser.set_defaults(func=verify_run_command)
