"""Verification planning and execution commands."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path
from typing import Any

from ..repo_policy import evaluate_repo_policy, load_repo_policy
from ..verification import (
    build_structured_guidance,
    build_verification_plan,
    redact_verification_plan,
    run_verification_command,
    run_verification_execution,
    verification_item_id,
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


def _resolved_changed_paths(
    args: argparse.Namespace, _payload: dict[str, Any]
) -> list[str] | None:
    explicit = list(getattr(args, "changed_path", []) or [])
    if explicit:
        return explicit
    # Session payload plans are snapshots. Verification must rediscover live Git
    # state after edits unless the caller supplies an explicit override.
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
        discovery_error=str(plan.get("discovery_error", "")),
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


def _public_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Return a CLI-safe bundle without raw env values or designated argv data."""
    public = dict(bundle)
    plan = bundle.get("verification_plan")
    if isinstance(plan, dict):
        public["verification_plan"] = redact_verification_plan(plan)
    return public


def _bind_verification_run(bundle: dict[str, Any]) -> None:
    """Give this invocation fresh item IDs so stale records cannot satisfy it."""
    plan = bundle.get("verification_plan")
    if not isinstance(plan, dict):
        return
    run_id = secrets.token_hex(16)
    plan["verification_run_id"] = run_id
    plan["preflight_item_id"] = f"verification-run-v1:{run_id}:preflight"
    for check in plan.get("selected_checks") or []:
        if not isinstance(check, dict):
            continue
        for field in ("execution_item_ids", "command_item_ids"):
            check[field] = [
                f"verification-run-v1:{run_id}:{item_id}"
                for item_id in check.get(field) or []
            ]
        review_item_id = str(check.get("review_item_id", "")).strip()
        if review_item_id:
            check["review_item_id"] = (
                f"verification-run-v1:{run_id}:{review_item_id}"
            )


def _refresh_session_verification_plan(
    *,
    meta: dict[str, Any],
    bundle: dict[str, Any],
    payload_path: Path | None,
    blocked_reason: str = "",
) -> None:
    if payload_path is None:
        return
    context_path = str(meta.get("context_path", "")).strip()
    client = str(meta.get("client", "")).strip()
    session_id = str(meta.get("session_id", "")).strip()
    if not context_path or not client or not session_id:
        return

    from ..session_harness import (
        _build_session_verification_state,
        write_client_session_payload_artifact,
    )

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    public_bundle = _public_bundle(bundle)
    payload["verification_plan"] = public_bundle["verification_plan"]
    payload["repo_policy"] = public_bundle["repo_policy"]
    payload["structured_guidance"] = public_bundle["structured_guidance"]
    activity = payload.get("activity")
    if not isinstance(activity, dict):
        activity = {}
    payload["activity"] = activity
    if blocked_reason:
        verification = activity.get("verification")
        verification = verification if isinstance(verification, dict) else {}
        records = list(verification.get("records") or [])
        records.append(
            {
                "status": "blocked",
                "check_name": "verification-preflight",
                "item_id": str(
                    public_bundle["verification_plan"].get(
                        "preflight_item_id", ""
                    )
                ),
                "summary": blocked_reason,
            }
        )
        verification["records"] = records
        activity["verification"] = verification
    activity["verification"] = _build_session_verification_state(
        payload,
        final=False,
    )

    config_raw = str(meta.get("config_path", "")).strip()
    config_path = Path(config_raw).expanduser().resolve() if config_raw else None
    manager = load_manager(config_path)
    write_client_session_payload_artifact(
        manager,
        Path(context_path).expanduser().resolve(),
        client=client,
        payload=payload,
        payload_path=payload_path,
    )


def _begin_verification_run(
    *,
    meta: dict[str, Any],
    bundle: dict[str, Any],
    payload_path: Path | None,
    blocked_reason: str = "",
) -> None:
    if blocked_reason:
        plan = bundle.get("verification_plan")
        if isinstance(plan, dict):
            plan["preflight_required"] = True
    _bind_verification_run(bundle)
    _refresh_session_verification_plan(
        meta=meta,
        bundle=bundle,
        payload_path=payload_path,
        blocked_reason=blocked_reason,
    )


def _invalid_verification_input(args: argparse.Namespace, exc: ValueError) -> int:
    message = f"Invalid verification input: {exc}"
    output = {
        "verification_plan": {"available": False},
        "results": [],
        "outcome": "blocked",
        "message": message,
    }
    if bool(getattr(args, "json", False)):
        print(json.dumps(output, indent=2))
    else:
        print(message, file=sys.stderr)
    return 2


def verify_plan_command(args: argparse.Namespace) -> int:
    try:
        meta, bundle, _payload_path = _build_plan_bundle(args)
    except ValueError as exc:
        return _invalid_verification_input(args, exc)
    public_bundle = _public_bundle(bundle)
    output = dict(public_bundle)
    output.update(meta)
    if args.json:
        print(json.dumps(output, indent=2))
        return 0

    plan = public_bundle["verification_plan"]
    policy = public_bundle["repo_policy"]
    structured = public_bundle["structured_guidance"]
    print(f"repo_root: {plan['repo_root']}")
    print(f"profile: {plan.get('profile') or '(none)'}")
    changed_paths = list(plan.get("changed_paths") or [])
    print(f"changed_paths: {len(changed_paths)}")
    if not plan.get("discovery_complete", True):
        print(f"discovery: incomplete: {plan.get('discovery_error', 'unknown error')}")
    for path in changed_paths[:12]:
        print(f"  - {path}")
    checks = list(plan.get("selected_checks") or [])
    if checks:
        print("checks:")
        for check in checks:
            name = str(check.get("name", "")).strip()
            matched_by = ", ".join(check.get("matched_by") or []) or "profile"
            print(f"  - {name} [{matched_by}]")
            for execution in check.get("executions") or []:
                argv = execution.get("argv") if isinstance(execution, dict) else []
                print(f"    argv: {json.dumps(argv or [])}")
            for command in check.get("commands") or []:
                legacy_status = (
                    "explicit opt-in enabled"
                    if plan.get("allow_legacy_shell")
                    else "blocked; explicit opt-in required"
                )
                print(f"    legacy shell (deprecated; {legacy_status}): {command}")
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
    for deprecation in plan.get("deprecations") or []:
        print(
            "deprecation: "
            f"{deprecation.get('message', '')} "
            f"Removal: {deprecation.get('removal_version', '')}."
        )
    return 0


def _filter_selected_checks(bundle: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    checks = list((bundle.get("verification_plan") or {}).get("selected_checks") or [])
    if not names:
        return checks
    normalized_names = [str(name).strip() for name in names]
    if any(not name for name in normalized_names):
        raise ValueError("--check names must be non-empty")
    selected = set(normalized_names)
    available = {str(check.get("name", "")).strip() for check in checks}
    unmatched = sorted(selected - available)
    if unmatched:
        raise ValueError(
            "--check names did not match selected verification checks: "
            + ", ".join(unmatched)
        )
    return [check for check in checks if str(check.get("name", "")).strip() in selected]


def _collect_runnable_items(
    checks: list[dict[str, Any]],
) -> tuple[list[tuple[str, bool, str, str, Any]], list[str]]:
    runnable: list[tuple[str, bool, str, str, Any]] = []
    required_without_commands: list[str] = []
    for check in checks:
        check_name = str(check.get("name", "")).strip()
        required = bool(check.get("required", True))
        executions = list(check.get("executions") or [])
        commands = [
            str(command).strip()
            for command in check.get("commands") or []
            if str(command).strip()
        ]
        if required and not executions and not commands:
            required_without_commands.append(check_name)
        execution_item_ids = list(check.get("execution_item_ids") or [])
        command_item_ids = list(check.get("command_item_ids") or [])
        for item_index, execution in enumerate(executions):
            item_id = (
                str(execution_item_ids[item_index])
                if item_index < len(execution_item_ids)
                else verification_item_id(check_name, "execution", item_index)
            )
            runnable.append(
                (check_name, required, item_id, "structured", execution)
            )
        for item_index, command in enumerate(commands):
            item_id = (
                str(command_item_ids[item_index])
                if item_index < len(command_item_ids)
                else verification_item_id(check_name, "legacy", item_index)
            )
            runnable.append(
                (check_name, required, item_id, "legacy_shell", command)
            )
    return runnable, required_without_commands


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
    check_name = str(result.get("check_name", ""))
    status = str(result.get("status", ""))
    digest_raw = result.get("digest")
    digest: dict[str, Any] = digest_raw if isinstance(digest_raw, dict) else {}
    _emit_session_event(
        manager=manager,
        context_path=Path(context_path).expanduser().resolve(),
        config_path=config_path,
        event_name="verification_recorded",
        client=client,
        session_id=session_id,
        cwd=Path(str(meta.get("cwd", Path.cwd()))).expanduser().resolve(),
        payload_file=payload_path,
        summary=f"{check_name}: {status}",
        verification_status=status,
        verification_command=str(
            result.get("audit_command") or result.get("command", "")
        ),
        seed_payload={
            "verification_check": check_name,
            "verification_item_id": str(result.get("verification_item_id", "")),
            "verification_digest": {
                "kind": str(digest.get("kind", "")),
                "line_count": digest.get("line_count"),
                "truncated": bool(digest.get("truncated", False)),
            },
            "verification_returncode": result.get("returncode"),
            "verification_request_hash": str(result.get("request_hash", "")),
            "verification_resolved_executable": str(result.get("resolved_executable", "")),
            "verification_redacted_argv": list(result.get("redacted_argv") or []),
            "verification_duration_seconds": result.get("duration_seconds"),
            "verification_timed_out": bool(result.get("timed_out", False)),
            "verification_stdout_truncated": bool(result.get("stdout_truncated", False)),
            "verification_stderr_truncated": bool(result.get("stderr_truncated", False)),
        },
    )


def verify_run_command(args: argparse.Namespace) -> int:
    try:
        meta, bundle, payload_path = _build_plan_bundle(args)
    except ValueError as exc:
        return _invalid_verification_input(args, exc)
    try:
        checks = _filter_selected_checks(bundle, list(getattr(args, "check", []) or []))
    except ValueError as exc:
        return _invalid_verification_input(args, exc)
    plan = bundle.get("verification_plan") or {}
    if not plan.get("discovery_complete", True):
        message = (
            "Verification scope discovery was incomplete: "
            f"{plan.get('discovery_error') or 'unknown git discovery error'}."
        )
        try:
            _begin_verification_run(
                meta=meta,
                bundle=bundle,
                payload_path=payload_path,
                blocked_reason=message,
            )
        except (OSError, ValueError) as exc:
            return _invalid_verification_input(args, ValueError(str(exc)))
        output = {
            **meta,
            **_public_bundle(bundle),
            "results": [],
            "outcome": "blocked",
            "message": message,
        }
        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print(message)
        return 2
    runnable, required_without_commands = _collect_runnable_items(checks)

    if required_without_commands:
        names = ", ".join(required_without_commands)
        message = (
            "Required verification checks have no runnable commands or executions: "
            f"{names}."
        )
        try:
            _begin_verification_run(
                meta=meta,
                bundle=bundle,
                payload_path=payload_path,
                blocked_reason=message,
            )
        except (OSError, ValueError) as exc:
            return _invalid_verification_input(args, ValueError(str(exc)))
        output = {
            **meta,
            **_public_bundle(bundle),
            "results": [],
            "outcome": "blocked",
            "message": message,
        }
        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print(output["message"])
        return 2

    if not runnable:
        message = "No runnable verification commands matched the current scope."
        try:
            _begin_verification_run(
                meta=meta,
                bundle=bundle,
                payload_path=payload_path,
                blocked_reason=message if args.require_checks else "",
            )
        except (OSError, ValueError) as exc:
            return _invalid_verification_input(args, ValueError(str(exc)))
        output = {
            **meta,
            **_public_bundle(bundle),
            "results": [],
            "outcome": "blocked" if args.require_checks else "passed",
            "message": message,
        }
        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print(output["message"])
        return 2 if args.require_checks else 0

    _bind_verification_run(bundle)
    runnable, _required_without_commands = _collect_runnable_items(checks)
    try:
        _refresh_session_verification_plan(
            meta=meta,
            bundle=bundle,
            payload_path=payload_path,
        )
    except (OSError, ValueError) as exc:
        return _invalid_verification_input(args, ValueError(str(exc)))

    results: list[dict[str, Any]] = []
    failed = False
    blocked = False
    repo_root = Path(bundle["verification_plan"]["repo_root"]).expanduser().resolve()
    allow_legacy_shell = bool(
        getattr(args, "allow_legacy_shell", False)
        or bundle["verification_plan"].get("allow_legacy_shell", False)
    )
    legacy_warning_emitted = False
    for check_name, required, item_id, execution_kind, execution in runnable:
        if execution_kind == "legacy_shell":
            if not legacy_warning_emitted:
                if allow_legacy_shell:
                    message = (
                        "legacy verification shell commands are deprecated and will be "
                        "removed in AFS 0.4.0."
                    )
                else:
                    message = (
                        "legacy verification shell commands are deprecated and blocked; "
                        "migrate to executions.argv or opt in explicitly."
                    )
                print(f"warning: {message}", file=sys.stderr)
                legacy_warning_emitted = True
            result = run_verification_command(
                repo_root=repo_root,
                check_name=check_name,
                command=str(execution),
                max_digest_items=args.max_digest_items,
                allow_legacy_shell=allow_legacy_shell,
            )
        else:
            result = run_verification_execution(
                repo_root=repo_root,
                check_name=check_name,
                execution=execution,
                max_digest_items=args.max_digest_items,
            )
        result = dict(result)
        result["verification_item_id"] = item_id
        if result.get("status") == "blocked" and not required:
            result["status"] = "skipped"
            result["summary"] = f"{check_name}: optional blocked check skipped"
            print(f"warning: {result['summary']}", file=sys.stderr)
        results.append(result)
        _record_result(args=args, meta=meta, payload_path=payload_path, result=result)
        if result["status"] == "blocked":
            blocked = True
            if not args.continue_on_fail:
                break
        elif result["status"] not in {"passed", "skipped"}:
            failed = True
            if not args.continue_on_fail:
                break

    output = {
        **meta,
        **_public_bundle(bundle),
        "results": results,
        "outcome": "blocked" if blocked else ("failed" if failed else "passed"),
    }
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        for result in results:
            print(f"[{result['status']}] {result['command']}")
            print(f"  {result['digest']['summary']}")
        print(f"outcome: {output['outcome']}")
    if blocked:
        return 2
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

    plan_parser = verify_sub.add_parser(
        "plan", help="Compute a verification plan for the current repo state."
    )
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

    run_parser = verify_sub.add_parser(
        "run", help="Execute the runnable commands from the current verification plan."
    )
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
    run_parser.add_argument(
        "--allow-legacy-shell",
        action="store_true",
        help="Explicitly run deprecated verification command strings through bash -lc.",
    )
    run_parser.add_argument("--json", action="store_true", help="Output JSON.")
    run_parser.set_defaults(func=verify_run_command)
