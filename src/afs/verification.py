"""Repo-aware verification planning and execution helpers."""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from .execution import (
    ArgvCommand,
    ExecutionPolicy,
    ExecutionRequest,
    LegacyShellCommand,
    execute_checked,
)
from .operator_digests import digest_operator_output
from .schema import (
    AFSConfig,
    VerificationCheckConfig,
    VerificationExecutionConfig,
    VerificationProfileConfig,
)

_PYTHON_SUFFIXES = {".py", ".pyi"}
_TS_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".mts", ".cts"}
_CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}


def _normalize_rel_path(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _pattern_variants(pattern: str) -> list[str]:
    normalized = _normalize_rel_path(pattern)
    if not normalized:
        return []
    variants = {normalized}
    pending = [normalized]
    while pending:
        current = pending.pop()
        next_variants: list[str] = []
        if "/**/" in current:
            next_variants.append(current.replace("/**/", "/", 1))
        if current.startswith("**/"):
            next_variants.append(current[3:])
        for candidate in next_variants:
            if candidate and candidate not in variants:
                variants.add(candidate)
                pending.append(candidate)
    return list(variants)


def _path_matches(path: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    posix_path = PurePosixPath(_normalize_rel_path(path))
    for raw_pattern in patterns:
        for pattern in _pattern_variants(raw_pattern):
            if posix_path.match(pattern) or posix_path.match(f"**/{pattern}"):
                return True
    return False


def _run_git(repo_root: Path, *args: str) -> str | None:
    git = shutil.which("git")
    if not git:
        return None
    try:
        request = ExecutionRequest(
            command=ArgvCommand((git, "-C", str(repo_root), *args)),
            caller="afs.verify",
            purpose="discover repository verification scope",
            cwd=repo_root,
            timeout_seconds=30,
        )
        policy = ExecutionPolicy(
            allowed_cwd_roots=(repo_root,),
            allowed_executables=frozenset({git}),
        )
        record = execute_checked(request, policy)
    except (TypeError, ValueError):
        return None
    if record.outcome != "completed":
        return None
    return record.stdout.strip()


def discover_git_repo_root(start_dir: str | Path | None = None) -> Path | None:
    """Return the git root containing *start_dir*, if any."""
    current = Path(start_dir or Path.cwd()).expanduser().resolve()
    output = _run_git(current, "rev-parse", "--show-toplevel")
    if not output:
        return None
    return Path(output).expanduser().resolve()


def discover_changed_paths(start_dir: str | Path | None = None) -> list[str]:
    """Return changed and untracked paths relative to the current git repo root."""
    repo_root = discover_git_repo_root(start_dir)
    if repo_root is None:
        return []
    output = _run_git(repo_root, "status", "--porcelain", "--untracked-files=all")
    if not output:
        return []
    changed_paths: list[str] = []
    seen: set[str] = set()
    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        if len(line) < 4:
            continue
        path_text = line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        normalized = _normalize_rel_path(path_text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        changed_paths.append(normalized)
    return changed_paths


def _iter_profile_checks(
    profiles: dict[str, VerificationProfileConfig],
    profile_name: str,
    *,
    seen_profiles: set[str] | None = None,
) -> list[VerificationCheckConfig]:
    current_seen = set(seen_profiles or set())
    normalized_name = str(profile_name or "").strip()
    if not normalized_name or normalized_name in current_seen:
        return []
    profile = profiles.get(normalized_name)
    if profile is None:
        return []
    current_seen.add(normalized_name)
    checks: list[VerificationCheckConfig] = []
    for included in profile.include_profiles:
        checks.extend(_iter_profile_checks(profiles, included, seen_profiles=current_seen))
    checks.extend(profile.checks)
    return checks


def _has_any_path(repo_root: Path, candidates: list[str]) -> bool:
    return any((repo_root / candidate).exists() for candidate in candidates)


def _build_builtin_checks(
    repo_root: Path,
    changed_paths: list[str],
) -> list[VerificationCheckConfig]:
    checks: list[VerificationCheckConfig] = []
    python_paths = [path for path in changed_paths if Path(path).suffix in _PYTHON_SUFFIXES]
    ts_paths = [path for path in changed_paths if Path(path).suffix in _TS_SUFFIXES]
    cpp_paths = [path for path in changed_paths if Path(path).suffix.lower() in _CPP_SUFFIXES]

    if python_paths or _has_any_path(repo_root, ["pyproject.toml", "requirements.txt"]):
        executions: list[VerificationExecutionConfig] = []
        if shutil.which("ruff"):
            targets = python_paths if python_paths else ["."]
            executions.append(VerificationExecutionConfig(argv=["ruff", "check", *targets]))
        if shutil.which("pytest") and (repo_root / "tests").exists():
            executions.append(VerificationExecutionConfig(argv=["pytest", "-q"]))
        if shutil.which("mypy") and _has_any_path(repo_root, ["mypy.ini", "pyproject.toml"]):
            executions.append(VerificationExecutionConfig(argv=["mypy", "."]))
        checks.append(
            VerificationCheckConfig(
                name="builtin-python",
                description="Auto-detected Python verification.",
                paths=["**/*.py", "**/*.pyi", "pyproject.toml", "requirements*.txt"],
                executions=executions,
                skills=["python-quality"],
            )
        )

    node_runner: list[str] = []
    if shutil.which("pnpm"):
        node_runner = ["pnpm", "exec"]
    elif shutil.which("npm"):
        node_runner = ["npm", "exec", "--"]
    elif shutil.which("yarn"):
        node_runner = ["yarn"]
    if ts_paths or _has_any_path(
        repo_root, ["package.json", "tsconfig.json", "tsconfig.base.json"]
    ):
        executions = []
        if node_runner:
            if _has_any_path(repo_root, ["tsconfig.json", "tsconfig.base.json"]):
                executions.append(
                    VerificationExecutionConfig(argv=[*node_runner, "tsc", "--noEmit"])
                )
            if _has_any_path(
                repo_root,
                [
                    ".eslintrc",
                    ".eslintrc.js",
                    ".eslintrc.cjs",
                    "eslint.config.js",
                    "eslint.config.mjs",
                ],
            ):
                targets = ts_paths if ts_paths else ["."]
                executions.append(
                    VerificationExecutionConfig(argv=[*node_runner, "eslint", *targets])
                )
            if _has_any_path(repo_root, ["vitest.config.ts", "vitest.config.js", "jest.config.js"]):
                executions.append(VerificationExecutionConfig(argv=[*node_runner, "vitest", "run"]))
        checks.append(
            VerificationCheckConfig(
                name="builtin-typescript",
                description="Auto-detected TypeScript or JavaScript verification.",
                paths=[
                    "**/*.ts",
                    "**/*.tsx",
                    "**/*.js",
                    "**/*.jsx",
                    "package.json",
                    "tsconfig*.json",
                ],
                executions=executions,
                skills=["typescript-quality"],
            )
        )

    if cpp_paths or _has_any_path(repo_root, ["CMakeLists.txt", "compile_commands.json"]):
        executions = []
        if (
            shutil.which("clang-tidy")
            and (repo_root / "compile_commands.json").exists()
            and cpp_paths
        ):
            executions.append(VerificationExecutionConfig(argv=["clang-tidy", *cpp_paths]))
        if shutil.which("ctest") and (
            (repo_root / "CTestTestfile.cmake").exists()
            or (repo_root / "build" / "CTestTestfile.cmake").exists()
        ):
            executions.append(VerificationExecutionConfig(argv=["ctest", "--output-on-failure"]))
        checks.append(
            VerificationCheckConfig(
                name="builtin-cpp",
                description="Auto-detected C or C++ verification.",
                paths=[
                    "**/*.c",
                    "**/*.cc",
                    "**/*.cpp",
                    "**/*.cxx",
                    "**/*.h",
                    "**/*.hh",
                    "**/*.hpp",
                    "**/*.hxx",
                    "CMakeLists.txt",
                    "compile_commands.json",
                ],
                executions=executions,
                skills=["cpp-quality"],
            )
        )
    return checks


def _matched_skill_names(matches: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(matches, list):
        return names
    for entry in matches:
        if isinstance(entry, dict):
            name = str(entry.get("name", "")).strip()
        else:
            name = str(entry).strip()
        if name:
            names.append(name)
    return names


def _collect_selected_checks(
    checks: list[VerificationCheckConfig],
    *,
    changed_paths: list[str],
    workflow: str,
    tool_profile: str,
    matched_skills: list[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for check in checks:
        if not check.name:
            continue
        matched_by: list[str] = []
        matched_paths: list[str] = []
        if check.paths:
            matched_paths = [path for path in changed_paths if _path_matches(path, check.paths)]
            if matched_paths:
                matched_by.append("paths")
        if check.skills and set(check.skills).intersection(matched_skills):
            matched_by.append("skills")
        if check.workflows and workflow in check.workflows:
            matched_by.append("workflow")
        if check.tool_profiles and tool_profile in check.tool_profiles:
            matched_by.append("tool_profile")
        if not any([check.paths, check.skills, check.workflows, check.tool_profiles]):
            matched_by.append("profile")

        if not matched_by:
            continue
        selected.append(
            {
                "name": check.name,
                "description": check.description,
                "required": check.required,
                "executions": [execution.to_dict() for execution in check.executions],
                "commands": list(check.commands),
                "paths": list(check.paths),
                "matched_by": matched_by,
                "matched_paths": matched_paths,
                "skills": list(check.skills),
            }
        )
    return selected


def _infer_languages(changed_paths: list[str]) -> list[str]:
    languages: list[str] = []
    if any(Path(path).suffix in _PYTHON_SUFFIXES for path in changed_paths):
        languages.append("python")
    if any(Path(path).suffix in _TS_SUFFIXES for path in changed_paths):
        languages.append("typescript")
    if any(Path(path).suffix.lower() in _CPP_SUFFIXES for path in changed_paths):
        languages.append("cpp")
    return languages


def recommended_structured_schema(workflow: str, *, policy_summary: dict[str, Any] | None = None) -> str:
    matched_risks = list((policy_summary or {}).get("matched_risks") or [])
    design_constraints = list((policy_summary or {}).get("design_constraints") or [])
    if matched_risks or design_constraints:
        return "design-brief"
    if workflow == "scan_fast":
        return "file-shortlist"
    if workflow == "edit_fast":
        return "edit-intent"
    if workflow == "review_deep":
        return "review-findings"
    if workflow == "root_cause_deep":
        return "verification-summary"
    return "implementation-plan"


def build_structured_guidance(
    *,
    model: str,
    workflow: str,
    policy_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recommended = recommended_structured_schema(workflow, policy_summary=policy_summary)
    followup = "verification-summary" if workflow in {"edit_fast", "root_cause_deep"} else "handoff-summary"
    repair_loop = [
        "Run one verification command at a time and keep the failing surface narrow.",
        "Compress noisy test or compiler output with `operator.digest` before retrying.",
        "Retry once with a narrower query, smaller file set, or structured schema before escalating scope.",
    ]
    if model == "gemini":
        repair_loop.append(
            "Stay retrieval-first on Flash, then escalate only after the evidence and failing output are compressed."
        )
    return {
        "recommended_schema": recommended,
        "followup_schema": followup,
        "repair_loop": repair_loop,
    }


def build_verification_plan(
    *,
    config: AFSConfig,
    cwd: str | Path,
    workflow: str,
    tool_profile: str,
    matched_skills: Any = None,
    changed_paths: list[str] | None = None,
    verification_profile: str = "",
    policy_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a repo-aware verification plan from config, git state, and skill matches."""
    resolved_cwd = Path(cwd).expanduser().resolve()
    repo_root = discover_git_repo_root(resolved_cwd) or resolved_cwd
    normalized_changed_paths = [
        _normalize_rel_path(path)
        for path in (
            changed_paths if changed_paths is not None else discover_changed_paths(resolved_cwd)
        )
        if _normalize_rel_path(path)
    ]
    matched_skill_names = _matched_skill_names(matched_skills)

    selected_profile = verification_profile.strip() or config.verification.default_profile.strip()
    configured_checks: list[VerificationCheckConfig] = []
    if config.verification.enabled and selected_profile:
        configured_checks = _iter_profile_checks(config.verification.profiles, selected_profile)
    if (
        not configured_checks
        and config.verification.enabled
        and config.verification.fallback_to_builtin
    ):
        configured_checks = _build_builtin_checks(repo_root, normalized_changed_paths)
        if configured_checks and not selected_profile:
            selected_profile = "builtin"

    selected_checks = _collect_selected_checks(
        configured_checks,
        changed_paths=normalized_changed_paths,
        workflow=workflow,
        tool_profile=tool_profile,
        matched_skills=matched_skill_names,
    )

    expected: list[str] = []
    legacy_command_count = 0
    structured_execution_count = 0
    for check in selected_checks:
        executions = list(check.get("executions") or [])
        commands = list(check.get("commands") or [])
        structured_execution_count += len(executions)
        legacy_command_count += len(commands)
        for execution in executions:
            argv = execution.get("argv") if isinstance(execution, dict) else []
            expected.append(f"{check['name']}: {shlex.join([str(arg) for arg in argv or []])}")
        for command in commands:
            expected.append(f"{check['name']}: {command} (legacy shell; opt-in required)")
        if not executions and not commands:
            expected.append(f"{check['name']}: review changed scope")

    deprecations: list[dict[str, Any]] = []
    if legacy_command_count:
        deprecations.append(
            {
                "kind": "legacy_shell",
                "count": legacy_command_count,
                "removal_version": "0.4.0",
                "opt_in_required": True,
                "message": (
                    "Legacy verification commands are deprecated and require "
                    "allow_legacy_shell or --allow-legacy-shell."
                ),
            }
        )

    return {
        "available": True,
        "repo_root": str(repo_root),
        "profile": selected_profile,
        "workflow": workflow,
        "tool_profile": tool_profile,
        "changed_paths": normalized_changed_paths,
        "inferred_languages": _infer_languages(normalized_changed_paths),
        "selected_checks": selected_checks,
        "required": any(bool(check.get("required")) for check in selected_checks),
        "allow_legacy_shell": config.verification.allow_legacy_shell,
        "command_count": structured_execution_count + legacy_command_count,
        "structured_execution_count": structured_execution_count,
        "legacy_command_count": legacy_command_count,
        "deprecations": deprecations,
        "expected": expected,
        "policy_risk_count": len(list((policy_summary or {}).get("matched_risks") or [])),
        "policy_violation_count": len(list((policy_summary or {}).get("anti_pattern_hits") or [])),
    }


def _blocked_verification_result(
    *,
    check_name: str,
    command: str,
    execution_kind: str,
    reason: str,
) -> dict[str, Any]:
    summary = f"{check_name}: {reason}"
    return {
        "check_name": check_name,
        "command": command,
        "execution_kind": execution_kind,
        "status": "blocked",
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "digest": {
            "kind": "generic",
            "summary": reason,
            "highlights": [reason],
            "truncated": False,
        },
        "summary": summary,
        "blocked_reason": reason,
        "audit_command": (
            "<legacy shell blocked>" if execution_kind == "legacy_shell" else command
        ),
        "redacted_argv": [],
    }


def _result_from_execution_record(
    *,
    check_name: str,
    command: str,
    execution_kind: str,
    record: Any,
    max_digest_items: int,
) -> dict[str, Any]:
    combined_output = (record.stdout or "") + (record.stderr or "")
    digest = digest_operator_output(combined_output, kind="auto", max_items=max_digest_items)
    if record.outcome == "completed":
        status = "passed"
    elif record.outcome == "blocked":
        status = "blocked"
    else:
        status = "failed"
    summary = f"{check_name}: {digest.get('summary', status)}"
    if record.reasons and not combined_output.strip():
        reason_summary = "; ".join(record.reasons)
        digest = dict(digest)
        digest["summary"] = reason_summary
        digest["highlights"] = list(record.reasons)
        summary = f"{check_name}: {reason_summary}"
    return {
        "check_name": check_name,
        "command": command,
        "execution_kind": execution_kind,
        "status": status,
        "outcome": record.outcome,
        "returncode": record.returncode,
        "stdout": record.stdout,
        "stderr": record.stderr,
        "digest": digest,
        "summary": summary,
        "request_hash": str(record.request_sha256),
        "resolved_executable": str(record.resolved_executable or ""),
        "duration_seconds": record.duration_seconds,
        "timed_out": record.outcome == "timed_out",
        "stdout_truncated": record.stdout_truncated,
        "stderr_truncated": record.stderr_truncated,
        "reason_codes": list(record.reason_codes),
        "reasons": list(record.reasons),
        "audit_command": shlex.join(record.redacted_argv),
        "redacted_argv": list(record.redacted_argv),
    }


def run_verification_execution(
    *,
    repo_root: str | Path,
    check_name: str,
    execution: VerificationExecutionConfig | dict[str, Any],
    max_digest_items: int = 5,
) -> dict[str, Any]:
    """Execute a structured verification argv through the checked broker."""
    resolved_root = Path(repo_root).expanduser().resolve()
    config = (
        execution
        if isinstance(execution, VerificationExecutionConfig)
        else VerificationExecutionConfig.from_dict(execution)
    )
    command = shlex.join(config.argv)
    if not config.argv:
        return _blocked_verification_result(
            check_name=check_name,
            command=command,
            execution_kind="structured",
            reason="structured verification execution requires a non-empty argv",
        )
    requested_cwd = Path(config.cwd).expanduser() if config.cwd else resolved_root
    if not requested_cwd.is_absolute():
        requested_cwd = resolved_root / requested_cwd
    allowed_env = frozenset([*config.inherit_env, *config.env.keys()])
    try:
        request = ExecutionRequest(
            command=ArgvCommand(tuple(config.argv)),
            caller="afs.verify",
            purpose=f"verification check {check_name}",
            cwd=requested_cwd,
            inherit_env=tuple(config.inherit_env),
            set_env=config.env,
            timeout_seconds=config.timeout_seconds,
            max_output_bytes=config.max_output_bytes,
            redact_argv_indices=tuple(config.redact_argv_indices),
        )
        policy = ExecutionPolicy(
            allowed_cwd_roots=(resolved_root,),
            allowed_env=allowed_env,
            allowed_executables=frozenset({config.argv[0]}),
        )
        record = execute_checked(request, policy)
    except (TypeError, ValueError) as exc:
        return _blocked_verification_result(
            check_name=check_name,
            command=command,
            execution_kind="structured",
            reason=f"invalid structured verification execution: {exc}",
        )
    return _result_from_execution_record(
        check_name=check_name,
        command=command,
        execution_kind="structured",
        record=record,
        max_digest_items=max_digest_items,
    )


def run_verification_command(
    *,
    repo_root: str | Path,
    check_name: str,
    command: str,
    max_digest_items: int = 5,
    allow_legacy_shell: bool = False,
) -> dict[str, Any]:
    """Execute a deprecated shell verification command through the checked broker."""
    resolved_root = Path(repo_root).expanduser().resolve()
    try:
        request = ExecutionRequest(
            command=LegacyShellCommand(command),
            caller="afs.verify",
            purpose=f"legacy verification check {check_name}",
            cwd=resolved_root,
        )
        policy = ExecutionPolicy(
            allowed_cwd_roots=(resolved_root,),
            allowed_executables=frozenset({shutil.which("bash") or "bash"}),
            allow_legacy_shell=allow_legacy_shell,
        )
        record = execute_checked(request, policy)
    except (TypeError, ValueError) as exc:
        return _blocked_verification_result(
            check_name=check_name,
            command=command,
            execution_kind="legacy_shell",
            reason=f"invalid legacy verification command: {exc}",
        )
    return _result_from_execution_record(
        check_name=check_name,
        command=command,
        execution_kind="legacy_shell",
        record=record,
        max_digest_items=max_digest_items,
    )
