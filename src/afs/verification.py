"""Repo-aware verification planning and execution helpers."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from .operator_digests import digest_operator_output
from .schema import AFSConfig, VerificationCheckConfig, VerificationProfileConfig

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
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


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
        commands: list[str] = []
        if shutil.which("ruff"):
            targets = " ".join(shlex.quote(path) for path in python_paths) if python_paths else "."
            commands.append(f"ruff check {targets}")
        if shutil.which("pytest") and (repo_root / "tests").exists():
            commands.append("pytest -q")
        if shutil.which("mypy") and _has_any_path(repo_root, ["mypy.ini", "pyproject.toml"]):
            commands.append("mypy .")
        checks.append(
            VerificationCheckConfig(
                name="builtin-python",
                description="Auto-detected Python verification.",
                paths=["**/*.py", "**/*.pyi", "pyproject.toml", "requirements*.txt"],
                commands=commands,
                skills=["python-quality"],
            )
        )

    node_runner = ""
    if shutil.which("pnpm"):
        node_runner = "pnpm exec"
    elif shutil.which("npm"):
        node_runner = "npm exec --"
    elif shutil.which("yarn"):
        node_runner = "yarn"
    if ts_paths or _has_any_path(repo_root, ["package.json", "tsconfig.json", "tsconfig.base.json"]):
        commands = []
        if node_runner:
            if _has_any_path(repo_root, ["tsconfig.json", "tsconfig.base.json"]):
                commands.append(f"{node_runner} tsc --noEmit")
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
                target = " ".join(shlex.quote(path) for path in ts_paths) if ts_paths else "."
                commands.append(f"{node_runner} eslint {target}")
            if _has_any_path(repo_root, ["vitest.config.ts", "vitest.config.js", "jest.config.js"]):
                commands.append(f"{node_runner} vitest run")
        checks.append(
            VerificationCheckConfig(
                name="builtin-typescript",
                description="Auto-detected TypeScript or JavaScript verification.",
                paths=["**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx", "package.json", "tsconfig*.json"],
                commands=commands,
                skills=["typescript-quality"],
            )
        )

    if cpp_paths or _has_any_path(repo_root, ["CMakeLists.txt", "compile_commands.json"]):
        commands = []
        if shutil.which("clang-tidy") and (repo_root / "compile_commands.json").exists() and cpp_paths:
            commands.append(
                "clang-tidy " + " ".join(shlex.quote(path) for path in cpp_paths)
            )
        if shutil.which("ctest") and (
            (repo_root / "CTestTestfile.cmake").exists()
            or (repo_root / "build" / "CTestTestfile.cmake").exists()
        ):
            commands.append("ctest --output-on-failure")
        checks.append(
            VerificationCheckConfig(
                name="builtin-cpp",
                description="Auto-detected C or C++ verification.",
                paths=["**/*.c", "**/*.cc", "**/*.cpp", "**/*.cxx", "**/*.h", "**/*.hh", "**/*.hpp", "**/*.hxx", "CMakeLists.txt", "compile_commands.json"],
                commands=commands,
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
        for path in (changed_paths if changed_paths is not None else discover_changed_paths(resolved_cwd))
        if _normalize_rel_path(path)
    ]
    matched_skill_names = _matched_skill_names(matched_skills)

    selected_profile = verification_profile.strip() or config.verification.default_profile.strip()
    configured_checks: list[VerificationCheckConfig] = []
    if config.verification.enabled and selected_profile:
        configured_checks = _iter_profile_checks(config.verification.profiles, selected_profile)
    if not configured_checks and config.verification.enabled and config.verification.fallback_to_builtin:
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
    for check in selected_checks:
        commands = list(check.get("commands") or [])
        if commands:
            for command in commands:
                expected.append(f"{check['name']}: {command}")
        else:
            expected.append(f"{check['name']}: review changed scope")

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
        "command_count": sum(len(list(check.get("commands") or [])) for check in selected_checks),
        "expected": expected,
        "policy_risk_count": len(list((policy_summary or {}).get("matched_risks") or [])),
        "policy_violation_count": len(list((policy_summary or {}).get("anti_pattern_hits") or [])),
    }


def run_verification_command(
    *,
    repo_root: str | Path,
    check_name: str,
    command: str,
    max_digest_items: int = 5,
) -> dict[str, Any]:
    """Execute a verification command and return a compact result payload."""
    resolved_root = Path(repo_root).expanduser().resolve()
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=resolved_root,
        capture_output=True,
        text=True,
        check=False,
    )
    combined_output = (result.stdout or "") + (result.stderr or "")
    digest = digest_operator_output(combined_output, kind="auto", max_items=max_digest_items)
    status = "passed" if result.returncode == 0 else "failed"
    summary = f"{check_name}: {digest.get('summary', status)}"
    return {
        "check_name": check_name,
        "command": command,
        "status": status,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "digest": digest,
        "summary": summary,
    }
