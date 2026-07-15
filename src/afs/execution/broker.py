"""Portable fail-closed process backend for typed execution requests."""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from ..protocols.canonical_json import canonical_json_bytes
from .models import (
    DEFAULT_INHERITED_ENV,
    ArgvCommand,
    ExecutionInspection,
    ExecutionPolicy,
    ExecutionRecord,
    ExecutionRequest,
    LegacyShellCommand,
)

_REDACTED = "<redacted>"
_TERMINATION_GRACE_SECONDS = 2.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _path_within(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def _effective_environment(
    request: ExecutionRequest,
    policy: ExecutionPolicy,
    source: Mapping[str, str],
) -> tuple[dict[str, str], list[str], list[str]]:
    reason_codes: list[str] = []
    reasons: list[str] = []
    denied_inherited = {
        name
        for name in request.inherit_env
        if name not in DEFAULT_INHERITED_ENV and name not in policy.allowed_env
    }
    # Baseline keys may be inherited automatically, but an untrusted request may
    # not override even PATH/HOME unless trusted policy explicitly permits it.
    denied_explicit = set(request.set_env) - set(policy.allowed_env)
    denied = sorted(denied_inherited | denied_explicit)
    if denied:
        reason_codes.append("environment_not_allowed")
        reasons.append("policy does not allow environment keys: " + ", ".join(denied))

    environment: dict[str, str] = {}
    for name in sorted(DEFAULT_INHERITED_ENV | set(request.inherit_env)):
        if name in source:
            environment[name] = str(source[name])
    environment.update(dict(request.set_env))
    return environment, reason_codes, reasons


def _resolve_cwd(request: ExecutionRequest) -> tuple[Path, str, str]:
    candidate = Path(request.cwd)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return candidate.resolve(), "cwd_not_found", "working directory does not exist"
    if not resolved.is_dir():
        return resolved, "cwd_not_directory", "working directory is not a directory"
    return resolved, "", ""


def _resolve_executable(raw: str, cwd: Path, environment: Mapping[str, str]) -> str:
    expanded = Path(raw).expanduser()
    path_like = expanded.is_absolute() or expanded.parent != Path(".")
    if path_like:
        candidate = expanded if expanded.is_absolute() else cwd / expanded
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            return ""
        if not resolved.is_file():
            return ""
        if os.name != "nt" and not os.access(resolved, os.X_OK):
            return ""
        return str(resolved)

    search_path = environment.get("PATH") or os.defpath
    found = shutil.which(raw, path=search_path)
    if not found:
        return ""
    try:
        return str(Path(found).resolve(strict=True))
    except (OSError, RuntimeError):
        return ""


def _executable_allowed(
    raw: str,
    resolved: str,
    cwd: Path,
    policy: ExecutionPolicy,
) -> bool:
    for entry in policy.allowed_executables:
        if entry == raw or entry == resolved:
            return True
        candidate = Path(entry).expanduser()
        if candidate.is_absolute() or candidate.parent != Path("."):
            candidate = candidate if candidate.is_absolute() else cwd / candidate
            try:
                if str(candidate.resolve(strict=True)) == resolved:
                    return True
            except (OSError, RuntimeError):
                continue
    return False


def _spawn_argv(request: ExecutionRequest, resolved_executable: str) -> tuple[str, ...]:
    if isinstance(request.command, ArgvCommand):
        return (resolved_executable, *request.command.argv[1:])
    return (resolved_executable, "-lc", request.command.command)


def _redact_argv(argv: tuple[str, ...], indices: tuple[int, ...]) -> tuple[str, ...]:
    redacted = list(argv)
    for index in indices:
        if index < len(redacted):
            redacted[index] = _REDACTED
    return tuple(redacted)


def _request_hash_payload(
    request: ExecutionRequest,
    *,
    resolved_executable: str,
    resolved_cwd: str,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": request.schema_version,
        "resolved_executable": resolved_executable,
        "command": request.command.to_dict(),
        "resolved_cwd": resolved_cwd,
        "caller": request.caller,
        "purpose": request.purpose,
        "environment": dict(environment),
        "inherit_env": list(request.inherit_env),
        "set_env_keys": sorted(request.set_env),
        "timeout_seconds": request.timeout_seconds,
        "max_output_bytes": request.max_output_bytes,
        "isolation": request.isolation,
        "network": request.network,
        "redact_argv_indices": list(request.redact_argv_indices),
    }


def inspect_execution(
    request: ExecutionRequest,
    policy: ExecutionPolicy,
    *,
    environ: Mapping[str, str] | None = None,
) -> ExecutionInspection:
    """Resolve and inspect a request without launching any process."""
    source_env = os.environ if environ is None else environ
    reason_codes: list[str] = []
    reasons: list[str] = []

    cwd, cwd_code, cwd_reason = _resolve_cwd(request)
    if cwd_code:
        reason_codes.append(cwd_code)
        reasons.append(cwd_reason)
    if not policy.allowed_cwd_roots:
        reason_codes.append("cwd_policy_empty")
        reasons.append("policy has no allowed working-directory roots")
    elif not _path_within(cwd, policy.allowed_cwd_roots):
        reason_codes.append("cwd_outside_allowed_roots")
        reasons.append("resolved working directory is outside policy roots")

    environment, env_codes, env_reasons = _effective_environment(
        request, policy, source_env
    )
    reason_codes.extend(env_codes)
    reasons.extend(env_reasons)

    if isinstance(request.command, LegacyShellCommand):
        raw_executable = "bash"
        if not policy.allow_legacy_shell:
            reason_codes.append("legacy_shell_not_allowed")
            reasons.append("policy does not allow deprecated legacy shell commands")
    else:
        raw_executable = request.command.argv[0]

    resolved_executable = _resolve_executable(raw_executable, cwd, environment)
    if not resolved_executable:
        reason_codes.append("executable_not_found")
        reasons.append(f"executable could not be resolved: {raw_executable}")
    elif not _executable_allowed(raw_executable, resolved_executable, cwd, policy):
        reason_codes.append("executable_not_allowed")
        reasons.append("resolved executable is not allowed by policy")

    if request.isolation != "process":
        reason_codes.append("unsupported_isolation")
        reasons.append(
            f"portable backend does not support isolation={request.isolation!r}"
        )
    if request.network != "inherit":
        reason_codes.append("unsupported_network")
        reasons.append(
            f"portable backend does not support network={request.network!r}"
        )

    spawn_argv = _spawn_argv(request, resolved_executable or raw_executable)
    invalid_indices = [
        index for index in request.redact_argv_indices if index >= len(spawn_argv)
    ]
    if invalid_indices:
        reason_codes.append("invalid_redaction_index")
        reasons.append(
            "redaction indices exceed the resolved argv: "
            + ", ".join(str(index) for index in invalid_indices)
        )

    resolved_cwd = str(cwd)
    env_sha256 = _sha256(environment)
    request_sha256 = _sha256(
        _request_hash_payload(
            request,
            resolved_executable=resolved_executable,
            resolved_cwd=resolved_cwd,
            environment=environment,
        )
    )
    return ExecutionInspection(
        allowed=not reason_codes,
        request_sha256=request_sha256,
        resolved_executable=resolved_executable,
        resolved_cwd=resolved_cwd,
        redacted_argv=_redact_argv(spawn_argv, request.redact_argv_indices),
        environment_keys=tuple(sorted(environment)),
        environment_sha256=env_sha256,
        timeout_seconds=request.timeout_seconds,
        max_output_bytes=request.max_output_bytes,
        isolation=request.isolation,
        network=request.network,
        reason_codes=tuple(reason_codes),
        reasons=tuple(reasons),
        _environment=environment,
    )


def _record_from_inspection(
    request: ExecutionRequest,
    inspection: ExecutionInspection,
    *,
    outcome: str,
    started_at: str,
    finished_at: str,
    duration_seconds: float,
    returncode: int | None,
    stdout: str = "",
    stderr: str = "",
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
    reason_codes: tuple[str, ...] | None = None,
    reasons: tuple[str, ...] | None = None,
) -> ExecutionRecord:
    return ExecutionRecord(
        outcome=outcome,
        request_sha256=inspection.request_sha256,
        resolved_executable=inspection.resolved_executable,
        resolved_cwd=inspection.resolved_cwd,
        redacted_argv=inspection.redacted_argv,
        environment_keys=inspection.environment_keys,
        environment_sha256=inspection.environment_sha256,
        timeout_seconds=request.timeout_seconds,
        max_output_bytes=request.max_output_bytes,
        isolation=request.isolation,
        network=request.network,
        caller=request.caller,
        purpose=request.purpose,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=max(0.0, duration_seconds),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        reason_codes=inspection.reason_codes if reason_codes is None else reason_codes,
        reasons=inspection.reasons if reasons is None else reasons,
    )


def _drain_stream(
    stream: BinaryIO,
    limit: int,
    output: bytearray,
    truncated: list[bool],
) -> None:
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            remaining = limit - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated[0] = True
    finally:
        stream.close()


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
            process.wait(timeout=_TERMINATION_GRACE_SECONDS)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=_TERMINATION_GRACE_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=_TERMINATION_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        process.kill()


def execute_checked(
    request: ExecutionRequest,
    policy: ExecutionPolicy,
    *,
    environ: Mapping[str, str] | None = None,
) -> ExecutionRecord:
    """Re-inspect immediately before spawning and return one bounded record."""
    started_at = _utc_now()
    start_clock = time.monotonic()
    inspection = inspect_execution(request, policy, environ=environ)
    if not inspection.allowed:
        finished_at = _utc_now()
        return _record_from_inspection(
            request,
            inspection,
            outcome="blocked",
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.monotonic() - start_clock,
            returncode=None,
        )

    spawn_argv = _spawn_argv(request, inspection.resolved_executable)
    popen_kwargs: dict[str, Any] = {
        "cwd": inspection.resolved_cwd,
        "env": dict(inspection._environment),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "shell": False,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        popen_kwargs["start_new_session"] = True

    try:
        process = subprocess.Popen(spawn_argv, **popen_kwargs)
    except OSError as exc:
        finished_at = _utc_now()
        return _record_from_inspection(
            request,
            inspection,
            outcome="spawn_error",
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.monotonic() - start_clock,
            returncode=None,
            reason_codes=("spawn_error",),
            reasons=(f"process spawn failed: {exc}",),
        )

    stdout_bytes = bytearray()
    stderr_bytes = bytearray()
    stdout_truncated = [False]
    stderr_truncated = [False]
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_thread = threading.Thread(
        target=_drain_stream,
        args=(process.stdout, request.max_output_bytes, stdout_bytes, stdout_truncated),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain_stream,
        args=(process.stderr, request.max_output_bytes, stderr_bytes, stderr_truncated),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    try:
        process.wait(timeout=request.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_tree(process)
    finally:
        try:
            process.wait(timeout=_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        stdout_thread.join(timeout=_TERMINATION_GRACE_SECONDS)
        stderr_thread.join(timeout=_TERMINATION_GRACE_SECONDS)

    stdout = bytes(stdout_bytes).decode("utf-8", errors="replace")
    stderr = bytes(stderr_bytes).decode("utf-8", errors="replace")
    finished_at = _utc_now()
    reason_codes: tuple[str, ...]
    reasons: tuple[str, ...]
    if timed_out:
        outcome = "timed_out"
        reason_codes = ("timeout",)
        reasons = (f"process exceeded timeout of {request.timeout_seconds:g} seconds",)
    elif process.returncode == 0:
        outcome = "completed"
        reason_codes = ()
        reasons = ()
    else:
        outcome = "failed"
        reason_codes = ("nonzero_exit",)
        reasons = (f"process exited with status {process.returncode}",)
    return _record_from_inspection(
        request,
        inspection,
        outcome=outcome,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=time.monotonic() - start_clock,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated[0],
        stderr_truncated=stderr_truncated[0],
        reason_codes=reason_codes,
        reasons=reasons,
    )
