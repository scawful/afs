from __future__ import annotations

import json
import os
import select
import sys
import time
from pathlib import Path

import psutil
import pytest

from afs.execution import (
    ArgvCommand,
    ExecutionInputError,
    ExecutionPolicy,
    ExecutionRequest,
    LegacyShellCommand,
    execute_checked,
    inspect_execution,
)
from afs.response_schemas import validate_structured_response


def _request(
    cwd: Path,
    *argv: str,
    timeout_seconds: float = 5,
    max_output_bytes: int = 1024 * 1024,
    inherit_env: tuple[str, ...] = (),
    set_env: dict[str, str] | None = None,
    isolation: str = "process",
    network: str = "inherit",
    redact_argv_indices: tuple[int, ...] = (),
) -> ExecutionRequest:
    return ExecutionRequest(
        command=ArgvCommand(tuple(argv)),
        caller="pytest",
        purpose="exercise the checked execution contract",
        cwd=cwd,
        inherit_env=inherit_env,
        set_env=set_env or {},
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        isolation=isolation,
        network=network,
        redact_argv_indices=redact_argv_indices,
    )


def _policy(
    root: Path,
    executable: str = sys.executable,
    *,
    allowed_env: frozenset[str] = frozenset(),
    allow_legacy_shell: bool = False,
) -> ExecutionPolicy:
    return ExecutionPolicy(
        allowed_cwd_roots=(root,),
        allowed_executables=frozenset({executable}),
        allowed_env=allowed_env,
        allow_legacy_shell=allow_legacy_shell,
    )


def test_inspection_is_pure_and_projection_matches_protocol(tmp_path: Path) -> None:
    sentinel = tmp_path / "must-not-exist"
    request = _request(
        tmp_path,
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(sentinel)!r}).touch()",
    )

    inspection = inspect_execution(request, _policy(tmp_path))

    assert inspection.allowed
    assert not sentinel.exists()
    assert validate_structured_response("v1/execution/inspection", inspection.to_dict()).valid


def test_structured_argv_is_literal_and_audit_redacts_secrets(tmp_path: Path) -> None:
    literal = "$(printf injected); echo nope"
    secret = "top-secret-value"
    script = (
        "import json,os,sys; "
        "print(json.dumps({'args':sys.argv[1:],"
        "'secret':os.environ['AFS_TEST_SECRET'],"
        "'pythonpath':os.environ.get('PYTHONPATH')}))"
    )
    request = _request(
        tmp_path,
        sys.executable,
        "-c",
        script,
        literal,
        secret,
        set_env={"AFS_TEST_SECRET": secret},
        redact_argv_indices=(4,),
    )

    record = execute_checked(
        request,
        _policy(tmp_path, allowed_env=frozenset({"AFS_TEST_SECRET"})),
        environ={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
            "PYTHONPATH": "must-not-leak",
        },
    )

    assert record.outcome == "completed"
    payload = json.loads(record.stdout)
    assert payload["args"] == [literal, secret]
    assert payload["secret"] == secret
    assert payload["pythonpath"] is None
    audit = record.audit_dict()
    assert audit["redacted_argv"][4] == "<redacted>"
    assert secret not in json.dumps(audit)
    assert "PYTHONPATH" not in audit["environment_keys"]
    assert validate_structured_response("v1/execution/record", audit).valid


def test_environment_values_change_request_hash_but_are_not_emitted(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        sys.executable,
        "-c",
        "pass",
        inherit_env=("AFS_TOKEN",),
    )
    policy = _policy(tmp_path, allowed_env=frozenset({"AFS_TOKEN"}))

    first = inspect_execution(request, policy, environ={"AFS_TOKEN": "first"})
    second = inspect_execution(request, policy, environ={"AFS_TOKEN": "s3cr3t-value-b"})

    assert first.request_sha256 != second.request_sha256
    assert first.environment_sha256 != second.environment_sha256
    assert "first" not in json.dumps(first.to_dict())
    assert "s3cr3t-value-b" not in json.dumps(second.to_dict())


def test_non_utf8_source_environment_fails_closed(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        sys.executable,
        "-c",
        "pass",
        inherit_env=("AFS_INVALID_TEXT",),
    )
    policy = _policy(tmp_path, allowed_env=frozenset({"AFS_INVALID_TEXT"}))

    with pytest.raises(ExecutionInputError, match="environment value.*UTF-8"):
        inspect_execution(
            request,
            policy,
            environ={"AFS_INVALID_TEXT": "\ud800"},
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("invalid\x00value", "must not contain NUL bytes"),
        ("x" * (1024 * 1024 + 1), "exceeds 1048576 characters"),
    ],
    ids=("nul", "oversized"),
)
def test_invalid_source_environment_values_fail_closed(
    tmp_path: Path,
    value: str,
    message: str,
) -> None:
    request = _request(
        tmp_path,
        sys.executable,
        "-c",
        "pass",
        inherit_env=("AFS_INVALID_VALUE",),
    )
    policy = _policy(tmp_path, allowed_env=frozenset({"AFS_INVALID_VALUE"}))

    with pytest.raises(ExecutionInputError, match=message):
        inspect_execution(
            request,
            policy,
            environ={"AFS_INVALID_VALUE": value},
        )


def test_windows_environment_lookup_preserves_system_root_and_rejects_case_aliases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from afs.execution import broker

    request = _request(tmp_path, sys.executable, "-c", "pass")
    policy = _policy(tmp_path)
    monkeypatch.setattr(broker.os, "name", "nt")

    environment, _, _ = broker._effective_environment(
        request,
        policy,
        {"SystemRoot": r"C:\Windows"},
    )
    assert environment["SYSTEMROOT"] == r"C:\Windows"

    with pytest.raises(ExecutionInputError, match="must not differ only by case"):
        broker._effective_environment(
            request,
            policy,
            {"PATH": "first", "Path": "second"},
        )


def test_cwd_outside_root_and_symlink_traversal_are_blocked(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    request = _request(link, sys.executable, "-c", "pass")
    inspection = inspect_execution(request, _policy(root))

    assert not inspection.allowed
    assert "cwd_outside_allowed_roots" in inspection.reason_codes


@pytest.mark.parametrize(
    ("isolation", "network", "reason"),
    [
        ("sandbox", "inherit", "unsupported_isolation"),
        ("container", "inherit", "unsupported_isolation"),
        ("process", "deny", "unsupported_network"),
    ],
)
def test_unsupported_backends_fail_closed(
    tmp_path: Path, isolation: str, network: str, reason: str
) -> None:
    request = _request(
        tmp_path,
        sys.executable,
        "-c",
        "pass",
        isolation=isolation,
        network=network,
    )

    record = execute_checked(request, _policy(tmp_path))

    assert record.outcome == "blocked"
    assert reason in record.reason_codes


def test_missing_and_disallowed_executables_are_blocked(tmp_path: Path) -> None:
    missing = _request(tmp_path, "definitely-not-an-afs-executable", "--version")
    missing_inspection = inspect_execution(
        missing,
        _policy(tmp_path, executable="definitely-not-an-afs-executable"),
    )
    assert not missing_inspection.allowed
    assert "executable_not_found" in missing_inspection.reason_codes

    existing = _request(tmp_path, sys.executable, "-c", "pass")
    denied_inspection = inspect_execution(existing, _policy(tmp_path, executable="other"))
    assert not denied_inspection.allowed
    assert "executable_not_allowed" in denied_inspection.reason_codes


def test_basename_allowlist_cannot_be_redirected_by_request_path(
    tmp_path: Path,
) -> None:
    executable_name = "afs-path-hijack.exe" if os.name == "nt" else "afs-path-hijack"
    trusted_dir = tmp_path / "trusted"
    attacker_dir = tmp_path / "attacker"
    trusted_dir.mkdir()
    attacker_dir.mkdir()
    for directory in (trusted_dir, attacker_dir):
        executable = directory / executable_name
        executable.write_bytes(b"not executed")
        executable.chmod(0o755)

    source_environment = {"PATH": str(trusted_dir), "HOME": str(tmp_path)}
    policy = ExecutionPolicy(
        allowed_cwd_roots=(tmp_path,),
        allowed_executables=frozenset({executable_name}),
        allowed_env=frozenset({"PATH"}),
    )
    trusted = inspect_execution(
        _request(tmp_path, executable_name),
        policy,
        environ=source_environment,
    )
    redirected = inspect_execution(
        _request(
            tmp_path,
            executable_name,
            set_env={"PATH": str(attacker_dir)},
        ),
        policy,
        environ=source_environment,
    )

    assert trusted.allowed
    assert trusted.resolved_executable == str((trusted_dir / executable_name).resolve())
    assert not redirected.allowed
    assert redirected.resolved_executable == str((attacker_dir / executable_name).resolve())
    assert "executable_not_allowed" in redirected.reason_codes


def test_spawn_failure_returns_a_record(tmp_path: Path, monkeypatch) -> None:
    from afs.execution import broker

    request = _request(tmp_path, sys.executable, "-c", "pass")

    def fail_spawn(*args, **kwargs):
        raise OSError("simulated executable race")

    monkeypatch.setattr(broker.subprocess, "Popen", fail_spawn)
    record = execute_checked(request, _policy(tmp_path))

    assert record.outcome == "spawn_error"
    assert record.returncode is None
    assert record.reason_codes == ("spawn_error",)
    assert "simulated executable race" in record.reasons[0]
    assert validate_structured_response("v1/execution/record", record.audit_dict()).valid


def test_legacy_shell_requires_explicit_policy(tmp_path: Path) -> None:
    request = ExecutionRequest(
        command=LegacyShellCommand("printf legacy"),
        caller="pytest",
        purpose="exercise legacy migration policy",
        cwd=tmp_path,
    )
    denied = ExecutionPolicy(
        allowed_cwd_roots=(tmp_path,),
        allowed_executables=frozenset({"bash"}),
    )
    assert execute_checked(request, denied).outcome == "blocked"

    if not any(Path(path).joinpath("bash").exists() for path in os.get_exec_path()):
        pytest.skip("bash is unavailable on this platform")
    allowed = ExecutionPolicy(
        allowed_cwd_roots=(tmp_path,),
        allowed_executables=frozenset({"bash"}),
        allow_legacy_shell=True,
    )
    record = execute_checked(request, allowed)
    assert record.outcome == "completed"
    assert record.stdout == "legacy"


def test_legacy_shell_policy_opt_in_requires_a_boolean(tmp_path: Path) -> None:
    with pytest.raises(ExecutionInputError, match="allow_legacy_shell must be a boolean"):
        ExecutionPolicy(
            allowed_cwd_roots=(tmp_path,),
            allowed_executables=frozenset({sys.executable}),
            allow_legacy_shell="false",  # type: ignore[arg-type]
        )


def test_executable_index_cannot_be_redacted(tmp_path: Path) -> None:
    with pytest.raises(ExecutionInputError, match="executable index 0"):
        _request(
            tmp_path,
            sys.executable,
            "-c",
            "pass",
            redact_argv_indices=(0,),
        )


@pytest.mark.parametrize("field", ["caller", "purpose", "cwd"])
def test_required_request_text_rejects_whitespace_only(
    tmp_path: Path,
    field: str,
) -> None:
    values = {
        "command": ArgvCommand((sys.executable,)),
        "caller": "pytest",
        "purpose": "exercise required request text",
        "cwd": tmp_path,
    }
    values[field] = "   "

    with pytest.raises(ExecutionInputError, match="non-empty string"):
        ExecutionRequest(**values)


def test_windows_batch_executables_are_blocked_from_structured_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from afs.execution import broker

    executable = tmp_path / "check.cmd"
    executable.write_text("echo must-not-run\n", encoding="utf-8")
    executable.chmod(0o755)
    request = _request(tmp_path, str(executable))
    monkeypatch.setattr(
        broker,
        "_is_unsupported_windows_batch_executable",
        lambda _resolved: True,
    )

    inspection = inspect_execution(request, _policy(tmp_path, str(executable)))

    assert not inspection.allowed
    assert "unsupported_batch_executable" in inspection.reason_codes


def _wait_for_child_exit(pid_file: Path, deadline_seconds: float = 3.0) -> bool:
    """Poll a recorded descendant PID until it has exited.

    The broker signals descendants before ``execute_checked`` returns, so the
    deadline only needs to absorb reap latency — keeping it tight also bounds
    kill latency, so a broker that defers descendant cleanup fails here. A
    zombie counts as exited (a pid-1 test runner may never reap orphans). The
    temporary-directory name in the command line proves that a matching PID is
    still our child; an inaccessible PID remains ambiguous and therefore never
    counts as exited.
    """
    child_pid = int(pid_file.read_text())
    reused_marker = pid_file.parent.name
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        try:
            proc = psutil.Process(child_pid)
            if proc.status() == psutil.STATUS_ZOMBIE:
                return True
            if reused_marker not in " ".join(proc.cmdline()):
                return True
        except psutil.NoSuchProcess:
            return True
        except psutil.AccessDenied:
            # AccessDenied does not prove that the child exited. Keep polling
            # and fail closed if its identity cannot be established.
            pass
        except SystemError as error:
            # psutil 7.2 can leak a chained PermissionError from proc_cmdline
            # on macOS instead of translating it to AccessDenied. Treat only
            # that permission-shaped native failure as ambiguous; unexpected
            # SystemErrors must still fail the test loudly.
            cause = error.__cause__ or error.__context__
            if not isinstance(cause, PermissionError):
                raise
        time.sleep(0.05)
    return False


def test_wait_for_child_exit_fails_closed_on_access_denied(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pid_file = tmp_path / "child-pid"
    pid_file.write_text(str(os.getpid()))

    def deny_process_access(pid: int):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", deny_process_access)

    assert not _wait_for_child_exit(pid_file, deadline_seconds=0.01)


def test_wait_for_child_exit_fails_closed_on_chained_permission_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pid_file = tmp_path / "child-pid"
    pid_file.write_text(str(os.getpid()))

    class PermissionProcess:
        def status(self) -> str:
            return psutil.STATUS_RUNNING

        def cmdline(self) -> list[str]:
            try:
                raise PermissionError("permission denied")
            except PermissionError as error:
                raise SystemError("proc_cmdline failed") from error

    monkeypatch.setattr(psutil, "Process", lambda _pid: PermissionProcess())

    assert not _wait_for_child_exit(pid_file, deadline_seconds=0.01)


def test_wait_for_child_exit_reraises_unrelated_system_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pid_file = tmp_path / "child-pid"
    pid_file.write_text(str(os.getpid()))

    class BrokenProcess:
        def status(self) -> str:
            raise SystemError("unexpected native failure")

    monkeypatch.setattr(psutil, "Process", lambda _pid: BrokenProcess())

    with pytest.raises(SystemError, match="unexpected native failure"):
        _wait_for_child_exit(pid_file, deadline_seconds=0.01)


def test_timeout_terminates_descendant_process(tmp_path: Path, monkeypatch) -> None:
    from afs.execution import broker

    if os.name != "nt":
        monkeypatch.setattr(broker, "_TERMINATION_GRACE_SECONDS", 0.1)
    ready = tmp_path / "child-ready"
    marker = tmp_path / "child-survived"
    child_pid_file = tmp_path / "child-pid"
    child_code = (
        "import time\n"
        "from pathlib import Path\n"
        f"Path({str(ready)!r}).touch()\n"
        "time.sleep(10)\n"
        f"Path({str(marker)!r}).touch()\n"
        "time.sleep(30)\n"
    )
    parent_code = (
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        f"proc = subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"Path({str(child_pid_file)!r}).write_text(str(proc.pid))\n"
        f"ready = Path({str(ready)!r})\n"
        "deadline = time.monotonic() + 20\n"
        "while not ready.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "time.sleep(30)\n"
    )
    request = _request(tmp_path, sys.executable, "-c", parent_code, timeout_seconds=3)

    record = execute_checked(request, _policy(tmp_path))

    assert record.outcome == "timed_out"
    assert record.timed_out
    assert ready.exists()
    assert _wait_for_child_exit(child_pid_file)
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group regression")
def test_successful_parent_cannot_leave_same_session_descendant(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "child-ready"
    marker = tmp_path / "child-survived"
    child_pid_file = tmp_path / "child-pid"
    child_code = (
        "import signal,time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(ready)!r}).touch()\n"
        "time.sleep(30)\n"
        f"Path({str(marker)!r}).touch()\n"
    )
    parent_code = (
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        "proc = subprocess.Popen(\n"
        f"    [sys.executable, '-c', {child_code!r}],\n"
        "    stdin=subprocess.DEVNULL,\n"
        "    stdout=subprocess.DEVNULL,\n"
        "    stderr=subprocess.DEVNULL,\n"
        ")\n"
        f"Path({str(child_pid_file)!r}).write_text(str(proc.pid))\n"
        f"ready = Path({str(ready)!r})\n"
        "deadline = time.monotonic() + 15\n"
        "while not ready.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
    )
    request = _request(tmp_path, sys.executable, "-c", parent_code, timeout_seconds=20)

    record = execute_checked(request, _policy(tmp_path))

    assert ready.exists()
    assert record.outcome == "completed"
    assert _wait_for_child_exit(child_pid_file)
    assert not marker.exists()


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(select, "kqueue"),
    reason="kqueue fallback is unavailable",
)
def test_kqueue_fallback_waits_without_reaping(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from afs.execution import broker

    monkeypatch.delattr(broker.os, "waitid", raising=False)
    request = _request(tmp_path, sys.executable, "-c", "print('kqueue')")

    record = execute_checked(request, _policy(tmp_path))

    assert record.outcome == "completed"
    assert record.stdout.strip() == "kqueue"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group regression")
def test_timeout_kills_descendant_that_ignores_sigterm(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from afs.execution import broker

    monkeypatch.setattr(broker, "_TERMINATION_GRACE_SECONDS", 0.1)
    ready = tmp_path / "child-ready"
    marker = tmp_path / "child-survived"
    child_pid_file = tmp_path / "child-pid"
    child_code = (
        "import signal,time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(ready)!r}).touch()\n"
        "time.sleep(30)\n"
        f"Path({str(marker)!r}).touch()\n"
    )
    parent_code = (
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        f"proc = subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"Path({str(child_pid_file)!r}).write_text(str(proc.pid))\n"
        f"ready = Path({str(ready)!r})\n"
        "deadline = time.monotonic() + 15\n"
        "while not ready.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "time.sleep(30)\n"
    )
    request = _request(
        tmp_path,
        sys.executable,
        "-c",
        parent_code,
        timeout_seconds=3,
    )

    record = execute_checked(request, _policy(tmp_path))

    assert ready.exists()
    assert record.outcome == "timed_out"
    assert _wait_for_child_exit(child_pid_file)
    assert not marker.exists()


def test_output_is_capped_and_invalid_utf8_is_replaced(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        sys.executable,
        "-c",
        "import os; os.write(1, b'\\xff' + b'x' * 1000); os.write(2, b'y' * 1000)",
        max_output_bytes=16,
    )

    record = execute_checked(request, _policy(tmp_path))

    assert record.outcome == "completed"
    assert record.stdout.startswith("\ufffd")
    assert len(record.stdout.encode("utf-8")) <= 18
    assert len(record.stderr) == 16
    assert record.stdout_truncated
    assert record.stderr_truncated


def test_request_rejects_unknown_fields_and_invalid_limits(tmp_path: Path) -> None:
    payload = _request(tmp_path, sys.executable, "-c", "pass").to_dict()
    payload["permissions"] = {"self_granted": True}
    with pytest.raises(ExecutionInputError, match="unknown fields"):
        ExecutionRequest.from_dict(payload)

    with pytest.raises(ExecutionInputError, match="timeout_seconds"):
        _request(tmp_path, sys.executable, timeout_seconds=3601)
