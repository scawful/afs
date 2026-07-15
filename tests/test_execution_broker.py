from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

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
    assert validate_structured_response(
        "v1/execution/inspection", inspection.to_dict()
    ).valid


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
    second = inspect_execution(
        request, policy, environ={"AFS_TOKEN": "s3cr3t-value-b"}
    )

    assert first.request_sha256 != second.request_sha256
    assert first.environment_sha256 != second.environment_sha256
    assert "first" not in json.dumps(first.to_dict())
    assert "s3cr3t-value-b" not in json.dumps(second.to_dict())


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
    assert validate_structured_response(
        "v1/execution/record", record.audit_dict()
    ).valid


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


def test_timeout_terminates_descendant_process(tmp_path: Path) -> None:
    marker = tmp_path / "child-survived"
    child_code = f"import time; time.sleep(1); open({str(marker)!r}, 'w').close()"
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        "time.sleep(30)"
    )
    request = _request(
        tmp_path,
        sys.executable,
        "-c",
        parent_code,
        timeout_seconds=0.15,
    )

    record = execute_checked(request, _policy(tmp_path))
    time.sleep(1.2)

    assert record.outcome == "timed_out"
    assert record.timed_out
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
