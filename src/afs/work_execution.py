"""Execution helpers for approved AFS work-assistant actions."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .work_assistant import WorkAssistantStore

PAYLOAD_VERSION = 1


class WorkApprovalExecutionError(RuntimeError):
    """Raised when an approved work action cannot be executed."""


def build_approval_payload(
    approval: dict[str, Any],
    *,
    context_root: Path,
    actor: str = "agent",
) -> dict[str, Any]:
    """Build the JSON payload passed to connector executors."""
    return {
        "version": PAYLOAD_VERSION,
        "context_root": str(context_root.expanduser().resolve()),
        "actor": actor,
        "approval": approval,
    }


def execute_approved_action(
    store: WorkAssistantStore,
    *,
    context_root: Path,
    approval_id: str,
    executor_command: list[str] | tuple[str, ...],
    actor: str = "agent",
    timeout: int = 60,
    dry_run: bool = False,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Execute one approved action by passing its JSON payload to a command.

    The command is invoked without a shell. A temporary approval JSON file is
    appended as the final argument and also exposed as
    ``AFS_WORK_APPROVAL_FILE``. On success the approval is marked ``applied``.
    On failure, the approval remains ``approved`` so the caller can retry.
    """
    approval = store.get_approval(approval_id)
    if approval is None:
        raise WorkApprovalExecutionError(f"approval not found: {approval_id}")
    if approval["status"] != "approved":
        raise WorkApprovalExecutionError(
            f"approval must be approved before execution: {approval_id}"
        )

    payload = build_approval_payload(approval, context_root=context_root, actor=actor)
    command = [str(part) for part in executor_command if str(part).strip()]
    if dry_run:
        store.record_activity(
            activity_type="approval_execution_dry_run",
            summary=f"Preview executor payload for {approval_id}",
            target_system=approval["target_system"],
            target_id=approval["target_id"],
            actor=actor,
            metadata={"approval_id": approval_id, "executor_command": command},
        )
        return {
            "status": "dry_run",
            "approval_id": approval_id,
            "payload": payload,
            "executor_command": command,
        }
    if not command:
        raise WorkApprovalExecutionError("executor command is required")

    resolved_command = [_resolve_executable(command[0]), *command[1:]]
    run_cwd = cwd.expanduser().resolve() if cwd else Path.cwd()
    with tempfile.TemporaryDirectory(prefix="afs-work-approval-") as temp_dir:
        payload_path = Path(temp_dir) / "approval.json"
        payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        env = os.environ.copy()
        env["AFS_CONTEXT_ROOT"] = str(context_root.expanduser().resolve())
        env["AFS_WORK_APPROVAL_ID"] = approval_id
        env["AFS_WORK_APPROVAL_FILE"] = str(payload_path)
        try:
            completed = subprocess.run(
                [*resolved_command, str(payload_path)],
                cwd=run_cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            result = _timeout_result(
                exc,
                approval_id=approval_id,
                executor_command=resolved_command,
            )
            store.record_approval_result(approval_id, result=result)
            store.record_activity(
                activity_type="approval_execution_failed",
                summary=f"Executor timed out for approved action {approval_id}",
                target_system=approval["target_system"],
                target_id=approval["target_id"],
                actor=actor,
                metadata={"approval_id": approval_id, "result": result},
            )
            return result

    result = _execution_result(
        completed,
        approval_id=approval_id,
        executor_command=resolved_command,
    )
    if completed.returncode == 0:
        store.record_approval_result(approval_id, result=result, status="applied")
        store.record_activity(
            activity_type="approval_applied",
            summary=f"Applied approved action {approval_id}",
            target_system=approval["target_system"],
            target_id=approval["target_id"],
            actor=actor,
            metadata={"approval_id": approval_id, "result": result},
        )
    else:
        store.record_approval_result(approval_id, result=result)
        store.record_activity(
            activity_type="approval_execution_failed",
            summary=f"Executor failed for approved action {approval_id}",
            target_system=approval["target_system"],
            target_id=approval["target_id"],
            actor=actor,
            metadata={"approval_id": approval_id, "result": result},
        )
    return result


def _resolve_executable(executable: str) -> str:
    expanded = Path(executable).expanduser()
    if expanded.parent != Path(".") or executable.startswith("."):
        if not expanded.exists():
            raise WorkApprovalExecutionError(f"executor not found: {executable}")
        return str(expanded.resolve())
    resolved = shutil.which(executable)
    if not resolved:
        raise WorkApprovalExecutionError(f"executor not found on PATH: {executable}")
    return resolved


def _execution_result(
    completed: subprocess.CompletedProcess[str],
    *,
    approval_id: str,
    executor_command: list[str],
) -> dict[str, Any]:
    stdout = completed.stdout.strip()
    parsed_stdout: Any = None
    if stdout:
        try:
            parsed_stdout = json.loads(stdout)
        except json.JSONDecodeError:
            parsed_stdout = None
    result: dict[str, Any] = {
        "approval_id": approval_id,
        "status": "applied" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "executor_command": executor_command,
        "stdout": stdout,
        "stderr": completed.stderr.strip(),
    }
    if parsed_stdout is not None:
        result["output"] = parsed_stdout
    return result


def _timeout_result(
    exc: subprocess.TimeoutExpired,
    *,
    approval_id: str,
    executor_command: list[str],
) -> dict[str, Any]:
    stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    return {
        "approval_id": approval_id,
        "status": "failed",
        "error": "executor timed out",
        "timeout": exc.timeout,
        "returncode": None,
        "executor_command": executor_command,
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
    }
