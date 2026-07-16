"""Execution helpers for approved AFS work-assistant actions."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .human_provenance import default_terminal_reader
from .work_assistant import EXTERNAL_WRITE_ACTIONS, WorkAssistantStore

PAYLOAD_VERSION = 1
COMMUNICATION_WRITE_ACTIONS = frozenset(
    {
        "post_ticket_comment",
        "post_code_review_comment",
        "post_doc_comment",
        "post_pr_comment",
        "post_pull_request_review",
        "publish_comment",
        "reply_to_comment",
        "send_email",
        "send_message",
        "submit_review",
    }
)


class WorkApprovalExecutionError(RuntimeError):
    """Raised when an approved work action cannot be executed."""


class HumanApprovalRequiredError(WorkApprovalExecutionError):
    """Raised when an external-write approval lacks interactive human confirmation.

    External and communication writes (see :data:`EXTERNAL_WRITE_ACTIONS`) must be
    confirmed by a person at a terminal before an agent may act on the user's behalf.
    The approval store accepts authorization only from the controlling-terminal
    broker; caller-supplied fields such as ``--by human`` are compatibility metadata,
    not proof. External writes also receive a fresh execution-time confirmation.
    """


# Generic sentinel stamped on a gated approval that carries no specific action verb
# (see work_assistant._normalize_approval_event). It is deliberately NOT a member of
# EXTERNAL_WRITE_ACTIONS, so a naive membership test let it slip past the human gate.
_GENERIC_EXTERNAL_WRITE = "external_write"
_LOCAL_TARGET_SYSTEMS = frozenset(
    {
        "",
        "afs",
        "context",
        "history",
        "hivemind",
        "items",
        "knowledge",
        "local",
        "memory",
        "scratchpad",
    }
)

# Verb stems that mark an action as an outward/mutating write even when the exact
# action name is not enumerated in EXTERNAL_WRITE_ACTIONS. Fail-safe: a novel outward
# action from a new connector (e.g. "post_status", "escalate_incident") still trips the
# human gate instead of sliding through just because it was not on the hardcoded list.
# Matched against underscore/hyphen-split tokens (not raw substrings) so benign names
# like "preview_doc" are not misread as "review".
_OUTWARD_ACTION_STEMS = frozenset(
    {
        "post",
        "send",
        "submit",
        "publish",
        "reply",
        "share",
        "comment",
        "email",
        "message",
        "notify",
        "mention",
        "review",
        "assign",
        "add",
        "archive",
        "change",
        "create",
        "delete",
        "edit",
        "escalate",
        "page",
        "merge",
        "close",
        "remove",
        "resolve",
        "update",
        "write",
    }
)


def action_requires_human_ack(action: str) -> bool:
    """True when an action is an external/communication write and needs a human ack.

    Fail-safe classification: the enumerated ``EXTERNAL_WRITE_ACTIONS`` (of which
    ``COMMUNICATION_WRITE_ACTIONS`` is a subset) are covered, plus the generic
    ``external_write`` sentinel and any action name whose tokens carry an outward verb
    stem. This closes the hole where a gated approval with a generic or novel action
    (``action_requires_human_ack("external_write")`` was ``False``) slipped past the
    terminal-confirmation gate. An empty action stays non-outward — upstream stamps the
    sentinel whenever the verb is unknown, so an empty string here means "no action".
    """
    normalized = str(action or "").strip()
    if not normalized:
        return False
    if normalized in EXTERNAL_WRITE_ACTIONS:
        return True
    lowered = normalized.lower()
    if lowered.replace("-", "_") == _GENERIC_EXTERNAL_WRITE:
        return True
    tokens = lowered.replace("-", "_").split("_")
    return any(token in _OUTWARD_ACTION_STEMS for token in tokens)


def approval_requires_human_ack(approval: dict[str, Any]) -> bool:
    """True when an approval row represents an outward/external write.

    Action-name stems are useful but not authoritative: new connectors can choose
    names like ``delete_ticket`` or ``update_crm_record`` before AFS knows those exact
    verbs. The approval target is the stronger backstop. Anything aimed at a non-local
    target system needs the terminal confirmation even if the action label looks
    harmless or internal.
    """
    if action_requires_human_ack(str(approval.get("action") or "")):
        return True
    target_system = str(approval.get("target_system") or "").strip().lower()
    return target_system not in _LOCAL_TARGET_SYSTEMS


def _default_tty_reader(tty_path: str | None) -> Callable[[str], str | None]:
    """Compatibility seam for the cross-platform controlling-terminal reader."""

    return default_terminal_reader(tty_path)


def confirm_human_approval(
    approval: dict[str, Any],
    *,
    tty_path: str | None = None,
    reader: Callable[[str], str | None] | None = None,
) -> None:
    """Require interactive human confirmation for an external-write approval.

    A no-op for non-external actions. For external writes it prompts on the
    controlling terminal and requires the operator to type the approval id. Raises
    :class:`HumanApprovalRequiredError` when no terminal is available or the typed
    value does not match — so approval cannot be granted from a non-interactive
    (agent) context. ``reader`` is injectable for testing.
    """
    action = str(approval.get("action") or "").strip()
    if not approval_requires_human_ack(approval):
        return
    approval_id = str(approval.get("approval_id") or "")
    target = f"{approval.get('target_system') or '?'}:{approval.get('target_id') or '?'}"
    summary = str(approval.get("summary") or "")
    prompt = "\n".join(
        [
            "",
            "=== HUMAN CONFIRMATION REQUIRED (external write) ===",
            f"  action:   {action}",
            f"  target:   {target}",
            f"  summary:  {summary}",
            f"  approval: {approval_id}",
            "Approving lets an agent perform this outward action on your behalf.",
            f"Type the approval id ({approval_id}) to confirm, anything else aborts: ",
        ]
    )
    read = reader or _default_tty_reader(tty_path)
    response = read(prompt)
    if response is None:
        raise HumanApprovalRequiredError(
            "external write requires interactive human confirmation, but no terminal "
            f"is available; refusing to approve {approval_id!r} in a non-interactive "
            "context. Re-run `afs work approvals approve` from an interactive terminal."
        )
    if response.strip() != approval_id:
        raise HumanApprovalRequiredError(
            f"human confirmation did not match approval id; not approving {approval_id!r}."
        )


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
    require_human_ack: bool = True,
    tty_path: str | None = None,
    confirm_reader: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Execute one approved action by passing its JSON payload to a command.

    The command is invoked without a shell. A temporary approval JSON file is
    appended as the final argument and also exposed as
    ``AFS_WORK_APPROVAL_FILE``. On success the approval is marked ``applied``.
    On failure, the approval remains ``approved`` so the caller can retry.

    Execution — not approval — is when the connector actually performs the outward
    action, so for external writes it demands a fresh interactive human confirmation
    here too (defense in depth over the approve-time check in ``cli/work.py``). This
    closes the hole where a row that reached ``approved`` by any path (a pre-existing
    row, a direct ``store.approve``, or an action the classifier missed) would execute
    with no confirmation. A headless agent has no controlling terminal, so it cannot
    satisfy the gate. ``confirm_reader`` is injectable for testing; a ``dry_run``
    preview never prompts. Pass ``require_human_ack=False`` only for a caller that has
    already confirmed out of band.
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
    if not approval.get("human_confirmed"):
        raise HumanApprovalRequiredError(
            f"approval {approval_id!r} was recorded programmatically and is not "
            "human-confirmed; re-run `afs work approvals approve` from a "
            "controlling terminal"
        )
    if not command:
        raise WorkApprovalExecutionError("executor command is required")

    if require_human_ack and approval_requires_human_ack(approval):
        confirm_human_approval(approval, tty_path=tty_path, reader=confirm_reader)

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
        sample_id = _record_applied_communication_sample(
            store,
            approval,
            result=result,
            actor=actor,
        )
        store.record_activity(
            activity_type="approval_applied",
            summary=f"Applied approved action {approval_id}",
            target_system=approval["target_system"],
            target_id=approval["target_id"],
            actor=actor,
            metadata={
                "approval_id": approval_id,
                "result": result,
                "communication_sample_id": sample_id,
            },
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


def _record_applied_communication_sample(
    store: WorkAssistantStore,
    approval: dict[str, Any],
    *,
    result: dict[str, Any],
    actor: str,
) -> str:
    action = str(approval.get("action") or "").strip()
    if action not in COMMUNICATION_WRITE_ACTIONS:
        return ""

    text = _extract_approved_text(approval.get("preview"), result.get("output"))
    if not text:
        return ""

    return store.record_communication_sample(
        text=text,
        source_system=str(approval.get("target_system") or ""),
        source_id=str(approval.get("target_id") or ""),
        channel=str(approval.get("target_system") or ""),
        purpose=action,
        style_notes=["approved external write"],
        provenance=[
            {
                "source": "work_approval",
                "approval_id": approval.get("approval_id"),
                "action": action,
                "target_system": approval.get("target_system"),
                "target_id": approval.get("target_id"),
                "actor": actor,
                "result_status": result.get("status"),
            }
        ],
        confidence=0.9,
        dedupe_key=f"comm_approval_{approval.get('approval_id')}",
    )


def _extract_approved_text(preview: Any, output: Any) -> str:
    for value in (output, preview):
        text = _extract_text_candidate(value)
        if text:
            return text
    return ""


def _extract_text_candidate(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""

    for key in (
        "body",
        "text",
        "comment",
        "message",
        "content",
        "draft",
        "reply",
        "review",
        "posted_text",
        "sent_text",
    ):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    # Some connector outputs wrap the submitted payload one level down.
    for key in ("preview", "details", "payload", "result"):
        nested = value.get(key)
        text = _extract_text_candidate(nested)
        if text:
            return text
    return ""


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
