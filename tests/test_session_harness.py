from __future__ import annotations

import json
import shlex
from argparse import Namespace
from pathlib import Path

import pytest

import afs.cli.core as cli_core_module
import afs.verification as verification_module
from afs.cli.core import (
    session_event_command,
    session_hook_command,
    session_prepare_client_command,
)
from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import (
    AFSConfig,
    GeneralConfig,
    ProfileConfig,
    ProfilesConfig,
    VerificationCheckConfig,
    VerificationConfig,
    VerificationExecutionConfig,
    VerificationProfileConfig,
)
from afs.session_harness import (
    _build_session_feedback_state,
    _build_session_verification_state,
    _verification_record_from_event,
    build_client_session_payload,
)
from afs.verification import verification_item_id


def _make_manager(tmp_path: Path) -> tuple[AFSManager, Path]:
    context_root = tmp_path / ".context"
    skill_root = tmp_path / "skills"
    skill_dir = skill_root / "agentic-background"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: agentic-background\n"
        "triggers: [background, agents, harness]\n"
        "enforcement:\n"
        "  - Stream background progress instead of hiding state.\n"
        "  - Preserve a clear handoff trail for long-running work.\n"
        "verification:\n"
        "  - Verify the harness artifacts and session prompt were written.\n"
        "---\n",
        encoding="utf-8",
    )

    config = AFSConfig(
        general=GeneralConfig(context_root=context_root),
        profiles=ProfilesConfig(
            active_profile="default",
            profiles={
                "default": ProfileConfig(skill_roots=[skill_root]),
            },
        ),
        verification=VerificationConfig(
            default_profile="repo",
            profiles={
                "repo": VerificationProfileConfig(
                    name="repo",
                    checks=[
                        VerificationCheckConfig(
                            name="harness-artifacts",
                            skills=["agentic-background"],
                            workflows=["edit_fast"],
                            executions=[
                                VerificationExecutionConfig(
                                    argv=[
                                        "python3",
                                        "-m",
                                        "pytest",
                                        "-q",
                                        "tests/test_session_harness.py",
                                    ]
                                )
                            ],
                        )
                    ],
                )
            },
        ),
    )
    manager = AFSManager(config=config)
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)

    scratchpad_root = manager.resolve_mount_root(context_root, MountType.SCRATCHPAD)
    scratchpad_root.mkdir(parents=True, exist_ok=True)
    (scratchpad_root / "state.md").write_text(
        "Investigate background-agent handoff state.",
        encoding="utf-8",
    )
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    knowledge_root.mkdir(parents=True, exist_ok=True)
    (knowledge_root / "agents.md").write_text(
        "Background agents should stream state to the harness.",
        encoding="utf-8",
    )
    policy_dir = tmp_path / ".afs"
    policy_dir.mkdir(exist_ok=True)
    (policy_dir / "policy.toml").write_text(
        "[review]\n"
        "focus = [\"order findings by severity\", \"call out missing tests\"]\n\n"
        "[design]\n"
        "constraints = [\"preserve artifact compatibility\"]\n\n"
        "[planning]\n"
        "principles = [\"keep plans reversible\"]\n",
        encoding="utf-8",
    )
    return manager, context_root


def test_session_prepare_client_command_outputs_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, context_root = _make_manager(tmp_path)
    config_path = tmp_path / "afs.toml"

    monkeypatch.setattr(
        cli_core_module,
        "_load_manager_context_and_config_path",
        lambda _args: (manager, context_root, config_path),
    )

    rc = session_prepare_client_command(
        Namespace(
            client="codex",
            session_id="sess-1",
            cwd=str(tmp_path),
            query="background agents harness",
            task="Improve the harness loop.",
            model="codex",
            workflow="edit_fast",
            tool_profile="edit_and_verify",
            pack_mode="focused",
            token_budget=700,
            include_content=False,
            max_query_results=4,
            max_embedding_results=2,
            skills_prompt="background agents harness",
            skills_top_k=5,
            no_session_pack=False,
            no_skills_match=False,
            no_write_artifacts=False,
            json=True,
            config=None,
            path=None,
            context_root=context_root,
            context_dir=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["client"] == "codex"
    assert payload["session_id"] == "sess-1"
    assert payload["bootstrap"]["artifact_paths"]["json"].endswith("session_bootstrap.json")
    assert Path(payload["bootstrap"]["artifact_paths"]["json"]).exists()
    assert payload["pack"]["artifact_paths"]["json"].endswith("session_pack_codex.json")
    assert Path(payload["pack"]["artifact_paths"]["json"]).exists()
    assert payload["skills"]["artifact_paths"]["json"].endswith("session_skills_codex.json")
    assert Path(payload["skills"]["artifact_paths"]["json"]).exists()
    assert payload["skills"]["matches"][0]["name"] == "agentic-background"
    assert payload["skills"]["matches"][0]["enforcement"] == [
        "Stream background progress instead of hiding state.",
        "Preserve a clear handoff trail for long-running work.",
    ]
    assert payload["skills"]["matches"][0]["verification"] == [
        "Verify the harness artifacts and session prompt were written.",
    ]
    assert payload["prompt"]["artifact_paths"]["text"].endswith("session_system_prompt_codex.txt")
    assert Path(payload["prompt"]["artifact_paths"]["text"]).exists()
    assert payload["prompt"]["artifact_paths"]["json"].endswith("session_system_prompt_codex.json")
    assert Path(payload["prompt"]["artifact_paths"]["json"]).exists()
    assert payload["cli_hints"]["workspace_path"] == str(tmp_path.resolve())
    quoted_workspace = shlex.quote(str(tmp_path.resolve()))
    assert payload["cli_hints"]["query_shortcut"] == f"afs query <text> --path {quoted_workspace}"
    assert (
        payload["cli_hints"]["query_canonical"]
        == f"afs context query <text> --path {quoted_workspace}"
    )
    assert payload["cli_hints"]["index_rebuild"] == f"afs index rebuild --path {quoted_workspace}"
    assert payload["cli_hints"]["agent_jobs_inbox"] == f"afs agent-jobs inbox --path {quoted_workspace}"
    assert payload["cli_hints"]["work_summary"] == f"afs work --path {quoted_workspace}"
    assert payload["cli_hints"]["work_approvals"] == f"afs work approvals list --path {quoted_workspace}"
    assert (
        payload["cli_hints"]["work_communication"]
        == f"afs work communication preflight --path {quoted_workspace}"
    )
    assert payload["cli_hints"]["verify_plan"].startswith("afs verify plan --payload-file ")
    assert payload["cli_hints"]["verify_run"].startswith("afs verify run --payload-file ")
    assert isinstance(payload["cli_hints"]["notes"], list)
    assert payload["verification_plan"]["profile"] == "repo"
    assert payload["verification_plan"]["selected_checks"][0]["name"] == "harness-artifacts"
    assert payload["verification_plan"]["selected_checks"][0]["executions"] == [
        {
            "argv": [
                "python3",
                "-m",
                "pytest",
                "-q",
                "tests/test_session_harness.py",
            ],
            "cwd": "",
            "timeout_seconds": 300.0,
            "max_output_bytes": 1024 * 1024,
            "inherit_env": [],
            "env": {},
            "redact_argv_indices": [],
        }
    ]
    assert payload["structured_guidance"]["recommended_schema"] == "design-brief"
    assert payload["structured_guidance"]["followup_schema"] == "verification-summary"
    assert payload["repo_policy"]["review_focus"] == [
        "order findings by severity",
        "call out missing tests",
    ]
    assert payload["repo_policy"]["design_constraints"] == ["preserve artifact compatibility"]
    assert payload["repo_policy"]["planning_principles"] == ["keep plans reversible"]
    prompt_text = Path(payload["prompt"]["artifact_paths"]["text"]).read_text(encoding="utf-8")
    assert "## Skill Enforcement" in prompt_text
    assert "- agentic-background: Stream background progress instead of hiding state." in prompt_text
    assert "## Skill Verification" in prompt_text
    assert "## Verification Plan" in prompt_text
    assert (
        "- harness-artifacts: python3 -m pytest -q tests/test_session_harness.py"
        in prompt_text
    )
    assert "## Repo Policy" in prompt_text
    assert "- order findings by severity" in prompt_text
    assert "## Structured Workflow" in prompt_text
    assert "Recommended schema: design-brief" in prompt_text
    assert "## Session Context" in prompt_text
    assert "Prompt contract:" in prompt_text
    assert payload["artifact_paths"]["json"].endswith("session_client_codex.json")
    assert Path(payload["artifact_paths"]["json"]).exists()
    assert payload["integration"]["notify_command"] == "afs session event"
    assert payload["activity"]["recent_events"] == []
    supported_names = {entry["name"] for entry in payload["integration"]["supported_events"]}
    assert "user_prompt_submit" in supported_names
    assert "task_completed" in supported_names


def test_prepared_session_redacts_verification_secrets_from_all_artifacts(
    tmp_path: Path,
) -> None:
    manager, context_root = _make_manager(tmp_path)
    check = manager.config.verification.profiles["repo"].checks[0]
    execution = check.executions[0]
    execution.argv.append("argv-secret")
    execution.redact_argv_indices = [len(execution.argv) - 1]
    execution.env = {"AFS_SECRET": "env-secret"}
    check.commands = ["printf legacy-secret"]

    payload = build_client_session_payload(
        manager,
        context_root,
        client="codex",
        session_id="session-redaction",
        cwd=tmp_path,
        model="codex",
        workflow="edit_fast",
        tool_profile="edit_and_verify",
        skills_prompt="background agents harness",
        write_artifacts=True,
    )

    check_payload = payload["verification_plan"]["selected_checks"][0]
    execution_payload = check_payload["executions"][0]
    assert execution_payload["argv"][-1] == "<redacted>"
    assert execution_payload["env"] == {"AFS_SECRET": "<redacted>"}
    assert check_payload["commands"] == ["<redacted legacy shell command>"]
    assert all(
        secret not in json.dumps(payload)
        for secret in ("argv-secret", "env-secret", "legacy-secret")
    )
    prompt_text = Path(payload["prompt"]["artifact_paths"]["text"]).read_text(
        encoding="utf-8"
    )
    persisted_payload = Path(payload["artifact_paths"]["json"]).read_text(
        encoding="utf-8"
    )
    for secret in ("argv-secret", "env-secret", "legacy-secret"):
        assert secret not in prompt_text
        assert secret not in persisted_payload
    assert "deprecated; blocked; explicit opt-in required" in prompt_text


def test_prepared_session_preserves_incomplete_git_discovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager, context_root = _make_manager(tmp_path)

    def fail_discovery(_start_dir=None):
        raise verification_module.VerificationDiscoveryError(
            "simulated repository discovery failure"
        )

    monkeypatch.setattr(
        verification_module,
        "discover_git_repo_root",
        lambda _start_dir=None: tmp_path,
    )
    monkeypatch.setattr(
        verification_module,
        "discover_changed_paths",
        fail_discovery,
    )

    payload = build_client_session_payload(
        manager,
        context_root,
        client="codex",
        session_id="session-discovery-failure",
        cwd=tmp_path,
        workflow="edit_fast",
        tool_profile="edit_and_verify",
        skills_prompt="background agents harness",
        write_artifacts=False,
    )

    assert payload["verification_plan"]["discovery_complete"] is False
    assert (
        payload["verification_plan"]["discovery_error"]
        == "simulated repository discovery failure"
    )


def test_blocked_verification_event_remains_blocked_in_session_state() -> None:
    record = _verification_record_from_event(
        event_name="verification_recorded",
        event_payload={
            "verification_status": "blocked",
            "verification_command": "python3 -c '<redacted>'",
            "summary": "required: blocked",
        },
        timestamp="2026-07-15T00:00:00Z",
    )
    payload = {
        "verification_plan": {
            "available": True,
            "selected_checks": [
                {
                    "name": "required",
                    "required": True,
                    "executions": [{"argv": ["python3"], "redact_argv_indices": []}],
                    "commands": [],
                }
            ],
        },
        "activity": {"verification": {"records": [record]}},
        "repo_policy": {},
    }

    state = _build_session_verification_state(payload, final=False)
    payload["activity"]["verification"] = state
    feedback = _build_session_feedback_state(payload)

    assert record["status"] == "blocked"
    assert state["status"] == "blocked"
    assert state["record_count"] == 1
    assert feedback["signals"] == ["verification_blocked"]
    assert feedback["counts"] == {"verification_blocked": 1}


def test_one_focused_check_cannot_satisfy_multiple_required_checks() -> None:
    first = _verification_record_from_event(
        event_name="verification_recorded",
        event_payload={
            "verification_status": "passed",
            "verification_command": "python3 -m pytest tests/easy.py",
            "verification_check": "easy",
            "verification_item_id": verification_item_id("easy", "execution", 0),
        },
        timestamp="2026-07-15T00:00:00Z",
    )
    second = _verification_record_from_event(
        event_name="verification_recorded",
        event_payload={
            "verification_status": "passed",
            "verification_command": "python3 -m pytest tests/hard.py",
            "verification_check": "hard",
            "verification_item_id": verification_item_id("hard", "execution", 0),
        },
        timestamp="2026-07-15T00:00:01Z",
    )
    payload = {
        "verification_plan": {
            "available": True,
            "selected_checks": [
                {
                    "name": "easy",
                    "required": True,
                    "executions": [{"argv": ["python3"], "redact_argv_indices": []}],
                    "commands": [],
                },
                {
                    "name": "hard",
                    "required": True,
                    "executions": [{"argv": ["python3"], "redact_argv_indices": []}],
                    "commands": [],
                },
            ],
        },
        "activity": {"verification": {"records": [first]}},
    }

    partial = _build_session_verification_state(payload, final=False)
    payload["activity"]["verification"]["records"].append(second)
    complete = _build_session_verification_state(payload, final=False)

    assert first["check_name"] == "easy"
    assert partial["status"] == "pending"
    assert partial["completed_required_checks"] == ["easy"]
    assert complete["status"] == "passed"
    assert complete["completed_required_checks"] == ["easy", "hard"]


def test_duplicate_item_passes_do_not_cover_a_distinct_required_item() -> None:
    first_item = verification_item_id("combo", "execution", 0)
    second_item = verification_item_id("combo", "execution", 1)
    duplicate = {
        "timestamp": "2026-07-15T00:00:00Z",
        "event": "verification_recorded",
        "status": "passed",
        "command": "first-command",
        "check_name": "combo",
        "item_id": first_item,
    }
    payload = {
        "verification_plan": {
            "available": True,
            "selected_checks": [
                {
                    "name": "combo",
                    "required": True,
                    "executions": [
                        {"argv": ["first-command"], "redact_argv_indices": []},
                        {"argv": ["second-command"], "redact_argv_indices": []},
                    ],
                    "execution_item_ids": [
                        first_item,
                        second_item,
                    ],
                    "commands": [],
                    "command_item_ids": [],
                }
            ],
        },
        "activity": {"verification": {"records": [duplicate, dict(duplicate)]}},
    }

    state = _build_session_verification_state(payload, final=True)

    assert state["status"] == "missing"
    assert state["completed_required_items"] == [first_item]
    assert "completed_required_checks" not in state


def test_fresh_verification_run_ignores_stale_item_records() -> None:
    base_item = verification_item_id("required", "execution", 0)
    current_item = f"verification-run-v1:current:{base_item}"
    payload = {
        "verification_plan": {
            "available": True,
            "verification_run_id": "current",
            "selected_checks": [
                {
                    "name": "required",
                    "required": True,
                    "executions": [
                        {"argv": ["python3"], "redact_argv_indices": []}
                    ],
                    "execution_item_ids": [current_item],
                    "commands": [],
                    "command_item_ids": [],
                }
            ],
        },
        "activity": {
            "verification": {
                "records": [
                    {
                        "status": "passed",
                        "check_name": "required",
                        "item_id": f"verification-run-v1:old:{base_item}",
                    }
                ]
            }
        },
    }

    pending = _build_session_verification_state(payload, final=False)
    payload["activity"]["verification"]["records"].append(
        {
            "status": "passed",
            "check_name": "required",
            "item_id": current_item,
        }
    )
    passed = _build_session_verification_state(payload, final=False)

    assert pending["status"] == "pending"
    assert passed["status"] == "passed"


def test_verification_coverage_retains_more_than_ten_required_items() -> None:
    item_ids = [verification_item_id("required", "execution", index) for index in range(11)]
    payload = {
        "verification_plan": {
            "available": True,
            "selected_checks": [
                {
                    "name": "required",
                    "required": True,
                    "executions": [
                        {"argv": ["true", str(index)], "redact_argv_indices": []}
                        for index in range(11)
                    ],
                    "execution_item_ids": item_ids,
                    "commands": [],
                    "command_item_ids": [],
                }
            ],
        },
        "activity": {
            "verification": {
                "records": [
                    {
                        "status": "passed",
                        "check_name": "required",
                        "item_id": item_id,
                    }
                    for item_id in item_ids
                ]
            }
        },
    }

    state = _build_session_verification_state(payload, final=True)

    assert state["status"] == "passed"
    assert state["record_count"] == 11
    assert state["completed_required_items"] == sorted(item_ids)


def test_empty_available_plan_preserves_workflow_verification_requirement() -> None:
    payload = {
        "verification_plan": {"available": True, "selected_checks": []},
        "pack": {"workflow": "edit_fast", "tool_profile": "edit_and_verify"},
        "prompt": {"model_family": "codex"},
        "activity": {"verification": {"records": []}},
    }

    state = _build_session_verification_state(payload, final=True)

    assert state["required"] is True
    assert state["status"] == "missing"


def test_current_optional_failure_keeps_session_failed() -> None:
    required_item = verification_item_id("required", "execution", 0)
    optional_item = verification_item_id("optional", "execution", 0)
    payload = {
        "verification_plan": {
            "available": True,
            "selected_checks": [
                {
                    "name": "required",
                    "required": True,
                    "executions": [{"argv": ["true"], "redact_argv_indices": []}],
                    "execution_item_ids": [required_item],
                    "commands": [],
                    "command_item_ids": [],
                },
                {
                    "name": "optional",
                    "required": False,
                    "executions": [{"argv": ["false"], "redact_argv_indices": []}],
                    "execution_item_ids": [optional_item],
                    "commands": [],
                    "command_item_ids": [],
                },
            ],
        },
        "activity": {
            "verification": {
                "records": [
                    {
                        "status": "passed",
                        "check_name": "required",
                        "item_id": required_item,
                    },
                    {
                        "status": "failed",
                        "check_name": "optional",
                        "item_id": optional_item,
                    },
                ]
            }
        },
    }

    state = _build_session_verification_state(payload, final=True)

    assert state["status"] == "failed"


def test_all_optional_current_failure_is_not_hidden_as_not_required() -> None:
    optional_item = verification_item_id("optional", "execution", 0)
    payload = {
        "verification_plan": {
            "available": True,
            "selected_checks": [
                {
                    "name": "optional",
                    "required": False,
                    "executions": [{"argv": ["false"], "redact_argv_indices": []}],
                    "execution_item_ids": [optional_item],
                    "commands": [],
                    "command_item_ids": [],
                }
            ],
        },
        "activity": {
            "verification": {
                "records": [
                    {
                        "status": "failed",
                        "check_name": "optional",
                        "item_id": optional_item,
                    }
                ]
            }
        },
    }

    state = _build_session_verification_state(payload, final=True)

    assert state["status"] == "failed"


def test_session_hook_command_uses_payload_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, context_root = _make_manager(tmp_path)
    config_path = tmp_path / "afs.toml"
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps({"client": "codex", "session_id": "sess-2"}),
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    def fake_run_grounding_hooks(*, event, payload, config, profile_name=None):
        calls["event"] = event
        calls["payload"] = payload

    def fake_log_session_event(op, session_id=None, metadata=None, payload=None, context_root=None):
        calls["log"] = {
            "op": op,
            "session_id": session_id,
            "metadata": metadata,
            "payload": payload,
            "context_root": context_root,
        }
        return "evt-1"

    monkeypatch.setattr(
        cli_core_module,
        "_load_manager_context_and_config_path",
        lambda _args: (manager, context_root, config_path),
    )
    monkeypatch.setattr("afs.grounding_hooks.run_grounding_hooks", fake_run_grounding_hooks)
    monkeypatch.setattr("afs.history.log_session_event", fake_log_session_event)

    rc = session_hook_command(
        Namespace(
            event="session_end",
            client=None,
            session_id=None,
            cwd=str(tmp_path),
            payload_file=str(payload_path),
            exit_code=3,
            reason="client_exit",
            json=True,
            config=None,
            path=None,
            context_root=context_root,
            context_dir=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["event"] == "session_end"
    assert calls["event"] == "session_end"
    hook_payload = calls["payload"]
    assert hook_payload["client"] == "codex"
    assert hook_payload["session_id"] == "sess-2"
    assert hook_payload["exit_code"] == 3
    assert hook_payload["reason"] == "client_exit"
    assert hook_payload["context_path"] == str(context_root)
    assert calls["log"]["op"] == "session_end"
    assert calls["log"]["payload"]["reason"] == "client_exit"


def test_session_event_command_updates_activity_payload(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, context_root = _make_manager(tmp_path)
    config_path = tmp_path / "afs.toml"

    monkeypatch.setattr(
        cli_core_module,
        "_load_manager_context_and_config_path",
        lambda _args: (manager, context_root, config_path),
    )

    prepare_args = Namespace(
        client="codex",
        session_id="sess-3",
        cwd=str(tmp_path),
        query="background agents harness",
        task="Improve the harness loop.",
        model="codex",
        workflow="edit_fast",
        tool_profile="edit_and_verify",
        pack_mode="focused",
        token_budget=400,
        include_content=False,
        max_query_results=4,
        max_embedding_results=2,
        skills_prompt="background agents harness",
        skills_top_k=5,
        no_session_pack=False,
        no_skills_match=False,
        no_write_artifacts=False,
        json=True,
        config=None,
        path=None,
        context_root=context_root,
        context_dir=None,
    )
    assert session_prepare_client_command(prepare_args) == 0
    prepared = json.loads(capsys.readouterr().out)
    payload_path = Path(prepared["artifact_paths"]["json"])

    calls: dict[str, object] = {}

    def fake_run_grounding_hooks(*, event, payload, config, profile_name=None):
        calls.setdefault("events", []).append((event, payload))

    def fake_log_session_event(op, session_id=None, metadata=None, payload=None, context_root=None):
        calls.setdefault("logs", []).append(
            {
                "op": op,
                "session_id": session_id,
                "metadata": metadata,
                "payload": payload,
                "context_root": context_root,
            }
        )
        return "evt-1"

    monkeypatch.setattr("afs.grounding_hooks.run_grounding_hooks", fake_run_grounding_hooks)
    monkeypatch.setattr("afs.history.log_session_event", fake_log_session_event)

    prompt_args = Namespace(
        event="user_prompt_submit",
        client=None,
        session_id=None,
        cwd=str(tmp_path),
        payload_file=str(payload_path),
        turn_id="turn-1",
        task_id=None,
        task_title=None,
        summary=None,
        status=None,
        reason=None,
        prompt="Investigate background agent progress reporting.",
        prompt_file=None,
        exit_code=None,
        json=True,
        config=None,
        path=None,
        context_root=context_root,
        context_dir=None,
    )
    assert session_event_command(prompt_args) == 0
    prompt_output = json.loads(capsys.readouterr().out)
    assert prompt_output["event"] == "user_prompt_submit"
    assert prompt_output["last_event"]["turn_id"] == "turn-1"

    task_args = Namespace(
        event="task_created",
        client=None,
        session_id=None,
        cwd=str(tmp_path),
        payload_file=str(payload_path),
        turn_id="turn-1",
        task_id="task-1",
        task_title="Investigate monitor sidecar",
        summary="Spawned a background monitor task.",
        status="queued",
        reason=None,
        verification_status=None,
        verification_command=None,
        prompt=None,
        prompt_file=None,
        exit_code=None,
        json=True,
        config=None,
        path=None,
        context_root=context_root,
        context_dir=None,
    )
    assert session_event_command(task_args) == 0
    task_output = json.loads(capsys.readouterr().out)
    assert task_output["active_tasks"][0]["task_id"] == "task-1"

    complete_args = Namespace(
        event="task_completed",
        client=None,
        session_id=None,
        cwd=str(tmp_path),
        payload_file=str(payload_path),
        turn_id="turn-1",
        task_id="task-1",
        task_title="Investigate monitor sidecar",
        summary="Monitor sidecar validated.",
        status=None,
        reason=None,
        verification_status="passed",
        verification_command="pytest -q tests/test_session_harness.py",
        prompt=None,
        prompt_file=None,
        exit_code=None,
        json=True,
        config=None,
        path=None,
        context_root=context_root,
        context_dir=None,
    )
    assert session_event_command(complete_args) == 0
    complete_output = json.loads(capsys.readouterr().out)
    assert complete_output["active_tasks"] == []

    updated_payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert updated_payload["activity"]["current_prompt"]["turn_id"] == "turn-1"
    assert updated_payload["activity"]["last_task"]["task_id"] == "task-1"
    assert updated_payload["activity"]["counters"]["user_prompt_submit"] == 1
    assert updated_payload["activity"]["counters"]["task_created"] == 1
    assert updated_payload["activity"]["counters"]["task_completed"] == 1
    assert updated_payload["activity"]["recent_events"][-1]["event"] == "task_completed"
    assert updated_payload["activity"]["verification"]["status"] == "passed"
    assert updated_payload["activity"]["verification"]["record_count"] == 1
    assert updated_payload["activity"]["verification"]["commands"] == [
        "pytest -q tests/test_session_harness.py"
    ]
    assert calls["events"][0][0] == "user_prompt_submit"
    assert calls["logs"][-1]["op"] == "task_completed"
    assert calls["logs"][-1]["metadata"]["workflow"] == "edit_fast"
    assert calls["logs"][-1]["metadata"]["tool_profile"] == "edit_and_verify"
    assert calls["logs"][-1]["metadata"]["outcome"] == "completed"
    assert calls["logs"][-1]["metadata"]["verification_status"] == "passed"
    assert calls["logs"][-1]["metadata"]["workflow_steps"] == [
        "user_prompt_submit",
        "task_created",
        "task_completed",
    ]
    assert "workflow_snapshot" in calls["logs"][-1]["payload"]
    assert "agentic-background" in calls["logs"][-1]["payload"]["workflow_snapshot"]["matched_skills"]


def test_session_hook_command_warns_when_required_verification_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, context_root = _make_manager(tmp_path)
    config_path = tmp_path / "afs.toml"

    monkeypatch.setattr(
        cli_core_module,
        "_load_manager_context_and_config_path",
        lambda _args: (manager, context_root, config_path),
    )

    assert session_prepare_client_command(
        Namespace(
            client="codex",
            session_id="sess-4",
            cwd=str(tmp_path),
            query="background agents harness",
            task="Improve the harness loop.",
            model="codex",
            workflow="edit_fast",
            tool_profile="edit_and_verify",
            pack_mode="focused",
            token_budget=700,
            include_content=False,
            max_query_results=4,
            max_embedding_results=2,
            skills_prompt="background agents harness",
            skills_top_k=5,
            no_session_pack=False,
            no_skills_match=False,
            no_write_artifacts=False,
            json=True,
            config=None,
            path=None,
            context_root=context_root,
            context_dir=None,
        )
    ) == 0
    prepared = json.loads(capsys.readouterr().out)
    payload_path = Path(prepared["artifact_paths"]["json"])

    rc = session_hook_command(
        Namespace(
            event="session_end",
            client=None,
            session_id=None,
            cwd=str(tmp_path),
            payload_file=str(payload_path),
            exit_code=0,
            reason="client_exit",
            verification_mode="warn",
            json=True,
            config=None,
            path=None,
            context_root=context_root,
            context_dir=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "warning"
    assert payload["verification"]["required"] is True
    assert payload["verification"]["status"] == "missing"


def test_session_hook_command_errors_in_strict_mode_when_verification_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, context_root = _make_manager(tmp_path)
    config_path = tmp_path / "afs.toml"

    monkeypatch.setattr(
        cli_core_module,
        "_load_manager_context_and_config_path",
        lambda _args: (manager, context_root, config_path),
    )

    assert session_prepare_client_command(
        Namespace(
            client="codex",
            session_id="sess-5",
            cwd=str(tmp_path),
            query="background agents harness",
            task="Improve the harness loop.",
            model="codex",
            workflow="edit_fast",
            tool_profile="edit_and_verify",
            pack_mode="focused",
            token_budget=700,
            include_content=False,
            max_query_results=4,
            max_embedding_results=2,
            skills_prompt="background agents harness",
            skills_top_k=5,
            no_session_pack=False,
            no_skills_match=False,
            no_write_artifacts=False,
            json=True,
            config=None,
            path=None,
            context_root=context_root,
            context_dir=None,
        )
    ) == 0
    prepared = json.loads(capsys.readouterr().out)
    payload_path = Path(prepared["artifact_paths"]["json"])

    rc = session_hook_command(
        Namespace(
            event="session_end",
            client=None,
            session_id=None,
            cwd=str(tmp_path),
            payload_file=str(payload_path),
            exit_code=0,
            reason="client_exit",
            verification_mode="error",
            json=True,
            config=None,
            path=None,
            context_root=context_root,
            context_dir=None,
        )
    )

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["verification"]["status"] == "missing"


@pytest.mark.parametrize(
    ("verification_mode", "expected_rc", "expected_status"),
    [("warn", 0, "warning"), ("error", 2, "error")],
)
@pytest.mark.parametrize("preflight_only", [False, True], ids=["check", "preflight"])
def test_session_hook_gates_blocked_required_verification(
    tmp_path: Path,
    monkeypatch,
    capsys,
    verification_mode: str,
    expected_rc: int,
    expected_status: str,
    preflight_only: bool,
) -> None:
    manager, context_root = _make_manager(tmp_path)
    config_path = tmp_path / "afs.toml"
    monkeypatch.setattr(
        cli_core_module,
        "_load_manager_context_and_config_path",
        lambda _args: (manager, context_root, config_path),
    )
    assert session_prepare_client_command(
        Namespace(
            client="codex",
            session_id=f"sess-blocked-{verification_mode}-{preflight_only}",
            cwd=str(tmp_path),
            query="background agents harness",
            task="Exercise the blocked verification gate.",
            model="codex",
            workflow="edit_fast",
            tool_profile="edit_and_verify",
            pack_mode="focused",
            token_budget=700,
            include_content=False,
            max_query_results=4,
            max_embedding_results=2,
            skills_prompt="background agents harness",
            skills_top_k=5,
            no_session_pack=False,
            no_skills_match=False,
            no_write_artifacts=False,
            json=True,
            config=None,
            path=None,
            context_root=context_root,
            context_dir=None,
        )
    ) == 0
    prepared = json.loads(capsys.readouterr().out)
    payload_path = Path(prepared["artifact_paths"]["json"])
    session_payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if preflight_only:
        item_id = "verification-run-v1:test:preflight"
        check_name = "verification-preflight"
        session_payload["verification_plan"] = {
            "available": True,
            "selected_checks": [],
            "preflight_required": True,
            "preflight_item_id": item_id,
            "verification_run_id": "test",
        }
    else:
        check = session_payload["verification_plan"]["selected_checks"][0]
        item_id = check["execution_item_ids"][0]
        check_name = check["name"]
    session_payload["activity"]["verification"]["records"] = [
        {
            "status": "blocked",
            "check_name": check_name,
            "item_id": item_id,
        }
    ]
    payload_path.write_text(json.dumps(session_payload, indent=2), encoding="utf-8")

    rc = session_hook_command(
        Namespace(
            event="session_end",
            client=None,
            session_id=None,
            cwd=str(tmp_path),
            payload_file=str(payload_path),
            exit_code=0,
            reason="client_exit",
            verification_mode=verification_mode,
            json=True,
            config=None,
            path=None,
            context_root=context_root,
            context_dir=None,
        )
    )

    output = json.loads(capsys.readouterr().out)
    assert rc == expected_rc
    assert output["status"] == expected_status
    assert output["verification"]["required"] is True
    assert output["verification"]["status"] == "blocked"
