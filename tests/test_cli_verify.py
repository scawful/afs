from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import afs.cli.core as cli_core_module
import afs.cli.verify as cli_verify_module
import afs.verification as verification_module
from afs.cli.verify import verify_plan_command, verify_run_command
from afs.schema import AFSConfig, VerificationExecutionConfig
from afs.verification import build_verification_plan, run_verification_execution


def test_changed_path_discovery_preserves_first_unstaged_filename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        verification_module,
        "discover_git_repo_root",
        lambda _start_dir=None: tmp_path,
    )
    monkeypatch.setattr(
        verification_module,
        "_run_git",
        lambda _root, *_args: " M ROADMAP.md\x00?? docs/new.md\x00",
    )

    assert verification_module.discover_changed_paths(tmp_path) == [
        "ROADMAP.md",
        "docs/new.md",
    ]


def test_changed_path_discovery_includes_both_sides_of_rename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        verification_module,
        "discover_git_repo_root",
        lambda _start_dir=None: tmp_path,
    )
    monkeypatch.setattr(
        verification_module,
        "_run_git",
        lambda _root, *_args: "R  docs/a.txt\x00src/a.py\x00",
    )

    assert verification_module.discover_changed_paths(tmp_path) == [
        "docs/a.txt",
        "src/a.py",
    ]


def test_git_discovery_rejects_truncated_broker_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(verification_module.shutil, "which", lambda _name: "/usr/bin/git")
    monkeypatch.setattr(
        verification_module,
        "execute_checked",
        lambda *_args, **_kwargs: SimpleNamespace(
            outcome="completed",
            stdout=" M ROADMAP.md\x00",
            stderr="",
            stdout_truncated=True,
            stderr_truncated=False,
        ),
    )

    with pytest.raises(
        verification_module.VerificationDiscoveryError,
        match="exceeded the execution broker limit",
    ):
        verification_module._run_git(tmp_path, "status", "--porcelain=v1", "-z")


def test_verify_run_blocks_when_changed_scope_discovery_is_incomplete(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    marker = tmp_path / "must-not-run"
    script = f"open({str(marker)!r}, 'w').close()"
    (tmp_path / "afs.toml").write_text(
        "[verification]\n"
        'default_profile = "repo"\n\n'
        "[verification.profiles.repo]\n\n"
        "[[verification.profiles.repo.checks]]\n"
        'name = "required"\n'
        "required = true\n"
        'paths = ["**/*.py"]\n\n'
        "[[verification.profiles.repo.checks.executions]]\n"
        f"argv = {json.dumps([sys.executable, '-c', script])}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        verification_module,
        "discover_git_repo_root",
        lambda _start_dir=None: tmp_path,
    )

    def fail_discovery(_start_dir=None):
        raise verification_module.VerificationDiscoveryError("simulated truncation")

    monkeypatch.setattr(verification_module, "discover_changed_paths", fail_discovery)

    rc = verify_run_command(_base_args(tmp_path, changed_path=[]))

    assert rc == 2
    assert not marker.exists()
    output = json.loads(capsys.readouterr().out)
    assert output["outcome"] == "blocked"
    assert output["verification_plan"]["discovery_complete"] is False


def test_verify_run_rediscovers_changes_after_cached_session_plan(
    tmp_path: Path,
    capsys,
) -> None:
    git = shutil.which("git")
    if not git:
        pytest.skip("git is unavailable")
    subprocess.run(
        [git, "init", "-q", str(tmp_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    (tmp_path / "afs.toml").write_text(
        "[verification]\n"
        'default_profile = "repo"\n\n'
        "[verification.profiles.repo]\n\n"
        "[[verification.profiles.repo.checks]]\n"
        'name = "required-python"\n'
        "required = true\n"
        'paths = ["src/**/*.py"]\n\n'
        "[[verification.profiles.repo.checks.executions]]\n"
        f"argv = {json.dumps([sys.executable, '-c', 'raise SystemExit(1)'])}\n",
        encoding="utf-8",
    )
    payload_path = tmp_path / "session.json"
    payload_path.write_text(
        json.dumps(
            {
                "cwd": str(tmp_path),
                "verification_plan": {
                    "changed_paths": [],
                    "discovery_complete": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("changed = True\n", encoding="utf-8")

    rc = verify_run_command(
        _base_args(
            tmp_path,
            payload_file=str(payload_path),
            changed_path=[],
            skill=[],
        )
    )

    assert rc == 1
    output = json.loads(capsys.readouterr().out)
    assert output["outcome"] == "failed"
    assert output["verification_plan"]["changed_paths"] == [
        "afs.toml",
        "session.json",
        "src/app.py",
    ]
    assert output["results"][0]["check_name"] == "required-python"


def test_verify_run_refreshes_session_payload_to_live_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload_path = tmp_path / "session.json"
    payload_path.write_text(
        json.dumps(
            {
                "context_path": str(tmp_path / ".context"),
                "client": "codex",
                "session_id": "session-1",
                "activity": {
                    "verification": {
                        "records": [{"status": "passed", "item_id": "stale-item"}]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    bundle = {
        "verification_plan": {
            "available": True,
            "selected_checks": [
                {
                    "name": "live-required",
                    "required": True,
                    "executions": [{"argv": ["python3"], "redact_argv_indices": []}],
                    "execution_item_ids": ["live-item"],
                    "commands": [],
                    "command_item_ids": [],
                }
            ],
        },
        "repo_policy": {},
        "structured_guidance": {},
    }
    captured: dict = {}

    def capture_payload(_manager, _context_path, *, payload, **_kwargs):
        captured.update(json.loads(json.dumps(payload)))
        return {"json": str(payload_path)}

    monkeypatch.setattr(cli_verify_module, "load_manager", lambda _path: object())
    monkeypatch.setattr(
        "afs.session_harness.write_client_session_payload_artifact",
        capture_payload,
    )

    cli_verify_module._refresh_session_verification_plan(
        meta={
            "context_path": str(tmp_path / ".context"),
            "client": "codex",
            "session_id": "session-1",
            "config_path": "",
        },
        bundle=bundle,
        payload_path=payload_path,
    )

    assert captured["verification_plan"]["selected_checks"][0]["name"] == "live-required"
    assert captured["activity"]["verification"]["status"] == "pending"


def test_blocked_preflight_replaces_stale_passed_session_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload_path = tmp_path / "session.json"
    payload_path.write_text(
        json.dumps(
            {
                "context_path": str(tmp_path / ".context"),
                "client": "codex",
                "session_id": "session-1",
                "verification_plan": {"available": True},
                "activity": {
                    "verification": {
                        "status": "passed",
                        "records": [{"status": "passed", "item_id": "stale-item"}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    bundle = {
        "verification_plan": {
            "available": True,
            "selected_checks": [],
        },
        "repo_policy": {},
        "structured_guidance": {},
    }
    captured: dict = {}

    def capture_payload(_manager, _context_path, *, payload, **_kwargs):
        captured.update(json.loads(json.dumps(payload)))
        return {"json": str(payload_path)}

    monkeypatch.setattr(cli_verify_module, "load_manager", lambda _path: object())
    monkeypatch.setattr(
        "afs.session_harness.write_client_session_payload_artifact",
        capture_payload,
    )

    cli_verify_module._begin_verification_run(
        meta={
            "context_path": str(tmp_path / ".context"),
            "client": "codex",
            "session_id": "session-1",
            "config_path": "",
        },
        bundle=bundle,
        payload_path=payload_path,
        blocked_reason="verification discovery failed",
    )

    plan = captured["verification_plan"]
    state = captured["activity"]["verification"]
    assert plan["preflight_required"] is True
    assert state["required"] is True
    assert state["status"] == "blocked"
    assert state["required_items"] == [plan["preflight_item_id"]]


def test_invalid_check_selection_does_not_mutate_session_payload(
    tmp_path: Path,
    capsys,
) -> None:
    (tmp_path / "afs.toml").write_text(
        "[verification]\n"
        'default_profile = "repo"\n\n'
        "[verification.profiles.repo]\n\n"
        "[[verification.profiles.repo.checks]]\n"
        'name = "required"\n'
        "required = true\n"
        'paths = ["src/**/*.py"]\n\n'
        "[[verification.profiles.repo.checks.executions]]\n"
        f"argv = {json.dumps([sys.executable, '-c', 'pass'])}\n",
        encoding="utf-8",
    )
    payload_path = tmp_path / "session.json"
    payload_path.write_text(
        json.dumps(
            {
                "context_path": str(tmp_path / ".context"),
                "client": "codex",
                "session_id": "session-1",
                "verification_plan": {"verification_run_id": "already-passed"},
                "activity": {"verification": {"status": "passed"}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    before = payload_path.read_bytes()

    rc = verify_run_command(
        _base_args(
            tmp_path,
            payload_file=str(payload_path),
            check=["missing"],
        )
    )

    assert rc == 2
    assert payload_path.read_bytes() == before
    assert json.loads(capsys.readouterr().out)["outcome"] == "blocked"


def _base_args(tmp_path: Path, **overrides) -> Namespace:
    payload = {
        "config": str(tmp_path / "afs.toml"),
        "cwd": str(tmp_path),
        "payload_file": None,
        "workflow": "edit_fast",
        "tool_profile": "edit_and_verify",
        "model": "gemini",
        "verification_profile": "repo",
        "repo_policy_file": None,
        "changed_path": ["src/app.py"],
        "skill": ["python-quality"],
        "json": True,
        "check": [],
        "max_digest_items": 5,
        "continue_on_fail": False,
        "require_checks": False,
        "allow_legacy_shell": False,
    }
    payload.update(overrides)
    return Namespace(**payload)


def test_verify_plan_command_includes_policy_and_structured_guidance(
    tmp_path: Path,
    capsys,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('debug')\n", encoding="utf-8")
    (tmp_path / ".afs").mkdir()
    (tmp_path / ".afs" / "policy.toml").write_text(
        "[review]\n"
        'focus = ["order findings by severity"]\n'
        'risk_categories = ["public-api"]\n'
        "\n"
        "[[review.path_risks]]\n"
        'name = "public-api"\n'
        'paths = ["src/**/*.py"]\n'
        'message = "Public Python surfaces need compatibility review."\n'
        "\n"
        "[design]\n"
        'constraints = ["preserve API compatibility"]\n'
        "\n"
        "[planning]\n"
        'principles = ["keep plans reversible"]\n'
        "\n"
        "[[anti_patterns]]\n"
        'name = "debug-print"\n'
        'paths = ["src/**/*.py"]\n'
        'pattern = "print("\n'
        'message = "Avoid debug prints in checked-in code."\n',
        encoding="utf-8",
    )
    (tmp_path / "afs.toml").write_text(
        "[verification]\n"
        'default_profile = "repo"\n\n'
        "[verification.profiles.repo]\n"
        "\n"
        "[[verification.profiles.repo.checks]]\n"
        'name = "python"\n'
        'paths = ["src/**/*.py"]\n'
        'commands = ["ruff check src"]\n'
        'skills = ["python-quality"]\n'
        'workflows = ["edit_fast"]\n'
        'tool_profiles = ["edit_and_verify"]\n',
        encoding="utf-8",
    )

    rc = verify_plan_command(_base_args(tmp_path))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verification_plan"]["profile"] == "repo"
    assert payload["verification_plan"]["selected_checks"][0]["name"] == "python"
    assert payload["repo_policy"]["matched_risks"][0]["name"] == "public-api"
    assert payload["repo_policy"]["anti_pattern_hits"][0]["name"] == "debug-print"
    assert payload["structured_guidance"]["recommended_schema"] == "design-brief"
    assert payload["structured_guidance"]["followup_schema"] == "verification-summary"
    assert payload["verification_plan"]["legacy_command_count"] == 1
    assert payload["verification_plan"]["deprecations"] == [
        {
            "kind": "legacy_shell",
            "count": 1,
            "removal_version": "0.4.0",
            "opt_in_required": True,
            "message": (
                "Legacy verification commands are deprecated and require "
                "allow_legacy_shell or --allow-legacy-shell."
            ),
        }
    ]


def test_verify_plan_redacts_duplicate_execution_and_legacy_surfaces(
    tmp_path: Path,
    capsys,
) -> None:
    _prepare_changed_python_file(tmp_path)
    (tmp_path / "afs.toml").write_text(
        "[verification]\n"
        'default_profile = "repo"\n\n'
        "[verification.profiles.repo]\n\n"
        "[[verification.profiles.repo.checks]]\n"
        'name = "python"\n'
        'paths = ["src/**/*.py"]\n'
        'commands = ["printf legacy-secret"]\n\n'
        "[[verification.profiles.repo.checks.executions]]\n"
        f"argv = {json.dumps([sys.executable, '-c', 'pass', 'argv-secret'])}\n"
        "redact_argv_indices = [3]\n"
        'env = { AFS_SECRET = "env-secret" }\n',
        encoding="utf-8",
    )

    rc = verify_plan_command(_base_args(tmp_path))

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    plan = payload["verification_plan"]
    check = plan["selected_checks"][0]
    execution = check["executions"][0]
    assert rc == 0
    assert execution["argv"][3] == "<redacted>"
    assert execution["env"] == {"AFS_SECRET": "<redacted>"}
    assert check["commands"] == ["<redacted legacy shell command>"]
    assert any("<redacted>" in item for item in plan["expected"])
    assert any("<redacted legacy shell command>" in item for item in plan["expected"])
    for secret in ("argv-secret", "env-secret", "legacy-secret"):
        assert secret not in captured.out


def test_verify_run_command_executes_selected_checks(
    tmp_path: Path,
    capsys,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "afs.toml").write_text(
        "[verification]\n"
        'default_profile = "repo"\n\n'
        "[verification.profiles.repo]\n"
        "\n"
        "[[verification.profiles.repo.checks]]\n"
        'name = "python"\n'
        'paths = ["src/**/*.py"]\n'
        'skills = ["python-quality"]\n'
        'workflows = ["edit_fast"]\n'
        'tool_profiles = ["edit_and_verify"]\n\n'
        "[[verification.profiles.repo.checks.executions]]\n"
        f"argv = {json.dumps([sys.executable, '-c', 'raise SystemExit(0)'])}\n",
        encoding="utf-8",
    )

    rc = verify_run_command(_base_args(tmp_path))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "passed"
    assert len(payload["results"]) == 1
    assert payload["verification_plan"]["structured_execution_count"] == 1
    assert payload["verification_plan"]["legacy_command_count"] == 0
    assert payload["verification_plan"]["deprecations"] == []
    assert payload["verification_plan"]["selected_checks"][0]["executions"][0][
        "argv"
    ] == [sys.executable, "-c", "raise SystemExit(0)"]
    assert payload["results"][0]["status"] == "passed"
    assert payload["results"][0]["command"] == shlex.join(
        [sys.executable, "-c", "raise SystemExit(0)"]
    )
    assert payload["results"][0]["request_hash"]
    assert payload["results"][0]["resolved_executable"] == str(Path(sys.executable).resolve())


def _write_check_config(
    tmp_path: Path,
    *,
    execution_argv: list[str] | None = None,
    command: str = "",
    required: bool = True,
    allow_legacy_shell: bool = False,
    timeout_seconds: float = 300.0,
    redact_argv_indices: list[int] | None = None,
) -> None:
    check_lines = [
        "[verification]",
        'default_profile = "repo"',
        f"allow_legacy_shell = {str(allow_legacy_shell).lower()}",
        "",
        "[verification.profiles.repo]",
        "",
        "[[verification.profiles.repo.checks]]",
        'name = "python"',
        'paths = ["src/**/*.py"]',
        f"required = {str(required).lower()}",
    ]
    if command:
        check_lines.append(f"commands = [{json.dumps(command)}]")
    check_lines.extend(
        [
            'skills = ["python-quality"]',
            'workflows = ["edit_fast"]',
            'tool_profiles = ["edit_and_verify"]',
        ]
    )
    if execution_argv is not None:
        check_lines.extend(
            [
                "",
                "[[verification.profiles.repo.checks.executions]]",
                f"argv = {json.dumps(execution_argv)}",
                f"timeout_seconds = {timeout_seconds}",
            ]
        )
        if redact_argv_indices is not None:
            check_lines.append(
                f"redact_argv_indices = {json.dumps(redact_argv_indices)}"
            )
    (tmp_path / "afs.toml").write_text("\n".join(check_lines) + "\n", encoding="utf-8")


def _prepare_changed_python_file(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "app.py").write_text("pass\n", encoding="utf-8")


def test_verify_run_structured_failure_returns_one(tmp_path: Path, capsys) -> None:
    _prepare_changed_python_file(tmp_path)
    _write_check_config(
        tmp_path,
        execution_argv=[sys.executable, "-c", "raise SystemExit(7)"],
    )

    rc = verify_run_command(_base_args(tmp_path))

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "failed"
    assert payload["results"][0]["status"] == "failed"
    assert payload["results"][0]["returncode"] == 7


def test_verify_run_structured_timeout_is_failure(tmp_path: Path, capsys) -> None:
    _prepare_changed_python_file(tmp_path)
    _write_check_config(
        tmp_path,
        execution_argv=[sys.executable, "-c", "import time; time.sleep(5)"],
        timeout_seconds=0.1,
    )

    rc = verify_run_command(_base_args(tmp_path))

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    result = payload["results"][0]
    assert payload["outcome"] == "failed"
    assert result["status"] == "failed"
    assert result["outcome"] == "timed_out"
    assert result["timed_out"] is True


def test_verify_run_blocks_required_check_without_runnable_execution(
    tmp_path: Path,
    capsys,
) -> None:
    _prepare_changed_python_file(tmp_path)
    _write_check_config(tmp_path)

    rc = verify_run_command(_base_args(tmp_path))

    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["outcome"] == "blocked"
    assert payload["results"] == []
    assert payload["message"] == (
        "Required verification checks have no runnable commands or executions: python."
    )


def test_verify_run_does_not_run_other_checks_when_required_check_is_not_runnable(
    tmp_path: Path,
    capsys,
) -> None:
    _prepare_changed_python_file(tmp_path)
    sentinel = tmp_path / "must-not-exist"
    execution_argv = [
        sys.executable,
        "-c",
        f"open({str(sentinel)!r}, 'w').close()",
    ]
    (tmp_path / "afs.toml").write_text(
        "[verification]\n"
        'default_profile = "repo"\n\n'
        "[verification.profiles.repo]\n\n"
        "[[verification.profiles.repo.checks]]\n"
        'name = "required-empty"\n'
        'paths = ["src/**/*.py"]\n'
        "required = true\n\n"
        "[[verification.profiles.repo.checks]]\n"
        'name = "optional-runnable"\n'
        'paths = ["src/**/*.py"]\n'
        "required = false\n\n"
        "[[verification.profiles.repo.checks.executions]]\n"
        f"argv = {json.dumps(execution_argv)}\n",
        encoding="utf-8",
    )

    rc = verify_run_command(_base_args(tmp_path))

    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["outcome"] == "blocked"
    assert "required-empty" in payload["message"]
    assert payload["results"] == []
    assert not sentinel.exists()


@pytest.mark.parametrize("check_names", [["missing"], ["python", "missing"]])
def test_verify_run_rejects_unmatched_explicit_check_names(
    tmp_path: Path,
    capsys,
    check_names: list[str],
) -> None:
    _prepare_changed_python_file(tmp_path)
    sentinel = tmp_path / "must-not-exist"
    _write_check_config(
        tmp_path,
        execution_argv=[sys.executable, "-c", f"open({str(sentinel)!r}, 'w').close()"],
    )

    rc = verify_run_command(_base_args(tmp_path, check=check_names))

    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["outcome"] == "blocked"
    assert payload["results"] == []
    assert "--check names did not match selected verification checks: missing" in payload[
        "message"
    ]
    assert not sentinel.exists()


def test_verify_run_blocks_required_legacy_command_by_default(
    tmp_path: Path,
    capsys,
) -> None:
    _prepare_changed_python_file(tmp_path)
    sentinel = tmp_path / "must-not-exist"
    _write_check_config(tmp_path, command=f"touch {shlex.quote(str(sentinel))}")

    rc = verify_run_command(_base_args(tmp_path))

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 2
    assert payload["outcome"] == "blocked"
    assert payload["results"][0]["status"] == "blocked"
    assert str(sentinel) not in payload["results"][0]["command"]
    assert "<redacted>" in payload["results"][0]["command"]
    assert "deprecated and blocked" in captured.err
    assert not sentinel.exists()


def test_verify_run_skips_optional_blocked_legacy_command(
    tmp_path: Path,
    capsys,
) -> None:
    _prepare_changed_python_file(tmp_path)
    sentinel = tmp_path / "must-not-exist"
    _write_check_config(
        tmp_path,
        command=f"touch {shlex.quote(str(sentinel))}",
        required=False,
    )

    rc = verify_run_command(_base_args(tmp_path))

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["outcome"] == "passed"
    assert payload["results"][0]["status"] == "skipped"
    assert "optional blocked check skipped" in captured.err
    assert not sentinel.exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="legacy shell requires bash")
@pytest.mark.parametrize(
    ("config_opt_in", "cli_opt_in"),
    [(True, False), (False, True)],
)
def test_verify_run_legacy_opt_in_warns_without_polluting_json(
    tmp_path: Path,
    capsys,
    config_opt_in: bool,
    cli_opt_in: bool,
) -> None:
    _prepare_changed_python_file(tmp_path)
    _write_check_config(
        tmp_path,
        command=f"{shlex.quote(sys.executable)} -c 'raise SystemExit(0)'",
        allow_legacy_shell=config_opt_in,
    )

    rc = verify_run_command(_base_args(tmp_path, allow_legacy_shell=cli_opt_in))

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["outcome"] == "passed"
    assert payload["results"][0]["status"] == "passed"
    assert "will be removed in AFS 0.4.0" in captured.err


def test_verify_run_records_execution_audit_metadata_without_output_or_env(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _prepare_changed_python_file(tmp_path)
    _write_check_config(
        tmp_path,
        execution_argv=[sys.executable, "-c", "print('not persisted')"],
        redact_argv_indices=[2],
    )
    with (tmp_path / "afs.toml").open("a", encoding="utf-8") as config_file:
        config_file.write('env = { AFS_SECRET = "session-secret" }\n')
    context_path = tmp_path / ".context"
    payload_path = tmp_path / "session-payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "client": "codex",
                "session_id": "session-1",
                "context_path": str(context_path),
            }
        ),
        encoding="utf-8",
    )
    captured_events: list[dict] = []

    def fake_emit_session_event(**kwargs):
        captured_events.append(kwargs)
        return {}

    monkeypatch.setattr(cli_verify_module, "load_manager", lambda _path: object())
    monkeypatch.setattr(cli_core_module, "_emit_session_event", fake_emit_session_event)

    rc = verify_run_command(_base_args(tmp_path, payload_file=str(payload_path)))

    assert rc == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    execution = output["verification_plan"]["selected_checks"][0]["executions"][0]
    assert execution["argv"][2] == "<redacted>"
    assert execution["env"] == {"AFS_SECRET": "<redacted>"}
    assert output["results"][0]["command"].endswith("-c '<redacted>'")
    assert "session-secret" not in captured.out
    seed = captured_events[0]["seed_payload"]
    assert seed["verification_request_hash"]
    assert seed["verification_resolved_executable"] == str(Path(sys.executable).resolve())
    assert seed["verification_redacted_argv"][2] == "<redacted>"
    assert seed["verification_duration_seconds"] >= 0
    assert seed["verification_timed_out"] is False
    assert seed["verification_stdout_truncated"] is False
    assert seed["verification_stderr_truncated"] is False
    assert seed["verification_digest"].keys() == {"kind", "line_count", "truncated"}
    assert captured_events[0]["summary"] == "python: passed"
    assert "not persisted" not in captured_events[0]["verification_command"]
    assert "<redacted>" in captured_events[0]["verification_command"]
    assert "stdout" not in seed
    assert "stderr" not in seed
    assert "env" not in seed


def test_verify_run_rejects_string_legacy_opt_in_without_execution(
    tmp_path: Path,
    capsys,
) -> None:
    _prepare_changed_python_file(tmp_path)
    sentinel = tmp_path / "must-not-exist"
    (tmp_path / "afs.toml").write_text(
        "[verification]\n"
        'default_profile = "repo"\n'
        'allow_legacy_shell = "false"\n\n'
        "[verification.profiles.repo]\n\n"
        "[[verification.profiles.repo.checks]]\n"
        'name = "python"\n'
        'paths = ["src/**/*.py"]\n'
        f"commands = [{json.dumps(f'touch {sentinel}')} ]\n",
        encoding="utf-8",
    )

    rc = verify_run_command(_base_args(tmp_path))

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 2
    assert payload["outcome"] == "blocked"
    assert "allow_legacy_shell must be a boolean" in payload["message"]
    assert captured.err == ""
    assert not sentinel.exists()


def test_invalid_structured_execution_error_redacts_designated_argv(
    tmp_path: Path,
) -> None:
    result = run_verification_execution(
        repo_root=tmp_path,
        check_name="invalid",
        execution=VerificationExecutionConfig(
            argv=[sys.executable, "-c", "pass", "session-secret"],
            timeout_seconds=-1,
            redact_argv_indices=[3],
        ),
    )

    assert result["status"] == "blocked"
    assert "session-secret" not in result["command"]
    assert "session-secret" not in result["audit_command"]
    assert "<redacted>" in result["command"]


def test_malformed_redaction_metadata_hides_entire_argv(tmp_path: Path) -> None:
    result = run_verification_execution(
        repo_root=tmp_path,
        check_name="invalid-redaction",
        execution={
            "argv": [sys.executable, "-c", "pass", "session-secret"],
            "redact_argv_indices": ["3"],
        },
    )

    assert result["status"] == "blocked"
    assert "session-secret" not in result["command"]
    assert sys.executable not in result["command"]
    assert result["command"].count("<redacted>") == 4


def test_builtin_ctest_uses_detected_build_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "CTestTestfile.cmake").write_text("# generated\n")
    monkeypatch.setattr(
        verification_module.shutil,
        "which",
        lambda executable: "/usr/bin/ctest" if executable == "ctest" else None,
    )

    plan = build_verification_plan(
        config=AFSConfig(),
        cwd=tmp_path,
        workflow="general",
        tool_profile="default",
        changed_paths=["src/main.cpp"],
    )

    check = next(
        item for item in plan["selected_checks"] if item["name"] == "builtin-cpp"
    )
    assert check["executions"] == [
        {
            "argv": ["ctest", "--output-on-failure"],
            "cwd": "build",
            "timeout_seconds": 300.0,
            "max_output_bytes": 1024 * 1024,
            "inherit_env": [],
            "env": {},
            "redact_argv_indices": [],
        }
    ]
