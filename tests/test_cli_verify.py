from __future__ import annotations

import json
import shlex
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import pytest

import afs.cli.core as cli_core_module
import afs.cli.verify as cli_verify_module
from afs.cli.verify import verify_plan_command, verify_run_command


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
    json.loads(capsys.readouterr().out)
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
