from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

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
        "focus = [\"order findings by severity\"]\n"
        "risk_categories = [\"public-api\"]\n"
        "\n"
        "[[review.path_risks]]\n"
        "name = \"public-api\"\n"
        "paths = [\"src/**/*.py\"]\n"
        "message = \"Public Python surfaces need compatibility review.\"\n"
        "\n"
        "[design]\n"
        "constraints = [\"preserve API compatibility\"]\n"
        "\n"
        "[planning]\n"
        "principles = [\"keep plans reversible\"]\n"
        "\n"
        "[[anti_patterns]]\n"
        "name = \"debug-print\"\n"
        "paths = [\"src/**/*.py\"]\n"
        "pattern = \"print(\"\n"
        "message = \"Avoid debug prints in checked-in code.\"\n",
        encoding="utf-8",
    )
    (tmp_path / "afs.toml").write_text(
        "[verification]\n"
        "default_profile = \"repo\"\n\n"
        "[verification.profiles.repo]\n"
        "\n"
        "[[verification.profiles.repo.checks]]\n"
        "name = \"python\"\n"
        "paths = [\"src/**/*.py\"]\n"
        "commands = [\"ruff check src\"]\n"
        "skills = [\"python-quality\"]\n"
        "workflows = [\"edit_fast\"]\n"
        "tool_profiles = [\"edit_and_verify\"]\n",
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


def test_verify_run_command_executes_selected_checks(
    tmp_path: Path,
    capsys,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "afs.toml").write_text(
        "[verification]\n"
        "default_profile = \"repo\"\n\n"
        "[verification.profiles.repo]\n"
        "\n"
        "[[verification.profiles.repo.checks]]\n"
        "name = \"python\"\n"
        "paths = [\"src/**/*.py\"]\n"
        "commands = [\"python3 -c \\\"raise SystemExit(0)\\\"\"]\n"
        "skills = [\"python-quality\"]\n"
        "workflows = [\"edit_fast\"]\n"
        "tool_profiles = [\"edit_and_verify\"]\n",
        encoding="utf-8",
    )

    rc = verify_run_command(_base_args(tmp_path))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "passed"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["status"] == "passed"
    assert payload["results"][0]["command"] == 'python3 -c "raise SystemExit(0)"'
