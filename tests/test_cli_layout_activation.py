from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

import afs.cli.layout as layout_cli
import afs.layout_activation as activation
import afs.layout_migration as migration
from afs.cli import build_parser
from afs.context_layout import build_migration_plan, write_manifest
from afs.human_provenance import _broker_for_reader


def _run(argv: list[str]) -> int:
    args = build_parser(argv).parse_args(argv)
    return args.func(args)


def _prepared(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "context"
    (source / "memory").mkdir(parents=True)
    (source / "memory" / "note.md").write_text("remember\n", encoding="utf-8")
    candidate = tmp_path / "context-v2"
    plan = build_migration_plan(source, candidate)
    plan_path = tmp_path / "plan.json"
    write_manifest(plan_path, plan)
    rationale = "Create the CLI activation candidate"
    scope = migration.layout_migration_authorization_scope(
        plan.plan_sha256,
        plan.transaction_id,
        rationale,
    )
    authorization = _broker_for_reader(lambda _prompt: "COPY").confirm_token(
        "COPY",
        "confirm",
        scope=scope,
    )
    assert authorization is not None
    migration.apply_migration(plan, rationale=rationale, authorization=authorization)
    return source, candidate, plan_path, tmp_path / "activation-state"


@pytest.fixture
def controlled_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AFS_CONTEXT_ROOT", raising=False)
    monkeypatch.setattr(activation, "_assert_quiescent", lambda *_roots: None)

    def exchange(left: Path, right: Path) -> None:
        temporary = left.parent / ".cli-test-exchange"
        os.replace(left, temporary)
        os.replace(right, left)
        os.replace(temporary, right)

    monkeypatch.setattr(activation, "_atomic_exchange", exchange)


def test_layout_activate_preview_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controlled_activation: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, _candidate, plan_path, state_dir = _prepared(tmp_path)
    monkeypatch.setattr(layout_cli, "_configured_context_root", lambda _args: source)

    assert (
        _run(
            [
                "layout",
                "activate",
                "--plan",
                str(plan_path),
                "--state-dir",
                str(state_dir),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "preview"
    assert payload["status"] == "ready"
    assert payload["active_root"] == str(source)
    assert not state_dir.exists()


def test_layout_activate_requires_rationale_and_controlling_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controlled_activation: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, candidate, plan_path, state_dir = _prepared(tmp_path)
    monkeypatch.setattr(layout_cli, "_configured_context_root", lambda _args: source)
    base = [
        "layout",
        "activate",
        "--plan",
        str(plan_path),
        "--state-dir",
        str(state_dir),
        "--apply",
        "--json",
    ]

    assert _run(base) == 2
    assert "rationale" in json.loads(capsys.readouterr().out)["error"].lower()
    monkeypatch.setattr(layout_cli, "_TTY_READER", lambda _prompt: None)
    assert _run([*base, "--because", "Activate after maintenance"]) == 2
    assert "controlling terminal" in json.loads(capsys.readouterr().out)["error"].lower()
    assert source.exists() and candidate.exists()
    assert not state_dir.exists()


def test_layout_activate_then_rollback_with_separate_human_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controlled_activation: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, candidate, plan_path, state_dir = _prepared(tmp_path)
    monkeypatch.setattr(layout_cli, "_configured_context_root", lambda _args: source)
    prompts: list[str] = []

    def confirm(prompt: str) -> str:
        prompts.append(prompt)
        match = re.search(r"Type '([^']+)'", prompt)
        assert match is not None
        return match.group(1)

    monkeypatch.setattr(layout_cli, "_TTY_READER", confirm)
    assert (
        _run(
            [
                "layout",
                "activate",
                "--plan",
                str(plan_path),
                "--state-dir",
                str(state_dir),
                "--apply",
                "--because",
                "Activate the reviewed candidate",
                "--json",
            ]
        )
        == 0
    )
    activated = json.loads(capsys.readouterr().out)
    assert activated["status"] == "activated"
    assert (source / ".afs" / "layout.toml").exists()
    assert not (candidate / ".afs" / "layout.toml").exists()

    assert (
        _run(
            [
                "layout",
                "rollback",
                "--state-dir",
                str(state_dir),
                "--apply",
                "--because",
                "Restore v1 while preserving v2",
                "--json",
            ]
        )
        == 0
    )
    rolled_back = json.loads(capsys.readouterr().out)
    assert rolled_back["status"] == "rolled_back"
    assert not (source / ".afs" / "layout.toml").exists()
    assert (candidate / ".afs" / "layout.toml").exists()
    assert "layout activation" in prompts[0]
    assert "layout rollback" in prompts[1]
    assert "no data will be merged or deleted" in prompts[1]
