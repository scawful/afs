from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import afs.cli as cli
import afs.cli.layout as layout_cli
from afs.cli import build_parser
from afs.context_layout import load_migration_plan


def _run(argv: list[str]) -> int:
    args = build_parser(argv).parse_args(argv)
    return args.func(args)


def _legacy_context(tmp_path: Path) -> Path:
    source = tmp_path / "context-v1"
    (source / "memory").mkdir(parents=True)
    (source / "memory" / "note.md").write_text("remember\n", encoding="utf-8")
    return source


def _write_plan(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = _legacy_context(tmp_path)
    destination = tmp_path / "context-v2"
    plan_path = tmp_path / "migration-plan.json"
    assert (
        _run(
            [
                "layout",
                "plan",
                "--context-root",
                str(source),
                "--destination-root",
                str(destination),
                "--output",
                str(plan_path),
                "--json",
            ]
        )
        == 0
    )
    return source, destination, plan_path


def test_mapping_file_resolves_only_explicit_unknown_top_level_entry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _legacy_context(tmp_path)
    (source / "legacy-guide.md").write_text("legacy\n", encoding="utf-8")
    destination = tmp_path / "context-v2"
    mapping_path = tmp_path / "mappings.json"
    mapping_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mappings": {"legacy-guide.md": "knowledge/common/imported/legacy-guide.md"},
            }
        ),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.json"

    assert (
        _run(
            [
                "layout",
                "plan",
                "--context-root",
                str(source),
                "--destination-root",
                str(destination),
                "--mapping-file",
                str(mapping_path),
                "--output",
                str(plan_path),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["ready"] is True
    assert payload["plan"]["explicit_mappings"] == [
        {
            "source": "legacy-guide.md",
            "destination": "knowledge/common/imported/legacy-guide.md",
        }
    ]
    plan = load_migration_plan(plan_path)
    assert plan.ready is True
    assert not destination.exists()


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (
            '{"schema_version":1,"mappings":{"a":"knowledge/common/a"},'
            '"mappings":{"b":"knowledge/common/b"}}',
            "duplicate field 'mappings'",
        ),
        (
            '{"schema_version":1,"mappings":{},"unexpected":true}',
            "unknown: unexpected",
        ),
        (
            '{"schema_version":2,"mappings":{}}',
            "schema_version must be 1",
        ),
    ],
)
def test_mapping_file_parser_is_strict(
    document: str,
    message: str,
    tmp_path: Path,
) -> None:
    mapping_path = tmp_path / "mappings.json"
    mapping_path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(message)):
        layout_cli._read_mapping_file(mapping_path)


def test_layout_plan_json_reports_invalid_mapping_as_blocked(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _legacy_context(tmp_path)
    mapping_path = tmp_path / "mappings.json"
    mapping_path.write_text('{"schema_version":2,"mappings":{}}', encoding="utf-8")

    assert (
        _run(
            [
                "layout",
                "plan",
                "--context-root",
                str(source),
                "--destination-root",
                str(tmp_path / "context-v2"),
                "--mapping-file",
                str(mapping_path),
                "--json",
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert "schema_version must be 1" in payload["error"]


def test_layout_commands_do_not_write_cli_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _legacy_context(tmp_path)
    calls: list[tuple[list[str], int]] = []
    monkeypatch.setattr(
        cli,
        "log_cli_invocation",
        lambda argv, code: calls.append((list(argv), code)),
    )

    assert cli.main(["layout", "audit", "--context-root", str(source), "--json"]) == 1
    assert calls == []
    capsys.readouterr()

    plan_path = tmp_path / "plan.json"
    assert (
        cli.main(
            [
                "layout",
                "plan",
                "--context-root",
                str(source),
                "--destination-root",
                str(tmp_path / "context-v2"),
                "--output",
                str(plan_path),
                "--json",
            ]
        )
        == 0
    )
    assert calls == []


def test_layout_plan_refuses_outputs_inside_source_or_candidate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _legacy_context(tmp_path)
    destination = tmp_path / "context-v2"
    base = [
        "layout",
        "plan",
        "--context-root",
        str(source),
        "--destination-root",
        str(destination),
    ]

    assert _run([*base, "--output", str(source / "plan.json")]) == 2
    assert "outside the migration source root" in capsys.readouterr().err
    assert _run([*base, "--output", str(destination / "plan.json")]) == 2
    assert "outside the migration destination root" in capsys.readouterr().err
    mapping_in_source = source / "mappings.json"
    mapping_in_source.write_text(
        '{"schema_version":1,"mappings":{}}',
        encoding="utf-8",
    )
    assert _run([*base, "--mapping-file", str(mapping_in_source)]) == 2
    assert "mapping file must be outside" in capsys.readouterr().err
    shared = tmp_path / "shared.json"
    assert _run([*base, "--output", str(shared), "--rollback-output", str(shared)]) == 2
    assert "must use distinct paths" in capsys.readouterr().err

    assert not (source / "plan.json").exists()
    assert not destination.exists()


def test_layout_migrate_preview_is_read_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, destination, plan_path = _write_plan(tmp_path)
    capsys.readouterr()
    before = sorted(path.relative_to(source).as_posix() for path in source.rglob("*"))

    assert _run(["layout", "migrate", "--plan", str(plan_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready"
    assert payload["mode"] == "preview"
    assert payload["transaction_id"].startswith("layout_")
    assert not destination.exists()
    assert before == sorted(path.relative_to(source).as_posix() for path in source.rglob("*"))


def test_layout_migrate_preflight_filesystem_error_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _source, destination, plan_path = _write_plan(tmp_path)
    capsys.readouterr()
    monkeypatch.setattr(
        layout_cli,
        "preflight_migration",
        lambda _plan: (_ for _ in ()).throw(PermissionError("permission denied")),
    )

    assert _run(["layout", "migrate", "--plan", str(plan_path), "--json"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "blocked", "error": "permission denied"}
    assert not destination.exists()


def test_layout_migrate_apply_requires_rationale_and_controlling_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _source, destination, plan_path = _write_plan(tmp_path)
    capsys.readouterr()

    assert _run(["layout", "migrate", "--plan", str(plan_path), "--apply"]) == 2
    assert "rationale is required" in capsys.readouterr().err
    assert not destination.exists()

    assert (
        _run(
            [
                "layout",
                "migrate",
                "--plan",
                str(plan_path),
                "--apply",
                "--json",
            ]
        )
        == 2
    )
    json_error = json.loads(capsys.readouterr().out)
    assert json_error["status"] == "blocked"
    assert "rationale is required" in json_error["error"]

    monkeypatch.setattr(layout_cli, "_TTY_READER", lambda _prompt: None)
    assert (
        _run(
            [
                "layout",
                "migrate",
                "--plan",
                str(plan_path),
                "--apply",
                "--because",
                "Create a verified candidate for human review",
            ]
        )
        == 2
    )
    assert "interactive human confirmation" in capsys.readouterr().err
    assert not destination.exists()


def test_layout_migrate_apply_keeps_source_and_publishes_candidate_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, destination, plan_path = _write_plan(tmp_path)
    capsys.readouterr()
    source_before = (source / "memory" / "note.md").read_bytes()

    def confirm(prompt: str) -> str | None:
        match = re.search(r"Type '(layout_[a-f0-9]+)'", prompt)
        return match.group(1) if match else None

    monkeypatch.setattr(layout_cli, "_TTY_READER", confirm)
    assert (
        _run(
            [
                "layout",
                "migrate",
                "--plan",
                str(plan_path),
                "--apply",
                "--because",
                "Create a verified candidate for human review",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "applied"
    assert payload["source_unchanged"] is True
    assert (destination / ".afs" / "layout.toml").is_file()
    assert (destination / "memory" / "common" / "note.md").read_bytes() == source_before
    assert (source / "memory" / "note.md").read_bytes() == source_before

    assert _run(["layout", "migrate", "--plan", str(plan_path), "--json"]) == 0
    retried = json.loads(capsys.readouterr().out)
    assert retried["status"] == "already_applied"
    assert retried["mode"] == "verified_existing_candidate"

    (destination / "memory" / "common" / "note.md").write_text(
        "tampered\n",
        encoding="utf-8",
    )
    assert _run(["layout", "migrate", "--plan", str(plan_path), "--json"]) == 2
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "blocked"
    assert "candidate" in blocked["error"].lower()


def test_layout_migrate_rechecks_source_after_human_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, destination, plan_path = _write_plan(tmp_path)
    capsys.readouterr()

    def mutate_then_confirm(prompt: str) -> str | None:
        (source / "memory" / "note.md").write_text("changed\n", encoding="utf-8")
        match = re.search(r"Type '(layout_[a-f0-9]+)'", prompt)
        return match.group(1) if match else None

    monkeypatch.setattr(layout_cli, "_TTY_READER", mutate_then_confirm)
    assert (
        _run(
            [
                "layout",
                "migrate",
                "--plan",
                str(plan_path),
                "--apply",
                "--because",
                "Create a verified candidate for human review",
                "--json",
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert "source" in payload["error"].lower()
    assert not destination.exists()
