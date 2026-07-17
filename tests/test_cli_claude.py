from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import afs.cli.claude as claude_cli
from afs.cli.claude import claude_session_report_command, claude_setup_command
from afs.context_layout import scaffold_v2
from afs.manager import AFSManager
from afs.project_registry import ProjectRegistry
from afs.schema import AFSConfig, GeneralConfig


def test_claude_setup_writes_to_resolved_project_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_path = tmp_path / "project"
    project_path.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    context_root = tmp_path / "context"
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
            )
        )
    )
    manager.ensure(path=project_path, context_root=context_root)

    monkeypatch.setattr(claude_cli, "load_manager", lambda _config_path=None: manager)
    monkeypatch.chdir(elsewhere)

    exit_code = claude_setup_command(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=str(context_root),
            context_dir=None,
            force=False,
        )
    )

    assert exit_code == 0
    assert (project_path / ".claude" / "settings.json").exists()
    assert (project_path / "CLAUDE.md").exists()
    assert not (elsewhere / ".claude" / "settings.json").exists()
    assert not (elsewhere / "CLAUDE.md").exists()


def test_claude_setup_writes_user_settings_when_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_path = tmp_path / "project"
    project_path.mkdir()
    context_root = tmp_path / "context"
    settings_path = tmp_path / "home" / ".claude" / "settings.json"
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
            )
        )
    )
    manager.ensure(path=project_path, context_root=context_root)

    monkeypatch.setattr(claude_cli, "load_manager", lambda _config_path=None: manager)

    exit_code = claude_setup_command(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=str(context_root),
            context_dir=None,
            scope="user",
            settings_path=str(settings_path),
            force=False,
        )
    )

    assert exit_code == 0
    assert settings_path.exists()
    settings = settings_path.read_text(encoding="utf-8")
    assert "AFS_CONFIG_PATH" not in settings
    assert "AFS_CONTEXT_ROOT" not in settings
    assert not (project_path / ".claude" / "settings.json").exists()
    assert not (project_path / "CLAUDE.md").exists()


def _session_report_args(
    *,
    project: Path,
    context_root: Path,
    scratchpad_path: str | None = None,
    common: bool = False,
) -> Namespace:
    return Namespace(
        session="42cc501a-6d5b-4087-af16-91152a81b866",
        claude_root=None,
        no_subagents=False,
        max_subagent_chars=4000,
        json=False,
        output=None,
        write_scratchpad=True,
        scratchpad_path=scratchpad_path,
        common=common,
        force=False,
        config=None,
        path=str(project),
        context_root=str(context_root),
        context_dir=None,
    )


def _stub_session_report(monkeypatch) -> None:  # noqa: ANN001
    report = SimpleNamespace(
        paths=SimpleNamespace(session_id="42cc501a-6d5b-4087-af16-91152a81b866")
    )
    monkeypatch.setattr(claude_cli, "build_session_report", lambda *args, **kwargs: report)
    monkeypatch.setattr(
        claude_cli,
        "render_session_report_markdown",
        lambda _report: "# Claude Session\n\nsummary\n",
    )


def test_claude_session_report_allocates_unique_v2_project_drafts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / ".context"
    project = tmp_path / "alpha"
    project.mkdir()
    scaffold_v2(context_root)
    record = ProjectRegistry(context_root).register(project)
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    _stub_session_report(monkeypatch)
    monkeypatch.setattr(claude_cli, "load_manager", lambda _path=None: manager)
    monkeypatch.setattr(
        "afs.artifacts._normalize_created_at",
        lambda _value: "2026-07-17T08:00:00.000000Z",
    )
    args = _session_report_args(project=project, context_root=context_root)

    assert claude_session_report_command(args) == 0
    assert claude_session_report_command(args) == 0

    project_notes = (
        context_root / "scratchpad" / "projects" / record.project_id / "notes"
    )
    reports = sorted(project_notes.glob("*.md"))
    assert len(reports) == 2
    assert reports[0].name != reports[1].name
    assert all("claude-session-42cc501a" in path.name for path in reports)
    assert all(path.read_text(encoding="utf-8").endswith("# Claude Session\n\nsummary\n") for path in reports)
    assert not (context_root / "scratchpad" / "common").exists()


def test_claude_session_report_isolates_v2_projects_and_requires_explicit_common(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    _stub_session_report(monkeypatch)
    monkeypatch.setattr(claude_cli, "load_manager", lambda _path=None: manager)

    assert claude_session_report_command(
        _session_report_args(project=alpha, context_root=context_root)
    ) == 0
    assert claude_session_report_command(
        _session_report_args(project=beta, context_root=context_root)
    ) == 0

    alpha_notes = context_root / "scratchpad" / "projects" / alpha_record.project_id / "notes"
    beta_notes = context_root / "scratchpad" / "projects" / beta_record.project_id / "notes"
    assert len(list(alpha_notes.glob("*.md"))) == 1
    assert len(list(beta_notes.glob("*.md"))) == 1
    assert not (context_root / "scratchpad" / "common").exists()

    assert claude_session_report_command(
        _session_report_args(project=alpha, context_root=context_root, common=True)
    ) == 0
    assert len(list((context_root / "scratchpad" / "common" / "notes").glob("*.md"))) == 1


def test_claude_session_report_rejects_explicit_v2_scratchpad_path(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    context_root = tmp_path / ".context"
    project = tmp_path / "alpha"
    project.mkdir()
    scaffold_v2(context_root)
    ProjectRegistry(context_root).register(project)
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    _stub_session_report(monkeypatch)
    monkeypatch.setattr(claude_cli, "load_manager", lambda _path=None: manager)

    result = claude_session_report_command(
        _session_report_args(
            project=project,
            context_root=context_root,
            scratchpad_path="custom/report.md",
        )
    )

    assert result == 2
    assert "not supported for v2 immutable drafts" in capsys.readouterr().out
    assert not any((context_root / "scratchpad").rglob("*.md"))


def test_claude_session_report_preserves_v1_scratchpad_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / ".context"
    project = tmp_path / "alpha"
    project.mkdir()
    (context_root / "scratchpad").mkdir(parents=True)
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    _stub_session_report(monkeypatch)
    monkeypatch.setattr(claude_cli, "load_manager", lambda _path=None: manager)

    result = claude_session_report_command(
        _session_report_args(
            project=project,
            context_root=context_root,
            scratchpad_path="custom/report.md",
        )
    )

    assert result == 0
    assert (context_root / "scratchpad" / "custom" / "report.md").read_text(
        encoding="utf-8"
    ) == "# Claude Session\n\nsummary\n"
