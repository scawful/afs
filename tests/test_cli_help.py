from __future__ import annotations

from pathlib import Path

import afs.cli as cli
from afs.cli import build_parser
from afs.cli._help import render_default_help, render_topic_help


def _write_base_config(path: Path) -> None:
    path.write_text(
        "[extensions]\n"
        "auto_discover = false\n\n"
        "[plugins]\n"
        "auto_discover = false\n",
        encoding="utf-8",
    )


def test_render_topic_help_suggests_leaf_command_typo(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "afs.toml"
    _write_base_config(config_path)
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("AFS_EXTENSION_DIRS", raising=False)
    monkeypatch.delenv("AFS_ENABLED_EXTENSIONS", raising=False)

    parser = build_parser()

    exit_code = render_topic_help(parser, ["session", "bootstap"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Unknown command: session bootstap" in captured.out
    assert "Closest matches:" in captured.out
    assert "session bootstrap" in captured.out


def test_render_topic_help_suggests_top_level_command_typo(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "afs.toml"
    _write_base_config(config_path)
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("AFS_EXTENSION_DIRS", raising=False)
    monkeypatch.delenv("AFS_ENABLED_EXTENSIONS", raising=False)

    parser = build_parser()

    exit_code = render_topic_help(parser, ["gmini"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Unknown command: gmini" in captured.out
    assert "Closest matches:" in captured.out
    assert "gemini" in captured.out


def test_render_default_help_mentions_context_query_and_index_rebuild(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "afs.toml"
    _write_base_config(config_path)
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("AFS_EXTENSION_DIRS", raising=False)
    monkeypatch.delenv("AFS_ENABLED_EXTENSIONS", raising=False)

    parser = build_parser()

    render_default_help(parser)

    out = capsys.readouterr().out
    assert "afs query" in out
    assert "afs context query" in out
    assert "afs index rebuild" in out


def test_render_topic_help_for_context_query_shows_examples_and_output_fields(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "afs.toml"
    _write_base_config(config_path)
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("AFS_EXTENSION_DIRS", raising=False)
    monkeypatch.delenv("AFS_ENABLED_EXTENSIONS", raising=False)

    parser = build_parser()

    exit_code = render_topic_help(parser, ["context", "query"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Examples:" in out
    assert "afs context query" in out
    assert "afs query sqlite" in out
    assert "Output fields:" in out
    assert "index_rebuild" in out


def test_main_emits_zsh_completion_source(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "afs.toml"
    _write_base_config(config_path)
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("_AFS_COMPLETE", "zsh_source")
    monkeypatch.delenv("AFS_EXTENSION_DIRS", raising=False)
    monkeypatch.delenv("AFS_ENABLED_EXTENSIONS", raising=False)
    monkeypatch.setattr(cli, "log_cli_invocation", lambda *_args, **_kwargs: None)

    exit_code = cli.main([])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "#compdef afs" in out
    assert "typeset -ga _afs_cmds_root" in out
    assert "context:Manage project contexts." in out
    assert "compdef _afs afs" in out
