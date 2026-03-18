from __future__ import annotations

from pathlib import Path

from afs.cli import build_parser


def _command_choices(parser) -> set[str]:
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if choices:
            return set(choices.keys())
    return set()


def test_core_parser_excludes_legacy_model_command_groups(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "afs.toml"
    config_path.write_text("[extensions]\nauto_discover = false\n", encoding="utf-8")
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("AFS_EXTENSION_DIRS", raising=False)
    monkeypatch.delenv("AFS_ENABLED_EXTENSIONS", raising=False)

    parser = build_parser()
    commands = _command_choices(parser)

    assert "context" in commands
    assert "mcp" in commands
    assert "profile" in commands
    assert "skills" in commands
    assert "health" in commands

    assert "claude" not in commands
    assert "training" not in commands
    assert "gateway" not in commands
    assert "vastai" not in commands
    assert "benchmark" not in commands
    assert "comparison" not in commands


def test_extension_cli_modules_can_restore_commands(
    monkeypatch,
    tmp_path: Path,
) -> None:
    extension_root = tmp_path / "extensions" / "legacy_scope"
    package_root = extension_root / "legacy_scope"
    package_root.mkdir(parents=True)

    (extension_root / "extension.toml").write_text(
        "name = \"legacy_scope\"\n"
        "cli_modules = [\"legacy_scope.cli\"]\n",
        encoding="utf-8",
    )
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "cli.py").write_text(
        "def register_parsers(subparsers):\n"
        "    parser = subparsers.add_parser('legacy-demo', help='legacy demo command')\n"
        "    parser.set_defaults(func=lambda _args: 0)\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "afs.toml"
    config_path.write_text("[extensions]\nauto_discover = false\n", encoding="utf-8")

    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("AFS_EXTENSION_DIRS", str(tmp_path / "extensions"))
    monkeypatch.setenv("AFS_ENABLED_EXTENSIONS", "legacy_scope")

    parser = build_parser()
    commands = _command_choices(parser)

    assert "legacy-demo" in commands


def test_profile_cli_modules_can_register_commands(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "profile_cli.py").write_text(
        "def register_parsers(subparsers):\n"
        "    parser = subparsers.add_parser('profile-demo', help='profile demo command')\n"
        "    parser.set_defaults(func=lambda _args: 0)\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "afs.toml"
    config_path.write_text(
        "[extensions]\n"
        "auto_discover = false\n\n"
        "[profiles]\n"
        "active_profile = \"work\"\n\n"
        "[profiles.work]\n"
        "cli_modules = [\"profile_cli\"]\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("AFS_EXTENSION_DIRS", raising=False)
    monkeypatch.delenv("AFS_ENABLED_EXTENSIONS", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))

    parser = build_parser()
    commands = _command_choices(parser)

    assert "profile-demo" in commands


def test_services_subcommands_accept_config_flag(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "afs.toml"
    config_path.write_text("[extensions]\nauto_discover = false\n", encoding="utf-8")
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))

    parser = build_parser()
    args = parser.parse_args(
        [
            "services",
            "render",
            "--config",
            str(config_path),
            "context-warm",
        ]
    )

    assert args.command == "services"
    assert args.services_command == "render"
    assert str(args.config) == str(config_path)
