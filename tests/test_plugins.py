from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import afs.plugins as plugins_module
from afs.cli.core import plugins_command
from afs.plugins import discover_plugins
from afs.schema import AFSConfig, PluginsConfig


def test_discover_plugins_in_custom_dir(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    package_dir = plugin_dir / "afs_plugin_demo"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    plugins = PluginsConfig(plugin_dirs=[plugin_dir], auto_discover=True)
    config = AFSConfig(plugins=plugins)

    names = discover_plugins(config)
    assert "afs_plugin_demo" in names


def test_plugins_command_uses_one_extension_report_snapshot(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "afs.toml"
    config_path.write_text(
        "[extensions]\nauto_discover = false\n"
        "[plugins]\nauto_discover = false\n",
        encoding="utf-8",
    )
    calls = 0

    def report(**_kwargs):
        nonlocal calls
        calls += 1
        return {
            "extensions": [
                {
                    "name": "example",
                    "path": str(tmp_path / "extension.toml"),
                    "api_version": 1,
                    "enabled": False,
                    "warnings": [],
                }
            ],
            "errors": [],
        }

    monkeypatch.setattr(plugins_module, "extension_load_report", report)

    rc = plugins_command(
        Namespace(config=str(config_path), load=False, json=True, details=False)
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert calls == 1
    assert payload["extensions"] == [
        {
            "name": "example",
            "manifest": str(tmp_path / "extension.toml"),
            "status": "discovered",
            "warnings": [],
        }
    ]
