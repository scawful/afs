from __future__ import annotations

from pathlib import Path

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
