"""Plugin discovery and loading helpers."""

from __future__ import annotations

import importlib
import logging
import pkgutil
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterable

from .schema import AFSConfig, PluginsConfig

logger = logging.getLogger(__name__)


def _iter_module_names(paths: list[Path] | None) -> set[str]:
    module_names: set[str] = set()
    if paths:
        search_paths = [str(path) for path in paths if path.exists()]
        for module in pkgutil.iter_modules(search_paths):
            module_names.add(module.name)
        return module_names
    for module in pkgutil.iter_modules():
        module_names.add(module.name)
    return module_names


def _filter_prefixes(names: Iterable[str], prefixes: list[str]) -> list[str]:
    return sorted(
        {
            name
            for name in names
            if any(name.startswith(prefix) for prefix in prefixes)
        }
    )


@contextmanager
def _prepend_sys_path(paths: list[Path]) -> Iterable[None]:
    if not paths:
        yield
        return
    path_strings = [str(path) for path in paths if path.exists()]
    if not path_strings:
        yield
        return
    original = list(sys.path)
    sys.path = path_strings + sys.path
    try:
        yield
    finally:
        sys.path = original


def _normalize_plugins_config(config: AFSConfig | PluginsConfig | dict | None) -> PluginsConfig:
    if config is None:
        return PluginsConfig()
    if isinstance(config, PluginsConfig):
        return config
    if isinstance(config, AFSConfig):
        return config.plugins
    if isinstance(config, dict):
        return PluginsConfig.from_dict(config.get("plugins", config))
    return PluginsConfig()


def discover_plugins(
    config: AFSConfig | PluginsConfig | dict | None = None,
    extra_paths: Iterable[Path] | None = None,
) -> list[str]:
    plugins_config = _normalize_plugins_config(config)
    names = set(plugins_config.enabled_plugins)

    if not plugins_config.auto_discover:
        return sorted(names)

    prefixes = plugins_config.auto_discover_prefixes or ["afs_plugin"]
    search_paths = list(plugins_config.plugin_dirs)
    if extra_paths:
        search_paths.extend(extra_paths)

    names.update(_filter_prefixes(_iter_module_names(search_paths), prefixes))
    names.update(_filter_prefixes(_iter_module_names(None), prefixes))
    return sorted(names)


def load_plugins(
    plugin_names: Iterable[str],
    plugin_dirs: Iterable[Path] | None = None,
) -> dict[str, ModuleType]:
    loaded: dict[str, ModuleType] = {}
    dirs = list(plugin_dirs or [])
    with _prepend_sys_path(dirs):
        for name in plugin_names:
            try:
                loaded[name] = importlib.import_module(name)
            except Exception as exc:
                logger.warning("Failed to load plugin %s: %s", name, exc)
    return loaded
