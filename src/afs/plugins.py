"""Plugin discovery and loading helpers."""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

from .schema import AFSConfig, PluginsConfig

logger = logging.getLogger(__name__)

_LOADED_PLUGINS: dict[str, ModuleType] = {}


def _merge_unique(items: Iterable[str], fallback: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for entry in list(items) + list(fallback):
        entry = entry.strip()
        if not entry or entry in seen:
            continue
        merged.append(entry)
        seen.add(entry)
    return merged


def _merge_unique_paths(items: Iterable[Path], fallback: Iterable[Path]) -> list[Path]:
    merged: list[Path] = []
    seen: set[str] = set()
    for entry in list(items) + list(fallback):
        entry_str = str(entry)
        if not entry_str or entry_str in seen:
            continue
        merged.append(entry)
        seen.add(entry_str)
    return merged


def _env_plugin_dirs() -> list[Path]:
    raw = os.environ.get("AFS_PLUGIN_DIRS")
    if not raw:
        return []
    entries = []
    for item in raw.split(os.pathsep):
        item = item.strip()
        if not item:
            continue
        entries.append(Path(item).expanduser().resolve())
    return entries


def _env_enabled_plugins() -> list[str]:
    raw = os.environ.get("AFS_ENABLED_PLUGINS")
    if not raw:
        return []
    entries = [item.strip() for item in re.split(r"[,\s]+", raw) if item.strip()]
    return entries


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
        plugins_config = PluginsConfig(
            enabled_plugins=list(config.enabled_plugins),
            plugin_dirs=list(config.plugin_dirs),
            auto_discover=config.auto_discover,
            auto_discover_prefixes=list(config.auto_discover_prefixes),
        )
    elif isinstance(config, AFSConfig):
        plugins = config.plugins
        plugins_config = PluginsConfig(
            enabled_plugins=list(plugins.enabled_plugins),
            plugin_dirs=list(plugins.plugin_dirs),
            auto_discover=plugins.auto_discover,
            auto_discover_prefixes=list(plugins.auto_discover_prefixes),
        )
    elif isinstance(config, dict):
        plugins_config = PluginsConfig.from_dict(config.get("plugins", config))
    else:
        plugins_config = PluginsConfig()

    env_dirs = _env_plugin_dirs()
    env_enabled = _env_enabled_plugins()
    if env_dirs:
        plugins_config.plugin_dirs = _merge_unique_paths(
            env_dirs, plugins_config.plugin_dirs
        )
    if env_enabled:
        plugins_config.enabled_plugins = _merge_unique(
            env_enabled, plugins_config.enabled_plugins
        )
    return plugins_config


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


def load_enabled_plugins(
    config: AFSConfig | PluginsConfig | dict | None = None,
    extra_paths: Iterable[Path] | None = None,
    *,
    force: bool = False,
) -> dict[str, ModuleType]:
    """Load plugins from config (cached by default)."""
    global _LOADED_PLUGINS
    if _LOADED_PLUGINS and not force:
        return dict(_LOADED_PLUGINS)

    if config is None:
        try:
            from .config import load_config_model

            config = load_config_model()
        except Exception:
            config = PluginsConfig()

    plugins_config = _normalize_plugins_config(config)
    plugin_names = discover_plugins(plugins_config, extra_paths=extra_paths)
    plugin_dirs = list(plugins_config.plugin_dirs)
    if extra_paths:
        plugin_dirs.extend(extra_paths)

    _LOADED_PLUGINS = load_plugins(plugin_names, plugin_dirs)
    return dict(_LOADED_PLUGINS)


def call_plugin_hook(
    hook: str,
    *args: Any,
    plugins: Iterable[ModuleType] | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Invoke a hook across loaded plugins, ignoring failures."""
    modules = list(plugins or load_enabled_plugins().values())
    results: list[Any] = []
    for module in modules:
        handler = getattr(module, hook, None)
        if callable(handler):
            try:
                results.append(handler(*args, **kwargs))
            except Exception as exc:
                name = getattr(module, "__name__", "unknown")
                logger.warning("Plugin hook %s failed in %s: %s", hook, name, exc)
    return results
