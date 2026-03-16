"""Extension manifest discovery and loading for AFS."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

from .schema import AFSConfig, ExtensionsConfig


@dataclass(frozen=True)
class ExtensionManifest:
    """Normalized extension manifest."""

    name: str
    root: Path
    manifest_path: Path
    description: str = ""
    knowledge_mounts: list[Path] = field(default_factory=list)
    skill_roots: list[Path] = field(default_factory=list)
    model_registries: list[Path] = field(default_factory=list)
    cli_modules: list[str] = field(default_factory=list)
    agent_modules: list[str] = field(default_factory=list)
    policies: list[str] = field(default_factory=list)
    hooks: dict[str, list[str]] = field(default_factory=dict)
    mcp_tools_module: str = ""
    mcp_tools_factory: str = "register_mcp_tools"
    mcp_server_module: str = ""
    mcp_server_factory: str = "register_mcp_server"


def _as_path_list(items: Any, root: Path) -> list[Path]:
    if not isinstance(items, list):
        return []
    values: list[Path] = []
    for entry in items:
        if not isinstance(entry, (str, Path)):
            continue
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = (root / path).resolve()
        else:
            path = path.resolve()
        values.append(path)
    return values


def _as_str_list(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [str(entry) for entry in items if isinstance(entry, str)]


def _env_extension_dirs() -> list[Path]:
    raw = os.environ.get("AFS_EXTENSION_DIRS", "").strip()
    if not raw:
        return []
    values: list[Path] = []
    for entry in raw.split(os.pathsep):
        if entry.strip():
            values.append(Path(entry).expanduser().resolve())
    return values


def _env_enabled_extensions() -> list[str]:
    raw = os.environ.get("AFS_ENABLED_EXTENSIONS", "").strip()
    if not raw:
        return []
    return [entry.strip() for entry in re.split(r"[,\s]+", raw) if entry.strip()]


def _default_extension_dirs() -> list[Path]:
    return [
        Path("extensions").expanduser().resolve(),
        Path("~/.config/afs/extensions").expanduser().resolve(),
        Path("~/.afs/extensions").expanduser().resolve(),
    ]


def _merge_unique_paths(*groups: list[Path]) -> list[Path]:
    merged: list[Path] = []
    seen: set[str] = set()
    for group in groups:
        for path in group:
            marker = str(path)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(path)
    return merged


def _merge_unique_str(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            value = value.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            merged.append(value)
    return merged


def resolve_extensions_config(config: AFSConfig | ExtensionsConfig | dict | None = None) -> ExtensionsConfig:
    """Resolve extension config with env and default dirs."""
    if config is None:
        resolved = ExtensionsConfig()
    elif isinstance(config, ExtensionsConfig):
        resolved = ExtensionsConfig(
            enabled_extensions=list(config.enabled_extensions),
            extension_dirs=list(config.extension_dirs),
            auto_discover=config.auto_discover,
        )
    elif isinstance(config, AFSConfig):
        source = config.extensions
        resolved = ExtensionsConfig(
            enabled_extensions=list(source.enabled_extensions),
            extension_dirs=list(source.extension_dirs),
            auto_discover=source.auto_discover,
        )
    elif isinstance(config, dict):
        resolved = ExtensionsConfig.from_dict(config.get("extensions", config))
    else:
        resolved = ExtensionsConfig()

    resolved.extension_dirs = _merge_unique_paths(
        _env_extension_dirs(),
        resolved.extension_dirs,
        _default_extension_dirs(),
    )
    resolved.enabled_extensions = _merge_unique_str(
        _env_enabled_extensions(),
        resolved.enabled_extensions,
    )
    return resolved


def _iter_manifest_paths(extension_dirs: list[Path]) -> list[Path]:
    manifests: list[Path] = []
    for extension_dir in extension_dirs:
        if not extension_dir.exists():
            continue

        direct_manifest = extension_dir / "extension.toml"
        if direct_manifest.exists():
            manifests.append(direct_manifest)
            continue

        try:
            children = list(extension_dir.iterdir())
        except OSError:
            continue

        for child in children:
            if not child.is_dir():
                continue
            manifest = child / "extension.toml"
            if manifest.exists():
                manifests.append(manifest)
    return manifests


def load_extension_manifest(path: Path) -> ExtensionManifest:
    """Load a single extension manifest from disk."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    root = path.parent.resolve()
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        name = root.name
    description = raw.get("description")
    if not isinstance(description, str):
        description = ""

    mounts = raw.get("mounts", {}) if isinstance(raw.get("mounts"), dict) else {}

    knowledge_mounts = _as_path_list(
        raw.get("knowledge_mounts", mounts.get("knowledge_mounts")),
        root,
    )
    skill_roots = _as_path_list(
        raw.get("skill_roots", mounts.get("skill_roots")),
        root,
    )
    model_registries = _as_path_list(
        raw.get("model_registries", mounts.get("model_registries")),
        root,
    )

    hooks_raw = raw.get("hooks")
    hooks: dict[str, list[str]] = {}
    if isinstance(hooks_raw, dict):
        for event, commands in hooks_raw.items():
            if isinstance(event, str):
                hooks[event] = _as_str_list(commands)

    mcp_tools_module = ""
    mcp_tools_factory = "register_mcp_tools"
    mcp_tools_raw = raw.get("mcp_tools")
    if isinstance(mcp_tools_raw, dict):
        module_value = mcp_tools_raw.get("module")
        if isinstance(module_value, str):
            mcp_tools_module = module_value.strip()
        factory_value = mcp_tools_raw.get("factory")
        if isinstance(factory_value, str) and factory_value.strip():
            mcp_tools_factory = factory_value.strip()
    else:
        module_value = raw.get("mcp_tools_module")
        if isinstance(module_value, str):
            mcp_tools_module = module_value.strip()

    mcp_server_module = ""
    mcp_server_factory = "register_mcp_server"
    mcp_server_raw = raw.get("mcp_server")
    if isinstance(mcp_server_raw, dict):
        module_value = mcp_server_raw.get("module")
        if isinstance(module_value, str):
            mcp_server_module = module_value.strip()
        factory_value = mcp_server_raw.get("factory")
        if isinstance(factory_value, str) and factory_value.strip():
            mcp_server_factory = factory_value.strip()
    else:
        module_value = raw.get("mcp_server_module")
        if isinstance(module_value, str):
            mcp_server_module = module_value.strip()

    return ExtensionManifest(
        name=name.strip(),
        root=root,
        manifest_path=path.resolve(),
        description=description.strip(),
        knowledge_mounts=knowledge_mounts,
        skill_roots=skill_roots,
        model_registries=model_registries,
        cli_modules=_as_str_list(raw.get("cli_modules")),
        agent_modules=_as_str_list(raw.get("agent_modules")),
        policies=_as_str_list(raw.get("policies")),
        hooks=hooks,
        mcp_tools_module=mcp_tools_module,
        mcp_tools_factory=mcp_tools_factory,
        mcp_server_module=mcp_server_module,
        mcp_server_factory=mcp_server_factory,
    )


def discover_extension_manifests(
    config: AFSConfig | ExtensionsConfig | dict | None = None,
    extra_dirs: list[Path] | None = None,
) -> dict[str, Path]:
    """Discover available extension manifests by name."""
    extension_config = resolve_extensions_config(config)
    extension_dirs = list(extension_config.extension_dirs)
    if extra_dirs:
        extension_dirs.extend([path.expanduser().resolve() for path in extra_dirs])

    discovered: dict[str, Path] = {}
    for manifest_path in _iter_manifest_paths(extension_dirs):
        try:
            manifest = load_extension_manifest(manifest_path)
        except Exception:
            continue
        discovered[manifest.name] = manifest.manifest_path
    return dict(sorted(discovered.items()))


def load_extensions(
    config: AFSConfig | ExtensionsConfig | dict | None = None,
    requested: list[str] | None = None,
    extra_dirs: list[Path] | None = None,
    *,
    allow_auto_discover: bool = True,
) -> dict[str, ExtensionManifest]:
    """Load enabled/requested extensions."""
    extension_config = resolve_extensions_config(config)
    requested_names = _merge_unique_str(
        requested or [],
        extension_config.enabled_extensions,
    )

    manifests_by_name = discover_extension_manifests(
        config=extension_config,
        extra_dirs=extra_dirs,
    )

    loaded: dict[str, ExtensionManifest] = {}
    if allow_auto_discover and extension_config.auto_discover and not requested_names:
        requested_names = sorted(manifests_by_name.keys())

    for name in requested_names:
        manifest_path = manifests_by_name.get(name)
        if not manifest_path:
            continue
        try:
            loaded[name] = load_extension_manifest(manifest_path)
        except Exception:
            continue
    return loaded
