"""Extension manifest discovery and loading for AFS."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import AFSConfig, ExtensionsConfig
from .toml_compat import tomllib


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
    python_paths: list[Path] = field(default_factory=list)
    cli_modules: list[str] = field(default_factory=list)
    agent_modules: list[str] = field(default_factory=list)
    policies: list[str] = field(default_factory=list)
    hooks: dict[str, list[str]] = field(default_factory=dict)
    manager_actions: list[str] = field(default_factory=list)
    mcp_tools_module: str = ""
    mcp_tools_factory: str = "register_mcp_tools"
    mcp_server_module: str = ""
    mcp_server_factory: str = "register_mcp_server"
    context_sources: list[dict[str, Any]] = field(default_factory=list)
    # Applies only to tools returned from the [mcp_tools] factory.
    mcp_tools_catalog: str = "full"

    @property
    def import_roots(self) -> list[Path]:
        """Python import roots for extension-owned implementation modules."""
        return _merge_unique_paths(self.python_paths, [self.root, self.root.parent])


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


def _as_context_source_specs(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    specs: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        module = item.get("module")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(module, str) or not module.strip():
            continue
        spec = {
            "name": name.strip(),
            "module": module.strip(),
        }
        factory = item.get("factory")
        if isinstance(factory, str) and factory.strip():
            spec["factory"] = factory.strip()
        description = item.get("description")
        if isinstance(description, str) and description.strip():
            spec["description"] = description.strip()
        kinds = item.get("kinds")
        if isinstance(kinds, list):
            spec["kinds"] = [kind for kind in kinds if isinstance(kind, str) and kind.strip()]
        specs.append(spec)
    return specs

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


def _env_extension_repo_roots() -> list[Path]:
    raw = os.environ.get("AFS_EXTENSION_REPO_ROOTS", "").strip()
    if not raw:
        return []
    values: list[Path] = []
    for entry in raw.split(os.pathsep):
        if entry.strip():
            values.append(Path(entry).expanduser().resolve())
    return values


def _env_extension_repo_prefixes() -> list[str]:
    raw = os.environ.get("AFS_EXTENSION_REPO_PREFIXES", "").strip()
    if not raw:
        return []
    return [entry.strip() for entry in re.split(r"[,\s]+", raw) if entry.strip()]


def _env_manifest_filenames() -> list[str]:
    raw = os.environ.get("AFS_EXTENSION_MANIFEST_FILENAMES", "").strip()
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
    workspace_extension_roots: list[Path] = []
    if config is None:
        resolved = ExtensionsConfig()
    elif isinstance(config, ExtensionsConfig):
        resolved = ExtensionsConfig(
            enabled_extensions=list(config.enabled_extensions),
            extension_dirs=list(config.extension_dirs),
            auto_discover=config.auto_discover,
            extension_repo_roots=list(config.extension_repo_roots),
            extension_repo_prefixes=list(config.extension_repo_prefixes),
            manifest_filenames=list(config.manifest_filenames),
        )
    elif isinstance(config, AFSConfig):
        source = config.extensions
        workspace_extension_roots = [
            workspace.path
            for workspace in config.general.workspace_directories
        ]
        resolved = ExtensionsConfig(
            enabled_extensions=list(source.enabled_extensions),
            extension_dirs=list(source.extension_dirs),
            auto_discover=source.auto_discover,
            extension_repo_roots=list(source.extension_repo_roots),
            extension_repo_prefixes=list(source.extension_repo_prefixes),
            manifest_filenames=list(source.manifest_filenames),
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
    resolved.extension_repo_roots = _merge_unique_paths(
        _env_extension_repo_roots(),
        resolved.extension_repo_roots,
        workspace_extension_roots,
    )
    resolved.extension_repo_prefixes = _merge_unique_str(
        _env_extension_repo_prefixes(),
        resolved.extension_repo_prefixes,
    )
    resolved.manifest_filenames = _merge_unique_str(
        _env_manifest_filenames(),
        resolved.manifest_filenames,
    )
    resolved.enabled_extensions = _merge_unique_str(
        _env_enabled_extensions(),
        resolved.enabled_extensions,
    )
    return resolved


def _iter_manifest_paths(extension_dirs: list[Path], manifest_filenames: list[str]) -> list[Path]:
    manifests: list[Path] = []
    for extension_dir in extension_dirs:
        if not extension_dir.exists():
            continue

        for filename in manifest_filenames:
            direct_manifest = extension_dir / filename
            if direct_manifest.exists():
                manifests.append(direct_manifest)
                break
        else:
            direct_manifest = None
        if direct_manifest is not None and direct_manifest.exists():
            continue

        try:
            children = list(extension_dir.iterdir())
        except OSError:
            continue

        for child in children:
            if not child.is_dir():
                continue
            for filename in manifest_filenames:
                manifest = child / filename
                if manifest.exists():
                    manifests.append(manifest)
                    break
    return manifests


def _iter_extension_repo_manifest_paths(
    repo_roots: list[Path],
    prefixes: list[str],
    manifest_filenames: list[str],
) -> list[Path]:
    manifests: list[Path] = []
    prefix_tuple = tuple(prefixes or [])
    for repo_root in repo_roots:
        if not repo_root.exists():
            continue
        candidates: list[Path] = []
        if repo_root.is_dir() and repo_root.name.startswith(prefix_tuple):
            candidates.append(repo_root)
        try:
            children = list(repo_root.iterdir())
        except OSError:
            children = []
        for child in children:
            if child.is_dir() and child.name.startswith(prefix_tuple):
                candidates.append(child)
        for candidate in candidates:
            for filename in manifest_filenames:
                manifest = candidate / filename
                if manifest.exists():
                    manifests.append(manifest)
                    break
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
    python_paths = _as_path_list(
        raw.get("python_paths", raw.get("import_paths")),
        root,
    )
    if not python_paths and (root / "src").is_dir():
        python_paths = [(root / "src").resolve()]

    hooks_raw = raw.get("hooks")
    hooks: dict[str, list[str]] = {}
    if isinstance(hooks_raw, dict):
        for event, commands in hooks_raw.items():
            if isinstance(event, str):
                hooks[event] = _as_str_list(commands)

    manager_raw = raw.get("manager")
    manager_actions: list[str] = []
    if isinstance(manager_raw, dict):
        manager_actions = _as_str_list(manager_raw.get("actions"))

    mcp_tools_module = ""
    mcp_tools_factory = "register_mcp_tools"
    mcp_tools_catalog = "full"
    mcp_tools_raw = raw.get("mcp_tools")
    if isinstance(mcp_tools_raw, dict):
        module_value = mcp_tools_raw.get("module")
        if isinstance(module_value, str):
            mcp_tools_module = module_value.strip()
        factory_value = mcp_tools_raw.get("factory")
        if isinstance(factory_value, str) and factory_value.strip():
            mcp_tools_factory = factory_value.strip()
        if "catalog" in mcp_tools_raw:
            catalog_value = mcp_tools_raw.get("catalog")
            if not isinstance(catalog_value, str):
                raise ValueError("[mcp_tools].catalog must be 'full' or 'slim'")
            mcp_tools_catalog = catalog_value.strip().lower()
            if mcp_tools_catalog not in {"full", "slim"}:
                raise ValueError("[mcp_tools].catalog must be 'full' or 'slim'")
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
        python_paths=python_paths,
        cli_modules=_as_str_list(raw.get("cli_modules")),
        agent_modules=_as_str_list(raw.get("agent_modules")),
        policies=_as_str_list(raw.get("policies")),
        hooks=hooks,
        manager_actions=manager_actions,
        mcp_tools_module=mcp_tools_module,
        mcp_tools_factory=mcp_tools_factory,
        mcp_server_module=mcp_server_module,
        mcp_server_factory=mcp_server_factory,
        context_sources=_as_context_source_specs(raw.get("context_sources")),
        mcp_tools_catalog=mcp_tools_catalog,
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
    manifest_paths = [
        *_iter_manifest_paths(extension_dirs, extension_config.manifest_filenames),
        *_iter_extension_repo_manifest_paths(
            extension_config.extension_repo_roots,
            extension_config.extension_repo_prefixes,
            extension_config.manifest_filenames,
        ),
    ]
    for manifest_path in manifest_paths:
        try:
            manifest = load_extension_manifest(manifest_path)
        except Exception:
            continue
        if manifest.name in discovered:
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
