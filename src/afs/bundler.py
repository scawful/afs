"""Profile bundle packaging: pack, install, inspect."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

from .config import load_config_model
from .extensions import resolve_extensions_config
from .profiles import resolve_active_profile
from .schema import AFSConfig, AgentConfig, BundleManifest, ProfileConfig

DEFAULT_EXTENSION_ROOT = Path.home() / ".config" / "afs" / "extensions"
CORE_SOURCE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BundlePackResult:
    path: Path
    file_count: int = 0
    size_bytes: int = 0
    bundled_modules: list[str] = field(default_factory=list)


@dataclass
class BundleInstallResult:
    extension_path: Path
    profile_name: str = ""
    profile_snippet_path: Path | None = None


@dataclass
class BundleInspectResult:
    manifest: BundleManifest
    resource_counts: dict[str, int] = field(default_factory=dict)
    bundled_modules: list[str] = field(default_factory=list)


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_array(values: Iterable[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def _count_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for file_path in directory.rglob("*") if file_path.is_file())


def _dir_size(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(file_path.stat().st_size for file_path in directory.rglob("*") if file_path.is_file())


def _write_agent_block(lines: list[str], table_name: str, agent: AgentConfig) -> None:
    lines.append("")
    lines.append(f"[[{table_name}]]")
    lines.append(f"name = {_toml_string(agent.name)}")
    lines.append(f"role = {_toml_string(agent.role)}")
    lines.append(f"backend = {_toml_string(agent.backend)}")
    lines.append(f"description = {_toml_string(agent.description)}")
    lines.append("tags = " + _toml_array(agent.tags))
    lines.append(f"auto_start = {str(agent.auto_start).lower()}")
    lines.append("triggers = " + _toml_array(agent.triggers))
    lines.append(f"schedule = {_toml_string(agent.schedule)}")
    lines.append(f"module = {_toml_string(agent.module)}")
    lines.append("watch_paths = " + _toml_array(str(path) for path in agent.watch_paths))


def _write_profile_section(lines: list[str], table_name: str, profile: ProfileConfig) -> None:
    lines.append("")
    lines.append(f"[{table_name}]")
    lines.append("inherits = " + _toml_array(profile.inherits))
    lines.append(
        "knowledge_mounts = " + _toml_array(str(path) for path in profile.knowledge_mounts)
    )
    lines.append("skill_roots = " + _toml_array(str(path) for path in profile.skill_roots))
    lines.append(
        "model_registries = " + _toml_array(str(path) for path in profile.model_registries)
    )
    lines.append("enabled_extensions = " + _toml_array(profile.enabled_extensions))
    lines.append("policies = " + _toml_array(profile.policies))
    lines.append("mcp_tools = " + _toml_array(profile.mcp_tools))
    lines.append("cli_modules = " + _toml_array(profile.cli_modules))
    for agent in profile.agent_configs:
        _write_agent_block(lines, f"{table_name}.agent_configs", agent)


def _write_bundle_toml(path: Path, manifest: BundleManifest) -> None:
    lines = [
        f"name = {_toml_string(manifest.name)}",
        f"version = {_toml_string(manifest.version)}",
        f"description = {_toml_string(manifest.description)}",
        f"author = {_toml_string(manifest.author)}",
        f"skills_dir = {_toml_string(manifest.skills_dir)}",
        f"knowledge_dir = {_toml_string(manifest.knowledge_dir)}",
        f"tools_dir = {_toml_string(manifest.tools_dir)}",
        f"agents_dir = {_toml_string(manifest.agents_dir)}",
        f"mcp_tools_dir = {_toml_string(manifest.mcp_tools_dir)}",
    ]
    _write_profile_section(lines, "profile", manifest.profile)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_bundle_toml(path: Path) -> BundleManifest:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return BundleManifest.from_dict(data)


def _bundle_profile_snapshot(profile_name: str, config: AFSConfig) -> ProfileConfig:
    resolved = resolve_active_profile(config, profile_name=profile_name)
    return ProfileConfig(
        knowledge_mounts=list(resolved.knowledge_mounts),
        skill_roots=list(resolved.skill_roots),
        model_registries=list(resolved.model_registries),
        enabled_extensions=list(resolved.enabled_extensions),
        policies=list(resolved.policies),
        mcp_tools=list(resolved.mcp_tools),
        cli_modules=list(resolved.cli_modules),
        agent_configs=list(resolved.agent_configs),
    )


@contextmanager
def _import_roots(search_roots: Iterable[Path]) -> Iterable[None]:
    candidates: list[str] = []
    for root in search_roots:
        candidates.extend([str(root), str(root.parent)])
    original = list(sys.path)
    sys.path = [entry for entry in candidates if Path(entry).exists()] + original
    try:
        yield
    finally:
        sys.path = original


def _module_search_roots(config: AFSConfig) -> list[Path]:
    roots: list[Path] = []
    extension_config = resolve_extensions_config(config)
    roots.extend(extension_config.extension_dirs)
    for root in config.plugins.plugin_dirs:
        roots.append(root)
    return [root.expanduser().resolve() for root in roots if root.exists()]


def _root_module_names(profile: ProfileConfig) -> list[str]:
    names: list[str] = []
    for module_name in profile.cli_modules:
        root_name = module_name.strip().split(".", 1)[0]
        if root_name:
            names.append(root_name)
    for module_name in profile.mcp_tools:
        root_name = module_name.strip().split(".", 1)[0]
        if root_name:
            names.append(root_name)
    for agent in profile.agent_configs:
        root_name = agent.module.strip().split(".", 1)[0]
        if root_name:
            names.append(root_name)

    ordered: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _copy_root_module(
    module_name: str,
    *,
    destination_root: Path,
    search_roots: list[Path],
) -> bool:
    with _import_roots(search_roots):
        spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False

    locations = list(spec.submodule_search_locations or [])
    origin = spec.origin
    source_path: Path | None = None
    destination_path: Path

    if locations:
        source_path = Path(locations[0]).resolve()
        destination_path = destination_root / module_name
    elif isinstance(origin, str) and origin not in {"built-in", "frozen"}:
        source_path = Path(origin).resolve()
        destination_path = destination_root / f"{module_name}.py"
    else:
        return False

    if source_path.is_relative_to(CORE_SOURCE_ROOT):
        return False

    if source_path.is_dir():
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
    else:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    return True


def _write_mcp_surface_module(
    extension_path: Path,
    extension_name: str,
    modules: list[str],
) -> str:
    module_name = f"afs_bundle_{_slug(extension_name)}_mcp"
    module_path = extension_path / f"{module_name}.py"
    module_list = ", ".join(repr(name) for name in modules)
    module_path.write_text(
        "\"\"\"Generated bundle MCP surface.\"\"\"\n\n"
        "from __future__ import annotations\n\n"
        "import importlib\n\n"
        f"MODULES = [{module_list}]\n\n"
        "def _normalize(payload):\n"
        "    if isinstance(payload, list):\n"
        "        return {\"tools\": list(payload), \"resources\": [], \"prompts\": []}\n"
        "    if not isinstance(payload, dict):\n"
        "        return {\"tools\": [], \"resources\": [], \"prompts\": []}\n"
        "    return {\n"
        "        \"tools\": list(payload.get(\"tools\") or []),\n"
        "        \"resources\": list(payload.get(\"resources\") or []),\n"
        "        \"prompts\": list(payload.get(\"prompts\") or []),\n"
        "    }\n\n"
        "def register_mcp_server(manager):\n"
        "    merged = {\"tools\": [], \"resources\": [], \"prompts\": []}\n"
        "    for module_name in MODULES:\n"
        "        module = importlib.import_module(module_name)\n"
        "        factory = getattr(module, \"register_mcp_server\", None)\n"
        "        if not callable(factory):\n"
        "            factory = getattr(module, \"register_mcp_tools\", None)\n"
        "        if not callable(factory):\n"
        "            continue\n"
        "        payload = _normalize(factory(manager))\n"
        "        merged[\"tools\"].extend(payload[\"tools\"])\n"
        "        merged[\"resources\"].extend(payload[\"resources\"])\n"
        "        merged[\"prompts\"].extend(payload[\"prompts\"])\n"
        "    return merged\n",
        encoding="utf-8",
    )
    return module_name


def _write_agent_registry_module(
    extension_path: Path,
    extension_name: str,
    agents: list[AgentConfig],
) -> str:
    module_name = f"afs_bundle_{_slug(extension_name)}_agents"
    module_path = extension_path / f"{module_name}.py"
    records = [
        {
            "name": agent.name,
            "description": agent.description,
            "module": agent.module,
        }
        for agent in agents
        if agent.name.strip() and agent.module.strip()
    ]
    record_lines = ",\n".join(
        "    {"
        + f"'name': {record['name']!r}, "
        + f"'description': {record['description']!r}, "
        + f"'module': {record['module']!r}"
        + "}"
        for record in records
    )
    module_path.write_text(
        "\"\"\"Generated bundle agent registry.\"\"\"\n\n"
        "from __future__ import annotations\n\n"
        "import importlib\n\n"
        f"AGENTS = [\n{record_lines}\n]\n\n"
        "def _entrypoint(module_name):\n"
        "    def _run(argv=None):\n"
        "        module = importlib.import_module(module_name)\n"
        "        main = getattr(module, 'main', None)\n"
        "        if not callable(main):\n"
        "            raise RuntimeError(f\"agent module missing main(): {module_name}\")\n"
        "        return main(argv)\n"
        "    return _run\n\n"
        "def register_agents():\n"
        "    return [\n"
        "        {\n"
        "            'name': agent['name'],\n"
        "            'description': agent['description'],\n"
        "            'entrypoint': _entrypoint(agent['module']),\n"
        "        }\n"
        "        for agent in AGENTS\n"
        "    ]\n",
        encoding="utf-8",
    )
    return module_name


def _write_extension_toml(
    extension_path: Path,
    ext_name: str,
    manifest: BundleManifest,
    *,
    mcp_surface_module: str | None,
    agent_registry_module: str | None,
) -> None:
    lines = [
        f"name = {_toml_string(ext_name)}",
        f"description = {_toml_string(manifest.description)}",
    ]
    if manifest.profile.policies:
        lines.append("policies = " + _toml_array(manifest.profile.policies))
    if (extension_path / manifest.knowledge_dir).exists():
        lines.append("knowledge_mounts = " + _toml_array([manifest.knowledge_dir]))
    if (extension_path / manifest.skills_dir).exists():
        lines.append("skill_roots = " + _toml_array([manifest.skills_dir]))
    if manifest.profile.cli_modules:
        lines.append("cli_modules = " + _toml_array(manifest.profile.cli_modules))
    if agent_registry_module:
        lines.append("agent_modules = " + _toml_array([agent_registry_module]))
    if mcp_surface_module:
        lines.append("")
        lines.append("[mcp_server]")
        lines.append(f"module = {_toml_string(mcp_surface_module)}")
        lines.append('factory = "register_mcp_server"')
    (extension_path / "extension.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_profile_snippet(
    extension_path: Path,
    profile_name: str,
    ext_name: str,
    profile: ProfileConfig,
) -> Path:
    snippet_path = extension_path / "profile-snippet.toml"
    bundled_profile = ProfileConfig(
        inherits=list(profile.inherits),
        enabled_extensions=[ext_name],
        policies=list(profile.policies),
        agent_configs=list(profile.agent_configs),
    )
    lines = []
    _write_profile_section(lines, f"profiles.{profile_name}", bundled_profile)
    snippet_path.write_text("\n".join(lines).lstrip() + "\n", encoding="utf-8")
    return snippet_path


def _resolve_install_root(
    config: AFSConfig,
    install_dir: Path | None,
) -> Path:
    if install_dir is not None:
        return install_dir.expanduser().resolve()
    extension_config = resolve_extensions_config(config)
    if extension_config.extension_dirs:
        return extension_config.extension_dirs[0].expanduser().resolve()
    return DEFAULT_EXTENSION_ROOT


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_") or "bundle"


def pack_bundle(
    profile_name: str,
    config: AFSConfig | None = None,
    output_path: Path | None = None,
) -> BundlePackResult:
    """Pack a profile into a portable bundle directory."""
    if config is None:
        config = load_config_model(merge_user=True)

    output_path = output_path or Path.cwd()
    bundle_dir = output_path / profile_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    profile_snapshot = _bundle_profile_snapshot(profile_name, config)
    manifest = BundleManifest(
        name=profile_name,
        description=f"Bundle for profile '{profile_name}'",
        profile=profile_snapshot,
    )

    knowledge_dir = bundle_dir / manifest.knowledge_dir
    knowledge_dir.mkdir(exist_ok=True)
    for mount in profile_snapshot.knowledge_mounts:
        source = mount.resolve() if mount.is_symlink() else mount
        if source.exists() and source.is_dir():
            shutil.copytree(
                source, knowledge_dir / mount.name,
                symlinks=False, dirs_exist_ok=True,
            )

    skills_dir = bundle_dir / manifest.skills_dir
    skills_dir.mkdir(exist_ok=True)
    for root in profile_snapshot.skill_roots:
        source = root.resolve() if root.is_symlink() else root
        if source.exists() and source.is_dir():
            shutil.copytree(
                source, skills_dir / root.name,
                symlinks=False, dirs_exist_ok=True,
            )

    (bundle_dir / manifest.agents_dir).mkdir(exist_ok=True)
    (bundle_dir / manifest.tools_dir).mkdir(exist_ok=True)
    (bundle_dir / manifest.mcp_tools_dir).mkdir(exist_ok=True)

    search_roots = _module_search_roots(config)
    bundled_modules: list[str] = []
    for root_module in _root_module_names(profile_snapshot):
        if _copy_root_module(root_module, destination_root=bundle_dir, search_roots=search_roots):
            bundled_modules.append(root_module)

    _write_bundle_toml(bundle_dir / "bundle.toml", manifest)

    return BundlePackResult(
        path=bundle_dir,
        file_count=_count_files(bundle_dir),
        size_bytes=_dir_size(bundle_dir),
        bundled_modules=bundled_modules,
    )


def install_bundle(
    bundle_path: Path,
    config: AFSConfig | None = None,
    name_override: str | None = None,
    install_dir: Path | None = None,
) -> BundleInstallResult:
    """Install a bundle as an AFS extension."""
    bundle_toml = bundle_path / "bundle.toml"
    if not bundle_toml.exists():
        raise FileNotFoundError(f"No bundle.toml found in {bundle_path}")

    if config is None:
        config = load_config_model(merge_user=True)

    manifest = _read_bundle_toml(bundle_toml)
    ext_name = name_override or manifest.name
    extensions_root = _resolve_install_root(config, install_dir)
    extension_path = extensions_root / ext_name
    extension_path.mkdir(parents=True, exist_ok=True)

    for item in bundle_path.iterdir():
        destination = extension_path / item.name
        if item.is_dir():
            shutil.copytree(item, destination, symlinks=False, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)

    mcp_surface_module = None
    if manifest.profile.mcp_tools:
        mcp_surface_module = _write_mcp_surface_module(
            extension_path,
            ext_name,
            list(manifest.profile.mcp_tools),
        )

    agent_registry_module = None
    if any(agent.name.strip() and agent.module.strip() for agent in manifest.profile.agent_configs):
        agent_registry_module = _write_agent_registry_module(
            extension_path,
            ext_name,
            manifest.profile.agent_configs,
        )

    _write_extension_toml(
        extension_path,
        ext_name,
        manifest,
        mcp_surface_module=mcp_surface_module,
        agent_registry_module=agent_registry_module,
    )
    profile_snippet_path = _write_profile_snippet(
        extension_path,
        manifest.name,
        ext_name,
        manifest.profile,
    )

    return BundleInstallResult(
        extension_path=extension_path,
        profile_name=manifest.name,
        profile_snippet_path=profile_snippet_path,
    )


def inspect_bundle(bundle_path: Path) -> BundleInspectResult:
    """Inspect a bundle directory and return manifest + resource counts."""
    bundle_toml = bundle_path / "bundle.toml"
    if not bundle_toml.exists():
        raise FileNotFoundError(f"No bundle.toml found in {bundle_path}")

    manifest = _read_bundle_toml(bundle_toml)
    resource_counts: dict[str, int] = {}
    for dir_attr in ("skills_dir", "knowledge_dir", "tools_dir", "agents_dir", "mcp_tools_dir"):
        dir_name = getattr(manifest, dir_attr)
        count = _count_files(bundle_path / dir_name)
        if count > 0:
            resource_counts[dir_name] = count

    bundled_modules: list[str] = []
    for module_name in _root_module_names(manifest.profile):
        if (bundle_path / module_name).exists() or (bundle_path / f"{module_name}.py").exists():
            bundled_modules.append(module_name)

    return BundleInspectResult(
        manifest=manifest,
        resource_counts=resource_counts,
        bundled_modules=bundled_modules,
    )


def list_bundles(config: AFSConfig | None = None) -> list[dict[str, Any]]:
    """List installed bundles (extensions with bundle.toml)."""
    if config is None:
        config = load_config_model(merge_user=True)

    extension_config = resolve_extensions_config(config)
    bundles: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for extensions_dir in extension_config.extension_dirs or [DEFAULT_EXTENSION_ROOT]:
        if not extensions_dir.exists():
            continue
        for ext_dir in sorted(extensions_dir.iterdir()):
            bundle_toml = ext_dir / "bundle.toml"
            if not ext_dir.is_dir() or not bundle_toml.exists():
                continue
            resolved_dir = ext_dir.resolve()
            if resolved_dir in seen_paths:
                continue
            seen_paths.add(resolved_dir)
            try:
                manifest = _read_bundle_toml(bundle_toml)
            except Exception:
                continue
            bundles.append(
                {
                    "name": manifest.name,
                    "version": manifest.version,
                    "path": str(ext_dir),
                    "description": manifest.description,
                    "profile": manifest.name,
                }
            )
    return bundles
