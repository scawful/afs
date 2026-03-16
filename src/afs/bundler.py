"""Profile bundle packaging: pack, install, inspect."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

from .config import load_config_model
from .profiles import resolve_active_profile
from .schema import AFSConfig, BundleManifest


@dataclass
class BundlePackResult:
    path: Path
    file_count: int = 0
    size_bytes: int = 0


@dataclass
class BundleInstallResult:
    extension_path: Path
    profile_name: str = ""


@dataclass
class BundleInspectResult:
    manifest: BundleManifest
    resource_counts: dict[str, int] = field(default_factory=dict)


def _count_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*") if f.is_file())


def _dir_size(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def _write_bundle_toml(path: Path, manifest: BundleManifest) -> None:
    lines = [
        f'name = "{manifest.name}"',
        f'version = "{manifest.version}"',
        f'description = "{manifest.description}"',
        f'author = "{manifest.author}"',
        f'skills_dir = "{manifest.skills_dir}"',
        f'knowledge_dir = "{manifest.knowledge_dir}"',
        f'tools_dir = "{manifest.tools_dir}"',
        f'agents_dir = "{manifest.agents_dir}"',
        f'mcp_tools_dir = "{manifest.mcp_tools_dir}"',
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_bundle_toml(path: Path) -> BundleManifest:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return BundleManifest.from_dict(data)


def pack_bundle(
    profile_name: str,
    config: AFSConfig | None = None,
    output_path: Path | None = None,
) -> BundlePackResult:
    """Pack a profile into a portable bundle directory."""
    if config is None:
        config = load_config_model(merge_user=True)

    resolved = resolve_active_profile(config, profile_name=profile_name)
    output_path = output_path or Path.cwd()
    bundle_dir = output_path / resolved.name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest = BundleManifest(
        name=resolved.name,
        description=f"Bundle for profile '{resolved.name}'",
    )

    # Copy knowledge mounts
    knowledge_dir = bundle_dir / manifest.knowledge_dir
    knowledge_dir.mkdir(exist_ok=True)
    for mount in resolved.knowledge_mounts:
        if mount.exists() and mount.is_dir():
            dest = knowledge_dir / mount.name
            if not dest.exists():
                shutil.copytree(mount, dest, dirs_exist_ok=True)

    # Copy skill roots
    skills_dir = bundle_dir / manifest.skills_dir
    skills_dir.mkdir(exist_ok=True)
    for root in resolved.skill_roots:
        if root.exists() and root.is_dir():
            dest = skills_dir / root.name
            if not dest.exists():
                shutil.copytree(root, dest, dirs_exist_ok=True)

    # Create stub dirs for agents/tools/mcp_tools
    (bundle_dir / manifest.agents_dir).mkdir(exist_ok=True)
    (bundle_dir / manifest.tools_dir).mkdir(exist_ok=True)
    (bundle_dir / manifest.mcp_tools_dir).mkdir(exist_ok=True)

    _write_bundle_toml(bundle_dir / "bundle.toml", manifest)

    file_count = _count_files(bundle_dir)
    size_bytes = _dir_size(bundle_dir)

    return BundlePackResult(path=bundle_dir, file_count=file_count, size_bytes=size_bytes)


def install_bundle(
    bundle_path: Path,
    config: AFSConfig | None = None,
    name_override: str | None = None,
) -> BundleInstallResult:
    """Install a bundle as an AFS extension."""
    bundle_toml = bundle_path / "bundle.toml"
    if not bundle_toml.exists():
        raise FileNotFoundError(f"No bundle.toml found in {bundle_path}")

    manifest = _read_bundle_toml(bundle_toml)
    ext_name = name_override or manifest.name

    extensions_dir = Path.home() / ".config" / "afs" / "extensions" / ext_name
    extensions_dir.mkdir(parents=True, exist_ok=True)

    # Copy bundle contents
    for item in bundle_path.iterdir():
        dest = extensions_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    # Generate extension.toml
    ext_toml = extensions_dir / "extension.toml"
    lines = [
        f'name = "{ext_name}"',
        f'description = "{manifest.description}"',
    ]
    if (extensions_dir / manifest.knowledge_dir).exists():
        lines.append(f'knowledge_mounts = ["{manifest.knowledge_dir}"]')
    if (extensions_dir / manifest.skills_dir).exists():
        lines.append(f'skill_roots = ["{manifest.skills_dir}"]')
    ext_toml.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return BundleInstallResult(
        extension_path=extensions_dir,
        profile_name=ext_name,
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
        dir_path = bundle_path / dir_name
        count = _count_files(dir_path)
        if count > 0:
            resource_counts[dir_name] = count

    return BundleInspectResult(manifest=manifest, resource_counts=resource_counts)


def list_bundles() -> list[dict[str, Any]]:
    """List installed bundles (extensions with bundle.toml)."""
    extensions_dir = Path.home() / ".config" / "afs" / "extensions"
    if not extensions_dir.exists():
        return []
    bundles: list[dict[str, Any]] = []
    for ext_dir in sorted(extensions_dir.iterdir()):
        bundle_toml = ext_dir / "bundle.toml"
        if bundle_toml.exists():
            try:
                manifest = _read_bundle_toml(bundle_toml)
                bundles.append({
                    "name": manifest.name,
                    "version": manifest.version,
                    "path": str(ext_dir),
                    "description": manifest.description,
                })
            except Exception:
                continue
    return bundles
