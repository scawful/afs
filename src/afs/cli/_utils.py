"""Shared CLI utilities and helpers."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..manager import AFSManager
    from ..models import MountType
    from ..schema import AFSConfig


AFS_DIRS = [
    "memory",
    "knowledge",
    "history",
    "scratchpad",
    "tools",
    "hivemind",
    "global",
    "items",
    "monorepo",
]


def _is_studio_root(path: Path) -> bool:
    return (path / "CMakeLists.txt").exists() and (path / "src").exists()


def _default_studio_build_dir(studio_root: Path) -> Path:
    if studio_root.name == "studio" and studio_root.parent.name == "apps":
        repo_root = studio_root.parent.parent
        if (repo_root / "src" / "afs").exists():
            return repo_root / "build" / "studio"
    return studio_root / "build"


def parse_mount_type(value: str) -> MountType:
    """Parse mount type from string."""
    from ..models import MountType
    try:
        return MountType(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Unknown mount type: {value}") from exc


def resolve_studio_root() -> Path:
    """Find the AFS studio source root."""
    candidates: list[Path] = []
    env_studio = os.getenv("AFS_STUDIO_ROOT")
    if env_studio:
        candidates.append(Path(env_studio).expanduser().resolve())
    env_root = os.getenv("AFS_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    trunk_root = os.getenv("TRUNK_ROOT")
    if trunk_root:
        trunk_path = Path(trunk_root).expanduser().resolve()
        candidates.append(trunk_path / "lab" / "afs_studio")
        candidates.append(trunk_path / "lab" / "afs")
    afs_root = Path(__file__).resolve().parents[3]
    candidates.append(afs_root.parent / "afs_studio")
    candidates.append(afs_root)

    for candidate in candidates:
        if _is_studio_root(candidate):
            return candidate
        studio_path = candidate / "apps" / "studio"
        if _is_studio_root(studio_path):
            return studio_path

    raise FileNotFoundError(
        "AFS studio source not found. Set AFS_STUDIO_ROOT or AFS_ROOT."
    )


def studio_binary_name() -> str:
    """Get platform-specific studio binary name."""
    return "afs_studio.exe" if os.name == "nt" else "afs_studio"


def studio_build_dir(root: Path, override: str | None) -> Path:
    """Get studio build directory."""
    return (
        Path(override).expanduser().resolve()
        if override
        else _default_studio_build_dir(root)
    )


def studio_binary_path(build_dir: Path, config: str | None) -> Path:
    """Get path to studio binary."""
    if config:
        candidate = build_dir / config / studio_binary_name()
        if candidate.exists():
            return candidate
    return build_dir / studio_binary_name()


def run_command(cmd: list[str]) -> int:
    """Run a subprocess command."""
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(f"command not found: {cmd[0]}")
        return 1
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


def studio_build(
    root: Path,
    build_dir: Path,
    build_type: str | None,
    config: str | None,
) -> int:
    """Build the AFS studio application."""
    cmake_cmd = ["cmake", "-S", str(root), "-B", str(build_dir)]
    if build_type:
        cmake_cmd.append(f"-DCMAKE_BUILD_TYPE={build_type}")
    status = run_command(cmake_cmd)
    if status != 0:
        return status
    build_cmd = ["cmake", "--build", str(build_dir), "--target", "afs_studio"]
    if config:
        build_cmd.extend(["--config", config])
    return run_command(build_cmd)


def load_manager(config_path: Path | None) -> AFSManager:
    """Load the AFS manager with configuration."""
    from ..config import load_config_model
    from ..manager import AFSManager

    config = load_config_model(config_path=config_path, merge_user=True)
    return AFSManager(config=config)


def resolve_context_paths(
    args: argparse.Namespace, manager: AFSManager
) -> tuple[Path, Path, Path | None, str | None]:
    """Resolve context paths from arguments."""
    project_path = Path(args.path).expanduser().resolve() if args.path else Path.cwd()
    context_root = (
        Path(args.context_root).expanduser().resolve() if args.context_root else None
    )
    context_dir = args.context_dir if args.context_dir else None
    context_path = manager.resolve_context_path(
        project_path,
        context_root=context_root,
        context_dir=context_dir,
    )
    return project_path, context_path, context_root, context_dir


def ensure_context_root(root: Path) -> None:
    """Create context root directory structure."""
    root.mkdir(parents=True, exist_ok=True)
    for name in AFS_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "workspaces").mkdir(parents=True, exist_ok=True)


def write_config(path: Path, config: AFSConfig) -> None:
    """Write configuration to TOML file, preserving unknown sections."""
    import tomlkit

    # Load existing document to preserve unknown sections/comments.
    doc: tomlkit.TOMLDocument
    if path.exists():
        try:
            doc = tomlkit.loads(path.read_text(encoding="utf-8"))
        except Exception:
            doc = tomlkit.document()
    else:
        doc = tomlkit.document()

    # --- [general] ---
    general = config.general
    gen_table = doc.get("general")
    if not isinstance(gen_table, dict):
        gen_table = tomlkit.table()
        doc["general"] = gen_table
    gen_table["context_root"] = str(general.context_root)
    gen_table["agent_workspaces_dir"] = str(general.agent_workspaces_dir)
    if general.mcp_allowed_roots:
        gen_table["mcp_allowed_roots"] = [str(p) for p in general.mcp_allowed_roots]
    if general.workspace_directories:
        ws_aot = tomlkit.aot()
        for ws in general.workspace_directories:
            ws_item = tomlkit.table()
            ws_item["path"] = str(ws.path)
            if ws.description:
                ws_item["description"] = ws.description
            ws_aot.append(ws_item)
        gen_table["workspace_directories"] = ws_aot

    # --- [plugins] ---
    plugins_table = doc.get("plugins")
    if not isinstance(plugins_table, dict):
        plugins_table = tomlkit.table()
        doc["plugins"] = plugins_table
    plugins_table["auto_discover"] = config.plugins.auto_discover
    plugins_table["auto_discover_prefixes"] = list(config.plugins.auto_discover_prefixes)
    plugins_table["enabled_plugins"] = list(config.plugins.enabled_plugins)
    plugins_table["plugin_dirs"] = [str(p) for p in config.plugins.plugin_dirs]

    # --- [extensions] ---
    ext_table = doc.get("extensions")
    if not isinstance(ext_table, dict):
        ext_table = tomlkit.table()
        doc["extensions"] = ext_table
    ext_table["auto_discover"] = config.extensions.auto_discover
    ext_table["enabled_extensions"] = list(config.extensions.enabled_extensions)
    ext_table["extension_dirs"] = [str(p) for p in config.extensions.extension_dirs]

    # --- [profiles] ---
    prof_table = doc.get("profiles")
    if not isinstance(prof_table, dict):
        prof_table = tomlkit.table()
        doc["profiles"] = prof_table
    prof_table["active_profile"] = config.profiles.active_profile
    prof_table["auto_apply"] = config.profiles.auto_apply
    for name, profile in config.profiles.profiles.items():
        p_table = tomlkit.table()
        p_table["inherits"] = list(profile.inherits)
        p_table["knowledge_mounts"] = [str(p) for p in profile.knowledge_mounts]
        p_table["skill_roots"] = [str(p) for p in profile.skill_roots]
        p_table["model_registries"] = [str(p) for p in profile.model_registries]
        p_table["enabled_extensions"] = list(profile.enabled_extensions)
        p_table["policies"] = list(profile.policies)
        p_table["mcp_tools"] = list(profile.mcp_tools)
        p_table["cli_modules"] = list(profile.cli_modules)
        if profile.agent_configs:
            agents_aot = tomlkit.aot()
            for agent in profile.agent_configs:
                a_table = tomlkit.table()
                a_table["name"] = agent.name
                a_table["role"] = agent.role
                a_table["backend"] = agent.backend
                a_table["description"] = agent.description
                a_table["tags"] = list(agent.tags)
                a_table["auto_start"] = agent.auto_start
                a_table["triggers"] = list(agent.triggers)
                a_table["schedule"] = agent.schedule
                a_table["module"] = agent.module
                a_table["watch_paths"] = [str(p) for p in agent.watch_paths]
                a_table["allowed_mounts"] = list(agent.allowed_mounts)
                a_table["allowed_tools"] = list(agent.allowed_tools)
                a_table["workspace_isolated"] = agent.workspace_isolated
                agents_aot.append(a_table)
            p_table["agent_configs"] = agents_aot
        prof_table[name] = p_table

    # --- [hooks] ---
    hooks_table = doc.get("hooks")
    if not isinstance(hooks_table, dict):
        hooks_table = tomlkit.table()
        doc["hooks"] = hooks_table
    hooks_table["before_context_read"] = list(config.hooks.before_context_read)
    hooks_table["after_context_write"] = list(config.hooks.after_context_write)
    hooks_table["before_agent_dispatch"] = list(config.hooks.before_agent_dispatch)

    # --- [cognitive] ---
    cog_table = doc.get("cognitive")
    if not isinstance(cog_table, dict):
        cog_table = tomlkit.table()
        doc["cognitive"] = cog_table
    cog_table["enabled"] = config.cognitive.enabled
    cog_table["record_emotions"] = config.cognitive.record_emotions
    cog_table["record_metacognition"] = config.cognitive.record_metacognition
    cog_table["record_goals"] = config.cognitive.record_goals
    cog_table["record_epistemic"] = config.cognitive.record_epistemic

    path.write_text(tomlkit.dumps(doc), encoding="utf-8")


def build_config(
    context_root: Path,
    workspace_path: Path | None,
    workspace_name: str | None,
) -> AFSConfig:
    """Build an AFS configuration object."""
    from ..schema import AFSConfig, GeneralConfig, WorkspaceDirectory

    general = GeneralConfig()
    general.context_root = context_root
    general.agent_workspaces_dir = context_root / "workspaces"
    if workspace_path:
        general.workspace_directories = [
            WorkspaceDirectory(path=workspace_path, description=workspace_name)
        ]
    return AFSConfig(general=general)
