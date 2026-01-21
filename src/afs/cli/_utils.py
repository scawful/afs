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
    """Write configuration to TOML file."""
    general = config.general
    lines: list[str] = [
        "[general]",
        f"context_root = \"{general.context_root}\"",
        f"agent_workspaces_dir = \"{general.agent_workspaces_dir}\"",
    ]

    if general.workspace_directories:
        for ws in general.workspace_directories:
            lines.append("")
            lines.append("[[general.workspace_directories]]")
            lines.append(f"path = \"{ws.path}\"")
            if ws.description:
                lines.append(f"description = \"{ws.description}\"")

    lines.append("")
    lines.append("[cognitive]")
    lines.append(f"enabled = {str(config.cognitive.enabled).lower()}")
    lines.append(f"record_emotions = {str(config.cognitive.record_emotions).lower()}")
    lines.append(
        f"record_metacognition = {str(config.cognitive.record_metacognition).lower()}"
    )
    lines.append(f"record_goals = {str(config.cognitive.record_goals).lower()}")
    lines.append(f"record_epistemic = {str(config.cognitive.record_epistemic).lower()}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
