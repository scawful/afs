"""Shared CLI utilities and helpers."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import load_runtime_config_model, resolve_runtime_config_path

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
    from ..manager import AFSManager

    config, _resolved_path = load_runtime_config_model(
        config_path=config_path,
        merge_user=True,
        start_dir=Path.cwd(),
    )
    return AFSManager(config=config)


def resolve_args_config_path(
    args: argparse.Namespace,
    *,
    start_dir: Path | None = None,
) -> Path | None:
    """Resolve config path from argparse args, env, or nearest repo config."""
    raw_config = getattr(args, "config", None)
    return resolve_runtime_config_path(
        Path(raw_config) if raw_config else None,
        start_dir=start_dir or Path.cwd(),
    )


def load_runtime_config_from_args(
    args: argparse.Namespace,
    *,
    merge_user: bool = True,
    start_dir: Path | None = None,
) -> tuple[AFSConfig, Path | None]:
    """Load runtime config using argparse config semantics."""
    raw_config = getattr(args, "config", None)
    config_path = Path(raw_config) if raw_config else None
    return load_runtime_config_model(
        config_path=config_path,
        merge_user=merge_user,
        start_dir=start_dir or Path.cwd(),
    )


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

    def _table() -> tomlkit.items.Table:
        return tomlkit.table()

    def _add_agent_array(
        parent: tomlkit.items.Table,
        key: str,
        agents: list,
    ) -> None:
        if not agents:
            return
        agents_aot = tomlkit.aot()
        for agent in agents:
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
        parent[key] = agents_aot

    # --- [general] ---
    general = config.general
    gen_table = _table()
    gen_table["context_root"] = str(general.context_root)
    if general.python_executable:
        gen_table["python_executable"] = str(general.python_executable)
    gen_table["discovery_ignore"] = list(general.discovery_ignore)
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
    doc["general"] = gen_table

    # --- [plugins] ---
    plugins_table = _table()
    plugins_table["auto_discover"] = config.plugins.auto_discover
    plugins_table["auto_discover_prefixes"] = list(config.plugins.auto_discover_prefixes)
    plugins_table["enabled_plugins"] = list(config.plugins.enabled_plugins)
    plugins_table["plugin_dirs"] = [str(p) for p in config.plugins.plugin_dirs]
    doc["plugins"] = plugins_table

    # --- [extensions] ---
    ext_table = _table()
    ext_table["auto_discover"] = config.extensions.auto_discover
    ext_table["enabled_extensions"] = list(config.extensions.enabled_extensions)
    ext_table["extension_dirs"] = [str(p) for p in config.extensions.extension_dirs]
    doc["extensions"] = ext_table

    # --- [profiles] ---
    prof_table = _table()
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
        _add_agent_array(p_table, "agent_configs", profile.agent_configs)
        prof_table[name] = p_table
    doc["profiles"] = prof_table

    # --- [hooks] ---
    hooks_table = _table()
    hooks_table["before_context_read"] = list(config.hooks.before_context_read)
    hooks_table["after_context_write"] = list(config.hooks.after_context_write)
    hooks_table["before_agent_dispatch"] = list(config.hooks.before_agent_dispatch)
    hooks_table["session_start"] = list(config.hooks.session_start)
    hooks_table["session_end"] = list(config.hooks.session_end)
    hooks_table["user_prompt_submit"] = list(config.hooks.user_prompt_submit)
    hooks_table["turn_started"] = list(config.hooks.turn_started)
    hooks_table["turn_completed"] = list(config.hooks.turn_completed)
    hooks_table["turn_failed"] = list(config.hooks.turn_failed)
    hooks_table["task_created"] = list(config.hooks.task_created)
    hooks_table["task_progress"] = list(config.hooks.task_progress)
    hooks_table["task_completed"] = list(config.hooks.task_completed)
    hooks_table["task_failed"] = list(config.hooks.task_failed)
    doc["hooks"] = hooks_table

    # --- [[directories]] ---
    directories_aot = tomlkit.aot()
    for directory in config.directories:
        dir_table = tomlkit.table()
        dir_table["name"] = directory.name
        dir_table["policy"] = directory.policy.value
        if directory.description:
            dir_table["description"] = directory.description
        if directory.role is not None:
            dir_table["role"] = directory.role.value
        directories_aot.append(dir_table)
    doc["directories"] = directories_aot

    # --- [cognitive] ---
    cog_table = _table()
    cog_table["enabled"] = config.cognitive.enabled
    cog_table["record_emotions"] = config.cognitive.record_emotions
    cog_table["record_metacognition"] = config.cognitive.record_metacognition
    cog_table["record_goals"] = config.cognitive.record_goals
    cog_table["record_epistemic"] = config.cognitive.record_epistemic
    doc["cognitive"] = cog_table

    # --- [orchestrator] ---
    orchestrator_table = _table()
    orchestrator_table["enabled"] = config.orchestrator.enabled
    orchestrator_table["max_agents"] = config.orchestrator.max_agents
    orchestrator_table["auto_routing"] = config.orchestrator.auto_routing
    _add_agent_array(
        orchestrator_table,
        "default_agents",
        config.orchestrator.default_agents,
    )
    doc["orchestrator"] = orchestrator_table

    # --- [services] ---
    services_table = _table()
    services_table["enabled"] = config.services.enabled
    if config.services.services:
        services_subtable = _table()
        for name, service in config.services.services.items():
            service_table = _table()
            service_table["enabled"] = service.enabled
            service_table["auto_start"] = service.auto_start
            if service.command:
                service_table["command"] = list(service.command)
            if service.context_filters:
                service_table["context_filters"] = [
                    str(path) for path in service.context_filters
                ]
            if service.working_directory is not None:
                service_table["working_directory"] = str(service.working_directory)
            if service.environment:
                env_table = _table()
                for key, value in service.environment.items():
                    env_table[key] = value
                service_table["environment"] = env_table
            services_subtable[name] = service_table
        services_table["services"] = services_subtable
    doc["services"] = services_table

    # --- [history] ---
    history_table = _table()
    history_table["enabled"] = config.history.enabled
    history_table["include_payloads"] = config.history.include_payloads
    history_table["max_inline_chars"] = config.history.max_inline_chars
    history_table["payload_dir_name"] = config.history.payload_dir_name
    history_table["redact_sensitive"] = config.history.redact_sensitive
    doc["history"] = history_table

    # --- [memory_export] ---
    memory_export_table = _table()
    memory_export_table["interval_seconds"] = config.memory_export.interval_seconds
    memory_export_table["dataset_output"] = str(config.memory_export.dataset_output)
    if config.memory_export.report_output is not None:
        memory_export_table["report_output"] = str(config.memory_export.report_output)
    memory_export_table["allow_raw"] = config.memory_export.allow_raw
    memory_export_table["allow_raw_tags"] = list(config.memory_export.allow_raw_tags)
    memory_export_table["default_instruction"] = config.memory_export.default_instruction
    memory_export_table["limit"] = config.memory_export.limit
    memory_export_table["require_quality"] = config.memory_export.require_quality
    memory_export_table["min_quality_score"] = config.memory_export.min_quality_score
    memory_export_table["score_profile"] = config.memory_export.score_profile
    memory_export_table["enable_asar"] = config.memory_export.enable_asar
    memory_export_table["auto_start"] = config.memory_export.auto_start
    if config.memory_export.routes:
        routes_aot = tomlkit.aot()
        for route in config.memory_export.routes:
            route_table = tomlkit.table()
            route_table["tags"] = list(route.tags)
            route_table["output"] = str(route.output)
            if route.domain:
                route_table["domain"] = route.domain
            routes_aot.append(route_table)
        memory_export_table["routes"] = routes_aot
    doc["memory_export"] = memory_export_table

    # --- [memory_consolidation] ---
    memory_consolidation_table = _table()
    memory_consolidation_table["enabled"] = config.memory_consolidation.enabled
    memory_consolidation_table["auto_start"] = config.memory_consolidation.auto_start
    memory_consolidation_table["interval_seconds"] = config.memory_consolidation.interval_seconds
    if config.memory_consolidation.report_output is not None:
        memory_consolidation_table["report_output"] = str(
            config.memory_consolidation.report_output
        )
    memory_consolidation_table["entries_filename"] = config.memory_consolidation.entries_filename
    memory_consolidation_table["summary_dir_name"] = config.memory_consolidation.summary_dir_name
    memory_consolidation_table["checkpoint_filename"] = config.memory_consolidation.checkpoint_filename
    memory_consolidation_table["max_events_per_run"] = config.memory_consolidation.max_events_per_run
    memory_consolidation_table["max_events_per_entry"] = config.memory_consolidation.max_events_per_entry
    memory_consolidation_table["include_event_types"] = list(
        config.memory_consolidation.include_event_types
    )
    memory_consolidation_table["write_markdown"] = config.memory_consolidation.write_markdown
    doc["memory_consolidation"] = memory_consolidation_table

    # --- [context_index] ---
    context_index_table = _table()
    context_index_table["enabled"] = config.context_index.enabled
    context_index_table["db_filename"] = config.context_index.db_filename
    context_index_table["auto_index"] = config.context_index.auto_index
    context_index_table["auto_refresh"] = config.context_index.auto_refresh
    context_index_table["include_content"] = config.context_index.include_content
    context_index_table["max_file_size_bytes"] = config.context_index.max_file_size_bytes
    context_index_table["max_content_chars"] = config.context_index.max_content_chars
    context_index_table["decay_hours"] = config.context_index.decay_hours
    doc["context_index"] = context_index_table

    # --- [sensitivity] ---
    sensitivity_table = _table()
    sensitivity_table["never_index"] = list(config.sensitivity.never_index)
    sensitivity_table["never_embed"] = list(config.sensitivity.never_embed)
    sensitivity_table["never_export"] = list(config.sensitivity.never_export)
    doc["sensitivity"] = sensitivity_table

    # --- [hivemind] ---
    hivemind_table = _table()
    hivemind_table["default_ttl_hours"] = config.hivemind.default_ttl_hours
    hivemind_table["reaper_enabled"] = config.hivemind.reaper_enabled
    doc["hivemind"] = hivemind_table

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
    if workspace_path:
        general.workspace_directories = [
            WorkspaceDirectory(path=workspace_path, description=workspace_name)
        ]
    return AFSConfig(general=general)
