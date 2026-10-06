"""Read-only diagnostics for extension declarations core cannot dispatch."""

from __future__ import annotations

from pathlib import Path

from .extensions import load_extension_manifest
from .toml_compat import tomllib

# Keep aligned with the session event CLI and run_grounding_hooks call sites.
CORE_HOOK_EVENTS = frozenset({
    "before_context_read", "after_context_write", "before_agent_dispatch",
    "session_start", "session_end", "user_prompt_submit", "turn_started",
    "turn_completed", "turn_failed", "task_created", "task_progress",
    "task_completed", "task_failed", "verification_recorded",
})


def extension_dispatch_warnings(manifest_path: Path) -> list[str]:
    """Find inert hooks and AFS package entry points without importing code."""
    manifest = load_extension_manifest(manifest_path)
    warnings = [
        f"hook {event!r} is not emitted by core AFS; it needs an explicit extension caller"
        for event in sorted(set(manifest.hooks) - CORE_HOOK_EVENTS)
    ]
    project_path = manifest.root / "pyproject.toml"
    if project_path.is_file():
        try:
            if project_path.stat().st_size > 1024 * 1024:
                warnings.append("pyproject.toml exceeds diagnostic size limit")
                return warnings
            raw = tomllib.loads(project_path.read_text(encoding="utf-8"))
            project = raw.get("project", {})
            groups = project.get("entry-points", {}) if isinstance(project, dict) else {}
            if isinstance(groups, dict):
                for group in sorted(groups):
                    if group == "afs" or group.startswith(("afs.", "afs_", "afs-")):
                        warnings.append(
                            f"package entry-point group {group!r} is not read by AFS; "
                            "declare its modules in extension.toml"
                        )
        except (OSError, ValueError) as exc:
            warnings.append(f"cannot inspect pyproject.toml: {type(exc).__name__}")
    return warnings
