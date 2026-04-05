"""AFS diagnostics engine — check and auto-fix common issues.

Used by ``afs doctor`` and the MCP server startup validation. Each check
function returns a :class:`DiagnosticResult` that optionally carries a
callable fix.

Usage::

    from afs.diagnostics import run_all_checks
    results = run_all_checks(auto_fix=True)
    for r in results:
        print(f"[{r.status}] {r.name}: {r.message}")
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .context_paths import resolve_agent_output_root
from .models import MountType

_REQUIRED_CONTEXT_MOUNTS = (
    MountType.MEMORY,
    MountType.KNOWLEDGE,
    MountType.SCRATCHPAD,
)
_SERVICE_DIAGNOSTIC_NAMES = (
    "context-warm",
    "context-watch",
    "agent-supervisor",
    "history-memory",
)
DOCTOR_SNAPSHOT_JSON = "doctor_snapshot.json"


@dataclass
class DiagnosticResult:
    """Result of a single diagnostic check."""

    name: str
    status: Literal["ok", "warn", "error"]
    message: str
    fix_available: bool = False
    fix_description: str = ""
    fix_applied: bool = False

    # Internal — not serialised.  Holds a callable that performs the fix.
    _fix_fn: Callable[[], str] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "fix_available": self.fix_available,
            "fix_description": self.fix_description,
            "fix_applied": self.fix_applied,
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _load_runtime(config_path: Path | None = None):
    from .config import load_runtime_config_model
    from .core import find_root, resolve_context_root
    from .manager import AFSManager

    config, _resolved_config_path = load_runtime_config_model(
        config_path=config_path,
        merge_user=True,
        start_dir=Path.cwd(),
    )
    linked_root = None if config_path else find_root(Path.cwd())
    context_root = resolve_context_root(config, linked_root)
    manager = AFSManager(config=config)
    return config, manager, context_root


def check_config(config_path: Path | None = None) -> DiagnosticResult:
    """Verify the AFS config loads without errors."""
    try:
        config, _manager, _context_root = _load_runtime(config_path)
    except Exception as exc:
        return DiagnosticResult(
            name="config",
            status="error",
            message=f"Config load failed: {exc}",
        )

    return DiagnosticResult(
        name="config",
        status="ok",
        message=(
            "Config loaded "
            f"(context_root={config.general.context_root}, "
            f"profile={config.profiles.active_profile})"
        ),
    )


def check_context_root(config_path: Path | None = None) -> DiagnosticResult:
    """Verify context root exists and has required mount directories."""
    try:
        _config, manager, context_root = _load_runtime(config_path)
    except Exception as exc:
        return DiagnosticResult(
            name="context_root",
            status="error",
            message=f"Cannot resolve context root: {exc}",
        )

    if not context_root.exists():
        def _fix() -> str:
            manager.ensure(context_root=context_root)
            return f"Created context root with mount dirs at {context_root}"

        return DiagnosticResult(
            name="context_root",
            status="error",
            message=f"Context root missing: {context_root}",
            fix_available=True,
            fix_description=f"afs context init at {context_root}",
            _fix_fn=_fix,
        )

    missing: list[tuple[MountType, Path]] = []
    for mount_type in _REQUIRED_CONTEXT_MOUNTS:
        mount_dir = manager.resolve_mount_root(context_root, mount_type)
        if not mount_dir.exists():
            missing.append((mount_type, mount_dir))

    if missing:
        def _fix_mounts() -> str:
            for _mount_type, mount_dir in missing:
                mount_dir.mkdir(parents=True, exist_ok=True)
            manager._ensure_metadata(context_root, Path.cwd().resolve())
            manager._ensure_cognitive_scaffold(context_root)
            return (
                "Created missing mount dirs: "
                + ", ".join(f"{mount_type.value}={mount_dir.name}" for mount_type, mount_dir in missing)
            )

        return DiagnosticResult(
            name="context_root",
            status="warn",
            message=(
                f"Missing mount dirs in {context_root}: "
                + ", ".join(f"{mount_type.value}={mount_dir.name}" for mount_type, mount_dir in missing)
            ),
            fix_available=True,
            fix_description=(
                "Create required mount dirs "
                + ", ".join(f"{mount_type.value}={mount_dir.name}" for mount_type, mount_dir in missing)
            ),
            _fix_fn=_fix_mounts,
        )

    return DiagnosticResult(
        name="context_root",
        status="ok",
        message=f"Context root OK: {context_root}",
    )


def check_context_health(config_path: Path | None = None) -> DiagnosticResult:
    """Verify mount health, profile-managed mounts, and provenance integrity."""
    try:
        config, manager, context_root = _load_runtime(config_path)
    except Exception as exc:
        return DiagnosticResult(
            name="context_health",
            status="error",
            message=f"Cannot evaluate context health: {exc}",
        )

    if not context_root.exists():
        return DiagnosticResult(
            name="context_health",
            status="warn",
            message=f"Context root missing, health check skipped: {context_root}",
        )

    try:
        health = manager.context_health(context_root)
    except Exception as exc:
        return DiagnosticResult(
            name="context_health",
            status="error",
            message=f"Context health check failed: {exc}",
        )

    issue_counts = {
        "broken": len(health["broken_mounts"]),
        "duplicates": len(health["duplicate_mount_sources"]),
        "profile_missing": len(health["profile"]["missing_mounts"]),
        "profile_missing_sources": len(health["profile"]["missing_sources"]),
        "profile_mismatched": len(health["profile"]["mismatched_mounts"]),
        "untracked": len(health["provenance"]["untracked_mounts"]),
        "stale_provenance": len(health["provenance"]["stale_records"]),
    }
    active_issues = [f"{name}={count}" for name, count in issue_counts.items() if count]

    if not active_issues:
        return DiagnosticResult(
            name="context_health",
            status="ok",
            message=(
                f"Context health OK (profile={health['profile']['name']}, "
                f"symlink_mounts={health['symlink_mounts']})"
            ),
        )

    def _fix() -> str:
        result = manager.repair_context(
            context_root,
            rebuild_index=config.context_index.enabled,
        )
        actions = result.get("applied_actions") or result.get("actions") or []
        if actions:
            return "; ".join(actions)
        return "Repaired context state"

    major_issues = (
        issue_counts["broken"]
        or issue_counts["duplicates"]
        or issue_counts["profile_missing"]
        or issue_counts["profile_missing_sources"]
        or issue_counts["profile_mismatched"]
    )
    return DiagnosticResult(
        name="context_health",
        status="error" if major_issues else "warn",
        message="Context issues detected: " + ", ".join(active_issues),
        fix_available=True,
        fix_description="Run context repair and rebuild stale index state",
        _fix_fn=_fix,
    )


def check_dependencies() -> DiagnosticResult:
    """Check optional dependencies that enable key features."""
    missing: list[str] = []
    available: list[str] = []

    checks = [
        ("google.genai", "google-genai", "Gemini embeddings"),
        ("psutil", "psutil", "system metrics"),
        ("httpx", "httpx", "HTTP embedding fallback"),
    ]

    for module_name, package, purpose in checks:
        try:
            importlib.import_module(module_name)
            available.append(package)
        except ImportError:
            missing.append(f"{package} ({purpose})")

    if missing:
        return DiagnosticResult(
            name="dependencies",
            status="warn",
            message=f"Optional deps missing: {', '.join(missing)}",
            fix_description=f"pip install {' '.join(m.split(' ')[0] for m in missing)}",
        )

    return DiagnosticResult(
        name="dependencies",
        status="ok",
        message=f"All optional deps available ({len(available)})",
    )


def check_mcp_registration() -> DiagnosticResult:
    """Check that AFS MCP server is registered in at least one client."""
    try:
        from .health.mcp_registration import find_afs_mcp_registrations

        registrations = find_afs_mcp_registrations()
    except Exception as exc:
        return DiagnosticResult(
            name="mcp_registration",
            status="warn",
            message=f"Cannot check MCP registrations: {exc}",
        )

    clients_registered = [
        client for client, paths in registrations.items() if paths
    ]

    if not clients_registered:
        return DiagnosticResult(
            name="mcp_registration",
            status="warn",
            message="AFS MCP not registered in any client",
            fix_description="Run: afs gemini setup  (or afs gemini setup --scope project)",
        )

    return DiagnosticResult(
        name="mcp_registration",
        status="ok",
        message=f"MCP registered in: {', '.join(clients_registered)}",
    )


def check_embedding_indexes(config_path: Path | None = None) -> DiagnosticResult:
    """Check embedding index health across knowledge mounts."""
    try:
        from .config import load_config_model
        from .profiles import resolve_active_profile

        config = load_config_model(config_path=config_path, merge_user=True)
        profile = resolve_active_profile(config)
    except Exception as exc:
        return DiagnosticResult(
            name="embeddings",
            status="warn",
            message=f"Cannot resolve profile for embedding check: {exc}",
        )

    knowledge_mounts = profile.knowledge_mounts
    if not knowledge_mounts:
        return DiagnosticResult(
            name="embeddings",
            status="ok",
            message="No knowledge mounts configured",
        )

    total_indexes = 0
    stale_indexes = 0
    missing_indexes: list[str] = []
    import time

    for mount in knowledge_mounts:
        mount_path = Path(mount).expanduser().resolve()
        if not mount_path.exists():
            continue
        index_file = mount_path / "embedding_index.json"
        if not index_file.exists():
            missing_indexes.append(str(mount_path))
            continue
        total_indexes += 1
        try:
            age = time.time() - index_file.stat().st_mtime
            if age > 7 * 24 * 3600:  # 7 days
                stale_indexes += 1
        except OSError:
            pass

    if missing_indexes:
        return DiagnosticResult(
            name="embeddings",
            status="warn",
            message=f"Missing embedding indexes in: {', '.join(missing_indexes)}",
            fix_description="Run: afs embeddings index --knowledge-path <path> --provider none --include '*.md'",
        )

    if stale_indexes:
        return DiagnosticResult(
            name="embeddings",
            status="warn",
            message=f"{stale_indexes}/{total_indexes} embedding indexes are stale (>7 days)",
            fix_description="Run: afs embeddings index --knowledge-path <path> --provider <provider> --include '*.md'",
        )

    return DiagnosticResult(
        name="embeddings",
        status="ok",
        message=f"{total_indexes} embedding index(es) healthy",
    )


def check_context_index(config_path: Path | None = None) -> DiagnosticResult:
    """Check whether the active context index exists and matches filesystem state."""
    try:
        config, manager, context_root = _load_runtime(config_path)
    except Exception as exc:
        return DiagnosticResult(
            name="context_index",
            status="error",
            message=f"Cannot resolve context index state: {exc}",
        )

    if not config.context_index.enabled:
        return DiagnosticResult(
            name="context_index",
            status="ok",
            message="Context index disabled in config",
        )

    if not context_root.exists():
        return DiagnosticResult(
            name="context_index",
            status="warn",
            message=f"Context root missing, index check skipped: {context_root}",
        )

    try:
        from .context_index import ContextSQLiteIndex

        index = ContextSQLiteIndex(manager, context_root)
    except Exception as exc:
        return DiagnosticResult(
            name="context_index",
            status="error",
            message=f"Context index unavailable: {exc}",
        )

    def _fix() -> str:
        rebuilt = manager.rebuild_context_index(context_root)
        written = int(rebuilt.get("rows_written", 0))
        deleted = int(rebuilt.get("rows_deleted", 0))
        return f"Rebuilt context index ({written} rows written, {deleted} rows deleted)"

    if not index.has_entries():
        diff = index.diff()
        pending = sum(
            len(diff.get(bucket, []))
            for bucket in ("added", "modified", "deleted")
        )
        if pending == 0:
            return DiagnosticResult(
                name="context_index",
                status="ok",
                message=f"Context index empty with no indexable content yet: {index.db_path}",
            )
        return DiagnosticResult(
            name="context_index",
            status="warn",
            message=f"Context index is empty: {index.db_path}",
            fix_available=True,
            fix_description="Rebuild the context index",
            _fix_fn=_fix,
        )

    if index.needs_refresh():
        return DiagnosticResult(
            name="context_index",
            status="warn",
            message=f"Context index is stale: {index.db_path}",
            fix_available=True,
            fix_description="Rebuild the context index to match filesystem changes",
            _fix_fn=_fix,
        )

    summary = index.summary()
    return DiagnosticResult(
        name="context_index",
        status="ok",
        message=f"Context index healthy ({summary.rows_written} entries at {index.db_path})",
    )


def check_extensions(config_path: Path | None = None) -> DiagnosticResult:
    """Check that enabled extensions load without errors."""
    try:
        from .config import load_config_model
        from .plugins import load_enabled_extensions

        config = load_config_model(config_path=config_path, merge_user=True)
        extensions = load_enabled_extensions(config=config)
    except Exception as exc:
        return DiagnosticResult(
            name="extensions",
            status="warn",
            message=f"Extension loading failed: {exc}",
        )

    if not extensions:
        return DiagnosticResult(
            name="extensions",
            status="ok",
            message="No extensions configured",
        )

    return DiagnosticResult(
        name="extensions",
        status="ok",
        message=f"{len(extensions)} extension(s) loaded: {', '.join(extensions.keys())}",
    )


def check_services(config_path: Path | None = None) -> DiagnosticResult:
    """Check configured auto-start background services."""
    try:
        config, _manager, _context_root = _load_runtime(config_path)
        from .services.manager import ServiceManager

        service_manager = ServiceManager(config=config, config_path=config_path)
    except Exception as exc:
        return DiagnosticResult(
            name="services",
            status="warn",
            message=f"Cannot inspect service state: {exc}",
        )

    configured: list[str] = []
    stopped: list[str] = []
    states: list[str] = []

    for name in _SERVICE_DIAGNOSTIC_NAMES:
        definition = service_manager.get_definition(name)
        if definition is None:
            continue
        if not definition.run_at_load:
            continue
        configured.append(name)
        status = service_manager.status(name)
        states.append(f"{name}={status.state.value}")
        if status.state.value != "running":
            stopped.append(name)

    if not configured:
        return DiagnosticResult(
            name="services",
            status="ok",
            message="No auto-start maintenance services configured",
        )

    if stopped:
        return DiagnosticResult(
            name="services",
            status="warn",
            message="Configured auto-start services not running: " + ", ".join(states),
            fix_description="Run `afs services status` and start or repair the stopped services",
        )

    return DiagnosticResult(
        name="services",
        status="ok",
        message="Auto-start services running: " + ", ".join(states),
    )


def check_python_environment() -> DiagnosticResult:
    """Check the Python environment is suitable for AFS."""
    version = sys.version_info
    issues: list[str] = []

    if version < (3, 10):
        issues.append(f"Python {version.major}.{version.minor} is below minimum 3.10")

    # Check that the afs package is properly installed.
    try:
        import afs  # noqa: F401
    except ImportError:
        issues.append("afs package not importable (pip install -e .)")

    # Check venv or system python.
    in_venv = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )
    if not in_venv:
        issues.append("Not running in a virtual environment")

    if issues:
        return DiagnosticResult(
            name="python",
            status="warn" if "below minimum" not in str(issues) else "error",
            message="; ".join(issues),
        )

    return DiagnosticResult(
        name="python",
        status="ok",
        message=f"Python {version.major}.{version.minor}.{version.micro} in venv",
    )


def check_mcp_server(config_path: Path | None = None) -> DiagnosticResult:
    """Check that the MCP server module loads and can build its registry."""
    try:
        from .config import load_config_model
        from .manager import AFSManager
        from .mcp_server import build_mcp_registry

        config = load_config_model(config_path=config_path, merge_user=True)
        manager = AFSManager(config=config)
        registry = build_mcp_registry(manager)
    except Exception as exc:
        return DiagnosticResult(
            name="mcp_server",
            status="error",
            message=f"MCP server build failed: {exc}",
        )

    errors = registry.load_errors
    if errors:
        error_summary = "; ".join(f"{k}: {v}" for k, v in list(errors.items())[:3])
        return DiagnosticResult(
            name="mcp_server",
            status="warn",
            message=f"{len(registry.tools)} tools, {len(errors)} load error(s): {error_summary}",
        )

    return DiagnosticResult(
        name="mcp_server",
        status="ok",
        message=f"{len(registry.tools)} tools registered",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_checks(
    checks: list[tuple[str, Callable[[], DiagnosticResult]]],
    *,
    auto_fix: bool = False,
) -> list[DiagnosticResult]:
    results: list[DiagnosticResult] = []
    for name, check_fn in checks:
        try:
            result = check_fn()
        except Exception as exc:
            result = DiagnosticResult(
                name=name,
                status="error",
                message=f"Check crashed: {exc}",
            )

        if auto_fix and result.fix_available and result._fix_fn is not None:
            try:
                fix_msg = result._fix_fn()
                result.fix_applied = True
                result.message = f"FIXED: {fix_msg}"
                result.status = "ok"
            except Exception as exc:
                result.message += f" (fix failed: {exc})"

        results.append(result)
    return results


def run_all_checks(
    config_path: Path | None = None,
    *,
    auto_fix: bool = False,
) -> list[DiagnosticResult]:
    """Run all diagnostic checks in dependency order.

    When *auto_fix* is True, applies available fixes and updates the result
    status accordingly.
    """
    checks: list[tuple[str, Callable[[], DiagnosticResult]]] = [
        ("python", lambda: check_python_environment()),
        ("config", lambda: check_config(config_path)),
        ("context_root", lambda: check_context_root(config_path)),
        ("context_health", lambda: check_context_health(config_path)),
        ("services", lambda: check_services(config_path)),
        ("dependencies", lambda: check_dependencies()),
        ("mcp_registration", lambda: check_mcp_registration()),
        ("embeddings", lambda: check_embedding_indexes(config_path)),
        ("extensions", lambda: check_extensions(config_path)),
        ("context_index", lambda: check_context_index(config_path)),
        ("mcp_server", lambda: check_mcp_server(config_path)),
    ]
    return _run_checks(checks, auto_fix=auto_fix)


def run_startup_checks(config_path: Path | None = None) -> list[DiagnosticResult]:
    """Run a lightweight subset of checks suitable for MCP server startup."""
    checks: list[tuple[str, Callable[[], DiagnosticResult]]] = [
        ("python", lambda: check_python_environment()),
        ("config", lambda: check_config(config_path)),
        ("context_root", lambda: check_context_root(config_path)),
        ("context_health", lambda: check_context_health(config_path)),
        ("dependencies", lambda: check_dependencies()),
        ("extensions", lambda: check_extensions(config_path)),
        ("context_index", lambda: check_context_index(config_path)),
        ("mcp_server", lambda: check_mcp_server(config_path)),
    ]
    return _run_checks(checks, auto_fix=False)


def format_results_text(results: list[DiagnosticResult]) -> str:
    """Render diagnostic results as human-readable text."""
    lines = ["AFS Doctor", ""]
    any_issues = False
    for r in results:
        icon = {"ok": "  ok", "warn": "WARN", "error": " ERR"}[r.status]
        lines.append(f"  [{icon}] {r.name}: {r.message}")
        if r.fix_available and not r.fix_applied:
            lines.append(f"         fix: {r.fix_description}")
            any_issues = True
        if r.status != "ok" and not r.fix_applied:
            any_issues = True

    lines.append("")
    if any_issues:
        lines.append("Run `afs doctor --fix` to auto-apply available fixes.")
    else:
        lines.append("All checks passed.")

    return "\n".join(lines)


def format_results_json(results: list[DiagnosticResult]) -> str:
    """Render diagnostic results as JSON."""
    return json.dumps(
        {"checks": [r.to_dict() for r in results]},
        indent=2,
    )


def write_doctor_snapshot(
    config_path: Path | None = None,
    *,
    results: list[DiagnosticResult] | None = None,
) -> Path | None:
    """Write the latest doctor snapshot into the active context agent output dir."""
    try:
        config, _manager, context_root = _load_runtime(config_path)
    except Exception:
        return None

    resolved_results = results if results is not None else run_all_checks(config_path=config_path)
    counts = {
        "ok": sum(1 for result in resolved_results if result.status == "ok"),
        "warn": sum(1 for result in resolved_results if result.status == "warn"),
        "error": sum(1 for result in resolved_results if result.status == "error"),
    }
    overall_status = "ok"
    if counts["error"]:
        overall_status = "error"
    elif counts["warn"]:
        overall_status = "warn"

    output_root = resolve_agent_output_root(context_root, config=config)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / DOCTOR_SNAPSHOT_JSON
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "name": "doctor_snapshot",
        "status": overall_status,
        "started_at": now,
        "finished_at": now,
        "duration_seconds": 0.0,
        "metrics": counts,
        "notes": [
            result.message
            for result in resolved_results
            if result.status != "ok"
        ][:5],
        "payload": {
            "checks": [result.to_dict() for result in resolved_results],
            "context_root": str(context_root),
            "config_path": str(config_path) if config_path else None,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
