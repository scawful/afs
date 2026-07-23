"""Extension manifest discovery and loading for AFS."""

from __future__ import annotations

import logging
import os
import re
import stat
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from .schema import AFSConfig, ExtensionsConfig
from .skills import normalize_skill_root
from .toml_compat import tomllib

logger = logging.getLogger(__name__)

EXTENSION_API_VERSION = 1
SUPPORTED_EXTENSION_API_VERSIONS = frozenset({1})
LEGACY_EXTENSION_SCHEMA_VERSION = "0.1"

_MAX_NAME_CHARS = 128
_MAX_TEXT_CHARS = 4096
_MAX_LIST_ITEMS = 256
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_ISSUES = 50
_MAX_ERROR_TEXT_CHARS = 512
_MODULE_NAME_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")
_EXTENSION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_KNOWN_MANIFEST_KEYS = frozenset(
    {
        "name",
        "description",
        "api_version",
        "schema_version",
        "mounts",
        "knowledge_mounts",
        "skill_roots",
        "model_registries",
        "python_paths",
        "import_paths",
        "cli_modules",
        "agent_modules",
        "policies",
        "hooks",
        "manager",
        "mcp_tools",
        "mcp_tools_module",
        "mcp_server",
        "mcp_server_module",
        "context_sources",
    }
)

_PATH_LIST_KEYS = (
    "knowledge_mounts",
    "skill_roots",
    "model_registries",
    "python_paths",
    "import_paths",
)
_MODULE_LIST_KEYS = (
    "cli_modules",
    "agent_modules",
)
_TEXT_LIST_KEYS = ("policies",)
_MOUNT_KEYS = frozenset({"knowledge_mounts", "skill_roots", "model_registries"})
_MANAGER_KEYS = frozenset({"actions"})
_MCP_TOOLS_KEYS = frozenset({"module", "factory", "catalog"})
_MCP_SERVER_KEYS = frozenset({"module", "factory"})
_CONTEXT_SOURCE_KEYS = frozenset({"name", "module", "factory", "description", "kinds"})


class ExtensionManifestError(ValueError):
    """A manifest failed validation; ``issues`` holds actionable messages."""

    def __init__(self, path: Path, issues: list[str]) -> None:
        self.path = Path(path)
        self.issues = list(issues)
        detail = "; ".join(self.issues) or "unknown manifest error"
        rendered_path = repr(str(self.path))
        if len(rendered_path) > _MAX_ERROR_TEXT_CHARS:
            rendered_path = (
                f"{rendered_path[:_MAX_ERROR_TEXT_CHARS]}... "
                f"({len(rendered_path) - _MAX_ERROR_TEXT_CHARS} characters omitted)"
            )
        super().__init__(f"{rendered_path}: {detail}")


def _bounded_text(value: Any, limit: int = _MAX_ERROR_TEXT_CHARS) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... ({len(text) - limit} characters omitted)"


def _bounded_repr(value: Any, limit: int = 160) -> str:
    return _bounded_text(repr(value), limit)


def _limit_issues(messages: list[str]) -> list[str]:
    if len(messages) <= _MAX_ISSUES:
        return messages
    omitted = len(messages) - _MAX_ISSUES
    return [*messages[:_MAX_ISSUES], f"{omitted} additional issue(s) omitted"]


def _unexpected_manifest_error(path: Path, exc: Exception) -> ExtensionManifestError:
    return ExtensionManifestError(
        path,
        [f"internal manifest loader error ({type(exc).__name__})"],
    )


def validate_extension_manifest(
    raw: dict[str, Any],
    path: Path,
) -> tuple[list[str], list[str]]:
    """Validate parsed manifest data; return actionable errors and warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    def unknown_keys(
        table: dict[str, Any],
        known: frozenset[str],
        prefix: str,
    ) -> None:
        for key in table:
            if key in known:
                continue
            rendered_key = _bounded_repr(key)
            suggestion = (
                get_close_matches(key, known, n=1)
                if isinstance(key, str) and len(key) <= 128
                else []
            )
            hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
            warnings.append(f"unknown key {prefix}{rendered_key}{hint}")

    def string_value(
        value: Any,
        label: str,
        *,
        required: bool = False,
        module_name: bool = False,
        identifier: bool = False,
        max_chars: int = _MAX_TEXT_CHARS,
        allow_multiline: bool = False,
    ) -> None:
        if value is None:
            if required:
                errors.append(f"{label} is required")
            return
        if not isinstance(value, str):
            errors.append(f"{label} must be a string (got {type(value).__name__})")
            return
        stripped = value.strip()
        if required and not stripped:
            errors.append(f"{label} must not be empty")
            return
        if "\x00" in value:
            errors.append(f"{label} must not contain NUL bytes")
        if any(
            (ord(character) < 32 and (not allow_multiline or character not in "\t\n\r"))
            or ord(character) == 127
            for character in value
        ):
            errors.append(f"{label} must not contain control characters")
        if len(value) > max_chars:
            errors.append(f"{label} exceeds {max_chars} characters")
        if stripped and module_name and not _MODULE_NAME_RE.fullmatch(stripped):
            errors.append(f"{label} must be a dotted Python module name")
        if stripped and identifier and not _IDENTIFIER_RE.fullmatch(stripped):
            errors.append(f"{label} must be a Python identifier")

    def string_list(
        value: Any,
        label: str,
        *,
        module_names: bool = False,
    ) -> None:
        if value is None:
            return
        if not isinstance(value, list):
            errors.append(f"{label} must be a list (got {type(value).__name__})")
            return
        if len(value) > _MAX_LIST_ITEMS:
            errors.append(f"{label} must contain at most {_MAX_LIST_ITEMS} entries")
        for index, item in enumerate(value[:_MAX_LIST_ITEMS]):
            string_value(
                item,
                f"{label}[{index}]",
                required=True,
                module_name=module_names,
            )

    api_version = raw.get("api_version")
    if api_version is not None:
        if isinstance(api_version, bool) or not isinstance(api_version, int):
            errors.append(
                f"api_version must be an integer (got {type(api_version).__name__}); "
                f"use api_version = {EXTENSION_API_VERSION}"
            )
        elif api_version not in SUPPORTED_EXTENSION_API_VERSIONS:
            supported = ", ".join(str(v) for v in sorted(SUPPORTED_EXTENSION_API_VERSIONS))
            errors.append(
                f"api_version {_bounded_repr(api_version)} is not supported by this AFS "
                f"(supports: {supported}); upgrade AFS or pin a compatible "
                "extension version"
            )

    schema_version = raw.get("schema_version")
    if schema_version is not None:
        if not isinstance(schema_version, str):
            errors.append(f"schema_version must be a string (got {type(schema_version).__name__})")
        elif schema_version.strip() != LEGACY_EXTENSION_SCHEMA_VERSION:
            errors.append(
                f"schema_version {_bounded_repr(schema_version)} is not supported; "
                f"use api_version = {EXTENSION_API_VERSION}"
            )

    string_value(
        raw.get("name"),
        "name",
        required="name" in raw,
        max_chars=_MAX_NAME_CHARS,
    )
    name = raw.get("name")
    if isinstance(name, str) and name.strip():
        if not _EXTENSION_NAME_RE.fullmatch(name.strip()):
            errors.append(
                "name must start with an alphanumeric character and contain only "
                "letters, numbers, '.', '_', or '-'"
            )
    string_value(raw.get("description"), "description", allow_multiline=True)
    string_value(
        raw.get("mcp_tools_module"),
        "mcp_tools_module",
        required="mcp_tools_module" in raw,
        module_name=True,
    )
    string_value(
        raw.get("mcp_server_module"),
        "mcp_server_module",
        required="mcp_server_module" in raw,
        module_name=True,
    )
    for key in _PATH_LIST_KEYS:
        string_list(raw.get(key), key)
    for key in _MODULE_LIST_KEYS:
        string_list(raw.get(key), key, module_names=True)
    for key in _TEXT_LIST_KEYS:
        string_list(raw.get(key), key)

    unknown_keys(raw, _KNOWN_MANIFEST_KEYS, "")

    mounts = raw.get("mounts")
    if mounts is not None:
        if not isinstance(mounts, dict):
            errors.append(f"[mounts] must be a table (got {type(mounts).__name__})")
        else:
            unknown_keys(mounts, _MOUNT_KEYS, "mounts.")
            for key in _MOUNT_KEYS:
                string_list(mounts.get(key), f"mounts.{key}")

    hooks = raw.get("hooks")
    if hooks is not None:
        if not isinstance(hooks, dict):
            errors.append(f"[hooks] must be a table (got {type(hooks).__name__})")
        else:
            for event, commands in hooks.items():
                string_value(event, "hooks event", required=True)
                safe_event = (
                    event
                    if isinstance(event, str)
                    and len(event) <= 128
                    and all(ord(character) >= 32 and ord(character) != 127 for character in event)
                    else _bounded_repr(event)
                )
                string_list(commands, f"hooks.{safe_event}")

    manager = raw.get("manager")
    if manager is not None:
        if not isinstance(manager, dict):
            errors.append(f"[manager] must be a table (got {type(manager).__name__})")
        else:
            unknown_keys(manager, _MANAGER_KEYS, "manager.")
            string_list(manager.get("actions"), "manager.actions")

    mcp_tools_raw = raw.get("mcp_tools")
    if mcp_tools_raw is not None and "mcp_tools_module" in raw:
        errors.append("use either [mcp_tools] or mcp_tools_module, not both")
    if mcp_tools_raw is not None:
        if not isinstance(mcp_tools_raw, dict):
            errors.append(f"[mcp_tools] must be a table (got {type(mcp_tools_raw).__name__})")
        else:
            unknown_keys(mcp_tools_raw, _MCP_TOOLS_KEYS, "mcp_tools.")
            string_value(
                mcp_tools_raw.get("module"),
                "mcp_tools.module",
                required=True,
                module_name=True,
            )
            string_value(
                mcp_tools_raw.get("factory"),
                "mcp_tools.factory",
                identifier=True,
            )
            if "catalog" in mcp_tools_raw:
                catalog = mcp_tools_raw.get("catalog")
                if not isinstance(catalog, str) or catalog.strip().lower() not in {
                    "full",
                    "slim",
                }:
                    errors.append("mcp_tools.catalog must be 'full' or 'slim'")

    mcp_server_raw = raw.get("mcp_server")
    if mcp_server_raw is not None and "mcp_server_module" in raw:
        errors.append("use either [mcp_server] or mcp_server_module, not both")
    if mcp_server_raw is not None:
        if not isinstance(mcp_server_raw, dict):
            errors.append(f"[mcp_server] must be a table (got {type(mcp_server_raw).__name__})")
        else:
            unknown_keys(mcp_server_raw, _MCP_SERVER_KEYS, "mcp_server.")
            string_value(
                mcp_server_raw.get("module"),
                "mcp_server.module",
                required=True,
                module_name=True,
            )
            string_value(
                mcp_server_raw.get("factory"),
                "mcp_server.factory",
                identifier=True,
            )

    context_sources = raw.get("context_sources")
    if context_sources is not None:
        if not isinstance(context_sources, list):
            errors.append(
                f"context_sources must be a list of tables (got {type(context_sources).__name__})"
            )
        else:
            if len(context_sources) > _MAX_LIST_ITEMS:
                errors.append(f"context_sources must contain at most {_MAX_LIST_ITEMS} entries")
            for index, source in enumerate(context_sources[:_MAX_LIST_ITEMS]):
                prefix = f"context_sources[{index}]"
                if not isinstance(source, dict):
                    errors.append(f"{prefix} must be a table")
                    continue
                unknown_keys(source, _CONTEXT_SOURCE_KEYS, f"{prefix}.")
                string_value(source.get("name"), f"{prefix}.name", required=True)
                string_value(
                    source.get("module"),
                    f"{prefix}.module",
                    required=True,
                    module_name=True,
                )
                string_value(
                    source.get("factory"),
                    f"{prefix}.factory",
                    identifier=True,
                )
                string_value(
                    source.get("description"),
                    f"{prefix}.description",
                    allow_multiline=True,
                )
                string_list(source.get("kinds"), f"{prefix}.kinds")

    return _limit_issues(errors), _limit_issues(warnings)


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
    api_version: int = EXTENSION_API_VERSION
    warnings: list[str] = field(default_factory=list)

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


def _as_skill_root_list(items: Any, root: Path) -> list[Path]:
    if not isinstance(items, list):
        return []
    values: list[Path] = []
    for entry in items:
        if not isinstance(entry, (str, Path)):
            continue
        candidate = Path(entry)
        try:
            expanded = candidate.expanduser()
        except (OSError, RuntimeError):
            values.append(candidate)
            continue
        if not expanded.is_absolute():
            expanded = root / expanded
        values.append(normalize_skill_root(expanded))
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
            try:
                marker = str(path.expanduser().resolve())
            except (OSError, RuntimeError, ValueError):
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
            children = sorted(extension_dir.iterdir(), key=lambda path: path.name)
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
            children = sorted(repo_root.iterdir(), key=lambda path: path.name)
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
    """Load and validate a single extension manifest from disk.

    Raises :class:`ExtensionManifestError` with actionable messages when the
    manifest cannot be parsed or fails validation; non-fatal issues are kept
    on ``ExtensionManifest.warnings``.
    """
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ExtensionManifestError(path, ["manifest must be a regular file"])
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            payload = handle.read(_MAX_MANIFEST_BYTES + 1)
        if len(payload) > _MAX_MANIFEST_BYTES:
            raise ExtensionManifestError(
                path,
                [f"manifest exceeds {_MAX_MANIFEST_BYTES} bytes"],
            )
        raw = tomllib.loads(payload.decode("utf-8"))
    except ExtensionManifestError:
        raise
    except OSError as exc:
        raise ExtensionManifestError(
            path, [f"cannot read manifest: {_bounded_repr(str(exc))}"]
        ) from exc
    except (ValueError, UnicodeError) as exc:
        raise ExtensionManifestError(
            path, [f"invalid TOML: {_bounded_repr(str(exc))}"]
        ) from exc
    except Exception as exc:
        raise _unexpected_manifest_error(path, exc) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(raw, dict):
        raise ExtensionManifestError(path, ["manifest must be a TOML table"])

    errors, warnings = validate_extension_manifest(raw, path)
    if errors:
        raise ExtensionManifestError(path, errors)

    api_version = raw.get("api_version")
    if not isinstance(api_version, int) or isinstance(api_version, bool):
        api_version = EXTENSION_API_VERSION

    try:
        root = path.parent.resolve()
        resolved_manifest_path = path.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExtensionManifestError(
            path,
            [f"cannot resolve manifest path: {_bounded_repr(str(exc))}"],
        ) from exc
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        name = root.name
        if len(name) > _MAX_NAME_CHARS or not _EXTENSION_NAME_RE.fullmatch(name):
            raise ExtensionManifestError(
                path,
                ["manifest directory is not a valid extension name; declare a valid name"],
            )
    description = raw.get("description")
    if not isinstance(description, str):
        description = ""

    mounts = raw.get("mounts", {}) if isinstance(raw.get("mounts"), dict) else {}

    try:
        knowledge_mounts = _as_path_list(
            raw.get("knowledge_mounts", mounts.get("knowledge_mounts")),
            root,
        )
        skill_roots = _as_skill_root_list(
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
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExtensionManifestError(
            path,
            [f"cannot resolve manifest field path: {_bounded_repr(str(exc))}"],
        ) from exc
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
            mcp_tools_catalog = str(mcp_tools_raw["catalog"]).strip().lower()
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
        manifest_path=resolved_manifest_path,
        api_version=api_version,
        warnings=warnings,
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
    for manifest_path in _all_manifest_paths(extension_config, extension_dirs):
        try:
            manifest = load_extension_manifest(manifest_path)
        except ExtensionManifestError as exc:
            logger.warning("Skipping extension manifest: %s", exc)
            continue
        except Exception as exc:
            error = _unexpected_manifest_error(manifest_path, exc)
            logger.warning("Skipping extension manifest: %s", error)
            continue
        if manifest.name in discovered:
            continue
        discovered[manifest.name] = manifest.manifest_path
    return dict(sorted(discovered.items()))


def _all_manifest_paths(
    extension_config: ExtensionsConfig,
    extension_dirs: list[Path] | None = None,
) -> list[Path]:
    dirs = _merge_unique_paths(
        extension_dirs or [],
        list(extension_config.extension_dirs),
    )
    return _merge_unique_paths(
        _iter_manifest_paths(dirs, extension_config.manifest_filenames),
        _iter_extension_repo_manifest_paths(
            extension_config.extension_repo_roots,
            extension_config.extension_repo_prefixes,
            extension_config.manifest_filenames,
        ),
    )


def extension_load_report(
    config: AFSConfig | ExtensionsConfig | dict | None = None,
    extra_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    """Load every discoverable manifest, capturing per-manifest failures.

    Returns ``{"extensions": [...], "errors": [...]}`` for surfacing through
    ``afs doctor`` and ``afs plugins`` — the load paths themselves only log
    and skip, so this is the one place broken manifests stay visible.
    """
    extension_config = resolve_extensions_config(config)
    extension_dirs = list(extension_config.extension_dirs)
    if extra_dirs:
        extension_dirs.extend([path.expanduser().resolve() for path in extra_dirs])

    explicitly_enabled = set(extension_config.enabled_extensions)
    extensions: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    for manifest_path in _all_manifest_paths(extension_config, extension_dirs):
        try:
            manifest = load_extension_manifest(manifest_path)
        except ExtensionManifestError as exc:
            errors.append({"path": str(manifest_path), "error": str(exc)})
            continue
        except Exception as exc:
            error = _unexpected_manifest_error(manifest_path, exc)
            errors.append({"path": str(manifest_path), "error": str(error)})
            continue
        if manifest.name in seen:
            continue
        seen.add(manifest.name)
        extensions.append(
            {
                "name": manifest.name,
                "path": str(manifest.manifest_path),
                "api_version": manifest.api_version,
                "enabled": manifest.name in explicitly_enabled,
                "warnings": list(manifest.warnings),
            }
        )
    if extension_config.auto_discover and not explicitly_enabled:
        for entry in extensions:
            entry["enabled"] = True
    discovered_names = {entry["name"] for entry in extensions}
    for name in sorted(explicitly_enabled - discovered_names):
        errors.append(
            {
                "path": "",
                "error": f"enabled extension {name!r} has no discoverable manifest",
            }
        )
    extensions.sort(key=lambda entry: entry["name"])
    return {"extensions": extensions, "errors": errors}


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
        except ExtensionManifestError as exc:
            logger.warning("Not loading extension %r: %s", name, exc)
            continue
        except Exception as exc:
            error = _unexpected_manifest_error(manifest_path, exc)
            logger.warning("Not loading extension %r: %s", name, error)
            continue
    return loaded
