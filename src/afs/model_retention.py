"""Read-only, policy-driven model retention auditing.

This module never infers that a model is reviewable from its name or age.
Only an explicit policy entry can produce ``review``; incomplete evidence
fails closed to ``unknown``.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import stat
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import psutil

from .chat_registry import resolve_chat_registry_paths
from .path_safety import assert_no_linklike_components, is_linklike, lexical_absolute
from .toml_compat import tomllib

SCHEMA = "afs.storage.models.v1"
POLICY_SCHEMA = "afs.model-retention.v1"
DEFAULT_RECENT_DAYS = 7
_NANOSECONDS_PER_DAY = 24 * 60 * 60 * 1_000_000_000
MAX_RECENT_DAYS = 3650
MAX_POLICY_BYTES = 1024 * 1024
MAX_POLICY_ENTRIES = 10_000
MAX_DISCOVERY_ENTRIES = 100_000
MAX_DISCOVERY_DEPTH = 32
MAX_ARTIFACT_ENTRIES = 50_000
MAX_DIRECTORY_ENTRIES = 10_000
MAX_MLX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_MLX_INDEX_BYTES = 8 * 1024 * 1024
MAX_REGISTRY_BYTES = 4 * 1024 * 1024
MAX_LSOF_PATHS_PER_COMMAND = 100
MAX_LSOF_COMMANDS = 128
_MIN_TIMESTAMP_NS = 946_684_800 * 1_000_000_000  # 2000-01-01
_MAX_TIMESTAMP_NS = 4_102_444_800 * 1_000_000_000  # 2100-01-01
_POLICY_KEYS = frozenset({"schema", "roots", "artifacts"})
_POLICY_ENTRY_KEYS = frozenset({"path", "decision", "because", "superseded_by"})
_REGISTRY_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "version",
        "defaults",
        "models",
        "routers",
        "profile_defaults",
        "domain_profiles",
        "mode_profiles",
    }
)
_REGISTRY_MODEL_KEYS = frozenset(
    {
        "name",
        "provider",
        "model_id",
        "role",
        "description",
        "tags",
        "parameters",
        "options",
        "system_prompt",
        "system_prompt_path",
        "base_url",
        "api_key_env",
        "thinking_tier",
        "aliases",
        "alias_for",
        "allow_auto_load",
        "deferred_tools",
        "domain",
        "effort",
        "lmstudio_load",
        "mode",
        "native_tools",
        "rollout_block_reason",
        "spawn_only",
        "spawnable_by",
        "tool_profile",
        "visibility",
        "gguf_path",
        "mlx_path",
        "path",
    }
)
_REGISTRY_ROUTER_KEYS = frozenset(
    {
        "name",
        "description",
        "strategy",
        "default_model",
        "models",
        "rules",
    }
)
_REGISTRY_ROUTER_RULE_KEYS = frozenset({"keywords", "model"})
_GGUF_SHARD = re.compile(r"-\d+-of-\d+\.gguf$", re.IGNORECASE)
_SAFETENSORS_SHARD = re.compile(
    r"^(?P<prefix>.*)-(?P<index>\d+)-of-(?P<total>\d+)\.safetensors$",
    re.IGNORECASE,
)

ModelStatus = Literal["keep", "review", "unknown"]
ArtifactKind = Literal["gguf", "mlx"]


class ModelRetentionError(ValueError):
    """Raised when model-retention inputs or policy are invalid."""


@dataclass(frozen=True)
class ActiveModelScan:
    """Result returned by an injected active-runtime reader."""

    available: bool
    references: tuple[str, ...] = ()
    detail: str = ""


ActiveReader = Callable[[], ActiveModelScan]


@dataclass(frozen=True)
class _PolicyEntry:
    path: Path
    decision: Literal["keep", "review"]
    because: str
    superseded_by: Path | None


@dataclass(frozen=True)
class _Policy:
    path: Path | None
    roots: tuple[Path, ...]
    entries: dict[Path, _PolicyEntry]


@dataclass
class _Artifact:
    path: Path
    root: Path
    kind: ArtifactKind
    logical_bytes: int
    allocated_bytes: int
    max_mtime_ns: int
    identities: Counter[tuple[int, int]]
    links: dict[tuple[int, int], int]
    measurement_issues: list[str]


@dataclass(frozen=True)
class _RegistryScan:
    available: bool
    references: tuple[str, ...]
    sources: tuple[str, ...]
    detail: str = ""


def _iso_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / 1_000_000_000, timezone.utc).isoformat()


def _allocated_bytes(path_stat: os.stat_result) -> int:
    """Return allocated bytes where supported, falling back conservatively."""

    blocks = getattr(path_stat, "st_blocks", None)
    if isinstance(blocks, int):
        return max(0, blocks * 512)
    return max(0, path_stat.st_size)


def _bounded_children(path: Path, *, limit: int) -> tuple[list[Path], bool]:
    expected = os.lstat(path)
    if is_linklike(expected) or not stat.S_ISDIR(expected.st_mode):
        raise NotADirectoryError(errno.ENOTDIR, "not a real directory", path)
    if os.scandir not in os.supports_fd:
        fallback_children: list[Path] = []
        truncated = False
        with os.scandir(path) as entries:
            for entry in entries:
                if len(fallback_children) >= limit:
                    fallback_children = []
                    truncated = True
                    break
                fallback_children.append(path / entry.name)
        observed = os.lstat(path)
        if (
            is_linklike(observed)
            or not stat.S_ISDIR(observed.st_mode)
            or (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino)
        ):
            raise OSError(errno.ESTALE, "directory changed while scanning", path)
        return sorted(fallback_children, key=os.fspath), truncated

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    children: list[Path] = []
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            expected.st_dev,
            expected.st_ino,
        ):
            raise OSError(errno.ESTALE, "directory changed while opening", path)
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if len(children) >= limit:
                    return [], True
                children.append(path / entry.name)
        return sorted(children, key=os.fspath), False
    finally:
        os.close(descriptor)


def _expand_path(value: str | os.PathLike[str], *, home: Path) -> Path:
    raw = os.fspath(value)
    if raw == "~":
        return home
    if raw.startswith(f"~{os.sep}"):
        return lexical_absolute(home / raw[2:])
    if raw.startswith("~"):
        raise ModelRetentionError(f"other-user home paths are not supported: {raw}")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = home / candidate
    return lexical_absolute(candidate)


def _validate_home(home: Path) -> Path:
    trusted = lexical_absolute(home)
    try:
        assert_no_linklike_components(trusted, boundary=None, allow_missing=False)
        path_stat = os.lstat(trusted)
    except (OSError, ValueError) as exc:
        raise ModelRetentionError(f"home is not a trusted directory: {trusted}") from exc
    if not stat.S_ISDIR(path_stat.st_mode) or is_linklike(path_stat):
        raise ModelRetentionError(f"home is not a trusted directory: {trusted}")
    return trusted


def _validate_bounded_path(path: Path, *, home: Path) -> Path:
    try:
        path.relative_to(home)
    except ValueError as exc:
        raise ModelRetentionError(f"path escapes the audited home: {path}") from exc
    return path


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ModelRetentionError(f"{field} must be an array of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ModelRetentionError(f"{field} must contain non-empty strings")
        result.append(item)
    return result


def _load_policy(path: Path | None, *, home: Path) -> _Policy:
    if path is None:
        return _Policy(path=None, roots=(), entries={})
    policy_path = _validate_bounded_path(
        _expand_path(path, home=home),
        home=home,
    )
    try:
        assert_no_linklike_components(policy_path, boundary=None, allow_missing=False)
        policy_stat = os.lstat(policy_path)
        if not stat.S_ISREG(policy_stat.st_mode) or is_linklike(policy_stat):
            raise ModelRetentionError(
                f"model-retention policy must be a regular file: {policy_path}"
            )
        if policy_stat.st_size > MAX_POLICY_BYTES:
            raise ModelRetentionError(f"model-retention policy exceeds {MAX_POLICY_BYTES} bytes")
        raw = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    except ModelRetentionError:
        raise
    except (OSError, UnicodeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise ModelRetentionError(f"cannot read model-retention policy: {policy_path}") from exc
    if raw.get("schema") != POLICY_SCHEMA:
        raise ModelRetentionError(f"policy schema must be {POLICY_SCHEMA!r}")
    unknown_policy_keys = sorted(set(raw) - _POLICY_KEYS)
    if unknown_policy_keys:
        raise ModelRetentionError(
            "unknown model-retention policy key(s): " + ", ".join(unknown_policy_keys)
        )

    roots = tuple(
        _validate_bounded_path(_expand_path(item, home=home), home=home)
        for item in _string_list(raw.get("roots", []), field="roots")
    )
    raw_entries = raw.get("artifacts", [])
    if not isinstance(raw_entries, list):
        raise ModelRetentionError("artifacts must be an array of tables")
    if len(raw_entries) > MAX_POLICY_ENTRIES:
        raise ModelRetentionError(f"artifacts exceeds the {MAX_POLICY_ENTRIES}-entry limit")
    entries: dict[Path, _PolicyEntry] = {}
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ModelRetentionError(f"artifacts[{index}] must be a table")
        unknown_entry_keys = sorted(set(raw_entry) - _POLICY_ENTRY_KEYS)
        if unknown_entry_keys:
            raise ModelRetentionError(
                f"unknown artifacts[{index}] key(s): " + ", ".join(unknown_entry_keys)
            )
        raw_path = raw_entry.get("path")
        decision = raw_entry.get("decision")
        because = raw_entry.get("because")
        superseded = raw_entry.get("superseded_by")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ModelRetentionError(f"artifacts[{index}].path must be a non-empty string")
        if decision not in {"keep", "review"}:
            raise ModelRetentionError(f"artifacts[{index}].decision must be keep or review")
        if not isinstance(because, str) or not because.strip():
            raise ModelRetentionError(f"artifacts[{index}].because must be non-empty")
        if superseded is not None and (not isinstance(superseded, str) or not superseded.strip()):
            raise ModelRetentionError(
                f"artifacts[{index}].superseded_by must be a non-empty string"
            )
        artifact_path = _validate_bounded_path(
            _expand_path(raw_path, home=home),
            home=home,
        )
        if artifact_path in entries:
            raise ModelRetentionError(f"duplicate artifact policy path: {artifact_path}")
        replacement = (
            _validate_bounded_path(_expand_path(superseded, home=home), home=home)
            if isinstance(superseded, str)
            else None
        )
        entries[artifact_path] = _PolicyEntry(
            path=artifact_path,
            decision=decision,
            because=because.strip(),
            superseded_by=replacement,
        )
    return _Policy(
        path=policy_path,
        roots=tuple(sorted(set(roots), key=os.fspath)),
        entries=entries,
    )


def default_active_reader(
    artifact_paths: Sequence[Path] = (),
) -> ActiveModelScan:
    """Read process command lines without changing process or server state."""

    references: set[str] = set()
    inaccessible = 0
    runtime_issues: list[str] = []
    saw_lm_studio = False
    saw_ollama = False
    try:
        current_uid = os.getuid() if hasattr(os, "getuid") else None
        current_username = psutil.Process().username() if current_uid is None else ""
        processes = psutil.process_iter()
        for process in processes:
            try:
                if current_uid is not None:
                    same_user = process.uids().effective == current_uid
                else:
                    same_user = process.username() == current_username
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except psutil.AccessDenied:
                inaccessible += 1
                continue
            try:
                command = process.cmdline()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except psutil.AccessDenied:
                if same_user:
                    inaccessible += 1
                continue
            arguments = [str(value) for value in command if value]
            references.update(arguments)
            if arguments:
                joined = " ".join(arguments)
                references.add(joined)
                lowered = joined.lower()
                saw_lm_studio = saw_lm_studio or (
                    "lm studio" in lowered and "crashpad" not in lowered
                )
                saw_ollama = saw_ollama or ("ollama" in lowered and " serve" in lowered)
    except (OSError, psutil.Error) as exc:
        return ActiveModelScan(
            available=False,
            detail=f"process_scan_failed:{type(exc).__name__}",
        )
    for command_name, status_arguments, runtime_seen in (
        ("lms", ("ps", "--json"), saw_lm_studio),
        ("ollama", ("ps",), saw_ollama),
    ):
        executable = shutil.which(command_name)
        if executable is None:
            if runtime_seen:
                runtime_issues.append(f"{command_name}_status_tool_unavailable")
            continue
        try:
            result = subprocess.run(
                [executable, *status_arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            if runtime_seen:
                runtime_issues.append(f"{command_name}_status_query_failed")
            continue
        if result.returncode != 0:
            if runtime_seen:
                runtime_issues.append(f"{command_name}_status_query_failed")
            continue
        if command_name == "lms":
            try:
                payload = json.loads(result.stdout)
            except ValueError:
                if runtime_seen:
                    runtime_issues.append("lms_status_invalid")
                continue
            pending: list[Any] = [payload]
            while pending:
                value = pending.pop()
                if isinstance(value, str) and value:
                    references.add(value)
                elif isinstance(value, dict):
                    pending.extend(value.values())
                elif isinstance(value, list):
                    pending.extend(value)
        else:
            for line in result.stdout.splitlines()[1:]:
                model_name = line.split(maxsplit=1)[0] if line.strip() else ""
                if model_name:
                    references.add(model_name)

    if artifact_paths:
        lsof = shutil.which("lsof")
        if lsof is None:
            runtime_issues.append("open_file_scan_unavailable")
        else:
            files: list[Path] = []
            directories: list[Path] = []
            for path in artifact_paths:
                try:
                    path_stat = os.lstat(path)
                except OSError:
                    runtime_issues.append("open_file_target_unavailable")
                    continue
                if stat.S_ISDIR(path_stat.st_mode):
                    directories.append(path)
                else:
                    files.append(path)
            commands: list[list[str]] = []
            for index in range(0, len(files), MAX_LSOF_PATHS_PER_COMMAND):
                commands.append(
                    [
                        lsof,
                        "-F0n",
                        "--",
                        *(
                            os.fspath(path)
                            for path in files[index : index + MAX_LSOF_PATHS_PER_COMMAND]
                        ),
                    ]
                )
            commands.extend([lsof, "-F0n", "+D", os.fspath(path)] for path in directories)
            if len(commands) > MAX_LSOF_COMMANDS:
                runtime_issues.append("open_file_scan_command_limit_exceeded")
            else:
                for command in commands:
                    try:
                        result = subprocess.run(
                            command,
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=3,
                        )
                    except (OSError, subprocess.SubprocessError):
                        runtime_issues.append("open_file_scan_failed")
                        break
                    if result.returncode not in {0, 1} or result.stderr.strip():
                        runtime_issues.append("open_file_scan_failed")
                        break
                    for field in result.stdout.replace("\0", "\n").splitlines():
                        if field.startswith("n") and len(field) > 1:
                            references.add(field[1:])

    if inaccessible or runtime_issues:
        detail_parts = []
        if inaccessible:
            detail_parts.append(f"{inaccessible}_processes_inaccessible")
        detail_parts.extend(runtime_issues)
        return ActiveModelScan(
            available=False,
            references=tuple(sorted(references)),
            detail=";".join(detail_parts),
        )
    return ActiveModelScan(available=True, references=tuple(sorted(references)))


def _load_registry_scan(
    registry_paths: Sequence[Path] | None,
    *,
    home: Path,
) -> _RegistryScan:
    requested: tuple[Path, ...]
    if registry_paths is None:
        requested = ()
    else:
        requested = tuple(
            _expand_path(path, home=home) if os.fspath(path).startswith("~") else path
            for path in registry_paths
        )
        if not requested:
            return _RegistryScan(
                available=False,
                references=(),
                sources=(),
                detail="no_registry_sources",
            )
        missing: list[str] = []
        for path in requested:
            if path.is_file():
                continue
            if path.is_dir() and any(
                (path / name).is_file() for name in ("chat_registry.toml", "registry.toml")
            ):
                continue
            missing.append(os.fspath(path))
        if missing:
            return _RegistryScan(
                available=False,
                references=(),
                sources=tuple(sorted(missing)),
                detail="explicit_registry_unavailable",
            )
    try:
        sources = resolve_chat_registry_paths(
            registry_paths=list(requested) if registry_paths is not None else None
        )
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        return _RegistryScan(
            available=False,
            references=(),
            sources=tuple(sorted(os.fspath(path) for path in requested)),
            detail=f"registry_resolution_failed:{type(exc).__name__}",
        )
    if not sources:
        return _RegistryScan(
            available=False,
            references=(),
            sources=(),
            detail="no_registry_sources",
        )

    references: set[str] = set()
    for source in sources:
        try:
            source_stat = os.lstat(source)
            if (
                is_linklike(source_stat)
                or not stat.S_ISREG(source_stat.st_mode)
                or source_stat.st_size > MAX_REGISTRY_BYTES
            ):
                raise ModelRetentionError("registry source is not a bounded regular file")
            payload = tomllib.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            return _RegistryScan(
                available=False,
                references=(),
                sources=tuple(sorted(os.fspath(path) for path in sources)),
                detail=f"registry_validation_failed:{type(exc).__name__}",
            )
        if set(payload) - _REGISTRY_TOP_LEVEL_KEYS:
            return _RegistryScan(
                available=False,
                references=(),
                sources=tuple(sorted(os.fspath(path) for path in sources)),
                detail="registry_top_level_keys_invalid",
            )
        models = payload.get("models")
        if not isinstance(models, list):
            return _RegistryScan(
                available=False,
                references=(),
                sources=tuple(sorted(os.fspath(path) for path in sources)),
                detail="registry_models_missing_or_invalid",
            )
        for model in models:
            if not isinstance(model, dict):
                return _RegistryScan(
                    available=False,
                    references=(),
                    sources=tuple(sorted(os.fspath(path) for path in sources)),
                    detail="registry_model_invalid",
                )
            if set(model) - _REGISTRY_MODEL_KEYS:
                return _RegistryScan(
                    available=False,
                    references=(),
                    sources=tuple(sorted(os.fspath(path) for path in sources)),
                    detail="registry_model_keys_invalid",
                )
            model_name = model.get("name")
            model_id = model.get("model_id", model_name)
            if not isinstance(model_name, str) or not model_name.strip():
                return _RegistryScan(
                    available=False,
                    references=(),
                    sources=tuple(sorted(os.fspath(path) for path in sources)),
                    detail="registry_model_invalid",
                )
            if not isinstance(model_id, str) or not model_id.strip():
                return _RegistryScan(
                    available=False,
                    references=(),
                    sources=tuple(sorted(os.fspath(path) for path in sources)),
                    detail="registry_model_invalid",
                )
            references.update((model_name.strip(), model_id.strip()))
            aliases = model.get("aliases", [])
            if not isinstance(aliases, list) or any(
                not isinstance(alias, str) or not alias.strip() for alias in aliases
            ):
                return _RegistryScan(
                    available=False,
                    references=(),
                    sources=tuple(sorted(os.fspath(path) for path in sources)),
                    detail="registry_model_aliases_invalid",
                )
            references.update(alias.strip() for alias in aliases)
            for field in ("gguf_path", "mlx_path", "path"):
                value = model.get(field)
                if isinstance(value, str) and value.strip():
                    references.add(value.strip())
        routers = payload.get("routers", [])
        if not isinstance(routers, list):
            return _RegistryScan(
                available=False,
                references=(),
                sources=tuple(sorted(os.fspath(path) for path in sources)),
                detail="registry_routers_invalid",
            )
        for router in routers:
            if not isinstance(router, dict) or set(router) - _REGISTRY_ROUTER_KEYS:
                return _RegistryScan(
                    available=False,
                    references=(),
                    sources=tuple(sorted(os.fspath(path) for path in sources)),
                    detail="registry_router_invalid",
                )
            router_name = router.get("name")
            if not isinstance(router_name, str) or not router_name.strip():
                return _RegistryScan(
                    available=False,
                    references=(),
                    sources=tuple(sorted(os.fspath(path) for path in sources)),
                    detail="registry_router_invalid",
                )
            for field in ("description", "strategy"):
                value = router.get(field)
                if value is not None and not isinstance(value, str):
                    return _RegistryScan(
                        available=False,
                        references=(),
                        sources=tuple(sorted(os.fspath(path) for path in sources)),
                        detail="registry_router_invalid",
                    )
            default_model = router.get("default_model")
            if default_model is not None:
                if not isinstance(default_model, str) or not default_model.strip():
                    return _RegistryScan(
                        available=False,
                        references=(),
                        sources=tuple(sorted(os.fspath(path) for path in sources)),
                        detail="registry_router_invalid",
                    )
                references.add(default_model.strip())
            router_models = router.get("models", [])
            if not isinstance(router_models, list) or any(
                not isinstance(model, str) or not model.strip() for model in router_models
            ):
                return _RegistryScan(
                    available=False,
                    references=(),
                    sources=tuple(sorted(os.fspath(path) for path in sources)),
                    detail="registry_router_invalid",
                )
            references.update(model.strip() for model in router_models)
            rules = router.get("rules", [])
            if not isinstance(rules, list):
                return _RegistryScan(
                    available=False,
                    references=(),
                    sources=tuple(sorted(os.fspath(path) for path in sources)),
                    detail="registry_router_invalid",
                )
            for rule in rules:
                if not isinstance(rule, dict) or set(rule) - _REGISTRY_ROUTER_RULE_KEYS:
                    return _RegistryScan(
                        available=False,
                        references=(),
                        sources=tuple(sorted(os.fspath(path) for path in sources)),
                        detail="registry_router_rule_invalid",
                    )
                keywords = rule.get("keywords")
                model = rule.get("model")
                if (
                    not isinstance(keywords, list)
                    or any(not isinstance(keyword, str) for keyword in keywords)
                    or not isinstance(model, str)
                    or not model.strip()
                ):
                    return _RegistryScan(
                        available=False,
                        references=(),
                        sources=tuple(sorted(os.fspath(path) for path in sources)),
                        detail="registry_router_rule_invalid",
                    )
                references.add(model.strip())
    return _RegistryScan(
        available=True,
        references=tuple(sorted(references)),
        sources=tuple(sorted(os.fspath(path) for path in sources)),
    )


def _is_mlx_directory(path: Path, *, device: int) -> bool:
    try:
        config_stat = os.lstat(path / "config.json")
        if (
            config_stat.st_dev != device
            or is_linklike(config_stat)
            or not stat.S_ISREG(config_stat.st_mode)
        ):
            return False
        children, _truncated = _bounded_children(
            path,
            limit=MAX_DIRECTORY_ENTRIES,
        )
        for child in children:
            if not child.name.endswith(".safetensors"):
                continue
            child_stat = os.lstat(child)
            if (
                child_stat.st_dev == device
                and not is_linklike(child_stat)
                and stat.S_ISREG(child_stat.st_mode)
            ):
                return True
    except OSError:
        return False
    return False


def _mlx_completeness_issues(path: Path, *, device: int) -> list[str]:
    """Validate bounded MLX shard indexes without reading model weights."""

    issues: set[str] = set()
    try:
        config = path / "config.json"
        config_stat = os.lstat(config)
        if (
            config_stat.st_dev != device
            or is_linklike(config_stat)
            or not stat.S_ISREG(config_stat.st_mode)
            or config_stat.st_size > MAX_MLX_CONFIG_BYTES
        ):
            issues.add("invalid_mlx_config")
        else:
            config_payload = json.loads(config.read_text(encoding="utf-8"))
            if not isinstance(config_payload, dict):
                issues.add("invalid_mlx_config")
    except (OSError, UnicodeError, ValueError):
        issues.add("invalid_mlx_config")
    try:
        children, truncated = _bounded_children(
            path,
            limit=MAX_DIRECTORY_ENTRIES,
        )
    except OSError as exc:
        return [f"mlx_index_scan_failed:{type(exc).__name__}"]
    if truncated:
        issues.add("mlx_directory_entry_limit_exceeded")
    indexes = [child for child in children if child.name.endswith(".safetensors.index.json")]
    weights = [child for child in children if child.name.endswith(".safetensors")]
    if not weights:
        issues.add("missing_mlx_weights")
    for weight in weights:
        try:
            weight_stat = os.lstat(weight)
        except OSError:
            issues.add("missing_mlx_weights")
            continue
        if weight_stat.st_size == 0:
            issues.add("empty_mlx_weight")
    shard_groups: dict[tuple[str, int], set[int]] = {}
    for weight in weights:
        match = _SAFETENSORS_SHARD.match(weight.name)
        if match is None:
            continue
        total = int(match.group("total"))
        shard_number = int(match.group("index"))
        shard_groups.setdefault((match.group("prefix"), total), set()).add(shard_number)
    for (_prefix, total), indexes_seen in shard_groups.items():
        if (
            total < 1
            or total > MAX_DIRECTORY_ENTRIES
            or len(indexes_seen) != total
            or any(index < 1 or index > total for index in indexes_seen)
        ):
            issues.add("incomplete_mlx_shard_set")
    for index in indexes:
        try:
            index_stat = os.lstat(index)
            if (
                index_stat.st_dev != device
                or is_linklike(index_stat)
                or not stat.S_ISREG(index_stat.st_mode)
            ):
                issues.add("invalid_mlx_index")
                continue
            if index_stat.st_size > MAX_MLX_INDEX_BYTES:
                issues.add("mlx_index_too_large")
                continue
            payload = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            issues.add("invalid_mlx_index")
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("weight_map"), dict):
            issues.add("invalid_mlx_index")
            continue
        shard_values = list(payload["weight_map"].values())
        if any(not isinstance(value, str) or not value for value in shard_values):
            issues.add("invalid_mlx_index")
            continue
        shards = {value for value in shard_values if isinstance(value, str)}
        if not shards:
            issues.add("invalid_mlx_index")
            continue
        for shard in shards:
            shard_path = Path(shard)
            if (
                shard_path.is_absolute()
                or ".." in shard_path.parts
                or not shard_path.name.endswith(".safetensors")
            ):
                issues.add("invalid_mlx_shard_path")
                continue
            try:
                shard_stat = os.lstat(path / shard_path)
            except OSError:
                issues.add("missing_mlx_shard")
                continue
            if (
                shard_stat.st_dev != device
                or is_linklike(shard_stat)
                or not stat.S_ISREG(shard_stat.st_mode)
                or shard_stat.st_size == 0
            ):
                issues.add("invalid_mlx_shard")
    return sorted(issues)


def _measure_artifact(
    path: Path,
    kind: ArtifactKind,
    *,
    device: int,
    root: Path,
) -> _Artifact:
    pending = [(path, 0)]
    seen: set[tuple[int, int]] = set()
    identities: Counter[tuple[int, int]] = Counter()
    links: dict[tuple[int, int], int] = {}
    logical = 0
    allocated = 0
    max_mtime = 0
    issues: set[str] = set()
    inspected = 0
    while pending:
        current, depth = pending.pop()
        inspected += 1
        if inspected > MAX_ARTIFACT_ENTRIES:
            issues.add("artifact_entry_limit_exceeded")
            break
        if depth > MAX_DISCOVERY_DEPTH:
            issues.add("artifact_depth_limit_exceeded")
            continue
        try:
            current_stat = os.lstat(current)
        except OSError as exc:
            issues.add(f"measurement_failed:{type(exc).__name__}")
            continue
        if is_linklike(current_stat):
            issues.add("contains_link")
            continue
        if current_stat.st_dev != device:
            issues.add("crosses_device")
            continue
        identity = (current_stat.st_dev, current_stat.st_ino)
        if stat.S_ISREG(current_stat.st_mode):
            identities[identity] += 1
            links[identity] = current_stat.st_nlink
        max_mtime = max(max_mtime, current_stat.st_mtime_ns)
        if identity not in seen:
            seen.add(identity)
            allocated += _allocated_bytes(current_stat)
            if stat.S_ISREG(current_stat.st_mode):
                logical += current_stat.st_size
        if stat.S_ISDIR(current_stat.st_mode):
            try:
                children, truncated = _bounded_children(
                    current,
                    limit=MAX_DIRECTORY_ENTRIES,
                )
            except OSError as exc:
                issues.add(f"measurement_failed:{type(exc).__name__}")
                continue
            if truncated:
                issues.add("artifact_directory_entry_limit_exceeded")
            pending.extend((child, depth + 1) for child in reversed(children))
    if kind == "mlx":
        issues.update(_mlx_completeness_issues(path, device=device))
    elif logical == 0:
        issues.add("empty_artifact")
    elif _GGUF_SHARD.search(path.name):
        issues.add("sharded_gguf_requires_bundle_policy")
    else:
        descriptor: int | None = None
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            header_stat = os.fstat(descriptor)
            if header_stat.st_dev != device or not stat.S_ISREG(header_stat.st_mode):
                issues.add("invalid_gguf_file")
            elif os.read(descriptor, 4) != b"GGUF":
                issues.add("invalid_gguf_header")
        except OSError:
            issues.add("gguf_header_unavailable")
        finally:
            if descriptor is not None:
                os.close(descriptor)
    return _Artifact(
        path=path,
        root=root,
        kind=kind,
        logical_bytes=logical,
        allocated_bytes=allocated,
        max_mtime_ns=max_mtime,
        identities=identities,
        links=links,
        measurement_issues=sorted(issues),
    )


def _discover_root(root: Path, *, home: Path) -> tuple[list[_Artifact], dict[str, Any]]:
    record: dict[str, Any] = {"path": os.fspath(root), "status": "ok", "issues": []}
    try:
        assert_no_linklike_components(root, boundary=home, allow_missing=True)
    except ValueError:
        record["status"] = "blocked"
        record["issues"] = ["root_contains_link"]
        return [], record
    try:
        root_stat = os.lstat(root)
    except FileNotFoundError:
        record["status"] = "missing"
        return [], record
    except OSError as exc:
        record["status"] = "unavailable"
        record["issues"] = [f"root_unavailable:{type(exc).__name__}"]
        return [], record
    if is_linklike(root_stat) or not stat.S_ISDIR(root_stat.st_mode):
        record["status"] = "blocked"
        record["issues"] = ["root_is_not_a_real_directory"]
        return [], record

    device = root_stat.st_dev
    pending = [(root, 0)]
    artifacts: list[_Artifact] = []
    issues: set[str] = set()
    inspected = 0
    while pending:
        current, depth = pending.pop()
        inspected += 1
        if inspected > MAX_DISCOVERY_ENTRIES:
            issues.add("discovery_entry_limit_exceeded")
            break
        if depth > MAX_DISCOVERY_DEPTH:
            issues.add("discovery_depth_limit_exceeded")
            continue
        try:
            current_stat = os.lstat(current)
        except OSError as exc:
            issues.add(f"discovery_failed:{type(exc).__name__}")
            continue
        if is_linklike(current_stat):
            issues.add("skipped_link")
            continue
        if current_stat.st_dev != device:
            issues.add("skipped_cross_device")
            continue
        if stat.S_ISREG(current_stat.st_mode):
            if current.suffix.lower() == ".gguf":
                artifacts.append(_measure_artifact(current, "gguf", device=device, root=root))
            continue
        if not stat.S_ISDIR(current_stat.st_mode):
            continue
        if _is_mlx_directory(current, device=device):
            artifacts.append(_measure_artifact(current, "mlx", device=device, root=root))
            continue
        try:
            children, truncated = _bounded_children(
                current,
                limit=MAX_DIRECTORY_ENTRIES,
            )
        except OSError as exc:
            issues.add(f"discovery_failed:{type(exc).__name__}")
            continue
        if truncated:
            issues.add("directory_entry_limit_exceeded")
        pending.extend((child, depth + 1) for child in reversed(children))
    record["issues"] = sorted(issues)
    if issues:
        record["status"] = "partial"
    return artifacts, record


def _reference_matches(reference: str, artifact: _Artifact) -> bool:
    candidate = reference.strip()
    if not candidate:
        return False
    absolute = os.fspath(artifact.path)
    names = {artifact.path.name}
    if artifact.path.suffix.lower() == ".gguf":
        names.add(artifact.path.stem)
    if candidate in names or candidate == absolute:
        return True
    normalized = candidate.removeprefix("file://").rstrip("/")
    if normalized == absolute:
        return True
    if artifact.kind == "mlx" and (
        normalized.startswith(f"{absolute}{os.sep}") or f"={absolute}{os.sep}" in normalized
    ):
        return True
    return any(normalized.endswith(f"/{name}") or f"={absolute}" in normalized for name in names)


def audit_model_retention(
    home: Path,
    *,
    roots: Sequence[Path] | None = None,
    policy_path: Path | None = None,
    registry_paths: Sequence[Path] | None = None,
    recent_days: int = DEFAULT_RECENT_DAYS,
    now_ns: int | None = None,
    active_reader: ActiveReader | None = None,
) -> dict[str, Any]:
    """Audit model artifacts without mutating files, processes, or services."""

    trusted_home = _validate_home(home)
    if type(recent_days) is not int or recent_days < 0 or recent_days > MAX_RECENT_DAYS:
        raise ModelRetentionError(
            f"recent_days must be an integer from 0 through {MAX_RECENT_DAYS}"
        )
    observed_now_ns = time.time_ns() if now_ns is None else now_ns
    if (
        type(observed_now_ns) is not int
        or observed_now_ns < _MIN_TIMESTAMP_NS
        or observed_now_ns > _MAX_TIMESTAMP_NS
    ):
        raise ModelRetentionError("now_ns must be a timestamp from 2000 through 2100")

    policy = _load_policy(policy_path, home=trusted_home)
    selected_roots = (
        tuple(
            _validate_bounded_path(_expand_path(path, home=trusted_home), home=trusted_home)
            for path in roots
        )
        if roots is not None
        else policy.roots or (trusted_home / "models" / "gguf", trusted_home / "models" / "mlx")
    )
    selected_roots = tuple(sorted(set(selected_roots), key=os.fspath))

    artifacts: list[_Artifact] = []
    root_records: list[dict[str, Any]] = []
    for root in selected_roots:
        discovered, record = _discover_root(root, home=trusted_home)
        artifacts.extend(discovered)
        root_records.append(record)
    root_status = {Path(record["path"]): record["status"] for record in root_records}
    artifacts = list({artifact.path: artifact for artifact in artifacts}.values())
    artifacts.sort(key=lambda item: os.fspath(item.path))
    artifacts_by_path = {artifact.path: artifact for artifact in artifacts}

    observed_links: Counter[tuple[int, int]] = Counter()
    identity_artifacts: Counter[tuple[int, int]] = Counter()
    for artifact in artifacts:
        observed_links.update(artifact.identities)
        identity_artifacts.update(artifact.identities.keys())

    try:
        active_scan = (
            default_active_reader(tuple(artifact.path for artifact in artifacts))
            if active_reader is None
            else active_reader()
        )
    except (OSError, RuntimeError, psutil.Error) as exc:
        active_scan = ActiveModelScan(
            available=False,
            detail=f"active_reader_failed:{type(exc).__name__}",
        )
    if not isinstance(active_scan, ActiveModelScan):
        raise ModelRetentionError("active_reader must return ActiveModelScan")
    registry_scan = _load_registry_scan(registry_paths, home=trusted_home)
    cutoff_ns = observed_now_ns - recent_days * _NANOSECONDS_PER_DAY

    active_matches = {
        artifact.path
        for artifact in artifacts
        if any(_reference_matches(reference, artifact) for reference in active_scan.references)
    }
    registry_matches = {
        artifact.path
        for artifact in artifacts
        if any(_reference_matches(reference, artifact) for reference in registry_scan.references)
    }
    protected: dict[Path, list[str]] = {}
    for artifact in artifacts:
        reasons: list[str] = []
        entry = policy.entries.get(artifact.path)
        if entry is not None and entry.decision == "keep":
            reasons.append("policy_keep")
        if artifact.max_mtime_ns > cutoff_ns:
            reasons.append(f"recent_within_{recent_days}_days")
        if artifact.path in active_matches:
            reasons.append("active_runtime_reference")
        if artifact.path in registry_matches:
            reasons.append("registry_reference")
        protected[artifact.path] = reasons

    artifact_records: list[dict[str, Any]] = []
    for artifact in artifacts:
        entry = policy.entries.get(artifact.path)
        evidence = list(protected[artifact.path])
        blockers: list[str] = []
        external_hardlink = any(
            link_count > observed_links[identity] for identity, link_count in artifact.links.items()
        )
        shared_with_other_artifact = any(
            identity_artifacts[identity] > 1 for identity in artifact.identities
        )
        status: ModelStatus
        if evidence:
            status = "keep"
        elif entry is None:
            status = "unknown"
            blockers.append("no_explicit_policy")
        elif entry.decision != "review":
            status = "unknown"
            blockers.append("policy_decision_not_actionable")
        else:
            replacement = entry.superseded_by
            if not active_scan.available:
                blockers.append("active_scan_unavailable")
            if not registry_scan.available:
                blockers.append("registry_scan_unavailable")
            if root_status.get(artifact.root) != "ok":
                blockers.append("root_inventory_incomplete")
            blockers.extend(artifact.measurement_issues)
            if external_hardlink:
                blockers.append("external_hardlink")
            if shared_with_other_artifact:
                blockers.append("shared_with_other_artifact")
            if replacement is None:
                blockers.append("replacement_not_declared")
            elif replacement == artifact.path:
                blockers.append("replacement_is_same_artifact")
            elif replacement not in protected:
                blockers.append("replacement_not_discovered")
            elif not protected[replacement]:
                blockers.append("replacement_not_protected")
            else:
                replacement_artifact = artifacts_by_path[replacement]
                if root_status.get(replacement_artifact.root) != "ok":
                    blockers.append("replacement_root_inventory_incomplete")
                blockers.extend(
                    f"replacement_{issue}" for issue in replacement_artifact.measurement_issues
                )
            status = "unknown" if blockers else "review"
            if status == "review":
                evidence.append("explicit_review_policy")
                evidence.append("protected_replacement")
                evidence.append("active_scan_clear")
                evidence.append("no_external_hardlinks")
                evidence.append("no_retained_hardlink_peer")
        artifact_records.append(
            {
                "path": os.fspath(artifact.path),
                "kind": artifact.kind,
                "status": status,
                "logical_bytes": artifact.logical_bytes,
                "allocated_bytes_upper_bound": artifact.allocated_bytes,
                "reclaimable_bytes_upper_bound": (
                    artifact.allocated_bytes if status == "review" else 0
                ),
                "mtime": _iso_from_ns(artifact.max_mtime_ns),
                "policy": (
                    {
                        "decision": entry.decision,
                        "because": entry.because,
                        "superseded_by": (
                            os.fspath(entry.superseded_by)
                            if entry.superseded_by is not None
                            else None
                        ),
                    }
                    if entry is not None
                    else None
                ),
                "evidence": sorted(set(evidence)),
                "blocked_reasons": sorted(set(blockers)),
            }
        )

    counts = Counter(record["status"] for record in artifact_records)
    discovered_paths = {artifact.path for artifact in artifacts}
    issues = [
        f"policy_artifact_not_discovered:{path}"
        for path in sorted(set(policy.entries) - discovered_paths, key=os.fspath)
    ]
    for root_record in root_records:
        for issue in root_record["issues"]:
            issues.append(f"root_issue:{root_record['path']}:{issue}")
    if not active_scan.available:
        issues.append(f"active_scan_unavailable:{active_scan.detail or 'unknown'}")
    if not registry_scan.available:
        issues.append(f"registry_scan_unavailable:{registry_scan.detail or 'unknown'}")
    return {
        "schema": SCHEMA,
        "generated_at": _iso_from_ns(observed_now_ns),
        "home": os.fspath(trusted_home),
        "recent_days": recent_days,
        "roots": root_records,
        "policy": {
            "path": os.fspath(policy.path) if policy.path is not None else None,
            "schema": POLICY_SCHEMA if policy.path is not None else None,
            "entries": len(policy.entries),
        },
        "active_scan": {
            "status": "ok" if active_scan.available else "unavailable",
            "detail": active_scan.detail,
            "reference_count": len(active_scan.references),
            "matched_artifacts": [os.fspath(path) for path in sorted(active_matches)],
        },
        "registry_scan": {
            "status": "ok" if registry_scan.available else "unavailable",
            "detail": registry_scan.detail,
            "sources": list(registry_scan.sources),
            "reference_count": len(registry_scan.references),
            "matched_artifacts": [os.fspath(path) for path in sorted(registry_matches)],
        },
        "summary": {
            "artifacts": len(artifact_records),
            "keep": counts["keep"],
            "review": counts["review"],
            "unknown": counts["unknown"],
            "allocated_bytes_upper_bound": sum(
                record["allocated_bytes_upper_bound"] for record in artifact_records
            ),
            "review_reclaimable_bytes_upper_bound": sum(
                record["reclaimable_bytes_upper_bound"] for record in artifact_records
            ),
        },
        "estimate_note": (
            "Allocated and reclaimable bytes are upper bounds; APFS clones, compression, "
            "and filesystem sharing can reduce actual reclaimed space."
        ),
        "issues": sorted(set(issues)),
        "artifacts": artifact_records,
    }
