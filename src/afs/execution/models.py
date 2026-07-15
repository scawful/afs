"""Immutable public models for policy-checked process execution."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

EXECUTION_SCHEMA_VERSION = "1.0"
DEFAULT_TIMEOUT_SECONDS = 300.0
MAX_TIMEOUT_SECONDS = 3600.0
DEFAULT_MAX_OUTPUT_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 10 * 1024 * 1024
DEFAULT_INHERITED_ENV = frozenset({"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"})

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_ARGV_ITEMS = 256
_MAX_ARGUMENT_CHARS = 32768
_MAX_SHELL_CHARS = 1024 * 1024
_MAX_ENV_ITEMS = 128
_MAX_ENV_VALUE_CHARS = 1024 * 1024
_MAX_REASON_CODE_CHARS = 128
_MAX_REASON_CHARS = 2048
_MAX_ERROR_CHARS = 4096


class ExecutionInputError(ValueError):
    """Raised when an execution request is structurally invalid."""


def _require_utf8(value: str, field_name: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ExecutionInputError(
            f"{field_name} must contain only UTF-8-encodable Unicode text"
        ) from exc


def _require_text(value: Any, field_name: str, *, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExecutionInputError(f"{field_name} must be a non-empty string")
    _require_utf8(value, field_name)
    if "\x00" in value:
        raise ExecutionInputError(f"{field_name} must not contain NUL bytes")
    if len(value) > max_length:
        raise ExecutionInputError(f"{field_name} exceeds {max_length} characters")
    return value


def _strict_keys(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ExecutionInputError(f"{label} contains unknown fields: {', '.join(unknown)}")


def _bounded_audit_text(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    digest = hashlib.sha256(value.encode("utf-8", errors="backslashreplace")).hexdigest()
    suffix = f"...<truncated:{digest}>"
    if len(suffix) >= max_length:
        return suffix[:max_length]
    return value[: max_length - len(suffix)] + suffix


@dataclass(frozen=True)
class ArgvCommand:
    """A structured command that is always spawned with ``shell=False``."""

    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized = tuple(self.argv)
        if not normalized:
            raise ExecutionInputError("command.argv must contain at least one argument")
        if len(normalized) > _MAX_ARGV_ITEMS:
            raise ExecutionInputError(
                f"command.argv must contain at most {_MAX_ARGV_ITEMS} arguments"
            )
        for index, value in enumerate(normalized):
            if not isinstance(value, str):
                raise ExecutionInputError(f"command.argv[{index}] must be a string")
            _require_utf8(value, f"command.argv[{index}]")
            if "\x00" in value:
                raise ExecutionInputError(
                    f"command.argv[{index}] must not contain NUL bytes"
                )
            if len(value) > _MAX_ARGUMENT_CHARS:
                raise ExecutionInputError(
                    f"command.argv[{index}] exceeds {_MAX_ARGUMENT_CHARS} characters"
                )
        if not normalized[0].strip():
            raise ExecutionInputError("command.argv[0] must name an executable")
        object.__setattr__(self, "argv", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "argv", "argv": list(self.argv)}


@dataclass(frozen=True)
class LegacyShellCommand:
    """Deprecated shell text, accepted only by an explicitly trusted policy."""

    command: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command",
            _require_text(self.command, "command.command", max_length=_MAX_SHELL_CHARS),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "legacy_shell", "command": self.command}


ExecutionCommand = ArgvCommand | LegacyShellCommand


def _command_from_dict(payload: Any) -> ExecutionCommand:
    if not isinstance(payload, Mapping):
        raise ExecutionInputError("command must be an object")
    kind = payload.get("kind")
    if kind == "argv":
        _strict_keys(payload, {"kind", "argv"}, "command")
        argv = payload.get("argv")
        if not isinstance(argv, list):
            raise ExecutionInputError("command.argv must be an array")
        return ArgvCommand(tuple(argv))
    if kind == "legacy_shell":
        _strict_keys(payload, {"kind", "command"}, "command")
        command = payload.get("command")
        if not isinstance(command, str):
            raise ExecutionInputError("command.command must be a string")
        return LegacyShellCommand(command)
    raise ExecutionInputError("command.kind must be 'argv' or 'legacy_shell'")


@dataclass(frozen=True)
class ExecutionRequest:
    """Untrusted intent submitted for inspection by a trusted policy."""

    command: ExecutionCommand
    caller: str
    purpose: str
    cwd: Path | str
    inherit_env: tuple[str, ...] = ()
    set_env: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    isolation: str = "process"
    network: str = "inherit"
    redact_argv_indices: tuple[int, ...] = ()
    schema_version: str = EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.command, (ArgvCommand, LegacyShellCommand)):
            raise ExecutionInputError("command must be ArgvCommand or LegacyShellCommand")
        if self.schema_version != EXECUTION_SCHEMA_VERSION:
            raise ExecutionInputError(
                f"schema_version must be {EXECUTION_SCHEMA_VERSION!r}"
            )
        object.__setattr__(
            self, "caller", _require_text(self.caller, "caller", max_length=256).strip()
        )
        object.__setattr__(
            self, "purpose", _require_text(self.purpose, "purpose", max_length=2048).strip()
        )

        raw_cwd = str(self.cwd)
        _require_text(raw_cwd, "cwd", max_length=32768)
        object.__setattr__(self, "cwd", Path(raw_cwd).expanduser())

        inherited = tuple(self.inherit_env)
        if len(inherited) > _MAX_ENV_ITEMS:
            raise ExecutionInputError(
                f"inherit_env must contain at most {_MAX_ENV_ITEMS} entries"
            )
        if len(set(inherited)) != len(inherited):
            raise ExecutionInputError("inherit_env must not contain duplicate names")
        for name in inherited:
            if (
                not isinstance(name, str)
                or len(name) > 256
                or _ENV_NAME_RE.fullmatch(name) is None
            ):
                raise ExecutionInputError(f"invalid inherited environment name: {name!r}")
        object.__setattr__(self, "inherit_env", inherited)

        if not isinstance(self.set_env, Mapping):
            raise ExecutionInputError("set_env must be an object")
        if len(self.set_env) > _MAX_ENV_ITEMS:
            raise ExecutionInputError(f"set_env must contain at most {_MAX_ENV_ITEMS} entries")
        normalized_env: dict[str, str] = {}
        for raw_name, raw_value in self.set_env.items():
            if (
                not isinstance(raw_name, str)
                or len(raw_name) > 256
                or _ENV_NAME_RE.fullmatch(raw_name) is None
            ):
                raise ExecutionInputError(f"invalid explicit environment name: {raw_name!r}")
            if not isinstance(raw_value, str):
                raise ExecutionInputError(f"set_env[{raw_name!r}] must be a string")
            _require_utf8(raw_value, f"set_env[{raw_name!r}]")
            if "\x00" in raw_value:
                raise ExecutionInputError(f"set_env[{raw_name!r}] must not contain NUL bytes")
            if len(raw_value) > _MAX_ENV_VALUE_CHARS:
                raise ExecutionInputError(
                    f"set_env[{raw_name!r}] exceeds {_MAX_ENV_VALUE_CHARS} characters"
                )
            normalized_env[raw_name] = raw_value
        object.__setattr__(self, "set_env", MappingProxyType(normalized_env))

        timeout = self.timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < float(timeout) <= MAX_TIMEOUT_SECONDS
        ):
            raise ExecutionInputError(
                f"timeout_seconds must be greater than 0 and at most {MAX_TIMEOUT_SECONDS:g}"
            )
        object.__setattr__(self, "timeout_seconds", float(timeout))

        output_limit = self.max_output_bytes
        if (
            isinstance(output_limit, bool)
            or not isinstance(output_limit, int)
            or not 0 < output_limit <= MAX_OUTPUT_BYTES
        ):
            raise ExecutionInputError(
                f"max_output_bytes must be greater than 0 and at most {MAX_OUTPUT_BYTES}"
            )

        if self.isolation not in {"process", "sandbox", "container"}:
            raise ExecutionInputError(
                "isolation must be 'process', 'sandbox', or 'container'"
            )
        if self.network not in {"inherit", "deny"}:
            raise ExecutionInputError("network must be 'inherit' or 'deny'")

        indices = tuple(self.redact_argv_indices)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise ExecutionInputError("redact_argv_indices must contain integers")
        if any(index < 0 for index in indices):
            raise ExecutionInputError("redact_argv_indices must not contain negative values")
        if any(index > 255 for index in indices):
            raise ExecutionInputError("redact_argv_indices must not contain values above 255")
        if len(set(indices)) != len(indices):
            raise ExecutionInputError("redact_argv_indices must not contain duplicates")
        object.__setattr__(self, "redact_argv_indices", tuple(sorted(indices)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExecutionRequest:
        if not isinstance(payload, Mapping):
            raise ExecutionInputError("execution request must be an object")
        allowed = {
            "schema_version",
            "command",
            "caller",
            "purpose",
            "cwd",
            "inherit_env",
            "set_env",
            "timeout_seconds",
            "max_output_bytes",
            "isolation",
            "network",
            "redact_argv_indices",
        }
        _strict_keys(payload, allowed, "execution request")
        required = {"schema_version", "command", "caller", "purpose", "cwd"}
        missing = sorted(required - set(payload))
        if missing:
            raise ExecutionInputError(
                "execution request is missing required fields: " + ", ".join(missing)
            )
        inherit_env = payload.get("inherit_env", [])
        set_env = payload.get("set_env", {})
        redact = payload.get("redact_argv_indices", [])
        if not isinstance(inherit_env, list):
            raise ExecutionInputError("inherit_env must be an array")
        if not isinstance(set_env, Mapping):
            raise ExecutionInputError("set_env must be an object")
        if not isinstance(redact, list):
            raise ExecutionInputError("redact_argv_indices must be an array")
        schema_version = payload.get("schema_version")
        caller = payload.get("caller")
        purpose = payload.get("purpose")
        cwd = payload.get("cwd")
        if not isinstance(schema_version, str):
            raise ExecutionInputError("schema_version must be a string")
        if not isinstance(caller, str):
            raise ExecutionInputError("caller must be a string")
        if not isinstance(purpose, str):
            raise ExecutionInputError("purpose must be a string")
        if not isinstance(cwd, str):
            raise ExecutionInputError("cwd must be a string")
        return cls(
            schema_version=schema_version,
            command=_command_from_dict(payload.get("command")),
            caller=caller,
            purpose=purpose,
            cwd=cwd,
            inherit_env=tuple(inherit_env),
            set_env=dict(set_env),
            timeout_seconds=payload.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
            max_output_bytes=payload.get(
                "max_output_bytes", DEFAULT_MAX_OUTPUT_BYTES
            ),
            isolation=payload.get("isolation", "process"),
            network=payload.get("network", "inherit"),
            redact_argv_indices=tuple(redact),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "command": self.command.to_dict(),
            "caller": self.caller,
            "purpose": self.purpose,
            "cwd": str(self.cwd),
            "inherit_env": list(self.inherit_env),
            "set_env": dict(self.set_env),
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "isolation": self.isolation,
            "network": self.network,
            "redact_argv_indices": list(self.redact_argv_indices),
        }


@dataclass(frozen=True)
class ExecutionPolicy:
    """Trusted permissions applied to an untrusted execution request."""

    allowed_cwd_roots: tuple[Path, ...]
    allowed_executables: frozenset[str]
    allowed_env: frozenset[str] = frozenset()
    allow_legacy_shell: bool = False

    def __post_init__(self) -> None:
        roots = tuple(Path(root).expanduser().resolve() for root in self.allowed_cwd_roots)
        object.__setattr__(self, "allowed_cwd_roots", roots)
        object.__setattr__(
            self,
            "allowed_executables",
            frozenset(str(value) for value in self.allowed_executables if str(value)),
        )
        object.__setattr__(
            self,
            "allowed_env",
            frozenset(str(value) for value in self.allowed_env if str(value)),
        )


@dataclass(frozen=True)
class ExecutionInspection:
    """Pure policy decision and resolved metadata; inspection never spawns."""

    allowed: bool
    request_sha256: str
    resolved_executable: str
    resolved_cwd: str
    redacted_argv: tuple[str, ...]
    environment_keys: tuple[str, ...]
    environment_sha256: str
    timeout_seconds: float
    max_output_bytes: int
    isolation: str
    network: str
    reason_codes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    schema_version: str = EXECUTION_SCHEMA_VERSION
    _environment: Mapping[str, str] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "redacted_argv", tuple(self.redacted_argv))
        object.__setattr__(self, "environment_keys", tuple(self.environment_keys))
        object.__setattr__(
            self,
            "reason_codes",
            tuple(
                _bounded_audit_text(code, _MAX_REASON_CODE_CHARS)
                for code in self.reason_codes
            ),
        )
        object.__setattr__(
            self,
            "reasons",
            tuple(_bounded_audit_text(reason, _MAX_REASON_CHARS) for reason in self.reasons),
        )
        object.__setattr__(self, "_environment", MappingProxyType(dict(self._environment)))

    @property
    def outcome(self) -> str:
        return "allowed" if self.allowed else "blocked"

    @property
    def env_keys(self) -> tuple[str, ...]:
        """Compatibility alias for the explicit protocol field name."""
        return self.environment_keys

    @property
    def env_sha256(self) -> str:
        """Compatibility alias for the explicit protocol field name."""
        return self.environment_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "allowed": self.allowed,
            "request_sha256": self.request_sha256,
            "resolved_executable": self.resolved_executable,
            "resolved_cwd": self.resolved_cwd,
            "redacted_argv": list(self.redacted_argv),
            "environment_keys": list(self.environment_keys),
            "environment_sha256": self.environment_sha256,
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "isolation": self.isolation,
            "network": self.network,
            "reason_codes": list(self.reason_codes),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ExecutionRecord:
    """Bounded execution result plus a redacted, persistable audit projection."""

    outcome: str
    request_sha256: str
    resolved_executable: str
    resolved_cwd: str
    redacted_argv: tuple[str, ...]
    environment_keys: tuple[str, ...]
    environment_sha256: str
    timeout_seconds: float
    max_output_bytes: int
    isolation: str
    network: str
    caller: str
    purpose: str
    started_at: str
    finished_at: str
    duration_seconds: float
    returncode: int | None
    stdout: str = field(default="", repr=False, compare=False)
    stderr: str = field(default="", repr=False, compare=False)
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    reason_codes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    schema_version: str = EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "redacted_argv", tuple(self.redacted_argv))
        object.__setattr__(self, "environment_keys", tuple(self.environment_keys))
        object.__setattr__(
            self,
            "reason_codes",
            tuple(
                _bounded_audit_text(code, _MAX_REASON_CODE_CHARS)
                for code in self.reason_codes
            ),
        )
        object.__setattr__(
            self,
            "reasons",
            tuple(_bounded_audit_text(reason, _MAX_REASON_CHARS) for reason in self.reasons),
        )

    @property
    def timed_out(self) -> bool:
        return self.outcome == "timed_out"

    @property
    def env_keys(self) -> tuple[str, ...]:
        """Compatibility alias for the explicit protocol field name."""
        return self.environment_keys

    @property
    def env_sha256(self) -> str:
        """Compatibility alias for the explicit protocol field name."""
        return self.environment_sha256

    def audit_dict(self) -> dict[str, Any]:
        """Return the protocol record without raw output or environment values."""
        return {
            "schema_version": self.schema_version,
            "outcome": self.outcome,
            "request_sha256": self.request_sha256,
            "resolved_executable": self.resolved_executable,
            "resolved_cwd": self.resolved_cwd,
            "redacted_argv": list(self.redacted_argv),
            "environment_keys": list(self.environment_keys),
            "environment_sha256": self.environment_sha256,
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "isolation": self.isolation,
            "network": self.network,
            "caller": self.caller,
            "purpose": self.purpose,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "reason_codes": list(self.reason_codes),
            "reasons": list(self.reasons),
            "error": _bounded_audit_text("; ".join(self.reasons), _MAX_ERROR_CHARS)
            if self.outcome in {"blocked", "spawn_error", "timed_out"}
            else "",
        }

    def to_dict(self, *, include_output: bool = False) -> dict[str, Any]:
        payload = self.audit_dict()
        if include_output:
            payload.update({"stdout": self.stdout, "stderr": self.stderr})
        return payload
