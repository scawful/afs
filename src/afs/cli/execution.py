"""Read-only CLI surface for policy-checking typed execution requests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from ..execution import (
    ExecutionInputError,
    ExecutionPolicy,
    ExecutionRequest,
    inspect_execution,
)

_MAX_REQUEST_BYTES = 1024 * 1024


def _reject_nonstandard_number(token: str) -> None:
    raise ValueError(f"non-standard JSON number is not allowed: {token}")


def _read_bounded_stream() -> str:
    text = sys.stdin.read(_MAX_REQUEST_BYTES + 1)
    if len(text.encode("utf-8")) > _MAX_REQUEST_BYTES:
        raise ExecutionInputError(
            f"execution request exceeds {_MAX_REQUEST_BYTES} bytes"
        )
    return text


def _read_request_argument(raw: str) -> str:
    if raw == "-":
        return _read_bounded_stream()
    if raw.lstrip().startswith("{"):
        if len(raw.encode("utf-8")) > _MAX_REQUEST_BYTES:
            raise ExecutionInputError(
                f"execution request exceeds {_MAX_REQUEST_BYTES} bytes"
            )
        return raw

    path = Path(raw).expanduser()
    try:
        stat = path.stat()
    except OSError as exc:
        raise ExecutionInputError(f"cannot read execution request {raw!r}: {exc}") from exc
    if not path.is_file():
        raise ExecutionInputError("execution request path must be a regular file")
    if stat.st_size > _MAX_REQUEST_BYTES:
        raise ExecutionInputError(
            f"execution request exceeds {_MAX_REQUEST_BYTES} bytes"
        )
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ExecutionInputError(f"cannot read execution request {raw!r}: {exc}") from exc


def _load_request(raw: str) -> ExecutionRequest:
    try:
        payload = json.loads(
            _read_request_argument(raw),
            parse_constant=_reject_nonstandard_number,
        )
    except json.JSONDecodeError as exc:
        raise ExecutionInputError(
            f"invalid execution request JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc
    except ValueError as exc:
        if isinstance(exc, ExecutionInputError):
            raise
        raise ExecutionInputError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise ExecutionInputError("execution request JSON must be an object")
    return ExecutionRequest.from_dict(payload)


def _derived_policy(args: argparse.Namespace) -> ExecutionPolicy:
    allowed_roots = tuple(Path(value).expanduser() for value in args.allowed_root)
    allowed_executables = frozenset(args.allowed_executable or [])
    allowed_env = frozenset(args.allowed_env or [])
    return ExecutionPolicy(
        allowed_cwd_roots=allowed_roots,
        allowed_executables=allowed_executables,
        allowed_env=allowed_env,
        allow_legacy_shell=bool(args.allow_legacy_shell),
    )


def execution_inspect_command(args: argparse.Namespace) -> int:
    """Inspect a request and return 0 allowed, 2 invalid, or 3 blocked."""
    try:
        request = _load_request(args.request)
        policy = _derived_policy(args)
        inspection = inspect_execution(request, policy, environ=os.environ)
    except (ExecutionInputError, OSError) as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "valid": False,
                        "allowed": False,
                        "error": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"invalid execution request: {exc}", file=sys.stderr)
        return 2

    payload: dict[str, Any] = inspection.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"allowed: {'yes' if inspection.allowed else 'no'}")
        print(f"request_sha256: {inspection.request_sha256}")
        print(f"resolved_executable: {inspection.resolved_executable or '(unresolved)'}")
        print(f"resolved_cwd: {inspection.resolved_cwd}")
        for code, reason in zip(
            inspection.reason_codes, inspection.reasons, strict=True
        ):
            print(f"reason: {code}: {reason}")
    return 0 if inspection.allowed else 3


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    execution_parser = subparsers.add_parser(
        "execution",
        help="Inspect typed execution requests without launching them.",
    )
    execution_sub = execution_parser.add_subparsers(dest="execution_command")
    inspect_parser = execution_sub.add_parser(
        "inspect",
        help="Resolve and policy-check a request without executing it.",
    )
    inspect_parser.add_argument(
        "--request",
        required=True,
        help="Request JSON file, inline JSON object, or '-' for stdin (max 1 MiB).",
    )
    inspect_parser.add_argument(
        "--allowed-root",
        action="append",
        required=True,
        help="Trusted working-directory root. Repeat to allow multiple roots.",
    )
    inspect_parser.add_argument(
        "--allowed-executable",
        action="append",
        default=[],
        help=(
            "Trusted executable name/path. Repeat as needed; omission denies all "
            "executables."
        ),
    )
    inspect_parser.add_argument(
        "--allowed-env",
        action="append",
        default=[],
        help=(
            "Trusted extra environment key. Repeat as needed; omission denies all "
            "non-baseline keys."
        ),
    )
    inspect_parser.add_argument(
        "--allow-legacy-shell",
        action="store_true",
        help="Allow inspection of deprecated bash -lc requests (never executes).",
    )
    inspect_parser.add_argument("--json", action="store_true", help="Output JSON.")
    inspect_parser.set_defaults(func=execution_inspect_command)
