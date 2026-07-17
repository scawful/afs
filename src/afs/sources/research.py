"""Bounded execution of explicitly selected extension research providers."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..execution import (
    MAX_OUTPUT_BYTES,
    ArgvCommand,
    ExecutionPolicy,
    ExecutionRequest,
    execute_checked,
)
from .models import ContextSourceRecord, ResearchRequest


class ResearchProviderError(RuntimeError):
    """Raised when a selected research provider fails or violates its contract."""


def normalize_research_records(
    request: ResearchRequest,
    records: Any,
    *,
    provider_name: str,
) -> tuple[ContextSourceRecord, ...]:
    """Validate provider evidence against result, byte, and URI boundaries."""

    if not isinstance(records, list):
        raise ResearchProviderError("research provider must return a list of records")
    if len(records) > request.max_results:
        raise ResearchProviderError(
            f"research provider returned {len(records)} results; maximum is "
            f"{request.max_results}"
        )
    normalized: list[ContextSourceRecord] = []
    total_bytes = 0
    for index, item in enumerate(records):
        if isinstance(item, ContextSourceRecord):
            record = item
        elif isinstance(item, dict):
            record = ContextSourceRecord.from_dict(item)
        else:
            raise ResearchProviderError(
                f"research result {index} must be a ContextSourceRecord or object"
            )
        if _contains_terminal_control(record.title) or _contains_terminal_control(
            record.uri
        ):
            raise ResearchProviderError(
                f"research result {index} title and URI must not contain control characters"
            )
        if not _allowed_research_uri(record.uri, request.allowed_domains):
            raise ResearchProviderError(
                f"research result {index} URI is outside the allowed HTTPS domains"
            )
        # Provider identity is selected by the caller and extension registry,
        # not self-asserted by untrusted result data.
        record = ContextSourceRecord(
            **{**record.to_dict(), "provider": provider_name}
        )
        try:
            encoded = json.dumps(
                record.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ResearchProviderError(
                f"research result {index} is not canonical JSON"
            ) from exc
        total_bytes += len(encoded)
        if total_bytes > request.max_bytes:
            raise ResearchProviderError(
                f"research provider output exceeds {request.max_bytes} bytes"
            )
        normalized.append(record)
    return tuple(normalized)


def _contains_terminal_control(value: str) -> bool:
    return any(unicodedata.category(character).startswith("C") for character in value)


def _allowed_research_uri(uri: str, allowed_domains: tuple[str, ...]) -> bool:
    if any(character.isspace() or character in '\\<>"' for character in uri):
        return False
    try:
        parsed = urlsplit(uri)
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    if (
        parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        return False
    host = parsed.hostname.casefold().rstrip(".")
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


def execute_research_provider(
    provider_name: str,
    request: ResearchRequest,
    *,
    config_path: Path | None = None,
) -> tuple[ContextSourceRecord, ...]:
    """Run one provider out of process with bounded time and captured output.

    Enabled extensions are trusted code. The process boundary enforces time
    and output limits and prevents non-selected provider imports; the selected
    provider is contractually responsible for honoring the domain allowlist
    during transport. AFS validates every returned URI again before exposure.
    """

    if not request.network_allowed:
        raise ResearchProviderError("research provider execution requires network consent")
    selected = provider_name.strip()
    if not selected:
        raise ResearchProviderError("research provider name must be non-empty")
    source_root = Path(__file__).resolve().parents[2]
    # Never run ``python -m afs...`` from the researched project: cwd wins
    # module resolution ahead of PYTHONPATH and could shadow the trusted
    # runner before consent/domain checks execute.
    working_directory = source_root
    explicit_config = config_path.expanduser().resolve() if config_path else None

    with tempfile.TemporaryDirectory(prefix="afs-research-") as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_text(
            json.dumps(request.to_dict(), sort_keys=True),
            encoding="utf-8",
        )
        argv = [
            sys.executable,
            "-m",
            "afs.sources.research_runner",
            "--provider",
            selected,
            "--request",
            str(request_path),
        ]
        if explicit_config is not None:
            argv.extend(["--config", str(explicit_config)])
        set_env = {"PYTHONPATH": str(source_root)}
        if explicit_config is not None:
            set_env["AFS_CONFIG_PATH"] = str(explicit_config)
        execution = ExecutionRequest(
            command=ArgvCommand(tuple(argv)),
            caller="afs.insights.research",
            purpose=f"bounded extension research via {selected}",
            cwd=working_directory,
            set_env=set_env,
            timeout_seconds=request.timeout_seconds,
            max_output_bytes=MAX_OUTPUT_BYTES,
            isolation="process",
            network="inherit",
            redact_argv_indices=(6,),
        )
        policy = ExecutionPolicy(
            allowed_cwd_roots=(working_directory,),
            allowed_executables=frozenset(
                {
                    sys.executable,
                    str(Path(sys.executable).expanduser().resolve()),
                    Path(sys.executable).name,
                }
            ),
            allowed_env=frozenset(set_env),
        )
        record = execute_checked(execution, policy, environ=os.environ)

    if record.outcome != "completed":
        detail = record.stderr.strip() or "; ".join(record.reasons)
        raise ResearchProviderError(
            f"research provider {selected!r} {record.outcome}: {detail[:500]}"
        )
    if record.stdout_truncated:
        raise ResearchProviderError("research provider output exceeded the execution limit")
    try:
        payload = json.loads(record.stdout)
    except json.JSONDecodeError as exc:
        raise ResearchProviderError("research provider returned invalid JSON") from exc
    return normalize_research_records(
        request,
        payload,
        provider_name=selected,
    )


__all__ = [
    "ResearchProviderError",
    "execute_research_provider",
    "normalize_research_records",
]
