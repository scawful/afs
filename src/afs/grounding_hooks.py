"""Grounding hooks for context and orchestration operations."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .history import log_event
from .profiles import merge_extension_hooks, resolve_active_profile
from .schema import AFSConfig

logger = logging.getLogger(__name__)

_PRE_HOOK_EVENTS = {
    "before_context_read",
    "before_agent_dispatch",
    "session_start",
    "user_prompt_submit",
    "turn_started",
}
_ZELDA_KEYWORDS = {
    "zelda",
    "alttp",
    "oracle-of-secrets",
    "oracle of secrets",
    "hyrule",
    "triforce",
    "snes",
}


def run_grounding_hooks(
    *,
    event: str,
    payload: dict[str, Any],
    config: AFSConfig,
    profile_name: str | None = None,
) -> None:
    """Run built-in and script hooks for the provided event."""
    profile = resolve_active_profile(config, profile_name=profile_name)
    commands = merge_extension_hooks(config, profile, event)
    context_root = _resolve_context_path(payload)
    status = "ok"
    error_message: str | None = None

    try:
        _enforce_profile_policies(event, payload, profile.policies)
        if not commands:
            return

        for command in commands:
            try:
                result = _run_hook_command(command, event, payload)
            except FileNotFoundError as exc:
                if event in _PRE_HOOK_EVENTS:
                    status = "blocked"
                    error_message = str(exc)
                    raise PermissionError(str(exc)) from exc
                logger.warning("Hook command missing for event %s: %s", event, exc)
                continue

            if result.returncode == 0:
                continue

            stderr = (result.stderr or "").strip()
            message = stderr or f"hook command failed with exit {result.returncode}"
            if event in _PRE_HOOK_EVENTS:
                status = "blocked"
                error_message = message
                raise PermissionError(message)
            logger.warning("Hook command failed for event %s: %s", event, message)
    except PermissionError:
        status = "blocked"
        raise
    except Exception as exc:
        status = "error"
        error_message = str(exc)
        raise
    finally:
        if context_root is not None:
            log_event(
                "hook",
                "afs.grounding_hooks",
                op=event,
                metadata={
                    "profile": profile.name,
                    "commands": list(commands),
                    "status": status,
                    "error": error_message,
                },
                context_root=context_root,
            )


def _enforce_profile_policies(event: str, payload: dict[str, Any], policies: list[str]) -> None:
    policy_set = {policy.strip().lower() for policy in policies if policy and policy.strip()}
    if "no_zelda" not in policy_set:
        return

    text_parts: list[str] = []
    for value in payload.values():
        if isinstance(value, str):
            text_parts.append(value.lower())
    combined = "\n".join(text_parts)

    if any(keyword in combined for keyword in _ZELDA_KEYWORDS):
        raise PermissionError(
            "Profile policy violation: no_zelda is active for this context."
        )


class _CompletedProcess:
    returncode: int
    stderr: str


def _run_hook_command(command: str, event: str, payload: dict[str, Any]) -> _CompletedProcess:
    argv = shlex.split(command)
    if not argv:
        raise FileNotFoundError("empty hook command")

    executable = argv[0]
    expanded = str(Path(executable).expanduser())
    if expanded.startswith("/") or expanded.startswith("."):
        argv[0] = expanded

    env = dict(os.environ)
    env["AFS_HOOK_EVENT"] = event

    payload_json = json.dumps(payload, ensure_ascii=True)
    return subprocess.run(
        argv,
        input=payload_json,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def _resolve_context_path(payload: dict[str, Any]) -> Path | None:
    raw = payload.get("context_path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return Path(raw).expanduser().resolve()
    except Exception:
        return None


def extract_path_tokens(text: str) -> list[str]:
    """Extract candidate file path tokens from free-form text."""
    if not text:
        return []
    candidates = re.findall(r"(?:~|\.|/)?[A-Za-z0-9_./-]+", text)
    return [token for token in candidates if "/" in token or token.startswith("~")]
