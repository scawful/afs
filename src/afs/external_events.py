"""Public event ingestion for extensions and external agents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .history import log_event


def emit_event(
    name: str, *, source: str, data: dict[str, Any], context_root: Path
) -> str:
    """Record an untrusted report without accepting core provenance fields."""
    for value in (name, source):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
            raise ValueError("event name and source must be 1-128 identifier characters")
    if not isinstance(data, dict):
        raise ValueError("event data must be a JSON object")
    if len(json.dumps(data, allow_nan=False).encode("utf-8")) > 32768:
        raise ValueError("event data exceeds 32768 bytes")
    event_id = log_event(
        "external",
        "afs.events.emit",
        op=name,
        metadata={"reported_source": source},
        payload={"data": data},
        context_root=context_root,
        include_payloads=True,
        redact_sensitive=True,
    )
    if event_id is None:
        raise RuntimeError("event was not recorded: history logging is disabled or unavailable")
    return event_id
