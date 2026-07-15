"""Versioned, language-neutral protocol schemas shipped with AFS."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

_PROTOCOL_SCHEMA_FILES: dict[str, tuple[str, str]] = {
    "v1/execution/inspection": (
        "afs.protocols.execution.v1",
        "inspection.schema.json",
    ),
    "v1/execution/record": ("afs.protocols.execution.v1", "record.schema.json"),
    "v1/execution/request": ("afs.protocols.execution.v1", "request.schema.json"),
    "v1/optimization/decision": (
        "afs.protocols.optimization.v1",
        "decision.schema.json",
    ),
    "v1/optimization/evaluation": (
        "afs.protocols.optimization.v1",
        "evaluation.schema.json",
    ),
    "v1/optimization/policy": ("afs.protocols.optimization.v1", "policy.schema.json"),
}


def load_protocol_schemas() -> dict[str, dict[str, Any]]:
    """Load the packaged JSON Schema contracts keyed by AFS schema name."""
    schemas: dict[str, dict[str, Any]] = {}
    for name, (package, filename) in _PROTOCOL_SCHEMA_FILES.items():
        schema_root = resources.files(package)
        payload = json.loads(schema_root.joinpath(filename).read_text(encoding="utf-8"))
        expected_id = f"afs://schemas/{name}"
        if payload.get("$id") != expected_id:
            raise ValueError(f"Protocol schema {filename} must use $id {expected_id!r}")
        schemas[name] = payload
    return schemas


def list_protocol_schema_names() -> list[str]:
    """Return stable names for the packaged protocol schemas."""
    return sorted(_PROTOCOL_SCHEMA_FILES)
