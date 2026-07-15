"""Versioned, language-neutral protocol schemas shipped with AFS."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

_PROTOCOL_SCHEMA_FILES = {
    "v1/optimization/decision": "decision.schema.json",
    "v1/optimization/evaluation": "evaluation.schema.json",
    "v1/optimization/policy": "policy.schema.json",
}


def load_protocol_schemas() -> dict[str, dict[str, Any]]:
    """Load the packaged JSON Schema contracts keyed by AFS schema name."""
    schema_root = resources.files("afs.protocols.optimization.v1")
    schemas: dict[str, dict[str, Any]] = {}
    for name, filename in _PROTOCOL_SCHEMA_FILES.items():
        payload = json.loads(schema_root.joinpath(filename).read_text(encoding="utf-8"))
        expected_id = f"afs://schemas/{name}"
        if payload.get("$id") != expected_id:
            raise ValueError(f"Protocol schema {filename} must use $id {expected_id!r}")
        schemas[name] = payload
    return schemas


def list_protocol_schema_names() -> list[str]:
    """Return stable names for the packaged protocol schemas."""
    return sorted(_PROTOCOL_SCHEMA_FILES)
