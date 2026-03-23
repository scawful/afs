"""Compact JSON response schemas for structured AFS workflows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

SCHEMA_MIME_TYPE = "application/schema+json"
SCHEMA_URI_PREFIX = "afs://schemas/"

_BASE_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

_SCHEMA_DEFINITIONS: dict[str, dict[str, Any]] = {
    "plan": {
        "$schema": _BASE_SCHEMA,
        "$id": f"{SCHEMA_URI_PREFIX}plan",
        "title": "AFS Plan",
        "description": "Short execution plan for a bounded task.",
        "type": "object",
        "required": ["goal", "steps", "completion_signal", "confidence"],
        "properties": {
            "goal": {"type": "string", "description": "One-sentence task goal."},
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 8,
                "description": "Flat ordered steps.",
            },
            "completion_signal": {
                "type": "string",
                "description": "How to know the task is done.",
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
        },
        "additionalProperties": False,
    },
    "file-shortlist": {
        "$schema": _BASE_SCHEMA,
        "$id": f"{SCHEMA_URI_PREFIX}file-shortlist",
        "title": "AFS File Shortlist",
        "description": "Small set of candidate files for the task.",
        "type": "object",
        "required": ["query", "paths", "rationale"],
        "properties": {
            "query": {"type": "string", "description": "Search or task focus."},
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 12,
                "uniqueItems": True,
            },
            "rationale": {
                "type": "string",
                "description": "Why these files were selected.",
            },
            "gaps": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
                "description": "Known missing context or ambiguity.",
            },
        },
        "additionalProperties": False,
    },
    "review-findings": {
        "$schema": _BASE_SCHEMA,
        "$id": f"{SCHEMA_URI_PREFIX}review-findings",
        "title": "AFS Review Findings",
        "description": "Review output focused on risks and gaps.",
        "type": "object",
        "required": ["highest_severity", "findings", "overall_risk"],
        "properties": {
            "highest_severity": {
                "type": "string",
                "enum": ["none", "low", "medium", "high", "critical"],
            },
            "findings": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 10,
            },
            "overall_risk": {
                "type": "string",
                "description": "One-line risk summary.",
            },
            "missing_coverage": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
        },
        "additionalProperties": False,
    },
    "edit-intent": {
        "$schema": _BASE_SCHEMA,
        "$id": f"{SCHEMA_URI_PREFIX}edit-intent",
        "title": "AFS Edit Intent",
        "description": "Planned edit scope before changing code.",
        "type": "object",
        "required": ["summary", "paths", "checks"],
        "properties": {
            "summary": {"type": "string", "description": "Edit summary."},
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 12,
                "uniqueItems": True,
            },
            "checks": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
                "description": "Planned validation commands or checks.",
            },
            "risks": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6,
            },
        },
        "additionalProperties": False,
    },
    "verification-summary": {
        "$schema": _BASE_SCHEMA,
        "$id": f"{SCHEMA_URI_PREFIX}verification-summary",
        "title": "AFS Verification Summary",
        "description": "Compact verification result after acting.",
        "type": "object",
        "required": ["outcome", "checks_run"],
        "properties": {
            "outcome": {
                "type": "string",
                "enum": ["passed", "failed", "blocked"],
            },
            "checks_run": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 10,
            },
            "failing_checks": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "residual_risk": {
                "type": "string",
                "description": "What is still uncertain after verification.",
            },
        },
        "additionalProperties": False,
    },
    "handoff-summary": {
        "$schema": _BASE_SCHEMA,
        "$id": f"{SCHEMA_URI_PREFIX}handoff-summary",
        "title": "AFS Handoff Summary",
        "description": "Compact cross-agent handoff contract.",
        "type": "object",
        "required": ["accomplished", "blocked", "next_steps"],
        "properties": {
            "accomplished": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "blocked": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "next_steps": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "owner": {
                "type": "string",
                "description": "Optional next owner or agent.",
            },
        },
        "additionalProperties": False,
    },
}


def list_response_schema_specs() -> list[dict[str, str]]:
    """Return MCP resource descriptors for the built-in response schemas."""
    resources: list[dict[str, str]] = []
    for name in sorted(_SCHEMA_DEFINITIONS):
        schema = _SCHEMA_DEFINITIONS[name]
        resources.append(
            {
                "uri": schema_uri(name),
                "name": schema["title"],
                "description": schema["description"],
                "mimeType": SCHEMA_MIME_TYPE,
            }
        )
    return resources


def get_response_schema(name: str) -> dict[str, Any]:
    """Return a copy of a named response schema."""
    schema = _SCHEMA_DEFINITIONS.get(name)
    if schema is None:
        raise KeyError(name)
    return deepcopy(schema)


def schema_uri(name: str) -> str:
    """Build the MCP resource URI for a response schema."""
    return f"{SCHEMA_URI_PREFIX}{name}"
