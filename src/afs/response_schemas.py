"""Compact JSON response schemas for structured AFS workflows."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

SCHEMA_MIME_TYPE = "application/schema+json"
SCHEMA_URI_PREFIX = "afs://schemas/"

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL)

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
    "design-brief": {
        "$schema": _BASE_SCHEMA,
        "$id": f"{SCHEMA_URI_PREFIX}design-brief",
        "title": "AFS Design Brief",
        "description": "Design framing for larger or riskier changes before editing.",
        "type": "object",
        "required": ["problem", "constraints", "invariants", "acceptance_criteria"],
        "properties": {
            "problem": {"type": "string", "description": "Problem statement or change driver."},
            "constraints": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "invariants": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "acceptance_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "rollback_plan": {
                "type": "string",
                "description": "How to back out safely if the change misbehaves.",
            },
        },
        "additionalProperties": False,
    },
    "implementation-plan": {
        "$schema": _BASE_SCHEMA,
        "$id": f"{SCHEMA_URI_PREFIX}implementation-plan",
        "title": "AFS Implementation Plan",
        "description": "Execution plan with risks and verification for software changes.",
        "type": "object",
        "required": ["summary", "steps", "verification", "risks"],
        "properties": {
            "summary": {"type": "string", "description": "One-line implementation summary."},
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 8,
            },
            "verification": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "risks": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "rollback_plan": {
                "type": "string",
                "description": "Fallback or rollback path if the plan fails.",
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


def list_response_schema_names() -> list[str]:
    """Return the built-in response schema names."""
    return sorted(_SCHEMA_DEFINITIONS)


def get_response_schema(name: str) -> dict[str, Any]:
    """Return a copy of a named response schema."""
    schema = _SCHEMA_DEFINITIONS.get(name)
    if schema is None:
        raise KeyError(name)
    return deepcopy(schema)


def schema_uri(name: str) -> str:
    """Build the MCP resource URI for a response schema."""
    return f"{SCHEMA_URI_PREFIX}{name}"


@dataclass
class SchemaValidationResult:
    """Outcome of validating a model response against a named response schema."""

    valid: bool
    schema: str
    errors: list[str] = field(default_factory=list)
    parse_error: str = ""
    parsed: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "schema": self.schema,
            "errors": list(self.errors),
            "parse_error": self.parse_error,
        }


def _strip_code_fence(text: str) -> str:
    """Drop a surrounding ```json ... ``` fence if present, else return as-is."""
    match = _CODE_FENCE_RE.match(text)
    return match.group(1) if match else text


def coerce_response_payload(data: Any) -> tuple[Any, str]:
    """Return ``(parsed, parse_error)`` for a model response.

    Accepts an already-parsed ``dict``/``list``, or JSON text (optionally wrapped
    in a ```json code fence, which models frequently add despite instructions).
    ``parse_error`` is empty on success.
    """
    if isinstance(data, (dict, list)):
        return data, ""
    if not isinstance(data, str):
        return None, f"expected a JSON object or string, got {type(data).__name__}"
    text = _strip_code_fence(data.strip())
    if not text:
        return None, "empty response"
    try:
        return json.loads(text), ""
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"


def validate_structured_response(name: str, data: Any) -> SchemaValidationResult:
    """Validate a model response against the named AFS response schema.

    This closes the loop the schemas were built for: they are advertised to models
    (as MCP resources and in the structured-workflow prompt) but were never checked.
    Raises :class:`KeyError` for an unknown schema name; otherwise always returns a
    result (never raises on malformed input — that surfaces as ``parse_error``).
    """
    schema = get_response_schema(name)
    parsed, parse_error = coerce_response_payload(data)
    if parse_error:
        return SchemaValidationResult(valid=False, schema=name, parse_error=parse_error)
    errors = _collect_schema_errors(schema, parsed)
    return SchemaValidationResult(
        valid=not errors, schema=name, errors=errors, parsed=parsed
    )


def build_schema_correction(result: SchemaValidationResult) -> str:
    """Render a compact correction message to feed back on a schema miss.

    This is the payload a host injects as a correction turn: it names the schema,
    lists the concrete violations, and points at the schema resource so the model
    can self-correct without guessing.
    """
    if result.valid:
        return ""
    lines = [
        f"Your previous response did not match the required `{result.schema}` schema.",
        "Return only JSON matching the schema — no prose, no markdown fences.",
    ]
    if result.parse_error:
        lines.append(f"- Parse error: {result.parse_error}")
    for err in result.errors[:10]:
        lines.append(f"- {err}")
    extra = len(result.errors) - 10
    if extra > 0:
        lines.append(f"- ...and {extra} more schema violation(s).")
    lines.append(f"Schema resource: {schema_uri(result.schema)}")
    return "\n".join(lines)


def _collect_schema_errors(schema: dict[str, Any], instance: Any) -> list[str]:
    """Collect human-readable schema violations, preferring jsonschema when present."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return _builtin_schema_errors(schema, instance)

    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(instance), key=lambda e: list(e.path)):
        location = "/".join(str(part) for part in error.path) or "(root)"
        errors.append(f"{location}: {error.message}")
    return errors


def _builtin_schema_errors(schema: dict[str, Any], instance: Any) -> list[str]:
    """Minimal structural fallback when jsonschema is unavailable.

    Covers the features the built-in response schemas actually use: object typing,
    ``required`` keys, ``additionalProperties: false``, and per-property type plus
    ``minItems``/``maxItems`` on arrays. Not a full JSON Schema implementation.
    """
    errors: list[str] = []
    if schema.get("type") == "object" and not isinstance(instance, dict):
        return [f"(root): expected object, got {type(instance).__name__}"]
    if not isinstance(instance, dict):
        return errors

    for key in schema.get("required", []):
        if key not in instance:
            errors.append(f"(root): missing required property '{key}'")

    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        for key in instance:
            if key not in properties:
                errors.append(f"(root): additional property '{key}' is not allowed")

    type_map = {
        "string": str,
        "array": list,
        "object": dict,
        "boolean": bool,
        "number": (int, float),
        "integer": int,
    }
    for key, spec in properties.items():
        if key not in instance or not isinstance(spec, dict):
            continue
        value = instance[key]
        expected = spec.get("type")
        py_type = type_map.get(expected) if isinstance(expected, str) else None
        if py_type is not None and not isinstance(value, py_type):
            errors.append(f"{key}: expected {expected}, got {type(value).__name__}")
            continue
        if expected == "array" and isinstance(value, list):
            min_items = spec.get("minItems")
            max_items = spec.get("maxItems")
            if isinstance(min_items, int) and len(value) < min_items:
                errors.append(f"{key}: expected at least {min_items} item(s)")
            if isinstance(max_items, int) and len(value) > max_items:
                errors.append(f"{key}: expected at most {max_items} item(s)")
            item_type = (spec.get("items") or {}).get("type")
            item_py = type_map.get(item_type) if isinstance(item_type, str) else None
            if item_py is not None:
                for idx, item in enumerate(value):
                    if not isinstance(item, item_py):
                        errors.append(f"{key}/{idx}: expected {item_type}")
    return errors
