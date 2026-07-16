"""Compact JSON response schemas for structured AFS workflows."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .protocols import load_protocol_schemas
from .protocols.canonical_json import (
    CanonicalJSONError,
    ensure_interoperable_json,
    strict_json_loads,
)

SCHEMA_MIME_TYPE = "application/schema+json"
SCHEMA_URI_PREFIX = "afs://schemas/"

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL)
_RFC3339_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$"
)

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
            "human_intent": {
                "type": "object",
                "minProperties": 1,
                "description": (
                    "Human-authored skeleton the plan expands on: goal, "
                    "non-goals, and done-when in the human's own words. "
                    "Agents must never write, fill, or edit this section — "
                    "reproduce it exactly as provided (or omit it when the "
                    "human gave none). Plan review diffs the agent-authored "
                    "sections against this intent."
                ),
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The outcome in the human's own words.",
                    },
                    "non_goals": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                    },
                    "done_when": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                    },
                },
                "additionalProperties": False,
            },
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

# Protocol schemas are standalone package data so non-Python clients can consume
# the exact same contracts that the CLI and MCP resource surface advertise.
_SCHEMA_DEFINITIONS.update(load_protocol_schemas())


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
        try:
            ensure_interoperable_json(data)
        except CanonicalJSONError as exc:
            return None, f"invalid JSON value: {exc}"
        return data, ""
    if not isinstance(data, str):
        return None, f"expected a JSON object or string, got {type(data).__name__}"
    text = _strip_code_fence(data.strip())
    if not text:
        return None, "empty response"
    try:
        return strict_json_loads(text), ""
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
    except CanonicalJSONError as exc:
        return None, f"invalid JSON: {exc}"


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


def _human_intent_subschema() -> dict[str, Any]:
    """The human_intent contract from the implementation-plan schema."""
    plan = _SCHEMA_DEFINITIONS.get("implementation-plan", {})
    return plan.get("properties", {}).get("human_intent", {"type": "object"})


def _canonical_json(value: Any) -> str | None:
    """Serialize to canonical JSON, or ``None`` for unserializable values.

    Comparison happens on the serialized form, not Python equality: ``True``
    and ``1`` (or ``1`` and ``1.0``) compare equal as Python objects but are
    different JSON documents, and a preservation check that conflates them
    can be gamed. ``allow_nan=False`` keeps non-finite floats out of the
    canonical form — ``NaN`` on both sides must not read as "preserved".
    """
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None


def verify_human_intent_preserved(skeleton: Any, expanded: Any) -> list[str]:
    """Check that an agent expansion left the human-authored skeleton untouched.

    ``skeleton`` is the human's original plan (or plan fragment) and
    ``expanded`` is the agent-produced plan. The ``human_intent`` section is
    the one part agents must never author: it must remain canonically
    unchanged, and it must not appear from nowhere. The skeleton's own
    ``human_intent`` must satisfy the schema contract — a malformed anchor is
    rejected rather than silently treated as absent. Returns a list of
    violations (empty when preserved).
    """
    if not isinstance(skeleton, dict):
        return ["skeleton must be a JSON object (the human-authored plan fragment)"]
    skeleton_intent = skeleton.get("human_intent")
    expanded_intent = expanded.get("human_intent") if isinstance(expanded, dict) else None

    if skeleton_intent is not None:
        if not isinstance(skeleton_intent, dict):
            return [
                "skeleton human_intent must be an object with goal/non_goals/"
                "done_when; a malformed trust anchor cannot be verified"
            ]
        skeleton_errors = _collect_schema_errors(
            _human_intent_subschema(), skeleton_intent
        )
        if skeleton_errors:
            return [
                f"skeleton human_intent is invalid: {error}"
                for error in skeleton_errors[:5]
            ]
        if expanded_intent is None:
            return ["human_intent was removed by the expansion; restore it verbatim"]
        skeleton_canonical = _canonical_json(skeleton_intent)
        expanded_canonical = _canonical_json(expanded_intent)
        # None means unserializable; two failures must never compare equal.
        if (
            skeleton_canonical is None
            or expanded_canonical is None
            or expanded_canonical != skeleton_canonical
        ):
            return [
                "human_intent was modified by the expansion; agents must "
                "reproduce it exactly as the human wrote it"
            ]
        return []
    if expanded_intent is not None:
        return [
            "human_intent was authored by the expansion; this section is "
            "human-written only — omit it and ask instead"
        ]
    return []


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
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError:
        return _builtin_schema_errors(schema, instance)

    format_checker = FormatChecker()
    format_checker.checkers["date-time"] = (_is_rfc3339_datetime, ())
    validator = Draft202012Validator(schema, format_checker=format_checker)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(instance), key=lambda e: list(e.path)):
        location = "/".join(str(part) for part in error.path) or "(root)"
        errors.append(f"{location}: {error.message}")
    return errors


def _is_rfc3339_datetime(value: object) -> bool:
    if not isinstance(value, str) or _RFC3339_DATETIME_RE.fullmatch(value) is None:
        return False
    normalized = value[:-1] + "+00:00" if value[-1:].casefold() == "z" else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _builtin_schema_errors(schema: dict[str, Any], instance: Any) -> list[str]:
    """Minimal structural fallback when jsonschema is unavailable.

    Covers the structural constraints used by AFS's built-in response and protocol
    schemas. It is intentionally not a complete JSON Schema implementation.
    """
    errors: list[str] = []
    _validate_schema_node(schema, instance, "(root)", errors)
    return errors


def _matches_json_type(expected: str, value: Any) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _validate_schema_node(
    schema: dict[str, Any],
    instance: Any,
    location: str,
    errors: list[str],
) -> None:
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{location}: expected constant {schema['const']!r}")
        return
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{location}: expected one of {schema['enum']!r}")
        return

    expected = schema.get("type")
    if isinstance(expected, str) and not _matches_json_type(expected, instance):
        errors.append(f"{location}: expected {expected}, got {type(instance).__name__}")
        return
    if isinstance(expected, list) and not any(
        isinstance(item, str) and _matches_json_type(item, instance) for item in expected
    ):
        errors.append(f"{location}: expected one of {expected!r}, got {type(instance).__name__}")
        return

    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        min_properties = schema.get("minProperties")
        max_properties = schema.get("maxProperties")
        if isinstance(min_properties, int) and len(instance) < min_properties:
            errors.append(f"{location}: expected at least {min_properties} properties")
        if isinstance(max_properties, int) and len(instance) > max_properties:
            errors.append(f"{location}: expected at most {max_properties} properties")
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{location}: missing required property '{key}'")
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            child_location = key if location == "(root)" else f"{location}/{key}"
            child_schema = properties.get(key) if isinstance(properties, dict) else None
            if isinstance(child_schema, dict):
                _validate_schema_node(child_schema, value, child_location, errors)
            elif additional is False:
                errors.append(f"{location}: additional property '{key}' is not allowed")
            elif isinstance(additional, dict):
                _validate_schema_node(additional, value, child_location, errors)

    if isinstance(instance, list):
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(instance) < min_items:
            errors.append(f"{location}: expected at least {min_items} item(s)")
        if isinstance(max_items, int) and len(instance) > max_items:
            errors.append(f"{location}: expected at most {max_items} item(s)")
        if schema.get("uniqueItems") is True:
            encoded = [json.dumps(item, sort_keys=True) for item in instance]
            if len(encoded) != len(set(encoded)):
                errors.append(f"{location}: expected unique items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(instance):
                _validate_schema_node(item_schema, item, f"{location}/{index}", errors)

    if isinstance(instance, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(instance) < min_length:
            errors.append(f"{location}: expected length >= {min_length}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, instance) is None:
            errors.append(f"{location}: does not match {pattern!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and instance < minimum:
            errors.append(f"{location}: expected value >= {minimum}")
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and instance > maximum:
            errors.append(f"{location}: expected value <= {maximum}")
