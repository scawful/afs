from __future__ import annotations

import pytest

from afs.response_schemas import (
    SchemaValidationResult,
    _builtin_schema_errors,
    build_schema_correction,
    coerce_response_payload,
    get_response_schema,
    validate_structured_response,
)

_VALID_PLAN = {
    "summary": "Add push hooks",
    "steps": ["wire hook", "add tests"],
    "verification": ["pytest -q"],
    "risks": ["token budget"],
}


def test_validate_accepts_conforming_payload() -> None:
    result = validate_structured_response("implementation-plan", _VALID_PLAN)
    assert result.valid is True
    assert result.errors == []
    assert result.parsed == _VALID_PLAN


def test_validate_reports_missing_required_fields() -> None:
    result = validate_structured_response("handoff-summary", {"accomplished": ["x"]})
    assert result.valid is False
    joined = " ".join(result.errors)
    assert "blocked" in joined
    assert "next_steps" in joined


def test_validate_rejects_additional_properties() -> None:
    payload = {**_VALID_PLAN, "surprise": 1}
    result = validate_structured_response("implementation-plan", payload)
    assert result.valid is False
    assert any("surprise" in err for err in result.errors)


def test_validate_parses_fenced_json_text() -> None:
    fenced = "```json\n" + '{"accomplished": ["a"], "blocked": [], "next_steps": ["b"]}' + "\n```"
    result = validate_structured_response("handoff-summary", fenced)
    assert result.valid is True
    assert result.parsed["accomplished"] == ["a"]


def test_validate_surfaces_parse_error_without_raising() -> None:
    result = validate_structured_response("handoff-summary", "not json at all {")
    assert result.valid is False
    assert result.errors == []
    assert "invalid JSON" in result.parse_error


def test_validate_unknown_schema_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        validate_structured_response("no-such-schema", {})


def test_coerce_passes_through_objects() -> None:
    parsed, error = coerce_response_payload({"a": 1})
    assert parsed == {"a": 1}
    assert error == ""


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_coerce_rejects_non_finite_parsed_values(value: float) -> None:
    parsed, error = coerce_response_payload({"value": value})

    assert parsed is None
    assert "non-finite" in error


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_coerce_rejects_nonstandard_json_number_tokens(token: str) -> None:
    parsed, error = coerce_response_payload(f'{{"value": {token}}}')

    assert parsed is None
    assert "non-standard JSON number" in error


def test_coerce_rejects_duplicate_object_members() -> None:
    parsed, error = coerce_response_payload('{"outer": {"value": 1, "value": 2}}')

    assert parsed is None
    assert "duplicate JSON object member 'value'" in error


def test_build_schema_correction_lists_violations_and_resource() -> None:
    result = validate_structured_response("handoff-summary", {"accomplished": ["x"]})
    correction = build_schema_correction(result)
    assert "`handoff-summary`" in correction
    assert "next_steps" in correction
    assert "afs://schemas/handoff-summary" in correction


def test_build_schema_correction_empty_when_valid() -> None:
    result = validate_structured_response("implementation-plan", _VALID_PLAN)
    assert build_schema_correction(result) == ""


def test_builtin_fallback_matches_core_rules() -> None:
    # The jsonschema-free fallback must catch the same common failures.
    schema = get_response_schema("implementation-plan")
    ok = _builtin_schema_errors(schema, _VALID_PLAN)
    assert ok == []

    missing = _builtin_schema_errors(schema, {"summary": "x"})
    assert any("steps" in err for err in missing)

    extra = _builtin_schema_errors(schema, {**_VALID_PLAN, "nope": True})
    assert any("nope" in err for err in extra)

    wrong_type = _builtin_schema_errors(schema, {**_VALID_PLAN, "summary": 123})
    assert any("summary" in err for err in wrong_type)


def test_builtin_fallback_checks_nested_protocol_rules() -> None:
    schema = get_response_schema("v1/optimization/evaluation")
    payload = {
        "schema_version": "1.0",
        "candidate_id": "candidate",
        "parent_id": "root",
        "artifact_sha256": "a" * 64,
        "evaluation_suite": {
            "name": "suite",
            "version": "1",
            "case_set_sha256": "b" * 64,
        },
        "provenance": {
            "run_id": "run",
            "evaluator": "eval",
            "evaluator_version": "1",
            "seed": 1,
            "environment_sha256": "c" * 64,
        },
        "metrics": [{"name": "quality", "unit": "ratio", "value": 1.0, "sample_count": 0}],
        "constraints": {"tests": "yes"},
    }

    errors = _builtin_schema_errors(schema, payload)

    assert any("sample_count" in error and ">= 1" in error for error in errors)
    assert any("constraints/tests" in error and "boolean" in error for error in errors)


def test_result_to_dict_shape() -> None:
    result = SchemaValidationResult(valid=False, schema="plan", errors=["e"], parse_error="")
    payload = result.to_dict()
    assert payload == {"valid": False, "schema": "plan", "errors": ["e"], "parse_error": ""}


# ---------------------------------------------------------------------------
# human_intent (skeleton-first planning)
# ---------------------------------------------------------------------------


def test_plan_accepts_human_intent_section() -> None:
    payload = {
        **_VALID_PLAN,
        "human_intent": {
            "goal": "friction on decisions, not mechanics",
            "non_goals": ["blocking headless agents"],
            "done_when": ["all five steps land with tests"],
        },
    }
    result = validate_structured_response("implementation-plan", payload)
    assert result.valid is True


def test_plan_human_intent_rejects_unknown_keys() -> None:
    payload = {**_VALID_PLAN, "human_intent": {"agent_notes": "sneaky"}}
    result = validate_structured_response("implementation-plan", payload)
    assert result.valid is False


def test_human_intent_preserved_checks() -> None:
    from afs.response_schemas import verify_human_intent_preserved

    intent = {"goal": "the human goal", "done_when": ["tests pass"]}
    skeleton = {"human_intent": intent, "summary": "seed"}
    faithful = {**_VALID_PLAN, "human_intent": dict(intent)}
    assert verify_human_intent_preserved(skeleton, faithful) == []

    edited = {**_VALID_PLAN, "human_intent": {**intent, "goal": "reworded"}}
    assert any("modified" in v for v in verify_human_intent_preserved(skeleton, edited))

    dropped = dict(_VALID_PLAN)
    assert any("removed" in v for v in verify_human_intent_preserved(skeleton, dropped))

    # No skeleton intent: the agent must not author one.
    fabricated = {**_VALID_PLAN, "human_intent": {"goal": "agent-invented"}}
    assert any(
        "authored" in v for v in verify_human_intent_preserved({}, fabricated)
    )
    assert verify_human_intent_preserved({}, dict(_VALID_PLAN)) == []


def test_human_intent_shape_bypasses_are_closed() -> None:
    """Adversarial probes: Python equality quirks and malformed anchors."""
    from afs.response_schemas import verify_human_intent_preserved

    # An empty section authored from nowhere is still authoring.
    assert any(
        "authored" in v
        for v in verify_human_intent_preserved({}, {**_VALID_PLAN, "human_intent": {}})
    )

    # A non-object anchor is rejected, never silently treated as absent.
    bad_anchor = {"human_intent": "just a string", "summary": "seed"}
    fabricated = {**_VALID_PLAN, "human_intent": {"goal": "agent-invented"}}
    assert any(
        "must be an object" in v
        for v in verify_human_intent_preserved(bad_anchor, fabricated)
    )

    # A skeleton violating the human_intent contract is invalid, not absent.
    invalid_anchor = {"human_intent": {"goal": 42}, "summary": "seed"}
    assert any(
        "invalid" in v
        for v in verify_human_intent_preserved(invalid_anchor, fabricated)
    )

    # A non-object skeleton document cannot anchor anything.
    assert any(
        "JSON object" in v
        for v in verify_human_intent_preserved(["not", "a", "dict"], fabricated)
    )

    # An explicitly empty anchor must stay exactly empty.
    empty_anchor = {"human_intent": {}, "summary": "seed"}
    filled = {**_VALID_PLAN, "human_intent": {"goal": "agent filled it in"}}
    assert any(
        "modified" in v for v in verify_human_intent_preserved(empty_anchor, filled)
    )
    kept_empty = {**_VALID_PLAN, "human_intent": {}}
    assert verify_human_intent_preserved(empty_anchor, kept_empty) == []


def test_canonical_json_distinguishes_python_equality_quirks() -> None:
    """True==1 and 1==1.0 in Python; the preservation check must not conflate
    them (they are different JSON documents)."""
    from afs.response_schemas import _canonical_json

    assert _canonical_json({"x": True}) != _canonical_json({"x": 1})
    assert _canonical_json({"x": 1}) != _canonical_json({"x": 1.0})
    # Key order is irrelevant — same document either way.
    assert _canonical_json({"a": "1", "b": "2"}) == _canonical_json({"b": "2", "a": "1"})
    assert _canonical_json(object()) is None
