from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from afs.optimization import OptimizationInputError, decide_optimization_step
from afs.response_schemas import (
    get_response_schema,
    list_response_schema_names,
    validate_structured_response,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = ROOT / "examples" / "optimization_gate"


def _example(name: str) -> dict:
    return json.loads((EXAMPLE_ROOT / name).read_text(encoding="utf-8"))


def test_packaged_optimization_schemas_are_valid_draft_2020_12() -> None:
    names = {
        "v1/optimization/decision",
        "v1/optimization/evaluation",
        "v1/optimization/policy",
    }
    assert names.issubset(list_response_schema_names())
    for name in names:
        schema = get_response_schema(name)
        Draft202012Validator.check_schema(schema)
        assert schema["$id"] == f"afs://schemas/{name}"


def test_example_records_validate_and_candidate_is_eligible_for_review() -> None:
    baseline = _example("baseline.json")
    candidate = _example("candidate.json")
    policy = _example("policy.json")

    assert validate_structured_response("v1/optimization/evaluation", baseline).valid
    assert validate_structured_response("v1/optimization/evaluation", candidate).valid
    assert validate_structured_response("v1/optimization/policy", policy).valid

    decision = decide_optimization_step(baseline, candidate, policy)

    assert decision["decision"] == "eligible_for_human_review"
    assert decision["requires_human_approval"] is True
    assert "candidate_improves_objective" in decision["reason_codes"]
    assert len(decision["decision_sha256"]) == 64
    assert validate_structured_response("v1/optimization/decision", decision).valid

    for metric in decision["metrics"]:
        assert ("min_delta" in metric) == (metric["role"] == "objective")


def test_decision_is_stable_across_semantically_irrelevant_input_order() -> None:
    baseline = _example("baseline.json")
    candidate = _example("candidate.json")
    policy = _example("policy.json")
    first = decide_optimization_step(baseline, candidate, policy)

    candidate["metrics"].reverse()
    policy["metrics"].reverse()
    policy["required_constraints"].reverse()
    second = decide_optimization_step(baseline, candidate, policy)

    assert second == first


def test_hashes_normalize_equivalent_json_number_spellings() -> None:
    baseline = _example("baseline.json")
    candidate = _example("candidate.json")
    policy = _example("policy.json")
    first = decide_optimization_step(baseline, candidate, policy)

    baseline_latency = next(metric for metric in baseline["metrics"] if metric["name"] == "latency")
    baseline_latency["value"] = 500
    second = decide_optimization_step(baseline, candidate, policy)

    assert second == first


def test_decision_does_not_mutate_inputs() -> None:
    inputs = (
        _example("baseline.json"),
        _example("candidate.json"),
        _example("policy.json"),
    )
    before = deepcopy(inputs)
    decide_optimization_step(*inputs)
    assert inputs == before


def test_failed_constraint_rejects_candidate() -> None:
    baseline = _example("baseline.json")
    candidate = _example("candidate.json")
    policy = _example("policy.json")
    candidate["constraints"]["tests_pass"] = False

    decision = decide_optimization_step(baseline, candidate, policy)

    assert decision["decision"] == "rejected"
    assert decision["failed_constraints"] == ["tests_pass"]
    assert decision["requires_human_approval"] is False


def test_guardrail_regression_rejects_candidate() -> None:
    baseline = _example("baseline.json")
    candidate = _example("candidate.json")
    policy = _example("policy.json")
    latency = next(metric for metric in candidate["metrics"] if metric["name"] == "latency")
    latency["value"] = 600.0

    decision = decide_optimization_step(baseline, candidate, policy)

    assert decision["decision"] == "rejected"
    assert "metric_regression" in decision["reason_codes"]


def test_under_sampled_evidence_is_inconclusive() -> None:
    baseline = _example("baseline.json")
    candidate = _example("candidate.json")
    policy = _example("policy.json")
    candidate["metrics"][0]["sample_count"] = 10

    decision = decide_optimization_step(baseline, candidate, policy)

    assert decision["decision"] == "inconclusive"
    assert "insufficient_samples" in decision["reason_codes"]


def test_missing_uncertainty_is_inconclusive_when_policy_uses_confidence_bound() -> None:
    baseline = _example("baseline.json")
    candidate = _example("candidate.json")
    policy = _example("policy.json")
    candidate["metrics"][0].pop("standard_error")

    decision = decide_optimization_step(baseline, candidate, policy)

    assert decision["decision"] == "inconclusive"
    assert "uncertainty_missing" in decision["reason_codes"]


def test_environment_mismatch_is_inconclusive_when_policy_requires_match() -> None:
    baseline = _example("baseline.json")
    candidate = _example("candidate.json")
    policy = _example("policy.json")
    candidate["provenance"]["environment_sha256"] = "e" * 64

    decision = decide_optimization_step(baseline, candidate, policy)

    assert decision["decision"] == "inconclusive"
    assert "environment_mismatch" in decision["reason_codes"]


def test_duplicate_metrics_fail_closed() -> None:
    baseline = _example("baseline.json")
    candidate = _example("candidate.json")
    policy = _example("policy.json")
    candidate["metrics"].append(deepcopy(candidate["metrics"][0]))

    with pytest.raises(OptimizationInputError, match="duplicate metric"):
        decide_optimization_step(baseline, candidate, policy)


def test_non_finite_metric_fails_closed() -> None:
    baseline = _example("baseline.json")
    candidate = _example("candidate.json")
    policy = _example("policy.json")
    candidate["metrics"][0]["value"] = float("nan")

    with pytest.raises(OptimizationInputError, match="non-finite"):
        decide_optimization_step(baseline, candidate, policy)


def test_objective_requires_positive_minimum_delta() -> None:
    baseline = _example("baseline.json")
    candidate = _example("candidate.json")
    policy = _example("policy.json")
    objective = next(metric for metric in policy["metrics"] if metric["role"] == "objective")
    objective["min_delta"] = 0.0

    with pytest.raises(OptimizationInputError, match="invalid policy"):
        decide_optimization_step(baseline, candidate, policy)

    del objective["min_delta"]

    with pytest.raises(OptimizationInputError, match="invalid policy"):
        decide_optimization_step(baseline, candidate, policy)


def test_policy_schema_enforces_role_specific_minimum_delta() -> None:
    policy = _example("policy.json")
    objective = next(metric for metric in policy["metrics"] if metric["role"] == "objective")
    guardrail = next(metric for metric in policy["metrics"] if metric["role"] == "guardrail")

    objective.pop("min_delta")
    assert not validate_structured_response("v1/optimization/policy", policy).valid

    objective["min_delta"] = 0.0
    assert not validate_structured_response("v1/optimization/policy", policy).valid

    objective["min_delta"] = 0.01
    guardrail["min_delta"] = 0.01
    assert not validate_structured_response("v1/optimization/policy", policy).valid


def test_guardrail_must_not_declare_minimum_delta() -> None:
    baseline = _example("baseline.json")
    candidate = _example("candidate.json")
    policy = _example("policy.json")
    guardrail = next(metric for metric in policy["metrics"] if metric["role"] == "guardrail")
    guardrail["min_delta"] = 0.0

    with pytest.raises(OptimizationInputError, match="invalid policy"):
        decide_optimization_step(baseline, candidate, policy)


def test_decision_schema_enforces_role_specific_minimum_delta() -> None:
    decision = decide_optimization_step(
        _example("baseline.json"),
        _example("candidate.json"),
        _example("policy.json"),
    )
    objective = next(
        metric for metric in decision["metrics"] if metric["role"] == "objective"
    )
    guardrail = next(
        metric for metric in decision["metrics"] if metric["role"] == "guardrail"
    )

    objective.pop("min_delta")
    assert not validate_structured_response("v1/optimization/decision", decision).valid

    objective["min_delta"] = 0.01
    guardrail["min_delta"] = 0.01
    assert not validate_structured_response("v1/optimization/decision", decision).valid


def test_exact_decimal_threshold_is_not_lost_to_binary_float_rounding() -> None:
    baseline = _example("baseline.json")
    candidate = _example("candidate.json")
    policy = _example("policy.json")
    objective = next(metric for metric in candidate["metrics"] if metric["name"] == "task_success")
    objective["value"] = 0.82
    policy["confidence_z"] = 0.0

    decision = decide_optimization_step(baseline, candidate, policy)
    objective_result = next(
        metric for metric in decision["metrics"] if metric["name"] == "task_success"
    )

    assert decision["decision"] == "eligible_for_human_review"
    assert objective_result["adjusted_delta"] == 0.02


def test_integer_outside_supported_numeric_range_fails_closed() -> None:
    baseline = _example("baseline.json")
    candidate = _example("candidate.json")
    policy = _example("policy.json")
    candidate["metrics"][0]["value"] = 10**400

    with pytest.raises(OptimizationInputError, match="supported finite range"):
        decide_optimization_step(baseline, candidate, policy)


def test_maximum_policy_size_still_produces_a_schema_valid_decision() -> None:
    baseline = _example("baseline.json")
    candidate = _example("candidate.json")
    policy = _example("policy.json")
    policy["confidence_z"] = 0.0
    policy["metrics"] = [
        {
            "name": f"quality_{index}",
            "unit": "ratio",
            "direction": "maximize",
            "role": "objective" if index == 0 else "guardrail",
            "max_regression": 0.0,
            "min_samples": 1,
            **({"min_delta": 0.01} if index == 0 else {}),
        }
        for index in range(256)
    ]

    decision = decide_optimization_step(baseline, candidate, policy)

    assert decision["decision"] == "inconclusive"
    assert len(decision["reasons"]) == 256
    assert validate_structured_response("v1/optimization/decision", decision).valid
