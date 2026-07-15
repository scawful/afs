"""Deterministic, side-effect-free gate for one optimization step."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from ..response_schemas import validate_structured_response

ALGORITHM_VERSION = "pareto-gate-1.0"
EVALUATION_SCHEMA = "v1/optimization/evaluation"
POLICY_SCHEMA = "v1/optimization/policy"
DECISION_SCHEMA = "v1/optimization/decision"


class OptimizationInputError(ValueError):
    """Raised when optimization evidence is invalid or ambiguous."""


def _ensure_finite(value: Any, location: str = "(root)") -> None:
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        try:
            binary64 = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise OptimizationInputError(
                f"{location}: number is outside the supported finite range"
            ) from exc
        if not math.isfinite(binary64):
            raise OptimizationInputError(f"{location}: non-finite numbers are not allowed")
    if isinstance(value, dict):
        for key, item in value.items():
            _ensure_finite(item, f"{location}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_finite(item, f"{location}/{index}")


def _decimal_number(value: Any, location: str) -> Decimal:
    """Return a bounded base-10 value for deterministic threshold arithmetic."""
    try:
        binary64 = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise OptimizationInputError(
            f"{location}: number is outside the supported finite range"
        ) from exc
    if not math.isfinite(binary64):
        raise OptimizationInputError(f"{location}: non-finite numbers are not allowed")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise OptimizationInputError(f"{location}: invalid decimal number") from exc


def _output_number(value: Decimal, location: str) -> float:
    """Convert a derived decimal to a finite JSON number or fail closed."""
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise OptimizationInputError(
            f"{location}: derived number is outside the supported finite range"
        ) from exc
    if not math.isfinite(result):
        raise OptimizationInputError(
            f"{location}: derived number is outside the supported finite range"
        )
    return result


def _normalize_for_canonical_json(value: Any, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_for_canonical_json(item, key) for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        normalized = [_normalize_for_canonical_json(item) for item in value]
        if parent_key == "metrics":
            return sorted(
                normalized,
                key=lambda item: str(item.get("name", "")) if isinstance(item, dict) else "",
            )
        if parent_key in {
            "failed_constraints",
            "missing_constraints",
            "reason_codes",
            "reasons",
            "required_constraints",
        }:
            return sorted(
                normalized,
                key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False),
            )
        return normalized
    return value


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    _ensure_finite(payload)
    normalized = _normalize_for_canonical_json(payload)
    return _encode_canonical_json(normalized).encode("utf-8")


def _canonical_number_text(value: int | float | Decimal) -> str:
    """Render one finite number using the AFS v1 plain-decimal hash format."""
    number = Decimal(str(value))
    if number == 0:
        return "0"

    sign, raw_digits, raw_exponent = number.as_tuple()
    if not isinstance(raw_exponent, int):
        raise OptimizationInputError("canonical JSON requires a finite number")
    exponent = raw_exponent
    digits = list(raw_digits)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    digit_text = "".join(str(digit) for digit in digits)

    if exponent >= 0:
        body = digit_text + "0" * exponent
    else:
        point = len(digit_text) + exponent
        if point > 0:
            body = f"{digit_text[:point]}.{digit_text[point:]}"
        else:
            body = f"0.{('0' * -point)}{digit_text}"
    return ("-" if sign else "") + body


def _encode_canonical_json(value: Any) -> str:
    """Encode JSON with stable object order and normalized numeric tokens."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float, Decimal)):
        return _canonical_number_text(value)
    if isinstance(value, list):
        return "[" + ",".join(_encode_canonical_json(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                json.dumps(str(key), ensure_ascii=False) + ":" + _encode_canonical_json(item)
                for key, item in sorted(value.items())
            )
            + "}"
        )
    raise OptimizationInputError(
        f"canonical JSON does not support values of type {type(value).__name__}"
    )


def canonical_json_text(payload: dict[str, Any]) -> str:
    """Render stable human-readable JSON for CLI output."""
    _ensure_finite(payload)
    normalized = _normalize_for_canonical_json(payload)
    return (
        json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _sha256_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _validated_payload(
    schema_name: str,
    payload: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    _ensure_finite(payload)
    result = validate_structured_response(schema_name, payload)
    if not result.valid:
        details = result.parse_error or "; ".join(result.errors)
        raise OptimizationInputError(f"invalid {label}: {details}")
    if not isinstance(result.parsed, dict):
        raise OptimizationInputError(f"invalid {label}: expected a JSON object")
    return deepcopy(result.parsed)


def _metric_map(payload: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for metric in payload["metrics"]:
        name = metric["name"]
        if name in metrics:
            raise OptimizationInputError(f"invalid {label}: duplicate metric {name!r}")
        metrics[name] = metric
    return metrics


def _append_reason(
    reason_codes: set[str],
    reasons: set[str],
    code: str,
    reason: str,
) -> None:
    reason_codes.add(code)
    reasons.add(reason)


def _comparison_mismatches(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    policy: dict[str, Any],
    reason_codes: set[str],
    reasons: set[str],
) -> bool:
    mismatch = False
    if candidate["parent_id"] != baseline["candidate_id"]:
        mismatch = True
        _append_reason(
            reason_codes,
            reasons,
            "lineage_mismatch",
            "candidate parent_id does not identify the baseline candidate",
        )

    baseline_suite = baseline["evaluation_suite"]
    candidate_suite = candidate["evaluation_suite"]
    if baseline_suite != candidate_suite:
        mismatch = True
        _append_reason(
            reason_codes,
            reasons,
            "evaluation_suite_mismatch",
            "baseline and candidate use different evaluation-suite evidence",
        )

    baseline_provenance = baseline["provenance"]
    candidate_provenance = candidate["provenance"]
    evaluator_keys = ("evaluator", "evaluator_version")
    if any(baseline_provenance[key] != candidate_provenance[key] for key in evaluator_keys):
        mismatch = True
        _append_reason(
            reason_codes,
            reasons,
            "evaluator_mismatch",
            "baseline and candidate use different evaluator implementations",
        )
    if baseline_provenance["seed"] != candidate_provenance["seed"]:
        mismatch = True
        _append_reason(
            reason_codes,
            reasons,
            "seed_mismatch",
            "baseline and candidate use different evaluation seeds",
        )
    if (
        policy["require_same_environment"]
        and baseline_provenance["environment_sha256"] != candidate_provenance["environment_sha256"]
    ):
        mismatch = True
        _append_reason(
            reason_codes,
            reasons,
            "environment_mismatch",
            "baseline and candidate use different environment fingerprints",
        )
    return mismatch


def decide_optimization_step(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    policy_payload: dict[str, Any],
) -> dict[str, Any]:
    """Compare immutable evidence without executing or mutating either candidate."""
    baseline = _validated_payload(EVALUATION_SCHEMA, baseline_payload, "baseline")
    candidate = _validated_payload(EVALUATION_SCHEMA, candidate_payload, "candidate")
    policy = _validated_payload(POLICY_SCHEMA, policy_payload, "policy")

    if baseline["candidate_id"] == candidate["candidate_id"]:
        raise OptimizationInputError("baseline and candidate IDs must differ")

    baseline_metrics = _metric_map(baseline, "baseline")
    candidate_metrics = _metric_map(candidate, "candidate")
    policy_metrics = _metric_map(policy, "policy")
    objective_policies = [
        metric for metric in policy_metrics.values() if metric["role"] == "objective"
    ]
    if not objective_policies:
        raise OptimizationInputError("invalid policy: at least one objective metric is required")
    for metric in objective_policies:
        if metric["min_delta"] <= 0:
            raise OptimizationInputError(
                f"invalid policy: objective metric {metric['name']!r} must use min_delta > 0"
            )

    reason_codes: set[str] = set()
    reasons: set[str] = set()
    incomplete = _comparison_mismatches(baseline, candidate, policy, reason_codes, reasons)

    candidate_constraints = candidate["constraints"]
    failed_constraints = sorted(
        name for name in policy["required_constraints"] if candidate_constraints.get(name) is False
    )
    missing_constraints = sorted(
        name for name in policy["required_constraints"] if name not in candidate_constraints
    )
    if failed_constraints:
        _append_reason(
            reason_codes,
            reasons,
            "constraint_failed",
            "candidate failed required constraints: " + ", ".join(failed_constraints),
        )
    if missing_constraints:
        incomplete = True
        _append_reason(
            reason_codes,
            reasons,
            "constraint_missing",
            "candidate is missing required constraints: " + ", ".join(missing_constraints),
        )

    metric_results: list[dict[str, Any]] = []
    objective_improved = False
    metric_regressed = False
    confidence_z = _decimal_number(policy["confidence_z"], "policy/confidence_z")

    for name in sorted(policy_metrics):
        metric_policy = policy_metrics[name]
        baseline_metric = baseline_metrics.get(name)
        candidate_metric = candidate_metrics.get(name)
        if baseline_metric is None or candidate_metric is None:
            incomplete = True
            _append_reason(
                reason_codes,
                reasons,
                "metric_missing",
                f"metric {name!r} is missing from baseline or candidate evidence",
            )
            continue

        expected_unit = metric_policy["unit"]
        if baseline_metric["unit"] != expected_unit or candidate_metric["unit"] != expected_unit:
            incomplete = True
            _append_reason(
                reason_codes,
                reasons,
                "metric_unit_mismatch",
                f"metric {name!r} does not use policy unit {expected_unit!r}",
            )
            continue

        baseline_value = _decimal_number(baseline_metric["value"], f"baseline/metrics/{name}/value")
        candidate_value = _decimal_number(
            candidate_metric["value"], f"candidate/metrics/{name}/value"
        )

        baseline_samples = int(baseline_metric["sample_count"])
        candidate_samples = int(candidate_metric["sample_count"])
        insufficient_samples = (
            baseline_samples < metric_policy["min_samples"]
            or candidate_samples < metric_policy["min_samples"]
        )
        missing_uncertainty = confidence_z > 0 and (
            "standard_error" not in baseline_metric or "standard_error" not in candidate_metric
        )

        with localcontext() as context:
            context.prec = 50
            adjusted_delta = candidate_value - baseline_value
            if metric_policy["direction"] == "minimize":
                adjusted_delta = -adjusted_delta

            combined_error = Decimal(0)
            if not missing_uncertainty and confidence_z > 0:
                baseline_error = _decimal_number(
                    baseline_metric["standard_error"],
                    f"baseline/metrics/{name}/standard_error",
                )
                candidate_error = _decimal_number(
                    candidate_metric["standard_error"],
                    f"candidate/metrics/{name}/standard_error",
                )
                combined_error = (baseline_error**2 + candidate_error**2).sqrt()
                conservative_delta = adjusted_delta - confidence_z * combined_error
            else:
                conservative_delta = adjusted_delta

        min_delta = _decimal_number(metric_policy["min_delta"], f"policy/metrics/{name}/min_delta")
        max_regression = _decimal_number(
            metric_policy["max_regression"], f"policy/metrics/{name}/max_regression"
        )

        status = "acceptable"
        if insufficient_samples or missing_uncertainty:
            status = "insufficient"
            incomplete = True
            if insufficient_samples:
                _append_reason(
                    reason_codes,
                    reasons,
                    "insufficient_samples",
                    f"metric {name!r} does not meet its minimum sample count",
                )
            if missing_uncertainty:
                _append_reason(
                    reason_codes,
                    reasons,
                    "uncertainty_missing",
                    f"metric {name!r} lacks standard-error evidence required by confidence_z",
                )
        elif conservative_delta < -max_regression:
            status = "regressed"
            metric_regressed = True
            _append_reason(
                reason_codes,
                reasons,
                "metric_regression",
                f"metric {name!r} exceeds its allowed regression",
            )
        elif metric_policy["role"] == "objective" and conservative_delta >= min_delta:
            status = "improved"
            objective_improved = True

        metric_results.append(
            {
                "name": name,
                "unit": expected_unit,
                "role": metric_policy["role"],
                "direction": metric_policy["direction"],
                "baseline_value": _output_number(baseline_value, f"metrics/{name}/baseline_value"),
                "candidate_value": _output_number(
                    candidate_value, f"metrics/{name}/candidate_value"
                ),
                "baseline_samples": baseline_samples,
                "candidate_samples": candidate_samples,
                "adjusted_delta": _output_number(adjusted_delta, f"metrics/{name}/adjusted_delta"),
                "conservative_delta": _output_number(
                    conservative_delta, f"metrics/{name}/conservative_delta"
                ),
                "min_delta": _output_number(min_delta, f"metrics/{name}/min_delta"),
                "max_regression": _output_number(max_regression, f"metrics/{name}/max_regression"),
                "status": status,
            }
        )

    if failed_constraints or metric_regressed:
        decision = "rejected"
    elif incomplete:
        decision = "inconclusive"
    elif not objective_improved:
        decision = "inconclusive"
        _append_reason(
            reason_codes,
            reasons,
            "no_objective_improvement",
            "candidate does not meet any objective minimum improvement",
        )
    else:
        decision = "eligible_for_human_review"
        _append_reason(
            reason_codes,
            reasons,
            "candidate_improves_objective",
            "candidate improves an objective without violating a guardrail",
        )
        _append_reason(
            reason_codes,
            reasons,
            "approval_required",
            "this recommendation is not authorization to mutate active state",
        )

    decision_payload: dict[str, Any] = {
        "schema_version": "1.0",
        "algorithm_version": ALGORITHM_VERSION,
        "policy_id": policy["policy_id"],
        "baseline_id": baseline["candidate_id"],
        "candidate_id": candidate["candidate_id"],
        "decision": decision,
        "requires_human_approval": decision == "eligible_for_human_review",
        "reason_codes": sorted(reason_codes),
        "reasons": sorted(reasons),
        "metrics": metric_results,
        "failed_constraints": failed_constraints,
        "missing_constraints": missing_constraints,
        "input_hashes": {
            "baseline_sha256": _sha256_payload(baseline),
            "candidate_sha256": _sha256_payload(candidate),
            "policy_sha256": _sha256_payload(policy),
        },
    }
    decision_payload["decision_sha256"] = _sha256_payload(decision_payload)

    validation = validate_structured_response(DECISION_SCHEMA, decision_payload)
    if not validation.valid:
        raise RuntimeError(
            "generated optimization decision violated its schema: " + "; ".join(validation.errors)
        )
    return decision_payload
