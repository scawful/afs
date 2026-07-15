from __future__ import annotations

import json
from copy import deepcopy
from importlib import resources
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from afs.optimization import decide_optimization_step
from afs.protocols import list_protocol_schema_names
from afs.protocols.canonical_json import (
    CanonicalJSONError,
    canonical_json_bytes,
    sha256_canonical_json,
)
from afs.response_schemas import (
    get_response_schema,
    list_response_schema_names,
    validate_structured_response,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = ROOT / "examples" / "optimization_gate"
EXECUTION_SCHEMA_NAMES = {
    "v1/execution/request",
    "v1/execution/inspection",
    "v1/execution/record",
}


def _request() -> dict:
    return {
        "schema_version": "1.0",
        "command": {"kind": "argv", "argv": ["python3", "-c", "print('ok')"]},
        "caller": "pytest",
        "purpose": "exercise the execution request contract",
        "cwd": "/workspace",
        "inherit_env": ["PATH", "HOME"],
        "set_env": {"AFS_TEST": "1"},
        "timeout_seconds": 30,
        "max_output_bytes": 4096,
        "isolation": "process",
        "network": "inherit",
        "redact_argv_indices": [2],
    }


def test_execution_schemas_are_registered_packaged_draft_2020_12_contracts() -> None:
    assert EXECUTION_SCHEMA_NAMES.issubset(list_protocol_schema_names())
    assert EXECUTION_SCHEMA_NAMES.issubset(list_response_schema_names())

    schema_root = resources.files("afs.protocols.execution.v1")
    assert schema_root.joinpath("request.schema.json").is_file()
    assert schema_root.joinpath("inspection.schema.json").is_file()
    assert schema_root.joinpath("record.schema.json").is_file()

    for name in EXECUTION_SCHEMA_NAMES:
        schema = get_response_schema(name)
        Draft202012Validator.check_schema(schema)
        assert schema["$id"] == f"afs://schemas/{name}"


def test_execution_request_accepts_structured_argv_and_explicit_legacy_shell() -> None:
    request = _request()
    assert validate_structured_response("v1/execution/request", request).valid

    legacy = deepcopy(request)
    legacy["command"] = {"kind": "legacy_shell", "command": "printf '%s\\n' ok"}
    legacy["redact_argv_indices"] = []
    assert validate_structured_response("v1/execution/request", legacy).valid


def test_execution_request_rejects_ambiguous_command_and_unknown_fields() -> None:
    ambiguous = _request()
    ambiguous["command"] = {
        "kind": "argv",
        "argv": ["python3"],
        "command": "python3",
    }
    result = validate_structured_response("v1/execution/request", ambiguous)
    assert not result.valid

    unknown = _request()
    unknown["allowed_cwd_roots"] = ["/workspace"]
    result = validate_structured_response("v1/execution/request", unknown)
    assert not result.valid
    assert any("allowed_cwd_roots" in error for error in result.errors)


def test_execution_request_allows_known_but_unsupported_isolation_modes() -> None:
    request = _request()
    request["isolation"] = "sandbox"
    request["network"] = "deny"

    # These values are syntactically valid so the trusted execution policy can
    # distinguish a blocked request from malformed input.
    assert validate_structured_response("v1/execution/request", request).valid


def test_canonical_json_normalizes_numbers_without_reordering_arrays() -> None:
    integer = {"timeout_seconds": 500, "argv": ["b", "a"]}
    decimal = {"argv": ["b", "a"], "timeout_seconds": 500.0}

    assert canonical_json_bytes(integer) == canonical_json_bytes(decimal)
    assert sha256_canonical_json(integer) == sha256_canonical_json(decimal)
    assert canonical_json_bytes(integer) == b'{"argv":["b","a"],"timeout_seconds":500}'


def test_canonical_json_rejects_non_finite_and_unsupported_values() -> None:
    with pytest.raises(CanonicalJSONError, match="non-finite"):
        canonical_json_bytes({"value": float("nan")})
    with pytest.raises(CanonicalJSONError, match="does not support"):
        canonical_json_bytes({"value": object()})
    with pytest.raises(CanonicalJSONError, match="keys must be strings"):
        canonical_json_bytes({1: "not-json"})


def test_optimization_decision_hash_remains_golden_after_canonical_extraction() -> None:
    inputs = [
        json.loads((EXAMPLE_ROOT / filename).read_text(encoding="utf-8"))
        for filename in ("baseline.json", "candidate.json", "policy.json")
    ]

    decision = decide_optimization_step(*inputs)

    assert (
        decision["decision_sha256"]
        == "fb0663a2cdc7125d2aa5f0c9297616c8c546978ce89ce0354d08bcc7ba33fdb9"
    )
