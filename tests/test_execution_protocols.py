from __future__ import annotations

import json
from copy import deepcopy
from importlib import resources
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from afs.execution import (
    DEFAULT_INHERITED_ENV,
    ArgvCommand,
    ExecutionInspection,
    ExecutionPolicy,
    ExecutionRequest,
    execute_checked,
    inspect_execution,
)
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


def test_execution_request_schema_rejects_lone_unicode_surrogates() -> None:
    request = _request()
    request["command"]["argv"][2] = "\ud800"

    result = validate_structured_response("v1/execution/request", request)

    assert not result.valid


def test_execution_request_allows_known_but_unsupported_isolation_modes() -> None:
    request = _request()
    request["isolation"] = "sandbox"
    request["network"] = "deny"

    # These values are syntactically valid so the trusted execution policy can
    # distinguish a blocked request from malformed input.
    assert validate_structured_response("v1/execution/request", request).valid


def test_execution_protocols_accept_maximum_effective_environment(
    tmp_path: Path,
) -> None:
    inherited = tuple(f"AFS_INHERITED_{index}" for index in range(128))
    explicit = {f"AFS_EXPLICIT_{index}": "1" for index in range(128)}
    environment = {
        **dict.fromkeys(inherited, "1"),
        **dict.fromkeys(DEFAULT_INHERITED_ENV, "1"),
    }
    request = ExecutionRequest(
        command=ArgvCommand(("afs-environment-boundary-test",)),
        caller="pytest",
        purpose="exercise the maximum effective environment cardinality",
        cwd=tmp_path,
        inherit_env=inherited,
        set_env=explicit,
        network="deny",
    )
    policy = ExecutionPolicy(
        allowed_cwd_roots=(tmp_path,),
        allowed_executables=frozenset({"afs-environment-boundary-test"}),
        allowed_env=frozenset((*inherited, *explicit)),
    )

    inspection = inspect_execution(request, policy, environ=environment)
    record = execute_checked(request, policy, environ=environment)

    assert len(inspection.environment_keys) == 261
    assert len(record.environment_keys) == 261
    assert validate_structured_response(
        "v1/execution/inspection", inspection.to_dict()
    ).valid
    assert validate_structured_response("v1/execution/record", record.audit_dict()).valid


def test_execution_protocol_bounds_denial_reasons_and_record_error(
    tmp_path: Path,
) -> None:
    inherited: list[str] = []
    for index in range(128):
        prefix = f"AFS_DENIED_{index:03d}_"
        inherited.append(prefix + "X" * (256 - len(prefix)))
    unresolved_executable = str(tmp_path / ("x" * 4096))
    environment = dict.fromkeys(inherited, "1")
    request = ExecutionRequest(
        command=ArgvCommand((unresolved_executable,)),
        caller="pytest",
        purpose="exercise bounded protocol denial reasons",
        cwd=tmp_path,
        inherit_env=tuple(inherited),
    )
    policy = ExecutionPolicy(
        allowed_cwd_roots=(tmp_path,),
        allowed_executables=frozenset({unresolved_executable}),
    )

    inspection = inspect_execution(request, policy, environ=environment)
    record = execute_checked(request, policy, environ=environment)
    audit = record.audit_dict()

    assert not inspection.allowed
    assert len(inspection.reasons[0]) == 2048
    assert len(inspection.reasons[1]) == 2048
    assert "...<truncated:" in inspection.reasons[0]
    assert "...<truncated:" in inspection.reasons[1]
    assert len(audit["error"]) == 4096
    assert "...<truncated:" in audit["error"]
    assert validate_structured_response(
        "v1/execution/inspection", inspection.to_dict()
    ).valid
    assert validate_structured_response("v1/execution/record", audit).valid


def test_truncated_protocol_reasons_remain_unique(tmp_path: Path) -> None:
    shared_prefix = "same-prefix-" + "x" * 4096
    inspection = ExecutionInspection(
        allowed=False,
        request_sha256="a" * 64,
        resolved_executable="",
        resolved_cwd=str(tmp_path),
        redacted_argv=(),
        environment_keys=(),
        environment_sha256="b" * 64,
        timeout_seconds=1,
        max_output_bytes=1,
        isolation="process",
        network="inherit",
        reason_codes=("first", "second"),
        reasons=(shared_prefix + "A", shared_prefix + "B"),
    )

    payload = inspection.to_dict()

    assert len(set(payload["reasons"])) == 2
    assert validate_structured_response("v1/execution/inspection", payload).valid


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
