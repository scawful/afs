"""Read-only optimization evidence commands."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

from ..optimization import (
    OptimizationInputError,
    canonical_json_text,
    decide_optimization_step,
)
from ..protocols.canonical_json import CanonicalJSONError, strict_json_loads

_MAX_INPUT_BYTES = 2 * 1024 * 1024
_DECISION_EXIT_CODES = {
    "eligible_for_human_review": 0,
    "rejected": 1,
    "inconclusive": 3,
}
_EXIT_INVALID_INPUT = 2
_EXIT_INTERNAL_ERROR = 4


def _load_json_object(path_value: str, label: str) -> dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OptimizationInputError(f"{label} is not a regular file: {path}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            raw_payload = stream.read(_MAX_INPUT_BYTES + 1)
    except OSError as exc:
        raise OptimizationInputError(f"cannot read {label} {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(raw_payload) > _MAX_INPUT_BYTES:
        raise OptimizationInputError(
            f"{label} exceeds the {_MAX_INPUT_BYTES}-byte input limit: {path}"
        )
    try:
        payload = strict_json_loads(raw_payload.decode("utf-8"))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        CanonicalJSONError,
        ValueError,
    ) as exc:
        raise OptimizationInputError(f"invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OptimizationInputError(f"invalid {label} JSON {path}: expected an object")
    return payload


def optimize_decide_command(args: argparse.Namespace) -> int:
    """Evaluate immutable evidence without executing or promoting a candidate."""
    try:
        baseline = _load_json_object(args.baseline, "baseline")
        candidate = _load_json_object(args.candidate, "candidate")
        policy = _load_json_object(args.policy, "policy")
        decision = decide_optimization_step(baseline, candidate, policy)
        rendered = canonical_json_text(decision) if args.json else None
    except OptimizationInputError as exc:
        print(f"afs optimize decide: {exc}", file=sys.stderr)
        return _EXIT_INVALID_INPUT
    except Exception as exc:
        # A gate crash must never share an exit code with an evidence verdict.
        print(
            f"afs optimize decide: internal error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return _EXIT_INTERNAL_ERROR

    if rendered is not None:
        sys.stdout.write(rendered)
    else:
        print(f"decision: {decision['decision']}")
        print(f"baseline: {decision['baseline_id']}")
        print(f"candidate: {decision['candidate_id']}")
        print(f"decision_sha256: {decision['decision_sha256']}")
        for reason in decision["reasons"]:
            print(f"  - {reason}")
        if decision["requires_human_approval"]:
            print("next: create a separately verified, exact-hash human approval record")

    return _DECISION_EXIT_CODES[decision["decision"]]


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Compare versioned optimization evidence without executing candidates.",
    )
    optimize_sub = optimize_parser.add_subparsers(dest="optimize_command")

    decide_parser = optimize_sub.add_parser(
        "decide",
        help="Return a deterministic review recommendation for one candidate step.",
    )
    decide_parser.add_argument("--baseline", required=True, help="Baseline evaluation JSON.")
    decide_parser.add_argument("--candidate", required=True, help="Candidate evaluation JSON.")
    decide_parser.add_argument("--policy", required=True, help="Optimization policy JSON.")
    decide_parser.add_argument("--json", action="store_true", help="Output canonical JSON.")
    decide_parser.set_defaults(func=optimize_decide_command)
