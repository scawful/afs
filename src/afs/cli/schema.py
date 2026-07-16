"""CLI commands for AFS response-schema validation.

These close the loop the response schemas were built for: they are advertised to
models but were never checked. `afs schema validate` lets a post-turn hook (or an
agent) validate structured output and get a concrete correction message back.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..protocols.canonical_json import strict_json_loads
from ..response_schemas import (
    build_schema_correction,
    get_response_schema,
    list_response_schema_names,
    validate_structured_response,
    verify_human_intent_preserved,
)


def schema_list_command(args: argparse.Namespace) -> int:
    names = list_response_schema_names()
    if getattr(args, "json", False):
        print(json.dumps(names, indent=2))
        return 0
    for name in names:
        print(name)
    return 0


def schema_show_command(args: argparse.Namespace) -> int:
    try:
        schema = get_response_schema(args.name)
    except KeyError:
        print(
            f"unknown schema: {args.name!r}; known: "
            + ", ".join(list_response_schema_names()),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(schema, indent=2))
    return 0


def _read_response_text(args: argparse.Namespace) -> str:
    text = getattr(args, "text", None)
    if text is not None:
        return text
    file_arg = getattr(args, "file", None)
    if file_arg and file_arg != "-":
        return Path(file_arg).expanduser().read_text(encoding="utf-8")
    # Default: read the response from stdin (how a post-turn hook pipes it in).
    binary_stream = getattr(sys.stdin, "buffer", None)
    if binary_stream is None:
        return sys.stdin.read()
    return binary_stream.read().decode("utf-8")


def _resolve_schema_name(args: argparse.Namespace) -> tuple[str, str]:
    """Return ``(schema_name, error)``. Resolves --schema directly or --workflow.

    Workflow resolution lets a post-turn hook validate against the right schema
    without knowing the workflow->schema mapping (it mirrors what the session
    prompt already recommended).
    """
    schema = getattr(args, "schema", None)
    if schema:
        return schema, ""
    workflow = getattr(args, "workflow", None)
    if workflow:
        from ..verification import recommended_structured_schema

        return recommended_structured_schema(workflow), ""
    return "", "one of --schema or --workflow is required"


def schema_validate_command(args: argparse.Namespace) -> int:
    """Validate a structured response; exit 0 if valid, 1 if not, 2 on bad usage."""
    schema_name, resolve_error = _resolve_schema_name(args)
    if resolve_error:
        print(resolve_error, file=sys.stderr)
        return 2

    known = set(list_response_schema_names())
    if schema_name not in known:
        print(
            f"unknown schema: {schema_name!r}; known: " + ", ".join(sorted(known)),
            file=sys.stderr,
        )
        return 2

    try:
        response_text = _read_response_text(args)
    except UnicodeError as exc:
        print(f"response input is not valid UTF-8: {exc}", file=sys.stderr)
        return 2
    result = validate_structured_response(schema_name, response_text)

    intent_violations: list[str] = []
    skeleton_arg = getattr(args, "skeleton", None)
    if skeleton_arg:
        try:
            skeleton_text = Path(skeleton_arg).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            print(f"cannot read skeleton: {exc}", file=sys.stderr)
            return 2
        # The skeleton is a trust anchor: parse it at least as strictly as the
        # response it verifies — no fence extraction or repairs, no duplicate
        # object keys (first key shown, last key verified), no NaN/Infinity —
        # so that what is verified is exactly what the human wrote.
        try:
            skeleton = strict_json_loads(skeleton_text)
        except ValueError as exc:
            print(f"invalid skeleton: {exc}", file=sys.stderr)
            return 2
        if not isinstance(skeleton, dict):
            print("invalid skeleton: must be a JSON object", file=sys.stderr)
            return 2
        intent_violations = verify_human_intent_preserved(skeleton, result.parsed)

    valid = result.valid and not intent_violations
    if getattr(args, "json", False):
        payload = result.to_dict()
        payload["valid"] = valid
        payload["human_intent_violations"] = intent_violations
        payload["correction"] = build_schema_correction(result)
        print(json.dumps(payload, indent=2))
    elif valid:
        print(f"valid: response matches `{schema_name}`")
    else:
        correction = build_schema_correction(result)
        if correction:
            print(correction)
        for violation in intent_violations:
            print(f"- {violation}")

    return 0 if valid else 1


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    schema_parser = subparsers.add_parser(
        "schema", help="Inspect and validate AFS structured-response schemas."
    )
    schema_sub = schema_parser.add_subparsers(dest="schema_command")

    list_parser = schema_sub.add_parser("list", help="List response schema names.")
    list_parser.add_argument("--json", action="store_true", help="Output JSON.")
    list_parser.set_defaults(func=schema_list_command)

    show_parser = schema_sub.add_parser("show", help="Print a response schema as JSON.")
    show_parser.add_argument("name", help="Schema name (see `afs schema list`).")
    show_parser.set_defaults(func=schema_show_command)

    validate_parser = schema_sub.add_parser(
        "validate",
        help="Validate a structured response against a schema (exit 1 on mismatch).",
    )
    target = validate_parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--schema", help="Schema name to validate against.")
    target.add_argument(
        "--workflow",
        help="Resolve the schema from a workflow (e.g. edit_fast, review_deep).",
    )
    source = validate_parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Response text to validate (inline).")
    source.add_argument(
        "--file", help="Read response from a file ('-' or omitted reads stdin)."
    )
    validate_parser.add_argument(
        "--skeleton",
        help="Human-authored skeleton plan (JSON file); fails validation if "
        "the response modified or authored its human_intent section.",
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full validation result (with correction) as JSON.",
    )
    validate_parser.set_defaults(func=schema_validate_command)
