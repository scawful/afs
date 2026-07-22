"""Fail-closed context layout inspection, planning, and migration commands."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import load_runtime_config_model
from ..context_layout import (
    MigrationPlan,
    audit_layout,
    build_migration_plan,
    build_rollback_manifest,
    load_migration_plan,
    write_manifest,
)
from ..core import resolve_context_root
from ..human_provenance import _broker_for_reader
from ..layout_activation import (
    ActivationApplyError,
    ActivationPreflight,
    ActivationPreflightError,
    RollbackApplyError,
    RollbackPreflight,
    RollbackPreflightError,
    activate_layout,
    activation_confirmation_token,
    layout_activation_authorization_scope,
    layout_rollback_authorization_scope,
    preflight_activation,
    preflight_rollback,
    rollback_confirmation_token,
    rollback_layout,
)
from ..layout_migration import (
    MigrationApplyError,
    MigrationPreflightError,
    apply_migration,
    layout_migration_authorization_scope,
    preflight_migration,
)

_TTY_READER = None
_MAX_MAPPING_BYTES = 1024 * 1024
_MAX_RATIONALE_CHARS = 4096
_MAX_RETENTION_REASON_CHARS = 1024


@dataclass(frozen=True)
class _MappingDocument:
    mappings: dict[str, str]
    retained_sources: dict[str, str]
    retained_paths: dict[str, str]


def _context_root(args: argparse.Namespace) -> Path:
    return Path(args.context_root or Path.home() / ".context").expanduser()


def _file_signature(path_stat: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        path_stat.st_dev,
        path_stat.st_ino,
        path_stat.st_mode,
        path_stat.st_size,
        path_stat.st_mtime_ns,
        path_stat.st_ctime_ns,
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"mapping file contains duplicate field {key!r}")
        result[key] = value
    return result


def _read_mapping_file(path: Path) -> _MappingDocument:
    """Read one small, stable, no-follow mapping document."""

    mapping_path = path.expanduser()
    try:
        before = os.lstat(mapping_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"mapping file does not exist: {mapping_path}") from None
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"mapping file must be a regular file: {mapping_path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(mapping_path, flags)
    chunks: list[bytes] = []
    total = 0
    try:
        opened = os.fstat(descriptor)
        if _file_signature(before) != _file_signature(opened):
            raise ValueError("mapping file changed while opening")
        while chunk := os.read(descriptor, 64 * 1024):
            total += len(chunk)
            if total > _MAX_MAPPING_BYTES:
                raise ValueError("mapping file exceeds the 1 MiB size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        final_path_stat = os.lstat(mapping_path)
    except FileNotFoundError as exc:
        raise ValueError("mapping file changed while reading") from exc
    if _file_signature(before) != _file_signature(after) or _file_signature(
        after
    ) != _file_signature(final_path_stat):
        raise ValueError("mapping file changed while reading")
    try:
        payload = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid mapping file JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("mapping file must be a JSON object")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise ValueError("mapping file schema_version must be 1 or 2")
    expected = {"schema_version", "mappings"}
    if schema_version == 2:
        expected.update({"retained_sources", "retained_paths"})
    keys = set(payload)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        raise ValueError(f"mapping file fields are invalid ({'; '.join(details)})")
    mappings = payload["mappings"]
    if not isinstance(mappings, dict):
        raise ValueError("mapping file field 'mappings' must be an object")
    if any(type(key) is not str or type(value) is not str for key, value in mappings.items()):
        raise ValueError("mapping file names and destinations must be strings")
    retentions: dict[str, dict[str, str]] = {
        "retained_sources": {},
        "retained_paths": {},
    }
    if schema_version == 2:
        for field_name in retentions:
            raw = payload[field_name]
            if not isinstance(raw, dict):
                raise ValueError(f"mapping file field {field_name!r} must be an object")
            if any(type(key) is not str or type(value) is not str for key, value in raw.items()):
                raise ValueError(f"mapping file {field_name} paths and reasons must be strings")
            normalized: dict[str, str] = {}
            for source, reason in raw.items():
                if (
                    not reason
                    or reason != reason.strip()
                    or len(reason) > _MAX_RETENTION_REASON_CHARS
                    or any(unicodedata.category(character).startswith("C") for character in reason)
                ):
                    raise ValueError(
                        f"mapping file {field_name} reason for {source!r} must be "
                        "reviewed non-empty text without surrounding whitespace, "
                        f"control characters, or more than {_MAX_RETENTION_REASON_CHARS} "
                        "characters"
                    )
                normalized[source] = reason
            retentions[field_name] = normalized
    return _MappingDocument(
        mappings=dict(mappings),
        retained_sources=retentions["retained_sources"],
        retained_paths=retentions["retained_paths"],
    )


def _terminal_text(value: Any, *, limit: int = 4096) -> str:
    printable = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in str(value)
    )
    return " ".join(printable.split())[:limit]


def _assert_manifest_path_outside_roots(
    path: Path,
    *,
    source: Path,
    destination: Path,
    label: str,
) -> None:
    resolved = path.expanduser().resolve(strict=False)
    source_resolved = source.expanduser().resolve(strict=False)
    destination_resolved = destination.expanduser().resolve(strict=False)
    if resolved == source_resolved or resolved.is_relative_to(source_resolved):
        raise ValueError(f"{label} must be outside the migration source root")
    if resolved == destination_resolved or resolved.is_relative_to(destination_resolved):
        raise ValueError(f"{label} must be outside the migration destination root")


def _rationale(args: argparse.Namespace) -> str | None:
    rationale = str(getattr(args, "because", "") or "").strip()
    if not rationale:
        _print_error_payload(
            {
                "status": "blocked",
                "error": "A rationale is required to apply a layout migration: pass "
                '--because "<why this candidate should be created>".',
            },
            as_json=args.json,
        )
        return None
    if len(rationale) > _MAX_RATIONALE_CHARS:
        _print_error_payload(
            {
                "status": "blocked",
                "error": "Layout migration rationale must be no more than "
                f"{_MAX_RATIONALE_CHARS} characters.",
            },
            as_json=args.json,
        )
        return None
    if any(unicodedata.category(character).startswith("C") for character in rationale):
        _print_error_payload(
            {
                "status": "blocked",
                "error": "Layout migration rationale must not contain control or "
                "formatting characters.",
            },
            as_json=args.json,
        )
        return None
    return rationale


def _transition_rationale(args: argparse.Namespace, action: str) -> str | None:
    rationale = str(getattr(args, "because", "") or "").strip()
    if not rationale:
        _print_error_payload(
            {
                "status": "blocked",
                "error": f'A rationale is required to {action}: pass --because "<why>".',
            },
            as_json=args.json,
        )
        return None
    if len(rationale) > _MAX_RATIONALE_CHARS:
        _print_error_payload(
            {
                "status": "blocked",
                "error": f"{action.title()} rationale must be no more than "
                f"{_MAX_RATIONALE_CHARS} characters.",
            },
            as_json=args.json,
        )
        return None
    if any(unicodedata.category(character).startswith("C") for character in rationale):
        _print_error_payload(
            {
                "status": "blocked",
                "error": f"{action.title()} rationale must not contain control or "
                "formatting characters.",
            },
            as_json=args.json,
        )
        return None
    return rationale


def _confirm_apply(plan: MigrationPlan, rationale: str):
    scope = layout_migration_authorization_scope(
        plan.plan_sha256,
        plan.transaction_id,
        rationale,
    )
    source_only_lines = [
        f"    - {_terminal_text(item.source)}: {_terminal_text(item.reason)}"
        for item in (*plan.retained_sources, *plan.retained_paths)
    ]
    prompt_lines = [
        "",
        "=== HUMAN CONFIRMATION REQUIRED (layout migration) ===",
        f"  transaction: {_terminal_text(plan.transaction_id)}",
        f"  plan sha256: {_terminal_text(plan.plan_sha256)}",
        f"  source root: {_terminal_text(plan.source_root)}",
        f"  destination: {_terminal_text(plan.destination_root)}",
        f"  source data: {plan.source_file_count} files / {plan.source_bytes} bytes",
        f"  candidate:   {plan.copy_file_count} files / {plan.copy_bytes} bytes",
        "  source-only: "
        f"{len(plan.retained_sources)} top-level sources and "
        f"{len(plan.retained_paths)} nested paths WILL NOT be copied into the candidate.",
    ]
    if source_only_lines:
        prompt_lines.extend(["  source-only exclusions:", *source_only_lines])
    prompt_lines.extend(
        [
            f"  because:     {_terminal_text(rationale)}",
            "  source will be retained; this does not activate the candidate.",
            f"Type '{plan.transaction_id}' to confirm, anything else aborts: ",
        ]
    )
    prompt = "\n".join(prompt_lines)
    return _broker_for_reader(_TTY_READER).confirm_token(
        plan.transaction_id,
        prompt,
        scope=scope,
    )


def _print_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    for key, value in payload.items():
        if isinstance(value, list):
            print(f"{key}:")
            if value:
                for item in value:
                    rendered = (
                        json.dumps(item, ensure_ascii=False, sort_keys=True)
                        if isinstance(item, dict)
                        else _terminal_text(item)
                    )
                    print(f"- {rendered}")
            else:
                print("- (none)")
        elif isinstance(value, bool):
            print(f"{key}: {str(value).lower()}")
        elif value is not None:
            print(f"{key}: {_terminal_text(value)}")


def _print_error_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    print(
        _terminal_text(payload.get("error", "layout migration blocked")),
        file=sys.stderr,
    )
    failed_destination = payload.get("failed_destination")
    if failed_destination:
        print(f"failed_destination: {_terminal_text(failed_destination)}", file=sys.stderr)


def layout_audit_command(args: argparse.Namespace) -> int:
    audit = audit_layout(_context_root(args))
    payload = audit.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"context_root: {_terminal_text(audit.context_root)}")
        print(f"layout_version: {audit.layout_version}")
        print(f"valid: {str(audit.valid).lower()}")
        print(f"migration_ready: {str(audit.migration_ready).lower()}")
        print("issues:")
        for issue in audit.issues:
            suffix = " (blocking)" if issue.blocking else ""
            print(f"- {_terminal_text(issue.code)}: {_terminal_text(issue.message)}{suffix}")
        if not audit.issues:
            print("- (none)")
    return 0 if audit.valid else 1


def layout_plan_command(args: argparse.Namespace) -> int:
    try:
        source = _context_root(args)
        destination = Path(args.destination_root).expanduser()
        artifact_paths = {
            label: Path(value).expanduser().resolve(strict=False)
            for label, value in (
                ("mapping file", args.mapping_file),
                ("migration plan output", args.output),
                ("rollback manifest output", args.rollback_output),
            )
            if value
        }
        if len(set(artifact_paths.values())) != len(artifact_paths):
            raise ValueError("mapping, plan, and rollback files must use distinct paths")
        if args.mapping_file:
            _assert_manifest_path_outside_roots(
                Path(args.mapping_file),
                source=source,
                destination=destination,
                label="mapping file",
            )
        if args.output:
            _assert_manifest_path_outside_roots(
                Path(args.output),
                source=source,
                destination=destination,
                label="migration plan output",
            )
        if args.rollback_output:
            _assert_manifest_path_outside_roots(
                Path(args.rollback_output),
                source=source,
                destination=destination,
                label="rollback manifest output",
            )
        decisions = _read_mapping_file(Path(args.mapping_file)) if args.mapping_file else None
        plan = build_migration_plan(
            source,
            destination,
            explicit_mappings=decisions.mappings if decisions else None,
            retained_sources=decisions.retained_sources if decisions else None,
            retained_paths=decisions.retained_paths if decisions else None,
        )
        rollback = build_rollback_manifest(plan)
        if args.output:
            write_manifest(Path(args.output), plan)
        if args.rollback_output:
            write_manifest(Path(args.rollback_output), rollback)
    except (OSError, ValueError) as exc:
        _print_error_payload(
            {"status": "blocked", "error": f"layout plan blocked: {exc}"},
            as_json=args.json,
        )
        return 2
    payload = {
        "plan": plan.to_dict(),
        "rollback": rollback.to_dict(),
        "plan_path": str(Path(args.output).expanduser().resolve()) if args.output else None,
        "rollback_path": str(Path(args.rollback_output).expanduser().resolve())
        if args.rollback_output
        else None,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"transaction_id: {plan.transaction_id}")
        print(f"plan_sha256: {plan.plan_sha256}")
        print(f"ready: {str(plan.ready).lower()}")
        print(f"source_fingerprint: {plan.source_fingerprint}")
        print(f"source_files: {plan.source_file_count}")
        print(f"source_bytes: {plan.source_bytes}")
        print(f"copy_files: {plan.copy_file_count}")
        print(f"copy_bytes: {plan.copy_bytes}")
        print(f"operations: {len(plan.operations)}")
        if plan.explicit_mappings:
            print("explicit_mappings:")
            for mapping in plan.explicit_mappings:
                print(
                    f"- {_terminal_text(mapping.source)} -> {_terminal_text(mapping.destination)}"
                )
        if plan.retained_sources:
            print("source_only_top_level_exclusions:")
            for retained in plan.retained_sources:
                print(f"- {_terminal_text(retained.source)} — {_terminal_text(retained.reason)}")
        if plan.retained_paths:
            print("source_only_nested_exclusions:")
            for retained in plan.retained_paths:
                print(f"- {_terminal_text(retained.source)} — {_terminal_text(retained.reason)}")
        if plan.blocking_entries:
            print("blocking_entries:")
            for entry in plan.blocking_entries:
                print(f"- {_terminal_text(entry)}")
        if args.output:
            print(f"plan_path: {payload['plan_path']}")
        if args.rollback_output:
            print(f"rollback_path: {payload['rollback_path']}")
    return 0 if plan.ready else 2


def layout_migrate_command(args: argparse.Namespace) -> int:
    if args.because and not args.apply:
        _print_error_payload(
            {
                "status": "blocked",
                "error": "--because is only valid together with --apply",
            },
            as_json=args.json,
        )
        return 2
    try:
        plan = load_migration_plan(Path(args.plan))
        preview = preflight_migration(plan)
    except (OSError, ValueError, MigrationPreflightError) as exc:
        payload = {"status": "blocked", "error": str(exc)}
        _print_error_payload(payload, as_json=args.json)
        return 2

    if preview.status == "already_applied":
        payload = preview.to_dict()
        payload["mode"] = "verified_existing_candidate"
        _print_payload(payload, as_json=args.json)
        return 0

    if not args.apply:
        payload = preview.to_dict()
        payload["mode"] = "preview"
        _print_payload(payload, as_json=args.json)
        return 0

    rationale = _rationale(args)
    if rationale is None:
        return 2
    authorization = _confirm_apply(plan, rationale)
    if authorization is None:
        _print_error_payload(
            {
                "status": "blocked",
                "error": "Layout migration requires interactive human confirmation through "
                "the controlling terminal; candidate creation was not started.",
            },
            as_json=args.json,
        )
        return 2
    try:
        result = apply_migration(
            plan,
            rationale=rationale,
            authorization=authorization,
        )
    except MigrationPreflightError as exc:
        payload = {"status": "blocked", "error": str(exc)}
        _print_error_payload(payload, as_json=args.json)
        return 2
    except MigrationApplyError as exc:
        failure_payload: dict[str, Any] = {
            "status": "failed",
            "error": str(exc),
            "failed_destination": str(exc.failed_destination)
            if exc.failed_destination is not None
            else None,
        }
        _print_error_payload(failure_payload, as_json=args.json)
        return 3
    _print_payload(result.to_dict(), as_json=args.json)
    return 0


def _configured_context_root(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).expanduser() if getattr(args, "config", None) else None
    config, _resolved_path = load_runtime_config_model(
        config_path=config_path,
        start_dir=Path.cwd(),
    )
    return resolve_context_root(config, None)


def _confirm_activation(preflight: ActivationPreflight, rationale: str):
    token = activation_confirmation_token(preflight, rationale)
    action = (
        "finalize the pending activation receipt"
        if preflight.status == "receipt_pending"
        else ("atomically exchange the v1 and v2 roots")
    )
    prompt = "\n".join(
        (
            "",
            "=== HUMAN CONFIRMATION REQUIRED (layout activation) ===",
            f"  activation: {_terminal_text(preflight.activation_id)}",
            f"  plan sha256: {_terminal_text(preflight.evidence.result.plan_hash)}",
            f"  action:      {action}",
            f"  active path: {_terminal_text(preflight.active_root)}",
            f"  inactive:    {_terminal_text(preflight.inactive_root)}",
            f"  because:     {_terminal_text(rationale)}",
            "  v1 will remain intact at the inactive path.",
            "  this does not merge data; rollback requires a separate human decision.",
            f"Type '{token}' to confirm, anything else aborts: ",
        )
    )
    return _broker_for_reader(_TTY_READER).confirm_token(
        token,
        prompt,
        scope=layout_activation_authorization_scope(preflight, rationale),
    )


def _confirm_rollback(preflight: RollbackPreflight, rationale: str):
    token = rollback_confirmation_token(preflight, rationale)
    action = (
        "finalize the pending rollback receipt"
        if preflight.status == "receipt_pending"
        else ("atomically restore v1")
    )
    prompt = "\n".join(
        (
            "",
            "=== HUMAN CONFIRMATION REQUIRED (layout rollback) ===",
            f"  activation:  {_terminal_text(preflight.activation_id)}",
            f"  action:      {action}",
            f"  active path: {_terminal_text(preflight.active_root)}",
            f"  inactive:    {_terminal_text(preflight.inactive_root)}",
            f"  because:     {_terminal_text(rationale)}",
            "  v2 and all v2-era writes will be preserved at the inactive path.",
            "  no data will be merged or deleted.",
            f"Type '{token}' to confirm, anything else aborts: ",
        )
    )
    return _broker_for_reader(_TTY_READER).confirm_token(
        token,
        prompt,
        scope=layout_rollback_authorization_scope(preflight, rationale),
    )


def layout_activate_command(args: argparse.Namespace) -> int:
    if args.because and not args.apply:
        _print_error_payload(
            {"status": "blocked", "error": "--because is only valid together with --apply"},
            as_json=args.json,
        )
        return 2
    try:
        plan = load_migration_plan(Path(args.plan))
        configured_root = _configured_context_root(args)
        preview = preflight_activation(
            plan,
            Path(args.state_dir),
            configured_root,
        )
    except (OSError, ValueError, ActivationPreflightError) as exc:
        _print_error_payload(
            {"status": "blocked", "error": f"layout activation blocked: {exc}"},
            as_json=args.json,
        )
        return 2
    if preview.status == "already_active":
        payload = preview.to_dict()
        payload["mode"] = "verified_active"
        _print_payload(payload, as_json=args.json)
        return 0
    if preview.status == "already_rolled_back":
        payload = preview.to_dict()
        payload["mode"] = "verified_rolled_back"
        _print_payload(payload, as_json=args.json)
        return 0
    if not args.apply:
        payload = preview.to_dict()
        payload["mode"] = "preview"
        _print_payload(payload, as_json=args.json)
        return 0
    rationale = _transition_rationale(args, "activate the v2 context")
    if rationale is None:
        return 2
    authorization = _confirm_activation(preview, rationale)
    if authorization is None:
        _print_error_payload(
            {
                "status": "blocked",
                "error": "Layout activation requires interactive human confirmation through "
                "the controlling terminal; roots were not exchanged.",
            },
            as_json=args.json,
        )
        return 2
    try:
        result = activate_layout(
            plan,
            Path(args.state_dir),
            configured_root,
            rationale=rationale,
            authorization=authorization,
        )
    except ActivationPreflightError as exc:
        _print_error_payload(
            {"status": "blocked", "error": str(exc)},
            as_json=args.json,
        )
        return 2
    except ActivationApplyError as exc:
        _print_error_payload(
            {"status": "failed", "error": str(exc)},
            as_json=args.json,
        )
        return 3
    _print_payload(result.to_dict(), as_json=args.json)
    return 0


def layout_rollback_command(args: argparse.Namespace) -> int:
    if args.because and not args.apply:
        _print_error_payload(
            {"status": "blocked", "error": "--because is only valid together with --apply"},
            as_json=args.json,
        )
        return 2
    try:
        configured_root = _configured_context_root(args)
        preview = preflight_rollback(Path(args.state_dir), configured_root)
    except (OSError, ValueError, RollbackPreflightError) as exc:
        _print_error_payload(
            {"status": "blocked", "error": f"layout rollback blocked: {exc}"},
            as_json=args.json,
        )
        return 2
    if preview.status == "already_rolled_back":
        payload = preview.to_dict()
        payload["mode"] = "verified_rolled_back"
        _print_payload(payload, as_json=args.json)
        return 0
    if not args.apply:
        payload = preview.to_dict()
        payload["mode"] = "preview"
        _print_payload(payload, as_json=args.json)
        return 0
    rationale = _transition_rationale(args, "rollback the v2 context")
    if rationale is None:
        return 2
    authorization = _confirm_rollback(preview, rationale)
    if authorization is None:
        _print_error_payload(
            {
                "status": "blocked",
                "error": "Layout rollback requires interactive human confirmation through "
                "the controlling terminal; roots were not exchanged.",
            },
            as_json=args.json,
        )
        return 2
    try:
        result = rollback_layout(
            Path(args.state_dir),
            configured_root,
            rationale=rationale,
            authorization=authorization,
        )
    except RollbackPreflightError as exc:
        _print_error_payload(
            {"status": "blocked", "error": str(exc)},
            as_json=args.json,
        )
        return 2
    except RollbackApplyError as exc:
        _print_error_payload(
            {"status": "failed", "error": str(exc)},
            as_json=args.json,
        )
        return 3
    _print_payload(result.to_dict(), as_json=args.json)
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    layout = subparsers.add_parser(
        "layout",
        help="Audit, migrate, activate, or rollback a versioned context layout.",
    )
    commands = layout.add_subparsers(dest="layout_command")

    audit = commands.add_parser("audit", help="Inspect layout health without changing files.")
    audit.add_argument("--context-root", help="Context root (default: ~/.context).")
    audit.add_argument("--json", action="store_true", help="Output JSON.")
    audit.set_defaults(func=layout_audit_command, _skip_cli_history=True)

    plan = commands.add_parser(
        "plan",
        help="Build a hash-bound v1-to-v2 plan without executing it.",
    )
    plan.add_argument("--context-root", help="Source context root (default: ~/.context).")
    plan.add_argument(
        "--destination-root",
        required=True,
        help="Separate, nonexistent v2 destination root.",
    )
    plan.add_argument(
        "--mapping-file",
        help=("Reviewed schema-v1/v2 JSON decisions for mappings and source-only exclusions."),
    )
    plan.add_argument("--output", help="Write the private JSON migration plan atomically.")
    plan.add_argument(
        "--rollback-output",
        help="Write an informational source-retention manifest atomically.",
    )
    plan.add_argument("--json", action="store_true", help="Output JSON.")
    plan.set_defaults(func=layout_plan_command, _skip_cli_history=True)

    migrate = commands.add_parser(
        "migrate",
        help="Preview or human-confirm creation of a separate v2 candidate.",
    )
    migrate.add_argument(
        "--plan",
        required=True,
        help="Private supported migration plan path.",
    )
    migrate.add_argument(
        "--apply",
        action="store_true",
        help="Create and verify the separate candidate after human confirmation.",
    )
    migrate.add_argument("--because", help="Human rationale required with --apply.")
    migrate.add_argument("--json", action="store_true", help="Output JSON.")
    migrate.set_defaults(func=layout_migrate_command, _skip_cli_history=True)

    activate = commands.add_parser(
        "activate",
        help="Preview or human-confirm atomic activation of a verified v2 candidate.",
    )
    activate.add_argument("--plan", required=True, help="Private completed migration plan path.")
    activate.add_argument(
        "--state-dir",
        required=True,
        help="External private directory for the activation journal and receipts.",
    )
    activate.add_argument("--config", help="Runtime config that must resolve the stable root.")
    activate.add_argument(
        "--apply",
        action="store_true",
        help="Atomically exchange v1 and v2 after human confirmation.",
    )
    activate.add_argument("--because", help="Human rationale required with --apply.")
    activate.add_argument("--json", action="store_true", help="Output JSON.")
    activate.set_defaults(func=layout_activate_command, _skip_cli_history=True)

    rollback = commands.add_parser(
        "rollback",
        help="Preview or human-confirm restoration of the preserved v1 root.",
    )
    rollback.add_argument(
        "--state-dir",
        required=True,
        help="External private activation state directory.",
    )
    rollback.add_argument("--config", help="Runtime config that must resolve the stable root.")
    rollback.add_argument(
        "--apply",
        action="store_true",
        help="Atomically restore v1 after a separate human confirmation.",
    )
    rollback.add_argument("--because", help="Human rationale required with --apply.")
    rollback.add_argument("--json", action="store_true", help="Output JSON.")
    rollback.set_defaults(func=layout_rollback_command, _skip_cli_history=True)
