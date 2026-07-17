"""CLI for provider-neutral context source adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..manager import AFSManager
from ..sources import (
    ContextSourceProvider,
    ResearchSourceProvider,
    assert_source_materialization_supported,
    discover_source_provider_specs,
    load_source_provider_by_name,
    materialize_source_records,
)
from ._utils import load_runtime_config_from_args, resolve_context_paths


def _load_manager_and_context(args: argparse.Namespace) -> tuple[Any, AFSManager, Path]:
    config, _config_path = load_runtime_config_from_args(args, start_dir=Path.cwd())
    manager = AFSManager(config=config)
    _project_path, context_path, _root, _dir = resolve_context_paths(args, manager)
    return config, manager, context_path


def _sources_list(args: argparse.Namespace) -> int:
    config, _resolved = load_runtime_config_from_args(args, start_dir=Path.cwd())
    specs = discover_source_provider_specs(config=config)
    payload = {
        "kinds": ["task", "ticket", "review", "doc", "message", "test", "hook", "trace"],
        "providers": [spec.to_dict() for spec in specs],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    if not specs:
        print("No extension-owned context source providers are enabled.")
        print("Providers can be declared with [[context_sources]] in extension.toml.")
        return 0
    for spec in specs:
        kinds = ", ".join(spec.kinds) if spec.kinds else "unspecified"
        print(f"{spec.name}: {spec.description or spec.module} [{kinds}]")
    return 0


def _sources_status(args: argparse.Namespace) -> int:
    config, _manager, _context_path = _load_manager_and_context(args)
    payload: dict[str, Any] = {"providers": []}
    for spec in discover_source_provider_specs(config=config):
        name = spec.name
        try:
            provider = load_source_provider_by_name(name, config=config)
        except Exception as exc:
            status = {"ok": False, "error": str(exc)}
            capabilities = {"sync": False, "research": False}
            kinds = list(spec.kinds)
        else:
            supports_sync = isinstance(provider, ContextSourceProvider)
            supports_research = isinstance(provider, ResearchSourceProvider)
            capabilities = {
                "sync": supports_sync,
                "research": supports_research,
            }
            kinds = list(getattr(provider, "kinds", spec.kinds) or [])
            if isinstance(provider, ContextSourceProvider):
                try:
                    status = provider.status()
                except Exception as exc:
                    status = {"ok": False, "error": str(exc)}
            elif supports_research:
                status = {"ok": True, "detail": "research-only provider"}
            else:
                status = {
                    "ok": False,
                    "error": "provider implements neither sync nor research",
                }
        payload["providers"].append(
            {
                "name": name,
                "kinds": kinds,
                "capabilities": capabilities,
                "status": status,
            }
        )
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if not payload["providers"]:
        print("No enabled context source providers loaded.")
        return 0
    for entry in payload["providers"]:
        status = entry["status"]
        ok = status.get("ok") if isinstance(status, dict) else None
        marker = "ok" if ok is not False else "error"
        capability_label = ", ".join(
            name for name, enabled in entry["capabilities"].items() if enabled
        ) or "no supported capability"
        print(f"{entry['name']}: {marker} [{capability_label}]")
    return 0


def _sources_sync(args: argparse.Namespace) -> int:
    config, _manager, context_path = _load_manager_and_context(args)
    try:
        assert_source_materialization_supported(context_path)
    except ValueError as exc:
        print(str(exc))
        return 2
    try:
        provider = load_source_provider_by_name(args.provider, config=config)
    except KeyError:
        print(f"unknown context source provider: {args.provider}")
        return 1
    except Exception as exc:
        print(f"failed to load context source provider {args.provider}: {exc}")
        return 2
    if not isinstance(provider, ContextSourceProvider):
        print(
            f"context source provider {args.provider} does not support sync; "
            "use it through the bounded research workflow instead"
        )
        return 2
    try:
        records = provider.sync(query=args.query or "", limit=max(1, int(args.limit)))
    except TypeError:
        records = provider.sync()  # type: ignore[call-arg]
    result = materialize_source_records(
        context_path=context_path,
        provider_name=args.provider,
        records=records,
        dry_run=not bool(args.apply),
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0
    action = "would write" if result.dry_run else "wrote"
    print(f"{action} {len(result.records)} record(s) under {result.target_dir}")
    if result.dry_run:
        print(
            "Dry run only. Re-run with --apply to write v1 "
            ".context/items/sources files."
        )
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("sources", help="Manage generic context source providers.")
    parser.add_argument("--config", help="Config path.")
    sub = parser.add_subparsers(dest="sources_command")

    list_parser = sub.add_parser("list", help="List extension-declared source providers.")
    list_parser.add_argument("--json", action="store_true", help="Print JSON.")
    list_parser.set_defaults(func=_sources_list)

    status_parser = sub.add_parser("status", help="Show loaded source provider status.")
    status_parser.add_argument("--path", help="Workspace/project path.")
    status_parser.add_argument("--context-root", help="Context root override.")
    status_parser.add_argument("--context-dir", help="Context dir name override.")
    status_parser.add_argument("--json", action="store_true", help="Print JSON.")
    status_parser.set_defaults(func=_sources_status)

    sync_parser = sub.add_parser(
        "sync",
        help="Sync provider records into v1 .context/items (v2 scoped ingestion pending).",
    )
    sync_parser.add_argument("--provider", required=True, help="Provider name.")
    sync_parser.add_argument("--query", default="", help="Provider query/filter.")
    sync_parser.add_argument("--limit", type=int, default=50, help="Maximum records to request.")
    sync_parser.add_argument("--path", help="Workspace/project path.")
    sync_parser.add_argument("--context-root", help="Context root override.")
    sync_parser.add_argument("--context-dir", help="Context dir name override.")
    sync_parser.add_argument(
        "--apply",
        action="store_true",
        help="Write records to v1 .context/items (unavailable for v2).",
    )
    sync_parser.add_argument("--json", action="store_true", help="Print JSON.")
    sync_parser.set_defaults(func=_sources_sync)
