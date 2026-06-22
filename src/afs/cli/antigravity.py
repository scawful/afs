"""Antigravity CLI integration helpers.

Antigravity CLI (``agy``) is the successor path for Gemini CLI style agentic
terminal workflows. AFS keeps this surface provider-neutral: it checks local
configuration, can write an AFS MCP entry, and leaves installation/auth to the
user unless explicitly performed outside AFS.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..antigravity_status import DEFAULT_ANTIGRAVITY_DB, antigravity_status
from ..health.mcp_registration import find_afs_mcp_registrations
from ..mcp_runtime import build_afs_mcp_entry

GEMINI_CLI_INDIVIDUAL_CUTOFF = "2026-06-18"
ANTIGRAVITY_INSTALL_COMMAND = "curl -fsSL https://antigravity.google/cli/install.sh | bash"


def _settings_candidates(project_path: Path | None = None) -> list[Path]:
    candidates = [Path.home() / ".gemini" / "antigravity-cli" / "settings.json"]
    if project_path is not None:
        candidates.append(project_path / ".antigravity" / "settings.json")
        candidates.append(project_path / ".gemini" / "antigravity-cli" / "settings.json")
    return candidates


def _default_settings_path(project_path: Path | None = None, *, scope: str = "user") -> Path:
    if scope == "project" and project_path is not None:
        return project_path.expanduser().resolve() / ".antigravity" / "settings.json"
    return (Path.home() / ".gemini" / "antigravity-cli" / "settings.json").expanduser().resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_version(binary: str) -> str:
    try:
        completed = subprocess.run(
            [binary, "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""


def _setup_payload(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    project_path = None
    if args.scope == "project":
        project_path = Path(args.project_path).expanduser().resolve() if args.project_path else Path.cwd().resolve()
    settings_path = Path(args.settings_path).expanduser().resolve() if args.settings_path else _default_settings_path(project_path, scope=args.scope)
    payload = _read_json(settings_path) if settings_path.exists() else {}
    servers = payload.setdefault("mcpServers", {})
    mcp_entry = build_afs_mcp_entry(
        "python-module" if args.python_module else "wrapper",
        cwd=project_path,
        prefer_repo_config=True,
    )
    planned = dict(payload)
    planned_servers = dict(servers)
    planned_servers["afs"] = mcp_entry
    planned["mcpServers"] = planned_servers
    return settings_path, payload, planned


def antigravity_setup_command(args: argparse.Namespace) -> int:
    settings_path, current, planned = _setup_payload(args)
    exists = "afs" in (current.get("mcpServers") if isinstance(current.get("mcpServers"), dict) else {})
    if args.json:
        print(json.dumps({
            "settings_path": str(settings_path),
            "would_update": bool(args.force or not exists),
            "apply": bool(args.apply),
            "install_command": ANTIGRAVITY_INSTALL_COMMAND,
            "mcp_entry": planned.get("mcpServers", {}).get("afs"),
        }, indent=2))
        return 0
    print("Antigravity CLI setup")
    print(f"  settings: {settings_path}")
    print(f"  binary:   {shutil.which(args.binary) or 'not found'}")
    print(f"  install:  {ANTIGRAVITY_INSTALL_COMMAND}")
    if exists and not args.force:
        print("AFS MCP entry already exists; use --force to overwrite.")
        return 0
    if not args.apply:
        print("Dry run only. Re-run with --apply to write the AFS MCP entry.")
        return 0
    _write_json(settings_path, planned)
    print(f"Wrote AFS MCP entry to {settings_path}")
    return 0


def antigravity_status_command(args: argparse.Namespace) -> int:
    project_path = Path(args.path).expanduser().resolve() if args.path else Path.cwd().resolve()
    binary = shutil.which(args.binary)
    settings_paths = _settings_candidates(project_path)
    existing_settings = [path for path in settings_paths if path.exists()]
    settings_payload = _read_json(existing_settings[0]) if existing_settings else {}
    registrations = find_afs_mcp_registrations(cwd=project_path)
    capture = antigravity_status(
        db_path=Path(args.db_path).expanduser().resolve() if args.db_path else DEFAULT_ANTIGRAVITY_DB,
    )
    payload: dict[str, Any] = {
        "binary": {"name": args.binary, "path": binary or "", "available": bool(binary)},
        "version": _run_version(binary) if binary and not args.skip_version else "",
        "settings": {
            "candidates": [str(path) for path in settings_paths],
            "existing": [str(path) for path in existing_settings],
            "has_mcp_servers": isinstance(settings_payload.get("mcpServers"), dict),
            "has_afs_mcp": "afs" in settings_payload.get("mcpServers", {}) if isinstance(settings_payload.get("mcpServers"), dict) else False,
            "trusted_workspaces": settings_payload.get("trustedWorkspaces", []),
        },
        "mcp_registered": registrations.get("antigravity", []),
        "capture": capture,
        "gemini_cli_cutoff": GEMINI_CLI_INDIVIDUAL_CUTOFF,
        "install_command": ANTIGRAVITY_INSTALL_COMMAND,
    }
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print("Antigravity CLI status")
    print(f"  agy:      {binary or 'not found'}")
    if payload["version"]:
        print(f"  version:  {payload['version']}")
    print(f"  settings: {', '.join(payload['settings']['existing']) or 'not found'}")
    print(f"  AFS MCP:  {'yes' if payload['settings']['has_afs_mcp'] or payload['mcp_registered'] else 'not registered'}")
    print(f"  capture:  {capture['payload_count']} payload(s), db_exists={capture['db_exists']}")
    if not binary:
        print(f"  install:  {ANTIGRAVITY_INSTALL_COMMAND}")
    return 0


def antigravity_models_command(args: argparse.Namespace) -> int:
    binary = shutil.which(args.binary)
    if not binary:
        print(f"{args.binary}: not found")
        print(f"install: {ANTIGRAVITY_INSTALL_COMMAND}")
        return 127
    completed = subprocess.run([binary, "models"], check=False)
    return int(completed.returncode)


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("antigravity", help="Manage Antigravity CLI integration.")
    sub = parser.add_subparsers(dest="antigravity_command")

    setup = sub.add_parser("setup", help="Write or preview an AFS MCP entry for Antigravity CLI.")
    setup.add_argument("--settings-path", help="Antigravity settings.json override.")
    setup.add_argument("--scope", choices=["user", "project"], default="user", help="Where to write settings.")
    setup.add_argument("--project-path", help="Project path for project-scoped setup.")
    setup.add_argument("--binary", default="agy", help="Antigravity CLI binary name/path.")
    setup.add_argument("--python-module", action="store_true", help="Use python -m afs.mcp_server instead of wrapper script.")
    setup.add_argument("--force", action="store_true", help="Overwrite existing AFS MCP entry.")
    setup.add_argument("--apply", action="store_true", help="Write the settings file.")
    setup.add_argument("--json", action="store_true", help="Print JSON plan.")
    setup.set_defaults(func=antigravity_setup_command)

    status = sub.add_parser("status", help="Show Antigravity CLI, settings, and capture DB status.")
    status.add_argument("--path", help="Workspace/project path for config detection.")
    status.add_argument("--binary", default="agy", help="Antigravity CLI binary name/path.")
    status.add_argument("--db-path", help="Antigravity state.vscdb override.")
    status.add_argument("--skip-version", action="store_true", help="Do not call agy --version.")
    status.add_argument("--json", action="store_true", help="Print JSON.")
    status.set_defaults(func=antigravity_status_command)

    models = sub.add_parser("models", help="Run `agy models` when Antigravity CLI is installed.")
    models.add_argument("--binary", default="agy", help="Antigravity CLI binary name/path.")
    models.set_defaults(func=antigravity_models_command)
