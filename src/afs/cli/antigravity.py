"""Antigravity CLI integration helpers.

Antigravity CLI (``agy``) is the successor path for Gemini CLI style agentic
terminal workflows. AFS keeps this surface provider-neutral: it checks local
configuration, can write an AFS MCP entry, and leaves installation/auth to the
user unless explicitly performed outside AFS.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..antigravity_status import DEFAULT_ANTIGRAVITY_DB, antigravity_status
from ..health.mcp_registration import find_afs_mcp_registrations
from ..mcp_runtime import build_afs_mcp_entry

GEMINI_CLI_INDIVIDUAL_CUTOFF = "2026-06-18"
ANTIGRAVITY_INSTALL_COMMAND = "curl -fsSL https://antigravity.google/cli/install.sh | bash"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        marker = str(path.expanduser())
        if marker in seen:
            continue
        seen.add(marker)
        result.append(path.expanduser())
    return result


def _settings_candidates(project_path: Path | None = None) -> list[Path]:
    candidates = [
        # Antigravity CLI 1.0.10 uses the migrated shared MCP config path.
        Path.home() / ".gemini" / "config" / "mcp_config.json",
        # Shared Gemini/Antigravity settings can still carry mcpServers.
        Path.home() / ".gemini" / "settings.json",
        # CLI-local settings are used for CLI preferences; keep detecting it for
        # older builds and hand-edited installs.
        Path.home() / ".gemini" / "antigravity-cli" / "settings.json",
        # Legacy/IDE paths seen in pre-migration Antigravity installs.
        Path.home() / ".gemini" / "antigravity" / "mcp_config.json",
        Path.home() / ".gemini" / "antigravity-ide" / "mcp_config.json",
        Path.home() / ".config" / "antigravity" / "settings.json",
    ]
    if project_path is not None:
        candidates.extend(
            [
                project_path / ".gemini" / "config" / "mcp_config.json",
                project_path / ".gemini" / "settings.json",
                project_path / ".antigravity" / "mcp_config.json",
                project_path / ".antigravity" / "settings.json",
                project_path / ".gemini" / "antigravity-cli" / "settings.json",
            ]
        )
    return _dedupe_paths(candidates)


def _default_settings_path(project_path: Path | None = None, *, scope: str = "user") -> Path:
    if scope == "project" and project_path is not None:
        return project_path.expanduser().resolve() / ".gemini" / "config" / "mcp_config.json"
    return (Path.home() / ".gemini" / "config" / "mcp_config.json").expanduser().resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _mcp_servers(payload: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("mcpServers", "mcp_servers"):
        servers = payload.get(key)
        if isinstance(servers, dict):
            return servers
    return None


def _payload_has_afs_mcp(payload: dict[str, Any]) -> bool:
    servers = _mcp_servers(payload)
    if not servers:
        return False
    for name, config in servers.items():
        if isinstance(name, str) and name.strip().lower() == "afs":
            return True
        if not isinstance(config, dict):
            continue
        command = str(config.get("command", "") or "")
        args = config.get("args")
        normalized_args = [str(arg) for arg in args] if isinstance(args, list) else []
        joined = " ".join([command, *normalized_args])
        if (
            "afs.mcp_server" in joined
            or re.search(r"(^|\s)-m\s+afs(\s|$)", joined)
            or re.search(r"(^|/)afs(\s|$)", joined)
        ):
            return True
    return False


def _trusted_workspaces(payloads: list[dict[str, Any]]) -> list[Any]:
    workspaces: list[Any] = []
    for payload in payloads:
        value = payload.get("trustedWorkspaces")
        if isinstance(value, list):
            workspaces.extend(value)
    return workspaces


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
        project_path = (
            Path(args.project_path).expanduser().resolve()
            if args.project_path
            else Path.cwd().resolve()
        )
    settings_path = (
        Path(args.settings_path).expanduser().resolve()
        if args.settings_path
        else _default_settings_path(project_path, scope=args.scope)
    )
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
    exists = _payload_has_afs_mcp(current)
    if args.json:
        print(json.dumps({
            "settings_path": str(settings_path),
            "config_kind": "mcp_config" if settings_path.name == "mcp_config.json" else "settings",
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
    settings_payloads = [_read_json(path) for path in existing_settings]
    paths_with_mcp_servers = [
        path
        for path, payload in zip(existing_settings, settings_payloads, strict=False)
        if _mcp_servers(payload)
    ]
    paths_with_afs_mcp = [
        path
        for path, payload in zip(existing_settings, settings_payloads, strict=False)
        if _payload_has_afs_mcp(payload)
    ]
    registrations = find_afs_mcp_registrations(cwd=project_path)
    capture = antigravity_status(
        db_path=(
            Path(args.db_path).expanduser().resolve()
            if args.db_path
            else DEFAULT_ANTIGRAVITY_DB
        ),
    )
    payload: dict[str, Any] = {
        "binary": {"name": args.binary, "path": binary or "", "available": bool(binary)},
        "version": _run_version(binary) if binary and not args.skip_version else "",
        "settings": {
            "candidates": [str(path) for path in settings_paths],
            "existing": [str(path) for path in existing_settings],
            "mcp_config_paths": [str(path) for path in paths_with_mcp_servers],
            "afs_mcp_paths": [str(path) for path in paths_with_afs_mcp],
            "has_mcp_servers": bool(paths_with_mcp_servers),
            "has_afs_mcp": bool(paths_with_afs_mcp),
            "trusted_workspaces": _trusted_workspaces(settings_payloads),
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
    print(f"  config:   {', '.join(payload['settings']['existing']) or 'not found'}")
    afs_mcp_status = (
        "yes"
        if payload["settings"]["has_afs_mcp"] or payload["mcp_registered"]
        else "not registered"
    )
    print(f"  AFS MCP:  {afs_mcp_status}")
    print(f"  capture:  {capture['payload_count']} payload(s), db_exists={capture['db_exists']}")
    if not binary:
        print(f"  install:  {ANTIGRAVITY_INSTALL_COMMAND}")
    return 0


def antigravity_models_command(args: argparse.Namespace) -> int:
    binary = shutil.which(args.binary)
    if not binary:
        if args.json:
            print(json.dumps({
                "binary": {"name": args.binary, "path": "", "available": False},
                "models": [],
                "install_command": ANTIGRAVITY_INSTALL_COMMAND,
            }, indent=2))
        else:
            print(f"{args.binary}: not found")
            print(f"install: {ANTIGRAVITY_INSTALL_COMMAND}")
        return 127
    try:
        completed = subprocess.run(
            [binary, "models"],
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired:
        if args.json:
            print(json.dumps({
                "binary": {"name": args.binary, "path": binary, "available": True},
                "models": [],
                "error": f"agy models timed out after {args.timeout}s",
            }, indent=2))
        else:
            print(f"agy models timed out after {args.timeout}s", file=sys.stderr)
        return 124
    if args.json:
        print(json.dumps({
            "binary": {"name": args.binary, "path": binary, "available": True},
            "returncode": completed.returncode,
            "models": _parse_models_output(completed.stdout),
            "stderr": completed.stderr.strip(),
        }, indent=2))
    else:
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(
                completed.stderr,
                end="" if completed.stderr.endswith("\n") else "\n",
                file=sys.stderr,
            )
    return int(completed.returncode)


def _parse_models_output(output: str) -> list[dict[str, str]]:
    models: list[dict[str, str]] = []
    for line in output.splitlines():
        raw = line.strip()
        if not raw:
            continue
        match = re.match(r"^(?P<name>.+?)(?:\s+\((?P<label>[^)]+)\))?$", raw)
        if not match:
            models.append({"raw": raw, "name": raw, "label": ""})
            continue
        models.append({
            "raw": raw,
            "name": match.group("name").strip(),
            "label": (match.group("label") or "").strip(),
        })
    return models


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("antigravity", help="Manage Antigravity CLI integration.")
    sub = parser.add_subparsers(dest="antigravity_command")

    setup = sub.add_parser("setup", help="Write or preview an AFS MCP entry for Antigravity CLI.")
    setup.add_argument(
        "--settings-path",
        "--config-path",
        dest="settings_path",
        help="Antigravity settings/mcp_config JSON override.",
    )
    setup.add_argument(
        "--scope",
        choices=["user", "project"],
        default="user",
        help="Where to write settings.",
    )
    setup.add_argument("--project-path", help="Project path for project-scoped setup.")
    setup.add_argument("--binary", default="agy", help="Antigravity CLI binary name/path.")
    setup.add_argument(
        "--python-module",
        action="store_true",
        help="Use python -m afs.mcp_server instead of wrapper script.",
    )
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
    models.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout for `agy models`, in seconds.",
    )
    models.add_argument("--json", action="store_true", help="Print parsed JSON model data.")
    models.set_defaults(func=antigravity_models_command)
