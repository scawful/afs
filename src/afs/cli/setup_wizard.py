"""Guided setup wizard for approachable AFS onboarding."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SetupStep:
    title: str
    command: list[str]
    note: str = ""
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "command": list(self.command),
            "shell": shlex.join(self.command),
            "note": self.note,
            "optional": self.optional,
        }


@dataclass(frozen=True)
class SetupPlan:
    workspace: Path
    context_root: Path
    config_path: Path
    config_scope: str
    context_mode: str
    steps: list[SetupStep]
    next_commands: list[list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace),
            "context_root": str(self.context_root),
            "config_path": str(self.config_path),
            "config_scope": self.config_scope,
            "context_mode": self.context_mode,
            "steps": [step.to_dict() for step in self.steps],
            "next_commands": [
                {"command": list(command), "shell": shlex.join(command)}
                for command in self.next_commands
            ],
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _afs_command() -> list[str]:
    env_cli = os.getenv("AFS_CLI", "").strip()
    if env_cli:
        return [env_cli]
    repo_script = _repo_root() / "scripts" / "afs"
    if repo_script.exists():
        return [str(repo_script)]
    return [sys.executable, "-m", "afs"]


def _default_config_path(workspace: Path, scope: str) -> Path:
    if scope == "user":
        return Path.home() / ".config" / "afs" / "config.toml"
    return workspace / "afs.toml"


def _default_context_root(workspace: Path, mode: str) -> Path:
    if mode == "project":
        return workspace / ".context"
    return Path.home() / ".context"


def _bool_choice(value: bool) -> str:
    return "yes" if value else "no"


def _ask_text(prompt: str, default: str) -> str:
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
    options = "/".join(choice.upper() if choice == default else choice for choice in choices)
    while True:
        answer = input(f"{prompt} ({options}): ").strip().lower()
        if not answer:
            return default
        matches = [choice for choice in choices if choice.startswith(answer)]
        if len(matches) == 1:
            return matches[0]
        print("Please choose one of: " + ", ".join(choices))


def _ask_bool(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def _interactive(args: argparse.Namespace) -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty() and not args.yes and not args.json)


def _collect_answers(args: argparse.Namespace) -> dict[str, Any]:
    interactive = _interactive(args)
    workspace_default = str(Path(args.workspace or Path.cwd()).expanduser().resolve())
    workspace = Path(
        _ask_text("Workspace or project path", workspace_default)
        if interactive and not args.workspace
        else workspace_default
    ).expanduser().resolve()

    config_scope = args.config_scope or "project"
    if interactive and not args.config_scope:
        config_scope = _ask_choice(
            "Where should AFS write configuration",
            ["project", "user"],
            "project",
        )

    context_mode = args.context_mode or ("project" if config_scope == "project" else "shared")
    if interactive and not args.context_mode:
        context_mode = _ask_choice(
            "Where should context files live",
            ["project", "shared"],
            context_mode,
        )

    context_default = str(_default_context_root(workspace, context_mode))
    context_root = Path(
        _ask_text("Context root", context_default)
        if interactive and not args.context_root
        else (args.context_root or context_default)
    ).expanduser().resolve()

    config_default = str(_default_config_path(workspace, config_scope))
    config_path = Path(
        _ask_text("Config path", config_default)
        if interactive and not args.config
        else (args.config or config_default)
    ).expanduser().resolve()

    link_default = bool(context_mode == "shared" and config_scope == "project")
    if args.link_context is not None:
        link_context = bool(args.link_context)
    elif interactive:
        link_context = _ask_bool("Create a .context link in the workspace", link_default)
    else:
        link_context = link_default

    shell_mode = args.shell
    if interactive and shell_mode == "ask":
        shell_mode = _ask_choice(
            "Install shell helpers",
            ["none", "helpers", "agent-hooks"],
            "helpers",
        )
    elif shell_mode == "ask":
        shell_mode = "helpers"

    mcp_mode = args.mcp
    if interactive and mcp_mode == "ask":
        mcp_mode = _ask_choice(
            "Register AFS with local MCP clients",
            ["none", "claude", "gemini", "antigravity", "both", "all"],
            "none",
        )
    elif mcp_mode == "ask":
        mcp_mode = "none"

    gws_mode = args.google_workspace
    if interactive and gws_mode == "ask":
        gws_mode = _ask_choice(
            "Google Workspace helper",
            ["skip", "check", "setup"],
            "skip",
        )
    elif gws_mode == "ask":
        gws_mode = "skip"

    worker = args.worker
    if worker is None:
        worker = _ask_bool("Install background job worker", False) if interactive else False

    return {
        "workspace": workspace,
        "context_root": context_root,
        "config_path": config_path,
        "config_scope": config_scope,
        "context_mode": context_mode,
        "link_context": link_context,
        "shell_mode": shell_mode,
        "mcp_mode": mcp_mode,
        "gws_mode": gws_mode,
        "worker": worker,
    }


def build_setup_plan(
    *,
    workspace: Path,
    context_root: Path,
    config_path: Path,
    config_scope: str,
    context_mode: str,
    link_context: bool,
    shell_mode: str,
    mcp_mode: str,
    gws_mode: str,
    worker: bool,
    force: bool = False,
    afs_command: list[str] | None = None,
) -> SetupPlan:
    afs = afs_command or _afs_command()
    root = _repo_root()
    workspace_name = workspace.name or "workspace"
    init_cmd = [
        *afs,
        "init",
        "--context-root",
        str(context_root),
        "--config",
        str(config_path),
        "--workspace-path",
        str(workspace),
        "--workspace-name",
        workspace_name,
    ]
    if link_context:
        init_cmd.append("--link-context")
    if force:
        init_cmd.append("--force")

    steps: list[SetupStep] = [
        SetupStep(
            "Write AFS config and create context directories",
            init_cmd,
            note="Existing config files are left unchanged unless --force is used.",
        ),
        SetupStep(
            "Repair context metadata and rebuild the search index",
            [*afs, "context", "repair", "--path", str(workspace), "--rebuild-index"],
        ),
    ]

    if shell_mode in {"helpers", "agent-hooks"}:
        shell_cmd = [
            *afs,
            "agent-hooks",
            "install-shell",
            "--afs-root",
            str(root),
            "--apply",
        ]
        if shell_mode == "helpers":
            shell_cmd.insert(-1, "--helpers-only")
        steps.append(
            SetupStep(
                "Install shell helpers",
                shell_cmd,
                note=(
                    "helpers-only keeps aliases/completion without routing AI harness commands"
                    if shell_mode == "helpers"
                    else "full hooks route supported AI harness commands through AFS wrappers"
                ),
            )
        )

    if worker:
        steps.append(
            SetupStep(
                "Install background job worker",
                [
                    *afs,
                    "agent-hooks",
                    "install-worker",
                    "--afs-root",
                    str(root),
                    "--path",
                    str(workspace),
                    "--apply",
                    "--load",
                ],
                optional=True,
            )
        )

    if mcp_mode in {"claude", "both", "all"}:
        steps.append(
            SetupStep(
                "Register Claude-compatible MCP config",
                [*afs, "claude", "setup", "--scope", config_scope, "--path", str(workspace)],
                optional=True,
            )
        )
    if mcp_mode in {"gemini", "both", "all"}:
        gemini_cmd = [*afs, "gemini", "setup", "--scope", config_scope]
        if config_scope == "project":
            gemini_cmd.extend(["--project-path", str(workspace)])
        steps.append(
            SetupStep(
                "Register Gemini MCP config",
                gemini_cmd,
                optional=True,
            )
        )

    if mcp_mode in {"antigravity", "all"}:
        antigravity_cmd = [*afs, "antigravity", "setup", "--scope", config_scope]
        if config_scope == "project":
            antigravity_cmd.extend(["--project-path", str(workspace)])
        steps.append(
            SetupStep(
                "Preview Antigravity CLI MCP config",
                antigravity_cmd,
                note="Add --apply to write the Antigravity settings file after reviewing the plan.",
                optional=True,
            )
        )

    if gws_mode == "check":
        steps.append(SetupStep("Check Google Workspace auth", [*afs, "gws", "status"], optional=True))
    elif gws_mode == "setup":
        gws_script = root / "scripts" / "setup_gws.sh"
        steps.append(
            SetupStep(
                "Run Google Workspace helper setup",
                [str(gws_script)],
                note="Uses user-provided OAuth/client credentials and scopes.",
                optional=True,
            )
        )

    next_commands = [
        [*afs, "next", "--intent", "setup", "--path", str(workspace)],
        [*afs, "manager", "open", "--path", str(workspace)],
        [*afs, "status", "--start-dir", str(workspace)],
        [*afs, "guide", "context"],
        [*afs, "query", "<text>", "--path", str(workspace)],
        [*afs, "doctor"],
    ]
    return SetupPlan(
        workspace=workspace,
        context_root=context_root,
        config_path=config_path,
        config_scope=config_scope,
        context_mode=context_mode,
        steps=steps,
        next_commands=next_commands,
    )


def _print_plan(plan: SetupPlan) -> None:
    print("AFS setup plan")
    print(f"  workspace:    {plan.workspace}")
    print(f"  context_root: {plan.context_root}")
    print(f"  config_path:  {plan.config_path}")
    print(f"  config_scope: {plan.config_scope}")
    print(f"  context_mode: {plan.context_mode}")
    print()
    print("Steps")
    for index, step in enumerate(plan.steps, start=1):
        marker = "optional" if step.optional else "write"
        print(f"{index}. {step.title} ({marker})")
        print(f"   {shlex.join(step.command)}")
        if step.note:
            print(f"   note: {step.note}")
    print()
    print("After setup")
    for command in plan.next_commands:
        print(f"  {shlex.join(command)}")


def _run_step(step: SetupStep) -> int:
    print(f"+ {shlex.join(step.command)}")
    completed = subprocess.run(step.command, check=False)
    return int(completed.returncode)


def setup_command(args: argparse.Namespace) -> int:
    answers = _collect_answers(args)
    plan = build_setup_plan(
        workspace=answers["workspace"],
        context_root=answers["context_root"],
        config_path=answers["config_path"],
        config_scope=answers["config_scope"],
        context_mode=answers["context_mode"],
        link_context=answers["link_context"],
        shell_mode=answers["shell_mode"],
        mcp_mode=answers["mcp_mode"],
        gws_mode=answers["gws_mode"],
        worker=answers["worker"],
        force=bool(args.force),
    )

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
        return 0

    _print_plan(plan)

    apply = bool(args.apply and not args.dry_run)
    if not apply and not args.dry_run and _interactive(args):
        print()
        apply = _ask_bool("Apply these steps now", False)

    if not apply:
        print()
        print("Dry run only. Re-run with --apply to execute the plan.")
        return 0

    print()
    print("Applying setup")
    for step in plan.steps:
        rc = _run_step(step)
        if rc != 0:
            print(f"step failed: {step.title}", file=sys.stderr)
            return rc
    print()
    print("Setup complete. Useful next commands:")
    for command in plan.next_commands:
        print(f"  {shlex.join(command)}")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("setup", help="Run the guided AFS setup wizard.")
    parser.add_argument("--workspace", help="Workspace/project path (default: current directory).")
    parser.add_argument("--context-root", help="Context root path.")
    parser.add_argument("--config", help="Config path to write.")
    parser.add_argument("--config-scope", choices=["project", "user"], help="Config scope.")
    parser.add_argument("--context-mode", choices=["project", "shared"], help="Where context files live.")
    parser.add_argument("--link-context", dest="link_context", action="store_true", help="Create a .context link in the workspace.")
    parser.add_argument("--no-link-context", dest="link_context", action="store_false", help="Do not create a .context link.")
    parser.set_defaults(link_context=None)
    parser.add_argument(
        "--shell",
        choices=["ask", "none", "helpers", "agent-hooks"],
        default="ask",
        help="Shell integration level.",
    )
    parser.add_argument(
        "--mcp",
        choices=["ask", "none", "claude", "gemini", "antigravity", "both", "all"],
        default="ask",
        help="MCP client registration.",
    )
    parser.add_argument(
        "--google-workspace",
        choices=["ask", "skip", "check", "setup"],
        default="ask",
        help="Optional Google Workspace helper handling.",
    )
    parser.add_argument("--worker", dest="worker", action="store_true", help="Install background job worker.")
    parser.add_argument("--no-worker", dest="worker", action="store_false", help="Skip background job worker.")
    parser.set_defaults(worker=None)
    parser.add_argument("--apply", action="store_true", help="Execute the generated setup plan.")
    parser.add_argument("--dry-run", action="store_true", help="Print the setup plan without executing it.")
    parser.add_argument("--yes", "-y", action="store_true", help="Accept defaults and do not prompt.")
    parser.add_argument("--force", action="store_true", help="Overwrite config when running init.")
    parser.add_argument("--json", action="store_true", help="Print the setup plan as JSON.")
    parser.set_defaults(func=setup_command)
