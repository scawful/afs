"""Small Python GUI manager for approachable AFS setup and agent state."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import load_runtime_config_model
from .extensions import resolve_extensions_config
from .manager import AFSManager
from .models import MountType
from .plugins import load_enabled_extensions
from .session_bootstrap import build_agent_discovery_path
from .tasks import Task, TaskQueue


@dataclass(frozen=True)
class ClientConfigState:
    name: str
    path: Path
    exists: bool
    registered: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "exists": self.exists,
            "registered": self.registered,
            "note": self.note,
        }


@dataclass(frozen=True)
class ExtensionState:
    name: str
    root: Path
    description: str = ""
    manager_actions: list[str] = field(default_factory=list)
    hooks: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root),
            "description": self.description,
            "manager_actions": list(self.manager_actions),
            "hooks": {key: list(value) for key, value in self.hooks.items()},
        }


@dataclass(frozen=True)
class ManagerSnapshot:
    workspace: Path
    context_path: Path
    context_exists: bool
    context_healthy: bool | None
    mount_counts: dict[str, int]
    clients: list[ClientConfigState]
    tasks: list[Task]
    extensions: list[ExtensionState]
    commands: dict[str, str]
    discovery_path: dict[str, Any]
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace),
            "context_path": str(self.context_path),
            "context_exists": self.context_exists,
            "context_healthy": self.context_healthy,
            "mount_counts": dict(self.mount_counts),
            "clients": [client.to_dict() for client in self.clients],
            "tasks": [task.to_dict() for task in self.tasks],
            "extensions": [extension.to_dict() for extension in self.extensions],
            "commands": dict(self.commands),
            "discovery_path": dict(self.discovery_path),
            "errors": list(self.errors),
        }


_CLIENTS = (
    ("Gemini project", ".gemini/settings.json"),
    ("Claude project", ".claude/settings.json"),
    ("Codex project", ".codex/config.toml"),
    ("OpenCode project", ".opencode/opencode.jsonc"),
    ("MCP project", ".mcp.json"),
)


def _contains_afs(value: Any) -> bool:
    if isinstance(value, dict):
        return any("afs" in str(key).lower() or _contains_afs(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_afs(item) for item in value)
    if isinstance(value, str):
        return "afs" in value.lower()
    return False


def _afs_command() -> str:
    """Return a shell-safe AFS command for manager buttons and hints."""
    repo_script = Path(__file__).resolve().parents[2] / "scripts" / "afs"
    if repo_script.exists():
        return shlex.quote(str(repo_script))
    discovered = shutil.which("afs")
    return shlex.quote(discovered) if discovered else "afs"


def _strip_jsonc_comments(text: str) -> str:
    """Strip JSONC comments without treating URL-like text inside strings as comments."""
    output: list[str] = []
    in_string = False
    quote = ""
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            index += 1
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            output.append(char)
            index += 1
            continue
        if char == "/" and next_char == "/":
            index = text.find("\n", index)
            if index == -1:
                break
            output.append("\n")
            index += 1
            continue
        if char == "/" and next_char == "*":
            end = text.find("*/", index + 2)
            if end == -1:
                break
            output.append("\n" * text[index:end].count("\n"))
            index = end + 2
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _loads_jsonish(text: str) -> Any:
    """Load JSON or simple JSONC used by editor/client settings."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        stripped = _strip_jsonc_comments(text)
        stripped = re.sub(r",\s*([}\]])", r"\1", stripped)
        return json.loads(stripped)


def _config_state(name: str, path: Path) -> ClientConfigState:
    if not path.exists():
        return ClientConfigState(name=name, path=path, exists=False, registered=False, note="missing")
    if path.is_dir():
        return ClientConfigState(name=name, path=path, exists=True, registered=False, note="directory")
    if path.suffix in {".json", ".jsonc"} or path.name == "settings.json":
        try:
            payload = _loads_jsonish(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return ClientConfigState(name=name, path=path, exists=True, registered=False, note=f"unreadable: {exc}")
        registered = _contains_afs(payload)
        return ClientConfigState(name=name, path=path, exists=True, registered=registered, note="afs entry found" if registered else "no afs entry")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return ClientConfigState(name=name, path=path, exists=True, registered=False, note=f"unreadable: {exc}")
    registered = "afs" in text.lower()
    return ClientConfigState(name=name, path=path, exists=True, registered=registered, note="afs text found" if registered else "no afs entry")


def collect_manager_snapshot(workspace: Path | str = Path("."), *, home: Path | None = None) -> ManagerSnapshot:
    """Collect the read model used by the GUI and by tests."""
    root = Path(workspace).expanduser().resolve()
    home = (home or Path.home()).expanduser().resolve()
    config, _config_path = load_runtime_config_model(merge_user=True, start_dir=root)
    manager = AFSManager(config=config)
    context_path = root / ".context"
    if not context_path.exists():
        context_path = config.general.context_root

    errors: list[str] = []
    mount_counts: dict[str, int] = {}
    healthy: bool | None = None
    tasks: list[Task] = []

    if context_path.exists():
        try:
            context = manager.list_context(context_path=context_path)
            mount_counts = {
                mount.value: len(context.get_mounts(mount))
                for mount in MountType
            }
            health = manager.context_health(context_path)
            healthy = bool(health.get("healthy"))
        except Exception as exc:  # pragma: no cover - defensive GUI read path
            errors.append(f"context: {exc}")
        try:
            tasks = TaskQueue(context_path).list()
        except Exception as exc:  # pragma: no cover - defensive GUI read path
            errors.append(f"tasks: {exc}")

    clients = [_config_state(name, root / rel) for name, rel in _CLIENTS]
    clients.extend(
        [
            _config_state("Gemini user", home / ".gemini" / "settings.json"),
            _config_state("Claude user", home / ".claude" / "settings.json"),
        ]
    )

    extension_config = resolve_extensions_config(config)
    loaded = load_enabled_extensions(config=config)
    extensions = [
        ExtensionState(
            name=manifest.name,
            root=manifest.root,
            description=manifest.description,
            manager_actions=list(manifest.manager_actions),
            hooks=dict(manifest.hooks),
        )
        for manifest in loaded.values()
    ]

    afs = _afs_command()
    quoted = shlex.quote(str(root))
    commands = {
        "open_manager": f"{afs} manager open --path {quoted}",
        "setup_preview": f"{afs} setup --workspace {quoted} --dry-run",
        "setup_apply": f"{afs} setup --workspace {quoted} --apply",
        "gemini_setup": f"{afs} gemini setup --scope project --project-path {quoted}",
        "claude_setup": f"{afs} claude setup --scope project --path {quoted}",
        "tasks": f"{afs} tasks list --path {quoted}",
        "extensions": "afs plugins --details",
    }
    if extension_config.enabled_extensions:
        commands["enabled_extensions"] = ", ".join(extension_config.enabled_extensions)

    return ManagerSnapshot(
        workspace=root,
        context_path=context_path,
        context_exists=context_path.exists(),
        context_healthy=healthy,
        mount_counts=mount_counts,
        clients=clients,
        tasks=tasks,
        extensions=extensions,
        commands=commands,
        discovery_path=build_agent_discovery_path(context_path),
        errors=errors,
    )


def _run_shell(command: str) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        shell=True,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.returncode, completed.stdout.strip()


def launch_manager(workspace: Path | str = Path(".")) -> int:
    """Launch the Tkinter AFS manager."""
    try:
        import tkinter as tk
        from tkinter import messagebox, simpledialog, ttk
    except Exception as exc:  # pragma: no cover - platform dependent
        print(f"AFS Manager GUI is unavailable: {exc}", file=sys.stderr)
        print("Try: afs manager snapshot --json --path .", file=sys.stderr)
        return 1

    root = Path(workspace).expanduser().resolve()
    window = tk.Tk()
    window.title("AFS Manager")
    window.geometry("920x640")

    status = tk.StringVar(value="Loading AFS state...")
    tree = ttk.Treeview(window, columns=("value",), show="tree headings")
    tree.heading("#0", text="Item")
    tree.heading("value", text="Value")
    tree.column("#0", width=360)
    tree.column("value", width=520)
    tree.pack(fill="both", expand=True, padx=10, pady=10)

    bar = ttk.Frame(window)
    bar.pack(fill="x", padx=10, pady=(0, 10))
    ttk.Label(bar, textvariable=status).pack(side="left", fill="x", expand=True)

    snapshot: ManagerSnapshot | None = None

    def fill() -> None:
        nonlocal snapshot
        snapshot = collect_manager_snapshot(root)
        tree.delete(*tree.get_children())
        workspace_id = tree.insert("", "end", text="Workspace", values=(str(snapshot.workspace),), open=True)
        tree.insert(workspace_id, "end", text="Context", values=(str(snapshot.context_path),))
        tree.insert(workspace_id, "end", text="Healthy", values=(str(snapshot.context_healthy),))
        mounts_id = tree.insert("", "end", text="Mounts", values=("counts",), open=True)
        for name, count in sorted(snapshot.mount_counts.items()):
            tree.insert(mounts_id, "end", text=name, values=(str(count),))
        clients_id = tree.insert("", "end", text="Agent client configs", values=(".gemini/.claude/.codex/.opencode",), open=True)
        for client in snapshot.clients:
            state = "registered" if client.registered else client.note
            tree.insert(clients_id, "end", text=client.name, values=(f"{state} — {client.path}",))
        tasks_id = tree.insert("", "end", text="Tasks", values=(f"{len(snapshot.tasks)} task(s)",), open=True)
        for task in snapshot.tasks[:30]:
            tree.insert(tasks_id, "end", text=f"{task.id} [{task.status}]", values=(task.title,))
        extensions_id = tree.insert("", "end", text="Extensions", values=(f"{len(snapshot.extensions)} loaded",), open=True)
        for ext in snapshot.extensions:
            ext_id = tree.insert(extensions_id, "end", text=ext.name, values=(str(ext.root),))
            if ext.description:
                tree.insert(ext_id, "end", text="description", values=(ext.description,))
            for action in ext.manager_actions:
                tree.insert(ext_id, "end", text="manager action", values=(action,))
            for event, commands in sorted(ext.hooks.items()):
                if commands:
                    tree.insert(ext_id, "end", text=f"hook:{event}", values=(", ".join(commands),))
        commands_id = tree.insert("", "end", text="Suggested commands", values=("copy/run from terminal",), open=True)
        for name, command in snapshot.commands.items():
            tree.insert(commands_id, "end", text=name, values=(command,))
        path_id = tree.insert("", "end", text="Agent discovery path", values=("deterministic, low-noise",), open=True)
        for step in snapshot.discovery_path.get("steps", []):
            if isinstance(step, dict):
                tree.insert(path_id, "end", text=str(step.get("step", "")), values=(str(step.get("tool", "")),))
        if snapshot.errors:
            errors_id = tree.insert("", "end", text="Errors", values=(f"{len(snapshot.errors)} issue(s)",), open=True)
            for err in snapshot.errors:
                tree.insert(errors_id, "end", text="error", values=(err,))
        status.set(f"AFS Manager ready: {snapshot.workspace}")

    def new_task() -> None:
        if snapshot is None or not snapshot.context_exists:
            messagebox.showerror("AFS Manager", "No .context root found for this workspace.")
            return
        title = simpledialog.askstring("New AFS task", "Task title:")
        if not title:
            return
        queue = TaskQueue(snapshot.context_path)
        queue.create(title, created_by="afs-manager")
        fill()

    def setup_client(command: str) -> None:
        if not messagebox.askyesno("AFS Manager", f"Run this command?\n\n{command}"):
            return
        code, output = _run_shell(command)
        if code == 0:
            messagebox.showinfo("AFS Manager", output or "Command completed.")
            fill()
            return
        messagebox.showerror("AFS Manager", output or f"Command failed: {code}")

    ttk.Button(bar, text="Refresh", command=fill).pack(side="right")
    ttk.Button(bar, text="New task", command=new_task).pack(side="right", padx=(0, 6))
    ttk.Button(
        bar,
        text="Setup Gemini",
        command=lambda: setup_client(f"{_afs_command()} gemini setup --scope project --project-path {shlex.quote(str(root))}"),
    ).pack(side="right", padx=(0, 6))
    ttk.Button(
        bar,
        text="Setup Claude",
        command=lambda: setup_client(f"{_afs_command()} claude setup --scope project --path {shlex.quote(str(root))}"),
    ).pack(side="right", padx=(0, 6))

    fill()
    window.mainloop()
    return 0
