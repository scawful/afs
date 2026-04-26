"""Installable hooks for routing local AI harnesses through AFS."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

SHELL_BLOCK_BEGIN = "# >>> afs agent hooks >>>"
SHELL_BLOCK_END = "# <<< afs agent hooks <<<"
DEFAULT_WORKER_LABEL = "com.afs.agent-jobs"


@dataclass(frozen=True)
class HookInstallResult:
    target: str
    changed: bool
    applied: bool
    loaded: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "changed": self.changed,
            "applied": self.applied,
            "loaded": self.loaded,
            "message": self.message,
        }


def default_shell_profile() -> Path:
    home = Path.home()
    zshrc = home / ".zshrc"
    if zshrc.exists():
        return zshrc
    return zshrc


def default_launchd_plist_path(label: str = DEFAULT_WORKER_LABEL) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def render_shell_profile_block(afs_root: Path) -> str:
    root = str(afs_root.expanduser().resolve())
    shell_init = shlex.quote(f"{root}/scripts/afs-shell-init.sh")
    agent_hooks = shlex.quote(f"{root}/scripts/afs-agent-hooks.sh")
    return "\n".join(
        [
            SHELL_BLOCK_BEGIN,
            f"[ -f {shell_init} ] && source {shell_init}",
            f"[ -f {agent_hooks} ] && source {agent_hooks}",
            SHELL_BLOCK_END,
            "",
        ]
    )


def _replace_or_append_block(text: str, block: str) -> tuple[str, bool]:
    begin = text.find(SHELL_BLOCK_BEGIN)
    end = text.find(SHELL_BLOCK_END)
    if begin != -1 and end != -1 and end >= begin:
        end += len(SHELL_BLOCK_END)
        if end < len(text) and text[end : end + 1] == "\n":
            end += 1
        updated = text[:begin] + block + text[end:]
    else:
        separator = "" if not text or text.endswith("\n") else "\n"
        updated = text + separator + block
    return updated, updated != text


def install_shell_profile_hooks(
    *,
    afs_root: Path,
    profile_path: Path | None = None,
    apply: bool = False,
) -> HookInstallResult:
    target = (profile_path or default_shell_profile()).expanduser()
    current = target.read_text(encoding="utf-8") if target.exists() else ""
    updated, changed = _replace_or_append_block(current, render_shell_profile_block(afs_root))
    if apply and changed:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(updated, encoding="utf-8")
    if apply and changed:
        message = "shell hooks installed"
    elif changed:
        message = "shell hooks would be installed"
    else:
        message = "shell hooks already current"
    return HookInstallResult(
        target=str(target),
        changed=changed,
        applied=apply and changed,
        message=message,
    )


def render_worker_command(
    *,
    afs_root: Path,
    context_path: Path,
    agent_name: str,
    command: str,
    poll_seconds: float,
) -> list[str]:
    root = afs_root.expanduser().resolve()
    return [
        str(root / "scripts" / "afs"),
        "agent-jobs",
        "work",
        "--loop",
        "--agent",
        agent_name,
        "--path",
        str(context_path.expanduser()),
        "--command",
        command,
        "--poll-seconds",
        str(poll_seconds),
    ]


def default_worker_command(afs_root: Path) -> str:
    wrapper = shlex.quote(str(afs_root.expanduser().resolve() / "scripts" / "afs-codex"))
    return f'{wrapper} --prompt-file "$AFS_AGENT_JOB_PROMPT_FILE" exec < "$AFS_AGENT_JOB_PROMPT_FILE"'


def render_launchd_plist(
    *,
    afs_root: Path,
    context_path: Path,
    agent_name: str = "local-worker",
    command: str | None = None,
    poll_seconds: float = 30.0,
    label: str = DEFAULT_WORKER_LABEL,
    log_dir: Path | None = None,
) -> bytes:
    root = afs_root.expanduser().resolve()
    logs = (log_dir or (Path.home() / ".config" / "afs" / "agent-jobs")).expanduser()
    worker_command = command or default_worker_command(root)
    args = render_worker_command(
        afs_root=root,
        context_path=context_path,
        agent_name=agent_name,
        command=worker_command,
        poll_seconds=poll_seconds,
    )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
        '<plist version="1.0">',
        "<dict>",
        "  <key>Label</key>",
        f"  <string>{_xml_escape(label)}</string>",
        "  <key>ProgramArguments</key>",
        "  <array>",
    ]
    lines.extend(f"    <string>{_xml_escape(arg)}</string>" for arg in args)
    lines.extend(
        [
            "  </array>",
            "  <key>EnvironmentVariables</key>",
            "  <dict>",
            "    <key>AFS_ROOT</key>",
            f"    <string>{_xml_escape(str(root))}</string>",
            "    <key>PATH</key>",
            "    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>",
            "  </dict>",
            "  <key>RunAtLoad</key>",
            "  <true/>",
            "  <key>KeepAlive</key>",
            "  <true/>",
            "  <key>ThrottleInterval</key>",
            "  <integer>60</integer>",
            "  <key>StandardOutPath</key>",
            f"  <string>{_xml_escape(str(logs / 'worker.stdout.log'))}</string>",
            "  <key>StandardErrorPath</key>",
            f"  <string>{_xml_escape(str(logs / 'worker.stderr.log'))}</string>",
            "  <key>Nice</key>",
            "  <integer>10</integer>",
            "</dict>",
            "</plist>",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def install_worker_launchd(
    *,
    afs_root: Path,
    context_path: Path,
    agent_name: str = "local-worker",
    command: str | None = None,
    poll_seconds: float = 30.0,
    label: str = DEFAULT_WORKER_LABEL,
    plist_path: Path | None = None,
    apply: bool = False,
    load: bool = False,
) -> HookInstallResult:
    target = (plist_path or default_launchd_plist_path(label)).expanduser()
    log_dir = Path.home() / ".config" / "afs" / "agent-jobs"
    payload = render_launchd_plist(
        afs_root=afs_root,
        context_path=context_path,
        agent_name=agent_name,
        command=command,
        poll_seconds=poll_seconds,
        label=label,
        log_dir=log_dir,
    )
    current = target.read_bytes() if target.exists() else b""
    changed = current != payload
    loaded = False
    message = "worker LaunchAgent already current"
    if changed:
        message = "worker LaunchAgent would be installed"
    if apply:
        target.parent.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        if changed:
            target.write_bytes(payload)
            message = "worker LaunchAgent installed"
        if load:
            domain = f"gui/{_uid()}"
            subprocess.run(["launchctl", "bootout", f"{domain}/{label}"], check=False, capture_output=True)
            result = subprocess.run(
                ["launchctl", "bootstrap", domain, str(target)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                stderr = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(f"launchctl bootstrap failed: {stderr}")
            loaded = True
            message = "worker LaunchAgent installed and loaded"
    return HookInstallResult(
        target=str(target),
        changed=changed,
        applied=apply and changed,
        loaded=loaded,
        message=message,
    )


def _uid() -> int:
    import os

    return os.getuid()


def shell_hooks_installed(profile_path: Path | None = None) -> bool:
    target = (profile_path or default_shell_profile()).expanduser()
    if not target.exists():
        return False
    text = target.read_text(encoding="utf-8", errors="replace")
    return SHELL_BLOCK_BEGIN in text and SHELL_BLOCK_END in text


def worker_launchd_installed(label: str = DEFAULT_WORKER_LABEL) -> bool:
    return default_launchd_plist_path(label).exists()
