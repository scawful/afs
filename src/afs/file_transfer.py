"""Single UTF-8 file delivery; the destination AFS process owns the hash check."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from typing import Any


def push_text(
    content: str,
    *,
    host: str,
    mount: str,
    destination: str,
    if_match: str,
    context_root: str,
    project_path: str | None = None,
    remote_afs: str = "afs",
    mkdirs: bool = False,
) -> dict[str, Any]:
    """Send content on stdin to a single conditional write, never a tree sync.

    No retry is automatic: a conflict or ambiguous SSH failure requires a fresh
    destination read. The host's SSH configuration supplies authentication.
    """
    if not re.fullmatch(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.-]*", host):
        raise ValueError("host must be an SSH hostname or configured alias, optionally user@host")
    if not re.fullmatch(r"[0-9a-fA-F]{64}|missing", if_match):
        raise ValueError("if_match must be a SHA-256 digest or 'missing'")
    if not destination or destination.startswith("-") or not remote_afs or remote_afs.startswith("-"):
        raise ValueError("destination and remote executable must be nonempty and cannot start with '-'")
    command = [
        remote_afs, "fs", "write", mount, destination,
        "--context-root", context_root, "--if-match", if_match,
        "--encoding", "utf-8", "--errors", "strict", "--json",
    ]
    if project_path:
        command += ["--path", project_path]
    if mkdirs:
        command.append("--mkdirs")
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "--", host, shlex.join(command)],
        input=content.encode("utf-8"), capture_output=True, timeout=60, check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"remote write failed ({result.returncode}): {detail.strip()}")
    receipt = json.loads(result.stdout)
    if not isinstance(receipt, dict) or receipt.get("written") is not True:
        raise ValueError("remote AFS did not return a write receipt; read destination before retrying")
    return {"host": host, "destination": destination, "receipt": receipt}
