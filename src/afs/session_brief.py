"""Bounded startup context for agents with native reasoning and skill loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .manager import AFSManager
from .models import MountType
from .profiles import resolve_active_profile
from .scopes import resolve_scope, visible_mount_roots
from .skills import resolve_skill_roots


def build_session_brief(
    manager: AFSManager, context_path: Path, *, project_path: Path | None = None
) -> dict[str, Any]:
    """Read two named notes per visible scope; do not walk or match skill trees."""
    from .session_bootstrap import _safe_scoped_candidate

    context_path = context_path.expanduser().resolve()
    scope = resolve_scope(context_path, requester_path=project_path)
    scratchpad_root = manager.resolve_mount_root(context_path, MountType.SCRATCHPAD)
    roots = visible_mount_roots(
        scratchpad_root, mount_type=MountType.SCRATCHPAD, scoped=scope
    )
    notes: list[dict[str, Any]] = []
    for root in roots:
        for name in ("state.md", "deferred.md"):
            path = _safe_scoped_candidate(root / name, root=root, scoped=scope)
            if path is None or not path.is_file():
                continue
            with path.open(encoding="utf-8", errors="replace") as stream:
                text = stream.read(1201)
            notes.append({"path": str(path), "text": text[:1200], "truncated": len(text) > 1200})
    profile = resolve_active_profile(manager.config)
    return {
        "compact": True,
        "context_path": str(context_path),
        "scope_id": scope.scope_id,
        "layout_version": scope.layout_version,
        "profile": profile.name,
        "scratchpad_roots": [str(root) for root in roots],
        "notes": notes,
        "skills": {
            "mode": "native",
            "roots": [str(root) for root in resolve_skill_roots(list(profile.skill_roots))],
            "matches": [],
        },
        "next_commands": [
            "afs context query --help",
            "afs session bootstrap --json --no-write-artifacts",
        ],
        "guidance": (
            "Use the host's skill loader. Query context when the task needs prior decisions. "
            "Read a truncated note only when relevant. Use the full bootstrap for diagnostics."
        ),
    }


def render_session_brief(summary: dict[str, Any]) -> str:
    lines = [f"AFS: {summary['context_path']}", f"Scope: {summary['scope_id']}"]
    for note in summary["notes"]:
        lines.extend(["", note["path"], note["text"]])
        if note["truncated"]:
            lines.append("[truncated; read the source when relevant]")
    lines.extend(["", "Skill roots:", *summary["skills"]["roots"], "", summary["guidance"]])
    return "\n".join(lines)
