"""Resolve conservative project/common authorization scopes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .context_layout import LAYOUT_VERSION, detect_layout_version
from .project_registry import COMMON_SCOPE_ID, ProjectRegistry


@dataclass(frozen=True)
class ResolvedScope:
    context_root: Path
    requester_path: Path | None
    layout_version: int
    scope_id: str
    project_id: str
    project_name: str


def resolve_scope(
    context_root: Path,
    *,
    requester_path: Path | None = None,
    common: bool = False,
) -> ResolvedScope:
    """Resolve a scope without inferring project access from a central root.

    Version 1 contexts remain a single common scope. A version 2 caller that
    does not provide a requester path also falls back to common; project scope
    is granted only when the path matches an explicit registry record.
    """

    root = context_root.expanduser().resolve()
    requester = requester_path.expanduser().resolve() if requester_path else None
    version = detect_layout_version(root)
    fallback_name = requester.name if requester is not None else root.name
    if version != LAYOUT_VERSION or common or requester is None:
        return ResolvedScope(
            context_root=root,
            requester_path=requester,
            layout_version=version,
            scope_id=COMMON_SCOPE_ID,
            project_id="",
            project_name=fallback_name,
        )

    record = ProjectRegistry(root).resolve(requester)
    if record is None:
        raise PermissionError(f"project is not registered in central context: {requester}")
    return ResolvedScope(
        context_root=root,
        requester_path=requester,
        layout_version=version,
        scope_id=record.scope_id,
        project_id=record.project_id,
        project_name=record.name,
    )


__all__ = ["ResolvedScope", "resolve_scope"]
