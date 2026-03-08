"""Compatibility shim for legacy Zelda knowledge adapter now owned by afs-scawful."""

from __future__ import annotations

try:
    from afs_scawful.knowledge.adapters.alttp_adapter import *  # type: ignore[F403]
except Exception as exc:  # pragma: no cover - compatibility path
    raise RuntimeError(
        "ALTTP knowledge adapter moved to the afs-scawful extension."
    ) from exc
