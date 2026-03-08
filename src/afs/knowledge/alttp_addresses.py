"""Compatibility shim for legacy Zelda knowledge now owned by afs-scawful."""

from __future__ import annotations

try:
    from afs_scawful.knowledge.alttp_addresses import *  # type: ignore[F403]
except Exception as exc:  # pragma: no cover - compatibility path
    raise RuntimeError(
        "ALTTP knowledge moved to the afs-scawful extension."
    ) from exc
