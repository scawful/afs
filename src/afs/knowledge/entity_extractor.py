"""Compatibility shim for legacy Zelda entity extraction now owned by afs-scawful."""

from __future__ import annotations

try:
    from afs_scawful.knowledge.entity_extractor import *  # type: ignore[F403]
except Exception as exc:  # pragma: no cover - compatibility path
    raise RuntimeError(
        "ALTTP entity extraction moved to the afs-scawful extension."
    ) from exc
