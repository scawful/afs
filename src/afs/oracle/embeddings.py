"""Compatibility shim for Oracle domain module now owned by afs-scawful."""

from __future__ import annotations

try:
    from afs_scawful.oracle.embeddings import *  # type: ignore[F403]
except Exception as exc:  # pragma: no cover - compatibility path
    raise RuntimeError(
        "Oracle domain module 'embeddings' moved to the afs-scawful extension."
    ) from exc
