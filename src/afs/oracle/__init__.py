"""Compatibility shim for Oracle domain modules now owned by afs-scawful."""

from __future__ import annotations

try:
    from afs_scawful.oracle import *  # type: ignore[F403]
except Exception as exc:  # pragma: no cover - compatibility path
    raise RuntimeError(
        "Oracle domain modules moved to the afs-scawful extension."
    ) from exc
