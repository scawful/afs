"""Bounded context-warm entrypoint used by the shipped supervisor defaults."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from . import context_warm

AGENT_NAME = "context-warm"
AGENT_DESCRIPTION = (
    "Audit discovered contexts without workspace writes, repairs, or embedding calls."
)

_SAFE_DEFAULT_ARGS = (
    "--skip-workspace-sync",
    "--skip-embeddings",
    "--max-contexts",
    "100",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the existing context audit with conservative background defaults."""
    forwarded = list(sys.argv[1:] if argv is None else argv)
    return context_warm.main([*_SAFE_DEFAULT_ARGS, *forwarded])


if __name__ == "__main__":
    raise SystemExit(main())
