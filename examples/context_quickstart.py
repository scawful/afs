#!/usr/bin/env python3
"""Minimal context quickstart for core AFS.

This example uses a temporary directory so it does not modify your checkout.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from afs.core import resolve_context_root


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="afs-example-") as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()
        context = workspace / ".context"
        for mount in ["memory", "knowledge", "tools", "scratchpad", "history", "items"]:
            (context / mount).mkdir(parents=True, exist_ok=True)

        resolved = resolve_context_root(workspace)
        print(f"workspace: {workspace}")
        print(f"context:   {resolved}")
        print("created mounts:")
        for child in sorted(context.iterdir()):
            if child.is_dir():
                print(f"  - {child.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
