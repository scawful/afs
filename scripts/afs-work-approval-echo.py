#!/usr/bin/env python3
"""Echo an AFS work approval payload without changing external systems."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: afs-work-approval-echo.py <approval-json>", file=sys.stderr)
        return 2
    payload_path = Path(argv[-1]).expanduser()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    approval = payload["approval"]
    print(
        json.dumps(
            {
                "ok": True,
                "approval_id": approval["approval_id"],
                "target_system": approval["target_system"],
                "target_id": approval["target_id"],
                "action": approval["action"],
                "note": "echo only; no external write performed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
