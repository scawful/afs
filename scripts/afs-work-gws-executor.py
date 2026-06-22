#!/usr/bin/env python3
"""Execute one approved AFS action through Google Workspace CLI."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from afs.work_gws_executor import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
