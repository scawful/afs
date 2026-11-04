#!/usr/bin/env bash
# One compact briefing; the native client owns skill loading and reasoning.
# Install as ~/.gemini/hooks/afs_start.sh after replacing <AFS_ROOT>.
set -euo pipefail
exec "<AFS_ROOT>/.venv/bin/python" "<AFS_ROOT>/scripts/harness/afs-session-start" --host gemini
