#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HOME"
exec "$ROOT/scripts/afs" manager open --path "$HOME"
