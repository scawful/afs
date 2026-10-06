#!/usr/bin/env bash
# training_watch.sh — Watch training data directories for changes
#
# Wraps `afs watch` to monitor training data directories and automatically:
#   1. Rebuild the dataset registry index
#   2. Run dataset QA summary on changed files
#   3. Print a concise watch-cycle summary for follow-on automation
#
# Usage:
#   ./scripts/training_watch.sh [--debounce SECONDS]
#
# Environment:
#   AFS_TRAINING_ROOT  Preferred training data root override
#   TRAINING_ROOT      Backwards-compatible training data root override
#   AFS_ROOT        Override AFS project root (default: script's parent dir)
#   QA_SCRIPT       Override QA summary script path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AFS_ROOT="${AFS_ROOT:-$(dirname "$SCRIPT_DIR")}"
TRAINING_ROOT="${AFS_TRAINING_ROOT:-${TRAINING_ROOT:-${AFS_CONTEXT_ROOT:-$HOME/.context}/scratchpad/common/training}}"
QA_SCRIPT="${QA_SCRIPT:-$AFS_ROOT/scripts/dataset_qa_summary.py}"
DEBOUNCE="30"
export AFS_ROOT TRAINING_ROOT QA_SCRIPT

if [[ "${1:-}" == "--debounce" ]] && [[ -n "${2:-}" ]]; then
    DEBOUNCE="$2"
fi

# Verify paths exist
if [[ ! -d "$TRAINING_ROOT" ]]; then
    echo "ERROR: Training root not found: $TRAINING_ROOT" >&2
    echo "Set TRAINING_ROOT to your training data directory." >&2
    exit 1
fi

# Build the on-change command
# This runs when afs watch detects changes in the watched directories
ON_CHANGE_CMD=$(cat <<'ONCMD'
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Training data changed, running QA..."

# Re-index datasets if registry module is available
python3 -c "
from pathlib import Path
try:
    from afs.training.registry import build_dataset_registry, write_dataset_registry
    root = Path('$TRAINING_ROOT')
    registry = build_dataset_registry(root)
    write_dataset_registry(registry, root / 'registry.json')
    print(f'Registry rebuilt: {len(registry)} datasets')
except ImportError:
    print('afs.training.registry not available, skipping registry rebuild')
except Exception as e:
    print(f'Registry rebuild error: {e}')
" 2>&1 || true

# Run QA summary on datasets directory if script exists
if [[ -f "$QA_SCRIPT" ]]; then
    # Find most recently modified JSONL files
    CHANGED=$(find "$TRAINING_ROOT" -name "*.jsonl" -mmin -5 2>/dev/null | head -3)
    for f in $CHANGED; do
        echo "QA: $f"
        python3 "$QA_SCRIPT" "$f" 2>&1 | tail -5 || true
    done
fi

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Training watch cycle complete"
ONCMD
)

echo "=== AFS Training Watch ==="
echo "Training root: $TRAINING_ROOT"
echo "AFS root:      $AFS_ROOT"
echo "Debounce:      ${DEBOUNCE}s"
echo "QA script:     $QA_SCRIPT"
echo "=========================="

# Run afs watch on the AFS context, which includes knowledge/tools/scratchpad
# The on-change hook handles training-specific actions
cd "$AFS_ROOT"
"${AFS_ROOT}/scripts/afs" watch \
    --debounce "$DEBOUNCE" \
    --on-change "$ON_CHANGE_CMD"
