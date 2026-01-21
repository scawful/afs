#!/bin/bash
# AFS Disk Cleanup Script
# Removes processed raw data and regenerable files
# SAFE: Only deletes files we've already converted/can regenerate

set -e

echo "🧹 AFS Disk Cleanup Script"
echo "=========================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

TOTAL_FREED=0

# Function to get directory size in MB
get_size_mb() {
    du -sm "$1" 2>/dev/null | awk '{print $1}'
}

# Function to safely remove and track space
safe_remove() {
    local path="$1"
    local desc="$2"

    if [ -e "$path" ]; then
        local size=$(get_size_mb "$path")
        echo -e "${YELLOW}Removing: $desc${NC}"
        echo "  Path: $path"
        echo "  Size: ${size}MB"
        rm -rf "$path"
        TOTAL_FREED=$((TOTAL_FREED + size))
        echo -e "${GREEN}✅ Removed${NC}"
        echo ""
    else
        echo -e "${YELLOW}⚠️  Not found: $desc${NC}"
        echo "  Path: $path"
        echo ""
    fi
}

echo "Step 1: Removing CodeSearchNet raw data (already processed to 624KB)"
echo "---------------------------------------------------------------------"
safe_remove "$HOME/.context/training/datasets/CodeSearchNet" "CodeSearchNet raw (4.8GB)"

echo "Step 2: Removing ToolBench raw data (already processed to 193MB)"
echo "----------------------------------------------------------------"
safe_remove "$HOME/.context/training/datasets/ToolBench" "ToolBench raw (516MB)"

echo "Step 3: Removing old test directories from December"
echo "---------------------------------------------------"
safe_remove "$HOME/.context/training/datasets/alttp_oracle_full_20251222_144223" "Old test 1"
safe_remove "$HOME/.context/training/datasets/alttp_oracle_full_20251222_144447" "Old test 2"
safe_remove "$HOME/.context/training/datasets/curated_hacks_pilot_20251222_122853" "Old pilot 1"
safe_remove "$HOME/.context/training/datasets/curated_hacks_pilot_20251222_123116" "Old pilot 2"
safe_remove "$HOME/.context/training/datasets/curated_hacks_pilot_20251222_125314" "Old pilot 3"
safe_remove "$HOME/.context/training/datasets/phase1_diversity_test_20251222_012112" "Old phase test 1"
safe_remove "$HOME/.context/training/datasets/phase1_diversity_test_20251222_012519" "Old phase test 2"
safe_remove "$HOME/.context/training/datasets/phase1_diversity_test_20251222_013401" "Old phase test 3"
safe_remove "$HOME/.context/training/datasets/pilot_test" "Old pilot test"
safe_remove "$HOME/.context/training/datasets/final_test_20251229_190848" "Old final test"
safe_remove "$HOME/.context/training/datasets/kg_disabled_test_20251229_190003" "Old KG test 1"
safe_remove "$HOME/.context/training/datasets/kg_full_credit_20251229_192419" "Old KG test 2"

echo "Step 4: Removing build artifacts (regenerable with cmake)"
echo "---------------------------------------------------------"
safe_remove "$HOME/src/lab/afs/build" "AFS build artifacts (447MB)"

echo "Step 5: Removing Python venv (reinstallable with pip)"
echo "-----------------------------------------------------"
safe_remove "$HOME/src/lab/afs/venv" "Python virtual environment (712MB)"

echo ""
echo "==========================================="
echo -e "${GREEN}✅ CLEANUP COMPLETE${NC}"
echo "==========================================="
echo ""
echo "Total space freed: ${TOTAL_FREED}MB (~$((TOTAL_FREED / 1024))GB)"
echo ""
echo "Checking disk space:"
df -h / | grep -E "Filesystem|/$"
echo ""
echo "What was kept:"
echo "  ✅ Processed JSONL files (~200MB)"
echo "  ✅ Enhanced datasets (1,867 samples)"
echo "  ✅ All scripts and documentation"
echo "  ✅ Training results and configs"
echo ""
echo "What was removed:"
echo "  🗑️  Raw source datasets (already processed)"
echo "  🗑️  Old test runs from December"
echo "  🗑️  Build artifacts (regenerable)"
echo "  🗑️  Python venv (reinstallable)"
echo ""
echo "To rebuild venv:"
echo "  cd ~/src/lab/afs"
echo "  python3 -m venv venv"
echo "  source venv/bin/activate"
echo "  pip install -r requirements.txt"
echo ""
