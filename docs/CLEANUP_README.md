# Disk Cleanup Guide

## Quick Start

Free up **~6.5GB** of disk space by removing processed raw data and regenerable files:

```bash
cd ~/src/lab/afs/scripts
./cleanup_disk.sh
```

## What Gets Removed

### 1. CodeSearchNet Raw Data (4.8GB)
- **Status:** Already processed to 624KB JSONL ✅
- **Safe to delete:** Yes
- **Location:** `~/.context/training/datasets/CodeSearchNet/`

### 2. ToolBench Raw Data (516MB)
- **Status:** Already processed to 193MB JSONL ✅
- **Safe to delete:** Yes
- **Location:** `~/.context/training/datasets/ToolBench/`

### 3. Old Test Directories (~100MB)
- **Status:** December 2025 test runs, no longer needed
- **Safe to delete:** Yes
- **Locations:**
  - `alttp_oracle_full_*` (multiple versions)
  - `curated_hacks_pilot_*`
  - `phase1_diversity_test_*`
  - `final_test_*`
  - `kg_*_test_*`

### 4. Build Artifacts (447MB)
- **Status:** Regenerable with `cmake --build build`
- **Safe to delete:** Yes
- **Location:** `~/src/lab/afs/build/`

### 5. Python Virtual Environment (712MB)
- **Status:** Reinstallable with pip
- **Safe to delete:** Yes
- **Location:** `~/src/lab/afs/venv/`

## What Gets Kept

✅ All processed JSONL files (~200MB)
✅ Enhanced datasets (1,867 samples)
✅ All scripts and source code
✅ Complete documentation
✅ Training configurations
✅ Evaluation results
✅ Model metadata

## Total Space Freed

**~6,577MB (~6.5GB)**

## If You Need To Rebuild

### Rebuild Python Environment
```bash
cd ~/src/lab/afs
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Rebuild C++ Build
```bash
cd ~/src/lab/afs
cmake -B build
cmake --build build
```

### Re-download Raw Datasets (if needed)
```bash
# CodeSearchNet (only if you need the raw data again)
cd ~/.context/training/datasets
git clone https://github.com/github/CodeSearchNet

# ToolBench (only if you need the raw data again)
huggingface-cli download tuandunghcmut/toolbench-v1 \
  --repo-type dataset \
  --local-dir ToolBench
```

## Safety

This script only removes:
- Raw source files we've already converted
- Old test runs from December
- Regenerable build artifacts
- Reinstallable dependencies

**It does NOT remove:**
- Any training data
- Enhanced datasets
- Scripts or documentation
- Model weights or configs
- Evaluation results

## Verification

After cleanup, verify important files are still there:

```bash
# Check processed datasets
ls -lh ~/src/lab/afs/models/*_enhanced.jsonl

# Check training data
ls -lh ~/.context/training/toolbench/processed/

# Check scripts
ls ~/src/lab/afs/scripts/*.py | wc -l

# Check docs
ls ~/src/lab/afs/docs/*.md | wc -l
```

All should show files present!
