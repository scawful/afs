# Training Infrastructure

Complete guide to the AFS training system for fine-tuning models like Majora and Veran.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Rehearsal Buffer System](#rehearsal-buffer-system)
4. [Oracle Training Generator](#oracle-training-generator)
5. [ToolBench Integration](#toolbench-integration)
6. [vast.ai Automation](#vastai-automation)
7. [Google Drive Backups](#google-drive-backups)
8. [Training New Models](#training-new-models)
9. [Adding New Datasets](#adding-new-datasets)
10. [Quality Pipeline](#quality-pipeline)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The AFS training infrastructure provides end-to-end tooling for:

- **Data generation** from codebases (Oracle, ALTTP disassembly)
- **Quality scoring** via ELECTRA discriminator and syntax validation
- **Rehearsal buffers** to prevent catastrophic forgetting
- **Dataset rebalancing** and split management
- **Cloud training** via vast.ai automation
- **Backup management** to Google Drive
- **Format conversion** for multiple training frameworks

**Key principle**: Portable JSONL datasets that work across machines and toolchains.

### Training Data Flow

```
Source Data → Export → Quality Scoring → Rehearsal Buffer → Split → Convert → Train
     ↓            ↓            ↓                 ↓            ↓         ↓        ↓
  AFS memory   JSONL    scored.jsonl      rehearsal.jsonl  train/   Alpaca   LoRA
  History                                                    val/    ShareGPT
  Gemini logs                                                test/   OpenAI
  Claude logs
  Codex logs
  Oracle code
  ToolBench
```

---

## Architecture

### Core Components

```
src/afs/
├── training/
│   ├── __init__.py           # Main exports (split_dataset, export_*, etc.)
│   ├── rehearsal.py          # RehearsalBuffer (prevent forgetting)
│   ├── scoring.py            # QualityScorer (ELECTRA + asar + entities)
│   ├── splitter.py           # Train/val/test splits
│   ├── rebalancer.py         # Dataset rebalancing by source/domain
│   ├── registry.py           # ModelRegistry (experiment tracking)
│   └── converters/           # Format converters (Alpaca, ShareGPT, OpenAI)
├── oracle/
│   └── training_generator.py # Extract from Oracle codebase
├── discriminator/
│   └── electra.py            # ASM quality discriminator
└── cli/
    └── training.py           # CLI commands
```

### Output Locations (gitignored)

```
~/src/lab/afs/
├── training_data/      # Exported datasets (JSONL)
├── models/             # Trained models (LoRA adapters)
├── distillation_data/  # Generated samples
├── benchmark_results/  # Evaluation results
└── data/               # Misc artifacts

~/.context/training/    # Training data cache
└── rehearsal/          # Rehearsal buffers by version
```

---

## Rehearsal Buffer System

**Problem**: Each training version (v1→v2→v3→v4) loses knowledge from previous versions (catastrophic forgetting).

**Solution**: Maintain high-quality samples from previous training runs and mix them into new training data.

### How It Works

```python
from afs.training import RehearsalBuffer

# 1. Build buffer from v4 training data
buffer = RehearsalBuffer()
buffer.load_from_jsonl(
    Path("models/veran_v4_training.jsonl"),
    version="v4"
)

# 2. Select top 30% by quality
buffer.select_top_samples(ratio=0.3)

# 3. Mix with new v5 data (30% rehearsal, 70% new)
new_samples = load_samples("models/veran_v5_new.jsonl")
mixed = buffer.merge_with_new_data(
    new_samples,
    rehearsal_ratio=0.3,
    shuffle=True,
    seed=42
)

# 4. Save buffer for v6
buffer.save(Path("~/.context/training/rehearsal/veran_v4.jsonl"))
```

### Configuration

```python
from afs.training import RehearsalBufferConfig

config = RehearsalBufferConfig(
    quality_threshold=0.5,  # Min quality score to include
    top_ratio=0.3,          # Keep top 30%
    enable_diversity=True,  # Diversity sampling
    max_per_domain=None,    # Max samples per domain
    track_provenance=True,  # Track version metadata
)

buffer = RehearsalBuffer(config=config)
```

### CLI Usage

```bash
# Build buffer from previous training
afs training rehearsal-build \
  --input models/veran_v4_training.jsonl \
  --version v4 \
  --output ~/.context/training/rehearsal/veran_v4.jsonl \
  --top-ratio 0.3

# Mix with new data
afs training rehearsal-mix \
  --buffer ~/.context/training/rehearsal/veran_v4.jsonl \
  --new-data models/veran_v5_new.jsonl \
  --rehearsal-ratio 0.3 \
  --output models/veran_v5_training.jsonl
```

### Best Practices

1. **Keep top 20-40%** of previous version by quality score
2. **Rehearsal ratio 0.2-0.4** (20-40% old, 60-80% new)
3. **Track provenance** to debug version-specific issues
4. **Diversity sample** across domains to avoid overfitting
5. **Save buffers** after each major version for future use

---

## Oracle Training Generator

Extracts training samples from the Oracle of Secrets codebase (SNES ROM hack).

### What It Generates

- **Documentation Q&A**: "What is [topic]?" → section content
- **Assembly explanations**: "What does [subroutine] do?" → comments + structure
- **Memory lookups**: "What address stores [variable]?" → $7EXXXX + description
- **Quest progression**: "How do I progress [quest]?" → step-by-step guide
- **Architecture**: "How does [system] work?" → design patterns

### Expected Output

~2,845 raw samples → ~2,156 after quality filtering (>0.6)

### Usage

```bash
python3 -m afs.oracle.training_generator \
  --oracle-path ~/src/hobby/oracle-of-secrets \
  --output ~/.context/training/oracle/majora_v1_raw.jsonl \
  --limit 1000  # Optional: for testing
```

### Programmatic Usage

```python
from pathlib import Path
from afs.oracle.training_generator import OracleTrainingGenerator

generator = OracleTrainingGenerator(
    oracle_path=Path("~/src/hobby/oracle-of-secrets")
)

with open("majora_v1_raw.jsonl", "w") as f:
    for sample in generator.generate():
        f.write(json.dumps(sample.to_dict()) + "\n")
```

### Customization

Subclass `OracleTrainingGenerator` to add domain-specific extraction:

```python
class CustomOracleGenerator(OracleTrainingGenerator):
    def generate_sprite_samples(self) -> Iterator[TrainingSample]:
        """Extract sprite creation patterns."""
        sprite_docs = self.docs_path / "Sprites"

        for sprite_file in sprite_docs.rglob("*.md"):
            # Parse sprite documentation
            # Generate Q&A pairs
            yield TrainingSample(
                instruction="How do I create a new sprite in Oracle?",
                output="...",
                domain="oracle_sprites",
                source=str(sprite_file.relative_to(self.oracle_path))
            )
```

---

## ToolBench Integration

ToolBench provides tool-use training data for teaching models to invoke CLI commands and APIs.

### Export from ToolBench

```bash
# Export ToolBench dataset to AFS format
afs training toolbench-export \
  --dataset-dir ~/data/toolbench \
  --output training_data/toolbench_train.jsonl \
  --split train \
  --limit 10000  # Optional
```

### Dataset Structure

ToolBench samples are converted to TrainingSample format:

```json
{
  "instruction": "Use the weather API to get current conditions for New York",
  "input": "API endpoint: /weather/current?city=New+York",
  "output": "curl -X GET 'https://api.weather.com/v1/current?city=New+York' -H 'Authorization: Bearer TOKEN'",
  "thinking": "The user wants current weather. I'll use the /current endpoint with the city parameter.",
  "domain": "toolbench_api",
  "source": "toolbench_train_001234"
}
```

### Mix with Other Data

```bash
# Rebalance ToolBench with domain-specific data
afs training rebalance \
  --input training_data/toolbench_train.jsonl \
  --input training_data/oracle_majora_v1.jsonl \
  --input training_data/memory_export.jsonl \
  --output training_data/mixed_training.jsonl \
  --weight toolbench=0.3 \
  --weight oracle=0.4 \
  --weight memory=0.3
```

---

## vast.ai Automation

Parallel cloud GPU training with automatic cost management.

### Prerequisites

```bash
# Install vast.ai CLI
pip install vastai

# Configure API key
vastai set api-key YOUR_KEY

# Test connection
vastai show instances
```

### Launch Training

```bash
# Single model
python3 scripts/vastai_setup.py \
  --model majora \
  --budget 50

# All models in parallel
python3 scripts/vastai_setup.py \
  --all-models \
  --budget 100

# Dry run (preview without launching)
python3 scripts/vastai_setup.py \
  --model majora \
  --dry-run
```

### GPU Configurations

```python
# Budget: RTX 3090 @ $0.30/hour
GPU_CONFIGS["budget"] = GPUConfig(
    gpu_name="RTX 3090",
    num_gpus=1,
    disk_space=50,
    max_price=0.30
)

# Balanced: RTX 4090 @ $0.50/hour
GPU_CONFIGS["balanced"] = GPUConfig(
    gpu_name="RTX 4090",
    num_gpus=1,
    disk_space=100,
    max_price=0.50
)

# Performance: A100 @ $1.50/hour
GPU_CONFIGS["performance"] = GPUConfig(
    gpu_name="A100",
    num_gpus=1,
    disk_space=200,
    max_price=1.50
)
```

### Training Jobs

```python
TRAINING_JOBS = {
    "majora": TrainingJob(
        model_name="majora-v1",
        training_script="scripts/train_majora_v1.py",
        training_data=Path("models/majora_v1_training.jsonl"),
        output_path=Path("/workspace/output/majora-v1-lora"),
        epochs=3,
        gpu_config="balanced",  # RTX 4090
        estimated_hours=4.0,
    ),
}
```

### Monitor Instances

```bash
# Check status
python3 scripts/vastai_setup.py --monitor

# Output:
# Active Training Instances
# ========================================
#
# Instance 12345 (majora-v1)
#   Status: running
#   Runtime: 2.3h
#   Cost: $1.15
```

### Cleanup

```bash
# Destroy all instances
python3 scripts/vastai_setup.py --cleanup
```

### Custom Training Scripts

Place training scripts in `scripts/` directory:

```python
# scripts/train_majora_v1.py
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    # Training logic here
    # Use unsloth, MLX, or any framework

if __name__ == "__main__":
    main()
```

---

## Google Drive Backups

Automated backup to Google Drive for disaster recovery.

### Prerequisites

1. Install Google Drive File Stream (macOS/Windows) or rclone (Linux)
2. Mount Google Drive at `~/Google Drive/My Drive/`

### Backup Structure

```
Google Drive/AFS_Backups/
├── training_data/
│   ├── majora_v1_20260114.tar.gz
│   ├── veran_v5_20260114.tar.gz
│   └── ...
├── models/
│   ├── majora-v1-lora_20260114.tar.gz
│   └── ...
├── evaluations/
│   └── ...
└── logs/
    └── backup_20260114_103045.log
```

### Usage

```bash
# Backup all training data from ~/.context/training/
python3 scripts/gdrive_backup.py --training-data

# Backup specific model
python3 scripts/gdrive_backup.py \
  --model majora-v1-lora \
  --path /workspace/output/majora-v1-lora

# Backup everything (training data + models)
python3 scripts/gdrive_backup.py --all

# List backups
python3 scripts/gdrive_backup.py --list

# Cleanup old backups (keep last 5)
python3 scripts/gdrive_backup.py --cleanup --keep 5

# Dry run (preview without executing)
python3 scripts/gdrive_backup.py --training-data --dry-run
```

### Automated Backups

Add to crontab for automatic backups:

```bash
# Backup training data daily at 2 AM
0 2 * * * cd ~/src/lab/afs && python3 scripts/gdrive_backup.py --training-data

# Cleanup old backups weekly
0 3 * * 0 cd ~/src/lab/afs && python3 scripts/gdrive_backup.py --cleanup --keep 5
```

### Restore from Backup

```bash
# Extract backup
cd ~/Google\ Drive/My\ Drive/AFS_Backups/models/
tar -xzf majora-v1-lora_20260114.tar.gz -C /workspace/output/
```

---

## Training New Models

Step-by-step guide to training a new model (e.g., Majora v2).

### 1. Export Training Data

```bash
# Export from multiple sources
afs training memory-export --output training_data/memory.jsonl
afs training history-export --output training_data/history.jsonl
afs training claude-export --output training_data/claude.jsonl
afs training gemini-export --output training_data/gemini.jsonl

# Generate from Oracle codebase
python3 -m afs.oracle.training_generator \
  --oracle-path ~/src/hobby/oracle-of-secrets \
  --output training_data/oracle.jsonl

# Export ToolBench
afs training toolbench-export \
  --dataset-dir ~/data/toolbench \
  --output training_data/toolbench.jsonl \
  --split train
```

### 2. Quality Scoring

```bash
# Score all datasets (adds quality_score field)
for dataset in training_data/*.jsonl; do
  afs discriminator score \
    --model models/electra \
    --input "$dataset" \
    --output "${dataset%.jsonl}_scored.jsonl"
done
```

### 3. Build Rehearsal Buffer (if retraining)

```bash
# Load previous version
afs training rehearsal-build \
  --input models/majora_v1_training.jsonl \
  --version v1 \
  --output ~/.context/training/rehearsal/majora_v1.jsonl \
  --top-ratio 0.3
```

### 4. Rebalance and Mix

```bash
# Mix datasets with rehearsal buffer
afs training rebalance \
  --input training_data/oracle_scored.jsonl \
  --input training_data/memory_scored.jsonl \
  --input training_data/toolbench_scored.jsonl \
  --input ~/.context/training/rehearsal/majora_v1.jsonl \
  --output training_data/majora_v2_mixed.jsonl \
  --weight oracle=0.35 \
  --weight memory=0.20 \
  --weight toolbench=0.25 \
  --weight rehearsal=0.20 \
  --min-quality-score 0.5
```

### 5. Split Dataset

```bash
# Create train/val/test splits
afs training prepare \
  --input training_data/majora_v2_mixed.jsonl \
  --output training_data/majora_v2_splits \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --stratify-by domain \
  --seed 42
```

### 6. Convert Format

```bash
# Convert to Alpaca format
afs training convert \
  --input training_data/majora_v2_splits/train.jsonl \
  --output training_data/majora_v2_splits/train_alpaca.jsonl \
  --format alpaca

# Or ShareGPT format
afs training convert \
  --input training_data/majora_v2_splits/train.jsonl \
  --output training_data/majora_v2_splits/train_sharegpt.jsonl \
  --format sharegpt
```

### 7. Train on vast.ai

```bash
# Launch training
python3 scripts/vastai_setup.py \
  --model majora \
  --budget 50

# Monitor progress
python3 scripts/vastai_setup.py --monitor
```

### 8. Backup Results

```bash
# Backup trained model
python3 scripts/gdrive_backup.py \
  --model majora-v2-lora \
  --path /workspace/output/majora-v2-lora

# Backup training data
python3 scripts/gdrive_backup.py --training-data
```

### 9. Register Experiment

```bash
# Track in registry
afs training registry-create \
  --name majora-v2-baseline \
  --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \
  --framework unsloth \
  --dataset training_data/majora_v2_mixed.jsonl \
  --tag majora \
  --tag oracle \
  --notes "First v2 training with rehearsal buffer"
```

---

## Adding New Datasets

### Custom Dataset Generators

Create a generator that yields `TrainingSample` objects:

```python
from pathlib import Path
from typing import Iterator
from afs.generators.base import BaseGenerator, TrainingSample

class MyCustomGenerator(BaseGenerator):
    """Generate training data from custom source."""

    def __init__(self, source_path: Path):
        super().__init__(name="custom", domain="custom")
        self.source_path = source_path

    def generate(self) -> Iterator[TrainingSample]:
        """Generate training samples."""
        for file in self.source_path.rglob("*.txt"):
            # Parse file
            content = file.read_text()

            # Create sample
            yield TrainingSample(
                instruction="Question based on content",
                output="Answer from content",
                thinking="Reasoning process",
                domain="custom_domain",
                source=str(file.relative_to(self.source_path)),
                _metadata={"file_type": "txt"}
            )

# Use the generator
generator = MyCustomGenerator(Path("~/my_data"))
samples = list(generator.generate())
```

### Export from Custom Sources

Create an export function:

```python
from pathlib import Path
from afs.training import export_to_jsonl
from afs.generators.base import TrainingSample

def export_my_data(
    source_path: Path,
    output_path: Path,
    limit: int | None = None
) -> int:
    """Export custom data to TrainingSample JSONL."""

    samples = []
    for item in parse_source(source_path):
        sample = TrainingSample(
            instruction=item["question"],
            output=item["answer"],
            domain="custom",
            source=item["id"]
        )
        samples.append(sample)

        if limit and len(samples) >= limit:
            break

    # Write to JSONL
    count = 0
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")
            count += 1

    return count
```

### Register CLI Command

Add to `src/afs/cli/training.py`:

```python
def training_mycustom_export_command(args: argparse.Namespace) -> int:
    """Export my custom data to TrainingSample JSONL."""
    from ..training import export_my_data

    result = export_my_data(
        source_path=Path(args.source),
        output_path=Path(args.output),
        limit=args.limit
    )

    print(f"Exported {result} samples")
    return 0

# In register_parsers():
train_mycustom = training_sub.add_parser(
    "mycustom-export",
    help="Export my custom data"
)
train_mycustom.add_argument("--source", required=True)
train_mycustom.add_argument("--output", required=True)
train_mycustom.add_argument("--limit", type=int)
train_mycustom.set_defaults(func=training_mycustom_export_command)
```

---

## Quality Pipeline

The quality pipeline combines multiple scoring signals for sample selection.

### Quality Scoring Components

```python
from afs.training.scoring import QualityScorer, ScoringConfig

config = ScoringConfig(
    weights=ScoringWeights(
        electra=0.4,   # ELECTRA discriminator (real vs fake)
        asar=0.3,      # Asar syntax validation
        entity=0.2,    # Entity coverage (known addresses)
        length=0.1     # Length/structure heuristics
    ),
    min_output_length=50,
    ideal_output_length=500,
    max_output_length=5000
)

scorer = QualityScorer(config=config)

# Score a sample
from afs.generators.base import TrainingSample

sample = TrainingSample(
    instruction="What does LDA do?",
    output="LDA loads the accumulator with a value from memory.",
    domain="asm"
)

quality = scorer.score(sample)
print(f"Overall: {quality.overall:.2f}")
print(f"ELECTRA: {quality.electra_score:.2f}")
print(f"Asar valid: {quality.asar_valid}")
print(f"Entity coverage: {quality.entity_coverage:.2f}")
```

### Filter by Quality

```bash
# Filter with discriminator
afs discriminator filter \
  --model models/electra \
  --input training_data/raw.jsonl \
  --output training_data/filtered.jsonl \
  --rejected training_data/rejected.jsonl \
  --min-score 0.6
```

### Quality Profiles

Use domain-specific scoring profiles:

```python
# Generic profile (balanced weights)
config = ScoringConfig.from_profile("generic")

# Dialogue profile (favor length and entity coverage)
config = ScoringConfig.from_profile("dialogue")

# ASM profile (favor ELECTRA and asar validation)
config = ScoringConfig.from_profile("asm")
```

### Batch Scoring

```python
from afs.training import score_jsonl

# Score entire dataset
score_jsonl(
    input_path=Path("training_data/raw.jsonl"),
    output_path=Path("training_data/scored.jsonl"),
    config=config,
    min_score=0.5  # Filter out scores < 0.5
)
```

---

## Troubleshooting

### Common Issues

#### 1. Low Quality Scores

**Problem**: Most samples score below 0.5

**Solutions**:
- Lower `min_quality_score` threshold: `--min-quality-score 0.3`
- Check ELECTRA model is trained on relevant domain
- Disable specific scoring components: `--no-asar` or `--no-electra`
- Use generic profile: `--score-profile generic`

#### 2. Catastrophic Forgetting

**Problem**: New model version loses knowledge from previous versions

**Solutions**:
- Build rehearsal buffer from previous version
- Increase rehearsal ratio: `--rehearsal-ratio 0.4`
- Mix more diverse domains in training data
- Train with more epochs (3-5 instead of 1-2)

#### 3. vast.ai Instance Fails

**Problem**: Training job fails on vast.ai

**Solutions**:
- Check logs: `vastai logs <instance_id>`
- Increase disk space: `disk_space=100` in GPU config
- Use more reliable GPU: Switch from "budget" to "balanced"
- Add retry logic in training script
- Pre-download models to avoid network timeouts

#### 4. Out of Memory (OOM)

**Problem**: Training crashes with CUDA OOM

**Solutions**:
- Reduce batch size: `--batch-size 8` instead of 32
- Use gradient accumulation: `--gradient-accumulation-steps 4`
- Enable gradient checkpointing
- Use smaller model or quantization (4-bit LoRA)
- Choose GPU with more VRAM (A100 40GB instead of RTX 3090 24GB)

#### 5. Dataset Imbalance

**Problem**: One domain dominates training

**Solutions**:
- Use rebalancing: `afs training rebalance --weight domain1=0.3 --weight domain2=0.7`
- Enable diversity sampling in rehearsal buffer
- Set `max_per_domain` in rehearsal config
- Oversample minority domains: `--allow-oversample`

#### 6. Quality Scoring is Slow

**Problem**: Scoring takes hours for large datasets

**Solutions**:
- Disable expensive components: `--no-asar` or `--no-electra`
- Use GPU for ELECTRA inference
- Process in batches: `--batch-size 128`
- Run in parallel: Split dataset and score concurrently
- Cache entity extractor results

#### 7. Google Drive Backup Fails

**Problem**: Backup script cannot write to Google Drive

**Solutions**:
- Check Google Drive File Stream is mounted: `ls ~/Google\ Drive/`
- Verify permissions: `touch ~/Google\ Drive/My\ Drive/test.txt`
- Use rclone as fallback: `rclone copy local_path gdrive:AFS_Backups/`
- Free up Google Drive storage space
- Check network connectivity

#### 8. Training Data is Corrupted

**Problem**: JSONL file has malformed JSON

**Solutions**:
- Validate JSONL: `python3 -c "import json; [json.loads(l) for l in open('data.jsonl')]"`
- Fix with jq: `cat data.jsonl | jq -c '.' > data_fixed.jsonl`
- Re-export with redaction: `--no-redact` might have encoding issues
- Check for null bytes: `grep -a '\x00' data.jsonl`

#### 9. Model Doesn't Improve

**Problem**: Loss plateaus, model doesn't learn

**Solutions**:
- Check data quality: Are samples diverse and correct?
- Increase learning rate: `--learning-rate 5e-4` instead of 1e-5
- Train for more epochs: 5-10 instead of 3
- Use better base model (7B instead of 3B)
- Add more training data (aim for 10k+ samples)
- Enable curriculum learning: Start with easier samples

#### 10. Can't Find Training Data

**Problem**: Export commands say "No data found"

**Solutions**:
- Check source paths exist: `ls ~/.context/memory`
- Verify AFS context is initialized: `afs status`
- Run discovery: `afs context discover --path ~/src`
- Use absolute paths: `--memory-root ~/.context/memory`
- Check config: `cat ~/.afs/config.toml`

---

## Quick Reference

### Essential Commands

```bash
# Export training data
afs training memory-export --output training_data/memory.jsonl
afs training history-export --output training_data/history.jsonl
afs training claude-export --output training_data/claude.jsonl

# Generate from Oracle
python3 -m afs.oracle.training_generator \
  --oracle-path ~/src/hobby/oracle-of-secrets \
  --output training_data/oracle.jsonl

# Quality scoring
afs discriminator filter \
  --model models/electra \
  --input training_data/raw.jsonl \
  --output training_data/filtered.jsonl \
  --min-score 0.6

# Rehearsal buffer
afs training rehearsal-build \
  --input models/v1_training.jsonl \
  --version v1 \
  --output ~/.context/training/rehearsal/v1.jsonl \
  --top-ratio 0.3

# Rebalance
afs training rebalance \
  --input training_data/oracle.jsonl \
  --input training_data/memory.jsonl \
  --output training_data/mixed.jsonl \
  --weight oracle=0.5 --weight memory=0.5

# Split
afs training prepare \
  --input training_data/mixed.jsonl \
  --output training_data/splits \
  --train-ratio 0.8 --val-ratio 0.1

# Convert format
afs training convert \
  --input training_data/splits/train.jsonl \
  --format alpaca

# Train on vast.ai
python3 scripts/vastai_setup.py --model majora --budget 50

# Backup
python3 scripts/gdrive_backup.py --training-data
```

---

## See Also

- [TRAINING_INFRA.md](./TRAINING_INFRA.md) - Legacy training docs
- [GGUF_CONVERSION.md](./GGUF_CONVERSION.md) - LoRA to GGUF conversion
- [NETWORK_INFERENCE.md](./NETWORK_INFERENCE.md) - LMStudio network setup
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Module overview
- [GLOSSARY.md](./GLOSSARY.md) - Term definitions
