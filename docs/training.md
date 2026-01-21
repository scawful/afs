# Training Guide

Complete walkthrough for the AFS training pipeline, from data preparation to model deployment.

## Table of Contents

1. [Overview](#overview)
2. [Data Export](#data-export)
3. [Quality Scoring](#quality-scoring)
4. [Rehearsal Buffers](#rehearsal-buffers)
5. [Dataset Rebalancing](#dataset-rebalancing)
6. [Splitting Data](#splitting-data)
7. [Format Conversion](#format-conversion)
8. [Training Models](#training-models)
9. [Experiment Tracking](#experiment-tracking)
10. [Troubleshooting](#troubleshooting)

## Overview

The training pipeline has 7 main stages:

```
┌──────────────┐
│ Raw Sources  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Export to JSONL  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Quality Scoring  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Rehearsal Buffer │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Rebalance        │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Split Data       │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Format Convert   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Train Model      │
└──────────────────┘
```

## Data Export

The first step is exporting training data from various sources.

### From Memory System

Export stored memories and contexts:

```bash
afs training memory-export \
  --output training_data/memory.jsonl \
  --memory-root ~/.context/memory
```

Output format:
```json
{
  "instruction": "Question or task",
  "output": "Memory content or answer",
  "domain": "memory",
  "source": "path/in/memory"
}
```

### From Chat History

Export previous conversation threads:

```bash
afs training history-export \
  --output training_data/history.jsonl \
  --history-root ~/.context/history
```

### From Oracle of Secrets

Extract training data from the Oracle ROM hack codebase:

```bash
python3 -m afs.oracle.training_generator \
  --oracle-path ~/src/hobby/oracle-of-secrets \
  --output training_data/oracle.jsonl
```

This generates:
- Documentation Q&A pairs
- Assembly subroutine explanations
- Memory address lookups
- Quest progression guides
- Architecture documentation

Expected output: ~2,200 samples after filtering

### From Gemini/Claude Logs

Export logs from other AI interactions:

```bash
afs training gemini-export --output training_data/gemini.jsonl
afs training claude-export --output training_data/claude.jsonl
afs training codex-export --output training_data/codex.jsonl
```

### Custom Data Sources

Create a custom generator:

```python
from pathlib import Path
from afs.generators.base import BaseGenerator, TrainingSample
from typing import Iterator

class MyGenerator(BaseGenerator):
    def __init__(self, source_path: Path):
        super().__init__(name="my_source", domain="custom")
        self.source_path = source_path

    def generate(self) -> Iterator[TrainingSample]:
        """Generate training samples."""
        for file in self.source_path.rglob("*.txt"):
            content = file.read_text()

            yield TrainingSample(
                instruction="Your question",
                output="Expected output",
                thinking="Reasoning (optional)",
                domain="custom",
                source=str(file.relative_to(self.source_path))
            )

# Export to JSONL
generator = MyGenerator(Path("~/my_data"))
with open("training_data/custom.jsonl", "w") as f:
    for sample in generator.generate():
        f.write(sample.to_json() + "\n")
```

## Quality Scoring

Quality scoring filters out low-quality samples and weights high-quality ones.

### Scoring Components

The scorer combines multiple signals:

```python
from afs.training.scoring import QualityScorer, ScoringConfig

config = ScoringConfig(
    weights=ScoringWeights(
        electra=0.4,      # ELECTRA discriminator (real vs generated)
        asar=0.3,         # Asar assembly syntax validation
        entity=0.2,       # Entity coverage (known addresses)
        length=0.1        # Output length/structure
    ),
    min_output_length=50,
    ideal_output_length=500,
    max_output_length=5000
)

scorer = QualityScorer(config=config)
```

### Score Individual Samples

```python
from afs.generators.base import TrainingSample

sample = TrainingSample(
    instruction="What does LDA do?",
    output="LDA loads the accumulator from memory.",
    domain="asm"
)

score = scorer.score(sample)
print(f"Overall: {score.overall:.2f}")
print(f"ELECTRA: {score.electra_score:.2f}")
print(f"Asar: {score.asar_valid} (syntax valid)")
print(f"Entity coverage: {score.entity_coverage:.2f}")
```

### Score Full Dataset

```bash
# Score all samples
afs discriminator score \
  --model models/electra \
  --input training_data/raw.jsonl \
  --output training_data/scored.jsonl

# Filter by score (keep >= 0.6)
afs discriminator filter \
  --input training_data/scored.jsonl \
  --output training_data/filtered.jsonl \
  --min-score 0.6
```

Output includes `quality_score` field (0-1 range).

### Scoring Profiles

Different scoring profiles for different domains:

```bash
# Generic (balanced)
afs discriminator score \
  --profile generic \
  --input training_data/raw.jsonl

# Assembly (favor ELECTRA + asar)
afs discriminator score \
  --profile asm \
  --input training_data/asm.jsonl

# Dialogue (favor length + entity)
afs discriminator score \
  --profile dialogue \
  --input training_data/dialogue.jsonl
```

## Rehearsal Buffers

Rehearsal buffers prevent **catastrophic forgetting** when training new model versions.

### Problem

Each training iteration (v1→v2→v3) loses knowledge from previous versions:

```
v1: Knows assembly, history, architecture
v2: Trained on new data → Loses assembly knowledge
v3: Trained on new data → Loses more assembly
```

### Solution

Mix high-quality samples from previous versions into new training data.

### Build Buffer

Extract top-quality samples from previous training:

```bash
afs training rehearsal-build \
  --input models/majora_v1_training.jsonl \
  --version v1 \
  --output ~/.context/training/rehearsal/majora_v1.jsonl \
  --top-ratio 0.3 \
  --min-score 0.7
```

This:
1. Loads all v1 training samples
2. Sorts by quality score
3. Keeps top 30% (highest quality)
4. Saves to rehearsal buffer

### Mix with New Data

Combine rehearsal buffer with new data:

```bash
afs training rehearsal-mix \
  --buffer ~/.context/training/rehearsal/majora_v1.jsonl \
  --new-data training_data/majora_v2_new.jsonl \
  --rehearsal-ratio 0.3 \
  --output training_data/majora_v2_training.jsonl \
  --shuffle \
  --seed 42
```

This creates training data that's:
- 30% old high-quality samples (from v1)
- 70% new samples (for learning)

### Programmatic Usage

```python
from afs.training import RehearsalBuffer, RehearsalBufferConfig

# Configure
config = RehearsalBufferConfig(
    quality_threshold=0.7,    # Only keep scores >= 0.7
    top_ratio=0.3,            # Keep top 30%
    enable_diversity=True,    # Diversity sampling
    max_per_domain=1000,      # Max samples per domain
    track_provenance=True     # Track version metadata
)

# Build buffer
buffer = RehearsalBuffer(config=config)
buffer.load_from_jsonl(Path("models/v1_training.jsonl"), version="v1")
buffer.select_top_samples(ratio=0.3)

# Mix with new data
new_samples = buffer.load_jsonl(Path("models/v2_new.jsonl"))
mixed = buffer.merge_with_new_data(
    new_samples,
    rehearsal_ratio=0.3,
    shuffle=True,
    seed=42
)

# Save for next version
buffer.save(Path("~/.context/training/rehearsal/v1.jsonl"))
```

### Best Practices

1. **Keep 20-40%** of previous version by quality score
2. **Rehearsal ratio 0.2-0.4** (20-40% old, 60-80% new)
3. **Track provenance** for debugging version-specific issues
4. **Diversity sample** across domains to avoid overfitting
5. **Save buffers** after each major version

## Dataset Rebalancing

Balance training data across domains and sources.

### Check Domain Distribution

Before training, understand your data:

```bash
afs training analyze \
  --input training_data/mixed.jsonl
```

Output:
```
Domain Distribution:
  oracle: 2,500 samples (50%)
  memory: 1,500 samples (30%)
  history: 1,000 samples (20%)
```

### Rebalance by Domain Weight

```bash
afs training rebalance \
  --input training_data/oracle.jsonl \
  --input training_data/memory.jsonl \
  --input training_data/history.jsonl \
  --output training_data/balanced.jsonl \
  --weight oracle=0.4 \
  --weight memory=0.4 \
  --weight history=0.2
```

This:
1. Loads all inputs
2. Samples from each to match weights
3. Shuffles and combines

### With Quality Filtering

```bash
afs training rebalance \
  --input training_data/oracle_scored.jsonl \
  --input training_data/memory_scored.jsonl \
  --output training_data/balanced_quality.jsonl \
  --weight oracle=0.5 \
  --weight memory=0.5 \
  --min-quality-score 0.6
```

Only includes samples with quality >= 0.6.

### Programmatic Usage

```python
from afs.training import rebalance_datasets

samples = rebalance_datasets(
    inputs=[
        Path("training_data/oracle.jsonl"),
        Path("training_data/memory.jsonl")
    ],
    weights={"oracle": 0.6, "memory": 0.4},
    min_quality_score=0.5,
    shuffle=True,
    seed=42
)

with open("training_data/rebalanced.jsonl", "w") as f:
    for sample in samples:
        f.write(sample.to_json() + "\n")
```

## Splitting Data

Create train/validation/test splits before training.

### Basic Split

```bash
afs training prepare \
  --input training_data/balanced.jsonl \
  --output training_data/splits \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

Creates:
```
training_data/splits/
├── train.jsonl   (80% of data)
├── val.jsonl     (10% of data)
└── test.jsonl    (10% of data)
```

### Stratified Split

Keep domain distribution across splits:

```bash
afs training prepare \
  --input training_data/balanced.jsonl \
  --output training_data/splits \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --stratify-by domain \
  --seed 42
```

Each split maintains the original domain ratios.

### Programmatic Usage

```python
from afs.training import split_dataset

splits = split_dataset(
    input_path=Path("training_data/balanced.jsonl"),
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    stratify_by="domain",
    seed=42
)

# Write splits
for split_name, samples in splits.items():
    path = Path(f"training_data/splits/{split_name}.jsonl")
    with open(path, "w") as f:
        for sample in samples:
            f.write(sample.to_json() + "\n")
```

## Format Conversion

Convert JSONL to framework-specific formats.

### Alpaca Format

For unsloth, MLX, and other frameworks:

```bash
afs training convert \
  --input training_data/splits/train.jsonl \
  --output training_data/splits/train_alpaca.jsonl \
  --format alpaca
```

Output format:
```json
{
  "instruction": "Question or task",
  "input": "Additional context (optional)",
  "output": "Expected response"
}
```

### ShareGPT Format

For vLLM and chat-based frameworks:

```bash
afs training convert \
  --input training_data/splits/train.jsonl \
  --output training_data/splits/train_sharegpt.jsonl \
  --format sharegpt
```

Output format:
```json
{
  "conversations": [
    {"from": "user", "value": "Question"},
    {"from": "assistant", "value": "Response"}
  ]
}
```

### OpenAI Format

For OpenAI fine-tuning API:

```bash
afs training convert \
  --input training_data/splits/train.jsonl \
  --output training_data/splits/train_openai.jsonl \
  --format openai
```

Output format:
```json
{
  "messages": [
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Response"}
  ]
}
```

## Training Models

Train models on prepared datasets.

### Local Training

#### MacBook Pro M5 (32GB RAM)

With 32GB unified memory, local LoRA fine-tuning is viable for smaller models:

**Recommended configurations:**

```bash
# 3B models - Fast iteration (uses ~12GB RAM)
python3 scripts/train_majora_v1.py \
  --model-name majora-v1 \
  --base-model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit \
  --data training_data/splits/train.jsonl \
  --output models/majora-v1-lora \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --gradient-accumulation-steps 4

# 7B models - Optimal balance (uses ~20GB RAM)
python3 scripts/train_majora_v1.py \
  --model-name majora-v1 \
  --base-model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \
  --data training_data/splits/train.jsonl \
  --output models/majora-v1-lora \
  --epochs 3 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --gradient-accumulation-steps 8
```

**Hardware considerations:**
- 3B models: ~1-2 hours for 1000 samples, comfortable with other apps running
- 7B models: ~3-4 hours for 1000 samples, close other applications
- 13B+ models: Still recommend vast.ai/cloud GPUs

This:
1. Loads base model in 4-bit quantization
2. Adds LoRA adapters
3. Trains on provided data
4. Saves LoRA weights to `models/majora-v1-lora`

#### Legacy Macs (16GB RAM)

Local training not recommended - use cloud training instead.

### Cloud Training (vast.ai)

Train on cloud GPUs for faster iteration:

```bash
python3 scripts/vastai_setup.py \
  --model majora \
  --budget 50
```

Features:
- Automatic GPU selection
- Cost management ($50 budget)
- Background training
- Automated checkpoint saving
- Google Drive backup

Check vast.ai for:
- GPU availability and pricing
- Available training jobs
- Current instances and costs

### Monitor Training

```bash
# Check local training
tail -f models/majora-v1-lora/training.log

# Check cloud training
python3 scripts/vastai_setup.py --monitor
```

### Evaluate During Training

Use validation split to track progress:

```bash
afs training evaluate \
  --model models/majora-v1-lora \
  --data training_data/splits/val.jsonl \
  --output models/majora-v1-lora/val_metrics.json
```

## Experiment Tracking

Track training runs for reproducibility and analysis.

### Register Experiment

```bash
afs training registry-create \
  --name majora-v2-rehearsal \
  --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \
  --framework unsloth \
  --dataset training_data/majora_v2_mixed.jsonl \
  --tag majora \
  --tag oracle \
  --tag rehearsal \
  --notes "v2 training with 30% rehearsal buffer from v1"
```

### View Registry

```bash
afs training registry-list
afs training registry-info majora-v2-rehearsal
```

### Metadata Tracking

Saved experiments include:
- Model name and version
- Base model used
- Framework (unsloth, MLX, etc.)
- Dataset and splits
- Hyperparameters
- Training metrics
- Timestamp and duration
- Tags for organization
- Notes and observations

## Troubleshooting

### Low Quality Scores

**Problem**: Most samples score below 0.5

**Solutions**:
1. Lower quality threshold:
   ```bash
   afs discriminator filter \
     --min-score 0.3
   ```

2. Check ELECTRA model is trained on relevant domain

3. Use domain-specific profile:
   ```bash
   afs discriminator score --profile asm
   ```

4. Disable expensive scorers:
   ```bash
   afs discriminator score --no-electra
   ```

### Catastrophic Forgetting

**Problem**: New model version loses knowledge from previous versions

**Solutions**:
1. Build and use rehearsal buffer (prevents 80% forgetting)
2. Increase rehearsal ratio: `--rehearsal-ratio 0.4`
3. Mix more diverse domains
4. Train for more epochs: 5-10 instead of 3
5. Use larger base model (13B vs 7B)

### Out of Memory (OOM)

**Problem**: Training crashes with memory errors

**Solutions for MacBook Pro M5 (32GB):**

1. **Model size matters:**
   - 3B models: Should never OOM (~12GB used)
   - 7B models: Reduce batch size to 2, gradient accumulation to 8
   - 13B+: Use vast.ai instead

2. Close background applications:
   ```bash
   # Check memory usage
   top -l 1 | grep PhysMem
   # Should have ~25-28GB free before training
   ```

3. Reduce batch size:
   ```bash
   --batch-size 2  # For 7B models
   --batch-size 1  # Last resort
   ```

4. Increase gradient accumulation:
   ```bash
   --gradient-accumulation-steps 8  # Maintains effective batch size
   ```

**Solutions for Cloud/GPU Training:**

1. Enable gradient checkpointing (in training script)

2. Use smaller model (3B vs 7B)

3. Use quantization (4-bit LoRA)

4. Choose GPU with more VRAM (A100 40GB vs RTX 3090 24GB)

### Dataset Imbalance

**Problem**: One domain dominates training (e.g., 80% Oracle, 20% other)

**Solutions**:
1. Use rebalancing:
   ```bash
   afs training rebalance \
     --weight oracle=0.3 \
     --weight memory=0.7
   ```

2. Enable diversity sampling in rehearsal buffer

3. Set per-domain caps:
   ```bash
   --max-per-domain 1000
   ```

4. Oversample minority domains

### Quality Scoring is Slow

**Problem**: Scoring takes hours for large datasets

**Solutions**:
1. Disable expensive components:
   ```bash
   afs discriminator score --no-electra
   ```

2. Use GPU for ELECTRA:
   ```bash
   --device cuda
   ```

3. Increase batch size:
   ```bash
   --batch-size 256
   ```

4. Run in parallel on multiple datasets

5. Cache results to avoid re-scoring

### Model Doesn't Improve

**Problem**: Loss plateaus, accuracy doesn't increase

**Solutions**:
1. Check data quality - are samples correct and diverse?
2. Increase learning rate: `--learning-rate 5e-4`
3. Train for more epochs: 5-10 instead of 3
4. Use better base model (13B instead of 7B)
5. Add more training data (aim for 10k+ samples)
6. Enable curriculum learning (start with easier samples)

### Can't Find Training Data

**Problem**: Export commands say "No data found"

**Solutions**:
1. Check source paths exist:
   ```bash
   ls ~/.context/memory
   ls ~/.context/history
   ```

2. Verify AFS is initialized:
   ```bash
   afs status
   ```

3. Run discovery:
   ```bash
   afs context discover --path ~/src
   ```

4. Use absolute paths for custom sources:
   ```bash
   --memory-root /Users/scawful/.context/memory
   ```

### JSONL File Corruption

**Problem**: Training fails with JSON parsing errors

**Solutions**:
1. Validate JSONL:
   ```bash
   python3 -c "import json; [json.loads(l) for l in open('data.jsonl')]"
   ```

2. Fix with jq:
   ```bash
   cat data.jsonl | jq -c '.' > data_fixed.jsonl
   ```

3. Check for encoding issues:
   ```bash
   file data.jsonl
   ```

4. Check for null bytes:
   ```bash
   grep -a $'\x00' data.jsonl
   ```

## Complete Training Workflow

Here's a complete example training a new model version:

```bash
# 1. Export data
afs training memory-export --output training_data/memory.jsonl
afs training history-export --output training_data/history.jsonl

# 2. Generate from Oracle
python3 -m afs.oracle.training_generator \
  --oracle-path ~/src/hobby/oracle-of-secrets \
  --output training_data/oracle.jsonl

# 3. Quality score
for dataset in training_data/{memory,history,oracle}.jsonl; do
  afs discriminator score \
    --model models/electra \
    --input "$dataset" \
    --output "${dataset%.jsonl}_scored.jsonl"
done

# 4. Build rehearsal buffer from v1
afs training rehearsal-build \
  --input models/majora_v1_training.jsonl \
  --version v1 \
  --output ~/.context/training/rehearsal/v1.jsonl \
  --top-ratio 0.3

# 5. Rebalance datasets
afs training rebalance \
  --input training_data/oracle_scored.jsonl \
  --input training_data/memory_scored.jsonl \
  --input training_data/history_scored.jsonl \
  --input ~/.context/training/rehearsal/v1.jsonl \
  --output training_data/majora_v2_mixed.jsonl \
  --weight oracle=0.35 \
  --weight memory=0.25 \
  --weight history=0.20 \
  --weight rehearsal=0.20

# 6. Split into train/val/test
afs training prepare \
  --input training_data/majora_v2_mixed.jsonl \
  --output training_data/majora_v2_splits \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --stratify-by domain \
  --seed 42

# 7. Convert to training format
afs training convert \
  --input training_data/majora_v2_splits/train.jsonl \
  --output training_data/majora_v2_splits/train_alpaca.jsonl \
  --format alpaca

# 8. Train locally or on vast.ai
python3 scripts/train_majora_v2.py \
  --data training_data/majora_v2_splits/train.jsonl \
  --output models/majora-v2-lora

# 9. Register experiment
afs training registry-create \
  --name majora-v2-with-rehearsal \
  --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \
  --dataset training_data/majora_v2_mixed.jsonl \
  --tag majora \
  --tag oracle \
  --notes "v2 with 20% rehearsal buffer"

# 10. Evaluate
python3 scripts/compare_models.py --models majora-v1 majora-v2
```

## See Also

- [Architecture & Modules](architecture.md)
- [Evaluation Guide](evaluation.md)
- [Deployment Guide](deployment.md)
- [API Reference](api.md)

---

**Last Updated:** January 2026
