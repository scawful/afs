# AFS Documentation Site

Welcome to the comprehensive documentation for the **AFS (Agentic File System) Training System**.

This documentation covers the complete machine learning training pipeline, from data preparation to model deployment.

## Documentation Structure

### Getting Started

- **[Main Documentation](index.md)** - Overview and quick start guide
- **[Training Guide](training.md)** - Step-by-step training pipeline walkthrough

### Core References

- **[Architecture](architecture.md)** - System design and module overview
- **[API Reference](api.md)** - Complete Python and CLI API
- **[Evaluation Guide](evaluation.md)** - Model evaluation and benchmarking
- **[Deployment Guide](deployment.md)** - GGUF conversion and serving

### Development

- **[Development Guide](development.md)** - Contributing and extending AFS

## Key Topics

### Training Pipeline

The complete end-to-end training system:

1. **Data Export** - Extract from memory, history, Oracle, etc.
2. **Quality Scoring** - ELECTRA-based discriminator + validation
3. **Rehearsal Buffers** - Prevent catastrophic forgetting
4. **Rebalancing** - Balance domains and sources
5. **Splitting** - Create train/val/test splits
6. **Format Conversion** - Convert to framework-specific formats
7. **Model Training** - Train on local machine or vast.ai
8. **Evaluation** - Comprehensive benchmark suite
9. **Deployment** - Convert to GGUF and deploy with LMStudio

### Model Architecture

Five specialized models for different tasks:

- **Majora** - Quest narrative and high-level design
- **Nayru** - 65816 assembly expertise
- **Veran** - Logic and debugging
- **Farore** - Planning and architecture
- **Din** - Creative generation and synthesis

### System Components

**Core Infrastructure:**
- Context management (hierarchical file system)
- Plugin system for extensibility
- Service orchestration (background agents)
- Configuration management

**Domain Capabilities:**
- Training data generators
- Quality discriminators
- Custom tokenizers
- Knowledge bases (address tables)
- Evaluation suites

## Quick Reference

### Installation

```bash
cd ~/src/lab/afs
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Training Data Export

```bash
# From various sources
afs training memory-export --output training_data/memory.jsonl
afs training history-export --output training_data/history.jsonl

# From Oracle codebase
python3 -m afs.oracle.training_generator \
  --oracle-path ~/src/hobby/oracle-of-secrets \
  --output training_data/oracle.jsonl
```

### Quality Filtering

```bash
# Score and filter
afs discriminator score \
  --model models/electra \
  --input training_data/raw.jsonl \
  --output training_data/scored.jsonl

afs discriminator filter \
  --input training_data/scored.jsonl \
  --output training_data/filtered.jsonl \
  --min-score 0.6
```

### Complete Workflow

```bash
# 1. Rehearsal buffer (prevent forgetting)
afs training rehearsal-build \
  --input models/v1_training.jsonl \
  --output ~/.context/training/rehearsal/v1.jsonl \
  --top-ratio 0.3

# 2. Rebalance datasets
afs training rebalance \
  --input training_data/oracle.jsonl \
  --input training_data/memory.jsonl \
  --output training_data/balanced.jsonl \
  --weight oracle=0.5 --weight memory=0.5

# 3. Create splits
afs training prepare \
  --input training_data/balanced.jsonl \
  --output training_data/splits \
  --train-ratio 0.8 --val-ratio 0.1

# 4. Train
python3 scripts/train_majora_v2.py \
  --data training_data/splits/train.jsonl \
  --output ~/models/adapters/afs/majora-v2-lora

# 5. Evaluate
python3 scripts/compare_models.py --models majora-v1 majora-v2

# 6. Deploy
python3 scripts/convert_to_gguf.py \
  --model ~/models/adapters/afs/majora-v2-lora \
  --output ~/models/gguf/majora-v2.gguf
```

## Common Tasks

### Generate Training Data

```python
from afs.oracle.training_generator import OracleTrainingGenerator
from pathlib import Path

generator = OracleTrainingGenerator(
    oracle_path=Path("~/src/hobby/oracle-of-secrets")
)

with open("training_data/oracle.jsonl", "w") as f:
    for sample in generator.generate():
        f.write(sample.to_json() + "\n")
```

### Quality Scoring

```python
from afs.training.scoring import QualityScorer, ScoringConfig

config = ScoringConfig.from_profile("asm")
scorer = QualityScorer(config=config)

sample = TrainingSample(
    instruction="What does LDA do?",
    output="LDA loads the accumulator.",
    domain="asm"
)

score = scorer.score(sample)
print(f"Quality: {score.overall:.2f}")
```

### Prevent Catastrophic Forgetting

```python
from afs.training import RehearsalBuffer

buffer = RehearsalBuffer()
buffer.load_from_jsonl(Path("v1_training.jsonl"), version="v1")
buffer.select_top_samples(ratio=0.3)

new_data = load_jsonl(Path("v2_new.jsonl"))
mixed = buffer.merge_with_new_data(new_data, rehearsal_ratio=0.3)
```

### Model Evaluation

```bash
# Deploy models
./scripts/deploy_to_lmstudio.sh

# Run evaluation
python3 scripts/compare_models.py

# View results
open ~/.context/training/evals/results/dashboard_*.html
```

## Directory Structure

```
~/src/lab/afs/
├── docs/                      # This documentation
│   ├── index.md              # Main documentation
│   ├── training.md           # Training guide
│   ├── evaluation.md         # Evaluation guide
│   ├── deployment.md         # Deployment guide
│   ├── architecture.md       # System architecture
│   ├── api.md               # API reference
│   ├── development.md       # Development guide
│   └── mkdocs.yml           # MkDocs configuration
│
├── src/afs/                  # Main package
│   ├── training/            # Training pipeline
│   ├── generators/          # Data generation
│   ├── discriminator/       # Quality filtering
│   ├── oracle/              # Oracle integration
│   ├── tokenizer/           # ASM tokenizer
│   ├── knowledge/           # Knowledge bases
│   ├── evaluation/          # Evaluation tools
│   └── cli/                 # Command-line interface
│
├── scripts/                  # Utility scripts
│   ├── train_majora_v*.py   # Training scripts
│   ├── compare_models.py    # Model evaluation
│   ├── convert_to_gguf.py   # GGUF conversion
│   └── deploy_to_lmstudio.sh # LMStudio setup
│
├── models/                   # Trained models (GGUF)
├── training_data/            # Exported datasets
├── benchmark_results/        # Evaluation results
└── mkdocs.yml               # MkDocs configuration
```

## Building Documentation

### Install MkDocs

```bash
pip install mkdocs mkdocs-material pymdown-extensions
```

### Build and Serve

```bash
# Build static site
mkdocs build

# Serve locally (http://localhost:8000)
mkdocs serve

# Build to specific directory
mkdocs build -d /var/www/html/afs-docs
```

### Deploy

Documentation can be deployed to:
- GitHub Pages (automatic)
- Any static hosting (AWS S3, etc.)
- Local server (nginx, Apache)

## Contributing to Documentation

### Adding New Pages

1. Create markdown file in `docs/` directory
2. Add to navigation in `mkdocs.yml`
3. Link from relevant index pages
4. Test with `mkdocs serve`

### Documentation Standards

- Use clear, concise language
- Include code examples for all features
- Add diagrams using Mermaid syntax
- Include troubleshooting sections
- Keep table of contents updated

### Example Page Template

```markdown
# Feature Name

Short description of the feature.

## Overview

Longer explanation and context.

## Quick Start

```bash
# Show how to use immediately
```

## Detailed Guide

Step-by-step instructions.

### Subsection

More details about specific aspect.

## Examples

```python
# Code examples
```

## Troubleshooting

**Problem:** Description

**Solutions:**
1. First solution
2. Second solution

## See Also

- [Related Guide](related.md)
```

## Navigation Tips

### Quick Links

- **Training from scratch**: See [Training Guide](training.md)
- **Troubleshooting issues**: See [Training Guide Troubleshooting](training.md#troubleshooting)
- **Model evaluation**: See [Evaluation Guide](evaluation.md)
- **Serving models**: See [Deployment Guide](deployment.md)
- **Python API**: See [API Reference](api.md)
- **System design**: See [Architecture](architecture.md)
- **Contributing**: See [Development Guide](development.md)

## Support

### Getting Help

1. **Search documentation** - Use the search feature to find relevant pages
2. **Check examples** - Look for code examples in the relevant guide
3. **Review API reference** - For specific function signatures and parameters
4. **Check troubleshooting** - Most guides have troubleshooting sections
5. **File an issue** - If problem not documented, file a GitHub issue

### Documentation Issues

Found a documentation problem? Help improve it:

1. Note the issue (typo, unclear section, missing example)
2. Check if it's already reported
3. Open a GitHub issue with:
   - Page location
   - Description of problem
   - Suggested fix if possible

## Version History

- **v1.0** (January 2026) - Initial comprehensive documentation

---

**Last Updated:** January 2026
**Maintainers:** AFS Development Team
