# AFS Training System Documentation

Welcome to the comprehensive documentation for the **AFS (Agentic File System) Training System** - a complete framework for training specialized AI models on domain-specific tasks.

## Overview

AFS provides end-to-end tooling for building, training, and deploying specialized models. The system is designed around five core expert models:

- **Majora** - Quest and narrative specialist
- **Nayru** - Assembly (65816) expert
- **Veran** - Logic and debugging specialist
- **Farore** - Planning and architecture expert
- **Din** - Creative and generation specialist

## Quick Start

### Installation

```bash
# Clone the repository
cd /Users/scawful/src/lab/afs

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .

# Verify installation
afs status
```

### First Training Run

```bash
# 1. Export training data
afs training memory-export --output training_data/memory.jsonl

# 2. Quality score
afs discriminator filter \
  --model models/electra \
  --input training_data/memory.jsonl \
  --output training_data/memory_scored.jsonl \
  --min-score 0.6

# 3. Split into train/val/test
afs training prepare \
  --input training_data/memory_scored.jsonl \
  --output training_data/splits \
  --train-ratio 0.8 \
  --val-ratio 0.1

# 4. Ready for training!
python3 scripts/train_majora_v1.py \
  --data training_data/splits/train.jsonl
```

## Documentation Sections

### Core Guides

- **[Training Guide](training.md)** - Complete training pipeline walkthrough
  - Data export and preparation
  - Quality scoring and filtering
  - Rehearsal buffers (preventing catastrophic forgetting)
  - Dataset rebalancing
  - Model training and experimentation

- **[Architecture & Modules](architecture.md)** - System design and components
  - Core AFS infrastructure
  - Domain capabilities (ALTTP/65816)
  - Training pipeline architecture
  - Model registry and experiment tracking

- **[Evaluation Suite](evaluation.md)** - Comprehensive evaluation framework
  - Unified evaluation suite (100+ questions)
  - Model comparison methodology
  - Meta-circular evaluation (models evaluating each other)
  - Screenshot-based evaluation
  - Benchmark suite

- **[Deployment Guide](deployment.md)** - Running models in production
  - GGUF conversion from LoRA adapters
  - LMStudio integration
  - Network inference setup
  - Model serving options

- **[API Reference](api.md)** - Python API and CLI reference
  - Core classes and functions
  - CLI command reference
  - Training modules
  - Evaluation tools

### Additional Resources

- **[Development Guide](development.md)** - Contributing to AFS
  - Setting up dev environment
  - Running tests
  - Code structure and conventions
  - Common workflows

## Key Concepts

### Training Data Pipeline

```
Raw Sources
    ↓
Export (memory, history, Oracle, etc.)
    ↓
Quality Scoring (ELECTRA discriminator)
    ↓
Rehearsal Buffer (prevent forgetting)
    ↓
Rebalancing (domain weights)
    ↓
Train/Val/Test Split
    ↓
Format Conversion (Alpaca, ShareGPT, etc.)
    ↓
Model Training
```

### Model Roles

| Model | Specialty | Best For |
|-------|-----------|----------|
| **Majora** | Quest narrative, high-level design | Architecture, story progression |
| **Nayru** | 65816 assembly, optimization | ASM generation/debugging |
| **Veran** | Rigorous logic, state management | Debugging, system design |
| **Farore** | Planning, multi-step reasoning | Architecture, planning |
| **Din** | Creative generation, synthesis | Novel problem-solving |

## System Architecture

AFS consists of two complementary tracks:

1. **Core Infrastructure**
   - Context management (`~/.context/`)
   - Plugin system
   - Service orchestration
   - Configuration management

2. **Domain Capabilities**
   - Training data generators
   - Quality discriminators
   - ASM tokenizers
   - Knowledge bases

## Common Tasks

### Generate Training Data

```bash
# From Oracle of Secrets
python3 -m afs.oracle.training_generator \
  --oracle-path ~/src/hobby/oracle-of-secrets \
  --output training_data/oracle.jsonl

# From memory system
afs training memory-export --output training_data/memory.jsonl

# From chat history
afs training history-export --output training_data/history.jsonl
```

### Quality Filtering

```bash
# Score with discriminator
afs discriminator score \
  --model models/electra \
  --input training_data/raw.jsonl \
  --output training_data/scored.jsonl

# Filter by score threshold
afs discriminator filter \
  --input training_data/scored.jsonl \
  --output training_data/filtered.jsonl \
  --min-score 0.6
```

### Prevent Forgetting

```bash
# Build rehearsal buffer from previous version
afs training rehearsal-build \
  --input models/v1_training.jsonl \
  --version v1 \
  --output ~/.context/training/rehearsal/v1.jsonl \
  --top-ratio 0.3

# Mix with new data (30% old, 70% new)
afs training rehearsal-mix \
  --buffer ~/.context/training/rehearsal/v1.jsonl \
  --new-data training_data/v2_new.jsonl \
  --rehearsal-ratio 0.3 \
  --output training_data/v2_training.jsonl
```

### Train Models

```bash
# Local training
python3 scripts/train_majora_v1.py \
  --data training_data/splits/train.jsonl \
  --output ~/models/adapters/afs/majora-v1-lora

# Cloud training (vast.ai)
python3 scripts/vastai_setup.py \
  --model majora \
  --budget 50
```

### Deploy & Run

```bash
# Convert to GGUF for LMStudio
python3 scripts/convert_to_gguf.py \
  --model ~/models/adapters/afs/majora-v1-lora \
  --output ~/models/gguf/majora.gguf

# Deploy to LMStudio
./scripts/deploy_to_lmstudio.sh

# Run evaluation
python3 scripts/compare_models.py \
  --models majora nayru veran
```

## Directory Structure

```
~/src/lab/afs/
├── src/afs/
│   ├── training/              # Training pipeline
│   ├── generators/            # Data generation
│   ├── discriminator/         # Quality filtering
│   ├── oracle/                # Oracle-of-Secrets integration
│   ├── tokenizer/             # Custom ASM tokenizer
│   ├── knowledge/             # ALTTP address tables
│   ├── evaluation/            # Evaluation tools
│   ├── cli/                   # Command-line interface
│   └── agents/                # Orchestrated agents
│
├── scripts/
│   ├── train_majora_v1.py     # Model training
│   ├── compare_models.py      # Model evaluation
│   ├── deploy_to_lmstudio.sh  # LMStudio setup
│   ├── vastai_setup.py        # Cloud training
│   └── gdrive_backup.py       # Backup management
│
├── models/                    # Trained models (GGUF)
├── training_data/             # Exported datasets (JSONL)
├── benchmark_results/         # Evaluation results
├── configs/                   # Model configurations
└── docs/                      # This documentation

~/.context/training/
├── rehearsal/                 # Rehearsal buffers
├── evals/                     # Evaluation suites
│   └── results/               # Evaluation results
└── exports/                   # Cached exports
```

## Troubleshooting

### Common Issues

**Q: "No training data found"**
- A: Check source paths exist: `ls ~/.context/memory`
- Run discovery: `afs context discover --path ~/src`
- Verify AFS initialization: `afs status`

**Q: "Low quality scores"**
- A: Lower threshold: `--min-quality-score 0.3`
- Check ELECTRA model is properly loaded
- Use domain-specific scoring profile: `--score-profile asm`

**Q: "CUDA out of memory"**
- A: Reduce batch size: `--batch-size 8`
- Enable gradient checkpointing
- Use smaller model (7B vs 13B)
- Choose GPU with more VRAM

**Q: "Model doesn't improve during training"**
- A: Check data quality - are samples correct and diverse?
- Increase learning rate: `--learning-rate 5e-4`
- Train for more epochs: 5-10 instead of 3
- Add more training data (aim for 10k+ samples)

See [Troubleshooting](training.md#troubleshooting) section in Training Guide for more solutions.

## Getting Help

### Documentation

- **[Training Guide](training.md)** - Step-by-step training walkthroughs
- **[Evaluation Guide](evaluation.md)** - Comprehensive evaluation methodology
- **[Deployment Guide](deployment.md)** - Model serving and inference
- **[API Reference](api.md)** - Complete Python and CLI API

### Command Help

```bash
afs --help                          # List all commands
afs training --help                 # Training commands
afs discriminator --help            # Quality filtering
afs context --help                  # Context management
afs agents --help                   # Agent orchestration
```

### Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_training.py -v

# Run with coverage
pytest --cov=src/afs
```

## Contributing

We welcome contributions! See [Development Guide](development.md) for:
- Setting up development environment
- Code standards and testing
- Pull request process
- Common development workflows

## Key Features

- **End-to-End Pipeline** - From raw data to trained models
- **Quality Control** - ELECTRA-based discriminator + syntax validation
- **Catastrophic Forgetting Prevention** - Rehearsal buffer system
- **Portable Datasets** - JSONL format works across machines and frameworks
- **Multi-Source Integration** - Combine data from memory, history, Oracle, etc.
- **Cloud Training** - vast.ai automation for parallel GPU training
- **Experiment Tracking** - Model registry for tracking training runs
- **Comprehensive Evaluation** - 100+ question benchmark suite
- **Meta-Circular Evaluation** - Models evaluating each other
- **Easy Deployment** - GGUF conversion and LMStudio integration

## Performance Tips

1. **Start small** - Test with 10% of data before full training
2. **Use rehearsal buffers** - Prevents catastrophic forgetting
3. **Quality matters** - Curated 1k samples beats raw 100k
4. **Diversity** - Mix domains to avoid overfitting
5. **Monitor metrics** - Track loss, validation accuracy, inference speed
6. **Version control** - Save experiment metadata with each training run

## License & Citation

Research-only. Not a product. See `/Users/scawful/src/lab/afs/AGENTS.md` for contribution guidelines and provenance requirements.

---

**Last Updated:** January 2026
**Version:** 1.0
**Documentation Index:** [Complete Guide](.)
