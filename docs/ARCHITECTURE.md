# AFS Architecture

> **See also:** `~/src/docs/ARCHITECTURE.md` for workspace-level architecture documentation.

## Overview

AFS (Agentic File System) is a framework for AI agent infrastructure with two complementary tracks:

1. **Core AFS** - Agent context management, orchestration, and services
2. **Domain Capabilities** - Specialized tooling for ALTTP/65816 assembly tasks

```
afs/
├── Core Infrastructure          Domain Capabilities
│   ├── manager.py              ├── generators/
│   ├── config.py               ├── discriminator/
│   ├── discovery.py            ├── tokenizer/
│   ├── orchestration.py        ├── training/
│   ├── plugins.py              └── knowledge/
│   ├── schema.py
│   └── services/
```

## Core AFS Infrastructure

### Context Management

The `.context/` directory structure for agent state:

```
.context/
├── scratchpad/          # Working memory (agent-writable)
│   ├── state.md         # Current agent state
│   ├── metacognition.json
│   └── goals.json
├── memory/              # Long-term constraints (read-only)
├── knowledge/           # Reference materials (read-only)
└── metadata.json        # Project metadata
```

**Key Classes:**
- `AFSManager` - Manages context roots, ensures directory structure
- `AFSValidator` - Validates context configurations
- `ContextRoot` - Represents a mounted context directory

### Configuration & Schema

- `AFSConfig` - Configuration loaded from `~/.config/afs/config.toml` and local `afs.toml`
- `DirectoryConfig` - Per-directory policy definitions
- `PolicyType` - Access policies (read-only, agent-writable, etc.)

### Plugins & Services

- Plugin discovery via entry points
- Service definitions for daemons (launchd/systemd adapters)
- Orchestrator for routing agent tasks

---

## Domain Capabilities (ALTTP/65816)

Specialized modules for working with SNES assembly code. These support training and using models that understand 65816 assembly.

### Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Data Flow                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Raw Assembly ──► generators/ ──► training/ ──► Trained Model  │
│                                                                 │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐              │
│  │ knowledge/ │   │ Augment    │   │ Converters │              │
│  │ (addresses)│   │ CoT Gen    │   │ (MLX, HF)  │              │
│  └────────────┘   │ Validation │   │ Registry   │              │
│                   └────────────┘   │ Splitter   │              │
│                                    └────────────┘              │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Two Training Paradigms:                                    ││
│  │                                                            ││
│  │ 1. DECODER (LLM fine-tuning)     2. ENCODER (pre-training) ││
│  │    - Qwen, DeepSeek, etc.           - BERT-style MLM       ││
│  │    - Instruction → Response         - Code understanding   ││
│  │    - Uses: converters/              - Uses: tokenizer/     ││
│  │                                       + asm_trainer.py     ││
│  └────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Module Descriptions

#### `generators/` - Training Data Generation

| Component | Purpose |
|-----------|---------|
| `base.py` | Base classes, I/O utilities, instruction cleaning |
| `asm_augment.py` | Augment training samples with paraphrases |
| `cot/` | Chain-of-Thought generation via LLMs (Gemini) |
| `data_cleaner.py` | Batch cleaning of malformed samples |
| `asar_validator.py` | Validate assembly syntax via asar |

#### `training/` - Model Training Utilities

| Component | Purpose |
|-----------|---------|
| `config.py` | Training configuration (LoRA, hyperparams) |
| `converters/` | Format data for frameworks (MLX, Alpaca, ChatML) |
| `splitter.py` | Train/val/test splitting with stratification |
| `registry.py` | Experiment tracking and A/B testing |
| `asm_trainer.py` | **Encoder training with ASM tokenizer** |

#### `tokenizer/` - Custom Assembly Tokenizer

| Component | Purpose |
|-----------|---------|
| `vocab.py` | Base vocabulary (opcodes, registers, directives) |
| `pretokenizer.py` | Regex-based semantic splitting |
| `asm_tokenizer.py` | HuggingFace-compatible tokenizer |

**Key Features:**
- Preserves semantic units (opcodes, addresses, indexed addressing)
- Trainable vocabulary expansion from corpus
- 5.6x compression vs character-level tokenization

#### `discriminator/` - Quality Filtering

| Component | Purpose |
|-----------|---------|
| `electra.py` | ELECTRA-based quality scoring |
| `fake_generators.py` | Generate synthetic errors for training |
| `data.py` | Dataset preparation |
| `filter.py` | Sample filtering by quality score |

#### `knowledge/` - Domain Knowledge

| Component | Purpose |
|-----------|---------|
| `alttp_addresses.py` | ALTTP RAM addresses, sprite tables, etc. |

---

## How to Use the ASM Tokenizer with Training

### For Encoder Pre-training (NEW)

Train a BERT-style model that understands assembly code:

```python
from afs.tokenizer import ASMTokenizer
from afs.training import ASMTrainer, ASMTrainerConfig

# Load or create tokenizer
tokenizer = ASMTokenizer()
tokenizer.train_on_corpus(assembly_texts, min_frequency=2)
tokenizer.save("models/asm-tokenizer")

# Train encoder
config = ASMTrainerConfig(
    hidden_size=256,
    num_layers=4,
    num_epochs=10,
)
trainer = ASMTrainer(tokenizer=tokenizer, config=config)
trainer.train(train_texts, val_texts)
```

**Use Cases:**
- Semantic search over assembly code
- Code similarity / clustering
- Pre-training foundation for downstream tasks

### For Decoder Fine-tuning (EXISTING)

The tokenizer could be used to pre-process assembly for LLM fine-tuning:

```python
from afs.tokenizer import ASMTokenizer
from afs.training import get_converter

# Tokenizer for analysis (not replacing LLM tokenizer)
asm_tokenizer = ASMTokenizer()

# Analyze token distribution in training data
for sample in training_data:
    tokens = asm_tokenizer.tokenize(sample["input"])
    # Check for unknown tokens, validate syntax, etc.

# Convert to LLM training format
converter = get_converter("mlx", include_cot=True)
converter.convert_dataset(input_path, output_path)
```

---

## Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                    AFS Integration Map                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Core AFS                    Domain Capabilities                │
│  ─────────                   ───────────────────                │
│                                                                 │
│  AFSManager ◄──────────────► knowledge/                        │
│  (context state)             (address tables in scratchpad)    │
│                                                                 │
│  Orchestrator ◄────────────► generators/                       │
│  (task routing)              (data generation agents)          │
│                                                                 │
│  Plugins ◄─────────────────► training/                         │
│  (capability discovery)      (model training plugins)          │
│                                                                 │
│  Services ◄────────────────► discriminator/                    │
│  (background daemons)        (quality scoring service)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## File Locations

| Category | Path | Description |
|----------|------|-------------|
| Core config | `~/.config/afs/config.toml` | Global AFS configuration |
| Project context | `.context/` | Per-project agent state |
| Trained models | `models/` | Saved tokenizers and models |
| Training data | `data/` | JSONL training datasets |
| Experiments | `~/.afs/model_registry.json` | Experiment tracking |
