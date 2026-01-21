# API Reference

Complete Python and CLI API reference for AFS training system.

## Table of Contents

1. [Python API](#python-api)
2. [CLI Commands](#cli-commands)
3. [Data Models](#data-models)
4. [Configuration](#configuration)
5. [Examples](#examples)

## Python API

### Training Module

#### RehearsalBuffer

Prevent catastrophic forgetting with rehearsal buffers.

```python
from afs.training import RehearsalBuffer, RehearsalBufferConfig
from pathlib import Path

# Create buffer with configuration
config = RehearsalBufferConfig(
    quality_threshold=0.7,      # Minimum quality score
    top_ratio=0.3,              # Keep top 30% of samples
    enable_diversity=True,      # Enable diversity sampling
    max_per_domain=1000,        # Max samples per domain
    track_provenance=True       # Track version metadata
)

buffer = RehearsalBuffer(config=config)

# Load training data
buffer.load_from_jsonl(
    Path("models/v1_training.jsonl"),
    version="v1"
)

# Select high-quality samples
buffer.select_top_samples(ratio=0.3)

# Mix with new data
new_samples = load_jsonl(Path("models/v2_new.jsonl"))
mixed = buffer.merge_with_new_data(
    new_samples,
    rehearsal_ratio=0.3,
    shuffle=True,
    seed=42
)

# Save for next version
buffer.save(Path("~/.context/training/rehearsal/v1.jsonl"))
```

#### QualityScorer

Score training samples for quality.

```python
from afs.training.scoring import QualityScorer, ScoringConfig, ScoringWeights
from afs.generators.base import TrainingSample

# Configure scorer
config = ScoringConfig(
    weights=ScoringWeights(
        electra=0.4,        # ELECTRA discriminator
        asar=0.3,           # ASM syntax validation
        entity=0.2,         # Entity coverage
        length=0.1          # Length heuristics
    ),
    min_output_length=50,
    ideal_output_length=500,
    max_output_length=5000
)

scorer = QualityScorer(config=config)

# Score individual sample
sample = TrainingSample(
    instruction="What does LDA do?",
    output="LDA loads the accumulator from memory.",
    domain="asm"
)

score = scorer.score(sample)
print(f"Overall: {score.overall:.2f}")
print(f"ELECTRA: {score.electra_score:.2f}")
print(f"Asar: {score.asar_valid}")
print(f"Entity coverage: {score.entity_coverage:.2f}")
```

#### split_dataset

Create train/val/test splits.

```python
from afs.training import split_dataset
from pathlib import Path

splits = split_dataset(
    input_path=Path("training_data/balanced.jsonl"),
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    stratify_by="domain",      # Maintain domain distribution
    seed=42
)

# Access splits
train_samples = splits["train"]
val_samples = splits["val"]
test_samples = splits["test"]

# Write to files
for split_name, samples in splits.items():
    with open(f"training_data/{split_name}.jsonl", "w") as f:
        for sample in samples:
            f.write(sample.to_json() + "\n")
```

#### rebalance_datasets

Balance training data across domains.

```python
from afs.training import rebalance_datasets
from pathlib import Path

samples = rebalance_datasets(
    inputs=[
        Path("training_data/oracle.jsonl"),
        Path("training_data/memory.jsonl"),
        Path("training_data/history.jsonl")
    ],
    weights={
        "oracle": 0.4,
        "memory": 0.3,
        "history": 0.3
    },
    min_quality_score=0.5,
    shuffle=True,
    seed=42
)

# Write rebalanced data
with open("training_data/rebalanced.jsonl", "w") as f:
    for sample in samples:
        f.write(sample.to_json() + "\n")
```

#### Format Converters

Convert to different training formats.

```python
from afs.training.converters import (
    AlpacaConverter,
    ShareGPTConverter,
    OpenAIConverter
)
from afs.generators.base import TrainingSample

sample = TrainingSample(
    instruction="Write a function",
    output="def my_function(): pass",
    domain="code"
)

# Alpaca format
alpaca_conv = AlpacaConverter()
alpaca_data = alpaca_conv.convert(sample)
# {"instruction": "...", "input": "", "output": "..."}

# ShareGPT format
sharegpt_conv = ShareGPTConverter()
sharegpt_data = sharegpt_conv.convert(sample)
# {"conversations": [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}]}

# OpenAI format
openai_conv = OpenAIConverter()
openai_data = openai_conv.convert(sample)
# {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Generators

#### OracleTrainingGenerator

Generate training data from Oracle of Secrets codebase.

```python
from afs.oracle.training_generator import OracleTrainingGenerator
from pathlib import Path
import json

generator = OracleTrainingGenerator(
    oracle_path=Path("~/src/hobby/oracle-of-secrets")
)

# Generate samples
with open("training_data/oracle.jsonl", "w") as f:
    for sample in generator.generate():
        f.write(json.dumps(sample.to_dict()) + "\n")

# Or use programmatically
samples = list(generator.generate())
high_quality = [s for s in samples if s.quality_score > 0.6]
print(f"Generated {len(samples)} total, {len(high_quality)} high-quality")
```

#### Custom Generator

Create custom training data generator.

```python
from afs.generators.base import BaseGenerator, TrainingSample
from pathlib import Path
from typing import Iterator

class MyDataGenerator(BaseGenerator):
    def __init__(self, source_path: Path):
        super().__init__(name="my_data", domain="custom")
        self.source_path = source_path

    def generate(self) -> Iterator[TrainingSample]:
        """Generate training samples from custom source."""
        for file in self.source_path.rglob("*.txt"):
            content = file.read_text()

            yield TrainingSample(
                instruction="Create question from content",
                output="Extract answer from content",
                thinking="Optional reasoning",
                domain="custom",
                source=str(file.relative_to(self.source_path)),
                _metadata={"file_type": "txt"}
            )

# Use generator
generator = MyDataGenerator(Path("~/my_data"))
for sample in generator.generate():
    print(sample.to_json())
```

### Discriminator

#### Quality Scoring

```python
from afs.discriminator import score_jsonl, filter_jsonl
from afs.training.scoring import ScoringConfig
from pathlib import Path

config = ScoringConfig.from_profile("asm")

# Score dataset
score_jsonl(
    input_path=Path("training_data/raw.jsonl"),
    output_path=Path("training_data/scored.jsonl"),
    config=config,
    device="cuda"  # Use GPU for speed
)

# Filter by quality score
filter_jsonl(
    input_path=Path("training_data/scored.jsonl"),
    output_path=Path("training_data/filtered.jsonl"),
    rejected_path=Path("training_data/rejected.jsonl"),
    min_score=0.6
)
```

### Knowledge Base

```python
from afs.knowledge import AddressDatabase

db = AddressDatabase()

# Query single address
player_health = db.query_address("player_health")
print(f"${player_health.address:06X} - {player_health.description}")

# Query address range
objects = db.query_range(0x7E0000, 0x7E1000)
for addr in objects:
    print(f"${addr.address:06X}: {addr.name}")

# Export as training data
training_samples = db.export_training_samples()
for sample in training_samples:
    print(sample.to_json())
```

### Tokenizer

```python
from afs.tokenizer import ASMTokenizer

tokenizer = ASMTokenizer()

# Tokenize code
code = "LDA #$00 ; Load zero"
tokens = tokenizer.tokenize(code)
print(tokens)  # ['LDA', '#$00', '']

# Encode/decode
encoded = tokenizer.encode(code)
decoded = tokenizer.decode(encoded)

# Works with blocks
asm_block = """
  LDA #$00
  STA $7E0000
  BRA done
"""
block_tokens = tokenizer.tokenize(asm_block)
```

### Deployment

```python
from afs.deployment import convert_lora_to_gguf, LMStudioClient

# Convert model
convert_lora_to_gguf(
    lora_path="models/majora-v1-lora",
    output_path="~/models/gguf/majora.gguf",
    base_model="unsloth/Qwen2.5-Coder-7B-Instruct",
    quantization="Q4_K_M",
    device="cuda"
)

# Query model
client = LMStudioClient(port=8000)
response = client.chat(
    prompt="What does LDA do?",
    temperature=0.7,
    max_tokens=500
)
print(response)
```

## CLI Commands

### Training Commands

#### Export Memory

```bash
afs training memory-export \
  --output training_data/memory.jsonl \
  --memory-root ~/.context/memory \
  --limit 1000
```

#### Export History

```bash
afs training history-export \
  --output training_data/history.jsonl \
  --history-root ~/.context/history
```

#### Score Quality

```bash
afs discriminator score \
  --model models/electra \
  --input training_data/raw.jsonl \
  --output training_data/scored.jsonl \
  --batch-size 32 \
  --device cuda
```

#### Filter by Quality

```bash
afs discriminator filter \
  --input training_data/scored.jsonl \
  --output training_data/filtered.jsonl \
  --rejected training_data/rejected.jsonl \
  --min-score 0.6
```

#### Build Rehearsal Buffer

```bash
afs training rehearsal-build \
  --input models/v1_training.jsonl \
  --version v1 \
  --output ~/.context/training/rehearsal/v1.jsonl \
  --top-ratio 0.3 \
  --min-score 0.7
```

#### Mix Rehearsal Buffer

```bash
afs training rehearsal-mix \
  --buffer ~/.context/training/rehearsal/v1.jsonl \
  --new-data training_data/v2_new.jsonl \
  --rehearsal-ratio 0.3 \
  --output training_data/v2_training.jsonl \
  --shuffle \
  --seed 42
```

#### Rebalance Datasets

```bash
afs training rebalance \
  --input training_data/oracle.jsonl \
  --input training_data/memory.jsonl \
  --input training_data/history.jsonl \
  --output training_data/balanced.jsonl \
  --weight oracle=0.4 \
  --weight memory=0.3 \
  --weight history=0.3 \
  --min-quality-score 0.5
```

#### Prepare Dataset

```bash
afs training prepare \
  --input training_data/balanced.jsonl \
  --output training_data/splits \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --stratify-by domain \
  --seed 42
```

#### Convert Format

```bash
afs training convert \
  --input training_data/splits/train.jsonl \
  --output training_data/splits/train_alpaca.jsonl \
  --format alpaca
```

Formats: `alpaca`, `sharegpt`, `openai`

#### Registry Commands

```bash
# Create experiment
afs training registry-create \
  --name majora-v2-baseline \
  --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit \
  --framework unsloth \
  --dataset training_data/majora_v2_mixed.jsonl \
  --tag majora \
  --tag oracle \
  --notes "First v2 training"

# List experiments
afs training registry-list

# Get details
afs training registry-info majora-v2-baseline
```

### Context Commands

#### Initialize Context

```bash
afs context init \
  --path ~/.context \
  --context-name default
```

#### Validate Context

```bash
afs context validate \
  --path ~/src
```

#### Discover Files

```bash
afs context discover \
  --path ~/src \
  --json
```

#### Ensure Structure

```bash
afs context ensure-all \
  --path ~/src
```

### Service Commands

#### Start Service

```bash
afs services start memory-export
afs services start context-warm
```

#### Stop Service

```bash
afs services stop memory-export
```

#### List Services

```bash
afs services list
```

#### Restart Service

```bash
AFS_CONTEXT_WARM_INTERVAL=3600 afs services restart context-warm
```

### Agent Commands

#### List Agents

```bash
afs agents list
```

#### Run Agent

```bash
afs agents run context-audit -- --path ~/src --output audit.json
afs agents run researcher -- input_dir output_file
afs agents run scribe-draft -- --prompt "Draft changelog"
```

### Embedding Commands

#### Index Documents

```bash
afs embeddings index \
  --project afs \
  --source ~/src/lab/afs/docs \
  --provider none
```

#### Search

```bash
afs embeddings search \
  --project afs \
  --query "context root" \
  --json
```

#### Evaluate

```bash
afs embeddings eval \
  --project afs \
  --query-file examples/embedding_eval.jsonl \
  --provider ollama \
  --model nomic-embed-text \
  --json
```

## Data Models

### TrainingSample

Standard training data format.

```python
from afs.generators.base import TrainingSample

sample = TrainingSample(
    instruction="Question or task",
    output="Expected response",
    input="Optional additional context",
    thinking="Optional reasoning process",
    domain="category_name",
    source="identifier or path",
    quality_score=0.85,
    _metadata={
        "language": "python",
        "difficulty": "medium"
    }
)

# Convert to JSON
json_str = sample.to_json()

# Convert to dict
data_dict = sample.to_dict()

# Convert to Alpaca format
alpaca = sample.to_alpaca()
```

### QualityScore

Quality scoring result.

```python
class QualityScore:
    overall: float              # 0-1 composite score
    electra_score: float        # ELECTRA discriminator
    asar_valid: bool            # Asar syntax validity
    entity_coverage: float      # Known entity coverage
    length_score: float         # Length heuristic
    components: dict            # Individual component scores
```

### RehearsalBufferConfig

Configuration for rehearsal buffers.

```python
class RehearsalBufferConfig:
    quality_threshold: float    # Min quality score (0-1)
    top_ratio: float            # Ratio of top samples to keep (0-1)
    enable_diversity: bool      # Enable diversity sampling
    max_per_domain: int | None  # Max samples per domain
    track_provenance: bool      # Track version metadata
    shuffle: bool               # Shuffle samples
    seed: int | None            # Random seed
```

## Configuration

### Global Config

Located at `~/.config/afs/config.toml`:

```toml
[general]
discovery_ignore = ["legacy", "archive", "archives"]
plugin_dirs = ["~/.config/afs/plugins", "~/.afs/plugins"]
enabled_plugins = []

[training]
default_batch_size = 32
default_learning_rate = 1e-5
default_epochs = 3

[discriminator]
model_path = "models/electra"
cache_dir = "~/.cache/afs/discriminator"

[orchestration]
port = 8000
workers = 4

[embeddings]
provider = "none"  # "none", "ollama", "hf"
model = "nomic-embed-text"
cache_dir = "~/.cache/afs/embeddings"
```

### Project Config

Local `afs.toml` in project directory:

```toml
[project]
name = "afs"
context_root = "~/.context"
workspace_name = "src"

[training]
batch_size = 64
learning_rate = 5e-5
epochs = 5

[evaluation]
sample_size = 100
timeout = 30
models = ["majora", "nayru", "veran"]
```

## Examples

### Complete Training Workflow

```python
from pathlib import Path
from afs.oracle.training_generator import OracleTrainingGenerator
from afs.training import (
    RehearsalBuffer,
    split_dataset,
    rebalance_datasets,
    score_jsonl
)
from afs.training.scoring import ScoringConfig
from afs.training.converters import AlpacaConverter

# 1. Generate data from Oracle
oracle_gen = OracleTrainingGenerator(
    oracle_path=Path("~/src/hobby/oracle-of-secrets")
)
oracle_samples = list(oracle_gen.generate())
print(f"Generated {len(oracle_samples)} samples")

# 2. Quality score
config = ScoringConfig.from_profile("generic")
score_jsonl(
    input_path=Path("training_data/oracle.jsonl"),
    output_path=Path("training_data/oracle_scored.jsonl"),
    config=config
)

# 3. Build rehearsal buffer
buffer = RehearsalBuffer()
buffer.load_from_jsonl(Path("models/v1_training.jsonl"), version="v1")
buffer.select_top_samples(ratio=0.3)

# 4. Rebalance with rehearsal
new_data = rebalance_datasets(
    inputs=[Path("training_data/oracle_scored.jsonl")],
    weights={"oracle": 1.0}
)
mixed = buffer.merge_with_new_data(new_data, rehearsal_ratio=0.3)

# 5. Split
splits = split_dataset(
    mixed,
    train_ratio=0.8,
    val_ratio=0.1,
    stratify_by="domain",
    seed=42
)

# 6. Convert format
converter = AlpacaConverter()
with open("training_data/train_alpaca.jsonl", "w") as f:
    for sample in splits["train"]:
        alpaca_data = converter.convert(sample)
        f.write(sample.to_json() + "\n")

print("Training data ready!")
```

### Evaluation Script

```python
import requests
import statistics
from pathlib import Path
import json

# Load evaluation suite
with open("~/.context/training/evals/unified_eval_suite.jsonl") as f:
    questions = [json.loads(line) for line in f]

# Query models
models = {
    "majora": "http://localhost:5000",
    "nayru": "http://localhost:5001",
    "veran": "http://localhost:5002"
}

results = {}

for model_name, endpoint in models.items():
    scores = []

    for question in questions[:10]:  # First 10 for quick test
        try:
            response = requests.post(
                f"{endpoint}/api/chat",
                json={"prompt": question["prompt"], "temperature": 0.7},
                timeout=30
            )
            result = response.json()

            # Simple scoring: 1 if model responded, 0 if error
            score = 1.0 if response.status_code == 200 else 0.0
            scores.append(score)
        except Exception as e:
            scores.append(0.0)

    results[model_name] = {
        "avg_score": statistics.mean(scores),
        "median_score": statistics.median(scores),
        "success_rate": sum(scores) / len(scores)
    }

# Print results
for model, metrics in results.items():
    print(f"{model}: {metrics['avg_score']:.2f}")
```

---

**Last Updated:** January 2026
**Version:** 1.0
