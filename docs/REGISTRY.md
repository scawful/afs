# Model Registry and Version Management System

A comprehensive system for tracking trained models with metadata, versions, and lineage. Provides model cataloging, version management, deployment tracking, and complete training history.

## Features

### 1. Model Catalog
- JSON database of all trained models
- Semantic versioning (v1, v2, v2.1, etc.)
- Automatic version incrementing
- Model metadata and ownership tracking

### 2. Training Metadata
- Framework information (unsloth, mlx, huggingface, etc.)
- Base model used
- Training parameters (samples, epochs, batch size, learning rate)
- Training duration and cost tracking
- Hardware information

### 3. Evaluation Metrics
- Standard metrics: accuracy, F1, perplexity, BLEU
- ROUGE scores with breakdown
- Inference speed (tokens/sec)
- Memory usage and latency
- Custom metric support

### 4. File Management
- Track LoRA weights location
- Track GGUF quantized models
- Track full checkpoints
- Config file paths

### 5. Lineage Tracking
- Parent-child relationships for fine-tuning chains
- Training data dependencies
- Git commit tracking for reproducibility
- Complete data lineage across versions

### 6. Deployment Management
- Mark versions as deployed
- Track deployment dates
- Rollback to previous versions
- Automatic deprecation of old versions

## Installation

The registry is built into the AFS package. No additional installation needed.

```python
from afs.registry import ModelRegistry, LineageTracker
```

## Quick Start

### Register a Model

```python
from afs.registry import ModelRegistry

registry = ModelRegistry()

# Register a new version (auto-increments to v1)
version = registry.register_model(
    model_name="majora",
    base_model="Qwen2.5-Coder-7B",
    framework="unsloth",
    samples=223,
    epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    training_data=["oracle", "toolbench"],
    lora_path="/path/to/majora-v1-lora",
    gguf_path="/path/to/majora-v1-Q8_0.gguf",
    evaluation_scores={
        "accuracy": 0.85,
        "f1_score": 0.82,
        "inference_speed_tokens_per_sec": 120,
    },
    duration_hours=3.0,
    cost_usd=0.80,
    hardware="M2 Max",
    notes="Initial training",
)

print(f"Registered {version.model_name}:{version.version}")
```

### Get Model Information

```python
# Get latest version
latest = registry.get_latest("majora")
print(f"Latest: {latest.version}")
print(f"Accuracy: {latest.evaluation_scores.accuracy}")

# Get specific version
v1 = registry.get_version("majora", "v1")

# List all versions
versions = registry.list_versions("majora")
for v in versions:
    print(f"{v.version}: {v.status.value}")
```

### Compare Versions

```python
# Compare two versions side-by-side
diff = registry.compare_versions("majora", "v1", "v2")

for key, values in diff.items():
    print(f"{key}: v1={values['v1']}, v2={values['v2']}")

# Example output:
# accuracy: v1=0.85, v2=0.87
# f1_score: v1=0.82, v2=0.84
```

### Deploy and Rollback

```python
# Mark version as deployed
registry.set_deployed("majora", "v1", deployed=True)

# Later, if v2 is better:
registry.register_model(
    model_name="majora",
    version="v2",
    # ... other params
)

# Rollback marks old version as deprecated
registry.rollback("majora", "v2")

# Check deployment status
latest = registry.get_latest("majora")
print(f"Current deployment: {latest.version} (deployed: {latest.deployed})")
```

### Update Evaluation Scores

```python
# After running evaluation pipeline
registry.update_evaluation_scores(
    "majora",
    "v1",
    accuracy=0.86,
    f1_score=0.83,
    perplexity=2.15,
    inference_speed_tokens_per_sec=125,
)
```

### Track Lineage

```python
from afs.registry import LineageTracker

tracker = LineageTracker()

# Register initial training
tracker.add_version(
    model_name="majora",
    version="v1",
    base_model="Qwen2.5-Coder-7B",
    training_data=["oracle", "toolbench"],
    git_commit="abc123def456",
)

# Register fine-tuned version
tracker.add_version(
    model_name="majora",
    version="v2",
    parent_version="v1",
    training_data=["oracle", "toolbench", "additional-data"],
    git_commit="def456ghi789",
)

# Get lineage tree
print(tracker.build_tree("majora"))

# Get ancestors (parent chain)
ancestors = tracker.get_ancestors("majora", "v3")
print(f"v3 comes from: {ancestors}")  # ['v2', 'v1']

# Get data lineage (all training data sources)
data_lineage = tracker.get_data_lineage("majora", "v3")
for version, sources in data_lineage.items():
    print(f"{version}: {sources}")
```

## Data Storage

### Default Locations

- **Registry**: `~/.context/training/registry.json`
- **Lineage**: `~/.context/training/lineage.json`

Custom paths can be specified:

```python
registry = ModelRegistry("/custom/path/registry.json")
tracker = LineageTracker("/custom/path/lineage.json")
```

### Registry Schema

```json
{
  "version": "1.0",
  "updated_at": "2026-01-14T12:00:00",
  "models": {
    "majora": {
      "metadata": {
        "model_name": "majora",
        "created_at": "2026-01-14T10:00:00",
        "description": "Code generation model",
        "owner": "scawful",
        "tags": ["coder", "production"],
        "notes": "Primary coder model"
      },
      "versions": {
        "v1": {
          "model_name": "majora",
          "version": "v1",
          "created_at": "2026-01-14T10:00:00",
          "status": "deployed",
          "lora_path": "/models/majora-v1-lora",
          "gguf_path": "/models/majora-v1-Q8_0.gguf",
          "training": {
            "framework": "unsloth",
            "base_model": "Qwen2.5-Coder-7B",
            "samples": 223,
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.0002,
            "duration_hours": 3.0,
            "cost_usd": 0.80,
            "hardware": "M2 Max"
          },
          "evaluation_scores": {
            "accuracy": 0.85,
            "f1_score": 0.82,
            "perplexity": null,
            "inference_speed_tokens_per_sec": 120,
            "custom_metrics": {}
          },
          "deployed": true,
          "deployed_at": "2026-01-14T11:00:00",
          "parent_version": null,
          "training_data_sources": ["oracle", "toolbench"],
          "git_commit": "abc123def456",
          "tags": ["production", "stable"],
          "notes": "Initial training"
        }
      }
    }
  }
}
```

### Lineage Schema

```json
{
  "version": "1.0",
  "updated_at": "2026-01-14T12:00:00",
  "lineages": {
    "majora": {
      "created_at": "2026-01-14T10:00:00",
      "versions": {
        "v1": {
          "created_at": "2026-01-14T10:00:00",
          "parent_version": null,
          "training_data": ["oracle", "toolbench"],
          "git_commit": "abc123def456",
          "base_model": "Qwen2.5-Coder-7B",
          "notes": "Initial training"
        },
        "v2": {
          "created_at": "2026-01-14T12:00:00",
          "parent_version": "v1",
          "training_data": ["oracle", "toolbench", "additional"],
          "git_commit": "def456ghi789",
          "base_model": "majora:v1",
          "notes": "Fine-tuned on additional data"
        }
      }
    }
  }
}
```

## CLI Usage

### List Models

```bash
afs registry list
```

Output:
```
Registered Models:
============================================================

majora
  Versions: 3 (2 deployed)
  ● v1       (deployed  ) acc=0.850
  ○ v2       (completed ) acc=0.870
  ● v3       (deployed  ) acc=0.880

oracle
  Versions: 2 (1 deployed)
  ● v1       (deployed  ) acc=0.820
```

### Show Model Details

```bash
afs registry info majora
```

### List All Versions

```bash
afs registry versions majora
```

### Compare Versions

```bash
afs registry compare majora v1 v2
```

### Register New Version

```bash
afs registry add majora v1 \
  --base-model "Qwen2.5-Coder-7B" \
  --samples 223 \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 0.0002 \
  --framework unsloth \
  --lora-path /path/to/lora \
  --gguf-path /path/to/gguf \
  --accuracy 0.85 \
  --f1-score 0.82
```

### Update Scores

```bash
afs registry score majora v1 \
  --accuracy 0.86 \
  --f1-score 0.83 \
  --speed 120
```

### Deploy Version

```bash
afs registry deploy majora v1
```

### Rollback

```bash
afs registry rollback majora v1
```

### View Lineage

```bash
afs registry tree majora
```

Output:
```
majora version tree:
└── v1
    ├── v2
    │   └── v3
    └── v2b (experimental branch)
```

```bash
afs registry lineage majora
```

## Version States

A model version can be in one of these states:

| Status | Meaning | Example |
|--------|---------|---------|
| `draft` | In preparation, not yet trained | Created but waiting for training |
| `training` | Currently training | Training in progress |
| `completed` | Training finished | Ready for evaluation/deployment |
| `deployed` | Currently in production | Used by end users |
| `deprecated` | Replaced by newer version | Old version after rollback |
| `archived` | Archived for historical reference | Kept for audit trail |

## Advanced Usage

### Custom Model Metadata

```python
registry.update_metadata(
    model_name="majora",
    description="Code generation model for Python",
    owner="scawful",
    tags=["production", "coder", "qwen"],
    notes="Primary model, stable performance",
)
```

### Custom Metrics

```python
registry.update_evaluation_scores(
    "majora",
    "v1",
    custom_metric_1=0.95,
    custom_metric_2=0.92,
)
```

### Batch Operations

```python
# Get all deployed versions
deployed = [v for v in registry.list_versions("majora") if v.deployed]

# Find best performing version
best = max(
    registry.list_versions("majora"),
    key=lambda v: v.evaluation_scores.accuracy or 0,
)

# Find cheapest training
cheapest = min(
    registry.list_versions("majora"),
    key=lambda v: v.training.cost_usd if v.training else float('inf'),
)
```

### Generate Reports

```python
# Registry summary
print(registry.summary())
print(registry.summary("majora"))  # Single model

# Lineage summary
tracker = LineageTracker()
print(tracker.summary())
print(tracker.build_tree("majora"))
```

## Best Practices

1. **Always track git commits**: Include the commit hash that trained each version for reproducibility
   ```python
   import subprocess
   git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
   registry.register_model(..., git_commit=git_hash)
   ```

2. **Use semantic versioning**: v1.0.0, v1.1.0, v2.0.0 for compatibility tracking
   ```python
   registry.register_model(model_name="majora", version="v1.1.0", ...)
   ```

3. **Track all training data**: Record dataset names for reproducibility
   ```python
   registry.register_model(
       model_name="majora",
       training_data=["oracle-v2", "toolbench-filtered"],
       ...
   )
   ```

4. **Update scores after evaluation**: Run evaluation pipeline, then update
   ```python
   scores = evaluate_model(model_path)
   registry.update_evaluation_scores("majora", "v1", **scores)
   ```

5. **Document changes in notes**: Add context for version differences
   ```python
   registry.register_model(
       model_name="majora",
       notes="Trained on 2x samples, added hard examples",
       ...
   )
   ```

6. **Use tags for categorization**: Mark experimental, production, etc.
   ```python
   registry.register_model(
       model_name="majora",
       tags=["experimental", "coder"],
       ...
   )
   ```

## Integration with Training Pipeline

```python
from afs.registry import ModelRegistry, LineageTracker

def train_model(model_name, training_config):
    """Train and register model."""
    registry = ModelRegistry()
    tracker = LineageTracker()

    # Train
    model_path, metrics = run_training(training_config)

    # Register
    version_info = registry.register_model(
        model_name=model_name,
        base_model=training_config["base_model"],
        samples=training_config["num_samples"],
        epochs=training_config["num_epochs"],
        framework=training_config["framework"],
        lora_path=model_path,
        evaluation_scores=metrics,
        git_commit=get_git_commit(),
        training_data=training_config["datasets"],
        notes=training_config.get("notes", ""),
    )

    # Track lineage
    tracker.add_version(
        model_name=model_name,
        version=version_info.version,
        parent_version=training_config.get("parent_version"),
        training_data=training_config["datasets"],
        git_commit=get_git_commit(),
    )

    return version_info
```

## Testing

Run the test suite:

```bash
pytest tests/test_registry.py -v
```

## See Also

- `afs.training.pipeline` - Training execution
- `afs.evaluation` - Evaluation metrics
- `afs.manager` - Model manager integration
