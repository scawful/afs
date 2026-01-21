# Development Guide

Guide for contributing to and developing the AFS training system.

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- Virtual environment (venv or conda)

### Setup Development Environment

```bash
# Clone repository (if not already done)
cd ~/src/lab/afs

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Verify Installation

```bash
# Run basic tests
pytest tests/ -v

# Check code style
black --check src/
pylint src/afs/

# Run type checking
mypy src/afs/
```

## Project Structure

```
src/afs/
├── __init__.py              # Package exports
├── config.py                # Configuration management
├── context_fs.py            # File system interface
├── manager.py               # Main AFS manager
├── discovery.py             # Project discovery
├── graph.py                 # Knowledge graph
├── orchestration.py         # Agent orchestration
│
├── cli/                     # Command-line interface
│   ├── __init__.py
│   ├── training.py          # Training commands
│   ├── context.py           # Context commands
│   ├── services.py          # Service commands
│   └── orchestrator.py      # Agent commands
│
├── training/                # Training pipeline
│   ├── __init__.py
│   ├── rehearsal.py         # Rehearsal buffers
│   ├── scoring.py           # Quality scoring
│   ├── splitter.py          # Data splitting
│   ├── rebalancer.py        # Dataset rebalancing
│   ├── registry.py          # Experiment tracking
│   └── converters/          # Format converters
│
├── generators/              # Data generation
│   ├── base.py              # Base generator class
│   ├── asm_augment.py       # ASM augmentation
│   └── cot/                 # Chain-of-thought
│
├── discriminator/           # Quality filtering
│   ├── electra.py           # ELECTRA scorer
│   └── asar.py              # Syntax validator
│
├── evaluation/              # Evaluation tools
│   ├── benchmark.py         # Benchmark suite
│   ├── meta_circular.py     # Model evaluation
│   └── screenshot.py        # Screenshot tests
│
├── oracle/                  # Oracle integration
│   └── training_generator.py
│
├── tokenizer/               # Custom tokenizers
│   └── asm.py
│
├── knowledge/               # Knowledge bases
│   ├── addresses.py         # Address tables
│   └── oracle.py            # Oracle knowledge
│
└── agents/                  # Agent implementations
    ├── researcher.py        # Researcher agent
    ├── context_auditor.py   # Context auditor
    └── context_warm.py      # Context warmer

tests/
├── test_training.py         # Training tests
├── test_generators.py       # Generator tests
├── test_scoring.py          # Scoring tests
├── test_cli.py              # CLI tests
└── test_deployment.py       # Deployment tests
```

## Code Standards

### Style Guide

Follow PEP 8 with these exceptions:

```python
# Line length: 100 characters max
# Use type hints for all functions
def process_data(data: list[str]) -> dict[str, int]:
    """Process data and return results."""
    pass

# Use meaningful variable names
# Good:
quality_scores = [0.8, 0.9, 0.7]

# Bad:
qs = [0.8, 0.9, 0.7]

# Use docstrings for all public functions
def generate_samples(source_path: Path) -> Iterator[TrainingSample]:
    """
    Generate training samples from source.

    Args:
        source_path: Path to source directory

    Returns:
        Iterator of TrainingSample objects

    Raises:
        FileNotFoundError: If source_path doesn't exist
    """
    pass
```

### Type Hints

Use type hints for better code clarity:

```python
from typing import Iterator, Optional
from pathlib import Path

def process_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    quality_threshold: float = 0.5
) -> int:
    """Process file and return number of samples."""
    pass
```

### Imports

Organize imports in standard order:

```python
# Standard library
import json
import logging
from pathlib import Path
from typing import Iterator, Optional

# Third-party
import requests
import numpy as np

# Local
from afs.generators.base import TrainingSample
from afs.training.scoring import QualityScorer
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_training.py -v

# Run specific test
pytest tests/test_training.py::test_split_dataset -v

# Run with coverage
pytest --cov=src/afs tests/

# Run with coverage report
pytest --cov=src/afs --cov-report=html tests/
```

### Writing Tests

```python
import pytest
from pathlib import Path
from afs.training import split_dataset
from afs.generators.base import TrainingSample

def test_split_dataset_basic():
    """Test basic train/val/test splitting."""
    # Arrange
    samples = [
        TrainingSample(
            instruction=f"q{i}",
            output=f"a{i}",
            domain="test"
        )
        for i in range(100)
    ]

    # Act
    splits = split_dataset(
        samples,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )

    # Assert
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10


def test_split_dataset_stratified():
    """Test stratified splitting preserves domain distribution."""
    # Create samples with domains
    samples = [
        TrainingSample(instruction=f"q{i}", output=f"a{i}", domain="oracle")
        for i in range(60)
    ] + [
        TrainingSample(instruction=f"q{i}", output=f"a{i}", domain="memory")
        for i in range(40)
    ]

    splits = split_dataset(
        samples,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        stratify_by="domain"
    )

    # Check domain distribution
    train_oracle = sum(1 for s in splits["train"] if s.domain == "oracle")
    val_oracle = sum(1 for s in splits["val"] if s.domain == "oracle")

    assert train_oracle / len(splits["train"]) == pytest.approx(0.6, abs=0.05)
    assert val_oracle / len(splits["val"]) == pytest.approx(0.6, abs=0.05)
```

### Fixtures

```python
@pytest.fixture
def sample_training_data(tmp_path):
    """Create sample training data for tests."""
    data_file = tmp_path / "training_data.jsonl"

    samples = [
        TrainingSample(
            instruction=f"Question {i}",
            output=f"Answer {i}",
            domain="test"
        )
        for i in range(100)
    ]

    with open(data_file, "w") as f:
        for sample in samples:
            f.write(sample.to_json() + "\n")

    return data_file


def test_quality_scorer(sample_training_data):
    """Test quality scoring with sample data."""
    from afs.training.scoring import QualityScorer, ScoringConfig

    config = ScoringConfig.from_profile("generic")
    scorer = QualityScorer(config=config)

    # Test with sample file
    with open(sample_training_data) as f:
        for line in f:
            sample = TrainingSample.from_json(line)
            score = scorer.score(sample)
            assert 0 <= score.overall <= 1
```

## Common Development Tasks

### Add New Training Data Source

1. Create generator class:

```python
# src/afs/generators/my_source.py
from afs.generators.base import BaseGenerator, TrainingSample
from pathlib import Path
from typing import Iterator

class MySourceGenerator(BaseGenerator):
    def __init__(self, source_path: Path):
        super().__init__(name="my_source", domain="custom")
        self.source_path = source_path

    def generate(self) -> Iterator[TrainingSample]:
        """Generate training samples."""
        for file in self.source_path.rglob("*.txt"):
            content = file.read_text()
            # Parse and generate samples
            yield TrainingSample(
                instruction="...",
                output="...",
                domain="custom",
                source=str(file)
            )
```

2. Add CLI command:

```python
# In src/afs/cli/training.py
def training_mysource_export_command(args: argparse.Namespace) -> int:
    """Export my source data."""
    from afs.generators.my_source import MySourceGenerator

    generator = MySourceGenerator(Path(args.source))
    count = 0

    with open(args.output, "w") as f:
        for sample in generator.generate():
            f.write(sample.to_json() + "\n")
            count += 1

    print(f"Exported {count} samples")
    return 0
```

3. Register command:

```python
# In register_parsers()
train_mysource = training_sub.add_parser(
    "mysource-export",
    help="Export my source data"
)
train_mysource.add_argument("--source", required=True)
train_mysource.add_argument("--output", required=True)
train_mysource.set_defaults(func=training_mysource_export_command)
```

4. Write tests:

```python
def test_mysource_generator(tmp_path):
    """Test my source generator."""
    from afs.generators.my_source import MySourceGenerator

    # Create test files
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # Generate samples
    gen = MySourceGenerator(tmp_path)
    samples = list(gen.generate())

    assert len(samples) > 0
    assert samples[0].domain == "custom"
```

### Add New Scoring Component

1. Implement scorer:

```python
# In src/afs/training/scoring.py
class MyScorer:
    def score(self, sample: TrainingSample) -> float:
        """Score sample using custom metric."""
        # Implement scoring logic
        return 0.8
```

2. Integrate into QualityScorer:

```python
class QualityScorer:
    def __init__(self, config: ScoringConfig):
        self.my_scorer = MyScorer()
        # ... other scorers

    def score(self, sample: TrainingSample) -> QualityScore:
        my_score = self.my_scorer.score(sample)
        # Combine with other scores
```

3. Add configuration option:

```python
# In ScoringConfig
@dataclass
class ScoringWeights:
    electra: float = 0.4
    asar: float = 0.3
    entity: float = 0.2
    length: float = 0.1
    my_metric: float = 0.0  # Add new weight
```

### Add New CLI Command

1. Create command function:

```python
# In src/afs/cli/training.py
def training_mycommand_command(args: argparse.Namespace) -> int:
    """Do something with training data."""
    from afs.training import do_something

    result = do_something(
        input_path=Path(args.input),
        output_path=Path(args.output),
        param1=args.param1
    )

    print(f"Success: {result}")
    return 0
```

2. Register command:

```python
def register_parsers(subparsers):
    # ... existing code

    train_mycommand = training_sub.add_parser(
        "mycommand",
        help="Do something with training data"
    )
    train_mycommand.add_argument("--input", required=True)
    train_mycommand.add_argument("--output", required=True)
    train_mycommand.add_argument("--param1", default="value")
    train_mycommand.set_defaults(func=training_mycommand_command)
```

3. Test command:

```bash
afs training mycommand --input data.jsonl --output output.jsonl
```

## Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Or for specific module
logging.getLogger("afs.training").setLevel(logging.DEBUG)
```

### Use IPython for Exploration

```bash
# Interactive exploration
ipython

# Load module
from afs.training import RehearsalBuffer
from pathlib import Path

# Explore
buffer = RehearsalBuffer()
buffer.load_from_jsonl(Path("data.jsonl"), version="v1")
# ... inspect buffer state
```

### Debug Failing Tests

```bash
# Run with Python debugger
pytest --pdb tests/test_training.py

# Run with extra verbose output
pytest -vv tests/test_training.py

# Stop on first failure
pytest -x tests/test_training.py
```

## Continuous Integration

### Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:

```bash
# Run manually
pre-commit run --all-files

# Automatically run before commit
git commit -m "..."
```

Checks:
- Code style (black)
- Linting (pylint, flake8)
- Type checking (mypy)
- YAML validation
- Trailing whitespace

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def process_data(
    data: list[dict[str, str]],
    output_path: Path,
    quality_threshold: float = 0.5
) -> int:
    """Process data and write to output file.

    Processes input data, applies quality filtering, and writes results
    to the specified output path. Returns the number of items processed.

    Args:
        data: List of dictionaries containing raw data
        output_path: Path where to write processed results
        quality_threshold: Minimum quality score (0-1) to include. Defaults to 0.5.

    Returns:
        Number of items successfully processed and written.

    Raises:
        ValueError: If quality_threshold is not between 0 and 1
        IOError: If output_path is not writable

    Example:
        >>> from pathlib import Path
        >>> data = [{"text": "hello"}]
        >>> count = process_data(data, Path("output.jsonl"))
        >>> print(f"Processed {count} items")
        Processed 1 items
    """
    if not 0 <= quality_threshold <= 1:
        raise ValueError("quality_threshold must be between 0 and 1")
    # ... implementation
```

### Update Documentation

When adding new features:

1. Add docstrings to code
2. Update relevant markdown files in `/docs/`
3. Add examples to API reference
4. Update table of contents

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

# Profile a function
cProfile.run("process_large_dataset()", "stats")

# Analyze results
p = pstats.Stats("stats")
p.sort_stats("cumulative").print_stats(20)
```

### Memory Profiling

```bash
pip install memory-profiler

python -m memory_profiler script.py
```

### Common Optimizations

1. **Batch processing**: Process samples in batches instead of one-by-one
2. **Caching**: Cache expensive computations
3. **Lazy loading**: Load data only when needed
4. **Parallelization**: Use multiprocessing for CPU-bound operations

```python
from concurrent.futures import ProcessPoolExecutor

def process_batch(batch):
    return [score_sample(s) for s in batch]

# Process in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_batch, data_batches)
```

## Release Process

### Version Bumping

```bash
# Check current version
cat setup.py | grep version

# Update version in setup.py
# Commit and tag
git tag v1.0.0
git push origin v1.0.0
```

### Creating Release

```bash
# Build distribution
python -m build

# Test installation
pip install dist/afs-1.0.0.tar.gz

# Upload to PyPI (if public)
python -m twine upload dist/afs-1.0.0.tar.gz
```

## Getting Help

### Documentation

- [Training Guide](training.md) - Complete training walkthrough
- [API Reference](api.md) - Python and CLI API
- [Architecture](architecture.md) - System design

### Contact

- Check GitHub issues for known problems
- Create new issue if bug not reported
- Refer to AGENTS.md for contribution guidelines

---

**Last Updated:** January 2026
