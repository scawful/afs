# Model Comparison Framework

A comprehensive framework for side-by-side evaluation and comparison of different model versions, variants, and implementations.

## Overview

The comparison framework enables you to:

- **Load 2-5 models simultaneously** for evaluation
- **Run identical prompts** through all models in parallel
- **Capture detailed metrics**: latency, token counts, generation speed
- **Score responses** on 5 independent dimensions: Correctness, Completeness, Clarity, Efficiency, Speed
- **Generate professional reports** in Markdown, JSON, and interactive HTML
- **Perform statistical analysis** including t-tests and effect size (Cohen's d)
- **Run comparisons** in 4 different modes: Head-to-Head, Tournament, Regression, A/B Test

## Features

### 1. Multiple Comparison Modes

| Mode | Use Case | Models | Example |
|------|----------|--------|---------|
| **Head-to-Head** | Direct 1v1 comparison | 2 | v5 vs v6 |
| **Tournament** | Rank multiple models | 2-5 | v5 vs v6 vs v7 vs v8 |
| **Regression** | Detect degradation | 2 | v5 (baseline) vs v6 (candidate) |
| **A/B Test** | Analyze production split | 2 | Real user traffic results |

### 2. Multi-Dimensional Scoring

Each response is scored across 5 dimensions on a 0.0-1.0 scale:

```python
response_scores = {
    "correctness": 0.95,      # Factual accuracy and validity
    "completeness": 0.88,     # How fully it addresses the prompt
    "clarity": 0.92,          # How well structured and readable
    "efficiency": 0.82,       # Token usage (fewer is better)
    "speed": 0.91,            # Generation speed (tokens/sec)
    "overall": 0.90           # Weighted average
}
```

### 3. Comprehensive Reporting

- **Markdown Tables**: Side-by-side comparison for documentation
- **JSON Export**: Machine-readable results for analysis
- **HTML Dashboard**: Interactive charts and visualizations
- **Statistical Tests**: Significance testing and effect sizes

### 4. Integration with AFS Evaluation Suite

Works seamlessly with existing AFS tools:

```python
from afs.comparison import ModelComparator
from afs.evaluation import SemanticEvaluator
```

## Quick Start

### CLI Usage

```bash
# Head-to-head comparison
python3 -m afs comparison compare \
  --models gemini-2.0-flash,claude-3.5-sonnet \
  --questions eval/questions.json \
  --output results/comparison

# Tournament (rank multiple models)
python3 -m afs comparison tournament \
  --models v5,v6,v7,v8 \
  --questions eval/questions.json

# Regression testing (new vs baseline)
python3 -m afs comparison regression \
  --baseline v5 \
  --candidate v6 \
  --questions eval/historical.json

# A/B test analysis
python3 -m afs comparison ab-test \
  --results ab_test_results.json
```

### Python API

```python
from afs.comparison import ModelComparator, ComparisonMode
from afs.generators.model_generator import create_generator

# Create comparator
comparator = ModelComparator(comparison_mode=ComparisonMode.TOURNAMENT)

# Load models
for model_name in ["v5", "v6", "v7"]:
    model = create_generator(model_type="api", model_name=model_name)
    comparator.load_model(model_name, lambda m=model: m)

# Run comparison
prompts = ["question 1", "question 2", ...]
report = comparator.run_prompts(prompts)

# Generate reports
comparator.generate_markdown_report()
comparator.save_report_json(Path("report.json"))
comparator.generate_html_report(Path("dashboard.html"))

# Analyze results
ranked = report.get_ranked_models()
for rank, (model_name, stats) in enumerate(ranked, 1):
    print(f"{rank}. {model_name}: {stats['mean_overall_score']:.3f}")
```

## Architecture

### Core Components

```
comparison/
├── framework.py           # Main comparison engine
│   ├── ModelComparator   # Orchestrator class
│   ├── ResponseScorer    # Scoring interface
│   ├── BasicScorer       # Default implementation
│   ├── ComparisonResult  # Per-prompt results
│   ├── ComparisonReport  # Aggregated report
│   └── StatisticalTester # Significance testing
├── __init__.py           # Module exports
└── README.md            # This file
```

### Data Flow

```
Models → Load
  ↓
Prompts → Evaluate
  ↓
Responses → Score
  ↓
Results → Aggregate
  ↓
Report → Generate (MD/JSON/HTML)
```

## Class Reference

### `ModelComparator`

Main comparison orchestrator.

**Methods:**
- `load_model(name, factory)` - Load a model for comparison
- `run_prompts(prompts, config)` - Execute comparison
- `generate_markdown_report()` - Generate markdown
- `generate_html_report(path)` - Generate interactive HTML
- `save_report_json(path)` - Export raw results

### `ResponseScorer`

Abstract base for response scoring.

**Methods:**
- `score(response, reference=None)` - Score a response

**Implementations:**
- `BasicScorer` - Heuristic-based scoring
- Custom scorers (override for domain-specific logic)

### `ComparisonReport`

Aggregated results across all prompts.

**Properties:**
- `comparison_mode` - Type of comparison
- `model_names` - List of models
- `results` - Per-prompt results
- `model_stats` - Aggregated statistics

**Methods:**
- `compute_statistics()` - Calculate aggregates
- `get_ranked_models()` - Sort by score
- `to_dict()` - Serialize to JSON

### `StatisticalTester`

Significance testing utilities.

**Static Methods:**
- `t_test(scores1, scores2)` - Independent t-test
- `effect_size(scores1, scores2)` - Cohen's d

## Usage Examples

### Example 1: Basic Head-to-Head

```python
from afs.comparison import ModelComparator, ComparisonMode

comparator = ModelComparator(ComparisonMode.HEAD_TO_HEAD)

# Load models
model_a = get_model("v5")
model_b = get_model("v6")
comparator.load_model("v5", lambda: model_a)
comparator.load_model("v6", lambda: model_b)

# Run comparison
prompts = ["prompt1", "prompt2", "prompt3"]
report = comparator.run_prompts(prompts)

# Check results
if report.get_ranked_models()[0][0] == "v6":
    print("v6 is better!")
    print(f"Confidence: {report.results[0].confidence_score:.1%}")
```

### Example 2: Custom Scoring

```python
from afs.comparison import ResponseScorer, ScoredResponse

class DomainScorer(ResponseScorer):
    def score(self, response, reference=None):
        scored = ScoredResponse(response)

        # Custom logic for your domain
        scored.correctness_score = self._check_correctness(response.response)
        scored.completeness_score = self._check_completeness(response.response)

        # ... other dimensions

        scored.overall_score = (
            0.4 * scored.correctness_score +
            0.3 * scored.completeness_score +
            # ... weighted average
        )
        return scored

# Use it
comparator = ModelComparator(scorer=DomainScorer())
```

### Example 3: Statistical Analysis

```python
from afs.comparison import StatisticalTester

# Extract scores
scores_v5 = [r.responses["v5"].overall_score for r in report.results]
scores_v6 = [r.responses["v6"].overall_score for r in report.results]

# Run t-test
t_stat, is_sig = StatisticalTester.t_test(scores_v5, scores_v6)

# Calculate effect size
cohen_d = StatisticalTester.effect_size(scores_v5, scores_v6)

if cohen_d > 0.8:
    print("Large practical improvement")
elif cohen_d > 0.5:
    print("Medium improvement")
elif abs(cohen_d) < 0.2:
    print("Negligible difference")
```

### Example 4: Tournament with Rankings

```python
comparator = ModelComparator(ComparisonMode.TOURNAMENT)

# Load 5 models
for version in ["v4", "v5", "v6", "v7", "v8"]:
    model = create_generator(model_name=version)
    comparator.load_model(version, lambda m=model: m)

# Run tournament
report = comparator.run_prompts(questions)

# Get rankings
ranked = report.get_ranked_models()
for rank, (model_name, stats) in enumerate(ranked, 1):
    medal = "🥇🥈🥉"[min(rank-1, 2)]
    print(f"{medal} {rank}. {model_name:10} "
          f"Score: {stats['mean_overall_score']:.3f} "
          f"Wins: {stats['win_count']}/{stats['total_comparisons']}")
```

## Understanding Results

### Ranking Interpretation

Models are ranked by **mean overall score** (higher is better):

```
Rank 1 (Winner)
  Mean: 0.892
  Median: 0.905
  Std Dev: 0.041
  Win Rate: 40%

Rank 2 (Close Second)
  Mean: 0.856
  Median: 0.871
  Std Dev: 0.053
  Win Rate: 30%
```

### Statistical Significance

**t-statistic and p-value:**
- If `|t| > 1.96`, difference is significant at α=0.05
- Larger |t| means more confidence in the difference

**Effect Size (Cohen's d):**
- `> 0.8`: Large practical difference
- `0.5-0.8`: Medium difference
- `0.2-0.5`: Small difference
- `< 0.2`: Negligible difference

### Metric Trade-offs

When interpreting results, remember:

- **Correctness vs Completeness**: Longer responses may be more complete but less correct
- **Efficiency vs Speed**: Fast models may use more tokens; efficient models may be slower
- **Win Rate vs Mean Score**: High win rate doesn't guarantee highest average score

## Best Practices

### Question Selection

1. **Diversity**: Mix simple and complex prompts
2. **Representativeness**: Cover your typical use cases
3. **Quantity**: Aim for 10-50 prompts for statistical power
4. **Clarity**: Unambiguous, well-defined requirements

### Fair Comparison

1. **Identical Parameters**: Same temperature, max_tokens across models
2. **Same Hardware**: Run on identical hardware if possible
3. **Deterministic Testing**: Use fixed seeds when available
4. **Repeat Testing**: Run multiple times for stability

### Custom Scoring

1. **Validate**: Test scorer on known-good examples
2. **Document**: Clearly describe scoring logic
3. **Weights**: Justify relative importance of dimensions
4. **Consistency**: Apply same logic across all models

## Troubleshooting

### Common Issues

**"Cannot load more than 5 models"**
- The framework limits to 5 simultaneous models for practical reasons
- Consider running separate tournaments if you have many models

**"No models loaded"**
- Call `load_model()` before `run_prompts()`
- Check for exceptions during model loading

**"Low confidence scores"**
- Models may be very similar in quality
- Run more prompts to increase statistical power
- Consider custom scoring if default scoring isn't capturing differences

**"Model generation failed"**
- Check model compatibility with generator interface
- Verify API credentials if using API models
- Review logs for specific errors

## Integration with CI/CD

Use the comparison framework in your pipeline:

```yaml
# .github/workflows/compare_models.yml
on: [push]
jobs:
  compare:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run regression test
        run: |
          python3 -m afs comparison regression \
            --baseline v5 --candidate ${{ github.sha }} \
            --questions eval/historical.json
```

## Performance Considerations

- **Model Loading**: Each model is loaded once and reused
- **Prompt Execution**: Sequential by default (can be parallelized)
- **Scoring**: Uses heuristics by default (can be customized)
- **Memory**: Keeps all results in memory (suitable for < 1000 prompts)

For large-scale comparisons (>1000 prompts), consider:
- Batching prompts
- Streaming results to disk
- Using multiprocessing

## Contributing

To extend the framework:

1. **Custom Scorers**: Inherit from `ResponseScorer`
2. **New Modes**: Add to `ComparisonMode` enum
3. **Report Formats**: Add generator methods
4. **Statistics**: Extend `StatisticalTester`

See `examples/comparison_custom_scorer.py` for examples.

## References

- **Framework Code**: `/Users/scawful/src/lab/afs/src/afs/comparison/framework.py`
- **CLI Commands**: `/Users/scawful/src/lab/afs/src/afs/cli/comparison.py`
- **Tutorial**: `/Users/scawful/src/lab/afs/examples/comparison_tutorial.py`
- **Full Documentation**: `/Users/scawful/src/lab/afs/docs/COMPARISON_FRAMEWORK.md`

## License

Part of the AFS (Assembly Fusion System) project.
