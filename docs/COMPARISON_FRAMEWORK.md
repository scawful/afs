# Model Comparison Framework

Comprehensive framework for side-by-side evaluation of different model versions.

## Features

### Comparison Modes

1. **Head-to-Head** (2 models): Direct comparison between two specific models
2. **Tournament** (2-5 models): Rank multiple models on identical prompts
3. **Regression Testing**: Detect performance degradation vs. baseline
4. **A/B Test Analysis**: Analyze production traffic split results

### Scoring Dimensions

Each response is scored on 5 independent dimensions (0.0-1.0 scale):

| Dimension | Meaning | Examples |
|-----------|---------|----------|
| **Correctness** | Factual accuracy and absence of errors | Syntax correctness, logical validity |
| **Completeness** | How fully the response addresses the prompt | Length, detail level, coverage |
| **Clarity** | How well structured and readable the output is | Formatting, line breaks, comments |
| **Efficiency** | Token usage (inverse scoring) | Shorter responses score higher |
| **Speed** | Generation speed (tokens/sec) | Faster throughput scores higher |

### Reports Generated

- **Markdown Report**: Side-by-side tables with all metrics
- **JSON Report**: Machine-readable results for further analysis
- **HTML Dashboard**: Interactive charts and visualizations
- **Statistical Tests**: t-tests and effect size (Cohen's d)

## Installation

The framework is built into AFS and requires no additional installation.

```bash
# Verify the framework is available
python3 -m afs comparison --help
```

## Usage Examples

### 1. Head-to-Head Comparison

Compare two specific models:

```bash
python3 -m afs comparison compare \
  --models gemini-2.0-flash,claude-3.5-sonnet \
  --questions eval/questions.json \
  --provider gemini \
  --output results/h2h_comparison
```

**Output:**
- `results/h2h_comparison/comparison_report.md` - Markdown tables
- `results/h2h_comparison/comparison_report.json` - Raw data
- `results/h2h_comparison/comparison_dashboard.html` - Interactive dashboard

### 2. Tournament Mode

Rank 3-5 models on the same questions:

```bash
python3 -m afs comparison tournament \
  --models v5,v6,v7,v8 \
  --questions eval/questions.json \
  --output results/tournament
```

Models are ranked by mean overall score. Output includes win rates and statistical significance.

### 3. Regression Testing

Detect performance degradation with a baseline:

```bash
python3 -m afs comparison regression \
  --baseline v5 \
  --candidate v6 \
  --questions eval/historical_baseline.json \
  --output results/regression_v5_vs_v6
```

Reports Cohen's d effect size:
- `d > 0.2`: Meaningful improvement ✓
- `-0.2 < d < 0.2`: No meaningful difference ≈
- `d < -0.2`: Regression ⚠️

### 4. A/B Test Analysis

Analyze production traffic split results:

```bash
python3 -m afs comparison ab-test \
  --results production_ab_results.json
```

Expected JSON format:

```json
{
  "model_a": "v5",
  "model_b": "v6",
  "results_a": [
    {"score": 0.85},
    {"score": 0.92},
    ...
  ],
  "results_b": [
    {"score": 0.88},
    {"score": 0.91},
    ...
  ]
}
```

## Question Format

Questions can be provided as JSON in two formats:

### Format 1: Array of Strings
```json
[
  "Create a simple loop in assembly",
  "Implement a basic function call",
  "Handle interrupts correctly"
]
```

### Format 2: Array of Objects
```json
[
  {
    "prompt": "Create a simple loop in assembly",
    "category": "control_flow"
  },
  {
    "prompt": "Implement a basic function call",
    "category": "functions"
  }
]
```

## Python API

### Basic Usage

```python
from afs.comparison import ModelComparator, ComparisonMode, BasicScorer
from afs.generators.model_generator import create_generator

# Create comparator
comparator = ModelComparator(
    comparison_mode=ComparisonMode.TOURNAMENT,
    scorer=BasicScorer()
)

# Load models
for model_name in ["v5", "v6", "v7"]:
    model = create_generator(
        model_type="api",
        model_name=model_name,
        api_provider="gemini"
    )
    comparator.load_model(model_name, lambda m=model: m)

# Run comparison
prompts = [
    "Create a simple loop",
    "Handle interrupts",
    "Implement function calls"
]

report = comparator.run_prompts(prompts)

# Generate reports
md_report = comparator.generate_markdown_report()
comparator.generate_html_report(Path("dashboard.html"))
comparator.save_report_json(Path("report.json"))
```

### Custom Scoring

Extend `ResponseScorer` for domain-specific evaluation:

```python
from afs.comparison import ResponseScorer, ScoredResponse

class AssemblyScorer(ResponseScorer):
    """Custom scorer for assembly code."""

    def score(self, response, reference=None):
        scored = ScoredResponse(model_response=response)

        # Check for compilation errors
        has_syntax = self._check_valid_syntax(response.response)
        scored.correctness_score = 0.95 if has_syntax else 0.2

        # Check completeness
        scored.completeness_score = self._check_completeness(response.response)

        # Check clarity (comments, labels, etc)
        scored.clarity_score = self._check_clarity(response.response)

        # Efficiency and speed (from response metadata)
        scored.efficiency_score = max(0, 1.0 - response.total_tokens/1000)
        scored.speed_score = min(response.tokens_per_second / 100, 1.0)

        # Weighted average
        scored.overall_score = (
            0.35 * scored.correctness_score +
            0.25 * scored.completeness_score +
            0.20 * scored.clarity_score +
            0.10 * scored.efficiency_score +
            0.10 * scored.speed_score
        )

        return scored

    def _check_valid_syntax(self, code):
        # Validate assembly syntax
        pass

    def _check_completeness(self, code):
        # Check if code fully addresses the prompt
        pass

    def _check_clarity(self, code):
        # Score based on comments, labels, formatting
        pass

# Use custom scorer
comparator = ModelComparator(scorer=AssemblyScorer())
```

### Statistical Analysis

Access detailed statistics:

```python
# Get ranked models
ranked = report.get_ranked_models()
for rank, (model_name, stats) in enumerate(ranked, 1):
    print(f"{rank}. {model_name}")
    print(f"   Mean Score: {stats['mean_overall_score']:.3f}")
    print(f"   Win Rate: {stats['win_rate']:.1%}")
    print(f"   Avg Latency: {stats['mean_latency_ms']:.0f}ms")

# Statistical testing
from afs.comparison import StatisticalTester

scores_v5 = [r.responses['v5'].overall_score for r in report.results]
scores_v6 = [r.responses['v6'].overall_score for r in report.results]

t_stat, is_significant = StatisticalTester.t_test(scores_v5, scores_v6)
effect_size = StatisticalTester.effect_size(scores_v5, scores_v6)

print(f"t-statistic: {t_stat:.3f}")
print(f"Significant (α=0.05): {is_significant}")
print(f"Effect size (Cohen's d): {effect_size:.3f}")
```

## Output Structure

### Markdown Report

```markdown
# Model Comparison Report

**Generated:** 2026-01-14T12:34:56.789000
**Comparison Mode:** tournament
**Prompts Evaluated:** 10

## Summary

| Model | Mean Score | Median Score | Std Dev | Win Rate | Avg Latency | Avg Tokens | Tokens/Sec |
|-------|------------|--------------|---------|----------|-------------|------------|------------|
| v7    | 0.892      | 0.905        | 0.041   | 40.0%    | 1250ms      | 256        | 204.8     |
| v6    | 0.856      | 0.871        | 0.053   | 30.0%    | 1180ms      | 245        | 207.6     |
| v5    | 0.821      | 0.838        | 0.062   | 20.0%    | 1320ms      | 271        | 205.3     |
| v8    | 0.798      | 0.812        | 0.068   | 10.0%    | 1450ms      | 289        | 199.3     |

## Detailed Results

### Prompt 1
```
Create a simple loop in assembly
```

**Winner:** v7 (confidence: 85%)

| Model | Overall | Correctness | Completeness | Clarity | Efficiency | Speed | Latency | Tokens |
|-------|---------|-------------|--------------|---------|------------|-------|---------|--------|
| v7    | 0.92    | 0.95        | 0.90         | 0.89    | 0.85       | 0.92  | 1200ms  | 245    |
| v6    | 0.88    | 0.92        | 0.85         | 0.87    | 0.82       | 0.90  | 1150ms  | 258    |
```

### JSON Report

```json
{
  "timestamp": "2026-01-14T12:34:56.789000",
  "comparison_mode": "tournament",
  "models": ["v5", "v6", "v7", "v8"],
  "prompt_count": 10,
  "model_statistics": {
    "v7": {
      "mean_overall_score": 0.892,
      "median_overall_score": 0.905,
      "stdev_overall_score": 0.041,
      "win_rate": 0.4,
      "win_count": 4,
      "total_comparisons": 10,
      "mean_latency_ms": 1250,
      "mean_tokens": 256,
      "mean_tokens_per_second": 204.8
    },
    ...
  },
  "results": [
    {
      "prompt": "Create a simple loop in assembly",
      "winner": "v7",
      "confidence_score": 0.85,
      "is_significant": true,
      "responses": {
        "v7": {
          "model_name": "v7",
          "response": "...",
          "latency_ms": 1200,
          "tokens": 245,
          "tokens_per_second": 204.2,
          "scores": {
            "correctness": 0.95,
            "completeness": 0.90,
            "clarity": 0.89,
            "efficiency": 0.85,
            "speed": 0.92
          },
          "overall_score": 0.92
        },
        ...
      }
    }
  ]
}
```

## Interpreting Results

### Ranking Models

Models are ranked by **mean overall score** across all prompts:

- **Rank 1**: Highest average performance
- **Win Rate**: Percentage of prompts where model scored highest
- **Confidence**: Gap between winner and runner-up

### Statistical Significance

- **Significant?**: YES means the difference is statistically meaningful (α=0.05)
- **Effect Size (Cohen's d)**:
  - `> 0.8`: Large practical difference
  - `0.5-0.8`: Medium practical difference
  - `0.2-0.5`: Small practical difference
  - `< 0.2`: Negligible difference

### Efficiency vs. Speed Trade-off

- **High Efficiency**: Fewer tokens (more concise)
- **High Speed**: More tokens/sec (faster generation)
- **Balance**: Both matter for real-world deployment

## Best Practices

### 1. Question Selection

- Use diverse, representative questions
- Include edge cases and difficult scenarios
- Mix simple and complex prompts
- Aim for 10-20 questions per comparison

### 2. Fair Comparison

- Keep generation parameters identical across models
- Use same temperature and max_tokens
- Run on same hardware if possible
- Repeat tests for stability

### 3. Interpreting Results

- Don't over-index on single metrics
- Consider statistical significance, not just means
- Look at variance and consistency
- Check if winner changes with different question sets

### 4. Custom Scoring

- Implement domain-specific scoring logic
- Validate scorer on gold-standard examples
- Use consistent weights across dimensions
- Document scoring rationale

## Troubleshooting

### "Failed to load model"
- Verify model name/path is correct
- Check API credentials are set
- Ensure model type matches actual model

### "No models loaded"
- Call `load_model()` before `run_prompts()`
- Check for exceptions during model loading

### "Comparison failed: ..."
- Check question JSON format
- Verify model response format
- Review logs for specific errors

### Low confidence scores
- Winners may be very close in quality
- Consider running more prompts
- Check if scorer thresholds are appropriate

## Integration with Evaluation Suite

The comparison framework integrates with existing evaluation tools:

```python
from afs.evaluation import SemanticEvaluator
from afs.comparison import ModelComparator

# Run semantic evaluation first
semantic_eval = SemanticEvaluator()
results = semantic_eval.evaluate_models(["v5", "v6"])

# Then compare side-by-side
comparator = ModelComparator()
# ... continue with comparison
```

## Performance Tips

- **Parallel Execution**: Use `ProcessPoolExecutor` for large question sets
- **Model Caching**: Keep models loaded across multiple prompt sets
- **API Batching**: Group API calls when possible
- **Memory Management**: Clear intermediate results for large datasets

## API Reference

See `/Users/scawful/src/lab/afs/src/afs/comparison/framework.py` for full API documentation.

## Examples

Complete working examples are available in:
- `examples/comparison_tutorial.py` - Basic usage
- `examples/comparison_custom_scorer.py` - Custom scoring
- `examples/comparison_regression_pipeline.py` - Regression workflow
