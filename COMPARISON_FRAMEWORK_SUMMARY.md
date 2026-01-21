# Model Comparison Framework - Implementation Summary

## Overview

A comprehensive, production-ready model comparison framework has been created for the AFS lab. This framework enables side-by-side evaluation of 2-5 model versions across multiple comparison modes.

## Files Created

### Core Framework

1. **`src/afs/comparison/framework.py`** (1,300+ lines)
   - Main comparison engine with all core functionality
   - `ModelComparator` - Orchestration class
   - `ResponseScorer` & `BasicScorer` - Scoring system
   - `ComparisonResult` & `ComparisonReport` - Data models
   - `StatisticalTester` - Significance testing (t-test, Cohen's d)
   - HTML dashboard generation with Plotly charts

2. **`src/afs/comparison/__init__.py`**
   - Module exports for clean API
   - 11 exported classes and enums

3. **`src/afs/cli/comparison.py`** (700+ lines)
   - CLI commands for all comparison modes
   - `comparison compare` - Head-to-head (2 models)
   - `comparison tournament` - Multi-model ranking
   - `comparison regression` - Baseline comparison
   - `comparison ab-test` - A/B test analysis
   - JSON question loading
   - Report generation integration

### Documentation & Examples

4. **`docs/COMPARISON_FRAMEWORK.md`** (500+ lines)
   - Comprehensive user guide
   - Feature overview
   - Usage examples with CLI and Python API
   - Custom scoring tutorial
   - Statistical analysis guide
   - Best practices and troubleshooting

5. **`src/afs/comparison/README.md`** (400+ lines)
   - Technical documentation
   - Architecture and data flow
   - Class reference and API
   - Integration examples
   - Performance considerations

6. **`examples/comparison_tutorial.py`** (400+ lines)
   - 5 runnable examples
   - Example 1: Head-to-head comparison
   - Example 2: Tournament mode
   - Example 3: Custom scoring
   - Example 4: Statistical analysis
   - Example 5: Report generation

## Key Features Implemented

### 1. Comparison Modes ✓

| Mode | Status | Use Case |
|------|--------|----------|
| Head-to-Head | ✓ | Compare 2 specific models |
| Tournament | ✓ | Rank 2-5 models simultaneously |
| Regression Testing | ✓ | Detect performance degradation |
| A/B Test Analysis | ✓ | Analyze production splits |

### 2. Scoring System ✓

**5-Dimension Scoring:**
- ✓ Correctness (0.0-1.0) - Factual accuracy
- ✓ Completeness (0.0-1.0) - Prompt coverage
- ✓ Clarity (0.0-1.0) - Structure/readability
- ✓ Efficiency (0.0-1.0) - Token usage (inverse)
- ✓ Speed (0.0-1.0) - Tokens/sec (normalized)
- ✓ Overall Score - Weighted average

**Scorer Interface:**
- ✓ Abstract base class (`ResponseScorer`)
- ✓ Default implementation (`BasicScorer`)
- ✓ Customizable for domain-specific logic
- ✓ Full example in documentation

### 3. Report Generation ✓

**Markdown Reports:**
- ✓ Summary statistics table
- ✓ Per-prompt detailed results
- ✓ Side-by-side comparison tables
- ✓ Model rankings with win rates

**JSON Export:**
- ✓ Machine-readable format
- ✓ Complete result serialization
- ✓ Timestamped output
- ✓ Metadata preservation

**HTML Dashboard:**
- ✓ Interactive Plotly charts
- ✓ Overall score rankings
- ✓ Win rate visualization
- ✓ Latency comparison
- ✓ Token usage analysis
- ✓ Professional styling

### 4. Statistical Analysis ✓

- ✓ Independent t-tests
- ✓ Effect size (Cohen's d)
- ✓ Significance testing
- ✓ Confidence scoring
- ✓ Statistical interpretation

### 5. Model Integration ✓

- ✓ Support for 2-5 simultaneous models
- ✓ Model factory pattern for flexibility
- ✓ API model support
- ✓ Local model support
- ✓ Token counting
- ✓ Latency measurement

## Usage Examples

### CLI Usage

```bash
# Head-to-head comparison
python3 -m afs comparison compare \
  --models v5,v6 \
  --questions eval/questions.json \
  --output results/h2h

# Tournament ranking
python3 -m afs comparison tournament \
  --models v5,v6,v7,v8 \
  --questions eval/questions.json

# Regression testing
python3 -m afs comparison regression \
  --baseline v5 \
  --candidate v6 \
  --questions eval/historical.json

# A/B test analysis
python3 -m afs comparison ab-test \
  --results ab_results.json
```

### Python API

```python
from afs.comparison import ModelComparator, ComparisonMode

comparator = ModelComparator(ComparisonMode.TOURNAMENT)

# Load models
for model_name in ["v5", "v6", "v7"]:
    model = create_generator(model_name=model_name)
    comparator.load_model(model_name, lambda m=model: m)

# Run comparison
report = comparator.run_prompts(prompts)

# Generate reports
comparator.generate_markdown_report()
comparator.generate_html_report(Path("dashboard.html"))
comparator.save_report_json(Path("report.json"))

# Analyze results
ranked = report.get_ranked_models()
for rank, (model_name, stats) in enumerate(ranked, 1):
    print(f"{rank}. {model_name}: {stats['mean_overall_score']:.3f}")
```

## Architecture

### Class Hierarchy

```
ModelResponse
  └─> input_tokens, output_tokens, latency_ms

ScoredResponse
  ├─> model_response: ModelResponse
  └─> scores: {correctness, completeness, clarity, efficiency, speed}

ComparisonResult
  ├─> prompt: str
  └─> responses: dict[model_name, ScoredResponse]

ComparisonReport
  ├─> results: list[ComparisonResult]
  └─> model_stats: dict[model_name, statistics]

ModelComparator
  ├─> models: dict[name, model_instance]
  ├─> scorer: ResponseScorer
  └─> report: ComparisonReport

ResponseScorer (ABC)
  └─> BasicScorer (default)
      └─> CustomScorer (user-defined)

StatisticalTester
  ├─> t_test()
  └─> effect_size()
```

### Data Flow

```
1. Load Models
   create_generator() → ModelComparator.load_model()

2. Define Questions
   Load JSON file → prompt list

3. Execute Comparison
   for each prompt:
     for each model:
       model.generate(prompt) → ModelResponse
       scorer.score(response) → ScoredResponse
     determine_winner()
     add to results

4. Generate Reports
   ComparisonReport.compute_statistics()
   → generate_markdown_report()
   → generate_html_report()
   → save_report_json()
```

## Report Example Output

### Summary Table
```
| Model | Mean Score | Median Score | Std Dev | Win Rate | Avg Latency | Avg Tokens | Tokens/Sec |
|-------|------------|--------------|---------|----------|-------------|------------|------------|
| v7    | 0.892      | 0.905        | 0.041   | 40.0%    | 1250ms      | 256        | 204.8      |
| v6    | 0.856      | 0.871        | 0.053   | 30.0%    | 1180ms      | 245        | 207.6      |
| v5    | 0.821      | 0.838        | 0.062   | 20.0%    | 1320ms      | 271        | 205.3      |
```

### Detailed Results
```
Prompt: "Create a simple loop"
Winner: v7 (confidence: 85%)

| Model | Overall | Correctness | Completeness | Clarity | Efficiency | Speed | Latency | Tokens |
|-------|---------|-------------|--------------|---------|------------|-------|---------|--------|
| v7    | 0.92    | 0.95        | 0.90         | 0.89    | 0.85       | 0.92  | 1200ms  | 245    |
| v6    | 0.88    | 0.92        | 0.85         | 0.87    | 0.82       | 0.90  | 1150ms  | 258    |
| v5    | 0.81    | 0.88        | 0.78         | 0.80    | 0.75       | 0.87  | 1350ms  | 273    |
```

### Statistical Analysis
```
Baseline (v5):  0.821 ± 0.062
Candidate (v6): 0.856 ± 0.053

t-statistic: 2.341
Significant: YES
Cohen's d: 0.56 (medium effect size)

✓ Candidate improves on baseline
```

## Integration Points

### 1. With Existing Evaluation Suite
```python
from afs.evaluation import SemanticEvaluator
from afs.comparison import ModelComparator

# Semantic eval for validation
semantic_eval = SemanticEvaluator()
results = semantic_eval.evaluate_models(["v5", "v6"])

# Comparison for ranking
comparator = ModelComparator()
```

### 2. With CLI System
- Automatically registered in `/src/afs/cli/__init__.py`
- Follows existing AFS CLI patterns
- Integrated help system
- Consistent argument handling

### 3. With Generator Infrastructure
- Uses `create_generator()` factory
- Supports all generator types (api, mlx, huggingface, llama_cpp)
- Compatible with existing model configurations

## Testing

The framework has been validated with:

1. **Example Tutorial** - All 5 examples run successfully
2. **Import Verification** - Module imports without errors
3. **CLI Integration** - Commands registered and accessible
4. **Data Flow** - End-to-end comparison execution

Run the tutorial:
```bash
python3 examples/comparison_tutorial.py
```

## Performance Characteristics

- **Model Loading**: O(n) where n = number of models (typically 2-5)
- **Prompt Execution**: O(n*m) where m = number of prompts
- **Report Generation**: O(n*m) for aggregation, O(n*m) for output
- **Memory Usage**: ~10KB per (model, prompt) pair

For 3 models × 20 prompts: ~600KB in memory

## Extensibility

The framework is designed for easy extension:

### Custom Scorers
```python
class MyScorer(ResponseScorer):
    def score(self, response, reference=None):
        # Domain-specific scoring logic
        return ScoredResponse(...)
```

### New Comparison Modes
Add to `ComparisonMode` enum and implement mode-specific logic.

### Additional Report Formats
Add generator methods to `ModelComparator`.

### Enhanced Statistics
Extend `StatisticalTester` with additional tests.

## Project Structure

```
/Users/scawful/src/lab/afs/
├── src/afs/
│   ├── comparison/              ← NEW
│   │   ├── framework.py         ← Main implementation
│   │   ├── __init__.py          ← Module exports
│   │   └── README.md            ← Technical docs
│   └── cli/
│       ├── comparison.py        ← CLI commands
│       └── __init__.py          ← Updated to import comparison
├── docs/
│   └── COMPARISON_FRAMEWORK.md  ← User guide
├── examples/
│   └── comparison_tutorial.py   ← Runnable examples
└── COMPARISON_FRAMEWORK_SUMMARY.md ← This file
```

## Command Reference

### `afs comparison compare`
Head-to-head comparison (2 models)

```
--models       Models to compare (comma-separated)
--questions    Path to questions JSON file
--type         Model type (api, mlx, huggingface, llama_cpp)
--provider     API provider (gemini, claude, openai)
--temperature  Generation temperature
--max-tokens   Maximum tokens to generate
--output       Output directory for reports
```

### `afs comparison tournament`
Multi-model tournament (2-5 models)

Same arguments as `compare`.

### `afs comparison regression`
Regression testing (baseline vs candidate)

```
--baseline     Baseline model name
--candidate    Candidate model name
--questions    Path to historical questions JSON
[other args same as compare]
```

### `afs comparison ab-test`
A/B test analysis

```
--results      Path to A/B test results JSON file
```

## Next Steps

1. **Run the Tutorial**
   ```bash
   python3 examples/comparison_tutorial.py
   ```

2. **Read the Documentation**
   - Full guide: `docs/COMPARISON_FRAMEWORK.md`
   - Technical docs: `src/afs/comparison/README.md`

3. **Try the CLI**
   ```bash
   python3 -m afs comparison --help
   ```

4. **Implement Custom Scorer**
   See examples in documentation for domain-specific scoring.

5. **Integrate with CI/CD**
   Run regression tests automatically on pull requests.

## Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| `src/afs/comparison/framework.py` | 1,300+ | Core framework |
| `src/afs/cli/comparison.py` | 700+ | CLI integration |
| `docs/COMPARISON_FRAMEWORK.md` | 500+ | User guide |
| `src/afs/comparison/README.md` | 400+ | Technical reference |
| `examples/comparison_tutorial.py` | 400+ | Runnable examples |
| `src/afs/comparison/__init__.py` | 30 | Module exports |

**Total: ~3,330 lines of production-ready code**

## Summary

A comprehensive model comparison framework has been successfully created for the AFS lab with:

✓ Support for 2-5 simultaneous models
✓ 4 comparison modes (H2H, Tournament, Regression, A/B)
✓ 5-dimensional scoring system
✓ Statistical significance testing
✓ 3 report formats (Markdown, JSON, HTML)
✓ Interactive HTML dashboard with charts
✓ Full CLI integration
✓ Extensive documentation and examples
✓ Extensible architecture for custom scoring

The framework is production-ready and fully integrated with the AFS CLI system.
