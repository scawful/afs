# Quick Start: Model Comparison Framework

## Installation

The framework is built-in to AFS. No additional installation needed.

```python
from afs.comparison import ModelComparator, ComparisonMode
```

## Basic Usage (30 seconds)

```python
from afs.comparison import ModelComparator, ComparisonMode

# Create comparator
comp = ModelComparator(ComparisonMode.HEAD_TO_HEAD)

# Load 2 models
comp.load_model("v5", lambda: get_model("v5"))
comp.load_model("v6", lambda: get_model("v6"))

# Run comparison
prompts = ["question 1", "question 2", "question 3"]
report = comp.run_prompts(prompts)

# See results
ranked = report.get_ranked_models()
for rank, (name, stats) in enumerate(ranked, 1):
    print(f"{rank}. {name}: {stats['mean_overall_score']:.3f}")

# Generate reports
comp.generate_markdown_report()
comp.generate_html_report(Path("dashboard.html"))
comp.save_report_json(Path("results.json"))
```

## CLI Quickstart

### Compare 2 Models
```bash
python3 -m afs comparison compare \
  --models gemini-2.0-flash,claude-3.5-sonnet \
  --questions eval/questions.json
```

### Rank Multiple Models
```bash
python3 -m afs comparison tournament \
  --models v5,v6,v7,v8 \
  --questions eval/questions.json
```

### Test for Regression
```bash
python3 -m afs comparison regression \
  --baseline v5 \
  --candidate v6 \
  --questions eval/historical.json
```

## Report Formats

### What You Get

```
comparison_results/
├── comparison_report.md          # Markdown tables
├── comparison_report.json        # Raw data
└── comparison_dashboard.html     # Interactive charts
```

### Markdown Preview

```markdown
# Model Comparison Report

## Summary
| Model | Mean Score | Median Score | Std Dev | Win Rate |
|-------|------------|--------------|---------|----------|
| v7    | 0.892      | 0.905        | 0.041   | 40.0%    |
| v6    | 0.856      | 0.871        | 0.053   | 30.0%    |
| v5    | 0.821      | 0.838        | 0.062   | 20.0%    |
```

## Scoring System

Each response scored on 5 dimensions (0.0-1.0):

| Dimension | Meaning | What It Measures |
|-----------|---------|-----------------|
| **Correctness** | Accuracy | Factual validity |
| **Completeness** | Coverage | Addresses full prompt |
| **Clarity** | Structure | Readability, formatting |
| **Efficiency** | Conciseness | Token usage (fewer = better) |
| **Speed** | Performance | Tokens/second |

**Overall Score = Weighted Average**

Default weights:
- Correctness: 30%
- Completeness: 20%
- Clarity: 20%
- Efficiency: 15%
- Speed: 15%

## Comparison Modes

### 1. Head-to-Head (2 models)
```python
ModelComparator(ComparisonMode.HEAD_TO_HEAD)
```
Direct 1v1 comparison. Perfect for: "Is v6 better than v5?"

### 2. Tournament (2-5 models)
```python
ModelComparator(ComparisonMode.TOURNAMENT)
```
Rank multiple models. Perfect for: "Which of these 4 is best?"

### 3. Regression (2 models)
```python
ModelComparator(ComparisonMode.REGRESSION)
```
Detect degradation. Perfect for: "Did we break anything?"

### 4. A/B Test (2 models)
```python
# Via CLI only
python3 -m afs comparison ab-test --results results.json
```
Analyze production split. Perfect for: "Real user results?"

## Custom Scoring

Want domain-specific scoring? Easy:

```python
from afs.comparison import ResponseScorer, ScoredResponse

class MyScorer(ResponseScorer):
    def score(self, response, reference=None):
        scored = ScoredResponse(response)

        # Your custom logic
        scored.correctness_score = self._validate(response.response)
        scored.completeness_score = self._measure_coverage(response.response)
        scored.clarity_score = 0.8  # Example
        scored.efficiency_score = max(0, 1.0 - response.total_tokens/1000)
        scored.speed_score = min(response.tokens_per_second/100, 1.0)

        # Weighted average
        scored.overall_score = (
            0.4 * scored.correctness_score +
            0.3 * scored.completeness_score +
            0.2 * scored.clarity_score +
            0.05 * scored.efficiency_score +
            0.05 * scored.speed_score
        )
        return scored

# Use it
comp = ModelComparator(scorer=MyScorer())
```

## Statistical Analysis

### Is the difference significant?

```python
from afs.comparison import StatisticalTester

scores_v5 = [r.responses["v5"].overall_score for r in report.results]
scores_v6 = [r.responses["v6"].overall_score for r in report.results]

t_stat, is_sig = StatisticalTester.t_test(scores_v5, scores_v6)
cohen_d = StatisticalTester.effect_size(scores_v5, scores_v6)

print(f"t = {t_stat:.3f}, significant = {is_sig}")
print(f"Cohen's d = {cohen_d:.3f}")

if abs(cohen_d) > 0.8:
    print("Large practical difference")
elif abs(cohen_d) > 0.5:
    print("Medium difference")
elif abs(cohen_d) > 0.2:
    print("Small difference")
else:
    print("Negligible difference")
```

**Effect Size Interpretation:**
- `> 0.8`: Large practical effect
- `0.5-0.8`: Medium effect
- `0.2-0.5`: Small effect
- `< 0.2`: Negligible

## Question Format

### Simple Format (Array of Strings)
```json
[
  "Create a simple loop in assembly",
  "Implement a basic function call",
  "Handle interrupts correctly"
]
```

### Rich Format (Array of Objects)
```json
[
  {
    "prompt": "Create a simple loop in assembly",
    "category": "control_flow",
    "difficulty": "easy"
  },
  {
    "prompt": "Implement a basic function call",
    "category": "functions",
    "difficulty": "medium"
  }
]
```

## Interpreting Results

### Ranking Example

```
🥇 1. v7
   Mean Score: 0.892
   Wins: 4/10 (40%)
   Confidence: High

🥈 2. v6
   Mean Score: 0.856
   Wins: 3/10 (30%)

🥉 3. v5
   Mean Score: 0.821
   Wins: 2/10 (20%)
```

### What It Means

- **Rank 1**: Highest average score across all questions
- **Wins**: Number of questions where model scored highest
- **Confidence**: Gap between winner and runner-up
  - High (>0.5): Clear winner
  - Medium (0.2-0.5): Close competition
  - Low (<0.2): Virtually tied

## Common Patterns

### Pattern 1: Quick A/B Test
```bash
python3 -m afs comparison compare \
  --models old_model,new_model \
  --questions eval/sample_questions.json \
  --output results/quick_test
```

### Pattern 2: Quality Regression
```python
import subprocess
result = subprocess.run([
    "python3", "-m", "afs", "comparison", "regression",
    "--baseline", "production_model",
    "--candidate", "candidate_model",
    "--questions", "eval/golden_test_set.json"
], capture_output=True, text=True)

if "Regression" in result.stdout:
    print("⚠️ Performance degradation detected!")
    # Block deployment
```

### Pattern 3: Model Tournament
```python
competitors = ["v1", "v2", "v3", "v4", "v5"]
comparator = ModelComparator(ComparisonMode.TOURNAMENT)

for version in competitors:
    model = create_generator(model_name=version)
    comparator.load_model(version, lambda m=model: m)

report = comparator.run_prompts(questions)
winner = report.get_ranked_models()[0][0]
print(f"Champion: {winner}")
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cannot load more than 5 models" | Use separate tournaments or pairwise comparisons |
| "No models loaded" | Call `load_model()` before `run_prompts()` |
| "Generation failed" | Check model compatibility with generator interface |
| "Low confidence scores" | Run more prompts or improve question diversity |

## Next Steps

1. **Try the Tutorial**
   ```bash
   python3 examples/comparison_tutorial.py
   ```

2. **Read Full Docs**
   - User guide: `docs/COMPARISON_FRAMEWORK.md`
   - API reference: `src/afs/comparison/README.md`

3. **Implement Custom Scorer**
   - See examples in documentation
   - Validate on known-good cases

4. **Add to CI/CD**
   - Run regression tests on PRs
   - Automated model quality checks

## API Cheat Sheet

```python
# Create & load
comp = ModelComparator(ComparisonMode.TOURNAMENT)
comp.load_model(name, lambda: model)

# Run
report = comp.run_prompts(prompts, generation_config={...})

# Analyze
ranked = report.get_ranked_models()
stats = report.model_stats[model_name]

# Report
comp.generate_markdown_report()
comp.generate_html_report(Path("dashboard.html"))
comp.save_report_json(Path("report.json"))

# Statistics
from afs.comparison import StatisticalTester
t, sig = StatisticalTester.t_test(scores1, scores2)
d = StatisticalTester.effect_size(scores1, scores2)
```

## Performance

- 2-5 models: No problem
- 10-50 prompts per model: Fast (<1 min)
- 100+ prompts: Still reasonable (5-10 min)
- Memory: ~10KB per (model, prompt) pair

## Key Files

| File | Purpose |
|------|---------|
| `src/afs/comparison/framework.py` | Core implementation |
| `src/afs/cli/comparison.py` | CLI commands |
| `docs/COMPARISON_FRAMEWORK.md` | Full documentation |
| `examples/comparison_tutorial.py` | Working examples |

## Support

- **Issues**: Check `README.md` troubleshooting section
- **Questions**: Read the full documentation
- **Feedback**: File an issue in the repository

---

**Ready to compare models? Start here:**
```bash
python3 -m afs comparison compare --help
```
