# Model Comparison Framework - Complete Implementation

## Status: ✓ COMPLETE AND PRODUCTION-READY

A comprehensive framework for side-by-side evaluation of 2-5 model versions with support for 4 comparison modes, 5-dimensional scoring, and 3+ report formats.

## Quick Start (2 minutes)

### Installation
Already included in AFS. No additional setup needed.

### Basic Usage
```python
from afs.comparison import ModelComparator, ComparisonMode

comp = ModelComparator(ComparisonMode.TOURNAMENT)
comp.load_model("v5", lambda: get_model("v5"))
comp.load_model("v6", lambda: get_model("v6"))

report = comp.run_prompts(["prompt1", "prompt2", "prompt3"])
comp.generate_html_report(Path("dashboard.html"))
```

### CLI Usage
```bash
python3 -m afs comparison compare \
  --models v5,v6 \
  --questions eval/questions.json

python3 -m afs comparison tournament \
  --models v5,v6,v7,v8 \
  --questions eval/questions.json

python3 -m afs comparison regression \
  --baseline v5 \
  --candidate v6 \
  --questions eval/historical.json
```

## What's Included

### Core Files
- **framework.py** (1,300+ lines): Main comparison engine
- **comparison.py** (700+ lines): CLI integration
- **__init__.py**: Module exports

### Documentation
- **QUICKSTART_COMPARISON.md**: Quick reference (read this first!)
- **docs/COMPARISON_FRAMEWORK.md**: Complete user guide
- **src/afs/comparison/README.md**: Technical reference
- **COMPARISON_FRAMEWORK_SUMMARY.md**: Implementation details

### Examples
- **examples/comparison_tutorial.py**: 5 runnable examples demonstrating:
  1. Head-to-head comparison
  2. Tournament mode
  3. Custom scoring
  4. Statistical analysis
  5. Report generation

Run with: `python3 examples/comparison_tutorial.py`

## Key Features

### 4 Comparison Modes
| Mode | Use Case | Models |
|------|----------|--------|
| **Head-to-Head** | 1v1 comparison | 2 |
| **Tournament** | Multi-model ranking | 2-5 |
| **Regression** | Detect degradation | 2 |
| **A/B Test** | Production analysis | 2 |

### 5-Dimension Scoring
- **Correctness** (accuracy)
- **Completeness** (coverage)
- **Clarity** (structure)
- **Efficiency** (conciseness)
- **Speed** (throughput)

Each scored 0.0-1.0 with weighted overall score.

### Report Formats
- **Markdown**: Tables with summary and detailed results
- **JSON**: Machine-readable serialization
- **HTML**: Interactive dashboard with Plotly charts

### Statistical Analysis
- Independent t-tests
- Cohen's d effect size
- Significance testing
- Confidence scoring
- Win rate analysis

## Architecture

### Class Structure
```
ModelComparator
  ├─ load_model(name, factory)
  ├─ run_prompts(prompts, config)
  ├─ generate_markdown_report()
  ├─ generate_html_report(path)
  └─ save_report_json(path)

ResponseScorer (interface)
  └─ BasicScorer (default implementation)

ComparisonReport
  ├─ results: ComparisonResult[]
  ├─ model_stats: dict
  └─ get_ranked_models()

StatisticalTester
  ├─ t_test(scores1, scores2)
  └─ effect_size(scores1, scores2)
```

### Data Flow
```
Models (2-5)
    ↓
Prompts (10-50)
    ↓
Generate → Capture Response
    ↓
Score (5 dimensions)
    ↓
Aggregate Statistics
    ↓
Generate Reports (MD/JSON/HTML)
```

## Usage Examples

### Example 1: Quick Head-to-Head
```bash
python3 -m afs comparison compare \
  --models old_version,new_version \
  --questions eval/sample.json
```

### Example 2: Tournament with Charts
```python
from afs.comparison import ModelComparator, ComparisonMode

comp = ModelComparator(ComparisonMode.TOURNAMENT)
for v in ["v5", "v6", "v7", "v8"]:
    comp.load_model(v, lambda: create_generator(v))

report = comp.run_prompts(questions)
comp.generate_html_report(Path("results/tournament.html"))
```

### Example 3: Custom Scoring
```python
from afs.comparison import ResponseScorer, ScoredResponse

class MyScorer(ResponseScorer):
    def score(self, response, reference=None):
        scored = ScoredResponse(response)
        # Your domain-specific logic
        scored.correctness_score = self.validate(response.response)
        # ... other dimensions
        scored.overall_score = weighted_average(...)
        return scored

comp = ModelComparator(scorer=MyScorer())
```

### Example 4: Regression Testing
```bash
python3 -m afs comparison regression \
  --baseline production_model \
  --candidate candidate_model \
  --questions eval/golden_set.json
```

Output shows:
- Cohen's d effect size
- Statistical significance
- Clear "improved", "degraded", or "no change" verdict

## Report Output

### Markdown Summary
```markdown
# Model Comparison Report

## Summary
| Model | Mean Score | Median Score | Std Dev | Win Rate | Avg Latency | Avg Tokens |
|-------|------------|--------------|---------|----------|-------------|------------|
| v7    | 0.892      | 0.905        | 0.041   | 40.0%    | 1250ms      | 256        |
| v6    | 0.856      | 0.871        | 0.053   | 30.0%    | 1180ms      | 245        |
| v5    | 0.821      | 0.838        | 0.062   | 20.0%    | 1320ms      | 271        |
```

### HTML Dashboard
- Interactive Plotly charts
- Overall scores bar chart
- Win rate visualization
- Latency comparison
- Token usage analysis
- Summary statistics table

### JSON Export
Complete serialized results for analysis and archiving.

## CLI Commands

```bash
# Compare 2 models head-to-head
python3 -m afs comparison compare \
  --models model1,model2 \
  --questions questions.json

# Rank multiple models
python3 -m afs comparison tournament \
  --models v5,v6,v7,v8 \
  --questions questions.json

# Detect performance regression
python3 -m afs comparison regression \
  --baseline baseline_model \
  --candidate candidate_model \
  --questions historical_questions.json

# Analyze A/B test results
python3 -m afs comparison ab-test \
  --results ab_test_results.json
```

## File Structure

```
/Users/scawful/src/lab/afs/
├── src/afs/comparison/
│   ├── framework.py         (Core implementation)
│   ├── __init__.py          (Module exports)
│   └── README.md            (Technical reference)
├── src/afs/cli/
│   └── comparison.py        (CLI commands)
├── docs/
│   └── COMPARISON_FRAMEWORK.md  (User guide)
├── examples/
│   └── comparison_tutorial.py   (Runnable examples)
├── QUICKSTART_COMPARISON.md     (Quick reference)
├── COMPARISON_FRAMEWORK_SUMMARY.md
└── README_COMPARISON.md         (This file)
```

## Getting Help

### Quick Reference
**Start here:** `QUICKSTART_COMPARISON.md`

### Full Documentation
**Learn more:** `docs/COMPARISON_FRAMEWORK.md`

### Technical Details
**Deep dive:** `src/afs/comparison/README.md`

### Code Examples
**See in action:** `examples/comparison_tutorial.py`

### Troubleshooting
See "Troubleshooting" section in `docs/COMPARISON_FRAMEWORK.md`

## Common Questions

### Q: Can I compare more than 5 models?
A: No, the framework limits to 5 simultaneous models for practical reasons. Use separate tournaments or pairwise comparisons.

### Q: What if my models don't match the generator interface?
A: Implement a wrapper that matches the generator interface. See custom scorer example.

### Q: How do I use custom scoring logic?
A: Inherit from `ResponseScorer` and implement `score()` method. See examples/comparison_tutorial.py.

### Q: Can I integrate this into CI/CD?
A: Yes! See integration examples in the documentation.

### Q: What's the minimum/maximum number of questions?
A: Works with any number, but 10-50 is recommended for statistical power.

## Next Steps

1. **Try an example:**
   ```bash
   python3 examples/comparison_tutorial.py
   ```

2. **Read the quick start:**
   ```bash
   cat QUICKSTART_COMPARISON.md
   ```

3. **Run a real comparison:**
   ```bash
   python3 -m afs comparison compare --help
   ```

4. **Implement custom scoring:**
   See docs for scorer example

5. **Integrate into your workflow:**
   See CI/CD examples in documentation

## Performance

- **Memory:** ~10KB per (model, prompt) pair
- **Speed:** Sequential evaluation (parallelizable)
- **Tested with:** 2-5 models, 5-20 prompts
- **Suitable for:** Model development, production monitoring, quality assurance

## Production Readiness

- ✓ Type hints throughout
- ✓ Comprehensive error handling
- ✓ Logging integration
- ✓ Full documentation
- ✓ Example code
- ✓ Tested and verified
- ✓ Extensible architecture

## Summary Statistics

| Metric | Value |
|--------|-------|
| Lines of Code | 3,900+ |
| Public Classes | 11 |
| Public Methods | 50+ |
| CLI Commands | 4 |
| Report Formats | 3 |
| Scoring Dimensions | 5 |
| Comparison Modes | 4 |
| Documentation Pages | 5 |
| Example Scripts | 1 (with 5 examples) |

## Support

For issues, questions, or feedback:
1. Check the troubleshooting section in the docs
2. Review the example code
3. Read the technical reference
4. Examine the framework source code

---

**Status: COMPLETE** ✓

Created: 2026-01-14
Framework Version: 1.0
AFS Integration: Complete
