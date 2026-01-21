# Evaluation Guide

Complete guide to evaluating AFS models using the comprehensive evaluation suite.

## Quick Start

```bash
# 1. Deploy models to LMStudio
./scripts/deploy_to_lmstudio.sh

# 2. Start model servers (in LMStudio UI)
# Load majora.gguf on port 5000
# Load nayru.gguf on port 5001
# Load veran.gguf on port 5002
# etc.

# 3. Run model comparison
python3 scripts/compare_models.py

# 4. View results
open ~/.context/training/evals/results/dashboard_*.html
```

## Evaluation Suite

The unified evaluation suite contains 100+ questions across 5 domains.

### Domain Breakdown

**Code Generation (10 questions)**
- Tests ability to write working code
- Easy: String reversal, prime checking, CSS spinner
- Medium: React component, decorator, REST API
- Hard: Binary search tree, LRU cache, Dijkstra's algorithm

Scoring: Feature presence (50%) + Code quality (50%)

**Debugging (10 questions)**
- Identify and fix bugs in existing code
- Categories: off-by-one errors, infinite loops, memory leaks, SQL injection, async/await, race conditions, CSS, array indexing

Scoring: Bug identification (50%) + Correct fix (50%)

**Assembly (65816) - 20 questions**

Generation (10):
- Memory copying, item detection, sound effects
- 16-bit arithmetic, VBlank waiting, jump tables
- 24-bit counters, memory clearing, DMA transfers
- Value swapping

Debugging (10):
- Mode mismatches, truncation issues, flag handling
- Stack corruption, invalid jumps, incorrect results
- Missing setup, indirect jumps, comparison logic
- OAM timing

Optimization (8):
- Zero-fill patterns, bit shifting, block moves
- Hardware multiplier, loop reduction, branchless code
- Bit clearing, shift divisions

Scoring: Correct technique (40%) + Efficiency (30%) + Explanation (30%)

**Oracle Knowledge (10 questions)**
- Domain-specific knowledge of Oracle of Secrets ROM hack
- Memory addresses and their meanings
- Code organization and file structure
- Architectural patterns and system concepts
- Known issues and solutions

Scoring: Accuracy (60%) + Completeness (40%)

**Cross-Domain (10 questions)**
- Integration between Python and Assembly
- Memory addressing comparison, debugging strategies
- Algorithm porting, test harness design
- Concurrency handling, API design, instrumentation
- Compiler design, trade-offs analysis, test generation

Scoring: Design quality (40%) + Technical accuracy (40%) + Explanation (20%)

**System Design (5 questions)**
- Large-scale architecture challenges
- ROM modding framework, CI/CD for assembly
- State synchronization, caching strategy
- Multi-model routing

Scoring: Architecture quality (50%) + Completeness (50%)

## Model Comparison

Run evaluation across all models:

```bash
# Compare all models with default sample size
python3 scripts/compare_models.py

# Compare specific models only
python3 scripts/compare_models.py --models majora nayru veran

# Use smaller sample for quick testing
python3 scripts/compare_models.py --sample-size 10

# Use custom evaluation subset
python3 scripts/compare_models.py --custom-file assembly_eval.jsonl

# Specify output directory
python3 scripts/compare_models.py --output ~/results
```

### Outputs

Model comparison generates:

1. **Markdown Reports** (`comparison_*.md`)
   - Readable table format
   - Scores by domain
   - Model summaries

2. **Interactive Dashboard** (`dashboard_*.html`)
   - Visual comparison charts
   - Domain breakdowns
   - Speed metrics
   - Sortable tables

3. **JSON Results** (`results_*.json`)
   - Detailed per-question scores
   - Response text
   - Timing information
   - Metadata

4. **Charts** (`comparison_charts_*.png`)
   - Bar charts comparing models
   - Domain performance
   - Speed vs accuracy tradeoff

All outputs saved to: `~/.context/training/evals/results/`

## Model Profiles

### Majora v1 (Quest Specialist)
- **Port:** 5000
- **Strengths:** Story progression, quest logic, high-level design
- **Weaknesses:** Low-level assembly, exact syntax
- **Best For:** Architecture, Oracle knowledge, game design

Expected scores:
- Code generation: 0.65
- Debugging: 0.70
- Assembly: 0.35
- Oracle: 0.85
- Cross-domain: 0.60

### Nayru (Assembly Expert)
- **Port:** 5001
- **Strengths:** 65816 assembly, optimization, hardware details
- **Weaknesses:** General code, UI design, high-level concepts
- **Best For:** Assembly generation/debugging, optimization

Expected scores:
- Code generation: 0.50
- Debugging: 0.45
- Assembly: 0.90
- Oracle: 0.55
- Cross-domain: 0.75

### Veran v5 (Logic Specialist)
- **Port:** 5002
- **Strengths:** Rigorous logic, debugging, state management
- **Weaknesses:** Creative tasks, natural language, UI
- **Best For:** Debugging, system design, logic puzzles

Expected scores:
- Code generation: 0.80
- Debugging: 0.85
- Assembly: 0.50
- Oracle: 0.60
- Cross-domain: 0.80

### Agahnim (General Purpose)
- **Port:** 5003
- **Strengths:** Balanced across domains, good documentation
- **Weaknesses:** No specialization, may miss deep insights
- **Best For:** General questions, cross-domain integration

Expected scores:
- Code generation: 0.70
- Debugging: 0.72
- Assembly: 0.55
- Oracle: 0.65
- Cross-domain: 0.72

### Hylia (Retrieval Specialist)
- **Port:** 5004
- **Strengths:** Finding information, documentation, memory
- **Weaknesses:** Creation, synthesis, optimization
- **Best For:** Knowledge questions, research, documentation

Expected scores:
- Code generation: 0.45
- Debugging: 0.50
- Assembly: 0.40
- Oracle: 0.80
- Cross-domain: 0.65

## Meta-Circular Evaluation

Use trained models to evaluate each other's responses.

### How It Works

1. Query target model with test question
2. Get response from target model
3. Use evaluator model to score response
4. Aggregate scores from multiple evaluators
5. Feedback into training data

```bash
# Run meta-circular evaluation
python3 scripts/meta_circular_evaluation.py

# Specific models as evaluators
python3 scripts/meta_circular_evaluation.py \
  --evaluators nayru veran agahnim

# Sample size
python3 scripts/meta_circular_evaluation.py --sample-size 50
```

### Outputs

1. **Summary Report** (`meta_circular_report_*.md`)
   - Consensus scores
   - Model strengths/weaknesses
   - Recommendations

2. **Detailed Results** (`meta_circular_results_*.json`)
   - Per-evaluator scores
   - Full evaluation text
   - Disagreement metrics

3. **Training Data** (`training_data_*.jsonl`)
   - Evaluator responses as training samples
   - Self-critique examples
   - Can be fine-tuned back into models

## Screenshot Evaluation

Evaluate models on screenshot-based tasks (UI/design questions):

```bash
python3 scripts/screenshot_evaluation.py \
  --screenshots-dir ~/screenshots \
  --models majora nayru veran
```

Useful for:
- UI generation quality
- Design understanding
- Visual debugging
- Layout reasoning

## Benchmark Results

View historical benchmark results:

```bash
# List all benchmark runs
ls -lh ~/.context/training/evals/results/

# Latest dashboard
open $(ls -t ~/.context/training/evals/results/dashboard_*.html | head -1)

# Compare runs
python3 scripts/analyze_benchmarks.py \
  --results ~/.context/training/evals/results/
```

## Custom Evaluation

### Create Custom Question Set

```bash
# Extract assembly questions only
jq 'select(.category | contains("asm"))' \
  ~/.context/training/evals/unified_eval_suite.jsonl \
  > assembly_eval.jsonl

# Extract by specific domains
jq 'select(.category == "oracle_knowledge")' \
  ~/.context/training/evals/unified_eval_suite.jsonl \
  > oracle_eval.jsonl
```

### Add New Questions

Edit `~/.context/training/evals/unified_eval_suite.jsonl`:

```json
{
  "id": "custom_001",
  "category": "assembly",
  "prompt": "Write 65816 assembly to copy 256 bytes from $000000 to $7E0000",
  "expected_features": [
    "LDA",
    "STA",
    "loop",
    "counter"
  ],
  "max_score": 1.0
}
```

Then run evaluation:

```bash
python3 scripts/compare_models.py \
  --custom-file assembly_eval.jsonl
```

## Evaluation Metrics

### Accuracy Metrics

**Average Score** (0-1)
- Mean correctness across all questions
- Best: 1.0, Worst: 0.0

**Median Score**
- Middle value (robust to outliers)
- Better than mean when outliers present

**Success Rate**
- Percentage of questions scoring >= 0.7
- Practical success metric

**Domain Scores**
- Average for each category
- Identify model specializations

### Speed Metrics

**Average Time per Question**
- Mean seconds to answer
- Useful for production constraints

**Percentile Times**
- P50, P95, P99 latencies
- Understand latency distribution

### Quality Metrics

**Feature Coverage**
- Percentage of expected features present
- Checklist scoring for code

**Code Quality**
- Readability, efficiency, correctness
- Multi-point scale (0-10)

**Explanation Quality**
- Clarity, completeness, accuracy
- Subjective scoring

## Troubleshooting

### Models Not Responding

```bash
# Test health
curl http://localhost:5000/api/chat -d '{"prompt": "test"}'

# Check if running
ps aux | grep lmstudio

# Verify connection from Python
python3 lmstudio_client.py
```

### Performance Issues

Solutions:
1. Reduce question sample size: `--sample-size 5`
2. Check GPU memory: `nvidia-smi` or `gpustat`
3. Close other applications
4. Reduce batch size in LMStudio
5. Use CPU for smaller models

### Missing Model Files

```bash
# Check available models
find ~/models/gguf -name "*.gguf"

# Deploy if needed
./scripts/deploy_to_lmstudio.sh

# Convert from LoRA if needed
python3 scripts/convert_to_gguf.py \
  --model ~/models/adapters/afs/majora-v1-lora \
  --output ~/models/gguf/majora.gguf
```

### Evaluation Timeouts

Increase timeout in `compare_models.py`:
```python
# In lmstudio_client.py
timeout = 60  # seconds instead of 30
```

Or use command-line flag:
```bash
python3 scripts/compare_models.py --timeout 60
```

### Inconsistent Results

Causes:
- Model temperature/randomness
- GPU variance
- Different batch sizes
- Memory pressure

Solutions:
1. Set seed: `--seed 42`
2. Set temperature: `--temperature 0.0`
3. Run multiple times and average
4. Keep machine quiet (no other processes)

## Best Practices

### Evaluation Guidelines

1. **Warm up models** - Run a few queries before evaluation
2. **Control environment** - Same machine, no other processes
3. **Multiple runs** - Average across 3+ evaluations
4. **Start small** - Test with 10-20 questions first
5. **Record metadata** - Save GPU model, batch size, temperature
6. **Compare fairly** - Same questions, same timeout
7. **Version control** - Save evaluation results with commit
8. **Domain focus** - Evaluate by domain not just overall

### Sampling Strategy

```
Start:     5 questions (5 min)
         ↓
Quality:  10 questions (10 min)
         ↓
Full:     50 questions (1 hour)
         ↓
Statistical: 100+ questions with CI
```

### Analyzing Results

```python
import json

# Load results
with open('results_*.json') as f:
    results = json.load(f)

# Compare models
for summary in results['summaries']:
    print(f"{summary['model']}: {summary['avg_score']:.3f}")

# Domain breakdown
for domain in results['domain_scores']:
    for model, scores in domain.items():
        print(f"{model} {domain}: {scores['avg']:.2f}")

# Find disagreements
for question in results['detailed_results']:
    scores = [r['score'] for r in question['responses']]
    if max(scores) - min(scores) > 0.5:
        print(f"Disagreement on: {question['id']}")
```

## Integration with Training

Meta-circular evaluation generates training data:

```bash
# Run meta-circular eval
python3 scripts/meta_circular_evaluation.py --sample-size 50

# Outputs to:
# ~/.context/training/evals/meta_circular/training_data_*.jsonl

# Use for fine-tuning
python3 scripts/train_majora_v2.py \
  --data ~/.context/training/evals/meta_circular/training_data_*.jsonl
```

## File Locations

```
/Users/scawful/src/lab/afs/
├── scripts/
│   ├── compare_models.py              # Main comparison
│   ├── deploy_to_lmstudio.sh          # Setup
│   ├── meta_circular_evaluation.py    # Model evaluation
│   ├── screenshot_evaluation.py       # Screenshot eval
│   └── analyze_benchmarks.py          # Analysis
├── curl_tests/                        # Generated by deploy_to_lmstudio.sh
├── lmstudio_client.py                 # Generated by deploy_to_lmstudio.sh
└── LMSTUDIO_SETUP.md                  # Generated by deploy_to_lmstudio.sh

/Users/scawful/models/gguf/
├── ollama/
├── afs/
└── embeddings/

~/.context/training/evals/
├── unified_eval_suite.jsonl           # 100+ questions
└── results/
    ├── comparison_*.md                # Reports
    ├── dashboard_*.html               # Interactive
    ├── results_*.json                 # Data
    ├── comparison_charts_*.png        # Charts
    └── meta_circular/
        ├── meta_circular_report_*.md
        ├── meta_circular_results_*.json
        └── training_data_*.jsonl
```

## Next Steps

1. Deploy models to LMStudio
2. Run initial comparison (10 questions)
3. Analyze results to identify specializations
4. Run full evaluation (100 questions)
5. Use meta-circular eval for training data
6. Fine-tune underperforming models
7. Iterate and improve

---

**Last Updated:** January 2026
