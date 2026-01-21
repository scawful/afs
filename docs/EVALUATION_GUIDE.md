# Comprehensive Model Evaluation Suite

## Overview

This guide covers the complete evaluation infrastructure for the AFS multi-model system. The suite includes:

1. **Unified Evaluation Suite** - 100+ questions across 5 domains
2. **Model Comparison Script** - Compare all 5 models on the same questions
3. **Meta-Circular Evaluation** - Models evaluate each other's responses
4. **LMStudio Deployment** - Deploy and run models locally

## Quick Start

### 1. Setup Evaluation Suite

The unified evaluation suite is already created at:
```
~/.context/training/evals/unified_eval_suite.jsonl
```

Contains 100 questions across categories:
- **Code Generation** (10 questions) - Python, JavaScript, C++, React, HTML/CSS
- **Debugging** (10 questions) - Finding and fixing bugs
- **Architecture** (2 questions) - System design
- **Assembly** (20 questions) - 65816 ASM generation/debugging/optimization
- **Oracle Knowledge** (10 questions) - ROM hack specific
- **Cross-Domain** (10 questions) - Multi-skill challenges
- **System Design** (5 questions) - Large-scale architecture

### 2. Deploy to LMStudio

```bash
cd /Users/scawful/src/lab/afs/scripts
./deploy_to_lmstudio.sh
```

This script:
- Checks for GGUF model files
- Links models to LMStudio directory
- Generates curl test scripts
- Creates Python API client
- Produces setup documentation

### 3. Start Model Servers

In LMStudio UI:

1. Load model: `majora.gguf`
2. Click "Chat" or "API Server"
3. Set port: `5000`
4. Start server
5. Repeat for other models on ports 5001-5004

### 4. Run Model Comparison

```bash
cd /Users/scawful/src/lab/afs
python3 scripts/compare_models.py

# Or compare specific models:
python3 scripts/compare_models.py --models majora nayru veran

# Or use sample size:
python3 scripts/compare_models.py --sample-size 10
```

Outputs:
- `~/.context/training/evals/results/comparison_*.md` - Markdown table
- `~/.context/training/evals/results/dashboard_*.html` - Interactive HTML
- `~/.context/training/evals/results/results_*.json` - Detailed JSON
- `~/.context/training/evals/results/comparison_charts_*.png` - Charts

## Evaluation Categories

### Code Generation (10 questions)
Tests ability to write working code:
- Easy: String reversal, prime checking, CSS spinner
- Medium: React component, decorator, REST API
- Hard: Binary search tree, LRU cache, Dijkstra's algorithm

**Scoring:** Feature presence (50%) + Output quality (50%)

### Debugging (10 questions)
Identify and fix bugs in existing code:
- Off-by-one errors
- Infinite loops
- Memory leaks
- SQL injection
- Async/await issues
- Race conditions
- CSS specificity
- Array indexing

**Scoring:** Bug identification (50%) + Correct fix (50%)

### Assembly (20 questions)
65816 assembly for SNES development:

#### Generation (10)
- Memory copying
- Item detection (hookshot check)
- Sound effects
- 16-bit arithmetic
- VBlank waiting
- Jump tables
- 24-bit counters
- Memory clearing
- DMA transfers
- Value swapping

#### Debugging (10)
- Mode mismatches
- Truncation issues
- Flag handling
- Stack corruption
- Invalid jumps
- Incorrect results
- Missing setup
- Indirect jumps
- Comparison logic
- OAM timing

#### Optimization (8)
- Zero-fill patterns
- Bit shifting
- Block moves
- Hardware multiplier
- Loop reduction
- Branchless code
- Bit clearing
- Shift divisions

**Scoring:** Correct technique (40%) + Efficiency (30%) + Explanation (30%)

### Oracle Knowledge (10 questions)
Domain-specific knowledge of Oracle of Secrets ROM hack:
- Memory addresses and their meanings
- Code organization and file structure
- Architectural patterns (namespaces, data-driven design)
- System concepts (V-Blank, NMI hooks)
- Known issues and solutions

**Scoring:** Accuracy (60%) + Completeness (40%)

### Cross-Domain (10 questions)
Integration between Python and Assembly:
- Memory addressing comparison
- Debugging strategies
- Algorithm porting
- Test harness design
- Concurrency handling
- API design
- Instrumentation
- Compiler design
- Trade-offs analysis
- Test generation

**Scoring:** Design quality (40%) + Technical accuracy (40%) + Explanation (20%)

### System Design (5 questions)
Large-scale architecture challenges:
- ROM modding framework
- CI/CD for assembly code
- State synchronization
- Caching strategy
- Multi-model routing

**Scoring:** Architecture quality (50%) + Completeness (50%)

## Metrics Explained

### Accuracy Metrics
- **Average Score** (0-1): Mean correctness across all questions
- **Median Score**: Middle value (more robust to outliers)
- **Success Rate**: % of questions with score >= 0.7

### Speed Metrics
- **Average Time**: Mean seconds per question
- **Faster is better** but accuracy matters most

### Quality Metrics
- **Feature Coverage**: % of expected features present
- **Code Quality**: Readability, efficiency, correctness
- **Explanation Quality**: Clarity and completeness

## Model Profiles

### Majora v1 (Quest Specialist)
- **Port:** 5000
- **Strengths:** Story progression, quest logic, high-level design
- **Weaknesses:** Low-level assembly, exact syntax
- **Best For:** Architecture, Oracle knowledge, game design

### Nayru (Assembly Expert)
- **Port:** 5001
- **Strengths:** 65816 assembly, optimization, hardware details
- **Weaknesses:** General code, UI design, high-level concepts
- **Best For:** Assembly generation/debugging, optimization

### Veran v5 (Logic Specialist)
- **Port:** 5002
- **Strengths:** Rigorous logic, debugging, state management
- **Weaknesses:** Creative tasks, natural language, UI
- **Best For:** Debugging, system design, logic puzzles

### Agahnim (General Purpose)
- **Port:** 5003
- **Strengths:** Balanced across domains, good documentation
- **Weaknesses:** No specialization, may miss deep insights
- **Best For:** General questions, cross-domain integration

### Hylia (Retrieval Specialist)
- **Port:** 5004
- **Strengths:** Finding information, documentation, memory
- **Weaknesses:** Creation, synthesis, optimization
- **Best For:** Knowledge questions, research, documentation

## Meta-Circular Evaluation

Use completed models to evaluate each other's responses:

```bash
python3 scripts/meta_circular_evaluation.py

# Specific models:
python3 scripts/meta_circular_evaluation.py --models nayru agahnim hylia

# Different sample size:
python3 scripts/meta_circular_evaluation.py --sample-size 20
```

### How It Works

1. Query target model: "What is X?"
2. Evaluator model scores response: "This is [score]/10 because..."
3. Multiple evaluators provide consensus
4. Results feed back into training data

### Outputs

- `meta_circular_report_*.md` - Summary of evaluations
- `meta_circular_results_*.json` - Detailed results
- `training_data_*.jsonl` - JSONL for fine-tuning

## Troubleshooting

### Models Not Responding

```bash
# Check health
python3 /Users/scawful/src/lab/afs/lmstudio_client.py

# Check connection
curl http://localhost:5000/chat -d '{"prompt": "test"}'
```

### Performance Issues

- Reduce question sample size: `--sample-size 5`
- Check GPU memory: `nvidia-smi` or `gpustat`
- Close other applications
- Reduce batch size

### Missing Model Files

```bash
# Check what models exist:
find /Users/scawful/models/gguf -name "*.gguf"

# Deploy if needed:
./scripts/deploy_to_lmstudio.sh
```

### Evaluation Timeouts

Increase timeout in `compare_models.py`:
```python
"timeout": 60  # 60 seconds instead of 30
```

## Advanced Usage

### Custom Evaluation Subset

Create custom JSONL with specific questions:

```bash
# Extract assembly questions only:
jq 'select(.category | contains("asm"))' \
  ~/.context/training/evals/unified_eval_suite.jsonl \
  > assembly_eval.jsonl

# Use custom set:
python3 scripts/compare_models.py --custom-file assembly_eval.jsonl
```

### Adding New Questions

Edit `unified_eval_suite.jsonl` and add lines:

```json
{"id": "custom_001", "category": "custom", "prompt": "Your question", "expected_features": ["feature1", "feature2"]}
```

### Statistical Analysis

Post-evaluation analysis:

```python
import json
import statistics

# Load results
with open('results_*.json') as f:
    results = json.load(f)

# Compare models
for summary in results['summaries']:
    print(f"{summary['model']}: {summary['avg_score']:.3f}")
```

### Filtering Results

```bash
# Just assembly questions
jq '.detailed_results | map(select(.category | startswith("asm")))' \
  results_*.json
```

## Benchmarking Best Practices

1. **Warm up models** - Run a few queries before evaluation
2. **Control environment** - Consistent hardware/timing
3. **Multiple runs** - Average across several evaluations
4. **Sample size** - Start small (10), then scale to full (100)
5. **Record metadata** - GPU model, batch size, temperature
6. **Compare fairly** - Same questions, same timeout

## Integration with Training

Meta-circular evaluation generates training data:

```bash
# Run meta-circular eval
python3 scripts/meta_circular_evaluation.py --sample-size 50

# Outputs training data to:
# ~/.context/training/evals/meta_circular/training_data_*.jsonl

# Use for fine-tuning:
python3 scripts/train_majora_v1.py \
  --training-data ~/.context/training/evals/meta_circular/training_data_*.jsonl
```

## File Locations

```
/Users/scawful/src/lab/afs/
├── scripts/
│   ├── compare_models.py              # Main comparison script
│   ├── deploy_to_lmstudio.sh          # LMStudio setup
│   └── meta_circular_evaluation.py    # Model-to-model evaluation
│
├── docs/
│   └── EVALUATION_GUIDE.md            # This file
│
├── curl_tests/                        # Generated by deploy_to_lmstudio.sh
├── lmstudio_client.py                 # Generated by deploy_to_lmstudio.sh
└── LMSTUDIO_SETUP.md                  # Generated by deploy_to_lmstudio.sh

/Users/scawful/models/gguf/
├── ollama/
├── afs/
└── embeddings/

~/.context/training/evals/
├── unified_eval_suite.jsonl           # 100+ question evaluation set
└── results/
    ├── comparison_*.md                # Markdown reports
    ├── dashboard_*.html               # Interactive dashboards
    ├── results_*.json                 # Detailed JSON results
    └── comparison_charts_*.png        # Chart images

~/.context/training/evals/meta_circular/
├── meta_circular_report_*.md          # Meta evaluation summaries
├── meta_circular_results_*.json       # Detailed evaluations
└── training_data_*.jsonl              # Generated training data
```

## Next Steps

1. Deploy models to LMStudio
2. Run initial comparison on sample (10 questions)
3. Analyze results and identify model strengths
4. Run full evaluation (100 questions)
5. Use meta-circular evaluation for training data
6. Fine-tune underperforming models
7. Iterate

---

**Last Updated:** 2025-01-14
**Suite Version:** 1.0 (100 questions)
**Supported Models:** 5 (Majora, Nayru, Veran, Agahnim, Hylia)
