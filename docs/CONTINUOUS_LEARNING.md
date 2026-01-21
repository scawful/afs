# Continuous Learning System

**Self-improving model infrastructure that feeds production usage back into training.**

## Overview

The continuous learning system creates a feedback loop where production model usage automatically improves the model over time. It captures every query, scores quality, collects user feedback, and triggers retraining when sufficient high-quality data accumulates.

```
Production Usage → Logging → Quality Scoring → Triggers → Retraining → A/B Testing → Auto-Promotion
       ↑                                                                                    ↓
       └────────────────────────────── Feedback Loop ──────────────────────────────────────┘
```

## Architecture

### Components

1. **Usage Logger** (`continuous/logger.py`)
   - SQLite database for all model queries/responses
   - Tracks quality scores, user feedback, latency
   - Efficient querying and deduplication

2. **Training Data Generator** (`continuous/generator.py`)
   - Converts production logs → training samples
   - Quality filtering (min score threshold)
   - Deduplication based on content hash
   - Format conversion (ChatML, Alpaca, completion)

3. **Retraining Triggers** (`continuous/trigger.py`)
   - **Sample count**: Trigger after N quality samples
   - **Scheduled**: Weekly/daily retraining
   - **Quality drop**: Retrain if quality drops >X%
   - **Error rate**: Trigger on high negative feedback

4. **A/B Testing** (`continuous/ab_test.py`)
   - Deploy new models as "challengers"
   - Route traffic proportionally (90% champion, 10% challenger)
   - Compare metrics across versions
   - Auto-promote if challenger >5% better

5. **Main Loop** (`continuous/loop.py`)
   - Orchestrates full cycle
   - Monitors triggers every hour
   - Executes retraining when needed
   - Manages A/B tests and promotions

## Quick Start

### 1. Basic Usage Logging

```python
from afs.continuous import UsageLogger

logger = UsageLogger(db_path="~/.context/training/continuous/usage.db")

# Log model usage
record_id = logger.log(
    query="What is 65816 assembly?",
    response="65816 is the CPU used in the SNES...",
    model="nayru-v1",
    expert="asm",
    latency_ms=150,
    quality_score=0.85
)

# Record user feedback (thumbs up/down)
logger.record_feedback(record_id, feedback=1)  # 1=good, -1=bad, 0=neutral
```

### 2. Generate Training Data

```python
from afs.continuous import TrainingDataGenerator, DataGeneratorConfig

config = DataGeneratorConfig(
    min_quality_score=0.7,
    min_user_feedback=1,  # Only positive feedback
    deduplicate=True,
    format_type="chatml"
)

generator = TrainingDataGenerator(logger, config)
result = generator.generate(output_path="training_data.jsonl")

print(f"Generated {result.final_count} training samples")
```

### 3. Automatic Retraining

```python
from afs.continuous import AutoRetrainer, TriggerConfig

config = TriggerConfig(
    min_new_samples=1000,      # Trigger after 1000 quality samples
    schedule_interval_hours=168,  # Or weekly
    quality_drop_threshold=0.1    # Or 10% quality drop
)

retrainer = AutoRetrainer(logger, trigger_config=config)

# Define your training function
def train_model(data_path):
    # Load data, train model, save weights
    return {"loss": 0.25, "accuracy": 0.85, "model_path": "model.pth"}

# Check and retrain if needed
result = retrainer.check_and_retrain(train_fn=train_model)
if result:
    print(f"Retraining triggered: {result['trigger']['reason']}")
```

### 4. A/B Testing

```python
from afs.continuous import ABTestManager, ABTestConfig

config = ABTestConfig(
    initial_challenger_traffic=0.1,  # 10% to new model
    min_improvement_threshold=0.05,   # 5% improvement needed
    enable_auto_promotion=True
)

ab_test = ABTestManager(logger, config)

# Deploy new model as challenger
ab_test.deploy_challenger(
    model_name="nayru-v2",
    model_path="models/nayru-v2.pth",
    traffic_weight=0.1
)

# Route requests
model = ab_test.route_request()  # Returns champion or challenger

# Compare performance
comparison = ab_test.compare_models()
if comparison:
    print(f"Winner: {comparison.winner}")
    print(f"Improvement: {comparison.improvement*100:.1f}%")

# Auto-promote if better
if ab_test.auto_promote_if_ready():
    print("New model promoted to champion!")
```

### 5. Full Continuous Learning Loop

```python
from afs.continuous import ContinuousLearningLoop, LoopConfig

config = LoopConfig(
    check_interval_seconds=3600,  # Check every hour
    enable_ab_testing=True,
    enable_auto_promotion=True
)

loop = ContinuousLearningLoop(config)

# Define training function
def train_model(data_path):
    # Your training logic here
    return {"model_path": "model.pth", "metrics": {...}}

# Run indefinitely
loop.run_loop(train_fn=train_model)

# Or run one iteration
result = loop.run_iteration(train_fn=train_model)
```

## Running the Main Loop

### As a daemon

```bash
# Start the loop
python scripts/continuous_loop.py

# With custom config
python scripts/continuous_loop.py --config config.json

# Check status
python scripts/continuous_loop.py --status

# Manual trigger
python scripts/continuous_loop.py --trigger
```

### Configuration file

```json
{
  "db_path": "~/.context/training/continuous/usage.db",
  "output_dir": "~/.context/training/continuous",
  "check_interval_seconds": 3600,
  "trigger_config": {
    "min_new_samples": 1000,
    "schedule_interval_hours": 168,
    "quality_drop_threshold": 0.1,
    "cooldown_hours": 24
  },
  "generator_config": {
    "min_quality_score": 0.7,
    "min_user_feedback": 1,
    "deduplicate": true,
    "format_type": "chatml"
  },
  "ab_test_config": {
    "initial_challenger_traffic": 0.1,
    "min_improvement_threshold": 0.05,
    "enable_auto_promotion": true,
    "promotion_threshold": 0.05,
    "min_duration_hours": 48
  },
  "enable_ab_testing": true,
  "enable_auto_promotion": true
}
```

## Database Schema

### Usage Table

```sql
CREATE TABLE usage (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    model TEXT NOT NULL,
    expert TEXT,
    latency_ms REAL,
    token_count INTEGER,
    quality_score REAL DEFAULT 0.0,
    user_feedback INTEGER,
    feedback_text TEXT,
    context_hash TEXT,
    dedupe_hash TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

## Trigger Types

### 1. Sample Count Trigger

Triggers when enough quality samples accumulate:

```python
TriggerConfig(
    enable_sample_count=True,
    min_new_samples=1000,
    min_quality_score=0.7
)
```

### 2. Scheduled Trigger

Triggers on a fixed schedule:

```python
TriggerConfig(
    enable_scheduled=True,
    schedule_interval_hours=168  # Weekly
)
```

### 3. Quality Drop Trigger

Triggers when quality drops significantly:

```python
TriggerConfig(
    enable_quality_drop=True,
    quality_drop_threshold=0.1,  # 10% drop
    quality_window_hours=24
)
```

### 4. Error Rate Trigger

Triggers on high negative feedback rate:

```python
TriggerConfig(
    enable_error_rate=True,
    error_rate_threshold=0.2,  # 20% errors
    error_window_hours=24
)
```

## Quality Scoring

Quality scores can come from:

1. **Pre-computed** - Scored during generation
2. **Discriminator model** - Trained quality classifier
3. **User feedback** - Implicit quality signal
4. **Ensemble** - Combination of multiple signals

Integrate with existing `afs.training.scoring`:

```python
from afs.training.scoring import QualityScorer, ScoringConfig

scorer = QualityScorer(config=ScoringConfig())
score = scorer.score_sample(sample)
logger.log(..., quality_score=score.overall)
```

## A/B Testing Metrics

Default metrics compared:

- `avg_quality_score` (weight: 0.5)
- `positive_feedback_rate` (weight: 0.3)
- `avg_latency_ms` (weight: -0.2, lower is better)

Customize:

```python
ABTestConfig(
    metrics=["avg_quality_score", "custom_metric"],
    metric_weights={
        "avg_quality_score": 0.6,
        "custom_metric": 0.4,
    }
)
```

## Integration with Existing Training Pipeline

The continuous learning system builds on `afs.training` and `afs.feedback`:

```python
# Continuous learning generates data
result = generator.generate("new_data.jsonl")

# Feed into existing training pipeline
from afs.training import run_pipeline, PipelineConfig

pipeline_config = PipelineConfig(
    input_paths=["new_data.jsonl"],
    output_dir="training_output",
    score_quality=True,
    deduplicate=True,
    split_data=True
)

pipeline_result = run_pipeline(
    input_paths=pipeline_config.input_paths,
    output_dir=pipeline_config.output_dir,
    config=pipeline_config
)
```

## Rehearsal Buffer Integration

Prevent catastrophic forgetting by merging with rehearsal buffer:

```python
DataGeneratorConfig(
    include_existing_data=True,
    existing_data_path="~/.context/training/rehearsal_buffer.jsonl"
)
```

See `afs.training.rehearsal` for rehearsal buffer management.

## Monitoring

### Get statistics

```python
stats = logger.get_statistics()
print(f"Total records: {stats['total']}")
print(f"Feedback rate: {stats['feedback_rate']*100:.1f}%")
print(f"Avg quality: {stats['avg_quality_score']:.3f}")
```

### Loop status

```python
summary = loop.get_status_summary()
print(f"Total retrains: {summary['status']['total_retrains']}")
print(f"Last retrain: {summary['status']['last_retrain']}")
print(f"Champion: {summary['status']['champion_model']}")
```

## Best Practices

1. **Start conservative** - Low traffic to challenger (10%), high promotion threshold (5%)
2. **Monitor closely** - Watch quality metrics during A/B tests
3. **Quality over quantity** - High quality threshold (>0.7) for training data
4. **Cooldown periods** - Avoid retraining too frequently (24h cooldown)
5. **Test duration** - Run A/B tests for 48+ hours before promotion
6. **Deduplication** - Always enable to avoid overfitting on repeated queries
7. **Feedback collection** - Aim for >10% feedback rate

## Troubleshooting

### No retraining triggered

- Check trigger thresholds (too high?)
- Verify quality scores are being logged
- Check cooldown period hasn't expired
- View trigger logs: `loop.trigger.check_triggers()`

### Low feedback rate

- Make feedback collection easier in UI
- Use uncertainty sampling to prioritize which requests to collect feedback on
- Consider implicit feedback signals (retry, edit, copy)

### Poor A/B test results

- Increase `min_samples_per_version` (more data needed)
- Extend `min_duration_hours` (test longer)
- Check if traffic split is working (monitor routing)
- Verify quality scorer is working correctly

### Database growing too large

- Implement retention policy (delete old records)
- Archive to cold storage after N days
- Increase deduplication threshold

## Examples

See `examples/continuous_learning_demo.py` for complete working examples of:
- Basic logging
- Data generation
- Trigger checking
- A/B testing
- Full loop execution

Run the demo:

```bash
python examples/continuous_learning_demo.py
```

## Files

```
src/afs/continuous/
├── __init__.py          # Public API
├── logger.py            # SQLite usage logging
├── generator.py         # Training data generation
├── trigger.py           # Automatic retraining triggers
├── ab_test.py           # A/B testing framework
└── loop.py              # Main continuous learning loop

scripts/
└── continuous_loop.py   # Main entry point

examples/
└── continuous_learning_demo.py  # Complete demo

docs/
└── CONTINUOUS_LEARNING.md  # This file

~/.context/training/continuous/
├── usage.db             # SQLite database
├── retrain_*.jsonl      # Generated training data
├── models/              # Model weights
├── ab_test_state.json   # A/B test state
└── loop_status.json     # Loop status
```

## Future Enhancements

- [ ] Active learning (uncertainty sampling)
- [ ] Multi-armed bandit for traffic routing
- [ ] Synthetic data generation for error cases
- [ ] Model drift detection
- [ ] Explainable quality scores
- [ ] Integration with model registry
- [ ] Distributed training support
- [ ] Real-time metrics dashboard

## References

- Related: `afs.feedback` - Original feedback collection module
- Related: `afs.training` - Training pipeline and scoring
- Related: `afs.active_learning` - Uncertainty sampling
- Paper: "Continuous Learning in Neural Networks" (Parisi et al.)
- Paper: "Online Learning to Rank" (Li et al.)
