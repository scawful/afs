# afs.continuous - Self-Improving Model Infrastructure

**Complete continuous learning system that feeds production usage back into training.**

## What's Inside

```
continuous/
├── logger.py       - SQLite usage tracking (queries, responses, feedback)
├── generator.py    - Training data generation with quality filtering
├── trigger.py      - Automatic retraining triggers (count, schedule, quality drop)
├── ab_test.py      - A/B testing framework (challenger vs champion)
├── loop.py         - Main orchestration loop
└── README.md       - This file
```

## Quick Example

```python
from afs.continuous import ContinuousLearningLoop, LoopConfig

# Initialize
loop = ContinuousLearningLoop(LoopConfig())

# Log production usage
record_id = loop.log_usage(
    query="What is 65816?",
    response="CPU for SNES...",
    model="nayru-v1",
    quality_score=0.85
)

# Record feedback
loop.record_feedback(record_id, feedback=1)  # Thumbs up!

# Define training
def train(data_path):
    # Your training logic
    return {"model_path": "model.pth", "loss": 0.25}

# Run continuous learning
loop.run_loop(train_fn=train)  # Monitors and retrains automatically
```

## Key Features

### 1. Usage Logging
- **SQLite database** for fast queries and persistence
- **Quality scoring** integration
- **User feedback** tracking (thumbs up/down)
- **Deduplication** based on content hash
- **Efficient querying** with indexes

### 2. Training Data Generation
- **Quality filtering** (min score threshold)
- **Feedback filtering** (positive only)
- **Deduplication** to avoid overfitting
- **Format conversion** (ChatML, Alpaca, completion)
- **Merge with existing data** to prevent forgetting

### 3. Automatic Triggers
- **Sample count**: Retrain after N quality samples
- **Scheduled**: Weekly/daily retraining
- **Quality drop**: Retrain if quality drops >X%
- **Error rate**: Trigger on high negative feedback
- **Cooldown**: Prevent retraining too frequently

### 4. A/B Testing
- **Champion vs Challenger** model versioning
- **Traffic routing** (90% champion, 10% challenger)
- **Metric comparison** (quality, feedback, latency)
- **Auto-promotion** if challenger >5% better
- **State persistence** across restarts

### 5. Main Loop
- **Hourly checks** for retraining conditions
- **Automatic execution** when triggers fire
- **A/B test management** (deploy, compare, promote)
- **Status tracking** (last retrain, total retrains, errors)
- **Graceful shutdown** on interrupt

## Component Details

### UsageLogger

```python
from afs.continuous import UsageLogger

logger = UsageLogger("usage.db")

# Log usage
record_id = logger.log(
    query="...",
    response="...",
    model="nayru-v1",
    quality_score=0.85
)

# Record feedback
logger.record_feedback(record_id, feedback=1)

# Query records
for record in logger.get_records(min_quality=0.7, with_feedback_only=True):
    print(record.query, record.quality_score)

# Get statistics
stats = logger.get_statistics()
print(f"Total: {stats['total']}, Feedback: {stats['with_feedback']}")
```

### TrainingDataGenerator

```python
from afs.continuous import TrainingDataGenerator, DataGeneratorConfig

config = DataGeneratorConfig(
    min_quality_score=0.7,
    min_user_feedback=1,
    deduplicate=True,
    format_type="chatml"
)

generator = TrainingDataGenerator(logger, config)
result = generator.generate("training.jsonl")

print(f"Generated {result.final_count} samples")
print(f"Quality stats: {result.quality_stats}")
```

### RetrainTrigger & AutoRetrainer

```python
from afs.continuous import AutoRetrainer, TriggerConfig

config = TriggerConfig(
    min_new_samples=1000,
    schedule_interval_hours=168,
    quality_drop_threshold=0.1,
    cooldown_hours=24
)

retrainer = AutoRetrainer(logger, trigger_config=config)

# Define training
def train_model(data_path):
    # Your training logic
    return {"model_path": "model.pth", "metrics": {...}}

# Check and retrain if needed
result = retrainer.check_and_retrain(train_fn=train_model)
if result:
    print(f"Retrained! Trigger: {result['trigger']['type']}")
```

### ABTestManager

```python
from afs.continuous import ABTestManager, ABTestConfig

config = ABTestConfig(
    initial_challenger_traffic=0.1,
    min_improvement_threshold=0.05,
    enable_auto_promotion=True
)

ab_test = ABTestManager(logger, config)

# Deploy challenger
ab_test.deploy_challenger("nayru-v2", Path("model.pth"), traffic_weight=0.1)

# Route requests
model = ab_test.route_request()  # Returns champion or challenger

# Compare performance
comparison = ab_test.compare_models()
if comparison and comparison.winner == ModelStatus.CHALLENGER:
    print(f"Challenger winning by {comparison.improvement*100:.1f}%")

# Auto-promote if ready
if ab_test.auto_promote_if_ready():
    print("New model promoted!")
```

### ContinuousLearningLoop

```python
from afs.continuous import ContinuousLearningLoop, LoopConfig

config = LoopConfig(
    db_path="usage.db",
    check_interval_seconds=3600,  # Check every hour
    enable_ab_testing=True,
    enable_auto_promotion=True
)

loop = ContinuousLearningLoop(config)

# Run indefinitely
loop.run_loop(train_fn=train_model)

# Or run one iteration
result = loop.run_iteration(train_fn=train_model)

# Get status
summary = loop.get_status_summary()
print(f"Retrains: {summary['status']['total_retrains']}")
print(f"Champion: {summary['status']['champion_model']}")
```

## Integration Patterns

### Pattern 1: Inference Logging

```python
class ModelService:
    def __init__(self):
        self.loop = ContinuousLearningLoop()

    def generate(self, query: str) -> tuple[str, str]:
        start = time.time()
        response = self.model.generate(query)
        latency = (time.time() - start) * 1000

        record_id = self.loop.log_usage(
            query=query,
            response=response,
            model="nayru-v1",
            latency_ms=latency,
            quality_score=self.scorer.score(response)
        )

        return response, record_id
```

### Pattern 2: Feedback Collection

```python
@app.post("/feedback")
def feedback(record_id: str, is_positive: bool):
    loop.record_feedback(
        record_id,
        feedback=1 if is_positive else -1
    )
```

### Pattern 3: Background Loop

```python
import threading

def run_continuous_learning():
    loop = ContinuousLearningLoop()
    loop.run_loop(train_fn=train_model)

# Start in background thread
thread = threading.Thread(target=run_continuous_learning, daemon=True)
thread.start()
```

### Pattern 4: Scheduled Checks

```python
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
loop = ContinuousLearningLoop()

def check_and_retrain():
    loop.run_iteration(train_fn=train_model)

# Check every hour
scheduler.add_job(check_and_retrain, 'interval', hours=1)
scheduler.start()
```

## Configuration

All components accept config objects:

```python
# Trigger config
TriggerConfig(
    enable_sample_count=True,
    min_new_samples=1000,
    enable_scheduled=True,
    schedule_interval_hours=168,
    enable_quality_drop=True,
    quality_drop_threshold=0.1,
    enable_error_rate=False,
    cooldown_hours=24
)

# Generator config
DataGeneratorConfig(
    min_quality_score=0.7,
    min_user_feedback=1,
    deduplicate=True,
    format_type="chatml",
    max_samples=None
)

# A/B test config
ABTestConfig(
    initial_challenger_traffic=0.1,
    min_samples_per_version=100,
    min_improvement_threshold=0.05,
    enable_auto_promotion=True,
    promotion_threshold=0.05,
    min_duration_hours=48
)

# Loop config
LoopConfig(
    db_path="usage.db",
    output_dir="continuous_output",
    check_interval_seconds=3600,
    trigger_config=trigger_config,
    generator_config=generator_config,
    ab_test_config=ab_test_config,
    enable_ab_testing=True,
    enable_auto_promotion=True
)
```

## Database Schema

SQLite table `usage`:

```sql
CREATE TABLE usage (
    id TEXT PRIMARY KEY,           -- Unique record ID
    timestamp TEXT NOT NULL,       -- ISO timestamp
    query TEXT NOT NULL,           -- User query
    response TEXT NOT NULL,        -- Model response
    model TEXT NOT NULL,           -- Model identifier
    expert TEXT,                   -- Expert/agent name
    latency_ms REAL,               -- Inference latency
    token_count INTEGER,           -- Token count
    quality_score REAL,            -- Quality score (0-1)
    user_feedback INTEGER,         -- User feedback (-1, 0, 1)
    feedback_text TEXT,            -- Optional feedback text
    context_hash TEXT,             -- Context identifier
    dedupe_hash TEXT,              -- Deduplication hash
    created_at TEXT                -- Creation timestamp
);
```

## Testing

```bash
# Run tests
pytest tests/test_continuous_learning.py -v

# Run demo
python examples/continuous_learning_demo.py
```

## Documentation

- [CONTINUOUS_LEARNING.md](../../../docs/CONTINUOUS_LEARNING.md) - Full documentation
- [CONTINUOUS_LEARNING_QUICKSTART.md](../../../docs/CONTINUOUS_LEARNING_QUICKSTART.md) - Quick start guide
- [examples/continuous_learning_demo.py](../../../examples/continuous_learning_demo.py) - Working examples

## Dependencies

- **SQLite** - Built-in Python database
- **afs.training.scoring** - Quality scoring (optional)
- **afs.generators.base** - TrainingSample format (optional)

## Performance

- **Logging**: ~1ms per record
- **Query**: ~100ms for 10k records with filters
- **Deduplication**: ~1s per 1000 records
- **Trigger check**: ~100ms
- **Database size**: ~1KB per record

## Best Practices

1. **Start with high thresholds** - 1000+ samples, 0.7+ quality
2. **Enable cooldown** - 24h minimum between retrains
3. **Monitor closely** - Check status regularly during A/B tests
4. **Test locally first** - Use `--no-train` flag for dry runs
5. **Collect feedback** - Aim for >10% feedback rate
6. **Deduplicate always** - Prevents overfitting on repeated queries
7. **Use rehearsal buffer** - Prevents catastrophic forgetting

## Troubleshooting

**No retraining triggered?**
- Check trigger thresholds (may be too high)
- Verify quality scores are being logged
- Check cooldown period

**Low feedback rate?**
- Make feedback UI more prominent
- Use uncertainty sampling for which queries to collect feedback on
- Consider implicit signals (retry, edit, copy)

**A/B test inconclusive?**
- Increase min_samples_per_version
- Run test longer (min_duration_hours)
- Check traffic routing is working

**Database too large?**
- Archive old records periodically
- Increase deduplication threshold
- Delete records older than N days

## Future Enhancements

- Active learning with uncertainty sampling
- Multi-armed bandit for traffic routing
- Synthetic data generation for failure cases
- Real-time metrics dashboard
- Distributed training support
- Model drift detection
- Explainable quality scores

## See Also

- `afs.feedback` - Original feedback collection module
- `afs.training` - Training pipeline and scoring
- `afs.active_learning` - Uncertainty sampling
- `afs.registry` - Model registry integration
