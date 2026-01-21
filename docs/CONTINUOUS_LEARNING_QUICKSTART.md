# Continuous Learning Quick Start

**Get your models improving themselves in 5 minutes!**

## What is This?

A complete self-improving model system that:
1. Logs every production query/response
2. Collects user feedback (thumbs up/down)
3. Automatically retrains when you have 1000+ quality samples
4. A/B tests the new model (10% traffic)
5. Auto-promotes if it's 5% better

## Installation

```bash
# Already installed in afs!
cd ~/src/lab/afs
```

## 5-Minute Demo

### 1. Log some usage

```python
from afs.continuous import UsageLogger

logger = UsageLogger("~/.context/training/continuous/usage.db")

# Log each model call
record_id = logger.log(
    query="What is 65816 assembly?",
    response="65816 is the CPU used in the SNES...",
    model="nayru-v1",
    quality_score=0.85
)

# Record user feedback
logger.record_feedback(record_id, feedback=1)  # 1=good, -1=bad
```

### 2. Run the continuous learning loop

```bash
# In one terminal, start the loop
python scripts/continuous_loop.py

# In another terminal, check status
python scripts/continuous_loop.py --status
```

### 3. See the magic happen

The loop will:
- Check every hour for retraining conditions
- Generate training data when thresholds are met
- Call your training function
- Deploy as challenger in A/B test
- Auto-promote if better

## Integration with Your App

### Add logging to your inference code

```python
from afs.continuous import ContinuousLearningLoop, LoopConfig

# Initialize once at startup
loop = ContinuousLearningLoop(LoopConfig())

# In your inference handler
def handle_query(query: str) -> str:
    start_time = time.time()

    # Your model inference
    response = model.generate(query)

    # Log usage
    latency_ms = (time.time() - start_time) * 1000
    record_id = loop.log_usage(
        query=query,
        response=response,
        model="nayru-v1",
        latency_ms=latency_ms,
        quality_score=compute_quality(response)
    )

    return response, record_id

# When user gives feedback
def handle_feedback(record_id: str, is_positive: bool):
    loop.record_feedback(record_id, feedback=1 if is_positive else -1)
```

### Define your training function

```python
def train_model(data_path):
    """Your training logic."""
    # 1. Load data
    dataset = load_jsonl(data_path)

    # 2. Train with LoRA
    trainer = Trainer(model, dataset)
    metrics = trainer.train()

    # 3. Save model
    model_path = save_model()

    # 4. Return metrics and path
    return {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "model_path": model_path
    }
```

### Run the loop with your function

```python
# Option 1: Run as daemon
loop.run_loop(train_fn=train_model)

# Option 2: Check manually
result = loop.run_iteration(train_fn=train_model)
if result["retrain_triggered"]:
    print(f"Retrained! Status: {result['retrain']['status']}")
```

## Configuration

Create `config.json`:

```json
{
  "check_interval_seconds": 3600,
  "trigger_config": {
    "min_new_samples": 1000,
    "schedule_interval_hours": 168
  },
  "generator_config": {
    "min_quality_score": 0.7,
    "min_user_feedback": 1
  },
  "ab_test_config": {
    "initial_challenger_traffic": 0.1,
    "promotion_threshold": 0.05
  }
}
```

Run with config:

```bash
python scripts/continuous_loop.py --config config.json
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION INFERENCE                     │
│                                                             │
│  User Query → Model → Response → Log to SQLite             │
│                                    ↓                         │
│                              Record Feedback                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   CONTINUOUS LEARNING LOOP                  │
│                                                             │
│  1. CHECK TRIGGERS (every hour)                            │
│     • 1000+ quality samples?                               │
│     • Weekly schedule?                                      │
│     • Quality drop >10%?                                    │
│                                                             │
│  2. GENERATE TRAINING DATA                                  │
│     • Filter by quality (>0.7)                             │
│     • Filter by feedback (positive only)                   │
│     • Deduplicate                                           │
│                                                             │
│  3. RETRAIN MODEL                                           │
│     • Call your training function                          │
│     • Save new model weights                               │
│                                                             │
│  4. A/B TEST                                                │
│     • Deploy as challenger (10% traffic)                   │
│     • Compare metrics for 48 hours                         │
│                                                             │
│  5. AUTO-PROMOTE                                            │
│     • If challenger >5% better                             │
│     • Promote to champion (100% traffic)                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                      IMPROVED MODEL                         │
│                                                             │
│  New model serves production traffic                        │
│  Cycle repeats...                                           │
└─────────────────────────────────────────────────────────────┘
```

## Metrics Dashboard

Check your stats:

```python
summary = loop.get_status_summary()

print(f"Total usage: {summary['usage_stats']['total']}")
print(f"Feedback rate: {summary['usage_stats']['feedback_rate']*100:.1f}%")
print(f"Avg quality: {summary['usage_stats']['avg_quality_score']:.3f}")
print(f"Total retrains: {summary['status']['total_retrains']}")
print(f"Champion model: {summary['status']['champion_model']}")
```

## FAQ

**Q: How often will it retrain?**
A: When you accumulate 1000+ quality samples, OR weekly, OR if quality drops >10%. Has 24h cooldown between retrains.

**Q: What if the new model is worse?**
A: It stays at 10% traffic. Only auto-promotes if >5% better after 48+ hours.

**Q: Can I test without actual training?**
A: Yes! Use `--no-train` flag to just prepare data:
```bash
python scripts/continuous_loop.py --trigger --no-train
```

**Q: How much disk space?**
A: SQLite DB grows ~1KB per query. 1M queries = ~1GB. Training data cleaned up after successful retrain.

**Q: Can I disable A/B testing?**
A: Yes! Set `enable_ab_testing=False` in config. New models go directly to production.

**Q: What about catastrophic forgetting?**
A: Use rehearsal buffer integration:
```python
DataGeneratorConfig(
    include_existing_data=True,
    existing_data_path="rehearsal_buffer.jsonl"
)
```

## Examples

Run the demo to see everything in action:

```bash
python examples/continuous_learning_demo.py
```

Demos include:
- Basic logging
- Data generation
- Trigger checking
- A/B testing
- Full loop iteration

## Next Steps

1. **Integrate logging** - Add `loop.log_usage()` to your inference code
2. **Collect feedback** - Add thumbs up/down buttons, call `loop.record_feedback()`
3. **Define training function** - Implement `train_model(data_path)`
4. **Start the loop** - Run `python scripts/continuous_loop.py`
5. **Monitor metrics** - Check status regularly with `--status` flag

## Full Documentation

See [CONTINUOUS_LEARNING.md](./CONTINUOUS_LEARNING.md) for complete docs.

## Questions?

Check the tests: `tests/test_continuous_learning.py`
