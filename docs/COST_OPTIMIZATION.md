# Cost Optimization and Analysis System

A comprehensive system for minimizing training costs while maximizing model quality. This system provides GPU price tracking, detailed cost analysis, budget management, and intelligent optimization recommendations.

## Features

### 1. GPU Price Tracking

Track GPU prices from various cloud providers (vast.ai, Lambda, Paperspace, etc.) with historical data and trend analysis.

**Key capabilities:**
- Hourly price scraping from vast.ai
- Price trend analysis (24h, 7d, 30d)
- Automatic alerts on price drops (>10%) and surges (>20%)
- Price recommendations based on availability and cost
- Historical price data persistence

**Example:**
```python
from afs.cost import GPUPriceTracker

tracker = GPUPriceTracker()

# Fetch latest prices
prices = tracker.fetch_vastai_prices()

# Find cheapest GPU
cheapest = tracker.get_cheapest_gpu(gpu_type="A100")
print(f"Cheapest A100: {cheapest[0].price_per_hour:.3f}/hr")

# Get recommendations
recs = tracker.get_price_recommendations(gpu_type="RTX 4090", max_price=0.50)
for gpu, reason in recs[:5]:
    print(f"{gpu.gpu_name}: ${gpu.price_per_hour:.3f}/hr - {reason}")
```

### 2. Cost Analysis

Detailed analysis of training costs, including per-sample, per-epoch, and per-token metrics.

**Key metrics:**
- Total training cost
- Cost per sample trained
- Cost per epoch
- Cost per million tokens
- Cost per accuracy point
- Efficiency score (0-1 based on accuracy per dollar)

**Example:**
```python
from afs.cost import CostAnalyzer, TrainingMetrics

analyzer = CostAnalyzer()

metrics = TrainingMetrics(
    run_id="run_001",
    model_name="bert-base",
    num_samples=100000,
    num_epochs=3,
    batch_size=32,
    learning_rate=2e-5,
    total_duration_hours=24.5,
    gpu_name="A100",
    gpu_price_per_hour=1.50,
    test_accuracy=0.94,
    tokens_processed=50_000_000,
)

report = analyzer.analyze_training_run(metrics)

print(f"Total cost: ${report.total_cost:.2f}")
print(f"Cost per sample: ${report.cost_per_sample:.6f}")
print(f"Cost per token: ${report.cost_per_token:.9f}")
print(f"Efficiency score: {report.efficiency_score:.2f}")
```

### 3. Budget Management

Set and monitor budget limits for different models with automatic alerts at thresholds.

**Threshold alerts:**
- 50% of budget used: informational
- 75% of budget used: warning
- 90% of budget used: critical warning
- 100% of budget: exceeded alert

**Example:**
```python
analyzer.set_budget("my_model", 1000.0)  # $1000 budget

# Check during training
alert = analyzer.check_budget("my_model", 750.0)  # 75% used
if alert:
    print(f"⚠ {alert.message}")
    # Can trigger automatic training stop
```

### 4. Cost Optimization Recommendations

Intelligent recommendations for optimizing training costs based on analysis patterns.

**Optimization categories:**

#### Batch Size Optimization
Recommends increasing batch size to improve throughput and reduce training time.
- Considers GPU VRAM constraints
- Estimates throughput improvement
- Calculates cost savings from reduced training time
- Includes risk assessment

#### Early Stopping
Recommends stopping training when validation loss plateaus.
- Monitors validation loss trends
- Sets patience threshold
- Calculates savings from stopped epochs
- Prevents unnecessary computation

#### Epoch Count Optimization
Identifies when accuracy gains plateau and recommends reducing epochs.
- Detects diminishing returns
- Calculates "knee point" in accuracy curve
- Estimates savings from fewer epochs
- Balances quality vs. cost

#### Dataset Size Optimization
Recommends reducing dataset size while maintaining accuracy.
- Builds accuracy vs. data size curves
- Identifies threshold (95% of best accuracy)
- Suggests dataset reduction
- Includes generalization risk assessment

#### Compute Tier Optimization
Recommends switching to more cost-effective GPU tiers.
- Compares cost-per-throughput ratios
- Calculates monthly savings
- Identifies 20%+ efficiency gains
- Includes capability compatibility risks

**Example:**
```python
from afs.cost import CostOptimizer

optimizer = CostOptimizer()

# Get batch size recommendation
rec = optimizer.recommend_batch_size(
    current_batch_size=32,
    gpu_vram_gb=40,
    model_param_count=12_000_000,
    current_throughput=100,  # samples/sec
    gpu_price_per_hour=1.50,
    epoch_hours=2.5,
)

if rec:
    print(f"{rec.title}")
    print(f"Estimated savings: ${rec.estimated_savings:.2f}")
    print(f"Confidence: {rec.confidence:.0%}")
    print(f"Action: {rec.action}")
    if rec.risks:
        print(f"Risks: {', '.join(rec.risks)}")

# Get all high-confidence recommendations
recs = optimizer.get_high_confidence_recommendations(confidence_threshold=0.8)
total_savings = optimizer.get_total_potential_savings()
print(f"Total potential savings: ${total_savings:.2f}")
```

### 5. Cost Reports and Forecasting

Generate comprehensive reports and forecast future costs.

**Reports include:**
- Cost comparison across models
- ROI analysis (accuracy improvement per dollar)
- Budget status dashboard
- Price trend analysis
- Recommendation summary

**Forecasting:**
- Project costs for planned training runs
- Compare against budget limits
- Identify overage risks
- Plan spending

**Example:**
```python
# Cost comparison
comparison = analyzer.get_cost_comparison()
for model, stats in comparison.items():
    print(f"{model}: ${stats['total_cost']:.2f} ({stats['runs']} runs)")

# ROI analysis
roi = analyzer.get_roi_analysis("my_model")
print(f"Investment: ${roi['total_investment']:.2f}")
print(f"Accuracy improvement: {roi['accuracy_improvement']:.2%}")
print(f"Improvement per dollar: {roi['improvement_per_dollar']:.4f}")

# Cost forecast
forecast = analyzer.forecast_cost("my_model", planned_runs=5)
print(f"Estimated cost for 5 runs: ${forecast['estimated_total']:.2f}")
print(f"Status: {forecast['budget_status']}")
```

## Command-Line Usage

### Generate Summary Report
```bash
python scripts/cost_report.py --summary
```

### Show GPU Prices
```bash
python scripts/cost_report.py --prices
```

### Fetch Latest Prices from vast.ai
```bash
python scripts/cost_report.py --fetch-prices
```

### Analyze Specific Model
```bash
python scripts/cost_report.py --model bert-base
```

### Show Optimization Recommendations
```bash
python scripts/cost_report.py --recommendations
```

### Monitor Budget Status
```bash
python scripts/cost_report.py --budget
```

### Forecast Future Costs
```bash
python scripts/cost_report.py --forecast my_model 5
```

### Set Budget Limit
```bash
python scripts/cost_report.py --set-budget my_model 1000.0
```

## Integration with Training Pipeline

### Track Costs During Training

```python
from afs.cost import CostAnalyzer, TrainingMetrics
import time

analyzer = CostAnalyzer()

# Set budget
analyzer.set_budget("my_model", 500.0)

# During/after training
start_time = time.time()
gpu_price_per_hour = 1.50

# ... training code ...

elapsed_hours = (time.time() - start_time) / 3600

metrics = TrainingMetrics(
    run_id="run_001",
    model_name="my_model",
    num_samples=100000,
    num_epochs=3,
    batch_size=32,
    learning_rate=2e-5,
    total_duration_hours=elapsed_hours,
    gpu_name="A100",
    gpu_price_per_hour=gpu_price_per_hour,
    test_accuracy=0.94,
)

report = analyzer.analyze_training_run(metrics)
print(f"Training cost: ${report.total_cost:.2f}")

# Check budget
alert = analyzer.check_budget("my_model", report.total_cost)
if alert:
    print(f"Budget alert: {alert.message}")
```

### Generate Recommendations After Training

```python
from afs.cost import CostOptimizer

optimizer = CostOptimizer()

# Generate recommendations based on training metrics
recs = []

# Batch size recommendation
rec = optimizer.recommend_batch_size(
    current_batch_size=32,
    gpu_vram_gb=40,
    model_param_count=metrics.model_param_count,
    current_throughput=metrics.num_samples / metrics.total_duration_hours / 3600,
    gpu_price_per_hour=gpu_price_per_hour,
    epoch_hours=elapsed_hours / metrics.num_epochs,
)
if rec:
    recs.append(rec)

# Early stopping recommendation
rec = optimizer.recommend_early_stopping(
    validation_loss_history=validation_losses,
    gpu_price_per_hour=gpu_price_per_hour,
    hours_per_epoch=elapsed_hours / metrics.num_epochs,
)
if rec:
    recs.append(rec)

# Print recommendations
for rec in recs:
    print(f"\n{rec.title}")
    print(f"Savings: ${rec.estimated_savings:.2f}")
    print(f"Action: {rec.action}")
```

## Data Storage

All data is stored in `~/.context/training/` with the following structure:

```
~/.context/training/
├── prices/
│   └── price_history.json          # Historical GPU prices
├── cost_history.json               # Training cost reports
├── budgets.json                    # Budget limits by model
└── recommendations/
    └── recommendations.json        # Optimization recommendations
```

Files are automatically created and updated as you use the system.

## Architecture

### GPUPriceTracker
Manages GPU price data from various providers:
- Fetches prices from vast.ai API
- Tracks historical price data
- Detects price anomalies
- Generates recommendations

### CostAnalyzer
Analyzes training costs and generates reports:
- Calculates cost metrics (per-sample, per-epoch, etc.)
- Tracks efficiency scores
- Manages budget limits
- Forecasts future costs
- Performs ROI analysis

### CostOptimizer
Generates optimization recommendations:
- Analyzes batch size tradeoffs
- Detects early stopping opportunities
- Identifies epoch diminishing returns
- Recommends dataset size reductions
- Suggests compute tier switches

## Performance Considerations

### Memory Usage
- Price histories are kept in memory during session
- Typically <100MB for 1000+ GPU types over 1 year
- Disk storage: ~1MB per 10k price points

### API Rate Limiting
- vast.ai API: ~60 requests/minute
- Recommend fetching prices hourly via scheduled task

### Recommendation Accuracy
- Recommendations are estimates based on training patterns
- Actual savings depend on specific training characteristics
- Confidence scores indicate reliability
- Use high-confidence (>0.8) recommendations first

## Best Practices

1. **Set budgets early**: Define budget limits before starting training
2. **Track prices regularly**: Fetch prices hourly for accurate recommendations
3. **Log detailed metrics**: Record all training parameters for better analysis
4. **Review recommendations**: Check recommendations after each training run
5. **Validate savings**: Test recommendations on small runs before full deployment
6. **Monitor alerts**: Set up monitoring for budget and price alerts

## Examples

### Complete Training Cost Optimization Workflow

```python
from afs.cost import (
    CostAnalyzer, CostOptimizer, GPUPriceTracker,
    TrainingMetrics, GPUPrice
)

# Initialize
tracker = GPUPriceTracker()
analyzer = CostAnalyzer()
optimizer = CostOptimizer()

# 1. Fetch latest prices
prices = tracker.fetch_vastai_prices()
for price in prices:
    tracker.track_price(price)

# 2. Find cheapest GPU
cheapest = tracker.get_cheapest_gpu(gpu_type="A100")
print(f"Using: {cheapest[0].gpu_name} @ ${cheapest[0].price_per_hour:.3f}/hr")

# 3. Set budget
analyzer.set_budget("my_model", 500.0)

# 4. Run training (pseudo-code)
# metrics = train_model(...)

# 5. Analyze cost
# report = analyzer.analyze_training_run(metrics)

# 6. Generate recommendations
# batch_size_rec = optimizer.recommend_batch_size(...)
# early_stop_rec = optimizer.recommend_early_stopping(...)

# 7. Review and plan next iteration
# comparison = analyzer.get_cost_comparison()
# forecast = analyzer.forecast_cost("my_model", planned_runs=5)
```

## Future Enhancements

- [ ] Integration with Weights & Biases for automatic logging
- [ ] Real-time cost monitoring dashboard
- [ ] Multi-GPU cluster cost tracking
- [ ] Automatic training pause/resume based on price changes
- [ ] ML-based cost prediction using historical patterns
- [ ] Cost allocation across team members
- [ ] Integration with cloud provider APIs (AWS, GCP, Azure)
- [ ] Automated optimization rule engine

## Related Documentation

- [Training Pipeline Documentation](TRAINING.md)
- [Architecture Overview](../ARCHITECTURE.md)
- [Configuration Guide](../configs/README.md)
