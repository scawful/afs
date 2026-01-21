# Cost Optimization Module

A comprehensive cost tracking and optimization system for ML training workloads.

## Quick Start

### Basic Usage

```python
from afs.cost import CostAnalyzer, GPUPriceTracker, CostOptimizer
from afs.cost import TrainingMetrics, GPUPrice

# Track GPU prices
tracker = GPUPriceTracker()
prices = tracker.fetch_vastai_prices()
for price in prices:
    tracker.track_price(price)

# Analyze training costs
analyzer = CostAnalyzer()
metrics = TrainingMetrics(
    run_id="my_run_001",
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
print(f"Cost: ${report.total_cost:.2f}")

# Set budget and check status
analyzer.set_budget("bert-base", 1000.0)
alert = analyzer.check_budget("bert-base", report.total_cost)

# Get optimization recommendations
optimizer = CostOptimizer()
rec = optimizer.recommend_batch_size(
    current_batch_size=32,
    gpu_vram_gb=40,
    model_param_count=110_000_000,
    current_throughput=100,
    gpu_price_per_hour=1.50,
    epoch_hours=24.5/3,
)
```

## Components

### 1. GPUPriceTracker (`tracker.py`)

Tracks GPU prices from cloud providers.

**Key Classes:**
- `GPUPrice`: Single GPU price data point
- `PriceHistory`: Historical price tracking
- `PriceAlert`: Price change alerts
- `GPUPriceTracker`: Main tracker class

**Key Methods:**
- `track_price()`: Track a new price point
- `fetch_vastai_prices()`: Fetch latest vast.ai prices
- `get_cheapest_gpu()`: Find cheapest GPU
- `get_price_recommendations()`: Get recommendations based on price history
- `get_alerts()`: Get recent price alerts
- `get_price_statistics()`: Get overall statistics

### 2. CostAnalyzer (`analyzer.py`)

Analyzes training costs and manages budgets.

**Key Classes:**
- `TrainingMetrics`: Training run metrics
- `TrainingCostReport`: Cost analysis report
- `BudgetAlert`: Budget threshold alerts
- `CostAnalyzer`: Main analyzer class

**Key Methods:**
- `analyze_training_run()`: Generate cost report
- `set_budget()`: Set budget limit for model
- `check_budget()`: Check if budget threshold crossed
- `get_cost_comparison()`: Compare costs across models
- `forecast_cost()`: Forecast future costs
- `get_roi_analysis()`: Calculate ROI

### 3. CostOptimizer (`optimizer.py`)

Generates cost optimization recommendations.

**Key Classes:**
- `OptimizationRecommendation`: A recommendation object
- `CostOptimizer`: Main optimizer class

**Key Methods:**
- `recommend_batch_size()`: Batch size optimization
- `recommend_early_stopping()`: Early stopping recommendation
- `recommend_epoch_count()`: Epoch count optimization
- `recommend_dataset_size()`: Dataset size recommendation
- `recommend_compute_tier()`: Compute tier switching
- `get_all_recommendations()`: Get recent recommendations
- `get_high_confidence_recommendations()`: Filter by confidence

## Data Models

### GPUPrice
```python
@dataclass
class GPUPrice:
    gpu_name: str
    provider: str
    price_per_hour: float
    availability: int = 0
    cuda_version: Optional[str] = None
    vram_gb: Optional[int] = None
    instance_id: Optional[str] = None
    region: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

### TrainingMetrics
```python
@dataclass
class TrainingMetrics:
    run_id: str
    model_name: str
    num_samples: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    total_duration_hours: float
    gpu_name: str
    gpu_price_per_hour: float
    validation_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    tokens_processed: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

### TrainingCostReport
```python
@dataclass
class TrainingCostReport:
    run_id: str
    model_name: str
    total_cost: float
    gpu_hours: float
    cost_per_sample: float
    cost_per_epoch: float
    cost_per_token: Optional[float] = None
    efficiency_score: float = 0.0
    cost_per_accuracy_point: Optional[float] = None
    metrics: Optional[TrainingMetrics] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

### OptimizationRecommendation
```python
@dataclass
class OptimizationRecommendation:
    category: str
    title: str
    description: str
    estimated_savings: float
    estimated_time_saved: Optional[float] = None
    confidence: float = 0.8
    action: str = ""
    risks: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

## Storage

All data is persisted to `~/.context/training/`:

```
~/.context/training/
├── prices/
│   └── price_history.json      # Historical GPU prices
├── cost_history.json           # Training cost reports
├── budgets.json                # Budget limits
└── recommendations.json        # Optimization recommendations
```

## Command-Line Interface

Use `scripts/cost_report.py` for CLI access:

```bash
# Show summary
python scripts/cost_report.py --summary

# Fetch prices
python scripts/cost_report.py --fetch-prices

# Show prices
python scripts/cost_report.py --prices

# Analyze model
python scripts/cost_report.py --model bert-base

# Show recommendations
python scripts/cost_report.py --recommendations

# Show budget status
python scripts/cost_report.py --budget

# Forecast costs
python scripts/cost_report.py --forecast bert-base 5

# Set budget
python scripts/cost_report.py --set-budget bert-base 1000.0
```

## Examples

See `/Users/scawful/src/lab/afs/docs/COST_OPTIMIZATION.md` for detailed examples.

## Testing

Run tests with:
```bash
python3 -m pytest tests/test_cost_system.py -v
```

## Integration

To integrate with your training pipeline:

```python
import time
from afs.cost import CostAnalyzer, TrainingMetrics

analyzer = CostAnalyzer()
analyzer.set_budget("my_model", 1000.0)

start = time.time()
# ... training code ...
duration = (time.time() - start) / 3600

metrics = TrainingMetrics(
    run_id="run_001",
    model_name="my_model",
    num_samples=len(train_dataset),
    num_epochs=3,
    batch_size=32,
    learning_rate=2e-5,
    total_duration_hours=duration,
    gpu_name="A100",
    gpu_price_per_hour=1.50,
    test_accuracy=accuracy,
)

report = analyzer.analyze_training_run(metrics)
print(f"Training cost: ${report.total_cost:.2f}")

alert = analyzer.check_budget("my_model", report.total_cost)
if alert:
    print(f"Budget alert: {alert.message}")
```

## Future Enhancements

- Integration with Weights & Biases
- Real-time dashboard
- Multi-GPU cluster tracking
- Automatic training pause/resume on price changes
- ML-based cost prediction
- Team cost allocation
- Cloud provider API integration

## License

MIT

## Contact

For issues or questions, see the main AFS documentation.
