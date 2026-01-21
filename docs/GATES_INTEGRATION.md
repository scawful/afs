# Quality Gates Integration with AFS

This document describes how to integrate the quality gates system with the existing AFS CLI and workflows.

## CLI Integration

The quality gates system is accessible via the `afs gates` command group:

```bash
afs gates check --context production --model veran:v6.2 ...
afs gates approve --model veran:v6.2 --context production ...
afs gates reject --model veran:v6.2 --context production --reason "..."
afs gates status --model veran:v6.2 ...
afs gates history --model veran:v6.2 ...
afs gates report --model veran:v6.2 --context production ...
```

### Registering with AFS CLI

The gates CLI is automatically registered in `/Users/scawful/src/lab/afs/src/afs/cli/__init__.py`:

```python
from . import gates  # Add this import

def build_parser():
    ...
    gates.register_parsers(subparsers)  # Register gates commands
```

Alternatively, it can be invoked directly:

```bash
python -m afs.gates check --context production --model veran:v6.2 ...
```

## Workflow Integration

### 1. Training Pipeline Integration

During training, after model evaluation:

```python
from afs.gates import QualityGate
from afs.training import ModelTrainer

trainer = ModelTrainer()
model = trainer.train(...)

# Evaluate model
eval_results = trainer.evaluate(model)

# Run quality gates
gate = QualityGate.development()  # Or staging/production
test_metrics = extract_test_metrics(eval_results)
model_metrics = extract_model_metrics(eval_results)

report = gate.check_all(
    model_name="veran",
    model_version=f"v{model.version}",
    test_metrics=test_metrics,
    model_metrics=model_metrics,
)

if not report.all_passed():
    logger.error(f"Quality gates failed: {report.summary()}")
    # Decide whether to continue or reject
```

### 2. Pre-Merge Validation

In CI/CD pipeline (GitHub Actions):

```yaml
name: Pre-Merge Quality Gates

on:
  pull_request:
    paths:
      - 'src/afs/**'
      - 'tests/**'

jobs:
  quality-gates:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov

      - name: Run tests with coverage
        run: |
          pytest --cov=afs --cov-report=json tests/

      - name: Check quality gates
        run: |
          python -m afs.gates check \
            --context staging \
            --model afs:pr-${{ github.event.pull_request.number }} \
            --test-metrics coverage.json \
            --security-scan security-scan.json
        continue-on-error: false
```

### 3. Pre-Deployment Approval

Before deploying to production:

```python
from afs.gates.registry_integration import DeploymentController

controller = DeploymentController()

# Check deployment eligibility
can_deploy, reason = controller.can_deploy(
    "veran",
    os.environ["MODEL_VERSION"],
    "production"
)

if not can_deploy:
    print(f"Cannot deploy: {reason}")
    sys.exit(1)

# Execute deployment
success = controller.execute_deployment(
    "veran",
    os.environ["MODEL_VERSION"],
    "production",
    deployment_target="prod-cluster",
)

if not success:
    sys.exit(1)

print("Deployment complete")
```

### 4. Post-Deployment Monitoring

After deployment, continuously monitor for regressions:

```python
from afs.gates import QualityGate, ModelMetrics

# Get production metrics
prod_metrics = get_production_metrics()
baseline_metrics = get_previous_version_metrics()

gate = QualityGate.production()
result = gate.check_model_quality(prod_metrics, baseline_metrics)

if not result.passed:
    # Anomaly detected
    controller.rollback(
        "veran",
        os.environ["MODEL_VERSION"],
        "production",
        reason=f"Regression detected: {result.message}"
    )
    alerts.send_critical("Production model rolled back")
```

## Integration with Existing Modules

### Quality Metrics

The gates system works with existing quality metrics:

```python
from afs.quality.metrics import QualityMetrics
from afs.gates import QualityGate, ModelMetrics

# Compute quality scores
quality_analyzer = QualityMetrics(domain="code")
instruction_clarity = quality_analyzer.compute_instruction_clarity(code)
output_correctness = quality_analyzer.compute_output_correctness(output)

# Convert to gate format
model_metrics = ModelMetrics(
    quality_score=output_correctness.overall_score(),
    accuracy=...,
    f1_score=...,
)

# Run gates
gate = QualityGate.staging()
result = gate.check_model_quality(model_metrics)
```

### Model Registry

Gates integrate with the existing registry:

```python
from afs.registry.database import ModelRegistry
from afs.gates.registry_integration import RegistryIntegration

# Existing registry
registry_db = ModelRegistry()
version = registry_db.get_version("veran", "v6.2")

# Gates approval
gates_registry = RegistryIntegration()
approval = gates_registry.get_approval_status("veran", "v6.2", "production")

# Check if deployable
if approval and approval.approved and version.status == "completed":
    # Ready to deploy
    pass
```

### Continuous Learning

Gates can prevent bad examples from being added to the training data:

```python
from afs.continuous.learning import ExampleValidator
from afs.gates import QualityGate

validator = ExampleValidator()
gate = QualityGate.development()

for example in new_examples:
    # Validate quality
    quality_score = validator.compute_quality(example)

    # Gate check
    metrics = ModelMetrics(quality_score=quality_score)
    result = gate.check_model_quality(metrics)

    if result.passed:
        # Add to training data
        store_example(example)
    else:
        # Reject with reason
        logger.warning(f"Example rejected: {result.message}")
```

## Custom Gate Policies

Define organization-specific gate policies:

```python
from afs.gates import QualityGate, GateThresholds

# Org-wide strict standards
ORG_PRODUCTION_THRESHOLDS = GateThresholds(
    min_test_pass_rate=0.99,
    min_code_coverage=0.95,
    min_quality_score=0.90,
    max_regression=0.01,
    max_latency_increase=0.05,
    max_critical_vulnerabilities=0,
    max_high_vulnerabilities=0,
    max_medium_vulnerabilities=0,
)

# Team-specific relaxed thresholds for experimental features
EXPERIMENTAL_FEATURE_THRESHOLDS = GateThresholds(
    min_test_pass_rate=0.85,
    min_code_coverage=0.70,
    min_quality_score=0.60,
    max_regression=0.30,
)

# Usage
if is_experimental:
    gate = QualityGate.staging(EXPERIMENTAL_FEATURE_THRESHOLDS)
else:
    gate = QualityGate.production(ORG_PRODUCTION_THRESHOLDS)
```

## Dashboard Integration

Display gate status in the AFS dashboard:

```python
from afs.gates.registry_integration import RegistryIntegration

registry = RegistryIntegration()

# Get all models
models = ["veran", "din", "nayru", "farore"]

for model_name in models:
    # Get approval status per context
    for context in ["development", "staging", "production"]:
        approvals = registry.list_approved_versions(model_name, context)

        # Display in dashboard
        dashboard_data[model_name][context] = {
            "approved_versions": approvals,
            "latest_approval": registry.get_approval_status(
                model_name,
                approvals[-1] if approvals else None,
                context
            ),
        }
```

## Notification Integration

Send alerts on gate failures:

```python
from afs.notifications import NotificationManager
from afs.gates import QualityGate

gate = QualityGate.production()
notifier = NotificationManager()

def on_gates_blocked(report):
    notifier.send(
        channel="slack",
        destination="#deployments",
        level="critical",
        message=f"Quality gates blocked deployment of {report.model_name}:{report.model_version}",
        details={
            "failed_checks": [
                f"{c.gate_name}: {c.message}"
                for c in report.failed_checks()
            ],
            "context": report.context.value,
            "timestamp": report.timestamp,
        },
    )

def on_production_regression(result):
    notifier.send(
        channel="pagerduty",
        level="warning",
        message=f"Production regression detected: {result.message}",
        service="model-deployments",
    )

gate.register_callback("gates_blocked", on_gates_blocked)
gate.register_callback("gate_failed", on_production_regression)
```

## Testing Integration

Test gate enforcement:

```python
from afs.gates import QualityGate, TestMetrics
import pytest

def test_model_meets_production_standards():
    """Verify model meets production quality gates."""
    gate = QualityGate.production()

    # Get model metrics from evaluation
    metrics = load_model_metrics()
    baseline = load_baseline_metrics()

    # Check gates
    report = gate.check_all(
        model_name="veran",
        model_version="v6.2",
        model_metrics=metrics,
        baseline_model_metrics=baseline,
    )

    # Assert all gates passed
    assert report.all_passed(), f"Gates failed: {report.failed_checks()}"

def test_gates_catch_regressions():
    """Verify gates catch model regressions."""
    gate = QualityGate.production()

    # Create regressed metrics
    bad_metrics = ModelMetrics(
        quality_score=0.75,  # Below production threshold
        baseline_quality_score=0.85,
    )

    # Should fail
    report = gate.check_all(
        model_name="veran",
        model_version="v6.2-bad",
        model_metrics=bad_metrics,
    )

    assert not report.all_passed()
    assert len(report.failed_checks()) > 0
```

## Performance Considerations

Gate checks are lightweight:

- Single gate check: < 50ms
- Full report: < 500ms
- File I/O: Network-dependent
- Memory: < 10MB per report

For CI/CD pipelines, run gates in parallel with other checks:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: pytest --cov ...

  gates:
    runs-on: ubuntu-latest
    steps:
      - run: python -m afs.gates check ...

  security:
    runs-on: ubuntu-latest
    steps:
      - run: trivy scan ...

  build:
    needs: [test, gates, security]
    runs-on: ubuntu-latest
    steps:
      - run: docker build ...
```

## Configuration

Gate configuration can be stored in project config:

```toml
# pyproject.toml or gates.toml

[tool.afs.gates]

[tool.afs.gates.development]
min_test_pass_rate = 0.80
min_code_coverage = 0.60
min_quality_score = 0.50

[tool.afs.gates.staging]
min_test_pass_rate = 0.95
min_code_coverage = 0.80
min_quality_score = 0.70

[tool.afs.gates.production]
min_test_pass_rate = 0.98
min_code_coverage = 0.90
min_quality_score = 0.85
max_critical_vulnerabilities = 0
```

## Troubleshooting

### Gates failing unexpectedly

1. Check threshold configuration
2. Verify metrics are being computed correctly
3. Compare against baseline metrics
4. Review gate reports in `.quality-gates/`

### Integration issues

1. Ensure AFS CLI is properly installed
2. Check that metrics files are in correct JSON format
3. Verify file permissions for report output
4. Check logs for detailed error messages

### Performance issues

1. Run gates check in parallel with other CI tasks
2. Consider caching metrics between runs
3. Use local file integration for faster turnaround

## See Also

- Quality Metrics: `afs.quality.metrics`
- Model Registry: `afs.registry.database`
- Continuous Learning: `afs.continuous.learning`
- Notifications: `afs.notifications`
- CLI: `afs.cli`
