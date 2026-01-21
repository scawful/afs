# Quality Gate System

Automated quality gate enforcement to prevent bad code and models from reaching production. The system provides configurable rules for different deployment contexts, integration with CI/CD pipelines, and comprehensive reporting.

## Overview

The quality gate system enforces multiple dimensions of quality before deployment:

- **Test Metrics**: Pass rate and code coverage
- **Model Quality**: Quality score, regression vs baseline, latency
- **Security**: Vulnerability scanning and assessment
- **Performance**: Memory usage, throughput
- **Deployment Approval**: Registry-based version approval

## Features

### 1. Context-Aware Gates

Different strictness levels for different environments:

```python
from afs.gates import QualityGate

# Development: Fast iteration, relaxed
dev_gate = QualityGate.development()  # 80% tests, 60% coverage

# Staging: Standard checks
staging_gate = QualityGate.staging()  # 95% tests, 80% coverage

# Production: Strict
prod_gate = QualityGate.production()  # 98% tests, 90% coverage
```

### 2. Configurable Thresholds

Customize thresholds per context:

```python
from afs.gates import QualityGate, GateThresholds

custom_thresholds = GateThresholds(
    min_test_pass_rate=0.92,
    min_code_coverage=0.85,
    min_quality_score=0.75,
    max_regression=0.03,
    max_latency_increase=0.15,
    max_critical_vulnerabilities=0,
    max_high_vulnerabilities=1,
    max_medium_vulnerabilities=5,
)

gate = QualityGate.staging(custom_thresholds)
```

### 3. Comprehensive Checks

#### Test Metrics
```python
from afs.gates import TestMetrics

metrics = TestMetrics(
    total_tests=1000,
    passed_tests=980,
    failed_tests=20,
    skipped_tests=0,
    duration_seconds=120.0,
    coverage_percent=88.5,
)

result = gate.check_tests(metrics)
print(f"Tests: {result.passed} - {result.message}")
```

#### Model Quality
```python
from afs.gates import ModelMetrics

current = ModelMetrics(
    quality_score=0.87,
    accuracy=0.92,
    f1_score=0.89,
    perplexity=25.3,
    latency_ms=125.0,
    throughput_tokens_per_sec=45.2,
    memory_mb=2048.0,
)

baseline = ModelMetrics(
    quality_score=0.85,
    latency_ms=120.0,
    memory_mb=2000.0,
)

result = gate.check_model_quality(current, baseline)
print(f"Model Quality: {result.passed}")
print(f"  Regression: {current.regression_percent():.2%}")
print(f"  Latency increase: {current.latency_increase_percent():.2%}")
```

#### Security Scans
```python
from afs.gates import SecurityScanResults

scan = SecurityScanResults(
    critical_vulnerabilities=0,
    high_vulnerabilities=1,
    medium_vulnerabilities=8,
    low_vulnerabilities=15,
    scan_tool="trivy",
)

result = gate.check_security(scan)
print(f"Security: {result.passed} - Total: {scan.total_vulnerabilities()}")
```

### 4. Complete Report

Run all checks at once:

```python
report = gate.check_all(
    model_name="veran",
    model_version="v6.2",
    test_metrics=metrics,
    model_metrics=current,
    security_results=scan,
    baseline_model_metrics=baseline,
)

print(gate.summary_string(report))
if not report.all_passed():
    for check in report.failed_checks():
        print(f"FAILED: {check.gate_name} - {check.message}")
```

## CI/CD Integration

### GitHub Actions

Automatically report gate results in GitHub Actions:

```python
from afs.gates.ci_integration import GitHubActionsIntegration

integration = GitHubActionsIntegration()

# Report results
integration.report(report)

# Block merge on failure
if not report.all_passed():
    integration.block_merge(report, "Quality gates failed")
    sys.exit(1)

# Approve deployment
integration.approve_deployment("veran", "v6.2")
```

Results appear as:
- Job outputs
- Check annotations
- Step summary

### GitHub Actions Workflow Example

```yaml
name: Quality Gates

on: [pull_request, push]

jobs:
  quality-gates:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run tests
        run: pytest --cov=afs tests/

      - name: Check quality gates
        run: |
          python -m afs.gates check \
            --context staging \
            --model veran:v6 \
            --test-metrics test-metrics.json \
            --metrics model-metrics.json \
            --security-scan security-scan.json
        continue-on-error: false
```

### Jenkins Integration

```python
from afs.gates.ci_integration import JenkinsIntegration

integration = JenkinsIntegration()
integration.report(report)

# Creates:
# - report-<timestamp>.json
# - junit-gates.xml (parseable by Jenkins)
```

### Local File Integration

For local development and testing:

```python
from afs.gates.ci_integration import LocalFileIntegration

integration = LocalFileIntegration(".quality-gates")
integration.report(report)

# Creates:
# - .quality-gates/report-<timestamp>.json
# - .quality-gates/approvals.jsonl
# - .quality-gates/blocks.jsonl
```

## Model Registry Integration

Manage version approval status based on quality gate results:

### Approve Versions

```python
from afs.gates.registry_integration import RegistryIntegration

registry = RegistryIntegration("models/registry")

# Approve based on gate report
success = registry.approve_version(
    report,
    approved_by="ci-system",
    notes="Production deployment approved",
)
```

### Check Approval Status

```python
# Single context
status = registry.get_approval_status("veran", "v6.2", "production")
if status and status.approved:
    print(f"Approved on {status.timestamp}")

# Full history
history = registry.get_approval_history("veran", "v6.2")
for record in history:
    status = "✓" if record.approved else "✗"
    print(f"{record.context:12} {status} {record.timestamp}")
```

### List Approved Versions

```python
approved = registry.list_approved_versions("veran", "production")
print(f"Approved for production: {approved}")
```

## Deployment Control

Manage deployments with automatic validation:

```python
from afs.gates.registry_integration import DeploymentController

controller = DeploymentController()

# Check if version can be deployed
can_deploy, reason = controller.can_deploy("veran", "v6.2", "production")
if not can_deploy:
    print(f"Cannot deploy: {reason}")
    sys.exit(1)

# Execute deployment
success = controller.execute_deployment(
    "veran", "v6.2", "production",
    deployment_target="prod-cluster"
)

# If needed, rollback
if detection.anomaly_detected():
    controller.rollback("veran", "v6.2", "production", reason="Anomaly detected")
```

## CLI Commands

### Check Gates

```bash
# Basic check
python -m afs.gates check \
  --context production \
  --model veran:v6.2 \
  --test-metrics test-metrics.json \
  --metrics model-metrics.json

# With baseline comparison
python -m afs.gates check \
  --context production \
  --model veran:v6.2 \
  --metrics model-metrics.json \
  --baseline baseline-metrics.json \
  --security-scan security-scan.json

# Strict mode
python -m afs.gates check \
  --context staging \
  --model veran:v6.2 \
  --test-metrics test-metrics.json \
  --strict
```

### Approve Versions

```bash
# Basic approval
python -m afs.gates approve \
  --model veran:v6.2 \
  --context production \
  --approved-by alice@example.com \
  --notes "Passed all production gates"

# From report
python -m afs.gates approve \
  --model veran:v6.2 \
  --context production \
  --report gate-report.json
```

### Reject Versions

```bash
python -m afs.gates reject \
  --model veran:v6.2 \
  --context production \
  --reason "Security vulnerabilities found" \
  --rejected-by alice@example.com
```

### Check Status

```bash
# Single context
python -m afs.gates status \
  --model veran:v6.2 \
  --context production

# All contexts
python -m afs.gates status \
  --model veran:v6.2

# JSON output
python -m afs.gates status \
  --model veran:v6.2 \
  --json
```

### View History

```bash
python -m afs.gates history \
  --model veran:v6.2 \
  --json
```

## Event Callbacks

Register callbacks for gate events:

```python
from afs.gates import QualityGate

gate = QualityGate.production()

def on_gate_passed(result):
    """Handle passed gate."""
    logger.info(f"Gate passed: {result.gate_name}")
    metrics.increment(f"gates.{result.gate_name}.passed")

def on_gate_failed(result):
    """Handle failed gate."""
    logger.error(f"Gate failed: {result.gate_name} - {result.message}")
    metrics.increment(f"gates.{result.gate_name}.failed")

def on_gates_blocked(report):
    """Handle blocked deployment."""
    logger.critical(f"Deployment blocked: {report.model_name}:{report.model_version}")
    alerts.send_pagerduty("Quality gate blocked deployment")

gate.register_callback("gate_passed", on_gate_passed)
gate.register_callback("gate_failed", on_gate_failed)
gate.register_callback("gates_blocked", on_gates_blocked)

# Callbacks triggered during checks
report = gate.check_all(...)
```

## Data Models

### GateThresholds
Configuration thresholds for quality gates:

```python
@dataclass
class GateThresholds:
    min_test_pass_rate: float = 0.95
    min_code_coverage: float = 0.80
    min_quality_score: float = 0.70
    max_regression: float = 0.05
    max_latency_increase: float = 0.20
    max_critical_vulnerabilities: int = 0
    max_high_vulnerabilities: int = 2
    max_medium_vulnerabilities: int = 10
    max_memory_increase_percent: float = 15.0
    min_throughput_tokens_per_sec: float = 10.0
```

### TestMetrics
Test execution results:

```python
@dataclass
class TestMetrics:
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    duration_seconds: float
    coverage_percent: float
```

### ModelMetrics
Model quality and performance:

```python
@dataclass
class ModelMetrics:
    quality_score: float
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    perplexity: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    memory_mb: Optional[float] = None
    baseline_quality_score: Optional[float] = None
    baseline_latency_ms: Optional[float] = None
    baseline_memory_mb: Optional[float] = None
```

### SecurityScanResults
Security vulnerability assessment:

```python
@dataclass
class SecurityScanResults:
    critical_vulnerabilities: int = 0
    high_vulnerabilities: int = 0
    medium_vulnerabilities: int = 0
    low_vulnerabilities: int = 0
    scan_timestamp: str = ...
    scan_tool: str = "unknown"
```

## Best Practices

### 1. Progressive Validation
```python
# Validate in stages
dev_gate = QualityGate.development()  # Catch obvious issues
staging_gate = QualityGate.staging()   # Standard validation
prod_gate = QualityGate.production()   # Strict validation
```

### 2. Baseline Tracking
```python
# Always compare against baseline
result = gate.check_model_quality(current_metrics, baseline_metrics)

# Monitor regression over time
if result.regression_percent() > 0.03:
    logger.warning("Quality regression detected")
```

### 3. Comprehensive Reports
```python
# Collect all metrics before gating
report = gate.check_all(
    model_name=name,
    model_version=version,
    test_metrics=test_data,      # From test runner
    model_metrics=eval_data,      # From evaluation
    security_results=scan_data,   # From security tools
    baseline_model_metrics=baseline,  # From previous version
)

# Use full report for decision making
if not report.all_passed():
    for check in report.failed_checks():
        print(f"Issue: {check.message}")
```

### 4. Notification Integration
```python
def on_production_blocked(report):
    """Send alerts on production gate failure."""
    channels.slack.post(
        channel="#deployments",
        text=f"🚫 Production deployment blocked for {report.model_name}",
        details=report.summary(),
    )
    channels.pagerduty.trigger(
        service="model-deployments",
        severity="critical",
    )

gate.register_callback("gates_blocked", on_production_blocked)
```

### 5. Audit Trail
```python
# All approvals/rejections are logged
approval = registry.get_approval_status(name, version, context)
if approval:
    print(f"Approved by {approval.approved_by} at {approval.timestamp}")
    print(f"Notes: {approval.notes}")
```

## Testing

Run the test suite:

```bash
pytest tests/test_quality_gates.py -v
```

Key test areas:
- Threshold enforcement (each context)
- Metric calculations
- Report generation
- CI/CD integrations
- Registry operations
- Callback triggering

## Architecture

```
gates/
├── quality_gates.py          # Core gate logic
├── ci_integration.py         # CI/CD pipeline integration
├── registry_integration.py    # Model registry interface
├── cli.py                    # Command-line interface
└── __init__.py              # Public API
```

### Core Components

1. **QualityGate**: Main enforcer with context-aware rules
2. **GateCheckResult**: Individual check result
3. **QualityGateReport**: Complete validation report
4. **CIPipelineIntegration**: Abstract CI/CD interface
   - GitHubActionsIntegration
   - JenkinsIntegration
   - LocalFileIntegration
5. **RegistryIntegration**: Model version approval tracking
6. **DeploymentController**: Deployment eligibility and control

## Integration Examples

### Full CI/CD Pipeline

```python
from afs.gates import QualityGate, TestMetrics, ModelMetrics
from afs.gates.ci_integration import GitHubActionsIntegration
from afs.gates.registry_integration import RegistryIntegration, DeploymentController

# Load metrics from CI environment
test_metrics = load_test_results()
model_metrics = load_evaluation_results()
security_scan = load_security_scan()

# Run production gates
gate = QualityGate.production()
report = gate.check_all(
    model_name="veran",
    model_version=os.environ["MODEL_VERSION"],
    test_metrics=test_metrics,
    model_metrics=model_metrics,
    security_results=security_scan,
)

# Report to CI
gh_integration = GitHubActionsIntegration()
gh_integration.report(report)

if not report.all_passed():
    gh_integration.block_merge(report, "Quality gates failed")
    sys.exit(1)

# Approve in registry
registry = RegistryIntegration()
if not registry.approve_version(report):
    sys.exit(1)

# Prepare deployment
controller = DeploymentController(registry)
if not controller.pre_deployment_check(
    "veran",
    os.environ["MODEL_VERSION"],
    "production"
):
    sys.exit(1)

print("✓ All gates passed - ready for deployment")
```

## Monitoring

The system can emit metrics and alerts:

```python
from monitoring import metrics, alerts

def setup_monitoring(gate: QualityGate):
    def on_check_failed(result):
        metrics.increment(f"gates.failed.{result.gate_name}")
        alerts.log_warning(f"Gate {result.gate_name} failed")

    def on_deployment_blocked(report):
        metrics.increment("gates.blocked")
        alerts.send_alert(
            level="critical",
            message=f"Deployment blocked: {report.model_name}"
        )

    gate.register_callback("gate_failed", on_check_failed)
    gate.register_callback("gates_blocked", on_deployment_blocked)
```

## Troubleshooting

### Gates passing locally but failing in CI

Check that CI environment provides all required metrics:
- Test results with coverage
- Model evaluation metrics
- Security scan results
- Baseline for comparison

### Approval not working

Verify registry path and permissions:
```bash
ls -la models/registry/
```

### Callbacks not firing

Ensure callbacks are registered before calling check methods:
```python
gate = QualityGate.production()
gate.register_callback("gate_passed", callback)  # Before check
gate.check_all(...)  # Callback will fire
```

## Performance

- Gate checks: < 100ms per gate
- Report generation: < 500ms
- File I/O: Network-dependent
- In-memory: < 10MB for typical report

## See Also

- Quality metrics: `afs.quality.metrics`
- Model registry: `afs.registry.database`
- Continuous learning: `afs.continuous.learning`
