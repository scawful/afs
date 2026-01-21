# Quality Gates Quick Reference

## Installation

The quality gates system is part of AFS. Import and use:

```python
from afs.gates import QualityGate, TestMetrics, ModelMetrics, SecurityScanResults
```

## Create a Gate

```python
# Development (relaxed)
gate = QualityGate.development()

# Staging (standard)
gate = QualityGate.staging()

# Production (strict)
gate = QualityGate.production()

# Custom thresholds
from afs.gates import GateThresholds
custom = GateThresholds(min_test_pass_rate=0.92, min_code_coverage=0.85)
gate = QualityGate.staging(custom)
```

## Check Tests

```python
from afs.gates import TestMetrics

metrics = TestMetrics(
    total_tests=100,
    passed_tests=95,
    failed_tests=5,
    skipped_tests=0,
    duration_seconds=60.0,
    coverage_percent=85.0,
)

result = gate.check_tests(metrics)
print(f"Status: {result.status}")  # PASSED or BLOCKED
print(f"Message: {result.message}")
```

## Check Model Quality

```python
from afs.gates import ModelMetrics

metrics = ModelMetrics(
    quality_score=0.87,
    accuracy=0.92,
    latency_ms=125.0,
    throughput_tokens_per_sec=50.0,
    baseline_quality_score=0.85,  # For regression comparison
    baseline_latency_ms=120.0,
)

result = gate.check_model_quality(metrics)
print(f"Regression: {metrics.regression_percent():.2%}")
print(f"Latency increase: {metrics.latency_increase_percent():.2%}")
```

## Check Security

```python
from afs.gates import SecurityScanResults

scan = SecurityScanResults(
    critical_vulnerabilities=0,
    high_vulnerabilities=1,
    medium_vulnerabilities=5,
    low_vulnerabilities=10,
    scan_tool="trivy",
)

result = gate.check_security(scan)
print(f"Vulnerabilities: {scan.total_vulnerabilities()}")
```

## Run All Checks

```python
report = gate.check_all(
    model_name="veran",
    model_version="v6.2",
    test_metrics=test_metrics,
    model_metrics=model_metrics,
    security_results=scan_results,
    baseline_model_metrics=baseline_metrics,
)

print(gate.summary_string(report))

if not report.all_passed():
    for check in report.failed_checks():
        print(f"FAILED: {check.gate_name} - {check.message}")
```

## Register Callbacks

```python
def on_gate_passed(result):
    print(f"Gate {result.gate_name} passed")

def on_gate_failed(result):
    print(f"Gate {result.gate_name} failed: {result.message}")

def on_gates_blocked(report):
    print(f"Deployment blocked: {report.model_name}")

gate.register_callback("gate_passed", on_gate_passed)
gate.register_callback("gate_failed", on_gate_failed)
gate.register_callback("gates_blocked", on_gates_blocked)
```

## CI/CD Integration

### GitHub Actions

```python
from afs.gates.ci_integration import GitHubActionsIntegration

integration = GitHubActionsIntegration()

# Report results
integration.report(report)

# Block merge on failure
if not report.all_passed():
    integration.block_merge(report, "Quality gates failed")

# Approve deployment
integration.approve_deployment("veran", "v6.2")
```

### Jenkins

```python
from afs.gates.ci_integration import JenkinsIntegration

integration = JenkinsIntegration()
integration.report(report)  # Creates JUnit XML
```

### Local Files

```python
from afs.gates.ci_integration import LocalFileIntegration

integration = LocalFileIntegration(".quality-gates")
integration.report(report)  # Saves JSON
```

## Registry & Approvals

```python
from afs.gates.registry_integration import RegistryIntegration

registry = RegistryIntegration()

# Approve a version
registry.approve_version(report, approved_by="alice@example.com")

# Check approval status
status = registry.get_approval_status("veran", "v6.2", "production")
if status and status.approved:
    print(f"Approved on {status.timestamp}")

# List approved versions
approved = registry.list_approved_versions("veran", "staging")
print(f"Approved versions: {approved}")

# Reject a version
registry.reject_version(
    "veran", "v6.2", "production",
    reason="Security vulnerabilities",
    rejected_by="alice@example.com"
)
```

## Deployment Control

```python
from afs.gates.registry_integration import DeploymentController

controller = DeploymentController()

# Check if deployable
can_deploy, reason = controller.can_deploy("veran", "v6.2", "production")
if can_deploy:
    # Execute deployment
    controller.execute_deployment("veran", "v6.2", "production")
else:
    print(f"Cannot deploy: {reason}")

# Rollback if needed
controller.rollback("veran", "v6.2", "production", reason="Anomaly detected")
```

## CLI Commands

```bash
# Check gates
python -m afs.gates check \
  --context production \
  --model veran:v6.2 \
  --test-metrics test-metrics.json \
  --metrics model-metrics.json \
  --baseline baseline-metrics.json \
  --security-scan security.json

# Approve
python -m afs.gates approve \
  --model veran:v6.2 \
  --context production \
  --approved-by alice@example.com \
  --notes "Passed all gates"

# Reject
python -m afs.gates reject \
  --model veran:v6.2 \
  --context production \
  --reason "Security issues" \
  --rejected-by alice@example.com

# Check status
python -m afs.gates status --model veran:v6.2
python -m afs.gates status --model veran:v6.2 --context production

# View history
python -m afs.gates history --model veran:v6.2 --json

# View reports
python -m afs.gates report --model veran:v6.2 --context production
```

## Thresholds by Context

### Development
```
min_test_pass_rate: 80%
min_code_coverage: 60%
min_quality_score: 0.50
max_regression: 20%
max_latency_increase: 50%
max_critical: 1
max_high: 5
max_medium: 20
```

### Staging (Default)
```
min_test_pass_rate: 95%
min_code_coverage: 80%
min_quality_score: 0.70
max_regression: 5%
max_latency_increase: 20%
max_critical: 0
max_high: 2
max_medium: 10
```

### Production
```
min_test_pass_rate: 98%
min_code_coverage: 90%
min_quality_score: 0.85
max_regression: 2%
max_latency_increase: 10%
max_critical: 0
max_high: 0
max_medium: 2
max_memory_increase: 5%
```

## Common Patterns

### Full Pipeline Check

```python
gate = QualityGate.production()

report = gate.check_all(
    model_name="veran",
    model_version=version,
    test_metrics=get_test_results(),
    model_metrics=get_eval_results(),
    security_results=get_security_scan(),
    baseline_model_metrics=get_baseline(),
)

if report.all_passed():
    registry.approve_version(report)
    controller.execute_deployment("veran", version, "production")
else:
    registry.reject_version("veran", version, "production",
                           reason=str(report.failed_checks()))
```

### Progressive Validation

```python
# Check in development first
dev_gate = QualityGate.development()
if not dev_gate.check_all(...).all_passed():
    return False

# Then staging
staging_gate = QualityGate.staging()
if not staging_gate.check_all(...).all_passed():
    return False

# Finally production
prod_gate = QualityGate.production()
if not prod_gate.check_all(...).all_passed():
    return False

# All passed, proceed with deployment
```

### Monitoring Regressions

```python
current = ModelMetrics(...)
baseline = ModelMetrics(...)

regression = current.regression_percent()
if regression > 0.05:  # 5% threshold
    alerts.warning(f"Quality regression: {regression:.2%}")
    controller.rollback(...)
```

## Data Files Format

### Test Metrics JSON
```json
{
  "total_tests": 1000,
  "passed_tests": 980,
  "failed_tests": 20,
  "skipped_tests": 0,
  "duration_seconds": 120.5,
  "coverage_percent": 88.5
}
```

### Model Metrics JSON
```json
{
  "quality_score": 0.87,
  "accuracy": 0.92,
  "f1_score": 0.89,
  "latency_ms": 127.0,
  "throughput_tokens_per_sec": 48.5,
  "memory_mb": 2100.0,
  "baseline_quality_score": 0.85,
  "baseline_latency_ms": 125.0
}
```

### Security Scan JSON
```json
{
  "critical_vulnerabilities": 0,
  "high_vulnerabilities": 1,
  "medium_vulnerabilities": 5,
  "low_vulnerabilities": 12,
  "scan_tool": "trivy"
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Gate always passes | Check thresholds match your context |
| Regression detected | Compare against correct baseline |
| CI integration not working | Verify GitHub environment variables set |
| Approval not recorded | Check permissions on registry directory |
| Metrics not loading | Verify JSON format matches expected schema |

## Links

- Full Documentation: `/Users/scawful/src/lab/afs/QUALITY_GATES_README.md`
- Integration Guide: `/Users/scawful/src/lab/afs/GATES_INTEGRATION.md`
- Examples: `/Users/scawful/src/lab/afs/examples/quality_gates_example.py`
- Tests: `/Users/scawful/src/lab/afs/tests/test_quality_gates.py`
