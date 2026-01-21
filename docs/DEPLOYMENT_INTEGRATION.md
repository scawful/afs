# Deployment Validator Integration Guide

Complete guide to integrating the pre-deployment validator into your deployment pipeline.

## Quick Start

### Install

The validator is part of AFS. No additional installation needed.

```bash
python -m afs.deployment.cli --help
```

### Basic Usage

```bash
# Validate a model before deployment
python -m afs.deployment.cli validate model.gguf \
    --model-name majora \
    --version v5 \
    --baseline v4 \
    --output-dir ./reports \
    --json \
    --markdown \
    --notify
```

## Integration Points

### 1. Model Registry

The validator integrates with the model registry to access version metadata:

```python
from pathlib import Path
from afs.registry.database import ModelRegistry
from afs.deployment import PreDeploymentValidator

# Get model from registry
registry = ModelRegistry()
version = registry.get_version("majora", "v5")

# Run validation
validator = PreDeploymentValidator(
    model_path=Path(version.gguf_path),
    model_name="majora",
    version="v5",
    baseline_version=version.parent_version
)

report = validator.validate_all()
```

### 2. Notification System

Integrate with the notification system for deployment alerts:

```python
from afs.notifications.base import NotificationManager
from afs.notifications.desktop import DesktopNotificationHandler
from afs.notifications.email import EmailNotificationHandler
from afs.notifications.slack import SlackNotificationHandler

# Setup notifications
notif = NotificationManager()
notif.register_handler("desktop", DesktopNotificationHandler())
notif.register_handler("email", EmailNotificationHandler())
notif.register_handler("slack", SlackNotificationHandler())

# Pass to validator
validator = PreDeploymentValidator(
    model_path="model.gguf",
    notification_manager=notif
)

report = validator.validate_all()
```

### 3. Deployment Pipeline

Integrate into your CI/CD deployment pipeline:

```python
#!/usr/bin/env python3
"""Deployment pipeline with pre-deployment validation."""

import sys
from pathlib import Path
from afs.deployment import PreDeploymentValidator
from afs.registry.database import ModelRegistry

def main():
    model_name = "majora"
    version = "v5"

    # Get model from registry
    registry = ModelRegistry()
    model_version = registry.get_version(model_name, version)

    if not model_version or not model_version.gguf_path:
        print(f"Model {model_name} {version} not found")
        return 1

    # Run validation
    validator = PreDeploymentValidator(
        model_path=Path(model_version.gguf_path),
        model_name=model_name,
        version=version,
        baseline_version=model_version.parent_version
    )

    report = validator.validate_all()

    # Save reports
    report_dir = Path("./deployment_reports")
    report.save_json(report_dir / f"{model_name}_{version}.json")
    report.save_markdown(report_dir / f"{model_name}_{version}.md")

    # Check if deployment should proceed
    if report.passed:
        print(f"✓ {model_name} {version} ready for deployment")
        # Continue with deployment
        return deploy_model(model_name, version)
    else:
        print(f"✗ {model_name} {version} validation failed")
        print(f"  Failed checks: {len(report.failed_checks)}")
        for check in report.failed_checks:
            print(f"    - {check.check_name}: {check.message}")

        rollback = validator.get_rollback_recommendation()
        if rollback:
            print(f"\n{rollback}")

        return 1

def deploy_model(model_name, version):
    """Deploy model to production."""
    print(f"Deploying {model_name} {version}...")
    # Your deployment logic here
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### 4. GitHub Actions Workflow

Example GitHub Actions workflow for continuous deployment:

```yaml
name: Deploy Model

on:
  workflow_dispatch:
    inputs:
      model_name:
        description: Model name
        required: true
      version:
        description: Version to deploy
        required: true
      baseline:
        description: Baseline version for comparison
        required: false

jobs:
  validate:
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
          pip install pytest

      - name: Run validation
        run: |
          python -m afs.deployment.cli validate \
            models/${{ inputs.model_name }}_${{ inputs.version }}.gguf \
            --model-name ${{ inputs.model_name }} \
            --version ${{ inputs.version }} \
            --baseline ${{ inputs.baseline }} \
            --output-dir ./reports \
            --json \
            --markdown

      - name: Upload reports
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: validation-reports
          path: reports/

      - name: Check validation status
        run: |
          python3 << 'EOF'
          import json
          with open('reports/${{ inputs.model_name }}_${{ inputs.version }}_*.json') as f:
              report = json.load(f)
              if report['summary']['overall_status'] != 'PASSED':
                  exit(1)
          EOF

  deploy:
    needs: validate
    runs-on: ubuntu-latest
    if: success()
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to production
        run: |
          # Your deployment script
          ./scripts/deploy.sh ${{ inputs.model_name }} ${{ inputs.version }}

      - name: Notify deployment
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: |
            Deployed ${{ inputs.model_name }} v${{ inputs.version }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### 5. Docker Integration

Run validator in Docker:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install -e .

ENTRYPOINT ["python", "-m", "afs.deployment.cli"]
CMD ["validate", "--help"]
```

Usage:

```bash
docker build -t afs-validator .

docker run -v /models:/models afs-validator validate \
  /models/majora_v5.gguf \
  --model-name majora \
  --version v5
```

## Handling Validation Failures

### Deployment Blocked

If validation fails, deployment is blocked automatically:

```python
if report.passed:
    # Safe to deploy
    deploy()
else:
    # Fix issues first
    for failure in report.failed_checks:
        print(f"❌ {failure.check_name}: {failure.message}")
        print(f"   Details: {failure.details}")
```

### Automatic Rollback

If a critical check fails, automatic rollback is recommended:

```python
rollback = validator.get_rollback_recommendation()
if rollback:
    print(rollback)
    # Trigger rollback
    registry.set_deployed(model_name, baseline_version, deployed=True)
```

### Escalation

For critical failures, escalate to DevOps team:

```python
if not report.passed and len(report.failed_checks) > 3:
    # Critical multi-check failure
    notif.notify(
        title="CRITICAL: Model validation failed",
        message=f"Multiple failures in {model_name} {version}",
        level=NotificationLevel.CRITICAL
    )
```

## Monitoring

### Validation Metrics

Track validation metrics over time:

```python
import json
from datetime import datetime

# Save validation report
timestamp = datetime.now().isoformat()
report_path = f"metrics/{model_name}_{version}_{timestamp}.json"
report.save_json(Path(report_path))

# Analyze trends
reports = Path("metrics").glob(f"{model_name}_*.json")
for report_file in reports:
    with open(report_file) as f:
        data = json.load(f)
        print(f"{data['timestamp']}: {data['summary']['overall_status']}")
```

### Dashboard Metrics

Expose validation metrics to monitoring dashboard:

```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
validation_total = Counter('afs_validation_total', 'Total validations', ['model', 'status'])
validation_duration = Histogram('afs_validation_duration_seconds', 'Validation duration')
validation_failed_checks = Gauge('afs_validation_failed_checks', 'Failed checks', ['model', 'version'])

# Record metrics
with validation_duration.time():
    report = validator.validate_all()

validation_total.labels(model=model_name, status=report.summary()['overall_status']).inc()
validation_failed_checks.labels(model=model_name, version=version).set(len(report.failed_checks))
```

## Best Practices

### 1. Always Validate Before Deployment

```python
# ❌ Bad: Skip validation
deploy_model(model_path)

# ✅ Good: Validate first
validator = PreDeploymentValidator(model_path)
report = validator.validate_all()
if report.passed:
    deploy_model(model_path)
```

### 2. Save Reports for Audit Trail

```python
# ✅ Good: Save both JSON and markdown
report_dir = Path("deployment_reports") / datetime.now().strftime("%Y-%m-%d")
report.save_json(report_dir / f"{model_name}_{version}.json")
report.save_markdown(report_dir / f"{model_name}_{version}.md")
```

### 3. Enable Notifications

```python
# ✅ Good: Get alerts on failures
validator = PreDeploymentValidator(
    model_path,
    notification_manager=notif
)
```

### 4. Compare Against Baseline

```python
# ✅ Good: Test for regressions
validator = PreDeploymentValidator(
    model_path,
    baseline_version="v4"  # Compare to previous version
)
```

### 5. Use Strict Mode for Critical Deployments

```bash
# ✅ Good: Require all checks to pass (no warnings)
python -m afs.deployment.cli validate model.gguf --strict
```

## Troubleshooting

### "Validation takes too long"

Inference tests (the slowest check) can be disabled by not calling `validate_all()`:

```python
# Run only fast checks
validator._check_file_exists()
validator._check_file_integrity()
validator._check_file_format()
validator._check_file_size()
validator._check_memory_requirements()
# Skip: validator._check_inference_capability()
# Skip: validator._check_response_quality()
# Skip: validator._check_latency()
```

### "Model fails to load"

Check if model format is correct:

```bash
# Verify GGUF format
file model.gguf
xxd -l 4 model.gguf  # Should show: 47 47 55 46 (GGUF)

# Try loading with llama-cpp
python -c "from llama_cpp import Llama; Llama('model.gguf')"
```

### "False positive on warnings"

Adjust thresholds for your deployment:

```python
validator = PreDeploymentValidator(model_path)

# Increase thresholds
validator.MIN_TOKENS_PER_SEC = 3.0  # Instead of 5.0
validator.WARN_VRAM_GB = 16.0  # Instead of 8.0

report = validator.validate_all()
```

## See Also

- [Deployment Validator Documentation](DEPLOYMENT_VALIDATOR.md)
- [Model Registry](REGISTRY_README.md)
- [Notifications System](NOTIFICATIONS_README.md)
