# AFS Deployment Module

Pre-deployment validation system for model deployment with rigorous quality checks.

## Quick Start

### Validate a Model
```bash
python -m afs.deployment.cli validate model.gguf \
    --model-name majora \
    --version v5 \
    --baseline v4 \
    --json \
    --markdown
```

### Programmatic API
```python
from afs.deployment import PreDeploymentValidator
from pathlib import Path

validator = PreDeploymentValidator(
    model_path=Path("model.gguf"),
    model_name="majora",
    version="v5",
    baseline_version="v4"
)

report = validator.validate_all()
if report.passed:
    print("✓ Ready for deployment")
    report.save_json(Path("report.json"))
else:
    print("✗ Validation failed")
    for failure in report.failed_checks:
        print(f"  - {failure.check_name}")
```

## Validation Checks

The validator runs 8 comprehensive checks:

| Check | Purpose | Status |
|-------|---------|--------|
| File Integrity | SHA256 hash verification | ✓ Implemented |
| Format Validation | GGUF format and magic bytes | ✓ Implemented |
| Size Checks | 100 MB - 100 GB bounds | ✓ Implemented |
| Memory Requirements | VRAM estimation and threshold | ✓ Implemented |
| Inference Test | Load model and run queries | ✓ Implemented |
| Response Quality | Check coherence and outputs | ✓ Implemented |
| Latency Test | Measure tokens/sec throughput | ✓ Implemented |
| Regression Test | Compare vs baseline version | ✓ Implemented |

## Features

- **8 validation categories** covering file integrity, format, size, memory, inference, quality, latency, and regression
- **Detailed reports** in JSON and markdown formats
- **Automatic rollback recommendations** for critical failures
- **Notification integration** with desktop, email, Slack
- **Model registry integration** for version management
- **CLI and programmatic APIs** for flexibility
- **Comprehensive test coverage** (21/21 tests passing)

## Files

- `validator.py` - Core validation engine (900+ lines)
- `cli.py` - Command-line interface (450+ lines)
- `__init__.py` - Module exports

## Documentation

- `../../docs/DEPLOYMENT_VALIDATOR.md` - Complete feature documentation
- `../../docs/DEPLOYMENT_INTEGRATION.md` - Integration guide
- `../../examples/deployment_validation_example.py` - Usage examples

## CLI Commands

```bash
# Validate a model
python -m afs.deployment.cli validate MODEL_PATH [OPTIONS]

# Check model status
python -m afs.deployment.cli registry-check --model-name NAME [--version VERSION]

# Rollback to previous version
python -m afs.deployment.cli rollback --model-name NAME --target-version VERSION

# Compare two versions
python -m afs.deployment.cli compare --model-name NAME --version1 V1 --version2 V2
```

See `cli.py` for complete option documentation.

## Report Format

### JSON Report
```json
{
  "model_path": "...",
  "model_name": "majora",
  "version": "v5",
  "timestamp": "2026-01-14T10:30:45.123456",
  "summary": {
    "total_checks": 8,
    "passed": 8,
    "warnings": 0,
    "failed": 0,
    "overall_status": "PASSED"
  },
  "results": [...]
}
```

### Markdown Report
```markdown
# Pre-Deployment Validation Report

**Model:** majora v5
**Status:** PASSED
**Passed:** 8/8
```

## API Reference

### PreDeploymentValidator

```python
class PreDeploymentValidator:
    def __init__(
        self,
        model_path: Path,
        model_name: str = "unknown",
        version: str = "v1",
        baseline_version: Optional[str] = None,
        notification_manager: Optional[NotificationManager] = None,
    )

    def validate_all() -> ValidationReport
        """Run all validation checks"""

    def get_rollback_recommendation() -> Optional[str]
        """Get rollback recommendation if validation failed"""
```

### ValidationReport

```python
class ValidationReport:
    model_path: Path
    model_name: str
    version: str
    baseline_version: Optional[str]
    results: list[ValidationResult]

    @property
    def passed() -> bool
        """True if all checks passed"""

    @property
    def failed_checks() -> list[ValidationResult]
        """All failed checks"""

    def summary() -> dict[str, Any]
        """Summary statistics"""

    def save_json(output_path: Path) -> None
    def save_markdown(output_path: Path) -> None
```

## Testing

```bash
# Run all tests
python -m pytest tests/test_deployment_validator.py -v

# Run specific test class
python -m pytest tests/test_deployment_validator.py::TestPreDeploymentValidator -v

# Run with coverage
python -m pytest tests/test_deployment_validator.py --cov=afs.deployment
```

## Integration Examples

### Model Registry
```python
from afs.registry.database import ModelRegistry
from afs.deployment import PreDeploymentValidator

registry = ModelRegistry()
version = registry.get_version("majora", "v5")
validator = PreDeploymentValidator(Path(version.gguf_path), ...)
```

### Notifications
```python
from afs.notifications.base import NotificationManager
from afs.notifications.desktop import DesktopNotificationHandler

notif = NotificationManager()
notif.register_handler("desktop", DesktopNotificationHandler())
validator = PreDeploymentValidator(..., notification_manager=notif)
```

## Performance

- **Total validation time**: 5-30 seconds
- **File integrity (SHA256)**: 1-10 seconds (depends on file size)
- **Other checks**: <20 seconds combined

## Configuration

Adjust validation thresholds:

```python
validator = PreDeploymentValidator(model_path)

# Thresholds
validator.MIN_MODEL_SIZE = 100 * 1024 * 1024  # 100 MB
validator.MAX_MODEL_SIZE = 100 * 1024 * 1024 * 1024  # 100 GB
validator.MIN_TOKENS_PER_SEC = 5.0
validator.WARN_TOKENS_PER_SEC = 10.0
validator.MIN_VRAM_GB = 4.0
validator.WARN_VRAM_GB = 8.0
```

## Best Practices

1. **Always validate before deployment**
2. **Save reports for audit trail**
3. **Enable notifications for critical failures**
4. **Use baseline version for regression testing**
5. **Review warnings, not just failures**
6. **Use strict mode (`--strict`) for critical deployments**

## Troubleshooting

### "Validation failed"
Check the detailed report:
```bash
cat report.json | python -m json.tool
```

### "High VRAM requirement"
This is a warning, not a failure. Verify your deployment hardware can support it.

### "Model won't load"
Check file format and try loading with llama-cpp-python:
```bash
python -c "from llama_cpp import Llama; Llama('model.gguf')"
```

## Related Documentation

- [Deployment Validator Features](../../docs/DEPLOYMENT_VALIDATOR.md)
- [Integration Guide](../../docs/DEPLOYMENT_INTEGRATION.md)
- [Model Registry](../registry/README.md)
- [Notifications](../notifications/README.md)

## License

MIT - See LICENSE file
