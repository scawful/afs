# Pre-Deployment Validation System

A comprehensive validation system that runs rigorous checks before deploying models to production, catching issues before they affect users.

## Overview

The pre-deployment validator provides multiple layers of quality assurance:

1. **File Integrity** - SHA256 verification
2. **Format Validation** - GGUF compatibility checking
3. **Size Checks** - Reasonable file size bounds
4. **Memory Requirements** - VRAM availability estimation
5. **Inference Tests** - Load model and run test queries
6. **Response Quality** - Check coherence of outputs
7. **Latency Tests** - Measure tokens/second throughput
8. **Regression Tests** - Compare against baseline version

Each check produces a pass/fail/warning result with detailed metrics.

## Installation

The validator is part of the AFS deployment module:

```bash
python -m afs.deployment.validator --help
```

Or use programmatically:

```python
from afs.deployment import PreDeploymentValidator

validator = PreDeploymentValidator(
    model_path="/path/to/model.gguf",
    model_name="majora",
    version="v5",
    baseline_version="v4"
)

report = validator.validate_all()
print(report.summary())
```

## CLI Usage

### Validate a Model

```bash
python -m afs.deployment.cli validate /path/to/model.gguf \
    --model-name majora \
    --version v5 \
    --baseline v4 \
    --output-dir ./reports \
    --json \
    --markdown \
    --notify
```

**Options:**
- `--model-name` - Name of the model
- `--version` - Version string (e.g., "v5")
- `--baseline` - Previous version for regression testing
- `--output-dir` - Directory to save reports
- `--json` - Generate JSON report
- `--markdown` - Generate markdown report
- `--notify` - Send notifications on failures
- `--strict` - Exit with error on any warnings

**Output:**
```
============================================================
Validation Report: majora v5
============================================================
Status: PASSED
Passed: 8/8
Warnings: 0
Failed: 0

✓ File Exists: Model file found: /path/to/model.gguf
✓ SHA256 Verification: File integrity verified
✓ GGUF Magic Bytes: Valid GGUF format detected
✓ File Size: Model size within acceptable range: 7.45 GB
✓ VRAM Requirements: Good throughput: 12.34 tokens/sec
✓ Model Loading: Successfully loaded and ran 3 test queries
✓ Response Quality: All 3 test responses appear coherent
✓ Inference Latency: Good throughput: 12.34 tokens/sec
```

### Check Model Status

```bash
# Check specific version
python -m afs.deployment.cli registry-check \
    --model-name majora \
    --version v5

# List all versions
python -m afs.deployment.cli registry-check \
    --model-name majora
```

### Compare Versions

```bash
python -m afs.deployment.cli compare \
    --model-name majora \
    --version1 v4 \
    --version2 v5
```

**Output:**
```
Comparing majora:
  v4 vs v5

  accuracy: 0.92 → 0.94 (↑ 2.2%)
  f1_score: 0.89 → 0.91 (↑ 2.2%)
  latency_ms: 45 → 42 (↓ 6.7%)
```

### Rollback Model

```bash
python -m afs.deployment.cli rollback \
    --model-name majora \
    --target-version v4
```

## Validation Checks

### 1. File Integrity (SHA256)

Verifies model file hasn't been corrupted during transfer.

```
Category: FILE_INTEGRITY
Check: SHA256 Verification
Details: Computes SHA256 hash of model file
Status: PASSED/FAILED
```

**What it checks:**
- File exists and is readable
- Computes and reports SHA256 hash
- Can be used to verify integrity after download

**Failure scenarios:**
- File not found
- File corrupted (different hash than recorded)

### 2. Format Validation (GGUF)

Ensures model file is in correct format for deployment.

```
Category: FORMAT_VALIDATION
Checks:
  - File Extension
  - GGUF Magic Bytes
Status: PASSED/WARNING/FAILED
```

**What it checks:**
- File extension matches expected format (.gguf, .bin, .pt, .safetensors)
- For GGUF files: magic bytes are "GGUF" (0x47474546)
- File structure is valid

**Failure scenarios:**
- Wrong file extension
- Invalid magic bytes in GGUF file
- Corrupted file structure

### 3. Size Checks

Validates file size is within reasonable bounds.

```
Category: SIZE_CHECKS
Check: File Size
Bounds: 100 MB - 100 GB
Status: PASSED/WARNING/FAILED
```

**What it checks:**
- Model size >= 100 MB (minimum viable model)
- Model size <= 100 GB (maximum storage)

**Warning scenarios:**
- Model < 100 MB (unusually small)

**Failure scenarios:**
- Model > 100 GB (too large)

### 4. Memory Requirements

Estimates VRAM needed and checks feasibility.

```
Category: MEMORY_REQUIREMENTS
Check: VRAM Requirements
Status: PASSED/WARNING/FAILED
Details:
  - Model size: 7.45 GB
  - Estimated parameters: 7.45B
  - Estimated VRAM: 14.9 GB
  - Threshold: 8 GB warning
```

**What it checks:**
- Estimates parameter count from file size
- Calculates VRAM needed (roughly 2GB per billion parameters)
- Compares against thresholds

**Warning scenarios:**
- VRAM > 8 GB (high memory requirement)

**Failure scenarios:**
- None (informational only)

### 5. Inference Test

Loads model and runs 3 test queries.

```
Category: INFERENCE_TEST
Check: Model Loading
Status: PASSED/FAILED
Details:
  - Queries: 3
  - Elapsed: 2.34 seconds
  - Avg response length: 125 tokens
```

**What it checks:**
- Model file can be loaded
- Model can run inference
- Basic functionality works
- Typical response length

**Failure scenarios:**
- Model fails to load
- Inference throws error
- Model appears broken

### 6. Response Quality

Checks that model outputs are coherent.

```
Category: RESPONSE_QUALITY
Check: Response Quality
Status: PASSED/WARNING
Details:
  - Responses checked: 3
  - Issues: None
```

**What it checks:**
- Response length is reasonable (>10 characters)
- No error messages in responses
- No excessive token repetition (sign of malfunction)
- Outputs appear coherent

**Warning scenarios:**
- Response too short (<10 chars)
- Contains "error" text
- High token repetition (>50%)

### 7. Latency Test

Measures inference throughput (tokens/second).

```
Category: LATENCY_TEST
Check: Inference Latency
Status: PASSED/WARNING/FAILED
Details:
  - Total tokens: 155
  - Responses: 3
  - Tokens per response: 51.7
  - Tokens/sec: 12.34
```

**What it checks:**
- Inference speed in tokens/sec
- Compares against minimum threshold (5 tokens/sec)
- Compares against optimal threshold (10 tokens/sec)

**Thresholds:**
- < 5 tokens/sec: FAILED
- 5-10 tokens/sec: WARNING
- > 10 tokens/sec: PASSED

### 8. Regression Test

Compares performance against baseline version on 20 standard questions.

```
Category: REGRESSION_TEST
Check: Regression Test
Status: PASSED/WARNING
Details:
  - Baseline: v4
  - Test questions: 20
  - Status: No significant degradation
```

**What it checks:**
- Loads both current and baseline model
- Runs 20 standard test questions on both
- Compares response metrics
- Flags if current version is worse

**Test questions cover:**
- ML fundamentals (what is machine learning, neural networks, etc.)
- Model architectures (transformers, attention, etc.)
- Training techniques (backprop, fine-tuning, etc.)
- Evaluation metrics (precision, recall, F1, ROC, etc.)

## Validation Report

### JSON Report

```json
{
  "model_path": "/path/to/model.gguf",
  "model_name": "majora",
  "version": "v5",
  "baseline_version": "v4",
  "timestamp": "2026-01-14T10:30:45.123456",
  "summary": {
    "total_checks": 8,
    "passed": 8,
    "warnings": 0,
    "failed": 0,
    "skipped": 0,
    "overall_status": "PASSED"
  },
  "results": [
    {
      "category": "file_integrity",
      "check_name": "File Exists",
      "status": "passed",
      "message": "Model file found: /path/to/model.gguf",
      "details": {
        "size_bytes": 8000000000
      },
      "timestamp": "2026-01-14T10:30:45.123456"
    },
    ...
  ],
  "test_queries": [
    "What is 2+2?",
    "Explain machine learning in one sentence.",
    "What is the capital of France?"
  ],
  "test_responses": [
    {
      "query": "What is 2+2?",
      "response": "[Mock response to: What is 2+2?...]",
      "tokens": 25,
      "stop_reason": "length"
    },
    ...
  ]
}
```

### Markdown Report

```markdown
# Pre-Deployment Validation Report

**Model:** majora v5
**Timestamp:** 2026-01-14T10:30:45.123456
**Status:** PASSED

## Summary
- Total Checks: 8
- Passed: 8
- Warnings: 0
- Failed: 0
- Skipped: 0

## Validation Results

### File Integrity
- ✓ **File Exists**: Model file found: /path/to/model.gguf
  - size_bytes: 8000000000

### Format Validation
- ✓ **GGUF Magic Bytes**: Valid GGUF format detected
  - magic: GGUF

### Size Checks
- ✓ **File Size**: Model size within acceptable range: 7.45 GB
  - size_bytes: 8000000000
  - size_gb: 7.45

### Memory Requirements
- ✓ **VRAM Requirements**: Good throughput: 12.34 tokens/sec
  - model_size_gb: 7.45
  - estimated_params_billions: 7.45
  - estimated_vram_gb: 14.9

...
```

## Rollback Recommendations

If validation fails, the system provides rollback recommendations:

```python
rollback = validator.get_rollback_recommendation()
# Output: "CRITICAL: Rollback to v4"
```

**Critical failures trigger rollback recommendation:**
- GGUF Magic Bytes failed (corrupted format)
- Maximum Size exceeded (won't fit)
- Model Loading failed (can't run)
- File doesn't exist

## Integration with Notifications

Enable notifications for critical failures:

```python
from afs.notifications.base import NotificationManager
from afs.notifications.desktop import DesktopNotificationHandler
from afs.notifications.email import EmailNotificationHandler

# Create notification manager
notif_mgr = NotificationManager()
notif_mgr.register_handler("desktop", DesktopNotificationHandler())
notif_mgr.register_handler("email", EmailNotificationHandler())

# Pass to validator
validator = PreDeploymentValidator(
    model_path="model.gguf",
    notification_manager=notif_mgr
)

report = validator.validate_all()
```

## Integration with Model Registry

The validator integrates with the model registry:

```python
from afs.registry.database import ModelRegistry

registry = ModelRegistry()

# Get model version
version = registry.get_version("majora", "v5")

# Check GGUF path
gguf_path = Path(version.gguf_path)

# Run validator
validator = PreDeploymentValidator(
    model_path=gguf_path,
    model_name="majora",
    version="v5",
    baseline_version="v4"
)

report = validator.validate_all()
```

## Best Practices

### Before Production Deployment

1. **Always run full validation:**
   ```bash
   python -m afs.deployment.cli validate model.gguf \
       --model-name majora \
       --version v5 \
       --baseline v4 \
       --json \
       --markdown \
       --notify
   ```

2. **Review all warnings**, not just failures:
   - High memory requirements
   - Slow latency
   - Quality issues

3. **Save reports for audit trail:**
   ```bash
   --output-dir ./deployment_reports
   ```

4. **Notify on deployment:**
   ```bash
   --notify
   ```

### Handling Failures

| Failure | Action | Rollback? |
|---------|--------|-----------|
| File not found | Check path, verify transfer | N/A |
| Invalid GGUF format | Regenerate GGUF, check conversion | Yes, to v4 |
| File too large | Optimize model, check disk | Yes, to v4 |
| Model won't load | Debug model structure, test locally | Yes, to v4 |
| Low throughput | Optimize inference, check hardware | Review |
| Low quality | Retrain, check data quality | Review |

### Handling Warnings

| Warning | Action | Deploy? |
|---------|--------|---------|
| High VRAM (>8GB) | Note requirement, check deployment target | Yes* |
| Slow latency (5-10 t/s) | Optimize if possible, note threshold | Yes |
| Small model (<100MB) | Verify intentional, check capabilities | Yes |
| Low response quality | Retrain if pattern, monitor | Review |

*Only if deployment hardware supports it.

## Advanced Usage

### Custom Validation Threshold

```python
validator = PreDeploymentValidator(
    model_path="model.gguf",
    model_name="majora",
    version="v5"
)

# Override thresholds
validator.MIN_TOKENS_PER_SEC = 8.0
validator.WARN_VRAM_GB = 16.0

report = validator.validate_all()
```

### Programmatic Integration

```python
from afs.deployment import PreDeploymentValidator
from afs.registry.database import ModelRegistry

# Get model from registry
registry = ModelRegistry()
version = registry.get_version("majora", "v5")

# Validate
validator = PreDeploymentValidator(
    model_path=Path(version.gguf_path),
    model_name="majora",
    version="v5",
    baseline_version=version.parent_version
)

report = validator.validate_all()

# Check results
if report.passed:
    print("✓ Ready to deploy!")
    # Deploy logic here
else:
    print("✗ Deployment blocked:")
    for fail in report.failed_checks:
        print(f"  - {fail.check_name}: {fail.message}")

    # Get rollback recommendation
    rollback = validator.get_rollback_recommendation()
    if rollback:
        print(f"\nRecommendation: {rollback}")
```

### Generate HTML Report

```python
report = validator.validate_all()

# Save JSON and generate HTML from template
report.save_json(Path("reports/validation.json"))

# Use your favorite template engine:
# jinja2, mako, etc.
```

## Troubleshooting

### "Model file not found"

```bash
# Check path is correct
ls -lh /path/to/model.gguf

# Use absolute path
python -m afs.deployment.cli validate $(pwd)/model.gguf
```

### "Invalid GGUF magic bytes"

The file isn't actually a GGUF file:

```bash
# Check file type
file model.gguf

# Check first 4 bytes
xxd -l 4 model.gguf
# Should show: 47 47 55 46 (GGUF in hex)
```

### "Inference test failed"

Model can't be loaded or run:

```bash
# Try loading manually
python -c "from llama_cpp import Llama; m = Llama('model.gguf')"

# Check model format and architecture
python -c "import gguf; reader = gguf.GGUFReader('model.gguf'); print(reader.fields.keys())"
```

### "High VRAM requirement"

Model needs more GPU memory than available:

```bash
# Check estimated requirements
python -m afs.deployment.cli validate model.gguf | grep VRAM

# Options:
# 1. Use smaller quantization (Q4 instead of Q5)
# 2. Deploy on GPU with more memory
# 3. Use CPU inference (slower but no VRAM limit)
```

## Testing

Run validator tests:

```bash
pytest tests/test_deployment_validator.py -v
```

Test coverage:
- ValidationResult creation and serialization
- ValidationReport creation, summary, save
- All validation checks (file, format, size, memory, inference, quality, latency, regression)
- Rollback recommendations
- Full validation flow

## Performance

Typical validation time: 5-30 seconds

- File integrity (SHA256): 1-10 seconds (depends on file size)
- Format validation: <1 second
- Size checks: <1 second
- Memory estimation: <1 second
- Inference test: 2-10 seconds (depends on model size and hardware)
- Response quality: <1 second
- Latency measurement: <1 second
- Regression test (20 questions): 5-20 seconds

## See Also

- [Model Registry](REGISTRY_README.md)
- [Training System](../src/afs/training)
- [Notifications](NOTIFICATIONS_README.md)
