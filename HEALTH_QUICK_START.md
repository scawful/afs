# AFS Health Check System - Quick Start Guide

## Installation

The health check system is built into AFS. Just activate your environment:

```bash
source .venv/bin/activate
```

## Quick Health Check

```bash
# Run a quick health check
afs health check

# Get detailed diagnostics
afs health check --level comprehensive

# Get JSON output for logging
afs health check --json
```

## Health Check Levels

```
afs health check --level basic           # 5-10s   (system only)
afs health check --level standard        # 15-30s  (full check) ← DEFAULT
afs health check --level comprehensive   # 45-90s  (benchmarks)
afs health check --level stress          # 90-120s (load test)
```

## Continuous Monitoring

```bash
# Monitor every 60 seconds
afs health monitor

# Monitor with custom interval
afs health monitor --interval 30

# Run for 24 hours with auto-healing
afs health monitor --duration 86400 --auto-heal

# Stop with Ctrl+C
```

## View Results

```bash
# Current status
afs health status

# View trends (last 24 hours)
afs health trend

# View trends for 7 days
afs health trend --hours 168

# View recent reports
afs health history

# Show last 20 reports
afs health history --limit 20

# Get JSON output
afs health status --json
afs history --json
```

## Understanding the Score

```
Green  (0.9-1.0)  ✓  Excellent   - System running optimally
Yellow (0.7-0.9)  ⚠  Good        - Minor issues detected
Orange (0.5-0.7)  ⚠⚠ Degraded    - Performance impact
Red    (0.0-0.5)  ✗  Critical    - Action required
```

## What Gets Checked

### System Health
- **CPU Usage**: Current CPU load
- **Memory Usage**: RAM utilization
- **Disk Space**: Storage availability
- **GPU VRAM**: GPU memory (if NVIDIA GPU present)

### Service Health
- **LMStudio API**: Model serving endpoint
- **MCP Servers**: Model context protocol servers
- **Python Dependencies**: Package integrity

### Model Health
- **Load Time**: How long to initialize model
- **Inference Latency**: Response time per query
- **Output Quality**: Response validation

### Data Health
- **Training Data**: File integrity check
- **Cache**: Freshness and validity

### Integration Health
- **External APIs**: Anthropic, OpenAI availability
- **Notifications**: Alert channel configuration

## Common Use Cases

### Development
```bash
# Daily health check before starting
afs health check

# Quick status check
afs health status
```

### Staging/Production
```bash
# Start monitoring daemon
afs health monitor --interval 60 --auto-heal &

# View trends daily
afs health trend --hours 24

# Review reports
afs health history
```

### CI/CD Pipeline
```bash
# Include in pre-deployment checks
afs health check --level comprehensive

# Exit with error if unhealthy
if [ $? -ne 0 ]; then
  echo "System health check failed"
  exit 1
fi
```

### Debugging Performance Issues
```bash
# Get comprehensive diagnostics
afs health check --level comprehensive --json > report.json

# Analyze trends
afs health trend --hours 168  # Last week

# Check history for patterns
afs health history
```

## Auto-Healing

Enable automatic fixes for detected issues:

```bash
# Auto-heal during check
afs health check --auto-heal

# Auto-heal during monitoring
afs health monitor --auto-heal
```

Auto-healing can:
- Clear corrupted caches
- Restart failed services
- Free memory if needed
- Retry failed requests
- Generate alerts

## Environment Setup

### LMStudio API
```bash
export LMSTUDIO_API_URL="http://localhost:1234"
```

### Custom Context Root
```bash
export AFS_CONTEXT_ROOT="~/.context"
```

## Report Location

Health reports saved to:
```
~/.context/health/
├── report-2024-01-14T10:30:00.json   # Individual reports
├── report-2024-01-14T10:29:00.json
└── trends.json                        # Trend history
```

## Python API

```python
from afs.health import EnhancedHealthChecker, HealthCheckLevel

# Create checker
checker = EnhancedHealthChecker()

# Run check
result = checker.check(level=HealthCheckLevel.COMPREHENSIVE)

# Print summary
print(result.summary())

# Access scores
for score in result.scores:
    print(f"{score.category}/{score.metric}: {score.score:.2f}")

# Save report
import json
with open("health_report.json", "w") as f:
    json.dump(result.to_dict(), f, indent=2)
```

## Monitoring Daemon

```python
from afs.health.daemon import HealthMonitoringDaemon
import asyncio

daemon = HealthMonitoringDaemon(
    check_interval_s=60,
    alert_threshold=0.1,
    auto_heal=True
)

# Run for 24 hours
asyncio.run(daemon.start(duration_s=86400))

# Get trends
trend = daemon.get_trend(hours=24)
print(f"Current: {trend['current']}, Average: {trend['average']}")
```

## Troubleshooting

### "VRAM check unavailable"
GPU tools not installed. Install NVIDIA drivers or ignore for CPU-only systems.

### "LMStudio API check failed"
Verify LMStudio is running:
```bash
curl http://localhost:1234/api/tags
# Should return model list
```

### High memory usage in stress test
Expected behavior. Stress test simulates load. Check trends to detect real leaks:
```bash
afs health trend --hours 168
```

### Healing actions not executing
Check:
1. `--auto-heal` flag is enabled
2. File permissions (cache directories writable)
3. System resources available

## Examples

```bash
# Quick health check
afs health check

# Full diagnostics with auto-healing
afs health check --level comprehensive --auto-heal

# Run stress test
afs health check --level stress

# Start 24-hour monitoring
afs health monitor --duration 86400 --auto-heal

# View status
afs health status

# See trends
afs health trend --hours 24

# Check history
afs health history

# Get JSON for integration
afs health check --json | jq '.overall_score'
```

## Best Practices

1. **Development**: Run `afs health check` daily
2. **Staging**: Run `afs health monitor --interval 300` (5-min checks)
3. **Production**: Run `afs health monitor --interval 60 --auto-heal` (1-min, auto-heal)
4. **CI/CD**: Include `afs health check --level comprehensive` before deployment
5. **Analysis**: Review `afs health trend --hours 168` weekly

## Next Steps

- Read full documentation: `src/afs/health/README.md`
- Run examples: `python3 examples/health_check_example.py`
- View tests: `pytest tests/test_health_checks.py -v`
- Check CLI help: `afs health --help`

## Support

For issues or questions:
1. Check `afs health status` for current state
2. Review `~/.context/health/` for recent reports
3. Run `afs health check --level comprehensive --json` for detailed diagnostics
4. Consult `HEALTH_SYSTEM_IMPLEMENTATION.md` for architecture details
