# AFS Health Check System

Comprehensive health monitoring and diagnostics for the AFS system with auto-healing capabilities.

## Features

### Multiple Check Levels

- **Basic**: Model loads, responds to ping (~5 seconds)
- **Standard**: Run 5 test queries, verify outputs (~15 seconds)
- **Comprehensive**: Full evaluation suite, performance benchmarks (~60 seconds)
- **Stress**: High-load testing, memory leak detection (~90 seconds)

### Health Categories

1. **System Health**
   - CPU usage
   - Memory usage
   - Disk space
   - GPU VRAM (if available)

2. **Service Health**
   - LMStudio API availability
   - MCP servers operational status
   - Python dependencies

3. **Model Health**
   - Model load time
   - Inference latency
   - Output quality (via test queries)

4. **Data Health**
   - Training data integrity
   - Cache validity and freshness

5. **Integration Health**
   - External API availability
   - Notification channel configuration

### Health Scoring

Scores range from 0.0 to 1.0 with color-coded status:

```
0.9-1.0: Excellent (GREEN)   ✓
0.7-0.9: Good (YELLOW)       ⚠
0.5-0.7: Degraded (ORANGE)   ⚠⚠
0.0-0.5: Critical (RED)      ✗
```

### Auto-Healing Actions

Automatically triggered when `--auto-heal` is enabled:

- Restart hung services
- Clear corrupted caches
- Free memory (kill idle processes)
- Retry failed requests with exponential backoff
- Rollback to last known good version (future)
- Alert ops team for manual intervention

### Continuous Monitoring

- Checks every 60 seconds (configurable)
- Alerts on score drops >0.1 (configurable)
- Logs all results with structured JSON
- Generates 24-hour trends automatically
- Keeps historical data for analysis

## Usage

### CLI Commands

#### Run a health check

```bash
# Quick basic check
afs health check

# Full diagnostic at standard level
afs health check --level standard

# Comprehensive check with auto-healing
afs health check --level comprehensive --auto-heal

# Stress test the system
afs health check --level stress

# Get JSON output
afs health check --json
```

#### Start monitoring daemon

```bash
# Monitor every 60 seconds
afs health monitor

# Custom interval (30 seconds) at comprehensive level
afs health monitor --interval 30 --level comprehensive

# Run for 24 hours with auto-healing
afs health monitor --duration 86400 --auto-heal

# Monitor with 10-minute interval
afs health monitor --interval 600
```

#### View status and history

```bash
# Get current health status
afs health status

# Show trends for last 24 hours
afs health trend --hours 24

# Show trend for last 7 days
afs health trend --hours 168

# List recent health reports (last 10)
afs health history

# Show last 50 reports
afs health history --limit 50

# Get JSON output
afs health history --json
```

### Python API

```python
from afs.health import EnhancedHealthChecker, HealthCheckLevel

# Create checker
checker = EnhancedHealthChecker()

# Run check
result = checker.check(
    level=HealthCheckLevel.COMPREHENSIVE,
    auto_heal=True,
    save_report=True
)

# Print summary
print(result.summary())

# Access metrics
for score in result.scores:
    print(f"{score.category}/{score.metric}: {score.score:.2f} - {score.message}")

# Get JSON for logging
import json
print(json.dumps(result.to_dict(), indent=2))
```

### Continuous Monitoring

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

# Get trend analysis
trend = daemon.get_trend(hours=24)
print(f"Average score: {trend['average']:.2f}")
print(f"Trend: {trend['trend']}")
```

## Configuration

### Environment Variables

```bash
# Custom LMStudio API URL
export LMSTUDIO_API_URL="http://localhost:1234"

# Custom context root
export AFS_CONTEXT_ROOT="~/.context"
```

### Config File (~/.config/afs/health.toml)

```toml
[thresholds]
cpu_threshold = 85.0          # Alert if CPU >85%
memory_threshold = 80.0        # Alert if memory >80%
disk_threshold = 85.0          # Alert if disk >85%
vram_threshold = 90.0          # Alert if VRAM >90%

[performance]
inference_latency_threshold_ms = 5000.0
model_load_timeout_s = 30.0
api_timeout_s = 10.0

[monitoring]
check_interval_s = 60
alert_threshold = 0.1
keep_reports_count = 100
trend_history_hours = 168  # 1 week

[stress_test]
load = 50                      # Parallel tasks
duration_s = 60                # Test duration
```

## Health Reports

Reports are saved to `~/.context/health/`:

```
~/.context/health/
├── report-2024-01-14T10:30:00.json  # Individual check reports
├── report-2024-01-14T10:29:00.json
├── trends.json                       # Historical trend data
└── daemon.log                        # Daemon logs (if enabled)
```

### Report Structure

```json
{
  "check_level": "comprehensive",
  "timestamp": "2024-01-14T10:30:00.123456",
  "overall_score": 0.87,
  "overall_status": "good",
  "duration_ms": 45230.5,
  "scores": [
    {
      "category": "system",
      "metric": "cpu_usage",
      "score": 0.92,
      "status": "excellent",
      "message": "CPU usage at 8.1%",
      "details": {
        "cpu_percent": 8.1,
        "threshold": 85.0
      },
      "timestamp": "2024-01-14T10:30:00.123456"
    }
  ],
  "checks": [
    {
      "name": "system_health",
      "passed": true,
      "duration_ms": 1250.3,
      "error": null,
      "timestamp": "2024-01-14T10:30:00.123456"
    }
  ],
  "healing_actions": [
    "Cleared cache: ~/.cache/afs"
  ],
  "trends": {
    "system/cpu_usage": [0.88, 0.89, 0.87, ...],
    "system/memory_usage": [0.71, 0.72, 0.71, ...]
  }
}
```

## Alerts and Notifications

When a score drops significantly (default: >0.1), the system:

1. Logs a warning with the drop details
2. Creates a notification event
3. Sends alerts via configured channels:
   - Email
   - Slack
   - Discord
   - Desktop notifications

To configure notifications, see `afs.notifications` module.

## Troubleshooting

### "VRAM check unavailable"

NVIDIA GPU tools not installed. Install:

```bash
pip install pynvml  # For NVIDIA GPUs
```

### "LMStudio API check failed"

Verify LMStudio is running:

```bash
curl http://localhost:1234/api/tags
```

### High memory usage in stress test

This is expected. The stress test simulates load. Monitor trends to detect memory leaks:

```bash
afs health trend --hours 168
```

### Auto-healing not working

Check that:
1. `--auto-heal` flag is enabled
2. Sufficient file permissions exist
3. Check logs: `~/.context/health/daemon.log`

## Architecture

```
EnhancedHealthChecker
├── System Health Check
│   ├── CPU usage
│   ├── Memory usage
│   ├── Disk usage
│   └── VRAM (GPU)
├── Service Health Check
│   ├── LMStudio API
│   ├── MCP servers
│   └── Python dependencies
├── Model Health Check
│   ├── Load time
│   ├── Inference latency
│   └── Output quality
├── Data Health Check
│   ├── Training data integrity
│   └── Cache validity
├── Integration Health Check
│   ├── External APIs
│   └── Notification channels
└── Stress Test
    └── High-load testing & memory leak detection

HealthMonitoringDaemon
├── Periodic health checks (60s intervals)
├── Score trend tracking (24h history)
├── Alert generation on drops
└── Report persistence
```

## Performance

Approximate execution times:

- Basic check: 5-10 seconds
- Standard check: 15-30 seconds
- Comprehensive check: 45-90 seconds
- Stress test: 90-120 seconds

Monitoring overhead: <1% CPU, ~10MB memory per daemon

## Best Practices

1. **Development**: Run `afs health check --level standard` daily
2. **Staging**: Run `afs health monitor --interval 300` (5 min intervals)
3. **Production**: Run `afs health monitor --interval 60 --auto-heal` (1 min, auto-heal)
4. **CI/CD**: Include `afs health check --level comprehensive` before deployment
5. **Analysis**: Review `afs health trend --hours 168` weekly for patterns

## Limitations

- GPU check requires NVIDIA-specific tools (nvidia-smi)
- Some metrics estimated (throughput, quality)
- Healing actions limited to safe operations (no aggressive kills)
- Stress test is lightweight (not production-grade load testing)

## Future Improvements

- Kubernetes integration for distributed health checks
- Machine learning-based anomaly detection
- Predictive alerting (detect issues before they occur)
- Custom health check plugins
- Integration with monitoring systems (Prometheus, Grafana)
- Automatic performance optimization recommendations

## Related

- `afs.notifications` - Alert configuration
- `afs.logging_config` - Structured logging
- `afs.manager` - Context management
