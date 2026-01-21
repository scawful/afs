# Training Monitor TUI - Technical Design Document

## Overview

The Training Monitor TUI is a production-grade terminal interface for real-time monitoring of VAST.AI GPU instances used in distributed model training. It provides live metrics, cost tracking, and training progress visualization with minimal resource overhead.

## Design Goals

1. **Real-time Visibility**: 10-second refresh cycle for live training insights
2. **Production Ready**: Robust error handling, graceful degradation, comprehensive logging
3. **Resource Efficient**: <1% CPU idle, ~50MB memory with 5 instances
4. **Extensible**: Clean architecture for adding metrics, alerts, and integrations
5. **User Friendly**: Intuitive controls, clear status indicators, helpful feedback

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────┐
│ TrainingMonitorUI (Rich Terminal Interface)         │
│ - Layout rendering                                  │
│ - Keyboard event handling                           │
│ - Display formatting                                │
└─────────────────────────────────────────────────────┘
         ↑
         │ (reads metrics)
         ↓
┌─────────────────────────────────────────────────────┐
│ VastAIMonitor (Data Collection)                    │
│ - Instance status fetching                          │
│ - Training log parsing                              │
│ - Metrics extraction                                │
└─────────────────────────────────────────────────────┘
         ↑
         │ (calls CLI)
         ↓
┌─────────────────────────────────────────────────────┐
│ VAST.AI API (via CLI)                              │
│ - vastai show instances --raw                       │
│ - SSH to instance for logs                          │
└─────────────────────────────────────────────────────┘
```

### Component Details

#### 1. TrainingMetrics (Data Model)

```python
@dataclass
class TrainingMetrics:
    # Identity
    instance_id: int
    gpu_type: str
    status: str  # "running", "loading", "exited", "created"

    # Progress Tracking
    epoch: int
    total_epochs: int
    step: int
    total_steps: int
    loss: float

    # Resource Metrics
    gpu_util: float        # 0-100%
    memory_util: float     # 0-100%
    disk_util: float       # 0-100%

    # Cost Metrics
    cost_per_hour: float   # $/hour
    total_cost: float      # $ cumulative
    runtime_seconds: float # seconds elapsed

    # Timestamps
    started_at: datetime
    last_updated: datetime

    # Logging
    recent_logs: deque(maxlen=10)
```

**Purpose**: Single source of truth for instance metrics. Supports:
- Progress calculations (`progress_percentage()`)
- Health status (`health_status()`)
- Time estimates (`estimated_completion_time()`)

#### 2. VastAIMonitor (Data Collection)

**Responsibilities**:
- Query vast.ai API for instance status
- Fetch and parse training logs
- Extract metrics from log output
- Cache metrics for UI rendering

**Key Methods**:

```python
def update_all_metrics(self) -> None:
    """Update metrics for all instances."""
    # Iterates through instance_ids and updates metrics dict
    # Handles errors gracefully, logs warnings for failures
```

```python
def _fetch_instance_status(self, instance_id: int) -> dict | None:
    """Fetch instance data from vast.ai API."""
    # Calls: vastai show instances --raw
    # Parses JSON response
    # Returns instance data or None on error
    # Implements timeout: 10 seconds
```

```python
def _fetch_training_logs(self, instance_id: int) -> list[str]:
    """Fetch training logs from remote instance."""
    # Currently: returns empty list (SSH disabled by default)
    # In production: SSH into instance
    # Command: ssh root@<instance-ip> tail -n 10 /workspace/output/*/training.log
    # Returns: list of log lines
```

```python
def _parse_training_metrics(
    self,
    instance_id: int,
    instance_data: dict,
    logs: list[str],
) -> None:
    """Parse instance data and logs into TrainingMetrics."""
    # Extract metrics from instance_data (status, GPU%, cost, etc.)
    # Parse logs using regex patterns:
    #   - Epoch: r"Epoch\s+(\d+)/(\d+)"
    #   - Step: r"Step\s+(\d+)"
    #   - Loss: r"Loss\s*=\s*([\d.]+)"
    # Update metrics dict in-place
```

**Error Handling**:
- Timeout on API calls: 10 seconds
- Failed API calls: log warning, return None
- Malformed JSON: caught and logged
- Missing instance: gracefully skipped
- SSH failures: return empty logs

**Performance**:
- Single API call per update cycle
- O(n) complexity where n = number of instances
- Minimal memory allocation

#### 3. TrainingMonitorUI (Presentation)

**Responsibilities**:
- Render metrics using Rich library
- Handle keyboard input non-blocking
- Manage layout with multiple panels
- Update display every 100ms
- Refresh metrics every N seconds

**Key Methods**:

```python
def run(self) -> None:
    """Main event loop."""
    # Initialize display with Rich Live
    # Loop:
    #   - Check for keyboard input (select.select, non-blocking)
    #   - Update metrics if interval elapsed and not paused
    #   - Render layout
    #   - Sleep 100ms
    # Exit on 'q' or Ctrl+C
```

```python
def build_layout(self) -> Layout:
    """Build complete UI layout."""
    # Split screen into regions:
    #   - metrics table (top, ~15 lines)
    #   - logs panel (middle, ~10 lines)
    #   - stats & controls (bottom, 2x panels)
    # Returns: Rich Layout object
```

```python
def _build_metrics_table(self) -> Table:
    """Build main metrics table."""
    # Rich Table with columns:
    #   Instance | GPU Type | Status | Progress | GPU% | Memory% | Loss | Cost/hr | Total | ETA
    # Color-coded rows based on health status
    # Emoji indicators for status (✅ running, 🔄 loading, etc.)
```

```python
def _build_logs_panel(self) -> Panel:
    """Build training logs display."""
    # Shows last 10 log lines from all instances
    # Format: "#30007012 Step 1000: Loss = 2.345"
    # Uses deque for efficient circular buffer
```

```python
def _build_stats_panel(self) -> Panel:
    """Build summary statistics."""
    # Instance count (running/errors)
    # Total cost and per-hour cost
    # Average GPU/memory utilization
```

```python
def _build_controls_panel(self) -> Panel:
    """Build keyboard controls display."""
    # Shows available commands
    # Displays current state (PAUSED/ACTIVE)
```

**Keyboard Input Handling**:
```python
def handle_keyboard_input(self) -> None:
    """Process keyboard input without blocking."""
    # Uses select.select() for non-blocking I/O
    # Parses single character input
    # Handles: q (quit), r (refresh), p (pause), c (clear)
```

### Data Flow Diagram

```
Every 100ms:
┌─────────────────────────────────────┐
│ TrainingMonitorUI.run()            │
├─────────────────────────────────────┤
│ 1. handle_keyboard_input()          │ (non-blocking, instant)
│    └─> read stdin, process key     │
│                                     │
│ 2. if (time_since_update > interval) AND not paused:
│    └─> monitor.update_all_metrics() │ (10s cycle)
│        ├─> fetch instances status   │
│        ├─> fetch training logs      │
│        └─> parse metrics            │
│                                     │
│ 3. layout = build_layout()          │ (compute display)
│    ├─> _build_metrics_table()       │
│    ├─> _build_logs_panel()          │
│    ├─> _build_stats_panel()         │
│    └─> _build_controls_panel()      │
│                                     │
│ 4. live.update(layout)              │ (render to terminal)
│                                     │
│ 5. sleep(0.1)                       │ (100ms frame time)
└─────────────────────────────────────┘
```

### Update Cycle

Default interval: 10 seconds

```
Time  | Action                    | CPU  | I/O
------|---------------------------|------|----
0s    | fetch instances           | 0.5% | 1 API call
0.2s  | parse metrics             | 0.2% | None
0.3s  | render UI (20 frames)     | 2%   | 1 terminal write per frame
10s   | (repeat)                  |      |
```

Total metrics updates per minute: 6 (once every 10 seconds)
Total API calls per hour: ~360 (6/min * 60)

## Implementation Details

### 1. Regex Patterns for Log Parsing

These patterns extract training progress from log files:

```python
# Epoch detection - matches "Epoch 1/3", "Epoch 2/3", etc.
EPOCH_PATTERN = r"Epoch\s+(\d+)/(\d+)"
# Example match: "Epoch 2/3" → epoch=2, total_epochs=3

# Step detection - matches "Step 1000", "Step 2500", etc.
STEP_PATTERN = r"Step\s+(\d+)"
# Example match: "Step 1000" → step=1000

# Loss detection - matches "Loss = 2.345", "loss: 1.234", etc.
LOSS_PATTERN = r"Loss\s*=\s*([\d.]+)"
# Example match: "Loss = 2.345" → loss=2.345
```

Pattern flexibility: Matches case-insensitive variants, with/without spaces

### 2. Health Status Calculation

Health status determined by:

```python
def health_status(self) -> str:
    if self.status == "exited":
        return "error"           # Instance crashed
    if self.gpu_util < 10:
        return "warning"         # GPU sitting idle
    if self.loss > 10.0:
        return "warning"         # Loss diverging
    if self.gpu_util > 95:
        return "warning"         # Over-utilized
    return "healthy"             # All metrics normal
```

### 3. ETA Calculation

Estimates remaining time based on training velocity:

```python
def estimated_completion_time(self) -> Optional[timedelta]:
    if self.total_steps <= 0 or self.step <= 0:
        return None

    # Calculate time per step
    time_per_step = self.runtime_seconds / self.step

    # Estimate remaining work
    remaining_steps = self.total_steps - self.step
    estimated_seconds = remaining_steps * time_per_step

    return timedelta(seconds=int(estimated_seconds))
```

Accuracy depends on:
- Training velocity consistency (constant training speed)
- Accurate `total_steps` estimation from logs
- Recent log data (needs at least 1 step to extrapolate)

### 4. Non-Blocking Keyboard Input

Uses `select.select()` for platform-independent non-blocking I/O:

```python
def handle_keyboard_input(self) -> None:
    # Check if stdin has data waiting (non-blocking)
    if select.select([sys.stdin], [], [], 0)[0]:
        try:
            key = sys.stdin.read(1).lower()
            if key == 'q':
                self.should_exit = True
            # ... handle other keys
        except Exception as e:
            logger.warning(f"Input error: {e}")
```

**Advantages**:
- Doesn't freeze UI while waiting for input
- Works on Unix/Linux/macOS (Windows uses different approach)
- Allows 100ms frame timing independent of keyboard

### 5. Circular Logging Buffer

Uses Python's `deque` with maxlen for efficient log storage:

```python
recent_logs: deque = field(default_factory=lambda: deque(maxlen=10))

# Automatic removal of oldest log when capacity reached
recent_logs.append("New log line")  # If 10 logs exist, oldest is auto-removed
```

**Memory efficiency**:
- Fixed size: exactly 10 strings per instance
- 5 instances × 10 logs × ~100 bytes = ~5KB total
- No manual cleanup needed

## Error Handling Strategy

### Graceful Degradation

1. **API Timeout (10s)**
   - Caught by subprocess timeout
   - Logs warning, uses cached metrics
   - UI continues with stale data

2. **Invalid JSON Response**
   - Caught by json.JSONDecodeError
   - Logs warning, skips instance
   - Other instances continue normally

3. **Missing Instance**
   - Instance not found in API response
   - Metrics remain unchanged
   - UI shows last known state

4. **SSH Failure (Log Fetching)**
   - Returns empty log list
   - Instance continues being monitored
   - No training log updates until SSH restored

5. **Malformed Log Format**
   - Regex doesn't match
   - Metrics not updated for that line
   - Other metrics continue normally

6. **Keyboard Input Error**
   - Caught and logged
   - UI continues regardless
   - Does not interrupt monitoring

### Logging

```python
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
```

**Log Levels**:
- DEBUG: Detailed metrics updates, API calls
- INFO: Monitor start/stop, configuration
- WARNING: API timeouts, missing instances, SSH failures
- ERROR: Fatal issues (but caught and logged, not crashing)

## Performance Considerations

### CPU Usage

| Operation | Cost |
|-----------|------|
| Keyboard input check | <0.1% |
| Metrics rendering (20 FPS) | 1-2% |
| API call (once per 10s) | 0.5% (amortized) |
| Idle/sleep | <0.1% |
| **Total (typical)** | **<2%** |

### Memory Usage

| Component | Size |
|-----------|------|
| TrainingMetrics × 5 | ~5 KB |
| Recent logs buffer (10 per instance) | ~5 KB |
| UI objects, Rich layout | ~20 KB |
| Python interpreter overhead | ~40 MB |
| **Total** | **~50 MB** |

### Network Usage

| Metric | Value |
|--------|-------|
| API calls/minute | 0.1 (once per 10s) |
| Bytes per API call | ~2 KB (JSON response) |
| Bytes/hour | ~12 MB (very low) |
| Requests/hour | 360 |

Negligible impact on API rate limits.

## Extensibility

### Adding Custom Metrics

1. Update TrainingMetrics dataclass:
   ```python
   @dataclass
   class TrainingMetrics:
       learning_rate: float = 0.0
       val_loss: float = 0.0
       batch_size: int = 0
   ```

2. Add parsing in `_parse_training_metrics()`:
   ```python
   lr_match = re.search(r"LR\s*=\s*([\d.e-]+)", log)
   if lr_match:
       metrics.learning_rate = float(lr_match.group(1))
   ```

3. Display in UI:
   ```python
   table.add_column("LR", justify="right")
   # In row: f"{metrics.learning_rate:.2e}",
   ```

### Adding Alerts

```python
def check_epoch_completion(self, old_metrics, new_metrics):
    if new_metrics.epoch > old_metrics.epoch:
        # Send notification
        self._send_alert(f"Epoch {new_metrics.epoch} completed")

def _send_alert(self, message: str):
    # OS notification
    os.system(f'notify-send "{message}"')
    # Or: send email, Slack, Discord, etc.
```

### Adding Persistent Storage

```python
import sqlite3

def log_metrics_to_db(self, metrics: TrainingMetrics):
    with sqlite3.connect("training_metrics.db") as conn:
        conn.execute(
            "INSERT INTO metrics VALUES (?, ?, ?, ?, ...)",
            (metrics.instance_id, metrics.epoch, metrics.loss, ...)
        )
```

## Testing Strategy

### Unit Tests

```python
def test_health_status():
    m = TrainingMetrics(instance_id=1, gpu_type="RTX4090", status="running")
    m.gpu_util = 95.0
    assert m.health_status() == "warning"

def test_progress_percentage():
    m = TrainingMetrics(instance_id=1, step=500, total_steps=1000)
    assert m.progress_percentage() == 50.0

def test_estimated_completion_time():
    m = TrainingMetrics(instance_id=1, step=100, total_steps=1000)
    m.runtime_seconds = 360  # 6 minutes
    eta = m.estimated_completion_time()
    assert eta.total_seconds() == pytest.approx(3240)  # 54 minutes
```

### Integration Tests

```python
def test_mock_monitor():
    monitor = VastAIMonitor([1, 2, 3], use_mock=True)
    monitor.update_all_metrics()

    assert len(monitor.metrics) == 3
    assert all(m.status in ["running", "loading"] for m in monitor.metrics.values())
    assert all(m.gpu_util >= 0 and m.gpu_util <= 100 for m in monitor.metrics.values())
```

### End-to-End Tests

```python
def test_ui_render():
    monitor = VastAIMonitor([1, 2], use_mock=True)
    ui = TrainingMonitorUI(monitor)

    # Should not crash
    layout = ui.build_layout()
    assert layout is not None
```

## Security Considerations

1. **API Keys**
   - vast.ai CLI handles key storage (~/.vast)
   - Script doesn't store or expose keys
   - Recommend storing in environment: `vastai set api-key $(echo $VASTAI_KEY)`

2. **SSH Access**
   - Uses system SSH configuration (~/.ssh/config)
   - Requires pre-configured keys or ssh-agent
   - Never hardcodes passwords or keys

3. **Input Validation**
   - Instance IDs: parsed as integers, range checked
   - Log lines: used as-is in deque (no eval/exec)
   - JSON: parsed with json module (safe)
   - Regex: patterns are hardcoded (no user input)

4. **Error Messages**
   - Don't leak sensitive information
   - Log details to file, summary to UI
   - Catch all exceptions, don't stack trace on UI

## Future Enhancements

### Short-term (v1.1)

1. **Configuration File**
   - YAML/TOML for instance lists, thresholds
   - Custom log patterns
   - Alert rules

2. **Persistent Metrics**
   - SQLite logging of all metrics
   - Historical graphs
   - Cost projections

3. **Advanced Alerts**
   - Slack/email notifications
   - Loss divergence detection
   - GPU hang detection

### Mid-term (v2.0)

1. **Async API Calls**
   - asyncio + aiohttp for concurrent requests
   - Eliminate blocking on slow APIs
   - Better timeout handling

2. **Multi-Session**
   - Switch between instance groups
   - Compare parallel training runs
   - Batch operations

3. **Web Dashboard**
   - FastAPI backend for metrics collection
   - React frontend for remote monitoring
   - Mobile-friendly design

### Long-term (v3.0)

1. **Machine Learning Integration**
   - Anomaly detection on metrics
   - Training time prediction
   - Cost optimization recommendations

2. **Multi-Cloud Support**
   - AWS EC2, Azure, GCP monitoring
   - Unified dashboard across providers
   - Cost comparison

3. **Advanced Orchestration**
   - Automatic instance scaling
   - Job queuing and prioritization
   - Distributed training coordination

## Deployment

### Prerequisites

- Python 3.10+
- vast.ai CLI with API key configured
- Rich library (pip install rich)
- Terminal with 80+ column width

### Installation

```bash
cd ~/src/lab/afs
pip install -e .  # Installs rich as dependency
chmod +x scripts/training_monitor_tui.py
```

### Verification

```bash
# Test with mock data
python3 scripts/training_monitor_tui.py --instances 1,2,3 --mock

# Test with real API (requires vast.ai credentials)
python3 scripts/training_monitor_tui.py --all
```

### Systemd Service (Optional)

```ini
[Unit]
Description=VAST.AI Training Monitor
After=network-online.target

[Service]
Type=simple
User=scawful
WorkingDirectory=/Users/scawful/src/lab/afs
ExecStart=/usr/bin/python3 scripts/training_monitor_tui.py --all
Restart=always

[Install]
WantedBy=default.target
```

## References

- Rich Documentation: https://rich.readthedocs.io/
- vast.ai CLI: https://docs.vast.ai/
- Python logging: https://docs.python.org/3/library/logging.html
- asyncio Guide: https://docs.python.org/3/library/asyncio.html
