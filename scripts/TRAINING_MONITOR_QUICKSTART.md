# Training Monitor TUI - Quick Start

## Installation (1 minute)

```bash
cd ~/src/lab/afs
pip install -e .  # Installs rich dependency
chmod +x scripts/training_monitor_tui.py
```

## Basic Usage

### Monitor Specific Instances
```bash
python3 scripts/training_monitor_tui.py --instances 30007012,30007267,30007268
```

### Monitor All Running Instances
```bash
python3 scripts/training_monitor_tui.py --all
```

### Test Without API (Mock Data)
```bash
python3 scripts/training_monitor_tui.py --instances 1,2,3 --mock
```

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Refresh now |
| `p` | Pause updates |
| `c` | Clear logs |

## Display Explained

```
┏━━━━┳━━━━━━━━━━┳━━━━━┳━━━━━━┳━━━━┳━━━━┳━━━━┳━━━━┳━━━━┳━━━━┓
┃ID  ┃GPU Type  ┃Sts  ┃Prog  ┃GPU%┃Mem%┃Loss┃$/hr┃Total┃ETA ┃
┡━━━━╇━━━━━━━━━━╇━━━━━╇━━━━━━╇━━━━╇━━━━╇━━━━╇━━━━╇━━━━╇━━━━┩
│3007│RTX 4090  │✅ ok│E1/3 │80% │42% │2.34│0.47│15.2│45m │
└────┴──────────┴─────┴──────┴────┴────┴────┴────┴────┴────┘
```

**Color coding**:
- 🟢 Green: Healthy
- 🟡 Yellow: Warning (low GPU, high loss, etc.)
- 🔴 Red: Error (instance exited)

## Common Scenarios

### Watch training during launch
```bash
python3 scripts/vastai_setup.py --all-models &
sleep 30
python3 scripts/training_monitor_tui.py --all
```

### Long-running background monitoring
```bash
nohup python3 scripts/training_monitor_tui.py --all > monitor.log 2>&1 &
tail -f monitor.log
```

### High-frequency monitoring
```bash
python3 scripts/training_monitor_tui.py --instances 30007012,30007267 --interval 5
```

### Cost-conscious monitoring
```bash
python3 scripts/training_monitor_tui.py --all --interval 30
```

## What You See

### Main Table
- Instance ID, GPU type, status
- Training progress: epoch/total and percentage
- GPU and memory utilization
- Training loss (should decrease)
- Hourly and total cost
- Estimated time to completion

### Summary Panel
- Running/error instance count
- Total and average hourly cost
- Average resource utilization

### Recent Logs
- Last 10 training outputs
- Shows step progress and loss updates

## Troubleshooting

### "No instances found"
```bash
# Check vast.ai is configured
vastai show instances

# Check API key is set
vastai show user
```

### "Command not found"
```bash
# Make sure vast.ai CLI is installed
which vastai
pip install vastai

# Set API key
vastai set api-key YOUR_KEY
```

### No logs showing
- Logs require SSH access to instances
- Instances need to write training logs
- Use `--mock` to test UI independently

## Advanced Options

### Help
```bash
python3 scripts/training_monitor_tui.py --help
```

### Custom intervals (seconds)
- Fast (5s): `--interval 5`
- Normal (10s): default
- Slow (30s): `--interval 30`

### Debug mode
```bash
python3 scripts/training_monitor_tui.py --instances 1,2,3 2>debug.log
```

## Performance

- **CPU**: <1% idle, <5% during updates
- **Memory**: ~50 MB
- **Network**: ~12 MB/hour (negligible)
- **Update cycle**: 10 seconds (configurable)

## Example Output

```
════════════════════════════════════════════════════════════════════════════════════════════════════
VAST.AI Training Monitor
════════════════════════════════════════════════════════════════════════════════════════════════════
Instance    GPU Type         Status        Progress  GPU%   Memory%  Loss     Cost/hr  Total Cost  ETA
────────────────────────────────────────────────────────────────────────────────────────────────────
30007012    RTX 4090         ✅ running     E1/3 65%  85.2%  42.1%    2.34     $0.467   $15.23      45m
30007267    RTX 3090         ✅ running     E2/3 25%  72.5%  38.9%    3.21     $0.287   $8.45       2h5m
30007268    A100             🔄 loading     E1/3  5%  12.1%  9.3%     N/A      $1.234   $2.10       12h
30007269    H100             ✅ running     E1/3 40%  88.3%  51.2%    1.89     $1.500   $18.75      1h30m
30007270    L40S             ✅ running     E3/3 95%  76.4%  44.6%    0.87     $0.812   $25.50      8m

════════════════════════════════════════════════════════════════════════════════════════════════════
Instance Stats                      Summary
─────────────────────────────────────────────────────────────────────────────────────────────────
Running: 4/5 | Errors: 0           Cost Summary
                                    Total: $70.03 | Per Hour: $4.300
Resource Utilization
Avg GPU: 66.9% | Avg Memory: 37.2% │ q - Quit | r - Refresh | p - Pause | c - Clear
════════════════════════════════════════════════════════════════════════════════════════════════════
```

## Files

- **Main script**: `training_monitor_tui.py` (21 KB)
- **Tests**: `test_training_monitor.py` (36 test cases)
- **Examples**: `training_monitor_examples.sh` (12+ scenarios)
- **Docs**:
  - `TRAINING_MONITOR_README.md` - Full user guide
  - `TRAINING_MONITOR_DESIGN.md` - Technical architecture
  - `TRAINING_MONITOR_SUMMARY.md` - Implementation details

## Next Steps

1. **Test it**: `python3 scripts/training_monitor_tui.py --instances 1,2,3 --mock`
2. **Use it**: Run with your actual instance IDs
3. **Customize**: Add custom alerts or metrics (see TRAINING_MONITOR_DESIGN.md)

## Support

See full documentation:
- User guide: `TRAINING_MONITOR_README.md`
- Technical details: `TRAINING_MONITOR_DESIGN.md`
- Implementation: `TRAINING_MONITOR_SUMMARY.md`
- Examples: `training_monitor_examples.sh`
