# Quick Start Guide - AFS Training Dashboard

## 30-Second Setup

```bash
cd /Users/scawful/src/lab/afs/dashboard
./serve.sh
```

That's it! The dashboard opens automatically in your browser.

## What Just Happened?

1. ✓ Created Python virtual environment
2. ✓ Installed Flask + Flask-CORS
3. ✓ Started backend server on `http://localhost:5000`
4. ✓ Opened dashboard in your default browser

## Dashboard Features

### Tabs
- **Overview** - Key metrics and training status table
- **Models** - Individual model cards with stats
- **Costs** - Cost breakdown and doughnut chart
- **Metrics** - GPU, loss, throughput, and memory graphs
- **Registry** - Model registry with download options

### Controls
- **Dark/Light Toggle** - Top right corner (moon icon)
- **Manual Refresh** - Top right corner (refresh icon)
- **Export CSV/JSON** - Bottom footer buttons
- **Auto-Refresh** - Every 30 seconds (automatic)

## 5 Models Monitored

```
┌─────────────┬──────────────────────────┬────────────┬─────────┐
│ Model       │ Purpose                  │ GPU Hours  │ Cost    │
├─────────────┼──────────────────────────┼────────────┼─────────┤
│ Majora v1   │ Oracle of Secrets expert │ 4.0 hrs    │ $0.96   │
│ Veran v5    │ Advanced verification    │ 3.5 hrs    │ $0.84   │
│ Din v4      │ Creative dialogue        │ 3.0 hrs    │ $0.72   │
│ Nayru v7    │ Assembly & architecture  │ 3.5 hrs    │ $0.84   │
│ Farore v6   │ Planning & decomposition │ 3.0 hrs    │ $0.72   │
└─────────────┴──────────────────────────┴────────────┴─────────┘
```

**Total:** 17 GPU hours = ~$4.08

## Files Created

| File | Size | Purpose |
|------|------|---------|
| `api.py` | 13 KB | Flask backend with REST API |
| `index.html` | 11 KB | Dashboard HTML structure |
| `app.js` | 23 KB | Frontend logic & real-time updates |
| `styles.css` | 18 KB | Dark/light mode styling |
| `serve.sh` | 3.9 KB | Automated launch script |
| `README.md` | 9.1 KB | Full documentation |

**Total:** ~78 KB

## Architecture

```
Browser (localhost:5000)
    ↓
index.html (HTML structure)
    ↓ calls
app.js (30-second auto-refresh)
    ↓ fetches from
api.py (Flask backend)
    ↓ reads
../models/
    ├── majora_v1_merged.jsonl
    ├── veran_v5_merged.jsonl
    ├── din_v2_merged.jsonl
    ├── nayru_v6_merged.jsonl
    └── farore_v6_merged.jsonl
```

## Key Metrics

### Real-Time
- Model training status (running/pending/completed)
- Training progress percentage
- GPU cost per hour
- Session elapsed time

### Graphs (24-hour history)
- GPU utilization (%)
- Memory usage (GB)
- Training loss curves
- Throughput (samples/sec, tokens/sec)

### Registry
- Model versions and sizes
- Evaluation scores
- Deployment status
- Training sample counts

## Customization

### Change Refresh Interval

Edit `app.js` line 17:
```javascript
REFRESH_INTERVAL: 30000,  // Change to 10000 for 10 seconds
```

### Change Port

Edit `api.py` line 244:
```python
app.run(host="0.0.0.0", port=5001)  # Use 5001 instead
```

### Add More Models

Edit `api.py` line 22-43:
```python
MODELS_CONFIG = {
    "your_model": {
        "name": "Your Model Name",
        "description": "What it does",
        "gpu_hours": 3.5,
        "cost_per_hour": 0.24,
        "status": "pending",
        "progress": 0,
    },
}
```

## Stopping the Dashboard

Press `Ctrl+C` in the terminal

```
Shutting down server...
Server stopped
```

## Troubleshooting

### "Port 5000 already in use"
```bash
# Kill the process on port 5000
lsof -ti :5000 | xargs kill -9
# Then run serve.sh again
```

### "Module not found: flask"
The script will auto-install. If manual:
```bash
pip install flask flask-cors
```

### "No data showing"
1. Verify model files exist:
   ```bash
   ls /Users/scawful/src/lab/afs/models/*merged.jsonl
   ```
2. Check API health:
   ```bash
   curl http://localhost:5000/api/health
   ```

### Browser won't open
Manually open: `http://localhost:5000`

## API Examples

### Check Server Health
```bash
curl http://localhost:5000/api/health
# {"status": "healthy", "timestamp": "2026-01-14T06:30:45.123456"}
```

### Get Training Status
```bash
curl http://localhost:5000/api/training/status
# Shows total cost, models completed, time elapsed
```

### Export Data as CSV
```bash
curl http://localhost:5000/api/export/csv > report.csv
```

## Features

✓ Real-time monitoring (30-second auto-refresh)
✓ 5 model tracking (Majora, Veran, Din, Nayru, Farore)
✓ Cost breakdown visualization
✓ GPU & memory monitoring
✓ Training loss curves
✓ Throughput metrics
✓ Model registry with evaluation scores
✓ Dark/light mode toggle
✓ Mobile responsive
✓ CSV/JSON export
✓ Auto-open in default browser
✓ No external dependencies (except Flask)

## Next Steps

1. **Monitoring:** Leave dashboard open while training
2. **Alerts:** Set up browser notifications (future enhancement)
3. **Data Export:** Download CSV reports regularly
4. **Integration:** Connect to real GPU monitoring APIs
5. **Deployment:** Deploy to server for team access

---

**Status:** Dashboard ready to monitor 5 AFS models
**Total Training Cost:** ~$4.08
**Estimated Duration:** ~17 GPU hours

Happy monitoring! 🚀
