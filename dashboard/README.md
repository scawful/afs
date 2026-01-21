# AFS Training Dashboard

Real-time web dashboard for monitoring AFS training system progress, costs, and performance metrics.

## Features

### 1. Live Training Status
- Real-time table showing all 5 models (Majora, Veran, Din, Nayru, Farore)
- GPU allocation, cost per hour, and training progress
- Color-coded status indicators (green=running, yellow=loading, red=failed)
- Estimated completion times
- Auto-refresh every 30 seconds

### 2. Cost Tracking
- Current hourly burn rate display
- Total session cost with historical breakdown
- Per-model cost visualization
- Interactive doughnut chart
- CSV/JSON export capabilities

### 3. Performance Metrics
- GPU utilization graphs (24-hour history)
- Training loss curves (training vs validation)
- Throughput metrics (samples/sec, tokens/sec)
- Memory usage gauges
- Real-time performance indicators

### 4. Model Registry
- Complete list of all trained models with versions
- Model status and deployment readiness
- Evaluation scores
- Training sample counts and file sizes
- Download links for GGUF files
- Quick-copy model keys

### 5. User Experience
- Dark mode (default) and light mode toggle
- Mobile-responsive design
- Smooth animations and transitions
- Real-time clock and session timer
- Export data as CSV or JSON
- Manual refresh control
- Scrollable tables for large datasets

## Quick Start

### Prerequisites
- Python 3.8+
- macOS, Linux, or Windows

### Installation

1. **Navigate to dashboard directory:**
   ```bash
   cd /Users/scawful/src/lab/afs/dashboard
   ```

2. **Run the launch script:**
   ```bash
   ./serve.sh
   ```

   The script will:
   - Create a Python virtual environment (if needed)
   - Install dependencies (Flask, Flask-CORS)
   - Start the Flask backend on `http://localhost:5000`
   - Open the dashboard in your default browser

3. **Access the dashboard:**
   - Open `http://localhost:5000` in your web browser
   - Dashboard auto-refreshes every 30 seconds

### Manual Setup (if serve.sh fails)

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or: .venv\Scripts\activate (Windows)

# Install dependencies
pip install flask flask-cors

# Start the server
python3 api.py

# Open http://localhost:5000 in your browser
```

## Architecture

### Frontend (`index.html`, `app.js`, `styles.css`)
- **Framework:** Vanilla JavaScript, HTML5, CSS3
- **Charts:** Chart.js for real-time graphs
- **Features:**
  - Tab-based navigation (Overview, Models, Costs, Metrics, Registry)
  - Dark/light mode with localStorage persistence
  - Responsive grid layout
  - Auto-refresh with 30-second interval
  - Smooth animations and transitions

### Backend (`api.py`)
- **Framework:** Flask with CORS support
- **REST API Endpoints:**
  - `GET /api/health` - Health check
  - `GET /api/training/status` - Overall session status
  - `GET /api/models/status` - All model status
  - `GET /api/models/<model_key>/status` - Single model status
  - `GET /api/costs/breakdown` - Cost breakdown by model
  - `GET /api/metrics/gpu-utilization` - GPU metrics (24h history)
  - `GET /api/metrics/training-loss` - Training loss curves
  - `GET /api/metrics/throughput` - Throughput metrics
  - `GET /api/models/registry` - Model registry with versions
  - `GET /api/export/csv` - Export all data as CSV
  - `GET /api/export/json` - Export all data as JSON
  - `POST /api/update-status` - Manual status updates

## Models Tracked

| Model | Purpose | GPU Hours | Cost/Hour |
|-------|---------|-----------|-----------|
| Majora v1 | Oracle of Secrets expert | 4.0 | $0.24 |
| Veran v5 | Advanced verification | 3.5 | $0.24 |
| Din v4 | Creative dialogue generation | 3.0 | $0.24 |
| Nayru v7 | Assembly & architecture | 3.5 | $0.24 |
| Farore v6 | Task planning & decomposition | 3.0 | $0.24 |

**Total Training Cost:** ~$4.32 (17 GPU hours)

## Data Sources

The dashboard reads data from:
- **Model files:** `models/*_merged.jsonl` (training samples)
- **Evaluation results:** `evaluations/results/*.json`
- **System metrics:** Simulated in backend (can be connected to real GPU monitoring)

## Customization

### Update Model Configuration

Edit the `MODELS_CONFIG` dictionary in `api.py`:

```python
MODELS_CONFIG = {
    "your_model": {
        "name": "Model Display Name",
        "description": "Model purpose",
        "gpu_hours": 3.5,
        "cost_per_hour": 0.24,
        "status": "pending",
        "progress": 0,
    },
    # ... more models
}
```

### Integrate Real GPU Metrics

Replace simulated data in `api.py`:
- `gpu_utilization()` - Connect to nvidia-smi or system monitoring
- `training_loss()` - Read from TensorBoard logs
- `throughput()` - Parse training logs

### Connect to Actual Model Status

Update the `calculate_model_status()` function to:
- Monitor actual training processes
- Parse real training logs
- Check model file timestamps
- Verify GGUF conversion status

## API Usage Examples

### Get Overall Status
```bash
curl http://localhost:5000/api/training/status
```

### Get All Model Status
```bash
curl http://localhost:5000/api/models/status
```

### Export Data as CSV
```bash
curl http://localhost:5000/api/export/csv > dashboard.csv
```

### Export Data as JSON
```bash
curl http://localhost:5000/api/export/json > dashboard.json
```

### Update Model Status
```bash
curl -X POST http://localhost:5000/api/update-status \
  -H "Content-Type: application/json" \
  -d '{
    "model_key": "majora",
    "status": "running",
    "progress": 45
  }'
```

## Styling

### Theme Variables

Located in `styles.css` `:root` selector:

```css
--color-primary: #2196F3         /* Primary accent color */
--color-success: #4CAF50         /* Success states */
--color-warning: #FFC107         /* Warning states */
--color-danger: #F44336          /* Error states */
--spacing-lg: 1.5rem             /* Spacing unit */
--radius-lg: 12px                /* Border radius */
```

### Dark Mode

Default dark theme with professional color scheme:
- Background: `#0F1419`
- Cards: `#1A1F28`
- Text: `#ECEFF1`
- Accents: Material Design Blue

### Light Mode

Alternatively toggle to light theme:
- Background: `#F5F7FA`
- Cards: `#FFFFFF`
- Text: `#212121`
- Same accent colors

## Performance Considerations

- **Data Size:** Handles 5 models + 24 hours of metrics efficiently
- **Update Frequency:** 30-second auto-refresh (configurable in `app.js`)
- **Chart Rendering:** Chart.js with 1000+ data points supported
- **Memory:** ~10-15 MB typical usage
- **Network:** ~50KB per refresh cycle

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 5000
lsof -ti :5000 | xargs kill -9

# Or use different port by editing api.py:
# app.run(host="0.0.0.0", port=5001)
```

### CORS Errors
Flask-CORS is enabled. If still seeing errors, check:
- Browser developer console (F12 → Console)
- Server is running (`http://localhost:5000/api/health`)
- Network tab shows requests with correct URLs

### No Data Displayed
1. Check server is running: `ps aux | grep api.py`
2. Check API responds: `curl http://localhost:5000/api/health`
3. Check browser console for errors: `F12 → Console`
4. Verify data files exist: `ls /Users/scawful/src/lab/afs/models/`

### Charts Not Loading
- Clear browser cache (Ctrl+Shift+Delete / Cmd+Shift+Delete)
- Check Chart.js CDN is accessible
- Verify data format in API response

## Browser Compatibility

- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Development

### Local Testing

```bash
# Start server in debug mode
FLASK_ENV=development python3 api.py

# Watch for API changes (auto-reload)
# Flask will restart on file changes
```

### Inspect Network Requests

1. Open DevTools: `F12`
2. Network tab
3. Refresh page
4. See all API calls and responses

### Modify Styling

Edit `styles.css` and refresh browser (Ctrl+R / Cmd+R)

### Update Charts

Edit `app.js` Chart.js configuration options

## Future Enhancements

- [ ] WebSocket real-time updates (vs 30-second polling)
- [ ] Historical cost trends (week/month views)
- [ ] Model comparison tools
- [ ] Alert system for training failures
- [ ] Notification webhooks (Slack, Discord)
- [ ] Database backend for persistent history
- [ ] Multi-user access control
- [ ] Training job queue visualization
- [ ] Automated model download/conversion status
- [ ] Integration with vast.ai API for live training status

## File Structure

```
/Users/scawful/src/lab/afs/dashboard/
├── index.html          # Main dashboard HTML
├── app.js             # Frontend logic & Chart.js integration
├── styles.css         # All styling (dark/light modes)
├── api.py             # Flask backend with REST API
├── serve.sh           # Launch script
└── README.md          # This file
```

## License

Same as AFS project

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review browser console logs (F12)
3. Check server logs in terminal
4. Verify model data files exist
5. Create an issue with error messages
