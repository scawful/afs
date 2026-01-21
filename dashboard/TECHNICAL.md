# AFS Training Dashboard - Technical Documentation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Browser (Client)                     │
│  ┌──────────────┐  ┌────────┐  ┌──────────────────────┐   │
│  │  index.html  │──│ app.js │──│   styles.css         │   │
│  │              │  │        │  │  (dark/light mode)   │   │
│  │  Dashboard   │  │Chart   │  │  animations          │   │
│  │  UI Layout   │  │.js     │  │  responsive grid     │   │
│  │              │  │        │  │                      │   │
│  └──────────────┘  └────────┘  └──────────────────────┘   │
│        ↑              ↑ ↓              ↑ ↓                    │
│        └──────────────┼────────────────┼──────────────┐      │
│                       │ HTTP Fetch     │              │      │
│                    ┌──▼──────────────────────┐        │      │
│                    │  Data Updates (30s)     │        │      │
│                    └──────────────────────────┘        │      │
│                                                        │      │
└────────────────────────────────────────────────────────┼──────┘
                                                         │
                                              ┌──────────▼──────────┐
                                              │   Flask Backend     │
                                              │     (api.py)        │
                                              ├────────────────────┤
                                              │  REST API Routes:   │
                                              │  ├─ /api/health    │
                                              │  ├─ /api/models/*  │
                                              │  ├─ /api/costs/*   │
                                              │  ├─ /api/metrics/* │
                                              │  └─ /api/export/*  │
                                              └──────────┬─────────┘
                                                         │
                            ┌────────────────────────────┼────────────────────────────┐
                            │                            │                            │
                    ┌───────▼────────┐        ┌─────────▼──────────┐     ┌──────────▼────────┐
                    │  models/*.json │        │ evaluations/*.json │     │  System Metrics   │
                    │  Training data │        │ Eval results       │     │  (GPU, Memory)    │
                    │  Sample counts │        │ Scores             │     │  Simulated/Real   │
                    └────────────────┘        └────────────────────┘     └───────────────────┘
```

## Component Details

### Frontend (Client-Side)

#### index.html
- **Lines:** 285
- **Structure:** 5 main tabs
  - Overview: Key metrics + status table
  - Models: Individual model cards
  - Costs: Cost breakdown + table
  - Metrics: Performance graphs
  - Registry: Model list with downloads

**Key Elements:**
```html
<div class="dashboard-header">
  <!-- Dark mode toggle, refresh button, last update indicator -->
</div>

<div class="dashboard-nav">
  <!-- Tab navigation buttons -->
</div>

<div class="dashboard-content">
  <!-- 5 tab sections with different content -->
</div>

<div class="dashboard-footer">
  <!-- Export controls, info -->
</div>
```

#### styles.css
- **Lines:** 600+
- **Features:**
  - CSS custom properties for theming
  - Dark mode (default) + light mode
  - Mobile responsive (breakpoints: 1200px, 768px, 480px)
  - Smooth animations and transitions
  - Grid layouts for responsive design
  - Custom scrollbar styling
  - Print stylesheet

**Color Scheme (Dark Mode):**
```css
--color-bg: #0F1419          /* Very dark blue-gray */
--color-bg-secondary: #1A1F28 /* Darker blue */
--color-card-bg: #1A1F28     /* Card background */
--color-primary: #2196F3     /* Material Blue */
--color-text: #ECEFF1        /* Very light blue-gray */
```

**Grid System:**
```css
.stats-grid {
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  /* Responsive 1-4 columns based on width */
}

.models-grid {
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  /* Model cards fit dynamically */
}
```

#### app.js
- **Lines:** 700+
- **Modules:**

1. **Initialization**
   ```javascript
   initializeDashboard()        // Setup on load
   setupEventListeners()        // Attach handlers
   applyTheme()                 // Apply dark/light mode
   ```

2. **Data Fetching**
   ```javascript
   refreshDashboard()           // Main refresh (30s interval)
   fetchAPI(endpoint)           // Generic fetch wrapper
   ```

3. **UI Updates**
   ```javascript
   updateOverview()             // Stats + table
   updateModelsGrid()           // Model cards
   updateCostsTable()           // Cost breakdown
   loadMetricsCharts()          // Performance graphs
   loadRegistry()               // Model registry
   ```

4. **Chart.js Integration**
   ```javascript
   updateCostChart()            // Doughnut chart
   updateGpuChart()             // Dual-axis GPU + memory
   updateLossChart()            // Training vs validation loss
   updateThroughputChart()      // Samples/sec + tokens/sec
   ```

5. **Utilities**
   ```javascript
   toggleTheme()                // Dark/light mode toggle
   exportData()                 // CSV/JSON export
   copyToClipboard()            // Model key copy
   getChartBg/Text/Grid()       // Theme-aware colors
   ```

**Auto-Refresh Architecture:**
```javascript
// Main refresh loop (30 seconds)
state.autoRefreshInterval = setInterval(refreshDashboard, 30000);

// Session time updates (1 second)
setInterval(updateSessionTime, 1000);

// Tab-specific loads
switchTab() → loadMetricsCharts() || loadRegistry()
```

### Backend (Server-Side)

#### api.py
- **Lines:** 360
- **Framework:** Flask 2.0+ with Flask-CORS
- **Data Source:** Filesystem (no database)

**Key Components:**

1. **Configuration**
   ```python
   MODELS_CONFIG = {dict}      # Model metadata
   TRAINING_STATE = {dict}     # Global training state
   PROJECT_ROOT = Path()       # Project directory
   MODELS_DIR = Path()         # Models directory
   EVALUATIONS_DIR = Path()    # Evaluations directory
   ```

2. **Data Loading Functions**
   ```python
   load_model_files()          # Read *_merged.jsonl files
   load_evaluation_results()   # Read evaluation/*.json
   calculate_model_status()    # Compute status for model
   ```

3. **API Endpoints (13 total)**

   **Health Check:**
   ```
   GET /api/health
   → {"status": "healthy", "timestamp": "..."}
   ```

   **Training Status:**
   ```
   GET /api/training/status
   → {
       "session_start": "...",
       "total_cost": 0.96,
       "models_completed": 1,
       "models_in_progress": 0,
       "models_pending": 4,
       "estimated_completion": "..."
     }
   ```

   **Models:**
   ```
   GET /api/models/status
   → {"models": [...], "timestamp": "..."}

   GET /api/models/<key>/status
   → {"key": "majora", "name": "...", "status": "...", ...}
   ```

   **Costs:**
   ```
   GET /api/costs/breakdown
   → {
       "breakdown": {
         "majora": {"total_cost": 0.96, ...},
         ...
       },
       "total_cost": 4.08,
       "hourly_rate": 1.20
     }
   ```

   **Metrics:**
   ```
   GET /api/metrics/gpu-utilization
   → {"metrics": [...24h data], "current_utilization": 82}

   GET /api/metrics/training-loss
   → {"metrics": [...loss data]}

   GET /api/metrics/throughput
   → {"metrics": [...throughput data]}
   ```

   **Registry:**
   ```
   GET /api/models/registry
   → {
       "models": [
         {
           "key": "majora",
           "name": "Majora v1",
           "status": "completed",
           "training_samples": 500,
           "file_size_mb": 2400,
           "evaluation_score": 0.85,
           "deployment_status": "ready",
           ...
         },
         ...
       ]
     }
   ```

   **Export:**
   ```
   GET /api/export/csv
   → (CSV file download)

   GET /api/export/json
   → (JSON file download)
   ```

   **Manual Updates:**
   ```
   POST /api/update-status
   → {"model_key": "majora", "status": "running", "progress": 45}
   ```

## Data Flow

### Initial Load
```
Page Load
  ↓
initializeDashboard()
  ├─ applyTheme() [localStorage]
  ├─ setupEventListeners()
  ├─ refreshDashboard() [immediate]
  └─ setInterval(refreshDashboard, 30000) [start polling]

refreshDashboard()
  ├─ fetchAPI('/api/training/status')
  ├─ fetchAPI('/api/models/status')
  ├─ fetchAPI('/api/costs/breakdown')
  └─ Update UI
```

### Real-Time Updates
```
Every 30 seconds:
  refreshDashboard()
    ├─ Fetch latest data from Flask
    ├─ Update overview stats
    ├─ Update models table
    ├─ Update costs table
    └─ updateLastRefresh()

Every 1 second:
  updateSessionTime()
    └─ Update elapsed time display

On Tab Switch:
  switchTab('metrics')
    └─ loadMetricsCharts() [fetch + render]
```

## Model Status Calculation

```python
def calculate_model_status(model_key):
    config = MODELS_CONFIG[model_key]
    model_files = load_model_files()
    eval_results = load_evaluation_results()

    return {
        "name": config["name"],
        "status": config["status"],
        "progress": config["progress"],
        "gpu_hours": config["gpu_hours"],
        "cost_per_hour": config["cost_per_hour"],
        "total_cost": gpu_hours * cost_per_hour,
        "training_samples": count_lines(*_merged.jsonl),
        "file_size_mb": stat().st_size,
        "evaluation_score": eval_results.get(...),
    }
```

## Chart Configuration

### Cost Chart (Doughnut)
```javascript
{
  type: 'doughnut',
  data: {
    labels: ['Majora v1', 'Veran v5', ...],
    datasets: [{
      data: [0.96, 0.84, 0.72, 0.84, 0.72],
      backgroundColor: ['#2196F3', '#4CAF50', '#FFC107', '#FF9800', '#F44336'],
    }]
  }
}
```

### GPU Chart (Dual-Axis Line)
```javascript
{
  type: 'line',
  data: {
    datasets: [
      {
        label: 'GPU Utilization (%)',
        yAxisID: 'y',
        data: [...],
      },
      {
        label: 'Memory (GB)',
        yAxisID: 'y1',
        data: [...],
      }
    ]
  },
  options: {
    scales: {
      y: { max: 100, title: 'Utilization (%)' },
      y1: { max: 32, title: 'Memory (GB)', position: 'right' }
    }
  }
}
```

## Performance

### Data Size
- **Model files:** ~500KB total (*_merged.jsonl)
- **API response:** ~5-50 KB per endpoint
- **Chart data:** 24h metrics = ~2-5 KB
- **Page size:** ~78 KB (HTML + CSS + JS)

### Load Times
- **Initial page:** ~500ms (DOM parse + Chart.js load)
- **API fetch:** ~50-100ms (local filesystem)
- **Chart render:** ~100-200ms (Chart.js)
- **Total refresh:** ~300-400ms

### Browser Memory
- **DOM:** ~2-3 MB
- **Charts:** ~5-8 MB (multiple Chart instances)
- **Data cache:** ~1-2 MB
- **Total:** ~10-15 MB typical

## Extension Points

### Add New Metric
1. Add endpoint in `api.py`
2. Add fetch call in `app.js`
3. Add chart in `index.html`
4. Add styling in `styles.css`

### Connect Real Data
Replace simulated functions:
- `gpu_utilization()` → nvidia-smi parsing
- `training_loss()` → TensorBoard logs
- `throughput()` → Training log parsing

### Add Database
Replace filesystem calls:
- `load_model_files()` → SQL query
- `load_evaluation_results()` → SQL query
- `TRAINING_STATE` → Database records

### Deploy to Production
1. Use production Flask server (Gunicorn, uWSGI)
2. Add reverse proxy (Nginx)
3. Enable HTTPS
4. Add authentication
5. Use database for persistence

## Debugging

### Browser Console
```javascript
// Check last update time
console.log(state.lastUpdate);

// Check chart state
console.log(state.charts);

// Force refresh
refreshDashboard();

// Check theme
console.log(state.darkMode);
```

### Server Logs
```bash
# Run with verbose output
FLASK_ENV=development python3 api.py

# Check requests
curl -v http://localhost:5000/api/health
```

### Network Inspection
```bash
# Check all requests
curl http://localhost:5000/api/training/status | jq

# Check endpoint availability
for endpoint in health training/status models/status costs/breakdown; do
  echo "Testing $endpoint..."
  curl -s http://localhost:5000/api/$endpoint | jq .
done
```

## Limitations & Future Work

### Current Limitations
- Simulated metrics (not connected to real GPU data)
- No persistent history (data lost on restart)
- Single-server deployment
- No authentication
- Manual status updates required

### Planned Enhancements
- [ ] WebSocket for real-time updates (vs 30-second polling)
- [ ] PostgreSQL/MongoDB for history
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] vast.ai API integration
- [ ] TensorBoard integration
- [ ] Slack/Discord webhooks
- [ ] User authentication & RBAC
- [ ] Multi-instance monitoring
- [ ] Email alerts
- [ ] Mobile app

---

**Version:** 1.0
**Created:** 2026-01-14
**Status:** Production-ready for simulation + monitoring
