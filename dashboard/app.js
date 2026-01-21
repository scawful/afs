/**
 * AFS Training Dashboard - Frontend Logic
 * Real-time monitoring with auto-refresh every 30 seconds
 */

// Configuration
const CONFIG = {
    API_BASE: 'http://localhost:5000',
    REFRESH_INTERVAL: 30000, // 30 seconds
    CHART_UPDATE_INTERVAL: 5000, // 5 seconds
};

// Global state
let state = {
    darkMode: localStorage.getItem('darkMode') !== 'false',
    lastUpdate: null,
    sessionStart: null,
    charts: {},
    autoRefreshInterval: null,
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
});

/**
 * Initialize dashboard on page load
 */
async function initializeDashboard() {
    // Apply theme
    applyTheme();
    setupEventListeners();

    // Initial data load
    await refreshDashboard();

    // Setup auto-refresh
    state.autoRefreshInterval = setInterval(refreshDashboard, CONFIG.REFRESH_INTERVAL);
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Tab navigation
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', (e) => switchTab(e.target.getAttribute('data-tab')));
    });

    // Theme toggle
    document.getElementById('toggleTheme').addEventListener('click', toggleTheme);

    // Manual refresh
    document.getElementById('refreshBtn').addEventListener('click', refreshDashboard);

    // Export buttons
    document.getElementById('exportCsv').addEventListener('click', () => exportData('csv'));
    document.getElementById('exportJson').addEventListener('click', () => exportData('json'));
}

/**
 * Switch between tabs
 */
function switchTab(tabName) {
    // Update nav
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabName).classList.add('active');

    // Load tab-specific data if needed
    if (tabName === 'metrics') {
        loadMetricsCharts();
    } else if (tabName === 'registry') {
        loadRegistry();
    }
}

/**
 * Apply theme
 */
function applyTheme() {
    const body = document.body;
    if (state.darkMode) {
        body.classList.remove('light-mode');
    } else {
        body.classList.add('light-mode');
    }
}

/**
 * Toggle dark/light theme
 */
function toggleTheme() {
    state.darkMode = !state.darkMode;
    localStorage.setItem('darkMode', state.darkMode);
    applyTheme();

    // Redraw charts with new colors
    Object.values(state.charts).forEach(chart => {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    });
    state.charts = {};

    // Reload metrics if visible
    if (document.getElementById('metrics').classList.contains('active')) {
        loadMetricsCharts();
    }
}

/**
 * Main refresh function
 */
async function refreshDashboard() {
    try {
        showLoading(true);

        // Fetch all data in parallel
        const [statusData, modelsData, costsData] = await Promise.all([
            fetchAPI('/api/training/status'),
            fetchAPI('/api/models/status'),
            fetchAPI('/api/costs/breakdown'),
        ]);

        // Update overview
        updateOverview(statusData, modelsData, costsData);

        // Update models grid
        updateModelsGrid(modelsData.models);

        // Update costs table
        updateCostsTable(costsData.breakdown);

        // Update last refresh time
        updateLastRefresh();

        showLoading(false);
    } catch (error) {
        console.error('Error refreshing dashboard:', error);
        showNotification('Error loading dashboard data', 'error');
        showLoading(false);
    }
}

/**
 * Update overview section
 */
function updateOverview(status, models, costs) {
    // Update stats
    document.getElementById('totalCost').textContent = `$${status.total_cost.toFixed(2)}`;
    document.getElementById('completedModels').textContent = status.models_completed;
    document.getElementById('inProgressModels').textContent = status.models_in_progress;

    // Update session time
    if (!state.sessionStart) {
        state.sessionStart = new Date(status.session_start);
    }
    updateSessionTime();

    // Update models table
    updateModelsTable(models.models);
}

/**
 * Update session time display
 */
function updateSessionTime() {
    if (!state.sessionStart) return;

    const now = new Date();
    const diff = now - state.sessionStart;
    const hours = Math.floor(diff / 3600000);
    const minutes = Math.floor((diff % 3600000) / 60000);
    const seconds = Math.floor((diff % 60000) / 1000);

    document.getElementById('sessionTime').textContent =
        `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

/**
 * Update models table in overview
 */
function updateModelsTable(models) {
    const tbody = document.getElementById('modelsTableBody');
    tbody.innerHTML = '';

    models.forEach(model => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${model.name}</strong></td>
            <td>
                <span class="status-badge ${model.status}">
                    ${model.status}
                </span>
            </td>
            <td>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${model.progress}%"></div>
                </div>
                <small>${model.progress}%</small>
            </td>
            <td>${model.gpu_hours}h</td>
            <td>$${model.total_cost.toFixed(4)}</td>
            <td>${model.training_samples || 'N/A'}</td>
        `;
        tbody.appendChild(row);
    });
}

/**
 * Update models grid
 */
function updateModelsGrid(models) {
    const grid = document.getElementById('modelsGrid');
    grid.innerHTML = '';

    models.forEach(model => {
        const card = document.createElement('div');
        card.className = 'model-card';
        card.innerHTML = `
            <h3>${model.name}</h3>
            <p>${model.description}</p>
            <div class="model-stats">
                <div class="model-stat">
                    <div class="model-stat-label">Status</div>
                    <div class="model-stat-value" style="color: ${getStatusColor(model.status)}">
                        ${model.status}
                    </div>
                </div>
                <div class="model-stat">
                    <div class="model-stat-label">Progress</div>
                    <div class="model-stat-value">${model.progress}%</div>
                </div>
                <div class="model-stat">
                    <div class="model-stat-label">GPU Hours</div>
                    <div class="model-stat-value">${model.gpu_hours}</div>
                </div>
                <div class="model-stat">
                    <div class="model-stat-label">Total Cost</div>
                    <div class="model-stat-value">$${model.total_cost.toFixed(4)}</div>
                </div>
            </div>
        `;
        grid.appendChild(card);
    });
}

/**
 * Update costs table
 */
function updateCostsTable(breakdown) {
    const tbody = document.getElementById('costTableBody');
    tbody.innerHTML = '';

    let total = 0;
    Object.entries(breakdown).forEach(([key, data]) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${data.model_name}</strong></td>
            <td>${data.gpu_hours}</td>
            <td>$${data.cost_per_hour.toFixed(4)}</td>
            <td>$${data.total_cost.toFixed(4)}</td>
        `;
        tbody.appendChild(row);
        total += data.total_cost;
    });

    // Add total row
    const totalRow = document.createElement('tr');
    totalRow.style.fontWeight = 'bold';
    totalRow.style.borderTop = '2px solid var(--color-border)';
    totalRow.innerHTML = `
        <td>TOTAL</td>
        <td></td>
        <td></td>
        <td>$${total.toFixed(4)}</td>
    `;
    tbody.appendChild(totalRow);

    // Update cost chart
    updateCostChart(breakdown);
}

/**
 * Update cost chart
 */
function updateCostChart(breakdown) {
    const ctx = document.getElementById('costChart');
    if (!ctx) return;

    const labels = [];
    const data = [];

    Object.entries(breakdown).forEach(([key, item]) => {
        labels.push(item.model_name);
        data.push(item.total_cost);
    });

    if (state.charts.costChart) {
        state.charts.costChart.destroy();
    }

    state.charts.costChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    '#2196F3',
                    '#4CAF50',
                    '#FFC107',
                    '#FF9800',
                    '#F44336',
                ],
                borderColor: getChartBg(),
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: getChartText(),
                        padding: 15,
                    },
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    callbacks: {
                        label: (context) => `$${context.parsed.toFixed(4)}`,
                    },
                },
            },
        },
    });
}

/**
 * Load metrics charts
 */
async function loadMetricsCharts() {
    try {
        const gpuData = await fetchAPI('/api/metrics/gpu-utilization');
        const lossData = await fetchAPI('/api/metrics/training-loss');
        const throughputData = await fetchAPI('/api/metrics/throughput');

        updateGpuChart(gpuData.metrics);
        updateLossChart(lossData.metrics);
        updateThroughputChart(throughputData.metrics);
        updateMemoryGauge(gpuData.current_memory, gpuData.current_utilization);
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

/**
 * Update GPU chart
 */
function updateGpuChart(metrics) {
    const ctx = document.getElementById('gpuChart');
    if (!ctx) return;

    const labels = metrics.map(m => new Date(m.timestamp).toLocaleTimeString());
    const utilization = metrics.map(m => m.utilization_percent);
    const memory = metrics.map(m => m.memory_used_gb);

    if (state.charts.gpuChart) {
        state.charts.gpuChart.destroy();
    }

    state.charts.gpuChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'GPU Utilization (%)',
                    data: utilization,
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                },
                {
                    label: 'Memory (GB)',
                    data: memory,
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    yAxisID: 'y1',
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false },
            plugins: {
                legend: {
                    labels: { color: getChartText() },
                },
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'Utilization (%)', color: getChartText() },
                    ticks: { color: getChartText() },
                    grid: { color: getChartGrid() },
                    max: 100,
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'Memory (GB)', color: getChartText() },
                    ticks: { color: getChartText() },
                    grid: { drawOnChartArea: false },
                    max: 32,
                },
                x: {
                    ticks: { color: getChartText() },
                    grid: { color: getChartGrid() },
                },
            },
        },
    });
}

/**
 * Update training loss chart
 */
function updateLossChart(metrics) {
    const ctx = document.getElementById('lossChart');
    if (!ctx) return;

    const steps = metrics.map(m => m.step);
    const losses = metrics.map(m => m.loss);
    const valLosses = metrics.map(m => m.validation_loss);

    if (state.charts.lossChart) {
        state.charts.lossChart.destroy();
    }

    state.charts.lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: steps,
            datasets: [
                {
                    label: 'Training Loss',
                    data: losses,
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                },
                {
                    label: 'Validation Loss',
                    data: valLosses,
                    borderColor: '#FFC107',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false },
            plugins: {
                legend: {
                    labels: { color: getChartText() },
                },
            },
            scales: {
                y: {
                    title: { display: true, text: 'Loss', color: getChartText() },
                    ticks: { color: getChartText() },
                    grid: { color: getChartGrid() },
                },
                x: {
                    title: { display: true, text: 'Training Step', color: getChartText() },
                    ticks: { color: getChartText() },
                    grid: { color: getChartGrid() },
                },
            },
        },
    });
}

/**
 * Update throughput chart
 */
function updateThroughputChart(metrics) {
    const ctx = document.getElementById('throughputChart');
    if (!ctx) return;

    const labels = metrics.map(m => new Date(m.timestamp).toLocaleTimeString());
    const sps = metrics.map(m => m.samples_per_second);
    const tps = metrics.map(m => m.tokens_per_second);

    if (state.charts.throughputChart) {
        state.charts.throughputChart.destroy();
    }

    state.charts.throughputChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Samples/Second',
                    data: sps,
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                },
                {
                    label: 'Tokens/Second',
                    data: tps,
                    borderColor: '#FF9800',
                    backgroundColor: 'rgba(255, 152, 0, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    yAxisID: 'y1',
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false },
            plugins: {
                legend: {
                    labels: { color: getChartText() },
                },
            },
            scales: {
                y: {
                    position: 'left',
                    title: { display: true, text: 'Samples/Second', color: getChartText() },
                    ticks: { color: getChartText() },
                    grid: { color: getChartGrid() },
                },
                y1: {
                    position: 'right',
                    title: { display: true, text: 'Tokens/Second', color: getChartText() },
                    ticks: { color: getChartText() },
                    grid: { drawOnChartArea: false },
                },
                x: {
                    ticks: { color: getChartText() },
                    grid: { color: getChartGrid() },
                },
            },
        },
    });
}

/**
 * Update memory gauge
 */
function updateMemoryGauge(used, utilization) {
    document.getElementById('memoryUsed').textContent = used.toFixed(1);
    document.getElementById('memoryPercent').textContent = Math.round(utilization) + '%';
    document.getElementById('memoryValue').textContent = `${used.toFixed(1)}/32GB`;
}

/**
 * Load model registry
 */
async function loadRegistry() {
    try {
        const data = await fetchAPI('/api/models/registry');
        updateRegistryGrid(data.models);
    } catch (error) {
        console.error('Error loading registry:', error);
    }
}

/**
 * Update registry grid
 */
function updateRegistryGrid(models) {
    const grid = document.getElementById('registryGrid');
    grid.innerHTML = '';

    models.forEach(model => {
        const card = document.createElement('div');
        card.className = 'registry-card';
        card.innerHTML = `
            <h3>${model.name}</h3>
            <div class="registry-info">
                <div class="registry-info-row">
                    <span class="registry-info-label">Version</span>
                    <span class="registry-info-value">${model.version}</span>
                </div>
                <div class="registry-info-row">
                    <span class="registry-info-label">Status</span>
                    <span class="registry-info-value">${model.status}</span>
                </div>
                <div class="registry-info-row">
                    <span class="registry-info-label">Samples</span>
                    <span class="registry-info-value">${model.training_samples}</span>
                </div>
                <div class="registry-info-row">
                    <span class="registry-info-label">Size</span>
                    <span class="registry-info-value">${model.file_size_mb.toFixed(1)} MB</span>
                </div>
                <div class="registry-info-row">
                    <span class="registry-info-label">Deployment</span>
                    <span class="registry-info-value">${model.deployment_status}</span>
                </div>
                ${model.evaluation_score ? `
                <div class="registry-info-row">
                    <span class="registry-info-label">Eval Score</span>
                    <span class="registry-info-value">${(model.evaluation_score * 100).toFixed(1)}%</span>
                </div>
                ` : ''}
            </div>
            <div class="registry-actions">
                <button class="btn-small" onclick="copyToClipboard('${model.key}')">Copy Key</button>
                ${model.deployment_status === 'ready' ? `
                <button class="btn-small" onclick="downloadModel('${model.download_url}')">Download</button>
                ` : `
                <button class="btn-small" disabled>Not Ready</button>
                `}
            </div>
        `;
        grid.appendChild(card);
    });
}

/**
 * Fetch API endpoint
 */
async function fetchAPI(endpoint) {
    const response = await fetch(`${CONFIG.API_BASE}${endpoint}`);
    if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
    }
    return await response.json();
}

/**
 * Update last refresh time
 */
function updateLastRefresh() {
    const now = new Date();
    state.lastUpdate = now;
    document.getElementById('lastUpdate').textContent = now.toLocaleTimeString();
}

/**
 * Show/hide loading modal
 */
function showLoading(show) {
    const modal = document.getElementById('loadingModal');
    if (show) {
        modal.classList.add('active');
    } else {
        modal.classList.remove('active');
    }
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    // Could integrate a toast library here
    console.log(`[${type.toUpperCase()}] ${message}`);
}

/**
 * Export data as CSV or JSON
 */
async function exportData(format) {
    try {
        const url = format === 'csv'
            ? `${CONFIG.API_BASE}/api/export/csv`
            : `${CONFIG.API_BASE}/api/export/json`;

        const response = await fetch(url);
        if (!response.ok) throw new Error('Export failed');

        const blob = await response.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = response.headers.get('Content-Disposition')?.split('filename="')[1]?.replace('"', '') ||
            `export.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(downloadUrl);
        a.remove();

        showNotification(`Exported as ${format.toUpperCase()}`, 'success');
    } catch (error) {
        console.error('Export error:', error);
        showNotification('Export failed', 'error');
    }
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard', 'success');
    });
}

/**
 * Download model
 */
function downloadModel(url) {
    window.open(url, '_blank');
}

/**
 * Get status color
 */
function getStatusColor(status) {
    const colors = {
        'running': '#4CAF50',
        'completed': '#4CAF50',
        'loading': '#FFC107',
        'pending': '#9DB3BF',
        'failed': '#F44336',
    };
    return colors[status] || '#666';
}

/**
 * Get chart colors based on theme
 */
function getChartBg() {
    return state.darkMode ? '#1A1F28' : '#FFFFFF';
}

function getChartText() {
    return state.darkMode ? '#ECEFF1' : '#212121';
}

function getChartGrid() {
    return state.darkMode ? 'rgba(55, 71, 79, 0.2)' : 'rgba(0, 0, 0, 0.1)';
}

// Update session time every second
setInterval(updateSessionTime, 1000);
