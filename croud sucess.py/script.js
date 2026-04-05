// ============================================================================
// AI Crowd Monitoring System - Frontend JavaScript
// ============================================================================

// Configuration
const CONFIG = {
    updateInterval: 100, // ms
    maxAlerts: 10,
    maxFaces: 5,
};

// Global State
let state = {
    peopleCount: 0,
    totalCount: 0,
    densityLevel: 'Low',
    fps: 0,
    anomalies: 0,
    alerts: [],
    detectedFaces: [],
    heatmapData: [],
    scatterPoints: [],
};

// Charts
let scatterChart = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
    setupEventListeners();
    updateClock();
    connectToBackend();
    setInterval(updateClock, 1000);
});

// ============================================================================
// Chart Initialization
// ============================================================================

function initializeCharts() {
    // Scatter Chart
    const scatterCtx = document.getElementById('scatterChart').getContext('2d');
    scatterChart = new Chart(scatterCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Detected Persons',
                data: [],
                backgroundColor: 'rgba(0, 212, 255, 0.6)',
                borderColor: 'rgba(0, 212, 255, 1)',
                borderWidth: 2,
                pointRadius: 6,
                pointHoverRadius: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: { color: '#e0e0ff' }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: { display: true, text: 'X Position', color: '#a0a0c0' },
                    ticks: { color: '#a0a0c0' },
                    grid: { color: 'rgba(42, 48, 80, 0.5)' },
                    min: 0,
                    max: 640,
                },
                y: {
                    title: { display: true, text: 'Y Position', color: '#a0a0c0' },
                    ticks: { color: '#a0a0c0' },
                    grid: { color: 'rgba(42, 48, 80, 0.5)' },
                    min: 0,
                    max: 480,
                }
            }
        }
    });
}

// ============================================================================
// Event Listeners
// ============================================================================

function setupEventListeners() {
    // Input source change
    document.querySelectorAll('input[name="input-source"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'upload') {
                document.getElementById('videoUpload').click();
            }
        });
    });

    // Video upload
    document.getElementById('uploadBtn').addEventListener('click', () => {
        document.getElementById('videoUpload').click();
    });

    document.getElementById('videoUpload').addEventListener('change', (e) => {
        uploadVideo(e.target.files[0]);
    });

    // Threshold change
    document.getElementById('threshold').addEventListener('input', (e) => {
        document.getElementById('thresholdValue').textContent = e.target.value;
        sendToBackend({ action: 'update_threshold', value: parseFloat(e.target.value) });
    });
}

// ============================================================================
// WebSocket Communication
// ============================================================================

let socket = null;

function connectToBackend() {
    // For real-time updates, we'll use polling with fetch
    // In production, use WebSocket for better performance
    pollBackendData();
}

function pollBackendData() {
    fetch('/api/detection-data')
        .then(response => response.json())
        .then(data => {
            updateUIWithData(data);
            setTimeout(pollBackendData, CONFIG.updateInterval);
        })
        .catch(error => {
            console.log('Backend not yet connected');
            updateConnectionStatus(false);
            setTimeout(pollBackendData, 2000);
        });
}

function sendToBackend(data) {
    fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    }).catch(err => console.error('Send failed:', err));
}

function uploadVideo(file) {
    if (!file) return;

    const formData = new FormData();
    formData.append('video', file);

    fetch('/api/upload-video', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Video uploaded:', data);
        addAlert('Video uploaded successfully', 'success');
    })
    .catch(error => {
        console.error('Upload failed:', error);
        addAlert('Video upload failed', 'error');
    });
}

// ============================================================================
// UI Updates
// ============================================================================

function updateUIWithData(data) {
    // Update connection status
    updateConnectionStatus(true);

    // Update counts
    state.peopleCount = data.people_count || 0;
    state.totalCount = data.total_count || 0;
    state.fps = data.fps || 0;
    state.anomalies = data.anomalies || 0;

    document.getElementById('peopleCount').textContent = state.peopleCount;
    document.getElementById('totalCount').textContent = state.totalCount;
    document.getElementById('fpsValue').textContent = state.fps.toFixed(1);
    document.getElementById('anomalies').textContent = state.anomalies;

    // Update density level
    state.densityLevel = state.peopleCount > 50 ? 'High' : (state.peopleCount > 20 ? 'Medium' : 'Low');
    document.getElementById('densityLevel').textContent = state.densityLevel;

    // Update visualizations
    if (data.heatmap_data) {
        drawHeatmap(data.heatmap_data);
    }

    if (data.scatter_points) {
        updateScatterChart(data.scatter_points);
    }

    // Update detected faces
    if (data.detected_faces) {
        updateDetectedFaces(data.detected_faces);
    }

    // Update alerts
    if (data.new_alerts) {
        data.new_alerts.forEach(alert => {
            addAlert(alert.message, alert.type);
        });
    }
}

function updateConnectionStatus(connected) {
    const badge = document.getElementById('connectionStatus');
    if (connected) {
        badge.textContent = '● Connected';
        badge.className = 'status-badge';
    } else {
        badge.textContent = '● Disconnected';
        badge.className = 'status-badge disconnected';
    }
}

function updateClock() {
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    document.getElementById('timestamp').textContent = timeString;
}

// ============================================================================
// Heatmap Visualization
// ============================================================================

function drawHeatmap(data) {
    const canvas = document.getElementById('heatmapCanvas');
    const ctx = canvas.getContext('2d');

    // Clear canvas
    ctx.fillStyle = '#0a0e27';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = 'rgba(42, 48, 80, 0.3)';
    ctx.lineWidth = 1;
    const gridSize = 40;

    for (let i = 0; i <= canvas.width; i += gridSize) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, canvas.height);
        ctx.stroke();
    }

    for (let i = 0; i <= canvas.height; i += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(canvas.width, i);
        ctx.stroke();
    }

    // Draw heatmap points
    if (Array.isArray(data) && data.length > 0) {
        data.forEach(point => {
            const intensity = Math.min(point.intensity || 0.5, 1);
            const radius = 20 * intensity;

            // Gradient from blue (low) to red (high)
            const hue = (1 - intensity) * 240; // 0-240 degrees
            ctx.fillStyle = `hsla(${hue}, 100%, 50%, ${0.6 * intensity})`;

            ctx.beginPath();
            ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    // Draw border
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.5)';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
}

// ============================================================================
// Scatter Chart Update
// ============================================================================

function updateScatterChart(points) {
    if (!scatterChart) return;

    scatterChart.data.datasets[0].data = points.map(p => ({
        x: p.x,
        y: p.y
    }));

    scatterChart.update('none'); // Update without animation
}

// ============================================================================
// Face Detection Display
// ============================================================================

function updateDetectedFaces(faces) {
    const facesList = document.getElementById('facesList');

    if (!faces || faces.length === 0) {
        facesList.innerHTML = '<p class="empty-state">No suspicious activity detected</p>';
        return;
    }

    facesList.innerHTML = faces.slice(0, CONFIG.maxFaces).map(face => `
        <div class="face-item">
            <img src="${face.image_base64}" alt="Face" class="face-thumbnail">
            <div class="face-info">
                <strong>${face.id || 'Unknown'}</strong>
                <div class="face-time">${formatTime(face.timestamp)}</div>
                <div class="face-confidence">Match: ${(face.confidence * 100).toFixed(1)}%</div>
            </div>
        </div>
    `).join('');
}

// ============================================================================
// Alerts Management
// ============================================================================

function addAlert(message, type = 'info') {
    const alertsList = document.getElementById('alertsList');

    // Clear empty state
    if (alertsList.querySelector('.empty-state')) {
        alertsList.innerHTML = '';
    }

    const alertElement = document.createElement('div');
    alertElement.className = `alert-item alert-${type}`;
    alertElement.innerHTML = `
        <div class="alert-text">
            ${type === 'error' ? '❌' : type === 'success' ? '✅' : '⚠️'} ${message}
        </div>
        <div class="alert-time">${new Date().toLocaleTimeString()}</div>
    `;

    alertsList.insertBefore(alertElement, alertsList.firstChild);

    // Keep only recent alerts
    while (alertsList.children.length > CONFIG.maxAlerts) {
        alertsList.removeChild(alertsList.lastChild);
    }

    state.alerts.push({
        message,
        type,
        timestamp: new Date()
    });
}

// ============================================================================
// Utility Functions
// ============================================================================

function formatTime(timestamp) {
    if (!timestamp) return 'just now';

    const date = new Date(timestamp * 1000);
    const now = new Date();
    const diff = now - date;

    if (diff < 60000) return 'just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;

    return date.toLocaleTimeString();
}

// ============================================================================
// Export data for debugging
// ============================================================================

window.getSystemState = () => state;
window.sendCommand = (cmd) => sendToBackend(cmd);
