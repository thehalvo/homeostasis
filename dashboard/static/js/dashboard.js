/**
 * Homeostasis Dashboard Main JavaScript
 * 
 * Handles real-time updates, UI interactions, and data visualization.
 */

// Initialize SocketIO connection
let socket;

// Dashboard charts
let errorRateChart;
let responseTimeChart;
let fixSuccessChart;
let memoryUsageChart;
let canaryMetricsChart;

// Store dashboard state
const dashboardState = {
    errors: [],
    fixes: [],
    approvals: [],
    metrics: {},
    canary: null
};

/**
 * Initialize the dashboard when the DOM is loaded
 */
document.addEventListener('DOMContentLoaded', function() {
    // Connect to SocketIO server
    connectWebSocket();
    
    // Load initial data
    loadDashboardData();
    
    // Set up refresh interval
    setInterval(refreshDashboardData, 10000); // 10 seconds
    
    // Set up event listeners
    setupEventListeners();
});

/**
 * Connect to WebSocket server for real-time updates
 */
function connectWebSocket() {
    socket = io();
    
    // Connection events
    socket.on('connect', function() {
        console.log('Connected to WebSocket server');
        
        // Subscribe to updates
        socket.emit('subscribe', {
            topics: ['errors', 'fixes', 'metrics']
        });
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from WebSocket server');
    });
    
    // Data update events
    socket.on('errors_update', function(data) {
        updateErrors(data.errors);
    });
    
    socket.on('fixes_update', function(data) {
        updateFixes(data.fixes);
    });
    
    socket.on('metrics_update', function(data) {
        updateMetrics(data.metrics);
    });
    
    socket.on('canary_update', function(data) {
        updateCanary(data.canary);
    });
}

/**
 * Load initial dashboard data from API
 */
function loadDashboardData() {
    // Fetch errors
    fetch('/api/errors')
        .then(response => response.json())
        .then(data => updateErrors(data.errors))
        .catch(error => console.error('Error fetching errors:', error));
    
    // Fetch fixes
    fetch('/api/fixes')
        .then(response => response.json())
        .then(data => updateFixes(data.fixes))
        .catch(error => console.error('Error fetching fixes:', error));
    
    // Fetch approvals
    fetch('/api/approvals')
        .then(response => response.json())
        .then(data => updateApprovals(data.approvals))
        .catch(error => console.error('Error fetching approvals:', error));
    
    // Fetch metrics
    fetch('/api/metrics')
        .then(response => response.json())
        .then(data => updateMetrics(data.metrics))
        .catch(error => console.error('Error fetching metrics:', error));
    
    // Fetch canary status
    fetch('/api/canary')
        .then(response => response.json())
        .then(data => updateCanary(data.canary))
        .catch(error => console.error('Error fetching canary status:', error));
}

/**
 * Refresh dashboard data
 */
function refreshDashboardData() {
    // Only refresh if we're not using WebSockets or WebSocket is disconnected
    if (!socket || !socket.connected) {
        loadDashboardData();
    }
}

/**
 * Update errors data and UI
 */
function updateErrors(errors) {
    if (!errors || !Array.isArray(errors)) {
        return;
    }
    
    // Update state
    dashboardState.errors = errors;
    
    // Update UI
    updateErrorsTable(errors);
    updateErrorsCount(errors);
}

/**
 * Update errors table
 */
function updateErrorsTable(errors) {
    const tableBody = document.getElementById('errors-table-body');
    if (!tableBody) {
        return;
    }
    
    // Clear existing rows
    tableBody.innerHTML = '';
    
    // Check if we have errors
    if (errors.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="7" class="text-center">No errors found</td>';
        tableBody.appendChild(row);
        return;
    }
    
    // Add rows for each error
    errors.forEach(error => {
        const row = document.createElement('tr');
        
        // Determine status badge
        let statusBadge = '';
        switch (error.status) {
            case 'fixed':
                statusBadge = '<span class="badge bg-success">Fixed</span>';
                break;
            case 'analyzing':
                statusBadge = '<span class="badge bg-warning">Analyzing</span>';
                break;
            case 'pending':
                statusBadge = '<span class="badge bg-info">Pending</span>';
                break;
            default:
                statusBadge = `<span class="badge bg-secondary">${error.status}</span>`;
        }
        
        // Format timestamp
        let timestamp = error.timestamp;
        if (timestamp) {
            const date = new Date(timestamp);
            timestamp = date.toLocaleString();
        }
        
        // Set row HTML
        row.innerHTML = `
            <td>${error.id}</td>
            <td>${timestamp}</td>
            <td>${error.service}</td>
            <td>${error.error_type}</td>
            <td>${error.message}</td>
            <td>${statusBadge}</td>
            <td>
                <button class="btn btn-sm btn-primary" onclick="viewError('${error.id}')">View</button>
            </td>
        `;
        
        tableBody.appendChild(row);
    });
}

/**
 * Update errors count in dashboard header
 */
function updateErrorsCount(errors) {
    const activeErrorsCount = document.getElementById('active-errors-count');
    if (!activeErrorsCount) {
        return;
    }
    
    // Count non-fixed errors
    const activeCount = errors.filter(error => error.status !== 'fixed').length;
    activeErrorsCount.textContent = activeCount;
    
    // Update system status
    const systemStatus = document.getElementById('system-status');
    if (systemStatus) {
        if (activeCount === 0) {
            systemStatus.textContent = 'Healthy';
            systemStatus.classList.remove('text-danger');
            systemStatus.classList.add('text-success');
        } else {
            systemStatus.textContent = 'Issues Detected';
            systemStatus.classList.remove('text-success');
            systemStatus.classList.add('text-danger');
        }
    }
}

/**
 * Update fixes data and UI
 */
function updateFixes(fixes) {
    if (!fixes || !Array.isArray(fixes)) {
        return;
    }
    
    // Update state
    dashboardState.fixes = fixes;
    
    // Update UI
    updateFixesTable(fixes);
    updateFixesCount(fixes);
}

/**
 * Update fixes table
 */
function updateFixesTable(fixes) {
    const tableBody = document.getElementById('fixes-table-body');
    if (!tableBody) {
        return;
    }
    
    // Clear existing rows
    tableBody.innerHTML = '';
    
    // Check if we have fixes
    if (fixes.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="7" class="text-center">No fixes found</td>';
        tableBody.appendChild(row);
        return;
    }
    
    // Add rows for each fix
    fixes.forEach(fix => {
        const row = document.createElement('tr');
        
        // Determine status badge
        let statusBadge = '';
        switch (fix.status) {
            case 'deployed':
                statusBadge = '<span class="badge bg-success">Deployed</span>';
                break;
            case 'pending':
                statusBadge = '<span class="badge bg-warning">Pending</span>';
                break;
            case 'testing':
                statusBadge = '<span class="badge bg-info">Testing</span>';
                break;
            case 'rejected':
                statusBadge = '<span class="badge bg-danger">Rejected</span>';
                break;
            default:
                statusBadge = `<span class="badge bg-secondary">${fix.status}</span>`;
        }
        
        // Format timestamp
        let timestamp = fix.timestamp;
        if (timestamp) {
            const date = new Date(timestamp);
            timestamp = date.toLocaleString();
        }
        
        // Format confidence
        let confidence = fix.confidence;
        if (typeof confidence === 'number') {
            confidence = `${Math.round(confidence * 100)}%`;
        }
        
        // Set row HTML
        row.innerHTML = `
            <td>${fix.id}</td>
            <td>${timestamp}</td>
            <td>${fix.service}</td>
            <td>${fix.error_id}</td>
            <td>${statusBadge}</td>
            <td>${confidence}</td>
            <td>
                <button class="btn btn-sm btn-primary" onclick="viewFix('${fix.id}')">View</button>
            </td>
        `;
        
        tableBody.appendChild(row);
    });
}

/**
 * Update fixes count in dashboard header
 */
function updateFixesCount(fixes) {
    const activeFixesCount = document.getElementById('active-fixes-count');
    if (!activeFixesCount) {
        return;
    }
    
    // Count active fixes
    const activeCount = fixes.filter(fix => fix.status !== 'deployed' && fix.status !== 'rejected').length;
    activeFixesCount.textContent = activeCount;
}

/**
 * Update approvals data and UI
 */
function updateApprovals(approvals) {
    if (!approvals || !Array.isArray(approvals)) {
        return;
    }
    
    // Update state
    dashboardState.approvals = approvals;
    
    // Update UI
    updateApprovalsTable(approvals);
}

/**
 * Update approvals table
 */
function updateApprovalsTable(approvals) {
    const tableBody = document.getElementById('approvals-table-body');
    if (!tableBody) {
        return;
    }
    
    // Clear existing rows
    tableBody.innerHTML = '';
    
    // Check if we have pending approvals
    const pendingApprovals = approvals.filter(approval => approval.status === 'pending');
    if (pendingApprovals.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="6" class="text-center">No pending approvals</td>';
        tableBody.appendChild(row);
        return;
    }
    
    // Add rows for each pending approval
    pendingApprovals.forEach(approval => {
        const row = document.createElement('tr');
        
        // Format timestamp
        let timestamp = approval.timestamp;
        if (timestamp) {
            const date = new Date(timestamp);
            timestamp = date.toLocaleString();
        }
        
        // Set row HTML
        row.innerHTML = `
            <td>${approval.id}</td>
            <td>${timestamp}</td>
            <td>${approval.fix_id}</td>
            <td>${approval.error_type || 'Unknown'}</td>
            <td>${approval.confidence || 'N/A'}</td>
            <td>
                <button class="btn btn-sm btn-success" onclick="approveFix('${approval.id}')">Approve</button>
                <button class="btn btn-sm btn-danger" onclick="rejectFix('${approval.id}')">Reject</button>
            </td>
        `;
        
        tableBody.appendChild(row);
    });
}

/**
 * Update metrics data and charts
 */
function updateMetrics(metrics) {
    if (!metrics) {
        return;
    }
    
    // Update state
    dashboardState.metrics = metrics;
    
    // Update success rate in dashboard header
    const successRateElement = document.getElementById('success-rate');
    if (successRateElement && metrics.fix_success_rate) {
        const successRate = metrics.fix_success_rate.current * 100;
        successRateElement.textContent = `${Math.round(successRate)}%`;
        
        // Update progress bar
        const progressBar = document.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.style.width = `${Math.round(successRate)}%`;
        }
    }
    
    // Update charts if they exist
    updateMetricsCharts(metrics);
}

/**
 * Update metrics charts
 */
function updateMetricsCharts(metrics) {
    // Update Error Rate Chart
    if (window.errorRateChart && metrics.error_rate) {
        const data = metrics.error_rate.history.concat(metrics.error_rate.current);
        errorRateChart.data.datasets[0].data = data.map(value => value * 100); // Convert to percentage
        errorRateChart.update();
    }
    
    // Update Response Time Chart
    if (window.responseTimeChart && metrics.response_time) {
        const data = metrics.response_time.history.concat(metrics.response_time.current);
        responseTimeChart.data.datasets[0].data = data;
        responseTimeChart.update();
    }
    
    // Update Fix Success Rate Chart
    if (window.fixSuccessChart && metrics.fix_success_rate) {
        const data = metrics.fix_success_rate.history.concat(metrics.fix_success_rate.current);
        fixSuccessChart.data.datasets[0].data = data.map(value => value * 100); // Convert to percentage
        fixSuccessChart.update();
    }
}

/**
 * Update canary deployment information
 */
function updateCanary(canary) {
    if (!canary) {
        return;
    }
    
    // Update state
    dashboardState.canary = canary;
    
    // TODO: Update canary UI
}

/**
 * Set up event listeners for interactive elements
 */
function setupEventListeners() {
    // TODO: Add event listeners for buttons and forms
}

/**
 * View error details
 */
function viewError(errorId) {
    // TODO: Show error details modal
    console.log(`View error: ${errorId}`);
}

/**
 * View fix details
 */
function viewFix(fixId) {
    // TODO: Show fix details modal
    console.log(`View fix: ${fixId}`);
}

/**
 * Approve a fix
 */
function approveFix(approvalId) {
    // TODO: Send approval request to API
    console.log(`Approve fix: ${approvalId}`);
}

/**
 * Reject a fix
 */
function rejectFix(approvalId) {
    // TODO: Send rejection request to API
    console.log(`Reject fix: ${approvalId}`);
}