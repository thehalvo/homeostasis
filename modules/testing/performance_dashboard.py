"""
Performance dashboard for visualizing regression test results.

This module provides a web-based dashboard for monitoring performance
trends and identifying regressions across the Homeostasis framework.
"""
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any
import statistics
from flask import Flask, render_template_string, jsonify, request


class PerformanceDashboard:
    """Web dashboard for performance monitoring."""
    
    def __init__(self, db_path: str = "performance_baselines.db"):
        self.db_path = db_path
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up Flask routes."""
        @self.app.route('/')
        def index():
            return render_template_string(DASHBOARD_TEMPLATE)
        
        @self.app.route('/api/metrics/<test_name>')
        def get_metrics(test_name: str):
            """Get metrics for a specific test."""
            metrics = self._load_test_metrics(test_name)
            return jsonify(metrics)
        
        @self.app.route('/api/tests')
        def get_tests():
            """Get list of all tests."""
            tests = self._get_test_list()
            return jsonify(tests)
        
        @self.app.route('/api/summary')
        def get_summary():
            """Get performance summary."""
            summary = self._generate_summary()
            return jsonify(summary)
        
        @self.app.route('/api/regressions')
        def get_regressions():
            """Get recent regressions."""
            days = int(request.args.get('days', 7))
            regressions = self._detect_recent_regressions(days)
            return jsonify(regressions)
        
        @self.app.route('/api/trends/<test_name>')
        def get_trends(test_name: str):
            """Get performance trends for a test."""
            days = int(request.args.get('days', 30))
            trends = self._calculate_trends(test_name, days)
            return jsonify(trends)
    
    def _load_test_metrics(self, test_name: str, days: int = 30) -> List[Dict]:
        """Load metrics for a specific test."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            SELECT duration, memory_delta, cpu_percent, timestamp, git_commit
            FROM metrics
            WHERE name = ? AND timestamp > ?
            ORDER BY timestamp
        """, (test_name, since.isoformat()))
        
        metrics = []
        for row in cursor.fetchall():
            metrics.append({
                "duration": row[0],
                "memory": row[1],
                "cpu": row[2],
                "timestamp": row[3],
                "commit": row[4]
            })
        
        conn.close()
        return metrics
    
    def _get_test_list(self) -> List[Dict]:
        """Get list of all tests with their status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT m.name, b.mean_duration, b.last_updated
            FROM metrics m
            LEFT JOIN baselines b ON m.name = b.name
            ORDER BY m.name
        """)
        
        tests = []
        for row in cursor.fetchall():
            tests.append({
                "name": row[0],
                "has_baseline": row[1] is not None,
                "baseline_duration": row[1],
                "last_updated": row[2]
            })
        
        conn.close()
        return tests
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate overall performance summary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total test count
        cursor.execute("SELECT COUNT(DISTINCT name) FROM metrics")
        total_tests = cursor.fetchone()[0]
        
        # Get recent metrics count
        since = datetime.now() - timedelta(days=1)
        cursor.execute("""
            SELECT COUNT(*) FROM metrics WHERE timestamp > ?
        """, (since.isoformat(),))
        recent_runs = cursor.fetchone()[0]
        
        # Get baseline coverage
        cursor.execute("SELECT COUNT(*) FROM baselines")
        baseline_count = cursor.fetchone()[0]
        
        # Calculate average performance across all tests
        cursor.execute("""
            SELECT AVG(duration), AVG(memory_delta), AVG(cpu_percent)
            FROM metrics
            WHERE timestamp > ?
        """, (since.isoformat(),))
        
        row = cursor.fetchone()
        avg_performance = {
            "duration": row[0] or 0,
            "memory": row[1] or 0,
            "cpu": row[2] or 0
        }
        
        conn.close()
        
        return {
            "total_tests": total_tests,
            "recent_runs": recent_runs,
            "baseline_coverage": baseline_count / total_tests * 100 if total_tests > 0 else 0,
            "average_performance": avg_performance,
            "last_update": datetime.now().isoformat()
        }
    
    def _detect_recent_regressions(self, days: int = 7) -> List[Dict]:
        """Detect regressions in the past N days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since = datetime.now() - timedelta(days=days)
        
        # Get recent metrics with baselines
        cursor.execute("""
            SELECT m.name, m.duration, m.timestamp, b.mean_duration, b.std_duration
            FROM metrics m
            JOIN baselines b ON m.name = b.name
            WHERE m.timestamp > ?
            ORDER BY m.timestamp DESC
        """, (since.isoformat(),))
        
        regressions = []
        for row in cursor.fetchall():
            name, duration, timestamp, baseline_mean, baseline_std = row
            
            # Check if duration is significantly worse than baseline
            z_score = (duration - baseline_mean) / (baseline_std + 0.001)
            
            if z_score > 2:  # More than 2 standard deviations
                regression_factor = duration / baseline_mean
                
                if regression_factor > 1.2:  # At least 20% slower
                    regressions.append({
                        "test": name,
                        "timestamp": timestamp,
                        "duration": duration,
                        "baseline": baseline_mean,
                        "factor": regression_factor,
                        "severity": "critical" if regression_factor > 1.5 else "warning"
                    })
        
        conn.close()
        return regressions
    
    def _calculate_trends(self, test_name: str, days: int = 30) -> Dict[str, Any]:
        """Calculate performance trends for a test."""
        metrics = self._load_test_metrics(test_name, days)
        
        if not metrics:
            return {"error": "No metrics found"}
        
        # Group by day
        daily_stats = {}
        for metric in metrics:
            date = metric["timestamp"][:10]  # YYYY-MM-DD
            if date not in daily_stats:
                daily_stats[date] = {
                    "durations": [],
                    "memories": [],
                    "cpus": []
                }
            
            daily_stats[date]["durations"].append(metric["duration"])
            daily_stats[date]["memories"].append(metric["memory"])
            daily_stats[date]["cpus"].append(metric["cpu"])
        
        # Calculate daily averages
        trend_data = []
        for date, stats in sorted(daily_stats.items()):
            trend_data.append({
                "date": date,
                "duration": statistics.mean(stats["durations"]),
                "memory": statistics.mean(stats["memories"]),
                "cpu": statistics.mean(stats["cpus"]),
                "samples": len(stats["durations"])
            })
        
        # Calculate trend line (simple linear regression)
        if len(trend_data) > 1:
            x_values = list(range(len(trend_data)))
            y_values = [d["duration"] for d in trend_data]
            
            n = len(x_values)
            x_mean = sum(x_values) / n
            y_mean = sum(y_values) / n
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)
            
            if denominator != 0:
                slope = numerator / denominator
                trend = "improving" if slope < -0.001 else "degrading" if slope > 0.001 else "stable"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "test_name": test_name,
            "trend": trend,
            "data": trend_data,
            "statistics": {
                "min_duration": min(m["duration"] for m in metrics),
                "max_duration": max(m["duration"] for m in metrics),
                "avg_duration": statistics.mean(m["duration"] for m in metrics),
                "std_duration": statistics.stdev(m["duration"] for m in metrics) if len(metrics) > 1 else 0
            }
        }
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the dashboard server."""
        self.app.run(host=host, port=port, debug=debug)


# HTML template for the dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Homeostasis Performance Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
        }
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
        }
        .card .value {
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }
        .card .unit {
            font-size: 14px;
            color: #999;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .chart-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .regressions {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .regression-item {
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #ff6b6b;
            background: #fff5f5;
            border-radius: 4px;
        }
        .regression-item.warning {
            border-left-color: #ffd93d;
            background: #fffbe5;
        }
        select {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Homeostasis Performance Dashboard</h1>
        
        <div class="summary-cards" id="summary-cards">
            <!-- Summary cards will be populated here -->
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <h3>Test Performance Trends</h3>
                <select id="test-selector" onchange="loadTestData()">
                    <option value="">Select a test...</option>
                </select>
                <canvas id="performance-chart"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>Overall Performance Distribution</h3>
                <canvas id="distribution-chart"></canvas>
            </div>
        </div>
        
        <div class="regressions">
            <h3>Recent Performance Regressions</h3>
            <div id="regressions-list">
                <!-- Regressions will be populated here -->
            </div>
        </div>
    </div>
    
    <script>
        let performanceChart = null;
        let distributionChart = null;
        
        // Load summary data
        async function loadSummary() {
            const response = await fetch('/api/summary');
            const data = await response.json();
            
            const cardsHtml = `
                <div class="card">
                    <h3>Total Tests</h3>
                    <div class="value">${data.total_tests}</div>
                </div>
                <div class="card">
                    <h3>Recent Runs (24h)</h3>
                    <div class="value">${data.recent_runs}</div>
                </div>
                <div class="card">
                    <h3>Baseline Coverage</h3>
                    <div class="value">${data.baseline_coverage.toFixed(1)}<span class="unit">%</span></div>
                </div>
                <div class="card">
                    <h3>Avg Duration</h3>
                    <div class="value">${(data.average_performance.duration * 1000).toFixed(1)}<span class="unit">ms</span></div>
                </div>
            `;
            
            document.getElementById('summary-cards').innerHTML = cardsHtml;
        }
        
        // Load test list
        async function loadTests() {
            const response = await fetch('/api/tests');
            const tests = await response.json();
            
            const selector = document.getElementById('test-selector');
            tests.forEach(test => {
                const option = document.createElement('option');
                option.value = test.name;
                option.textContent = test.name;
                selector.appendChild(option);
            });
        }
        
        // Load test data
        async function loadTestData() {
            const testName = document.getElementById('test-selector').value;
            if (!testName) return;
            
            const response = await fetch(`/api/trends/${testName}?days=30`);
            const data = await response.json();
            
            // Update performance chart
            if (performanceChart) {
                performanceChart.destroy();
            }
            
            const ctx = document.getElementById('performance-chart').getContext('2d');
            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.data.map(d => d.date),
                    datasets: [{
                        label: 'Duration (ms)',
                        data: data.data.map(d => d.duration * 1000),
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }, {
                        label: 'Memory (MB)',
                        data: data.data.map(d => d.memory),
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1,
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            grid: {
                                drawOnChartArea: false,
                            },
                        },
                    }
                }
            });
        }
        
        // Load regressions
        async function loadRegressions() {
            const response = await fetch('/api/regressions?days=7');
            const regressions = await response.json();
            
            const listElement = document.getElementById('regressions-list');
            
            if (regressions.length === 0) {
                listElement.innerHTML = '<p style="color: #28a745;">✅ No performance regressions detected in the last 7 days!</p>';
                return;
            }
            
            const regressionsHtml = regressions.map(r => `
                <div class="regression-item ${r.severity}">
                    <strong>${r.test}</strong><br>
                    Duration: ${(r.duration * 1000).toFixed(1)}ms 
                    (baseline: ${(r.baseline * 1000).toFixed(1)}ms)<br>
                    Regression factor: ${r.factor.toFixed(2)}x slower<br>
                    <small>${new Date(r.timestamp).toLocaleString()}</small>
                </div>
            `).join('');
            
            listElement.innerHTML = regressionsHtml;
        }
        
        // Initialize dashboard
        async function init() {
            await loadSummary();
            await loadTests();
            await loadRegressions();
            
            // Refresh data every 30 seconds
            setInterval(() => {
                loadSummary();
                loadRegressions();
            }, 30000);
        }
        
        init();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    # Run the dashboard
    dashboard = PerformanceDashboard()
    dashboard.run(debug=True)