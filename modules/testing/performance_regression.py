"""
Performance regression testing framework for Homeostasis.

This module provides infrastructure for detecting performance regressions
across all components of the self-healing system, including error detection,
analysis, patch generation, and deployment.
"""
import time
import json
import os
import statistics
import sqlite3
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import threading
import concurrent.futures
from contextlib import contextmanager
import psutil
import traceback
import warnings


@dataclass
class PerformanceMetric:
    """Container for performance metrics."""
    name: str
    duration: float
    memory_delta: float
    cpu_percent: float
    timestamp: datetime
    git_commit: str
    environment: Dict[str, Any]
    metadata: Dict[str, Any] = None


@dataclass
class PerformanceBaseline:
    """Baseline performance metrics for comparison."""
    name: str
    mean_duration: float
    std_duration: float
    p50_duration: float
    p95_duration: float
    p99_duration: float
    mean_memory: float
    mean_cpu: float
    sample_count: int
    last_updated: datetime
    git_commit: str


@dataclass
class RegressionResult:
    """Result of regression detection."""
    test_name: str
    metric_type: str  # duration, memory, cpu
    baseline_value: float
    current_value: float
    regression_factor: float
    is_regression: bool
    confidence: float
    details: Dict[str, Any]


class PerformanceTracker:
    """Track performance metrics during test execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.start_cpu_time = None
        self.cpu_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start(self):
        """Start tracking performance metrics."""
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss
        self.start_cpu_time = self.process.cpu_times()
        self.cpu_samples = []
        self.monitoring = True
        
        # Start CPU monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_cpu)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self) -> Tuple[float, float, float]:
        """Stop tracking and return metrics."""
        self.monitoring = False
        
        # Calculate duration
        duration = time.perf_counter() - self.start_time
        
        # Calculate memory delta
        end_memory = self.process.memory_info().rss
        memory_delta = (end_memory - self.start_memory) / 1024 / 1024  # MB
        
        # Calculate average CPU usage
        avg_cpu = statistics.mean(self.cpu_samples) if self.cpu_samples else 0
        
        return duration, memory_delta, avg_cpu
    
    def _monitor_cpu(self):
        """Monitor CPU usage in background thread."""
        while self.monitoring:
            try:
                cpu_percent = self.process.cpu_percent(interval=0.1)
                self.cpu_samples.append(cpu_percent)
            except:
                pass
            time.sleep(0.1)


class PerformanceRegressionDetector:
    """Detect performance regressions by comparing against baselines."""
    
    def __init__(self, db_path: str = "performance_baselines.db"):
        self.db_path = db_path
        self._init_database()
        
        # Regression thresholds
        self.thresholds = {
            "duration": {
                "warning": 1.2,    # 20% slower
                "critical": 1.5,   # 50% slower
                "confidence": 0.95
            },
            "memory": {
                "warning": 1.3,    # 30% more memory
                "critical": 2.0,   # 2x memory
                "confidence": 0.90
            },
            "cpu": {
                "warning": 1.25,   # 25% more CPU
                "critical": 1.75,  # 75% more CPU
                "confidence": 0.90
            }
        }
    
    def _init_database(self):
        """Initialize SQLite database for storing baselines."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS baselines (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                mean_duration REAL,
                std_duration REAL,
                p50_duration REAL,
                p95_duration REAL,
                p99_duration REAL,
                mean_memory REAL,
                mean_cpu REAL,
                sample_count INTEGER,
                last_updated TIMESTAMP,
                git_commit TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY,
                name TEXT,
                duration REAL,
                memory_delta REAL,
                cpu_percent REAL,
                timestamp TIMESTAMP,
                git_commit TEXT,
                environment TEXT,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO metrics (name, duration, memory_delta, cpu_percent, 
                               timestamp, git_commit, environment, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metric.name,
            metric.duration,
            metric.memory_delta,
            metric.cpu_percent,
            metric.timestamp,
            metric.git_commit,
            json.dumps(metric.environment),
            json.dumps(metric.metadata) if metric.metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def update_baseline(self, name: str, metrics: List[PerformanceMetric]):
        """Update baseline from recent metrics."""
        if not metrics:
            return
        
        # Calculate statistics
        durations = [m.duration for m in metrics]
        memories = [m.memory_delta for m in metrics]
        cpus = [m.cpu_percent for m in metrics]
        
        baseline = PerformanceBaseline(
            name=name,
            mean_duration=statistics.mean(durations),
            std_duration=statistics.stdev(durations) if len(durations) > 1 else 0,
            p50_duration=statistics.median(durations),
            p95_duration=statistics.quantiles(durations, n=20)[18] if len(durations) > 1 else durations[0],
            p99_duration=statistics.quantiles(durations, n=100)[98] if len(durations) > 1 else durations[0],
            mean_memory=statistics.mean(memories),
            mean_cpu=statistics.mean(cpus),
            sample_count=len(metrics),
            last_updated=datetime.now(),
            git_commit=metrics[-1].git_commit
        )
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO baselines 
            (name, mean_duration, std_duration, p50_duration, p95_duration, p99_duration,
             mean_memory, mean_cpu, sample_count, last_updated, git_commit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            baseline.name,
            baseline.mean_duration,
            baseline.std_duration,
            baseline.p50_duration,
            baseline.p95_duration,
            baseline.p99_duration,
            baseline.mean_memory,
            baseline.mean_cpu,
            baseline.sample_count,
            baseline.last_updated,
            baseline.git_commit
        ))
        
        conn.commit()
        conn.close()
    
    def get_baseline(self, name: str) -> Optional[PerformanceBaseline]:
        """Get baseline for a test."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM baselines WHERE name = ?
        """, (name,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return PerformanceBaseline(
            name=row[1],
            mean_duration=row[2],
            std_duration=row[3],
            p50_duration=row[4],
            p95_duration=row[5],
            p99_duration=row[6],
            mean_memory=row[7],
            mean_cpu=row[8],
            sample_count=row[9],
            last_updated=datetime.fromisoformat(row[10]),
            git_commit=row[11]
        )
    
    def detect_regression(self, current_metric: PerformanceMetric) -> List[RegressionResult]:
        """Detect if current metric represents a regression."""
        baseline = self.get_baseline(current_metric.name)
        
        if not baseline:
            return []
        
        results = []
        
        # Check duration regression
        if baseline.mean_duration > 0:
            duration_factor = current_metric.duration / baseline.mean_duration
            z_score = abs(current_metric.duration - baseline.mean_duration) / (baseline.std_duration + 0.001)
            confidence = min(1.0, z_score / 3.0)  # Normalize to 0-1
            
            is_regression = duration_factor > self.thresholds["duration"]["warning"]
            
            results.append(RegressionResult(
                test_name=current_metric.name,
                metric_type="duration",
                baseline_value=baseline.mean_duration,
                current_value=current_metric.duration,
                regression_factor=duration_factor,
                is_regression=is_regression,
                confidence=confidence,
                details={
                    "baseline_p95": baseline.p95_duration,
                    "baseline_p99": baseline.p99_duration,
                    "severity": "critical" if duration_factor > self.thresholds["duration"]["critical"] else "warning"
                }
            ))
        
        # Check memory regression
        if abs(baseline.mean_memory) > 0.1:  # Only check if baseline uses significant memory
            memory_factor = abs(current_metric.memory_delta / baseline.mean_memory)
            
            is_regression = memory_factor > self.thresholds["memory"]["warning"]
            
            results.append(RegressionResult(
                test_name=current_metric.name,
                metric_type="memory",
                baseline_value=baseline.mean_memory,
                current_value=current_metric.memory_delta,
                regression_factor=memory_factor,
                is_regression=is_regression,
                confidence=0.8,  # Memory measurements are less reliable
                details={
                    "severity": "critical" if memory_factor > self.thresholds["memory"]["critical"] else "warning"
                }
            ))
        
        # Check CPU regression
        if baseline.mean_cpu > 0:
            cpu_factor = current_metric.cpu_percent / baseline.mean_cpu
            
            is_regression = cpu_factor > self.thresholds["cpu"]["warning"]
            
            results.append(RegressionResult(
                test_name=current_metric.name,
                metric_type="cpu",
                baseline_value=baseline.mean_cpu,
                current_value=current_metric.cpu_percent,
                regression_factor=cpu_factor,
                is_regression=is_regression,
                confidence=0.7,  # CPU measurements can be noisy
                details={
                    "severity": "critical" if cpu_factor > self.thresholds["cpu"]["critical"] else "warning"
                }
            ))
        
        return results


class PerformanceRegressionTester:
    """Main class for running performance regression tests."""
    
    def __init__(self, detector: PerformanceRegressionDetector = None):
        self.detector = detector or PerformanceRegressionDetector()
        self.tracker = PerformanceTracker()
        self.current_git_commit = self._get_git_commit()
        self.environment = self._get_environment()
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()[:8]
        except:
            return "unknown"
    
    def _get_environment(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            "platform": os.uname().sysname,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "cpu_count": psutil.cpu_count(),
            "total_memory": psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
        }
    
    @contextmanager
    def measure(self, test_name: str, metadata: Dict[str, Any] = None):
        """Context manager for measuring performance."""
        self.tracker.start()
        
        try:
            yield
        finally:
            duration, memory_delta, cpu_percent = self.tracker.stop()
            
            # Record metric
            metric = PerformanceMetric(
                name=test_name,
                duration=duration,
                memory_delta=memory_delta,
                cpu_percent=cpu_percent,
                timestamp=datetime.now(),
                git_commit=self.current_git_commit,
                environment=self.environment,
                metadata=metadata
            )
            
            self.detector.record_metric(metric)
            
            # Detect regressions
            regressions = self.detector.detect_regression(metric)
            
            # Report regressions
            for regression in regressions:
                if regression.is_regression:
                    self._report_regression(regression)
    
    def _report_regression(self, regression: RegressionResult):
        """Report a detected regression."""
        severity = regression.details.get("severity", "warning")
        
        msg = (
            f"Performance regression detected in '{regression.test_name}' "
            f"({regression.metric_type}): {regression.regression_factor:.2f}x "
            f"baseline ({regression.baseline_value:.3f} -> {regression.current_value:.3f})"
        )
        
        if severity == "critical":
            raise AssertionError(msg)
        else:
            warnings.warn(msg, PerformanceWarning)
    
    def benchmark(self, func: Callable, test_name: str, iterations: int = 10,
                  warmup: int = 2, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a benchmark with multiple iterations."""
        # Warmup runs
        for _ in range(warmup):
            func()
        
        # Collect metrics
        metrics = []
        for i in range(iterations):
            self.tracker.start()
            
            try:
                result = func()
            except Exception as e:
                raise RuntimeError(f"Benchmark '{test_name}' failed: {e}")
            
            duration, memory_delta, cpu_percent = self.tracker.stop()
            
            metric = PerformanceMetric(
                name=test_name,
                duration=duration,
                memory_delta=memory_delta,
                cpu_percent=cpu_percent,
                timestamp=datetime.now(),
                git_commit=self.current_git_commit,
                environment=self.environment,
                metadata={**(metadata or {}), "iteration": i}
            )
            
            metrics.append(metric)
            self.detector.record_metric(metric)
        
        # Update baseline if this is a baseline run
        if os.environ.get("UPDATE_PERFORMANCE_BASELINE") == "true":
            self.detector.update_baseline(test_name, metrics)
        
        # Check for regressions
        avg_metric = PerformanceMetric(
            name=test_name,
            duration=statistics.mean(m.duration for m in metrics),
            memory_delta=statistics.mean(m.memory_delta for m in metrics),
            cpu_percent=statistics.mean(m.cpu_percent for m in metrics),
            timestamp=datetime.now(),
            git_commit=self.current_git_commit,
            environment=self.environment,
            metadata=metadata
        )
        
        regressions = self.detector.detect_regression(avg_metric)
        
        # Prepare results
        results = {
            "name": test_name,
            "iterations": iterations,
            "duration": {
                "mean": statistics.mean(m.duration for m in metrics),
                "std": statistics.stdev(m.duration for m in metrics) if len(metrics) > 1 else 0,
                "min": min(m.duration for m in metrics),
                "max": max(m.duration for m in metrics),
            },
            "memory": {
                "mean": statistics.mean(m.memory_delta for m in metrics),
                "std": statistics.stdev(m.memory_delta for m in metrics) if len(metrics) > 1 else 0,
            },
            "cpu": {
                "mean": statistics.mean(m.cpu_percent for m in metrics),
                "std": statistics.stdev(m.cpu_percent for m in metrics) if len(metrics) > 1 else 0,
            },
            "regressions": [asdict(r) for r in regressions if r.is_regression]
        }
        
        # Report critical regressions
        for regression in regressions:
            if regression.is_regression and regression.details.get("severity") == "critical":
                self._report_regression(regression)
        
        return results


class PerformanceWarning(UserWarning):
    """Warning for performance regressions."""
    pass


# Decorator for easy performance testing
def performance_test(name: str = None, iterations: int = 10, warmup: int = 2):
    """Decorator for marking performance tests."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            test_name = name or func.__name__
            tester = PerformanceRegressionTester()
            
            def test_func():
                return func(*args, **kwargs)
            
            return tester.benchmark(test_func, test_name, iterations, warmup)
        
        return wrapper
    return decorator