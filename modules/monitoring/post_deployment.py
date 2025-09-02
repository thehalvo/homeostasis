"""
Post-deployment monitoring for tracking fix effectiveness.

This module provides utilities for:
1. Monitoring service health after deployment
2. Tracking error patterns after fixes
3. Collecting metrics to evaluate fix success
"""
import os
import sys
import time
import json
import threading
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.monitoring.logger import MonitoringLogger


class PostDeploymentMonitor:
    """
    Monitors service health after deployment.
    """
    
    def __init__(self, 
                config: Dict[str, Any] = None, 
                log_level: str = "INFO"):
        """
        Initialize the post-deployment monitor.
        
        Args:
            config: Configuration dictionary
            log_level: Logging level
        """
        self.logger = MonitoringLogger("post_deployment_monitor", log_level=log_level)
        
        # Default configuration
        self.config = {
            "check_interval": 60,  # seconds
            "duration": 3600,  # 1 hour monitoring period
            "metrics": ["response_time", "error_rate", "memory_usage"],
            "alert_thresholds": {
                "error_rate": 0.05,  # 5% error rate
                "response_time": 500,  # 500ms
                "memory_usage": 512  # 512MB
            }
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # Data storage for metrics
        self.metrics_data = {}
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # Alert handlers
        self.alert_handlers = []
        
        self.logger.info("Initialized post-deployment monitor")
    
    def register_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Register a function to handle alerts.
        
        Args:
            handler: Function that takes alert message and data
        """
        self.alert_handlers.append(handler)
    
    def _send_alert(self, message: str, data: Dict[str, Any] = None) -> None:
        """
        Send an alert to all registered handlers.
        
        Args:
            message: Alert message
            data: Alert data
        """
        self.logger.warning(f"ALERT: {message}", data=data)
        
        for handler in self.alert_handlers:
            try:
                handler(message, data or {})
            except Exception as e:
                self.logger.exception(e, message=f"Error in alert handler: {str(e)}")
    
    def check_health(self, health_url: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Check service health by requesting a health endpoint.
        
        Args:
            health_url: URL of the health endpoint
            timeout: Request timeout in seconds
            
        Returns:
            Health status with metrics
        """
        try:
            start_time = time.time()
            response = requests.get(health_url, timeout=timeout)
            elapsed = time.time() - start_time
            
            health_data = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_code": response.status_code,
                "response_time": elapsed * 1000,  # Convert to ms
                "timestamp": time.time()
            }
            
            # Try to parse response body if it's JSON
            try:
                response_json = response.json()
                health_data.update(response_json)
            except Exception:
                pass
                
            return health_data
            
        except requests.RequestException as e:
            self.logger.warning(f"Health check failed: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def collect_metrics(self, service_url: str) -> Dict[str, Any]:
        """
        Collect metrics from a service.
        
        Args:
            service_url: Base URL of the service
            
        Returns:
            Collected metrics
        """
        metrics = {
            "timestamp": time.time()
        }
        
        # Response time (health check)
        health_data = self.check_health(f"{service_url}/health")
        if "response_time" in health_data:
            metrics["response_time"] = health_data["response_time"]
        
        # Error rate (sample requests to endpoints)
        error_count = 0
        request_count = 0
        
        # Try common endpoints
        for endpoint in ["/", "/api", "/status"]:
            try:
                response = requests.get(f"{service_url}{endpoint}", timeout=5)
                request_count += 1
                if response.status_code >= 400:
                    error_count += 1
            except requests.RequestException:
                request_count += 1
                error_count += 1
        
        if request_count > 0:
            metrics["error_rate"] = error_count / request_count
        
        # Memory usage (if exposed via metrics endpoint)
        try:
            metrics_response = requests.get(f"{service_url}/metrics", timeout=5)
            if metrics_response.status_code == 200:
                try:
                    metrics_data = metrics_response.json()
                    if "memory_usage" in metrics_data:
                        metrics["memory_usage"] = metrics_data["memory_usage"]
                except Exception:
                    pass
        except requests.RequestException:
            pass
            
        return metrics
    
    def _monitoring_thread_func(self, service_url: str, patch_id: str, duration: int, interval: int) -> None:
        """
        Thread function for continuous monitoring.
        
        Args:
            service_url: Base URL of the service
            patch_id: ID of the patch to monitor
            duration: Monitoring duration in seconds
            interval: Check interval in seconds
        """
        self.logger.info(f"Starting monitoring for patch {patch_id} at {service_url}")
        
        end_time = time.time() + duration
        
        # Initialize metrics entry
        if patch_id not in self.metrics_data:
            self.metrics_data[patch_id] = []
        
        try:
            while time.time() < end_time and not self.stop_monitoring.is_set():
                # Collect metrics
                metrics = self.collect_metrics(service_url)
                
                # Add to metrics data
                self.metrics_data[patch_id].append(metrics)
                
                # Check for alert conditions
                self._check_alerts(metrics, patch_id)
                
                # Wait for the next interval
                self.stop_monitoring.wait(interval)
        finally:
            self.is_monitoring = False
            self.logger.info(f"Monitoring completed for patch {patch_id}")
            
            # Save metrics
            self._save_metrics(patch_id)
    
    def _check_alerts(self, metrics: Dict[str, Any], patch_id: str) -> None:
        """
        Check metrics against alert thresholds.
        
        Args:
            metrics: Collected metrics
            patch_id: ID of the patch being monitored
        """
        thresholds = self.config["alert_thresholds"]
        
        # Check response time
        if "response_time" in metrics and "response_time" in thresholds:
            if metrics["response_time"] > thresholds["response_time"]:
                self._send_alert(
                    f"Response time exceeded threshold for patch {patch_id}",
                    {
                        "patch_id": patch_id,
                        "metric": "response_time",
                        "value": metrics["response_time"],
                        "threshold": thresholds["response_time"]
                    }
                )
        
        # Check error rate
        if "error_rate" in metrics and "error_rate" in thresholds:
            if metrics["error_rate"] > thresholds["error_rate"]:
                self._send_alert(
                    f"Error rate exceeded threshold for patch {patch_id}",
                    {
                        "patch_id": patch_id,
                        "metric": "error_rate",
                        "value": metrics["error_rate"],
                        "threshold": thresholds["error_rate"]
                    }
                )
        
        # Check memory usage
        if "memory_usage" in metrics and "memory_usage" in thresholds:
            if metrics["memory_usage"] > thresholds["memory_usage"]:
                self._send_alert(
                    f"Memory usage exceeded threshold for patch {patch_id}",
                    {
                        "patch_id": patch_id,
                        "metric": "memory_usage",
                        "value": metrics["memory_usage"],
                        "threshold": thresholds["memory_usage"]
                    }
                )
    
    def _save_metrics(self, patch_id: str) -> None:
        """
        Save metrics data to disk.
        
        Args:
            patch_id: ID of the patch
        """
        if patch_id not in self.metrics_data:
            return
            
        # Create metrics directory
        metrics_dir = project_root / "logs" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to file
        metrics_file = metrics_dir / f"{patch_id}_{int(time.time())}.json"
        
        try:
            with open(metrics_file, "w") as f:
                json.dump(self.metrics_data[patch_id], f, indent=2)
                
            self.logger.info(f"Saved metrics for patch {patch_id} to {metrics_file}")
            
        except Exception as e:
            self.logger.exception(e, message=f"Failed to save metrics for patch {patch_id}")
    
    def start_monitoring(self, service_url: str, patch_id: str) -> bool:
        """
        Start monitoring a service after deployment.
        
        Args:
            service_url: Base URL of the service
            patch_id: ID of the deployed patch
            
        Returns:
            True if monitoring started, False otherwise
        """
        if self.is_monitoring:
            self.logger.warning("Monitoring is already in progress")
            return False
            
        self.is_monitoring = True
        self.stop_monitoring.clear()
        
        # Start the monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_thread_func,
            args=(
                service_url,
                patch_id,
                self.config["duration"],
                self.config["check_interval"]
            )
        )
        
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info(f"Started monitoring for patch {patch_id}")
        return True
    
    def stop_monitoring_service(self) -> None:
        """Stop the monitoring thread."""
        if not self.is_monitoring:
            return
            
        self.logger.info("Stopping monitoring")
        self.stop_monitoring.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            
        self.is_monitoring = False
    
    def get_metrics(self, patch_id: str) -> List[Dict[str, Any]]:
        """
        Get collected metrics for a patch.
        
        Args:
            patch_id: ID of the patch
            
        Returns:
            List of metrics data points
        """
        return self.metrics_data.get(patch_id, [])
    
    def analyze_metrics(self, patch_id: str) -> Dict[str, Any]:
        """
        Analyze collected metrics for a patch.
        
        Args:
            patch_id: ID of the patch
            
        Returns:
            Analysis results
        """
        metrics = self.get_metrics(patch_id)
        
        if not metrics:
            return {"success": False, "message": "No metrics data available"}
            
        results = {
            "patch_id": patch_id,
            "num_data_points": len(metrics),
            "start_time": metrics[0]["timestamp"],
            "end_time": metrics[-1]["timestamp"]
        }
        
        # Calculate statistics for each metric
        for metric_name in self.config["metrics"]:
            # Extract values for this metric
            values = [m.get(metric_name) for m in metrics if metric_name in m]
            
            if not values:
                continue
                
            # Calculate statistics
            results[metric_name] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "values": values
            }
            
            # Check if any values exceeded thresholds
            if metric_name in self.config["alert_thresholds"]:
                threshold = self.config["alert_thresholds"][metric_name]
                exceeded = [v for v in values if v > threshold]
                
                results[metric_name]["exceeded_threshold"] = len(exceeded) > 0
                results[metric_name]["exceeded_count"] = len(exceeded)
                results[metric_name]["threshold"] = threshold
        
        # Overall success assessment
        success = True
        for metric_name in self.config["metrics"]:
            if metric_name in results and results[metric_name].get("exceeded_threshold", False):
                success = False
                break
                
        results["success"] = success
        
        return results
    
    def load_metrics_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load metrics history from disk.
        
        Returns:
            Dictionary of patch ID to metrics data
        """
        metrics_dir = project_root / "logs" / "metrics"
        if not metrics_dir.exists():
            return {}
            
        metrics_history = {}
        
        # Find all metrics files
        for metrics_file in metrics_dir.glob("*.json"):
            try:
                # Extract patch ID from filename
                filename = metrics_file.name
                patch_id = filename.split("_")[0]
                
                # Load metrics data
                with open(metrics_file, "r") as f:
                    metrics_data = json.load(f)
                    
                if patch_id not in metrics_history:
                    metrics_history[patch_id] = []
                    
                metrics_history[patch_id].extend(metrics_data)
                
            except Exception as e:
                self.logger.exception(e, message=f"Failed to load metrics from {metrics_file}")
        
        return metrics_history


class SuccessRateTracker:
    """
    Tracks success rate of deployed fixes.
    """
    
    def __init__(self, 
                history_file: Optional[Path] = None,
                log_level: str = "INFO"):
        """
        Initialize the success rate tracker.
        
        Args:
            history_file: File to store fix history
            log_level: Logging level
        """
        self.logger = MonitoringLogger("success_rate_tracker", log_level=log_level)
        
        # Set up history file
        self.history_file = history_file or (project_root / "logs" / "fix_history.json")
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load history
        self.history = self._load_history()
        
        self.logger.info("Initialized success rate tracker")
    
    def _load_history(self) -> Dict[str, Any]:
        """
        Load fix history from file.
        
        Returns:
            Fix history
        """
        if not self.history_file.exists():
            return {"fixes": [], "stats": {}}
            
        try:
            with open(self.history_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.exception(e, message=f"Failed to load fix history from {self.history_file}")
            return {"fixes": [], "stats": {}}
    
    def _save_history(self) -> None:
        """Save fix history to file."""
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            self.logger.exception(e, message=f"Failed to save fix history to {self.history_file}")
    
    def record_fix(self, 
                  patch_id: str, 
                  bug_id: str, 
                  success: bool,
                  metrics: Dict[str, Any] = None) -> None:
        """
        Record a fix and its success or failure.
        
        Args:
            patch_id: ID of the patch
            bug_id: ID of the bug fixed
            success: Whether the fix was successful
            metrics: Optional metrics data
        """
        # Create the fix record
        fix_record = {
            "patch_id": patch_id,
            "bug_id": bug_id,
            "success": success,
            "timestamp": time.time(),
            "metrics": metrics or {}
        }
        
        # Add to history
        self.history["fixes"].append(fix_record)
        
        # Update statistics
        stats = self.history["stats"]
        
        # Overall stats
        if "overall" not in stats:
            stats["overall"] = {"total": 0, "success": 0, "failure": 0}
            
        stats["overall"]["total"] += 1
        if success:
            stats["overall"]["success"] += 1
        else:
            stats["overall"]["failure"] += 1
            
        # Bug-specific stats
        if bug_id not in stats:
            stats[bug_id] = {"total": 0, "success": 0, "failure": 0}
            
        stats[bug_id]["total"] += 1
        if success:
            stats[bug_id]["success"] += 1
        else:
            stats[bug_id]["failure"] += 1
            
        # Calculate success rates
        for key, data in stats.items():
            data["success_rate"] = data["success"] / data["total"] if data["total"] > 0 else 0
        
        # Save history
        self._save_history()
        
        self.logger.info(
            f"Recorded fix {patch_id} for bug {bug_id}: {'success' if success else 'failure'}",
            success_rate=stats["overall"]["success_rate"]
        )
    
    def get_success_rate(self, bug_id: Optional[str] = None) -> float:
        """
        Get the success rate for fixes.
        
        Args:
            bug_id: Optional bug ID to get specific success rate
            
        Returns:
            Success rate (0.0 to 1.0)
        """
        stats = self.history["stats"]
        
        if bug_id:
            if bug_id in stats:
                return stats[bug_id]["success_rate"]
            return 0.0
            
        if "overall" in stats:
            return stats["overall"]["success_rate"]
            
        return 0.0
    
    def get_fix_stats(self) -> Dict[str, Any]:
        """
        Get statistics about fixes.
        
        Returns:
            Fix statistics
        """
        return self.history["stats"]
    
    def get_recent_fixes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent fixes.
        
        Args:
            limit: Maximum number of fixes to return
            
        Returns:
            List of recent fixes
        """
        fixes = self.history["fixes"]
        
        # Sort by timestamp (newest first)
        sorted_fixes = sorted(fixes, key=lambda x: x["timestamp"], reverse=True)
        
        return sorted_fixes[:limit]


if __name__ == "__main__":
    # Example usage
    monitor = PostDeploymentMonitor()
    
    # Register alert handler
    def alert_handler(message, data):
        print(f"ALERT: {message}")
        print(f"Data: {data}")
        
    monitor.register_alert_handler(alert_handler)
    
    # Check health
    health_data = monitor.check_health("http://localhost:8000/health")
    print(f"Health check: {health_data}")
    
    # Collect metrics
    metrics = monitor.collect_metrics("http://localhost:8000")
    print(f"Collected metrics: {metrics}")
    
    # Start monitoring (runs in background)
    monitoring_started = monitor.start_monitoring("http://localhost:8000", "example-patch-1")
    
    if monitoring_started:
        # Skip input prompt in test mode
        if not (os.environ.get('USE_MOCK_TESTS') == 'true' or os.environ.get('HOMEOSTASIS_TEST_MODE') == 'true'):
            print("Monitoring started. Press Enter to stop...")
            input()
        
        # Stop monitoring
        monitor.stop_monitoring_service()
        
        # Analyze metrics
        analysis = monitor.analyze_metrics("example-patch-1")
        print(f"Metrics analysis: {analysis}")
    
    # Example of success rate tracking
    tracker = SuccessRateTracker()
    
    # Record successful fix
    tracker.record_fix("patch-1", "bug_1", True, {"response_time": 150})
    
    # Record failed fix
    tracker.record_fix("patch-2", "bug_2", False, {"error_rate": 0.2})
    
    # Get success rate
    overall_rate = tracker.get_success_rate()
    print(f"Overall success rate: {overall_rate * 100:.2f}%")
    
    # Get specific bug success rate
    bug_rate = tracker.get_success_rate("bug_1")
    print(f"Bug 1 success rate: {bug_rate * 100:.2f}%")
    
    # Get statistics
    stats = tracker.get_fix_stats()
    print(f"Fix statistics: {stats}")
    
    # Get recent fixes
    recent_fixes = tracker.get_recent_fixes()
    print(f"Recent fixes: {recent_fixes}")