"""
Metrics collection for evaluating fix effectiveness.

This module provides utilities for:
1. Collecting performance and behavioral metrics
2. Storing and analyzing metric data
3. Generating reports on fix effectiveness
"""
import os
import sys
import time
import json
import statistics
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.monitoring.logger import MonitoringLogger


class MetricsCollector:
    """
    Collects and analyzes metrics for fix effectiveness.
    """
    
    def __init__(self, 
                storage_dir: Optional[Path] = None,
                history_limit: int = 100, 
                log_level: str = "INFO"):
        """
        Initialize the metrics collector.
        
        Args:
            storage_dir: Directory to store metrics data
            history_limit: Maximum number of metric records to keep
            log_level: Logging level
        """
        self.logger = MonitoringLogger("metrics_collector", log_level=log_level)
        
        # Set up storage directory
        self.storage_dir = storage_dir or (project_root / "logs" / "metrics")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.history_limit = history_limit
        
        # Initialize storage for in-memory metrics
        self.metrics = {}
        self.grouped_metrics = {}
        
        # Load existing metrics
        self._load_metrics()
        
        self.logger.info(f"Initialized metrics collector with storage at {self.storage_dir}")
    
    def _load_metrics(self) -> None:
        """Load metrics from disk."""
        try:
            # Load all metric files
            for metric_file in self.storage_dir.glob("*.json"):
                try:
                    with open(metric_file, "r") as f:
                        data = json.load(f)
                        
                    # Extract metric type from filename
                    metric_type = metric_file.stem.split("_")[0]
                    
                    # Store in memory
                    if metric_type not in self.metrics:
                        self.metrics[metric_type] = []
                        
                    self.metrics[metric_type].extend(data)
                    
                except Exception as e:
                    self.logger.exception(e, message=f"Failed to load metrics from {metric_file}")
                    
            # Group metrics by entity
            self._group_metrics()
            
            self.logger.info(f"Loaded metrics: {', '.join(self.metrics.keys())}")
            
        except Exception as e:
            self.logger.exception(e, message="Failed to load metrics")
    
    def _group_metrics(self) -> None:
        """Group metrics by entity (patch ID, bug ID, etc.)."""
        self.grouped_metrics = {}
        
        for metric_type, metrics in self.metrics.items():
            self.grouped_metrics[metric_type] = {}
            
            for metric in metrics:
                # Identify entity (patch_id, bug_id, etc.)
                entity_id = None
                for key in ["patch_id", "bug_id", "fix_id", "id"]:
                    if key in metric:
                        entity_id = metric[key]
                        break
                        
                if not entity_id:
                    continue
                    
                # Group by entity
                if entity_id not in self.grouped_metrics[metric_type]:
                    self.grouped_metrics[metric_type][entity_id] = []
                    
                self.grouped_metrics[metric_type][entity_id].append(metric)
    
    def _save_metrics(self, metric_type: str) -> None:
        """
        Save metrics to disk.
        
        Args:
            metric_type: Type of metrics to save
        """
        if metric_type not in self.metrics:
            return
            
        try:
            # Create filename with timestamp
            timestamp = int(time.time())
            filename = f"{metric_type}_{timestamp}.json"
            file_path = self.storage_dir / filename
            
            # Save metrics
            with open(file_path, "w") as f:
                json.dump(self.metrics[metric_type], f, indent=2)
                
            self.logger.info(f"Saved {len(self.metrics[metric_type])} {metric_type} metrics to {file_path}")
            
            # Clean up old files if needed
            self._cleanup_old_files(metric_type)
            
        except Exception as e:
            self.logger.exception(e, message=f"Failed to save {metric_type} metrics")
    
    def _cleanup_old_files(self, metric_type: str) -> None:
        """
        Clean up old metric files to stay within history limit.
        
        Args:
            metric_type: Type of metrics to clean up
        """
        # Find all files for this metric type
        files = list(self.storage_dir.glob(f"{metric_type}_*.json"))
        
        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest files if we exceed the limit
        if len(files) > self.history_limit:
            for file in files[:-self.history_limit]:
                try:
                    file.unlink()
                    self.logger.debug(f"Removed old metrics file: {file}")
                except Exception as e:
                    self.logger.exception(e, message=f"Failed to remove old file: {file}")
    
    def record_metric(self, 
                     metric_type: str, 
                     metric_data: Dict[str, Any],
                     save: bool = True) -> None:
        """
        Record a new metric.
        
        Args:
            metric_type: Type of metric (e.g., "test", "deployment", "fix")
            metric_data: Metric data to record
            save: Whether to save metrics to disk immediately
        """
        # Add timestamp if not present
        if "timestamp" not in metric_data:
            metric_data["timestamp"] = time.time()
            
        # Initialize metric type if not present
        if metric_type not in self.metrics:
            self.metrics[metric_type] = []
            
        # Add the metric
        self.metrics[metric_type].append(metric_data)
        
        # Update grouped metrics
        if metric_type not in self.grouped_metrics:
            self.grouped_metrics[metric_type] = {}
            
        # Identify entity (patch_id, bug_id, etc.)
        entity_id = None
        for key in ["patch_id", "bug_id", "fix_id", "id"]:
            if key in metric_data:
                entity_id = metric_data[key]
                break
                
        if entity_id:
            if entity_id not in self.grouped_metrics[metric_type]:
                self.grouped_metrics[metric_type][entity_id] = []
                
            self.grouped_metrics[metric_type][entity_id].append(metric_data)
        
        self.logger.info(f"Recorded {metric_type} metric for {entity_id or 'unknown'}")
        
        # Save if requested
        if save:
            self._save_metrics(metric_type)
    
    def record_test_metric(self, 
                          patch_id: str, 
                          success: bool,
                          duration: float,
                          memory_usage: Optional[float] = None,
                          other_data: Dict[str, Any] = None) -> None:
        """
        Record a test metric.
        
        Args:
            patch_id: ID of the patch being tested
            success: Whether the test was successful
            duration: Test duration in seconds
            memory_usage: Optional memory usage in MB
            other_data: Additional data to record
        """
        metric_data = {
            "patch_id": patch_id,
            "success": success,
            "duration": duration,
            "timestamp": time.time()
        }
        
        if memory_usage is not None:
            metric_data["memory_usage"] = memory_usage
            
        if other_data:
            metric_data.update(other_data)
            
        self.record_metric("test", metric_data)
    
    def record_deployment_metric(self, 
                               patch_id: str, 
                               success: bool,
                               duration: float,
                               other_data: Dict[str, Any] = None) -> None:
        """
        Record a deployment metric.
        
        Args:
            patch_id: ID of the patch being deployed
            success: Whether the deployment was successful
            duration: Deployment duration in seconds
            other_data: Additional data to record
        """
        metric_data = {
            "patch_id": patch_id,
            "success": success,
            "duration": duration,
            "timestamp": time.time()
        }
            
        if other_data:
            metric_data.update(other_data)
            
        self.record_metric("deployment", metric_data)
    
    def record_fix_metric(self, 
                        patch_id: str, 
                        bug_id: str,
                        success: bool,
                        response_time: Optional[float] = None,
                        error_rate: Optional[float] = None,
                        other_data: Dict[str, Any] = None) -> None:
        """
        Record a fix effectiveness metric.
        
        Args:
            patch_id: ID of the patch
            bug_id: ID of the bug being fixed
            success: Whether the fix was successful
            response_time: Optional service response time in ms
            error_rate: Optional error rate (0.0 to 1.0)
            other_data: Additional data to record
        """
        metric_data = {
            "patch_id": patch_id,
            "bug_id": bug_id,
            "success": success,
            "timestamp": time.time()
        }
        
        if response_time is not None:
            metric_data["response_time"] = response_time
            
        if error_rate is not None:
            metric_data["error_rate"] = error_rate
            
        if other_data:
            metric_data.update(other_data)
            
        self.record_metric("fix", metric_data)
    
    def get_metrics(self, 
                   metric_type: str, 
                   entity_id: Optional[str] = None,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get metrics data.
        
        Args:
            metric_type: Type of metrics to get
            entity_id: Optional entity ID to filter by
            start_time: Optional start timestamp to filter by
            end_time: Optional end timestamp to filter by
            
        Returns:
            List of metrics data
        """
        if metric_type not in self.metrics:
            return []
            
        # Get metrics
        if entity_id and metric_type in self.grouped_metrics and entity_id in self.grouped_metrics[metric_type]:
            metrics = self.grouped_metrics[metric_type][entity_id]
        else:
            metrics = self.metrics[metric_type]
            
        # Apply time filters
        if start_time is not None or end_time is not None:
            filtered_metrics = []
            
            for metric in metrics:
                timestamp = metric.get("timestamp", 0)
                
                if start_time is not None and timestamp < start_time:
                    continue
                    
                if end_time is not None and timestamp > end_time:
                    continue
                    
                filtered_metrics.append(metric)
                
            return filtered_metrics
        
        return metrics
    
    def analyze_metrics(self, 
                       metric_type: str, 
                       entity_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze metrics and generate statistics.
        
        Args:
            metric_type: Type of metrics to analyze
            entity_id: Optional entity ID to filter by
            
        Returns:
            Analysis results
        """
        metrics = self.get_metrics(metric_type, entity_id)
        
        if not metrics:
            return {"count": 0}
            
        # Basic stats
        result = {
            "count": len(metrics),
            "timespan": {
                "start": min(m.get("timestamp", 0) for m in metrics),
                "end": max(m.get("timestamp", 0) for m in metrics)
            }
        }
        
        # Success rate
        success_values = [m.get("success", False) for m in metrics if "success" in m]
        if success_values:
            success_count = sum(1 for v in success_values if v)
            result["success_rate"] = success_count / len(success_values)
            result["success_count"] = success_count
            result["failure_count"] = len(success_values) - success_count
        
        # Numeric metrics
        numeric_fields = ["duration", "memory_usage", "response_time", "error_rate"]
        
        for field in numeric_fields:
            values = [m.get(field) for m in metrics if field in m and m.get(field) is not None]
            
            if not values:
                continue
                
            try:
                result[field] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "median": statistics.median(values),
                    "stddev": statistics.stdev(values) if len(values) > 1 else 0
                }
            except (TypeError, statistics.StatisticsError) as e:
                self.logger.warning(f"Failed to calculate statistics for {field}: {str(e)}")
        
        return result
    
    def generate_report(self, 
                       metric_types: List[str] = None, 
                       output_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report on metrics.
        
        Args:
            metric_types: Types of metrics to include (default: all)
            output_file: Optional file to write the report
            
        Returns:
            Report data
        """
        if metric_types is None:
            metric_types = list(self.metrics.keys())
            
        report = {
            "timestamp": time.time(),
            "date": datetime.datetime.now().isoformat(),
            "metric_types": metric_types,
            "metrics": {}
        }
        
        # Generate report for each metric type
        for metric_type in metric_types:
            if metric_type not in self.metrics:
                continue
                
            # Overall analysis
            overall = self.analyze_metrics(metric_type)
            
            # Per-entity analysis
            entities = {}
            if metric_type in self.grouped_metrics:
                for entity_id in self.grouped_metrics[metric_type]:
                    entities[entity_id] = self.analyze_metrics(metric_type, entity_id)
            
            report["metrics"][metric_type] = {
                "overall": overall,
                "entities": entities
            }
        
        # Write report to file if requested
        if output_file:
            try:
                with open(output_file, "w") as f:
                    json.dump(report, f, indent=2)
                    
                self.logger.info(f"Wrote metrics report to {output_file}")
                
            except Exception as e:
                self.logger.exception(e, message=f"Failed to write report to {output_file}")
        
        return report
    
    def get_effectiveness_score(self, patch_id: str) -> float:
        """
        Calculate an effectiveness score for a patch.
        
        This score combines test success, deployment success, and fix success
        into a single value from 0.0 (complete failure) to 1.0 (completely effective).
        
        Args:
            patch_id: ID of the patch
            
        Returns:
            Effectiveness score (0.0 to 1.0)
        """
        # Weights for each component
        weights = {
            "test": 0.3,
            "deployment": 0.3,
            "fix": 0.4
        }
        
        scores = {}
        
        # Calculate test score
        test_metrics = self.get_metrics("test", patch_id)
        if test_metrics:
            success_rate = sum(1 for m in test_metrics if m.get("success", False)) / len(test_metrics)
            scores["test"] = success_rate
        
        # Calculate deployment score
        deployment_metrics = self.get_metrics("deployment", patch_id)
        if deployment_metrics:
            success_rate = sum(1 for m in deployment_metrics if m.get("success", False)) / len(deployment_metrics)
            scores["deployment"] = success_rate
        
        # Calculate fix score
        fix_metrics = self.get_metrics("fix", patch_id)
        if fix_metrics:
            success_rate = sum(1 for m in fix_metrics if m.get("success", False)) / len(fix_metrics)
            scores["fix"] = success_rate
        
        # Calculate weighted score
        total_weight = 0
        weighted_score = 0
        
        for metric_type, weight in weights.items():
            if metric_type in scores:
                weighted_score += scores[metric_type] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
            
        return weighted_score / total_weight


if __name__ == "__main__":
    # Example usage
    collector = MetricsCollector()
    
    # Record test metrics
    collector.record_test_metric(
        patch_id="test-patch-1",
        success=True,
        duration=1.5,
        memory_usage=256,
        other_data={"test_type": "unit"}
    )
    
    collector.record_test_metric(
        patch_id="test-patch-1",
        success=True,
        duration=3.2,
        memory_usage=512,
        other_data={"test_type": "integration"}
    )
    
    # Record deployment metric
    collector.record_deployment_metric(
        patch_id="test-patch-1",
        success=True,
        duration=5.8,
        other_data={"environment": "staging"}
    )
    
    # Record fix metrics
    collector.record_fix_metric(
        patch_id="test-patch-1",
        bug_id="bug-123",
        success=True,
        response_time=120,
        error_rate=0.01
    )
    
    # Get metrics
    test_metrics = collector.get_metrics("test", "test-patch-1")
    print(f"Test metrics: {test_metrics}")
    
    # Analyze metrics
    analysis = collector.analyze_metrics("test", "test-patch-1")
    print(f"Test metrics analysis: {analysis}")
    
    # Generate report
    report = collector.generate_report()
    print(f"Metrics report: {report}")
    
    # Get effectiveness score
    score = collector.get_effectiveness_score("test-patch-1")
    print(f"Effectiveness score: {score:.2f}")