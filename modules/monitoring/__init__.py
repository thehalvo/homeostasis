# Monitoring module package
"""
Monitoring module for tracking service behavior and errors.

This module provides tools for monitoring services, including:
- Log extraction and parsing
- Error detection and tracking
- Post-deployment monitoring
- Metrics collection
- Alerting for unexpected behavior
- Feedback loop for fix quality improvement
"""

__version__ = "0.2.0"

# Core monitoring components
from modules.monitoring.logger import MonitoringLogger
from modules.monitoring.extractor import get_latest_errors, get_error_summary
from modules.monitoring.post_deployment import PostDeploymentMonitor, SuccessRateTracker
from modules.monitoring.metrics_collector import MetricsCollector
from modules.monitoring.feedback_loop import FeedbackLoop, FixImprovement
from modules.monitoring.alert_system import AlertManager, AnomalyDetector
from modules.monitoring.distributed_monitoring import DistributedMonitor

__all__ = [
    'MonitoringLogger',
    'get_latest_errors',
    'get_error_summary',
    'PostDeploymentMonitor',
    'SuccessRateTracker',
    'MetricsCollector',
    'FeedbackLoop',
    'FixImprovement',
    'AlertManager',
    'AnomalyDetector',
    'DistributedMonitor'
]