"""
Canary deployment implementation for Homeostasis.

Provides functionality for gradually rolling out fixes to a subset of traffic
and monitoring their performance before full deployment.
"""

import enum
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.monitoring.metrics_collector import MetricsCollector
from modules.monitoring.post_deployment import PostDeploymentMonitor
from modules.security.audit import get_audit_logger

logger = logging.getLogger(__name__)


class CanaryStatus(enum.Enum):
    """Enumeration of canary deployment statuses."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class CanaryMetrics:
    """Metrics for a canary deployment."""

    error_rate: float = 0.0
    response_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    success_rate: float = 1.0


class CanaryDeployment:
    """
    Manages canary deployments for safe, gradual rollout of fixes.

    Canary deployments involve directing a small percentage of traffic to the new
    version (the "canary"), monitoring its performance, and gradually increasing
    the percentage if metrics remain healthy.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize canary deployment manager.

        Args:
            config: Configuration dictionary for canary settings
        """
        self.config = config or {}

        # Set default values
        self.initial_percentage = self.config.get("percentage", 10)
        self.percentage_increment = self.config.get("increment", 10)
        self.interval_seconds = self.config.get("interval", 300)  # 5 minutes
        self.max_percentage = self.config.get("max_percentage", 100)
        self.metrics_threshold = self.config.get(
            "success_threshold", 0.95
        )  # 95% success

        # Initialize metrics collector
        self.metrics_collector = MetricsCollector()

        # Initialize monitoring
        self.post_deployment_monitor = PostDeploymentMonitor(config=self.config)

        # Set initial state
        self.reset_state()

        # Create the data directory if it doesn't exist
        self.data_dir = Path("logs/canary")
        os.makedirs(self.data_dir, exist_ok=True)

    def reset_state(self) -> None:
        """Reset the state of the canary deployment."""
        self.service_name: Optional[str] = None
        self.fix_id: Optional[str] = None
        self.status = CanaryStatus.NOT_STARTED
        self.current_percentage = 0
        self.start_time: Optional[datetime] = None
        self.last_increment_time: Optional[datetime] = None
        self.completion_time: Optional[datetime] = None
        self.metrics_history: List[Dict[str, Any]] = []
        self.current_metrics = CanaryMetrics()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = False

    def start(
        self, service_name: str, fix_id: str, initial_percentage: Optional[int] = None
    ) -> bool:
        """Start a canary deployment.

        Args:
            service_name: Name of the service being deployed
            fix_id: ID of the fix being deployed
            initial_percentage: Optional override for initial percentage

        Returns:
            bool: True if canary deployment started successfully
        """
        # Check if already running
        if self.status in [CanaryStatus.IN_PROGRESS, CanaryStatus.PAUSED]:
            logger.warning(
                f"Canary deployment already in progress for {self.service_name}"
            )
            return False

        # Reset and initialize state
        self.reset_state()
        self.service_name = service_name
        self.fix_id = fix_id
        self.status = CanaryStatus.IN_PROGRESS
        self.start_time = datetime.now()
        self.last_increment_time = self.start_time
        self.current_percentage = initial_percentage or self.initial_percentage

        # Start monitoring
        self._start_monitoring()

        # Log the start of canary deployment
        logger.info(
            f"Started canary deployment for {service_name} fix {fix_id} "
            f"at {self.current_percentage}%"
        )

        # Log to audit log if available
        try:
            get_audit_logger().log_event(
                event_type="canary_deployment_started",
                details={
                    "service_name": service_name,
                    "fix_id": fix_id,
                    "initial_percentage": self.current_percentage,
                    "metrics_threshold": self.metrics_threshold,
                },
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")

        # Save initial state
        self._save_state()

        return True

    def _start_monitoring(self) -> None:
        """Start the background thread for monitoring canary metrics."""
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            return

        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

    def _stop_monitoring(self) -> None:
        """Stop the background monitoring thread."""
        if self.monitoring_thread:
            self.stop_monitoring = True
            self.monitoring_thread.join(timeout=5)
            self.monitoring_thread = None

    def _monitoring_loop(self) -> None:
        """Background thread to monitor canary metrics and auto-increment."""
        try:
            while not self.stop_monitoring and self.status == CanaryStatus.IN_PROGRESS:
                # Collect current metrics
                self._update_metrics()

                # Check if it's time to increment
                current_time = datetime.now()
                if self.last_increment_time is not None:
                    seconds_since_last = (
                        current_time - self.last_increment_time
                    ).total_seconds()
                else:
                    # First iteration, use start_time
                    seconds_since_last = (
                        current_time - self.start_time
                    ).total_seconds() if self.start_time else 0

                if seconds_since_last >= self.interval_seconds:
                    # Check if metrics are healthy
                    if self._are_metrics_healthy():
                        # Auto-increment the percentage
                        self._increment_percentage()
                    else:
                        # Metrics are unhealthy, auto-rollback
                        logger.warning("Metrics are unhealthy, auto-rolling back")
                        self.rollback()
                        break

                # Save the current state
                self._save_state()

                # Sleep for a bit to avoid excessive CPU usage
                time.sleep(min(30, self.interval_seconds / 10))

        except Exception as e:
            logger.exception(f"Error in canary monitoring thread: {str(e)}")
            self.status = CanaryStatus.FAILED
            self._save_state()

    def _update_metrics(self) -> None:
        """Update the current metrics for the canary deployment."""
        try:
            # Get metrics from the metrics collector
            if not self.fix_id:
                return
            metrics_list = self.metrics_collector.get_metrics("fix", self.fix_id)

            # Update current metrics - use the latest metric if available
            if metrics_list and len(metrics_list) > 0:
                metrics = metrics_list[-1]  # Get the most recent metric
                self.current_metrics = CanaryMetrics(
                    error_rate=metrics.get("error_rate", 0.0),
                    response_time=metrics.get("response_time", 0.0),
                    memory_usage=metrics.get("memory_usage", 0.0),
                    cpu_usage=metrics.get("cpu_usage", 0.0),
                    success_rate=1.0 - metrics.get("error_rate", 0.0),
                )

                # Add to history
                self.metrics_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "percentage": self.current_percentage,
                        "metrics": {
                            "error_rate": self.current_metrics.error_rate,
                            "response_time": self.current_metrics.response_time,
                            "memory_usage": self.current_metrics.memory_usage,
                            "cpu_usage": self.current_metrics.cpu_usage,
                            "success_rate": self.current_metrics.success_rate,
                        },
                    }
                )

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")

    def _are_metrics_healthy(self) -> bool:
        """Check if the current metrics are healthy.

        Returns:
            bool: True if metrics are healthy
        """
        # Basic check for success rate exceeding threshold
        return bool(self.current_metrics.success_rate >= self.metrics_threshold)

    def _increment_percentage(self) -> bool:
        """Increment the canary percentage.

        Returns:
            bool: True if incremented, False if already at max
        """
        if self.current_percentage >= self.max_percentage:
            # Already at max, complete the deployment
            self.complete()
            return False

        # Calculate next percentage
        next_percentage = min(
            self.current_percentage + self.percentage_increment, self.max_percentage
        )

        logger.info(
            f"Incrementing canary deployment from {self.current_percentage}% "
            f"to {next_percentage}%"
        )

        # Update state
        self.current_percentage = next_percentage
        self.last_increment_time = datetime.now()

        # Log to audit log if available
        try:
            get_audit_logger().log_event(
                event_type="canary_deployment_incremented",
                details={
                    "service_name": self.service_name,
                    "fix_id": self.fix_id,
                    "previous_percentage": self.current_percentage
                    - self.percentage_increment,
                    "current_percentage": self.current_percentage,
                    "metrics": {
                        "error_rate": self.current_metrics.error_rate,
                        "success_rate": self.current_metrics.success_rate,
                    },
                },
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")

        # Check if we've reached 100%
        if self.current_percentage >= self.max_percentage:
            # Auto-complete
            self.complete()
            return False

        # Save the updated state
        self._save_state()

        return True

    def _save_state(self) -> None:
        """Save the current state of the canary deployment."""
        if not self.service_name or not self.fix_id:
            return

        state_file = self.data_dir / f"canary_{self.service_name}_{self.fix_id}.json"

        try:
            state_data = {
                "service_name": self.service_name,
                "fix_id": self.fix_id,
                "status": self.status.value,
                "current_percentage": self.current_percentage,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "last_increment_time": (
                    self.last_increment_time.isoformat()
                    if self.last_increment_time
                    else None
                ),
                "completion_time": (
                    self.completion_time.isoformat() if self.completion_time else None
                ),
                "config": {
                    "initial_percentage": self.initial_percentage,
                    "percentage_increment": self.percentage_increment,
                    "interval_seconds": self.interval_seconds,
                    "max_percentage": self.max_percentage,
                    "metrics_threshold": self.metrics_threshold,
                },
                "current_metrics": {
                    "error_rate": self.current_metrics.error_rate,
                    "response_time": self.current_metrics.response_time,
                    "memory_usage": self.current_metrics.memory_usage,
                    "cpu_usage": self.current_metrics.cpu_usage,
                    "success_rate": self.current_metrics.success_rate,
                },
                "metrics_history": self.metrics_history,
            }

            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving canary state: {str(e)}")

    def _load_state(self, service_name: str, fix_id: str) -> bool:
        """Load the state of a canary deployment.

        Args:
            service_name: Name of the service
            fix_id: ID of the fix

        Returns:
            bool: True if state was loaded successfully
        """
        state_file = self.data_dir / f"canary_{service_name}_{fix_id}.json"

        if not state_file.exists():
            return False

        try:
            with open(state_file, "r") as f:
                state_data = json.load(f)

            # Restore state
            self.service_name = state_data["service_name"]
            self.fix_id = state_data["fix_id"]
            self.status = CanaryStatus(state_data["status"])
            self.current_percentage = state_data["current_percentage"]
            self.start_time = (
                datetime.fromisoformat(state_data["start_time"])
                if state_data["start_time"]
                else None
            )
            self.last_increment_time = (
                datetime.fromisoformat(state_data["last_increment_time"])
                if state_data["last_increment_time"]
                else None
            )
            self.completion_time = (
                datetime.fromisoformat(state_data["completion_time"])
                if state_data["completion_time"]
                else None
            )

            # Restore config
            config = state_data.get("config", {})
            self.initial_percentage = config.get(
                "initial_percentage", self.initial_percentage
            )
            self.percentage_increment = config.get(
                "percentage_increment", self.percentage_increment
            )
            self.interval_seconds = config.get(
                "interval_seconds", self.interval_seconds
            )
            self.max_percentage = config.get("max_percentage", self.max_percentage)
            self.metrics_threshold = config.get(
                "metrics_threshold", self.metrics_threshold
            )

            # Restore metrics
            metrics = state_data.get("current_metrics", {})
            self.current_metrics = CanaryMetrics(
                error_rate=metrics.get("error_rate", 0.0),
                response_time=metrics.get("response_time", 0.0),
                memory_usage=metrics.get("memory_usage", 0.0),
                cpu_usage=metrics.get("cpu_usage", 0.0),
                success_rate=metrics.get("success_rate", 1.0),
            )

            # Restore metrics history
            self.metrics_history = state_data.get("metrics_history", [])

            return True

        except Exception as e:
            logger.error(f"Error loading canary state: {str(e)}")
            return False

    def promote(self) -> bool:
        """Manually promote the canary to the next percentage.

        Returns:
            bool: True if promoted, False if already at max or not in progress
        """
        if self.status != CanaryStatus.IN_PROGRESS:
            logger.warning(f"Cannot promote canary: status is {self.status.value}")
            return False

        return self._increment_percentage()

    def pause(self) -> bool:
        """Pause the canary deployment.

        Returns:
            bool: True if paused, False if not in progress
        """
        if self.status != CanaryStatus.IN_PROGRESS:
            logger.warning(f"Cannot pause canary: status is {self.status.value}")
            return False

        # Update status
        self.status = CanaryStatus.PAUSED

        # Stop monitoring
        self._stop_monitoring()

        logger.info(
            f"Paused canary deployment for {self.service_name} at {self.current_percentage}%"
        )

        # Log to audit log if available
        try:
            get_audit_logger().log_event(
                event_type="canary_deployment_paused",
                details={
                    "service_name": self.service_name,
                    "fix_id": self.fix_id,
                    "current_percentage": self.current_percentage,
                },
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")

        # Save the updated state
        self._save_state()

        return True

    def resume(self) -> bool:
        """Resume a paused canary deployment.

        Returns:
            bool: True if resumed, False if not paused
        """
        if self.status != CanaryStatus.PAUSED:
            logger.warning(f"Cannot resume canary: status is {self.status.value}")
            return False

        # Update status
        self.status = CanaryStatus.IN_PROGRESS
        self.last_increment_time = datetime.now()

        # Restart monitoring
        self._start_monitoring()

        logger.info(
            f"Resumed canary deployment for {self.service_name} at {self.current_percentage}%"
        )

        # Log to audit log if available
        try:
            get_audit_logger().log_event(
                event_type="canary_deployment_resumed",
                details={
                    "service_name": self.service_name,
                    "fix_id": self.fix_id,
                    "current_percentage": self.current_percentage,
                },
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")

        # Save the updated state
        self._save_state()

        return True

    def complete(self) -> bool:
        """Complete the canary deployment.

        Returns:
            bool: True if completed, False if not in progress or already completed
        """
        if self.status not in [CanaryStatus.IN_PROGRESS, CanaryStatus.PAUSED]:
            logger.warning(f"Cannot complete canary: status is {self.status.value}")
            return False

        # Set to 100% and update status
        self.current_percentage = self.max_percentage
        self.status = CanaryStatus.COMPLETED
        self.completion_time = datetime.now()

        # Stop monitoring
        self._stop_monitoring()

        duration = (
            (self.completion_time - self.start_time).total_seconds()
            if self.start_time and self.completion_time
            else 0
        )
        logger.info(
            f"Completed canary deployment for {self.service_name} after "
            f"{duration} seconds"
        )

        # Log to audit log if available
        try:
            get_audit_logger().log_event(
                event_type="canary_deployment_completed",
                details={
                    "service_name": self.service_name,
                    "fix_id": self.fix_id,
                    "duration_seconds": (
                        (self.completion_time - self.start_time).total_seconds()
                        if self.start_time and self.completion_time
                        else 0
                    ),
                    "final_metrics": {
                        "error_rate": self.current_metrics.error_rate,
                        "success_rate": self.current_metrics.success_rate,
                    },
                },
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")

        # Save the final state
        self._save_state()

        return True

    def rollback(self) -> bool:
        """Roll back the canary deployment.

        Returns:
            bool: True if rolled back, False if already completed or rolled back
        """
        if self.status not in [CanaryStatus.IN_PROGRESS, CanaryStatus.PAUSED]:
            logger.warning(f"Cannot roll back canary: status is {self.status.value}")
            return False

        # Update status
        self.status = CanaryStatus.ROLLED_BACK
        self.completion_time = datetime.now()

        # Stop monitoring
        self._stop_monitoring()

        duration = (
            (self.completion_time - self.start_time).total_seconds()
            if self.start_time and self.completion_time
            else 0
        )
        logger.warning(
            f"Rolled back canary deployment for {self.service_name} at {self.current_percentage}% "
            f"after {duration} seconds"
        )

        # Log to audit log if available
        try:
            get_audit_logger().log_event(
                event_type="canary_deployment_rolled_back",
                details={
                    "service_name": self.service_name,
                    "fix_id": self.fix_id,
                    "percentage_at_rollback": self.current_percentage,
                    "duration_seconds": (
                        (self.completion_time - self.start_time).total_seconds()
                        if self.start_time and self.completion_time
                        else 0
                    ),
                    "metrics_at_rollback": {
                        "error_rate": self.current_metrics.error_rate,
                        "success_rate": self.current_metrics.success_rate,
                    },
                },
            )
        except Exception as e:
            logger.debug(f"Could not log to audit log: {str(e)}")

        # Save the final state
        self._save_state()

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the canary deployment.

        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "service_name": self.service_name,
            "fix_id": self.fix_id,
            "status": self.status.value if self.status else None,
            "current_percentage": self.current_percentage,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_increment_time": (
                self.last_increment_time.isoformat()
                if self.last_increment_time
                else None
            ),
            "completion_time": (
                self.completion_time.isoformat() if self.completion_time else None
            ),
            "elapsed_time": (
                (datetime.now() - self.start_time).total_seconds()
                if self.start_time
                else 0
            ),
            "metrics": {
                "error_rate": self.current_metrics.error_rate,
                "response_time": self.current_metrics.response_time,
                "memory_usage": self.current_metrics.memory_usage,
                "cpu_usage": self.current_metrics.cpu_usage,
                "success_rate": self.current_metrics.success_rate,
            },
            "config": {
                "initial_percentage": self.initial_percentage,
                "percentage_increment": self.percentage_increment,
                "interval_seconds": self.interval_seconds,
                "max_percentage": self.max_percentage,
                "metrics_threshold": self.metrics_threshold,
            },
        }


# Singleton instance for app-wide use
_canary_deployment = None


def get_canary_deployment(config: Optional[Dict[str, Any]] = None) -> CanaryDeployment:
    """Get or create the singleton CanaryDeployment instance.

    Args:
        config: Optional configuration to initialize the manager with

    Returns:
        CanaryDeployment: The canary deployment instance
    """
    global _canary_deployment
    if _canary_deployment is None:
        _canary_deployment = CanaryDeployment(config)
    return _canary_deployment
