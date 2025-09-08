"""
Health Checks Module

This module provides health checking functionality for monitoring system health.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthChecker:
    """
    Performs health checks on services and components.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the health checker.

        Args:
            config: Configuration for health checks
        """
        self.config = config or {}
        self.checks = []
        self.results = {}

    def register_check(self, name: str, check_func) -> None:
        """
        Register a health check.

        Args:
            name: Name of the health check
            check_func: Function that returns HealthStatus
        """
        self.checks.append((name, check_func))

    def run_checks(self) -> Dict[str, Any]:
        """
        Run all registered health checks.

        Returns:
            Dictionary of health check results
        """
        results = {}
        overall_status = HealthStatus.HEALTHY

        for name, check_func in self.checks:
            try:
                status = check_func()
                results[name] = {
                    "status": status.value,
                    "timestamp": datetime.now().isoformat(),
                    "error": None,
                }

                # Update overall status
                if status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif (
                    status == HealthStatus.DEGRADED and
                    overall_status != HealthStatus.UNHEALTHY
                ):
                    overall_status = HealthStatus.DEGRADED

            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                }
                overall_status = HealthStatus.UNHEALTHY

        self.results = {
            "overall_status": overall_status.value,
            "checks": results,
            "timestamp": datetime.now().isoformat(),
        }

        return self.results

    def get_status(self) -> HealthStatus:
        """
        Get the overall health status.

        Returns:
            Overall health status
        """
        if not self.results:
            return HealthStatus.UNKNOWN

        status_str = self.results.get("overall_status", "unknown")
        return HealthStatus(status_str)

    def is_healthy(self) -> bool:
        """
        Check if the system is healthy.

        Returns:
            True if healthy, False otherwise
        """
        return self.get_status() == HealthStatus.HEALTHY

    def check_service_health(
        self, service_name: str, metrics: Dict[str, Any]
    ) -> HealthStatus:
        """
        Check health of a specific service based on metrics.

        Args:
            service_name: Name of the service
            metrics: Service metrics

        Returns:
            Health status of the service
        """
        # Simple health check based on common metrics
        error_rate = metrics.get("error_rate", 0)
        response_time = metrics.get("response_time", 0)
        cpu_usage = metrics.get("cpu_usage", 0)
        memory_usage = metrics.get("memory_usage", 0)

        # Define thresholds
        if error_rate > 0.1 or response_time > 5000:  # 10% errors or 5s response time
            return HealthStatus.UNHEALTHY
        elif (
            error_rate > 0.05 or
            response_time > 2000 or
            cpu_usage > 80 or
            memory_usage > 80
        ):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
