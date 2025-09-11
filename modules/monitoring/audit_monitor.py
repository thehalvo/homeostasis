"""
Audit Monitoring for Homeostasis.

This module provides comprehensive monitoring and analysis of audit logs to track
healing activity patterns, detect suspicious activities, and generate reports on
system behavior and security posture.
"""

import datetime
import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.security.audit import log_event

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """Represents a parsed audit event for easier analysis."""

    event_id: str
    timestamp: datetime.datetime
    event_type: str
    status: str
    severity: str
    hostname: str
    user: str
    details: Dict[str, Any]
    source_ip: Optional[str] = None

    @classmethod
    def from_json(cls, event_json: Dict[str, Any]) -> "AuditEvent":
        """Create an AuditEvent instance from JSON data.

        Args:
            event_json: The JSON representation of an audit event

        Returns:
            AuditEvent: A new instance
        """
        # Parse timestamp
        timestamp_str = event_json.get("timestamp")
        if timestamp_str:
            # Handle 'Z' in ISO format
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str[:-1]
            timestamp = datetime.datetime.fromisoformat(timestamp_str)
        else:
            timestamp = datetime.datetime.now()

        # Extract source_ip from details if present
        source_ip = None
        details = event_json.get("details", {})
        if "source_ip" in details:
            source_ip = details["source_ip"]

        return cls(
            event_id=event_json.get("event_id", ""),
            timestamp=timestamp,
            event_type=event_json.get("event_type", ""),
            status=event_json.get("status", ""),
            severity=event_json.get("severity", ""),
            hostname=event_json.get("hostname", ""),
            user=event_json.get("user", ""),
            details=details,
            source_ip=source_ip or event_json.get("source_ip"),
        )


class AuditMonitor:
    """
    Monitors audit logs for healing activities, generates reports,
    and detects anomalies or potential security issues.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the audit monitor.

        Args:
            config: Configuration dictionary for the monitor
        """
        self.config = config or {}

        # Get audit log path from config or use default
        self.audit_log_path = self.config.get("log_file", "logs/audit.log")

        # Initialize last read position
        self.last_position = 0

        # Initialize in-memory event cache
        self.event_cache = []
        self.event_cache_size = self.config.get("event_cache_size", 1000)

        # Initialize statistics
        self.event_counts = Counter()
        self.user_activity = defaultdict(Counter)
        self.fix_activity = defaultdict(list)
        self.error_events = []

        # Time tracking
        self.last_analysis_time = datetime.datetime.now()

        # Check if audit log exists
        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        """Ensure the audit log file exists."""
        log_path = Path(self.audit_log_path)
        if not log_path.exists():
            # Create parent directories if they don't exist
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Create empty file
            log_path.touch()
            logger.info(f"Created empty audit log file at {self.audit_log_path}")

    def _parse_log_line(self, line: str) -> Optional[AuditEvent]:
        """Parse an audit log line into an AuditEvent.

        Args:
            line: A line from the audit log

        Returns:
            Optional[AuditEvent]: Parsed event or None if parsing failed
        """
        try:
            # Extract JSON part (everything after the timestamp and log level)
            match = re.match(
                r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[[^\]]+\] (.+)", line
            )
            if not match:
                return None

            json_str = match.group(1)
            event_data = json.loads(json_str)

            return AuditEvent.from_json(event_data)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse audit log line: {e}")
            return None

    def read_new_events(self) -> List[AuditEvent]:
        """Read new events from the audit log file.

        Returns:
            List[AuditEvent]: New events
        """
        new_events = []

        try:
            with open(self.audit_log_path, "r") as f:
                # Seek to the last read position
                f.seek(self.last_position)

                # Read new lines
                for line in f:
                    event = self._parse_log_line(line)
                    if event:
                        new_events.append(event)

                        # Update event cache (FIFO)
                        self.event_cache.append(event)
                        if len(self.event_cache) > self.event_cache_size:
                            self.event_cache.pop(0)

                # Update last position
                self.last_position = f.tell()
        except Exception as e:
            logger.error(f"Error reading audit log: {e}")

        return new_events

    def update_statistics(self, events: List[AuditEvent]) -> None:
        """Update statistics with new events.

        Args:
            events: New events to process
        """
        for event in events:
            # Update event counts
            self.event_counts[event.event_type] += 1

            # Update user activity
            self.user_activity[event.user][event.event_type] += 1

            # Track fix-related events
            if "fix_" in event.event_type or event.event_type in (
                "deployment",
                "suggestion_deployed",
            ):
                fix_id = event.details.get("fix_id")
                if fix_id:
                    self.fix_activity[fix_id].append(event)

            # Track error events
            if event.severity in ("error", "critical") or event.status == "failure":
                self.error_events.append(event)

                # Keep only the most recent errors (limit to 100)
                if len(self.error_events) > 100:
                    self.error_events.pop(0)

    def check_for_anomalies(self) -> List[Dict[str, Any]]:
        """Check for anomalous audit events.

        Returns:
            List[Dict[str, Any]]: Detected anomalies
        """
        anomalies = []

        # Check time since last analysis
        now = datetime.datetime.now()
        time_window = (now - self.last_analysis_time).total_seconds() / 60  # in minutes

        # Calculate activity rates
        for user, events in self.user_activity.items():
            # Check for unusual activity spikes
            total_events = sum(events.values())
            events_per_minute = total_events / max(time_window, 1)

            # Threshold for anomaly detection
            threshold = self.config.get("activity_threshold", 10)  # events per minute

            if events_per_minute > threshold:
                anomalies.append(
                    {
                        "type": "high_activity",
                        "user": user,
                        "events_per_minute": events_per_minute,
                        "threshold": threshold,
                        "timestamp": now.isoformat(),
                    }
                )

        # Check for suspicious patterns in fix deployment
        for fix_id, events in self.fix_activity.items():
            # Check if there are rapid approval-deployment sequences
            approval_events = [e for e in events if e.event_type == "fix_approved"]
            deployment_events = [
                e for e in events if e.event_type in ("fix_deployed", "deployment")
            ]

            for approval in approval_events:
                for deployment in deployment_events:
                    # If deployment was less than 5 seconds after approval, flag it
                    if (
                        0 <
                        (deployment.timestamp - approval.timestamp).total_seconds() <
                        5
                    ):
                        anomalies.append(
                            {
                                "type": "rapid_approval_deployment",
                                "fix_id": fix_id,
                                "approver": approval.user,
                                "deployer": deployment.user,
                                "approval_time": approval.timestamp.isoformat(),
                                "deployment_time": deployment.timestamp.isoformat(),
                                "time_difference_seconds": (
                                    deployment.timestamp - approval.timestamp
                                ).total_seconds(),
                            }
                        )

        # Check for failed deployments
        failed_deployments = [
            e
            for e in self.error_events
            if e.event_type == "deployment" and e.status == "failure"
        ]

        if failed_deployments:
            anomalies.append(
                {
                    "type": "failed_deployments",
                    "count": len(failed_deployments),
                    "events": [
                        {
                            "fix_id": e.details.get("fix_id", "unknown"),
                            "user": e.user,
                            "timestamp": e.timestamp.isoformat(),
                            "details": e.details,
                        }
                        for e in failed_deployments
                    ],
                }
            )

        # Update last analysis time
        self.last_analysis_time = now

        return anomalies

    def generate_activity_report(self, time_period: str = "day") -> Dict[str, Any]:
        """Generate a report of healing activities.

        Args:
            time_period: Time period for the report ('hour', 'day', 'week')

        Returns:
            Dict[str, Any]: Activity report
        """
        # Determine cutoff time
        now = datetime.datetime.now()
        cutoff_map = {
            "hour": now - datetime.timedelta(hours=1),
            "day": now - datetime.timedelta(days=1),
            "week": now - datetime.timedelta(weeks=1),
            "month": now - datetime.timedelta(days=30),
        }
        cutoff_time = cutoff_map.get(time_period, cutoff_map["day"])

        # Filter events by time
        recent_events = [e for e in self.event_cache if e.timestamp >= cutoff_time]

        # Count events by type
        event_counts = Counter()
        for event in recent_events:
            event_counts[event.event_type] += 1

        # Count activities by user
        user_activity = defaultdict(Counter)
        for event in recent_events:
            user_activity[event.user][event.event_type] += 1

        # Count healing activities
        healing_activities = {
            "errors_detected": sum(
                1 for e in recent_events if e.event_type == "error_detected"
            ),
            "fixes_generated": sum(
                1 for e in recent_events if e.event_type == "fix_generated"
            ),
            "fixes_deployed": sum(
                1
                for e in recent_events
                if e.event_type in ("fix_deployed", "deployment")
            ),
            "fixes_approved": sum(
                1 for e in recent_events if e.event_type == "fix_approved"
            ),
            "fixes_rejected": sum(
                1 for e in recent_events if e.event_type == "fix_rejected"
            ),
            "success_rate": self._calculate_success_rate(recent_events),
        }

        # Generate summary
        return {
            "time_period": time_period,
            "start_time": cutoff_time.isoformat(),
            "end_time": now.isoformat(),
            "total_events": len(recent_events),
            "event_counts": dict(event_counts),
            "user_activity": {
                user: dict(activities) for user, activities in user_activity.items()
            },
            "healing_activities": healing_activities,
            "error_events": [
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type,
                    "user": e.user,
                    "details": e.details,
                }
                for e in recent_events
                if e.severity in ("error", "critical") or e.status == "failure"
            ],
        }

    def _calculate_success_rate(self, events: List[AuditEvent]) -> float:
        """Calculate healing activity success rate.

        Args:
            events: List of events to analyze

        Returns:
            float: Success rate (0.0 - 1.0)
        """
        deployment_events = [
            e for e in events if e.event_type in ("fix_deployed", "deployment")
        ]

        if not deployment_events:
            return 0.0

        successful = sum(1 for e in deployment_events if e.status == "success")
        return successful / len(deployment_events)

    def export_audit_data(self, output_path: str, format: str = "json") -> bool:
        """Export audit data to a file.

        Args:
            output_path: Path to write the export
            format: Format to export ('json' or 'csv')

        Returns:
            bool: True if export was successful
        """
        try:
            if format == "json":
                with open(output_path, "w") as f:
                    json.dump(
                        [self._event_to_dict(e) for e in self.event_cache], f, indent=2
                    )
            elif format == "csv":
                with open(output_path, "w") as f:
                    # Write header
                    f.write(
                        "event_id,timestamp,event_type,status,severity,hostname,user,source_ip\n"
                    )

                    # Write data
                    for event in self.event_cache:
                        f.write(
                            f"{event.event_id},{event.timestamp.isoformat()},{event.event_type},"
                            f"{event.status},{event.severity},{event.hostname},{event.user},"
                            f"{event.source_ip or ''}\n"
                        )
            else:
                logger.error(f"Unsupported export format: {format}")
                return False

            logger.info(f"Exported audit data to {output_path} in {format} format")
            return True

        except Exception as e:
            logger.error(f"Error exporting audit data: {e}")
            return False

    def _event_to_dict(self, event: AuditEvent) -> Dict[str, Any]:
        """Convert an AuditEvent to a dictionary for serialization.

        Args:
            event: The event to convert

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "status": event.status,
            "severity": event.severity,
            "hostname": event.hostname,
            "user": event.user,
            "details": event.details,
            "source_ip": event.source_ip,
        }

    def monitor(self, interval: int = 60) -> None:
        """Continuously monitor audit logs.

        Args:
            interval: Monitoring interval in seconds
        """
        logger.info(f"Starting audit monitoring with interval {interval} seconds")

        try:
            while True:
                # Read new events
                new_events = self.read_new_events()

                if new_events:
                    logger.info(f"Read {len(new_events)} new audit events")

                    # Update statistics
                    self.update_statistics(new_events)

                    # Check for anomalies
                    anomalies = self.check_for_anomalies()
                    if anomalies:
                        logger.warning(
                            f"Detected {len(anomalies)} anomalies in audit logs"
                        )

                        # Log anomalies to audit log
                        for anomaly in anomalies:
                            log_event(
                                event_type="audit_anomaly_detected",
                                details={
                                    "anomaly_type": anomaly["type"],
                                    "anomaly_details": anomaly,
                                },
                                severity="warning",
                            )

                # Sleep for the specified interval
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Audit monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in audit monitoring: {e}")


# Singleton instance for app-wide use
_audit_monitor = None


def get_audit_monitor(config: Dict[str, Any] = None) -> AuditMonitor:
    """Get or create the singleton AuditMonitor instance.

    Args:
        config: Optional configuration for the monitor

    Returns:
        AuditMonitor: The audit monitor instance
    """
    global _audit_monitor
    if _audit_monitor is None:
        _audit_monitor = AuditMonitor(config)
    return _audit_monitor


def generate_activity_report(time_period: str = "day") -> Dict[str, Any]:
    """Generate a report of healing activities.

    Args:
        time_period: Time period for the report

    Returns:
        Dict[str, Any]: Activity report
    """
    # Ensure monitor has updated data
    monitor = get_audit_monitor()
    monitor.read_new_events()

    return monitor.generate_activity_report(time_period)


def check_for_anomalies() -> List[Dict[str, Any]]:
    """Check for anomalous audit events.

    Returns:
        List[Dict[str, Any]]: Detected anomalies
    """
    # Ensure monitor has updated data
    monitor = get_audit_monitor()
    monitor.read_new_events()

    return monitor.check_for_anomalies()


def export_audit_data(output_path: str, format: str = "json") -> bool:
    """Export audit data to a file.

    Args:
        output_path: Path to write the export
        format: Format to export ('json' or 'csv')

    Returns:
        bool: True if export was successful
    """
    monitor = get_audit_monitor()
    return monitor.export_audit_data(output_path, format)
