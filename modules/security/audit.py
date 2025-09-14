"""
Audit logging module for Homeostasis.

Provides structured logging for security-relevant events to maintain
a comprehensive audit trail for compliance and forensics.
"""

import datetime
import json
import logging
import os
import socket
import uuid
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """Handles structured audit logging for security events."""

    def __init__(self, config: Optional[Dict[Any, Any]] = None):
        """Initialize the audit logger.

        Args:
            config: Configuration dictionary for audit logging settings
        """
        self.config = config or {}

        # Create audit logger
        self.audit_logger = self._setup_logger()

        # Get hostname for logging
        self.hostname = socket.gethostname()

    def _setup_logger(self) -> logging.Logger:
        """Set up the audit logger.

        Returns:
            logging.Logger: Configured logger
        """
        # Create logger
        audit_logger = logging.getLogger("homeostasis.audit")
        audit_logger.setLevel(logging.INFO)

        # Prevent propagation to root logger
        audit_logger.propagate = False

        # Get log file path
        log_file = self.config.get("log_file", "logs/audit.log")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Set up appropriate handler based on rotation setting
        rotation = self.config.get("rotation", "daily")
        retention = self.config.get("retention", 90)  # days

        handler: logging.Handler
        if rotation == "size":
            # Rotate based on size
            max_bytes = self.config.get("max_bytes", 10 * 1024 * 1024)  # 10 MB
            backup_count = self.config.get("backup_count", 10)

            handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
        else:
            # Rotate based on time
            when_map = {
                "hourly": "H",
                "daily": "midnight",
                "weekly": "W0",
                "monthly": "M",
            }

            handler = TimedRotatingFileHandler(
                log_file,
                when=when_map.get(rotation, "midnight"),
                interval=1,
                backupCount=retention,
            )

        # Set formatter for structured logging
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        audit_logger.addHandler(handler)

        return audit_logger

    def log_event(
        self,
        event_type: str,
        user: Optional[str] = None,
        details: Optional[Dict[Any, Any]] = None,
        status: str = "success",
        severity: str = "info",
    ) -> str:
        """Log an audit event.

        Args:
            event_type: Type of event (e.g., 'login', 'fix_deployed')
            user: Username or identifier of the user performing the action
            details: Additional event details
            status: Event status ('success', 'failure', 'attempt')
            severity: Event severity ('info', 'warning', 'error', 'critical')

        Returns:
            str: Event ID
        """
        # Generate event ID
        event_id = str(uuid.uuid4())

        # Build event data
        event_data = {
            "event_id": event_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "status": status,
            "severity": severity,
            "hostname": self.hostname,
            "user": user or "system",
            "details": details or {},
        }

        # Add source IP if available
        if details and "source_ip" in details:
            event_data["source_ip"] = details["source_ip"]

        # Log the event as JSON
        self.audit_logger.info(json.dumps(event_data))

        return event_id

    def log_login(
        self,
        username: str,
        status: str = "success",
        source_ip: Optional[str] = None,
        details: Optional[Dict[Any, Any]] = None,
    ) -> str:
        """Log a login event.

        Args:
            username: Username attempting to log in
            status: Login status ('success', 'failure', 'attempt')
            source_ip: Source IP address
            details: Additional login details

        Returns:
            str: Event ID
        """
        event_details = details or {}
        if source_ip:
            event_details["source_ip"] = source_ip

        severity = "info" if status == "success" else "warning"

        return self.log_event(
            event_type="login",
            user=username,
            details=event_details,
            status=status,
            severity=severity,
        )

    def log_logout(
        self,
        username: str,
        session_duration: Optional[int] = None,
        source_ip: Optional[str] = None,
    ) -> str:
        """Log a logout event.

        Args:
            username: Username logging out
            session_duration: Session duration in seconds
            source_ip: Source IP address

        Returns:
            str: Event ID
        """
        details: Dict[str, Any] = {}
        if session_duration is not None:
            details["session_duration"] = session_duration
        if source_ip:
            details["source_ip"] = source_ip

        return self.log_event(event_type="logout", user=username, details=details)

    def log_access(
        self,
        username: str,
        resource: str,
        action: str,
        status: str = "success",
        source_ip: Optional[str] = None,
        details: Optional[Dict[Any, Any]] = None,
    ) -> str:
        """Log a resource access event.

        Args:
            username: Username accessing the resource
            resource: Resource being accessed
            action: Action being performed
            status: Access status ('success', 'failure', 'attempt')
            source_ip: Source IP address
            details: Additional access details

        Returns:
            str: Event ID
        """
        event_details = details or {}
        event_details.update({"resource": resource, "action": action})

        if source_ip:
            event_details["source_ip"] = source_ip

        severity = "info" if status == "success" else "warning"

        return self.log_event(
            event_type="access",
            user=username,
            details=event_details,
            status=status,
            severity=severity,
        )

    def log_fix(
        self,
        fix_id: str,
        event_type: str,
        user: Optional[str] = None,
        status: str = "success",
        details: Optional[Dict[Any, Any]] = None,
    ) -> str:
        """Log a fix-related event.

        Args:
            fix_id: Identifier for the fix
            event_type: Type of fix event ('fix_generated', 'fix_approved', etc.)
            user: Username involved in the event
            status: Event status ('success', 'failure', 'attempt')
            details: Additional fix details

        Returns:
            str: Event ID
        """
        event_details = details or {}
        event_details["fix_id"] = fix_id

        return self.log_event(
            event_type=event_type,
            user=user or "system",
            details=event_details,
            status=status,
        )

    def log_deployment(
        self,
        fix_id: str,
        environment: str,
        user: Optional[str] = None,
        status: str = "success",
        details: Optional[Dict[Any, Any]] = None,
    ) -> str:
        """Log a deployment event.

        Args:
            fix_id: Identifier for the fix being deployed
            environment: Environment being deployed to
            user: Username performing the deployment
            status: Deployment status ('success', 'failure', 'attempt')
            details: Additional deployment details

        Returns:
            str: Event ID
        """
        event_details = details or {}
        event_details.update({"fix_id": fix_id, "environment": environment})

        severity = "info"
        if status == "failure":
            severity = "error"

        return self.log_event(
            event_type="deployment",
            user=user or "system",
            details=event_details,
            status=status,
            severity=severity,
        )

    def log_config_change(
        self,
        user: str,
        config_section: str,
        change_description: str,
        old_value: Any = None,
        new_value: Any = None,
    ) -> str:
        """Log a configuration change event.

        Args:
            user: Username making the change
            config_section: Section of configuration being changed
            change_description: Description of the change
            old_value: Previous configuration value
            new_value: New configuration value

        Returns:
            str: Event ID
        """
        details = {
            "config_section": config_section,
            "change_description": change_description,
        }

        # Only include values if they're serializable
        try:
            if old_value is not None:
                # Try to serialize to ensure it's JSON-compatible
                json.dumps({"value": old_value})
                details["old_value"] = old_value
        except (TypeError, OverflowError):
            details["old_value"] = str(old_value)

        try:
            if new_value is not None:
                # Try to serialize to ensure it's JSON-compatible
                json.dumps({"value": new_value})
                details["new_value"] = new_value
        except (TypeError, OverflowError):
            details["new_value"] = str(new_value)

        return self.log_event(event_type="config_change", user=user, details=details)

    def log_security_event(
        self,
        event_type: str,
        user: Optional[str] = None,
        source_ip: Optional[str] = None,
        severity: str = "warning",
        details: Optional[Dict[Any, Any]] = None,
    ) -> str:
        """Log a security-related event.

        Args:
            event_type: Type of security event
            user: Username associated with the event
            source_ip: Source IP address
            severity: Event severity ('info', 'warning', 'error', 'critical')
            details: Additional security event details

        Returns:
            str: Event ID
        """
        event_details = details or {}
        if source_ip:
            event_details["source_ip"] = source_ip

        return self.log_event(
            event_type=f"security_{event_type}",
            user=user or "unknown",
            details=event_details,
            status="detected",
            severity=severity,
        )


# Singleton instance for app-wide use
_audit_logger = None


def get_audit_logger(config: Optional[Dict[Any, Any]] = None) -> AuditLogger:
    """Get or create the singleton AuditLogger instance.

    Args:
        config: Optional configuration to initialize the logger with

    Returns:
        AuditLogger: The audit logger instance
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(config)
    return _audit_logger


def log_event(
    event_type: str,
    user: Optional[str] = None,
    details: Optional[Dict[Any, Any]] = None,
    status: str = "success",
    severity: str = "info",
) -> str:
    """Log an audit event.

    Args:
        event_type: Type of event
        user: Username or identifier
        details: Additional details
        status: Event status
        severity: Event severity

    Returns:
        str: Event ID
    """
    return get_audit_logger().log_event(
        event_type=event_type,
        user=user,
        details=details,
        status=status,
        severity=severity,
    )


def log_login(
    username: str,
    status: str = "success",
    source_ip: Optional[str] = None,
    details: Optional[Dict[Any, Any]] = None,
) -> str:
    """Log a login event.

    Args:
        username: Username
        status: Login status
        source_ip: Source IP
        details: Additional details

    Returns:
        str: Event ID
    """
    return get_audit_logger().log_login(
        username=username, status=status, source_ip=source_ip, details=details
    )


def log_fix(
    fix_id: str,
    event_type: str,
    user: Optional[str] = None,
    status: str = "success",
    details: Optional[Dict[Any, Any]] = None,
) -> str:
    """Log a fix-related event.

    Args:
        fix_id: Fix identifier
        event_type: Type of fix event
        user: Username
        status: Event status
        details: Additional details

    Returns:
        str: Event ID
    """
    return get_audit_logger().log_fix(
        fix_id=fix_id, event_type=event_type, user=user, status=status, details=details
    )
