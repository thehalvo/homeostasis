"""
Human-in-the-Loop Feedback System for Homeostasis.

This module provides advanced human interaction capabilities including
configurable review gates, intelligent notifications, escalation paths,
and comprehensive feedback collection for continuous improvement.
"""

import json
import logging
import smtplib
import uuid
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from modules.monitoring.logger import MonitoringLogger
from modules.security.approval import ApprovalManager, ApprovalStatus, ApprovalType
from modules.security.audit import get_audit_logger, log_event

logger = logging.getLogger(__name__)


class ReviewGateType(Enum):
    """Types of review gates."""

    AUTOMATIC = "automatic"  # Automatic approval based on rules
    MANUAL = "manual"  # Requires human review
    CONDITIONAL = "conditional"  # Conditional based on context
    ESCALATED = "escalated"  # Escalated to higher authority


class NotificationChannel(Enum):
    """Available notification channels."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    SMS = "sms"


class FeedbackType(Enum):
    """Types of feedback."""

    APPROVAL = "approval"
    REJECTION = "rejection"
    MODIFICATION = "modification"
    ESCALATION = "escalation"
    TIMEOUT = "timeout"


class ReviewGate:
    """Represents a configurable review gate."""

    def __init__(
        self,
        gate_id: str,
        name: str,
        gate_type: ReviewGateType,
        conditions: Dict[str, Any],
        reviewers: List[str],
        timeout_minutes: int = 60,
        escalation_path: List[str] = None,
    ):
        """
        Initialize a review gate.

        Args:
            gate_id: Unique identifier for the gate
            name: Human-readable name
            gate_type: Type of review gate
            conditions: Conditions that trigger this gate
            reviewers: List of reviewer usernames/groups
            timeout_minutes: Timeout before escalation
            escalation_path: Escalation hierarchy
        """
        self.gate_id = gate_id
        self.name = name
        self.gate_type = gate_type
        self.conditions = conditions
        self.reviewers = reviewers
        self.timeout_minutes = timeout_minutes
        self.escalation_path = escalation_path or []

    def matches_context(self, context: Dict[str, Any]) -> bool:
        """
        Check if this gate matches the given context.

        Args:
            context: Context to evaluate

        Returns:
            bool: True if gate should be triggered
        """
        for key, expected_value in self.conditions.items():
            if key not in context:
                return False

            actual_value = context[key]

            # Handle different comparison types
            if isinstance(expected_value, dict):
                # Complex condition (e.g., {"operator": ">=", "value": 0.8})
                operator = expected_value.get("operator", "==")
                value = expected_value.get("value")

                if operator == "==":
                    if actual_value != value:
                        return False
                elif operator == "!=":
                    if actual_value == value:
                        return False
                elif operator == ">=":
                    if actual_value < value:
                        return False
                elif operator == "<=":
                    if actual_value > value:
                        return False
                elif operator == ">":
                    if actual_value <= value:
                        return False
                elif operator == "<":
                    if actual_value >= value:
                        return False
                elif operator == "in":
                    if actual_value not in value:
                        return False
                elif operator == "contains":
                    if value not in actual_value:
                        return False
            else:
                # Simple equality check
                if actual_value != expected_value:
                    return False

        return True


class NotificationManager:
    """Manages notifications across different channels."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the notification manager.

        Args:
            config: Notification configuration
        """
        self.config = config or {}
        self.monitoring_logger = MonitoringLogger("notification_manager")

        # Channel configurations
        self.email_config = self.config.get("email", {})
        self.slack_config = self.config.get("slack", {})
        self.webhook_config = self.config.get("webhook", {})
        self.sms_config = self.config.get("sms", {})

    def send_notification(
        self,
        channel: NotificationChannel,
        recipients: List[str],
        subject: str,
        message: str,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        Send a notification through the specified channel.

        Args:
            channel: Notification channel
            recipients: List of recipients
            subject: Notification subject
            message: Notification message
            metadata: Additional metadata

        Returns:
            bool: True if notification sent successfully
        """
        try:
            if channel == NotificationChannel.EMAIL:
                return self._send_email(recipients, subject, message, metadata)
            elif channel == NotificationChannel.SLACK:
                return self._send_slack(recipients, subject, message, metadata)
            elif channel == NotificationChannel.WEBHOOK:
                return self._send_webhook(recipients, subject, message, metadata)
            elif channel == NotificationChannel.DASHBOARD:
                return self._send_dashboard(recipients, subject, message, metadata)
            elif channel == NotificationChannel.SMS:
                return self._send_sms(recipients, subject, message, metadata)
            else:
                self.monitoring_logger.warning(
                    f"Unsupported notification channel: {channel}"
                )
                return False

        except Exception as e:
            self.monitoring_logger.error(
                f"Failed to send {channel.value} notification: {e}"
            )
            return False

    def _send_email(
        self,
        recipients: List[str],
        subject: str,
        message: str,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Send email notification."""
        if not self.email_config.get("enabled", False):
            return False

        smtp_server = self.email_config.get("smtp_server")
        smtp_port = self.email_config.get("smtp_port", 587)
        username = self.email_config.get("username")
        password = self.email_config.get("password")
        from_email = self.email_config.get("from_email", username)

        if not all([smtp_server, username, password]):
            self.monitoring_logger.warning("Email configuration incomplete")
            return False

        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = from_email
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject

            # Add metadata to message if provided
            if metadata:
                message += "\n\nAdditional Information:\n"
                for key, value in metadata.items():
                    message += f"- {key}: {value}\n"

            msg.attach(MIMEText(message, "plain"))

            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()

            self.monitoring_logger.info(f"Email sent to {len(recipients)} recipients")
            return True

        except Exception as e:
            self.monitoring_logger.error(f"Failed to send email: {e}")
            return False

    def _send_slack(
        self,
        recipients: List[str],
        subject: str,
        message: str,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Send Slack notification."""
        if not self.slack_config.get("enabled", False):
            return False

        webhook_url = self.slack_config.get("webhook_url")
        if not webhook_url:
            self.monitoring_logger.warning("Slack webhook URL not configured")
            return False

        try:
            # Format message for Slack
            slack_message = {
                "text": subject,
                "attachments": [
                    {
                        "color": "warning",
                        "fields": [
                            {"title": "Message", "value": message, "short": False}
                        ],
                    }
                ],
            }

            # Add metadata fields if provided
            if metadata:
                for key, value in metadata.items():
                    slack_message["attachments"][0]["fields"].append(
                        {
                            "title": key.replace("_", " ").title(),
                            "value": str(value),
                            "short": True,
                        }
                    )

            # Send to each recipient (channel/user)
            success_count = 0
            for recipient in recipients:
                recipient_message = slack_message.copy()
                recipient_message["channel"] = recipient

                response = requests.post(webhook_url, json=recipient_message, timeout=10)
                if response.status_code == 200:
                    success_count += 1

            self.monitoring_logger.info(
                f"Slack notification sent to {success_count}/{len(recipients)} recipients"
            )
            return success_count > 0

        except Exception as e:
            self.monitoring_logger.error(f"Failed to send Slack notification: {e}")
            return False

    def _send_webhook(
        self,
        recipients: List[str],
        subject: str,
        message: str,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Send webhook notification."""
        if not self.webhook_config.get("enabled", False):
            return False

        try:
            payload = {
                "subject": subject,
                "message": message,
                "recipients": recipients,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
            }

            # Send to configured webhook URLs
            webhook_urls = self.webhook_config.get("urls", [])
            success_count = 0

            for url in webhook_urls:
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    success_count += 1

            self.monitoring_logger.info(
                f"Webhook notification sent to {success_count}/{len(webhook_urls)} endpoints"
            )
            return success_count > 0

        except Exception as e:
            self.monitoring_logger.error(f"Failed to send webhook notification: {e}")
            return False

    def _send_dashboard(
        self,
        recipients: List[str],
        subject: str,
        message: str,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Send dashboard notification (store for dashboard display)."""
        try:
            # Store notification for dashboard display
            notification_data = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "recipients": recipients,
                "subject": subject,
                "message": message,
                "metadata": metadata or {},
                "read": False,
            }

            # Store in notifications file
            notifications_file = Path("logs/dashboard_notifications.json")
            notifications_file.parent.mkdir(exist_ok=True)

            notifications = []
            if notifications_file.exists():
                with open(notifications_file, "r") as f:
                    notifications = json.load(f)

            notifications.append(notification_data)

            # Keep only last 100 notifications
            notifications = notifications[-100:]

            with open(notifications_file, "w") as f:
                json.dump(notifications, f, indent=2)

            self.monitoring_logger.info(
                f"Dashboard notification stored for {len(recipients)} recipients"
            )
            return True

        except Exception as e:
            self.monitoring_logger.error(f"Failed to store dashboard notification: {e}")
            return False

    def _send_sms(
        self,
        recipients: List[str],
        subject: str,
        message: str,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Send SMS notification (placeholder - requires SMS service integration)."""
        # This would integrate with services like Twilio, AWS SNS, etc.
        self.monitoring_logger.info(
            f"SMS notification would be sent to {len(recipients)} recipients"
        )
        return True


class HumanInLoopFeedbackSystem:
    """
    Advanced human-in-the-loop feedback system with configurable review gates,
    intelligent notifications, and comprehensive feedback collection.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the human-in-the-loop feedback system.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.monitoring_logger = MonitoringLogger("human_in_loop_feedback")
        self.audit_logger = get_audit_logger()

        # Initialize notification manager
        notification_config = self.config.get("notifications", {})
        self.notification_manager = NotificationManager(notification_config)

        # Initialize approval manager
        approval_config = self.config.get("approval", {})
        self.approval_manager = ApprovalManager(approval_config)

        # Load review gates
        self.review_gates = {}
        self._load_review_gates()

        # Feedback storage
        self.feedback_storage = Path(
            self.config.get("feedback_storage", "logs/feedback")
        )
        self.feedback_storage.mkdir(parents=True, exist_ok=True)

        # Active reviews tracking
        self.active_reviews = {}

    def _load_review_gates(self) -> None:
        """Load review gates from configuration."""
        gates_config = self.config.get("review_gates", {})

        # Default review gates
        default_gates = {
            "critical_system_fix": ReviewGate(
                gate_id="critical_system_fix",
                name="Critical System Fix Review",
                gate_type=ReviewGateType.MANUAL,
                conditions={
                    "environment": "production",
                    "severity": {"operator": ">=", "value": "high"},
                    "fix_type": {
                        "operator": "in",
                        "value": ["security", "database_schema", "authentication"],
                    },
                },
                reviewers=["security_team", "senior_engineers"],
                timeout_minutes=120,
                escalation_path=["tech_lead", "cto"],
            ),
            "high_confidence_fix": ReviewGate(
                gate_id="high_confidence_fix",
                name="High Confidence Fix Review",
                gate_type=ReviewGateType.AUTOMATIC,
                conditions={
                    "confidence": {"operator": ">=", "value": 0.9},
                    "test_coverage": {"operator": ">=", "value": 0.8},
                    "environment": {"operator": "!=", "value": "production"},
                },
                reviewers=[],
                timeout_minutes=0,
            ),
            "low_confidence_fix": ReviewGate(
                gate_id="low_confidence_fix",
                name="Low Confidence Fix Review",
                gate_type=ReviewGateType.MANUAL,
                conditions={"confidence": {"operator": "<", "value": 0.7}},
                reviewers=["engineers", "qa_team"],
                timeout_minutes=60,
                escalation_path=["tech_lead"],
            ),
            "large_scope_fix": ReviewGate(
                gate_id="large_scope_fix",
                name="Large Scope Fix Review",
                gate_type=ReviewGateType.ESCALATED,
                conditions={
                    "files_modified": {"operator": ">", "value": 5},
                    "lines_changed": {"operator": ">", "value": 100},
                },
                reviewers=["senior_engineers"],
                timeout_minutes=180,
                escalation_path=["tech_lead", "engineering_manager"],
            ),
        }

        # Add default gates
        for gate_id, gate in default_gates.items():
            self.review_gates[gate_id] = gate

        # Load custom gates from config
        for gate_id, gate_config in gates_config.items():
            gate = ReviewGate(
                gate_id=gate_id,
                name=gate_config.get("name", gate_id),
                gate_type=ReviewGateType(gate_config.get("gate_type", "manual")),
                conditions=gate_config.get("conditions", {}),
                reviewers=gate_config.get("reviewers", []),
                timeout_minutes=gate_config.get("timeout_minutes", 60),
                escalation_path=gate_config.get("escalation_path", []),
            )
            self.review_gates[gate_id] = gate

    def evaluate_review_gates(self, context: Dict[str, Any]) -> List[ReviewGate]:
        """
        Evaluate which review gates should be triggered for the given context.

        Args:
            context: Context information about the fix/change

        Returns:
            List[ReviewGate]: List of triggered review gates
        """
        triggered_gates = []

        for gate in self.review_gates.values():
            if gate.matches_context(context):
                triggered_gates.append(gate)
                self.monitoring_logger.info(
                    f"Review gate triggered: {gate.name}",
                    gate_id=gate.gate_id,
                    gate_type=gate.gate_type.value,
                )

        # Sort by gate type priority (automatic first, then manual, conditional, escalated)
        priority_order = {
            ReviewGateType.AUTOMATIC: 1,
            ReviewGateType.MANUAL: 2,
            ReviewGateType.CONDITIONAL: 3,
            ReviewGateType.ESCALATED: 4,
        }

        triggered_gates.sort(key=lambda g: priority_order.get(g.gate_type, 999))

        return triggered_gates

    def create_review_request(
        self,
        session_id: str,
        fix_id: str,
        context: Dict[str, Any],
        requestor: str = "system",
    ) -> Optional[str]:
        """
        Create a review request based on triggered gates.

        Args:
            session_id: Healing session ID
            fix_id: Fix ID
            context: Context information
            requestor: User/system requesting the review

        Returns:
            Optional[str]: Review request ID if created, None if no review needed
        """
        # Evaluate review gates
        triggered_gates = self.evaluate_review_gates(context)

        if not triggered_gates:
            self.monitoring_logger.info("No review gates triggered, auto-approving")
            return None

        # Check for automatic approval
        if triggered_gates[0].gate_type == ReviewGateType.AUTOMATIC:
            self.monitoring_logger.info("Automatic approval gate triggered")
            self._log_feedback(
                fix_id,
                FeedbackType.APPROVAL,
                "system",
                "Automatically approved based on review gate conditions",
            )
            return None

        # Create approval request for manual/conditional/escalated gates
        primary_gate = triggered_gates[0]

        # Generate comprehensive title and description
        title = f"Review Required: {context.get('fix_type', 'Unknown')} fix for {context.get('file_path', 'unknown file')}"

        description = self._generate_review_description(context, triggered_gates)

        # Determine timeout
        timeout_seconds = primary_gate.timeout_minutes * 60

        # Create approval request
        approval_request = self.approval_manager.create_request(
            request_type=ApprovalType.FIX_DEPLOYMENT,
            requester=requestor,
            title=title,
            description=description,
            data={
                "session_id": session_id,
                "fix_id": fix_id,
                "context": context,
                "triggered_gates": [g.gate_id for g in triggered_gates],
                "primary_gate": primary_gate.gate_id,
            },
            required_approvers=1,  # Can be made configurable
            expiry=timeout_seconds,
        )

        request_id = approval_request.request_id

        # Track active review
        self.active_reviews[request_id] = {
            "session_id": session_id,
            "fix_id": fix_id,
            "primary_gate": primary_gate,
            "triggered_gates": triggered_gates,
            "created_at": datetime.utcnow(),
            "context": context,
            "notifications_sent": [],
        }

        # Send notifications
        self._send_review_notifications(request_id, primary_gate, context)

        # Schedule escalation if needed
        if primary_gate.escalation_path:
            self._schedule_escalation(request_id, primary_gate)

        self.monitoring_logger.info(
            "Review request created",
            request_id=request_id,
            fix_id=fix_id,
            primary_gate=primary_gate.gate_id,
        )

        return request_id

    def _generate_review_description(
        self, context: Dict[str, Any], gates: List[ReviewGate]
    ) -> str:
        """Generate a comprehensive review description."""
        description = "A healing action requires human review.\n\n"

        # Add context information
        description += "**Context Information:**\n"
        for key, value in context.items():
            description += f"- {key.replace('_', ' ').title()}: {value}\n"

        description += "\n**Triggered Review Gates:**\n"
        for gate in gates:
            description += f"- {gate.name} ({gate.gate_type.value})\n"

        # Add risk assessment
        risk_level = self._assess_risk_level(context)
        description += f"\n**Risk Assessment:** {risk_level}\n"

        # Add recommendations
        recommendations = self._generate_recommendations(context, gates)
        if recommendations:
            description += f"\n**Recommendations:**\n{recommendations}\n"

        return description

    def _assess_risk_level(self, context: Dict[str, Any]) -> str:
        """Assess the risk level of the change."""
        risk_score = 0

        # Environment risk
        if context.get("environment") == "production":
            risk_score += 3
        elif context.get("environment") == "staging":
            risk_score += 1

        # Confidence risk
        confidence = context.get("confidence", 1.0)
        if confidence < 0.5:
            risk_score += 3
        elif confidence < 0.7:
            risk_score += 2
        elif confidence < 0.9:
            risk_score += 1

        # Scope risk
        files_modified = context.get("files_modified", 1)
        if files_modified > 10:
            risk_score += 3
        elif files_modified > 5:
            risk_score += 2
        elif files_modified > 1:
            risk_score += 1

        # Fix type risk
        critical_types = ["security", "authentication", "database_schema"]
        if context.get("fix_type") in critical_types:
            risk_score += 2

        # Determine risk level
        if risk_score >= 7:
            return "Very High"
        elif risk_score >= 5:
            return "High"
        elif risk_score >= 3:
            return "Medium"
        elif risk_score >= 1:
            return "Low"
        else:
            return "Very Low"

    def _generate_recommendations(
        self, context: Dict[str, Any], gates: List[ReviewGate]
    ) -> str:
        """Generate recommendations for the reviewer."""
        recommendations = []

        # Confidence-based recommendations
        confidence = context.get("confidence", 1.0)
        if confidence < 0.7:
            recommendations.append("- Consider additional testing before approval")
            recommendations.append("- Review the fix logic carefully")

        # Environment-based recommendations
        if context.get("environment") == "production":
            recommendations.append("- Ensure rollback plan is in place")
            recommendations.append("- Consider gradual deployment if possible")

        # Scope-based recommendations
        files_modified = context.get("files_modified", 1)
        if files_modified > 5:
            recommendations.append("- Review all modified files for consistency")
            recommendations.append("- Ensure comprehensive test coverage")

        return "\n".join(recommendations) if recommendations else ""

    def _send_review_notifications(
        self, request_id: str, gate: ReviewGate, context: Dict[str, Any]
    ) -> None:
        """Send notifications to reviewers."""
        subject = f"Review Required: {context.get('fix_type', 'Fix')} - {gate.name}"

        message = f"""
A healing action requires your review.

Request ID: {request_id}
Gate: {gate.name}
Fix Type: {context.get('fix_type', 'Unknown')}
Environment: {context.get('environment', 'Unknown')}
Confidence: {context.get('confidence', 'Unknown')}

Please review this request in the Homeostasis dashboard or CLI.

Command: homeostasis --approve {request_id}
"""

        metadata = {
            "request_id": request_id,
            "gate_id": gate.gate_id,
            "priority": (
                "high" if gate.gate_type == ReviewGateType.ESCALATED else "normal"
            ),
        }

        # Send notifications through configured channels
        notification_config = self.config.get("notifications", {})

        for channel_name, channel_config in notification_config.items():
            if not channel_config.get("enabled", False):
                continue

            try:
                channel = NotificationChannel(channel_name)
                recipients = gate.reviewers

                success = self.notification_manager.send_notification(
                    channel=channel,
                    recipients=recipients,
                    subject=subject,
                    message=message,
                    metadata=metadata,
                )

                if success:
                    self.active_reviews[request_id]["notifications_sent"].append(
                        {
                            "channel": channel_name,
                            "recipients": recipients,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

            except ValueError:
                self.monitoring_logger.warning(
                    f"Unknown notification channel: {channel_name}"
                )

    def _schedule_escalation(self, request_id: str, gate: ReviewGate) -> None:
        """Schedule escalation for the request."""
        # This could be implemented with a background task scheduler
        # For now, we'll just log the intention
        self.monitoring_logger.info(
            f"Escalation scheduled for request {request_id} in {gate.timeout_minutes} minutes",
            escalation_path=gate.escalation_path,
        )

        # Store escalation info for manual processing
        escalation_data = {
            "request_id": request_id,
            "escalate_at": (
                datetime.utcnow() + timedelta(minutes=gate.timeout_minutes)
            ).isoformat(),
            "escalation_path": gate.escalation_path,
        }

        escalation_file = self.feedback_storage / "escalations.json"
        escalations = []

        if escalation_file.exists():
            with open(escalation_file, "r") as f:
                escalations = json.load(f)

        escalations.append(escalation_data)

        with open(escalation_file, "w") as f:
            json.dump(escalations, f, indent=2)

    def process_feedback(
        self,
        request_id: str,
        feedback_type: FeedbackType,
        reviewer: str,
        details: Dict[str, Any] = None,
    ) -> bool:
        """
        Process feedback from a reviewer.

        Args:
            request_id: Review request ID
            feedback_type: Type of feedback
            reviewer: Username of the reviewer
            details: Additional feedback details

        Returns:
            bool: True if feedback processed successfully
        """
        if request_id not in self.active_reviews:
            self.monitoring_logger.warning(f"Review request {request_id} not found")
            return False

        review_data = self.active_reviews[request_id]
        fix_id = review_data["fix_id"]

        # Log the feedback
        self._log_feedback(fix_id, feedback_type, reviewer, details)

        # Update approval request based on feedback type
        try:
            if feedback_type == FeedbackType.APPROVAL:
                self.approval_manager.approve_request(
                    request_id, reviewer, details.get("comment") if details else None
                )
            elif feedback_type == FeedbackType.REJECTION:
                self.approval_manager.reject_request(
                    request_id, reviewer, details.get("reason") if details else None
                )
            elif feedback_type == FeedbackType.MODIFICATION:
                # Add comment with modification details
                self.approval_manager.add_comment(
                    request_id,
                    reviewer,
                    f"Modification requested: {details.get('changes', 'No details provided')}",
                )
            elif feedback_type == FeedbackType.ESCALATION:
                # Handle escalation
                self._handle_escalation(request_id, reviewer, details)

        except Exception as e:
            self.monitoring_logger.error(f"Failed to process feedback: {e}")
            return False

        # Remove from active reviews if final decision
        if feedback_type in [FeedbackType.APPROVAL, FeedbackType.REJECTION]:
            del self.active_reviews[request_id]

        self.monitoring_logger.info(
            f"Feedback processed: {feedback_type.value}",
            request_id=request_id,
            reviewer=reviewer,
            fix_id=fix_id,
        )

        return True

    def _log_feedback(
        self,
        fix_id: str,
        feedback_type: FeedbackType,
        reviewer: str,
        details: Any = None,
    ) -> None:
        """Log feedback for analysis and improvement."""
        feedback_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "fix_id": fix_id,
            "feedback_type": feedback_type.value,
            "reviewer": reviewer,
            "details": details,
        }

        # Store in feedback file
        feedback_file = self.feedback_storage / "feedback_log.jsonl"
        with open(feedback_file, "a") as f:
            f.write(json.dumps(feedback_record) + "\n")

        # Log to audit system
        log_event(
            event_type=f"feedback_{feedback_type.value}",
            user=reviewer,
            details={
                "fix_id": fix_id,
                "feedback_type": feedback_type.value,
                **(details if isinstance(details, dict) else {"details": details}),
            },
        )

    def _handle_escalation(
        self, request_id: str, reviewer: str, details: Dict[str, Any] = None
    ) -> None:
        """Handle escalation of a review request."""
        review_data = self.active_reviews[request_id]
        primary_gate = review_data["primary_gate"]

        if not primary_gate.escalation_path:
            self.monitoring_logger.warning(
                f"No escalation path defined for gate {primary_gate.gate_id}"
            )
            return

        # Update reviewers to escalation path
        escalated_reviewers = primary_gate.escalation_path

        # Send escalation notifications
        subject = f"ESCALATED: Review Required - {review_data['context'].get('fix_type', 'Fix')}"
        message = f"""
This review request has been escalated.

Original Request ID: {request_id}
Escalated by: {reviewer}
Reason: {details.get('reason', 'No reason provided') if details else 'Timeout or manual escalation'}

Please review this request urgently.
"""

        # Send to escalated reviewers
        for channel_name, channel_config in self.config.get(
            "notifications", {}
        ).items():
            if channel_config.get("enabled", False):
                try:
                    channel = NotificationChannel(channel_name)
                    self.notification_manager.send_notification(
                        channel=channel,
                        recipients=escalated_reviewers,
                        subject=subject,
                        message=message,
                        metadata={"escalated": True, "priority": "urgent"},
                    )
                except ValueError:
                    continue

    def get_pending_reviews(self, reviewer: str = None) -> List[Dict[str, Any]]:
        """
        Get list of pending reviews.

        Args:
            reviewer: Filter by specific reviewer

        Returns:
            List[Dict[str, Any]]: List of pending reviews
        """
        pending_reviews = []

        for request_id, review_data in self.active_reviews.items():
            primary_gate = review_data["primary_gate"]

            # Filter by reviewer if specified
            if reviewer and reviewer not in primary_gate.reviewers:
                continue

            approval_request = self.approval_manager.get_request(request_id)
            if approval_request and approval_request.status == ApprovalStatus.PENDING:
                pending_reviews.append(
                    {
                        "request_id": request_id,
                        "fix_id": review_data["fix_id"],
                        "session_id": review_data["session_id"],
                        "gate_name": primary_gate.name,
                        "gate_type": primary_gate.gate_type.value,
                        "context": review_data["context"],
                        "created_at": review_data["created_at"].isoformat(),
                        "title": approval_request.title,
                        "description": approval_request.description,
                    }
                )

        return pending_reviews

    def get_feedback_statistics(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Get feedback statistics for analysis.

        Args:
            days_back: Number of days to analyze

        Returns:
            Dict[str, Any]: Feedback statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        feedback_file = self.feedback_storage / "feedback_log.jsonl"

        if not feedback_file.exists():
            return {"total_feedback": 0}

        stats = {
            "total_feedback": 0,
            "by_type": {},
            "by_reviewer": {},
            "response_times": [],
            "approval_rate": 0.0,
        }

        approvals = 0
        rejections = 0

        try:
            with open(feedback_file, "r") as f:
                for line in f:
                    record = json.loads(line.strip())
                    timestamp = datetime.fromisoformat(record["timestamp"])

                    if timestamp < cutoff_date:
                        continue

                    stats["total_feedback"] += 1

                    # Count by type
                    feedback_type = record["feedback_type"]
                    stats["by_type"][feedback_type] = (
                        stats["by_type"].get(feedback_type, 0) + 1
                    )

                    # Count by reviewer
                    reviewer = record["reviewer"]
                    stats["by_reviewer"][reviewer] = (
                        stats["by_reviewer"].get(reviewer, 0) + 1
                    )

                    # Track approvals/rejections
                    if feedback_type == "approval":
                        approvals += 1
                    elif feedback_type == "rejection":
                        rejections += 1

            # Calculate approval rate
            total_decisions = approvals + rejections
            if total_decisions > 0:
                stats["approval_rate"] = approvals / total_decisions

        except Exception as e:
            self.monitoring_logger.error(f"Failed to analyze feedback statistics: {e}")

        return stats


# Singleton instance
_human_in_loop_system = None


def get_human_in_loop_system(
    config: Dict[str, Any] = None,
) -> HumanInLoopFeedbackSystem:
    """
    Get or create the singleton HumanInLoopFeedbackSystem instance.

    Args:
        config: Optional configuration

    Returns:
        HumanInLoopFeedbackSystem: The human-in-the-loop feedback system instance
    """
    global _human_in_loop_system
    if _human_in_loop_system is None:
        _human_in_loop_system = HumanInLoopFeedbackSystem(config)
    return _human_in_loop_system


# Convenience functions
def create_review_request(
    session_id: str, fix_id: str, context: Dict[str, Any], requestor: str = "system"
) -> Optional[str]:
    """Create a review request based on triggered gates."""
    return get_human_in_loop_system().create_review_request(
        session_id, fix_id, context, requestor
    )


def process_feedback(
    request_id: str,
    feedback_type: FeedbackType,
    reviewer: str,
    details: Dict[str, Any] = None,
) -> bool:
    """Process feedback from a reviewer."""
    return get_human_in_loop_system().process_feedback(
        request_id, feedback_type, reviewer, details
    )


def get_pending_reviews(reviewer: str = None) -> List[Dict[str, Any]]:
    """Get list of pending reviews."""
    return get_human_in_loop_system().get_pending_reviews(reviewer)
