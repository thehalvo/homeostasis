"""
Multi-Team Collaboration Module

Provides collaboration features for enterprise teams including notifications,
change approval workflows, team metrics, and integration with collaboration platforms.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Notification priority levels"""

    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    INFO = "info"


class NotificationChannel(Enum):
    """Notification channels"""

    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"


class ApprovalStatus(Enum):
    """Approval request status"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TeamRole(Enum):
    """Team roles for collaboration"""

    ADMIN = "admin"
    DEVELOPER = "developer"
    OPERATOR = "operator"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


@dataclass
class TeamMember:
    """Represents a team member"""

    member_id: str
    name: str
    email: str
    teams: List[str] = field(default_factory=list)
    roles: List[TeamRole] = field(default_factory=list)
    notification_preferences: Dict[str, Any] = field(default_factory=dict)
    active: bool = True

    def can_approve(self) -> bool:
        """Check if member can approve changes"""
        return TeamRole.ADMIN in self.roles or TeamRole.REVIEWER in self.roles


@dataclass
class Team:
    """Represents a team"""

    team_id: str
    name: str
    description: str
    members: List[str] = field(default_factory=list)  # Member IDs
    notification_channels: Dict[NotificationChannel, Dict[str, Any]] = field(
        default_factory=dict
    )
    on_call_rotation: List[str] = field(default_factory=list)  # Member IDs in rotation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Notification:
    """Represents a notification"""

    notification_id: str
    title: str
    message: str
    priority: NotificationPriority
    timestamp: datetime
    recipients: List[str] = field(default_factory=list)  # Team or member IDs
    channels: List[NotificationChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    healing_action_id: Optional[str] = None
    require_acknowledgment: bool = False
    acknowledged_by: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "notification_id": self.notification_id,
            "title": self.title,
            "message": self.message,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "recipients": self.recipients,
            "channels": [c.value for c in self.channels],
            "metadata": self.metadata,
            "healing_action_id": self.healing_action_id,
            "require_acknowledgment": self.require_acknowledgment,
            "acknowledged_by": self.acknowledged_by,
        }


@dataclass
class ApprovalRequest:
    """Represents an approval request"""

    request_id: str
    title: str
    description: str
    requested_by: str
    requested_at: datetime
    expires_at: datetime
    approvers: List[str] = field(default_factory=list)  # Required approvers
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: List[str] = field(default_factory=list)
    rejected_by: List[str] = field(default_factory=list)
    comments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    healing_action_id: Optional[str] = None
    minimum_approvals: int = 1

    def is_approved(self) -> bool:
        """Check if request is approved"""
        return len(self.approved_by) >= self.minimum_approvals

    def is_expired(self) -> bool:
        """Check if request is expired"""
        return datetime.utcnow() > self.expires_at


@dataclass
class TeamMetrics:
    """Team performance metrics"""

    team_id: str
    period_start: datetime
    period_end: datetime
    healing_actions_triggered: int = 0
    healing_actions_successful: int = 0
    healing_actions_failed: int = 0
    average_response_time: float = 0.0  # seconds
    average_resolution_time: float = 0.0  # seconds
    alerts_acknowledged: int = 0
    changes_approved: int = 0
    changes_rejected: int = 0
    on_call_hours: Dict[str, float] = field(default_factory=dict)  # member_id -> hours


class TeamCollaborationHub:
    """Central hub for team collaboration features"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._teams: Dict[str, Team] = {}
        self._members: Dict[str, TeamMember] = {}
        self._notifications: Dict[str, Notification] = {}
        self._approval_requests: Dict[str, ApprovalRequest] = {}
        self._metrics: Dict[str, List[TeamMetrics]] = defaultdict(list)
        self._notification_service = TeamNotificationService(config)
        self._approval_workflow = ChangeApprovalWorkflow(config)
        self._metrics_collector = TeamMetricsCollector(config)

    async def create_team(self, team: Team) -> bool:
        """Create a new team"""
        try:
            self._teams[team.team_id] = team
            logger.info(f"Created team: {team.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create team: {e}")
            return False

    async def add_team_member(self, member: TeamMember) -> bool:
        """Add a team member"""
        try:
            self._members[member.member_id] = member

            # Add member to their teams
            for team_id in member.teams:
                if team_id in self._teams:
                    if member.member_id not in self._teams[team_id].members:
                        self._teams[team_id].members.append(member.member_id)

            logger.info(f"Added team member: {member.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add team member: {e}")
            return False

    async def send_notification(self, notification: Notification) -> bool:
        """Send a notification to teams/members"""
        try:
            self._notifications[notification.notification_id] = notification

            # Expand recipients
            all_members = self._expand_recipients(notification.recipients)

            # Send through notification service
            success = await self._notification_service.send(
                notification, all_members, self._teams, self._members
            )

            if success:
                # Track metrics
                await self._metrics_collector.record_notification(notification)

            return success

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

    async def request_approval(self, request: ApprovalRequest) -> bool:
        """Create an approval request"""
        try:
            self._approval_requests[request.request_id] = request

            # Notify approvers
            notification = Notification(
                notification_id=f"approval_{request.request_id}",
                title=f"Approval Required: {request.title}",
                message=request.description,
                priority=NotificationPriority.HIGH,
                timestamp=datetime.utcnow(),
                recipients=request.approvers,
                channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
                metadata={"approval_request_id": request.request_id},
                require_acknowledgment=True,
            )

            await self.send_notification(notification)

            # Track metrics
            await self._metrics_collector.record_approval_request(request)

            return True

        except Exception as e:
            logger.error(f"Failed to create approval request: {e}")
            return False

    async def process_approval(
        self, request_id: str, approver_id: str, approved: bool, comment: str = ""
    ) -> bool:
        """Process an approval decision"""
        try:
            request = self._approval_requests.get(request_id)
            if not request:
                logger.error(f"Approval request not found: {request_id}")
                return False

            if request.status != ApprovalStatus.PENDING:
                logger.warning(f"Request {request_id} is not pending")
                return False

            # Check if approver is authorized
            if approver_id not in request.approvers:
                logger.error(
                    f"User {approver_id} not authorized to approve request {request_id}"
                )
                return False

            # Record decision
            if approved:
                request.approved_by.append(approver_id)
            else:
                request.rejected_by.append(approver_id)

            # Add comment
            if comment:
                request.comments.append(
                    {
                        "approver_id": approver_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "approved": approved,
                        "comment": comment,
                    }
                )

            # Check if request is now approved or rejected
            if request.is_approved():
                request.status = ApprovalStatus.APPROVED
                await self._notify_approval_status(request, True)

                # Execute approved action if configured
                if request.healing_action_id:
                    await self._approval_workflow.execute_approved_action(
                        request.healing_action_id
                    )

            elif len(request.rejected_by) > 0:
                request.status = ApprovalStatus.REJECTED
                await self._notify_approval_status(request, False)

            # Track metrics
            await self._metrics_collector.record_approval_decision(
                request, approver_id, approved
            )

            return True

        except Exception as e:
            logger.error(f"Failed to process approval: {e}")
            return False

    async def get_team_metrics(
        self, team_id: str, start_date: datetime, end_date: datetime
    ) -> Optional[TeamMetrics]:
        """Get team metrics for a period"""
        try:
            return await self._metrics_collector.get_team_metrics(
                team_id, start_date, end_date, self._teams, self._members
            )
        except Exception as e:
            logger.error(f"Failed to get team metrics: {e}")
            return None

    async def get_on_call_member(self, team_id: str) -> Optional[str]:
        """Get current on-call team member"""
        team = self._teams.get(team_id)
        if not team or not team.on_call_rotation:
            return None

        # Simple rotation based on current time
        # In production, this would be more sophisticated
        current_hour = datetime.utcnow().hour
        current_day = datetime.utcnow().day

        rotation_index = (current_day * 24 + current_hour) % len(team.on_call_rotation)
        return team.on_call_rotation[rotation_index]

    def _expand_recipients(self, recipients: List[str]) -> Set[str]:
        """Expand team IDs to member IDs"""
        all_members = set()

        for recipient in recipients:
            if recipient in self._teams:
                # It's a team ID
                all_members.update(self._teams[recipient].members)
            elif recipient in self._members:
                # It's a member ID
                all_members.add(recipient)

        return all_members

    async def _notify_approval_status(self, request: ApprovalRequest, approved: bool):
        """Notify about approval status"""
        status = "approved" if approved else "rejected"

        notification = Notification(
            notification_id=f"approval_status_{request.request_id}",
            title=f"Approval Request {status.capitalize()}: {request.title}",
            message=f"The approval request has been {status}.",
            priority=NotificationPriority.HIGH,
            timestamp=datetime.utcnow(),
            recipients=[request.requested_by] + request.approvers,
            channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
            metadata={"approval_request_id": request.request_id, "status": status},
        )

        await self.send_notification(notification)


class TeamNotificationService:
    """Service for sending notifications to teams"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._channel_adapters = {}
        self._initialize_adapters()

    def _initialize_adapters(self):
        """Initialize notification channel adapters"""
        if self.config.get("slack"):
            self._channel_adapters[NotificationChannel.SLACK] = SlackIntegration(
                self.config["slack"]
            )

        if self.config.get("teams"):
            self._channel_adapters[NotificationChannel.TEAMS] = (
                MicrosoftTeamsIntegration(self.config["teams"])
            )

        # Add more adapters as needed

    async def send(
        self,
        notification: Notification,
        recipients: Set[str],
        teams: Dict[str, Team],
        members: Dict[str, TeamMember],
    ) -> bool:
        """Send notification through configured channels"""
        success_count = 0

        for channel in notification.channels:
            if channel in self._channel_adapters:
                try:
                    # Get channel-specific configuration for teams
                    channel_configs = []

                    for recipient_id in recipients:
                        if recipient_id in members:
                            member = members[recipient_id]
                            # Check member's notification preferences
                            if channel.value in member.notification_preferences.get(
                                "channels", []
                            ):
                                channel_configs.append(member.notification_preferences)

                        # Also check team configurations
                        for team_id in members.get(
                            recipient_id, TeamMember("", "", "")
                        ).teams:
                            if (
                                team_id in teams
                                and channel in teams[team_id].notification_channels
                            ):
                                channel_configs.append(
                                    teams[team_id].notification_channels[channel]
                                )

                    # Send through adapter
                    adapter = self._channel_adapters[channel]
                    if await adapter.send_notification(notification, channel_configs):
                        success_count += 1

                except Exception as e:
                    logger.error(
                        f"Failed to send notification via {channel.value}: {e}"
                    )

        return success_count > 0


class ChangeApprovalWorkflow:
    """Workflow for change approvals"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.approval_timeout = config.get("approval_timeout", 3600)  # seconds
        self.auto_approve_low_risk = config.get("auto_approve_low_risk", False)
        self._pending_actions = {}

    async def create_approval_request(
        self, healing_action: Dict[str, Any], approvers: List[str]
    ) -> ApprovalRequest:
        """Create approval request for a healing action"""
        risk_level = self._assess_risk(healing_action)

        # Auto-approve low risk if configured
        if self.auto_approve_low_risk and risk_level == "low":
            return None

        request = ApprovalRequest(
            request_id=f"req_{healing_action['action_id']}",
            title=f"Healing Action: {healing_action.get('description', 'Unknown')}",
            description=self._format_action_description(healing_action),
            requested_by="homeostasis_system",
            requested_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=self.approval_timeout),
            approvers=approvers,
            metadata={
                "risk_level": risk_level,
                "action_type": healing_action.get("type"),
                "affected_resources": healing_action.get("affected_resources", []),
            },
            healing_action_id=healing_action["action_id"],
            minimum_approvals=self._determine_required_approvals(risk_level),
        )

        # Store pending action
        self._pending_actions[healing_action["action_id"]] = healing_action

        return request

    async def execute_approved_action(self, action_id: str) -> bool:
        """Execute an approved healing action"""
        action = self._pending_actions.get(action_id)
        if not action:
            logger.error(f"Action not found: {action_id}")
            return False

        try:
            # Here you would integrate with the actual healing system
            # For now, just log
            logger.info(f"Executing approved healing action: {action_id}")

            # Clean up
            del self._pending_actions[action_id]

            return True

        except Exception as e:
            logger.error(f"Failed to execute approved action: {e}")
            return False

    def _assess_risk(self, healing_action: Dict[str, Any]) -> str:
        """Assess risk level of a healing action"""
        # Simple risk assessment based on action type and scope
        action_type = healing_action.get("type", "")
        affected_count = len(healing_action.get("affected_resources", []))

        high_risk_actions = ["delete", "restart", "modify_config", "rollback"]
        medium_risk_actions = ["scale", "update", "patch"]

        if any(risk in action_type.lower() for risk in high_risk_actions):
            return "high"
        elif any(risk in action_type.lower() for risk in medium_risk_actions):
            return "medium"
        elif affected_count > 10:
            return "high"
        elif affected_count > 5:
            return "medium"
        else:
            return "low"

    def _determine_required_approvals(self, risk_level: str) -> int:
        """Determine number of required approvals based on risk"""
        risk_approvals = {"low": 1, "medium": 2, "high": 3}
        return risk_approvals.get(risk_level, 1)

    def _format_action_description(self, healing_action: Dict[str, Any]) -> str:
        """Format healing action description for approval request"""
        parts = [
            f"Action Type: {healing_action.get('type', 'Unknown')}",
            f"Description: {healing_action.get('description', 'No description')}",
            f"Affected Resources: {len(healing_action.get('affected_resources', []))}",
            f"Error Type: {healing_action.get('error_type', 'Unknown')}",
            f"Proposed Fix: {healing_action.get('proposed_fix', 'Not specified')}",
        ]

        return "\n".join(parts)


class TeamMetricsCollector:
    """Collector for team collaboration metrics"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._metrics_store = defaultdict(lambda: defaultdict(int))
        self._response_times = defaultdict(list)
        self._resolution_times = defaultdict(list)

    async def record_notification(self, notification: Notification):
        """Record notification metrics"""
        for recipient in notification.recipients:
            self._metrics_store[recipient]["notifications_sent"] += 1

            if notification.priority == NotificationPriority.URGENT:
                self._metrics_store[recipient]["urgent_notifications"] += 1

    async def record_approval_request(self, request: ApprovalRequest):
        """Record approval request metrics"""
        for approver in request.approvers:
            self._metrics_store[approver]["approval_requests_received"] += 1

    async def record_approval_decision(
        self, request: ApprovalRequest, approver_id: str, approved: bool
    ):
        """Record approval decision metrics"""
        if approved:
            self._metrics_store[approver_id]["approvals_granted"] += 1
        else:
            self._metrics_store[approver_id]["approvals_rejected"] += 1

        # Calculate response time
        response_time = (datetime.utcnow() - request.requested_at).total_seconds()
        self._response_times[approver_id].append(response_time)

    async def record_healing_action(self, team_id: str, action: Dict[str, Any]):
        """Record healing action metrics"""
        self._metrics_store[team_id]["healing_actions_triggered"] += 1

        if action.get("status") == "success":
            self._metrics_store[team_id]["healing_actions_successful"] += 1
        elif action.get("status") == "failed":
            self._metrics_store[team_id]["healing_actions_failed"] += 1

        # Record resolution time if available
        if action.get("start_time") and action.get("end_time"):
            resolution_time = (
                action["end_time"] - action["start_time"]
            ).total_seconds()
            self._resolution_times[team_id].append(resolution_time)

    async def get_team_metrics(
        self,
        team_id: str,
        start_date: datetime,
        end_date: datetime,
        teams: Dict[str, Team],
        members: Dict[str, TeamMember],
    ) -> TeamMetrics:
        """Get aggregated team metrics"""
        metrics = TeamMetrics(
            team_id=team_id, period_start=start_date, period_end=end_date
        )

        # Aggregate metrics for team members
        team = teams.get(team_id)
        if team:
            for member_id in team.members:
                member_metrics = self._metrics_store.get(member_id, {})

                metrics.healing_actions_triggered += member_metrics.get(
                    "healing_actions_triggered", 0
                )
                metrics.healing_actions_successful += member_metrics.get(
                    "healing_actions_successful", 0
                )
                metrics.healing_actions_failed += member_metrics.get(
                    "healing_actions_failed", 0
                )
                metrics.alerts_acknowledged += member_metrics.get(
                    "alerts_acknowledged", 0
                )
                metrics.changes_approved += member_metrics.get("approvals_granted", 0)
                metrics.changes_rejected += member_metrics.get("approvals_rejected", 0)

                # Calculate average response time
                if member_id in self._response_times:
                    member_response_times = self._response_times[member_id]
                    if member_response_times:
                        avg_response = sum(member_response_times) / len(
                            member_response_times
                        )
                        metrics.average_response_time = max(
                            metrics.average_response_time, avg_response
                        )

        # Calculate team-wide averages
        if team_id in self._resolution_times and self._resolution_times[team_id]:
            resolution_times = self._resolution_times[team_id]
            metrics.average_resolution_time = sum(resolution_times) / len(
                resolution_times
            )

        return metrics


class SlackIntegration:
    """Slack notification integration"""

    def __init__(self, config: Dict[str, Any]):
        self.webhook_url = config.get("webhook_url")
        self.bot_token = config.get("bot_token")
        self.default_channel = config.get("default_channel", "#alerts")

    async def send_notification(
        self, notification: Notification, channel_configs: List[Dict[str, Any]]
    ) -> bool:
        """Send notification to Slack"""
        try:
            # Build Slack message
            slack_message = self._build_slack_message(notification)

            # Send to configured channels
            for config in channel_configs:
                channel = config.get("slack_channel", self.default_channel)

                if self.webhook_url:
                    # Use webhook
                    response = requests.post(
                        self.webhook_url,
                        json={"channel": channel, **slack_message},
                        timeout=30,
                    )
                    response.raise_for_status()

                elif self.bot_token:
                    # Use Slack API
                    headers = {
                        "Authorization": f"Bearer {self.bot_token}",
                        "Content-Type": "application/json",
                    }

                    response = requests.post(
                        "https://slack.com/api/chat.postMessage",
                        headers=headers,
                        json={"channel": channel, **slack_message},
                        timeout=30,
                    )
                    response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    def _build_slack_message(self, notification: Notification) -> Dict[str, Any]:
        """Build Slack message format"""
        # Color based on priority
        color_map = {
            NotificationPriority.URGENT: "#FF0000",
            NotificationPriority.HIGH: "#FF9900",
            NotificationPriority.NORMAL: "#0099FF",
            NotificationPriority.LOW: "#00CC00",
            NotificationPriority.INFO: "#CCCCCC",
        }

        message = {
            "text": notification.title,
            "attachments": [
                {
                    "color": color_map.get(notification.priority, "#0099FF"),
                    "fields": [
                        {
                            "title": "Message",
                            "value": notification.message,
                            "short": False,
                        },
                        {
                            "title": "Priority",
                            "value": notification.priority.value,
                            "short": True,
                        },
                        {
                            "title": "Time",
                            "value": notification.timestamp.strftime(
                                "%Y-%m-%d %H:%M:%S UTC"
                            ),
                            "short": True,
                        },
                    ],
                    "footer": "Homeostasis Healing System",
                    "ts": int(notification.timestamp.timestamp()),
                }
            ],
        }

        # Add action buttons if approval required
        if notification.metadata.get("approval_request_id"):
            message["attachments"][0]["actions"] = [
                {
                    "name": "approve",
                    "text": "Approve",
                    "type": "button",
                    "value": notification.metadata["approval_request_id"],
                    "style": "primary",
                },
                {
                    "name": "reject",
                    "text": "Reject",
                    "type": "button",
                    "value": notification.metadata["approval_request_id"],
                    "style": "danger",
                },
            ]

        return message


class MicrosoftTeamsIntegration:
    """Microsoft Teams notification integration"""

    def __init__(self, config: Dict[str, Any]):
        self.webhook_url = config.get("webhook_url")
        self.tenant_id = config.get("tenant_id")
        self.client_id = config.get("client_id")
        self.client_secret = config.get("client_secret")

    async def send_notification(
        self, notification: Notification, channel_configs: List[Dict[str, Any]]
    ) -> bool:
        """Send notification to Microsoft Teams"""
        try:
            # Build Teams message card
            teams_card = self._build_teams_card(notification)

            # Send to webhook
            if self.webhook_url:
                response = requests.post(self.webhook_url, json=teams_card, timeout=30)
                response.raise_for_status()

            # For advanced scenarios, use Graph API
            # This would require OAuth token management

            return True

        except Exception as e:
            logger.error(f"Failed to send Teams notification: {e}")
            return False

    def _build_teams_card(self, notification: Notification) -> Dict[str, Any]:
        """Build Teams message card format"""
        # Theme color based on priority
        color_map = {
            NotificationPriority.URGENT: "FF0000",
            NotificationPriority.HIGH: "FF9900",
            NotificationPriority.NORMAL: "0099FF",
            NotificationPriority.LOW: "00CC00",
            NotificationPriority.INFO: "CCCCCC",
        }

        card = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "themeColor": color_map.get(notification.priority, "0099FF"),
            "summary": notification.title,
            "sections": [
                {
                    "activityTitle": notification.title,
                    "activitySubtitle": f"Priority: {notification.priority.value}",
                    "activityImage": "https://example.com/homeostasis-icon.png",
                    "facts": [
                        {
                            "name": "Time",
                            "value": notification.timestamp.strftime(
                                "%Y-%m-%d %H:%M:%S UTC"
                            ),
                        },
                        {"name": "System", "value": "Homeostasis Healing System"},
                    ],
                    "text": notification.message,
                }
            ],
        }

        # Add action buttons if approval required
        if notification.metadata.get("approval_request_id"):
            card["potentialAction"] = [
                {
                    "@type": "HttpPOST",
                    "name": "Approve",
                    "target": f"{self.webhook_url}/approve",
                    "body": json.dumps(
                        {
                            "request_id": notification.metadata["approval_request_id"],
                            "action": "approve",
                        }
                    ),
                },
                {
                    "@type": "HttpPOST",
                    "name": "Reject",
                    "target": f"{self.webhook_url}/reject",
                    "body": json.dumps(
                        {
                            "request_id": notification.metadata["approval_request_id"],
                            "action": "reject",
                        }
                    ),
                },
            ]

        return card
