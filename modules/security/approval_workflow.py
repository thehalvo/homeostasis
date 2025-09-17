"""
Enhanced Approval Workflow Engine for Homeostasis Enterprise Governance.

This module provides a comprehensive approval workflow system for managing
critical healing actions in mission-critical and regulated environments.

Features:
- Multi-stage approval workflows
- Conditional approval routing
- Escalation mechanisms
- SLA management
- Integration with external approval systems
"""

import datetime
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .approval import ApprovalManager
from .audit import get_audit_logger
from .rbac import get_rbac_manager
from .user_management import get_user_management

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""

    DRAFT = "draft"
    ACTIVE = "active"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXPIRED = "expired"


class ApprovalCondition(Enum):
    """Conditions for approval routing."""

    ALWAYS = "always"
    RISK_LEVEL = "risk_level"
    PATCH_SIZE = "patch_size"
    ERROR_TYPE = "error_type"
    ENVIRONMENT = "environment"
    TIME_BASED = "time_based"
    CUSTOM = "custom"


class EscalationTrigger(Enum):
    """Triggers for escalation."""

    SLA_BREACH = "sla_breach"
    REJECTION = "rejection"
    HIGH_RISK = "high_risk"
    MANUAL = "manual"


@dataclass
class WorkflowStage:
    """Represents a stage in an approval workflow."""

    stage_id: str
    name: str
    description: str
    approver_roles: List[str]
    approver_users: List[str] = field(default_factory=list)
    min_approvals: int = 1
    max_approvals: Optional[int] = None
    timeout_hours: int = 24
    condition: ApprovalCondition = ApprovalCondition.ALWAYS
    condition_params: Dict = field(default_factory=dict)
    next_stage: Optional[str] = None
    escalation_stage: Optional[str] = None
    auto_approve: bool = False
    notifications: Dict = field(default_factory=dict)


@dataclass
class WorkflowTemplate:
    """Template for approval workflows."""

    template_id: str
    name: str
    description: str
    category: str
    stages: List[WorkflowStage]
    initial_stage: str
    metadata: Dict = field(default_factory=dict)
    is_active: bool = True
    created_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )


@dataclass
class WorkflowInstance:
    """Instance of an executing workflow."""

    instance_id: str
    template_id: str
    request_id: str
    current_stage: str
    status: WorkflowStatus
    initiated_by: str
    initiated_at: str
    completed_at: Optional[str] = None
    stage_history: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    sla_deadline: Optional[str] = None


@dataclass
class ApprovalDecision:
    """Represents an approval decision."""

    decision_id: str
    workflow_instance_id: str
    stage_id: str
    approver_id: str
    decision: str  # 'approve', 'reject', 'delegate'
    comments: str
    decided_at: str
    delegate_to: Optional[str] = None
    conditions_met: Dict = field(default_factory=dict)


class ApprovalWorkflowEngine:
    """
    Advanced approval workflow engine for enterprise governance.

    Manages multi-stage approval workflows with conditional routing,
    escalation, and SLA management.
    """

    def __init__(self, config: Optional[Dict] = None, storage_path: Optional[str] = None):
        """Initialize the workflow engine.

        Args:
            config: Configuration dictionary
            storage_path: Path to store workflow data
        """
        self.config = config or {}
        self.storage_path = Path(storage_path or "data/workflows")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Get managers
        self.approval_manager = ApprovalManager(config)
        self.audit_logger = get_audit_logger(config)
        self.rbac_manager = get_rbac_manager(config)
        self.user_management = get_user_management(config)

        # Initialize stores
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.instances: Dict[str, WorkflowInstance] = {}
        self.decisions: Dict[str, List[ApprovalDecision]] = {}

        # Condition evaluators
        self.condition_evaluators: Dict[ApprovalCondition, Callable] = {
            ApprovalCondition.ALWAYS: self._evaluate_always,
            ApprovalCondition.RISK_LEVEL: self._evaluate_risk_level,
            ApprovalCondition.PATCH_SIZE: self._evaluate_patch_size,
            ApprovalCondition.ERROR_TYPE: self._evaluate_error_type,
            ApprovalCondition.ENVIRONMENT: self._evaluate_environment,
            ApprovalCondition.TIME_BASED: self._evaluate_time_based,
        }

        # Load existing data
        self._load_workflow_data()

        # Initialize default templates if none exist
        if not self.templates:
            self._create_default_templates()

    def create_workflow_template(
        self, name: str, description: str, category: str, stages: List[Dict]
    ) -> str:
        """Create a new workflow template.

        Args:
            name: Template name
            description: Template description
            category: Template category
            stages: List of stage configurations

        Returns:
            Template ID
        """
        template_id = self._generate_template_id()

        # Convert stage dicts to WorkflowStage objects
        workflow_stages = []
        for stage_config in stages:
            stage = WorkflowStage(
                stage_id=stage_config.get("stage_id", self._generate_stage_id()),
                name=stage_config["name"],
                description=stage_config.get("description", ""),
                approver_roles=stage_config.get("approver_roles", []),
                approver_users=stage_config.get("approver_users", []),
                min_approvals=stage_config.get("min_approvals", 1),
                max_approvals=stage_config.get("max_approvals"),
                timeout_hours=stage_config.get("timeout_hours", 24),
                condition=ApprovalCondition(stage_config.get("condition", "always")),
                condition_params=stage_config.get("condition_params", {}),
                next_stage=stage_config.get("next_stage"),
                escalation_stage=stage_config.get("escalation_stage"),
                auto_approve=stage_config.get("auto_approve", False),
                notifications=stage_config.get("notifications", {}),
            )
            workflow_stages.append(stage)

        # Validate stage references
        stage_ids = {stage.stage_id for stage in workflow_stages}
        for stage in workflow_stages:
            if stage.next_stage and stage.next_stage not in stage_ids:
                raise ValueError(f"Invalid next_stage reference: {stage.next_stage}")
            if stage.escalation_stage and stage.escalation_stage not in stage_ids:
                raise ValueError(
                    f"Invalid escalation_stage reference: {stage.escalation_stage}"
                )

        # Create template
        template = WorkflowTemplate(
            template_id=template_id,
            name=name,
            description=description,
            category=category,
            stages=workflow_stages,
            initial_stage=workflow_stages[0].stage_id if workflow_stages else "",
        )

        self.templates[template_id] = template
        self._save_workflow_data()

        # Log event
        self.audit_logger.log_event(
            event_type="workflow_template_created",
            user="system",
            details={
                "template_id": template_id,
                "name": name,
                "category": category,
                "stages": len(workflow_stages),
            },
        )

        return template_id

    def initiate_workflow(
        self,
        template_id: str,
        request_id: str,
        initiated_by: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Initiate a workflow instance.

        Args:
            template_id: Workflow template ID
            request_id: Associated approval request ID
            initiated_by: User ID initiating the workflow
            metadata: Additional metadata

        Returns:
            Workflow instance ID
        """
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Workflow template {template_id} not found")

        if not template.is_active:
            raise ValueError(f"Workflow template {template_id} is not active")

        # Get the approval request
        approval_request = self.approval_manager.get_request(request_id)
        if not approval_request:
            raise ValueError(f"Approval request {request_id} not found")

        # Create workflow instance
        instance_id = self._generate_instance_id()
        now = datetime.datetime.utcnow()

        # Calculate SLA deadline based on total timeout
        total_timeout = sum(stage.timeout_hours for stage in template.stages)
        sla_deadline = now + datetime.timedelta(hours=total_timeout)

        instance = WorkflowInstance(
            instance_id=instance_id,
            template_id=template_id,
            request_id=request_id,
            current_stage=template.initial_stage,
            status=WorkflowStatus.ACTIVE,
            initiated_by=initiated_by,
            initiated_at=now.isoformat(),
            metadata=metadata or {},
            sla_deadline=sla_deadline.isoformat(),
        )

        self.instances[instance_id] = instance
        self._save_workflow_data()

        # Start the first stage
        self._start_stage(instance_id, template.initial_stage)

        # Log event
        self.audit_logger.log_event(
            event_type="workflow_initiated",
            user=initiated_by,
            details={
                "instance_id": instance_id,
                "template_id": template_id,
                "request_id": request_id,
            },
        )

        return instance_id

    def process_approval_decision(
        self, instance_id: str, approver_id: str, decision: str, comments: Optional[str] = None
    ) -> bool:
        """Process an approval decision for a workflow stage.

        Args:
            instance_id: Workflow instance ID
            approver_id: Approver user ID
            decision: Decision ('approve', 'reject', 'delegate')
            comments: Decision comments

        Returns:
            True if decision processed successfully
        """
        instance = self.instances.get(instance_id)
        if not instance:
            raise ValueError(f"Workflow instance {instance_id} not found")

        if instance.status != WorkflowStatus.ACTIVE:
            raise ValueError(f"Workflow {instance_id} is not active")

        # Get current stage
        template = self.templates[instance.template_id]
        current_stage = next(
            (s for s in template.stages if s.stage_id == instance.current_stage), None
        )
        if not current_stage:
            raise ValueError(f"Current stage {instance.current_stage} not found")

        # Validate approver
        user = self.user_management.users.get(approver_id)
        if not user:
            raise ValueError(f"Approver {approver_id} not found")

        # Check if approver has permission
        has_permission = approver_id in current_stage.approver_users or any(
            role in current_stage.approver_roles for role in user["roles"]
        )

        if not has_permission:
            raise ValueError(
                f"User {approver_id} is not authorized to approve this stage"
            )

        # Create decision record
        decision_id = self._generate_decision_id()
        approval_decision = ApprovalDecision(
            decision_id=decision_id,
            workflow_instance_id=instance_id,
            stage_id=current_stage.stage_id,
            approver_id=approver_id,
            decision=decision,
            comments=comments or "",
            decided_at=datetime.datetime.utcnow().isoformat(),
        )

        if instance_id not in self.decisions:
            self.decisions[instance_id] = []
        self.decisions[instance_id].append(approval_decision)

        # Update stage history
        instance.stage_history.append(
            {
                "stage_id": current_stage.stage_id,
                "stage_name": current_stage.name,
                "decision": decision,
                "approver_id": approver_id,
                "decided_at": approval_decision.decided_at,
            }
        )

        # Process the decision
        if decision == "approve":
            self._handle_approval(instance_id, current_stage)
        elif decision == "reject":
            self._handle_rejection(instance_id, current_stage)
        elif decision == "delegate":
            # TODO: Implement delegation logic
            pass

        self._save_workflow_data()

        # Log event
        self.audit_logger.log_event(
            event_type="workflow_decision",
            user=approver_id,
            details={
                "instance_id": instance_id,
                "stage_id": current_stage.stage_id,
                "decision": decision,
            },
        )

        return True

    def get_pending_approvals(self, user_id: str) -> List[Dict]:
        """Get pending approvals for a user.

        Args:
            user_id: User ID

        Returns:
            List of pending approval information
        """
        user = self.user_management.users.get(user_id)
        if not user:
            return []

        pending_approvals = []

        for instance in self.instances.values():
            if instance.status != WorkflowStatus.ACTIVE:
                continue

            template = self.templates.get(instance.template_id)
            if not template:
                continue

            current_stage = next(
                (s for s in template.stages if s.stage_id == instance.current_stage),
                None,
            )
            if not current_stage:
                continue

            # Check if user can approve this stage
            can_approve = user_id in current_stage.approver_users or any(
                role in current_stage.approver_roles for role in user["roles"]
            )

            if not can_approve:
                continue

            # Check if user already decided on this stage
            stage_decisions = [
                d
                for d in self.decisions.get(instance.instance_id, [])
                if d.stage_id == current_stage.stage_id and d.approver_id == user_id
            ]

            if stage_decisions:
                continue

            # Get approval request details
            approval_request = self.approval_manager.get_request(instance.request_id)
            if not approval_request:
                continue

            pending_approvals.append(
                {
                    "workflow_instance_id": instance.instance_id,
                    "request_id": instance.request_id,
                    "stage": current_stage.name,
                    "stage_description": current_stage.description,
                    "request_title": approval_request.title,
                    "request_description": approval_request.description,
                    "initiated_at": instance.initiated_at,
                    "sla_deadline": instance.sla_deadline,
                    "template_name": template.name,
                }
            )

        return pending_approvals

    def check_sla_breaches(self) -> List[str]:
        """Check for SLA breaches and trigger escalations.

        Returns:
            List of instance IDs that breached SLA
        """
        breached_instances = []
        now = datetime.datetime.utcnow()

        for instance in self.instances.values():
            if instance.status != WorkflowStatus.ACTIVE:
                continue

            if instance.sla_deadline:
                deadline = datetime.datetime.fromisoformat(instance.sla_deadline)
                if now > deadline:
                    breached_instances.append(instance.instance_id)
                    self._trigger_escalation(
                        instance.instance_id, EscalationTrigger.SLA_BREACH
                    )

        return breached_instances

    def get_workflow_status(self, instance_id: str) -> Dict:
        """Get detailed workflow status.

        Args:
            instance_id: Workflow instance ID

        Returns:
            Workflow status information
        """
        instance = self.instances.get(instance_id)
        if not instance:
            raise ValueError(f"Workflow instance {instance_id} not found")

        template = self.templates.get(instance.template_id)
        if not template:
            raise ValueError(f"Template {instance.template_id} not found")

        # Get current stage info
        current_stage = next(
            (s for s in template.stages if s.stage_id == instance.current_stage), None
        )

        # Get decisions for current stage
        current_stage_decisions = [
            d
            for d in self.decisions.get(instance_id, [])
            if d.stage_id == instance.current_stage
        ]

        # Calculate progress
        total_stages = len(template.stages)
        completed_stages = len(set(h["stage_id"] for h in instance.stage_history))
        progress = (completed_stages / total_stages * 100) if total_stages > 0 else 0

        return {
            "instance_id": instance_id,
            "template_name": template.name,
            "status": instance.status.value,
            "current_stage": current_stage.name if current_stage else None,
            "progress": progress,
            "initiated_at": instance.initiated_at,
            "sla_deadline": instance.sla_deadline,
            "stage_history": instance.stage_history,
            "current_stage_decisions": [
                {
                    "approver_id": d.approver_id,
                    "decision": d.decision,
                    "decided_at": d.decided_at,
                }
                for d in current_stage_decisions
            ],
        }

    def _start_stage(self, instance_id: str, stage_id: str):
        """Start a workflow stage."""
        instance = self.instances[instance_id]
        template = self.templates[instance.template_id]
        stage = next((s for s in template.stages if s.stage_id == stage_id), None)

        if not stage:
            raise ValueError(f"Stage {stage_id} not found")

        # Check if stage should be auto-approved
        if stage.auto_approve:
            self._auto_approve_stage(instance_id, stage)
            return

        # Check stage condition
        if not self._evaluate_stage_condition(instance_id, stage):
            # Skip to next stage if condition not met
            if stage.next_stage:
                self._start_stage(instance_id, stage.next_stage)
            else:
                self._complete_workflow(instance_id, True)
            return

        # Send notifications
        self._send_stage_notifications(instance_id, stage)

        # Set stage timeout
        if stage.timeout_hours:
            deadline = datetime.datetime.utcnow() + datetime.timedelta(
                hours=stage.timeout_hours
            )
            instance.metadata[f"stage_{stage_id}_deadline"] = deadline.isoformat()

    def _handle_approval(self, instance_id: str, stage: WorkflowStage):
        """Handle stage approval."""
        instance = self.instances[instance_id]

        # Check if minimum approvals reached
        stage_decisions = [
            d
            for d in self.decisions.get(instance_id, [])
            if d.stage_id == stage.stage_id and d.decision == "approve"
        ]

        if len(stage_decisions) >= stage.min_approvals:
            # Move to next stage or complete
            if stage.next_stage:
                instance.current_stage = stage.next_stage
                self._start_stage(instance_id, stage.next_stage)
            else:
                self._complete_workflow(instance_id, True)

    def _handle_rejection(self, instance_id: str, stage: WorkflowStage):
        """Handle stage rejection."""
        instance = self.instances[instance_id]

        # Trigger escalation if configured
        if stage.escalation_stage:
            instance.current_stage = stage.escalation_stage
            self._start_stage(instance_id, stage.escalation_stage)
            self._trigger_escalation(instance_id, EscalationTrigger.REJECTION)
        else:
            # Reject the entire workflow
            self._complete_workflow(instance_id, False)

    def _complete_workflow(self, instance_id: str, approved: bool):
        """Complete a workflow."""
        instance = self.instances[instance_id]
        instance.status = WorkflowStatus.COMPLETED
        instance.completed_at = datetime.datetime.utcnow().isoformat()

        # Update the approval request
        if approved:
            self.approval_manager.approve_request(
                instance.request_id,
                "workflow_system",
                f"Approved through workflow {instance_id}",
            )
        else:
            self.approval_manager.reject_request(
                instance.request_id,
                "workflow_system",
                f"Rejected through workflow {instance_id}",
            )

        # Log completion
        self.audit_logger.log_event(
            event_type="workflow_completed",
            user="system",
            details={"instance_id": instance_id, "approved": approved},
        )

    def _auto_approve_stage(self, instance_id: str, stage: WorkflowStage):
        """Auto-approve a stage."""
        decision_id = self._generate_decision_id()
        approval_decision = ApprovalDecision(
            decision_id=decision_id,
            workflow_instance_id=instance_id,
            stage_id=stage.stage_id,
            approver_id="system",
            decision="approve",
            comments="Auto-approved by system",
            decided_at=datetime.datetime.utcnow().isoformat(),
        )

        if instance_id not in self.decisions:
            self.decisions[instance_id] = []
        self.decisions[instance_id].append(approval_decision)

        self._handle_approval(instance_id, stage)

    def _evaluate_stage_condition(self, instance_id: str, stage: WorkflowStage) -> bool:
        """Evaluate if a stage condition is met."""
        evaluator = self.condition_evaluators.get(stage.condition)
        if not evaluator:
            return True

        return bool(evaluator(instance_id, stage))

    def _evaluate_always(self, instance_id: str, stage: WorkflowStage) -> bool:
        """Always true condition."""
        return True

    def _evaluate_risk_level(self, instance_id: str, stage: WorkflowStage) -> bool:
        """Evaluate risk level condition."""
        instance = self.instances[instance_id]
        approval_request = self.approval_manager.get_request(instance.request_id)

        if not approval_request:
            return False

        required_levels = stage.condition_params.get("risk_levels", [])
        request_metadata = (
            approval_request.data if hasattr(approval_request, "data") else {}
        )
        current_level = request_metadata.get("risk_level", "low")

        return current_level in required_levels

    def _evaluate_patch_size(self, instance_id: str, stage: WorkflowStage) -> bool:
        """Evaluate patch size condition."""
        instance = self.instances[instance_id]
        approval_request = self.approval_manager.get_request(instance.request_id)

        if not approval_request:
            return False

        min_size = stage.condition_params.get("min_size", 0)
        max_size = stage.condition_params.get("max_size", float("inf"))

        request_metadata = (
            approval_request.data if hasattr(approval_request, "data") else {}
        )
        patch_size = int(request_metadata.get("patch_size", 0))

        return min_size <= patch_size <= max_size

    def _evaluate_error_type(self, instance_id: str, stage: WorkflowStage) -> bool:
        """Evaluate error type condition."""
        instance = self.instances[instance_id]
        approval_request = self.approval_manager.get_request(instance.request_id)

        if not approval_request:
            return False

        required_types = stage.condition_params.get("error_types", [])
        request_metadata = (
            approval_request.data if hasattr(approval_request, "data") else {}
        )
        error_type = request_metadata.get("error_type", "")

        return error_type in required_types

    def _evaluate_environment(self, instance_id: str, stage: WorkflowStage) -> bool:
        """Evaluate environment condition."""
        instance = self.instances[instance_id]
        approval_request = self.approval_manager.get_request(instance.request_id)

        if not approval_request:
            return False

        required_envs = stage.condition_params.get("environments", [])
        request_metadata = (
            approval_request.data if hasattr(approval_request, "data") else {}
        )
        environment = request_metadata.get("environment", "")

        return environment in required_envs

    def _evaluate_time_based(self, instance_id: str, stage: WorkflowStage) -> bool:
        """Evaluate time-based condition."""
        # Check business hours, day of week, etc.
        now = datetime.datetime.utcnow()

        # Example: Only during business hours
        business_hours_only = stage.condition_params.get("business_hours_only", False)
        if business_hours_only:
            hour = now.hour
            if hour < 9 or hour > 17:  # 9 AM to 5 PM
                return False

        # Example: Not on weekends
        no_weekends = stage.condition_params.get("no_weekends", False)
        if no_weekends:
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False

        return True

    def _trigger_escalation(self, instance_id: str, trigger: EscalationTrigger):
        """Trigger workflow escalation."""
        # Log escalation event
        self.audit_logger.log_event(
            event_type="workflow_escalation",
            user="system",
            details={"instance_id": instance_id, "trigger": trigger.value},
        )

        # Send escalation notifications
        # TODO: Implement notification logic

    def _send_stage_notifications(self, instance_id: str, stage: WorkflowStage):
        """Send notifications for a stage."""
        # TODO: Implement notification logic
        # This would integrate with email, Slack, etc.
        pass

    def _create_default_templates(self):
        """Create default workflow templates."""
        # High-risk security patch workflow
        security_stages = [
            {
                "name": "Security Review",
                "description": "Initial security team review",
                "approver_roles": ["security_reviewer"],
                "min_approvals": 2,
                "timeout_hours": 4,
            },
            {
                "name": "Architecture Review",
                "description": "Architecture team review for system impact",
                "approver_roles": ["architect", "tech_lead"],
                "min_approvals": 1,
                "timeout_hours": 8,
                "condition": "risk_level",
                "condition_params": {"risk_levels": ["high", "critical"]},
            },
            {
                "name": "Management Approval",
                "description": "Management approval for critical changes",
                "approver_roles": ["manager", "director"],
                "min_approvals": 1,
                "timeout_hours": 12,
                "condition": "risk_level",
                "condition_params": {"risk_levels": ["critical"]},
            },
        ]

        self.create_workflow_template(
            name="Security Patch Workflow",
            description="Multi-stage approval for security patches",
            category="security",
            stages=security_stages,
        )

        # Standard bug fix workflow
        bugfix_stages = [
            {
                "name": "Peer Review",
                "description": "Peer developer review",
                "approver_roles": ["developer", "reviewer"],
                "min_approvals": 1,
                "timeout_hours": 24,
            },
            {
                "name": "QA Validation",
                "description": "QA team validation",
                "approver_roles": ["qa_engineer"],
                "min_approvals": 1,
                "timeout_hours": 12,
                "auto_approve": False,
            },
        ]

        self.create_workflow_template(
            name="Standard Bug Fix Workflow",
            description="Standard approval process for bug fixes",
            category="bugfix",
            stages=bugfix_stages,
        )

        # Emergency patch workflow
        emergency_stages = [
            {
                "name": "Emergency Approval",
                "description": "Fast-track emergency approval",
                "approver_roles": ["emergency_approver", "oncall_engineer"],
                "min_approvals": 1,
                "timeout_hours": 1,
            }
        ]

        self.create_workflow_template(
            name="Emergency Patch Workflow",
            description="Fast-track approval for emergency patches",
            category="emergency",
            stages=emergency_stages,
        )

    def _generate_template_id(self) -> str:
        """Generate unique template ID."""
        return f"wft_{uuid.uuid4().hex[:12]}"

    def _generate_instance_id(self) -> str:
        """Generate unique instance ID."""
        return f"wfi_{uuid.uuid4().hex[:12]}"

    def _generate_stage_id(self) -> str:
        """Generate unique stage ID."""
        return f"stg_{uuid.uuid4().hex[:8]}"

    def _generate_decision_id(self) -> str:
        """Generate unique decision ID."""
        return f"dec_{uuid.uuid4().hex[:12]}"

    def _load_workflow_data(self):
        """Load workflow data from storage."""
        # Load templates
        templates_file = self.storage_path / "templates.json"
        if templates_file.exists():
            with open(templates_file, "r") as f:
                templates_data = json.load(f)
                for template_data in templates_data:
                    stages = []
                    for stage_data in template_data["stages"]:
                        stage = WorkflowStage(**stage_data)
                        stages.append(stage)

                    template = WorkflowTemplate(
                        template_id=template_data["template_id"],
                        name=template_data["name"],
                        description=template_data["description"],
                        category=template_data["category"],
                        stages=stages,
                        initial_stage=template_data["initial_stage"],
                        metadata=template_data.get("metadata", {}),
                        is_active=template_data.get("is_active", True),
                        created_at=template_data.get("created_at"),
                        updated_at=template_data.get("updated_at"),
                    )
                    self.templates[template.template_id] = template

        # Load instances
        instances_file = self.storage_path / "instances.json"
        if instances_file.exists():
            with open(instances_file, "r") as f:
                instances_data = json.load(f)
                for instance_data in instances_data:
                    instance = WorkflowInstance(
                        instance_id=instance_data["instance_id"],
                        template_id=instance_data["template_id"],
                        request_id=instance_data["request_id"],
                        current_stage=instance_data["current_stage"],
                        status=WorkflowStatus(instance_data["status"]),
                        initiated_by=instance_data["initiated_by"],
                        initiated_at=instance_data["initiated_at"],
                        completed_at=instance_data.get("completed_at"),
                        stage_history=instance_data.get("stage_history", []),
                        metadata=instance_data.get("metadata", {}),
                        sla_deadline=instance_data.get("sla_deadline"),
                    )
                    self.instances[instance.instance_id] = instance

        # Load decisions
        decisions_file = self.storage_path / "decisions.json"
        if decisions_file.exists():
            with open(decisions_file, "r") as f:
                decisions_data = json.load(f)
                for instance_id, decisions_list in decisions_data.items():
                    self.decisions[instance_id] = []
                    for decision_data in decisions_list:
                        decision = ApprovalDecision(**decision_data)
                        self.decisions[instance_id].append(decision)

    def _save_workflow_data(self):
        """Save workflow data to storage."""
        # Save templates
        templates_data = []
        for template in self.templates.values():
            template_dict = {
                "template_id": template.template_id,
                "name": template.name,
                "description": template.description,
                "category": template.category,
                "stages": [
                    {
                        "stage_id": stage.stage_id,
                        "name": stage.name,
                        "description": stage.description,
                        "approver_roles": stage.approver_roles,
                        "approver_users": stage.approver_users,
                        "min_approvals": stage.min_approvals,
                        "max_approvals": stage.max_approvals,
                        "timeout_hours": stage.timeout_hours,
                        "condition": stage.condition.value,
                        "condition_params": stage.condition_params,
                        "next_stage": stage.next_stage,
                        "escalation_stage": stage.escalation_stage,
                        "auto_approve": stage.auto_approve,
                        "notifications": stage.notifications,
                    }
                    for stage in template.stages
                ],
                "initial_stage": template.initial_stage,
                "metadata": template.metadata,
                "is_active": template.is_active,
                "created_at": template.created_at,
                "updated_at": template.updated_at,
            }
            templates_data.append(template_dict)

        with open(self.storage_path / "templates.json", "w") as f:
            json.dump(templates_data, f, indent=2)

        # Save instances
        instances_data = []
        for instance in self.instances.values():
            instance_dict = {
                "instance_id": instance.instance_id,
                "template_id": instance.template_id,
                "request_id": instance.request_id,
                "current_stage": instance.current_stage,
                "status": instance.status.value,
                "initiated_by": instance.initiated_by,
                "initiated_at": instance.initiated_at,
                "completed_at": instance.completed_at,
                "stage_history": instance.stage_history,
                "metadata": instance.metadata,
                "sla_deadline": instance.sla_deadline,
            }
            instances_data.append(instance_dict)

        with open(self.storage_path / "instances.json", "w") as f:
            json.dump(instances_data, f, indent=2)

        # Save decisions
        decisions_data = {}
        for instance_id, decisions_list in self.decisions.items():
            decisions_data[instance_id] = [
                {
                    "decision_id": d.decision_id,
                    "workflow_instance_id": d.workflow_instance_id,
                    "stage_id": d.stage_id,
                    "approver_id": d.approver_id,
                    "decision": d.decision,
                    "comments": d.comments,
                    "decided_at": d.decided_at,
                    "delegate_to": d.delegate_to,
                    "conditions_met": d.conditions_met,
                }
                for d in decisions_list
            ]

        with open(self.storage_path / "decisions.json", "w") as f:
            json.dump(decisions_data, f, indent=2)


# Singleton instance
_workflow_engine = None


def get_workflow_engine(config: Optional[Dict] = None) -> ApprovalWorkflowEngine:
    """Get or create the singleton ApprovalWorkflowEngine instance."""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = ApprovalWorkflowEngine(config)
    return _workflow_engine
