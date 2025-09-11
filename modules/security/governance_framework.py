"""
Enterprise Governance Framework for Homeostasis.

This module provides a unified interface for all governance capabilities,
integrating RBAC, user management, approval workflows, compliance reporting,
identity providers, and policy enforcement.

This is the main entry point for enterprise governance features.
"""

import datetime
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from .approval import ApprovalType, get_approval_manager
from .approval_workflow import get_workflow_engine
from .audit import get_audit_logger
from .auth import get_auth_manager
from .compliance_reporting import ComplianceFramework, get_compliance_reporting
from .governance_manager import GovernanceManager
from .identity_providers import get_identity_integration
from .policy_enforcement import PolicyEvaluationContext, get_policy_engine
from .rbac import get_rbac_manager
from .user_management import get_user_management

if TYPE_CHECKING:
    from .regulated_industries import RegulatedIndustry

logger = logging.getLogger(__name__)


class GovernanceCapability(Enum):
    """Governance capabilities."""

    RBAC = "rbac"
    USER_MANAGEMENT = "user_management"
    APPROVAL_WORKFLOWS = "approval_workflows"
    COMPLIANCE_REPORTING = "compliance_reporting"
    POLICY_ENFORCEMENT = "policy_enforcement"
    IDENTITY_FEDERATION = "identity_federation"
    AUDIT_LOGGING = "audit_logging"
    REGULATED_INDUSTRIES = "regulated_industries"


@dataclass
class GovernanceConfig:
    """Configuration for governance framework."""

    enabled_capabilities: List[GovernanceCapability] = field(
        default_factory=lambda: list(GovernanceCapability)
    )
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    require_approval_for_production: bool = True
    enforce_policies: bool = True
    enable_sso: bool = False
    audit_retention_days: int = 365
    session_timeout_minutes: int = 60
    mfa_required_roles: List[str] = field(default_factory=lambda: ["admin", "operator"])
    metadata: Dict = field(default_factory=dict)


@dataclass
class HealingActionRequest:
    """Request for a healing action that requires governance."""

    request_id: str
    action_type: str
    error_context: Dict
    patch_content: str
    environment: str
    service_name: str
    language: str
    requested_by: Optional[str] = None
    risk_assessment: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class GovernanceDecision:
    """Decision from governance evaluation."""

    allowed: bool
    reason: str
    approval_required: bool = False
    approval_request_id: Optional[str] = None
    workflow_instance_id: Optional[str] = None
    policy_violations: List[str] = field(default_factory=list)
    compliance_issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class EnterpriseGovernanceFramework:
    """
    Unified enterprise governance framework for Homeostasis.

    Integrates all governance components to provide comprehensive
    control over healing actions in enterprise and regulated environments.
    """

    def __init__(self, config: Union[Dict, GovernanceConfig] = None):
        """Initialize the governance framework.

        Args:
            config: Configuration dictionary or GovernanceConfig object
        """
        if isinstance(config, dict):
            # Extract only the fields that GovernanceConfig accepts
            governance_config_fields = {
                "enabled_capabilities",
                "compliance_frameworks",
                "require_approval_for_production",
                "enforce_policies",
                "enable_sso",
                "audit_retention_days",
                "session_timeout_minutes",
                "mfa_required_roles",
                "metadata",
            }
            # Filter config to only include valid fields
            filtered_config = {
                k: v for k, v in config.items() if k in governance_config_fields
            }
            # Store extra fields in metadata for backward compatibility
            extra_fields = {
                k: v for k, v in config.items() if k not in governance_config_fields
            }
            if extra_fields:
                filtered_config["metadata"] = {
                    **filtered_config.get("metadata", {}),
                    **extra_fields,
                }
            self.config = GovernanceConfig(**filtered_config)
        else:
            self.config = config or GovernanceConfig()

        # Initialize all managers
        self._initialize_managers()

        # Set up initial configuration
        self._configure_framework()

        logger.info(
            f"Enterprise Governance Framework initialized with capabilities: "
            f"{[c.value for c in self.config.enabled_capabilities]}"
        )

    def evaluate_healing_action(
        self, request: HealingActionRequest
    ) -> GovernanceDecision:
        """Evaluate a healing action against all governance rules.

        Args:
            request: Healing action request

        Returns:
            Governance decision
        """
        decision = GovernanceDecision(allowed=True, reason="")

        # 1. Check user permissions
        if request.requested_by:
            if not self._check_user_permissions(request):
                decision.allowed = False
                decision.reason = "User lacks required permissions"
                return decision

        # 2. Evaluate policies
        if GovernanceCapability.POLICY_ENFORCEMENT in self.config.enabled_capabilities:
            policy_result = self._evaluate_policies(request)
            if not policy_result[0]:
                decision.allowed = False
                decision.reason = policy_result[1]
                decision.policy_violations = policy_result[2].get("violations", [])

                # Check if approval can override
                if policy_result[2].get("approval_reasons"):
                    decision.approval_required = True
                else:
                    return decision

        # 3. Check compliance requirements
        if (
            GovernanceCapability.COMPLIANCE_REPORTING
            in self.config.enabled_capabilities
        ):
            compliance_result = self._check_compliance_requirements(request)
            if compliance_result:
                decision.compliance_issues = compliance_result
                decision.approval_required = True

        # 3a. Check regulated industry requirements
        if (
            GovernanceCapability.REGULATED_INDUSTRIES
            in self.config.enabled_capabilities
        ):
            industry_result = self._check_industry_compliance(request)
            if not industry_result["passed"]:
                decision.allowed = False
                decision.reason = "Industry compliance validation failed"
                decision.compliance_issues.extend(industry_result["issues"])
                decision.metadata["industry_validation"] = industry_result

                # Check if approval can override
                if industry_result.get("approval_allowed"):
                    decision.approval_required = True
                else:
                    return decision

        # 4. Assess risk and determine if approval needed
        risk_level = self._assess_risk(request)
        request.risk_assessment = {"level": risk_level}

        if self._requires_approval(request, risk_level):
            decision.approval_required = True

        # 5. If approval required, initiate workflow
        if decision.approval_required:
            approval_result = self._initiate_approval_workflow(request)
            decision.approval_request_id = approval_result["approval_request_id"]
            decision.workflow_instance_id = approval_result.get("workflow_instance_id")
            decision.allowed = False
            decision.reason = "Approval required for this action"

        # 6. Log the evaluation
        self._log_governance_evaluation(request, decision)

        return decision

    def authenticate_user(
        self, username: str, password: str, method: str = "password", **kwargs
    ) -> Optional[Dict]:
        """Authenticate a user.

        Args:
            username: Username
            password: Password (or token for SSO)
            method: Authentication method
            **kwargs: Additional authentication parameters

        Returns:
            Authentication result if successful
        """
        if method == "password":
            return self.user_management.authenticate_user(
                username,
                password,
                ip_address=kwargs.get("ip_address"),
                user_agent=kwargs.get("user_agent"),
            )
        elif method == "ldap":
            return self.identity_integration.authenticate_ldap(
                username, password, provider_id=kwargs.get("provider_id")
            )
        elif method == "sso":
            # Handle SSO callback
            return self.identity_integration.complete_authentication(
                session_id=kwargs.get("session_id"), callback_data=kwargs
            )
        else:
            raise ValueError(f"Unsupported authentication method: {method}")

    def initiate_sso(self, provider_name: str, redirect_uri: str) -> Dict:
        """Initiate SSO authentication.

        Args:
            provider_name: SSO provider name
            redirect_uri: Redirect URI after authentication

        Returns:
            SSO initiation response
        """
        # Find provider by name
        provider_id = None
        for pid, provider in self.identity_integration.providers.items():
            if provider.name == provider_name:
                provider_id = pid
                break

        if not provider_id:
            raise ValueError(f"SSO provider '{provider_name}' not found")

        return self.identity_integration.initiate_authentication(
            provider_id=provider_id, redirect_uri=redirect_uri
        )

    def get_user_dashboard(self, user_id: str) -> Dict:
        """Get governance dashboard for a user.

        Args:
            user_id: User ID

        Returns:
            Dashboard data
        """
        user = self.user_management.users.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        dashboard = {
            "user": {
                "user_id": user_id,
                "username": user["username"],
                "roles": user["roles"],
                "groups": user["groups"],
                "permissions": list(self.user_management.get_user_permissions(user_id)),
            },
            "pending_approvals": [],
            "recent_actions": [],
            "compliance_status": {},
            "policy_metrics": {},
        }

        # Get pending approvals
        if GovernanceCapability.APPROVAL_WORKFLOWS in self.config.enabled_capabilities:
            dashboard["pending_approvals"] = self.workflow_engine.get_pending_approvals(
                user_id
            )

        # Get compliance status
        if (
            GovernanceCapability.COMPLIANCE_REPORTING
            in self.config.enabled_capabilities
        ):
            dashboard["compliance_status"] = (
                self.compliance_reporting.get_compliance_dashboard()
            )

        # Get policy metrics
        if GovernanceCapability.POLICY_ENFORCEMENT in self.config.enabled_capabilities:
            dashboard["policy_metrics"] = self.policy_engine.get_policy_metrics()

        return dashboard

    def generate_compliance_report(
        self, framework: ComplianceFramework, report_type: str = "assessment"
    ) -> str:
        """Generate a compliance report.

        Args:
            framework: Compliance framework
            report_type: Type of report

        Returns:
            Report ID
        """
        return self.compliance_reporting.generate_compliance_report(
            framework=framework, report_type=report_type
        )

    def manage_user(self, action: str, user_id: str = None, **params) -> Dict:
        """Manage users (create, update, deactivate).

        Args:
            action: Management action (create, update, deactivate)
            user_id: User ID (for update/deactivate)
            **params: Action-specific parameters

        Returns:
            Action result
        """
        if action == "create":
            user_id = self.user_management.create_user(**params)
            return {"user_id": user_id, "status": "created"}

        elif action == "update":
            if not user_id:
                raise ValueError("user_id required for update")
            self.user_management.update_user(user_id, **params)
            return {"user_id": user_id, "status": "updated"}

        elif action == "deactivate":
            if not user_id:
                raise ValueError("user_id required for deactivate")
            self.user_management.deactivate_user(user_id, params.get("reason"))
            return {"user_id": user_id, "status": "deactivated"}

        else:
            raise ValueError(f"Unknown user management action: {action}")

    def configure_regulated_industry(
        self, industry: "RegulatedIndustry", requirements: List, config: Dict
    ) -> bool:
        """Configure a regulated industry with compliance requirements.

        Args:
            industry: Industry type
            requirements: List of compliance requirements
            config: Industry configuration

        Returns:
            True if successful
        """
        if (
            GovernanceCapability.REGULATED_INDUSTRIES
            not in self.config.enabled_capabilities
        ):
            raise ValueError("Regulated industries capability not enabled")

        return self.regulated_industries.configure_industry(
            industry, requirements, config
        )

    def get_industry_dashboard(self, industry: "RegulatedIndustry") -> Dict:
        """Get compliance dashboard for a regulated industry.

        Args:
            industry: Industry type

        Returns:
            Dashboard data
        """
        if (
            GovernanceCapability.REGULATED_INDUSTRIES
            not in self.config.enabled_capabilities
        ):
            raise ValueError("Regulated industries capability not enabled")

        return self.regulated_industries.get_industry_dashboard(industry)

    def configure_policy(self, name: str, rules: List[Dict], **params) -> str:
        """Configure a governance policy.

        Args:
            name: Policy name
            rules: Policy rules
            **params: Additional policy parameters

        Returns:
            Policy ID
        """
        from .policy_enforcement import PolicyScope, PolicyType

        return self.policy_engine.create_policy(
            name=name,
            description=params.get("description", ""),
            type=PolicyType(params.get("type", "healing")),
            scope=PolicyScope(params.get("scope", "global")),
            rules=rules,
            compliance_frameworks=params.get("compliance_frameworks", []),
        )

    def get_governance_metrics(self) -> Dict:
        """Get overall governance metrics.

        Returns:
            Governance metrics
        """
        metrics = {
            "users": {
                "total": len(self.user_management.users),
                "active": len(
                    [
                        u
                        for u in self.user_management.users.values()
                        if u["status"] == "active"
                    ]
                ),
            },
            "approvals": {
                "pending": len(self.approval_manager.list_requests(status="pending")),
                "total": len(self.approval_manager.requests),
            },
            "compliance": {},
            "policies": {},
            "authentication": {},
        }

        # Add compliance metrics
        if (
            GovernanceCapability.COMPLIANCE_REPORTING
            in self.config.enabled_capabilities
        ):
            compliance_dash = self.compliance_reporting.get_compliance_dashboard()
            metrics["compliance"] = {
                "frameworks": compliance_dash.get("framework_summary", {}),
                "findings": compliance_dash.get("total_findings", 0),
            }

        # Add policy metrics
        if GovernanceCapability.POLICY_ENFORCEMENT in self.config.enabled_capabilities:
            metrics["policies"] = self.policy_engine.get_policy_metrics()

        return metrics

    def _initialize_managers(self):
        """Initialize all governance managers."""
        base_config = {
            "audit_retention_days": self.config.audit_retention_days,
            "session_timeout": self.config.session_timeout_minutes * 60,
            "mfa_required_roles": self.config.mfa_required_roles,
        }

        # Core managers (always needed)
        self.audit_logger = get_audit_logger(base_config)
        self.auth_manager = get_auth_manager(base_config)
        self.rbac_manager = get_rbac_manager(base_config)

        # Capability-specific managers
        if GovernanceCapability.USER_MANAGEMENT in self.config.enabled_capabilities:
            self.user_management = get_user_management(base_config)

        if GovernanceCapability.APPROVAL_WORKFLOWS in self.config.enabled_capabilities:
            self.approval_manager = get_approval_manager(base_config)
            self.workflow_engine = get_workflow_engine(base_config)

        if (
            GovernanceCapability.COMPLIANCE_REPORTING
            in self.config.enabled_capabilities
        ):
            self.compliance_reporting = get_compliance_reporting(base_config)

        if GovernanceCapability.POLICY_ENFORCEMENT in self.config.enabled_capabilities:
            self.policy_engine = get_policy_engine(base_config)

        if GovernanceCapability.IDENTITY_FEDERATION in self.config.enabled_capabilities:
            self.identity_integration = get_identity_integration(base_config)

        # LLM governance manager (always needed)
        self.governance_manager = GovernanceManager(base_config)

        # Regulated industries support
        if (
            GovernanceCapability.REGULATED_INDUSTRIES
            in self.config.enabled_capabilities
        ):
            from .regulated_industries import get_regulated_industries

            self.regulated_industries = get_regulated_industries(base_config)
            # Set the governance framework reference to avoid circular dependency
            if hasattr(self.regulated_industries, "set_governance_framework"):
                self.regulated_industries.set_governance_framework(self)

    def _configure_framework(self):
        """Configure the framework based on settings."""
        # Set up default admin user if none exists
        if hasattr(self, "user_management") and not self.user_management.users:
            try:
                admin_id = self.user_management.create_user(
                    username="admin",
                    email="admin@homeostasis.local",
                    password="ChangeMe123!",  # Should be changed on first login
                    full_name="System Administrator",
                    roles=["admin"],
                )
                logger.info(f"Created default admin user: {admin_id}")
            except Exception as e:
                logger.warning(f"Could not create default admin user: {str(e)}")

        # Configure compliance frameworks
        if self.config.compliance_frameworks and hasattr(self, "compliance_reporting"):
            for framework in self.config.compliance_frameworks:
                # Ensure framework controls are initialized
                logger.info(f"Configured compliance framework: {framework.value}")

    def _check_user_permissions(self, request: HealingActionRequest) -> bool:
        """Check if user has permissions for the healing action."""
        if not request.requested_by:
            return True  # System actions allowed

        user = self.user_management.users.get(request.requested_by)
        if not user:
            return False

        # Check specific permissions based on action type
        required_permissions = {
            "database_change": "patch.database",
            "security_fix": "patch.security",
            "config_change": "patch.config",
            "code_fix": "patch.code",
        }

        required_permission = required_permissions.get(
            request.action_type, "patch.request"
        )
        permissions = self.user_management.get_user_permissions(request.requested_by)

        return required_permission in permissions

    def _evaluate_policies(
        self, request: HealingActionRequest
    ) -> Tuple[bool, str, Dict]:
        """Evaluate policies for the healing action."""
        context = PolicyEvaluationContext(
            healing_action=request.action_type,
            error_type=request.error_context.get("error_type", ""),
            error_details=request.error_context,
            patch_content=request.patch_content,
            environment=request.environment,
            service_name=request.service_name,
            language=request.language,
            user_id=request.requested_by,
            risk_level=request.risk_assessment.get("level", "low"),
            patch_size=len(request.patch_content),
            metadata=request.metadata,
        )

        return self.policy_engine.enforce_healing_policy(context)

    def _check_compliance_requirements(
        self, request: HealingActionRequest
    ) -> List[str]:
        """Check compliance requirements for the healing action."""
        compliance_issues = []

        # Check each configured compliance framework
        for framework in self.config.compliance_frameworks:
            # Simplified check - in practice would be more sophisticated
            if framework == ComplianceFramework.HIPAA:
                if (
                    "patient" in request.patch_content.lower() or
                    "medical" in request.patch_content.lower()
                ):
                    compliance_issues.append("HIPAA: Potential PHI exposure in patch")

            elif framework == ComplianceFramework.PCI_DSS:
                if (
                    "card" in request.patch_content.lower() or
                    "payment" in request.patch_content.lower()
                ):
                    compliance_issues.append(
                        "PCI-DSS: Potential cardholder data handling"
                    )

            elif framework == ComplianceFramework.SOC2:
                if request.environment == "production" and not request.requested_by:
                    compliance_issues.append(
                        "SOC2: Production changes require authenticated user"
                    )

        return compliance_issues

    def _assess_risk(self, request: HealingActionRequest) -> str:
        """Assess risk level of the healing action."""
        risk_score = 0

        # Environment risk
        if request.environment == "production":
            risk_score += 3
        elif request.environment == "staging":
            risk_score += 1

        # Action type risk
        high_risk_actions = ["database_change", "security_fix", "authentication_change"]
        if request.action_type in high_risk_actions:
            risk_score += 2

        # Patch size risk
        if len(request.patch_content) > 1000:
            risk_score += 2
        elif len(request.patch_content) > 500:
            risk_score += 1

        # Service criticality
        critical_services = request.metadata.get("critical_services", [])
        if request.service_name in critical_services:
            risk_score += 2

        # Determine risk level
        if risk_score >= 5:
            return "high"
        elif risk_score >= 3:
            return "medium"
        else:
            return "low"

    def _requires_approval(
        self, request: HealingActionRequest, risk_level: str
    ) -> bool:
        """Determine if approval is required."""
        # Always require approval for production if configured
        if (
            self.config.require_approval_for_production and
            request.environment == "production"
        ):
            return True

        # Require approval for high risk
        if risk_level == "high":
            return True

        # Check if action type requires approval
        approval_required_actions = ["database_change", "security_fix", "config_change"]
        if request.action_type in approval_required_actions:
            return True

        return False

    def _initiate_approval_workflow(self, request: HealingActionRequest) -> Dict:
        """Initiate approval workflow for the healing action."""
        # Create approval request
        approval_request = self.approval_manager.create_request(
            request_type=ApprovalType.FIX_DEPLOYMENT,
            requester=request.requested_by or "system",
            title=f"Healing Action: {request.action_type}",
            description=f"Automated fix for {request.error_context.get('error_type', 'error')} "
            f"in {request.service_name} ({request.environment})",
            data={
                "healing_request_id": request.request_id,
                "patch_content": request.patch_content,
                "risk_level": request.risk_assessment.get("level", "medium"),
                "environment": request.environment,
                "service": request.service_name,
            },
        )

        # Select workflow template based on risk and environment
        if (
            request.environment == "production" and
            request.risk_assessment.get("level") == "high"
        ):
            template_name = "Security Patch Workflow"
        elif request.environment == "production":
            template_name = "Standard Bug Fix Workflow"
        else:
            template_name = "Emergency Patch Workflow"

        # Find template
        template_id = None
        for tid, template in self.workflow_engine.templates.items():
            if template.name == template_name:
                template_id = tid
                break

        workflow_instance_id = None
        if template_id:
            # Initiate workflow
            workflow_instance_id = self.workflow_engine.initiate_workflow(
                template_id=template_id,
                request_id=approval_request.request_id,
                initiated_by=request.requested_by or "system",
                metadata={
                    "healing_request": request.request_id,
                    "auto_initiated": True,
                },
            )

        return {
            "approval_request_id": approval_request.request_id,
            "workflow_instance_id": workflow_instance_id,
        }

    def _check_industry_compliance(self, request: HealingActionRequest) -> Dict:
        """Check regulated industry compliance requirements."""
        # Build context for industry validation
        context = {
            "user": request.requested_by or "system",
            "user_role": (
                self._get_user_primary_role(request.requested_by)
                if request.requested_by
                else "system"
            ),
            "environment": request.environment,
            "system": request.service_name,
            "audit_enabled": True,
            "data_types": self._infer_data_types(request),
        }

        # Build action details
        action = {
            "id": request.request_id,
            "type": request.action_type,
            "patch": request.patch_content,
            "target_system": request.service_name,
        }

        # Validate against industry requirements
        validation = self.regulated_industries.validate_healing_action(action, context)

        # Build result
        result = {"passed": validation.passed, "issues": [], "approval_allowed": False}

        # Add failures as issues
        for failure in validation.failures:
            result["issues"].append(
                f"{failure['requirement']}: {failure['message']} - {failure['remediation']}"
            )

        # Add critical warnings as issues
        for warning in validation.warnings:
            if warning.get("severity") == "high":
                result["issues"].append(
                    f"{warning['requirement']}: {warning['message']}"
                )

        # Determine if approval can override
        critical_failures = [
            f for f in validation.failures if f.get("severity") == "critical"
        ]
        if not critical_failures and validation.failures:
            result["approval_allowed"] = True

        return result

    def _get_user_primary_role(self, user_id: str) -> str:
        """Get user's primary role."""
        if not user_id:
            return "system"

        user = self.user_management.users.get(user_id)
        if user and user.get("roles"):
            return user["roles"][0]

        return "user"

    def _infer_data_types(self, request: HealingActionRequest) -> List[str]:
        """Infer data types from request context."""
        data_types = []

        # Check service name
        service_lower = request.service_name.lower()
        if any(
            term in service_lower for term in ["patient", "medical", "health", "ehr"]
        ):
            data_types.append("phi")
            data_types.append("medical")

        if any(
            term in service_lower
            for term in ["payment", "billing", "transaction", "card"]
        ):
            data_types.append("financial")
            data_types.append("payment")

        if any(
            term in service_lower for term in ["classified", "secret", "government"]
        ):
            data_types.append("government")

        if any(
            term in service_lower for term in ["pharma", "clinical", "trial", "drug"]
        ):
            data_types.append("pharmaceutical")

        if any(
            term in service_lower
            for term in ["telecom", "utility", "network", "emergency"]
        ):
            data_types.append("telecom")

        # Check patch content for indicators
        patch_lower = request.patch_content.lower()
        if any(
            term in patch_lower for term in ["patient", "diagnosis", "prescription"]
        ):
            if "phi" not in data_types:
                data_types.append("phi")

        if any(term in patch_lower for term in ["card", "payment", "transaction"]):
            if "payment" not in data_types:
                data_types.append("payment")

        return data_types

    def _log_governance_evaluation(
        self, request: HealingActionRequest, decision: GovernanceDecision
    ):
        """Log governance evaluation."""
        self.audit_logger.log_event(
            event_type="governance_evaluation",
            user=request.requested_by or "system",
            details={
                "request_id": request.request_id,
                "action_type": request.action_type,
                "environment": request.environment,
                "service": request.service_name,
                "decision": "allowed" if decision.allowed else "denied",
                "reason": decision.reason,
                "approval_required": decision.approval_required,
                "policy_violations": decision.policy_violations,
                "compliance_issues": decision.compliance_issues,
            },
            status="success" if decision.allowed else "denied",
        )


# Singleton instance
_governance_framework = None


def get_governance_framework(
    config: Union[Dict, GovernanceConfig] = None,
) -> EnterpriseGovernanceFramework:
    """Get or create the singleton EnterpriseGovernanceFramework instance."""
    global _governance_framework
    if _governance_framework is None:
        _governance_framework = EnterpriseGovernanceFramework(config)
    return _governance_framework


# Convenience functions for common operations


def evaluate_healing_action(
    action_type: str, error_context: Dict, patch_content: str, **kwargs
) -> GovernanceDecision:
    """Evaluate a healing action against governance rules.

    Args:
        action_type: Type of healing action
        error_context: Error context information
        patch_content: Proposed patch content
        **kwargs: Additional parameters

    Returns:
        Governance decision
    """
    framework = get_governance_framework()

    request = HealingActionRequest(
        request_id=kwargs.get(
            "request_id", f"req_{datetime.datetime.utcnow().timestamp()}"
        ),
        action_type=action_type,
        error_context=error_context,
        patch_content=patch_content,
        environment=kwargs.get("environment", "development"),
        service_name=kwargs.get("service_name", "unknown"),
        language=kwargs.get("language", "unknown"),
        requested_by=kwargs.get("requested_by"),
        metadata=kwargs.get("metadata", {}),
    )

    return framework.evaluate_healing_action(request)


def require_approval(func):
    """Decorator to require governance approval for a function.

    The decorated function should accept a 'context' parameter that includes:
    - action_type: Type of action being performed
    - error_context: Error information
    - patch_content: Proposed changes
    - environment: Target environment
    - service_name: Service name
    - language: Programming language
    """

    def wrapper(*args, **kwargs):
        # Extract context from kwargs or first positional arg
        context = kwargs.get("context")
        if not context and args:
            context = args[0] if isinstance(args[0], dict) else None

        if not context:
            raise ValueError("Context required for governance approval")

        # Evaluate governance
        decision = evaluate_healing_action(
            action_type=context.get("action_type", "unknown"),
            error_context=context.get("error_context", {}),
            patch_content=context.get("patch_content", ""),
            environment=context.get("environment", "development"),
            service_name=context.get("service_name", "unknown"),
            language=context.get("language", "unknown"),
            requested_by=context.get("user_id"),
        )

        if not decision.allowed:
            raise PermissionError(f"Governance denied: {decision.reason}")

        # Add governance decision to context
        context["governance_decision"] = decision

        # Call original function
        return func(*args, **kwargs)

    return wrapper
