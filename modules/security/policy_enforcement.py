"""
Policy Enforcement Engine for Homeostasis Enterprise Governance.

This module provides comprehensive policy enforcement for healing actions,
ensuring all automated fixes comply with organizational policies and
regulatory requirements.

Features:
- Dynamic policy evaluation
- Context-aware policy application
- Policy versioning and history
- Real-time policy violations detection
- Integration with approval workflows
"""

import datetime
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .approval_workflow import get_workflow_engine
from .audit import get_audit_logger
from .compliance_reporting import ComplianceFramework, get_compliance_reporting
from .rbac import get_rbac_manager
from .user_management import get_user_management

logger = logging.getLogger(__name__)


class PolicyType(Enum):
    """Types of policies."""

    HEALING = "healing"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"
    CUSTOM = "custom"


class PolicyAction(Enum):
    """Policy enforcement actions."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    REQUIRE_REVIEW = "require_review"
    LOG_ONLY = "log_only"
    QUARANTINE = "quarantine"


class PolicyScope(Enum):
    """Policy scope levels."""

    GLOBAL = "global"
    ENVIRONMENT = "environment"
    SERVICE = "service"
    LANGUAGE = "language"
    ERROR_TYPE = "error_type"


class PolicyPriority(Enum):
    """Policy priority levels."""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFORMATIONAL = 5


@dataclass
class PolicyCondition:
    """Represents a policy condition."""

    condition_id: str
    field: str
    operator: str  # equals, not_equals, contains, regex, greater_than, less_than
    value: Union[str, int, float, List, Dict]
    case_sensitive: bool = True
    negate: bool = False


@dataclass
class PolicyRule:
    """Represents a policy rule."""

    rule_id: str
    name: str
    description: str
    conditions: List[PolicyCondition]
    action: PolicyAction
    condition_logic: str = "AND"  # AND, OR
    action_params: Dict = field(default_factory=dict)
    priority: PolicyPriority = PolicyPriority.MEDIUM
    metadata: Dict = field(default_factory=dict)


@dataclass
class Policy:
    """Represents a governance policy."""

    policy_id: str
    name: str
    description: str
    type: PolicyType
    scope: PolicyScope
    rules: List[PolicyRule]
    enabled: bool = True
    version: int = 1
    created_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )
    created_by: str = "system"
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class PolicyEvaluationContext:
    """Context for policy evaluation."""

    healing_action: str
    error_type: str
    error_details: Dict
    patch_content: str
    environment: str
    service_name: str
    language: str
    user_id: Optional[str] = None
    risk_level: str = "low"
    patch_size: int = 0
    affected_files: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class PolicyEvaluationResult:
    """Result of policy evaluation."""

    policy_id: str
    rule_id: Optional[str]
    action: PolicyAction
    reason: str
    violations: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class PolicyViolation:
    """Represents a policy violation."""

    violation_id: str
    policy_id: str
    rule_id: str
    context: PolicyEvaluationContext
    detected_at: str
    severity: str
    description: str
    remediation_required: bool = False
    remediation_status: str = "pending"


class PolicyEnforcementEngine:
    """
    Policy enforcement engine for governing healing actions.

    Ensures all automated fixes comply with organizational policies
    and regulatory requirements.
    """

    def __init__(self, config: Optional[Dict[Any, Any]] = None, storage_path: Optional[str] = None):
        """Initialize the policy enforcement engine.

        Args:
            config: Configuration dictionary
            storage_path: Path to store policy data
        """
        self.config = config or {}
        self.storage_path = Path(storage_path or "data/policies")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Get managers
        self.audit_logger = get_audit_logger(config)
        self.rbac_manager = get_rbac_manager(config)
        self.user_management = get_user_management(config)
        self.workflow_engine = get_workflow_engine(config)
        self.compliance_reporting = get_compliance_reporting(config)

        # Initialize stores
        self.policies: Dict[str, Policy] = {}
        self.policy_history: Dict[str, List[Policy]] = {}
        self.violations: Dict[str, PolicyViolation] = {}

        # Condition evaluators
        self.condition_evaluators: Dict[str, Callable] = {
            "equals": self._evaluate_equals,
            "not_equals": self._evaluate_not_equals,
            "contains": self._evaluate_contains,
            "regex": self._evaluate_regex,
            "greater_than": self._evaluate_greater_than,
            "less_than": self._evaluate_less_than,
            "in": self._evaluate_in,
            "not_in": self._evaluate_not_in,
            "starts_with": self._evaluate_starts_with,
            "ends_with": self._evaluate_ends_with,
        }

        # Load existing data
        self._load_policy_data()

        # Initialize default policies if none exist
        if not self.policies:
            self._create_default_policies()

    def evaluate_healing_action(
        self, context: PolicyEvaluationContext
    ) -> List[PolicyEvaluationResult]:
        """Evaluate a healing action against all applicable policies.

        Args:
            context: Evaluation context

        Returns:
            List of evaluation results
        """
        results = []
        applicable_policies = self._get_applicable_policies(context)

        for policy in applicable_policies:
            if not policy.enabled:
                continue

            result = self._evaluate_policy(policy, context)
            if result:
                results.append(result)

                # Log policy evaluation
                self.audit_logger.log_event(
                    event_type="policy_evaluation",
                    user=context.user_id or "system",
                    details={
                        "policy_id": policy.policy_id,
                        "action": result.action.value,
                        "healing_action": context.healing_action,
                        "service": context.service_name,
                    },
                )

                # Record violation if denied
                if result.action == PolicyAction.DENY:
                    self._record_violation(policy, result, context)

        # Sort by priority (most restrictive first)
        results.sort(key=lambda r: r.action.value)

        return results

    def enforce_healing_policy(
        self, context: PolicyEvaluationContext
    ) -> Tuple[bool, Optional[str], Dict]:
        """Enforce policies for a healing action.

        Args:
            context: Evaluation context

        Returns:
            Tuple of (allowed, reason, metadata)
        """
        results = self.evaluate_healing_action(context)

        if not results:
            # No policies apply, allow by default
            return True, None, {}

        # Check for any DENY actions
        for result in results:
            if result.action == PolicyAction.DENY:
                return (
                    False,
                    result.reason,
                    {"policy_id": result.policy_id, "violations": result.violations},
                )

        # Check for REQUIRE_APPROVAL actions
        approval_required = [
            r for r in results if r.action == PolicyAction.REQUIRE_APPROVAL
        ]
        if approval_required:
            # Initiate approval workflow
            metadata = {
                "policies": [r.policy_id for r in approval_required],
                "approval_reasons": [r.reason for r in approval_required],
            }
            return False, "Approval required", metadata

        # Check for QUARANTINE actions
        quarantine_required = [
            r for r in results if r.action == PolicyAction.QUARANTINE
        ]
        if quarantine_required:
            metadata = {
                "policies": [r.policy_id for r in quarantine_required],
                "quarantine_reasons": [r.reason for r in quarantine_required],
            }
            return False, "Quarantined for review", metadata

        # All policies allow or only require logging
        return True, None, {}

    def create_policy(
        self,
        name: str,
        description: str,
        type: PolicyType,
        scope: PolicyScope,
        rules: List[Dict],
        compliance_frameworks: Optional[List[ComplianceFramework]] = None,
        created_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new policy.

        Args:
            name: Policy name
            description: Policy description
            type: Policy type
            scope: Policy scope
            rules: List of rule configurations
            compliance_frameworks: Associated compliance frameworks
            created_by: User creating the policy

        Returns:
            Policy ID
        """
        policy_id = self._generate_policy_id()

        # Convert rule dicts to PolicyRule objects
        policy_rules = []
        for rule_config in rules:
            # Convert condition dicts to PolicyCondition objects
            conditions = []
            for cond_config in rule_config.get("conditions", []):
                condition = PolicyCondition(
                    condition_id=self._generate_condition_id(),
                    field=cond_config["field"],
                    operator=cond_config["operator"],
                    value=cond_config["value"],
                    case_sensitive=cond_config.get("case_sensitive", True),
                    negate=cond_config.get("negate", False),
                )
                conditions.append(condition)

            rule = PolicyRule(
                rule_id=self._generate_rule_id(),
                name=rule_config["name"],
                description=rule_config.get("description", ""),
                conditions=conditions,
                condition_logic=rule_config.get("condition_logic", "AND"),
                action=PolicyAction(rule_config["action"]),
                action_params=rule_config.get("action_params", {}),
                priority=PolicyPriority(
                    rule_config.get("priority", PolicyPriority.MEDIUM.value)
                ),
                metadata=rule_config.get("metadata", {}),
            )
            policy_rules.append(rule)

        # Create policy
        policy = Policy(
            policy_id=policy_id,
            name=name,
            description=description,
            type=type,
            scope=scope,
            rules=policy_rules,
            compliance_frameworks=compliance_frameworks or [],
            created_by=created_by,
            metadata=metadata or {},
        )

        self.policies[policy_id] = policy
        self._save_policy_data()

        # Log policy creation
        self.audit_logger.log_event(
            event_type="policy_created",
            user=created_by,
            details={
                "policy_id": policy_id,
                "name": name,
                "type": type.value,
                "scope": scope.value,
                "rules_count": len(policy_rules),
            },
        )

        return policy_id

    def update_policy(
        self, policy_id: str, updates: Dict, updated_by: str = "system"
    ) -> bool:
        """Update an existing policy.

        Args:
            policy_id: Policy ID
            updates: Fields to update
            updated_by: User updating the policy

        Returns:
            True if successful
        """
        policy = self.policies.get(policy_id)
        if not policy:
            raise ValueError(f"Policy {policy_id} not found")

        # Save current version to history
        if policy_id not in self.policy_history:
            self.policy_history[policy_id] = []
        self.policy_history[policy_id].append(policy)

        # Create new version
        new_policy = Policy(
            policy_id=policy_id,
            name=updates.get("name", policy.name),
            description=updates.get("description", policy.description),
            type=updates.get("type", policy.type),
            scope=updates.get("scope", policy.scope),
            rules=updates.get("rules", policy.rules),
            enabled=updates.get("enabled", policy.enabled),
            version=policy.version + 1,
            created_at=policy.created_at,
            updated_at=datetime.datetime.utcnow().isoformat(),
            created_by=policy.created_by,
            compliance_frameworks=updates.get(
                "compliance_frameworks", policy.compliance_frameworks
            ),
            tags=updates.get("tags", policy.tags),
            metadata=updates.get("metadata", policy.metadata),
        )

        self.policies[policy_id] = new_policy
        self._save_policy_data()

        # Log policy update
        self.audit_logger.log_event(
            event_type="policy_updated",
            user=updated_by,
            details={
                "policy_id": policy_id,
                "version": new_policy.version,
                "updates": list(updates.keys()),
            },
        )

        return True

    def disable_policy(
        self, policy_id: str, reason: str, disabled_by: str = "system"
    ) -> bool:
        """Disable a policy.

        Args:
            policy_id: Policy ID
            reason: Reason for disabling
            disabled_by: User disabling the policy

        Returns:
            True if successful
        """
        policy = self.policies.get(policy_id)
        if not policy:
            raise ValueError(f"Policy {policy_id} not found")

        policy.enabled = False
        policy.metadata["disabled_reason"] = reason
        policy.metadata["disabled_by"] = disabled_by
        policy.metadata["disabled_at"] = datetime.datetime.utcnow().isoformat()

        self._save_policy_data()

        # Log policy disable
        self.audit_logger.log_event(
            event_type="policy_disabled",
            user=disabled_by,
            details={"policy_id": policy_id, "reason": reason},
        )

        return True

    def get_policy_violations(
        self,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        policy_id: Optional[str] = None,
    ) -> List[PolicyViolation]:
        """Get policy violations.

        Args:
            start_date: Start date filter
            end_date: End date filter
            policy_id: Filter by policy ID

        Returns:
            List of violations
        """
        violations = []

        for violation in self.violations.values():
            # Apply filters
            if policy_id and violation.policy_id != policy_id:
                continue

            violation_date = datetime.datetime.fromisoformat(violation.detected_at)

            if start_date and violation_date < start_date:
                continue

            if end_date and violation_date > end_date:
                continue

            violations.append(violation)

        return sorted(violations, key=lambda v: v.detected_at, reverse=True)

    def get_policy_metrics(self) -> Dict[str, Any]:
        """Get policy enforcement metrics.

        Returns:
            Policy metrics
        """
        total_evaluations = 0
        evaluations_by_action = {action.value: 0 for action in PolicyAction}

        # Get evaluation counts from audit logs
        # In a real implementation, this would query the audit log system

        # Violation metrics
        total_violations = len(self.violations)
        open_violations = len(
            [v for v in self.violations.values() if v.remediation_status == "pending"]
        )

        # Policy metrics
        enabled_policies = len([p for p in self.policies.values() if p.enabled])
        policies_by_type: Dict[str, int] = {}
        for policy in self.policies.values():
            if policy.enabled:
                policies_by_type[policy.type.value] = (
                    policies_by_type.get(policy.type.value, 0) + 1
                )

        return {
            "total_policies": len(self.policies),
            "enabled_policies": enabled_policies,
            "policies_by_type": policies_by_type,
            "total_violations": total_violations,
            "open_violations": open_violations,
            "evaluations": {
                "total": total_evaluations,
                "by_action": evaluations_by_action,
            },
        }

    def _get_applicable_policies(
        self, context: PolicyEvaluationContext
    ) -> List[Policy]:
        """Get policies applicable to the context."""
        applicable = []

        for policy in self.policies.values():
            if not policy.enabled:
                continue

            # Check scope
            if policy.scope == PolicyScope.GLOBAL:
                applicable.append(policy)
            elif (
                policy.scope == PolicyScope.ENVIRONMENT
                and policy.metadata.get("environment") == context.environment
            ):
                applicable.append(policy)
            elif (
                policy.scope == PolicyScope.SERVICE
                and policy.metadata.get("service") == context.service_name
            ):
                applicable.append(policy)
            elif (
                policy.scope == PolicyScope.LANGUAGE
                and policy.metadata.get("language") == context.language
            ):
                applicable.append(policy)
            elif (
                policy.scope == PolicyScope.ERROR_TYPE
                and policy.metadata.get("error_type") == context.error_type
            ):
                applicable.append(policy)

        return applicable

    def _evaluate_policy(
        self, policy: Policy, context: PolicyEvaluationContext
    ) -> Optional[PolicyEvaluationResult]:
        """Evaluate a single policy."""
        for rule in policy.rules:
            if self._evaluate_rule(rule, context):
                # Rule matched, return result
                violations = []

                # Collect specific violations
                if rule.action == PolicyAction.DENY:
                    for condition in rule.conditions:
                        violations.append(
                            f"{condition.field} {condition.operator} {condition.value}"
                        )

                return PolicyEvaluationResult(
                    policy_id=policy.policy_id,
                    rule_id=rule.rule_id,
                    action=rule.action,
                    reason=f"{policy.name}: {rule.description}",
                    violations=violations,
                    metadata=rule.action_params,
                )

        return None

    def _evaluate_rule(
        self, rule: PolicyRule, context: PolicyEvaluationContext
    ) -> bool:
        """Evaluate a policy rule."""
        results = []

        for condition in rule.conditions:
            result = self._evaluate_condition(condition, context)
            results.append(result)

        # Apply condition logic
        if rule.condition_logic == "AND":
            return all(results)
        elif rule.condition_logic == "OR":
            return any(results)
        else:
            return False

    def _evaluate_condition(
        self, condition: PolicyCondition, context: PolicyEvaluationContext
    ) -> bool:
        """Evaluate a single condition."""
        # Get field value from context
        field_value = self._get_field_value(condition.field, context)

        # Get evaluator
        evaluator = self.condition_evaluators.get(condition.operator)
        if not evaluator:
            logger.warning(f"Unknown operator: {condition.operator}")
            return False

        # Evaluate condition
        result = bool(evaluator(field_value, condition.value, condition.case_sensitive))

        # Apply negation if needed
        if condition.negate:
            result = not result

        return result

    def _get_field_value(self, field: str, context: PolicyEvaluationContext) -> Any:
        """Get field value from context."""
        # Support dot notation for nested fields
        parts = field.split(".")
        value = context

        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value

    def _evaluate_equals(
        self, field_value: Any, condition_value: Any, case_sensitive: bool
    ) -> bool:
        """Evaluate equals condition."""
        if (
            isinstance(field_value, str)
            and isinstance(condition_value, str)
            and not case_sensitive
        ):
            return field_value.lower() == condition_value.lower()
        return bool(field_value == condition_value)

    def _evaluate_not_equals(
        self, field_value: Any, condition_value: Any, case_sensitive: bool
    ) -> bool:
        """Evaluate not equals condition."""
        return not self._evaluate_equals(field_value, condition_value, case_sensitive)

    def _evaluate_contains(
        self, field_value: Any, condition_value: Any, case_sensitive: bool
    ) -> bool:
        """Evaluate contains condition."""
        if not isinstance(field_value, str) or not isinstance(condition_value, str):
            return False

        if not case_sensitive:
            return condition_value.lower() in field_value.lower()
        return condition_value in field_value

    def _evaluate_regex(
        self, field_value: Any, condition_value: Any, case_sensitive: bool
    ) -> bool:
        """Evaluate regex condition."""
        if not isinstance(field_value, str) or not isinstance(condition_value, str):
            return False

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            return bool(re.search(condition_value, field_value, flags))
        except re.error:
            logger.error(f"Invalid regex pattern: {condition_value}")
            return False

    def _evaluate_greater_than(
        self, field_value: Any, condition_value: Any, case_sensitive: bool
    ) -> bool:
        """Evaluate greater than condition."""
        try:
            return float(field_value) > float(condition_value)
        except (TypeError, ValueError):
            return False

    def _evaluate_less_than(
        self, field_value: Any, condition_value: Any, case_sensitive: bool
    ) -> bool:
        """Evaluate less than condition."""
        try:
            return float(field_value) < float(condition_value)
        except (TypeError, ValueError):
            return False

    def _evaluate_in(
        self, field_value: Any, condition_value: Any, case_sensitive: bool
    ) -> bool:
        """Evaluate in condition."""
        if not isinstance(condition_value, (list, tuple, set)):
            return False

        if isinstance(field_value, str) and not case_sensitive:
            field_value = field_value.lower()
            condition_value = [
                v.lower() if isinstance(v, str) else v for v in condition_value
            ]

        return field_value in condition_value

    def _evaluate_not_in(
        self, field_value: Any, condition_value: Any, case_sensitive: bool
    ) -> bool:
        """Evaluate not in condition."""
        return not self._evaluate_in(field_value, condition_value, case_sensitive)

    def _evaluate_starts_with(
        self, field_value: Any, condition_value: Any, case_sensitive: bool
    ) -> bool:
        """Evaluate starts with condition."""
        if not isinstance(field_value, str) or not isinstance(condition_value, str):
            return False

        if not case_sensitive:
            return field_value.lower().startswith(condition_value.lower())
        return field_value.startswith(condition_value)

    def _evaluate_ends_with(
        self, field_value: Any, condition_value: Any, case_sensitive: bool
    ) -> bool:
        """Evaluate ends with condition."""
        if not isinstance(field_value, str) or not isinstance(condition_value, str):
            return False

        if not case_sensitive:
            return field_value.lower().endswith(condition_value.lower())
        return field_value.endswith(condition_value)

    def _record_violation(
        self,
        policy: Policy,
        result: PolicyEvaluationResult,
        context: PolicyEvaluationContext,
    ):
        """Record a policy violation."""
        violation_id = self._generate_violation_id()

        # Determine severity based on policy priority
        severity_map = {
            PolicyPriority.CRITICAL: "critical",
            PolicyPriority.HIGH: "high",
            PolicyPriority.MEDIUM: "medium",
            PolicyPriority.LOW: "low",
            PolicyPriority.INFORMATIONAL: "info",
        }

        # Find the rule that triggered the violation
        rule = next((r for r in policy.rules if r.rule_id == result.rule_id), None)
        severity = severity_map.get(
            rule.priority if rule else PolicyPriority.MEDIUM, "medium"
        )

        violation = PolicyViolation(
            violation_id=violation_id,
            policy_id=policy.policy_id,
            rule_id=result.rule_id or "",
            context=context,
            detected_at=datetime.datetime.utcnow().isoformat(),
            severity=severity,
            description=result.reason,
            remediation_required=severity in ["critical", "high"],
        )

        self.violations[violation_id] = violation
        self._save_policy_data()

        # Add evidence to compliance reporting
        self.compliance_reporting.add_evidence(
            control_id="policy_enforcement",
            evidence_type="policy_violation",
            description=f"Policy violation: {result.reason}",
            data={
                "violation_id": violation_id,
                "policy_id": policy.policy_id,
                "severity": severity,
                "context": {
                    "service": context.service_name,
                    "environment": context.environment,
                    "error_type": context.error_type,
                },
            },
        )

    def _create_default_policies(self):
        """Create default policies."""
        # Security policy - No credential exposure
        security_rules = [
            {
                "name": "No Hardcoded Credentials",
                "description": "Prevent patches containing hardcoded credentials",
                "conditions": [
                    {
                        "field": "patch_content",
                        "operator": "regex",
                        "value": r'(password|api_key|secret|token)\s*=\s*["\'][^"\']+["\']',
                        "case_sensitive": False,
                    }
                ],
                "action": "deny",
                "priority": PolicyPriority.CRITICAL.value,
            },
            {
                "name": "No SQL Injection Risk",
                "description": "Prevent patches with SQL injection vulnerabilities",
                "conditions": [
                    {
                        "field": "patch_content",
                        "operator": "regex",
                        "value": r'(SELECT|INSERT|UPDATE|DELETE).*\+.*[\'"]',
                        "case_sensitive": False,
                    }
                ],
                "action": "require_review",
                "priority": PolicyPriority.HIGH.value,
            },
        ]

        self.create_policy(
            name="Security Best Practices",
            description="Enforce security best practices in patches",
            type=PolicyType.SECURITY,
            scope=PolicyScope.GLOBAL,
            rules=security_rules,
            compliance_frameworks=[
                ComplianceFramework.SOC2,
                ComplianceFramework.ISO_27001,
            ],
        )

        # Production protection policy
        prod_rules = [
            {
                "name": "Production Database Protection",
                "description": "Require approval for database changes in production",
                "conditions": [
                    {
                        "field": "environment",
                        "operator": "equals",
                        "value": "production",
                    },
                    {
                        "field": "patch_content",
                        "operator": "regex",
                        "value": r"(ALTER|DROP|TRUNCATE|DELETE)",
                        "case_sensitive": False,
                    },
                ],
                "action": "require_approval",
                "priority": PolicyPriority.HIGH.value,
            },
            {
                "name": "Production Large Patch Review",
                "description": "Require review for large patches in production",
                "conditions": [
                    {
                        "field": "environment",
                        "operator": "equals",
                        "value": "production",
                    },
                    {"field": "patch_size", "operator": "greater_than", "value": 500},
                ],
                "action": "require_review",
                "priority": PolicyPriority.MEDIUM.value,
            },
        ]

        self.create_policy(
            name="Production Environment Protection",
            description="Additional safeguards for production environment",
            type=PolicyType.OPERATIONAL,
            scope=PolicyScope.ENVIRONMENT,
            rules=prod_rules,
            metadata={"environment": "production"},
        )

        # Compliance policy - HIPAA
        hipaa_rules = [
            {
                "name": "PHI Data Protection",
                "description": "Prevent exposure of PHI in patches",
                "conditions": [
                    {
                        "field": "service_name",
                        "operator": "in",
                        "value": [
                            "patient_service",
                            "medical_records",
                            "healthcare_api",
                        ],
                    },
                    {
                        "field": "patch_content",
                        "operator": "regex",
                        "value": r"(ssn|dob|patient|diagnosis|medical)",
                        "case_sensitive": False,
                    },
                ],
                "action": "require_approval",
                "priority": PolicyPriority.CRITICAL.value,
                "action_params": {"approval_template": "hipaa_compliance_review"},
            }
        ]

        self.create_policy(
            name="HIPAA Compliance",
            description="Ensure HIPAA compliance for healthcare services",
            type=PolicyType.COMPLIANCE,
            scope=PolicyScope.SERVICE,
            rules=hipaa_rules,
            compliance_frameworks=[ComplianceFramework.HIPAA],
            metadata={
                "services": ["patient_service", "medical_records", "healthcare_api"]
            },
        )

        # Language-specific policy - Python
        python_rules = [
            {
                "name": "Python Dangerous Functions",
                "description": "Prevent use of dangerous Python functions",
                "conditions": [
                    {"field": "language", "operator": "equals", "value": "python"},
                    {
                        "field": "patch_content",
                        "operator": "regex",
                        "value": r"(exec|eval|__import__|compile)\s*\(",
                        "case_sensitive": True,
                    },
                ],
                "action": "deny",
                "priority": PolicyPriority.HIGH.value,
            }
        ]

        self.create_policy(
            name="Python Security Policy",
            description="Python-specific security policies",
            type=PolicyType.SECURITY,
            scope=PolicyScope.LANGUAGE,
            rules=python_rules,
            metadata={"language": "python"},
        )

    def _generate_policy_id(self) -> str:
        """Generate unique policy ID."""
        import uuid

        return f"pol_{uuid.uuid4().hex[:12]}"

    def _generate_rule_id(self) -> str:
        """Generate unique rule ID."""
        import uuid

        return f"rul_{uuid.uuid4().hex[:8]}"

    def _generate_condition_id(self) -> str:
        """Generate unique condition ID."""
        import uuid

        return f"cnd_{uuid.uuid4().hex[:8]}"

    def _generate_violation_id(self) -> str:
        """Generate unique violation ID."""
        import uuid

        return f"vio_{uuid.uuid4().hex[:12]}"

    def _load_policy_data(self):
        """Load policy data from storage."""
        # Load policies
        policies_file = self.storage_path / "policies.json"
        if policies_file.exists():
            with open(policies_file, "r") as f:
                policies_data = json.load(f)
                for policy_data in policies_data:
                    # Reconstruct rules
                    rules = []
                    for rule_data in policy_data["rules"]:
                        # Reconstruct conditions
                        conditions = []
                        for cond_data in rule_data["conditions"]:
                            condition = PolicyCondition(**cond_data)
                            conditions.append(condition)

                        rule = PolicyRule(
                            rule_id=rule_data["rule_id"],
                            name=rule_data["name"],
                            description=rule_data["description"],
                            conditions=conditions,
                            condition_logic=rule_data["condition_logic"],
                            action=PolicyAction(rule_data["action"]),
                            action_params=rule_data.get("action_params", {}),
                            priority=PolicyPriority(
                                rule_data.get("priority", PolicyPriority.MEDIUM.value)
                            ),
                            metadata=rule_data.get("metadata", {}),
                        )
                        rules.append(rule)

                    policy = Policy(
                        policy_id=policy_data["policy_id"],
                        name=policy_data["name"],
                        description=policy_data["description"],
                        type=PolicyType(policy_data["type"]),
                        scope=PolicyScope(policy_data["scope"]),
                        rules=rules,
                        enabled=policy_data.get("enabled", True),
                        version=policy_data.get("version", 1),
                        created_at=policy_data["created_at"],
                        updated_at=policy_data["updated_at"],
                        created_by=policy_data.get("created_by", "system"),
                        compliance_frameworks=[
                            ComplianceFramework(f)
                            for f in policy_data.get("compliance_frameworks", [])
                        ],
                        tags=policy_data.get("tags", []),
                        metadata=policy_data.get("metadata", {}),
                    )
                    self.policies[policy.policy_id] = policy

        # Load violations
        violations_file = self.storage_path / "violations.json"
        if violations_file.exists():
            with open(violations_file, "r") as f:
                violations_data = json.load(f)
                for violation_data in violations_data:
                    # Reconstruct context
                    context_data = violation_data["context"]
                    context = PolicyEvaluationContext(**context_data)

                    violation = PolicyViolation(
                        violation_id=violation_data["violation_id"],
                        policy_id=violation_data["policy_id"],
                        rule_id=violation_data["rule_id"],
                        context=context,
                        detected_at=violation_data["detected_at"],
                        severity=violation_data["severity"],
                        description=violation_data["description"],
                        remediation_required=violation_data.get(
                            "remediation_required", False
                        ),
                        remediation_status=violation_data.get(
                            "remediation_status", "pending"
                        ),
                    )
                    self.violations[violation.violation_id] = violation

    def _save_policy_data(self):
        """Save policy data to storage."""
        # Save policies
        policies_data = []
        for policy in self.policies.values():
            # Convert rules to dict
            rules_data = []
            for rule in policy.rules:
                # Convert conditions to dict
                conditions_data = []
                for condition in rule.conditions:
                    cond_dict = {
                        "condition_id": condition.condition_id,
                        "field": condition.field,
                        "operator": condition.operator,
                        "value": condition.value,
                        "case_sensitive": condition.case_sensitive,
                        "negate": condition.negate,
                    }
                    conditions_data.append(cond_dict)

                rule_dict = {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "conditions": conditions_data,
                    "condition_logic": rule.condition_logic,
                    "action": rule.action.value,
                    "action_params": rule.action_params,
                    "priority": rule.priority.value,
                    "metadata": rule.metadata,
                }
                rules_data.append(rule_dict)

            policy_dict = {
                "policy_id": policy.policy_id,
                "name": policy.name,
                "description": policy.description,
                "type": policy.type.value,
                "scope": policy.scope.value,
                "rules": rules_data,
                "enabled": policy.enabled,
                "version": policy.version,
                "created_at": policy.created_at,
                "updated_at": policy.updated_at,
                "created_by": policy.created_by,
                "compliance_frameworks": [
                    f.value for f in policy.compliance_frameworks
                ],
                "tags": policy.tags,
                "metadata": policy.metadata,
            }
            policies_data.append(policy_dict)

        with open(self.storage_path / "policies.json", "w") as f:
            json.dump(policies_data, f, indent=2)

        # Save violations
        violations_data = []
        for violation in self.violations.values():
            # Convert context to dict
            context_dict = {
                "healing_action": violation.context.healing_action,
                "error_type": violation.context.error_type,
                "error_details": violation.context.error_details,
                "patch_content": violation.context.patch_content,
                "environment": violation.context.environment,
                "service_name": violation.context.service_name,
                "language": violation.context.language,
                "user_id": violation.context.user_id,
                "risk_level": violation.context.risk_level,
                "patch_size": violation.context.patch_size,
                "affected_files": violation.context.affected_files,
                "metadata": violation.context.metadata,
            }

            violation_dict = {
                "violation_id": violation.violation_id,
                "policy_id": violation.policy_id,
                "rule_id": violation.rule_id,
                "context": context_dict,
                "detected_at": violation.detected_at,
                "severity": violation.severity,
                "description": violation.description,
                "remediation_required": violation.remediation_required,
                "remediation_status": violation.remediation_status,
            }
            violations_data.append(violation_dict)

        with open(self.storage_path / "violations.json", "w") as f:
            json.dump(violations_data, f, indent=2)


# Singleton instance
_policy_engine = None


def get_policy_engine(config: Optional[Dict[Any, Any]] = None) -> PolicyEnforcementEngine:
    """Get or create the singleton PolicyEnforcementEngine instance."""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = PolicyEnforcementEngine(config)
    return _policy_engine
