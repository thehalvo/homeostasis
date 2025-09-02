"""
Governance and Access Control Manager for Homeostasis LLM Integration.

This module provides role-based access control, approval workflows, and governance
for LLM-generated patches to ensure compliance and security in enterprise environments.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..security.security_config import SecurityConfig, get_security_config


logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles in the system."""
    ADMIN = "admin"
    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    OPERATOR = "operator"
    VIEWER = "viewer"


class PatchRiskLevel(Enum):
    """Risk levels for patches."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(Enum):
    """Approval status for patches."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class PatchCategory(Enum):
    """Categories of patches for governance."""
    SECURITY_FIX = "security_fix"
    BUG_FIX = "bug_fix"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    FEATURE_ADDITION = "feature_addition"
    REFACTORING = "refactoring"
    CONFIGURATION_CHANGE = "configuration_change"
    DEPENDENCY_UPDATE = "dependency_update"


@dataclass
class User:
    """User in the governance system."""
    user_id: str
    username: str
    email: str
    roles: List[UserRole]
    permissions: Set[str] = field(default_factory=set)
    is_active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_login: Optional[str] = None


@dataclass
class ApprovalRequest:
    """Request for patch approval."""
    request_id: str
    context_id: str
    patch_content: str
    patch_category: PatchCategory
    risk_level: PatchRiskLevel
    requested_by: str
    requested_at: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    expires_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernancePolicy:
    """Governance policy for patch management."""
    policy_id: str
    name: str
    description: str
    patch_categories: List[PatchCategory]
    risk_levels: List[PatchRiskLevel]
    required_approver_roles: List[UserRole]
    min_approvers: int = 1
    max_approval_time_hours: int = 24
    allow_self_approval: bool = False
    require_multi_stage_approval: bool = False
    compliance_frameworks: List[str] = field(default_factory=list)
    is_active: bool = True


class GovernanceManager:
    """
    Governance and access control manager for LLM patch generation.
    
    Features:
    - Role-based access control
    - Patch approval workflows
    - Risk assessment and categorization
    - Compliance policy enforcement
    - Audit logging
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None, storage_dir: Optional[Path] = None):
        """Initialize the governance manager."""
        self.config = config or get_security_config()
        self.storage_dir = storage_dir or Path.cwd() / "governance"
        self.storage_dir.mkdir(exist_ok=True)
        
        # In-memory stores (in production, these would be backed by a database)
        self.users: Dict[str, User] = {}
        self.approval_requests: Dict[str, ApprovalRequest] = {}
        self.policies: Dict[str, GovernancePolicy] = {}
        
        # Load data from storage
        self._load_governance_data()
        
        # Initialize default policies if none exist
        if not self.policies:
            self._create_default_policies()
        
        logger.info(f"Governance Manager initialized with {len(self.policies)} policies")
    
    def register_user(self, username: str, email: str, roles: List[UserRole]) -> str:
        """
        Register a new user in the governance system.
        
        Args:
            username: Username
            email: Email address
            roles: List of roles for the user
            
        Returns:
            User ID
        """
        user_id = str(uuid.uuid4())
        
        # Generate permissions based on roles
        permissions = self._generate_permissions_for_roles(roles)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles,
            permissions=permissions
        )
        
        self.users[user_id] = user
        self._save_users()
        
        logger.info(f"Registered user {username} with roles: {[r.value for r in roles]}")
        return user_id
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """
        Check if a user has a specific permission.
        
        Args:
            user_id: User ID
            permission: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return False
        
        return permission in user.permissions
    
    def assess_patch_risk(self, patch_content: str, context_metadata: Dict[str, Any]) -> Tuple[PatchRiskLevel, PatchCategory]:
        """
        Assess the risk level and category of a patch.
        
        Args:
            patch_content: The patch content
            context_metadata: Metadata about the error context
            
        Returns:
            Tuple of (risk_level, category)
        """
        # Analyze patch content for risk indicators
        risk_score = 0
        category = PatchCategory.BUG_FIX  # Default
        
        # Check for high-risk patterns
        high_risk_patterns = [
            r'(?i)(security|auth|password|token|credential)',
            r'(?i)(admin|root|privilege)',
            r'(?i)(database|sql|query)',
            r'(?i)(network|socket|connection)',
            r'(?i)(file.*delete|rm.*-rf|unlink)',
            r'(?i)(exec|eval|system|subprocess)',
        ]
        
        for pattern in high_risk_patterns:
            import re
            if re.search(pattern, patch_content):
                risk_score += 3
                if 'security' in pattern or 'auth' in pattern:
                    category = PatchCategory.SECURITY_FIX
        
        # Check for medium-risk patterns
        medium_risk_patterns = [
            r'(?i)(config|settings|environment)',
            r'(?i)(import|dependency|require)',
            r'(?i)(performance|optimization)',
        ]
        
        for pattern in medium_risk_patterns:
            if re.search(pattern, patch_content):
                risk_score += 2
                if 'config' in pattern:
                    category = PatchCategory.CONFIGURATION_CHANGE
                elif 'performance' in pattern:
                    category = PatchCategory.PERFORMANCE_IMPROVEMENT
                elif 'dependency' in pattern:
                    category = PatchCategory.DEPENDENCY_UPDATE
        
        # Determine risk level based on score
        if risk_score >= 6:
            risk_level = PatchRiskLevel.CRITICAL
        elif risk_score >= 4:
            risk_level = PatchRiskLevel.HIGH
        elif risk_score >= 2:
            risk_level = PatchRiskLevel.MEDIUM
        else:
            risk_level = PatchRiskLevel.LOW
        
        # Consider error context metadata
        error_category = context_metadata.get("error_classification", {}).get("category", "")
        if error_category == "security":
            risk_level = PatchRiskLevel.HIGH
            category = PatchCategory.SECURITY_FIX
        
        return risk_level, category
    
    def request_patch_approval(self, 
                             context_id: str,
                             patch_content: str,
                             requested_by: str,
                             context_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Request approval for a patch.
        
        Args:
            context_id: Context ID for the patch
            patch_content: The patch content
            requested_by: User ID of the requestor
            context_metadata: Metadata about the error context
            
        Returns:
            Approval request ID
        """
        # Assess patch risk and category
        risk_level, category = self.assess_patch_risk(patch_content, context_metadata or {})
        
        # Find applicable policy
        policy = self._find_applicable_policy(category, risk_level)
        if not policy:
            raise ValueError(f"No applicable policy found for category {category.value} and risk {risk_level.value}")
        
        # Check if approval is required
        if not self._requires_approval(policy, requested_by):
            # Auto-approve for low-risk patches or if user has self-approval rights
            return self._auto_approve_patch(context_id, patch_content, requested_by, category, risk_level)
        
        # Create approval request
        request_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=policy.max_approval_time_hours)
        
        approval_request = ApprovalRequest(
            request_id=request_id,
            context_id=context_id,
            patch_content=patch_content,
            patch_category=category,
            risk_level=risk_level,
            requested_by=requested_by,
            requested_at=datetime.now().isoformat(),
            expires_at=expires_at.isoformat(),
            metadata={
                "policy_id": policy.policy_id,
                "context_metadata": context_metadata,
                "required_approver_roles": [r.value for r in policy.required_approver_roles],
                "min_approvers": policy.min_approvers
            }
        )
        
        self.approval_requests[request_id] = approval_request
        self._save_approval_requests()
        
        # Send notifications to potential approvers
        self._notify_approvers(approval_request, policy)
        
        logger.info(f"Created approval request {request_id} for patch in context {context_id}")
        return request_id
    
    def approve_patch(self, request_id: str, approver_id: str, comments: Optional[str] = None) -> bool:
        """
        Approve a patch request.
        
        Args:
            request_id: Approval request ID
            approver_id: User ID of the approver
            comments: Optional approval comments
            
        Returns:
            True if approved successfully, False otherwise
        """
        request = self.approval_requests.get(request_id)
        if not request:
            raise ValueError(f"Approval request {request_id} not found")
        
        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Request {request_id} is not pending (status: {request.status.value})")
        
        # Check if request has expired
        if request.expires_at and datetime.fromisoformat(request.expires_at) < datetime.now():
            request.status = ApprovalStatus.EXPIRED
            self._save_approval_requests()
            raise ValueError(f"Request {request_id} has expired")
        
        # Check if approver has permission
        policy = self.policies.get(request.metadata.get("policy_id", ""))
        if not policy:
            raise ValueError(f"Policy not found for request {request_id}")
        
        approver = self.users.get(approver_id)
        if not approver:
            raise ValueError(f"Approver {approver_id} not found")
        
        # Check if approver has required role
        has_required_role = any(role in policy.required_approver_roles for role in approver.roles)
        if not has_required_role:
            raise ValueError("Approver does not have required role for this patch category")
        
        # Check self-approval policy
        if not policy.allow_self_approval and approver_id == request.requested_by:
            raise ValueError("Self-approval is not allowed for this patch category")
        
        # Approve the request
        request.status = ApprovalStatus.APPROVED
        request.approved_by = approver_id
        request.approved_at = datetime.now().isoformat()
        if comments:
            request.metadata["approval_comments"] = comments
        
        self._save_approval_requests()
        
        logger.info(f"Approved patch request {request_id} by user {approver_id}")
        return True
    
    def reject_patch(self, request_id: str, rejector_id: str, reason: str) -> bool:
        """
        Reject a patch request.
        
        Args:
            request_id: Approval request ID
            rejector_id: User ID of the rejector
            reason: Rejection reason
            
        Returns:
            True if rejected successfully, False otherwise
        """
        request = self.approval_requests.get(request_id)
        if not request:
            raise ValueError(f"Approval request {request_id} not found")
        
        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Request {request_id} is not pending")
        
        # Check if rejector has permission (same as approval permission)
        policy = self.policies.get(request.metadata.get("policy_id", ""))
        if not policy:
            raise ValueError(f"Policy not found for request {request_id}")
        
        rejector = self.users.get(rejector_id)
        if not rejector:
            raise ValueError(f"Rejector {rejector_id} not found")
        
        has_required_role = any(role in policy.required_approver_roles for role in rejector.roles)
        if not has_required_role:
            raise ValueError("User does not have required role to reject this patch")
        
        # Reject the request
        request.status = ApprovalStatus.REJECTED
        request.rejection_reason = reason
        request.metadata["rejected_by"] = rejector_id
        request.metadata["rejected_at"] = datetime.now().isoformat()
        
        self._save_approval_requests()
        
        logger.info(f"Rejected patch request {request_id} by user {rejector_id}: {reason}")
        return True
    
    def get_pending_approvals(self, user_id: Optional[str] = None) -> List[ApprovalRequest]:
        """
        Get pending approval requests, optionally filtered by user.
        
        Args:
            user_id: Optional user ID to filter by approver permissions
            
        Returns:
            List of pending approval requests
        """
        pending_requests = [
            req for req in self.approval_requests.values()
            if req.status == ApprovalStatus.PENDING
        ]
        
        # Filter by user permissions if specified
        if user_id:
            user = self.users.get(user_id)
            if user:
                filtered_requests = []
                for req in pending_requests:
                    policy = self.policies.get(req.metadata.get("policy_id", ""))
                    if policy and any(role in policy.required_approver_roles for role in user.roles):
                        filtered_requests.append(req)
                pending_requests = filtered_requests
        
        return pending_requests
    
    def get_governance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of governance activities.
        
        Returns:
            Governance summary
        """
        total_requests = len(self.approval_requests)
        pending_requests = len([r for r in self.approval_requests.values() if r.status == ApprovalStatus.PENDING])
        approved_requests = len([r for r in self.approval_requests.values() if r.status == ApprovalStatus.APPROVED])
        rejected_requests = len([r for r in self.approval_requests.values() if r.status == ApprovalStatus.REJECTED])
        
        return {
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "total_policies": len(self.policies),
            "active_policies": len([p for p in self.policies.values() if p.is_active]),
            "approval_requests": {
                "total": total_requests,
                "pending": pending_requests,
                "approved": approved_requests,
                "rejected": rejected_requests
            },
            "recent_activity": self._get_recent_activity()
        }
    
    def _generate_permissions_for_roles(self, roles: List[UserRole]) -> Set[str]:
        """Generate permissions based on user roles."""
        permissions = set()
        
        for role in roles:
            if role == UserRole.ADMIN:
                permissions.update([
                    "patch.approve", "patch.reject", "patch.request",
                    "user.manage", "policy.manage", "system.admin"
                ])
            elif role == UserRole.DEVELOPER:
                permissions.update([
                    "patch.request", "patch.view"
                ])
            elif role == UserRole.REVIEWER:
                permissions.update([
                    "patch.approve", "patch.reject", "patch.view"
                ])
            elif role == UserRole.OPERATOR:
                permissions.update([
                    "patch.request", "patch.view", "system.operate"
                ])
            elif role == UserRole.VIEWER:
                permissions.update([
                    "patch.view"
                ])
        
        return permissions
    
    def _find_applicable_policy(self, category: PatchCategory, risk_level: PatchRiskLevel) -> Optional[GovernancePolicy]:
        """Find the applicable governance policy for a patch."""
        for policy in self.policies.values():
            if (policy.is_active and 
                category in policy.patch_categories and 
                risk_level in policy.risk_levels):
                return policy
        return None
    
    def _requires_approval(self, policy: GovernancePolicy, user_id: str) -> bool:
        """Check if approval is required based on policy and user."""
        # For now, all patches require approval unless it's a low-risk patch and user has admin role
        user = self.users.get(user_id)
        if user and UserRole.ADMIN in user.roles and policy.allow_self_approval:
            return False
        return True
    
    def _auto_approve_patch(self, context_id: str, patch_content: str, requested_by: str, 
                          category: PatchCategory, risk_level: PatchRiskLevel) -> str:
        """Auto-approve a patch that doesn't require manual approval."""
        request_id = str(uuid.uuid4())
        
        approval_request = ApprovalRequest(
            request_id=request_id,
            context_id=context_id,
            patch_content=patch_content,
            patch_category=category,
            risk_level=risk_level,
            requested_by=requested_by,
            requested_at=datetime.now().isoformat(),
            status=ApprovalStatus.APPROVED,
            approved_by="system",
            approved_at=datetime.now().isoformat(),
            metadata={"auto_approved": True}
        )
        
        self.approval_requests[request_id] = approval_request
        self._save_approval_requests()
        
        logger.info(f"Auto-approved patch request {request_id} for context {context_id}")
        return request_id
    
    def _notify_approvers(self, request: ApprovalRequest, policy: GovernancePolicy):
        """Send notifications to potential approvers."""
        # In a real implementation, this would send emails/notifications
        approver_users = [
            user for user in self.users.values()
            if any(role in policy.required_approver_roles for role in user.roles)
        ]
        
        logger.info(f"Notifying {len(approver_users)} potential approvers for request {request.request_id}")
    
    def _create_default_policies(self):
        """Create default governance policies."""
        # High-risk security patches
        security_policy = GovernancePolicy(
            policy_id="security_high_risk",
            name="High Risk Security Patches",
            description="Patches that affect security, authentication, or authorization",
            patch_categories=[PatchCategory.SECURITY_FIX],
            risk_levels=[PatchRiskLevel.HIGH, PatchRiskLevel.CRITICAL],
            required_approver_roles=[UserRole.ADMIN, UserRole.REVIEWER],
            min_approvers=2,
            max_approval_time_hours=4,
            allow_self_approval=False,
            require_multi_stage_approval=True
        )
        
        # Regular bug fixes
        bugfix_policy = GovernancePolicy(
            policy_id="bugfix_standard",
            name="Standard Bug Fixes",
            description="Regular bug fixes and improvements",
            patch_categories=[PatchCategory.BUG_FIX, PatchCategory.PERFORMANCE_IMPROVEMENT],
            risk_levels=[PatchRiskLevel.LOW, PatchRiskLevel.MEDIUM],
            required_approver_roles=[UserRole.REVIEWER, UserRole.ADMIN],
            min_approvers=1,
            max_approval_time_hours=24,
            allow_self_approval=True
        )
        
        # Configuration changes
        config_policy = GovernancePolicy(
            policy_id="config_changes",
            name="Configuration Changes",
            description="Changes to configuration files and settings",
            patch_categories=[PatchCategory.CONFIGURATION_CHANGE],
            risk_levels=[PatchRiskLevel.LOW, PatchRiskLevel.MEDIUM, PatchRiskLevel.HIGH],
            required_approver_roles=[UserRole.OPERATOR, UserRole.ADMIN],
            min_approvers=1,
            max_approval_time_hours=12,
            allow_self_approval=False
        )
        
        self.policies = {
            security_policy.policy_id: security_policy,
            bugfix_policy.policy_id: bugfix_policy,
            config_policy.policy_id: config_policy
        }
        
        self._save_policies()
        logger.info("Created default governance policies")
    
    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent governance activity."""
        recent_requests = sorted(
            self.approval_requests.values(),
            key=lambda r: r.requested_at,
            reverse=True
        )[:10]
        
        return [
            {
                "request_id": req.request_id,
                "category": req.patch_category.value,
                "risk_level": req.risk_level.value,
                "status": req.status.value,
                "requested_at": req.requested_at
            }
            for req in recent_requests
        ]
    
    def _load_governance_data(self):
        """Load governance data from storage."""
        try:
            # Load users
            users_file = self.storage_dir / "users.json"
            if users_file.exists():
                with open(users_file, 'r') as f:
                    users_data = json.load(f)
                    for user_data in users_data:
                        user = User(
                            user_id=user_data["user_id"],
                            username=user_data["username"],
                            email=user_data["email"],
                            roles=[UserRole(role) for role in user_data["roles"]],
                            permissions=set(user_data.get("permissions", [])),
                            is_active=user_data.get("is_active", True),
                            created_at=user_data.get("created_at", ""),
                            last_login=user_data.get("last_login")
                        )
                        self.users[user.user_id] = user
            
            # Load policies
            policies_file = self.storage_dir / "policies.json"
            if policies_file.exists():
                with open(policies_file, 'r') as f:
                    policies_data = json.load(f)
                    for policy_data in policies_data:
                        policy = GovernancePolicy(
                            policy_id=policy_data["policy_id"],
                            name=policy_data["name"],
                            description=policy_data["description"],
                            patch_categories=[PatchCategory(cat) for cat in policy_data["patch_categories"]],
                            risk_levels=[PatchRiskLevel(level) for level in policy_data["risk_levels"]],
                            required_approver_roles=[UserRole(role) for role in policy_data["required_approver_roles"]],
                            min_approvers=policy_data.get("min_approvers", 1),
                            max_approval_time_hours=policy_data.get("max_approval_time_hours", 24),
                            allow_self_approval=policy_data.get("allow_self_approval", False),
                            require_multi_stage_approval=policy_data.get("require_multi_stage_approval", False),
                            compliance_frameworks=policy_data.get("compliance_frameworks", []),
                            is_active=policy_data.get("is_active", True)
                        )
                        self.policies[policy.policy_id] = policy
            
            # Load approval requests
            requests_file = self.storage_dir / "approval_requests.json"
            if requests_file.exists():
                with open(requests_file, 'r') as f:
                    requests_data = json.load(f)
                    for req_data in requests_data:
                        request = ApprovalRequest(
                            request_id=req_data["request_id"],
                            context_id=req_data["context_id"],
                            patch_content=req_data["patch_content"],
                            patch_category=PatchCategory(req_data["patch_category"]),
                            risk_level=PatchRiskLevel(req_data["risk_level"]),
                            requested_by=req_data["requested_by"],
                            requested_at=req_data["requested_at"],
                            status=ApprovalStatus(req_data["status"]),
                            approved_by=req_data.get("approved_by"),
                            approved_at=req_data.get("approved_at"),
                            rejection_reason=req_data.get("rejection_reason"),
                            expires_at=req_data.get("expires_at"),
                            metadata=req_data.get("metadata", {})
                        )
                        self.approval_requests[request.request_id] = request
            
        except Exception as e:
            logger.warning(f"Error loading governance data: {e}")
    
    def _save_users(self):
        """Save users to storage."""
        try:
            users_data = [
                {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "roles": [role.value for role in user.roles],
                    "permissions": list(user.permissions),
                    "is_active": user.is_active,
                    "created_at": user.created_at,
                    "last_login": user.last_login
                }
                for user in self.users.values()
            ]
            
            with open(self.storage_dir / "users.json", 'w') as f:
                json.dump(users_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def _save_policies(self):
        """Save policies to storage."""
        try:
            policies_data = [
                {
                    "policy_id": policy.policy_id,
                    "name": policy.name,
                    "description": policy.description,
                    "patch_categories": [cat.value for cat in policy.patch_categories],
                    "risk_levels": [level.value for level in policy.risk_levels],
                    "required_approver_roles": [role.value for role in policy.required_approver_roles],
                    "min_approvers": policy.min_approvers,
                    "max_approval_time_hours": policy.max_approval_time_hours,
                    "allow_self_approval": policy.allow_self_approval,
                    "require_multi_stage_approval": policy.require_multi_stage_approval,
                    "compliance_frameworks": policy.compliance_frameworks,
                    "is_active": policy.is_active
                }
                for policy in self.policies.values()
            ]
            
            with open(self.storage_dir / "policies.json", 'w') as f:
                json.dump(policies_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving policies: {e}")
    
    def _save_approval_requests(self):
        """Save approval requests to storage."""
        try:
            requests_data = [
                {
                    "request_id": req.request_id,
                    "context_id": req.context_id,
                    "patch_content": req.patch_content,
                    "patch_category": req.patch_category.value,
                    "risk_level": req.risk_level.value,
                    "requested_by": req.requested_by,
                    "requested_at": req.requested_at,
                    "status": req.status.value,
                    "approved_by": req.approved_by,
                    "approved_at": req.approved_at,
                    "rejection_reason": req.rejection_reason,
                    "expires_at": req.expires_at,
                    "metadata": req.metadata
                }
                for req in self.approval_requests.values()
            ]
            
            with open(self.storage_dir / "approval_requests.json", 'w') as f:
                json.dump(requests_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving approval requests: {e}")


def create_governance_manager(config: Optional[SecurityConfig] = None, 
                            storage_dir: Optional[Path] = None) -> GovernanceManager:
    """Create and return a configured governance manager."""
    return GovernanceManager(config, storage_dir)