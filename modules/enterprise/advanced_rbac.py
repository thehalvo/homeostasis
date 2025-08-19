"""
Advanced Role-Based Access Control (RBAC) for Enterprise

Extends the basic RBAC system with custom roles, hierarchical permissions,
dynamic role assignment, and attribute-based access control (ABAC) features.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable

from modules.security.rbac import RBACManager, PermissionDeniedError
from modules.security.audit import get_audit_logger
from modules.security.user_management import get_user_management

logger = logging.getLogger(__name__)


class PermissionType(Enum):
    """Types of permissions"""
    RESOURCE = "resource"
    ACTION = "action"
    DATA = "data"
    SYSTEM = "system"
    CUSTOM = "custom"


class RoleType(Enum):
    """Types of roles"""
    SYSTEM = "system"  # Built-in system roles
    PREDEFINED = "predefined"  # Predefined template roles
    CUSTOM = "custom"  # User-created custom roles
    DYNAMIC = "dynamic"  # Dynamically assigned roles


@dataclass
class Permission:
    """Enhanced permission with additional metadata"""
    name: str
    type: PermissionType
    description: str
    resource: Optional[str] = None
    actions: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Role:
    """Enhanced role with hierarchical support"""
    name: str
    type: RoleType
    description: str
    permissions: Set[str] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)  # Role inheritance
    child_roles: Set[str] = field(default_factory=set)
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    modified_at: Optional[datetime] = None
    modified_by: Optional[str] = None


@dataclass
class RoleAssignment:
    """Role assignment with temporal and contextual constraints"""
    user_id: str
    role_name: str
    assigned_at: datetime
    assigned_by: str
    expires_at: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)  # Contextual constraints
    conditions: Dict[str, Any] = field(default_factory=dict)  # Dynamic conditions


@dataclass
class AccessPolicy:
    """Access policy for attribute-based access control"""
    policy_id: str
    name: str
    description: str
    resource_pattern: str
    conditions: List[Dict[str, Any]]  # List of condition rules
    effect: str  # "allow" or "deny"
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedRBACManager:
    """
    Advanced RBAC manager with enterprise features including:
    - Custom role creation and management
    - Hierarchical role inheritance
    - Dynamic role assignment
    - Attribute-based access control (ABAC)
    - Temporal role assignments
    - Contextual permissions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize advanced RBAC manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Base RBAC manager
        self.base_rbac = RBACManager(config)
        
        # Enhanced stores
        self.permissions: Dict[str, Permission] = {}
        self.roles: Dict[str, Role] = {}
        self.role_assignments: Dict[str, List[RoleAssignment]] = {}
        self.access_policies: Dict[str, AccessPolicy] = {}
        
        # Managers
        self.audit_logger = get_audit_logger()
        self.user_management = get_user_management()
        
        # Permission evaluators
        self.condition_evaluators: Dict[str, Callable] = {}
        
        # Initialize default components
        self._initialize_permissions()
        self._initialize_roles()
        self._register_default_evaluators()
        
        logger.info("Initialized advanced RBAC manager")
    
    def _initialize_permissions(self):
        """Initialize enhanced permissions"""
        # Import base permissions
        for perm_name, perm_desc in self.base_rbac.permissions.items():
            self.permissions[perm_name] = Permission(
                name=perm_name,
                type=PermissionType.SYSTEM,
                description=perm_desc
            )
        
        # Add enterprise permissions
        enterprise_permissions = [
            Permission(
                name="manage_custom_roles",
                type=PermissionType.SYSTEM,
                description="Create and manage custom roles"
            ),
            Permission(
                name="manage_access_policies",
                type=PermissionType.SYSTEM,
                description="Manage attribute-based access policies"
            ),
            Permission(
                name="delegate_permissions",
                type=PermissionType.SYSTEM,
                description="Delegate permissions to other users"
            ),
            Permission(
                name="audit_access",
                type=PermissionType.SYSTEM,
                description="View access audit logs"
            ),
            Permission(
                name="manage_sla",
                type=PermissionType.SYSTEM,
                description="Manage SLA configurations"
            ),
            Permission(
                name="view_compliance_reports",
                type=PermissionType.SYSTEM,
                description="View compliance reports"
            ),
            Permission(
                name="manage_compliance",
                type=PermissionType.SYSTEM,
                description="Manage compliance configurations"
            ),
            Permission(
                name="manage_data_classification",
                type=PermissionType.DATA,
                description="Manage data classification policies"
            )
        ]
        
        for perm in enterprise_permissions:
            self.permissions[perm.name] = perm
    
    def _initialize_roles(self):
        """Initialize enhanced roles with hierarchy"""
        # Import base roles
        for role_name, role_info in self.base_rbac.roles.items():
            self.roles[role_name] = Role(
                name=role_name,
                type=RoleType.SYSTEM,
                description=role_info['description'],
                permissions=role_info['permissions']
            )
        
        # Add enterprise roles
        enterprise_roles = [
            Role(
                name="compliance_officer",
                type=RoleType.PREDEFINED,
                description="Compliance and audit management",
                permissions={
                    'view_compliance_reports', 'manage_compliance',
                    'audit_access', 'view_audit_logs', 'view_dashboard'
                }
            ),
            Role(
                name="security_admin",
                type=RoleType.PREDEFINED,
                description="Security administration",
                permissions={
                    'manage_users', 'manage_roles', 'manage_access_policies',
                    'audit_access', 'view_audit_logs', 'manage_data_classification'
                },
                parent_roles={'admin'}  # Inherits from admin
            ),
            Role(
                name="sla_manager",
                type=RoleType.PREDEFINED,
                description="SLA configuration and monitoring",
                permissions={
                    'manage_sla', 'view_dashboard', 'view_metrics',
                    'view_analyses', 'view_fixes'
                }
            ),
            Role(
                name="delegated_admin",
                type=RoleType.PREDEFINED,
                description="Limited administrative privileges",
                permissions={
                    'manage_users', 'delegate_permissions', 'view_audit_logs'
                },
                conditions={
                    'max_delegation_depth': 1,
                    'restricted_permissions': ['manage_roles', 'manage_config']
                }
            )
        ]
        
        for role in enterprise_roles:
            self.roles[role.name] = role
            
            # Update parent-child relationships
            for parent_role in role.parent_roles:
                if parent_role in self.roles:
                    self.roles[parent_role].child_roles.add(role.name)
    
    def _register_default_evaluators(self):
        """Register default condition evaluators"""
        # Time-based evaluator
        self.register_condition_evaluator('time_range', self._evaluate_time_range)
        
        # IP-based evaluator
        self.register_condition_evaluator('ip_whitelist', self._evaluate_ip_whitelist)
        
        # Resource ownership evaluator
        self.register_condition_evaluator('resource_owner', self._evaluate_resource_owner)
        
        # Department/group evaluator
        self.register_condition_evaluator('department', self._evaluate_department)
        
        # Data classification evaluator
        self.register_condition_evaluator('data_classification', self._evaluate_data_classification)
    
    def create_custom_role(self, name: str, description: str, permissions: List[str],
                          parent_roles: Optional[List[str]] = None,
                          conditions: Optional[Dict[str, Any]] = None,
                          created_by: str = None) -> bool:
        """Create a custom role.
        
        Args:
            name: Role name
            description: Role description
            permissions: List of permission names
            parent_roles: Optional parent roles for inheritance
            conditions: Optional conditions for the role
            created_by: User creating the role
            
        Returns:
            True if role was created successfully
        """
        # Validate role name
        if name in self.roles:
            raise ValueError(f"Role '{name}' already exists")
        
        # Validate permissions
        invalid_perms = set(permissions) - set(self.permissions.keys())
        if invalid_perms:
            raise ValueError(f"Invalid permissions: {invalid_perms}")
        
        # Validate parent roles
        if parent_roles:
            invalid_parents = set(parent_roles) - set(self.roles.keys())
            if invalid_parents:
                raise ValueError(f"Invalid parent roles: {invalid_parents}")
        
        # Create role
        role = Role(
            name=name,
            type=RoleType.CUSTOM,
            description=description,
            permissions=set(permissions),
            parent_roles=set(parent_roles) if parent_roles else set(),
            conditions=conditions or {},
            created_by=created_by
        )
        
        self.roles[name] = role
        
        # Update parent-child relationships
        if parent_roles:
            for parent in parent_roles:
                self.roles[parent].child_roles.add(name)
        
        # Log creation
        self.audit_logger.log_event(
            event_type='custom_role_created',
            user=created_by or 'system',
            details={
                'role_name': name,
                'permissions': list(permissions),
                'parent_roles': parent_roles
            }
        )
        
        return True
    
    def update_custom_role(self, name: str, description: Optional[str] = None,
                          permissions: Optional[List[str]] = None,
                          parent_roles: Optional[List[str]] = None,
                          conditions: Optional[Dict[str, Any]] = None,
                          modified_by: str = None) -> bool:
        """Update a custom role.
        
        Args:
            name: Role name to update
            description: New description
            permissions: New permissions list
            parent_roles: New parent roles
            conditions: New conditions
            modified_by: User modifying the role
            
        Returns:
            True if role was updated successfully
        """
        if name not in self.roles:
            raise ValueError(f"Role '{name}' does not exist")
        
        role = self.roles[name]
        
        # Only custom roles can be updated
        if role.type != RoleType.CUSTOM:
            raise ValueError(f"Cannot modify {role.type.value} role '{name}'")
        
        # Update fields
        if description is not None:
            role.description = description
        
        if permissions is not None:
            # Validate permissions
            invalid_perms = set(permissions) - set(self.permissions.keys())
            if invalid_perms:
                raise ValueError(f"Invalid permissions: {invalid_perms}")
            role.permissions = set(permissions)
        
        if parent_roles is not None:
            # Remove from old parents
            for parent in role.parent_roles:
                if parent in self.roles:
                    self.roles[parent].child_roles.discard(name)
            
            # Validate new parents
            invalid_parents = set(parent_roles) - set(self.roles.keys())
            if invalid_parents:
                raise ValueError(f"Invalid parent roles: {invalid_parents}")
            
            # Update relationships
            role.parent_roles = set(parent_roles)
            for parent in parent_roles:
                self.roles[parent].child_roles.add(name)
        
        if conditions is not None:
            role.conditions = conditions
        
        role.modified_at = datetime.utcnow()
        role.modified_by = modified_by
        
        # Log update
        self.audit_logger.log_event(
            event_type='custom_role_updated',
            user=modified_by or 'system',
            details={
                'role_name': name,
                'updated_fields': {
                    k: v for k, v in {
                        'description': description,
                        'permissions': permissions,
                        'parent_roles': parent_roles,
                        'conditions': conditions
                    }.items() if v is not None
                }
            }
        )
        
        return True
    
    def delete_custom_role(self, name: str, deleted_by: str = None) -> bool:
        """Delete a custom role.
        
        Args:
            name: Role name to delete
            deleted_by: User deleting the role
            
        Returns:
            True if role was deleted successfully
        """
        if name not in self.roles:
            return False
        
        role = self.roles[name]
        
        # Only custom roles can be deleted
        if role.type != RoleType.CUSTOM:
            raise ValueError(f"Cannot delete {role.type.value} role '{name}'")
        
        # Check if role is assigned to any users
        for user_assignments in self.role_assignments.values():
            if any(a.role_name == name for a in user_assignments):
                raise ValueError(f"Cannot delete role '{name}' - still assigned to users")
        
        # Remove from parent-child relationships
        for parent in role.parent_roles:
            if parent in self.roles:
                self.roles[parent].child_roles.discard(name)
        
        for child in role.child_roles:
            if child in self.roles:
                self.roles[child].parent_roles.discard(name)
        
        # Delete role
        del self.roles[name]
        
        # Log deletion
        self.audit_logger.log_event(
            event_type='custom_role_deleted',
            user=deleted_by or 'system',
            details={'role_name': name}
        )
        
        return True
    
    def assign_role_with_conditions(self, user_id: str, role_name: str,
                                  assigned_by: str, expires_at: Optional[datetime] = None,
                                  context: Optional[Dict[str, Any]] = None,
                                  conditions: Optional[Dict[str, Any]] = None) -> bool:
        """Assign a role to a user with temporal and contextual constraints.
        
        Args:
            user_id: User ID
            role_name: Role to assign
            assigned_by: User assigning the role
            expires_at: Optional expiration time
            context: Optional context constraints
            conditions: Optional dynamic conditions
            
        Returns:
            True if role was assigned successfully
        """
        if role_name not in self.roles:
            raise ValueError(f"Role '{role_name}' does not exist")
        
        # Create assignment
        assignment = RoleAssignment(
            user_id=user_id,
            role_name=role_name,
            assigned_at=datetime.utcnow(),
            assigned_by=assigned_by,
            expires_at=expires_at,
            context=context or {},
            conditions=conditions or {}
        )
        
        # Store assignment
        if user_id not in self.role_assignments:
            self.role_assignments[user_id] = []
        
        self.role_assignments[user_id].append(assignment)
        
        # Update user's roles in base system
        user = self.user_management.get_user(user_id)
        if user:
            current_roles = set(user.get('roles', []))
            current_roles.add(role_name)
            self.user_management.update_user(user_id, roles=list(current_roles))
        
        # Log assignment
        self.audit_logger.log_event(
            event_type='role_assigned',
            user=assigned_by,
            details={
                'user_id': user_id,
                'role_name': role_name,
                'expires_at': expires_at.isoformat() if expires_at else None,
                'context': context,
                'conditions': conditions
            }
        )
        
        return True
    
    def revoke_role(self, user_id: str, role_name: str, revoked_by: str) -> bool:
        """Revoke a role from a user.
        
        Args:
            user_id: User ID
            role_name: Role to revoke
            revoked_by: User revoking the role
            
        Returns:
            True if role was revoked successfully
        """
        if user_id not in self.role_assignments:
            return False
        
        # Remove assignments
        original_count = len(self.role_assignments[user_id])
        self.role_assignments[user_id] = [
            a for a in self.role_assignments[user_id]
            if a.role_name != role_name
        ]
        
        if len(self.role_assignments[user_id]) == original_count:
            return False  # No assignment found
        
        # Update user's roles in base system
        user = self.user_management.get_user(user_id)
        if user:
            current_roles = set(user.get('roles', []))
            current_roles.discard(role_name)
            self.user_management.update_user(user_id, roles=list(current_roles))
        
        # Log revocation
        self.audit_logger.log_event(
            event_type='role_revoked',
            user=revoked_by,
            details={
                'user_id': user_id,
                'role_name': role_name
            }
        )
        
        return True
    
    def get_effective_permissions(self, user_info: Dict[str, Any],
                                context: Optional[Dict[str, Any]] = None) -> Set[str]:
        """Get all effective permissions for a user including inherited and conditional.
        
        Args:
            user_info: User information
            context: Current context for evaluation
            
        Returns:
            Set of effective permission names
        """
        user_id = user_info.get('username') or user_info.get('user_id')
        permissions = set()
        
        # Get active role assignments
        active_roles = self._get_active_roles(user_id, context)
        
        # Collect permissions from all active roles
        for role_name in active_roles:
            role = self.roles.get(role_name)
            if role:
                # Direct permissions
                permissions.update(role.permissions)
                
                # Inherited permissions
                inherited_perms = self._get_inherited_permissions(role_name)
                permissions.update(inherited_perms)
        
        # Apply ABAC policies
        permissions = self._apply_access_policies(user_info, permissions, context)
        
        return permissions
    
    def _get_active_roles(self, user_id: str, context: Optional[Dict[str, Any]] = None) -> Set[str]:
        """Get currently active roles for a user."""
        active_roles = set()
        
        # Get assignments
        if user_id in self.role_assignments:
            now = datetime.utcnow()
            
            for assignment in self.role_assignments[user_id]:
                # Check expiration
                if assignment.expires_at and now > assignment.expires_at:
                    continue
                
                # Check context constraints
                if assignment.context and context:
                    if not self._match_context(assignment.context, context):
                        continue
                
                # Check dynamic conditions
                if assignment.conditions:
                    if not self._evaluate_conditions(assignment.conditions, {
                        'user_id': user_id,
                        'context': context,
                        'assignment': assignment
                    }):
                        continue
                
                active_roles.add(assignment.role_name)
        
        # Also check base user roles
        user = self.user_management.get_user(user_id)
        if user:
            active_roles.update(user.get('roles', []))
        
        return active_roles
    
    def _get_inherited_permissions(self, role_name: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """Get all inherited permissions from parent roles."""
        if visited is None:
            visited = set()
        
        if role_name in visited:
            return set()  # Prevent circular inheritance
        
        visited.add(role_name)
        permissions = set()
        
        role = self.roles.get(role_name)
        if role:
            for parent_role in role.parent_roles:
                parent = self.roles.get(parent_role)
                if parent:
                    permissions.update(parent.permissions)
                    permissions.update(self._get_inherited_permissions(parent_role, visited))
        
        return permissions
    
    def has_permission_with_context(self, user_info: Dict[str, Any], permission: str,
                                  resource: Optional[str] = None,
                                  context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if user has permission considering context and resource.
        
        Args:
            user_info: User information
            permission: Permission to check
            resource: Optional resource identifier
            context: Optional context for evaluation
            
        Returns:
            True if user has the permission
        """
        # Get effective permissions
        effective_perms = self.get_effective_permissions(user_info, context)
        
        # Check basic permission
        if permission not in effective_perms:
            return False
        
        # Check resource-specific permissions
        if resource:
            perm_obj = self.permissions.get(permission)
            if perm_obj and perm_obj.resource:
                if not self._match_resource(perm_obj.resource, resource):
                    return False
        
        # Check permission conditions
        perm_obj = self.permissions.get(permission)
        if perm_obj and perm_obj.conditions:
            if not self._evaluate_conditions(perm_obj.conditions, {
                'user': user_info,
                'resource': resource,
                'context': context
            }):
                return False
        
        return True
    
    def create_access_policy(self, name: str, description: str,
                           resource_pattern: str, conditions: List[Dict[str, Any]],
                           effect: str = "allow", priority: int = 0,
                           created_by: str = None) -> str:
        """Create an access policy for ABAC.
        
        Args:
            name: Policy name
            description: Policy description
            resource_pattern: Resource pattern (supports wildcards)
            conditions: List of condition rules
            effect: "allow" or "deny"
            priority: Policy priority (higher = more important)
            created_by: User creating the policy
            
        Returns:
            Policy ID
        """
        policy_id = f"policy_{datetime.utcnow().timestamp()}"
        
        policy = AccessPolicy(
            policy_id=policy_id,
            name=name,
            description=description,
            resource_pattern=resource_pattern,
            conditions=conditions,
            effect=effect,
            priority=priority,
            enabled=True,
            metadata={
                'created_at': datetime.utcnow().isoformat(),
                'created_by': created_by
            }
        )
        
        self.access_policies[policy_id] = policy
        
        # Log creation
        self.audit_logger.log_event(
            event_type='access_policy_created',
            user=created_by or 'system',
            details={
                'policy_id': policy_id,
                'name': name,
                'resource_pattern': resource_pattern,
                'effect': effect
            }
        )
        
        return policy_id
    
    def _apply_access_policies(self, user_info: Dict[str, Any],
                             permissions: Set[str],
                             context: Optional[Dict[str, Any]] = None) -> Set[str]:
        """Apply ABAC policies to modify permissions."""
        # Sort policies by priority
        sorted_policies = sorted(
            [p for p in self.access_policies.values() if p.enabled],
            key=lambda x: x.priority,
            reverse=True
        )
        
        modified_perms = permissions.copy()
        
        for policy in sorted_policies:
            # Check if policy applies
            if self._evaluate_policy_conditions(policy.conditions, {
                'user': user_info,
                'permissions': permissions,
                'context': context
            }):
                if policy.effect == "allow":
                    # Add permissions based on resource pattern
                    # This is simplified - in practice would be more sophisticated
                    pass
                elif policy.effect == "deny":
                    # Remove permissions based on resource pattern
                    # This is simplified - in practice would be more sophisticated
                    pass
        
        return modified_perms
    
    def register_condition_evaluator(self, condition_type: str,
                                   evaluator: Callable[[Dict[str, Any], Dict[str, Any]], bool]):
        """Register a custom condition evaluator.
        
        Args:
            condition_type: Type of condition
            evaluator: Function that evaluates the condition
        """
        self.condition_evaluators[condition_type] = evaluator
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a set of conditions."""
        for cond_type, cond_value in conditions.items():
            evaluator = self.condition_evaluators.get(cond_type)
            if evaluator:
                if not evaluator(cond_value, context):
                    return False
            else:
                # Unknown condition type - default to True for safety
                logger.warning(f"Unknown condition type: {cond_type}")
        
        return True
    
    def _evaluate_policy_conditions(self, conditions: List[Dict[str, Any]], context: Dict[str, Any]) -> bool:
        """Evaluate policy conditions (list of condition rules)."""
        # All conditions must be true (AND logic)
        for condition in conditions:
            if not self._evaluate_conditions(condition, context):
                return False
        return True
    
    def _match_context(self, required_context: Dict[str, Any], actual_context: Dict[str, Any]) -> bool:
        """Check if actual context matches required context."""
        for key, value in required_context.items():
            if key not in actual_context:
                return False
            if actual_context[key] != value:
                return False
        return True
    
    def _match_resource(self, pattern: str, resource: str) -> bool:
        """Check if resource matches pattern (supports wildcards)."""
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        return bool(re.match(f"^{regex_pattern}$", resource))
    
    # Default condition evaluators
    def _evaluate_time_range(self, time_range: Dict[str, str], context: Dict[str, Any]) -> bool:
        """Evaluate time range condition."""
        now = datetime.utcnow()
        
        if 'start' in time_range:
            start = datetime.fromisoformat(time_range['start'])
            if now < start:
                return False
        
        if 'end' in time_range:
            end = datetime.fromisoformat(time_range['end'])
            if now > end:
                return False
        
        return True
    
    def _evaluate_ip_whitelist(self, whitelist: List[str], context: Dict[str, Any]) -> bool:
        """Evaluate IP whitelist condition."""
        user_ip = context.get('user', {}).get('ip_address')
        if not user_ip:
            return False
        
        return user_ip in whitelist
    
    def _evaluate_resource_owner(self, owner_field: str, context: Dict[str, Any]) -> bool:
        """Evaluate resource ownership condition."""
        user_id = context.get('user', {}).get('user_id')
        resource = context.get('resource', {})
        
        if not user_id or not resource:
            return False
        
        return resource.get(owner_field) == user_id
    
    def _evaluate_department(self, departments: List[str], context: Dict[str, Any]) -> bool:
        """Evaluate department membership condition."""
        user_dept = context.get('user', {}).get('department')
        if not user_dept:
            return False
        
        return user_dept in departments
    
    def _evaluate_data_classification(self, allowed_levels: List[str], context: Dict[str, Any]) -> bool:
        """Evaluate data classification condition."""
        resource_classification = context.get('resource', {}).get('classification')
        if not resource_classification:
            return True  # No classification = public
        
        return resource_classification in allowed_levels
    
    def export_roles(self) -> Dict[str, Any]:
        """Export all roles configuration."""
        export_data = {
            'roles': {},
            'permissions': {},
            'policies': {}
        }
        
        # Export roles
        for role_name, role in self.roles.items():
            export_data['roles'][role_name] = {
                'type': role.type.value,
                'description': role.description,
                'permissions': list(role.permissions),
                'parent_roles': list(role.parent_roles),
                'conditions': role.conditions,
                'metadata': role.metadata
            }
        
        # Export permissions
        for perm_name, perm in self.permissions.items():
            export_data['permissions'][perm_name] = {
                'type': perm.type.value,
                'description': perm.description,
                'resource': perm.resource,
                'actions': perm.actions,
                'conditions': perm.conditions
            }
        
        # Export policies
        for policy_id, policy in self.access_policies.items():
            export_data['policies'][policy_id] = {
                'name': policy.name,
                'description': policy.description,
                'resource_pattern': policy.resource_pattern,
                'conditions': policy.conditions,
                'effect': policy.effect,
                'priority': policy.priority,
                'enabled': policy.enabled
            }
        
        return export_data
    
    def import_roles(self, import_data: Dict[str, Any], imported_by: str = None) -> Dict[str, int]:
        """Import roles configuration.
        
        Args:
            import_data: Roles configuration to import
            imported_by: User performing the import
            
        Returns:
            Statistics of imported items
        """
        stats = {
            'roles_imported': 0,
            'permissions_imported': 0,
            'policies_imported': 0,
            'errors': []
        }
        
        # Import permissions
        for perm_name, perm_data in import_data.get('permissions', {}).items():
            try:
                if perm_name not in self.permissions:
                    self.permissions[perm_name] = Permission(
                        name=perm_name,
                        type=PermissionType(perm_data['type']),
                        description=perm_data['description'],
                        resource=perm_data.get('resource'),
                        actions=perm_data.get('actions', []),
                        conditions=perm_data.get('conditions', {})
                    )
                    stats['permissions_imported'] += 1
            except Exception as e:
                stats['errors'].append(f"Permission {perm_name}: {str(e)}")
        
        # Import roles
        for role_name, role_data in import_data.get('roles', {}).items():
            try:
                if role_name not in self.roles and role_data['type'] == 'custom':
                    self.create_custom_role(
                        name=role_name,
                        description=role_data['description'],
                        permissions=role_data['permissions'],
                        parent_roles=role_data.get('parent_roles'),
                        conditions=role_data.get('conditions'),
                        created_by=imported_by
                    )
                    stats['roles_imported'] += 1
            except Exception as e:
                stats['errors'].append(f"Role {role_name}: {str(e)}")
        
        # Import policies
        for policy_name, policy_data in import_data.get('policies', {}).items():
            try:
                self.create_access_policy(
                    name=policy_data['name'],
                    description=policy_data['description'],
                    resource_pattern=policy_data['resource_pattern'],
                    conditions=policy_data['conditions'],
                    effect=policy_data['effect'],
                    priority=policy_data.get('priority', 0),
                    created_by=imported_by
                )
                stats['policies_imported'] += 1
            except Exception as e:
                stats['errors'].append(f"Policy {policy_name}: {str(e)}")
        
        return stats


# Factory function
def create_advanced_rbac_manager(config: Dict[str, Any] = None) -> AdvancedRBACManager:
    """Create advanced RBAC manager"""
    return AdvancedRBACManager(config)