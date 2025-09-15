"""
Role-Based Access Control (RBAC) for Homeostasis.

Provides permission management for securing access to different features
and functionality in production environments.
"""

import logging
from typing import Dict, List, Optional, Set, TypedDict, cast

logger = logging.getLogger(__name__)


class RoleInfo(TypedDict):
    """Type definition for role information."""

    description: str
    permissions: Set[str]


class PermissionDeniedError(Exception):
    """Exception raised when a user lacks required permissions."""

    pass


class RBACManager:
    """Manages role-based access control for Homeostasis."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the RBAC manager.

        Args:
            config: Configuration dictionary for RBAC settings
        """
        self.config = config or {}
        self.permissions: Dict[str, str] = {}
        self.roles: Dict[str, RoleInfo] = {}

        # Define default roles and permissions
        self._init_default_roles()

        # Load custom roles from config
        self._load_custom_roles()

    def _init_default_roles(self):
        """Initialize default roles and permissions."""
        # Default permissions
        self.permissions = {
            # Monitoring permissions
            "view_dashboard": "View monitoring dashboard",
            "view_logs": "View system logs",
            "view_metrics": "View system metrics",
            # Analysis permissions
            "view_analyses": "View error analyses",
            "customize_rules": "Customize analysis rules",
            # Fix permissions
            "view_fixes": "View proposed fixes",
            "approve_fixes": "Approve fixes for deployment",
            "reject_fixes": "Reject proposed fixes",
            "deploy_fixes": "Deploy fixes to production",
            # System permissions
            "manage_users": "Manage user accounts",
            "manage_roles": "Manage roles and permissions",
            "manage_config": "Manage system configuration",
            "view_audit_logs": "View audit logs",
            # Advanced permissions
            "create_templates": "Create fix templates",
            "customize_ml": "Customize ML algorithms",
        }

        # Default roles with associated permissions
        self.roles = {
            "admin": {
                "description": "Administrator with full access",
                "permissions": set(self.permissions.keys()),
            },
            "operator": {
                "description": "System operator who can monitor and approve fixes",
                "permissions": {
                    "view_dashboard",
                    "view_logs",
                    "view_metrics",
                    "view_analyses",
                    "view_fixes",
                    "approve_fixes",
                    "reject_fixes",
                    "deploy_fixes",
                    "view_audit_logs",
                },
            },
            "developer": {
                "description": "Developer who can customize rules and templates",
                "permissions": {
                    "view_dashboard",
                    "view_logs",
                    "view_metrics",
                    "view_analyses",
                    "view_fixes",
                    "customize_rules",
                    "create_templates",
                },
            },
            "analyst": {
                "description": "Analyst who can view metrics and analyses",
                "permissions": {
                    "view_dashboard",
                    "view_logs",
                    "view_metrics",
                    "view_analyses",
                    "view_fixes",
                },
            },
            "reader": {
                "description": "Read-only access to monitoring data",
                "permissions": {"view_dashboard", "view_logs", "view_metrics"},
            },
        }

    def _load_custom_roles(self):
        """Load custom roles from configuration."""
        custom_roles = self.config.get("roles", {})

        for role_name, role_config in custom_roles.items():
            # Skip if role already exists and custom roles aren't allowed to override
            if role_name in self.roles and not self.config.get(
                "allow_role_override", False
            ):
                logger.warning(
                    f"Role '{role_name}' already exists and cannot be overridden"
                )
                continue

            # Create or update role
            self.roles[role_name] = cast(
                RoleInfo,
                {
                    "description": role_config.get(
                        "description", f"Custom role: {role_name}"
                    ),
                    "permissions": set(role_config.get("permissions", [])),
                },
            )

    def list_roles(self) -> Dict[str, Dict]:
        """List all available roles.

        Returns:
            Dict: Dictionary of role names to role details
        """
        # Return a copy to prevent modification
        return {
            role_name: {
                "description": role_info["description"],
                "permissions": list(role_info["permissions"]),
            }
            for role_name, role_info in self.roles.items()
        }

    def list_permissions(self) -> Dict[str, str]:
        """List all available permissions.

        Returns:
            Dict: Dictionary of permission names to descriptions
        """
        return dict(self.permissions)

    def get_role_permissions(self, role_name: str) -> Set[str]:
        """Get permissions for a specific role.

        Args:
            role_name: Name of the role

        Returns:
            Set[str]: Set of permission names

        Raises:
            ValueError: If role does not exist
        """
        if role_name not in self.roles:
            raise ValueError(f"Role '{role_name}' does not exist")

        return set(self.roles[role_name]["permissions"])

    def has_permission(self, user_info: Dict, permission: str) -> bool:
        """Check if a user has a specific permission.

        Args:
            user_info: User information dictionary
            permission: Permission to check

        Returns:
            bool: True if user has permission, False otherwise
        """
        # Admin users have all permissions
        if "admin" in user_info.get("roles", []):
            return True

        # Check user roles for the permission
        user_roles = user_info.get("roles", [])
        for role in user_roles:
            if role in self.roles and permission in self.roles[role]["permissions"]:
                return True

        return False

    def require_permission(self, user_info: Dict, permission: str) -> None:
        """Require that a user has a specific permission.

        Args:
            user_info: User information dictionary
            permission: Permission to require

        Raises:
            PermissionDeniedError: If user does not have the required permission
        """
        if not self.has_permission(user_info, permission):
            raise PermissionDeniedError(
                f"User '{user_info.get('username', 'Unknown')}' does not have "
                f"the required permission: {permission}"
            )

    def create_role(
        self, role_name: str, description: str, permissions: List[str]
    ) -> bool:
        """Create a new role.

        Args:
            role_name: Name of the new role
            description: Description of the role
            permissions: List of permissions for the role

        Returns:
            bool: True if role was created, False if it already exists

        Raises:
            ValueError: If any of the permissions do not exist
        """
        # Check if role already exists
        if role_name in self.roles:
            return False

        # Validate permissions
        invalid_permissions = set(permissions) - set(self.permissions.keys())
        if invalid_permissions:
            raise ValueError(f"Invalid permissions: {invalid_permissions}")

        # Create role
        self.roles[role_name] = {
            "description": description,
            "permissions": set(permissions),
        }

        return True

    def update_role(
        self,
        role_name: str,
        description: Optional[str] = None,
        permissions: Optional[List[str]] = None,
    ) -> bool:
        """Update an existing role.

        Args:
            role_name: Name of the role to update
            description: New description (if None, keep current)
            permissions: New permissions (if None, keep current)

        Returns:
            bool: True if role was updated, False if it doesn't exist

        Raises:
            ValueError: If any of the permissions do not exist
        """
        # Check if role exists
        if role_name not in self.roles:
            return False

        # Validate permissions if provided
        if permissions is not None:
            invalid_permissions = set(permissions) - set(self.permissions.keys())
            if invalid_permissions:
                raise ValueError(f"Invalid permissions: {invalid_permissions}")

            self.roles[role_name]["permissions"] = set(permissions)

        # Update description if provided
        if description is not None:
            self.roles[role_name]["description"] = description

        return True

    def delete_role(self, role_name: str) -> bool:
        """Delete a role.

        Args:
            role_name: Name of the role to delete

        Returns:
            bool: True if role was deleted, False if it doesn't exist
        """
        # Check if role exists
        if role_name not in self.roles:
            return False

        # Don't allow deleting default roles
        if role_name in ["admin", "operator", "developer", "analyst", "reader"]:
            logger.warning(f"Cannot delete default role '{role_name}'")
            return False

        # Delete role
        del self.roles[role_name]
        return True

    def add_permission_to_role(self, role_name: str, permission: str) -> bool:
        """Add a permission to a role.

        Args:
            role_name: Name of the role
            permission: Permission to add

        Returns:
            bool: True if permission was added, False if role doesn't exist

        Raises:
            ValueError: If the permission does not exist
        """
        # Check if role exists
        if role_name not in self.roles:
            return False

        # Validate permission
        if permission not in self.permissions:
            raise ValueError(f"Invalid permission: {permission}")

        # Add permission
        self.roles[role_name]["permissions"].add(permission)
        return True

    def remove_permission_from_role(self, role_name: str, permission: str) -> bool:
        """Remove a permission from a role.

        Args:
            role_name: Name of the role
            permission: Permission to remove

        Returns:
            bool: True if permission was removed, False if role doesn't exist or
                 permission wasn't in the role
        """
        # Check if role exists
        if role_name not in self.roles:
            return False

        # Remove permission if it exists
        if permission in self.roles[role_name]["permissions"]:
            self.roles[role_name]["permissions"].remove(permission)
            return True

        return False


# Singleton instance for app-wide use
_rbac_manager = None


def get_rbac_manager(config: Optional[Dict] = None) -> RBACManager:
    """Get or create the singleton RBACManager instance.

    Args:
        config: Optional configuration to initialize the manager with

    Returns:
        RBACManager: The RBAC manager instance
    """
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager(config)
    return _rbac_manager


def has_permission(user_info: Dict, permission: str) -> bool:
    """Check if a user has a specific permission.

    Args:
        user_info: User information dictionary
        permission: Permission to check

    Returns:
        bool: True if user has permission, False otherwise
    """
    return get_rbac_manager().has_permission(user_info, permission)


def require_permission(user_info: Dict, permission: str) -> None:
    """Require that a user has a specific permission.

    Args:
        user_info: User information dictionary
        permission: Permission to require

    Raises:
        PermissionDeniedError: If user does not have the required permission
    """
    return get_rbac_manager().require_permission(user_info, permission)
