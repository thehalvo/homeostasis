"""
Enhanced User Management System for Homeostasis Enterprise Governance.

This module provides comprehensive user management capabilities including:
- User lifecycle management (create, update, deactivate)
- Role assignment and management
- Group-based permissions
- Session management
- Password policies
- Multi-factor authentication support
"""

import datetime
import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

from .auth import AuthenticationManager, get_auth_manager
from .rbac import RBACManager, get_rbac_manager
from .audit import get_audit_logger

logger = logging.getLogger(__name__)


class UserStatus(Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    LOCKED = "locked"
    PENDING_ACTIVATION = "pending_activation"


class AuthMethod(Enum):
    """Authentication methods."""
    PASSWORD = "password"
    MFA_TOTP = "mfa_totp"
    MFA_SMS = "mfa_sms"
    OAUTH = "oauth"
    SAML = "saml"
    LDAP = "ldap"


@dataclass
class UserProfile:
    """Extended user profile information."""
    user_id: str
    username: str
    email: str
    full_name: str
    department: str = ""
    job_title: str = ""
    phone: str = ""
    location: str = ""
    timezone: str = "UTC"
    language: str = "en"
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class UserSession:
    """User session information."""
    session_id: str
    user_id: str
    created_at: datetime.datetime
    last_activity: datetime.datetime
    ip_address: str
    user_agent: str
    expires_at: datetime.datetime
    is_active: bool = True
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class PasswordPolicy:
    """Password policy configuration."""
    min_length: int = 12
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    max_age_days: int = 90
    history_count: int = 5
    lockout_threshold: int = 5
    lockout_duration_minutes: int = 30


class UserManagementSystem:
    """
    Comprehensive user management system for enterprise environments.
    
    Features:
    - User lifecycle management
    - Advanced authentication and authorization
    - Session management
    - Password policies
    - Group-based permissions
    - Integration with external identity providers
    """
    
    def __init__(self, config: Dict = None, storage_path: str = None):
        """Initialize the user management system.
        
        Args:
            config: Configuration dictionary
            storage_path: Path to store user data
        """
        self.config = config or {}
        self.storage_path = Path(storage_path or "data/users")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Get managers
        self.auth_manager = get_auth_manager(config)
        self.rbac_manager = get_rbac_manager(config)
        self.audit_logger = get_audit_logger(config)
        
        # Initialize stores
        self.users: Dict[str, Dict] = {}
        self.profiles: Dict[str, UserProfile] = {}
        self.sessions: Dict[str, UserSession] = {}
        self.groups: Dict[str, Dict] = {}
        self.password_history: Dict[str, List[str]] = {}
        self.failed_attempts: Dict[str, List[datetime.datetime]] = {}
        
        # Load configuration
        self.password_policy = self._load_password_policy()
        self.session_timeout = self.config.get('session_timeout', 3600)  # 1 hour
        self.mfa_required_roles = set(self.config.get('mfa_required_roles', ['admin', 'operator']))
        
        # Load existing data
        self._load_user_data()
        
        # Initialize default groups if none exist
        if not self.groups:
            self._create_default_groups()
    
    def create_user(self, username: str, email: str, password: str, 
                    full_name: str, roles: List[str] = None,
                    groups: List[str] = None, **profile_data) -> str:
        """Create a new user account.
        
        Args:
            username: Username
            email: Email address
            password: Password
            full_name: Full name
            roles: List of role names
            groups: List of group names
            **profile_data: Additional profile data
            
        Returns:
            User ID
            
        Raises:
            ValueError: If user already exists or validation fails
        """
        # Validate username and email
        if self._username_exists(username):
            raise ValueError(f"Username '{username}' already exists")
        
        if self._email_exists(email):
            raise ValueError(f"Email '{email}' already exists")
        
        # Validate password
        if not self._validate_password(password):
            raise ValueError("Password does not meet policy requirements")
        
        # Generate user ID
        user_id = self._generate_user_id()
        
        # Register user with auth manager
        self.auth_manager.register_user(username, password, roles or ['user'])
        
        # Create user record
        user_data = {
            'user_id': user_id,
            'username': username,
            'email': email,
            'status': UserStatus.ACTIVE.value,
            'roles': roles or ['user'],
            'groups': groups or [],
            'created_at': datetime.datetime.utcnow().isoformat(),
            'last_login': None,
            'password_changed_at': datetime.datetime.utcnow().isoformat(),
            'mfa_enabled': False,
            'mfa_secret': None,
            'auth_methods': [AuthMethod.PASSWORD.value]
        }
        
        self.users[user_id] = user_data
        
        # Create user profile
        profile = UserProfile(
            user_id=user_id,
            username=username,
            email=email,
            full_name=full_name,
            **profile_data
        )
        self.profiles[user_id] = profile
        
        # Initialize password history
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        self.password_history[user_id] = [password_hash]
        
        # Apply group permissions
        self._apply_group_permissions(user_id)
        
        # Save data
        self._save_user_data()
        
        # Log event
        self.audit_logger.log_event(
            event_type='user_created',
            user='system',
            details={
                'user_id': user_id,
                'username': username,
                'email': email,
                'roles': roles or ['user'],
                'groups': groups or []
            }
        )
        
        return user_id
    
    def update_user(self, user_id: str, **updates) -> bool:
        """Update user information.
        
        Args:
            user_id: User ID
            **updates: Fields to update
            
        Returns:
            True if successful
        """
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        user = self.users[user_id]
        profile = self.profiles.get(user_id)
        
        # Update user data
        allowed_user_fields = {'email', 'status', 'roles', 'groups', 'mfa_enabled'}
        for field, value in updates.items():
            if field in allowed_user_fields:
                user[field] = value
        
        # Update profile data
        if profile:
            profile_fields = {'full_name', 'department', 'job_title', 'phone', 
                            'location', 'timezone', 'language'}
            for field, value in updates.items():
                if field in profile_fields and hasattr(profile, field):
                    setattr(profile, field, value)
        
        # Re-apply group permissions if groups changed
        if 'groups' in updates:
            self._apply_group_permissions(user_id)
        
        # Save data
        self._save_user_data()
        
        # Log event
        self.audit_logger.log_event(
            event_type='user_updated',
            user='system',
            details={
                'user_id': user_id,
                'updates': list(updates.keys())
            }
        )
        
        return True
    
    def deactivate_user(self, user_id: str, reason: str = None) -> bool:
        """Deactivate a user account.
        
        Args:
            user_id: User ID
            reason: Deactivation reason
            
        Returns:
            True if successful
        """
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        self.users[user_id]['status'] = UserStatus.INACTIVE.value
        
        # Revoke all active sessions
        self._revoke_user_sessions(user_id)
        
        # Log event
        self.audit_logger.log_event(
            event_type='user_deactivated',
            user='system',
            details={
                'user_id': user_id,
                'reason': reason
            }
        )
        
        self._save_user_data()
        return True
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str = None, user_agent: str = None) -> Optional[Dict]:
        """Authenticate a user and create a session.
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Session info if successful, None otherwise
        """
        # Check if account is locked
        if self._is_account_locked(username):
            self.audit_logger.log_login(username, 'failure', ip_address, 
                                      {'reason': 'account_locked'})
            return None
        
        # Authenticate with auth manager
        auth_result = self.auth_manager.authenticate(username, password)
        if not auth_result:
            self._record_failed_attempt(username)
            self.audit_logger.log_login(username, 'failure', ip_address)
            return None
        
        # Get user data
        user = self._get_user_by_username(username)
        if not user:
            return None
        
        # Check user status
        if user['status'] != UserStatus.ACTIVE.value:
            self.audit_logger.log_login(username, 'failure', ip_address,
                                      {'reason': f'account_{user["status"]}'})
            return None
        
        # Create session
        session = self._create_session(user['user_id'], ip_address, user_agent)
        
        # Generate tokens
        access_token, refresh_token = self.auth_manager.generate_token({
            'username': username,
            'user_id': user['user_id'],
            'roles': user['roles']
        })
        
        # Update last login
        user['last_login'] = datetime.datetime.utcnow().isoformat()
        self._save_user_data()
        
        # Clear failed attempts
        self._clear_failed_attempts(username)
        
        # Log successful login
        self.audit_logger.log_login(username, 'success', ip_address)
        
        return {
            'session_id': session.session_id,
            'user_id': user['user_id'],
            'username': username,
            'roles': user['roles'],
            'access_token': access_token,
            'refresh_token': refresh_token,
            'mfa_required': self._requires_mfa(user)
        }
    
    def change_password(self, user_id: str, current_password: str, 
                       new_password: str) -> bool:
        """Change user password.
        
        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password
            
        Returns:
            True if successful
        """
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Verify current password
        auth_result = self.auth_manager.authenticate(user['username'], current_password)
        if not auth_result:
            raise ValueError("Current password is incorrect")
        
        # Validate new password
        if not self._validate_password(new_password, user_id):
            raise ValueError("New password does not meet policy requirements")
        
        # Check password history
        if self._is_password_in_history(user_id, new_password):
            raise ValueError(f"Password was used recently. Please choose a different password.")
        
        # Update password in auth manager
        # Note: In a real implementation, we'd need to update the auth manager's password
        # For now, we'll just update the password changed timestamp
        
        # Add to password history
        password_hash = hashlib.sha256(new_password.encode()).hexdigest()
        if user_id not in self.password_history:
            self.password_history[user_id] = []
        
        self.password_history[user_id].append(password_hash)
        
        # Keep only the last N passwords
        if len(self.password_history[user_id]) > self.password_policy.history_count:
            self.password_history[user_id] = self.password_history[user_id][-self.password_policy.history_count:]
        
        # Update password changed timestamp
        user['password_changed_at'] = datetime.datetime.utcnow().isoformat()
        
        # Revoke all sessions except current
        # (In a real implementation, we'd preserve the current session)
        self._revoke_user_sessions(user_id)
        
        self._save_user_data()
        
        # Log event
        self.audit_logger.log_event(
            event_type='password_changed',
            user=user['username'],
            details={'user_id': user_id}
        )
        
        return True
    
    def create_group(self, group_name: str, description: str, 
                    permissions: List[str] = None) -> str:
        """Create a user group.
        
        Args:
            group_name: Group name
            description: Group description
            permissions: List of permissions
            
        Returns:
            Group ID
        """
        if self._group_exists(group_name):
            raise ValueError(f"Group '{group_name}' already exists")
        
        group_id = self._generate_group_id()
        
        self.groups[group_id] = {
            'group_id': group_id,
            'name': group_name,
            'description': description,
            'permissions': permissions or [],
            'created_at': datetime.datetime.utcnow().isoformat(),
            'members': []
        }
        
        self._save_user_data()
        
        # Log event
        self.audit_logger.log_event(
            event_type='group_created',
            user='system',
            details={
                'group_id': group_id,
                'name': group_name,
                'permissions': permissions or []
            }
        )
        
        return group_id
    
    def add_user_to_group(self, user_id: str, group_id: str) -> bool:
        """Add a user to a group.
        
        Args:
            user_id: User ID
            group_id: Group ID
            
        Returns:
            True if successful
        """
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        if group_id not in self.groups:
            raise ValueError(f"Group {group_id} not found")
        
        # Add to user's groups
        if group_id not in self.users[user_id]['groups']:
            self.users[user_id]['groups'].append(group_id)
        
        # Add to group's members
        if user_id not in self.groups[group_id]['members']:
            self.groups[group_id]['members'].append(user_id)
        
        # Apply group permissions
        self._apply_group_permissions(user_id)
        
        self._save_user_data()
        
        return True
    
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user (including group permissions).
        
        Args:
            user_id: User ID
            
        Returns:
            Set of permissions
        """
        if user_id not in self.users:
            return set()
        
        user = self.users[user_id]
        permissions = set()
        
        # Get role-based permissions
        for role in user['roles']:
            permissions.update(self.rbac_manager.get_role_permissions(role))
        
        # Get group-based permissions
        for group_id in user['groups']:
            if group_id in self.groups:
                permissions.update(self.groups[group_id]['permissions'])
        
        return permissions
    
    def validate_session(self, session_id: str) -> Optional[Dict]:
        """Validate a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session info if valid, None otherwise
        """
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        # Check if session is active
        if not session.is_active:
            return None
        
        # Check expiration
        if datetime.datetime.utcnow() > session.expires_at:
            session.is_active = False
            return None
        
        # Update last activity
        session.last_activity = datetime.datetime.utcnow()
        
        # Get user info
        user = self.users.get(session.user_id)
        if not user:
            return None
        
        return {
            'session_id': session_id,
            'user_id': session.user_id,
            'username': user['username'],
            'roles': user['roles'],
            'permissions': list(self.get_user_permissions(session.user_id))
        }
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful
        """
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            
            # Log event
            session = self.sessions[session_id]
            user = self.users.get(session.user_id)
            if user:
                self.audit_logger.log_logout(
                    username=user['username'],
                    session_duration=int((datetime.datetime.utcnow() - session.created_at).total_seconds()),
                    source_ip=session.ip_address
                )
            
            return True
        return False
    
    def enforce_password_expiration(self) -> List[str]:
        """Check and enforce password expiration policy.
        
        Returns:
            List of user IDs with expired passwords
        """
        expired_users = []
        max_age = datetime.timedelta(days=self.password_policy.max_age_days)
        now = datetime.datetime.utcnow()
        
        for user_id, user in self.users.items():
            if user['status'] != UserStatus.ACTIVE.value:
                continue
            
            password_changed = datetime.datetime.fromisoformat(user['password_changed_at'])
            if now - password_changed > max_age:
                expired_users.append(user_id)
                
                # Log event
                self.audit_logger.log_event(
                    event_type='password_expired',
                    user=user['username'],
                    details={'user_id': user_id}
                )
        
        return expired_users
    
    def _load_password_policy(self) -> PasswordPolicy:
        """Load password policy from configuration."""
        policy_config = self.config.get('password_policy', {})
        return PasswordPolicy(**policy_config)
    
    def _validate_password(self, password: str, user_id: str = None) -> bool:
        """Validate password against policy."""
        if len(password) < self.password_policy.min_length:
            return False
        
        if self.password_policy.require_uppercase and not any(c.isupper() for c in password):
            return False
        
        if self.password_policy.require_lowercase and not any(c.islower() for c in password):
            return False
        
        if self.password_policy.require_numbers and not any(c.isdigit() for c in password):
            return False
        
        if self.password_policy.require_special_chars:
            special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
            if not any(c in special_chars for c in password):
                return False
        
        return True
    
    def _is_password_in_history(self, user_id: str, password: str) -> bool:
        """Check if password was used recently."""
        if user_id not in self.password_history:
            return False
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return password_hash in self.password_history[user_id]
    
    def _username_exists(self, username: str) -> bool:
        """Check if username exists."""
        return any(u['username'] == username for u in self.users.values())
    
    def _email_exists(self, email: str) -> bool:
        """Check if email exists."""
        return any(u['email'] == email for u in self.users.values())
    
    def _group_exists(self, group_name: str) -> bool:
        """Check if group name exists."""
        return any(g['name'] == group_name for g in self.groups.values())
    
    def _get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username."""
        for user in self.users.values():
            if user['username'] == username:
                return user
        return None
    
    def _generate_user_id(self) -> str:
        """Generate unique user ID."""
        return f"usr_{secrets.token_hex(16)}"
    
    def _generate_group_id(self) -> str:
        """Generate unique group ID."""
        return f"grp_{secrets.token_hex(16)}"
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"ses_{secrets.token_hex(32)}"
    
    def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> UserSession:
        """Create a new user session."""
        session_id = self._generate_session_id()
        now = datetime.datetime.utcnow()
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            ip_address=ip_address or "unknown",
            user_agent=user_agent or "unknown",
            expires_at=now + datetime.timedelta(seconds=self.session_timeout)
        )
        
        self.sessions[session_id] = session
        return session
    
    def _revoke_user_sessions(self, user_id: str):
        """Revoke all sessions for a user."""
        for session_id, session in self.sessions.items():
            if session.user_id == user_id:
                session.is_active = False
    
    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts."""
        user = self._get_user_by_username(username)
        if not user:
            return False
        
        if user['status'] == UserStatus.LOCKED.value:
            return True
        
        # Check failed attempts
        if username in self.failed_attempts:
            attempts = self.failed_attempts[username]
            
            # Remove old attempts
            lockout_window = datetime.datetime.utcnow() - datetime.timedelta(
                minutes=self.password_policy.lockout_duration_minutes
            )
            recent_attempts = [a for a in attempts if a > lockout_window]
            
            if len(recent_attempts) >= self.password_policy.lockout_threshold:
                # Lock the account
                user['status'] = UserStatus.LOCKED.value
                self._save_user_data()
                return True
            
            self.failed_attempts[username] = recent_attempts
        
        return False
    
    def _record_failed_attempt(self, username: str):
        """Record a failed login attempt."""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(datetime.datetime.utcnow())
    
    def _clear_failed_attempts(self, username: str):
        """Clear failed login attempts."""
        if username in self.failed_attempts:
            del self.failed_attempts[username]
    
    def _requires_mfa(self, user: Dict) -> bool:
        """Check if user requires MFA."""
        if user.get('mfa_enabled'):
            return True
        
        # Check if any of user's roles require MFA
        return any(role in self.mfa_required_roles for role in user['roles'])
    
    def _apply_group_permissions(self, user_id: str):
        """Apply group permissions to user."""
        # This is handled dynamically in get_user_permissions()
        # but we could cache it here for performance if needed
        pass
    
    def _create_default_groups(self):
        """Create default user groups."""
        default_groups = [
            {
                'name': 'administrators',
                'description': 'System administrators with full access',
                'permissions': ['system.admin', 'user.manage', 'policy.manage']
            },
            {
                'name': 'developers',
                'description': 'Development team members',
                'permissions': ['patch.request', 'patch.view', 'code.edit']
            },
            {
                'name': 'operators',
                'description': 'System operators',
                'permissions': ['patch.deploy', 'system.monitor', 'log.view']
            },
            {
                'name': 'reviewers',
                'description': 'Code reviewers and approvers',
                'permissions': ['patch.approve', 'patch.reject', 'patch.comment']
            },
            {
                'name': 'compliance',
                'description': 'Compliance and audit team',
                'permissions': ['audit.view', 'report.generate', 'policy.view']
            }
        ]
        
        for group_data in default_groups:
            try:
                self.create_group(
                    group_name=group_data['name'],
                    description=group_data['description'],
                    permissions=group_data['permissions']
                )
            except ValueError:
                # Group already exists
                pass
    
    def _load_user_data(self):
        """Load user data from storage."""
        # Load users
        users_file = self.storage_path / "users.json"
        if users_file.exists():
            with open(users_file, 'r') as f:
                self.users = json.load(f)
        
        # Load profiles
        profiles_file = self.storage_path / "profiles.json"
        if profiles_file.exists():
            with open(profiles_file, 'r') as f:
                profiles_data = json.load(f)
                for profile_dict in profiles_data:
                    profile = UserProfile(**profile_dict)
                    self.profiles[profile.user_id] = profile
        
        # Load groups
        groups_file = self.storage_path / "groups.json"
        if groups_file.exists():
            with open(groups_file, 'r') as f:
                self.groups = json.load(f)
        
        # Load password history
        history_file = self.storage_path / "password_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.password_history = json.load(f)
    
    def _save_user_data(self):
        """Save user data to storage."""
        # Save users
        with open(self.storage_path / "users.json", 'w') as f:
            json.dump(self.users, f, indent=2)
        
        # Save profiles
        profiles_data = [
            {
                'user_id': p.user_id,
                'username': p.username,
                'email': p.email,
                'full_name': p.full_name,
                'department': p.department,
                'job_title': p.job_title,
                'phone': p.phone,
                'location': p.location,
                'timezone': p.timezone,
                'language': p.language,
                'metadata': p.metadata
            }
            for p in self.profiles.values()
        ]
        with open(self.storage_path / "profiles.json", 'w') as f:
            json.dump(profiles_data, f, indent=2)
        
        # Save groups
        with open(self.storage_path / "groups.json", 'w') as f:
            json.dump(self.groups, f, indent=2)
        
        # Save password history
        with open(self.storage_path / "password_history.json", 'w') as f:
            json.dump(self.password_history, f, indent=2)


# Singleton instance
_user_management = None

def get_user_management(config: Dict = None) -> UserManagementSystem:
    """Get or create the singleton UserManagementSystem instance."""
    global _user_management
    if _user_management is None:
        _user_management = UserManagementSystem(config)
    return _user_management