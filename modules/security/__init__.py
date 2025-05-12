"""
Security module for Homeostasis.

This module provides security features for production deployments of Homeostasis,
including authentication, authorization, encryption, API security, audit logging,
and approval workflows.
"""

from modules.security.auth import (
    AuthenticationError,
    AuthenticationManager,
    authenticate,
    generate_token,
    get_auth_manager,
    verify_token,
)
from modules.security.rbac import (
    PermissionDeniedError,
    RBACManager,
    get_rbac_manager,
    has_permission,
    require_permission,
)
from modules.security.api_security import (
    APISecurityManager,
    RateLimitExceededError,
    get_api_security_manager,
    secure_endpoint,
)
from modules.security.encryption import (
    EncryptionError,
    EncryptionManager,
    decrypt,
    decrypt_to_json,
    decrypt_to_string,
    encrypt,
    get_encryption_manager,
)
from modules.security.audit import (
    AuditLogger,
    get_audit_logger,
    log_event,
    log_login,
    log_fix,
)
from modules.security.approval import (
    ApprovalError,
    ApprovalManager,
    ApprovalStatus,
    ApprovalType,
    ApprovalRequest,
    create_approval_request,
    get_approval_manager,
    needs_approval,
)
from modules.security.security_config import (
    SecurityConfigError,
    SecurityConfig,
    get_security_config,
    get_config,
    set_config,
    save_config,
)
from modules.security.validators import (
    validate_email,
    validate_username,
    validate_password_strength,
    validate_url,
    validate_ip_address,
    check_for_xss,
    check_for_sql_injection,
    sanitize_input,
    sanitize_filename,
    validate_and_sanitize_json,
)

__all__ = [
    # Authentication
    'AuthenticationError',
    'AuthenticationManager',
    'authenticate',
    'generate_token',
    'get_auth_manager',
    'verify_token',
    
    # RBAC
    'PermissionDeniedError',
    'RBACManager',
    'get_rbac_manager',
    'has_permission',
    'require_permission',
    
    # API Security
    'APISecurityManager',
    'RateLimitExceededError',
    'get_api_security_manager',
    'secure_endpoint',
    
    # Encryption
    'EncryptionError',
    'EncryptionManager',
    'decrypt',
    'decrypt_to_json',
    'decrypt_to_string',
    'encrypt',
    'get_encryption_manager',
    
    # Audit Logging
    'AuditLogger',
    'get_audit_logger',
    'log_event',
    'log_login',
    'log_fix',
    
    # Approval Workflow
    'ApprovalError',
    'ApprovalManager',
    'ApprovalStatus',
    'ApprovalType',
    'ApprovalRequest',
    'create_approval_request',
    'get_approval_manager',
    'needs_approval',
    
    # Security Configuration
    'SecurityConfigError',
    'SecurityConfig',
    'get_security_config',
    'get_config',
    'set_config',
    'save_config',
    
    # Validators
    'validate_email',
    'validate_username',
    'validate_password_strength',
    'validate_url',
    'validate_ip_address',
    'check_for_xss',
    'check_for_sql_injection',
    'sanitize_input',
    'sanitize_filename',
    'validate_and_sanitize_json',
]