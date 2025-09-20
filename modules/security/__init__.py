"""
Security module for Homeostasis.

This module provides security features for production deployments of Homeostasis,
including authentication, authorization, encryption, API security, audit logging,
and approval workflows.

CRITICAL: This module includes MLflow security patches for CVE-2024-37052 through
CVE-2024-37060. The import hook is automatically applied when this module is imported.
"""

# Apply MLflow security patches immediately
from modules.security.mlflow_import_hook import ensure_mlflow_security

ensure_mlflow_security()

from modules.security.api_security import (  # noqa: E402
    APISecurityManager,
    RateLimitExceededError,
    get_api_security_manager,
    secure_endpoint,
)
from modules.security.approval import (  # noqa: E402
    ApprovalError,
    ApprovalManager,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalType,
    create_approval_request,
    get_approval_manager,
    needs_approval,
)
from modules.security.audit import (  # noqa: E402
    AuditLogger,
    get_audit_logger,
    log_event,
    log_fix,
    log_login,
)
from modules.security.auth import (  # noqa: E402
    AuthenticationError,
    AuthenticationManager,
    authenticate,
    generate_token,
    get_auth_manager,
    verify_token,
)
from modules.security.encryption import (  # noqa: E402
    EncryptionError,
    EncryptionManager,
    decrypt,
    decrypt_to_json,
    decrypt_to_string,
    encrypt,
    get_encryption_manager,
)
from modules.security.mlflow_security import (  # noqa: E402
    MLflowSecurityConfig,
    ModelSandbox,
    SecurityError,
    create_secure_mlflow_config,
    load_model_securely,
    predict_securely,
    secure_model_loader,
)
from modules.security.rbac import (  # noqa: E402
    PermissionDeniedError,
    RBACManager,
    get_rbac_manager,
    has_permission,
    require_permission,
)
from modules.security.security_config import (  # noqa: E402
    SecurityConfig,
    SecurityConfigError,
    get_config,
    get_security_config,
    save_config,
    set_config,
)
from modules.security.validators import (  # noqa: E402
    check_for_sql_injection,
    check_for_xss,
    sanitize_filename,
    sanitize_input,
    validate_and_sanitize_json,
    validate_email,
    validate_ip_address,
    validate_password_strength,
    validate_url,
    validate_username,
)

__all__ = [
    # Authentication
    "AuthenticationError",
    "AuthenticationManager",
    "authenticate",
    "generate_token",
    "get_auth_manager",
    "verify_token",
    # RBAC
    "PermissionDeniedError",
    "RBACManager",
    "get_rbac_manager",
    "has_permission",
    "require_permission",
    # API Security
    "APISecurityManager",
    "RateLimitExceededError",
    "get_api_security_manager",
    "secure_endpoint",
    # Encryption
    "EncryptionError",
    "EncryptionManager",
    "decrypt",
    "decrypt_to_json",
    "decrypt_to_string",
    "encrypt",
    "get_encryption_manager",
    # Audit Logging
    "AuditLogger",
    "get_audit_logger",
    "log_event",
    "log_login",
    "log_fix",
    # Approval Workflow
    "ApprovalError",
    "ApprovalManager",
    "ApprovalStatus",
    "ApprovalType",
    "ApprovalRequest",
    "create_approval_request",
    "get_approval_manager",
    "needs_approval",
    # Security Configuration
    "SecurityConfigError",
    "SecurityConfig",
    "get_security_config",
    "get_config",
    "set_config",
    "save_config",
    # Validators
    "validate_email",
    "validate_username",
    "validate_password_strength",
    "validate_url",
    "validate_ip_address",
    "check_for_xss",
    "check_for_sql_injection",
    "sanitize_input",
    "sanitize_filename",
    "validate_and_sanitize_json",
    # MLflow Security
    "MLflowSecurityConfig",
    "ModelSandbox",
    "SecurityError",
    "create_secure_mlflow_config",
    "load_model_securely",
    "predict_securely",
    "secure_model_loader",
    # MLflow Import Hook
    "ensure_mlflow_security",
    "disable_mlflow_imports",
    "SecurityWarning",
]
