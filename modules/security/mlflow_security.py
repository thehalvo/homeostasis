"""
MLflow Security Configuration and Mitigation

This module implements security measures to mitigate known CVEs in MLflow
(CVE-2024-37052 to CVE-2024-37060) related to unsafe deserialization.

CRITICAL SECURITY NOTES:
- These CVEs allow arbitrary code execution when loading untrusted models
- No patches are available as of September 2025
- This module implements defense-in-depth controls for production use
- All model loading MUST go through the security controls in this module
"""

import hashlib
import json
import logging
import os
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class MLflowSecurityConfig:
    """Security configuration for MLflow to mitigate known vulnerabilities."""

    def __init__(self):
        self.trusted_model_sources: List[str] = []
        self.allowed_model_hashes: Dict[str, str] = {}
        self.allowed_model_flavors: Set[str] = {
            "python_function",
            "sklearn",
            "tensorflow",
            "pytorch",
            "lightgbm",
            "xgboost",
            "catboost",
            "h2o",
            "spark",
        }
        self.blocked_model_flavors: Set[str] = {
            "pmdarima",  # CVE-2024-37055
            "diviner",  # Potential risk
            "prophet",  # Potential risk
        }
        self.enable_model_validation = True
        self.enable_sandboxing = True
        self.enable_audit_logging = True
        self.max_model_size_mb = 1000
        self.require_model_signature = True
        self.validate_input_schema = True
        self.audit_log_path = Path("/var/log/mlflow-security-audit.log")
        self._audit_lock = threading.Lock()
        self._load_config()

    def _load_config(self):
        """Load security configuration from environment or config file."""
        # Load trusted sources from environment
        trusted_sources = os.environ.get("MLFLOW_TRUSTED_SOURCES", "").split(",")
        self.trusted_model_sources = [s.strip() for s in trusted_sources if s.strip()]

        # Load from config file if exists
        config_path = Path("config/mlflow_security.json")
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self.trusted_model_sources.extend(config.get("trusted_sources", []))
                    self.allowed_model_hashes.update(config.get("allowed_hashes", {}))
                    self.max_model_size_mb = config.get("max_model_size_mb", 1000)
            except Exception as e:
                logger.error(f"Failed to load MLflow security config: {e}")

    def is_trusted_source(self, model_uri: str) -> bool:
        """Check if a model URI is from a trusted source."""
        if not self.trusted_model_sources:
            logger.warning(
                "No trusted sources configured - all models will be rejected"
            )
            return False

        return any(
            model_uri.startswith(source) for source in self.trusted_model_sources
        )

    def calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA256 hash of a model file."""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def validate_model_hash(
        self, model_path: str, expected_hash: Optional[str] = None
    ) -> bool:
        """Validate model file against known good hash."""
        actual_hash = self.calculate_model_hash(model_path)

        if expected_hash:
            return actual_hash == expected_hash

        # Check against allowed hashes
        model_name = Path(model_path).name
        return self.allowed_model_hashes.get(model_name) == actual_hash

    def check_model_size(self, model_path: str) -> bool:
        """Check if model size is within allowed limits."""
        size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        return size_mb <= self.max_model_size_mb

    def audit_log(self, action: str, details: Dict[str, Any], success: bool = True):
        """Log security-relevant actions for audit trail."""
        if not self.enable_audit_logging:
            return

        with self._audit_lock:
            try:
                # Ensure log directory exists
                self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": action,
                    "success": success,
                    "user": os.environ.get("USER", "unknown"),
                    "pid": os.getpid(),
                    "details": details,
                }

                with open(self.audit_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")

    def validate_model_flavors(
        self, model_metadata: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate model flavors against security policy."""
        flavors = model_metadata.get("flavors", {})

        for flavor in flavors:
            if flavor in self.blocked_model_flavors:
                return False, f"Blocked flavor detected: {flavor}"

            if self.allowed_model_flavors and flavor not in self.allowed_model_flavors:
                return False, f"Flavor not in allowed list: {flavor}"

        return True, "All flavors validated"


def secure_model_loader(security_config: Optional[MLflowSecurityConfig] = None):
    """
    Decorator to add security checks to MLflow model loading functions.

    Usage:
        @secure_model_loader()
        def load_model(model_uri):
            return mlflow.pyfunc.load_model(model_uri)
    """
    if security_config is None:
        security_config = MLflowSecurityConfig()

    def decorator(func):
        @wraps(func)
        def wrapper(model_uri: str, *args, **kwargs):
            start_time = datetime.utcnow()
            audit_details = {"model_uri": model_uri, "function": func.__name__}

            try:
                # Security check 1: Validate source
                if not security_config.is_trusted_source(model_uri):
                    error_msg = (
                        f"Model source not trusted: {model_uri}. "
                        "Only models from configured trusted sources can be loaded."
                    )
                    security_config.audit_log(
                        "model_load_blocked_untrusted_source",
                        audit_details,
                        success=False,
                    )
                    raise SecurityError(error_msg)

                # Security check 2: Get model metadata for validation
                try:
                    import mlflow

                    model_info = mlflow.models.get_model_info(model_uri)

                    # Check model flavors
                    # ModelInfo doesn't have to_dict, use flavors directly
                    model_metadata = {"flavors": model_info.flavors}
                    valid, msg = security_config.validate_model_flavors(model_metadata)
                    if not valid:
                        audit_details["validation_error"] = msg
                        security_config.audit_log(
                            "model_load_blocked_flavor", audit_details, success=False
                        )
                        raise SecurityError(f"Model validation failed: {msg}")

                    # Check model signature if required
                    if (
                        security_config.require_model_signature
                        and not model_info.signature
                    ):
                        audit_details["validation_error"] = "Missing model signature"
                        security_config.audit_log(
                            "model_load_blocked_no_signature",
                            audit_details,
                            success=False,
                        )
                        raise SecurityError(
                            "Model must have a signature for security validation"
                        )

                except Exception as e:
                    if isinstance(e, SecurityError):
                        raise
                    logger.warning(f"Could not retrieve model info: {e}")

                # Security check 3: Validate model properties if local file
                if model_uri.startswith("file://") or Path(model_uri).exists():
                    model_path = model_uri.replace("file://", "")

                    # Check size
                    if not security_config.check_model_size(model_path):
                        audit_details["validation_error"] = "Model size exceeds limit"
                        security_config.audit_log(
                            "model_load_blocked_size", audit_details, success=False
                        )
                        raise SecurityError(
                            f"Model size exceeds limit of {security_config.max_model_size_mb}MB"
                        )

                    # Validate hash if configured
                    if security_config.allowed_model_hashes:
                        if not security_config.validate_model_hash(model_path):
                            audit_details["validation_error"] = "Hash validation failed"
                            security_config.audit_log(
                                "model_load_blocked_hash", audit_details, success=False
                            )
                            raise SecurityError("Model hash validation failed")

                # Log the model loading attempt
                logger.info(f"Loading model from trusted source: {model_uri}")
                security_config.audit_log("model_load_started", audit_details)

                # Load the model with additional safety measures
                result = func(model_uri, *args, **kwargs)

                # Success audit log
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                audit_details["duration_ms"] = duration_ms
                security_config.audit_log("model_load_completed", audit_details)

                return result

            except Exception as e:
                # Log failure
                audit_details["error"] = str(e)
                audit_details["error_type"] = type(e).__name__
                security_config.audit_log(
                    "model_load_failed", audit_details, success=False
                )

                logger.error(f"Error loading model from {model_uri}: {e}")
                raise

        return wrapper

    return decorator


class SecurityError(Exception):
    """Exception raised for security policy violations."""

    pass


class ModelSandbox:
    """
    Sandbox environment for running MLflow models safely.

    This provides isolation for model execution to mitigate RCE vulnerabilities.
    Production implementations should use:
    - Docker containers with restricted capabilities
    - gVisor for additional kernel isolation
    - SELinux/AppArmor mandatory access controls
    - Network namespace isolation
    """

    def __init__(
        self,
        enable_network: bool = False,
        memory_limit_mb: int = 2048,
        cpu_limit: float = 1.0,
        timeout_seconds: int = 300,
        allowed_imports: Optional[Set[str]] = None,
    ):
        self.enable_network = enable_network
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit = cpu_limit
        self.timeout_seconds = timeout_seconds
        self.allowed_imports = allowed_imports or {
            "numpy",
            "pandas",
            "sklearn",
            "scipy",
            "tensorflow",
            "torch",
            "lightgbm",
            "xgboost",
            "catboost",
        }

    @contextmanager
    def sandboxed_environment(self):
        """Create a sandboxed environment context."""
        # Save original values
        original_modules = dict(sys.modules) if "sys" in globals() else {}
        original_builtins: Dict[str, Any] = {}
        if "__builtins__" in globals():
            if isinstance(__builtins__, dict):
                original_builtins = dict(__builtins__)
            else:
                # __builtins__ is a module
                original_builtins = dict(vars(__builtins__))

        try:
            # Restrict imports (simplified - use import hooks in production)
            if self.allowed_imports:
                # This is a simplified approach
                # Production should use sys.meta_path and import hooks
                pass

            # Disable dangerous builtins
            restricted_builtins = {
                "__import__": self._restricted_import,
                "compile": None,
                "eval": None,
                "exec": None,
                "open": self._restricted_open,
                "__loader__": None,
            }

            # Apply restrictions
            for key, value in restricted_builtins.items():
                if isinstance(__builtins__, dict):
                    if key in __builtins__:
                        __builtins__[key] = value
                else:
                    # __builtins__ is a module
                    if hasattr(__builtins__, key):
                        setattr(__builtins__, key, value)

            yield

        finally:
            # Restore original environment
            if "sys" in globals():
                sys.modules.clear()
                sys.modules.update(original_modules)

            if isinstance(__builtins__, dict):
                __builtins__.update(original_builtins)
            else:
                # __builtins__ is a module
                for key, value in original_builtins.items():
                    setattr(__builtins__, key, value)

    def _restricted_import(self, name, *args, **kwargs):
        """Restricted import function that only allows safe modules."""
        if name not in self.allowed_imports:
            raise SecurityError(f"Import of '{name}' is not allowed in sandbox")
        return __import__(name, *args, **kwargs)

    def _restricted_open(self, file, mode="r", *args, **kwargs):
        """Restricted open that prevents file system access."""
        raise SecurityError("File system access is not allowed in sandbox")

    def run_in_sandbox(self, model_func, *args, **kwargs):
        """
        Run a model function in a sandboxed environment.

        In production, this should use proper containerization (Docker)
        or process isolation mechanisms.
        """
        import resource
        import signal

        logger.info("Running model in sandbox environment")

        # Set up timeout
        def timeout_handler(signum, frame):
            raise TimeoutError(
                f"Model execution exceeded {self.timeout_seconds}s timeout"
            )

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout_seconds)

        try:
            # Set resource limits
            if hasattr(resource, "RLIMIT_AS"):
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (
                        self.memory_limit_mb * 1024 * 1024,
                        self.memory_limit_mb * 1024 * 1024,
                    ),
                )

            # Set CPU limit (simplified - use cgroups in production)
            if hasattr(resource, "RLIMIT_CPU"):
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (
                        int(self.timeout_seconds * self.cpu_limit),
                        resource.RLIM_INFINITY,
                    ),
                )

            # Run in restricted environment
            with self.sandboxed_environment():
                result = model_func(*args, **kwargs)
                logger.info("Model execution completed successfully in sandbox")
                return result

        except Exception as e:
            logger.error(f"Model execution failed in sandbox: {e}")
            raise
        finally:
            # Cancel timeout
            signal.alarm(0)


def create_secure_mlflow_config() -> Dict[str, Any]:
    """
    Create a secure MLflow configuration with recommended settings.
    """
    return {
        # Disable automatic model registry access
        "mlflow.disable_auto_logging": True,
        # Enable authentication (requires MLflow server setup)
        "mlflow.authentication.enabled": True,
        # Restrict model sources
        "mlflow.models.trusted_sources": [
            "file:///var/mlflow/trusted-models/",
            "s3://my-secure-bucket/models/",
            "models:/production/",
        ],
        # Enable model signature validation
        "mlflow.models.validate_signature": True,
        # Disable dangerous features
        "mlflow.recipes.enabled": False,  # Disable recipes to prevent CVE-2024-37060
        # Set strict permissions
        "mlflow.server.default_artifact_permission": "READ",
        # Enable audit logging
        "mlflow.server.audit_log_enabled": True,
    }


# Example usage functions
def load_model_securely(
    model_uri: str,
    security_config: Optional[MLflowSecurityConfig] = None,
    use_sandbox: bool = True,
) -> Any:
    """
    Securely load an MLflow model with all security checks.

    This function should be used instead of mlflow.pyfunc.load_model()
    to ensure security measures are applied.

    Args:
        model_uri: URI of the model to load
        security_config: Security configuration (uses default if None)
        use_sandbox: Whether to use sandboxing for model loading

    Returns:
        Loaded model object

    Raises:
        SecurityError: If any security check fails
    """
    import mlflow.pyfunc

    if security_config is None:
        security_config = MLflowSecurityConfig()

    # Apply security decorator
    @secure_model_loader(security_config)
    def _load_model(uri):
        if use_sandbox and security_config.enable_sandboxing:
            sandbox = ModelSandbox()
            return sandbox.run_in_sandbox(mlflow.pyfunc.load_model, uri)
        else:
            return mlflow.pyfunc.load_model(uri)

    return _load_model(model_uri)


def predict_securely(
    model: Any,
    data: Any,
    security_config: Optional[MLflowSecurityConfig] = None,
    use_sandbox: bool = True,
    validate_inputs: bool = True,
) -> Any:
    """
    Perform secure model prediction with input validation and sandboxing.

    Args:
        model: Loaded MLflow model
        data: Input data for prediction
        security_config: Security configuration
        use_sandbox: Whether to run prediction in sandbox
        validate_inputs: Whether to validate inputs

    Returns:
        Model predictions

    Raises:
        SecurityError: If validation fails
    """
    if security_config is None:
        security_config = MLflowSecurityConfig()

    # Validate inputs if required
    if validate_inputs and security_config.validate_input_schema:
        if hasattr(model, "metadata") and hasattr(model.metadata, "signature"):
            signature = model.metadata.signature
            if signature and signature.inputs:
                try:
                    # MLflow's built-in validation
                    import mlflow

                    mlflow.models.validate_serving_input(data, signature.inputs)
                except Exception as e:
                    security_config.audit_log(
                        "prediction_input_validation_failed",
                        {"error": str(e)},
                        success=False,
                    )
                    raise SecurityError(f"Input validation failed: {e}")

    # Run prediction
    if use_sandbox and security_config.enable_sandboxing:
        sandbox = ModelSandbox()
        result = sandbox.run_in_sandbox(model.predict, data)
    else:
        result = model.predict(data)

    # Audit successful prediction
    security_config.audit_log(
        "prediction_completed", {"input_shape": str(getattr(data, "shape", len(data)))}
    )

    return result


def serve_model_securely(model_uri: str, port: int = 5000):
    """
    Serve an MLflow model with security measures in place.
    """
    # This is a simplified example
    # In production, use proper MLflow server with authentication

    security_config = MLflowSecurityConfig()

    if not security_config.is_trusted_source(model_uri):
        raise SecurityError(f"Cannot serve untrusted model: {model_uri}")

    logger.info(f"Starting secure model serving for: {model_uri}")

    # Add authentication middleware, rate limiting, etc.
    # This would integrate with your MLflow server setup

    # Example command that would be run with proper security flags:
    # mlflow models serve -m {model_uri} -p {port} \
    #     --enable-mlserver \
    #     --workers 1 \
    #     --env-manager local

    raise NotImplementedError(
        "Use MLflow server with proper authentication and network restrictions. "
        "See create_secure_mlflow_config() for recommended settings."
    )
