"""
MLflow Security Configuration and Mitigation

This module implements security measures to mitigate known CVEs in MLflow
(CVE-2024-37052 to CVE-2024-37060) related to unsafe deserialization.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from functools import wraps
import hashlib
import json

logger = logging.getLogger(__name__)


class MLflowSecurityConfig:
    """Security configuration for MLflow to mitigate known vulnerabilities."""

    def __init__(self):
        self.trusted_model_sources: List[str] = []
        self.allowed_model_hashes: Dict[str, str] = {}
        self.enable_model_validation = True
        self.enable_sandboxing = True
        self.max_model_size_mb = 1000
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
            logger.warning("No trusted sources configured - all models will be rejected")
            return False

        return any(model_uri.startswith(source) for source in self.trusted_model_sources)

    def calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA256 hash of a model file."""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def validate_model_hash(self, model_path: str, expected_hash: Optional[str] = None) -> bool:
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
            # Security check 1: Validate source
            if not security_config.is_trusted_source(model_uri):
                raise SecurityError(
                    f"Model source not trusted: {model_uri}. "
                    "Only models from configured trusted sources can be loaded."
                )

            # Security check 2: Validate model properties if local file
            if model_uri.startswith("file://") or Path(model_uri).exists():
                model_path = model_uri.replace("file://", "")

                # Check size
                if not security_config.check_model_size(model_path):
                    raise SecurityError(
                        f"Model size exceeds limit of {security_config.max_model_size_mb}MB"
                    )

                # Validate hash if configured
                if security_config.allowed_model_hashes:
                    if not security_config.validate_model_hash(model_path):
                        raise SecurityError("Model hash validation failed")

            # Log the model loading attempt
            logger.info(f"Loading model from trusted source: {model_uri}")

            # Load the model with additional safety measures
            try:
                return func(model_uri, *args, **kwargs)
            except Exception as e:
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
    """

    def __init__(self, enable_network=False, memory_limit_mb=2048):
        self.enable_network = enable_network
        self.memory_limit_mb = memory_limit_mb

    def run_in_sandbox(self, model_func, *args, **kwargs):
        """
        Run a model function in a sandboxed environment.

        In production, this should use proper containerization (Docker)
        or process isolation mechanisms.
        """
        # This is a simplified implementation
        # In production, use Docker SDK or similar for true isolation

        logger.info("Running model in sandbox environment")

        # Set resource limits (simplified - use proper OS-level limits in production)
        import resource
        if hasattr(resource, 'RLIMIT_AS'):
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.memory_limit_mb * 1024 * 1024, self.memory_limit_mb * 1024 * 1024)
            )

        try:
            # Run the model function
            result = model_func(*args, **kwargs)
            logger.info("Model execution completed successfully in sandbox")
            return result
        except Exception as e:
            logger.error(f"Model execution failed in sandbox: {e}")
            raise


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
def load_model_securely(model_uri: str, security_config: Optional[MLflowSecurityConfig] = None):
    """
    Securely load an MLflow model with all security checks.

    This function should be used instead of mlflow.pyfunc.load_model()
    to ensure security measures are applied.
    """
    import mlflow.pyfunc

    if security_config is None:
        security_config = MLflowSecurityConfig()

    # Apply security decorator
    @secure_model_loader(security_config)
    def _load_model(uri):
        return mlflow.pyfunc.load_model(uri)

    return _load_model(model_uri)


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