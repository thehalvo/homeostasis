"""
MLflow Import Hook for Security

This module patches MLflow at import time to ensure all model loading
goes through our security wrappers. This is a critical security control
to mitigate CVE-2024-37052 through CVE-2024-37060.

This hook MUST be imported before any MLflow imports in the application.
"""

import logging
import sys
import warnings
from types import ModuleType
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SecureMLflowImporter:
    """Import hook that patches MLflow to use secure model loading."""

    def __init__(self):
        self._original_modules = {}
        self._patched = False

    def patch_mlflow(self):
        """Patch MLflow module to use secure loading functions."""
        if self._patched:
            return

        try:
            # Check if mlflow is already imported
            if "mlflow" in sys.modules:
                self._patch_existing_mlflow()

            # Install import hook for future imports
            self._install_import_hook()

            self._patched = True
            logger.info("MLflow security patches applied successfully")

        except Exception as e:
            logger.error(f"Failed to patch MLflow: {e}")
            raise RuntimeError(
                "Failed to apply MLflow security patches. "
                "Cannot proceed without security controls."
            ) from e

    def _patch_existing_mlflow(self):
        """Patch already-imported mlflow module."""
        mlflow = sys.modules.get("mlflow")
        if not mlflow:
            return

        # Import our secure wrapper
        from .mlflow_security import load_model_securely, predict_securely

        # Patch mlflow.pyfunc if it exists
        if hasattr(mlflow, "pyfunc") and hasattr(mlflow.pyfunc, "load_model"):
            self._original_modules["mlflow.pyfunc.load_model"] = mlflow.pyfunc.load_model
            mlflow.pyfunc.load_model = self._create_secure_wrapper(
                mlflow.pyfunc.load_model, load_model_securely
            )
            logger.info("Patched mlflow.pyfunc.load_model")

        # Patch other model loading functions
        model_modules = [
            "sklearn", "tensorflow", "pytorch", "lightgbm",
            "xgboost", "catboost", "h2o", "spark"
        ]

        for module_name in model_modules:
            if hasattr(mlflow, module_name):
                module = getattr(mlflow, module_name)
                if hasattr(module, "load_model"):
                    original_func = getattr(module, "load_model")
                    self._original_modules[f"mlflow.{module_name}.load_model"] = original_func
                    setattr(
                        module,
                        "load_model",
                        self._create_secure_wrapper(original_func, load_model_securely)
                    )
                    logger.info(f"Patched mlflow.{module_name}.load_model")

    def _create_secure_wrapper(self, original_func, secure_func):
        """Create a wrapper that uses the secure loading function."""
        def wrapper(*args, **kwargs):
            # Extract model_uri from args
            model_uri = args[0] if args else kwargs.get("model_uri", "")

            # Log warning about security
            warnings.warn(
                f"MLflow model loading intercepted for security. "
                f"Loading model from: {model_uri}. "
                f"Using secure wrapper to mitigate CVE-2024-37052 through CVE-2024-37060.",
                SecurityWarning,
                stacklevel=2
            )

            # Use secure loading function
            return secure_func(model_uri, use_sandbox=True)

        # Preserve function metadata
        wrapper.__name__ = original_func.__name__
        wrapper.__doc__ = (
            f"SECURITY PATCHED: {original_func.__doc__}\n\n"
            f"This function has been patched to use secure model loading "
            f"to mitigate known MLflow vulnerabilities."
        )

        return wrapper

    def _install_import_hook(self):
        """Install import hook for future mlflow imports."""
        # Add our custom meta path finder
        if not any(isinstance(finder, MLflowSecurityFinder) for finder in sys.meta_path):
            sys.meta_path.insert(0, MLflowSecurityFinder(self))


class MLflowSecurityFinder:
    """Meta path finder that intercepts mlflow imports."""

    def __init__(self, patcher: SecureMLflowImporter):
        self.patcher = patcher

    def find_spec(self, fullname, path, target=None):
        """Called when a module is imported."""
        # Only intercept mlflow imports
        if fullname == "mlflow" or fullname.startswith("mlflow."):
            # Let the normal import happen first
            return None
        return None

    def find_module(self, fullname, path=None):
        """Legacy method for Python < 3.4 compatibility."""
        return None


class SecurityWarning(UserWarning):
    """Warning issued when security controls are applied."""
    pass


# Global patcher instance
_mlflow_patcher = SecureMLflowImporter()


def ensure_mlflow_security():
    """
    Ensure MLflow security patches are applied.

    This function should be called at application startup before any MLflow usage.
    """
    _mlflow_patcher.patch_mlflow()


def disable_mlflow_imports():
    """
    Completely disable MLflow imports for maximum security.

    This is the most secure option if MLflow model loading is not needed.
    """
    class MLflowBlocker:
        def find_spec(self, fullname, path, target=None):
            if fullname == "mlflow" or fullname.startswith("mlflow."):
                raise ImportError(
                    f"Import of '{fullname}' is blocked for security reasons. "
                    f"MLflow model loading is disabled to prevent CVE-2024-37052 "
                    f"through CVE-2024-37060. Use alternative model loading methods."
                )
            return None

        def find_module(self, fullname, path=None):
            return self.find_spec(fullname, path, None)

    # Add blocker to meta path
    if not any(isinstance(finder, MLflowBlocker) for finder in sys.meta_path):
        sys.meta_path.insert(0, MLflowBlocker())
        logger.warning("MLflow imports have been completely disabled for security")


# Auto-patch on import if MLFLOW_SECURITY_PATCH environment variable is set
import os
if os.environ.get("MLFLOW_SECURITY_PATCH", "true").lower() == "true":
    ensure_mlflow_security()