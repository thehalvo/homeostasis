"""
Tests for MLflow import hook security patches.
"""

import os

# Disable auto-patching for tests - MUST be done before any imports
os.environ["MLFLOW_SECURITY_PATCH"] = "false"

import sys
import unittest
from unittest.mock import MagicMock, patch, Mock
import warnings

# Note: We'll handle mlflow_security mocking differently to avoid affecting other tests

# Save the original module if it exists
_original_mlflow_security = sys.modules.get("modules.security.mlflow_security")


def _cleanup_mlflow_security():
    """Cleanup function to restore original mlflow_security module."""
    if _original_mlflow_security is not None:
        sys.modules["modules.security.mlflow_security"] = _original_mlflow_security
    else:
        sys.modules.pop("modules.security.mlflow_security", None)


# Create a mock mlflow_security module temporarily
mock_mlflow_security = Mock()
mock_mlflow_security.load_model_securely = Mock(return_value="secure_model")
sys.modules["modules.security.mlflow_security"] = mock_mlflow_security

try:
    # Import the mlflow_import_hook module directly to avoid security.__init__.py
    import importlib.util
    from pathlib import Path

    # Get the path relative to this test file
    test_dir = Path(__file__).parent
    repo_root = test_dir.parent
    mlflow_hook_path = repo_root / "modules" / "security" / "mlflow_import_hook.py"

    spec = importlib.util.spec_from_file_location(
        "modules.security.mlflow_import_hook",
        str(mlflow_hook_path),
    )
    mlflow_import_hook = importlib.util.module_from_spec(spec)
    mlflow_import_hook.__package__ = "modules.security"
    sys.modules["modules.security.mlflow_import_hook"] = mlflow_import_hook
    spec.loader.exec_module(mlflow_import_hook)
finally:
    # Clean up immediately after import
    _cleanup_mlflow_security()


class TestMLflowImportHook(unittest.TestCase):
    """Test MLflow import hook security patches."""

    def setUp(self):
        """Set up test environment."""
        # Remove mlflow from modules if present
        mlflow_modules = [key for key in sys.modules.keys() if key.startswith("mlflow")]
        for module in mlflow_modules:
            del sys.modules[module]

    def test_patch_existing_mlflow(self):
        """Test patching already imported mlflow."""
        # Create mock mlflow module
        mock_mlflow = MagicMock()
        mock_pyfunc = MagicMock()

        # Create a proper function to mock load_model
        def mock_load_model(*args, **kwargs):
            return "original_model"

        mock_load_model.__doc__ = "Original load_model function"
        mock_pyfunc.load_model = mock_load_model
        mock_mlflow.pyfunc = mock_pyfunc

        # Add other expected modules as None
        for module in [
            "sklearn",
            "tensorflow",
            "pytorch",
            "lightgbm",
            "xgboost",
            "catboost",
            "h2o",
            "spark",
        ]:
            setattr(mock_mlflow, module, None)

        # Mock the secure loading function
        mock_load_securely = MagicMock(return_value="secure_model")

        with patch.dict(
            sys.modules, {"mlflow": mock_mlflow, "mlflow.pyfunc": mock_pyfunc}
        ):
            # Test without triggering import-time patching
            with patch.dict(os.environ, {"MLFLOW_SECURITY_PATCH": "false"}):
                # Clear any cached imports
                if "modules.security.mlflow_import_hook" in sys.modules:
                    del sys.modules["modules.security.mlflow_import_hook"]

                # Use the already imported module with mocked security
                # Note: load_model_securely is already mocked during module import
                patcher = mlflow_import_hook.SecureMLflowImporter()
                patcher._patch_existing_mlflow()

                # Verify original function was wrapped
                self.assertIn("SECURITY PATCHED", mock_mlflow.pyfunc.load_model.__doc__)

    def test_security_warning_on_load(self):
        """Test that security warning is issued when loading models."""
        # Note: load_model_securely is already mocked during module import
        _mlflow_patcher = mlflow_import_hook._mlflow_patcher
        SecurityWarning = mlflow_import_hook.SecurityWarning

        # Create a proper function to test
        def original_func(*args, **kwargs):
            return "model"

        original_func.__doc__ = "Original function"

        # Create secure wrapper
        wrapper = _mlflow_patcher._create_secure_wrapper(
            original_func, mock_mlflow_security.load_model_securely
        )

        # Reset the mock before testing
        mock_mlflow_security.load_model_securely.reset_mock()

        # Call wrapper and check warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wrapper("test_model_uri")

            # Should have security warning
            self.assertEqual(len(w), 1)
            self.assertIsInstance(w[0].message, SecurityWarning)
            self.assertIn("MLflow model loading intercepted", str(w[0].message))

        # Should use secure loading
        mock_mlflow_security.load_model_securely.assert_called_once_with(
            "test_model_uri", use_sandbox=True
        )
        self.assertEqual(result, "secure_model")

    def test_disable_mlflow_imports(self):
        """Test completely disabling mlflow imports."""
        # Note: load_model_securely is already mocked during module import
        disable_mlflow_imports = mlflow_import_hook.disable_mlflow_imports

        # Disable imports
        disable_mlflow_imports()

        # Try to import mlflow
        with self.assertRaises(ImportError) as cm:
            import mlflow  # noqa: F401

        self.assertIn("blocked for security reasons", str(cm.exception))
        self.assertIn("CVE-2024-37052", str(cm.exception))

    def test_auto_patch_on_import(self):
        """Test that patches are applied automatically."""
        # Skip this test since we're mocking the security module
        self.skipTest("Skipping auto-patch test as we're mocking the security module")

    def test_patch_error_handling(self):
        """Test error handling in patching."""
        with patch.dict(os.environ, {"MLFLOW_SECURITY_PATCH": "false"}):
            # Create a fresh patcher instance that hasn't been patched yet
            patcher = mlflow_import_hook.SecureMLflowImporter()
            patcher._patched = False  # Reset patched state

            # Add a mock mlflow to sys.modules so _patch_existing_mlflow will be called
            mock_mlflow = MagicMock()

            # Force an error during patching
            with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
                with patch.object(
                    patcher,
                    "_patch_existing_mlflow",
                    side_effect=Exception("Test error"),
                ):
                    # Patch logger locally
                    with patch.object(mlflow_import_hook, "logger") as mock_logger:
                        with self.assertRaises(RuntimeError) as cm:
                            patcher.patch_mlflow()

                        self.assertIn(
                            "Failed to apply MLflow security patches", str(cm.exception)
                        )
                        mock_logger.error.assert_called()

    def test_multiple_model_flavors_patched(self):
        """Test that multiple MLflow model flavors are patched."""
        # Create mock mlflow with multiple flavors
        mock_mlflow = MagicMock()

        # Add pyfunc module
        mock_pyfunc = MagicMock()

        def mock_pyfunc_load_model(*args, **kwargs):
            return "pyfunc_model"

        mock_pyfunc_load_model.__doc__ = "Original pyfunc load_model function"
        mock_pyfunc.load_model = mock_pyfunc_load_model
        mock_mlflow.pyfunc = mock_pyfunc

        model_flavors = [
            "sklearn",
            "tensorflow",
            "pytorch",
            "lightgbm",
            "xgboost",
            "catboost",
            "h2o",
            "spark",
        ]

        for flavor in model_flavors:
            flavor_module = MagicMock()

            # Create a proper function to mock load_model
            def make_mock_load_model(flavor_name):
                def mock_load_model(*args, **kwargs):
                    return f"{flavor_name}_model"

                mock_load_model.__doc__ = f"Original {flavor_name} load_model function"
                return mock_load_model

            flavor_module.load_model = make_mock_load_model(flavor)
            setattr(mock_mlflow, flavor, flavor_module)

        # Mock the secure loading function
        mock_load_securely = MagicMock(return_value="secure_model")

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with patch.dict(os.environ, {"MLFLOW_SECURITY_PATCH": "false"}):
                # Clear any cached imports
                if "modules.security.mlflow_import_hook" in sys.modules:
                    del sys.modules["modules.security.mlflow_import_hook"]

                # Create fresh patcher instance
                patcher = mlflow_import_hook.SecureMLflowImporter()
                patcher._patch_existing_mlflow()

                # Check all flavors were patched
                for flavor in model_flavors:
                    flavor_module = getattr(mock_mlflow, flavor)
                    self.assertIn("SECURITY PATCHED", flavor_module.load_model.__doc__)


if __name__ == "__main__":
    unittest.main()
