"""
Tests for MLflow import hook security patches.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import warnings

# Set environment to ensure patches are applied
os.environ["MLFLOW_SECURITY_PATCH"] = "true"


class TestMLflowImportHook(unittest.TestCase):
    """Test MLflow import hook security patches."""

    def setUp(self):
        """Set up test environment."""
        # Remove mlflow from modules if present
        mlflow_modules = [key for key in sys.modules.keys() if key.startswith("mlflow")]
        for module in mlflow_modules:
            del sys.modules[module]

    @patch.dict(sys.modules, {"mlflow": MagicMock(), "mlflow.pyfunc": MagicMock()})
    def test_patch_existing_mlflow(self):
        """Test patching already imported mlflow."""
        # Create mock mlflow module
        mock_mlflow = sys.modules["mlflow"]
        mock_mlflow.pyfunc.load_model = MagicMock()

        # Import hook should patch it
        from modules.security.mlflow_import_hook import ensure_mlflow_security
        ensure_mlflow_security()

        # Verify original function was wrapped
        self.assertIn("SECURITY PATCHED", mock_mlflow.pyfunc.load_model.__doc__)

    def test_security_warning_on_load(self):
        """Test that security warning is issued when loading models."""
        from modules.security.mlflow_import_hook import _mlflow_patcher, SecurityWarning

        # Create a mock function
        original_func = MagicMock(return_value="model")
        original_func.__name__ = "load_model"
        original_func.__doc__ = "Original function"

        # Create secure wrapper
        with patch("modules.security.mlflow_security.load_model_securely") as mock_secure:
            mock_secure.return_value = "secure_model"
            wrapper = _mlflow_patcher._create_secure_wrapper(
                original_func,
                mock_secure
            )

            # Call wrapper and check warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = wrapper("test_model_uri")

                # Should have security warning
                self.assertEqual(len(w), 1)
                self.assertIsInstance(w[0].message, SecurityWarning)
                self.assertIn("MLflow model loading intercepted", str(w[0].message))

            # Should use secure loading
            mock_secure.assert_called_once_with("test_model_uri", use_sandbox=True)
            self.assertEqual(result, "secure_model")

    def test_disable_mlflow_imports(self):
        """Test completely disabling mlflow imports."""
        from modules.security.mlflow_import_hook import disable_mlflow_imports

        # Disable imports
        disable_mlflow_imports()

        # Try to import mlflow
        with self.assertRaises(ImportError) as cm:
            import mlflow

        self.assertIn("blocked for security reasons", str(cm.exception))
        self.assertIn("CVE-2024-37052", str(cm.exception))

    def test_auto_patch_on_import(self):
        """Test that patches are applied automatically."""
        # Import security module should trigger patching
        import modules.security

        # Verify ensure_mlflow_security was called
        self.assertIn("ensure_mlflow_security", dir(modules.security))

    @patch("modules.security.mlflow_import_hook.logger")
    def test_patch_error_handling(self, mock_logger):
        """Test error handling in patching."""
        from modules.security.mlflow_import_hook import SecureMLflowImporter

        patcher = SecureMLflowImporter()

        # Force an error during patching
        with patch.object(patcher, "_patch_existing_mlflow", side_effect=Exception("Test error")):
            with self.assertRaises(RuntimeError) as cm:
                patcher.patch_mlflow()

            self.assertIn("Failed to apply MLflow security patches", str(cm.exception))
            mock_logger.error.assert_called()

    def test_multiple_model_flavors_patched(self):
        """Test that multiple MLflow model flavors are patched."""
        # Create mock mlflow with multiple flavors
        mock_mlflow = MagicMock()
        model_flavors = ["sklearn", "tensorflow", "pytorch", "lightgbm"]

        for flavor in model_flavors:
            flavor_module = MagicMock()
            flavor_module.load_model = MagicMock()
            setattr(mock_mlflow, flavor, flavor_module)

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from modules.security.mlflow_import_hook import ensure_mlflow_security
            ensure_mlflow_security()

            # Check all flavors were patched
            for flavor in model_flavors:
                flavor_module = getattr(mock_mlflow, flavor)
                self.assertIn("SECURITY PATCHED", flavor_module.load_model.__doc__)


if __name__ == "__main__":
    unittest.main()