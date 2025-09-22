#!/usr/bin/env python3
"""
Test Dashboard LLM Integration

Tests the web dashboard LLM key management functionality.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dashboard.app import DashboardServer
from modules.llm_integration.api_key_manager import APIKeyManager, KeyValidationError


class TestDashboardLLMIntegration(unittest.TestCase):
    """Test cases for dashboard LLM integration."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)

        # Create patches for all dashboard dependencies
        self.patches = [
            patch('dashboard.app.MetricsCollector'),
            patch('dashboard.app.LLMMetricsCollector'),
            patch('dashboard.app.SecurityGuardrails'),
            patch('dashboard.app.CostTracker'),
            patch('dashboard.app.get_suggestion_manager'),
            patch('dashboard.app.get_auth_manager'),
            patch('dashboard.app.get_rbac_manager'),
            patch('dashboard.app.get_api_security_manager'),
            patch('dashboard.app.get_canary_deployment'),
            patch('dashboard.app.APIKeyManager'),  # Patch APIKeyManager where it's used
        ]

        # Also patch KeyValidationError to be accessible
        self.key_validation_error_patch = patch('dashboard.app.KeyValidationError', KeyValidationError)
        self.key_validation_error_patch.start()

        # Start all patches
        self.mocks = []
        for p in self.patches:
            mock = p.start()
            mock.return_value = MagicMock()
            self.mocks.append(mock)

        # Set up specific mock behaviors
        self.api_key_manager_mock = self.mocks[-1]  # Last one is APIKeyManager
        mock_instance = MagicMock()
        mock_instance.list_keys.return_value = {
            "openai": {"configured": True, "type": "environment"},
            "anthropic": {"configured": False, "type": None},
            "openrouter": {"configured": False, "type": None}
        }
        mock_instance.get_available_secrets_managers.return_value = ["environment", "file"]
        mock_instance.validate_key.return_value = True
        mock_instance.set_key.return_value = True
        mock_instance.get_key.return_value = "test-key"
        mock_instance.remove_key.return_value = True
        mock_instance.SUPPORTED_PROVIDERS = ["openai", "anthropic", "openrouter"]
        self.api_key_manager_mock.return_value = mock_instance

        # Create dashboard server with mocked dependencies
        self.dashboard = DashboardServer(debug=True)

        # Create test client
        self.client = self.dashboard.app.test_client()

    def tearDown(self):
        """Clean up patches."""
        for p in self.patches:
            p.stop()
        self.key_validation_error_patch.stop()

    def test_llm_keys_api_endpoint(self):
        """Test LLM keys status API endpoint."""
        response = self.client.get("/api/llm-keys")
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response data: {response.get_data(as_text=True)}")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("providers", data)
        self.assertIn("secrets_managers", data)

        # Check that all supported providers are included
        expected_providers = ["openai", "anthropic", "openrouter"]
        for provider in expected_providers:
            self.assertIn(provider, data["providers"])

    def test_set_openai_key(self):
        """Test setting OpenAI API key."""
        response = self.client.post(
            "/api/llm-keys/openai",
            json={"api_key": "sk-test1234567890123456789012345678901234567890"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("Openai API key set successfully", data["message"])

    def test_set_invalid_key_format(self):
        """Test setting API key with invalid format."""
        # Mock validate_key to raise KeyValidationError
        self.api_key_manager_mock.return_value.validate_key.side_effect = KeyValidationError(
            "Invalid OpenAI API key format. OpenAI keys should start with 'sk-'. Please check your key and try again."
        )

        response = self.client.post(
            "/api/llm-keys/openai", json={"api_key": "invalid-key"}
        )

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Invalid OpenAI API key format", data["message"])

    def test_set_empty_key(self):
        """Test setting empty API key."""
        response = self.client.post("/api/llm-keys/openai", json={"api_key": ""})

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("API key cannot be empty", data["message"])

    def test_missing_key_in_request(self):
        """Test request without API key."""
        response = self.client.post("/api/llm-keys/openai", json={})

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("API key is required", data["message"])

    def test_test_key_endpoint(self):
        """Test API key testing endpoint."""

        # Test with key in request body
        response = self.client.post(
            "/api/llm-keys/openai/test",
            json={"api_key": "sk-test1234567890123456789012345678901234567890"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("Openai API key is valid", data["message"])

    def test_test_key_not_found(self):
        """Test testing key when no key is configured."""
        # Mock get_key to return None (no key configured)
        self.api_key_manager_mock.return_value.get_key.return_value = None

        response = self.client.post("/api/llm-keys/openai/test", json={})

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("No openai API key found to test", data["message"])

    def test_remove_key_not_found(self):
        """Test removing key when none exists."""
        # Mock remove_key to return False (no key exists)
        self.api_key_manager_mock.return_value.remove_key.return_value = False

        response = self.client.delete("/api/llm-keys/openai")

        # Should return 404 when no key is found
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("No openai API key found to remove", data["message"])

    def test_test_all_keys_endpoint(self):
        """Test testing all API keys endpoint."""
        # Mock get_key to return None for all providers (no keys configured)
        self.api_key_manager_mock.return_value.get_key.return_value = None

        response = self.client.post("/api/llm-keys/test-all")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("results", data)

        # All providers should be in results
        expected_providers = ["openai", "anthropic", "openrouter"]
        for provider in expected_providers:
            self.assertIn(provider, data["results"])
            # Should indicate no key configured
            self.assertFalse(data["results"][provider]["success"])
            self.assertIn("No", data["results"][provider]["message"])

    def test_configuration_page_includes_llm_tab(self):
        """Test that configuration page includes LLM keys tab."""
        response = self.client.get("/config")
        self.assertEqual(response.status_code, 200)

        # Check that the LLM keys tab is present
        content = response.get_data(as_text=True)
        self.assertIn("llm-keys-tab", content)
        self.assertIn("LLM Keys", content)
        self.assertIn("LLM API Keys Management", content)


class TestAPIKeyManagerCoordination(unittest.TestCase):
    """Test coordination between CLI and dashboard."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)

    def test_shared_key_manager(self):
        """Test that CLI and dashboard share the same key storage."""
        # Create API key manager (simulates CLI usage)
        cli_manager = APIKeyManager(config_dir=self.config_dir)

        # Create another manager (simulates dashboard usage)
        dashboard_manager = APIKeyManager(config_dir=self.config_dir)

        # Set key via CLI manager
        with patch("modules.llm_integration.api_key_manager.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            cli_manager.set_key(
                "openai", "sk-test1234567890123456789012345678901234567890"
            )

        # Check that dashboard manager can see the key
        keys_status = dashboard_manager.list_keys()
        self.assertTrue(keys_status["openai"]["encrypted_storage"])

        # Remove key via dashboard manager
        dashboard_manager.remove_key("openai")

        # Check that CLI manager sees the removal
        cli_keys_status = cli_manager.list_keys()
        self.assertFalse(cli_keys_status["openai"]["encrypted_storage"])


if __name__ == "__main__":
    # Set up test environment
    os.environ["TESTING"] = "1"

    # Run tests
    unittest.main()
