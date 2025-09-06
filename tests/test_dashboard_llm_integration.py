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
from modules.llm_integration.api_key_manager import APIKeyManager


class TestDashboardLLMIntegration(unittest.TestCase):
    """Test cases for dashboard LLM integration."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)

        # Create dashboard server
        self.dashboard = DashboardServer(debug=True)

        # Create test client
        self.client = self.dashboard.app.test_client()

    def test_llm_keys_api_endpoint(self):
        """Test LLM keys status API endpoint."""
        response = self.client.get("/api/llm-keys")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("providers", data)
        self.assertIn("secrets_managers", data)

        # Check that all supported providers are included
        expected_providers = ["openai", "anthropic", "openrouter"]
        for provider in expected_providers:
            self.assertIn(provider, data["providers"])

    @patch("dashboard.app.APIKeyManager")
    def test_set_openai_key(self, mock_api_key_manager):
        """Test setting OpenAI API key."""
        # Mock the APIKeyManager instance
        mock_manager = MagicMock()
        mock_manager.validate_key.return_value = True
        mock_manager.set_key.return_value = True
        mock_api_key_manager.return_value = mock_manager

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

    @patch("modules.llm_integration.api_key_manager.requests.get")
    def test_test_key_endpoint(self, mock_get):
        """Test API key testing endpoint."""
        # Mock successful validation
        mock_get.return_value.status_code = 200

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
        response = self.client.post("/api/llm-keys/openai/test", json={})

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("No openai API key found to test", data["message"])

    def test_remove_key_not_found(self):
        """Test removing key when none exists."""
        response = self.client.delete("/api/llm-keys/openai")

        # Should return 404 when no key is found
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("No openai API key found to remove", data["message"])

    def test_test_all_keys_endpoint(self):
        """Test testing all API keys endpoint."""
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
