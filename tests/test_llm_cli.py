#!/usr/bin/env python3
"""
Test script for LLM CLI functionality.
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

# Add project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_api_key_manager():
    """Test the API key manager functionality."""
    from modules.llm_integration.api_key_manager import APIKeyManager

    print("Testing API Key Manager...")

    # Use a temporary directory for testing and mock the password
    with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
        "getpass.getpass", return_value="test_password"
    ):
        temp_path = Path(temp_dir)
        manager = APIKeyManager(config_dir=temp_path)

        # Test listing keys (should be empty)
        keys = manager.list_keys()
        print(f"Initial keys: {keys}")
        assert all(
            not any(sources.values()) for sources in keys.values()
        ), "Should have no keys initially"

        # Test setting a key (without validation)
        test_key = "sk-test123456789"
        manager.set_key("openai", test_key, validate=False)

        # Test retrieving the key
        retrieved_key = manager.get_key("openai")
        assert (
            retrieved_key == test_key
        ), f"Retrieved key doesn't match: {retrieved_key} != {test_key}"

        # Test listing keys again
        keys = manager.list_keys()
        print(f"Keys after setting OpenAI: {keys}")
        assert keys["openai"], "OpenAI key should be available"

        # Test masked key
        masked = manager.get_masked_key("openai")
        print(f"Masked key: {masked}")
        assert (
            masked and masked != test_key
        ), "Masked key should be different from original"

        # Test removing key
        success = manager.remove_key("openai")
        assert success, "Should successfully remove key"

        # Verify key is gone
        retrieved_key = manager.get_key("openai")
        assert retrieved_key is None, "Key should be None after removal"

        print("✓ API Key Manager tests passed!")


def test_provider_abstraction():
    """Test the provider abstraction layer."""
    from modules.llm_integration.provider_abstraction import (LLMMessage,
                                                              LLMRequest,
                                                              ProviderFactory)

    print("Testing Provider Abstraction...")

    # Test creating providers
    providers = ProviderFactory.get_supported_providers()
    print(f"Supported providers: {providers}")
    assert "openai" in providers
    assert "anthropic" in providers
    assert "openrouter" in providers

    # Test message and request creation
    message = LLMMessage(role="user", content="Hello")
    request = LLMRequest(messages=[message], max_tokens=10)

    assert message.role == "user"
    assert message.content == "Hello"
    assert request.messages[0] == message
    assert request.max_tokens == 10

    print("✓ Provider Abstraction tests passed!")


def test_cli_help():
    """Test CLI help functionality."""
    print("Testing CLI help...")

    # Import the CLI module
    from modules.llm_integration.llm_cli import create_llm_cli_parser

    parser = create_llm_cli_parser()

    # Test that help can be generated without errors
    try:
        help_text = parser.format_help()
        assert "set-key" in help_text
        assert "list-keys" in help_text
        assert "validate-key" in help_text
        print("✓ CLI help generation works!")
    except Exception as e:
        print(f"✗ CLI help generation failed: {e}")
        raise


def test_multi_provider_config():
    """Test multi-provider configuration functionality."""
    from modules.llm_integration.api_key_manager import APIKeyManager
    from modules.llm_integration.provider_abstraction import LLMManager

    print("Testing Multi-Provider Configuration...")

    # Use a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        manager = APIKeyManager(config_dir=temp_path, use_external_secrets=False)

        # Test default configuration
        config = manager.get_provider_config_summary()
        print(f"Default config: {config}")
        assert config["active_provider"] is None
        assert config["fallback_enabled"] is True
        assert config["total_configured_providers"] == 0

        # Test setting active provider
        manager.set_active_provider("anthropic")
        active = manager.get_active_provider()
        assert (
            active == "anthropic"
        ), f"Active provider should be anthropic, got {active}"

        # Test setting active provider to auto
        manager.set_active_provider(None)
        active = manager.get_active_provider()
        assert active is None, "Active provider should be None for auto-selection"

        # Test fallback order
        original_order = manager.get_fallback_order()
        assert isinstance(original_order, list), "Fallback order should be a list"

        new_order = ["openai", "anthropic", "openrouter"]
        manager.set_fallback_order(new_order)
        retrieved_order = manager.get_fallback_order()
        assert (
            retrieved_order == new_order
        ), f"Fallback order doesn't match: {retrieved_order} != {new_order}"

        # Test fallback enabled/disabled
        manager.set_enable_fallback(False)
        assert not manager.is_fallback_enabled(), "Fallback should be disabled"

        manager.set_enable_fallback(True)
        assert manager.is_fallback_enabled(), "Fallback should be enabled"

        # Test OpenRouter unified mode
        manager.set_openrouter_unified_mode(
            True, proxy_to_anthropic=True, proxy_to_openai=False
        )
        unified_config = manager.get_openrouter_unified_config()
        assert unified_config["enabled"] is True
        assert unified_config["proxy_to_anthropic"] is True
        assert unified_config["proxy_to_openai"] is False

        # Test provider policies
        manager.set_provider_policies(
            cost_preference="low",
            latency_preference="high",
            reliability_preference="balanced",
        )
        policies = manager.get_provider_policies()
        assert policies["cost_preference"] == "low"
        assert policies["latency_preference"] == "high"
        assert policies["reliability_preference"] == "balanced"

        # Test LLM Manager with provider selection
        llm_manager = LLMManager(manager)
        available = llm_manager.get_available_providers()
        assert isinstance(available, list), "Available providers should be a list"

        # Test recommended provider order
        recommended = llm_manager.get_recommended_provider_order()
        assert isinstance(recommended, list), "Recommended order should be a list"

        print("✓ Multi-Provider Configuration tests passed!")


def test_cli_multi_provider_commands():
    """Test multi-provider CLI commands."""
    print("Testing Multi-Provider CLI commands...")

    # Import the CLI module
    from modules.llm_integration.llm_cli import create_llm_cli_parser

    parser = create_llm_cli_parser()

    # Test that new commands are available in help
    try:
        help_text = parser.format_help()

        # Check for new multi-provider commands
        multi_provider_commands = [
            "set-active-provider",
            "get-active-provider",
            "set-fallback-order",
            "get-fallback-order",
            "set-fallback-enabled",
            "set-openrouter-unified",
            "set-provider-policies",
            "get-provider-policies",
            "auto-configure",
            "provider-status",
        ]

        for command in multi_provider_commands:
            assert command in help_text, f"Command {command} not found in help"

        print("✓ Multi-Provider CLI commands available!")

    except Exception as e:
        print(f"✗ Multi-Provider CLI command test failed: {e}")
        raise


def main():
    """Run all tests."""
    print("Running LLM CLI tests...")
    print("=" * 40)

    try:
        test_api_key_manager()
        test_provider_abstraction()
        test_cli_help()
        test_multi_provider_config()
        test_cli_multi_provider_commands()

        print("\n" + "=" * 40)
        print("✓ All tests passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
