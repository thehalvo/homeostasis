#!/usr/bin/env python3
"""
Test script for LLM CLI functionality.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_api_key_manager():
    """Test the API key manager functionality."""
    from modules.llm_integration.api_key_manager import APIKeyManager, KeyValidationError
    
    print("Testing API Key Manager...")
    
    # Use a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        manager = APIKeyManager(config_dir=temp_path)
        
        # Test listing keys (should be empty)
        keys = manager.list_keys()
        print(f"Initial keys: {keys}")
        assert all(not has_key for has_key in keys.values()), "Should have no keys initially"
        
        # Test setting a key (without validation)
        test_key = "sk-test123456789"
        manager.set_key("openai", test_key, validate=False)
        
        # Test retrieving the key
        retrieved_key = manager.get_key("openai")
        assert retrieved_key == test_key, f"Retrieved key doesn't match: {retrieved_key} != {test_key}"
        
        # Test listing keys again
        keys = manager.list_keys()
        print(f"Keys after setting OpenAI: {keys}")
        assert keys["openai"], "OpenAI key should be available"
        
        # Test masked key
        masked = manager.get_masked_key("openai")
        print(f"Masked key: {masked}")
        assert masked and masked != test_key, "Masked key should be different from original"
        
        # Test removing key
        success = manager.remove_key("openai")
        assert success, "Should successfully remove key"
        
        # Verify key is gone
        retrieved_key = manager.get_key("openai")
        assert retrieved_key is None, "Key should be None after removal"
        
        print("✓ API Key Manager tests passed!")


def test_provider_abstraction():
    """Test the provider abstraction layer."""
    from modules.llm_integration.provider_abstraction import (
        LLMMessage, LLMRequest, ProviderFactory
    )
    
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


def main():
    """Run all tests."""
    print("Running LLM CLI tests...")
    print("=" * 40)
    
    try:
        test_api_key_manager()
        test_provider_abstraction()
        test_cli_help()
        
        print("\n" + "=" * 40)
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()