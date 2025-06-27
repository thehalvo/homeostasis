#!/usr/bin/env python3
"""
CLI interface for LLM integration management.
"""

import argparse
import getpass
import sys
from typing import Optional

from .api_key_manager import APIKeyManager, KeyValidationError


def cmd_set_key(args: argparse.Namespace) -> None:
    """Handle the set-key command."""
    manager = APIKeyManager()
    
    provider = args.provider.lower()
    
    # Get API key from argument or prompt
    if args.key:
        api_key = args.key
    else:
        api_key = getpass.getpass(f"Enter your {provider.upper()} API key: ")
    
    if not api_key.strip():
        print("Error: API key cannot be empty")
        sys.exit(1)
    
    try:
        # Validate unless explicitly disabled
        validate = not args.no_validate
        manager.set_key(provider, api_key.strip(), validate=validate)
        
        if not validate:
            print("⚠️  Key saved without validation")
            
    except KeyValidationError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def cmd_list_keys(args: argparse.Namespace) -> None:
    """Handle the list-keys command."""
    manager = APIKeyManager()
    
    try:
        keys = manager.list_keys()
        
        print("LLM API Keys Status:")
        print("-" * 30)
        
        for provider, has_key in keys.items():
            status = "✓ Available" if has_key else "✗ Not set"
            
            if has_key and args.show_masked:
                masked_key = manager.get_masked_key(provider)
                if masked_key:
                    status += f" ({masked_key})"
            
            print(f"{provider.capitalize():>12}: {status}")
        
        print("\nNote: Keys can be stored in:")
        print("  • Encrypted local storage (~/.homeostasis/)")
        print("  • Environment variables (HOMEOSTASIS_<PROVIDER>_API_KEY)")
        
    except Exception as e:
        print(f"Error listing keys: {e}")
        sys.exit(1)


def cmd_remove_key(args: argparse.Namespace) -> None:
    """Handle the remove-key command."""
    manager = APIKeyManager()
    
    provider = args.provider.lower()
    
    try:
        if manager.remove_key(provider):
            print(f"✓ Removed API key for {provider}")
        else:
            print(f"No stored key found for {provider}")
            
    except Exception as e:
        print(f"Error removing key: {e}")
        sys.exit(1)


def cmd_validate_key(args: argparse.Namespace) -> None:
    """Handle the validate-key command."""
    manager = APIKeyManager()
    
    provider = args.provider.lower()
    
    try:
        api_key = manager.get_key(provider)
        if not api_key:
            print(f"Error: No API key found for {provider}")
            print("Use 'homeostasis set-key' to configure an API key")
            sys.exit(1)
        
        print(f"Validating {provider} API key...")
        
        if manager.validate_key(provider, api_key):
            print(f"✓ {provider.capitalize()} API key is valid")
        
    except KeyValidationError as e:
        print(f"✗ Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error validating key: {e}")
        sys.exit(1)


def cmd_test_providers(args: argparse.Namespace) -> None:
    """Handle the test-providers command."""
    manager = APIKeyManager()
    
    providers = args.providers if args.providers else manager.SUPPORTED_PROVIDERS
    
    print("Testing LLM provider connections...")
    print("=" * 40)
    
    results = {}
    for provider in providers:
        provider = provider.lower()
        if provider not in manager.SUPPORTED_PROVIDERS:
            print(f"⚠️  {provider}: Unsupported provider")
            continue
        
        try:
            api_key = manager.get_key(provider)
            if not api_key:
                print(f"✗ {provider.capitalize()}: No API key configured")
                results[provider] = False
                continue
            
            print(f"   Testing {provider}...", end="")
            
            if manager.validate_key(provider, api_key):
                print(" ✓ Success")
                results[provider] = True
            
        except KeyValidationError as e:
            print(f" ✗ Failed ({e})")
            results[provider] = False
        except Exception as e:
            print(f" ✗ Error ({e})")
            results[provider] = False
    
    print("\nSummary:")
    print("-" * 20)
    working_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    print(f"Working providers: {working_count}/{total_count}")
    
    if working_count == 0:
        print("\n⚠️  No working providers found!")
        print("Configure API keys using: homeostasis set-key <provider>")
        sys.exit(1)
    elif working_count < total_count:
        print("\n⚠️  Some providers are not working.")
        print("Check your API keys and network connection.")


def create_llm_cli_parser() -> argparse.ArgumentParser:
    """
    Create the CLI parser for LLM integration commands.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Homeostasis LLM Integration CLI",
        prog="homeostasis"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # set-key command
    set_key_parser = subparsers.add_parser(
        "set-key",
        help="Set API key for an LLM provider"
    )
    set_key_parser.add_argument(
        "provider",
        choices=APIKeyManager.SUPPORTED_PROVIDERS,
        help="LLM provider name"
    )
    set_key_parser.add_argument(
        "--key", "-k",
        help="API key (if not provided, will be prompted securely)"
    )
    set_key_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip API key validation"
    )
    set_key_parser.set_defaults(func=cmd_set_key)
    
    # list-keys command
    list_keys_parser = subparsers.add_parser(
        "list-keys",
        help="List configured API keys"
    )
    list_keys_parser.add_argument(
        "--show-masked",
        action="store_true",
        help="Show masked versions of the keys"
    )
    list_keys_parser.set_defaults(func=cmd_list_keys)
    
    # remove-key command
    remove_key_parser = subparsers.add_parser(
        "remove-key",
        help="Remove API key for a provider"
    )
    remove_key_parser.add_argument(
        "provider",
        choices=APIKeyManager.SUPPORTED_PROVIDERS,
        help="LLM provider name"
    )
    remove_key_parser.set_defaults(func=cmd_remove_key)
    
    # validate-key command
    validate_key_parser = subparsers.add_parser(
        "validate-key",
        help="Validate an API key"
    )
    validate_key_parser.add_argument(
        "provider",
        choices=APIKeyManager.SUPPORTED_PROVIDERS,
        help="LLM provider name"
    )
    validate_key_parser.set_defaults(func=cmd_validate_key)
    
    # test-providers command
    test_providers_parser = subparsers.add_parser(
        "test-providers",
        help="Test all configured providers"
    )
    test_providers_parser.add_argument(
        "--providers", "-p",
        nargs="+",
        choices=APIKeyManager.SUPPORTED_PROVIDERS,
        help="Specific providers to test (default: all)"
    )
    test_providers_parser.set_defaults(func=cmd_test_providers)
    
    return parser


def main() -> None:
    """Main entry point for the LLM CLI."""
    parser = create_llm_cli_parser()
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()