#!/usr/bin/env python3
"""
CLI interface for LLM integration management.
"""

import argparse
import getpass
import sys

from .api_key_manager import APIKeyManager, KeyValidationError
from .provider_abstraction import LLMManager


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
    
    # Detect and suggest corrections for common issues
    issues = manager.detect_key_issues(provider, api_key)
    if issues:
        print("⚠️  Detected potential issues with the API key:")
        for issue in issues:
            print(f"   • {issue}")
        
        suggested_correction = manager.suggest_key_correction(provider, api_key)
        if suggested_correction and suggested_correction != api_key:
            print(f"\n💡 Suggested correction: {suggested_correction[:10]}...")
            
            if not args.key:  # Only prompt for interactive input
                response = input("Apply this correction? (y/N): ").lower().strip()
                if response in ['y', 'yes']:
                    api_key = suggested_correction
                    print("✓ Applied correction")
        
        if not args.no_validate:
            print("\nProceeding with validation...")
    
    try:
        # Validate unless explicitly disabled
        validate = not args.no_validate
        manager.set_key(provider, api_key.strip(), validate=validate)
        
        if not validate:
            print("⚠️  Key saved without validation")
            
    except KeyValidationError as e:
        print(f"Error: {e}")
        
        # If validation failed, offer to save without validation
        if not args.no_validate and not args.key:  # Only for interactive input
            response = input("\nSave key without validation? (y/N): ").lower().strip()
            if response in ['y', 'yes']:
                try:
                    manager.set_key(provider, api_key.strip(), validate=False)
                    print("⚠️  Key saved without validation")
                    return
                except Exception as save_error:
                    print(f"Failed to save key: {save_error}")
        
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def cmd_list_keys(args: argparse.Namespace) -> None:
    """Handle the list-keys command."""
    manager = APIKeyManager()
    
    try:
        keys_info = manager.list_keys()
        
        print("LLM API Keys Status:")
        print("=" * 50)
        
        for provider, sources in keys_info.items():
            print(f"\n{provider.capitalize()}:")
            
            # Check if any source has the key
            has_any_key = any(sources.values())
            overall_status = "✓ Available" if has_any_key else "✗ Not set"
            
            if has_any_key and args.show_masked:
                masked_key = manager.get_masked_key(provider)
                if masked_key:
                    overall_status += f" ({masked_key})"
            
            print(f"  Status: {overall_status}")
            
            # Show detailed source information
            if args.verbose or not has_any_key:
                print("  Sources:")
                print(f"    Environment Variable: {'✓' if sources['environment'] else '✗'}")
                print(f"    External Secrets:     {'✓' if sources['external_secrets'] else '✗'}")
                print(f"    Encrypted Storage:    {'✓' if sources['encrypted_storage'] else '✗'}")
        
        # Show available secrets managers
        if args.verbose:
            available_managers = manager.get_available_secrets_managers()
            if available_managers:
                print("\nAvailable External Secrets Managers:")
                for name, manager_type in available_managers.items():
                    print(f"  • {name.upper()}: {manager_type}")
            else:
                print("\nNo external secrets managers configured")
        
        print("\nKey Storage Options:")
        print("  • Environment variables: HOMEOSTASIS_<PROVIDER>_API_KEY")
        print("  • Encrypted local storage: ~/.homeostasis/llm_keys.enc")
        print("  • External secrets managers: AWS Secrets Manager, Azure Key Vault, HashiCorp Vault")
        
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


def cmd_set_active_provider(args: argparse.Namespace) -> None:
    """Handle the set-active-provider command."""
    manager = APIKeyManager()
    
    provider = args.provider.lower() if args.provider != 'auto' else None
    
    try:
        manager.set_active_provider(provider)
    except KeyValidationError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def cmd_get_active_provider(args: argparse.Namespace) -> None:
    """Handle the get-active-provider command."""
    manager = APIKeyManager()
    
    try:
        active = manager.get_active_provider()
        if active:
            print(f"Active provider: {active}")
        else:
            print("Active provider: auto-selection")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_set_fallback_order(args: argparse.Namespace) -> None:
    """Handle the set-fallback-order command."""
    manager = APIKeyManager()
    
    try:
        manager.set_fallback_order(args.providers)
    except KeyValidationError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def cmd_get_fallback_order(args: argparse.Namespace) -> None:
    """Handle the get-fallback-order command."""
    manager = APIKeyManager()
    
    try:
        order = manager.get_fallback_order()
        enabled = manager.is_fallback_enabled()
        
        print(f"Fallback enabled: {'Yes' if enabled else 'No'}")
        print(f"Fallback order: {' → '.join(order)}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_set_fallback_enabled(args: argparse.Namespace) -> None:
    """Handle the set-fallback-enabled command."""
    manager = APIKeyManager()
    
    try:
        manager.set_enable_fallback(args.enabled)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_set_openrouter_unified(args: argparse.Namespace) -> None:
    """Handle the set-openrouter-unified command."""
    manager = APIKeyManager()
    
    try:
        manager.set_openrouter_unified_mode(
            args.enabled, 
            args.proxy_anthropic, 
            args.proxy_openai
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_set_provider_policies(args: argparse.Namespace) -> None:
    """Handle the set-provider-policies command."""
    manager = APIKeyManager()
    
    try:
        manager.set_provider_policies(
            cost_preference=args.cost,
            latency_preference=args.latency,
            reliability_preference=args.reliability
        )
    except KeyValidationError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def cmd_get_provider_policies(args: argparse.Namespace) -> None:
    """Handle the get-provider-policies command."""
    manager = APIKeyManager()
    
    try:
        policies = manager.get_provider_policies()
        print("Provider Selection Policies:")
        print(f"  Cost preference: {policies.get('cost_preference', 'balanced')}")
        print(f"  Latency preference: {policies.get('latency_preference', 'balanced')}")
        print(f"  Reliability preference: {policies.get('reliability_preference', 'high')}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_auto_configure(args: argparse.Namespace) -> None:
    """Handle the auto-configure command."""
    manager = APIKeyManager()
    llm_manager = LLMManager(manager)
    
    try:
        llm_manager.auto_configure_fallback_order()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_provider_status(args: argparse.Namespace) -> None:
    """Handle the provider-status command."""
    manager = APIKeyManager()
    
    try:
        summary = manager.get_provider_config_summary()
        
        print("Multi-Provider Configuration Status:")
        print("=" * 50)
        
        print(f"\nActive Provider: {summary['active_provider'] or 'Auto-selection'}")
        print(f"Fallback Enabled: {'Yes' if summary['fallback_enabled'] else 'No'}")
        print(f"Available Providers: {', '.join(summary['available_providers']) if summary['available_providers'] else 'None'}")
        print(f"Total Configured: {summary['total_configured_providers']}")
        
        if summary['fallback_order']:
            print(f"Fallback Order: {' → '.join(summary['fallback_order'])}")
        
        print(f"OpenRouter Unified: {'Enabled' if summary['openrouter_unified'] else 'Disabled'}")
        
        if args.verbose and summary['provider_policies']:
            print("\nProvider Policies:")
            for key, value in summary['provider_policies'].items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Show OpenRouter unified config if enabled
        if summary['openrouter_unified']:
            unified_config = manager.get_openrouter_unified_config()
            print("\nOpenRouter Unified Configuration:")
            print(f"  Proxy to Anthropic: {'Yes' if unified_config.get('proxy_to_anthropic') else 'No'}")
            print(f"  Proxy to OpenAI: {'Yes' if unified_config.get('proxy_to_openai') else 'No'}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


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
    list_keys_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed source information"
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
    
    # Multi-provider configuration commands
    
    # set-active-provider command
    set_active_parser = subparsers.add_parser(
        "set-active-provider",
        help="Set the active LLM provider"
    )
    provider_choices = APIKeyManager.SUPPORTED_PROVIDERS + ['auto']
    set_active_parser.add_argument(
        "provider",
        choices=provider_choices,
        help="Provider name or 'auto' for auto-selection"
    )
    set_active_parser.set_defaults(func=cmd_set_active_provider)
    
    # get-active-provider command
    get_active_parser = subparsers.add_parser(
        "get-active-provider",
        help="Get the current active provider"
    )
    get_active_parser.set_defaults(func=cmd_get_active_provider)
    
    # set-fallback-order command
    set_fallback_parser = subparsers.add_parser(
        "set-fallback-order",
        help="Set the fallback order for providers"
    )
    set_fallback_parser.add_argument(
        "providers",
        nargs="+",
        choices=APIKeyManager.SUPPORTED_PROVIDERS,
        help="Provider names in fallback order"
    )
    set_fallback_parser.set_defaults(func=cmd_set_fallback_order)
    
    # get-fallback-order command
    get_fallback_parser = subparsers.add_parser(
        "get-fallback-order",
        help="Get the current fallback order"
    )
    get_fallback_parser.set_defaults(func=cmd_get_fallback_order)
    
    # set-fallback-enabled command
    set_fallback_enabled_parser = subparsers.add_parser(
        "set-fallback-enabled",
        help="Enable or disable fallback"
    )
    set_fallback_enabled_parser.add_argument(
        "enabled",
        type=lambda x: x.lower() in ['true', '1', 'yes', 'on'],
        help="Enable fallback (true/false)"
    )
    set_fallback_enabled_parser.set_defaults(func=cmd_set_fallback_enabled)
    
    # set-openrouter-unified command
    set_openrouter_parser = subparsers.add_parser(
        "set-openrouter-unified",
        help="Configure OpenRouter unified mode"
    )
    set_openrouter_parser.add_argument(
        "enabled",
        type=lambda x: x.lower() in ['true', '1', 'yes', 'on'],
        help="Enable OpenRouter unified mode (true/false)"
    )
    set_openrouter_parser.add_argument(
        "--proxy-anthropic",
        action="store_true",
        default=True,
        help="Proxy Anthropic requests through OpenRouter"
    )
    set_openrouter_parser.add_argument(
        "--proxy-openai",
        action="store_true", 
        default=True,
        help="Proxy OpenAI requests through OpenRouter"
    )
    set_openrouter_parser.set_defaults(func=cmd_set_openrouter_unified)
    
    # set-provider-policies command
    set_policies_parser = subparsers.add_parser(
        "set-provider-policies",
        help="Set provider selection policies"
    )
    set_policies_parser.add_argument(
        "--cost",
        choices=['low', 'balanced', 'high'],
        default='balanced',
        help="Cost preference (default: balanced)"
    )
    set_policies_parser.add_argument(
        "--latency",
        choices=['low', 'balanced', 'high'],
        default='balanced',
        help="Latency preference (default: balanced)"
    )
    set_policies_parser.add_argument(
        "--reliability",
        choices=['low', 'balanced', 'high'],
        default='high',
        help="Reliability preference (default: high)"
    )
    set_policies_parser.set_defaults(func=cmd_set_provider_policies)
    
    # get-provider-policies command
    get_policies_parser = subparsers.add_parser(
        "get-provider-policies",
        help="Get current provider selection policies"
    )
    get_policies_parser.set_defaults(func=cmd_get_provider_policies)
    
    # auto-configure command
    auto_configure_parser = subparsers.add_parser(
        "auto-configure",
        help="Automatically configure provider fallback order based on policies"
    )
    auto_configure_parser.set_defaults(func=cmd_auto_configure)
    
    # provider-status command
    provider_status_parser = subparsers.add_parser(
        "provider-status",
        help="Show multi-provider configuration status"
    )
    provider_status_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information"
    )
    provider_status_parser.set_defaults(func=cmd_provider_status)
    
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