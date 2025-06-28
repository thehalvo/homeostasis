#!/usr/bin/env python3
"""
API Key Manager for LLM Integration

Handles secure storage, validation, and management of API keys for various LLM providers.
"""

import os
import json
import base64
import getpass
from pathlib import Path
from typing import Dict, Optional, List, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import requests

from .secrets_managers import secrets_registry, SecretsManagerError


class KeyValidationError(Exception):
    """Raised when API key validation fails."""
    pass


class APIKeyManager:
    """
    Manages API keys for LLM providers with secure storage and validation.
    """
    
    SUPPORTED_PROVIDERS = ['openai', 'anthropic', 'openrouter']
    CONFIG_DIR = Path.home() / '.homeostasis'
    KEYS_FILE = CONFIG_DIR / 'llm_keys.enc'
    SALT_FILE = CONFIG_DIR / 'salt'
    
    def __init__(self, config_dir: Optional[Path] = None, use_external_secrets: bool = True):
        """
        Initialize the API key manager.
        
        Args:
            config_dir: Optional custom configuration directory
            use_external_secrets: Whether to use external secrets managers
        """
        if config_dir:
            self.config_dir = config_dir
            self.keys_file = config_dir / 'llm_keys.enc'
            self.salt_file = config_dir / 'salt'
        else:
            self.config_dir = self.CONFIG_DIR
            self.keys_file = self.KEYS_FILE
            self.salt_file = self.SALT_FILE
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        self._cipher_suite = None
        self._keys = {}
        self._use_external_secrets = use_external_secrets
    
    def _get_password(self) -> str:
        """
        Get password for encryption/decryption.
        
        Returns:
            Password string
        """
        # For now, use a simple approach. In production, consider more secure methods
        return getpass.getpass("Enter password for key storage: ")
    
    def _get_cipher_suite(self) -> Fernet:
        """
        Get or create cipher suite for encryption/decryption.
        
        Returns:
            Fernet cipher suite
        """
        if self._cipher_suite is not None:
            return self._cipher_suite
        
        # Get or create salt
        if self.salt_file.exists():
            salt = self.salt_file.read_bytes()
        else:
            salt = os.urandom(16)
            self.salt_file.write_bytes(salt)
        
        # Get password and derive key
        password = self._get_password()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self._cipher_suite = Fernet(key)
        
        return self._cipher_suite
    
    def _load_keys(self) -> Dict[str, str]:
        """
        Load encrypted keys from storage.
        
        Returns:
            Dictionary of provider -> API key mappings
        """
        if not self.keys_file.exists():
            return {}
        
        try:
            cipher_suite = self._get_cipher_suite()
            encrypted_data = self.keys_file.read_bytes()
            decrypted_data = cipher_suite.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            raise KeyValidationError(f"Failed to load keys: {e}")
    
    def _save_keys(self, keys: Dict[str, str]) -> None:
        """
        Save keys to encrypted storage.
        
        Args:
            keys: Dictionary of provider -> API key mappings
        """
        try:
            cipher_suite = self._get_cipher_suite()
            data = json.dumps(keys).encode()
            encrypted_data = cipher_suite.encrypt(data)
            self.keys_file.write_bytes(encrypted_data)
        except Exception as e:
            raise KeyValidationError(f"Failed to save keys: {e}")
    
    def set_key(self, provider: str, api_key: str, validate: bool = True) -> None:
        """
        Set API key for a provider.
        
        Args:
            provider: Provider name (openai, anthropic, openrouter)
            api_key: API key string
            validate: Whether to validate the key
        
        Raises:
            KeyValidationError: If provider is unsupported or validation fails
        """
        provider = provider.lower()
        if provider not in self.SUPPORTED_PROVIDERS:
            raise KeyValidationError(f"Unsupported provider: {provider}. Supported: {self.SUPPORTED_PROVIDERS}")
        
        if validate:
            self.validate_key(provider, api_key)
        
        # Load existing keys
        keys = self._load_keys()
        keys[provider] = api_key
        
        # Save updated keys
        self._save_keys(keys)
        self._keys[provider] = api_key
        
        # Try to store in external secrets manager if available
        if self._use_external_secrets:
            external_success = self.set_key_in_external_secrets(provider, api_key)
            if external_success:
                print(f"✓ API key for {provider} saved to encrypted storage and external secrets")
            else:
                print(f"✓ API key for {provider} saved to encrypted storage")
        else:
            print(f"✓ API key for {provider} saved successfully")
    
    def get_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            API key if found, None otherwise
        """
        provider = provider.lower()
        
        # Check memory cache first
        if provider in self._keys:
            return self._keys[provider]
        
        # Check environment variables
        env_var = f"HOMEOSTASIS_{provider.upper()}_API_KEY"
        env_key = os.getenv(env_var)
        if env_key:
            self._keys[provider] = env_key
            return env_key
        
        # Check external secrets managers
        if self._use_external_secrets:
            external_key = self._get_key_from_external_secrets(provider)
            if external_key:
                self._keys[provider] = external_key
                return external_key
        
        # Load from encrypted storage
        try:
            keys = self._load_keys()
            key = keys.get(provider)
            if key:
                self._keys[provider] = key
            return key
        except Exception:
            return None
    
    def _get_key_from_external_secrets(self, provider: str) -> Optional[str]:
        """Get API key from external secrets managers."""
        try:
            manager = secrets_registry.get_preferred_manager()
            if manager:
                secret_name = f"{provider}_api_key"
                return manager.get_secret(secret_name)
        except SecretsManagerError:
            pass  # Fall back to other methods
        except Exception:
            pass  # Fall back to other methods
        
        return None
    
    def set_key_in_external_secrets(self, provider: str, api_key: str) -> bool:
        """Set API key in external secrets manager."""
        if not self._use_external_secrets:
            return False
        
        try:
            manager = secrets_registry.get_preferred_manager()
            if manager:
                secret_name = f"{provider}_api_key"
                return manager.set_secret(secret_name, api_key)
        except SecretsManagerError as e:
            print(f"⚠️  External secrets manager error: {e}")
        except Exception as e:
            print(f"⚠️  Failed to store key in external secrets: {e}")
        
        return False
    
    def get_available_secrets_managers(self) -> Dict[str, str]:
        """Get available external secrets managers."""
        available = secrets_registry.get_available_managers()
        return {name: type(manager).__name__ for name, manager in available.items()}
    
    def list_keys(self) -> Dict[str, Dict[str, bool]]:
        """
        List available keys (without revealing the actual keys).
        
        Returns:
            Dictionary of provider -> source availability mappings
        """
        result = {}
        
        for provider in self.SUPPORTED_PROVIDERS:
            sources = {}
            
            # Check environment variable
            env_var = f"HOMEOSTASIS_{provider.upper()}_API_KEY"
            sources['environment'] = bool(os.getenv(env_var))
            
            # Check external secrets managers
            sources['external_secrets'] = False
            if self._use_external_secrets:
                try:
                    external_key = self._get_key_from_external_secrets(provider)
                    sources['external_secrets'] = bool(external_key)
                except Exception:
                    pass
            
            # Check encrypted storage
            sources['encrypted_storage'] = False
            try:
                keys = self._load_keys()
                sources['encrypted_storage'] = provider in keys
            except Exception:
                pass
            
            result[provider] = sources
        
        return result
    
    def remove_key(self, provider: str) -> bool:
        """
        Remove API key for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            True if key was removed, False if not found
        """
        provider = provider.lower()
        
        try:
            keys = self._load_keys()
            if provider in keys:
                del keys[provider]
                self._save_keys(keys)
                
                # Remove from memory cache
                if provider in self._keys:
                    del self._keys[provider]
                
                print(f"✓ API key for {provider} removed successfully")
                return True
            else:
                print(f"No stored key found for {provider}")
                return False
        except Exception as e:
            print(f"Failed to remove key for {provider}: {e}")
            return False
    
    def validate_key(self, provider: str, api_key: str) -> bool:
        """
        Validate an API key by making a lightweight request to the provider.
        
        Args:
            provider: Provider name
            api_key: API key to validate
            
        Returns:
            True if valid
            
        Raises:
            KeyValidationError: If validation fails
        """
        provider = provider.lower()
        
        try:
            if provider == 'openai':
                return self._validate_openai_key(api_key)
            elif provider == 'anthropic':
                return self._validate_anthropic_key(api_key)
            elif provider == 'openrouter':
                return self._validate_openrouter_key(api_key)
            else:
                raise KeyValidationError(f"Validation not implemented for provider: {provider}")
        except requests.RequestException as e:
            raise KeyValidationError(f"Network error during validation: {e}")
        except Exception as e:
            raise KeyValidationError(f"Validation failed: {e}")
    
    def _validate_openai_key(self, api_key: str) -> bool:
        """Validate OpenAI API key."""
        # Check key format first
        if not api_key.startswith('sk-'):
            raise KeyValidationError(
                "Invalid OpenAI API key format. "
                "OpenAI keys should start with 'sk-'. "
                "Please check your key and try again."
            )
        
        if len(api_key) < 45:
            raise KeyValidationError(
                "OpenAI API key appears too short. "
                "Valid keys are typically 51+ characters. "
                "Please verify you copied the complete key."
            )
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Use the models endpoint for validation
        response = requests.get(
            'https://api.openai.com/v1/models',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            return True
        elif response.status_code == 401:
            raise KeyValidationError(
                "Invalid OpenAI API key. "
                "Please check: 1) Key is correct and complete, "
                "2) Key hasn't expired, 3) Account has sufficient credits. "
                "Get a new key from https://platform.openai.com/api-keys"
            )
        elif response.status_code == 429:
            raise KeyValidationError(
                "OpenAI API rate limit exceeded during validation. "
                "Please wait a moment and try again."
            )
        else:
            raise KeyValidationError(f"OpenAI API validation failed with status {response.status_code}. "
                                   f"Please check your internet connection and try again.")
    
    def _validate_anthropic_key(self, api_key: str) -> bool:
        """Validate Anthropic API key."""
        # Check key format first
        if not api_key.startswith('sk-ant-'):
            raise KeyValidationError(
                "Invalid Anthropic API key format. "
                "Anthropic keys should start with 'sk-ant-'. "
                "Please check your key and try again."
            )
        
        if len(api_key) < 60:
            raise KeyValidationError(
                "Anthropic API key appears too short. "
                "Valid keys are typically 90+ characters. "
                "Please verify you copied the complete key."
            )
        
        headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        # Use a minimal completion request for validation
        data = {
            'model': 'claude-3-haiku-20240307',
            'max_tokens': 1,
            'messages': [{'role': 'user', 'content': 'Hi'}]
        }
        
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            return True
        elif response.status_code == 401:
            raise KeyValidationError(
                "Invalid Anthropic API key. "
                "Please check: 1) Key is correct and complete, "
                "2) Key hasn't been revoked, 3) Account has API access. "
                "Get a new key from https://console.anthropic.com/"
            )
        elif response.status_code == 429:
            raise KeyValidationError(
                "Anthropic API rate limit exceeded during validation. "
                "Please wait a moment and try again."
            )
        elif response.status_code == 400:
            try:
                error_data = response.json()
                if 'error' in error_data and 'type' in error_data['error']:
                    error_type = error_data['error']['type']
                    if error_type == 'invalid_request_error':
                        raise KeyValidationError(
                            "Anthropic API request error. "
                            "Your key may be valid but have restricted permissions."
                        )
                raise KeyValidationError(f"Anthropic API error: {error_data}")
            except (ValueError, KeyError):
                raise KeyValidationError("Anthropic API returned invalid response during validation.")
        else:
            raise KeyValidationError(f"Anthropic API validation failed with status {response.status_code}. "
                                   f"Please check your internet connection and try again.")
    
    def _validate_openrouter_key(self, api_key: str) -> bool:
        """Validate OpenRouter API key."""
        # Check key format first
        if not api_key.startswith('sk-or-'):
            raise KeyValidationError(
                "Invalid OpenRouter API key format. "
                "OpenRouter keys should start with 'sk-or-'. "
                "Please check your key and try again."
            )
        
        if len(api_key) < 40:
            raise KeyValidationError(
                "OpenRouter API key appears too short. "
                "Valid keys are typically 60+ characters. "
                "Please verify you copied the complete key."
            )
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Use the models endpoint for validation
        response = requests.get(
            'https://openrouter.ai/api/v1/models',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            return True
        elif response.status_code == 401:
            raise KeyValidationError(
                "Invalid OpenRouter API key. "
                "Please check: 1) Key is correct and complete, "
                "2) Key hasn't been revoked, 3) Account has sufficient credits. "
                "Get a new key from https://openrouter.ai/keys"
            )
        elif response.status_code == 429:
            raise KeyValidationError(
                "OpenRouter API rate limit exceeded during validation. "
                "Please wait a moment and try again."
            )
        else:
            raise KeyValidationError(f"OpenRouter API validation failed with status {response.status_code}. "
                                   f"Please check your internet connection and try again.")
    
    def get_masked_key(self, provider: str) -> Optional[str]:
        """
        Get a masked version of the API key for display purposes.
        
        Args:
            provider: Provider name
            
        Returns:
            Masked key (e.g., "sk-...xyz") or None if not found
        """
        key = self.get_key(provider)
        if not key:
            return None
        
        if len(key) <= 8:
            return "*" * len(key)
        
        return f"{key[:3]}...{key[-4:]}"
    
    def detect_key_issues(self, provider: str, api_key: str) -> List[str]:
        """
        Detect potential issues with an API key format.
        
        Args:
            provider: Provider name
            api_key: API key to analyze
            
        Returns:
            List of detected issues
        """
        issues = []
        provider = provider.lower()
        
        # Common issues for all providers
        if not api_key.strip():
            issues.append("API key is empty")
            return issues
        
        if api_key != api_key.strip():
            issues.append("API key has leading/trailing whitespace")
        
        if '\n' in api_key or '\r' in api_key:
            issues.append("API key contains line breaks")
        
        if ' ' in api_key:
            issues.append("API key contains spaces (may be incomplete)")
        
        # Provider-specific checks
        if provider == 'openai':
            if not api_key.startswith('sk-'):
                issues.append("OpenAI keys should start with 'sk-'")
            if len(api_key) < 45:
                issues.append("OpenAI key appears too short (should be 51+ characters)")
            elif len(api_key) > 60:
                issues.append("OpenAI key appears too long (typically ~51 characters)")
                
        elif provider == 'anthropic':
            if not api_key.startswith('sk-ant-'):
                issues.append("Anthropic keys should start with 'sk-ant-'")
            if len(api_key) < 60:
                issues.append("Anthropic key appears too short (should be 90+ characters)")
            elif len(api_key) > 120:
                issues.append("Anthropic key appears too long (typically ~100 characters)")
                
        elif provider == 'openrouter':
            if not api_key.startswith('sk-or-'):
                issues.append("OpenRouter keys should start with 'sk-or-'")
            if len(api_key) < 40:
                issues.append("OpenRouter key appears too short (should be 60+ characters)")
            elif len(api_key) > 80:
                issues.append("OpenRouter key appears too long (typically ~64 characters)")
        
        return issues
    
    def suggest_key_correction(self, provider: str, api_key: str) -> Optional[str]:
        """
        Suggest corrections for common API key issues.
        
        Args:
            provider: Provider name
            api_key: API key to analyze
            
        Returns:
            Corrected key if possible, None otherwise
        """
        if not api_key:
            return None
        
        # Clean up common issues
        corrected = api_key.strip()
        corrected = corrected.replace('\n', '').replace('\r', '')
        corrected = corrected.replace(' ', '')
        
        # If we made changes, return the corrected version
        if corrected != api_key:
            return corrected
        
        return None