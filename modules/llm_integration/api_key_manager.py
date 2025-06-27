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
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the API key manager.
        
        Args:
            config_dir: Optional custom configuration directory
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
        
        # Load from encrypted storage
        try:
            keys = self._load_keys()
            key = keys.get(provider)
            if key:
                self._keys[provider] = key
            return key
        except Exception:
            return None
    
    def list_keys(self) -> Dict[str, bool]:
        """
        List available keys (without revealing the actual keys).
        
        Returns:
            Dictionary of provider -> has_key mappings
        """
        result = {}
        
        for provider in self.SUPPORTED_PROVIDERS:
            # Check environment variable
            env_var = f"HOMEOSTASIS_{provider.upper()}_API_KEY"
            has_env_key = bool(os.getenv(env_var))
            
            # Check encrypted storage
            has_stored_key = False
            try:
                keys = self._load_keys()
                has_stored_key = provider in keys
            except Exception:
                pass
            
            result[provider] = has_env_key or has_stored_key
        
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
            raise KeyValidationError("Invalid OpenAI API key")
        else:
            raise KeyValidationError(f"OpenAI API validation failed: {response.status_code}")
    
    def _validate_anthropic_key(self, api_key: str) -> bool:
        """Validate Anthropic API key."""
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
            raise KeyValidationError("Invalid Anthropic API key")
        else:
            raise KeyValidationError(f"Anthropic API validation failed: {response.status_code}")
    
    def _validate_openrouter_key(self, api_key: str) -> bool:
        """Validate OpenRouter API key."""
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
            raise KeyValidationError("Invalid OpenRouter API key")
        else:
            raise KeyValidationError(f"OpenRouter API validation failed: {response.status_code}")
    
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