"""
Encryption utilities for Homeostasis.

Provides encryption and decryption functionalities for secure data handling
in production environments.
"""

import base64
import binascii
import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    """Exception raised for encryption-related errors."""
    pass


class EncryptionManager:
    """Manages encryption operations for Homeostasis."""

    def __init__(self, config: Dict = None):
        """Initialize the encryption manager.
        
        Args:
            config: Configuration dictionary for encryption settings
        """
        self.config = config or {}
        
        # Initialize Fernet for symmetric encryption
        self.fernet_key = self._get_fernet_key()
        self.fernet = Fernet(self.fernet_key)
        
    def _get_fernet_key(self) -> bytes:
        """Get or generate a Fernet key for encryption.
        
        Returns:
            bytes: The Fernet key
        """
        # Try to get from config
        if self.config.get('encryption_key'):
            key = self.config['encryption_key']
            if isinstance(key, str):
                key = key.encode('utf-8')
            
            # Ensure the key is valid for Fernet (32 url-safe base64-encoded bytes)
            return self._prepare_key(key)
            
        # Try to get from environment
        env_key = os.environ.get('HOMEOSTASIS_ENCRYPTION_KEY')
        if env_key:
            key = env_key.encode('utf-8')
            return self._prepare_key(key)
            
        # Generate a random key if not configured
        # In production, this should be persisted and shared across instances
        logger.warning("No encryption key configured. Generating random key. "
                      "This is insecure for production environments.")
        return Fernet.generate_key()
    
    def _prepare_key(self, key: bytes) -> bytes:
        """Prepare a key for use with Fernet.
        
        Args:
            key: The input key
            
        Returns:
            bytes: A valid Fernet key
        """
        # If key is already a valid Fernet key (url-safe base64-encoded 32 bytes)
        try:
            if len(base64.urlsafe_b64decode(key + b'=' * (-len(key) % 4))) == 32:
                return key
        except binascii.Error:
            pass
        
        # If key is not a valid Fernet key, derive one using PBKDF2
        salt = self.config.get('key_salt', b'homeostasis_salt')
        if isinstance(salt, str):
            salt = salt.encode('utf-8')
            
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        derived_key = kdf.derive(key)
        return base64.urlsafe_b64encode(derived_key)
        
    def encrypt(self, data: Union[str, bytes, Dict]) -> str:
        """Encrypt data using Fernet (symmetric encryption).
        
        Args:
            data: Data to encrypt (string, bytes, or JSON-serializable dict)
            
        Returns:
            str: Base64-encoded encrypted data
            
        Raises:
            EncryptionError: If encryption fails
        """
        try:
            # Convert data to bytes
            if isinstance(data, dict):
                data = json.dumps(data)
            if isinstance(data, str):
                data = data.encode('utf-8')
                
            # Encrypt
            encrypted = self.fernet.encrypt(data)
            return encrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise EncryptionError(f"Encryption failed: {str(e)}")
        
    def decrypt(self, encrypted_data: Union[str, bytes]) -> bytes:
        """Decrypt Fernet-encrypted data.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            bytes: Decrypted data
            
        Raises:
            EncryptionError: If decryption fails
        """
        try:
            # Convert to bytes if needed
            if isinstance(encrypted_data, str):
                encrypted_data = encrypted_data.encode('utf-8')
                
            # Decrypt
            return self.fernet.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise EncryptionError(f"Decryption failed: {str(e)}")
        
    def decrypt_to_string(self, encrypted_data: Union[str, bytes]) -> str:
        """Decrypt to string.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            str: Decrypted string
            
        Raises:
            EncryptionError: If decryption fails
        """
        decrypted = self.decrypt(encrypted_data)
        try:
            return decrypted.decode('utf-8')
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode decrypted bytes to string: {str(e)}")
            raise EncryptionError(f"Failed to decode decrypted bytes to string: {str(e)}")
        
    def decrypt_to_json(self, encrypted_data: Union[str, bytes]) -> Dict:
        """Decrypt to JSON.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Dict: Decrypted JSON
            
        Raises:
            EncryptionError: If decryption or JSON parsing fails
        """
        decrypted_str = self.decrypt_to_string(encrypted_data)
        try:
            return json.loads(decrypted_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse decrypted string as JSON: {str(e)}")
            raise EncryptionError(f"Failed to parse decrypted string as JSON: {str(e)}")
        
    def hash_password(self, password: str) -> Tuple[bytes, bytes]:
        """Hash a password with a random salt.
        
        Args:
            password: Password to hash
            
        Returns:
            Tuple[bytes, bytes]: (password_hash, salt)
        """
        salt = os.urandom(16)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256', 
            password.encode('utf-8'), 
            salt, 
            100000
        )
        return password_hash, salt
        
    def verify_password(self, password: str, password_hash: bytes, salt: bytes) -> bool:
        """Verify a password against its hash.
        
        Args:
            password: Password to verify
            password_hash: Stored password hash
            salt: Salt used for hashing
            
        Returns:
            bool: True if password matches, False otherwise
        """
        computed_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        return hmac.compare_digest(computed_hash, password_hash)
        
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a secure random token.
        
        Args:
            length: Length of the token in bytes
            
        Returns:
            str: Base64-encoded secure token
        """
        token = os.urandom(length)
        return base64.urlsafe_b64encode(token).decode('utf-8').rstrip('=')


# Singleton instance for app-wide use
_encryption_manager = None

def get_encryption_manager(config: Dict = None) -> EncryptionManager:
    """Get or create the singleton EncryptionManager instance.
    
    Args:
        config: Optional configuration to initialize the manager with
        
    Returns:
        EncryptionManager: The encryption manager instance
    """
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager(config)
    return _encryption_manager

def encrypt(data: Union[str, bytes, Dict]) -> str:
    """Encrypt data.
    
    Args:
        data: Data to encrypt
        
    Returns:
        str: Encrypted data as a string
        
    Raises:
        EncryptionError: If encryption fails
    """
    return get_encryption_manager().encrypt(data)

def decrypt(encrypted_data: Union[str, bytes]) -> bytes:
    """Decrypt data.
    
    Args:
        encrypted_data: Encrypted data
        
    Returns:
        bytes: Decrypted data
        
    Raises:
        EncryptionError: If decryption fails
    """
    return get_encryption_manager().decrypt(encrypted_data)

def decrypt_to_string(encrypted_data: Union[str, bytes]) -> str:
    """Decrypt to string.
    
    Args:
        encrypted_data: Encrypted data
        
    Returns:
        str: Decrypted string
        
    Raises:
        EncryptionError: If decryption fails
    """
    return get_encryption_manager().decrypt_to_string(encrypted_data)

def decrypt_to_json(encrypted_data: Union[str, bytes]) -> Dict:
    """Decrypt to JSON.
    
    Args:
        encrypted_data: Encrypted data
        
    Returns:
        Dict: Decrypted JSON
        
    Raises:
        EncryptionError: If decryption or JSON parsing fails
    """
    return get_encryption_manager().decrypt_to_json(encrypted_data)