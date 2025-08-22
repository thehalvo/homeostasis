"""
Authentication system for Homeostasis.

Provides authentication mechanisms for securing access to Homeostasis components
and functionality in production environments.
"""

import base64
import datetime
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import jwt
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_TOKEN_EXPIRY = 3600  # 1 hour
DEFAULT_REFRESH_TOKEN_EXPIRY = 86400 * 7  # 7 days


class AuthenticationError(Exception):
    """Exception raised for authentication errors."""
    pass


class AuthenticationManager:
    """Manages authentication for Homeostasis."""

    def __init__(self, config: Dict = None):
        """Initialize the authentication manager.
        
        Args:
            config: Configuration dictionary for auth settings
        """
        self.config = config or {}
        
        # Generate or load secret key from config or environment
        self.secret_key = self._get_secret_key()
        
        # Initialize Fernet key for encryption
        self.fernet_key = self._get_fernet_key()
        self.fernet = Fernet(self.fernet_key)
        
        # User store - in production this would be a database
        # For this implementation, we use an in-memory store as a placeholder
        self.users = {}
        
        # Token blacklist for revoked tokens
        self.token_blacklist = set()

    def _get_secret_key(self) -> bytes:
        """Get or generate a secret key for JWT signing.
        
        Returns:
            bytes: The secret key
        """
        # Try to get from config
        if self.config.get('jwt_secret'):
            secret = self.config['jwt_secret']
            if isinstance(secret, str):
                return secret.encode('utf-8')
            return secret
            
        # Try to get from environment
        env_secret = os.environ.get('HOMEOSTASIS_JWT_SECRET')
        if env_secret:
            return env_secret.encode('utf-8')
            
        # Generate a random key if not configured
        # In production, this should be persisted and shared across instances
        logger.warning("No JWT secret configured. Generating random secret. "
                      "This is insecure for production environments.")
        return os.urandom(32)

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
            if len(key) != 32:
                key = hashlib.sha256(key).digest()
            return base64.urlsafe_b64encode(key)
            
        # Try to get from environment
        env_key = os.environ.get('HOMEOSTASIS_ENCRYPTION_KEY')
        if env_key:
            key = env_key.encode('utf-8')
            if len(key) != 32:
                key = hashlib.sha256(key).digest()
            return base64.urlsafe_b64encode(key)
            
        # Generate a random key if not configured
        # In production, this should be persisted and shared across instances
        logger.warning("No encryption key configured. Generating random key. "
                      "This is insecure for production environments.")
        return Fernet.generate_key()

    def register_user(self, username: str, password: str, roles: List[str] = None) -> bool:
        """Register a new user.
        
        Args:
            username: Username for the new user
            password: Password for the new user
            roles: List of roles to assign to the user
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        if username in self.users:
            return False
            
        # In production, use a secure password hashing library like bcrypt or Argon2
        # This is a simplified example
        salt = os.urandom(16)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256', 
            password.encode('utf-8'), 
            salt, 
            100000
        )
        
        self.users[username] = {
            'password_hash': password_hash,
            'salt': salt,
            'roles': roles or ['user'],
            'created_at': datetime.datetime.utcnow().isoformat()
        }
        
        return True

    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate a user with username and password.
        
        Args:
            username: Username of the user
            password: Password to verify
            
        Returns:
            Optional[Dict]: User info if authentication successful, None otherwise
        """
        if username not in self.users:
            return None
            
        user = self.users[username]
        salt = user['salt']
        stored_hash = user['password_hash']
        
        # Calculate hash of provided password
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        
        if not hmac.compare_digest(password_hash, stored_hash):
            return None
            
        return {
            'username': username,
            'roles': user['roles']
        }

    def generate_token(self, user_info: Dict) -> Tuple[str, str]:
        """Generate JWT token and refresh token for a user.
        
        Args:
            user_info: User information to encode in the token
            
        Returns:
            Tuple[str, str]: (access_token, refresh_token)
        """
        now = int(time.time())
        
        # Access token payload
        payload = {
            'sub': user_info['username'],
            'roles': user_info['roles'],
            'iat': now,
            'exp': now + self.config.get('token_expiry', DEFAULT_TOKEN_EXPIRY),
            'type': 'access'
        }
        
        # Generate access token
        access_token = jwt.encode(
            payload,
            self.secret_key,
            algorithm='HS256'
        )
        
        # Refresh token payload
        refresh_payload = {
            'sub': user_info['username'],
            'iat': now,
            'exp': now + self.config.get('refresh_token_expiry', DEFAULT_REFRESH_TOKEN_EXPIRY),
            'type': 'refresh'
        }
        
        # Generate refresh token
        refresh_token = jwt.encode(
            refresh_payload,
            self.secret_key,
            algorithm='HS256'
        )
        
        return access_token, refresh_token

    def verify_token(self, token: str) -> Dict:
        """Verify a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Dict: Token payload if valid
            
        Raises:
            AuthenticationError: If token is invalid or expired
        """
        if token in self.token_blacklist:
            raise AuthenticationError("Token has been revoked")
            
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=['HS256']
            )
            
            # Check token type
            if payload.get('type') != 'access':
                raise AuthenticationError("Invalid token type")
                
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")

    def refresh_access_token(self, refresh_token: str) -> str:
        """Generate new access token using refresh token.
        
        Args:
            refresh_token: Refresh token to use
            
        Returns:
            str: New access token
            
        Raises:
            AuthenticationError: If refresh token is invalid
        """
        try:
            payload = jwt.decode(
                refresh_token,
                self.secret_key,
                algorithms=['HS256']
            )
            
            # Check token type
            if payload.get('type') != 'refresh':
                raise AuthenticationError("Invalid token type")
                
            # Get user info
            username = payload['sub']
            if username not in self.users:
                raise AuthenticationError("User not found")
                
            user_info = {
                'username': username,
                'roles': self.users[username]['roles']
            }
            
            # Generate new access token
            access_token, _ = self.generate_token(user_info)
            return access_token
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Refresh token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid refresh token: {str(e)}")

    def revoke_token(self, token: str) -> None:
        """Revoke a token by adding it to the blacklist.
        
        Args:
            token: Token to revoke
        """
        self.token_blacklist.add(token)

    def encrypt_data(self, data: Union[str, bytes, Dict]) -> str:
        """Encrypt sensitive data.
        
        Args:
            data: Data to encrypt (string, bytes, or JSON-serializable dict)
            
        Returns:
            str: Encrypted data as a string
        """
        if isinstance(data, dict):
            data = json.dumps(data)
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        encrypted = self.fernet.encrypt(data)
        return encrypted.decode('utf-8')

    def decrypt_data(self, encrypted_data: str) -> bytes:
        """Decrypt encrypted data.
        
        Args:
            encrypted_data: Encrypted data as a string
            
        Returns:
            bytes: Decrypted data
            
        Raises:
            Exception: If decryption fails
        """
        if isinstance(encrypted_data, str):
            encrypted_data = encrypted_data.encode('utf-8')
            
        return self.fernet.decrypt(encrypted_data)

    def decrypt_json(self, encrypted_data: str) -> Dict:
        """Decrypt encrypted JSON data.
        
        Args:
            encrypted_data: Encrypted JSON data as a string
            
        Returns:
            Dict: Decrypted JSON data as a dictionary
            
        Raises:
            Exception: If decryption or JSON parsing fails
        """
        decrypted = self.decrypt_data(encrypted_data)
        return json.loads(decrypted)


# Singleton instance for app-wide use
_auth_manager = None

def get_auth_manager(config: Dict = None) -> AuthenticationManager:
    """Get or create the singleton AuthenticationManager instance.
    
    Args:
        config: Optional configuration to initialize the manager with
        
    Returns:
        AuthenticationManager: The authentication manager instance
    """
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager(config)
    return _auth_manager

def authenticate(username: str, password: str) -> Optional[Dict]:
    """Authenticate a user.
    
    Args:
        username: Username to authenticate
        password: Password to verify
        
    Returns:
        Optional[Dict]: User info if authentication successful, None otherwise
    """
    return get_auth_manager().authenticate(username, password)

def generate_token(user_info: Dict) -> Tuple[str, str]:
    """Generate JWT access and refresh tokens.
    
    Args:
        user_info: User information to encode in the token
        
    Returns:
        Tuple[str, str]: (access_token, refresh_token)
    """
    return get_auth_manager().generate_token(user_info)

def verify_token(token: str) -> Dict:
    """Verify a JWT token.
    
    Args:
        token: JWT token to verify
        
    Returns:
        Dict: Token payload if valid
        
    Raises:
        AuthenticationError: If token is invalid
    """
    return get_auth_manager().verify_token(token)