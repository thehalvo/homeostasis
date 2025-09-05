"""
API Security module for Homeostasis.

Provides API security features including rate limiting, token validation,
and protection against common API attacks.
"""

import collections
import hashlib
import hmac
import logging
import re
import time
from functools import wraps
from typing import Dict, Optional, Tuple

from modules.security.auth import AuthenticationError, get_auth_manager
from modules.security.rbac import PermissionDeniedError, require_permission

logger = logging.getLogger(__name__)


class RateLimitExceededError(Exception):
    """Exception raised when a rate limit is exceeded."""
    pass


class RateLimiter:
    """Rate limiter for API endpoints."""

    def __init__(self, config: Dict = None):
        """Initialize the rate limiter.
        
        Args:
            config: Configuration dictionary for rate limiter settings
        """
        self.config = config or {}
        
        # Default rate limits
        self.default_rate_limits = {
            'global': (100, 60),  # 100 requests per 60 seconds
            'user': (20, 60),     # 20 requests per 60 seconds per user
            'ip': (50, 60),       # 50 requests per 60 seconds per IP
        }
        
        # Override defaults with config values if provided
        if 'rate_limits' in self.config:
            self.default_rate_limits.update(self.config['rate_limits'])
        
        # Initialize storage for request counts
        # In a production environment, this should use Redis or a similar distributed store
        self.request_counts = {
            'global': collections.defaultdict(int),
            'user': collections.defaultdict(lambda: collections.defaultdict(int)),
            'ip': collections.defaultdict(lambda: collections.defaultdict(int)),
        }
        
        # Store time windows
        self.time_windows = {
            'global': collections.defaultdict(float),
            'user': collections.defaultdict(lambda: collections.defaultdict(float)),
            'ip': collections.defaultdict(lambda: collections.defaultdict(float)),
        }
        
    def check_rate_limit(self, scope: str, identifier: str = None, 
                         endpoint: str = None) -> bool:
        """Check if a request is within rate limits.
        
        Args:
            scope: The scope of the rate limit ('global', 'user', or 'ip')
            identifier: The identifier within the scope (username or IP)
            endpoint: The specific endpoint being accessed
            
        Returns:
            bool: True if within rate limits, False if exceeded
        """
        now = time.time()
        
        # Get rate limit for this scope and endpoint
        limit, window = self._get_rate_limit(scope, endpoint)
        
        # Update counts based on scope
        if scope == 'global':
            return self._check_global_limit(endpoint, limit, window, now)
        else:
            return self._check_scoped_limit(scope, identifier, endpoint, limit, window, now)
    
    def _get_rate_limit(self, scope: str, endpoint: str = None) -> Tuple[int, int]:
        """Get the rate limit for a specific scope and endpoint.
        
        Args:
            scope: The scope of the rate limit
            endpoint: The specific endpoint
            
        Returns:
            Tuple[int, int]: (requests_limit, time_window_seconds)
        """
        # Check for endpoint-specific limit
        if endpoint and 'endpoint_limits' in self.config:
            endpoint_limits = self.config['endpoint_limits']
            if endpoint in endpoint_limits and scope in endpoint_limits[endpoint]:
                return endpoint_limits[endpoint][scope]
        
        # Fall back to default for this scope
        return self.default_rate_limits.get(scope, (100, 60))
    
    def _check_global_limit(self, endpoint: str, limit: int, window: int, now: float) -> bool:
        """Check global rate limit.
        
        Args:
            endpoint: The endpoint being accessed
            limit: Number of requests allowed
            window: Time window in seconds
            now: Current timestamp
            
        Returns:
            bool: True if within rate limits, False if exceeded
        """
        key = endpoint or 'global'
        
        # If window has expired, reset counter
        if now - self.time_windows['global'][key] > window:
            self.request_counts['global'][key] = 1
            self.time_windows['global'][key] = now
            return True
        
        # Increment counter
        self.request_counts['global'][key] += 1
        
        # Check if limit exceeded
        return self.request_counts['global'][key] <= limit
    
    def _check_scoped_limit(self, scope: str, identifier: str, endpoint: str, 
                            limit: int, window: int, now: float) -> bool:
        """Check rate limit for a specific scope and identifier.
        
        Args:
            scope: The scope of the rate limit
            identifier: The identifier within the scope
            endpoint: The endpoint being accessed
            limit: Number of requests allowed
            window: Time window in seconds
            now: Current timestamp
            
        Returns:
            bool: True if within rate limits, False if exceeded
        """
        key = endpoint or 'all'
        
        # If window has expired, reset counter
        if now - self.time_windows[scope][identifier][key] > window:
            self.request_counts[scope][identifier][key] = 1
            self.time_windows[scope][identifier][key] = now
            return True
        
        # Increment counter
        self.request_counts[scope][identifier][key] += 1
        
        # Check if limit exceeded
        return self.request_counts[scope][identifier][key] <= limit
        
    def reset_counters(self):
        """Reset all rate limit counters."""
        self.request_counts = {
            'global': collections.defaultdict(int),
            'user': collections.defaultdict(lambda: collections.defaultdict(int)),
            'ip': collections.defaultdict(lambda: collections.defaultdict(int)),
        }
        self.time_windows = {
            'global': collections.defaultdict(float),
            'user': collections.defaultdict(lambda: collections.defaultdict(float)),
            'ip': collections.defaultdict(lambda: collections.defaultdict(float)),
        }


class APISecurityManager:
    """Manages API security for Homeostasis."""

    def __init__(self, config: Dict = None):
        """Initialize the API security manager.
        
        Args:
            config: Configuration dictionary for API security settings
        """
        self.config = config or {}
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(config)
        
        # Initialize token validator
        self.auth_manager = get_auth_manager(config)
        
        # Initialize blocked IPs set
        self.blocked_ips = set(self.config.get('blocked_ips', []))
        
        # Initialize suspicious patterns
        self.suspicious_patterns = self.config.get('suspicious_patterns', [
            r'(?i)(?:\'|\%27)(?:\s|\+|\/\*.*\*\/)*(?:(?:select|union|delete|insert|drop|update))',  # SQL injection
            r'(?i)<script[^>]*>.*?</script>',  # XSS
            r'(?i)javascript\s*:',  # XSS through javascript: URLs
            r'(?:\.\.\/)+',  # Path traversal
            r'(?:/etc/passwd)',  # Sensitive file access attempt
        ])
        self.compiled_patterns = [re.compile(p) for p in self.suspicious_patterns]
        
    def check_ip(self, ip_address: str) -> bool:
        """Check if an IP address is blocked.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            bool: True if IP is allowed, False if blocked
        """
        return ip_address not in self.blocked_ips
        
    def block_ip(self, ip_address: str) -> None:
        """Block an IP address.
        
        Args:
            ip_address: IP address to block
        """
        self.blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP address: {ip_address}")
        
    def unblock_ip(self, ip_address: str) -> bool:
        """Unblock an IP address.
        
        Args:
            ip_address: IP address to unblock
            
        Returns:
            bool: True if IP was unblocked, False if it wasn't blocked
        """
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            logger.info(f"Unblocked IP address: {ip_address}")
            return True
        return False
        
    def check_request_content(self, content: str) -> Tuple[bool, Optional[str]]:
        """Check request content for suspicious patterns.
        
        Args:
            content: Request content to check
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, reason_if_invalid)
        """
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(content):
                return False, f"Suspicious pattern detected: {self.suspicious_patterns[i]}"
        return True, None
        
    def verify_token(self, token: str) -> Dict:
        """Verify a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Dict: Token payload if valid
            
        Raises:
            AuthenticationError: If token is invalid
        """
        return self.auth_manager.verify_token(token)
        
    def check_rate_limit(self, scope: str, identifier: str = None, 
                         endpoint: str = None) -> bool:
        """Check if a request is within rate limits.
        
        Args:
            scope: The scope of the rate limit ('global', 'user', or 'ip')
            identifier: The identifier within the scope (username or IP)
            endpoint: The specific endpoint being accessed
            
        Returns:
            bool: True if within rate limits, False if exceeded
        """
        return self.rate_limiter.check_rate_limit(scope, identifier, endpoint)
        
    def create_secure_headers(self) -> Dict[str, str]:
        """Create secure HTTP headers for API responses.
        
        Returns:
            Dict[str, str]: Dictionary of secure HTTP headers
        """
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'Content-Security-Policy': "default-src 'self'",
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Referrer-Policy': 'no-referrer'
        }
    
    def generate_csrf_token(self, user_id: str) -> str:
        """Generate a CSRF token for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            str: CSRF token
        """
        # In a production environment, this should use a more sophisticated approach
        # and store the token securely (e.g., in Redis)
        timestamp = str(int(time.time()))
        token_data = f"{user_id}:{timestamp}"
        
        # Generate token using HMAC
        token = hashlib.sha256((token_data + self.auth_manager.secret_key.decode('utf-8')).encode('utf-8')).hexdigest()
        return f"{token}:{timestamp}"
        
    def verify_csrf_token(self, token: str, user_id: str, max_age: int = 3600) -> bool:
        """Verify a CSRF token.
        
        Args:
            token: CSRF token to verify
            user_id: User identifier
            max_age: Maximum age of token in seconds
            
        Returns:
            bool: True if token is valid, False otherwise
        """
        try:
            hmac_token, timestamp = token.split(':', 1)
            timestamp = int(timestamp)
            
            # Check if token has expired
            if int(time.time()) - timestamp > max_age:
                return False
                
            # Recreate token for comparison
            token_data = f"{user_id}:{timestamp}"
            expected_token = hashlib.sha256((token_data + self.auth_manager.secret_key.decode('utf-8')).encode('utf-8')).hexdigest()
            
            # Compare tokens
            return hmac.compare_digest(hmac_token, expected_token)
        except (ValueError, TypeError):
            return False


# Singleton instance for app-wide use
_api_security_manager = None


def get_api_security_manager(config: Dict = None) -> APISecurityManager:
    """Get or create the singleton APISecurityManager instance.
    
    Args:
        config: Optional configuration to initialize the manager with
        
    Returns:
        APISecurityManager: The API security manager instance
    """
    global _api_security_manager
    if _api_security_manager is None:
        _api_security_manager = APISecurityManager(config)
    return _api_security_manager


def secure_endpoint(permission: str = None, rate_limit: bool = True):
    """Decorator for securing API endpoints.
    
    Args:
        permission: Permission required for this endpoint
        rate_limit: Whether to apply rate limiting
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get request context (specific implementation depends on the web framework)
            request = kwargs.get('request')
            if request is None:
                # Try to find request in positional args
                for arg in args:
                    if hasattr(arg, 'headers') and hasattr(arg, 'remote_addr'):
                        request = arg
                        break
            
            if request is None:
                raise ValueError("Could not find request object")
                
            # Get security manager
            security_manager = get_api_security_manager()
            
            # Check IP address
            ip_address = getattr(request, 'remote_addr', 'unknown')
            if not security_manager.check_ip(ip_address):
                logger.warning(f"Request from blocked IP: {ip_address}")
                return {'error': 'Access denied'}, 403
                
            # Apply rate limiting if enabled
            if rate_limit:
                # Check global rate limit
                if not security_manager.check_rate_limit('global'):
                    logger.warning("Global rate limit exceeded")
                    raise RateLimitExceededError("Rate limit exceeded")
                    
                # Check IP-based rate limit
                if not security_manager.check_rate_limit('ip', ip_address):
                    logger.warning(f"IP rate limit exceeded for {ip_address}")
                    raise RateLimitExceededError("Rate limit exceeded")
                    
                # Check user-based rate limit if authenticated
                if hasattr(request, 'user') and request.user:
                    user_id = request.user.get('username', 'anonymous')
                    if not security_manager.check_rate_limit('user', user_id):
                        logger.warning(f"User rate limit exceeded for {user_id}")
                        raise RateLimitExceededError("Rate limit exceeded")
            
            # Check authentication and permissions if required
            if permission is not None:
                # Get token from request
                auth_header = getattr(request, 'headers', {}).get('Authorization', '')
                token = auth_header.replace('Bearer ', '') if auth_header.startswith('Bearer ') else None
                
                if not token:
                    raise AuthenticationError("Authentication required")
                    
                # Verify token
                try:
                    user_info = security_manager.verify_token(token)
                    
                    # Check permission
                    require_permission(user_info, permission)
                except (AuthenticationError, PermissionDeniedError) as e:
                    logger.warning(f"Access denied: {str(e)}")
                    return {'error': str(e)}, 403
            
            # Call the original function
            return func(*args, **kwargs)
            
        return wrapper
        
    return decorator