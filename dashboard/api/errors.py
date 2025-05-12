"""
API Error classes for Homeostasis Dashboard.
"""

class APIError(Exception):
    """Base class for API errors."""
    pass

class ConnectionError(APIError):
    """Error when connecting to API."""
    pass

class NotFoundError(APIError):
    """Error when resource is not found."""
    pass

class AuthenticationError(APIError):
    """Error during authentication."""
    pass