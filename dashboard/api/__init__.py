"""
API module for Homeostasis Dashboard.

This module provides communication between the dashboard and the Homeostasis system.
"""

from dashboard.api.client import HomeostasisClient
from dashboard.api.errors import (
    APIError,
    AuthenticationError,
    ConnectionError,
    NotFoundError,
)

__all__ = [
    "HomeostasisClient",
    "APIError",
    "ConnectionError",
    "NotFoundError",
    "AuthenticationError",
]
