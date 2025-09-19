"""
Root conftest.py for all tests.

Sets up environment variables and global test configuration.
"""

import os
import sys
import warnings

# Ensure we're using mock tests by default
os.environ.setdefault("USE_MOCK_TESTS", "true")

# Disable performance tracking during tests to avoid threading issues
os.environ.setdefault("DISABLE_PERFORMANCE_TRACKING", "true")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*asyncio_default_fixture_loop_scope.*")

# Configure pytest-asyncio
import pytest

pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for asyncio tests."""
    import asyncio

    return asyncio.DefaultEventLoopPolicy()