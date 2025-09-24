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
import pytest  # noqa: E402

pytest_plugins = ["pytest_asyncio"]




@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for asyncio tests."""
    import asyncio

    return asyncio.DefaultEventLoopPolicy()


# Simple test progress reporting for CI
def pytest_runtest_logstart(nodeid, location):
    """Called when a test starts."""
    if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
        print(f"\n[TEST START] {nodeid}", flush=True)


def pytest_runtest_logfinish(nodeid, location):
    """Called when a test finishes."""
    if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
        print(f"[TEST DONE] {nodeid}", flush=True)


@pytest.fixture(autouse=True)
def clear_analyzer_cache(request):
    """Clear analyzer singleton cache before each test to ensure isolation."""
    from modules.analysis.analyzer import Analyzer

    # Skip cache clearing for stress tests to avoid memory issues
    if "stress" in request.node.nodeid or "test_concurrent_high_volume" in request.node.nodeid:
        yield
        return

    # Clear before test
    Analyzer.clear_cache()
    yield
    # Clear after test
    Analyzer.clear_cache()


@pytest.fixture(autouse=True)
def limit_memory_usage():
    """Limit memory usage for tests to prevent OOM on GitHub Actions."""
    import gc

    # Force garbage collection before each test
    gc.collect()
    yield
    # Force garbage collection after each test
    gc.collect()


