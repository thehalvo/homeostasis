"""
Pytest fixtures for end-to-end healing scenario tests.

Provides shared fixtures for test environment setup, scenario runners,
and metrics collection.
"""
import pytest
import tempfile
from pathlib import Path

from tests.e2e.healing_scenarios.test_utilities import (
    TestEnvironment,
    HealingScenarioRunner,
    MetricsCollector
)


@pytest.fixture
def test_environment():
    """Create and manage an isolated test environment."""
    env = TestEnvironment()
    env.setup()
    yield env
    # Cleanup
    env.cleanup()


@pytest.fixture
def scenario_runner(test_environment):
    """Create a healing scenario runner."""
    return HealingScenarioRunner(test_environment)


@pytest.fixture
def metrics_collector():
    """Create a metrics collector for test instrumentation."""
    return MetricsCollector()