"""
Chaos Engineering Tests for Homeostasis

This package contains comprehensive chaos engineering tests to validate
system resilience and self-healing capabilities.

Test Categories:
- Network Chaos: Latency, packet loss, partitions
- Resource Chaos: CPU, memory, disk pressure
- Service Chaos: Failures, cascades, timeouts

Usage:
    # Run all chaos tests
    pytest tests/chaos/

    # Run specific intensity
    pytest tests/chaos/ -m chaos_low
    pytest tests/chaos/ -m chaos_medium
    pytest tests/chaos/ -m chaos_high

    # Run specific category
    pytest tests/chaos/test_network_chaos.py
    pytest tests/chaos/test_resource_chaos.py
    pytest tests/chaos/test_service_chaos.py

    # Run with chaos runner
    python tests/chaos/chaos_runner.py --intensity medium --duration 30
"""

import pytest

# Pytest markers for chaos test intensity
pytest.mark.chaos_low = pytest.mark.chaos_low
pytest.mark.chaos_medium = pytest.mark.chaos_medium
pytest.mark.chaos_high = pytest.mark.chaos_high

# Pytest markers for chaos test categories
pytest.mark.chaos_network = pytest.mark.chaos_network
pytest.mark.chaos_resource = pytest.mark.chaos_resource
pytest.mark.chaos_service = pytest.mark.chaos_service
