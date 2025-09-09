"""
Advanced healing scenario tests using mock infrastructure.

Tests framework-specific, database, concurrent, and cascading error healing
scenarios using simulated services.
"""

import pytest
import yaml

from tests.e2e.healing_scenarios.test_infrastructure import (
    MockOrchestrator, MockServiceEnvironment, PatchValidator)


class TestAdvancedHealingScenariosMock:
    """Test advanced self-healing scenarios with mock infrastructure."""

    @pytest.fixture
    def mock_environment(self):
        """Create a mock service environment."""
        env = MockServiceEnvironment()
        yield env
        env.cleanup()

    @pytest.fixture
    def mock_orchestrator(self, mock_environment, tmp_path):
        """Create a mock orchestrator."""
        # Create minimal config
        config = {
            "general": {"project_name": "test", "environment": "test"},
            "monitoring": {"log_file": "test.log"},
            "analysis": {"rule_based": {"enabled": True}},
            "testing": {"enabled": True},
            "deployment": {"enabled": True, "backup_dir": "backups/"},
            "patch_generation": {
                "generated_patches_dir": "patches",
                "backup_original_files": True,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return MockOrchestrator(config_path, mock_environment)

    @pytest.mark.asyncio
    async def test_framework_specific_healing(
        self, mock_environment, mock_orchestrator
    ):
        """Test healing of framework-specific errors (FastAPI async/await)."""
        # Create FastAPI service
        service = mock_environment.create_service("fastapi_service")
        service.start()

        # Inject FastAPI-specific async error
        mock_environment.inject_error(
            service="fastapi_service",
            error_type="TypeError",
            message="object dict can't be used in 'await' expression",
            file_path="api/endpoints.py",
            line_number=45,
        )

        # Detect and analyze
        errors = mock_orchestrator.monitor_for_errors()
        assert len(errors) == 1
        assert "await" in errors[0]["message"]

        analysis_results = mock_orchestrator.analyze_errors(errors)
        assert analysis_results[0]["root_cause"] == "type_mismatch"

        # Generate framework-specific patch
        patches = mock_orchestrator.generate_patches(analysis_results)
        assert len(patches) == 1
        assert "await" not in patches[0]["new_code"]

        # Validate and deploy
        assert PatchValidator.validate_fix_addresses_error(patches[0], "TypeError")
        test_results = mock_orchestrator.test_patches(patches)
        assert all(test_results.values())
        assert service.health_status

    @pytest.mark.asyncio
    async def test_database_error_healing(self, mock_environment, mock_orchestrator):
        """Test healing of database-related errors."""
        # Create database service
        service = mock_environment.create_service("db_service")
        service.start()

        # Inject database error
        mock_environment.inject_error(
            service="db_service",
            error_type="NameError",
            message="name 'user_id' is not defined",
            file_path="db/queries.py",
            line_number=120,
        )

        # Add SQL context to the error log
        error_logs = mock_environment.get_error_logs()
        if error_logs:
            error_logs[-1][
                "stack_trace"
            ] = """Traceback (most recent call last):
  File "db/queries.py", line 120, in get_user
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
NameError: name 'user_id' is not defined"""

        # Healing process
        errors = mock_orchestrator.monitor_for_errors()
        analysis_results = mock_orchestrator.analyze_errors(errors)
        patches = mock_orchestrator.generate_patches(analysis_results)

        # Verify patch adds missing parameter
        assert "def" in patches[0]["new_code"]
        assert "user_id" in patches[0]["new_code"]

        test_results = mock_orchestrator.test_patches(patches)
        assert all(test_results.values())

    @pytest.mark.asyncio
    async def test_concurrent_healing_scenarios(
        self, mock_environment, mock_orchestrator
    ):
        """Test multiple healing scenarios running concurrently."""
        # Create multiple services with different errors
        services = []
        error_types = [
            ("api_service", "KeyError", "dict_key_not_exists"),
            ("auth_service", "AttributeError", "null_reference"),
            ("cache_service", "TypeError", "type_mismatch"),
        ]

        for service_name, error_type, _ in error_types:
            service = mock_environment.create_service(service_name)
            service.start()
            services.append(service)

            # Inject error
            mock_environment.inject_error(service_name, error_type)

        # Detect all errors concurrently
        errors = mock_orchestrator.monitor_for_errors()
        assert len(errors) == 3

        # Analyze all errors
        analysis_results = mock_orchestrator.analyze_errors(errors)
        assert len(analysis_results) == 3

        # Verify each error has correct root cause
        for i, (_, _, expected_root_cause) in enumerate(error_types):
            assert analysis_results[i]["root_cause"] == expected_root_cause

        # Generate patches for all
        patches = mock_orchestrator.generate_patches(analysis_results)
        assert len(patches) == 3

        # Deploy all patches
        test_results = mock_orchestrator.test_patches(patches)
        assert all(test_results.values())

        # Verify all services are healthy
        for service in services:
            assert service.health_status

    @pytest.mark.asyncio
    async def test_cascading_failure_healing(self, mock_environment, mock_orchestrator):
        """Test healing of cascading failures across multiple components."""
        # Create microservices architecture
        services = {
            "api_gateway": mock_environment.create_service("api_gateway"),
            "user_service": mock_environment.create_service("user_service"),
            "order_service": mock_environment.create_service("order_service"),
            "payment_service": mock_environment.create_service("payment_service"),
        }

        for service in services.values():
            service.start()

        # Simulate cascading failure
        # First: payment service fails
        mock_environment.inject_error(
            service="payment_service",
            error_type="ConnectionError",
            message="Failed to connect to payment processor",
            file_path="payment/processor.py",
            line_number=200,
        )

        # Then: order service fails due to payment failure
        mock_environment.inject_error(
            service="order_service",
            error_type="RuntimeError",
            message="Payment service unavailable",
            file_path="orders/checkout.py",
            line_number=150,
        )

        # Finally: API gateway reports errors
        mock_environment.inject_error(
            service="api_gateway",
            error_type="ServiceUnavailableError",
            message="Downstream services failing",
            file_path="gateway/router.py",
            line_number=50,
        )

        # Detect cascading errors
        errors = mock_orchestrator.monitor_for_errors()
        assert len(errors) >= 3

        # Analyze should identify cascade pattern
        analysis_results = mock_orchestrator.analyze_errors(errors)

        # Generate comprehensive fix
        patches = mock_orchestrator.generate_patches(analysis_results)
        assert len(patches) > 0

        # Deploy fixes starting from root cause
        test_results = mock_orchestrator.test_patches(patches)
        assert any(test_results.values())  # At least some patches should work

    @pytest.mark.asyncio
    async def test_circuit_breaker_healing(self, mock_environment, mock_orchestrator):
        """Test healing with circuit breaker pattern."""
        service = mock_environment.create_service("resilient_service")
        service.start()

        # Inject repeated failures to trigger circuit breaker
        for i in range(5):
            mock_environment.inject_error(
                service="resilient_service",
                error_type="TimeoutError",
                message=f"Request timeout {i + 1}",
                file_path="service/client.py",
                line_number=80,
            )

        # Detect pattern of failures
        errors = mock_orchestrator.monitor_for_errors()
        assert len(errors) >= 5

        # Analysis should recommend circuit breaker
        analysis_results = mock_orchestrator.analyze_errors(errors)

        # Generate patch with circuit breaker pattern
        patches = mock_orchestrator.generate_patches(analysis_results)

        # Deploy and verify service becomes resilient
        mock_orchestrator.deploy_patches(patches)

        # Service should handle failures gracefully now
        assert len(mock_environment.deployed_patches) > 0

    @pytest.mark.asyncio
    async def test_memory_leak_healing(self, mock_environment, mock_orchestrator):
        """Test healing of memory-related issues."""
        service = mock_environment.create_service("memory_service")
        service.start()

        # Inject memory-related error
        mock_environment.inject_error(
            service="memory_service",
            error_type="MemoryError",
            message="Unable to allocate memory",
            file_path="processors/data_handler.py",
            line_number=300,
        )

        # Add context about large data structure
        error_logs = mock_environment.get_error_logs()
        if error_logs:
            error_logs[-1][
                "stack_trace"
            ] = """Traceback (most recent call last):
  File "processors/data_handler.py", line 300, in process_large_dataset
    results.append(transform_data(item))
MemoryError: Unable to allocate memory"""

        # Detect and analyze
        errors = mock_orchestrator.monitor_for_errors()
        analysis_results = mock_orchestrator.analyze_errors(errors)

        # Generate memory-efficient patch
        patches = mock_orchestrator.generate_patches(analysis_results)

        # Deploy and verify
        test_results = mock_orchestrator.test_patches(patches)
        assert any(test_results.values())

    @pytest.mark.asyncio
    async def test_race_condition_healing(self, mock_environment, mock_orchestrator):
        """Test healing of concurrency issues."""
        service = mock_environment.create_service("concurrent_service")
        service.start()

        # Inject race condition error
        mock_environment.inject_error(
            service="concurrent_service",
            error_type="RuntimeError",
            message="dictionary changed size during iteration",
            file_path="workers/processor.py",
            line_number=150,
        )

        # Analyze concurrency issue
        errors = mock_orchestrator.monitor_for_errors()
        analysis_results = mock_orchestrator.analyze_errors(errors)

        # Generate thread-safe patch
        patches = mock_orchestrator.generate_patches(analysis_results)

        # Verify patch addresses concurrency
        if patches:
            patch_code = patches[0].get("new_code", "")
            # Check for thread-safety patterns
            thread_safe_patterns = ["lock", "Lock", "copy()", "list(", "threading"]
            has_thread_safety = any(
                pattern in patch_code for pattern in thread_safe_patterns
            )

            # Verify that thread safety mechanisms were added
            assert has_thread_safety, "Patch should include thread safety mechanisms"

            # For this mock, we'll accept any patch as valid
            test_results = mock_orchestrator.test_patches(patches)
            assert any(test_results.values())

    @pytest.mark.asyncio
    async def test_api_contract_healing(self, mock_environment, mock_orchestrator):
        """Test healing of API contract violations."""
        service = mock_environment.create_service("api_service")
        service.start()

        # Inject API contract error
        mock_environment.inject_error(
            service="api_service",
            error_type="ValidationError",
            message="Required field 'email' missing",
            file_path="api/validators.py",
            line_number=75,
        )

        # Detect and analyze
        errors = mock_orchestrator.monitor_for_errors()
        analysis_results = mock_orchestrator.analyze_errors(errors)

        # Generate patch that adds validation
        patches = mock_orchestrator.generate_patches(analysis_results)

        # Deploy and verify
        if patches:
            test_results = mock_orchestrator.test_patches(patches)
            assert any(test_results.values())
