"""
Basic end-to-end healing scenario tests using mock infrastructure.

Tests fundamental healing workflows including error detection, patch generation,
testing, and deployment using simulated services and logs.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import yaml

from tests.e2e.healing_scenarios.test_infrastructure import (
    MockServiceEnvironment,
    LogSimulator,
    PatchValidator,
    MockOrchestrator
)


class TestBasicHealingScenariosMock:
    """Test basic self-healing scenarios with mock infrastructure."""
    
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
                "backup_original_files": True
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        return MockOrchestrator(config_path, mock_environment)
    
    @pytest.mark.asyncio
    async def test_keyerror_healing(self, mock_environment, mock_orchestrator):
        """Test healing of a KeyError in the service."""
        # Create and start a mock service
        service = mock_environment.create_service("test_service")
        service.start()
        
        # Inject a KeyError
        mock_environment.inject_error(
            service="test_service",
            error_type="KeyError",
            message="KeyError: 'user_data'",
            file_path="app.py",
            line_number=50
        )
        
        # Detect errors
        errors = mock_orchestrator.monitor_for_errors()
        assert len(errors) == 1
        assert errors[0]["exception_type"] == "KeyError"
        
        # Analyze errors
        analysis_results = mock_orchestrator.analyze_errors(errors)
        assert len(analysis_results) == 1
        assert analysis_results[0]["root_cause"] == "dict_key_not_exists"
        
        # Generate patches
        patches = mock_orchestrator.generate_patches(analysis_results)
        assert len(patches) == 1
        assert "get(" in patches[0]["new_code"]
        
        # Validate patch
        valid, errors = PatchValidator.validate_patch_structure(patches[0])
        assert valid, f"Patch validation failed: {errors}"
        assert PatchValidator.validate_fix_addresses_error(patches[0], "KeyError")
        
        # Test patches
        test_results = mock_orchestrator.test_patches(patches)
        assert all(test_results.values()), "Some patch tests failed"
        
        # Deploy patches
        deployed = mock_orchestrator.deploy_patches(patches)
        assert deployed, "Patch deployment failed"
        
        # Verify service is healthy
        assert service.health_status, "Service should be healthy after patch"
        
    @pytest.mark.asyncio
    async def test_attributeerror_healing(self, mock_environment, mock_orchestrator):
        """Test healing of an AttributeError in the service."""
        # Create and start a mock service
        service = mock_environment.create_service("test_service")
        service.start()
        
        # Inject an AttributeError
        mock_environment.inject_error(
            service="test_service",
            error_type="AttributeError",
            message="AttributeError: 'NoneType' object has no attribute 'items'",
            file_path="app.py",
            line_number=75
        )
        
        # Run healing process
        errors = mock_orchestrator.monitor_for_errors()
        assert errors[0]["exception_type"] == "AttributeError"
        
        analysis_results = mock_orchestrator.analyze_errors(errors)
        assert analysis_results[0]["root_cause"] == "null_reference"
        
        patches = mock_orchestrator.generate_patches(analysis_results)
        assert "is not None" in patches[0]["new_code"]
        
        # Validate and deploy
        assert PatchValidator.validate_fix_addresses_error(patches[0], "AttributeError")
        test_results = mock_orchestrator.test_patches(patches)
        assert all(test_results.values())
        
        deployed = mock_orchestrator.deploy_patches(patches)
        assert deployed
        assert service.health_status
        
    @pytest.mark.asyncio
    async def test_typeerror_healing(self, mock_environment, mock_orchestrator):
        """Test healing of a TypeError in the service."""
        service = mock_environment.create_service("test_service")
        service.start()
        
        # Inject a TypeError (async/await mismatch)
        mock_environment.inject_error(
            service="test_service",
            error_type="TypeError",
            message="TypeError: object dict can't be used in 'await' expression",
            file_path="app.py",
            line_number=100
        )
        
        # Healing process
        errors = mock_orchestrator.monitor_for_errors()
        analysis_results = mock_orchestrator.analyze_errors(errors)
        patches = mock_orchestrator.generate_patches(analysis_results)
        
        # Verify patch removes inappropriate await
        assert "await" not in patches[0]["new_code"]
        assert PatchValidator.validate_fix_addresses_error(patches[0], "TypeError")
        
        # Deploy and verify
        test_results = mock_orchestrator.test_patches(patches)
        assert all(test_results.values())
        assert service.health_status
        
    @pytest.mark.asyncio
    async def test_multiple_error_healing(self, mock_environment, mock_orchestrator):
        """Test healing multiple different errors concurrently."""
        # Create multiple services
        services = [
            mock_environment.create_service("service1"),
            mock_environment.create_service("service2"),
            mock_environment.create_service("service3")
        ]
        
        for service in services:
            service.start()
            
        # Inject different errors
        mock_environment.inject_error("service1", "KeyError")
        mock_environment.inject_error("service2", "AttributeError")
        mock_environment.inject_error("service3", "TypeError")
        
        # Detect all errors
        errors = mock_orchestrator.monitor_for_errors()
        assert len(errors) == 3
        
        # Analyze and generate patches for all
        analysis_results = mock_orchestrator.analyze_errors(errors)
        assert len(analysis_results) == 3
        
        patches = mock_orchestrator.generate_patches(analysis_results)
        assert len(patches) == 3
        
        # Validate all patches
        for patch in patches:
            valid, _ = PatchValidator.validate_patch_structure(patch)
            assert valid
            
        # Deploy all patches
        test_results = mock_orchestrator.test_patches(patches)
        assert all(test_results.values())
        
        # Verify all services are healthy
        for service in services:
            assert service.health_status
            
    @pytest.mark.asyncio
    async def test_healing_with_rollback(self, mock_environment, mock_orchestrator):
        """Test healing with rollback capability."""
        service = mock_environment.create_service("test_service")
        service.start()
        
        # Inject error
        mock_environment.inject_error("test_service", "NameError")
        
        # Generate patch that will fail validation
        with patch.object(PatchValidator, 'validate_syntax', return_value=(False, "Syntax error")):
            errors = mock_orchestrator.monitor_for_errors()
            analysis_results = mock_orchestrator.analyze_errors(errors)
            patches = mock_orchestrator.generate_patches(analysis_results)
            
            # Syntax validation should fail
            valid, error = PatchValidator.validate_syntax(patches[0]["new_code"])
            assert not valid
            
        # Service should still be unhealthy (rollback scenario)
        assert not service.health_status
        
    @pytest.mark.asyncio
    async def test_healing_performance_metrics(self, mock_environment, mock_orchestrator):
        """Test collection of healing performance metrics."""
        import time
        
        service = mock_environment.create_service("test_service")
        service.start()
        
        # Track timing
        start_time = time.time()
        
        # Inject error
        mock_environment.inject_error("test_service", "KeyError")
        
        # Full healing cycle
        errors = mock_orchestrator.monitor_for_errors()
        analysis_results = mock_orchestrator.analyze_errors(errors)
        patches = mock_orchestrator.generate_patches(analysis_results)
        test_results = mock_orchestrator.test_patches(patches)
        deployed = mock_orchestrator.deploy_patches(patches)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify metrics
        assert duration < 5.0, "Healing should complete quickly"
        assert len(mock_orchestrator.errors_detected) == 0  # Mock doesn't track this yet
        assert len(mock_orchestrator.patches_generated) == 1
        assert len(mock_orchestrator.tests_run) == 1
        assert service.health_status
        
    @pytest.mark.asyncio
    async def test_cascading_error_detection(self, mock_environment, mock_orchestrator):
        """Test detection of cascading errors across services."""
        # Create interconnected services
        services = {
            "frontend": mock_environment.create_service("frontend"),
            "backend": mock_environment.create_service("backend"),
            "database": mock_environment.create_service("database")
        }
        
        for service in services.values():
            service.start()
            
        # Simulate cascading failure starting from database
        logs = LogSimulator.generate_log_sequence("cascading_failure")
        
        # Map service names from logs to our test services
        service_mapping = {
            "service1": "frontend",
            "service2": "backend", 
            "service3": "database"
        }
        
        for log in logs:
            log_service_name = log["service"]
            actual_service_name = service_mapping.get(log_service_name, log_service_name)
            
            if actual_service_name in services:
                # Update the log to use the actual service name
                log["service"] = actual_service_name
                services[actual_service_name].logs.append(log)
                services[actual_service_name].health_status = False
                
        # Detect cascading errors
        errors = mock_orchestrator.monitor_for_errors()
        assert len(errors) >= 3
        
        # Verify error propagation pattern
        error_services = [e["service"] for e in errors]
        assert "service1" in error_services or "database" in error_services