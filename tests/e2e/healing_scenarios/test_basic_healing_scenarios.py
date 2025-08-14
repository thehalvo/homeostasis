"""
Basic end-to-end healing scenario tests.

Tests fundamental healing workflows including error detection, patch generation,
testing, and deployment.
"""
import asyncio
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.e2e.healing_scenarios.test_utilities import (
    HealingScenario,
    HealingScenarioRunner,
    TestEnvironment,
    MetricsCollector,
    check_service_healthy,
    check_error_fixed,
    check_no_syntax_errors
)


class TestBasicHealingScenarios:
    """Test basic self-healing scenarios."""
    
    @pytest.mark.asyncio
    async def test_keyerror_healing(self, test_environment, scenario_runner, metrics_collector):
        """Test healing of a KeyError in the service."""
        # Define the scenario
        def trigger_keyerror():
            test_environment.inject_error(
                error_type="KeyError",
                file_path="app.py",
                error_code=""
            )
            test_environment.stop_service()
            test_environment.start_service()
            test_environment.trigger_error()
            
        scenario = HealingScenario(
            name="KeyError Healing",
            description="Service tries to access a missing dictionary key",
            error_type="KeyError",
            target_service="test_service",
            error_trigger=trigger_keyerror,
            validation_checks=[
                check_service_healthy,
                check_error_fixed,
                lambda: check_no_syntax_errors(test_environment.service_path / "app.py")
            ],
            expected_fix_type="keyerror_fix"
        )
        
        # Run the scenario
        result = await scenario_runner.run_scenario(scenario)
        
        # Record metrics
        metrics_collector.record_healing_duration(scenario.name, result.duration)
        metrics_collector.record_success_rate(scenario.name, result.success)
        
        # Assertions
        assert result.error_detected, "Error should have been detected"
        assert result.patch_generated, "Patch should have been generated"
        assert result.patch_applied, "Patch should have been applied"
        assert result.tests_passed, "Tests should have passed"
        assert result.deployment_successful, "Deployment should have succeeded"
        assert result.success, "Overall healing should have succeeded"
        
    @pytest.mark.asyncio
    async def test_attributeerror_healing(self, test_environment, scenario_runner, metrics_collector):
        """Test healing of an AttributeError in the service."""
        # Define the scenario
        def trigger_attributeerror():
            test_environment.inject_error(
                error_type="AttributeError",
                file_path="app.py",
                error_code=""
            )
            test_environment.stop_service()
            test_environment.start_service()
            test_environment.trigger_error()
            
        scenario = HealingScenario(
            name="AttributeError Healing",
            description="Service tries to access attribute on None object",
            error_type="AttributeError",
            target_service="test_service",
            error_trigger=trigger_attributeerror,
            validation_checks=[
                check_service_healthy,
                check_error_fixed,
                lambda: check_no_syntax_errors(test_environment.service_path / "app.py")
            ],
            expected_fix_type="attribute_error_fix"
        )
        
        # Run the scenario
        result = await scenario_runner.run_scenario(scenario)
        
        # Record metrics
        metrics_collector.record_healing_duration(scenario.name, result.duration)
        metrics_collector.record_success_rate(scenario.name, result.success)
        
        # Assertions
        assert result.error_detected, "Error should have been detected"
        assert result.patch_generated, "Patch should have been generated"
        assert result.patch_applied, "Patch should have been applied"
        assert result.tests_passed, "Tests should have passed"
        assert result.deployment_successful, "Deployment should have succeeded"
        assert result.success, "Overall healing should have succeeded"
        
    @pytest.mark.asyncio
    async def test_typeerror_healing(self, test_environment, scenario_runner, metrics_collector):
        """Test healing of a TypeError in the service."""
        # Define the scenario
        def trigger_typeerror():
            test_environment.inject_error(
                error_type="TypeError",
                file_path="app.py",
                error_code=""
            )
            test_environment.stop_service()
            test_environment.start_service()
            test_environment.trigger_error()
            
        scenario = HealingScenario(
            name="TypeError Healing",
            description="Service performs invalid operation between types",
            error_type="TypeError",
            target_service="test_service",
            error_trigger=trigger_typeerror,
            validation_checks=[
                check_service_healthy,
                check_error_fixed,
                lambda: check_no_syntax_errors(test_environment.service_path / "app.py")
            ],
            expected_fix_type="type_error_fix"
        )
        
        # Run the scenario
        result = await scenario_runner.run_scenario(scenario)
        
        # Record metrics
        metrics_collector.record_healing_duration(scenario.name, result.duration)
        metrics_collector.record_success_rate(scenario.name, result.success)
        
        # Log details for debugging
        if not result.success:
            print(f"Healing failed. Logs:")
            for log in result.logs:
                print(f"  {log}")
                
        # Assertions
        assert result.error_detected, "Error should have been detected"
        assert result.patch_generated, "Patch should have been generated"
        assert result.patch_applied, "Patch should have been applied"
        assert result.tests_passed, "Tests should have passed"
        assert result.deployment_successful, "Deployment should have succeeded"
        assert result.success, "Overall healing should have succeeded"
        
    @pytest.mark.asyncio
    async def test_healing_with_rollback(self, test_environment, scenario_runner, metrics_collector):
        """Test healing with rollback when tests fail."""
        # Define a scenario that will fail tests
        def trigger_complex_error():
            error_code = '''
@app.get("/error")
async def trigger_error():
    # This will cause an error and the fix might break other functionality
    data = {"key1": "value1"}
    result = data["missing_key"]
    # Additional logic that might be broken by naive fix
    important_calculation = len(data) * 100
    return {"result": result, "calculation": important_calculation}
'''
            test_environment.inject_error(
                error_type="Custom",
                file_path="app.py",
                error_code=error_code
            )
            test_environment.stop_service()
            test_environment.start_service()
            test_environment.trigger_error()
            
        scenario = HealingScenario(
            name="Healing with Rollback",
            description="Complex error that might require rollback",
            error_type="KeyError",
            target_service="test_service",
            error_trigger=trigger_complex_error,
            validation_checks=[
                check_service_healthy,
                lambda: check_no_syntax_errors(test_environment.service_path / "app.py")
            ],
            expected_fix_type="keyerror_fix"
        )
        
        # Run the scenario
        result = await scenario_runner.run_scenario(scenario)
        
        # Record metrics
        metrics_collector.record_healing_duration(scenario.name, result.duration)
        metrics_collector.record_success_rate(scenario.name, result.success)
        
        # This scenario might fail or succeed depending on the fix quality
        assert result.error_detected, "Error should have been detected"
        assert result.patch_generated, "Patch should have been generated"
        
    @pytest.mark.asyncio
    async def test_multiple_error_healing(self, test_environment, scenario_runner, metrics_collector):
        """Test healing when multiple errors occur in sequence."""
        errors_triggered = []
        
        def trigger_first_error():
            test_environment.inject_error(
                error_type="KeyError",
                file_path="app.py",
                error_code=""
            )
            test_environment.stop_service()
            test_environment.start_service()
            test_environment.trigger_error()
            errors_triggered.append("KeyError")
            
        # First scenario - KeyError
        scenario1 = HealingScenario(
            name="Multiple Errors - Phase 1",
            description="First error in a sequence",
            error_type="KeyError",
            target_service="test_service",
            error_trigger=trigger_first_error,
            validation_checks=[check_service_healthy],
            expected_fix_type="keyerror_fix"
        )
        
        result1 = await scenario_runner.run_scenario(scenario1)
        
        # After first healing, inject a different error
        def trigger_second_error():
            # Add another endpoint with a different error
            new_endpoint = '''

@app.get("/error2")
async def trigger_error2():
    obj = None
    return {"result": obj.missing_attribute}
'''
            app_path = test_environment.service_path / "app.py"
            content = app_path.read_text()
            app_path.write_text(content + new_endpoint)
            
            test_environment.stop_service()
            test_environment.start_service()
            
            # Trigger the new error
            import requests
            try:
                requests.get("http://localhost:8000/error2", timeout=5)
            except:
                pass
            errors_triggered.append("AttributeError")
            
        # Second scenario - AttributeError
        scenario2 = HealingScenario(
            name="Multiple Errors - Phase 2",
            description="Second error after first healing",
            error_type="AttributeError",
            target_service="test_service",
            error_trigger=trigger_second_error,
            validation_checks=[check_service_healthy],
            expected_fix_type="attribute_error_fix"
        )
        
        result2 = await scenario_runner.run_scenario(scenario2)
        
        # Record metrics
        metrics_collector.record_healing_duration("Multiple Error Sequence", 
                                                result1.duration + result2.duration)
        metrics_collector.record_success_rate("Multiple Error Sequence", 
                                            result1.success and result2.success)
        
        # Both healings should succeed
        assert result1.success, "First healing should succeed"
        assert result2.success, "Second healing should succeed"
        assert len(errors_triggered) == 2, "Both errors should have been triggered"
        

    @pytest.mark.asyncio 
    async def test_healing_performance_metrics(self, test_environment, scenario_runner, metrics_collector):
        """Test that healing completes within reasonable time limits."""
        import time
        
        def trigger_simple_error():
            test_environment.inject_error(
                error_type="KeyError",
                file_path="app.py", 
                error_code=""
            )
            test_environment.stop_service()
            test_environment.start_service()
            test_environment.trigger_error()
            
        scenario = HealingScenario(
            name="Performance Test",
            description="Measure healing performance",
            error_type="KeyError",
            target_service="test_service",
            error_trigger=trigger_simple_error,
            validation_checks=[check_service_healthy],
            expected_fix_type="keyerror_fix",
            timeout=120  # 2 minute timeout
        )
        
        start_time = time.time()
        result = await scenario_runner.run_scenario(scenario)
        total_time = time.time() - start_time
        
        # Record detailed timing metrics
        metrics_collector.record_healing_duration(scenario.name, result.duration)
        
        # Performance assertions
        assert result.success, "Healing should succeed"
        assert result.duration < 60, f"Healing took too long: {result.duration}s"
        assert total_time < scenario.timeout, f"Scenario exceeded timeout: {total_time}s"
        
        # Check individual phase timings from logs
        phase_timings = {}
        for log in result.logs:
            if "Error detected" in log:
                phase_timings['detection'] = log
            elif "Patch generated" in log:
                phase_timings['generation'] = log
            elif "Tests passed" in log:
                phase_timings['testing'] = log
            elif "Deployment successful" in log:
                phase_timings['deployment'] = log
                
        assert len(phase_timings) >= 4, "All healing phases should be recorded"
        

    def test_metrics_summary(self, metrics_collector):
        """Test metrics collection and summary generation."""
        # Add some test data
        metrics_collector.record_healing_duration("Test Scenario 1", 45.5)
        metrics_collector.record_healing_duration("Test Scenario 2", 52.3)
        metrics_collector.record_success_rate("Test Scenario 1", True)
        metrics_collector.record_success_rate("Test Scenario 1", True)
        metrics_collector.record_success_rate("Test Scenario 1", False)
        metrics_collector.record_success_rate("Test Scenario 2", True)
        
        summary = metrics_collector.get_summary()
        
        assert summary['total_scenarios'] == 2
        assert summary['average_healing_duration'] == pytest.approx(48.9, 0.1)
        assert summary['scenario_success_rates']['Test Scenario 1'] == pytest.approx(0.667, 0.01)
        assert summary['scenario_success_rates']['Test Scenario 2'] == 1.0
        assert summary['overall_success_rate'] == 0.75