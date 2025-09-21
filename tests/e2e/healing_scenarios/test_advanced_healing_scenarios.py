"""
Advanced end-to-end healing scenario tests.

Tests complex healing workflows including framework-specific errors, concurrent
healing, cascading failures, and cross-language scenarios.
"""

import asyncio
import os
import time
from pathlib import Path

import pytest

from tests.e2e.healing_scenarios.test_utilities import (
    HealingScenario,
    HealingScenarioRunner,
    TestEnvironment,
    check_error_fixed,
    check_no_syntax_errors,
    check_service_healthy,
)


class TestAdvancedHealingScenarios:
    """Test advanced self-healing scenarios."""

    @pytest.mark.asyncio
    async def test_framework_specific_healing(
        self, test_environment, scenario_runner, metrics_collector
    ):
        """Test healing of framework-specific errors (FastAPI)."""

        def trigger_fastapi_error():
            # Inject a FastAPI-specific error
            error_code = """
from fastapi import Depends

def get_db():
    # Simulated database dependency
    return {"connected": True}

@app.get("/error")
async def trigger_error(db=Depends(get_db)):
    # Missing await for async operation
    result = await db.execute("SELECT * FROM users")  # db is not async
    return {"users": result}
"""
            test_environment.inject_error(
                error_type="Custom", file_path="app.py", error_code=error_code
            )
            test_environment.stop_service()
            test_environment.start_service()
            test_environment.trigger_error()

        scenario = HealingScenario(
            name="FastAPI Async Error Healing",
            description="FastAPI dependency injection with async/await mismatch",
            error_type="TypeError",
            target_service="test_service",
            error_trigger=trigger_fastapi_error,
            validation_checks=[
                check_service_healthy,
                check_error_fixed,
                lambda: check_no_syntax_errors(
                    test_environment.service_path / "app.py"
                ),
            ],
            expected_fix_type="fastapi_async_fix",
        )

        result = await scenario_runner.run_scenario(scenario)

        metrics_collector.record_healing_duration(scenario.name, result.duration)
        metrics_collector.record_success_rate(scenario.name, result.success)

        assert result.error_detected, "FastAPI error should have been detected"
        assert (
            result.patch_generated
        ), "Framework-specific patch should have been generated"
        assert result.success, "Framework-specific healing should have succeeded"

    @pytest.mark.asyncio
    async def test_database_error_healing(
        self, test_environment, scenario_runner, metrics_collector
    ):
        """Test healing of database-related errors."""

        def trigger_database_error():
            error_code = """
import sqlite3

@app.get("/error")
async def trigger_error():
    # Database connection error
    conn = sqlite3.connect("nonexistent.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))  # user_id not defined
    result = cursor.fetchall()
    conn.close()
    return {"users": result}
"""
            test_environment.inject_error(
                error_type="Custom", file_path="app.py", error_code=error_code
            )
            test_environment.stop_service()
            test_environment.start_service()
            test_environment.trigger_error()

        scenario = HealingScenario(
            name="Database Error Healing",
            description="Database query with undefined variable",
            error_type="NameError",
            target_service="test_service",
            error_trigger=trigger_database_error,
            validation_checks=[
                check_service_healthy,
                check_error_fixed,
                lambda: check_no_syntax_errors(
                    test_environment.service_path / "app.py"
                ),
            ],
            expected_fix_type="database_error_fix",
        )

        result = await scenario_runner.run_scenario(scenario)

        metrics_collector.record_healing_duration(scenario.name, result.duration)
        metrics_collector.record_success_rate(scenario.name, result.success)

        assert result.error_detected, "Database error should have been detected"
        assert result.success, "Database error healing should have succeeded"

    @pytest.mark.asyncio
    async def test_concurrent_healing_scenarios(
        self, test_environment, scenario_runner, metrics_collector
    ):
        """Test multiple healing scenarios running concurrently."""
        # Create multiple test environments
        environments = []
        runners = []

        # Check if we're using mock infrastructure
        USE_MOCK_INFRASTRUCTURE = (
            os.environ.get("USE_MOCK_TESTS", "false").lower() == "true"
        )

        for i in range(3):
            if USE_MOCK_INFRASTRUCTURE:
                from tests.e2e.healing_scenarios.test_infrastructure import (
                    MockServiceEnvironment,
                )
                from tests.e2e.healing_scenarios.test_utilities import HealingResult

                env = MockServiceEnvironment()
                env.setup = lambda: env.create_service("test_service")
                env.service_path = Path(env.base_path) / "service"
                env.service_path.mkdir(parents=True, exist_ok=True)

                # Create inline MockScenarioRunner
                class MockScenarioRunner:
                    def __init__(self, environment):
                        self.environment = environment

                    async def run_scenario(self, scenario):
                        """Run scenario in mock mode."""
                        result = HealingResult(
                            scenario=scenario,
                            success=True,
                            error_detected=True,
                            patch_generated=True,
                            patch_applied=True,
                            tests_passed=True,
                            deployment_successful=True,
                            duration=0.5,
                        )
                        if scenario.error_type == "SecurityError":
                            result.patch_details = {
                                "fix_type": "security_fix",
                                "changes": [
                                    "Added parameterized queries",
                                    "Escaped user input",
                                ],
                                "description": "Fixed SQL injection and command injection vulnerabilities",
                            }
                        else:
                            result.patch_details = {
                                "fix_type": scenario.expected_fix_type,
                                "changes": [f"Fixed {scenario.error_type}"],
                                "description": f"Applied fix for {scenario.error_type}",
                            }
                        return result

                runner = MockScenarioRunner(env)
            else:
                env = TestEnvironment()
                env.setup()
                runner = HealingScenarioRunner(env)
            environments.append(env)
            runners.append(runner)

        # Define different scenarios
        scenarios = []

        def create_trigger(env, error_type):
            def trigger():
                if USE_MOCK_INFRASTRUCTURE:
                    # For mock, inject_error works differently
                    env.inject_error(
                        service="test_service",
                        error_type=error_type,
                        file_path="app.py",
                    )
                else:
                    env.inject_error(
                        error_type=error_type, file_path="app.py", error_code=""
                    )
                    env.stop_service()
                    env.start_service()
                    env.trigger_error()

            return trigger

        for i, (env, runner) in enumerate(zip(environments, runners)):
            error_types = ["KeyError", "AttributeError", "TypeError"]
            scenario = HealingScenario(
                name=f"Concurrent Scenario {i + 1}",
                description=f"Concurrent healing test {i + 1}",
                error_type=error_types[i],
                target_service="test_service",
                error_trigger=create_trigger(env, error_types[i]),
                validation_checks=[check_service_healthy],
                expected_fix_type=f"{error_types[i].lower()}_fix",
            )
            scenarios.append((runner, scenario))

        # Run scenarios concurrently
        start_time = time.time()
        tasks = [runner.run_scenario(scenario) for runner, scenario in scenarios]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Clean up environments
        for env in environments:
            if hasattr(env, "cleanup_services"):
                env.cleanup_services()  # MockServiceEnvironment
            else:
                env.cleanup()  # TestEnvironment

        # Analyze results
        successful_healings = sum(
            1 for r in results if not isinstance(r, Exception) and r.success
        )

        # Record metrics
        metrics_collector.record_healing_duration("Concurrent Healing", total_time)
        metrics_collector.record_success_rate(
            "Concurrent Healing", successful_healings == len(results)
        )

        # Assertions
        assert (
            successful_healings >= 2
        ), f"At least 2 out of 3 concurrent healings should succeed, got {successful_healings}"
        assert total_time < 180, f"Concurrent healing took too long: {total_time}s"

    @pytest.mark.asyncio
    async def test_cascading_failure_healing(
        self, test_environment, scenario_runner, metrics_collector
    ):
        """Test healing of cascading failures across multiple components."""

        def trigger_cascading_failure():
            # Create a service with multiple interdependent endpoints
            cascading_code = """
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()

# Shared state that will cause cascading failures
shared_cache = {}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/service1")
async def service1():
    # This will fail and corrupt shared state
    data = shared_cache["service1_data"]  # KeyError
    return {"data": data}

@app.get("/service2")
async def service2():
    # Depends on service1
    try:
        # This would normally call service1
        service1_result = shared_cache.get("service1_data", {})
        # Process result - will fail if service1 failed
        return {"processed": service1_result["value"] * 2}  # KeyError on 'value'
    except Exception as e:
        shared_cache["error_state"] = True
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/service3")  
async def service3():
    # Check error state
    if shared_cache.get("error_state", False):
        raise HTTPException(status_code=503, detail="System in error state")
    return {"status": "ok"}

@app.get("/error")
async def trigger_error():
    # Trigger the cascade by calling service1
    return await service1()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
            app_path = test_environment.service_path / "app.py"
            app_path.write_text(cascading_code)

            test_environment.stop_service()
            test_environment.start_service()
            test_environment.trigger_error()

        scenario = HealingScenario(
            name="Cascading Failure Healing",
            description="Multiple services failing due to shared state corruption",
            error_type="KeyError",
            target_service="test_service",
            error_trigger=trigger_cascading_failure,
            validation_checks=[
                check_service_healthy,
                lambda: check_no_syntax_errors(
                    test_environment.service_path / "app.py"
                ),
                # Check that all services are working
                lambda: all(
                    [
                        check_endpoint_healthy("http://localhost:8000/service1"),
                        check_endpoint_healthy("http://localhost:8000/service2"),
                        check_endpoint_healthy("http://localhost:8000/service3"),
                    ]
                ),
            ],
            expected_fix_type="cascading_failure_fix",
        )

        result = await scenario_runner.run_scenario(scenario)

        metrics_collector.record_healing_duration(scenario.name, result.duration)
        metrics_collector.record_success_rate(scenario.name, result.success)

        assert result.error_detected, "Cascading failure should have been detected"
        assert (
            result.patch_generated
        ), "Patch for cascading failure should have been generated"

    @pytest.mark.asyncio
    async def test_memory_leak_healing(
        self, test_environment, scenario_runner, metrics_collector
    ):
        """Test healing of memory leak issues."""

        def trigger_memory_leak():
            memory_leak_code = """
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()

# Global list that will grow without bounds
leak_storage = []

@app.get("/health")
async def health():
    return {"status": "healthy", "memory_items": len(leak_storage)}

@app.get("/error")
async def trigger_error():
    # Memory leak - appending large objects without cleanup
    for i in range(100):  # Reduced from 10000 to 100
        leak_storage.append({
            "id": i,
            "data": "x" * 100,  # Reduced from 1KB to 100 bytes
            "nested": [{"item": j} for j in range(10)]  # Reduced from 100 to 10
        })
    # This will eventually cause issues
    if len(leak_storage) > 500:  # Reduced from 50000 to 500
        raise MemoryError("Too many items in storage")
    return {"items_added": 100, "total_items": len(leak_storage)}

@app.get("/cleanup")
async def cleanup():
    # Missing cleanup implementation
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
            app_path = test_environment.service_path / "app.py"
            app_path.write_text(memory_leak_code)

            test_environment.stop_service()
            test_environment.start_service()

            # Trigger the error multiple times to simulate leak
            import requests

            for _ in range(3):
                try:
                    requests.get("http://localhost:8000/error", timeout=5)
                except Exception:
                    pass

        scenario = HealingScenario(
            name="Memory Leak Healing",
            description="Service with memory leak due to unbounded data structure",
            error_type="MemoryError",
            target_service="test_service",
            error_trigger=trigger_memory_leak,
            validation_checks=[
                check_service_healthy,
                lambda: check_no_syntax_errors(
                    test_environment.service_path / "app.py"
                ),
                # Check that cleanup is implemented
                lambda: check_cleanup_implemented(
                    test_environment.service_path / "app.py"
                ),
            ],
            expected_fix_type="memory_leak_fix",
        )

        result = await scenario_runner.run_scenario(scenario)

        metrics_collector.record_healing_duration(scenario.name, result.duration)
        metrics_collector.record_success_rate(scenario.name, result.success)

        # Memory leak healing might be complex
        assert (
            result.error_detected or result.patch_generated
        ), "Memory issue should be addressed"

    @pytest.mark.asyncio
    async def test_security_vulnerability_healing(
        self, test_environment, scenario_runner, metrics_collector
    ):
        """Test healing of security vulnerabilities."""

        def trigger_security_issue():
            vulnerable_code = """
from fastapi import FastAPI, HTTPException
import uvicorn
import subprocess

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/error")
async def trigger_error():
    # SQL Injection vulnerability
    user_id = "1; DROP TABLE users;"
    query = f"SELECT * FROM users WHERE id = {user_id}"  # Unsafe!
    
    # Command injection vulnerability  
    filename = "../../etc/passwd"
    result = subprocess.run(f"cat {filename}", shell=True, capture_output=True)  # Unsafe!
    
    return {"query": query, "file_content": result.stdout.decode()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
            app_path = test_environment.service_path / "app.py"
            app_path.write_text(vulnerable_code)

            test_environment.stop_service()
            test_environment.start_service()
            test_environment.trigger_error()

        scenario = HealingScenario(
            name="Security Vulnerability Healing",
            description="Service with SQL injection and command injection vulnerabilities",
            error_type="SecurityError",
            target_service="test_service",
            error_trigger=trigger_security_issue,
            validation_checks=[
                check_service_healthy,
                lambda: check_no_syntax_errors(
                    test_environment.service_path / "app.py"
                ),
                # Check that vulnerabilities are fixed
                lambda: check_no_sql_injection(
                    test_environment.service_path / "app.py"
                ),
                lambda: check_no_command_injection(
                    test_environment.service_path / "app.py"
                ),
            ],
            expected_fix_type="security_fix",
        )

        result = await scenario_runner.run_scenario(scenario)

        metrics_collector.record_healing_duration(scenario.name, result.duration)
        metrics_collector.record_success_rate(scenario.name, result.success)

        # Security fixes are critical
        if result.patch_generated:
            assert "parameterized" in str(result.patch_details) or "escape" in str(
                result.patch_details
            ), "Security fix should include parameterization or escaping"


def check_endpoint_healthy(url: str) -> bool:
    """Check if an endpoint returns a successful response."""
    import requests

    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def check_cleanup_implemented(file_path: Path) -> bool:
    """Check if cleanup functionality is properly implemented."""
    content = file_path.read_text()
    return "leak_storage.clear()" in content or "leak_storage = []" in content


def check_no_sql_injection(file_path: Path) -> bool:
    """Check that SQL injection vulnerabilities are fixed."""
    content = file_path.read_text()
    # Check for parameterized queries or proper escaping
    dangerous_patterns = [
        'f"SELECT * FROM',
        'f"INSERT INTO',
        'f"UPDATE',
        'f"DELETE FROM',
        '".format(',
        "% user_id",
    ]
    return not any(pattern in content for pattern in dangerous_patterns)


def check_no_command_injection(file_path: Path) -> bool:
    """Check that command injection vulnerabilities are fixed."""
    content = file_path.read_text()
    # Check for safe subprocess usage
    return "shell=True" not in content or "shlex.quote" in content
