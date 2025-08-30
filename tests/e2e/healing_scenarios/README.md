# End-to-End Healing Scenario Tests

This directory contains comprehensive end-to-end tests for the Homeostasis self-healing system. These tests simulate real-world error scenarios and validate that the system can detect, diagnose, patch, test, and deploy fixes automatically.

## Overview

The E2E tests cover:

1. **Basic Healing Scenarios** - Common errors like KeyError, AttributeError, TypeError
2. **Advanced Healing Scenarios** - Framework-specific errors, cascading failures, security vulnerabilities
3. **Cross-Language Healing** - Healing across Python, JavaScript, Go, Java services
4. **Performance Testing** - Ensuring healing completes within acceptable time limits
5. **Concurrent Healing** - Multiple healing processes running simultaneously

## Test Structure

```
tests/e2e/healing_scenarios/
├── __init__.py                        # Package initialization
├── test_utilities.py                  # Test utilities and fixtures
├── test_basic_healing_scenarios.py    # Basic healing tests
├── test_advanced_healing_scenarios.py # Advanced healing tests
├── test_cross_language_healing.py     # Cross-language tests
├── run_e2e_tests.py                  # Test runner script
└── README.md                         # This file
```

## Running Tests

### Local Development

1. **Run all tests:**
   ```bash
   python tests/e2e/healing_scenarios/run_e2e_tests.py
   ```

2. **Run specific test suite:**
   ```bash
   python tests/e2e/healing_scenarios/run_e2e_tests.py --suite basic
   ```

3. **Run with verbose output:**
   ```bash
   python tests/e2e/healing_scenarios/run_e2e_tests.py --verbose
   ```

4. **Run tests in parallel:**
   ```bash
   python tests/e2e/healing_scenarios/run_e2e_tests.py --parallel
   ```

### Using Docker

1. **Build and run tests in Docker:**
   ```bash
   docker-compose -f tests/e2e/docker-compose.yml up --build
   ```

2. **Run specific test suite in Docker:**
   ```bash
   docker-compose -f tests/e2e/docker-compose.yml run test-runner \
     python run_e2e_tests.py --suite advanced
   ```

### CI/CD Integration

Tests automatically run on:
- Push to main/develop branches
- Pull requests
- Daily schedule (2 AM UTC)
- Manual workflow dispatch

## Test Scenarios

### Basic Healing Scenarios

1. **KeyError Healing**
   - Simulates missing dictionary key access
   - Validates proper null checking is added

2. **AttributeError Healing**
   - Simulates attribute access on None
   - Validates defensive programming fixes

3. **TypeError Healing**
   - Simulates type mismatches
   - Validates type conversion fixes

### Advanced Healing Scenarios

1. **Framework-Specific Errors**
   - FastAPI async/await issues
   - Django ORM errors
   - Flask routing problems

2. **Database Errors**
   - Connection failures
   - Query syntax errors
   - Transaction issues

3. **Cascading Failures**
   - Service dependency failures
   - Shared state corruption
   - Circuit breaker patterns

4. **Security Vulnerabilities**
   - SQL injection
   - Command injection
   - XSS vulnerabilities

### Cross-Language Healing

1. **JavaScript/Node.js**
   - Undefined property access
   - Promise rejection handling
   - Module import errors

2. **Go**
   - Nil pointer dereference
   - Channel deadlocks
   - Interface type assertions

3. **Java**
   - NullPointerException
   - ClassCastException
   - ConcurrentModificationException

## Test Utilities

### TestEnvironment

Creates isolated test environments with:
- Temporary service directories
- Custom configurations
- Log collection
- Service lifecycle management

### HealingScenarioRunner

Executes healing scenarios and tracks:
- Error detection
- Patch generation
- Test execution
- Deployment success
- Performance metrics

### MetricsCollector

Collects and analyzes:
- Healing duration
- Success rates
- Resource usage
- Performance trends

## Writing New Tests

1. **Create a new test file:**
   ```python
   # test_custom_healing_scenarios.py
   import pytest
   from tests.e2e.healing_scenarios.test_utilities import (
       HealingScenario,
       HealingScenarioRunner,
       TestEnvironment
   )
   ```

2. **Define a healing scenario:**
   ```python
   def trigger_custom_error():
       # Code to trigger the error
       pass
       
   scenario = HealingScenario(
       name="Custom Error Healing",
       description="Description of the error",
       error_type="ErrorType",
       target_service="service_name",
       error_trigger=trigger_custom_error,
       validation_checks=[check_service_healthy],
       expected_fix_type="custom_fix"
   )
   ```

3. **Run and validate:**
   ```python
   @pytest.mark.asyncio
   async def test_custom_healing(test_environment, scenario_runner):
       result = await scenario_runner.run_scenario(scenario)
       assert result.success
   ```

## Test Configuration

### Environment Variables

- `TEST_ENV`: Test environment (local/docker/ci)
- `LOG_LEVEL`: Logging verbosity (DEBUG/INFO/WARNING/ERROR)
- `HEALING_TIMEOUT`: Maximum time for healing cycle (seconds)
- `USE_MOCK_TESTS`: Use mock infrastructure instead of real services (true/false, default: false)
  - When set to `true`, tests use simulated services for faster execution
  - Recommended for local development and rapid iteration
  - Example: `USE_MOCK_TESTS=true pytest tests/e2e/healing_scenarios/`

### Configuration Files

- `config.test.yaml`: Test-specific orchestrator configuration
- `prometheus.yml`: Metrics collection configuration
- `grafana/dashboards/`: Test monitoring dashboards

## Monitoring and Observability

### Metrics

- Healing success rate
- Average healing duration
- Error detection accuracy
- Patch quality metrics

### Dashboards

Access Grafana dashboards at http://localhost:3000 when running with Docker Compose.

### Logs

- Application logs: `logs/app.log`
- Orchestrator logs: `logs/orchestrator.log`
- Test results: `test_results/`

## Troubleshooting

### Common Issues

1. **Service startup failures**
   - Check port availability (8000, 8001, 8002)
   - Verify dependencies are installed
   - Review service logs

2. **Timeout errors**
   - Increase `--timeout` parameter
   - Check system resources
   - Review healing cycle logs

3. **Docker issues**
   - Ensure Docker daemon is running
   - Check available disk space
   - Review container logs with `docker-compose logs`

### Debug Mode

Enable debug logging:
```bash
LOG_LEVEL=DEBUG python run_e2e_tests.py --verbose
```

## Contributing

1. Add tests for new healing capabilities
2. Ensure tests are idempotent and isolated
3. Document complex test scenarios
4. Run full test suite before submitting PR
5. Update this README with new test descriptions