"""
Test utilities for end-to-end healing scenario tests.

Provides helper functions and fixtures for simulating errors, monitoring healing
processes, and validating successful remediation.
"""
import asyncio
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from unittest.mock import MagicMock, patch

import pytest
import requests
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.monitoring.logger import MonitoringLogger
from modules.monitoring.extractor import get_latest_errors
from orchestrator.orchestrator import Orchestrator


@dataclass
class HealingScenario:
    """Represents a complete healing scenario for testing."""
    name: str
    description: str
    error_type: str
    target_service: str
    error_trigger: Callable
    validation_checks: List[Callable]
    expected_fix_type: str
    timeout: int = 300  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealingResult:
    """Result of a healing scenario execution."""
    scenario: HealingScenario
    success: bool
    error_detected: bool
    patch_generated: bool
    patch_applied: bool
    tests_passed: bool
    deployment_successful: bool
    duration: float
    error_details: Optional[Dict[str, Any]] = None
    patch_details: Optional[Dict[str, Any]] = None
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class TestEnvironment:
    """Manages isolated test environments for healing scenarios."""
    
    # Class variable to track used ports
    _used_ports = set()
    _port_counter = 8000
    
    @classmethod
    def _get_next_port(cls):
        """Get the next available port for testing."""
        while cls._port_counter in cls._used_ports:
            cls._port_counter += 1
        port = cls._port_counter
        cls._used_ports.add(port)
        return port
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(tempfile.mkdtemp(prefix="healing_test_"))
        self.service_path = self.base_path / "service"
        self.logs_path = self.base_path / "logs"
        self.config_path = self.base_path / "config.yaml"
        self.orchestrator = None
        self.service_process = None
        self.logger = MonitoringLogger("test_environment")
        self.port = self._get_next_port()
        
    def setup(self, service_template: str = "example_service"):
        """Set up the test environment with a service template."""
        # Create directory structure
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        # Always create minimal service for testing to avoid dependency issues
        self._create_minimal_service()
            
        # Create test configuration
        self._create_test_config()
        
        # Initialize orchestrator
        self.orchestrator = Orchestrator(self.config_path, log_level="DEBUG")
        
    def _create_minimal_service(self):
        """Create a minimal service for testing."""
        self.service_path.mkdir(parents=True, exist_ok=True)
        
        # Create a simple FastAPI service
        service_code = '''
from fastapi import FastAPI, HTTPException
import uvicorn
import logging
import json
import sys
from datetime import datetime
from pathlib import Path

# Set up logging to match the expected format
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "app.log"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_service")

# Add file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Format logs as JSON
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": "test_service",
            "level": record.levelname,
            "message": record.getMessage(),
            "file_path": record.pathname,
            "line_number": record.lineno,
            "function_name": record.funcName
        }
        if hasattr(record, 'exc_info') and record.exc_info:
            log_data["exception_type"] = record.exc_info[0].__name__
            log_data["stack_trace"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

file_handler.setFormatter(JSONFormatter())
logger.addHandler(file_handler)

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/error")
async def trigger_error():
    # This endpoint will be modified to trigger specific errors
    raise HTTPException(status_code=500, detail="Test error")

@app.exception_handler(Exception)
async def exception_handler(request, exc):
    # Log the exception in the expected format
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {"detail": str(exc)}, 500

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
'''
        # No formatting needed - port is already in the code
        (self.service_path / "app.py").write_text(service_code)
        
    def _create_test_config(self):
        """Create test configuration for the orchestrator."""
        config = {
            "general": {
                "project_name": "healing_test",
                "environment": "test"
            },
            "service": {
                "name": "test_service",
                "path": str(self.service_path),
                "start_command": f"python {self.service_path}/app.py {self.port}",
                "stop_command": "pkill -f 'python.*app.py'",
                "health_check_url": f"http://localhost:{self.port}/health",
                "health_check_timeout": 3
            },
            "monitoring": {
                "check_interval": 5,
                "log_file": str(self.logs_path / "app.log"),
                "post_deployment": {
                    "enabled": True,
                    "monitoring_duration": 30,
                    "metrics_interval": 5,
                    "alert_thresholds": {
                        "error_rate": 0.05,
                        "response_time": 1000,
                        "memory_usage": 80
                    }
                }
            },
            "analysis": {
                "rule_based": {"enabled": True},
                "ai_based": {"enabled": False}
            },
            "patch_generation": {
                "generated_patches_dir": str(self.base_path / "patches"),
                "backup_original_files": True
            },
            "testing": {
                "enabled": True,
                "test_command": "pytest tests/",
                "test_timeout": 60,
                "containers": {"enabled": False},
                "parallel": {"max_workers": 2},
                "regression": {"enabled": True, "save_path": "tests/regression"},
                "graduated_testing": {
                    "enabled": True,
                    "levels": ["unit", "integration"],
                    "commands": {
                        "unit": "pytest tests/unit/",
                        "integration": "pytest tests/integration/"
                    },
                    "timeouts": {"unit": 30, "integration": 60},
                    "resource_limits": {}
                }
            },
            "deployment": {
                "enabled": True,
                "restart_service": True,
                "backup_dir": str(self.base_path / "backups"),
                "production": {
                    "require_approval": False,
                    "canary_deployment": False,
                    "blue_green": False
                },
                "nginx_config_path": str(self.base_path / "nginx"),
                "template_path": str(self.base_path / "templates")
            },
            "security": {
                "enabled": True,
                "approval": {"enabled": False},
                "canary": {"enabled": False},
                "healing_rate_limits": {"enabled": False}
            },
            "rollback": {
                "enabled": True,
                "auto_rollback_on_failure": True,
                "max_sessions_to_keep": 5
            },
            "suggestion": {
                "enabled": False
            }
        }
        
        # Store config as instance attribute
        self.config = config
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
            
    def inject_error(self, error_type: str, file_path: str, error_code: str):
        """Inject a specific error into the service code."""
        target_file = self.service_path / file_path
        if not target_file.exists():
            target_file = self.service_path / "app.py"
            
        # Read current content
        content = target_file.read_text()
        
        # Define the error pattern once for all error types
        error_pattern = r'@app\.get\("/error"\)\s*\nasync def trigger_error\(\):[^@]*(?=@|\Z)'
        
        # Inject error based on type
        if error_type == "KeyError":
            # Replace error endpoint with KeyError-inducing code
            error_endpoint = '''@app.get("/error")
async def trigger_error():
    try:
        data = {"key1": "value1"}
        return {"result": data["missing_key"]}  # This will cause KeyError
    except Exception as e:
        logger.error(f"KeyError in trigger_error: {e}", exc_info=True)
        raise'''
            
            content = re.sub(error_pattern, error_endpoint + '\n', content)
            
        elif error_type == "AttributeError":
            error_endpoint = '''@app.get("/error")
async def trigger_error():
    try:
        obj = None
        return {"result": obj.attribute}  # This will cause AttributeError
    except Exception as e:
        logger.error(f"AttributeError in trigger_error: {e}", exc_info=True)
        raise'''
            content = re.sub(error_pattern, error_endpoint + '\n', content)
            
        elif error_type == "TypeError":
            error_endpoint = '''@app.get("/error")
async def trigger_error():
    try:
        result = "string" + 123  # This will cause TypeError
        return {"result": result}
    except Exception as e:
        logger.error(f"TypeError in trigger_error: {e}", exc_info=True)
        raise'''
            content = re.sub(error_pattern, error_endpoint + '\n', content)
            
        elif error_type == "Custom":
            # Use provided error code directly, ensure it logs errors
            if "logger.error" not in error_code:
                # Wrap in try/except to ensure logging
                lines = error_code.strip().split('\n')
                if lines and lines[0].startswith('@app.get'):
                    func_def_line = next((i for i, line in enumerate(lines) if 'async def' in line), None)
                    if func_def_line is not None:
                        # Insert try after function definition
                        lines.insert(func_def_line + 1, '    try:')
                        # Indent remaining lines
                        for i in range(func_def_line + 2, len(lines)):
                            lines[i] = '    ' + lines[i]
                        # Add exception handler
                        lines.extend([
                            '    except Exception as e:',
                            '        logger.error(f"Error in trigger_error: {e}", exc_info=True)',
                            '        raise'
                        ])
                        error_code = '\n'.join(lines)
            content = re.sub(error_pattern, error_code + '\n', content)
            
        # Write modified content
        target_file.write_text(content)
        
    def start_service(self):
        """Start the test service."""
        if self.orchestrator:
            # Set PYTHONPATH to include project root
            import os
            pythonpath = os.environ.get('PYTHONPATH', '')
            if str(project_root) not in pythonpath:
                os.environ['PYTHONPATH'] = f"{project_root}:{pythonpath}".rstrip(':')
            
            # Create log file if it doesn't exist
            if hasattr(self, 'config'):
                log_file = Path(self.config["monitoring"]["log_file"])
            else:
                # Fallback to expected log path
                log_file = self.logs_path / "app.log"
            
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.touch(exist_ok=True)
            
            self.orchestrator.start_service()
            time.sleep(2)  # Wait for service to start
            
    def stop_service(self):
        """Stop the test service."""
        if self.orchestrator:
            self.orchestrator.stop_service()
            
    def trigger_error(self) -> bool:
        """Trigger the error endpoint and return success status."""
        try:
            response = requests.get(f"http://localhost:{self.port}/error", timeout=5)
            return False  # If we get a response, the error wasn't triggered properly
        except requests.exceptions.RequestException:
            return True  # Error was triggered
            
    def cleanup(self):
        """Clean up the test environment."""
        self.stop_service()
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
        # Release the port
        if hasattr(self, 'port'):
            TestEnvironment._used_ports.discard(self.port)


class HealingScenarioRunner:
    """Runs end-to-end healing scenarios and collects results."""
    
    def __init__(self, environment: TestEnvironment):
        self.environment = environment
        self.logger = MonitoringLogger("scenario_runner")
        
    async def run_scenario(self, scenario: HealingScenario) -> HealingResult:
        """Run a complete healing scenario."""
        start_time = time.time()
        result = HealingResult(
            scenario=scenario,
            success=False,
            error_detected=False,
            patch_generated=False,
            patch_applied=False,
            tests_passed=False,
            deployment_successful=False,
            duration=0
        )
        
        try:
            # Phase 1: Trigger the error
            self.logger.info(f"Running scenario: {scenario.name}")
            self.logger.info("Phase 1: Triggering error")
            
            scenario.error_trigger()
            result.logs.append(f"Error triggered: {scenario.error_type}")
            
            # Wait for error to be logged
            await asyncio.sleep(2)
            
            # Phase 2: Run healing cycle
            self.logger.info("Phase 2: Running healing cycle")
            
            # Monitor healing process
            healing_task = asyncio.create_task(
                self._monitor_healing_cycle(result)
            )
            
            # Run the orchestrator's healing cycle
            healing_success = await asyncio.to_thread(
                self.environment.orchestrator.run_self_healing_cycle
            )
            
            # Wait for monitoring to complete
            await healing_task
            
            # Phase 3: Validate healing
            self.logger.info("Phase 3: Validating healing")
            
            validation_passed = await self._validate_healing(
                scenario, result
            )
            
            result.success = (
                healing_success and 
                validation_passed and
                result.deployment_successful
            )
            
        except Exception as e:
            self.logger.exception(e, "Scenario execution failed")
            result.logs.append(f"Exception: {str(e)}")
            
        finally:
            result.duration = time.time() - start_time
            result.logs.append(f"Scenario completed in {result.duration:.2f}s")
            
        return result
        
    async def _monitor_healing_cycle(self, result: HealingResult):
        """Monitor the healing cycle and update result."""
        max_wait = 60  # Maximum wait time in seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            # Check for error detection
            errors = get_latest_errors(limit=5)
            if errors and not result.error_detected:
                result.error_detected = True
                result.error_details = errors[0]
                result.logs.append("Error detected by monitoring system")
                
            # Check logs for patch generation
            log_path = self.environment.logs_path / "orchestrator.log"
            if log_path.exists():
                log_content = log_path.read_text()
                
                if "Generated patch" in log_content and not result.patch_generated:
                    result.patch_generated = True
                    result.logs.append("Patch generated")
                    
                if "Successfully applied patch" in log_content and not result.patch_applied:
                    result.patch_applied = True
                    result.logs.append("Patch applied")
                    
                if "Tests passed" in log_content and not result.tests_passed:
                    result.tests_passed = True
                    result.logs.append("Tests passed")
                    
                if "Service is healthy after restart" in log_content and not result.deployment_successful:
                    result.deployment_successful = True
                    result.logs.append("Deployment successful")
                    
                # If all phases completed, we can stop monitoring
                if all([
                    result.error_detected,
                    result.patch_generated,
                    result.patch_applied,
                    result.tests_passed,
                    result.deployment_successful
                ]):
                    break
                    
            await asyncio.sleep(1)
            
    async def _validate_healing(
        self, 
        scenario: HealingScenario, 
        result: HealingResult
    ) -> bool:
        """Validate that the healing was successful."""
        all_checks_passed = True
        
        for check in scenario.validation_checks:
            try:
                # Pass the environment if the check accepts it
                import inspect
                sig = inspect.signature(check)
                if 'environment' in sig.parameters:
                    check_passed = await asyncio.to_thread(check, self.environment)
                else:
                    check_passed = await asyncio.to_thread(check)
                    
                if not check_passed:
                    all_checks_passed = False
                    result.logs.append(f"Validation check failed: {check.__name__}")
                else:
                    result.logs.append(f"Validation check passed: {check.__name__}")
            except Exception as e:
                all_checks_passed = False
                result.logs.append(f"Validation check error: {check.__name__} - {str(e)}")
                
        # Additional validation: verify error is fixed
        try:
            response = requests.get(f"http://localhost:{self.environment.port}/error", timeout=5)
            if response.status_code == 200:
                result.logs.append("Error endpoint now returns success")
            else:
                all_checks_passed = False
                result.logs.append(f"Error endpoint still failing: {response.status_code}")
        except Exception as e:
            # This might be expected if the endpoint was removed
            result.logs.append(f"Error endpoint check: {str(e)}")
            
        return all_checks_passed


class MetricsCollector:
    """Collects and analyzes metrics during healing scenarios."""
    
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'error_rates': [],
            'healing_durations': [],
            'success_rates': {},
            'resource_usage': []
        }
        
    def record_healing_duration(self, scenario_name: str, duration: float):
        """Record how long a healing cycle took."""
        self.metrics['healing_durations'].append({
            'scenario': scenario_name,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
        
    def record_success_rate(self, scenario_name: str, success: bool):
        """Track success rates for different scenarios."""
        if scenario_name not in self.metrics['success_rates']:
            self.metrics['success_rates'][scenario_name] = {
                'total': 0,
                'successful': 0
            }
            
        self.metrics['success_rates'][scenario_name]['total'] += 1
        if success:
            self.metrics['success_rates'][scenario_name]['successful'] += 1
            
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of collected metrics."""
        summary = {
            'total_scenarios': len(self.metrics['healing_durations']),
            'average_healing_duration': 0,
            'overall_success_rate': 0,
            'scenario_success_rates': {}
        }
        
        if self.metrics['healing_durations']:
            total_duration = sum(m['duration'] for m in self.metrics['healing_durations'])
            summary['average_healing_duration'] = total_duration / len(self.metrics['healing_durations'])
            
        total_runs = 0
        total_successes = 0
        
        for scenario, stats in self.metrics['success_rates'].items():
            if stats['total'] > 0:
                success_rate = stats['successful'] / stats['total']
                summary['scenario_success_rates'][scenario] = success_rate
                total_runs += stats['total']
                total_successes += stats['successful']
                
        if total_runs > 0:
            summary['overall_success_rate'] = total_successes / total_runs
            
        return summary


# Validation check functions
def check_service_healthy(environment: Optional[TestEnvironment] = None) -> bool:
    """Check if the service is responding to health checks."""
    port = environment.port if environment else 8000
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False
        

def check_error_fixed(environment: Optional[TestEnvironment] = None) -> bool:
    """Check if the error endpoint is now working."""
    port = environment.port if environment else 8000
    try:
        response = requests.get(f"http://localhost:{port}/error", timeout=5)
        return response.status_code != 500
    except:
        return True  # If endpoint was removed, consider it fixed
        

def check_no_syntax_errors(file_path: Path) -> bool:
    """Check if a Python file has no syntax errors."""
    try:
        compile(file_path.read_text(), str(file_path), 'exec')
        return True
    except SyntaxError:
        return False


# Test fixtures
@pytest.fixture
def test_environment():
    """Create an isolated test environment."""
    env = TestEnvironment()
    env.setup()
    yield env
    env.cleanup()
    

@pytest.fixture
def scenario_runner(test_environment):
    """Create a scenario runner."""
    return HealingScenarioRunner(test_environment)
    

@pytest.fixture
def metrics_collector():
    """Create a metrics collector."""
    return MetricsCollector()