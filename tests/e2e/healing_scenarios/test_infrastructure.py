"""
Test infrastructure for simulating service environments and healing scenarios.

This module provides mock implementations of services, log generation, and patch
validation to enable testing without requiring real service infrastructure.
"""

import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MockService:
    """Represents a mock service with controllable behavior."""

    name: str
    port: int
    status: str = "stopped"
    health_status: bool = True
    injected_errors: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error_count: int = 0
    request_count: int = 0

    def start(self):
        """Start the mock service."""
        self.status = "running"
        self.log_event("INFO", f"Service {self.name} started on port {self.port}")

    def stop(self):
        """Stop the mock service."""
        self.status = "stopped"
        self.log_event("INFO", f"Service {self.name} stopped")

    def inject_error(
        self,
        error_type: str,
        message: str,
        file_path: str = "app.py",
        line_number: int = 50,
        stack_trace: str = None,
    ):
        """Inject an error into the service."""
        error = {
            "type": error_type,
            "message": message,
            "file_path": file_path,
            "line_number": line_number,
            "stack_trace": stack_trace or f"Traceback at {file_path}:{line_number}",
        }
        self.injected_errors.append(error)
        self.health_status = False

        # Log the error
        self.log_event(
            "ERROR",
            message,
            {
                "exception_type": error_type,
                "file_path": file_path,
                "line_number": line_number,
                "stack_trace": stack_trace,
            },
        )

    def log_event(self, level: str, message: str, extra: Dict[str, Any] = None):
        """Log an event for the service."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.name,
            "level": level,
            "message": message,
            "file_path": extra.get("file_path", "app.py") if extra else "app.py",
            "line_number": extra.get("line_number", 1) if extra else 1,
        }

        # For ERROR level logs, properly structure error details
        if level == "ERROR" and extra:
            log_entry["error_details"] = {
                "exception_type": extra.get("exception_type", "Unknown"),
                "message": message,
                "file_path": extra.get("file_path", "app.py"),
                "line_number": extra.get("line_number", 1),
            }
            if "stack_trace" in extra:
                log_entry["stack_trace"] = extra["stack_trace"]
            # Don't duplicate these fields at the top level
            log_entry["exception_type"] = extra.get("exception_type", "Unknown")
        elif extra:
            log_entry.update(extra)

        self.logs.append(log_entry)

    def get_recent_logs(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent logs from the service."""
        return self.logs[-count:]

    def clear_errors(self):
        """Clear injected errors and restore health."""
        self.injected_errors.clear()
        self.health_status = True
        self.log_event("INFO", "Errors cleared, service healthy")


class MockServiceEnvironment:
    """Manages a mock service environment for testing."""

    def __init__(self):
        self.services: Dict[str, MockService] = {}
        self.deployed_patches: List[Dict[str, Any]] = []
        self.test_results: Dict[str, bool] = {}
        self.base_path = Path(tempfile.mkdtemp(prefix="mock_healing_test_"))
        self.logs_path = self.base_path / "logs"
        self.logs_path.mkdir(parents=True, exist_ok=True)
        self._port_counter = 8000

    def create_service(self, name: str, port: Optional[int] = None) -> MockService:
        """Create a new mock service."""
        if port is None:
            port = self._port_counter
            self._port_counter += 1

        service = MockService(name=name, port=port)
        self.services[name] = service
        return service

    def start_service(self, name: str):
        """Start a mock service."""
        if name in self.services:
            self.services[name].start()

    def stop_service(self, name: str):
        """Stop a mock service."""
        if name in self.services:
            self.services[name].stop()

    def inject_error(
        self,
        service: str,
        error_type: str,
        message: str = None,
        file_path: str = "app.py",
        line_number: int = 50,
    ):
        """Inject an error into a service."""
        if service not in self.services:
            raise ValueError(f"Service {service} not found")

        # Generate realistic error messages
        if message is None:
            service_obj = self.services[service]
            language = getattr(service_obj, "language", "python")
            message = self._generate_error_message(error_type, language)

        self.services[service].inject_error(
            error_type=error_type,
            message=message,
            file_path=file_path,
            line_number=line_number,
        )

        # Track injected errors for monitoring tests
        if not hasattr(self, "_injected_errors"):
            self._injected_errors = []
        self._injected_errors.append(
            {
                "service": service,
                "error_type": error_type,
                "message": message,
                "file_path": file_path,
                "line_number": line_number,
            }
        )

    def _generate_error_message(self, error_type: str, language: str = "python") -> str:
        """Generate a realistic error message for the given error type."""
        if language == "python":
            error_messages = {
                "KeyError": "KeyError: 'missing_key'",
                "AttributeError": "AttributeError: 'NoneType' object has no attribute 'data'",
                "TypeError": "TypeError: object dict can't be used in 'await' expression",
                "NameError": "NameError: name 'user_id' is not defined",
                "ValueError": "ValueError: invalid literal for int() with base 10: 'abc'",
                "IndexError": "IndexError: list index out of range",
                "ZeroDivisionError": "ZeroDivisionError: division by zero",
                "ImportError": "ImportError: No module named 'missing_module'",
                "RuntimeError": "RuntimeError: maximum recursion depth exceeded",
            }
        elif language == "javascript":
            error_messages = {
                "TypeError": "TypeError: Cannot read property 'data' of undefined",
                "ReferenceError": "ReferenceError: user_id is not defined",
                "SyntaxError": "SyntaxError: Unexpected token '}' in JSON at position 42",
                "RangeError": "RangeError: Maximum call stack size exceeded",
            }
        elif language == "go":
            error_messages = {
                "panic": "panic: runtime error: invalid memory address or nil pointer dereference",
                "error": "error: undefined: someVariable",
                "compile": "compile error: cannot use x (type int) as type string",
            }
        elif language == "java":
            error_messages = {
                "NullPointerException": "java.lang.NullPointerException: Cannot invoke method on null object",
                "ArrayIndexOutOfBoundsException": "java.lang.ArrayIndexOutOfBoundsException: Index 10 out of bounds for length 5",
                "ClassCastException": "java.lang.ClassCastException: java.lang.String cannot be cast to java.lang.Integer",
            }
        else:
            return f"{error_type}: Generic error message for {language}"

        return error_messages.get(error_type, f"{error_type}: Generic error message")

    def create_cross_language_service(
        self, name: str, language: str, port: int
    ) -> MockService:
        """Create a mock service for cross-language testing."""
        service = MockService(name=name, port=port)
        # Add language-specific metadata
        service.language = language
        service.file_extension = {
            "javascript": ".js",
            "go": ".go",
            "java": ".java",
            "python": ".py",
        }.get(language, ".txt")
        self.services[name] = service
        return service

    def stop_default_service(self):
        """Stop the default test_service."""
        if "test_service" in self.services:
            self.stop_service("test_service")

    def start_default_service(self):
        """Start the default test_service."""
        if "test_service" in self.services:
            self.start_service("test_service")

    def trigger_error(self):
        """Trigger an error in the default test_service."""
        # In mock mode, errors are already injected via inject_error
        pass

    def get_logs(self, service: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get logs from services."""
        logs = []

        if service:
            if service in self.services:
                logs.extend(self.services[service].logs)
        else:
            # Aggregate logs from all services
            for svc in self.services.values():
                logs.extend(svc.logs)

        # Sort by timestamp
        logs.sort(key=lambda x: x.get("timestamp", ""))
        return logs

    def get_error_logs(self) -> List[Dict[str, Any]]:
        """Get only error logs."""
        all_logs = self.get_logs()
        error_logs = [log for log in all_logs if log.get("level") == "ERROR"]

        # For monitoring tests, ensure we have proper error logs
        if hasattr(self, "_injected_errors"):
            # Add any injected errors that might not have logs yet
            for injected in self._injected_errors:
                # Check if this error already has a log by matching service, message, and file_path
                has_log = any(
                    log.get("service") == injected["service"]
                    and log.get("message") == injected["message"]
                    and log.get("file_path") == injected["file_path"]
                    for log in error_logs
                )
                if not has_log:
                    # Create a log entry for this injected error
                    error_logs.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "level": "ERROR",
                            "message": injected["message"],
                            "error_details": {
                                "exception_type": injected["error_type"],
                                "message": injected["message"],
                                "file_path": injected["file_path"],
                                "line_number": injected["line_number"],
                            },
                            "service": injected["service"],
                        }
                    )

        return error_logs

    def deploy_patch(self, patch: Dict[str, Any]) -> bool:
        """Deploy a patch to the mock environment."""
        # Simulate patch deployment
        self.deployed_patches.append(patch)

        # Check if patch addresses the errors
        success = self._validate_patch(patch)

        if success:
            # Clear errors from affected service
            for service in self.services.values():
                for error in service.injected_errors:
                    if error.get("file_path") == patch.get("file_path"):
                        service.clear_errors()
                        break

        return success

    def _validate_patch(self, patch: Dict[str, Any]) -> bool:
        """Validate if a patch would fix the errors."""
        # Simple validation based on patch type
        if "old_code" in patch and "new_code" in patch:
            # Check if the patch addresses common error patterns
            new_code = patch.get("new_code", "")

            # KeyError fixes
            if "get(" in new_code or "if" in new_code and "in" in new_code:
                return True

            # AttributeError fixes
            if "if" in new_code and "is not None" in new_code:
                return True

            # TypeError fixes
            old_code = patch.get("old_code", "")
            if "await" in old_code and "await" not in new_code:
                return True

            # NameError fixes
            if "def" in new_code and "(" in new_code:
                return True

            # ConnectionError fixes
            if "try:" in new_code and "ConnectionError" in new_code:
                return True

            # TimeoutError fixes
            if "timeout=" in new_code or "retry" in new_code:
                return True

            # MemoryError fixes
            if "chunk" in new_code or "batch" in new_code:
                return True

            # RuntimeError fixes
            if "list(" in new_code or "copy(" in new_code:
                return True

            # ValidationError fixes
            if "if" in new_code and ("in" in new_code or "ValidationError" in new_code):
                return True

            # ServiceUnavailableError fixes
            if "is_healthy()" in new_code or "fallback" in new_code:
                return True

        return False

    def run_tests(self, test_name: str) -> bool:
        """Run tests for a deployed patch."""
        # Simulate test execution
        time.sleep(0.1)  # Simulate test runtime

        # Check if services are healthy
        all_healthy = all(svc.health_status for svc in self.services.values())

        self.test_results[test_name] = all_healthy
        return all_healthy

    def cleanup(self):
        """Clean up the mock environment."""
        import shutil

        # Stop all services first
        for service in self.services.values():
            if service.status == "running":
                service.stop()
        self.services.clear()
        # Clean up temporary directories
        if self.base_path.exists():
            shutil.rmtree(self.base_path)

    def cleanup_services(self):
        """Alias for cleanup for compatibility."""
        self.cleanup()


class LogSimulator:
    """Simulates realistic log generation for testing."""

    @staticmethod
    def generate_error_log(
        error_type: str,
        service: str = "test_service",
        file_path: str = "app.py",
        line_number: int = 50,
        custom_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a realistic error log entry."""
        error_patterns = {
            "KeyError": {
                "message": custom_message or "KeyError: 'user_data'",
                "stack_trace": f"""Traceback (most recent call last):
  File "{file_path}", line {line_number}, in process_request
    user_info = data['user_data']
KeyError: 'user_data'""",
            },
            "AttributeError": {
                "message": custom_message
                or "AttributeError: 'NoneType' object has no attribute 'items'",
                "stack_trace": f"""Traceback (most recent call last):
  File "{file_path}", line {line_number}, in handle_response
    for key, value in response.items():
AttributeError: 'NoneType' object has no attribute 'items'""",
            },
            "TypeError": {
                "message": custom_message
                or "TypeError: object dict can't be used in 'await' expression",
                "stack_trace": f"""Traceback (most recent call last):
  File "{file_path}", line {line_number}, in async_handler
    result = await db.execute(query)
TypeError: object dict can't be used in 'await' expression""",
            },
            "NameError": {
                "message": custom_message or "NameError: name 'user_id' is not defined",
                "stack_trace": f"""Traceback (most recent call last):
  File "{file_path}", line {line_number}, in get_user
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
NameError: name 'user_id' is not defined""",
            },
        }

        pattern = error_patterns.get(
            error_type,
            {
                "message": f"{error_type}: Generic error",
                "stack_trace": f"Traceback at {file_path}:{line_number}",
            },
        )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "service": service,
            "level": "ERROR",
            "message": pattern["message"],
            "exception_type": error_type,
            "file_path": file_path,
            "line_number": line_number,
            "stack_trace": pattern["stack_trace"],
            "function_name": "unknown_function",
        }

    @staticmethod
    def generate_log_sequence(scenario: str) -> List[Dict[str, Any]]:
        """Generate a sequence of logs for a specific scenario."""
        sequences = {
            "cascading_failure": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "service": "service1",
                    "level": "ERROR",
                    "message": "KeyError: 'service1_data'",
                    "exception_type": "KeyError",
                    "file_path": "service1.py",
                    "line_number": 100,
                },
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "service": "service2",
                    "level": "ERROR",
                    "message": "ConnectionError: Failed to connect to service1",
                    "exception_type": "ConnectionError",
                    "file_path": "service2.py",
                    "line_number": 50,
                },
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "service": "service3",
                    "level": "ERROR",
                    "message": "TimeoutError: Request to service2 timed out",
                    "exception_type": "TimeoutError",
                    "file_path": "service3.py",
                    "line_number": 75,
                },
            ],
            "concurrent_errors": [
                LogSimulator.generate_error_log("KeyError", "service1"),
                LogSimulator.generate_error_log("AttributeError", "service2"),
                LogSimulator.generate_error_log("TypeError", "service3"),
            ],
        }

        return sequences.get(scenario, [])


class PatchValidator:
    """Validates generated patches for correctness."""

    @staticmethod
    def validate_syntax(
        code: str, language: str = "python"
    ) -> Tuple[bool, Optional[str]]:
        """Validate syntax of generated code."""
        if language == "python":
            try:
                compile(code, "<string>", "exec")
                return True, None
            except SyntaxError as e:
                return False, str(e)

        # Add more language validators as needed
        return True, None

    @staticmethod
    def validate_patch_structure(patch: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate patch has required structure."""
        errors = []
        required_fields = ["file_path", "description"]

        for field in required_fields:
            if field not in patch:
                errors.append(f"Missing required field: {field}")

        # Check for either old_code/new_code or patches array
        if "old_code" not in patch and "patches" not in patch:
            errors.append("Patch must have either old_code/new_code or patches array")

        if "old_code" in patch and "new_code" not in patch:
            errors.append("Patch with old_code must also have new_code")

        return len(errors) == 0, errors

    @staticmethod
    def validate_fix_addresses_error(patch: Dict[str, Any], error_type: str) -> bool:
        """Check if patch addresses the specific error type."""
        new_code = patch.get("new_code", "")

        validators = {
            "KeyError": lambda code: any(
                pattern in code
                for pattern in [".get(", "if ", "in ", "try:", "except KeyError"]
            ),
            "AttributeError": lambda code: any(
                pattern in code
                for pattern in [
                    "is not None",
                    "hasattr(",
                    "getattr(",
                    "try:",
                    "except AttributeError",
                ]
            ),
            "TypeError": lambda code: any(
                pattern in code
                for pattern in [
                    "isinstance(",
                    "type(",
                    "try:",
                    "except TypeError",
                    "await",
                ]
            )
            or "await" not in code,  # Fix for async/await type errors
            "NameError": lambda code: any(
                pattern in code
                for pattern in ["def ", "global ", "nonlocal ", "import ", "from "]
            ),
        }

        validator = validators.get(error_type)
        if validator:
            return validator(new_code)

        return True  # Default to valid for unknown error types


# Mock implementations for testing without real infrastructure
class MockOrchestrator:
    """Mock orchestrator for testing."""

    def __init__(self, config_path: Path, environment: MockServiceEnvironment):
        self.config_path = config_path
        self.environment = environment
        self.errors_detected = []
        self.patches_generated = []
        self.tests_run = []

    def monitor_for_errors(self) -> List[Dict[str, Any]]:
        """Monitor for errors in the mock environment."""
        return self.environment.get_error_logs()

    def analyze_errors(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze errors and return analysis results."""
        results = []
        for error in errors:
            results.append(
                {
                    "error_id": f"err_{len(results)}",
                    "error_type": error.get("exception_type", "Unknown"),
                    "root_cause": self._determine_root_cause(error),
                    "confidence": 0.9,
                    "file_path": error.get("file_path", "app.py"),
                    "line_number": error.get("line_number", 1),
                }
            )
        return results

    def _determine_root_cause(self, error: Dict[str, Any]) -> str:
        """Determine root cause based on error type."""
        error_type = error.get("exception_type", "")
        root_causes = {
            "KeyError": "dict_key_not_exists",
            "AttributeError": "null_reference",
            "TypeError": "type_mismatch",
            "NameError": "undefined_variable",
            "ConnectionError": "network_failure",
            "TimeoutError": "timeout_exceeded",
            "MemoryError": "memory_exhaustion",
            "RuntimeError": "runtime_issue",
            "ValidationError": "validation_failure",
            "ServiceUnavailableError": "service_unavailable",
        }
        return root_causes.get(error_type, "unknown")

    def generate_patches(
        self, analysis_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate patches based on analysis."""
        patches = []
        for result in analysis_results:
            patch = self._generate_patch_for_error(result)
            if patch:
                patches.append(patch)
                self.patches_generated.append(patch)
        return patches

    def _generate_patch_for_error(
        self, analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate a patch for a specific error."""
        error_type = analysis.get("error_type")
        file_path = analysis.get("file_path", "app.py")

        patch_templates = {
            "KeyError": {
                "old_code": "data = cache[key]",
                "new_code": "data = cache.get(key, {})",
                "description": "Use .get() to safely access dictionary",
            },
            "AttributeError": {
                "old_code": "result = obj.method()",
                "new_code": "result = obj.method() if obj is not None else None",
                "description": "Add None check before attribute access",
            },
            "TypeError": {
                "old_code": "result = await sync_function()",
                "new_code": "result = sync_function()",
                "description": "Remove await from non-async function",
            },
            "NameError": {
                "old_code": "def process():",
                "new_code": "def process(user_id=None):",
                "description": "Add missing parameter",
            },
            "ConnectionError": {
                "old_code": "response = connect_to_service()",
                "new_code": "try:\n    response = connect_to_service()\nexcept ConnectionError:\n    response = None",
                "description": "Add connection error handling",
            },
            "TimeoutError": {
                "old_code": "result = make_request(timeout=1)",
                "new_code": "result = make_request(timeout=5, retry=3)",
                "description": "Increase timeout and add retry",
            },
            "MemoryError": {
                "old_code": "results.append(transform_data(item))",
                "new_code": "for chunk in chunks(items, 1000):\n    results.extend(transform_data(chunk))",
                "description": "Process data in chunks to reduce memory usage",
            },
            "RuntimeError": {
                "old_code": "for key in data:",
                "new_code": "for key in list(data.keys()):",
                "description": "Create copy to avoid modification during iteration",
            },
            "ValidationError": {
                "old_code": "process_request(data)",
                "new_code": "if 'email' in data:\n    process_request(data)\nelse:\n    raise ValidationError('email required')",
                "description": "Add field validation",
            },
            "ServiceUnavailableError": {
                "old_code": "return service.call()",
                "new_code": "if service.is_healthy():\n    return service.call()\nelse:\n    return fallback_response()",
                "description": "Add service health check and fallback",
            },
        }

        template = patch_templates.get(error_type)
        if template:
            return {
                "file_path": file_path,
                "old_code": template["old_code"],
                "new_code": template["new_code"],
                "description": template["description"],
            }
        return None

    def test_patches(self, patches: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Test patches in the environment."""
        results = {}
        for i, patch in enumerate(patches):
            test_name = f"test_patch_{i}"
            # Deploy patch to environment
            deployed = self.environment.deploy_patch(patch)
            if deployed:
                # Run tests
                passed = self.environment.run_tests(test_name)
                results[test_name] = passed
                self.tests_run.append((test_name, passed))
            else:
                results[test_name] = False
        return results

    def deploy_patches(self, patches: List[Dict[str, Any]]) -> bool:
        """Deploy patches to production."""
        # In mock mode, just mark as deployed
        for patch in patches:
            self.environment.deploy_patch(patch)
        return True
