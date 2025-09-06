"""
Cross-language integration test scenarios for Homeostasis.

This module tests the interaction between different language plugins,
error handling across language boundaries, and polyglot application support.
"""

import os
import sys

import pytest

# Add the modules directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.analysis.cross_language_orchestrator import CrossLanguageOrchestrator
from modules.analysis.language_plugin_system import LanguagePluginSystem


class TestCrossLanguageOrchestration:
    """Test cases for cross-language orchestration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = CrossLanguageOrchestrator()
        self.plugin_system = LanguagePluginSystem()

    def test_python_calling_c_extension_error(self):
        """Test Python calling C extension with segfault."""
        error_data = {
            "id": "cross-lang-001",
            "primary_language": "python",
            "error_chain": [
                {
                    "language": "python",
                    "error_type": "SystemError",
                    "message": "error return without exception set",
                    "file": "wrapper.py",
                    "line": 45,
                    "function": "call_native_function",
                },
                {
                    "language": "cpp",  # C extension
                    "error_type": "SegmentationFault",
                    "signal": "SIGSEGV",
                    "address": "0x0",
                    "file": "native_module.c",
                    "line": 120,
                    "function": "process_data",
                },
            ],
        }

        analysis = self.orchestrator.analyze_cross_language_error(error_data)

        assert analysis["root_cause_language"] == "cpp"
        assert analysis["propagation_path"] == ["cpp", "python"]
        assert "null pointer" in analysis["root_cause"].lower()
        assert len(analysis["fixes"]) == 2  # Fix for C and Python parts

    def test_javascript_calling_rust_wasm_error(self):
        """Test JavaScript calling Rust WASM with panic."""
        error_data = {
            "id": "cross-lang-002",
            "primary_language": "javascript",
            "error_chain": [
                {
                    "language": "javascript",
                    "error_type": "RuntimeError",
                    "message": "unreachable executed",
                    "file": "app.js",
                    "line": 200,
                    "context": "WebAssembly",
                },
                {
                    "language": "rust",
                    "error_type": "Panic",
                    "message": "index out of bounds: the len is 5 but the index is 10",
                    "file": "lib.rs",
                    "line": 50,
                    "function": "process_array",
                },
            ],
        }

        analysis = self.orchestrator.analyze_cross_language_error(error_data)

        assert analysis["root_cause_language"] == "rust"
        assert "bounds check" in analysis["fixes"]["rust"]["suggestion"]
        assert "error handling" in analysis["fixes"]["javascript"]["suggestion"].lower()

    def test_java_jni_cpp_error(self):
        """Test Java JNI calling C++ with exception."""
        error_data = {
            "id": "cross-lang-003",
            "primary_language": "java",
            "error_chain": [
                {
                    "language": "java",
                    "error_type": "java.lang.UnsatisfiedLinkError",
                    "message": "Native method threw exception",
                    "file": "NativeWrapper.java",
                    "line": 30,
                    "method": "nativeProcess",
                },
                {
                    "language": "cpp",
                    "error_type": "std::bad_alloc",
                    "message": "memory allocation failed",
                    "file": "jni_impl.cpp",
                    "line": 75,
                    "function": "Java_NativeWrapper_nativeProcess",
                },
            ],
        }

        analysis = self.orchestrator.analyze_cross_language_error(error_data)

        assert analysis["root_cause_language"] == "cpp"
        assert "memory" in analysis["root_cause"]
        assert "try-catch" in analysis["fixes"]["java"]["suggestion"]
        assert "exception handling" in analysis["fixes"]["cpp"]["suggestion"]

    def test_go_cgo_c_error(self):
        """Test Go CGO calling C with error."""
        error_data = {
            "id": "cross-lang-004",
            "primary_language": "go",
            "error_chain": [
                {
                    "language": "go",
                    "error_type": "CGOError",
                    "message": "unexpected fault address",
                    "file": "wrapper.go",
                    "line": 20,
                },
                {
                    "language": "cpp",  # C code
                    "error_type": "BufferOverflow",
                    "message": "buffer overflow detected",
                    "file": "utils.c",
                    "line": 40,
                    "function": "process_buffer",
                },
            ],
        }

        analysis = self.orchestrator.analyze_cross_language_error(error_data)

        assert analysis["root_cause_language"] == "cpp"
        assert "buffer overflow" in analysis["root_cause"]
        assert "bounds check" in analysis["fixes"]["cpp"]["suggestion"]


class TestPolyglotApplicationErrors:
    """Test cases for polyglot application error scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = CrossLanguageOrchestrator()

    def test_microservice_communication_error(self):
        """Test error propagation across microservices in different languages."""
        error_data = {
            "id": "polyglot-001",
            "error_type": "DistributedError",
            "services": [
                {
                    "name": "api-gateway",
                    "language": "javascript",
                    "error": {
                        "type": "HTTPError",
                        "status": 500,
                        "message": "Internal server error from user-service",
                    },
                },
                {
                    "name": "user-service",
                    "language": "java",
                    "error": {
                        "type": "DatabaseException",
                        "message": "Connection pool exhausted",
                    },
                },
                {
                    "name": "database-proxy",
                    "language": "go",
                    "error": {
                        "type": "ConnectionError",
                        "message": "too many connections",
                    },
                },
            ],
        }

        analysis = self.orchestrator.analyze_distributed_error(error_data)

        assert analysis["root_cause_service"] == "database-proxy"
        assert len(analysis["service_fixes"]) == 3
        assert (
            "connection pool" in analysis["service_fixes"]["user-service"]["suggestion"]
        )

    def test_data_serialization_mismatch(self):
        """Test data serialization errors between languages."""
        error_data = {
            "id": "polyglot-002",
            "error_type": "SerializationError",
            "producer": {
                "language": "python",
                "format": "msgpack",
                "error": {
                    "type": "EncodingError",
                    "message": "datetime object is not serializable",
                },
            },
            "consumer": {
                "language": "rust",
                "format": "msgpack",
                "error": {"type": "DecodingError", "message": "invalid msgpack format"},
            },
        }

        analysis = self.orchestrator.analyze_serialization_error(error_data)

        assert "datetime" in analysis["root_cause"]
        assert (
            "ISO format" in analysis["producer_fix"]["suggestion"]
            or "timestamp" in analysis["producer_fix"]["suggestion"]
        )

    def test_shared_memory_concurrency_error(self):
        """Test shared memory concurrency errors between processes."""
        error_data = {
            "id": "polyglot-003",
            "error_type": "SharedMemoryError",
            "processes": [
                {
                    "pid": 1234,
                    "language": "cpp",
                    "error": {
                        "type": "SegmentationFault",
                        "address": "0x7fff00001000",
                        "operation": "write",
                    },
                },
                {
                    "pid": 5678,
                    "language": "rust",
                    "error": {
                        "type": "AccessViolation",
                        "address": "0x7fff00001000",
                        "operation": "read",
                    },
                },
            ],
            "shared_memory": {"segment": "/dev/shm/app_shared", "size": 4096},
        }

        analysis = self.orchestrator.analyze_shared_memory_error(error_data)

        assert "synchronization" in analysis["root_cause"]
        assert (
            "mutex" in analysis["fixes"]["cpp"]["suggestion"]
            or "semaphore" in analysis["fixes"]["cpp"]["suggestion"]
        )


class TestLanguageInteroperability:
    """Test cases for language interoperability issues."""

    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = CrossLanguageOrchestrator()

    def test_ffi_type_mismatch(self):
        """Test FFI type mismatch between languages."""
        error_data = {
            "id": "interop-001",
            "error_type": "FFITypeMismatch",
            "caller": {
                "language": "python",
                "expected_type": "c_char_p",
                "passed_type": "str",
            },
            "callee": {
                "language": "cpp",
                "expected_type": "const char*",
                "received": "invalid pointer",
            },
        }

        analysis = self.orchestrator.analyze_ffi_error(error_data)

        assert "type conversion" in analysis["root_cause"]
        assert (
            "encode" in analysis["caller_fix"]["suggestion"]
            or "bytes" in analysis["caller_fix"]["suggestion"]
        )

    def test_grpc_schema_mismatch(self):
        """Test gRPC schema mismatch between services."""
        error_data = {
            "id": "interop-002",
            "error_type": "gRPCError",
            "client": {
                "language": "go",
                "proto_version": "3.1.0",
                "error": "unknown field 'user_id'",
            },
            "server": {
                "language": "java",
                "proto_version": "3.0.0",
                "expected_field": "userId",
            },
        }

        analysis = self.orchestrator.analyze_grpc_error(error_data)

        assert "schema" in analysis["root_cause"] or "version" in analysis["root_cause"]
        assert "proto" in analysis["fix"]["suggestion"]

    def test_rest_api_contract_violation(self):
        """Test REST API contract violation between services."""
        error_data = {
            "id": "interop-003",
            "error_type": "APIContractViolation",
            "endpoint": "/api/users",
            "client": {
                "language": "typescript",
                "expected_response": {
                    "id": "number",
                    "name": "string",
                    "email": "string",
                },
            },
            "server": {
                "language": "ruby",
                "actual_response": {
                    "id": "string",  # Changed from number
                    "full_name": "string",  # Changed from name
                    "email": "string",
                },
            },
        }

        analysis = self.orchestrator.analyze_api_contract_error(error_data)

        assert (
            "contract" in analysis["root_cause"] or "schema" in analysis["root_cause"]
        )
        assert len(analysis["mismatches"]) == 2


class TestCrossLanguageDebugging:
    """Test cases for cross-language debugging scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = CrossLanguageOrchestrator()

    def test_stack_trace_correlation(self):
        """Test correlating stack traces across language boundaries."""
        error_data = {
            "id": "debug-001",
            "combined_stack_trace": [
                {
                    "language": "python",
                    "frames": [
                        {"file": "app.py", "line": 100, "function": "handle_request"},
                        {"file": "service.py", "line": 50, "function": "process_data"},
                    ],
                },
                {
                    "language": "cpp",
                    "frames": [
                        {
                            "file": "native.cpp",
                            "line": 200,
                            "function": "transform_data",
                        },
                        {
                            "file": "algorithm.cpp",
                            "line": 75,
                            "function": "apply_transform",
                        },
                    ],
                },
                {
                    "language": "python",
                    "frames": [
                        {"file": "wrapper.py", "line": 25, "function": "call_native"}
                    ],
                },
            ],
        }

        correlation = self.orchestrator.correlate_stack_traces(error_data)

        assert correlation["entry_point"]["language"] == "python"
        assert correlation["error_origin"]["language"] == "cpp"
        assert len(correlation["call_path"]) == 5

    def test_distributed_tracing_error(self):
        """Test distributed tracing across services."""
        error_data = {
            "id": "debug-002",
            "trace_id": "abc123",
            "spans": [
                {
                    "id": "span1",
                    "service": "frontend",
                    "language": "javascript",
                    "status": "ok",
                    "duration_ms": 50,
                },
                {
                    "id": "span2",
                    "service": "api",
                    "language": "go",
                    "status": "error",
                    "duration_ms": 500,
                    "error": {
                        "type": "timeout",
                        "message": "context deadline exceeded",
                    },
                },
                {
                    "id": "span3",
                    "service": "database",
                    "language": "rust",
                    "status": "ok",
                    "duration_ms": 450,
                },
            ],
        }

        analysis = self.orchestrator.analyze_distributed_trace(error_data)

        assert analysis["bottleneck_service"] == "database"
        assert analysis["error_service"] == "api"
        assert "timeout" in analysis["root_cause"]


class TestCrossLanguagePerformance:
    """Test cases for cross-language performance issues."""

    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = CrossLanguageOrchestrator()

    def test_serialization_performance_bottleneck(self):
        """Test serialization performance issues between languages."""
        error_data = {
            "id": "perf-001",
            "issue_type": "performance",
            "metric": "latency",
            "pipeline": [
                {
                    "stage": "python_serialize",
                    "language": "python",
                    "format": "json",
                    "duration_ms": 500,
                    "data_size_mb": 10,
                },
                {"stage": "network_transfer", "duration_ms": 100},
                {
                    "stage": "java_deserialize",
                    "language": "java",
                    "format": "json",
                    "duration_ms": 800,
                    "data_size_mb": 10,
                },
            ],
        }

        analysis = self.orchestrator.analyze_performance_issue(error_data)

        assert analysis["bottleneck"] == "java_deserialize"
        assert (
            "binary format" in analysis["suggestion"]
            or "protobuf" in analysis["suggestion"]
        )

    def test_memory_leak_across_ffi(self):
        """Test memory leak in FFI boundary."""
        error_data = {
            "id": "perf-002",
            "issue_type": "memory_leak",
            "languages": ["python", "rust"],
            "observations": [
                {"time": "00:00", "python_heap_mb": 100, "rust_heap_mb": 50},
                {"time": "01:00", "python_heap_mb": 500, "rust_heap_mb": 100},
                {"time": "02:00", "python_heap_mb": 900, "rust_heap_mb": 150},
            ],
            "ffi_calls_per_hour": 10000,
        }

        analysis = self.orchestrator.analyze_memory_leak(error_data)

        assert analysis["leak_location"] == "python"
        assert "reference" in analysis["root_cause"] or "gc" in analysis["root_cause"]
        assert "cleanup" in analysis["fix"]["suggestion"]


class TestCrossLanguageSecurityIssues:
    """Test cases for security issues across language boundaries."""

    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = CrossLanguageOrchestrator()

    def test_injection_through_language_boundary(self):
        """Test injection vulnerability through language boundary."""
        error_data = {
            "id": "sec-001",
            "vulnerability_type": "injection",
            "flow": [
                {
                    "language": "javascript",
                    "component": "frontend",
                    "input": "'; DROP TABLE users; --",
                    "validation": False,
                },
                {
                    "language": "python",
                    "component": "api",
                    "processing": "pass-through",
                    "sanitization": False,
                },
                {
                    "language": "java",
                    "component": "database-service",
                    "query_construction": "string concatenation",
                    "vulnerable": True,
                },
            ],
        }

        analysis = self.orchestrator.analyze_security_issue(error_data)

        assert analysis["vulnerability_confirmed"] is True
        assert len(analysis["fixes"]) == 3
        assert "validation" in analysis["fixes"]["javascript"]["action"]
        assert "sanitization" in analysis["fixes"]["python"]["action"]
        assert "prepared statement" in analysis["fixes"]["java"]["action"]

    def test_buffer_overflow_propagation(self):
        """Test buffer overflow propagation through FFI."""
        error_data = {
            "id": "sec-002",
            "vulnerability_type": "buffer_overflow",
            "source": {
                "language": "go",
                "function": "processUserInput",
                "buffer_size": 256,
                "input_size": 1024,
            },
            "propagation": [
                {
                    "language": "cpp",
                    "function": "native_process",
                    "receives": "unsafe pointer",
                    "bounds_check": False,
                }
            ],
        }

        analysis = self.orchestrator.analyze_security_issue(error_data)

        assert analysis["severity"] == "critical"
        assert "bounds check" in analysis["fixes"]["go"]["action"]
        assert "validation" in analysis["fixes"]["cpp"]["action"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
