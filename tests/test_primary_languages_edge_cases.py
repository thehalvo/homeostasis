"""
Edge case and corner case test coverage for primary languages.

This module contains comprehensive test cases for unusual, boundary, and
extreme scenarios that might occur in primary language error handling.
"""
import pytest
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Add the modules directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules', 'analysis'))

from language_plugin_system import LanguagePluginSystem
from comprehensive_error_detector import ComprehensiveErrorDetector


class TestPythonEdgeCases:
    """Edge cases for Python error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin_system = LanguagePluginSystem()
        self.plugin = self.plugin_system.get_plugin("python")
    
    def test_unicode_in_error_messages(self):
        """Test handling of unicode characters in error messages."""
        error_data = {
            "language": "python",
            "error_type": "UnicodeDecodeError",
            "message": "'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte 🐍",
            "file": "文件.py",  # Unicode filename
            "line": 42,
            "encoding": "utf-8"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert analysis is not None
        assert "encoding" in analysis["suggestion"].lower()
    
    def test_extremely_long_stack_trace(self):
        """Test handling of extremely deep stack traces."""
        error_data = {
            "language": "python",
            "error_type": "RecursionError",
            "message": "maximum recursion depth exceeded",
            "stack_trace": [
                {"function": f"recursive_func", "file": "deep.py", "line": i}
                for i in range(1000)  # 1000 levels deep
            ]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert analysis is not None
        assert analysis["root_cause"] == "python_recursion_error"
        assert "recursion limit" in analysis["suggestion"].lower()
    
    def test_circular_import_with_complex_path(self):
        """Test circular import detection with complex import paths."""
        error_data = {
            "language": "python",
            "error_type": "ImportError",
            "message": "cannot import name 'ClassA' from partially initialized module 'package.subpackage.module_a'",
            "import_chain": [
                "package.subpackage.module_a",
                "package.subpackage.module_b",
                "package.another.module_c",
                "package.subpackage.module_a"  # Circular
            ]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert "circular import" in analysis["root_cause"]
        assert "refactor" in analysis["suggestion"].lower()
    
    def test_metaclass_conflict_error(self):
        """Test metaclass conflict errors."""
        error_data = {
            "language": "python",
            "error_type": "TypeError",
            "message": "metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases",
            "file": "meta.py",
            "line": 30,
            "classes": ["ClassA", "ClassB"]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert "metaclass" in analysis["root_cause"]
    
    def test_generator_exhaustion_error(self):
        """Test generator exhaustion edge case."""
        error_data = {
            "language": "python",
            "error_type": "StopIteration",
            "message": "generator already exhausted",
            "file": "gen.py",
            "line": 50,
            "context": "inside async generator"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert analysis is not None
        assert "generator" in analysis["suggestion"].lower()


class TestJavaScriptEdgeCases:
    """Edge cases for JavaScript error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin_system = LanguagePluginSystem()
        self.plugin = self.plugin_system.get_plugin("javascript")
    
    def test_async_stack_trace_reconstruction(self):
        """Test async/await stack trace reconstruction."""
        error_data = {
            "language": "javascript",
            "error_type": "TypeError",
            "message": "Cannot read property 'data' of undefined",
            "async_stack_trace": [
                {"function": "async fetchData", "file": "api.js", "line": 100},
                {"function": "async processUser", "file": "user.js", "line": 50},
                {"function": "Promise.then", "file": "<anonymous>"},
                {"function": "async handleRequest", "file": "handler.js", "line": 25}
            ]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert analysis is not None
        assert "async" in str(analysis)
    
    def test_circular_reference_in_json(self):
        """Test circular reference in JSON serialization."""
        error_data = {
            "language": "javascript",
            "error_type": "TypeError",
            "message": "Converting circular structure to JSON",
            "file": "serializer.js",
            "line": 75,
            "object_path": "obj.parent.child.parent"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert "circular" in analysis["root_cause"]
        assert "JSON.stringify" in analysis["suggestion"]
    
    def test_maximum_call_stack_with_tail_recursion(self):
        """Test maximum call stack with tail recursion."""
        error_data = {
            "language": "javascript",
            "error_type": "RangeError",
            "message": "Maximum call stack size exceeded",
            "optimization": "tail-call-optimization-not-available",
            "browser": "Chrome",
            "stack_depth": 10473
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert "stack" in analysis["root_cause"]
        assert "iteration" in analysis["suggestion"].lower() or "loop" in analysis["suggestion"].lower()
    
    def test_proxy_trap_infinite_recursion(self):
        """Test Proxy trap causing infinite recursion."""
        error_data = {
            "language": "javascript",
            "error_type": "RangeError",
            "message": "Maximum call stack size exceeded",
            "context": "Proxy get trap",
            "file": "proxy.js",
            "line": 30
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert analysis is not None
        assert "proxy" in analysis["suggestion"].lower() or "trap" in analysis["suggestion"].lower()


class TestJavaEdgeCases:
    """Edge cases for Java error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin_system = LanguagePluginSystem()
        self.plugin = self.plugin_system.get_plugin("java")
    
    def test_classloader_deadlock(self):
        """Test classloader deadlock scenario."""
        error_data = {
            "language": "java",
            "error_type": "DeadlockException",
            "message": "Deadlock detected in classloading",
            "threads": [
                {
                    "name": "Thread-1",
                    "state": "BLOCKED",
                    "waiting_for": "ClassA",
                    "holding": "ClassB"
                },
                {
                    "name": "Thread-2",
                    "state": "BLOCKED",
                    "waiting_for": "ClassB",
                    "holding": "ClassA"
                }
            ]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert "deadlock" in analysis["root_cause"]
        assert analysis["severity"] == "critical"
    
    def test_annotation_processor_circular_dependency(self):
        """Test annotation processor circular dependency."""
        error_data = {
            "language": "java",
            "error_type": "CompilationError",
            "message": "Annotation processor 'com.example.MyProcessor' depends on itself",
            "phase": "annotation-processing",
            "round": 5
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert "circular" in analysis["root_cause"] or "annotation" in analysis["root_cause"]
    
    def test_jvm_segfault_in_jni(self):
        """Test JVM segfault in JNI code."""
        error_data = {
            "language": "java",
            "error_type": "JVMCrash",
            "message": "SIGSEGV (0xb) at pc=0x00007fff5fc01000",
            "problematic_frame": "C  [libsystem.dylib+0x1000]  memcpy+0x10",
            "jni_method": "Java_com_example_Native_processData"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert analysis["severity"] == "critical"
        assert "JNI" in analysis["suggestion"] or "native" in analysis["suggestion"].lower()
    
    def test_memory_leak_in_threadlocal(self):
        """Test memory leak in ThreadLocal usage."""
        error_data = {
            "language": "java",
            "error_type": "OutOfMemoryError",
            "message": "Java heap space",
            "heap_dump_analysis": {
                "suspect": "ThreadLocal",
                "retained_size": "2.5GB",
                "instance_count": 1000000
            }
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert "ThreadLocal" in analysis["suggestion"]
        assert "remove()" in analysis["suggestion"] or "clean" in analysis["suggestion"].lower()


class TestCppEdgeCases:
    """Edge cases for C++ error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin_system = LanguagePluginSystem()
        self.plugin = self.plugin_system.get_plugin("cpp")
    
    def test_template_instantiation_recursion_limit(self):
        """Test template instantiation recursion limit."""
        error_data = {
            "language": "cpp",
            "error_type": "CompilationError",
            "message": "template instantiation depth exceeds maximum of 900",
            "file": "template_meta.hpp",
            "line": 100,
            "template_chain": ["Fibonacci<900>", "Fibonacci<899>", "..."]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert "template" in analysis["root_cause"]
        assert "depth" in analysis["suggestion"] or "limit" in analysis["suggestion"]
    
    def test_undefined_behavior_optimization(self):
        """Test undefined behavior eliminated by optimizer."""
        error_data = {
            "language": "cpp",
            "error_type": "RuntimeError",
            "message": "Segmentation fault",
            "compiler_warning": "assuming signed overflow does not occur",
            "optimization_level": "-O3",
            "undefined_behavior": "signed integer overflow"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert "undefined behavior" in analysis["root_cause"]
        assert analysis["severity"] == "high"
    
    def test_coroutine_promise_type_mismatch(self):
        """Test C++20 coroutine promise type mismatch."""
        error_data = {
            "language": "cpp",
            "error_type": "CompilationError",
            "message": "no member named 'promise_type' in 'std::coroutine_traits'",
            "file": "async.cpp",
            "line": 50,
            "cpp_version": "c++20"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert "coroutine" in analysis["root_cause"] or "promise_type" in analysis["suggestion"]
    
    def test_module_partition_circular_dependency(self):
        """Test C++20 module partition circular dependency."""
        error_data = {
            "language": "cpp",
            "error_type": "CompilationError",
            "message": "module partition forms circular dependency",
            "modules": ["math:core", "math:advanced", "math:core"],
            "cpp_version": "c++20"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert "circular" in analysis["root_cause"]
        assert "module" in analysis["suggestion"]


class TestGoEdgeCases:
    """Edge cases for Go error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin_system = LanguagePluginSystem()
        self.plugin = self.plugin_system.get_plugin("go")
    
    def test_cgo_panic_recovery(self):
        """Test panic recovery across CGO boundary."""
        error_data = {
            "language": "go",
            "error_type": "CGOPanic",
            "message": "panic during CGO call",
            "recovered": False,
            "c_function": "process_data",
            "signal": "SIGSEGV"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert "CGO" in analysis["root_cause"] or "cgo" in analysis["root_cause"]
        assert analysis["severity"] == "critical"
    
    def test_goroutine_leak_detection(self):
        """Test goroutine leak with detailed analysis."""
        error_data = {
            "language": "go",
            "error_type": "GoroutineLeak",
            "message": "goroutine leak detected",
            "count": 10000,
            "growth_rate": "100/sec",
            "stack_analysis": {
                "blocked_on_channel": 9500,
                "blocked_on_mutex": 400,
                "running": 100
            }
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert "leak" in analysis["root_cause"]
        assert "channel" in analysis["suggestion"] or "close" in analysis["suggestion"]
    
    def test_reflection_panic_with_nil_interface(self):
        """Test reflection panic with nil interface."""
        error_data = {
            "language": "go",
            "error_type": "Panic",
            "message": "reflect: call of reflect.Value.Type on zero Value",
            "file": "reflector.go",
            "line": 75,
            "operation": "reflect.ValueOf(nil).Type()"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert "reflect" in analysis["root_cause"] or "reflection" in analysis["root_cause"]
        assert "nil check" in analysis["suggestion"] or "IsValid()" in analysis["suggestion"]
    
    def test_unsafe_pointer_arithmetic_overflow(self):
        """Test unsafe pointer arithmetic overflow."""
        error_data = {
            "language": "go",
            "error_type": "RuntimeError",
            "message": "runtime error: unsafe pointer arithmetic",
            "file": "unsafe_ops.go",
            "line": 30,
            "pointer_value": "0xc000000000",
            "offset": "0x7fffffffffffffff"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert "unsafe" in analysis["root_cause"]
        assert analysis["severity"] == "critical"


class TestMultiLanguageEdgeCases:
    """Edge cases involving multiple languages or cross-language scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin_system = LanguagePluginSystem()
    
    def test_mixed_encoding_in_error_chain(self):
        """Test error chain with mixed character encodings."""
        # This could happen in a polyglot system
        error_chain = [
            {
                "language": "python",
                "message": "UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80"
            },
            {
                "language": "java",
                "message": "java.nio.charset.MalformedInputException: Input length = 1"
            },
            {
                "language": "javascript",
                "message": "URIError: URI malformed"
            }
        ]
        
        for error in error_chain:
            plugin = self.plugin_system.get_plugin(error["language"])
            if plugin:
                analysis = plugin.analyze_error(error)
                assert "encoding" in str(analysis).lower() or "decode" in str(analysis).lower()
    
    def test_numeric_overflow_across_languages(self):
        """Test numeric overflow handling in different languages."""
        test_cases = [
            {
                "language": "python",
                "error_type": "OverflowError",
                "message": "int too large to convert to float",
                "value": "10**308"
            },
            {
                "language": "java",
                "error_type": "ArithmeticException",
                "message": "integer overflow",
                "value": "Integer.MAX_VALUE + 1"
            },
            {
                "language": "javascript",
                "error_type": "RangeError",
                "message": "BigInt too large",
                "value": "2n ** 1000000n"
            },
            {
                "language": "go",
                "error_type": "RuntimeError",
                "message": "integer divide by zero",
                "operation": "math.MaxInt64 / 0"
            }
        ]
        
        for test_case in test_cases:
            plugin = self.plugin_system.get_plugin(test_case["language"])
            if plugin:
                analysis = plugin.analyze_error(test_case)
                assert analysis is not None
                assert "overflow" in str(analysis).lower() or "arithmetic" in str(analysis).lower()
    
    def test_memory_corruption_patterns(self):
        """Test similar memory corruption patterns across languages."""
        corruption_patterns = [
            {
                "language": "cpp",
                "pattern": "double-free",
                "message": "free(): double free detected"
            },
            {
                "language": "rust",
                "pattern": "use-after-free",
                "message": "use of moved value"
            },
            {
                "language": "go",
                "pattern": "data-race",
                "message": "DATA RACE: concurrent map access"
            }
        ]
        
        for pattern in corruption_patterns:
            plugin = self.plugin_system.get_plugin(pattern["language"])
            if plugin:
                error = {
                    "language": pattern["language"],
                    "error_type": "MemoryError",
                    "message": pattern["message"]
                }
                analysis = plugin.analyze_error(error)
                assert analysis["severity"] in ["high", "critical"]


class TestPlatformSpecificEdgeCases:
    """Platform-specific edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin_system = LanguagePluginSystem()
    
    def test_windows_path_length_limit(self):
        """Test Windows path length limit errors."""
        # This affects multiple languages on Windows
        languages = ["python", "java", "csharp"]
        
        for lang in languages:
            plugin = self.plugin_system.get_plugin(lang)
            if plugin:
                error = {
                    "language": lang,
                    "error_type": "FileSystemError",
                    "message": f"The system cannot find the path specified",
                    "path_length": 300,
                    "platform": "Windows",
                    "path": "C:\\" + "very\\long\\" * 50 + "path.txt"
                }
                
                analysis = plugin.analyze_error(error)
                assert analysis is not None
                assert "path" in analysis["suggestion"].lower() or "length" in analysis["suggestion"].lower()
    
    def test_linux_file_descriptor_limit(self):
        """Test Linux file descriptor limit errors."""
        error_data = {
            "language": "python",
            "error_type": "OSError",
            "message": "Too many open files",
            "errno": 24,
            "platform": "Linux",
            "open_files": 1024,
            "ulimit": 1024
        }
        
        plugin = self.plugin_system.get_plugin("python")
        analysis = plugin.analyze_error(error_data)
        
        assert "file descriptor" in analysis["suggestion"] or "ulimit" in analysis["suggestion"]
    
    def test_macos_sandbox_violation(self):
        """Test macOS sandbox violation errors."""
        error_data = {
            "language": "swift",
            "error_type": "SandboxViolation",
            "message": "deny file-read-data /private/var/db/receipts",
            "platform": "macOS",
            "entitlement": "com.apple.security.files.user-selected.read-only"
        }
        
        plugin = self.plugin_system.get_plugin("swift")
        if plugin:
            analysis = plugin.analyze_error(error_data)
            assert "sandbox" in str(analysis).lower() or "entitlement" in str(analysis).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])