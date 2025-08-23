"""
Test cases for Go language plugin.

This module contains comprehensive test cases for the Go plugin,
including compilation errors, runtime panics, goroutine issues,
channel operations, interface errors, and framework-specific errors.
"""
import pytest
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the modules directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.analysis.plugins.go_plugin import GoLanguagePlugin, GoErrorHandler as GoExceptionHandler, GoPatchGenerator
from modules.analysis.language_adapters import GoErrorAdapter


class TestGoErrorAdapter:
    """Test cases for Go error adapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = GoErrorAdapter()
    
    def test_to_standard_format_compilation_error(self):
        """Test Go compilation error conversion to standard format."""
        go_error = {
            "type": "CompilationError",
            "file": "main.go",
            "line": 15,
            "column": 10,
            "message": "undefined: fmt.Printlnx",
            "package": "main"
        }
        
        standard_error = self.adapter.to_standard_format(go_error)
        
        assert standard_error["language"] == "go"
        assert standard_error["error_type"] == "CompilationError"
        assert standard_error["message"] == "undefined: fmt.Printlnx"
        assert standard_error["file"] == "main.go"
        assert standard_error["line"] == 15
        assert standard_error["column"] == 10
    
    def test_to_standard_format_panic(self):
        """Test Go panic conversion."""
        go_error = {
            "type": "Panic",
            "message": "runtime error: index out of range [3] with length 2",
            "stacktrace": [
                "goroutine 1 [running]:",
                "main.main()",
                "\t/home/user/project/main.go:10 +0x27"
            ]
        }
        
        standard_error = self.adapter.to_standard_format(go_error)
        
        assert standard_error["language"] == "go"
        assert standard_error["error_type"] == "Panic"
        assert "index out of range" in standard_error["message"]
        assert len(standard_error["stack_trace"]) == 1
        assert standard_error["stack_trace"][0]["file"] == "/home/user/project/main.go"
        assert standard_error["stack_trace"][0]["line"] == 10
        assert standard_error["stack_trace"][0]["function"] == "main.main"
    
    def test_to_standard_format_nil_pointer(self):
        """Test nil pointer dereference error conversion."""
        go_error = {
            "type": "Panic",
            "message": "runtime error: invalid memory address or nil pointer dereference",
            "signal": "SIGSEGV",
            "stacktrace": [
                "goroutine 5 [running]:",
                "main.processData(0x0)",
                "\t/app/processor.go:25 +0x39",
                "main.worker(0xc0000ae000)",
                "\t/app/worker.go:15 +0x65"
            ]
        }
        
        standard_error = self.adapter.to_standard_format(go_error)
        
        assert standard_error["language"] == "go"
        assert standard_error["error_type"] == "Panic"
        assert "nil pointer dereference" in standard_error["message"]
        assert standard_error["signal"] == "SIGSEGV"
        assert len(standard_error["stack_trace"]) == 2
    
    def test_to_standard_format_deadlock(self):
        """Test goroutine deadlock error conversion."""
        go_error = {
            "type": "Deadlock",
            "message": "fatal error: all goroutines are asleep - deadlock!",
            "goroutines": [
                {
                    "id": 1,
                    "state": "chan receive",
                    "location": "main.go:20"
                },
                {
                    "id": 2,
                    "state": "chan send",
                    "location": "worker.go:35"
                }
            ]
        }
        
        standard_error = self.adapter.to_standard_format(go_error)
        
        assert standard_error["language"] == "go"
        assert standard_error["error_type"] == "Deadlock"
        assert "deadlock" in standard_error["message"]
        assert "goroutines" in standard_error
    
    def test_from_standard_format(self):
        """Test conversion from standard format to Go format."""
        standard_error = {
            "id": "test-123",
            "language": "go",
            "error_type": "CompilationError",
            "message": "cannot use x (type int) as type string",
            "file": "converter.go",
            "line": 45
        }
        
        go_error = self.adapter.from_standard_format(standard_error)
        
        assert go_error["type"] == "CompilationError"
        assert go_error["file"] == "converter.go"
        assert go_error["line"] == 45


class TestGoExceptionHandler:
    """Test cases for Go exception handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GoExceptionHandler()
    
    def test_analyze_nil_pointer_dereference(self):
        """Test analysis of nil pointer dereference."""
        error_data = {
            "error_type": "Panic",
            "message": "runtime error: invalid memory address or nil pointer dereference",
            "stack_trace": [
                {"function": "processUser", "file": "user.go", "line": 30}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "go_nil_pointer_dereference"
        assert analysis["category"] == "runtime"
        assert analysis["severity"] == "high"
        assert "nil check" in analysis["suggestion"].lower()
    
    def test_analyze_index_out_of_range(self):
        """Test analysis of index out of range error."""
        error_data = {
            "error_type": "Panic",
            "message": "runtime error: index out of range [5] with length 3",
            "stack_trace": [
                {"function": "getElement", "file": "array.go", "line": 15}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "go_index_out_of_range"
        assert analysis["category"] == "runtime"
        assert "bounds check" in analysis["suggestion"].lower()
    
    def test_analyze_type_assertion_failure(self):
        """Test analysis of type assertion failure."""
        error_data = {
            "error_type": "Panic",
            "message": "interface conversion: interface {} is string, not int",
            "stack_trace": [
                {"function": "convertValue", "file": "converter.go", "line": 50}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "go_type_assertion_failure"
        assert analysis["category"] == "runtime"
        assert "type assertion" in analysis["suggestion"].lower() or "type switch" in analysis["suggestion"].lower()
    
    def test_analyze_channel_deadlock(self):
        """Test analysis of channel deadlock."""
        error_data = {
            "error_type": "Deadlock",
            "message": "fatal error: all goroutines are asleep - deadlock!",
            "goroutines": [{"state": "chan receive"}]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "go_channel_deadlock"
        assert analysis["category"] == "concurrency"
        assert analysis["severity"] == "critical"
        assert "channel" in analysis["suggestion"].lower()
    
    def test_analyze_race_condition(self):
        """Test analysis of race condition."""
        error_data = {
            "error_type": "DataRace",
            "message": "WARNING: DATA RACE",
            "stack_trace": [
                {"function": "incrementCounter", "file": "counter.go", "line": 25}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "race" in analysis["root_cause"]
        assert analysis["category"] == "concurrency"
        assert analysis["severity"] == "high"
        assert "mutex" in analysis["suggestion"].lower() or "atomic" in analysis["suggestion"].lower()
    
    def test_analyze_map_concurrent_access(self):
        """Test analysis of concurrent map access."""
        error_data = {
            "error_type": "Panic",
            "message": "fatal error: concurrent map read and map write",
            "stack_trace": [
                {"function": "updateCache", "file": "cache.go", "line": 40}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "go_concurrent_map_access"
        assert analysis["category"] == "concurrency"
        assert "sync.Map" in analysis["suggestion"] or "mutex" in analysis["suggestion"].lower()
    
    def test_analyze_compilation_error(self):
        """Test analysis of compilation error."""
        error_data = {
            "error_type": "CompilationError",
            "message": "cannot use x (type []int) as type []string in argument to processStrings",
            "file": "processor.go",
            "line": 20
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "type" in analysis["root_cause"]
        assert analysis["category"] == "compilation"
        assert "type" in analysis["suggestion"].lower()
    
    def test_analyze_import_cycle(self):
        """Test analysis of import cycle error."""
        error_data = {
            "error_type": "CompilationError",
            "message": "import cycle not allowed",
            "package": "github.com/user/project/pkg/util"
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "go_import_cycle"
        assert analysis["category"] == "compilation"
        assert "refactor" in analysis["suggestion"].lower() or "dependency" in analysis["suggestion"].lower()


class TestGoPatchGenerator:
    """Test cases for Go patch generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = GoPatchGenerator()
    
    def test_generate_nil_check_patch(self):
        """Test generating nil check patch."""
        analysis = {
            "root_cause": "go_nil_pointer_dereference",
            "suggestion": "Add nil check before dereferencing pointer"
        }
        
        error_context = {
            "file": "user.go",
            "line": 30,
            "code_snippet": "return user.Name",
            "variable": "user"
        }
        
        patch = self.generator.generate_patch(analysis, error_context)
        
        assert patch is not None
        assert "if user != nil {" in patch["content"]
        assert patch["type"] == "code_modification"
    
    def test_generate_bounds_check_patch(self):
        """Test generating bounds check patch."""
        analysis = {
            "root_cause": "go_index_out_of_range",
            "suggestion": "Add bounds check before array access"
        }
        
        error_context = {
            "file": "array.go",
            "line": 15,
            "code_snippet": "value := arr[index]",
            "array": "arr",
            "index": "index"
        }
        
        patch = self.generator.generate_patch(analysis, error_context)
        
        assert patch is not None
        assert "if index < len(arr)" in patch["content"] or "len(arr) > index" in patch["content"]
    
    def test_generate_type_assertion_patch(self):
        """Test generating safe type assertion patch."""
        analysis = {
            "root_cause": "go_type_assertion_failure",
            "suggestion": "Use safe type assertion with ok check"
        }
        
        error_context = {
            "file": "converter.go",
            "line": 50,
            "code_snippet": "intVal := val.(int)",
            "interface_var": "val",
            "target_type": "int"
        }
        
        patch = self.generator.generate_patch(analysis, error_context)
        
        assert patch is not None
        assert ", ok :=" in patch["content"]
        assert "if ok {" in patch["content"] or "if !ok {" in patch["content"]
    
    def test_generate_mutex_patch(self):
        """Test generating mutex protection patch."""
        analysis = {
            "root_cause": "go_race_condition",
            "suggestion": "Add mutex protection for shared variable"
        }
        
        error_context = {
            "file": "counter.go",
            "line": 25,
            "code_snippet": "counter++",
            "shared_var": "counter"
        }
        
        patch = self.generator.generate_patch(analysis, error_context)
        
        assert patch is not None
        assert "sync.Mutex" in patch["imports"] or "sync.RWMutex" in patch["imports"]
        assert "Lock()" in patch["content"]
        assert "Unlock()" in patch["content"]


class TestGoLanguagePlugin:
    """Test cases for Go language plugin integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = GoLanguagePlugin()
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        metadata = self.plugin.get_metadata()
        
        assert metadata["name"] == "Go Language Support"
        assert metadata["language"] == "go"
        assert "1.0.0" in metadata["version"]
        assert len(metadata["supported_frameworks"]) > 0
    
    def test_can_handle_go_error(self):
        """Test if plugin can handle Go errors."""
        go_error = {
            "language": "go",
            "error_type": "Panic",
            "message": "runtime error: index out of range"
        }
        
        assert self.plugin.can_handle(go_error) is True
    
    def test_can_handle_non_go_error(self):
        """Test if plugin rejects non-Go errors."""
        python_error = {
            "language": "python",
            "error_type": "ValueError",
            "message": "invalid literal"
        }
        
        assert self.plugin.can_handle(python_error) is False
    
    def test_analyze_error(self):
        """Test full error analysis flow."""
        error_data = {
            "language": "go",
            "error_type": "Panic",
            "message": "runtime error: invalid memory address or nil pointer dereference",
            "stack_trace": [
                {"function": "handleRequest", "file": "handler.go", "line": 100}
            ]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis is not None
        assert "root_cause" in analysis
        assert "category" in analysis
        assert "severity" in analysis
        assert "suggestion" in analysis
    
    def test_framework_detection_gin(self):
        """Test Gin framework detection."""
        error_data = {
            "language": "go",
            "error_type": "Panic",
            "stack_trace": [
                {"function": "github.com/gin-gonic/gin.(*Context).Next", "file": "context.go"}
            ]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis.get("framework") == "gin"
    
    def test_framework_detection_echo(self):
        """Test Echo framework detection."""
        error_data = {
            "language": "go",
            "error_type": "Panic",
            "stack_trace": [
                {"function": "github.com/labstack/echo/v4.(*Echo).ServeHTTP", "file": "echo.go"}
            ]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis.get("framework") == "echo"


class TestGoFrameworkSpecific:
    """Test cases for Go framework-specific errors."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GoExceptionHandler()
    
    def test_gin_binding_error(self):
        """Test Gin framework binding error."""
        error_data = {
            "error_type": "BindingError",
            "message": "Key: 'User.Email' Error:Field validation for 'Email' failed on the 'required' tag",
            "framework": "gin"
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["framework"] == "gin"
        assert "validation" in analysis["suggestion"].lower()
    
    def test_gorm_error(self):
        """Test GORM ORM error."""
        error_data = {
            "error_type": "DatabaseError",
            "message": "record not found",
            "framework": "gorm",
            "stack_trace": [
                {"function": "gorm.io/gorm.(*DB).First", "file": "finisher_api.go"}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["framework"] == "gorm"
        assert "ErrRecordNotFound" in analysis["suggestion"] or "not found" in analysis["suggestion"].lower()
    
    def test_fiber_error(self):
        """Test Fiber framework error."""
        error_data = {
            "error_type": "RuntimeError",
            "message": "Unhandled error in Fiber handler",
            "framework": "fiber",
            "stack_trace": [
                {"function": "github.com/gofiber/fiber/v2.(*App).next", "file": "router.go"}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["framework"] == "fiber"


class TestGoEdgeCases:
    """Test cases for Go edge cases and corner cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = GoErrorAdapter()
        self.handler = GoExceptionHandler()
    
    def test_parse_build_constraint_error(self):
        """Test parsing build constraint errors."""
        go_error = {
            "type": "CompilationError",
            "file": "platform_linux.go",
            "message": "build constraints exclude all Go files in /path/to/package"
        }
        
        standard_error = self.adapter.to_standard_format(go_error)
        
        assert standard_error["error_type"] == "CompilationError"
        assert "build constraints" in standard_error["message"]
    
    def test_parse_cgo_error(self):
        """Test parsing CGO-related errors."""
        go_error = {
            "type": "CGOError",
            "file": "wrapper.go",
            "line": 10,
            "message": "could not determine kind of name for C.some_function"
        }
        
        standard_error = self.adapter.to_standard_format(go_error)
        
        assert standard_error["error_type"] == "CGOError"
        assert "C." in standard_error["message"]
    
    def test_goroutine_leak_detection(self):
        """Test goroutine leak detection."""
        error_data = {
            "error_type": "GoroutineLeak",
            "message": "goroutine leak detected: 1000 goroutines running",
            "stack_summary": {
                "total": 1000,
                "states": {
                    "chan receive": 950,
                    "IO wait": 50
                }
            }
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "leak" in analysis["root_cause"]
        assert analysis["severity"] == "high"
        assert "goroutine" in analysis["suggestion"].lower()
    
    def test_context_timeout_error(self):
        """Test context timeout error handling."""
        error_data = {
            "error_type": "ContextError",
            "message": "context deadline exceeded",
            "stack_trace": [
                {"function": "processWithTimeout", "file": "timeout.go", "line": 45}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "timeout" in analysis["root_cause"] or "deadline" in analysis["root_cause"]
        assert "timeout" in analysis["suggestion"].lower() or "deadline" in analysis["suggestion"].lower()


class TestGoPerformanceAndSecurity:
    """Test cases for Go performance and security issues."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = GoExceptionHandler()
    
    def test_memory_leak_detection(self):
        """Test memory leak detection."""
        error_data = {
            "error_type": "MemoryLeak",
            "message": "possible memory leak detected",
            "metrics": {
                "heap_alloc": "500MB",
                "num_gc": 1000
            }
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "memory" in analysis["root_cause"]
        assert analysis["category"] == "performance"
        assert "profile" in analysis["suggestion"].lower() or "pprof" in analysis["suggestion"].lower()
    
    def test_sql_injection_detection(self):
        """Test SQL injection vulnerability detection."""
        error_data = {
            "error_type": "SecurityVulnerability",
            "message": "potential SQL injection detected",
            "code_snippet": 'db.Query("SELECT * FROM users WHERE id = " + userInput)',
            "file": "database.go",
            "line": 50
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "injection" in analysis["root_cause"] or "security" in analysis["root_cause"]
        assert analysis["severity"] == "critical"
        assert "prepared statement" in analysis["suggestion"].lower() or "parameterized" in analysis["suggestion"].lower()
    
    def test_infinite_loop_detection(self):
        """Test infinite loop detection."""
        error_data = {
            "error_type": "InfiniteLoop",
            "message": "possible infinite loop detected",
            "cpu_usage": "100%",
            "duration": "5m",
            "stack_trace": [
                {"function": "processLoop", "file": "processor.go", "line": 30}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "loop" in analysis["root_cause"]
        assert analysis["category"] == "performance"
        assert "condition" in analysis["suggestion"].lower() or "termination" in analysis["suggestion"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])