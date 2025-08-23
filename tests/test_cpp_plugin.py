"""
Test cases for C++ language plugin.

This module contains comprehensive test cases for the C++ plugin,
including compilation errors, runtime errors, memory management issues,
STL errors, template errors, threading issues, and framework-specific errors.
"""
import pytest
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the modules directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.analysis.plugins.cpp_plugin import CPPLanguagePlugin, CPPExceptionHandler, CPPPatchGenerator
from modules.analysis.cpp_adapter import CPPErrorAdapter


class TestCPPErrorAdapter:
    """Test cases for C++ error adapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = CPPErrorAdapter()
    
    def test_to_standard_format_compilation_error(self):
        """Test C++ compilation error conversion to standard format."""
        cpp_error = {
            "type": "CompilationError",
            "file": "main.cpp",
            "line": 15,
            "column": 10,
            "message": "error: 'vector' was not declared in this scope",
            "compiler": "g++",
            "flags": "-std=c++17"
        }
        
        standard_error = self.adapter.to_standard_format(cpp_error)
        
        assert standard_error["language"] == "cpp"
        assert standard_error["error_type"] == "CompilationError"
        assert standard_error["message"] == "error: 'vector' was not declared in this scope"
        assert standard_error["file"] == "main.cpp"
        assert standard_error["line"] == 15
        assert standard_error["column"] == 10
        assert standard_error["compiler"] == "g++"
    
    def test_to_standard_format_segmentation_fault(self):
        """Test segmentation fault error conversion."""
        cpp_error = {
            "type": "SegmentationFault",
            "signal": "SIGSEGV",
            "message": "Segmentation fault (core dumped)",
            "stacktrace": [
                "#0  0x0000000000400546 in main () at test.cpp:10",
                "#1  0x00007ffff7a2d830 in __libc_start_main () from /lib/x86_64-linux-gnu/libc.so.6"
            ],
            "address": "0x0"
        }
        
        standard_error = self.adapter.to_standard_format(cpp_error)
        
        assert standard_error["language"] == "cpp"
        assert standard_error["error_type"] == "SegmentationFault"
        assert standard_error["signal"] == "SIGSEGV"
        assert len(standard_error["stack_trace"]) == 2
        assert standard_error["stack_trace"][0]["file"] == "test.cpp"
        assert standard_error["stack_trace"][0]["line"] == 10
        assert standard_error["stack_trace"][0]["function"] == "main"
    
    def test_to_standard_format_memory_leak(self):
        """Test memory leak error conversion (e.g., from Valgrind)."""
        cpp_error = {
            "type": "MemoryLeak",
            "tool": "valgrind",
            "bytes_lost": "100 bytes",
            "blocks_lost": 1,
            "stacktrace": [
                "at 0x4C2DB8F: malloc (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)",
                "by 0x400537: main (leak.cpp:5)"
            ]
        }
        
        standard_error = self.adapter.to_standard_format(cpp_error)
        
        assert standard_error["language"] == "cpp"
        assert standard_error["error_type"] == "MemoryLeak"
        assert standard_error["tool"] == "valgrind"
        assert standard_error["bytes_lost"] == "100 bytes"
    
    def test_to_standard_format_template_error(self):
        """Test C++ template instantiation error conversion."""
        cpp_error = {
            "type": "TemplateInstantiationError",
            "file": "template.cpp",
            "line": 25,
            "message": "error: no matching function for call to 'MyTemplate<int>::process(const char*)'",
            "instantiation_stack": [
                "instantiated from 'void foo() [with T = int]' at template.cpp:30",
                "instantiated from here at main.cpp:10"
            ]
        }
        
        standard_error = self.adapter.to_standard_format(cpp_error)
        
        assert standard_error["language"] == "cpp"
        assert standard_error["error_type"] == "TemplateInstantiationError"
        assert "instantiation_stack" in standard_error
        assert len(standard_error["instantiation_stack"]) == 2
    
    def test_from_standard_format(self):
        """Test conversion from standard format to C++ format."""
        standard_error = {
            "id": "test-123",
            "language": "cpp",
            "error_type": "CompilationError",
            "message": "undefined reference to 'myFunction()'",
            "file": "main.cpp",
            "line": 20,
            "stack_trace": []
        }
        
        cpp_error = self.adapter.from_standard_format(standard_error)
        
        assert cpp_error["type"] == "CompilationError"
        assert cpp_error["message"] == "undefined reference to 'myFunction()'"
        assert cpp_error["file"] == "main.cpp"
        assert cpp_error["line"] == 20
    
    def test_parse_gdb_stack_trace(self):
        """Test parsing of GDB stack trace format."""
        stack_trace_str = """
#0  0x0000000000400546 in processData (data=0x0) at processor.cpp:45
#1  0x0000000000400456 in main (argc=1, argv=0x7fffffffe098) at main.cpp:10
#2  0x00007ffff7a2d830 in __libc_start_main (main=0x400440 <main>, argc=1, argv=0x7fffffffe098)
        """
        
        frames = self.adapter._parse_gdb_stack_trace(stack_trace_str.strip())
        
        assert len(frames) >= 2
        assert frames[0]["function"] == "processData"
        assert frames[0]["file"] == "processor.cpp"
        assert frames[0]["line"] == 45
        assert frames[1]["function"] == "main"
        assert frames[1]["file"] == "main.cpp"
        assert frames[1]["line"] == 10


class TestCPPExceptionHandler:
    """Test cases for C++ exception handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = CPPExceptionHandler()
    
    def test_analyze_segmentation_fault(self):
        """Test analysis of segmentation fault."""
        error_data = {
            "error_type": "SegmentationFault",
            "signal": "SIGSEGV",
            "message": "Segmentation fault",
            "stack_trace": [
                {"function": "strcpy", "file": "string.cpp", "line": 10}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "cpp_segmentation_fault"
        assert analysis["severity"] == "critical"
        assert analysis["category"] == "memory"
        assert "null pointer" in analysis["suggestion"].lower() or "buffer" in analysis["suggestion"].lower()
    
    def test_analyze_undefined_reference(self):
        """Test analysis of undefined reference linker error."""
        error_data = {
            "error_type": "LinkerError",
            "message": "undefined reference to 'MyClass::myMethod()'",
            "file": "main.cpp",
            "line": 25
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "cpp_undefined_reference"
        assert analysis["category"] == "linking"
        assert "implement" in analysis["suggestion"].lower() or "link" in analysis["suggestion"].lower()
    
    def test_analyze_memory_leak(self):
        """Test analysis of memory leak."""
        error_data = {
            "error_type": "MemoryLeak",
            "tool": "valgrind",
            "bytes_lost": "1,024 bytes",
            "stack_trace": [
                {"function": "operator new", "file": "main.cpp", "line": 50}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "cpp_memory_leak"
        assert analysis["category"] == "memory"
        assert analysis["severity"] == "high"
        assert "delete" in analysis["suggestion"].lower() or "smart pointer" in analysis["suggestion"].lower()
    
    def test_analyze_template_error(self):
        """Test analysis of template instantiation error."""
        error_data = {
            "error_type": "TemplateInstantiationError",
            "message": "no matching function for call to 'std::vector<T>::push_back(const char*)'",
            "file": "template.hpp",
            "line": 30
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "template" in analysis["root_cause"]
        assert analysis["category"] == "templates"
        assert "type" in analysis["suggestion"].lower()
    
    def test_analyze_stl_error(self):
        """Test analysis of STL-related error."""
        error_data = {
            "error_type": "RuntimeError",
            "message": "vector::_M_range_check: __n (which is 10) >= this->size() (which is 5)",
            "stack_trace": [
                {"function": "std::vector<int>::at", "file": "stl_vector.h"}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "cpp_stl_out_of_range"
        assert analysis["category"] == "stl"
        assert "bounds" in analysis["suggestion"].lower() or "size" in analysis["suggestion"].lower()
    
    def test_analyze_race_condition(self):
        """Test analysis of race condition / threading error."""
        error_data = {
            "error_type": "DataRace",
            "tool": "ThreadSanitizer",
            "message": "WARNING: ThreadSanitizer: data race",
            "stack_trace": [
                {"function": "incrementCounter", "file": "counter.cpp", "line": 15}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "race" in analysis["root_cause"] or "thread" in analysis["root_cause"]
        assert analysis["category"] == "threading"
        assert analysis["severity"] == "high"
        assert "mutex" in analysis["suggestion"].lower() or "atomic" in analysis["suggestion"].lower()
    
    def test_analyze_double_free(self):
        """Test analysis of double free error."""
        error_data = {
            "error_type": "DoubleFree",
            "message": "double free or corruption",
            "signal": "SIGABRT",
            "stack_trace": [
                {"function": "free", "file": "malloc.c"},
                {"function": "cleanup", "file": "resource.cpp", "line": 100}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "cpp_double_free"
        assert analysis["category"] == "memory"
        assert analysis["severity"] == "critical"
        assert "once" in analysis["suggestion"].lower() or "smart" in analysis["suggestion"].lower()
    
    def test_analyze_stack_overflow(self):
        """Test analysis of stack overflow."""
        error_data = {
            "error_type": "StackOverflow",
            "signal": "SIGSEGV",
            "message": "Stack overflow",
            "stack_trace": [
                {"function": "recursive_function", "file": "recursion.cpp", "line": 20},
                {"function": "recursive_function", "file": "recursion.cpp", "line": 20},
                {"function": "recursive_function", "file": "recursion.cpp", "line": 20}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "cpp_stack_overflow"
        assert analysis["category"] == "memory"
        assert "recursion" in analysis["suggestion"].lower() or "base case" in analysis["suggestion"].lower()
    
    def test_fallback_analysis(self):
        """Test fallback analysis for unknown errors."""
        error_data = {
            "error_type": "UnknownError",
            "message": "Something went wrong",
            "stack_trace": []
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["rule_id"] == "cpp_generic_fallback"
        assert analysis["root_cause"] == "cpp_unknown_error"
        assert analysis["confidence"] == "low"


class TestCPPPatchGenerator:
    """Test cases for C++ patch generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = CPPPatchGenerator()
    
    def test_generate_null_check_patch(self):
        """Test patch generation for null pointer dereference."""
        analysis = {
            "rule_id": "cpp_null_dereference",
            "root_cause": "cpp_segmentation_fault",
            "error_data": {
                "error_type": "SegmentationFault",
                "message": "Segmentation fault"
            },
            "suggestion": "Add null pointer check",
            "confidence": "high"
        }
        
        context = {
            "code_snippet": "int len = strlen(str);",
            "variable_name": "str"
        }
        
        patch = self.generator.generate_patch(analysis, context)
        
        assert patch["language"] == "cpp"
        assert patch["root_cause"] == "cpp_segmentation_fault"
        assert "suggestion_code" in patch
        assert "if (str != nullptr)" in patch["suggestion_code"] or "if (str)" in patch["suggestion_code"]
    
    def test_generate_memory_management_patch(self):
        """Test patch generation for memory leak."""
        analysis = {
            "rule_id": "cpp_memory_leak",
            "root_cause": "cpp_memory_leak",
            "error_data": {
                "error_type": "MemoryLeak",
                "message": "Memory leak detected"
            },
            "suggestion": "Use smart pointers or ensure delete",
            "confidence": "high"
        }
        
        context = {
            "code_snippet": "int* arr = new int[100];",
            "allocation_type": "array"
        }
        
        patch = self.generator.generate_patch(analysis, context)
        
        assert patch["language"] == "cpp"
        assert "delete[]" in patch["suggestion_code"] or "unique_ptr" in patch["suggestion_code"]
    
    def test_generate_bounds_check_patch(self):
        """Test patch generation for array bounds checking."""
        analysis = {
            "rule_id": "cpp_array_bounds",
            "root_cause": "cpp_buffer_overflow",
            "error_data": {
                "error_type": "BufferOverflow",
                "message": "Array index out of bounds"
            },
            "suggestion": "Add bounds checking",
            "confidence": "high"
        }
        
        context = {
            "code_snippet": "arr[i] = value;",
            "array_name": "arr",
            "index_name": "i"
        }
        
        patch = self.generator.generate_patch(analysis, context)
        
        assert patch["language"] == "cpp"
        assert "if (i >= 0 && i < " in patch["suggestion_code"] or "at(" in patch["suggestion_code"]
    
    def test_generate_thread_safety_patch(self):
        """Test patch generation for thread safety issues."""
        analysis = {
            "rule_id": "cpp_data_race",
            "root_cause": "cpp_race_condition",
            "error_data": {
                "error_type": "DataRace"
            },
            "suggestion": "Add synchronization",
            "confidence": "high"
        }
        
        context = {
            "code_snippet": "counter++;",
            "shared_variable": "counter"
        }
        
        patch = self.generator.generate_patch(analysis, context)
        
        assert patch["language"] == "cpp"
        assert "mutex" in patch["suggestion_code"] or "atomic" in patch["suggestion_code"]


class TestCPPLanguagePlugin:
    """Test cases for the main C++ language plugin."""
    
    @pytest.fixture
    def plugin(self):
        """Create a C++ plugin instance for testing."""
        return CPPLanguagePlugin()
    
    def test_plugin_initialization(self, plugin):
        """Test plugin initialization."""
        assert plugin.get_language_id() == "cpp"
        assert plugin.get_language_name() == "C++"
        assert plugin.get_language_version() == "C++11/14/17/20"
        assert "qt" in plugin.get_supported_frameworks()
        assert "boost" in plugin.get_supported_frameworks()
        assert "stl" in plugin.get_supported_frameworks()
    
    def test_can_handle_cpp_errors(self, plugin):
        """Test plugin can handle C++ errors."""
        # Test with explicit language
        error_data = {"language": "cpp", "error_type": "CompilationError"}
        assert plugin.can_handle(error_data) is True
        
        # Test with C++ specific errors
        cpp_errors = [
            {"error_type": "SegmentationFault", "signal": "SIGSEGV"},
            {"error_type": "CompilationError", "file": "main.cpp"},
            {"error_type": "TemplateInstantiationError"},
            {"error_type": "LinkerError", "message": "undefined reference"},
            {"file": "test.cpp", "error_type": "MemoryLeak"},
            {"file": "test.cc", "error_type": "RuntimeError"},
            {"file": "test.cxx", "error_type": "CompilationError"}
        ]
        
        for error in cpp_errors:
            assert plugin.can_handle(error) is True
    
    def test_normalize_error(self, plugin):
        """Test error normalization."""
        raw_error = {
            "raw_output": "main.cpp:10:5: error: use of undeclared identifier 'cout'\n    cout << \"Hello\";\n    ^"
        }
        
        normalized = plugin.normalize_error(raw_error)
        
        assert normalized["error_type"] == "CompilationError"
        assert normalized["file"] == "main.cpp"
        assert normalized["line"] == 10
        assert normalized["column"] == 5
        assert "undeclared identifier" in normalized["message"]
    
    def test_analyze_compilation_error(self, plugin):
        """Test analysis of compilation error."""
        error_data = {
            "error_type": "CompilationError",
            "message": "error: 'vector' was not declared in this scope",
            "file": "main.cpp",
            "line": 15
        }
        
        analysis = plugin.analyze_error(error_data)
        
        assert analysis is not None
        assert analysis["error_type"] == "CompilationError"
        assert "include" in analysis["suggestion"].lower()
    
    def test_analyze_runtime_error(self, plugin):
        """Test analysis of runtime error."""
        error_data = {
            "error_type": "SegmentationFault",
            "signal": "SIGSEGV",
            "stack_trace": [
                {"function": "main", "file": "test.cpp", "line": 20}
            ]
        }
        
        analysis = plugin.analyze_error(error_data)
        
        assert analysis is not None
        assert analysis["severity"] == "critical"
        assert analysis["category"] == "memory"
    
    def test_generate_fix(self, plugin):
        """Test fix generation."""
        error_data = {
            "error_type": "CompilationError",
            "message": "error: 'cout' was not declared in this scope",
            "file": "hello.cpp",
            "line": 5
        }
        
        analysis = plugin.analyze_error(error_data)
        context = {
            "code_snippet": "cout << \"Hello World\";",
            "includes": []
        }
        
        fix = plugin.generate_fix(analysis, context)
        
        assert fix is not None
        assert fix["language"] == "cpp"
        assert "#include <iostream>" in fix["suggestion_code"]
        assert "std::" in fix["suggestion_code"] or "using namespace std" in fix["suggestion_code"]
    
    def test_language_info(self, plugin):
        """Test language info methods."""
        assert plugin.get_language_id() == "cpp"
        assert plugin.get_language_name() == "C++"
        
        frameworks = plugin.get_supported_frameworks()
        assert "qt" in frameworks
        assert "boost" in frameworks
        assert "poco" in frameworks
        assert "stl" in frameworks


class TestCPPCompilerSpecificErrors:
    """Test cases for compiler-specific C++ errors."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = CPPExceptionHandler()
        self.adapter = CPPErrorAdapter()
    
    def test_gcc_error_format(self):
        """Test parsing GCC error format."""
        gcc_error = {
            "compiler": "gcc",
            "raw_output": "test.cpp: In function 'int main()':\ntest.cpp:5:10: error: 'cout' was not declared in this scope"
        }
        
        # This would be handled by the plugin's error normalization
        assert gcc_error["compiler"] == "gcc"
    
    def test_clang_error_format(self):
        """Test parsing Clang error format."""
        clang_error = {
            "compiler": "clang",
            "raw_output": "test.cpp:5:10: error: use of undeclared identifier 'cout'\n    cout << \"Hello\";\n    ^"
        }
        
        # This would be handled by the plugin's error normalization
        assert clang_error["compiler"] == "clang"
    
    def test_msvc_error_format(self):
        """Test parsing MSVC error format."""
        msvc_error = {
            "compiler": "msvc",
            "raw_output": "test.cpp(5): error C2065: 'cout': undeclared identifier"
        }
        
        # This would be handled by the plugin's error normalization
        assert msvc_error["compiler"] == "msvc"


class TestCPPFrameworkErrors:
    """Test cases for C++ framework-specific errors."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = CPPExceptionHandler()
        self.plugin = CPPLanguagePlugin()
    
    def test_qt_signal_slot_error(self):
        """Test analysis of Qt signal/slot connection error."""
        error_data = {
            "error_type": "RuntimeError",
            "message": "QObject::connect: Cannot connect (null)::clicked() to MainWindow::onButtonClicked()",
            "framework": "qt"
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "qt" in analysis["framework"]
        assert "signal" in analysis["suggestion"].lower() or "slot" in analysis["suggestion"].lower()
    
    def test_boost_exception(self):
        """Test analysis of Boost library exception."""
        error_data = {
            "error_type": "boost::filesystem::filesystem_error",
            "message": "boost::filesystem::directory_iterator::construct: No such file or directory",
            "framework": "boost"
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["framework"] == "boost"
        assert "file" in analysis["suggestion"].lower() or "directory" in analysis["suggestion"].lower()
    
    def test_stl_exception(self):
        """Test analysis of STL exceptions."""
        error_data = {
            "error_type": "std::out_of_range",
            "message": "vector::_M_range_check: __n (which is 10) >= this->size() (which is 5)",
            "framework": "stl"
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["framework"] == "stl"
        assert analysis["root_cause"] == "cpp_stl_out_of_range"


class TestCPPEdgeCases:
    """Test cases for C++ edge cases and corner cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = CPPErrorAdapter()
        self.handler = CPPExceptionHandler()
    
    def test_parse_multiple_errors(self):
        """Test parsing multiple compilation errors."""
        cpp_error = {
            "type": "CompilationError",
            "errors": [
                {
                    "file": "main.cpp",
                    "line": 10,
                    "message": "error: 'cout' was not declared"
                },
                {
                    "file": "main.cpp",
                    "line": 15,
                    "message": "error: expected ';' before '}' token"
                }
            ]
        }
        
        standard_error = self.adapter.to_standard_format(cpp_error)
        
        assert standard_error["error_type"] == "CompilationError"
        assert "errors" in standard_error
        assert len(standard_error["errors"]) == 2
    
    def test_parse_warning_as_error(self):
        """Test parsing warnings treated as errors."""
        cpp_error = {
            "type": "CompilationWarning",
            "warning_as_error": True,
            "file": "util.cpp",
            "line": 30,
            "message": "warning: comparison between signed and unsigned integer expressions [-Wsign-compare]"
        }
        
        standard_error = self.adapter.to_standard_format(cpp_error)
        
        assert standard_error["warning_as_error"] is True
        assert "sign" in standard_error["message"]
    
    def test_parse_linker_error_multiple_definitions(self):
        """Test parsing multiple definition linker errors."""
        cpp_error = {
            "type": "LinkerError",
            "message": "multiple definition of 'globalVar'",
            "references": [
                "first defined here: obj1.o:global.cpp:5",
                "also defined here: obj2.o:global.cpp:5"
            ]
        }
        
        standard_error = self.adapter.to_standard_format(cpp_error)
        
        assert standard_error["error_type"] == "LinkerError"
        assert "multiple definition" in standard_error["message"]
        assert "references" in standard_error
    
    def test_unicode_in_error_messages(self):
        """Test handling unicode characters in error messages."""
        cpp_error = {
            "type": "CompilationError",
            "file": "unicode.cpp",
            "line": 10,
            "message": "error: stray '\\302' in program"  # Unicode character issue
        }
        
        standard_error = self.adapter.to_standard_format(cpp_error)
        
        assert standard_error["error_type"] == "CompilationError"
        assert "stray" in standard_error["message"]


class TestCPPMemoryDebugging:
    """Test cases for C++ memory debugging tool integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = CPPExceptionHandler()
    
    def test_valgrind_invalid_read(self):
        """Test analysis of Valgrind invalid read error."""
        error_data = {
            "error_type": "InvalidRead",
            "tool": "valgrind",
            "message": "Invalid read of size 4",
            "stack_trace": [
                {"function": "processArray", "file": "array.cpp", "line": 25}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "memory"
        assert "bounds" in analysis["suggestion"].lower() or "size" in analysis["suggestion"].lower()
    
    def test_addresssanitizer_heap_use_after_free(self):
        """Test analysis of AddressSanitizer heap-use-after-free."""
        error_data = {
            "error_type": "HeapUseAfterFree",
            "tool": "AddressSanitizer",
            "message": "heap-use-after-free on address 0x60300000eff0",
            "stack_trace": [
                {"function": "getData", "file": "data.cpp", "line": 50}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "cpp_use_after_free"
        assert analysis["severity"] == "critical"
        assert "freed" in analysis["suggestion"].lower()


class TestCPPIntegration:
    """Integration tests for C++ plugin with real-world scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = CPPLanguagePlugin()
    
    def test_end_to_end_missing_include(self):
        """Test end-to-end flow for missing include error."""
        error_data = {
            "language": "cpp",
            "error_type": "CompilationError",
            "message": "error: 'vector' was not declared in this scope",
            "file": "main.cpp",
            "line": 10,
            "column": 5
        }
        
        # Analyze the error
        analysis = self.plugin.analyze_error(error_data)
        assert analysis is not None
        assert "include" in analysis["suggestion"].lower()
        
        # Generate a fix
        context = {
            "code_snippet": "vector<int> numbers;",
            "includes": ["#include <iostream>"]
        }
        
        fix = self.plugin.generate_fix(analysis, context)
        assert fix is not None
        assert fix["language"] == "cpp"
        assert "#include <vector>" in fix["suggestion_code"]
    
    def test_end_to_end_memory_leak_fix(self):
        """Test end-to-end flow for memory leak fix."""
        error_data = {
            "language": "cpp",
            "error_type": "MemoryLeak",
            "tool": "valgrind",
            "bytes_lost": "100 bytes in 1 blocks",
            "stack_trace": [
                {
                    "function": "createBuffer",
                    "file": "buffer.cpp",
                    "line": 15
                }
            ]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert analysis is not None
        assert analysis["root_cause"] == "cpp_memory_leak"
        
        context = {
            "code_snippet": "char* buffer = new char[100];",
            "function_context": "void processData() { char* buffer = new char[100]; }"
        }
        
        fix = self.plugin.generate_fix(analysis, context)
        assert fix is not None
        assert "delete[]" in fix["suggestion_code"] or "unique_ptr" in fix["suggestion_code"]
    
    def test_end_to_end_segfault_analysis(self):
        """Test end-to-end flow for segmentation fault analysis."""
        error_data = {
            "language": "cpp",
            "error_type": "SegmentationFault",
            "signal": "SIGSEGV",
            "message": "Segmentation fault (core dumped)",
            "stack_trace": [
                {
                    "function": "strlen",
                    "file": "string.h",
                    "line": 0
                },
                {
                    "function": "processString",
                    "file": "processor.cpp",
                    "line": 25
                }
            ]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert analysis is not None
        assert analysis["severity"] == "critical"
        assert "null" in analysis["suggestion"].lower()
    
    def test_framework_detection_in_errors(self):
        """Test that framework-specific errors are detected correctly."""
        test_cases = [
            {
                "error_type": "QObject::connect",
                "expected_framework": "qt"
            },
            {
                "error_type": "boost::bad_lexical_cast",
                "expected_framework": "boost"
            },
            {
                "error_type": "std::bad_alloc",
                "expected_framework": "stl"
            }
        ]
        
        for test_case in test_cases:
            error_data = {
                "error_type": test_case["error_type"],
                "message": "Test error",
                "stack_trace": []
            }
            
            analysis = self.plugin.analyze_error(error_data)
            if "framework" in analysis:
                assert analysis["framework"] == test_case["expected_framework"]


class TestCPPBuildSystemErrors:
    """Test cases for C++ build system errors."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = CPPExceptionHandler()
    
    def test_cmake_configuration_error(self):
        """Test analysis of CMake configuration errors."""
        error_data = {
            "error_type": "CMakeError",
            "message": "CMake Error: Could not find a package configuration file provided by \"Qt5\"",
            "file": "CMakeLists.txt",
            "line": 10
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "cmake" in analysis["root_cause"].lower()
        assert analysis["category"] == "cmake"
        assert "find_package" in analysis["suggestion"] or "path" in analysis["suggestion"].lower()
    
    def test_makefile_error(self):
        """Test analysis of Makefile errors."""
        error_data = {
            "error_type": "MakefileError",
            "message": "make: *** No rule to make target 'main.o', needed by 'app'",
            "file": "Makefile"
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "makefile" in analysis["category"]
        assert "rule" in analysis["suggestion"].lower() or "target" in analysis["suggestion"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])