"""
Regression test suites for previously fixed primary language bugs.

This module contains regression tests to ensure that previously fixed bugs
in primary language error handling do not resurface. Each test case represents
a real bug that was found and fixed in the past.
"""
import pytest
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch

# Add the modules directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules', 'analysis'))

from language_plugin_system import LanguagePluginSystem
from comprehensive_error_detector import ComprehensiveErrorDetector


class TestPythonRegressions:
    """Regression tests for Python error handling bugs."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin_system = LanguagePluginSystem()
        self.plugin = self.plugin_system.get_plugin("python")
    
    def test_regression_async_generator_stopiteration(self):
        """
        Bug: AsyncGenerator StopIteration was incorrectly categorized as syntax error.
        Fixed: Version 1.2.3
        """
        error_data = {
            "language": "python",
            "error_type": "StopAsyncIteration",
            "message": "async generator raised StopAsyncIteration",
            "file": "async_gen.py",
            "line": 25,
            "python_version": "3.8"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        # Ensure it's not categorized as syntax error
        assert analysis["category"] != "syntax"
        assert analysis["category"] == "runtime"
        assert "async" in analysis["suggestion"].lower()
    
    def test_regression_walrus_operator_syntax(self):
        """
        Bug: Walrus operator := in Python 3.8+ was flagged as syntax error in older Python detection.
        Fixed: Version 1.3.0
        """
        error_data = {
            "language": "python",
            "error_type": "SyntaxError",
            "message": "invalid syntax",
            "file": "modern.py",
            "line": 10,
            "code_snippet": "if (n := len(items)) > 10:",
            "python_version": "3.7"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        # Should recognize this as version incompatibility
        assert "version" in analysis["suggestion"].lower() or "3.8" in analysis["suggestion"]
        assert ":=" in analysis["suggestion"] or "walrus" in analysis["suggestion"].lower()
    
    def test_regression_circular_import_false_positive(self):
        """
        Bug: False positive circular import detection when modules had similar names.
        Fixed: Version 1.4.1
        """
        error_data = {
            "language": "python",
            "error_type": "ImportError",
            "message": "cannot import name 'helper' from 'utils.helper'",
            "file": "utils/helper_functions.py",
            "line": 5,
            "import_chain": [
                "main.py",
                "utils.helper",
                "utils.helper_functions"  # Not actually circular
            ]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        # Should not detect as circular import
        assert "circular import" not in analysis["root_cause"]
    
    def test_regression_dataclass_field_error(self):
        """
        Bug: Dataclass field errors were not properly detected with default_factory.
        Fixed: Version 1.5.2
        """
        error_data = {
            "language": "python",
            "error_type": "TypeError",
            "message": "Field cannot have a default factory",
            "file": "models.py",
            "line": 45,
            "decorator": "@dataclass",
            "field_name": "items"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert "dataclass" in analysis["root_cause"] or "field" in analysis["root_cause"]
        assert "default_factory" in analysis["suggestion"] or "field()" in analysis["suggestion"]


class TestJavaScriptRegressions:
    """Regression tests for JavaScript error handling bugs."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin_system = LanguagePluginSystem()
        self.plugin = self.plugin_system.get_plugin("javascript")
    
    def test_regression_optional_chaining_syntax(self):
        """
        Bug: Optional chaining ?. was incorrectly parsed in older environments.
        Fixed: Version 1.3.5
        """
        error_data = {
            "language": "javascript",
            "error_type": "SyntaxError",
            "message": "Unexpected token '?'",
            "file": "data.js",
            "line": 30,
            "code_snippet": "const value = obj?.property?.nested",
            "environment": "node_12"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert "optional chaining" in analysis["suggestion"].lower() or "?." in analysis["suggestion"]
        assert "version" in analysis["suggestion"].lower() or "compatibility" in analysis["suggestion"].lower()
    
    def test_regression_promise_constructor_antipattern(self):
        """
        Bug: Promise constructor anti-pattern was not detected.
        Fixed: Version 1.4.0
        """
        error_data = {
            "language": "javascript",
            "error_type": "UnhandledPromiseRejection",
            "message": "Promise rejected with no catch handler",
            "file": "async.js",
            "line": 50,
            "code_context": "return new Promise(async (resolve, reject) => {"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert "async" in analysis["suggestion"].lower()
        assert "Promise" in analysis["suggestion"]
    
    def test_regression_bigint_type_coercion(self):
        """
        Bug: BigInt type coercion errors were not properly handled.
        Fixed: Version 1.5.0
        """
        error_data = {
            "language": "javascript",
            "error_type": "TypeError",
            "message": "Cannot mix BigInt and other types",
            "file": "calc.js",
            "line": 15,
            "operation": "1n + 1"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert "BigInt" in analysis["root_cause"] or "bigint" in analysis["root_cause"]
        assert "convert" in analysis["suggestion"].lower() or "Number(" in analysis["suggestion"]


class TestJavaRegressions:
    """Regression tests for Java error handling bugs."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin_system = LanguagePluginSystem()
        self.plugin = self.plugin_system.get_plugin("java")
    
    def test_regression_lambda_type_inference(self):
        """
        Bug: Lambda type inference errors in generic contexts were misdiagnosed.
        Fixed: Version 1.4.3
        """
        error_data = {
            "language": "java",
            "error_type": "CompilationError",
            "message": "incompatible types: bad return type in lambda expression",
            "file": "StreamProcessor.java",
            "line": 78,
            "context": "stream.map(x -> x.toString())",
            "expected_type": "String",
            "actual_type": "Object"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert "lambda" in analysis["root_cause"] or "type inference" in analysis["suggestion"].lower()
        assert "generic" in analysis["suggestion"].lower() or "explicit" in analysis["suggestion"].lower()
    
    def test_regression_concurrent_modification_in_stream(self):
        """
        Bug: ConcurrentModificationException in streams was attributed to wrong cause.
        Fixed: Version 1.5.1
        """
        error_data = {
            "language": "java",
            "error_type": "ConcurrentModificationException",
            "message": "null",
            "stack_trace": [
                {"class": "ArrayList$Itr", "method": "checkForComodification"},
                {"class": "StreamSupport", "method": "stream"}
            ],
            "context": "parallel_stream"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert "stream" in analysis["suggestion"].lower()
        assert "concurrent" in analysis["root_cause"] or "modification" in analysis["root_cause"]
    
    def test_regression_record_pattern_matching(self):
        """
        Bug: Java 14+ record pattern matching errors were not recognized.
        Fixed: Version 1.6.0
        """
        error_data = {
            "language": "java",
            "error_type": "CompilationError",
            "message": "patterns in switch are not supported in -source 11",
            "file": "PatternMatch.java",
            "line": 40,
            "java_version": "11",
            "feature": "record_patterns"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert "version" in analysis["suggestion"].lower()
        assert "14" in analysis["suggestion"] or "record" in analysis["suggestion"].lower()


class TestCppRegressions:
    """Regression tests for C++ error handling bugs."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin_system = LanguagePluginSystem()
        self.plugin = self.plugin_system.get_plugin("cpp")
    
    def test_regression_template_dependent_name(self):
        """
        Bug: Template dependent name lookup errors were incorrectly categorized.
        Fixed: Version 1.3.7
        """
        error_data = {
            "language": "cpp",
            "error_type": "CompilationError",
            "message": "need 'typename' before dependent type name",
            "file": "template.hpp",
            "line": 25,
            "template_context": "template<typename T> class Container"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert "typename" in analysis["suggestion"]
        assert "template" in analysis["root_cause"] or "dependent" in analysis["root_cause"]
    
    def test_regression_move_after_move(self):
        """
        Bug: Use-after-move was not detected in certain contexts.
        Fixed: Version 1.4.5
        """
        error_data = {
            "language": "cpp",
            "error_type": "RuntimeError",
            "message": "use of moved value",
            "file": "resource.cpp",
            "line": 60,
            "static_analysis": "clang-tidy",
            "check": "bugprone-use-after-move"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert "move" in analysis["root_cause"] or "moved" in analysis["root_cause"]
        assert "after move" in analysis["suggestion"].lower()
    
    def test_regression_concepts_constraint_error(self):
        """
        Bug: C++20 concepts constraint errors were not properly parsed.
        Fixed: Version 1.6.2
        """
        error_data = {
            "language": "cpp",
            "error_type": "CompilationError",
            "message": "constraints not satisfied",
            "file": "concepts.cpp",
            "line": 35,
            "concept": "Sortable",
            "cpp_version": "c++20"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert "concept" in analysis["root_cause"] or "constraint" in analysis["root_cause"]
        assert analysis["category"] == "compilation"


class TestGoRegressions:
    """Regression tests for Go error handling bugs."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin_system = LanguagePluginSystem()
        self.plugin = self.plugin_system.get_plugin("go")
    
    def test_regression_interface_nil_comparison(self):
        """
        Bug: Interface nil comparison was incorrectly diagnosed.
        Fixed: Version 1.3.9
        """
        error_data = {
            "language": "go",
            "error_type": "LogicError",
            "message": "interface is not nil but underlying value is nil",
            "file": "validator.go",
            "line": 45,
            "interface_type": "error",
            "comparison": "err != nil"
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert "interface" in analysis["root_cause"]
        assert "nil" in analysis["suggestion"]
    
    def test_regression_goroutine_variable_capture(self):
        """
        Bug: Goroutine variable capture in loops was not detected.
        Fixed: Version 1.4.8
        """
        error_data = {
            "language": "go",
            "error_type": "DataRace",
            "message": "loop variable captured by func literal",
            "file": "worker.go",
            "line": 30,
            "loop_var": "i",
            "goroutine": True
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert "capture" in analysis["root_cause"] or "closure" in analysis["root_cause"]
        assert "loop" in analysis["suggestion"].lower()
    
    def test_regression_embedded_struct_ambiguity(self):
        """
        Bug: Embedded struct field ambiguity was not properly handled.
        Fixed: Version 1.5.5
        """
        error_data = {
            "language": "go",
            "error_type": "CompilationError",
            "message": "ambiguous selector",
            "file": "model.go",
            "line": 90,
            "field": "ID",
            "embedded_types": ["User", "Admin"]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert "ambiguous" in analysis["root_cause"] or "embedded" in analysis["root_cause"]
        assert "explicit" in analysis["suggestion"].lower()


class TestMultiLanguageRegressions:
    """Regression tests for cross-language issues."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin_system = LanguagePluginSystem()
    
    def test_regression_unicode_normalization_mismatch(self):
        """
        Bug: Unicode normalization differences between languages caused false errors.
        Fixed: Version 1.5.8
        """
        test_cases = [
            {
                "language": "python",
                "error_type": "UnicodeError",
                "message": "unicode normalization mismatch",
                "string1": "café",  # NFC
                "string2": "café"   # NFD
            },
            {
                "language": "javascript",
                "error_type": "ComparisonError",
                "message": "strings appear equal but are not",
                "string1": "café",
                "string2": "café"
            }
        ]
        
        for test_case in test_cases:
            plugin = self.plugin_system.get_plugin(test_case["language"])
            if plugin:
                analysis = plugin.analyze_error(test_case)
                assert "normaliz" in analysis["suggestion"].lower() or "unicode" in analysis["suggestion"].lower()
    
    def test_regression_timezone_serialization_errors(self):
        """
        Bug: Timezone handling differences between languages caused errors.
        Fixed: Version 1.6.3
        """
        test_cases = [
            {
                "language": "python",
                "error_type": "ValueError",
                "message": "naive datetime with tzinfo",
                "context": "timezone_conversion"
            },
            {
                "language": "java",
                "error_type": "DateTimeException",
                "message": "Unable to obtain ZonedDateTime",
                "context": "timezone_parsing"
            },
            {
                "language": "javascript",
                "error_type": "RangeError",
                "message": "Invalid time zone specified",
                "context": "Intl.DateTimeFormat"
            }
        ]
        
        for test_case in test_cases:
            plugin = self.plugin_system.get_plugin(test_case["language"])
            if plugin:
                analysis = plugin.analyze_error(test_case)
                assert "timezone" in analysis["suggestion"].lower() or "tz" in analysis["suggestion"].lower()


class TestFrameworkSpecificRegressions:
    """Regression tests for framework-specific bugs."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin_system = LanguagePluginSystem()
    
    def test_regression_django_middleware_ordering(self):
        """
        Bug: Django middleware ordering errors were not detected.
        Fixed: Version 1.4.9
        """
        error_data = {
            "language": "python",
            "error_type": "ImproperlyConfigured",
            "message": "Middleware must be in correct order",
            "framework": "django",
            "middleware_order": [
                "SecurityMiddleware",
                "SessionMiddleware",
                "CommonMiddleware"  # CSRF should be here
            ]
        }
        
        plugin = self.plugin_system.get_plugin("python")
        analysis = plugin.analyze_error(error_data)
        
        assert "middleware" in analysis["root_cause"]
        assert "order" in analysis["suggestion"].lower()
    
    def test_regression_react_hooks_conditional(self):
        """
        Bug: React hooks conditional usage was not properly detected.
        Fixed: Version 1.5.4
        """
        error_data = {
            "language": "javascript",
            "error_type": "ReactHookError",
            "message": "Rendered fewer hooks than expected",
            "framework": "react",
            "hook": "useState",
            "file": "Component.jsx",
            "line": 25,
            "in_conditional": True
        }
        
        plugin = self.plugin_system.get_plugin("javascript")
        analysis = plugin.analyze_error(error_data)
        
        assert "hook" in analysis["root_cause"]
        assert "conditional" in analysis["suggestion"].lower()
    
    def test_regression_spring_circular_dependency(self):
        """
        Bug: Spring circular dependency detection was incomplete.
        Fixed: Version 1.5.7
        """
        error_data = {
            "language": "java",
            "error_type": "BeanCurrentlyInCreationException",
            "message": "Error creating bean: Circular reference",
            "framework": "spring",
            "beans": ["serviceA", "serviceB", "serviceC", "serviceA"]
        }
        
        plugin = self.plugin_system.get_plugin("java")
        analysis = plugin.analyze_error(error_data)
        
        assert "circular" in analysis["root_cause"]
        assert "@Lazy" in analysis["suggestion"] or "refactor" in analysis["suggestion"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])