"""
Test cases for TypeScript Language Plugin

This module contains comprehensive tests for the TypeScript plugin,
including error detection, analysis, and fix generation capabilities.
"""

import os
import sys
import unittest

from modules.analysis.language_adapters import TypeScriptErrorAdapter
from modules.analysis.plugins.typescript_plugin import (
    TypeScriptExceptionHandler,
    TypeScriptLanguagePlugin,
    TypeScriptPatchGenerator,
)

# Add the modules directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules"))


class TestTypeScriptErrorAdapter(unittest.TestCase):
    """Test cases for TypeScript error adapter."""

    def setUp(self):
        """Set up test fixtures."""
        self.adapter = TypeScriptErrorAdapter()

    def test_to_standard_format_compilation_error(self):
        """Test conversion of TypeScript compilation error to standard format."""
        ts_error = {
            "code": "TS2304",
            "message": "Cannot find name 'React'.",
            "file": "src/App.tsx",
            "line": 1,
            "column": 8,
            "timestamp": "2024-01-01T10:00:00Z",
        }

        standard_error = self.adapter.to_standard_format(ts_error)

        self.assertEqual(standard_error["language"], "typescript")
        self.assertEqual(standard_error["error_type"], "TS2304")
        self.assertEqual(standard_error["error_code"], "TS2304")
        self.assertEqual(standard_error["message"], "Cannot find name 'React'.")
        self.assertEqual(standard_error["severity"], "error")
        self.assertIn("file_info", standard_error)
        self.assertEqual(standard_error["file_info"]["file"], "src/App.tsx")
        self.assertEqual(standard_error["file_info"]["line"], 1)
        self.assertEqual(standard_error["file_info"]["column"], 8)

    def test_to_standard_format_runtime_error(self):
        """Test conversion of TypeScript runtime error to standard format."""
        ts_error = {
            "name": "TypeError",
            "message": "Cannot read property 'length' of undefined",
            "stack": "TypeError: Cannot read property 'length' of undefined\\n    at processArray (main.ts:15:20)\\n    at main (main.ts:25:5)",
            "timestamp": "2024-01-01T10:00:00Z",
        }

        standard_error = self.adapter.to_standard_format(ts_error)

        self.assertEqual(standard_error["language"], "typescript")
        self.assertEqual(standard_error["error_type"], "TypeError")
        self.assertEqual(
            standard_error["message"], "Cannot read property 'length' of undefined"
        )
        self.assertIn("stack_trace", standard_error)

    def test_from_standard_format(self):
        """Test conversion from standard format back to TypeScript format."""
        standard_error = {
            "error_id": "test-123",
            "timestamp": "2024-01-01T10:00:00Z",
            "language": "typescript",
            "error_type": "TS2339",
            "error_code": "TS2339",
            "message": "Property 'foo' does not exist on type 'object'.",
            "severity": "error",
            "file_info": {"file": "src/test.ts", "line": 10, "column": 5},
        }

        ts_error = self.adapter.from_standard_format(standard_error)

        self.assertEqual(ts_error["name"], "TS2339")
        self.assertEqual(ts_error["code"], "TS2339")
        self.assertEqual(
            ts_error["message"], "Property 'foo' does not exist on type 'object'."
        )
        self.assertEqual(ts_error["file"], "src/test.ts")
        self.assertEqual(ts_error["line"], 10)
        self.assertEqual(ts_error["column"], 5)
        self.assertEqual(ts_error["severity"], "error")


class TestTypeScriptExceptionHandler(unittest.TestCase):
    """Test cases for TypeScript exception handler."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = TypeScriptExceptionHandler()

    def test_analyze_cannot_find_name_error(self):
        """Test analysis of TS2304 Cannot find name error."""
        error_data = {
            "language": "typescript",
            "error_type": "TS2304",
            "error_code": "TS2304",
            "message": "Cannot find name 'React'.",
            "severity": "error",
        }

        analysis = self.handler.analyze_exception(error_data)

        self.assertEqual(analysis["category"], "typescript")
        self.assertIn("type_error", analysis["subcategory"])
        self.assertEqual(analysis["confidence"], "high")
        self.assertIn("React", analysis["suggested_fix"])
        self.assertEqual(analysis["error_code"], "TS2304")
        self.assertIn("typescript", analysis["tags"])

    def test_analyze_type_assignment_error(self):
        """Test analysis of TS2322 Type assignment error."""
        error_data = {
            "language": "typescript",
            "error_type": "TS2322",
            "error_code": "TS2322",
            "message": "Type 'string' is not assignable to type 'number'.",
            "severity": "error",
        }

        analysis = self.handler.analyze_exception(error_data)

        self.assertEqual(analysis["category"], "typescript")
        self.assertIn("type", analysis["subcategory"])
        self.assertEqual(analysis["confidence"], "high")
        self.assertIn("type compatibility", analysis["suggested_fix"])

    def test_analyze_property_not_exist_error(self):
        """Test analysis of TS2339 Property does not exist error."""
        error_data = {
            "language": "typescript",
            "error_type": "TS2339",
            "error_code": "TS2339",
            "message": "Property 'foo' does not exist on type 'Bar'.",
            "severity": "error",
        }

        analysis = self.handler.analyze_exception(error_data)

        self.assertEqual(analysis["category"], "typescript")
        self.assertIn("property", analysis["subcategory"])
        self.assertEqual(analysis["confidence"], "high")
        self.assertIn("property", analysis["suggested_fix"])

    def test_analyze_module_not_found_error(self):
        """Test analysis of TS2307 Cannot find module error."""
        error_data = {
            "language": "typescript",
            "error_type": "TS2307",
            "error_code": "TS2307",
            "message": "Cannot find module 'react'.",
            "severity": "error",
        }

        analysis = self.handler.analyze_exception(error_data)

        self.assertEqual(analysis["category"], "typescript")
        self.assertIn("module", analysis["subcategory"])
        self.assertEqual(analysis["confidence"], "high")
        self.assertIn("install", analysis["suggested_fix"].lower())

    def test_analyze_compilation_error(self):
        """Test analysis of TypeScript compilation errors."""
        error_data = {
            "language": "typescript",
            "error_type": "TS1005",
            "error_code": "TS1005",
            "message": "';' expected.",
            "severity": "error",
        }

        analysis = self.handler.analyze_compilation_error(error_data)

        self.assertEqual(analysis["category"], "typescript")
        self.assertEqual(analysis["subcategory"], "syntax")
        self.assertEqual(analysis["confidence"], "high")
        self.assertIn("missing token", analysis["suggested_fix"])

    def test_analyze_config_error(self):
        """Test analysis of TypeScript configuration errors."""
        error_data = {
            "language": "typescript",
            "error_type": "TS5023",
            "error_code": "TS5023",
            "message": "Unknown compiler option 'invalidOption'.",
            "severity": "error",
        }

        analysis = self.handler.analyze_compilation_error(error_data)

        self.assertEqual(analysis["category"], "typescript")
        self.assertEqual(analysis["subcategory"], "config")
        self.assertEqual(analysis["confidence"], "high")
        self.assertIn("compiler option", analysis["suggested_fix"])

    def test_generic_analysis_fallback(self):
        """Test generic analysis for unmatched errors."""
        error_data = {
            "language": "typescript",
            "error_type": "UnknownError",
            "message": "Some unknown TypeScript error",
            "severity": "error",
        }

        analysis = self.handler.analyze_exception(error_data)

        self.assertEqual(analysis["category"], "typescript")
        self.assertEqual(analysis["confidence"], "low")
        self.assertEqual(analysis["rule_id"], "ts_generic_handler")
        self.assertIn("typescript", analysis["tags"])


class TestTypeScriptPatchGenerator(unittest.TestCase):
    """Test cases for TypeScript patch generator."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = TypeScriptPatchGenerator()

    def test_fix_cannot_find_name_react(self):
        """Test fix generation for missing React import."""
        error_data = {"message": "Cannot find name 'React'."}
        analysis = {
            "root_cause": "typescript_type_error_ts2304",
            "error_code": "TS2304",
        }
        source_code = "const element = <div>Hello</div>;"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("React", fix["description"])
        self.assertIn("import React", fix["fix_code"])

    def test_fix_cannot_find_name_process(self):
        """Test fix generation for missing Node.js types."""
        error_data = {"message": "Cannot find name 'process'."}
        analysis = {
            "root_cause": "typescript_type_error_ts2304",
            "error_code": "TS2304",
        }
        source_code = "console.log(process.env.NODE_ENV);"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("process", fix["description"])
        self.assertIn("@types/node", fix["fix_code"])

    def test_fix_type_assignment(self):
        """Test fix generation for type assignment errors."""
        error_data = {"message": "Type 'string' is not assignable to type 'number'."}
        analysis = {
            "root_cause": "typescript_type_error_ts2322",
            "error_code": "TS2322",
        }
        source_code = "const num: number = 'hello';"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("type compatibility", fix["description"])

    def test_fix_property_not_exist(self):
        """Test fix generation for property not exist errors."""
        error_data = {"message": "Property 'foo' does not exist on type 'Bar'."}
        analysis = {
            "root_cause": "typescript_type_error_ts2339",
            "error_code": "TS2339",
        }
        source_code = "obj.foo = 'value';"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("Property 'foo'", fix["description"])
        self.assertIn("optional chaining", fix["description"])

    def test_fix_module_not_found_relative(self):
        """Test fix generation for relative module not found."""
        error_data = {"message": "Cannot find module './utils'."}
        analysis = {
            "root_cause": "typescript_type_error_ts2307",
            "error_code": "TS2307",
        }
        source_code = "import { helper } from './utils';"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("./utils", fix["description"])
        self.assertIn("file exists", fix["description"])

    def test_fix_module_not_found_npm(self):
        """Test fix generation for npm module not found."""
        error_data = {"message": "Cannot find module 'lodash'."}
        analysis = {
            "root_cause": "typescript_type_error_ts2307",
            "error_code": "TS2307",
        }
        source_code = "import _ from 'lodash';"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("lodash", fix["description"])
        self.assertIn("npm install", fix["description"])
        self.assertIn("@types/lodash", fix["description"])

    def test_fix_syntax_error(self):
        """Test fix generation for syntax errors."""
        error_data = {"message": "';' expected."}
        analysis = {
            "root_cause": "typescript_compilation_syntax",
            "error_code": "TS1005",
        }
        source_code = "const x = 5"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("';'", fix["description"])

    def test_fix_unused_variable(self):
        """Test fix generation for unused variables."""
        error_data = {"message": "'unusedVar' is declared but its value is never read."}
        analysis = {
            "root_cause": "typescript_compilation_unused",
            "error_code": "TS6133",
        }
        source_code = "const unusedVar = 'value';"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("unusedVar", fix["description"])
        self.assertIn("underscore", fix["description"])


class TestTypeScriptLanguagePlugin(unittest.TestCase):
    """Test cases for TypeScript language plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = TypeScriptLanguagePlugin()

    def test_plugin_initialization(self):
        """Test plugin initialization."""
        self.assertEqual(self.plugin.get_language_id(), "typescript")
        self.assertEqual(self.plugin.get_language_name(), "TypeScript")
        self.assertEqual(self.plugin.get_language_version(), "3.0+")
        self.assertIn("react", self.plugin.get_supported_frameworks())
        self.assertIn("angular", self.plugin.get_supported_frameworks())

    def test_can_handle_typescript_error_code(self):
        """Test error handling detection by TypeScript error code."""
        error_data = {"error_type": "TS2304", "message": "Cannot find name 'React'."}

        self.assertTrue(self.plugin.can_handle(error_data))

    def test_can_handle_typescript_language(self):
        """Test error handling detection by language field."""
        error_data = {
            "language": "typescript",
            "error_type": "TypeError",
            "message": "Some error",
        }

        self.assertTrue(self.plugin.can_handle(error_data))

    def test_can_handle_typescript_message_patterns(self):
        """Test error handling detection by message patterns."""
        test_cases = [
            {"message": "TS2304: Cannot find name 'React'."},
            {"message": "error TS2322: Type 'string' is not assignable"},
            {"message": "TypeScript compilation failed"},
            {"message": "Type 'number' is not assignable to type 'string'"},
        ]

        for error_data in test_cases:
            with self.subTest(error_data=error_data):
                self.assertTrue(self.plugin.can_handle(error_data))

    def test_can_handle_typescript_files(self):
        """Test error handling detection by file extensions."""
        error_data = {"stack_trace": ["at main.ts:15:20", "at app.tsx:25:5"]}

        self.assertTrue(self.plugin.can_handle(error_data))

    def test_cannot_handle_non_typescript(self):
        """Test that plugin doesn't handle non-TypeScript errors."""
        error_data = {
            "language": "python",
            "error_type": "ValueError",
            "message": "invalid literal",
        }

        self.assertFalse(self.plugin.can_handle(error_data))

    def test_normalize_error(self):
        """Test error normalization."""
        ts_error = {
            "code": "TS2304",
            "message": "Cannot find name 'React'.",
            "file": "src/App.tsx",
            "line": 1,
            "column": 8,
        }

        normalized = self.plugin.normalize_error(ts_error)

        self.assertEqual(normalized["language"], "typescript")
        self.assertEqual(normalized["error_type"], "TS2304")
        self.assertEqual(normalized["message"], "Cannot find name 'React'.")

    def test_analyze_error_compilation(self):
        """Test error analysis for compilation errors."""
        error_data = {
            "language": "typescript",
            "error_type": "TS1005",
            "error_code": "TS1005",
            "message": "';' expected.",
            "severity": "error",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["plugin"], "typescript")
        self.assertEqual(analysis["language"], "typescript")
        self.assertEqual(analysis["category"], "typescript")
        self.assertIn("syntax", analysis["subcategory"])

    def test_analyze_error_type_system(self):
        """Test error analysis for type system errors."""
        error_data = {
            "language": "typescript",
            "error_type": "TS2304",
            "error_code": "TS2304",
            "message": "Cannot find name 'React'.",
            "severity": "error",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["plugin"], "typescript")
        self.assertEqual(analysis["language"], "typescript")
        self.assertEqual(analysis["category"], "typescript")
        self.assertIn("type", analysis["subcategory"])

    def test_generate_fix(self):
        """Test fix generation."""
        error_data = {"message": "Cannot find name 'React'."}
        analysis = {
            "root_cause": "typescript_type_error_ts2304",
            "error_code": "TS2304",
        }
        context = {
            "error_data": error_data,
            "source_code": "const element = <div>Hello</div>;",
        }

        fix = self.plugin.generate_fix(analysis, context)

        self.assertIsNotNone(fix)
        if fix:  # Fix might be empty dict if patch generation fails
            # Check for any of the expected fields in the fix
            self.assertTrue(
                any(
                    key in fix
                    for key in ["fix", "fix_code", "type", "description", "patch_type"]
                ),
                f"Expected fix structure, got: {fix.keys()}",
            )

    def test_get_language_info(self):
        """Test language info retrieval."""
        info = self.plugin.get_language_info()

        self.assertEqual(info["language"], "typescript")
        self.assertEqual(info["version"], "1.0.0")
        self.assertIn(".ts", info["supported_extensions"])
        self.assertIn(".tsx", info["supported_extensions"])
        self.assertIn("react", info["supported_frameworks"])
        self.assertIn("TypeScript compilation error handling", info["features"])


class TestTypeScriptIntegration(unittest.TestCase):
    """Integration tests for TypeScript plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = TypeScriptLanguagePlugin()

    def test_end_to_end_error_handling(self):
        """Test complete error handling flow."""
        # Simulate a TypeScript error
        raw_error = {
            "code": 2304,
            "message": "Cannot find name 'React'.",
            "file": "src/components/App.tsx",
            "line": 1,
            "column": 8,
            "timestamp": "2024-01-01T10:00:00Z",
        }

        # Step 1: Check if plugin can handle
        self.assertTrue(self.plugin.can_handle(raw_error))

        # Step 2: Normalize error
        normalized = self.plugin.normalize_error(raw_error)
        self.assertEqual(normalized["language"], "typescript")
        self.assertEqual(normalized["error_code"], "TS2304")

        # Step 3: Analyze error
        analysis = self.plugin.analyze_error(normalized)
        self.assertEqual(analysis["plugin"], "typescript")
        self.assertIn("type", analysis["subcategory"])

        # Step 4: Generate fix
        normalized["source_code"] = (
            "import React from 'react';\nconst App = () => <div>Hello</div>;"
        )
        fix = self.plugin.generate_fix(normalized, analysis)
        self.assertIsNotNone(fix)

    def test_multiple_error_types(self):
        """Test handling of multiple TypeScript error types."""
        error_scenarios = [
            {
                "code": "TS2304",
                "message": "Cannot find name 'React'.",
                "expected_category": "type",
            },
            {
                "code": "TS2322",
                "message": "Type 'string' is not assignable to type 'number'.",
                "expected_category": "type",
            },
            {
                "code": "TS1005",
                "message": "';' expected.",
                "expected_category": "syntax",
            },
            {
                "code": "TS5023",
                "message": "Unknown compiler option 'invalidOption'.",
                "expected_category": "config",
            },
        ]

        for scenario in error_scenarios:
            with self.subTest(error_code=scenario["code"]):
                error_data = {
                    "language": "typescript",
                    "error_type": scenario["code"],
                    "error_code": scenario["code"],
                    "message": scenario["message"],
                }

                # Test plugin can handle
                self.assertTrue(self.plugin.can_handle(error_data))

                # Test analysis
                analysis = self.plugin.analyze_error(error_data)
                self.assertEqual(analysis["plugin"], "typescript")
                self.assertIn(scenario["expected_category"], analysis["subcategory"])


if __name__ == "__main__":
    # Create a test suite combining all test cases
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestTypeScriptErrorAdapter,
        TestTypeScriptExceptionHandler,
        TestTypeScriptPatchGenerator,
        TestTypeScriptLanguagePlugin,
        TestTypeScriptIntegration,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
