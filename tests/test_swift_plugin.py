"""
Test cases for Swift Language Plugin

This module contains comprehensive tests for the Swift language plugin,
including error detection, analysis, and patch generation for various
Swift error types and frameworks.
"""

import os

# Import the Swift plugin and related classes
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from modules.analysis.plugins.swift_dependency_analyzer import SwiftDependencyAnalyzer
from modules.analysis.plugins.swift_plugin import (
    SwiftExceptionHandler,
    SwiftLanguagePlugin,
    SwiftPatchGenerator,
)


class TestSwiftExceptionHandler(unittest.TestCase):
    """Test cases for SwiftExceptionHandler."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = SwiftExceptionHandler()

    def test_force_unwrap_nil_detection(self):
        """Test detection of force unwrapping nil optionals."""
        error_data = {
            "error_type": "Fatal Error",
            "message": "fatal error: unexpectedly found nil while unwrapping an Optional value",
            "stack_trace": ["MyApp.swift:42:5"],
        }

        result = self.handler.analyze_exception(error_data)

        self.assertEqual(result["category"], "optionals")
        self.assertEqual(result["subcategory"], "force_unwrap")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["severity"], "high")
        self.assertIn("swift_force_unwrap_nil", result["root_cause"])

    def test_array_index_out_of_bounds_detection(self):
        """Test detection of array index out of bounds errors."""
        error_data = {
            "error_type": "Fatal Error",
            "message": "fatal error: Index out of range",
            "stack_trace": ["ViewController.swift:25:12"],
        }

        result = self.handler.analyze_exception(error_data)

        self.assertEqual(result["category"], "collections")
        self.assertEqual(result["subcategory"], "index_out_of_bounds")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["severity"], "high")
        self.assertIn("swift_array_index_out_of_bounds", result["root_cause"])

    def test_exc_bad_access_detection(self):
        """Test detection of EXC_BAD_ACCESS errors."""
        error_data = {
            "error_type": "EXC_BAD_ACCESS",
            "message": "EXC_BAD_ACCESS (code=1, address=0x0)",
            "stack_trace": ["0x1000 MyApp +42"],
        }

        result = self.handler.analyze_exception(error_data)

        self.assertEqual(result["category"], "memory")
        self.assertEqual(result["severity"], "critical")
        self.assertIn("swift_exc_bad_access", result["root_cause"])

    def test_main_thread_checker_detection(self):
        """Test detection of main thread checker violations."""
        error_data = {
            "error_type": "Runtime Warning",
            "message": "Main Thread Checker: UI API called on a background thread",
            "stack_trace": ["NetworkManager.swift:156:8"],
        }

        result = self.handler.analyze_exception(error_data)

        self.assertEqual(result["category"], "threading")
        self.assertEqual(result["subcategory"], "main_thread_violation")
        self.assertEqual(result["confidence"], "high")
        self.assertIn("swift_main_thread_violation", result["root_cause"])

    def test_task_cancellation_detection(self):
        """Test detection of task cancellation errors."""
        error_data = {
            "error_type": "CancellationError",
            "message": "Task was cancelled",
            "stack_trace": ["AsyncManager.swift:89:15"],
        }

        result = self.handler.analyze_exception(error_data)

        self.assertEqual(result["category"], "concurrency")
        self.assertEqual(result["subcategory"], "task_cancellation")
        self.assertIn("swift_task_cancelled", result["root_cause"])

    def test_fatal_error_analysis(self):
        """Test specific fatal error analysis."""
        error_data = {
            "error_type": "Fatal Error",
            "message": "fatal error: unexpectedly found nil while unwrapping an Optional value",
            "stack_trace": ["ContentView.swift:67:22"],
        }

        result = self.handler.analyze_fatal_error(error_data)

        self.assertEqual(result["category"], "fatal")
        self.assertEqual(result["subcategory"], "force_unwrap")
        self.assertEqual(result["confidence"], "high")
        self.assertIn("Replace force unwrapping", result["suggested_fix"])

    def test_generic_analysis_fallback(self):
        """Test generic analysis for unmatched errors."""
        error_data = {
            "error_type": "Unknown Error",
            "message": "Some unknown Swift error occurred",
            "stack_trace": [],
        }

        result = self.handler.analyze_exception(error_data)

        self.assertEqual(result["category"], "swift")
        self.assertEqual(result["subcategory"], "unknown")
        self.assertEqual(result["confidence"], "low")
        self.assertEqual(result["rule_id"], "swift_generic_handler")


class TestSwiftPatchGenerator(unittest.TestCase):
    """Test cases for SwiftPatchGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = SwiftPatchGenerator()

    def test_force_unwrap_patch_generation(self):
        """Test patch generation for force unwrapping errors."""
        error_data = {
            "message": "unexpectedly found nil while unwrapping an Optional value",
            "stack_trace": [{"file": "test.swift", "line": 3}],
        }

        analysis = {"root_cause": "swift_force_unwrap_nil", "confidence": "high"}

        source_code = """func processUser() {
    let user = getUser()
    print(user!.name)  // This line causes the error
}"""

        result = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "line_replacement")
        self.assertIn("Replaced force unwrapping", result["description"])

    def test_array_bounds_patch_generation(self):
        """Test patch generation for array bounds errors."""
        error_data = {
            "message": "Index out of range",
            "stack_trace": [{"file": "test.swift", "line": 3}],
        }

        analysis = {
            "root_cause": "swift_array_index_out_of_bounds",
            "confidence": "high",
        }

        source_code = """func getFirstItem() {
    let items = getItems()
    return items[0]  // This line causes the error
}"""

        result = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(result)
        self.assertIn("bounds checking", result["description"])

    def test_weak_reference_suggestion(self):
        """Test suggestion generation for retain cycle issues."""
        error_data = {"message": "retain cycle detected"}
        analysis = {"root_cause": "swift_weak_reference_cycle"}
        source_code = "closure code"

        result = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "suggestion")
        self.assertIn("weak self", result["description"])

    def test_main_thread_suggestion(self):
        """Test suggestion generation for main thread violations."""
        error_data = {"message": "UI API called on background thread"}
        analysis = {"root_cause": "swift_main_thread_violation"}
        source_code = "ui update code"

        result = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "suggestion")
        self.assertIn("main thread", result["description"])

    def test_template_based_patch(self):
        """Test template-based patch generation."""
        error_data = {"message": "force unwrap error"}
        analysis = {"root_cause": "swift_force_unwrap_nil"}
        source_code = "test code"

        # Mock the templates to exist
        self.generator.templates = {"safe_optional_unwrapping": "template content"}

        result = self.generator._template_based_patch(error_data, analysis, source_code)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "template")
        self.assertIn("template content", result["template"])


class TestSwiftDependencyAnalyzer(unittest.TestCase):
    """Test cases for SwiftDependencyAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SwiftDependencyAnalyzer()

    def test_missing_package_swift(self):
        """Test analysis when Package.swift is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.analyzer.analyze_project_dependencies(temp_dir)

            self.assertFalse(result["has_spm"])
            self.assertIn("No Package.swift file found", result["error"])
            self.assertIn("swift package init", result["suggestions"][0])

    def test_package_swift_parsing(self):
        """Test parsing of Package.swift file."""
        package_content = """// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "TestPackage",
    platforms: [.iOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/Alamofire/Alamofire.git", from: "5.0.0")
    ],
    targets: [
        .target(name: "TestPackage", dependencies: ["Alamofire"])
    ]
)"""

        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode="w", suffix=".swift", delete=False) as f:
            f.write(package_content)
            package_file = Path(f.name)

        try:
            package_info = self.analyzer._parse_package_swift(package_file)
        finally:
            os.unlink(package_file)

        self.assertEqual(package_info["name"], "TestPackage")
        self.assertEqual(len(package_info["dependencies"]), 1)
        self.assertEqual(
            package_info["dependencies"][0]["url"],
            "https://github.com/Alamofire/Alamofire.git",
        )
        self.assertIn("iOS", package_info["platforms"])

    def test_missing_module_error_analysis(self):
        """Test analysis of 'No such module' errors."""
        error_data = {
            "message": "No such module 'Alamofire'",
            "error_type": "Compilation Error",
        }

        result = self.analyzer.analyze_dependency_error(error_data, "/test/path")

        self.assertEqual(result["category"], "dependency")
        self.assertEqual(result["subcategory"], "missing_module")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["module_name"], "Alamofire")
        self.assertEqual(result["suggested_package"], "Alamofire")

    def test_package_swift_syntax_error_analysis(self):
        """Test analysis of Package.swift syntax errors."""
        error_data = {
            "message": "Package.swift:5:12: error: expected ']'",
            "error_type": "Syntax Error",
        }

        result = self.analyzer.analyze_dependency_error(error_data, "/test/path")

        self.assertEqual(result["category"], "dependency")
        self.assertEqual(result["subcategory"], "package_swift_syntax")
        self.assertIn("Package.swift syntax", result["suggested_fix"])

    def test_version_conflict_error_analysis(self):
        """Test analysis of version conflict errors."""
        error_data = {
            "message": "dependency version conflict detected",
            "error_type": "Resolution Error",
        }

        result = self.analyzer.analyze_dependency_error(error_data, "/test/path")

        self.assertEqual(result["category"], "dependency")
        self.assertEqual(result["subcategory"], "version_conflict")
        self.assertIn("version conflicts", result["suggested_fix"])


class TestSwiftLanguagePlugin(unittest.TestCase):
    """Test cases for SwiftLanguagePlugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = SwiftLanguagePlugin()

    def test_plugin_initialization(self):
        """Test plugin initialization."""
        self.assertEqual(self.plugin.get_language_id(), "swift")
        self.assertEqual(self.plugin.get_language_name(), "Swift")
        self.assertEqual(self.plugin.get_language_version(), "5.0+")
        self.assertIn("uikit", self.plugin.get_supported_frameworks())
        self.assertIn("swiftui", self.plugin.get_supported_frameworks())

    def test_can_handle_swift_errors(self):
        """Test that plugin can handle Swift errors."""
        # Test explicit language
        error_data = {"language": "swift"}
        self.assertTrue(self.plugin.can_handle(error_data))

        # Test runtime detection
        error_data = {"runtime": "iOS 15.0"}
        self.assertTrue(self.plugin.can_handle(error_data))

        # Test error pattern detection
        error_data = {"message": "fatal error: unexpectedly found nil"}
        self.assertTrue(self.plugin.can_handle(error_data))

        # Test file extension detection
        error_data = {"stack_trace": "MyApp.swift:42:5"}
        self.assertTrue(self.plugin.can_handle(error_data))

        # Test negative case
        error_data = {"language": "python", "message": "ImportError"}
        self.assertFalse(self.plugin.can_handle(error_data))

    def test_error_analysis_integration(self):
        """Test full error analysis integration."""
        error_data = {
            "error_type": "Fatal Error",
            "message": "fatal error: unexpectedly found nil while unwrapping an Optional value",
            "stack_trace": ["ContentView.swift:42:5"],
            "runtime": "iOS 15.0",
        }

        result = self.plugin.analyze_error(error_data)

        self.assertEqual(result["plugin"], "swift")
        self.assertEqual(result["language"], "swift")
        self.assertIn("plugin_version", result)
        self.assertEqual(result["category"], "fatal")

    def test_dependency_error_analysis(self):
        """Test dependency error analysis integration."""
        error_data = {
            "message": "No such module 'Alamofire'",
            "error_type": "Compilation Error",
            "language": "swift",
        }

        result = self.plugin.analyze_error(error_data)

        self.assertEqual(result["plugin"], "swift")
        self.assertEqual(result["category"], "dependency")
        self.assertEqual(result["subcategory"], "missing_module")

    def test_fix_generation(self):
        """Test fix generation."""
        error_data = {
            "message": "unexpectedly found nil while unwrapping an Optional value",
            "stack_trace": [{"file": "test.swift", "line": 42}],
        }

        analysis = {"root_cause": "swift_force_unwrap_nil", "confidence": "high"}

        source_code = "let value = optional!"

        result = self.plugin.generate_fix(error_data, analysis, source_code)

        # Should return a fix or None
        if result:
            self.assertIn("type", result)

    @patch("modules.analysis.plugins.swift_plugin.SwiftDependencyAnalyzer")
    def test_dependency_analysis(self, mock_analyzer_class):
        """Test dependency analysis."""
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_project_dependencies.return_value = {
            "has_spm": True,
            "dependencies": [],
        }

        # Create new plugin instance with mocked analyzer
        plugin = SwiftLanguagePlugin()

        result = plugin.analyze_dependencies("/test/path")

        mock_analyzer.analyze_project_dependencies.assert_called_once_with("/test/path")
        self.assertTrue(result["has_spm"])

    def test_language_info(self):
        """Test language info retrieval."""
        info = self.plugin.get_language_info()

        self.assertEqual(info["language"], "swift")
        self.assertIn("iOS application error handling", info["features"])
        self.assertIn("SwiftUI error detection", info["features"])
        self.assertIn("ios", info["platforms"])
        self.assertIn("macos", info["platforms"])


class TestSwiftPluginIntegration(unittest.TestCase):
    """Integration tests for Swift plugin components."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = SwiftLanguagePlugin()

    def test_end_to_end_force_unwrap_analysis(self):
        """Test end-to-end analysis of force unwrap error."""
        error_data = {
            "timestamp": "2023-01-01T12:00:00Z",
            "error_type": "Fatal Error",
            "message": "fatal error: unexpectedly found nil while unwrapping an Optional value",
            "stack_trace": ["ContentView.swift:67:22"],
            "runtime": "iOS 15.0",
            "context": {"project_path": "/test/project"},
        }

        # Analyze the error
        analysis = self.plugin.analyze_error(error_data)

        # Verify analysis results
        self.assertEqual(analysis["plugin"], "swift")
        self.assertEqual(analysis["category"], "fatal")
        self.assertEqual(analysis["subcategory"], "force_unwrap")
        self.assertEqual(analysis["confidence"], "high")

        # Generate fix
        source_code = """struct ContentView: View {
    var body: some View {
        Text(user!.name)  // Line 67
    }
}"""

        fix = self.plugin.generate_fix(error_data, analysis, source_code)

        if fix:
            self.assertIn("type", fix)
            self.assertIn("description", fix)

    def test_end_to_end_dependency_analysis(self):
        """Test end-to-end analysis of dependency error."""
        error_data = {
            "timestamp": "2023-01-01T12:00:00Z",
            "error_type": "Compilation Error",
            "message": "No such module 'Alamofire'",
            "language": "swift",
            "context": {"project_path": "/test/project"},
        }

        # Analyze the error
        analysis = self.plugin.analyze_error(error_data)

        # Verify analysis results
        self.assertEqual(analysis["plugin"], "swift")
        self.assertEqual(analysis["category"], "dependency")
        self.assertEqual(analysis["subcategory"], "missing_module")
        self.assertIn("suggestions", analysis)

    def test_concurrency_error_analysis(self):
        """Test analysis of Swift concurrency errors."""
        error_data = {
            "error_type": "CancellationError",
            "message": "Task was cancelled",
            "stack_trace": ["AsyncNetworkManager.swift:89:15"],
            "language": "swift",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["plugin"], "swift")
        self.assertIn("concurrency", analysis["category"])


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
