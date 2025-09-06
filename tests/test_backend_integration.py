"""
Backend Language Integration Tests

This module tests the integration between different backend language plugins,
error adaptations, and cross-language capabilities.
"""

import logging
import os
import sys
from pathlib import Path

import pytest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.analysis.cross_language_orchestrator import CrossLanguageOrchestrator
from modules.analysis.language_plugin_system import get_all_plugins, load_all_plugins
from modules.analysis.language_test_framework import (
    LanguageTestRunner,
    LanguageTestSuite,
)
from modules.analysis.shared_error_schema import (
    SharedErrorSchema,
    denormalize_error,
    normalize_error,
)


@pytest.fixture
def test_suite():
    """Load the backend test suite."""
    suite = LanguageTestSuite("Backend Test Suite")
    test_file = Path(__file__).parent / "test_cases" / "backend_test_suite.json"

    if not test_file.exists():
        pytest.skip(f"Test suite file not found: {test_file}")

    suite.load_from_file(test_file)
    return suite


@pytest.fixture
def test_runner():
    """Create a test runner with all plugins loaded."""
    # Load all available plugins first
    load_all_plugins()
    return LanguageTestRunner(load_plugins=False)


class TestBackendLanguageIntegration:
    """
    Test class for backend language integration.

    This class contains tests for:
    1. Language plugin functionality
    2. Error schema conversion
    3. Cross-language capabilities
    """

    def test_load_plugins(self):
        """Test that plugins can be loaded successfully."""
        # Reset plugins to ensure fresh test
        from modules.analysis.language_plugin_system import plugin_registry

        plugin_registry.plugins = {}

        # Load plugins
        count = load_all_plugins()
        plugins = get_all_plugins()

        # Check that plugins were loaded
        assert count > 0, "No plugins were loaded"
        assert len(plugins) > 0, "No plugins were registered"

        # Check for at least Python and Java plugins
        assert "python" in plugins, "Python plugin not loaded"
        assert "java" in plugins, "Java plugin not loaded"

    def test_shared_error_schema(self):
        """Test shared error schema functionality."""
        schema = SharedErrorSchema()

        # Test language detection
        test_errors = {
            "python": {
                "exception_type": "ValueError",
                "message": "Invalid value",
                "traceback": [
                    "Traceback (most recent call last):",
                    '  File "test.py", line 1',
                ],
            },
            "java": {
                "exception_class": "java.lang.IllegalArgumentException",
                "message": "Invalid argument",
                "stack_trace": "at com.example.Test.method(Test.java:10)",
            },
            "javascript": {
                "name": "TypeError",
                "message": "Cannot read property",
                "stack": "at function (file.js:10:5)",
            },
            "go": {
                "error_type": "runtime error",
                "message": "nil pointer dereference",
                "stack_trace": "goroutine 1 [running]:\nmain.test()\n\t/app/main.go:10",
            },
        }

        for expected_lang, error_data in test_errors.items():
            detected = schema.detect_language(error_data)
            assert (
                detected == expected_lang
            ), f"Expected {expected_lang}, got {detected}"

        # Test normalization
        for lang, error_data in test_errors.items():
            normalized = schema.normalize_error(error_data, lang)

            # Check required fields
            assert "error_id" in normalized
            assert "timestamp" in normalized
            assert "language" in normalized
            assert "error_type" in normalized
            assert "message" in normalized

            # Check language identification
            assert normalized["language"] == lang

        # Test round-trip conversion
        for lang, error_data in test_errors.items():
            normalized = schema.normalize_error(error_data, lang)
            denormalized = schema.denormalize_error(normalized, lang)

            # Check that essential information is preserved
            if lang == "python":
                assert denormalized.get("exception_type") == error_data.get(
                    "exception_type"
                )
            elif lang == "java":
                assert denormalized.get("exception_class") == error_data.get(
                    "exception_class"
                )
            elif lang == "javascript":
                assert denormalized.get("name") == error_data.get("name")
            elif lang == "go":
                assert denormalized.get("error_type") == error_data.get("error_type")

            assert denormalized.get("message") == error_data.get("message")

    def test_test_suite_loading(self, test_suite):
        """Test that the test suite loads correctly."""
        assert len(test_suite.test_cases) > 0, "No test cases were loaded"

        # Check for specific test cases
        languages = set(tc.language for tc in test_suite.test_cases)
        assert "python" in languages, "No Python test cases found"
        assert "java" in languages, "No Java test cases found"
        assert "javascript" in languages, "No JavaScript test cases found"
        assert "go" in languages, "No Go test cases found"

        # Test filtering
        python_cases = test_suite.filter_by_language("python")
        assert len(python_cases) > 0, "No Python test cases after filtering"

        java_cases = test_suite.filter_by_language("java")
        assert len(java_cases) > 0, "No Java test cases after filtering"

    def test_language_specific_analysis(self, test_runner, test_suite):
        """Test that language-specific analyzers work correctly."""
        # Test cases by language
        for lang in ["python", "java", "javascript", "go"]:
            cases = test_suite.filter_by_language(lang)
            if not cases:
                pytest.skip(f"No test cases for language: {lang}")

            plugin = test_runner.get_plugin(lang)
            if not plugin:
                pytest.skip(f"Plugin not available for language: {lang}")

            # Run test cases
            for case in cases:
                analysis = plugin.analyze_error(case.error_data)

                # Check that analysis produced results
                assert (
                    analysis
                ), f"No analysis results for {lang} test case: {case.name}"

                # Check expected analysis fields if provided
                for key, expected_value in case.expected_analysis.items():
                    assert key in analysis, f"Missing expected field {key} in analysis"
                    assert (
                        analysis[key] == expected_value
                    ), f"Expected {key}={expected_value}, got {analysis[key]}"

    def test_cross_language_conversion(self, test_suite):
        """Test converting errors between languages."""
        schema = SharedErrorSchema()
        languages = schema.get_supported_languages()

        # Only test languages that we have test cases for
        test_languages = set(tc.language for tc in test_suite.test_cases)
        languages = [lang for lang in languages if lang in test_languages]

        # Test conversion between each language pair
        for source_lang in languages:
            source_cases = test_suite.filter_by_language(source_lang)
            if not source_cases:
                continue

            for target_lang in languages:
                if source_lang == target_lang:
                    continue

                for case in source_cases:
                    # Convert to normalized format
                    normalized = normalize_error(case.error_data, source_lang)

                    # Convert to target language
                    target_error = denormalize_error(normalized, target_lang)

                    # Convert back to normalized format
                    normalized_roundtrip = normalize_error(target_error, target_lang)

                    # Check key fields preserved in roundtrip
                    assert (
                        normalized_roundtrip["error_type"] == normalized["error_type"]
                    ), f"Error type changed: {normalized['error_type']} -> {normalized_roundtrip['error_type']}"

                    assert (
                        normalized_roundtrip["message"] == normalized["message"]
                    ), f"Message changed: {normalized['message']} -> {normalized_roundtrip['message']}"

    def test_orchestrator_integration(self, test_suite):
        """Test the cross-language orchestrator with multiple languages."""
        orchestrator = CrossLanguageOrchestrator()

        # Check supported languages
        supported = orchestrator.get_supported_languages()
        assert len(supported) >= 2, "Orchestrator should support at least 2 languages"

        # Test analysis of errors from different languages
        for case in test_suite.test_cases:
            try:
                analysis = orchestrator.analyze_error(case.error_data, case.language)

                # Check that analysis produced results
                assert (
                    analysis
                ), f"No analysis results for {case.language} test case: {case.name}"

                # Check that language was automatically detected
                detected_analysis = orchestrator.analyze_error(case.error_data)
                assert (
                    detected_analysis
                ), f"Language detection failed for test case: {case.name}"

                # Results should be similar between explicit and auto-detection
                assert detected_analysis.get("root_cause") == analysis.get(
                    "root_cause"
                ), "Analysis results differ between explicit and auto-detected language"
            except Exception as e:
                pytest.fail(
                    f"Error analyzing {case.language} test case {case.name}: {e}"
                )

    def test_run_full_test_suite(self, test_runner, test_suite):
        """Run the full test suite using the test runner."""
        # Only run if we have a reasonable number of plugins available
        plugins = get_all_plugins()
        if len(plugins) < 2:
            pytest.skip("Not enough plugins available for full test suite")

        # Run the test suite
        results = test_runner.run_test_suite(test_suite)

        # Check basic result metrics
        assert len(results) == len(test_suite.test_cases), "Not all test cases were run"

        # Count passed tests
        passed = sum(1 for r in results if r.passed)

        # At least some tests should pass if plugins are working correctly
        assert passed > 0, "No tests passed"

        # Report pass percentage
        pass_percentage = (passed / len(results)) * 100
        logging.info(f"Test pass percentage: {pass_percentage:.1f}%")

    def test_cross_language_capabilities(self, test_runner, test_suite):
        """Test cross-language capabilities of the system."""
        # Only run if we have a reasonable number of plugins available
        plugins = get_all_plugins()
        if len(plugins) < 2:
            pytest.skip("Not enough plugins available for cross-language tests")

        # Run cross-language tests
        results = test_runner.run_cross_language_tests(test_suite)

        # Check basic result metrics
        assert "conversion_tests" in results, "No conversion test results"
        assert "pattern_detection_tests" in results, "No pattern detection test results"

        # Some conversion tests should be attempted
        assert len(results["conversion_tests"]) > 0, "No conversion tests were run"

        # Count successful conversions
        successful_conversions = sum(
            1 for r in results["conversion_tests"] if r["passed"]
        )
        if len(results["conversion_tests"]) > 0:
            success_rate = (
                successful_conversions / len(results["conversion_tests"])
            ) * 100
            logging.info(f"Cross-language conversion success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create runner
    runner = LanguageTestRunner()

    # Load test suite
    suite = LanguageTestSuite("Backend Integration")
    suite_path = Path(__file__).parent / "test_cases" / "backend_test_suite.json"

    if suite_path.exists():
        suite.load_from_file(suite_path)
        logging.info(f"Loaded {len(suite.test_cases)} test cases from {suite_path}")

        # Run tests
        results = runner.run_test_suite(suite)

        # Print results
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)

        logging.info(f"Test Results: {passed}/{len(results)} passed, {failed} failed")

        # Print failures
        if failed > 0:
            logging.warning("Failed tests:")
            for result in results:
                if not result.passed:
                    logging.warning(
                        f"  {result.test_case.name} ({result.test_case.language})"
                    )
                    for error in result.errors:
                        logging.warning(f"    - {error}")

        # Run cross-language tests
        cross_lang_results = runner.run_cross_language_tests(suite)

        # Print cross-language results
        logging.info("Cross-Language Test Results:")
        logging.info(
            f"  Conversion tests: {len(cross_lang_results['conversion_tests'])}"
        )
        successful = sum(
            1 for r in cross_lang_results["conversion_tests"] if r.get("passed")
        )
        logging.info(f"    Passed: {successful}")

        logging.info(
            f"  Pattern detection tests: {len(cross_lang_results['pattern_detection_tests'])}"
        )
        with_similar = sum(
            1
            for r in cross_lang_results["pattern_detection_tests"]
            if r.get("found_similar")
        )
        logging.info(f"    Found similar errors: {with_similar}")

        logging.info(
            f"  Cross suggestion tests: {len(cross_lang_results['cross_suggestion_tests'])}"
        )
        with_suggestions = sum(
            1
            for r in cross_lang_results["cross_suggestion_tests"]
            if r.get("found_suggestions")
        )
        logging.info(f"    Found suggestions: {with_suggestions}")
    else:
        logging.error(f"Test suite file not found: {suite_path}")
        sys.exit(1)
