"""
Backend Language Testing Framework

This module provides a unified testing framework for backend languages in Homeostasis.
It defines base classes and tools for testing language plugins, error adapters,
and cross-language capabilities.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .cross_language_orchestrator import CrossLanguageOrchestrator
from .language_adapters import ErrorAdapterFactory, convert_to_standard_format
from .language_plugin_system import LanguagePlugin, get_plugin, load_all_plugins

logger = logging.getLogger(__name__)


class LanguageTestCase:
    """
    A test case for language plugins that includes input data, expected results,
    and metadata for test organization.
    """

    def __init__(
        self,
        name: str,
        language: str,
        error_data: Dict[str, Any],
        expected_analysis: Optional[Dict[str, Any]] = None,
        expected_fix: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize a test case for a language plugin.

        Args:
            name: Descriptive name for the test case
            language: Language identifier (e.g., "python", "java")
            error_data: Error data for the test case
            expected_analysis: Expected analysis results (optional)
            expected_fix: Expected fix results (optional)
            context: Additional context for fix generation (optional)
            tags: Tags for categorizing the test case (optional)
        """
        self.name = name
        self.language = language.lower()
        self.error_data = error_data
        self.expected_analysis = expected_analysis or {}
        self.expected_fix = expected_fix or {}
        self.context = context or {}
        self.tags = tags or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert the test case to a dictionary for serialization."""
        return {
            "name": self.name,
            "language": self.language,
            "error_data": self.error_data,
            "expected_analysis": self.expected_analysis,
            "expected_fix": self.expected_fix,
            "context": self.context,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LanguageTestCase":
        """Create a test case from a dictionary."""
        return cls(
            name=data.get("name", "Unnamed Test"),
            language=data.get("language", "unknown"),
            error_data=data.get("error_data", {}),
            expected_analysis=data.get("expected_analysis"),
            expected_fix=data.get("expected_fix"),
            context=data.get("context"),
            tags=data.get("tags"),
        )


class LanguageTestResult:
    """
    Results of running a language test case.
    """

    def __init__(
        self,
        test_case: LanguageTestCase,
        passed: bool,
        actual_analysis: Optional[Dict[str, Any]] = None,
        actual_fix: Optional[Dict[str, Any]] = None,
        errors: Optional[List[str]] = None,
    ):
        """
        Initialize a test result.

        Args:
            test_case: The test case that was run
            passed: Whether the test passed
            actual_analysis: Actual analysis results from the test
            actual_fix: Actual fix results from the test
            errors: List of error messages if the test failed
        """
        self.test_case = test_case
        self.passed = passed
        self.actual_analysis = actual_analysis
        self.actual_fix = actual_fix
        self.errors = errors or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert the test result to a dictionary for serialization."""
        return {
            "test_case": self.test_case.to_dict(),
            "passed": self.passed,
            "actual_analysis": self.actual_analysis,
            "actual_fix": self.actual_fix,
            "errors": self.errors,
        }


class LanguageTestSuite:
    """
    A collection of language test cases.
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize a test suite.

        Args:
            name: Name of the test suite
            description: Description of the test suite
        """
        self.name = name
        self.description = description
        self.test_cases: List[LanguageTestCase] = []

    def add_test_case(self, test_case: LanguageTestCase) -> None:
        """
        Add a test case to the suite.

        Args:
            test_case: Test case to add
        """
        self.test_cases.append(test_case)

    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """
        Load test cases from a JSON file.

        Args:
            file_path: Path to the JSON file containing test cases
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Test file does not exist: {file_path}")
            return

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

                if "name" in data:
                    self.name = data["name"]

                if "description" in data:
                    self.description = data["description"]

                if "test_cases" in data and isinstance(data["test_cases"], list):
                    for tc_data in data["test_cases"]:
                        test_case = LanguageTestCase.from_dict(tc_data)
                        self.add_test_case(test_case)

                    logger.info(
                        f"Loaded {len(self.test_cases)} test cases from {file_path}"
                    )
                else:
                    logger.warning(f"No test cases found in {file_path}")
        except Exception as e:
            logger.error(f"Error loading test cases from {file_path}: {e}")

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save test cases to a JSON file.

        Args:
            file_path: Path to save the JSON file
        """
        file_path = Path(file_path)

        data = {
            "name": self.name,
            "description": self.description,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
        }

        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.test_cases)} test cases to {file_path}")
        except Exception as e:
            logger.error(f"Error saving test cases to {file_path}: {e}")

    def filter_by_language(self, language: str) -> List[LanguageTestCase]:
        """
        Filter test cases by language.

        Args:
            language: Language identifier

        Returns:
            List of test cases for the specified language
        """
        return [tc for tc in self.test_cases if tc.language == language.lower()]

    def filter_by_tags(self, tags: List[str]) -> List[LanguageTestCase]:
        """
        Filter test cases by tags.

        Args:
            tags: List of tags to filter by

        Returns:
            List of test cases that have at least one of the specified tags
        """
        return [tc for tc in self.test_cases if any(tag in tc.tags for tag in tags)]


class LanguageTestRunner:
    """
    Runner for language test cases.
    """

    def __init__(self, load_plugins: bool = True):
        """
        Initialize the test runner.

        Args:
            load_plugins: Whether to load all available plugins at initialization
        """
        self.orchestrator = CrossLanguageOrchestrator()

        # Cache for loaded plugins
        self.plugins = {}

        if load_plugins:
            # Load and register all available plugins
            plugins_loaded = load_all_plugins()
            logger.info(f"Loaded {plugins_loaded} plugins for testing")

    def get_plugin(self, language: str) -> Optional[LanguagePlugin]:
        """
        Get a plugin for a language, with caching.

        Args:
            language: Language identifier

        Returns:
            Language plugin instance or None if not found
        """
        language = language.lower()

        if language not in self.plugins:
            plugin = get_plugin(language)
            if plugin:
                self.plugins[language] = plugin
                return plugin
            else:
                logger.warning(f"No plugin available for language: {language}")
                return None
        else:
            return self.plugins[language]

    def run_test_case(self, test_case: LanguageTestCase) -> LanguageTestResult:
        """
        Run a single test case.

        Args:
            test_case: Test case to run

        Returns:
            Test result
        """
        errors = []
        actual_analysis = None
        actual_fix = None

        # Get the plugin for this language
        plugin = self.get_plugin(test_case.language)
        if not plugin:
            return LanguageTestResult(
                test_case=test_case,
                passed=False,
                errors=[f"No plugin available for language: {test_case.language}"],
            )

        try:
            # Run the analysis
            try:
                actual_analysis = plugin.analyze_error(test_case.error_data)
            except Exception as e:
                errors.append(f"Error during analysis: {e}")
                actual_analysis = {}

            # Run the fix generation if context is provided
            if test_case.context and actual_analysis:
                try:
                    actual_fix = plugin.generate_fix(actual_analysis, test_case.context)
                except Exception as e:
                    errors.append(f"Error during fix generation: {e}")
                    actual_fix = {}

            # Verify analysis results
            if test_case.expected_analysis and actual_analysis:
                # Check that expected keys exist and have the expected values
                for key, expected_value in test_case.expected_analysis.items():
                    if key not in actual_analysis:
                        errors.append(f"Missing expected key in analysis: {key}")
                    elif actual_analysis[key] != expected_value:
                        errors.append(
                            f"Analysis mismatch for {key}: "
                            f"expected {expected_value}, got {actual_analysis[key]}"
                        )

            # Verify fix results
            if test_case.expected_fix and actual_fix:
                # Check that expected keys exist and have the expected values
                for key, expected_value in test_case.expected_fix.items():
                    if key not in actual_fix:
                        errors.append(f"Missing expected key in fix: {key}")
                    elif actual_fix[key] != expected_value:
                        errors.append(
                            f"Fix mismatch for {key}: "
                            f"expected {expected_value}, got {actual_fix[key]}"
                        )

            # Check if test passed
            passed = len(errors) == 0

            return LanguageTestResult(
                test_case=test_case,
                passed=passed,
                actual_analysis=actual_analysis,
                actual_fix=actual_fix,
                errors=errors,
            )

        except Exception as e:
            logger.error(f"Unexpected error running test case: {e}")
            return LanguageTestResult(
                test_case=test_case, passed=False, errors=[f"Unexpected error: {e}"]
            )

    def run_test_suite(self, test_suite: LanguageTestSuite) -> List[LanguageTestResult]:
        """
        Run all test cases in a test suite.

        Args:
            test_suite: Test suite to run

        Returns:
            List of test results
        """
        results = []

        for test_case in test_suite.test_cases:
            logger.info(f"Running test case: {test_case.name}")
            result = self.run_test_case(test_case)
            results.append(result)

            if result.passed:
                logger.info(f"Test passed: {test_case.name}")
            else:
                logger.warning(f"Test failed: {test_case.name}")
                for error in result.errors:
                    logger.warning(f"  Error: {error}")

        # Log summary
        passed = sum(1 for r in results if r.passed)
        logger.info(f"Test summary: {passed}/{len(results)} tests passed")

        return results

    def run_cross_language_tests(self, test_suite: LanguageTestSuite) -> Dict[str, Any]:
        """
        Run cross-language tests from a test suite.

        This method tests the following capabilities:
        1. Error conversion between languages
        2. Cross-language error pattern detection
        3. Suggestion of fixes across language boundaries

        Args:
            test_suite: Test suite to run

        Returns:
            Dictionary with test results
        """
        results = {
            "conversion_tests": [],
            "pattern_detection_tests": [],
            "cross_suggestion_tests": [],
        }

        # Get available languages from test cases
        languages = set(tc.language for tc in test_suite.test_cases)
        language_pairs = [(l1, l2) for l1 in languages for l2 in languages if l1 != l2]

        # 1. Test error conversion between languages
        for source_lang, target_lang in language_pairs:
            source_cases = test_suite.filter_by_language(source_lang)
            if not source_cases:
                continue

            for test_case in source_cases:
                try:
                    # Convert error to standard format
                    standard_error = convert_to_standard_format(
                        test_case.error_data, source_lang
                    )

                    # Convert from standard to target format
                    target_adapter = ErrorAdapterFactory.get_adapter(target_lang)
                    target_error = target_adapter.from_standard_format(standard_error)
                    
                    # Validate cross-language conversion
                    assert target_error is not None, f"Failed to convert to {target_lang}"
                    assert "error_type" in target_error, "Missing error_type in converted error"
                    assert "message" in target_error, "Missing message in converted error"

                    # Convert back to source format to verify roundtrip
                    source_adapter = ErrorAdapterFactory.get_adapter(source_lang)
                    roundtrip_error = source_adapter.from_standard_format(
                        standard_error
                    )

                    # Check that key information is preserved in roundtrip
                    preserved = True
                    error_message = None

                    if (
                        "error_type" in test_case.error_data
                        and "error_type" in roundtrip_error
                    ):
                        if (
                            test_case.error_data["error_type"]
                            != roundtrip_error["error_type"]
                        ):
                            preserved = False
                            error_message = f"Error type changed: {test_case.error_data['error_type']} -> {roundtrip_error['error_type']}"

                    results["conversion_tests"].append(
                        {
                            "test_case": test_case.name,
                            "source_language": source_lang,
                            "target_language": target_lang,
                            "passed": preserved,
                            "error": error_message,
                        }
                    )
                except Exception as e:
                    results["conversion_tests"].append(
                        {
                            "test_case": test_case.name,
                            "source_language": source_lang,
                            "target_language": target_lang,
                            "passed": False,
                            "error": str(e),
                        }
                    )

        # 2. Test cross-language error pattern detection
        for test_case in test_suite.test_cases:
            try:
                # Use cross-language orchestrator to find similar errors
                similar_errors = self.orchestrator.find_similar_errors(
                    test_case.error_data, test_case.language, max_results=5
                )

                # Check if any similar errors were found
                results["pattern_detection_tests"].append(
                    {
                        "test_case": test_case.name,
                        "language": test_case.language,
                        "found_similar": len(similar_errors) > 0,
                        "similar_count": len(similar_errors),
                        "similar_languages": [
                            e.get("language", "unknown") for e in similar_errors
                        ],
                    }
                )
            except Exception as e:
                results["pattern_detection_tests"].append(
                    {
                        "test_case": test_case.name,
                        "language": test_case.language,
                        "found_similar": False,
                        "error": str(e),
                    }
                )

        # 3. Test cross-language fix suggestions
        for test_case in test_suite.test_cases:
            if test_case.context:  # Only test with cases that have context
                try:
                    # Get suggestions from other languages
                    suggestions = self.orchestrator.suggest_cross_language_fixes(
                        test_case.error_data, test_case.language
                    )

                    results["cross_suggestion_tests"].append(
                        {
                            "test_case": test_case.name,
                            "language": test_case.language,
                            "found_suggestions": len(suggestions) > 0,
                            "suggestion_count": len(suggestions),
                            "suggestion_languages": [
                                s.get("language", "unknown") for s in suggestions
                            ],
                        }
                    )
                except Exception as e:
                    results["cross_suggestion_tests"].append(
                        {
                            "test_case": test_case.name,
                            "language": test_case.language,
                            "found_suggestions": False,
                            "error": str(e),
                        }
                    )

        return results


class TestCaseGeneratorBase(ABC):
    """
    Base class for language-specific test case generators.

    This allows each language plugin to provide its own test case generator
    for creating realistic test cases from real-world examples.
    """

    @abstractmethod
    def generate_test_cases(self) -> List[LanguageTestCase]:
        """
        Generate test cases for a specific language.

        Returns:
            List of generated test cases
        """
        pass

    @abstractmethod
    def get_language(self) -> str:
        """
        Get the language identifier for this generator.

        Returns:
            Language identifier
        """
        pass


# Utility function to discover test case generators from plugins
def discover_test_generators() -> Dict[str, TestCaseGeneratorBase]:
    """
    Discover test case generators from registered plugins.

    Returns:
        Dictionary mapping language IDs to test case generators
    """
    generators = {}

    # Load all plugins first
    load_all_plugins()

    # Get all registered plugins
    from .language_plugin_system import get_all_plugins

    plugins = get_all_plugins()

    for lang_id, plugin in plugins.items():
        # Check if the plugin has a test generator
        if hasattr(plugin, "get_test_generator") and callable(
            getattr(plugin, "get_test_generator")
        ):
            try:
                generator = plugin.get_test_generator()
                if isinstance(generator, TestCaseGeneratorBase):
                    generators[lang_id] = generator
                    logger.info(f"Found test generator for language: {lang_id}")
            except Exception as e:
                logger.warning(f"Error getting test generator for {lang_id}: {e}")

    return generators


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Example usage
    suite = LanguageTestSuite(
        "Example Test Suite", "An example test suite for demonstration"
    )

    # Example test case for Java
    java_test = LanguageTestCase(
        name="Java NullPointerException Test",
        language="java",
        error_data={
            "error_type": "java.lang.NullPointerException",
            "message": 'Cannot invoke "String.length()" because "str" is null',
            "stack_trace": [
                "at com.example.StringProcessor.processString(StringProcessor.java:42)",
                "at com.example.Main.main(Main.java:25)",
            ],
        },
        expected_analysis={"root_cause": "java_null_pointer", "confidence": "high"},
        context={
            "code_snippet": "String result = str.length();",
            "method_params": "String str",
        },
        tags=["npe", "java-core"],
    )

    # Example test case for Go
    go_test = LanguageTestCase(
        name="Go Nil Pointer Test",
        language="go",
        error_data={
            "error_type": "runtime error",
            "message": "nil pointer dereference",
            "stack_trace": [
                "goroutine 1 [running]:",
                "main.processValue()",
                "\t/app/main.go:25",
                "main.main()",
                "\t/app/main.go:12",
            ],
        },
        expected_analysis={"root_cause": "go_nil_pointer", "confidence": "high"},
        context={"code_snippet": "result := value.GetData()"},
        tags=["nil-pointer", "go-core"],
    )

    # Add test cases to the suite
    suite.add_test_case(java_test)
    suite.add_test_case(go_test)

    # Save the test suite to a file
    test_dir = Path(__file__).parent / "tests"
    test_dir.mkdir(exist_ok=True)
    suite.save_to_file(test_dir / "example_test_suite.json")

    # Create a test runner
    runner = LanguageTestRunner()

    # Run the test suite
    results = runner.run_test_suite(suite)

    # Run cross-language tests
    cross_lang_results = runner.run_cross_language_tests(suite)

    # Print summary
    print("\nTest Summary:")
    print(f"  Total tests: {len(results)}")
    print(f"  Passed: {sum(1 for r in results if r.passed)}")
    print(f"  Failed: {sum(1 for r in results if not r.passed)}")

    # Print cross-language test summary
    print("\nCross-Language Test Summary:")
    print(f"  Conversion tests: {len(cross_lang_results['conversion_tests'])}")
    print(
        f"  Pattern detection tests: {len(cross_lang_results['pattern_detection_tests'])}"
    )
    print(
        f"  Cross-suggestion tests: {len(cross_lang_results['cross_suggestion_tests'])}"
    )
