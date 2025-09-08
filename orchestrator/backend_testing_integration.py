"""
Backend Testing Integration for Orchestrator

This module integrates the backend language testing framework with the main Homeostasis
orchestrator, enabling automated testing, validation, and metrics collection for
cross-language error handling and fixes.
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from modules.analysis.enhanced_cross_language_orchestrator import (
    EnhancedCrossLanguageOrchestrator,
)
from modules.analysis.language_plugin_system import load_all_plugins

# Import testing framework
from modules.analysis.language_test_framework import (
    LanguageTestCase,
    LanguageTestRunner,
    LanguageTestSuite,
)
from modules.analysis.shared_rule_system import (
    initialize_shared_rules,
    shared_rule_registry,
)

# Import orchestrator
from orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class BackendTestingManager:
    """
    Manager for backend language testing integrated with the orchestrator.

    This class provides capabilities for:
    1. Running backend test suites
    2. Validating fixes across languages
    3. Collecting and reporting metrics
    4. Automated regression testing
    """

    def __init__(
        self, orchestrator: Orchestrator, test_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the backend testing manager.

        Args:
            orchestrator: Main orchestrator instance
            test_dir: Directory for test cases and results
        """
        self.orchestrator = orchestrator

        # Set up test directory
        if test_dir is None:
            test_dir = (
                Path(os.path.dirname(os.path.abspath(__file__))) /
                ".." /
                "tests" /
                "test_cases"
            )
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(exist_ok=True, parents=True)

        # Create enhanced cross-language orchestrator
        self.cl_orchestrator = EnhancedCrossLanguageOrchestrator()

        # Create test runner
        self.test_runner = LanguageTestRunner(load_plugins=False)

        # Initialize test suites
        self.test_suites = {}
        self._load_test_suites()

        # Metrics tracking
        self.metrics = {
            "languages": {},
            "fixes": {"total_attempted": 0, "successful": 0, "by_language": {}},
            "cross_language": {
                "conversions": {"total": 0, "successful": 0},
                "fixes": {"total": 0, "successful": 0},
            },
            "tests": {"total_run": 0, "passed": 0, "by_language": {}},
        }

    def initialize(self) -> None:
        """Initialize the testing system."""
        # Load all plugins
        load_all_plugins()

        # Initialize shared rules
        initialize_shared_rules()

        # Update metrics with supported languages
        supported_languages = self.cl_orchestrator.get_supported_languages()
        for lang in supported_languages:
            self.metrics["languages"][lang] = {
                "plugin_loaded": True,
                "test_cases": 0,
                "fixes_attempted": 0,
                "fixes_successful": 0,
            }

            # Count test cases for each language
            for suite_name, suite in self.test_suites.items():
                cases = suite.filter_by_language(lang)
                self.metrics["languages"][lang]["test_cases"] += len(cases)

                # Initialize test metrics for this language
                if lang not in self.metrics["tests"]["by_language"]:
                    self.metrics["tests"]["by_language"][lang] = {
                        "total": 0,
                        "passed": 0,
                    }

                # Initialize fix metrics for this language
                if lang not in self.metrics["fixes"]["by_language"]:
                    self.metrics["fixes"]["by_language"][lang] = {
                        "total": 0,
                        "successful": 0,
                    }

        logger.info(
            f"Backend testing initialized with {len(supported_languages)} languages"
        )

    def run_backend_tests(
        self, suite_name: Optional[str] = None, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run backend language tests.

        Args:
            suite_name: Optional test suite name (runs all suites if None)
            language: Optional language to filter by

        Returns:
            Test results
        """
        start_time = time.time()
        results = {"suites": {}, "summary": {}}

        # Determine suites to run
        if suite_name:
            if suite_name not in self.test_suites:
                logger.error(f"Test suite not found: {suite_name}")
                return {"error": f"Test suite not found: {suite_name}"}

            suites_to_run = {suite_name: self.test_suites[suite_name]}
        else:
            suites_to_run = self.test_suites

        total_passed = 0
        total_tests = 0

        # Run each test suite
        for name, suite in suites_to_run.items():
            logger.info(f"Running test suite: {name}")

            # Filter by language if specified
            if language:
                # Create a new suite with only the specified language
                filtered_suite = LanguageTestSuite(f"{name}-{language}")
                language_cases = suite.filter_by_language(language)

                for case in language_cases:
                    filtered_suite.add_test_case(case)

                if not language_cases:
                    logger.warning(
                        f"No test cases for language {language} in suite {name}"
                    )
                    continue

                suite_to_test = filtered_suite
            else:
                suite_to_test = suite

            # Run the test suite
            test_results = self.test_runner.run_test_suite(suite_to_test)

            # Count results
            suite_passed = sum(1 for r in test_results if r.passed)
            suite_total = len(test_results)

            total_passed += suite_passed
            total_tests += suite_total

            # Update metrics
            self.metrics["tests"]["total_run"] += suite_total
            self.metrics["tests"]["passed"] += suite_passed

            # Update language-specific metrics
            for result in test_results:
                test_lang = result.test_case.language.lower()

                if test_lang in self.metrics["tests"]["by_language"]:
                    self.metrics["tests"]["by_language"][test_lang]["total"] += 1
                    if result.passed:
                        self.metrics["tests"]["by_language"][test_lang]["passed"] += 1

            # Store results
            results["suites"][name] = {
                "total": suite_total,
                "passed": suite_passed,
                "pass_percentage": (
                    (suite_passed / suite_total * 100) if suite_total > 0 else 0
                ),
                "tests": [r.to_dict() for r in test_results],
            }

        # Add summary
        results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": total_passed,
            "pass_percentage": (
                (total_passed / total_tests * 100) if total_tests > 0 else 0
            ),
            "execution_time": time.time() - start_time,
        }

        logger.info(f"Test summary: {total_passed}/{total_tests} tests passed")

        return results

    def run_cross_language_tests(
        self, suite_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run cross-language integration tests.

        Args:
            suite_name: Optional test suite name (runs all suites if None)

        Returns:
            Test results
        """
        start_time = time.time()
        results = {"suites": {}, "summary": {}}

        # Determine suites to run
        if suite_name:
            if suite_name not in self.test_suites:
                logger.error(f"Test suite not found: {suite_name}")
                return {"error": f"Test suite not found: {suite_name}"}

            suites_to_run = {suite_name: self.test_suites[suite_name]}
        else:
            suites_to_run = self.test_suites

        total_conversions = 0
        successful_conversions = 0
        total_cross_fixes = 0
        successful_cross_fixes = 0

        # Run each test suite
        for name, suite in suites_to_run.items():
            logger.info(f"Running cross-language tests for suite: {name}")

            # Run cross-language tests
            cross_results = self.test_runner.run_cross_language_tests(suite)

            # Count conversion results
            suite_conversions = len(cross_results["conversion_tests"])
            suite_successful_conv = sum(
                1 for r in cross_results["conversion_tests"] if r.get("passed")
            )

            total_conversions += suite_conversions
            successful_conversions += suite_successful_conv

            # Count cross fix results
            suite_cross_fixes = len(cross_results["cross_suggestion_tests"])
            suite_successful_cf = sum(
                1
                for r in cross_results["cross_suggestion_tests"]
                if r.get("found_suggestions")
            )

            total_cross_fixes += suite_cross_fixes
            successful_cross_fixes += suite_successful_cf

            # Update metrics
            self.metrics["cross_language"]["conversions"]["total"] += suite_conversions
            self.metrics["cross_language"]["conversions"][
                "successful"
            ] += suite_successful_conv
            self.metrics["cross_language"]["fixes"]["total"] += suite_cross_fixes
            self.metrics["cross_language"]["fixes"]["successful"] += suite_successful_cf

            # Store results
            results["suites"][name] = {
                "conversions": {
                    "total": suite_conversions,
                    "successful": suite_successful_conv,
                    "success_percentage": (
                        (suite_successful_conv / suite_conversions * 100)
                        if suite_conversions > 0
                        else 0
                    ),
                    "details": cross_results["conversion_tests"],
                },
                "cross_fixes": {
                    "total": suite_cross_fixes,
                    "successful": suite_successful_cf,
                    "success_percentage": (
                        suite_successful_cf / suite_cross_fixes * 100
                        if suite_cross_fixes > 0
                        else 0
                    ),
                    "details": cross_results["cross_suggestion_tests"],
                },
                "pattern_detection": {
                    "details": cross_results["pattern_detection_tests"]
                },
            }

        # Add summary
        results["summary"] = {
            "conversions": {
                "total": total_conversions,
                "successful": successful_conversions,
                "success_percentage": (
                    (successful_conversions / total_conversions * 100)
                    if total_conversions > 0
                    else 0
                ),
            },
            "cross_fixes": {
                "total": total_cross_fixes,
                "successful": successful_cross_fixes,
                "success_percentage": (
                    successful_cross_fixes / total_cross_fixes * 100
                    if total_cross_fixes > 0
                    else 0
                ),
            },
            "execution_time": time.time() - start_time,
        }

        logger.info(
            f"Cross-language summary: "
            f"{successful_conversions}/{total_conversions} successful conversions, "
            f"{successful_cross_fixes}/{total_cross_fixes} successful cross-language fixes"
        )

        return results

    def validate_fix(
        self, error_data: Dict[str, Any], fix_data: Dict[str, Any], language: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a fix for correctness.

        Args:
            error_data: Error data
            fix_data: Fix data
            language: Language identifier

        Returns:
            Tuple of (is_valid, validation_details)
        """
        validation = {"is_valid": False, "details": {}, "messages": []}

        # Check that the fix is for the correct language
        if fix_data.get("language") != language:
            validation["messages"].append(
                f"Fix language mismatch: expected {language}, got {fix_data.get('language')}"
            )
            return False, validation

        # Make sure fix has a suggestion or patch code
        if "suggestion" not in fix_data and "patch_code" not in fix_data:
            validation["messages"].append("Fix has no suggestion or patch code")
            return False, validation

        # For patch code, check that it's valid code for the language
        if "patch_code" in fix_data:
            # Basic syntax validation (could be expanded with actual parsing)
            syntax_valid = self._validate_syntax(fix_data["patch_code"], language)
            validation["details"]["syntax_valid"] = syntax_valid

            if not syntax_valid:
                validation["messages"].append("Fix patch code has syntax errors")
                return False, validation

        # For suggestions, check that they are appropriate for the error
        if "suggestion" in fix_data:
            # Check if suggestion matches the error type
            error_type = self._get_error_type(error_data, language)
            suggestion_appropriate = self._check_suggestion_relevance(
                fix_data["suggestion"], error_type, language
            )
            validation["details"]["suggestion_appropriate"] = suggestion_appropriate

            if not suggestion_appropriate:
                validation["messages"].append(
                    f"Suggestion may not be appropriate for {error_type} error"
                )
                # Don't fail for this, just warn

        # Check confidence level
        if "confidence" in fix_data:
            # Log but don't fail for low confidence
            if fix_data["confidence"] == "low":
                validation["messages"].append("Fix has low confidence level")

        # All checks passed
        validation["is_valid"] = True
        validation["messages"].append("Fix validation successful")

        return True, validation

    def generate_regression_test(
        self,
        error_data: Dict[str, Any],
        fix_data: Dict[str, Any],
        language: str,
        test_name: Optional[str] = None,
    ) -> LanguageTestCase:
        """
        Generate a regression test case from an error and its fix.

        Args:
            error_data: Error data
            fix_data: Fix data
            language: Language identifier
            test_name: Optional name for the test case

        Returns:
            Generated test case
        """
        # Generate a default name if not provided
        if not test_name:
            error_type = self._get_error_type(error_data, language)
            sanitized_type = re.sub(r"[^a-zA-Z0-9_]", "_", error_type)
            test_name = (
                f"{language.capitalize()}_{sanitized_type}_Test_{int(time.time())}"
            )

        # Extract expected analysis from the fix
        expected_analysis = {"root_cause": fix_data.get("root_cause", "unknown")}

        if "confidence" in fix_data:
            expected_analysis["confidence"] = fix_data["confidence"]

        # Create the test case
        test_case = LanguageTestCase(
            name=test_name,
            language=language,
            error_data=error_data,
            expected_analysis=expected_analysis,
            context=fix_data.get("context", {}),
            tags=[language, fix_data.get("root_cause", "unknown"), "regression"],
        )

        return test_case

    def add_regression_test(
        self, test_case: LanguageTestCase, suite_name: str = "regression"
    ) -> None:
        """
        Add a regression test to a test suite.

        Args:
            test_case: Test case to add
            suite_name: Test suite name
        """
        # Get or create the suite
        if suite_name not in self.test_suites:
            self.test_suites[suite_name] = LanguageTestSuite(
                suite_name, "Regression tests"
            )

        # Add the test case
        self.test_suites[suite_name].add_test_case(test_case)

        # Save the updated suite
        self._save_test_suite(suite_name)

        logger.info(f"Added regression test {test_case.name} to suite {suite_name}")

    def track_fix_attempt(
        self,
        error_data: Dict[str, Any],
        fix_data: Dict[str, Any],
        language: str,
        success: bool,
    ) -> None:
        """
        Track a fix attempt for metrics collection.

        Args:
            error_data: Error data
            fix_data: Fix data
            language: Language identifier
            success: Whether the fix was successful
        """
        # Update total counts
        self.metrics["fixes"]["total_attempted"] += 1
        if success:
            self.metrics["fixes"]["successful"] += 1

        # Update language-specific counts
        language = language.lower()
        if language in self.metrics["fixes"]["by_language"]:
            self.metrics["fixes"]["by_language"][language]["total"] += 1
            if success:
                self.metrics["fixes"]["by_language"][language]["successful"] += 1

        # Track if this was a cross-language fix
        if fix_data.get("used_cross_language"):
            self.metrics["cross_language"]["fixes"]["total"] += 1
            if success:
                self.metrics["cross_language"]["fixes"]["successful"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get testing metrics.

        Returns:
            Dictionary of metrics
        """
        # Calculate additional metrics
        metrics = self.metrics.copy()

        # Add success rates
        metrics["fixes"]["success_rate"] = (
            (metrics["fixes"]["successful"] / metrics["fixes"]["total_attempted"]) * 100
            if metrics["fixes"]["total_attempted"] > 0
            else 0
        )

        metrics["tests"]["pass_rate"] = (
            (metrics["tests"]["passed"] / metrics["tests"]["total_run"]) * 100
            if metrics["tests"]["total_run"] > 0
            else 0
        )

        metrics["cross_language"]["conversions"]["success_rate"] = (
            (
                metrics["cross_language"]["conversions"]["successful"] /
                metrics["cross_language"]["conversions"]["total"]
            ) *
            100
            if metrics["cross_language"]["conversions"]["total"] > 0
            else 0
        )

        metrics["cross_language"]["fixes"]["success_rate"] = (
            (
                metrics["cross_language"]["fixes"]["successful"] /
                metrics["cross_language"]["fixes"]["total"]
            ) *
            100
            if metrics["cross_language"]["fixes"]["total"] > 0
            else 0
        )

        # Add orchestrator metrics
        metrics["orchestrator"] = self.cl_orchestrator.get_metrics()

        # Add shared rules metrics
        metrics["shared_rules"] = {
            "total": len(shared_rule_registry.rules),
            "by_language": {},
        }

        # Count shared rules by language
        for language in metrics["languages"]:
            lang_rules = shared_rule_registry.get_rules_for_language(language)
            metrics["shared_rules"]["by_language"][language] = len(lang_rules)

        return metrics

    def _load_test_suites(self):
        """Load test suites from the test directory."""
        # Find all JSON files in the test directory
        for file_path in self.test_dir.glob("*.json"):
            try:
                # Create a new test suite
                suite = LanguageTestSuite(file_path.stem)

                # Load test cases from file
                suite.load_from_file(file_path)

                # Add to test suites
                if suite.test_cases:
                    self.test_suites[file_path.stem] = suite
                    logger.info(
                        f"Loaded test suite: {file_path.stem} with {len(suite.test_cases)} test cases"
                    )
            except Exception as e:
                logger.error(f"Error loading test suite from {file_path}: {e}")

    def _save_test_suite(self, suite_name: str):
        """
        Save a test suite to a file.

        Args:
            suite_name: Test suite name
        """
        if suite_name not in self.test_suites:
            logger.warning(f"Cannot save unknown test suite: {suite_name}")
            return

        suite = self.test_suites[suite_name]
        file_path = self.test_dir / f"{suite_name}.json"

        try:
            suite.save_to_file(file_path)
            logger.info(f"Saved test suite to {file_path}")
        except Exception as e:
            logger.error(f"Error saving test suite to {file_path}: {e}")

    def _validate_syntax(self, code: str, language: str) -> bool:
        """
        Validate syntax of code for a specific language.

        Args:
            code: Code to validate
            language: Language identifier

        Returns:
            True if syntax is valid
        """
        # This is a simple syntax check that could be expanded with actual parsing
        # For now, we just check for common syntax errors

        if language == "python":
            try:
                import ast

                ast.parse(code)
                return True
            except SyntaxError:
                return False
        elif language == "javascript":
            # Check for common JS syntax errors
            common_errors = [
                r".*?unexpected end of input",
                r".*?unterminated string literal",
                r".*?missing [)]",
                r".*?undefined is not a function",
            ]
            for error_pattern in common_errors:
                if re.search(error_pattern, code, re.IGNORECASE):
                    return False
            return True
        elif language == "java":
            # Check for common Java syntax errors
            common_errors = [
                r".*?expecting.*?[{};]",
                r".*?illegal start of expression",
                r".*?missing .*?[{}();]",
            ]
            for error_pattern in common_errors:
                if re.search(error_pattern, code, re.IGNORECASE):
                    return False
            return True
        elif language == "go":
            # Check for common Go syntax errors
            common_errors = [
                r".*?unexpected.*?[{};]",
                r".*?undefined:",
                r".*?syntax error",
            ]
            for error_pattern in common_errors:
                if re.search(error_pattern, code, re.IGNORECASE):
                    return False
            return True
        else:
            # For other languages, assume valid
            return True

    def _get_error_type(self, error_data: Dict[str, Any], language: str) -> str:
        """
        Get the error type from error data.

        Args:
            error_data: Error data
            language: Language identifier

        Returns:
            Error type
        """
        if language == "python":
            return error_data.get("exception_type", "Unknown")
        elif language == "javascript":
            return error_data.get("name", "Unknown")
        elif language == "java":
            return error_data.get("exception_class", "Unknown")
        elif language == "go":
            return error_data.get("error_type", "Unknown")
        else:
            return error_data.get("error_type", "Unknown")

    def _check_suggestion_relevance(
        self, suggestion: str, error_type: str, language: str
    ) -> bool:
        """
        Check if a suggestion is relevant for an error type.

        Args:
            suggestion: Suggestion text
            error_type: Error type
            language: Language identifier

        Returns:
            True if the suggestion is relevant
        """
        # Convert error type to lowercase for matching
        error_type_lower = error_type.lower()

        # Extract keywords from error type
        keywords = []
        if language == "python":
            if "keyerror" in error_type_lower:
                keywords = ["key", "dict", "dictionary", "check", "exist"]
            elif "indexerror" in error_type_lower:
                keywords = ["index", "list", "array", "bounds", "range"]
            elif (
                "attributeerror" in error_type_lower and "nonetype" in error_type_lower
            ):
                keywords = ["none", "null", "check", "attribute"]
            elif "typeerror" in error_type_lower:
                keywords = ["type", "convert", "cast"]
        elif language == "javascript":
            if "typeerror" in error_type_lower and "undefined" in suggestion.lower():
                keywords = ["null", "undefined", "check", "property"]
            elif "referenceerror" in error_type_lower:
                keywords = ["defined", "variable", "declaration"]
        elif language == "java":
            if "nullpointerexception" in error_type_lower:
                keywords = ["null", "check", "nullpointerexception"]
            elif "indexoutofboundsexception" in error_type_lower:
                keywords = ["index", "bounds", "length", "size"]
            elif "classcastexception" in error_type_lower:
                keywords = ["cast", "type", "instanceof"]
            elif "concurrentmodificationexception" in error_type_lower:
                keywords = ["concurrent", "modify", "iterator", "synchronized"]
        elif language == "go":
            if "nil pointer dereference" in error_type_lower:
                keywords = ["nil", "pointer", "check"]
            elif "index out of range" in error_type_lower:
                keywords = ["index", "range", "bounds", "length"]

        # Check if suggestion contains any of the keywords
        suggestion_lower = suggestion.lower()
        return any(keyword in suggestion_lower for keyword in keywords)


def integrate_backend_testing(orchestrator: Orchestrator) -> BackendTestingManager:
    """
    Integrate backend testing with the orchestrator.

    Args:
        orchestrator: Main orchestrator instance

    Returns:
        Backend testing manager
    """
    # Create and initialize the manager
    manager = BackendTestingManager(orchestrator)
    manager.initialize()

    # Register with the orchestrator (if needed)
    # This is a placeholder for any orchestrator-specific registration

    return manager


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create orchestrator
    orchestrator = Orchestrator(config_path=None)

    # Integrate backend testing
    testing_manager = integrate_backend_testing(orchestrator)

    # Run backend tests
    logger.info("Running backend language tests...")
    test_results = testing_manager.run_backend_tests()

    # Run cross-language tests
    logger.info("Running cross-language tests...")
    cross_results = testing_manager.run_cross_language_tests()

    # Get metrics
    metrics = testing_manager.get_metrics()

    # Display summary
    logger.info("\nTest Summary:")
    logger.info(f"  Total tests: {metrics['tests']['total_run']}")
    logger.info(f"  Passed tests: {metrics['tests']['passed']}")
    logger.info(f"  Pass rate: {metrics['tests']['pass_rate']:.1f}%")

    logger.info("\nCross-Language Summary:")
    logger.info(
        f"  Conversion success rate: {metrics['cross_language']['conversions']['success_rate']:.1f}%"
    )
    logger.info(
        f"  Cross-fix success rate: {metrics['cross_language']['fixes']['success_rate']:.1f}%"
    )

    # Show language metrics
    logger.info("\nLanguage Coverage:")
    for lang, lang_metrics in metrics["languages"].items():
        if lang_metrics["plugin_loaded"]:
            logger.info(f"  {lang}: {lang_metrics['test_cases']} test cases")

    # Show shared rules
    logger.info(f"\nShared rules: {metrics['shared_rules']['total']} total")
