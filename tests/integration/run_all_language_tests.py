#!/usr/bin/env python3
"""
Comprehensive integration test runner for all 40+ supported languages.

This script runs integration tests across all supported languages,
generates reports, and validates the self-healing capabilities.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.integration.language_integration_framework import (  # noqa: E402
    IntegrationTestOrchestrator,
    IntegrationTestResult,
)
from tests.integration.language_runners import LANGUAGE_RUNNERS  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# All supported languages
ALL_LANGUAGES = [
    "python",
    "javascript",
    "typescript",
    "java",
    "go",
    "rust",
    "ruby",
    "php",
    "csharp",
    "swift",
    "kotlin",
    "scala",
    "elixir",
    "erlang",
    "clojure",
    "haskell",
    "fsharp",
    "lua",
    "r",
    "matlab",
    "julia",
    "nim",
    "crystal",
    "zig",
    "powershell",
    "bash",
    "sql",
    "yaml_json",
    "terraform",
    "dockerfile",
    "ansible",
    "cpp",
    "objc",
    "perl",
    "dart",
    "groovy",
    "vb",
    "fortran",
    "cobol",
    "pascal",
    "ada",
    "d",
    "ocaml",
    "scheme",
    "racket",
    "prolog",
]

# Languages grouped by category
LANGUAGE_CATEGORIES = {
    "web": ["javascript", "typescript", "php", "ruby", "python"],
    "systems": ["rust", "go", "cpp", "c", "zig", "nim"],
    "mobile": ["swift", "kotlin", "java", "objc", "dart"],
    "functional": ["haskell", "fsharp", "elixir", "erlang", "clojure", "scala"],
    "scripting": ["python", "ruby", "perl", "lua", "powershell", "bash"],
    "data_science": ["python", "r", "julia", "matlab"],
    "infrastructure": ["terraform", "ansible", "dockerfile", "yaml_json"],
    "enterprise": ["java", "csharp", "cobol", "fortran"],
    "modern": ["rust", "go", "kotlin", "swift", "typescript", "zig"],
}


class ComprehensiveTestRunner:
    """Runs integration tests across all languages with detailed reporting."""

    def __init__(self, test_dir: Path = None):
        self.test_dir = test_dir or Path(__file__).parent / "test_suites"
        self.orchestrator = IntegrationTestOrchestrator()
        self.results: Dict[str, List[IntegrationTestResult]] = {}
        self.start_time = None
        self.end_time = None

    async def run_language_tests(
        self,
        languages: List[str] = None,
        categories: List[str] = None,
        parallel: bool = True,
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        """Run tests for specified languages or categories."""
        self.start_time = datetime.now()

        # Determine which languages to test
        target_languages = set()

        if languages:
            target_languages.update(languages)

        if categories:
            for category in categories:
                if category in LANGUAGE_CATEGORIES:
                    target_languages.update(LANGUAGE_CATEGORIES[category])

        if not target_languages:
            # Test all languages with available runners
            target_languages = set(LANGUAGE_RUNNERS.keys())

        logger.info(
            f"Running tests for {len(target_languages)} languages: {sorted(target_languages)}"
        )

        # Load test suites
        self.orchestrator.load_test_suites(self.test_dir)

        # Filter test suites by target languages
        filtered_suites = {
            lang: cases
            for lang, cases in self.orchestrator.test_suites.items()
            if lang in target_languages
        }
        self.orchestrator.test_suites = filtered_suites

        # Run tests
        self.results = await self.orchestrator.run_all_tests(
            parallel=parallel, max_workers=max_workers
        )

        self.end_time = datetime.now()

        # Generate comprehensive report
        report = self.generate_comprehensive_report()

        return report

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a detailed test report with insights."""
        duration = (self.end_time - self.start_time).total_seconds()

        # Calculate statistics
        total_tests = sum(len(results) for results in self.results.values())
        passed_tests = sum(
            1
            for results in self.results.values()
            for result in results
            if result.passed
        )
        failed_tests = total_tests - passed_tests

        # Language-specific statistics
        language_stats = {}
        for language, results in self.results.items():
            language_stats[language] = {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed),
                "pass_rate": (
                    sum(1 for r in results if r.passed) / len(results) * 100
                    if results
                    else 0
                ),
                "avg_duration": (
                    sum(r.duration for r in results) / len(results) if results else 0
                ),
                "error_types": self._collect_error_types(results),
                "fix_types": self._collect_fix_types(results),
            }

        # Category statistics
        category_stats = {}
        for category, languages in LANGUAGE_CATEGORIES.items():
            category_languages = [lang for lang in languages if lang in self.results]
            if category_languages:
                category_results = []
                for lang in category_languages:
                    category_results.extend(self.results[lang])

                category_stats[category] = {
                    "languages": category_languages,
                    "total": len(category_results),
                    "passed": sum(1 for r in category_results if r.passed),
                    "failed": sum(1 for r in category_results if not r.passed),
                    "pass_rate": sum(1 for r in category_results if r.passed)
                    / len(category_results)
                    * 100,
                }

        # Error analysis
        all_errors = []
        for results in self.results.values():
            for result in results:
                all_errors.extend(result.detected_errors)

        error_distribution = self._analyze_error_distribution(all_errors)

        # Fix effectiveness
        fix_stats = self._analyze_fix_effectiveness()

        report = {
            "summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration": duration,
                "languages_tested": len(self.results),
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "pass_rate": (
                    (passed_tests / total_tests * 100) if total_tests > 0 else 0
                ),
            },
            "language_statistics": language_stats,
            "category_statistics": category_stats,
            "error_analysis": error_distribution,
            "fix_effectiveness": fix_stats,
            "detailed_results": self._format_detailed_results(),
        }

        return report

    def _collect_error_types(
        self, results: List[IntegrationTestResult]
    ) -> Dict[str, int]:
        """Collect error types from test results."""
        error_types = {}
        for result in results:
            for error in result.detected_errors:
                error_type = error.get("error_type", "Unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
        return error_types

    def _collect_fix_types(
        self, results: List[IntegrationTestResult]
    ) -> Dict[str, int]:
        """Collect fix types from test results."""
        fix_types = {}
        for result in results:
            for fix in result.generated_fixes:
                fix_type = fix.get("fix_type", "Unknown")
                fix_types[fix_type] = fix_types.get(fix_type, 0) + 1
        return fix_types

    def _analyze_error_distribution(
        self, errors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze the distribution of errors."""
        error_types = {}
        error_categories = {
            "null_safety": [
                "NullPointerException",
                "AttributeError",
                "TypeError",
                "nil",
            ],
            "bounds_checking": [
                "IndexError",
                "IndexOutOfBounds",
                "ArrayIndexOutOfBoundsException",
            ],
            "type_safety": ["TypeError", "ClassCastException", "type mismatch"],
            "concurrency": ["race condition", "deadlock", "concurrent"],
            "memory": ["memory leak", "out of memory", "segmentation fault"],
            "syntax": ["SyntaxError", "ParseError", "compilation error"],
        }

        category_counts = {cat: 0 for cat in error_categories}

        for error in errors:
            error_type = error.get("error_type", "Unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1

            # Categorize error
            for category, keywords in error_categories.items():
                if any(keyword.lower() in error_type.lower() for keyword in keywords):
                    category_counts[category] += 1
                    break

        return {
            "error_types": error_types,
            "error_categories": category_counts,
            "total_errors": len(errors),
        }

    def _analyze_fix_effectiveness(self) -> Dict[str, Any]:
        """Analyze how effective the fixes were."""
        total_errors = 0
        total_fixes_generated = 0
        total_fixes_applied = 0
        successful_fixes = 0

        fix_success_by_type = {}

        for results in self.results.values():
            for result in results:
                total_errors += len(result.detected_errors)
                total_fixes_generated += len(result.generated_fixes)
                total_fixes_applied += len(result.applied_fixes)

                if result.passed and result.applied_fixes:
                    successful_fixes += len(result.applied_fixes)

                for fix in result.generated_fixes:
                    fix_type = fix.get("fix_type", "Unknown")
                    if fix_type not in fix_success_by_type:
                        fix_success_by_type[fix_type] = {
                            "generated": 0,
                            "successful": 0,
                        }
                    fix_success_by_type[fix_type]["generated"] += 1

                    if result.passed and fix in result.applied_fixes:
                        fix_success_by_type[fix_type]["successful"] += 1

        return {
            "total_errors_detected": total_errors,
            "total_fixes_generated": total_fixes_generated,
            "total_fixes_applied": total_fixes_applied,
            "successful_fixes": successful_fixes,
            "fix_generation_rate": (
                (total_fixes_generated / total_errors * 100) if total_errors > 0 else 0
            ),
            "fix_success_rate": (
                (successful_fixes / total_fixes_applied * 100)
                if total_fixes_applied > 0
                else 0
            ),
            "fix_success_by_type": fix_success_by_type,
        }

    def _format_detailed_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Format detailed results for each test."""
        detailed = {}

        for language, results in self.results.items():
            detailed[language] = []
            for result in results:
                test_detail = {
                    "test_name": result.test_case.name,
                    "description": result.test_case.description,
                    "test_type": result.test_case.test_type,
                    "passed": result.passed,
                    "duration": result.duration,
                    "errors_detected": len(result.detected_errors),
                    "fixes_generated": len(result.generated_fixes),
                    "fixes_applied": len(result.applied_fixes),
                    "tags": result.test_case.tags,
                }

                if not result.passed:
                    test_detail["failure_reason"] = result.error_messages

                detailed[language].append(test_detail)

        return detailed

    def save_report(self, report: Dict[str, Any], output_path: Path):
        """Save the report to a file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to {output_path}")

    def print_summary(self, report: Dict[str, Any]):
        """Print a summary of the test results."""
        summary = report["summary"]

        print("\n" + "=" * 80)
        print("HOMEOSTASIS INTEGRATION TEST REPORT")
        print("=" * 80)
        print(f"Duration: {summary['duration']:.2f} seconds")
        print(f"Languages Tested: {summary['languages_tested']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ({summary['pass_rate']:.1f}%)")
        print(f"Failed: {summary['failed']}")
        print()

        # Language summary
        print("Language Results:")
        print("-" * 60)
        for language, stats in sorted(report["language_statistics"].items()):
            print(
                f"{language:15} - Total: {stats['total']:3d}, "
                f"Passed: {stats['passed']:3d}, "
                f"Failed: {stats['failed']:3d}, "
                f"Pass Rate: {stats['pass_rate']:5.1f}%"
            )

        # Category summary
        if report["category_statistics"]:
            print("\nCategory Results:")
            print("-" * 60)
            for category, stats in sorted(report["category_statistics"].items()):
                print(
                    f"{category:15} - Languages: {len(stats['languages']):2d}, "
                    f"Tests: {stats['total']:3d}, "
                    f"Pass Rate: {stats['pass_rate']:5.1f}%"
                )

        # Fix effectiveness
        fix_stats = report["fix_effectiveness"]
        print("\nFix Effectiveness:")
        print("-" * 60)
        print(f"Errors Detected: {fix_stats['total_errors_detected']}")
        print(
            f"Fixes Generated: {fix_stats['total_fixes_generated']} "
            f"({fix_stats['fix_generation_rate']:.1f}%)"
        )
        print(f"Fixes Applied: {fix_stats['total_fixes_applied']}")
        print(
            f"Successful Fixes: {fix_stats['successful_fixes']} "
            f"({fix_stats['fix_success_rate']:.1f}%)"
        )

        print("\n" + "=" * 80)


async def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run integration tests for Homeostasis across all supported languages"
    )
    parser.add_argument(
        "--languages", "-l", nargs="+", help="Specific languages to test"
    )
    parser.add_argument(
        "--categories",
        "-c",
        nargs="+",
        choices=list(LANGUAGE_CATEGORIES.keys()),
        help="Test languages by category",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        default=True,
        help="Run tests in parallel (default: True)",
    )
    parser.add_argument(
        "--sequential", "-s", action="store_true", help="Run tests sequentially"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Output path for the test report"
    )
    parser.add_argument(
        "--test-dir", "-t", type=Path, help="Directory containing test suites"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create test runner
    runner = ComprehensiveTestRunner(test_dir=args.test_dir)

    # Run tests
    parallel = not args.sequential
    report = await runner.run_language_tests(
        languages=args.languages,
        categories=args.categories,
        parallel=parallel,
        max_workers=args.workers,
    )

    # Save report if output specified
    if args.output:
        runner.save_report(report, args.output)
    else:
        # Save to default location with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"test_reports/integration_test_report_{timestamp}.json")
        runner.save_report(report, output_path)

    # Print summary
    runner.print_summary(report)

    # Exit with appropriate code
    if report["summary"]["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
