"""
Comprehensive Language Integration Test Framework for Homeostasis

This framework provides infrastructure for testing all 40+ supported languages
with real-world scenarios, cross-language interactions, and framework-specific tests.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.analysis.cross_language_orchestrator import CrossLanguageOrchestrator
from modules.analysis.language_plugin_system import (
    LanguagePlugin,
    LanguagePluginRegistry,
)
from modules.monitoring.metrics_collector import MetricsCollector
from modules.testing.container_manager import ContainerManager
from modules.testing.security_scanner import (
    SecurityScanResult,
    SecurityTestOrchestrator,
)

logger = logging.getLogger(__name__)


@dataclass
class IntegrationTestCase:
    """Represents an integration test case for a language."""

    name: str
    language: str
    description: str
    test_type: str  # 'single', 'cross_language', 'framework', 'deployment'
    source_code: Dict[str, str]  # filename -> content
    expected_errors: List[Dict[str, Any]]
    expected_fixes: List[Dict[str, Any]]
    environment: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    timeout: int = 300  # seconds
    container_config: Optional[Dict[str, Any]] = None


@dataclass
class IntegrationTestResult:
    """Results from running an integration test."""

    test_case: IntegrationTestCase
    passed: bool
    duration: float
    detected_errors: List[Dict[str, Any]]
    generated_fixes: List[Dict[str, Any]]
    applied_fixes: List[Dict[str, Any]]
    test_output: str
    error_messages: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    container_logs: Optional[str] = None
    security_scan_results: Optional[Dict[str, SecurityScanResult]] = None
    security_report: Optional[Dict[str, Any]] = None


class LanguageIntegrationTestRunner(ABC):
    """Abstract base class for language-specific test runners."""

    def __init__(self, language: str, plugin: LanguagePlugin):
        self.language = language
        self.plugin = plugin
        self.container_manager = ContainerManager()
        self.metrics_collector = MetricsCollector()
        self.security_orchestrator = SecurityTestOrchestrator()
        self.enable_security_scanning = True

    @abstractmethod
    async def setup_environment(self, test_case: IntegrationTestCase) -> Path:
        """Set up the test environment for the language."""
        pass

    @abstractmethod
    async def execute_code(
        self, test_dir: Path, test_case: IntegrationTestCase
    ) -> Tuple[int, str, str]:
        """Execute the test code and return exit code, stdout, stderr."""
        pass

    @abstractmethod
    async def validate_environment(self, test_dir: Path) -> bool:
        """Validate that the language environment is properly set up."""
        pass

    async def run_test(self, test_case: IntegrationTestCase) -> IntegrationTestResult:
        """Run a complete integration test."""
        start_time = datetime.now()
        result = IntegrationTestResult(
            test_case=test_case,
            passed=False,
            duration=0,
            detected_errors=[],
            generated_fixes=[],
            applied_fixes=[],
            test_output="",
        )

        test_dir = None
        try:
            # Set up test environment
            test_dir = await self.setup_environment(test_case)

            # Validate environment
            if not await self.validate_environment(test_dir):
                result.error_messages.append("Environment validation failed")
                return result

            # Execute code to trigger errors
            exit_code, stdout, stderr = await self.execute_code(test_dir, test_case)
            result.test_output = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

            # Detect errors using the plugin
            if exit_code != 0:
                error_data = self._extract_errors(stdout, stderr, exit_code)
                for error in error_data:
                    analysis = self.plugin.analyze_error(error)
                    result.detected_errors.append(analysis)

                    # Generate fixes
                    fix = self.plugin.generate_fix(
                        analysis, {"test_dir": str(test_dir)}
                    )
                    if fix:
                        result.generated_fixes.append(fix)

                        # Apply fix
                        if await self._apply_fix(test_dir, fix):
                            result.applied_fixes.append(fix)

                            # Re-run to verify fix
                            exit_code2, _, _ = await self.execute_code(
                                test_dir, test_case
                            )
                            if exit_code2 == 0:
                                result.passed = True

            # Collect metrics
            result.metrics = await self._collect_metrics(test_dir)

            # Run security scan if enabled
            if self.enable_security_scanning and test_dir:
                try:
                    logger.info(f"Running security scan for test {test_case.name}")
                    scan_results = await self.security_orchestrator.run_security_scan(
                        test_dir
                    )
                    result.security_scan_results = scan_results
                    result.security_report = (
                        self.security_orchestrator.generate_security_report(
                            scan_results
                        )
                    )

                    # Check for critical vulnerabilities
                    if (
                        result.security_report
                        and result.security_report["summary"]["by_severity"]["critical"]
                        > 0
                    ):
                        result.error_messages.append(
                            f"Critical security vulnerabilities found: {result.security_report['summary']['by_severity']['critical']}"
                        )
                except Exception as e:
                    logger.error(f"Security scan failed: {e}")
                    result.error_messages.append(f"Security scan error: {str(e)}")

        except Exception as e:
            logger.error(f"Error running test {test_case.name}: {e}")
            result.error_messages.append(str(e))

        finally:
            # Cleanup
            if test_dir and test_dir.exists():
                shutil.rmtree(test_dir)

            result.duration = (datetime.now() - start_time).total_seconds()

        return result

    def _extract_errors(
        self, stdout: str, stderr: str, exit_code: int
    ) -> List[Dict[str, Any]]:
        """Extract error information from output."""
        errors = []

        # Basic error extraction - can be overridden by specific runners
        if stderr:
            errors.append(
                {
                    "error_type": "RuntimeError",
                    "message": stderr.strip(),
                    "exit_code": exit_code,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return errors

    async def _apply_fix(self, test_dir: Path, fix: Dict[str, Any]) -> bool:
        """Apply a generated fix to the test code."""
        try:
            file_path = test_dir / fix.get("file_path", "")
            if not file_path.exists():
                return False

            # Apply the fix
            if "patch" in fix:
                # Apply patch
                with open(file_path, "r") as f:
                    original = f.read()

                # Simple patch application (can be enhanced)
                patched = fix["patch"].get("new_code", original)

                with open(file_path, "w") as f:
                    f.write(patched)

                return True

        except Exception as e:
            logger.error(f"Error applying fix: {e}")

        return False

    async def _collect_metrics(self, test_dir: Path) -> Dict[str, Any]:
        """Collect metrics from the test run."""
        metrics = {
            "language": self.language,
            "timestamp": datetime.now().isoformat(),
            "test_dir_size": sum(
                f.stat().st_size for f in test_dir.rglob("*") if f.is_file()
            ),
        }

        return metrics


class PythonIntegrationTestRunner(LanguageIntegrationTestRunner):
    """Python-specific integration test runner."""

    async def setup_environment(self, test_case: IntegrationTestCase) -> Path:
        """Set up Python test environment."""
        test_dir = Path(tempfile.mkdtemp(prefix="homeostasis_python_test_"))

        # Write source files
        for filename, content in test_case.source_code.items():
            file_path = test_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

        # Create virtual environment
        venv_path = test_dir / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

        # Install dependencies
        if test_case.dependencies:
            pip_path = venv_path / "bin" / "pip"
            for dep in test_case.dependencies:
                subprocess.run([str(pip_path), "install", dep], check=True)

        return test_dir

    async def execute_code(
        self, test_dir: Path, test_case: IntegrationTestCase
    ) -> Tuple[int, str, str]:
        """Execute Python code."""
        venv_python = test_dir / "venv" / "bin" / "python"
        main_file = test_dir / "main.py"

        if not main_file.exists():
            # Find the first .py file
            py_files = list(test_dir.glob("*.py"))
            if py_files:
                main_file = py_files[0]

        process = subprocess.Popen(
            [str(venv_python), str(main_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=test_dir,
            env={**os.environ, **test_case.environment},
        )

        try:
            stdout, stderr = process.communicate(timeout=test_case.timeout)
            return process.returncode, stdout.decode(), stderr.decode()
        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "", "Test timeout exceeded"

    async def validate_environment(self, test_dir: Path) -> bool:
        """Validate Python environment."""
        venv_python = test_dir / "venv" / "bin" / "python"
        if not venv_python.exists():
            return False

        # Check Python version
        result = subprocess.run(
            [str(venv_python), "--version"], capture_output=True, text=True
        )

        return result.returncode == 0


class JavaScriptIntegrationTestRunner(LanguageIntegrationTestRunner):
    """JavaScript/Node.js integration test runner."""

    async def setup_environment(self, test_case: IntegrationTestCase) -> Path:
        """Set up JavaScript test environment."""
        test_dir = Path(tempfile.mkdtemp(prefix="homeostasis_js_test_"))

        # Write source files
        for filename, content in test_case.source_code.items():
            file_path = test_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

        # Create package.json if needed
        if test_case.dependencies and not (test_dir / "package.json").exists():
            package_json = {
                "name": "homeostasis-test",
                "version": "1.0.0",
                "dependencies": {dep: "*" for dep in test_case.dependencies},
            }
            (test_dir / "package.json").write_text(json.dumps(package_json, indent=2))

        # Install dependencies
        if test_case.dependencies:
            subprocess.run(["npm", "install"], cwd=test_dir, check=True)

        return test_dir

    async def execute_code(
        self, test_dir: Path, test_case: IntegrationTestCase
    ) -> Tuple[int, str, str]:
        """Execute JavaScript code."""
        main_file = test_dir / "index.js"

        if not main_file.exists():
            # Find the first .js file
            js_files = list(test_dir.glob("*.js"))
            if js_files:
                main_file = js_files[0]

        process = subprocess.Popen(
            ["node", str(main_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=test_dir,
            env={**os.environ, **test_case.environment},
        )

        try:
            stdout, stderr = process.communicate(timeout=test_case.timeout)
            return process.returncode, stdout.decode(), stderr.decode()
        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "", "Test timeout exceeded"

    async def validate_environment(self, test_dir: Path) -> bool:
        """Validate Node.js environment."""
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)

        return result.returncode == 0


class IntegrationTestOrchestrator:
    """Orchestrates integration tests across all languages."""

    def __init__(self):
        self.plugin_registry = LanguagePluginRegistry()
        self.runners: Dict[str, LanguageIntegrationTestRunner] = {}
        self.cross_language_orchestrator = CrossLanguageOrchestrator()
        self.test_suites: Dict[str, List[IntegrationTestCase]] = {}
        self._initialize_runners()

    def _initialize_runners(self):
        """Initialize test runners for all supported languages."""
        # Map of language to runner class
        runner_classes = {
            "python": PythonIntegrationTestRunner,
            "javascript": JavaScriptIntegrationTestRunner,
            # Add more runners as implemented
        }

        # Load all plugins and create runners
        for language_id, plugin in self.plugin_registry.plugins.items():
            runner_class = runner_classes.get(
                language_id, LanguageIntegrationTestRunner
            )
            self.runners[language_id] = runner_class(language_id, plugin)

    async def run_all_tests(
        self, parallel: bool = True, max_workers: int = 4
    ) -> Dict[str, List[IntegrationTestResult]]:
        """Run all integration tests."""
        results = {}

        if parallel:
            # Run tests in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                for language, test_cases in self.test_suites.items():
                    if language not in self.runners:
                        logger.warning(f"No runner for language: {language}")
                        continue

                    runner = self.runners[language]
                    for test_case in test_cases:
                        future = executor.submit(
                            asyncio.run, runner.run_test(test_case)
                        )
                        futures.append((language, future))

                # Collect results
                for language, future in futures:
                    if language not in results:
                        results[language] = []
                    try:
                        result = future.result()
                        results[language].append(result)
                    except Exception as e:
                        logger.error(f"Error running test for {language}: {e}")

        else:
            # Run tests sequentially
            for language, test_cases in self.test_suites.items():
                if language not in self.runners:
                    continue

                results[language] = []
                runner = self.runners[language]

                for test_case in test_cases:
                    result = await runner.run_test(test_case)
                    results[language].append(result)

        return results

    def load_test_suites(self, test_dir: Path):
        """Load test suites from directory structure."""
        for language_dir in test_dir.iterdir():
            if not language_dir.is_dir():
                continue

            language = language_dir.name
            test_cases = []

            # Load test cases from JSON files
            for test_file in language_dir.glob("*.json"):
                with open(test_file) as f:
                    test_data = json.load(f)

                if isinstance(test_data, list):
                    for case_data in test_data:
                        test_case = IntegrationTestCase(**case_data)
                        test_cases.append(test_case)
                else:
                    test_case = IntegrationTestCase(**test_data)
                    test_cases.append(test_case)

            if test_cases:
                self.test_suites[language] = test_cases

    def generate_report(
        self, results: Dict[str, List[IntegrationTestResult]]
    ) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_languages": len(results),
                "total_tests": sum(len(r) for r in results.values()),
                "passed": sum(1 for r in results.values() for t in r if t.passed),
                "failed": sum(1 for r in results.values() for t in r if not t.passed),
            },
            "languages": {},
        }

        for language, test_results in results.items():
            language_summary = {
                "total": len(test_results),
                "passed": sum(1 for r in test_results if r.passed),
                "failed": sum(1 for r in test_results if not r.passed),
                "average_duration": (
                    sum(r.duration for r in test_results) / len(test_results)
                    if test_results
                    else 0
                ),
                "test_details": [],
            }

            for result in test_results:
                test_detail = {
                    "name": result.test_case.name,
                    "passed": result.passed,
                    "duration": result.duration,
                    "error_count": len(result.detected_errors),
                    "fix_count": len(result.generated_fixes),
                    "applied_fix_count": len(result.applied_fixes),
                    "errors": result.error_messages,
                }

                # Add security scan summary if available
                if result.security_report:
                    test_detail["security_summary"] = {
                        "total_vulnerabilities": result.security_report["summary"][
                            "total_vulnerabilities"
                        ],
                        "critical": result.security_report["summary"]["by_severity"][
                            "critical"
                        ],
                        "high": result.security_report["summary"]["by_severity"][
                            "high"
                        ],
                        "medium": result.security_report["summary"]["by_severity"][
                            "medium"
                        ],
                        "low": result.security_report["summary"]["by_severity"]["low"],
                    }

                language_summary["test_details"].append(test_detail)

            report["languages"][language] = language_summary

        return report


# Test execution helpers
async def run_integration_tests(
    test_dir: Path = None, languages: List[str] = None
) -> Dict[str, Any]:
    """Run integration tests for specified languages or all languages."""
    orchestrator = IntegrationTestOrchestrator()

    # Load test suites
    if test_dir is None:
        test_dir = Path(__file__).parent / "test_suites"
    orchestrator.load_test_suites(test_dir)

    # Filter by languages if specified
    if languages:
        orchestrator.test_suites = {
            lang: cases
            for lang, cases in orchestrator.test_suites.items()
            if lang in languages
        }

    # Run tests
    results = await orchestrator.run_all_tests()

    # Generate report
    report = orchestrator.generate_report(results)

    return report


if __name__ == "__main__":
    # Run all integration tests
    languages = sys.argv[1:] if len(sys.argv) > 1 else None
    report = asyncio.run(run_integration_tests(languages=languages))

    print(json.dumps(report, indent=2))
