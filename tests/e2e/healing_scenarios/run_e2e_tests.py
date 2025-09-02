#!/usr/bin/env python3
"""
End-to-End Healing Scenario Test Runner

Runs all healing scenario tests and generates a comprehensive report.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))



class E2ETestRunner:
    """Runs end-to-end healing tests with reporting."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("test_results")
        self.output_dir.mkdir(exist_ok=True)
        self.start_time = None
        self.end_time = None
        
    def run_tests(
        self, 
        test_suite: Optional[str] = None,
        verbose: bool = False,
        parallel: bool = False,
        timeout: int = 3600
    ) -> Dict[str, any]:
        """Run the test suite and collect results."""
        self.start_time = datetime.now()
        
        # Prepare pytest arguments
        pytest_args = [
            str(Path(__file__).parent),  # Test directory
            "-v" if verbose else "-q",
            f"--tb={'short' if not verbose else 'long'}",
            f"--timeout={timeout}",
            "--json-report",
            f"--json-report-file={self.output_dir / 'report.json'}",
            "--html=" + str(self.output_dir / "report.html"),
            "--self-contained-html"
        ]
        
        # Add specific test suite if requested
        if test_suite:
            if test_suite == "basic":
                pytest_args.append("test_basic_healing_scenarios.py")
            elif test_suite == "advanced":
                pytest_args.append("test_advanced_healing_scenarios.py")
            elif test_suite == "cross-language":
                pytest_args.append("test_cross_language_healing.py")
                
        # Add parallelization if requested
        if parallel:
            pytest_args.extend(["-n", "auto"])
            
        # Run tests
        exit_code = pytest.main(pytest_args)
        
        self.end_time = datetime.now()
        
        # Generate comprehensive report
        report = self._generate_report(exit_code)
        
        # Save report
        self._save_report(report)
        
        return report
        
    def _generate_report(self, exit_code: int) -> Dict[str, any]:
        """Generate a comprehensive test report."""
        report = {
            "summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration": (self.end_time - self.start_time).total_seconds(),
                "exit_code": exit_code,
                "status": "success" if exit_code == 0 else "failure"
            },
            "test_results": {},
            "metrics": {},
            "recommendations": []
        }
        
        # Load pytest JSON report if available
        json_report_path = self.output_dir / "report.json"
        if json_report_path.exists():
            with open(json_report_path) as f:
                pytest_report = json.load(f)
                
            # Extract test results
            report["summary"]["total_tests"] = pytest_report["summary"]["total"]
            report["summary"]["passed"] = pytest_report["summary"].get("passed", 0)
            report["summary"]["failed"] = pytest_report["summary"].get("failed", 0)
            report["summary"]["skipped"] = pytest_report["summary"].get("skipped", 0)
            
            # Process individual test results
            for test in pytest_report.get("tests", []):
                test_name = test["nodeid"].split("::")[-1]
                report["test_results"][test_name] = {
                    "outcome": test["outcome"],
                    "duration": test.get("duration", 0),
                    "error": test.get("call", {}).get("longrepr") if test["outcome"] == "failed" else None
                }
                
        # Add recommendations based on results
        report["recommendations"] = self._generate_recommendations(report)
        
        return report
        
    def _generate_recommendations(self, report: Dict[str, any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check overall success rate
        if report["summary"].get("total_tests", 0) > 0:
            success_rate = report["summary"].get("passed", 0) / report["summary"]["total_tests"]
            
            if success_rate < 0.8:
                recommendations.append(
                    f"Success rate is {success_rate:.1%}. Consider improving error detection rules."
                )
                
            if success_rate < 0.5:
                recommendations.append(
                    "Critical: Less than 50% of healing scenarios succeed. Review healing logic."
                )
                
        # Check for specific failing patterns
        failed_tests = [
            name for name, result in report["test_results"].items()
            if result["outcome"] == "failed"
        ]
        
        if any("cross_language" in test for test in failed_tests):
            recommendations.append(
                "Cross-language healing needs improvement. Consider adding language-specific rules."
            )
            
        if any("security" in test for test in failed_tests):
            recommendations.append(
                "Security vulnerability healing failed. This is critical and needs immediate attention."
            )
            
        if any("concurrent" in test for test in failed_tests):
            recommendations.append(
                "Concurrent healing scenarios failed. Review synchronization and resource management."
            )
            
        # Performance recommendations
        slow_tests = [
            (name, result["duration"]) 
            for name, result in report["test_results"].items()
            if result.get("duration", 0) > 60
        ]
        
        if slow_tests:
            recommendations.append(
                f"Found {len(slow_tests)} slow healing scenarios (>60s). Consider optimization."
            )
            
        return recommendations
        
    def _save_report(self, report: Dict[str, any]):
        """Save the report to multiple formats."""
        # JSON format
        with open(self.output_dir / "e2e_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        # Human-readable format
        with open(self.output_dir / "e2e_report.txt", "w") as f:
            f.write("=" * 80 + "\n")
            f.write("End-to-End Healing Scenario Test Report\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Start Time: {report['summary']['start_time']}\n")
            f.write(f"End Time: {report['summary']['end_time']}\n")
            f.write(f"Duration: {report['summary']['duration']:.2f} seconds\n")
            f.write(f"Status: {report['summary']['status'].upper()}\n")
            
            if "total_tests" in report["summary"]:
                f.write(f"\nTests Run: {report['summary']['total_tests']}\n")
                f.write(f"Passed: {report['summary']['passed']}\n")
                f.write(f"Failed: {report['summary']['failed']}\n")
                f.write(f"Skipped: {report['summary']['skipped']}\n")
                
            # Test Results
            f.write("\n\nTEST RESULTS\n")
            f.write("-" * 40 + "\n")
            
            for test_name, result in report["test_results"].items():
                status = "✓" if result["outcome"] == "passed" else "✗"
                f.write(f"{status} {test_name} ({result['duration']:.2f}s)\n")
                
                if result.get("error"):
                    f.write(f"  Error: {result['error'][:200]}...\n")
                    
            # Recommendations
            if report["recommendations"]:
                f.write("\n\nRECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")
                
                for i, rec in enumerate(report["recommendations"], 1):
                    f.write(f"{i}. {rec}\n")
                    
        print(f"\nReports saved to: {self.output_dir}")
        

def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end healing scenario tests"
    )
    
    parser.add_argument(
        "--suite",
        choices=["all", "basic", "advanced", "cross-language"],
        default="all",
        help="Test suite to run"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_results"),
        help="Directory for test results"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Overall test timeout in seconds"
    )
    
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode - fail on any test failure"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = E2ETestRunner(output_dir=args.output_dir)
    
    # Run tests
    test_suite = None if args.suite == "all" else args.suite
    report = runner.run_tests(
        test_suite=test_suite,
        verbose=args.verbose,
        parallel=args.parallel,
        timeout=args.timeout
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if "total_tests" in report["summary"]:
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Duration: {report['summary']['duration']:.2f}s")
        
        success_rate = (
            report['summary']['passed'] / report['summary']['total_tests']
            if report['summary']['total_tests'] > 0 else 0
        )
        print(f"Success Rate: {success_rate:.1%}")
        
    # Exit with appropriate code
    if args.ci and report["summary"]["exit_code"] != 0:
        sys.exit(1)
    else:
        sys.exit(0)
        

if __name__ == "__main__":
    main()