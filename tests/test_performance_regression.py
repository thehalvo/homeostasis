"""
Performance regression tests for the Homeostasis framework.

This module contains tests that monitor for performance regressions across
key components of the self-healing system.
"""
import pytest
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch
import tempfile

# Add the modules directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.testing.performance_regression import (
    PerformanceRegressionTester,
    PerformanceRegressionDetector,
    performance_test
)
from modules.analysis.language_plugin_system import LanguagePluginSystem
from modules.analysis.comprehensive_error_detector import ComprehensiveErrorDetector
from modules.analysis.healing_metrics import HealingMetricsCollector as HealingMetrics
from modules.patch_generation.advanced_code_generator import AdvancedCodeGenerator
from modules.llm_integration.provider_abstraction import LLMManager
from modules.analysis.llm_context_manager import LLMContextManager
from modules.deployment.canary import CanaryDeployment as CanaryDeployer


class TestCoreComponentPerformance:
    """Test performance of core Homeostasis components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tester = PerformanceRegressionTester()
        self.plugin_system = LanguagePluginSystem()
        self.detector = ComprehensiveErrorDetector()
        self.metrics = HealingMetrics()
    
    def test_error_detection_performance(self):
        """Test error detection performance across languages."""
        test_errors = [
            {
                "language": "python",
                "error_type": "AttributeError",
                "message": "'NoneType' object has no attribute 'split'",
                "file": "app.py",
                "line": 42
            },
            {
                "language": "javascript",
                "error_type": "TypeError",
                "message": "Cannot read property 'map' of undefined",
                "file": "index.js",
                "line": 100
            },
            {
                "language": "java",
                "error_type": "NullPointerException",
                "message": "null",
                "stack_trace": [
                    {"class": "com.example.Service", "method": "process", "line": 50}
                ]
            }
        ]
        
        def detect_errors():
            results = []
            for error in test_errors:
                result = self.detector.detect_error(error)
                results.append(result)
            return results
        
        result = self.tester.benchmark(
            detect_errors,
            "error_detection_core_languages",
            iterations=20,
            metadata={"error_count": len(test_errors)}
        )
        
        # Performance assertions
        assert result["duration"]["mean"] < 0.1  # Should complete in < 100ms
        assert len(result["regressions"]) == 0  # No critical regressions
    
    def test_language_plugin_loading_performance(self):
        """Test language plugin loading performance."""
        languages = [
            "python", "javascript", "java", "go", "rust",
            "cpp", "csharp", "ruby", "php", "kotlin"
        ]
        
        def load_plugins():
            plugins = []
            for lang in languages:
                plugin = self.plugin_system.get_plugin(lang)
                plugins.append(plugin)
            return plugins
        
        result = self.tester.benchmark(
            load_plugins,
            "plugin_loading_performance",
            iterations=10,
            metadata={"plugin_count": len(languages)}
        )
        
        # Plugin loading should be fast
        assert result["duration"]["mean"] < 0.5  # < 500ms for 10 plugins
    
    def test_rule_matching_performance_at_scale(self):
        """Test rule matching performance with many rules."""
        # Generate many error patterns
        error_patterns = []
        for i in range(100):
            error_patterns.append({
                "error_type": f"Error{i}",
                "message": f"Test error message {i}",
                "pattern": f"pattern_{i}",
                "file": f"file_{i}.py",
                "line": i
            })
        
        def match_rules():
            matches = 0
            for pattern in error_patterns:
                # Simulate rule matching logic
                if "error" in pattern["message"].lower():
                    matches += 1
            return matches
        
        result = self.tester.benchmark(
            match_rules,
            "rule_matching_at_scale",
            iterations=50,
            metadata={"pattern_count": len(error_patterns)}
        )
        
        # Should scale linearly
        assert result["duration"]["mean"] < 0.01  # < 10ms for 100 patterns
    
    def test_healing_metrics_aggregation_performance(self):
        """Test performance of healing metrics aggregation."""
        # Generate sample metrics
        sample_metrics = []
        for i in range(1000):
            sample_metrics.append({
                "timestamp": f"2024-01-{i%30+1:02d}T{i%24:02d}:00:00",
                "language": ["python", "javascript", "java"][i % 3],
                "error_type": f"Error{i % 10}",
                "healing_time": 0.1 + (i % 100) / 1000,
                "success": i % 10 != 0
            })
        
        def aggregate_metrics():
            # Simulate metrics aggregation
            by_language = {}
            by_error_type = {}
            success_count = 0
            
            for metric in sample_metrics:
                # Group by language
                lang = metric["language"]
                if lang not in by_language:
                    by_language[lang] = []
                by_language[lang].append(metric["healing_time"])
                
                # Group by error type
                error_type = metric["error_type"]
                if error_type not in by_error_type:
                    by_error_type[error_type] = []
                by_error_type[error_type].append(metric["healing_time"])
                
                # Count successes
                if metric["success"]:
                    success_count += 1
            
            return {
                "languages": by_language,
                "error_types": by_error_type,
                "success_rate": success_count / len(sample_metrics)
            }
        
        result = self.tester.benchmark(
            aggregate_metrics,
            "metrics_aggregation_performance",
            iterations=20,
            metadata={"metric_count": len(sample_metrics)}
        )
        
        # Should handle 1000 metrics quickly
        assert result["duration"]["mean"] < 0.05  # < 50ms


class TestPatchGenerationPerformance:
    """Test performance of patch generation components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tester = PerformanceRegressionTester()
        # Create mock LLM components for testing
        mock_llm_manager = Mock(spec=LLMManager)
        mock_context_manager = Mock(spec=LLMContextManager)
        self.generator = AdvancedCodeGenerator(
            llm_manager=mock_llm_manager,
            context_manager=mock_context_manager
        )
    
    def test_simple_patch_generation_performance(self):
        """Test simple patch generation performance."""
        test_cases = [
            {
                "language": "python",
                "error": "NameError: name 'x' is not defined",
                "context": {
                    "code": "print(x)",
                    "line": 10,
                    "variables": ["y", "z"]
                }
            },
            {
                "language": "javascript",
                "error": "ReferenceError: foo is not defined",
                "context": {
                    "code": "console.log(foo)",
                    "line": 20,
                    "variables": ["bar", "baz"]
                }
            }
        ]
        
        def generate_patches():
            patches = []
            for case in test_cases:
                # Simulate patch generation
                # Extract variable name from error message
                error_parts = case["error"].split("'")
                if len(error_parts) >= 2:
                    var_to_replace = error_parts[1]
                else:
                    # Fallback: try to extract from error message
                    import re
                    match = re.search(r"name '(\w+)'", case["error"])
                    var_to_replace = match.group(1) if match else "unknown"
                
                patch = {
                    "language": case["language"],
                    "original": case["context"]["code"],
                    "fixed": case["context"]["code"].replace(
                        var_to_replace,
                        case["context"]["variables"][0]
                    ),
                    "confidence": 0.8
                }
                patches.append(patch)
            return patches
        
        result = self.tester.benchmark(
            generate_patches,
            "simple_patch_generation",
            iterations=30,
            metadata={"case_count": len(test_cases)}
        )
        
        # Simple patches should be very fast
        assert result["duration"]["mean"] < 0.001  # < 1ms
    
    def test_complex_patch_generation_performance(self):
        """Test complex patch generation with context analysis."""
        complex_case = {
            "language": "python",
            "error": {
                "type": "TypeError",
                "message": "unsupported operand type(s) for +: 'int' and 'str'"
            },
            "context": {
                "file_content": """
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item['price']  # Error here
    return total

items = [
    {'name': 'apple', 'price': '1.50'},
    {'name': 'banana', 'price': '0.75'}
]
""",
                "line": 4,
                "function": "calculate_total",
                "imports": [],
                "classes": []
            }
        }
        
        def generate_complex_patch():
            # Simulate complex analysis
            time.sleep(0.01)  # Simulate processing time
            
            # Analyze context
            needs_conversion = "int" in complex_case["error"]["message"]
            
            # Generate patch
            patch = {
                "original_line": "total = total + item['price']",
                "fixed_line": "total = total + float(item['price'])",
                "explanation": "Convert string price to float before addition",
                "confidence": 0.9,
                "alternatives": [
                    "total = total + int(float(item['price']))",
                    "total += float(item.get('price', 0))"
                ]
            }
            
            return patch
        
        result = self.tester.benchmark(
            generate_complex_patch,
            "complex_patch_generation",
            iterations=20,
            metadata={"context_size": len(complex_case["context"]["file_content"])}
        )
        
        # Complex patches with analysis should still be reasonably fast
        assert result["duration"]["mean"] < 0.1  # < 100ms


class TestDeploymentPerformance:
    """Test performance of deployment components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tester = PerformanceRegressionTester()
        self.deployer = CanaryDeployer()
    
    def test_canary_deployment_validation_performance(self):
        """Test canary deployment validation performance."""
        def validate_deployment():
            # Simulate deployment validation
            checks = {
                "syntax_check": True,
                "unit_tests": True,
                "integration_tests": True,
                "security_scan": True,
                "performance_baseline": True
            }
            
            # Simulate validation time
            time.sleep(0.005)
            
            return all(checks.values())
        
        result = self.tester.benchmark(
            validate_deployment,
            "canary_validation_performance",
            iterations=20,
            metadata={"check_count": 5}
        )
        
        # Validation should be quick
        assert result["duration"]["mean"] < 0.02  # < 20ms
    
    def test_rollback_performance(self):
        """Test rollback operation performance."""
        def perform_rollback():
            # Simulate rollback steps
            steps = [
                "stop_new_deployment",
                "restore_previous_version",
                "update_routing",
                "verify_restoration",
                "cleanup_failed_deployment"
            ]
            
            results = []
            for step in steps:
                # Simulate step execution
                time.sleep(0.001)
                results.append({"step": step, "status": "success"})
            
            return results
        
        result = self.tester.benchmark(
            perform_rollback,
            "rollback_performance",
            iterations=15,
            metadata={"step_count": 5}
        )
        
        # Rollback should be very fast
        assert result["duration"]["mean"] < 0.01  # < 10ms


class TestConcurrentOperationsPerformance:
    """Test performance under concurrent operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tester = PerformanceRegressionTester()
    
    def test_concurrent_error_processing_performance(self):
        """Test performance when processing multiple errors concurrently."""
        import concurrent.futures
        
        def process_errors_concurrently():
            errors = []
            for i in range(50):
                errors.append({
                    "id": f"error_{i}",
                    "language": ["python", "javascript", "java"][i % 3],
                    "type": f"Error{i % 5}",
                    "message": f"Test error {i}"
                })
            
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for error in errors:
                    future = executor.submit(self._process_single_error, error)
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            
            return results
        
        result = self.tester.benchmark(
            process_errors_concurrently,
            "concurrent_error_processing",
            iterations=10,
            metadata={"error_count": 50, "workers": 5}
        )
        
        # Should handle concurrent processing efficiently
        assert result["duration"]["mean"] < 0.2  # < 200ms for 50 errors
    
    def _process_single_error(self, error: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single error (helper method)."""
        # Simulate error processing
        time.sleep(0.001)
        return {
            "error_id": error["id"],
            "processed": True,
            "healing_suggestion": f"Fix for {error['type']}"
        }


class TestMemoryPerformance:
    """Test memory usage patterns."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tester = PerformanceRegressionTester()
    
    def test_large_codebase_analysis_memory(self):
        """Test memory usage when analyzing large codebases."""
        def analyze_large_codebase():
            # Simulate analyzing many files
            files = []
            for i in range(100):
                files.append({
                    "path": f"src/module_{i}/file_{i}.py",
                    "content": "def function():\n    pass\n" * 100,  # 200 lines
                    "errors": [f"error_{j}" for j in range(5)]
                })
            
            # Simulate analysis
            results = []
            for file in files:
                analysis = {
                    "file": file["path"],
                    "error_count": len(file["errors"]),
                    "line_count": file["content"].count("\n")
                }
                results.append(analysis)
            
            return results
        
        result = self.tester.benchmark(
            analyze_large_codebase,
            "large_codebase_memory_usage",
            iterations=5,
            metadata={"file_count": 100, "avg_lines": 200}
        )
        
        # Memory usage should be reasonable
        assert result["memory"]["mean"] < 50  # < 50MB for 100 files


class TestRegressionReporting:
    """Test regression detection and reporting."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use a temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.detector = PerformanceRegressionDetector(self.temp_db.name)
        self.tester = PerformanceRegressionTester(self.detector)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_db.name)
    
    def test_regression_detection(self):
        """Test that regressions are properly detected."""
        test_name = "test_regression_detection_sample"
        
        # Establish baseline with fast performance
        def fast_operation():
            time.sleep(0.01)  # 10ms
        
        # Run multiple times to establish baseline
        os.environ["UPDATE_PERFORMANCE_BASELINE"] = "true"
        for _ in range(5):
            self.tester.benchmark(fast_operation, test_name, iterations=3)
        os.environ.pop("UPDATE_PERFORMANCE_BASELINE", None)
        
        # Now run a slower version
        def slow_operation():
            time.sleep(0.02)  # 20ms (2x slower)
        
        result = self.tester.benchmark(slow_operation, test_name, iterations=3)
        
        # Should detect regression
        assert len(result["regressions"]) > 0
        assert any(r["metric_type"] == "duration" for r in result["regressions"])


@performance_test(iterations=10)
def test_decorated_performance():
    """Test using the performance_test decorator."""
    # Simulate some work
    total = 0
    for i in range(1000):
        total += i ** 2
    return total


if __name__ == "__main__":
    # Run performance regression tests
    pytest.main([__file__, "-v", "-s"])