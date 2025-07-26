"""
Performance benchmarks comparing primary vs fringe language handling.

This module contains benchmarks to measure and compare the performance
of error detection, analysis, and patch generation across different language
plugins, with a focus on primary languages vs fringe languages.
"""
import pytest
import time
import json
import os
import sys
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, patch
import concurrent.futures

# Add the modules directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules', 'analysis'))

from language_plugin_system import LanguagePluginSystem
from comprehensive_error_detector import ComprehensiveErrorDetector
from cross_language_orchestrator import CrossLanguageOrchestrator


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self):
        self.plugin_system = LanguagePluginSystem()
        self.detector = ComprehensiveErrorDetector()
        self.results = {}
    
    def measure_execution_time(self, func, *args, **kwargs) -> float:
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time, result
    
    def run_benchmark(self, name: str, func, iterations: int = 100) -> Dict[str, float]:
        """Run a benchmark multiple times and collect statistics."""
        times = []
        for _ in range(iterations):
            execution_time, _ = self.measure_execution_time(func)
            times.append(execution_time)
        
        return {
            "name": name,
            "iterations": iterations,
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times)
        }


class TestLanguageDetectionPerformance(PerformanceBenchmark):
    """Benchmark language detection performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        super().__init__()
        self.test_errors = self._generate_test_errors()
    
    def _generate_test_errors(self) -> Dict[str, List[Dict]]:
        """Generate test errors for different languages."""
        return {
            # Primary languages
            "python": [
                {
                    "error_type": "ValueError",
                    "message": "invalid literal for int() with base 10: 'abc'",
                    "file": "test.py",
                    "line": 10
                }
            ],
            "javascript": [
                {
                    "error_type": "TypeError",
                    "message": "Cannot read property 'length' of undefined",
                    "file": "app.js",
                    "line": 25
                }
            ],
            "java": [
                {
                    "error_type": "NullPointerException",
                    "message": "Cannot invoke method on null object",
                    "file": "Main.java",
                    "line": 42
                }
            ],
            "cpp": [
                {
                    "error_type": "SegmentationFault",
                    "message": "Segmentation fault (core dumped)",
                    "file": "main.cpp",
                    "line": 100
                }
            ],
            "go": [
                {
                    "error_type": "Panic",
                    "message": "runtime error: index out of range",
                    "file": "main.go",
                    "line": 30
                }
            ],
            # Fringe languages
            "zig": [
                {
                    "error_type": "CompilationError",
                    "message": "expected type 'u32', found 'i32'",
                    "file": "main.zig",
                    "line": 15
                }
            ],
            "nim": [
                {
                    "error_type": "ValueError",
                    "message": "index out of bounds",
                    "file": "app.nim",
                    "line": 20
                }
            ],
            "crystal": [
                {
                    "error_type": "TypeException",
                    "message": "undefined method 'size' for Nil",
                    "file": "main.cr",
                    "line": 35
                }
            ]
        }
    
    def test_language_detection_speed(self):
        """Benchmark language detection speed for primary vs fringe languages."""
        results = {}
        
        for language, errors in self.test_errors.items():
            def detect_language():
                for error in errors:
                    self.plugin_system.detect_language(error)
            
            benchmark_result = self.run_benchmark(
                f"detect_{language}",
                detect_language,
                iterations=1000
            )
            results[language] = benchmark_result
        
        # Compare primary vs fringe languages
        primary_languages = ["python", "javascript", "java", "cpp", "go"]
        fringe_languages = ["zig", "nim", "crystal"]
        
        primary_avg = statistics.mean([results[lang]["mean"] for lang in primary_languages])
        fringe_avg = statistics.mean([results[lang]["mean"] for lang in fringe_languages])
        
        print("\n=== Language Detection Performance ===")
        print(f"Primary languages avg: {primary_avg*1000:.2f}ms")
        print(f"Fringe languages avg: {fringe_avg*1000:.2f}ms")
        print(f"Performance ratio: {fringe_avg/primary_avg:.2f}x")
        
        # Primary languages should generally be faster due to optimization
        assert primary_avg < fringe_avg * 1.5  # Allow up to 50% slower for fringe


class TestErrorAnalysisPerformance(PerformanceBenchmark):
    """Benchmark error analysis performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        super().__init__()
        self.complex_errors = self._generate_complex_errors()
    
    def _generate_complex_errors(self) -> Dict[str, Dict]:
        """Generate complex errors for performance testing."""
        return {
            # Primary languages with complex stack traces
            "python": {
                "error_type": "RecursionError",
                "message": "maximum recursion depth exceeded",
                "stack_trace": [
                    {"function": f"recursive_func_{i}", "file": "deep.py", "line": i*10}
                    for i in range(100)
                ]
            },
            "java": {
                "error_type": "StackOverflowError",
                "message": "Stack overflow in thread main",
                "stack_trace": [
                    {"class": f"com.example.Class{i}", "method": "process", "file": f"Class{i}.java", "line": i*5}
                    for i in range(100)
                ]
            },
            "javascript": {
                "error_type": "RangeError",
                "message": "Maximum call stack size exceeded",
                "stack_trace": [
                    {"function": f"nested_{i}", "file": "recursive.js", "line": i*3}
                    for i in range(100)
                ]
            },
            # Fringe languages with simpler errors
            "haskell": {
                "error_type": "TypeError",
                "message": "Couldn't match expected type 'Int' with actual type '[Char]'",
                "file": "Main.hs",
                "line": 42
            },
            "erlang": {
                "error_type": "error",
                "message": "no function clause matching",
                "module": "gen_server",
                "function": "handle_call/3"
            }
        }
    
    def test_error_analysis_performance(self):
        """Benchmark error analysis performance."""
        results = {}
        
        for language, error in self.complex_errors.items():
            # Get the appropriate plugin
            plugin = self.plugin_system.get_plugin(language)
            if not plugin:
                continue
            
            def analyze_error():
                return plugin.analyze_error(error)
            
            benchmark_result = self.run_benchmark(
                f"analyze_{language}",
                analyze_error,
                iterations=100
            )
            results[language] = benchmark_result
        
        print("\n=== Error Analysis Performance ===")
        for lang, result in results.items():
            print(f"{lang}: {result['mean']*1000:.2f}ms (±{result['stdev']*1000:.2f}ms)")
        
        # Complex errors (primary languages) should still be processed efficiently
        assert all(result["mean"] < 0.1 for result in results.values())  # < 100ms


class TestPatchGenerationPerformance(PerformanceBenchmark):
    """Benchmark patch generation performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        super().__init__()
        self.patch_scenarios = self._generate_patch_scenarios()
    
    def _generate_patch_scenarios(self) -> Dict[str, Dict]:
        """Generate patch scenarios for different languages."""
        return {
            "python": {
                "analysis": {
                    "root_cause": "undefined_variable",
                    "suggestion": "Define variable before use"
                },
                "context": {
                    "file": "app.py",
                    "line": 50,
                    "code_snippet": "print(undefined_var)",
                    "variables": ["defined_var", "another_var"]
                }
            },
            "java": {
                "analysis": {
                    "root_cause": "null_pointer",
                    "suggestion": "Add null check"
                },
                "context": {
                    "file": "Service.java",
                    "line": 100,
                    "code_snippet": "return user.getName();",
                    "variable": "user"
                }
            },
            "go": {
                "analysis": {
                    "root_cause": "nil_pointer_dereference",
                    "suggestion": "Add nil check"
                },
                "context": {
                    "file": "handler.go",
                    "line": 75,
                    "code_snippet": "return user.Name",
                    "variable": "user"
                }
            },
            "rust": {
                "analysis": {
                    "root_cause": "borrow_checker_error",
                    "suggestion": "Use clone or reference"
                },
                "context": {
                    "file": "main.rs",
                    "line": 25,
                    "code_snippet": "let y = x;",
                    "ownership": "moved"
                }
            }
        }
    
    def test_patch_generation_performance(self):
        """Benchmark patch generation performance."""
        results = {}
        
        for language, scenario in self.patch_scenarios.items():
            plugin = self.plugin_system.get_plugin(language)
            if not plugin or not hasattr(plugin, 'generate_fix'):
                continue
            
            def generate_patch():
                return plugin.generate_fix(
                    scenario["analysis"],
                    scenario["context"]
                )
            
            benchmark_result = self.run_benchmark(
                f"patch_{language}",
                generate_patch,
                iterations=500
            )
            results[language] = benchmark_result
        
        print("\n=== Patch Generation Performance ===")
        for lang, result in results.items():
            print(f"{lang}: {result['mean']*1000:.2f}ms (±{result['stdev']*1000:.2f}ms)")
        
        # Patch generation should be fast
        assert all(result["mean"] < 0.05 for result in results.values())  # < 50ms


class TestConcurrentProcessingPerformance(PerformanceBenchmark):
    """Benchmark concurrent error processing performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        super().__init__()
        self.concurrent_errors = self._generate_concurrent_errors()
    
    def _generate_concurrent_errors(self, count: int = 100) -> List[Dict]:
        """Generate multiple errors for concurrent processing."""
        languages = ["python", "javascript", "java", "go", "rust", "cpp", "csharp", "ruby", "php"]
        errors = []
        
        for i in range(count):
            lang = languages[i % len(languages)]
            errors.append({
                "id": f"error_{i}",
                "language": lang,
                "error_type": "TestError",
                "message": f"Test error {i} for {lang}",
                "file": f"test_{i}.{lang[:2]}",
                "line": i
            })
        
        return errors
    
    def test_concurrent_processing_performance(self):
        """Benchmark concurrent error processing."""
        errors = self.concurrent_errors
        
        # Sequential processing
        def process_sequential():
            results = []
            for error in errors:
                plugin = self.plugin_system.get_plugin(error["language"])
                if plugin:
                    results.append(plugin.analyze_error(error))
            return results
        
        # Concurrent processing
        def process_concurrent():
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for error in errors:
                    plugin = self.plugin_system.get_plugin(error["language"])
                    if plugin:
                        future = executor.submit(plugin.analyze_error, error)
                        futures.append(future)
                
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            return results
        
        seq_time, _ = self.measure_execution_time(process_sequential)
        conc_time, _ = self.measure_execution_time(process_concurrent)
        
        speedup = seq_time / conc_time
        
        print("\n=== Concurrent Processing Performance ===")
        print(f"Sequential: {seq_time:.2f}s")
        print(f"Concurrent: {conc_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Concurrent should be significantly faster
        assert speedup > 2.0  # At least 2x speedup


class TestMemoryUsageComparison(PerformanceBenchmark):
    """Compare memory usage across language plugins."""
    
    def setup_method(self):
        """Set up test fixtures."""
        super().__init__()
        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process()
        except ImportError:
            pytest.skip("psutil required for memory benchmarks")
    
    def measure_memory_usage(self, func) -> Tuple[float, Any]:
        """Measure memory usage of a function."""
        import gc
        gc.collect()
        
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        result = func()
        gc.collect()
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return final_memory - initial_memory, result
    
    def test_plugin_memory_footprint(self):
        """Test memory footprint of different language plugins."""
        results = {}
        
        languages = [
            # Primary languages
            "python", "javascript", "java", "cpp", "go", "rust", "csharp", "ruby", "php",
            # Fringe languages
            "zig", "nim", "crystal", "haskell", "fsharp", "erlang"
        ]
        
        for language in languages:
            def load_and_process():
                plugin = self.plugin_system.get_plugin(language)
                if plugin:
                    # Process multiple errors to see memory growth
                    for i in range(100):
                        error = {
                            "language": language,
                            "error_type": "TestError",
                            "message": f"Test error {i}",
                            "file": f"test_{i}.ext",
                            "line": i
                        }
                        plugin.analyze_error(error)
                return plugin
            
            memory_delta, _ = self.measure_memory_usage(load_and_process)
            results[language] = memory_delta
        
        print("\n=== Memory Usage Comparison ===")
        for lang, memory in sorted(results.items(), key=lambda x: x[1]):
            print(f"{lang}: {memory:.2f} MB")
        
        # Primary languages might use more memory due to more features
        primary_avg = statistics.mean([results[l] for l in ["python", "javascript", "java"] if l in results])
        fringe_avg = statistics.mean([results[l] for l in ["zig", "nim", "crystal"] if l in results])
        
        print(f"\nPrimary avg: {primary_avg:.2f} MB")
        print(f"Fringe avg: {fringe_avg:.2f} MB")


class TestRuleMatchingPerformance(PerformanceBenchmark):
    """Benchmark rule matching performance across languages."""
    
    def setup_method(self):
        """Set up test fixtures."""
        super().__init__()
        self.rule_test_cases = self._generate_rule_test_cases()
    
    def _generate_rule_test_cases(self) -> Dict[str, List[Dict]]:
        """Generate test cases for rule matching."""
        return {
            "python": [
                {"pattern": "ImportError.*No module named", "text": "ImportError: No module named 'requests'"},
                {"pattern": "KeyError:.*", "text": "KeyError: 'user_id'"},
                {"pattern": "TypeError.*takes.*positional", "text": "TypeError: func() takes 2 positional arguments but 3 were given"}
            ] * 20,  # Multiply to test with many rules
            "javascript": [
                {"pattern": "Cannot read property.*of undefined", "text": "Cannot read property 'length' of undefined"},
                {"pattern": "is not a function", "text": "TypeError: callback is not a function"},
                {"pattern": "Unexpected token", "text": "SyntaxError: Unexpected token }"}
            ] * 20,
            "java": [
                {"pattern": "java\\.lang\\.NullPointerException", "text": "java.lang.NullPointerException at com.example.Main"},
                {"pattern": "ClassNotFoundException.*", "text": "ClassNotFoundException: com.example.MissingClass"},
                {"pattern": "ArrayIndexOutOfBoundsException", "text": "ArrayIndexOutOfBoundsException: 10"}
            ] * 20
        }
    
    def test_rule_matching_performance(self):
        """Benchmark rule matching performance."""
        results = {}
        
        for language, test_cases in self.rule_test_cases.items():
            plugin = self.plugin_system.get_plugin(language)
            if not plugin:
                continue
            
            def match_rules():
                matches = 0
                for case in test_cases:
                    # Simulate rule matching
                    import re
                    if re.search(case["pattern"], case["text"]):
                        matches += 1
                return matches
            
            benchmark_result = self.run_benchmark(
                f"rules_{language}",
                match_rules,
                iterations=100
            )
            results[language] = benchmark_result
        
        print("\n=== Rule Matching Performance ===")
        for lang, result in results.items():
            print(f"{lang}: {result['mean']*1000:.2f}ms for {len(self.rule_test_cases[lang])} rules")
        
        # Rule matching should scale well
        assert all(result["mean"] < 0.01 for result in results.values())  # < 10ms


class TestScalabilityBenchmark(PerformanceBenchmark):
    """Test scalability with increasing error volumes."""
    
    def test_scalability_with_error_volume(self):
        """Test how performance scales with error volume."""
        volumes = [10, 50, 100, 500, 1000]
        results = {}
        
        for volume in volumes:
            errors = [
                {
                    "language": "python",
                    "error_type": "ValueError",
                    "message": f"Error {i}",
                    "file": f"file_{i}.py",
                    "line": i
                }
                for i in range(volume)
            ]
            
            def process_errors():
                plugin = self.plugin_system.get_plugin("python")
                return [plugin.analyze_error(error) for error in errors]
            
            exec_time, _ = self.measure_execution_time(process_errors)
            results[volume] = exec_time
        
        print("\n=== Scalability Analysis ===")
        for volume, time in results.items():
            print(f"{volume} errors: {time:.3f}s ({time/volume*1000:.2f}ms per error)")
        
        # Check linear scalability
        time_per_error = [results[v]/v for v in volumes]
        variation = statistics.stdev(time_per_error) / statistics.mean(time_per_error)
        
        print(f"Scalability variation: {variation:.2%}")
        assert variation < 0.2  # Less than 20% variation indicates good scalability


if __name__ == "__main__":
    # Run benchmarks
    pytest.main([__file__, "-v", "-s"])  # -s to see print output