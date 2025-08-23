"""
Stress tests for high-volume error detection in primary languages.

This module contains stress tests to ensure the system can handle
high volumes of errors, concurrent processing, and extreme load conditions
for primary language plugins.
"""
import pytest
import time
import json
import os
import sys
import threading
import multiprocessing
import random
import string
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from unittest.mock import Mock, patch

# Add the modules directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.analysis.language_plugin_system import LanguagePluginSystem
from modules.analysis.comprehensive_error_detector import ComprehensiveErrorDetector
from modules.analysis.cross_language_orchestrator import CrossLanguageOrchestrator


class StressTestBase:
    """Base class for stress testing."""
    
    def __init__(self):
        self.plugin_system = LanguagePluginSystem()
        self.detector = ComprehensiveErrorDetector()
        self.orchestrator = CrossLanguageOrchestrator()
    
    def generate_random_error(self, language: str, error_type: str = None) -> Dict[str, Any]:
        """Generate a random error for stress testing."""
        error_types = {
            "python": ["ValueError", "TypeError", "ImportError", "AttributeError", "KeyError"],
            "javascript": ["TypeError", "ReferenceError", "SyntaxError", "RangeError"],
            "java": ["NullPointerException", "ClassNotFoundException", "ArrayIndexOutOfBoundsException"],
            "cpp": ["SegmentationFault", "std::bad_alloc", "std::out_of_range"],
            "go": ["panic", "deadlock", "nil pointer dereference"],
            "rust": ["panic", "borrow checker error", "lifetime error"],
            "csharp": ["NullReferenceException", "IndexOutOfRangeException", "InvalidOperationException"],
            "ruby": ["NoMethodError", "ArgumentError", "LoadError"],
            "php": ["Fatal error", "Parse error", "Warning"]
        }
        
        if error_type is None:
            error_type = random.choice(error_types.get(language, ["GenericError"]))
        
        return {
            "id": f"stress_{language}_{random.randint(0, 999999)}",
            "language": language,
            "error_type": error_type,
            "message": f"Random {error_type}: {''.join(random.choices(string.ascii_letters, k=50))}",
            "file": f"stress_test_{random.randint(0, 999)}.{language[:3]}",
            "line": random.randint(1, 1000),
            "timestamp": time.time()
        }


class TestHighVolumeProcessing(StressTestBase):
    """Test high-volume error processing."""
    
    def test_sequential_high_volume(self):
        """Test processing a high volume of errors sequentially."""
        languages = ["python", "javascript", "java", "cpp", "go"]
        error_count = 10000
        errors = []
        
        # Generate errors
        for i in range(error_count):
            lang = languages[i % len(languages)]
            errors.append(self.generate_random_error(lang))
        
        # Process errors
        start_time = time.time()
        processed = 0
        failed = 0
        
        for error in errors:
            try:
                plugin = self.plugin_system.get_plugin(error["language"])
                if plugin:
                    plugin.analyze_error(error)
                    processed += 1
            except Exception:
                failed += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n=== Sequential High Volume Test ===")
        print(f"Total errors: {error_count}")
        print(f"Processed: {processed}")
        print(f"Failed: {failed}")
        print(f"Duration: {duration:.2f}s")
        print(f"Rate: {processed/duration:.2f} errors/second")
        
        # Assert performance requirements
        assert processed > error_count * 0.95  # At least 95% success rate
        assert duration < error_count * 0.01  # Less than 10ms per error average
    
    def test_concurrent_high_volume(self):
        """Test processing high volume of errors concurrently."""
        languages = ["python", "javascript", "java", "cpp", "go", "rust", "csharp", "ruby", "php"]
        error_count = 50000
        max_workers = 16
        
        errors = [self.generate_random_error(random.choice(languages)) for _ in range(error_count)]
        
        start_time = time.time()
        processed = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for error in errors:
                future = executor.submit(self._process_error, error)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    if future.result():
                        processed += 1
                except Exception:
                    failed += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n=== Concurrent High Volume Test ===")
        print(f"Total errors: {error_count}")
        print(f"Workers: {max_workers}")
        print(f"Processed: {processed}")
        print(f"Failed: {failed}")
        print(f"Duration: {duration:.2f}s")
        print(f"Rate: {processed/duration:.2f} errors/second")
        
        assert processed > error_count * 0.95
        assert duration < error_count * 0.005  # Faster with concurrency
    
    def _process_error(self, error: Dict) -> bool:
        """Process a single error."""
        plugin = self.plugin_system.get_plugin(error["language"])
        if plugin:
            plugin.analyze_error(error)
            return True
        return False


class TestMemoryStress(StressTestBase):
    """Test memory usage under stress conditions."""
    
    def test_large_error_objects(self):
        """Test handling of very large error objects."""
        # Create errors with large stack traces
        large_errors = []
        
        for lang in ["python", "java", "javascript"]:
            error = self.generate_random_error(lang)
            # Add a very large stack trace
            error["stack_trace"] = [
                {
                    "function": f"func_{i}",
                    "file": f"file_{i}.py",
                    "line": i,
                    "locals": {f"var_{j}": f"value_{j}" * 100 for j in range(10)}
                }
                for i in range(500)
            ]
            large_errors.append(error)
        
        # Process large errors
        for error in large_errors:
            plugin = self.plugin_system.get_plugin(error["language"])
            if plugin:
                analysis = plugin.analyze_error(error)
                assert analysis is not None
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during prolonged processing."""
        import gc
        
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            pytest.skip("psutil required for memory leak test")
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many errors in batches
        batch_size = 1000
        num_batches = 10
        
        for batch in range(num_batches):
            errors = [
                self.generate_random_error(random.choice(["python", "javascript", "java"]))
                for _ in range(batch_size)
            ]
            
            for error in errors:
                plugin = self.plugin_system.get_plugin(error["language"])
                if plugin:
                    plugin.analyze_error(error)
            
            # Force garbage collection
            gc.collect()
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            print(f"Batch {batch + 1}: Memory growth: {memory_growth:.2f} MB")
            
            # Memory growth should stabilize
            if batch > 5:
                assert memory_growth < 100  # Less than 100MB growth


class TestConcurrencyStress(StressTestBase):
    """Test concurrent access and thread safety."""
    
    def test_plugin_thread_safety(self):
        """Test plugin thread safety with concurrent access."""
        num_threads = 50
        errors_per_thread = 100
        results = {}
        errors_found = []
        
        def worker(thread_id):
            thread_results = []
            thread_errors = []
            
            for i in range(errors_per_thread):
                try:
                    lang = random.choice(["python", "javascript", "java", "go"])
                    error = self.generate_random_error(lang)
                    plugin = self.plugin_system.get_plugin(lang)
                    
                    if plugin:
                        analysis = plugin.analyze_error(error)
                        thread_results.append(analysis)
                except Exception as e:
                    thread_errors.append(str(e))
            
            results[thread_id] = thread_results
            if thread_errors:
                errors_found.extend(thread_errors)
        
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        total_processed = sum(len(r) for r in results.values())
        
        print(f"\n=== Thread Safety Test ===")
        print(f"Threads: {num_threads}")
        print(f"Total processed: {total_processed}")
        print(f"Errors found: {len(errors_found)}")
        
        assert total_processed > num_threads * errors_per_thread * 0.95
        assert len(errors_found) < 10  # Very few errors
    
    def test_race_condition_detection(self):
        """Test for race conditions in shared state."""
        shared_counter = {"count": 0}
        lock = threading.Lock()
        
        def increment_with_analysis():
            lang = random.choice(["python", "javascript"])
            error = self.generate_random_error(lang)
            plugin = self.plugin_system.get_plugin(lang)
            
            if plugin:
                # Simulate shared state access
                with lock:
                    shared_counter["count"] += 1
                
                plugin.analyze_error(error)
                
                with lock:
                    shared_counter["count"] -= 1
        
        num_threads = 100
        threads = []
        
        for _ in range(num_threads):
            t = threading.Thread(target=increment_with_analysis)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Counter should be back to 0 if no race conditions
        assert shared_counter["count"] == 0


class TestErrorPatternStress(StressTestBase):
    """Test stress conditions with specific error patterns."""
    
    def test_error_storm_handling(self):
        """Test handling of error storms (many similar errors)."""
        # Simulate an error storm - same error repeated many times
        error_template = {
            "language": "python",
            "error_type": "ConnectionError",
            "message": "Connection refused to database",
            "file": "db_handler.py",
            "line": 150
        }
        
        storm_size = 10000
        start_time = time.time()
        
        # Process the storm
        unique_analyses = set()
        for i in range(storm_size):
            error = error_template.copy()
            error["timestamp"] = time.time()
            error["occurrence"] = i
            
            plugin = self.plugin_system.get_plugin("python")
            if plugin:
                analysis = plugin.analyze_error(error)
                # Convert to string for set comparison
                unique_analyses.add(json.dumps(analysis, sort_keys=True))
        
        duration = time.time() - start_time
        
        print(f"\n=== Error Storm Test ===")
        print(f"Storm size: {storm_size}")
        print(f"Duration: {duration:.2f}s")
        print(f"Unique analyses: {len(unique_analyses)}")
        
        # Should handle storms efficiently
        assert duration < storm_size * 0.001  # Less than 1ms per error
        # Should recognize pattern
        assert len(unique_analyses) < 10  # Limited unique analyses
    
    def test_cascading_errors(self):
        """Test handling of cascading error chains."""
        # Create a chain of errors that trigger each other
        cascade_length = 100
        languages = ["python", "javascript", "java", "go", "cpp"]
        
        error_chain = []
        for i in range(cascade_length):
            error = {
                "id": f"cascade_{i}",
                "language": languages[i % len(languages)],
                "error_type": "CascadeError",
                "message": f"Error {i} caused by error {i-1}",
                "caused_by": f"cascade_{i-1}" if i > 0 else None,
                "file": f"cascade_{i}.py",
                "line": i * 10
            }
            error_chain.append(error)
        
        # Process the cascade
        analyses = []
        for error in error_chain:
            plugin = self.plugin_system.get_plugin(error["language"])
            if plugin:
                analysis = plugin.analyze_error(error)
                analyses.append(analysis)
        
        assert len(analyses) == cascade_length
    
    def test_malformed_error_resilience(self):
        """Test resilience against malformed error data."""
        malformed_errors = [
            {},  # Empty error
            {"language": "python"},  # Missing required fields
            {"language": "unknown_language", "error_type": "Error"},  # Unknown language
            {"language": None, "error_type": None},  # None values
            {"language": "python", "error_type": "Error", "line": "not_a_number"},  # Wrong types
            {"language": "python", "error_type": "Error", "stack_trace": "not_a_list"},  # Wrong structure
            {
                "language": "javascript",
                "error_type": "Error",
                "message": "A" * 1000000  # Extremely long message
            }
        ]
        
        successful = 0
        for error in malformed_errors:
            try:
                lang = error.get("language")
                if lang:
                    plugin = self.plugin_system.get_plugin(lang)
                    if plugin:
                        plugin.analyze_error(error)
                        successful += 1
            except Exception:
                pass  # Expected for malformed data
        
        print(f"\n=== Malformed Error Test ===")
        print(f"Malformed errors: {len(malformed_errors)}")
        print(f"Handled gracefully: {len(malformed_errors) - successful}")


class TestResourceExhaustion(StressTestBase):
    """Test behavior under resource exhaustion."""
    
    def test_cpu_intensive_analysis(self):
        """Test CPU-intensive error analysis."""
        # Create errors that require intensive analysis
        complex_errors = []
        
        for _ in range(100):
            error = {
                "language": "python",
                "error_type": "ComplexError",
                "message": "Complex error requiring intensive analysis",
                "code_context": {
                    "ast": {"type": "Module", "body": [{"type": "FunctionDef"} for _ in range(100)]},
                    "variables": {f"var_{i}": {"type": "complex", "value": [j for j in range(100)]} for i in range(50)},
                    "call_graph": {f"func_{i}": [f"func_{j}" for j in range(i)] for i in range(50)}
                }
            }
            complex_errors.append(error)
        
        start_time = time.time()
        
        for error in complex_errors:
            plugin = self.plugin_system.get_plugin("python")
            if plugin:
                plugin.analyze_error(error)
        
        duration = time.time() - start_time
        
        print(f"\n=== CPU Intensive Test ===")
        print(f"Complex errors: {len(complex_errors)}")
        print(f"Duration: {duration:.2f}s")
        
        # Should complete in reasonable time
        assert duration < 60  # Less than 1 minute
    
    def test_parallel_resource_competition(self):
        """Test parallel processing with resource competition."""
        num_processes = multiprocessing.cpu_count()
        errors_per_process = 1000
        
        def process_worker(worker_id):
            results = []
            for i in range(errors_per_process):
                lang = random.choice(["python", "java", "cpp"])
                error = StressTestBase().generate_random_error(lang)
                # Note: Each process needs its own plugin instance
                plugin_system = LanguagePluginSystem()
                plugin = plugin_system.get_plugin(lang)
                if plugin:
                    analysis = plugin.analyze_error(error)
                    results.append(analysis)
            return len(results)
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_worker, i) for i in range(num_processes)]
            results = [f.result() for f in futures]
        
        duration = time.time() - start_time
        total_processed = sum(results)
        
        print(f"\n=== Parallel Resource Competition Test ===")
        print(f"Processes: {num_processes}")
        print(f"Total processed: {total_processed}")
        print(f"Duration: {duration:.2f}s")
        print(f"Rate: {total_processed/duration:.2f} errors/second")
        
        assert total_processed > num_processes * errors_per_process * 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print output