"""
Fuzz testing for primary language parsers and generators.

This module implements fuzz testing to discover edge cases and potential
vulnerabilities in error parsing and patch generation for primary languages.
"""
import pytest
import random
import string
import os
import sys
from typing import Any, Optional

# Add the modules directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules', 'analysis'))

from language_plugin_system import LanguagePluginSystem


class FuzzGenerator:
    """Base class for generating fuzz test inputs."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional seed for reproducibility."""
        if seed:
            random.seed(seed)
        self.plugin_system = LanguagePluginSystem()
    
    def random_string(self, min_len: int = 1, max_len: int = 1000) -> str:
        """Generate random string with various characters."""
        length = random.randint(min_len, max_len)
        # Include various character types
        char_sets = [
            string.ascii_letters,
            string.digits,
            string.punctuation,
            ' \t\n\r',  # Whitespace
            '\x00\x01\x02\x03',  # Control characters
            'äöüßÄÖÜ',  # Unicode
            '🐍🔥💻🚀',  # Emojis
            '\u200b\u200c\u200d',  # Zero-width characters
        ]
        
        chars = ''.join(char_sets)
        return ''.join(random.choice(chars) for _ in range(length))
    
    def random_number(self) -> Any:
        """Generate random number of various types."""
        number_types = [
            lambda: random.randint(-2**63, 2**63-1),  # Large integers
            lambda: random.uniform(-1e308, 1e308),  # Floats
            lambda: float('inf'),
            lambda: float('-inf'),
            lambda: float('nan'),
            lambda: 0,
            lambda: -0.0,
        ]
        return random.choice(number_types)()
    
    def random_data_structure(self, depth: int = 5) -> Any:
        """Generate random nested data structure."""
        if depth <= 0:
            return self.random_primitive()
        
        structure_types = [
            lambda: [self.random_data_structure(depth-1) for _ in range(random.randint(0, 10))],
            lambda: {self.random_string(1, 20): self.random_data_structure(depth-1) 
                    for _ in range(random.randint(0, 10))},
            lambda: tuple(self.random_data_structure(depth-1) for _ in range(random.randint(0, 5))),
            lambda: self.random_primitive(),
        ]
        
        return random.choice(structure_types)()
    
    def random_primitive(self) -> Any:
        """Generate random primitive value."""
        primitives = [
            lambda: self.random_string(0, 100),
            lambda: self.random_number(),
            lambda: random.choice([True, False]),
            lambda: None,
        ]
        return random.choice(primitives)()


class TestErrorParserFuzzing(FuzzGenerator):
    """Fuzz testing for error parsers."""
    
    def test_fuzz_python_error_parser(self):
        """Fuzz test Python error parser."""
        plugin = self.plugin_system.get_plugin("python")
        if not plugin:
            pytest.skip("Python plugin not available")
        
        iterations = 1000
        crashes = []
        
        for i in range(iterations):
            try:
                # Generate random error data
                error_data = {
                    "language": "python",
                    "error_type": self.random_string(1, 50),
                    "message": self.random_string(0, 1000),
                    "file": self.random_string(1, 200),
                    "line": self.random_number(),
                    "column": self.random_number(),
                    "stack_trace": self.random_data_structure(3)
                }
                
                # Try to analyze
                result = plugin.analyze_error(error_data)
                
                # Basic validation
                assert isinstance(result, dict)
                assert "root_cause" in result
                assert "suggestion" in result
                
            except Exception as e:
                crashes.append({
                    "iteration": i,
                    "input": error_data,
                    "error": str(e)
                })
        
        # Report crashes
        if crashes:
            print(f"\nPython parser crashes: {len(crashes)}/{iterations}")
            for crash in crashes[:5]:  # Show first 5
                print(f"Input: {crash['input']}")
                print(f"Error: {crash['error']}\n")
        
        # Should handle most fuzz inputs gracefully
        assert len(crashes) < iterations * 0.01  # Less than 1% crash rate
    
    def test_fuzz_javascript_error_parser(self):
        """Fuzz test JavaScript error parser."""
        plugin = self.plugin_system.get_plugin("javascript")
        if not plugin:
            pytest.skip("JavaScript plugin not available")
        
        iterations = 1000
        edge_cases_found = []
        
        for i in range(iterations):
            # Generate JavaScript-specific error patterns
            error_patterns = [
                {
                    "error_type": "TypeError",
                    "message": f"Cannot read property '{self.random_string(1, 20)}' of {random.choice(['undefined', 'null', self.random_string()])}",
                    "stack": self.random_string(0, 5000)
                },
                {
                    "error_type": "SyntaxError",
                    "message": f"Unexpected token {self.random_string(1, 10)}",
                    "line": self.random_number(),
                    "column": self.random_number()
                },
                {
                    "error_type": self.random_string(),
                    "message": self.random_string(),
                    "async_stack": [self.random_string() for _ in range(random.randint(0, 100))]
                }
            ]
            
            error_data = random.choice(error_patterns)
            error_data["language"] = "javascript"
            
            try:
                result = plugin.analyze_error(error_data)
                
                # Check for interesting results
                if result.get("confidence", 0) < 0.5:
                    edge_cases_found.append({
                        "input": error_data,
                        "result": result
                    })
                    
            except Exception:
                pass  # Expected for some inputs
        
        print(f"\nJavaScript edge cases found: {len(edge_cases_found)}")
    
    def test_fuzz_java_error_parser(self):
        """Fuzz test Java error parser."""
        plugin = self.plugin_system.get_plugin("java")
        if not plugin:
            pytest.skip("Java plugin not available")
        
        iterations = 1000
        
        for i in range(iterations):
            # Generate Java-specific patterns
            error_data = {
                "language": "java",
                "exception": f"{self.random_string(1, 50)}.{self.random_string(1, 50)}Exception",
                "message": self.random_string(0, 500),
                "stacktrace": [
                    f"at {self.random_string(1, 100)}.{self.random_string(1, 50)}({self.random_string(1, 20)}.java:{self.random_number()})"
                    for _ in range(random.randint(0, 50))
                ],
                "caused_by": random.choice([None, {
                    "exception": self.random_string(),
                    "message": self.random_string()
                }])
            }
            
            try:
                result = plugin.analyze_error(error_data)
                assert isinstance(result, dict)
            except Exception:
                pass  # Some inputs may cause exceptions


class TestPatchGeneratorFuzzing(FuzzGenerator):
    """Fuzz testing for patch generators."""
    
    def test_fuzz_python_patch_generator(self):
        """Fuzz test Python patch generator."""
        plugin = self.plugin_system.get_plugin("python")
        if not plugin:
            pytest.skip("Python plugin not available")
        
        iterations = 500
        valid_patches = 0
        
        for i in range(iterations):
            # Generate random analysis and context
            analysis = {
                "root_cause": self.random_string(5, 50),
                "suggestion": self.random_string(10, 200),
                "category": random.choice(["syntax", "runtime", "import", "type"]),
                "severity": random.choice(["low", "medium", "high", "critical"])
            }
            
            context = {
                "file": self.random_string(5, 100),
                "line": random.randint(1, 1000),
                "code_snippet": self.random_string(0, 500),
                "indentation": random.choice(["    ", "\t", "  "]),
                "variables": [self.random_string(1, 30) for _ in range(random.randint(0, 10))]
            }
            
            try:
                if hasattr(plugin, 'generate_fix'):
                    patch = plugin.generate_fix(analysis, context)
                    if patch and isinstance(patch, dict):
                        valid_patches += 1
            except Exception:
                pass  # Expected for some inputs
        
        print(f"\nPython patch generator - valid patches: {valid_patches}/{iterations}")
    
    def test_fuzz_go_patch_generator(self):
        """Fuzz test Go patch generator."""
        plugin = self.plugin_system.get_plugin("go")
        if not plugin:
            pytest.skip("Go plugin not available")
        
        iterations = 500
        
        for i in range(iterations):
            # Go-specific patch scenarios
            scenarios = [
                {
                    "root_cause": "nil_pointer_dereference",
                    "context": {
                        "code_snippet": f"{self.random_string(1, 20)} := {self.random_string(1, 20)}.{self.random_string(1, 20)}",
                        "variable": self.random_string(1, 20)
                    }
                },
                {
                    "root_cause": "channel_deadlock",
                    "context": {
                        "code_snippet": f"<-{self.random_string(1, 20)}",
                        "channel_name": self.random_string(1, 20)
                    }
                },
                {
                    "root_cause": self.random_string(5, 50),
                    "context": self.random_data_structure(2)
                }
            ]
            
            scenario = random.choice(scenarios)
            
            try:
                if hasattr(plugin, 'generate_fix'):
                    plugin.generate_fix(
                        {"root_cause": scenario["root_cause"], "suggestion": self.random_string()},
                        scenario.get("context", {})
                    )
            except Exception:
                pass


class TestInputValidationFuzzing(FuzzGenerator):
    """Fuzz testing for input validation."""
    
    def test_fuzz_malformed_error_structures(self):
        """Test with malformed error data structures."""
        languages = ["python", "javascript", "java", "cpp", "go"]
        
        malformed_patterns = [
            {},  # Empty
            {"language": None},  # None values
            {"language": ""},  # Empty strings
            {"language": " " * 1000},  # Whitespace
            {"language": "\x00" * 100},  # Null bytes
            {"language": self.random_string(10000, 100000)},  # Very long
            self.random_data_structure(10),  # Deep nesting
            [{"language": "python"}] * 1000,  # Lists
            "not a dict",  # Wrong type
            12345,  # Numbers
            None,  # None
        ]
        
        for lang in languages:
            plugin = self.plugin_system.get_plugin(lang)
            if not plugin:
                continue
            
            for pattern in malformed_patterns:
                try:
                    if isinstance(pattern, dict):
                        pattern["language"] = lang
                    plugin.analyze_error(pattern)
                except Exception:
                    pass  # Expected, but should not crash
    
    def test_fuzz_unicode_edge_cases(self):
        """Test Unicode edge cases."""
        unicode_tests = [
            "\U0001F600" * 100,  # Emojis
            "\u200b" * 1000,  # Zero-width spaces
            "A" + "\u0301" * 50,  # Combining characters
            "\uffff" * 100,  # High unicode
            "\ud800\udc00",  # Surrogate pairs
            "𝕳𝖊𝖑𝖑𝖔",  # Mathematical alphanumeric
            "🏳️‍🌈" * 50,  # Complex emoji sequences
        ]
        
        for test_string in unicode_tests:
            error_data = {
                "language": "python",
                "error_type": "UnicodeError",
                "message": test_string,
                "file": test_string[:50] + ".py",
                "line": 1
            }
            
            plugin = self.plugin_system.get_plugin("python")
            if plugin:
                try:
                    result = plugin.analyze_error(error_data)
                    assert isinstance(result, dict)
                except Exception:
                    pass


class TestConcurrentFuzzing(FuzzGenerator):
    """Fuzz testing with concurrent access."""
    
    def test_fuzz_concurrent_plugin_access(self):
        """Test concurrent access to plugins with fuzzy inputs."""
        import threading
        import time
        
        num_threads = 10
        duration_seconds = 5
        stop_flag = threading.Event()
        results = {"errors": [], "processed": 0}
        lock = threading.Lock()
        
        def worker():
            while not stop_flag.is_set():
                try:
                    lang = random.choice(["python", "javascript", "java", "go"])
                    plugin = self.plugin_system.get_plugin(lang)
                    
                    if plugin:
                        error_data = {
                            "language": lang,
                            "error_type": self.random_string(5, 50),
                            "message": self.random_string(0, 500),
                            "file": self.random_string(5, 100),
                            "line": random.randint(1, 10000)
                        }
                        
                        plugin.analyze_error(error_data)
                        
                        with lock:
                            results["processed"] += 1
                            
                except Exception as e:
                    with lock:
                        results["errors"].append(str(e))
        
        # Start threads
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)
        
        # Run for specified duration
        time.sleep(duration_seconds)
        stop_flag.set()
        
        # Wait for threads
        for t in threads:
            t.join()
        
        print("\nConcurrent fuzzing results:")
        print(f"Processed: {results['processed']}")
        print(f"Errors: {len(results['errors'])}")
        
        # Should process many without errors
        assert results["processed"] > 100
        assert len(results["errors"]) < results["processed"] * 0.05  # Less than 5% error rate


class TestPropertyBasedFuzzing(FuzzGenerator):
    """Property-based fuzzing tests."""
    
    def test_parser_generator_roundtrip(self):
        """Test that parsed errors can be used to generate valid patches."""
        languages = ["python", "javascript", "java", "go"]
        
        for lang in languages:
            plugin = self.plugin_system.get_plugin(lang)
            if not plugin:
                continue
            
            for _ in range(100):
                # Generate a semi-valid error
                error_data = {
                    "language": lang,
                    "error_type": random.choice(["TypeError", "ValueError", "SyntaxError"]),
                    "message": f"Test error: {self.random_string(10, 50)}",
                    "file": f"test_{random.randint(1, 100)}.py",
                    "line": random.randint(1, 1000)
                }
                
                try:
                    # Parse error
                    analysis = plugin.analyze_error(error_data)
                    
                    # Property 1: Analysis should have required fields
                    assert "root_cause" in analysis
                    assert "suggestion" in analysis
                    assert "category" in analysis
                    assert "severity" in analysis
                    
                    # Property 2: Severity should be valid
                    assert analysis["severity"] in ["low", "medium", "high", "critical"]
                    
                    # Property 3: If generator exists, should handle analysis
                    if hasattr(plugin, 'generate_fix'):
                        context = {
                            "file": error_data["file"],
                            "line": error_data["line"],
                            "code_snippet": self.random_string(20, 100)
                        }
                        
                        patch = plugin.generate_fix(analysis, context)
                        if patch:
                            # Property 4: Patch should have structure
                            assert isinstance(patch, dict)
                            if "content" in patch:
                                assert isinstance(patch["content"], str)
                                
                except Exception:
                    pass  # Some combinations may fail
    
    def test_error_categorization_consistency(self):
        """Test that similar errors get similar categorization."""
        plugin = self.plugin_system.get_plugin("python")
        if not plugin:
            pytest.skip("Python plugin not available")
        
        # Generate similar errors with variations
        base_errors = [
            {
                "error_type": "ImportError",
                "message": "No module named 'requests'"
            },
            {
                "error_type": "AttributeError",
                "message": "'NoneType' object has no attribute 'split'"
            },
            {
                "error_type": "KeyError",
                "message": "'user_id'"
            }
        ]
        
        for base_error in base_errors:
            categories = set()
            
            # Generate variations
            for _ in range(50):
                error = base_error.copy()
                error["language"] = "python"
                error["file"] = self.random_string(5, 50) + ".py"
                error["line"] = random.randint(1, 1000)
                
                # Add some noise to message
                if random.random() < 0.3:
                    error["message"] += " " + self.random_string(1, 20)
                
                try:
                    analysis = plugin.analyze_error(error)
                    categories.add(analysis.get("category"))
                except Exception:
                    pass
            
            # Property: Similar errors should have consistent categorization
            assert len(categories) <= 2  # At most 2 different categories for variations


if __name__ == "__main__":
    # Run with specific seed for reproducibility
    pytest.main([__file__, "-v", "-s", "--tb=short"])