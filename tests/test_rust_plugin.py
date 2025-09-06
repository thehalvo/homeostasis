"""
Tests for the Rust Language Plugin
"""

import sys
import unittest
from pathlib import Path

# Add the project root directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.analysis.language_adapters import RustErrorAdapter
from modules.analysis.plugins.rust_plugin import (
    RustErrorHandler,
    RustLanguagePlugin,
    RustPatchGenerator,
)


class TestRustPlugin(unittest.TestCase):
    """Test cases for the Rust language plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = RustLanguagePlugin()
        self.adapter = RustErrorAdapter()
        self.error_handler = RustErrorHandler()
        self.patch_generator = RustPatchGenerator()

    def test_plugin_initialization(self):
        """Test plugin initialization."""
        self.assertEqual(self.plugin.get_language_id(), "rust")
        self.assertEqual(self.plugin.get_language_name(), "Rust")
        self.assertEqual(self.plugin.get_language_version(), "1.0+")

        # Check supported frameworks
        frameworks = self.plugin.get_supported_frameworks()
        self.assertIn("actix", frameworks)
        self.assertIn("rocket", frameworks)
        self.assertIn("tokio", frameworks)
        self.assertIn("base", frameworks)

    def test_error_normalization(self):
        """Test conversion to standard format."""
        # Create a sample Rust error
        rust_error = {
            "error_type": "Panic",
            "message": "thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', src/main.rs:15:13",
            "backtrace": [
                "0: rust_panic",
                "1: core::panicking::panic_fmt",
                "   at /rustc/d5a82bbd26e1ad8b7401f6a718a9c57c96905483/library/core/src/panicking.rs:142:14",
                "2: core::option::expect_none_failed",
                "   at /rustc/d5a82bbd26e1ad8b7401f6a718a9c57c96905483/library/core/src/option.rs:1267:5",
                "3: core::option::Option<T>::unwrap",
                "   at /rustc/d5a82bbd26e1ad8b7401f6a718a9c57c96905483/library/core/src/option.rs:450:21",
                "4: example::main",
                "   at ./src/main.rs:15:13",
                "5: core::ops::function::FnOnce::call_once",
                "   at /rustc/d5a82bbd26e1ad8b7401f6a718a9c57c96905483/library/core/src/ops/function.rs:227:5",
            ],
            "rust_version": "1.60.0",
        }

        # Convert to standard format
        standard_error = self.plugin.normalize_error(rust_error)

        # Check conversion
        self.assertEqual(standard_error["language"], "rust")
        self.assertEqual(standard_error["error_type"], "Panic")
        self.assertEqual(
            standard_error["message"],
            "thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', src/main.rs:15:13",
        )
        self.assertIn("stack_trace", standard_error)
        self.assertEqual(standard_error["language_version"], "1.60.0")

    def test_error_denormalization(self):
        """Test conversion from standard format."""
        # Create a sample standard error
        standard_error = {
            "error_id": "12345",
            "language": "rust",
            "error_type": "Panic",
            "message": "thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', src/main.rs:15:13",
            "stack_trace": [
                {
                    "function": "unwrap",
                    "module": "core::option::Option<T>",
                    "file": "/rustc/d5a82bbd26e1ad8b7401f6a718a9c57c96905483/library/core/src/option.rs",
                    "line": 450,
                },
                {
                    "function": "main",
                    "module": "example",
                    "file": "./src/main.rs",
                    "line": 15,
                },
            ],
            "language_version": "1.60.0",
        }

        # Convert from standard format
        rust_error = self.plugin.denormalize_error(standard_error)

        # Check conversion
        self.assertEqual(rust_error["error_type"], "Panic")
        self.assertEqual(
            rust_error["message"],
            "thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', src/main.rs:15:13",
        )
        self.assertIn("backtrace", rust_error)
        self.assertEqual(rust_error["rust_version"], "1.60.0")

    def test_error_analysis_unwrap_none(self):
        """Test analysis of unwrap on None error."""
        # Create a sample Rust error
        rust_error = {
            "error_type": "Panic",
            "message": "thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', src/main.rs:15:13",
            "language": "rust",
            "backtrace": [
                "0: rust_panic",
                "1: core::panicking::panic_fmt",
                "   at /rustc/d5a82bbd26e1ad8b7401f6a718a9c57c96905483/library/core/src/panicking.rs:142:14",
                "2: core::option::expect_none_failed",
                "   at /rustc/d5a82bbd26e1ad8b7401f6a718a9c57c96905483/library/core/src/option.rs:1267:5",
                "3: core::option::Option<T>::unwrap",
                "   at /rustc/d5a82bbd26e1ad8b7401f6a718a9c57c96905483/library/core/src/option.rs:450:21",
                "4: example::main",
                "   at ./src/main.rs:15:13",
            ],
        }

        # Analyze the error
        analysis = self.plugin.analyze_error(rust_error)

        # Check analysis results
        self.assertEqual(analysis["root_cause"], "rust_unwrap_none")
        self.assertEqual(analysis["error_type"], "Panic")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["severity"], "high")
        self.assertTrue("suggestion" in analysis)

    def test_error_analysis_index_out_of_bounds(self):
        """Test analysis of index out of bounds error."""
        # Create a sample Rust error
        rust_error = {
            "error_type": "Panic",
            "message": "thread 'main' panicked at 'index out of bounds: the len is 3 but the index is 5', src/main.rs:12:17",
            "language": "rust",
            "backtrace": [
                "0: rust_panic",
                "1: core::panicking::panic_fmt",
                "   at /rustc/d5a82bbd26e1ad8b7401f6a718a9c57c96905483/library/core/src/panicking.rs:142:14",
                "2: core::panicking::panic_bounds_check",
                "   at /rustc/d5a82bbd26e1ad8b7401f6a718a9c57c96905483/library/core/src/panicking.rs:280:5",
                "3: example::main",
                "   at ./src/main.rs:12:17",
            ],
        }

        # Analyze the error
        analysis = self.plugin.analyze_error(rust_error)

        # Check analysis results
        self.assertEqual(analysis["root_cause"], "rust_index_out_of_bounds")
        self.assertEqual(analysis["error_type"], "Panic")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["severity"], "high")
        self.assertTrue("suggestion" in analysis)

    def test_fix_generation_unwrap_none(self):
        """Test fix generation for unwrap on None error."""
        # Create a sample analysis result
        analysis = {
            "error_data": {
                "error_type": "Panic",
                "message": "thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', src/main.rs:15:13",
                "stack_trace": [
                    {
                        "function": "unwrap",
                        "module": "core::option::Option<T>",
                        "file": "/rustc/d5a82bbd26e1ad8b7401f6a718a9c57c96905483/library/core/src/option.rs",
                        "line": 450,
                    },
                    {
                        "function": "main",
                        "module": "example",
                        "file": "./src/main.rs",
                        "line": 15,
                    },
                ],
            },
            "rule_id": "rust_unwrap_on_none",
            "error_type": "Panic",
            "root_cause": "rust_unwrap_none",
            "description": "Called unwrap() on a None value",
            "suggestion": "Use unwrap_or() or match expressions...",
            "confidence": "high",
            "severity": "high",
            "category": "runtime",
        }

        context = {"code_snippet": "let value = optional.unwrap();"}

        # Generate fix
        fix = self.plugin.generate_fix(analysis, context)

        # Check fix data
        self.assertEqual(fix["patch_id"], "rust_rust_unwrap_on_none")
        self.assertEqual(fix["language"], "rust")
        self.assertTrue("suggestion_code" in fix or "patch_code" in fix)
        self.assertEqual(fix["root_cause"], "rust_unwrap_none")

    def test_fix_generation_index_out_of_bounds(self):
        """Test fix generation for index out of bounds error."""
        # Create a sample analysis result
        analysis = {
            "error_data": {
                "error_type": "Panic",
                "message": "thread 'main' panicked at 'index out of bounds: the len is 3 but the index is 5', src/main.rs:12:17",
                "stack_trace": [
                    {
                        "function": "panic_bounds_check",
                        "module": "core::panicking",
                        "file": "/rustc/d5a82bbd26e1ad8b7401f6a718a9c57c96905483/library/core/src/panicking.rs",
                        "line": 280,
                    },
                    {
                        "function": "main",
                        "module": "example",
                        "file": "./src/main.rs",
                        "line": 12,
                    },
                ],
            },
            "rule_id": "rust_index_out_of_bounds",
            "error_type": "Panic",
            "root_cause": "rust_index_out_of_bounds",
            "description": "Attempted to access an index beyond the bounds of a collection",
            "suggestion": "Check bounds or use .get()",
            "confidence": "high",
            "severity": "high",
            "category": "runtime",
        }

        context = {"code_snippet": "let value = array[index];"}

        # Generate fix
        fix = self.plugin.generate_fix(analysis, context)

        # Check fix data
        self.assertEqual(fix["patch_id"], "rust_rust_index_out_of_bounds")
        self.assertEqual(fix["language"], "rust")
        self.assertTrue("suggestion_code" in fix or "patch_code" in fix)
        self.assertEqual(fix["root_cause"], "rust_index_out_of_bounds")

    def test_rule_loading(self):
        """Test loading of rule files."""
        # Get all rules
        rules = self.error_handler.rules

        # Ensure we have rules
        self.assertTrue(len(rules) > 10, "Should have loaded at least 10 rules")

        # Check for specific rule categories
        rule_ids = [rule.get("id") for rule in rules]
        self.assertIn("rust_unwrap_on_none", rule_ids)
        self.assertIn("rust_index_out_of_bounds", rule_ids)
        self.assertIn("rust_division_by_zero", rule_ids)

    def test_actix_error_analysis(self):
        """Test analysis of Actix Web framework error."""
        # Create a sample Actix error
        actix_error = {
            "error_type": "actix_web::Error",
            "message": "Failed to extract path parameters from request: Invalid parameter format",
            "language": "rust",
            "framework": "actix",
            "backtrace": [
                "0: actix_web::extract::path::PathExtractor::extract",
                "   at /usr/src/actix-web/src/extract/path.rs:109:18",
                "1: actix_web::handler::Factory::handle",
                "   at /usr/src/actix-web/src/handler.rs:273:39",
                "2: example::handlers::get_user",
                "   at ./src/handlers.rs:42:5",
            ],
        }

        # Analyze the error
        analysis = self.plugin.analyze_error(actix_error)

        # Check for framework-specific analysis
        self.assertEqual(analysis["category"], "framework")
        self.assertEqual(analysis["framework"], "actix")
        self.assertTrue("suggestion" in analysis)


if __name__ == "__main__":
    unittest.main()
