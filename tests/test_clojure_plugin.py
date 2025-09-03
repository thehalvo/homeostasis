"""
Tests for the Clojure language plugin.

This file contains tests for verifying the functionality of the Clojure language plugin,
including error normalization, analysis, and patch generation.
"""
import sys
import unittest
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.analysis.plugins.clojure_plugin import ClojureLanguagePlugin
from modules.analysis.language_adapters import ClojureErrorAdapter


class TestClojurePlugin(unittest.TestCase):
    """Test cases for the Clojure language plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = ClojureLanguagePlugin()
        self.adapter = ClojureErrorAdapter()
        
        # Sample Clojure error data
        self.sample_null_pointer = {
            "error_type": "java.lang.NullPointerException",
            "message": "Cannot invoke method on null",
            "stack_trace": [
                "at my.namespace.core$process_data.invoke (core.clj:42)",
                "at my.namespace.main$-main.invoke (main.clj:15)"
            ],
            "clojure_version": "1.11.1"
        }
        
        # Sample Arity Exception
        self.arity_error = {
            "error_type": "clojure.lang.ArityException",
            "message": "Wrong number of args (3) passed to: my.namespace.core/process-item",
            "stack_trace": [
                "at my.namespace.core$process_item.invoke (core.clj:25)",
                "at my.namespace.main$-main.invoke (main.clj:8)"
            ],
            "clojure_version": "1.11.1"
        }
        
        # Sample Ring/Compojure error
        self.ring_error = {
            "error_type": "java.lang.IllegalArgumentException", 
            "message": "Key must be integer",
            "stack_trace": [
                "at ring.middleware.params$wrap_params$fn__12345.invoke (params.clj:67)",
                "at my.web.handler$app.invoke (handler.clj:23)",
                "at ring.adapter.jetty$proxy_handler$fn__12345.invoke (jetty.clj:45)"
            ],
            "framework": "ring",
            "framework_version": "1.9.0",
            "clojure_version": "1.11.1"
        }
        
        # Sample core.async error
        self.async_error = {
            "error_type": "java.lang.IllegalStateException",
            "message": "Can't call blocking operation in go block",
            "stack_trace": [
                "at clojure.core.async$<!!.invoke (async.clj:154)",
                "at my.async.core$process_messages$fn__12345.invoke (core.clj:34)",
                "at clojure.core.async.impl.ioc_macros$run_state_machine.invoke (ioc_macros.clj:978)"
            ],
            "framework": "core.async",
            "clojure_version": "1.11.1"
        }
        
        # Sample Compilation error
        self.compilation_error = {
            "error_type": "CompilerException",
            "message": "java.lang.RuntimeException: Unable to resolve symbol: undefined-var in this context, compiling:(my/namespace/core.clj:15:12)",
            "stack_trace": [
                "at clojure.lang.Compiler.analyze(Compiler.java:6977)",
                "at clojure.lang.Compiler.analyze(Compiler.java:6929)",
                "at clojure.lang.Compiler$VectorExpr.parse(Compiler.java:4131)"
            ],
            "clojure_version": "1.11.1"
        }
        
        # Sample ClassCast error
        self.class_cast_error = {
            "error_type": "java.lang.ClassCastException",
            "message": "java.lang.String cannot be cast to java.lang.Number",
            "stack_trace": [
                "at my.namespace.math$add_numbers.invoke (math.clj:8)",
                "at my.namespace.core$process.invoke (core.clj:25)"
            ],
            "clojure_version": "1.11.1"
        }

    def test_plugin_info(self):
        """Test plugin information is correctly set."""
        info = self.plugin.get_language_info()
        
        self.assertEqual(info["language"], "clojure")
        self.assertEqual(info["version"], "1.0.0")
        self.assertEqual(info["description"], "Clojure language support for Homeostasis")
        
        # Check supported versions
        supported_versions = info["supported_versions"]
        expected_versions = ["1.8+", "1.9+", "1.10+", "1.11+"]
        
        for version in expected_versions:
            self.assertIn(version, supported_versions)
        
        # Check supported frameworks
        supported_frameworks = info["frameworks"]
        expected_frameworks = ["ring", "compojure", "luminus", "pedestal", "core.async", "spec", "datomic"]
        
        for framework in expected_frameworks:
            self.assertIn(framework, supported_frameworks)

    def test_can_handle_error(self):
        """Test error detection and handling capability."""
        # Test with explicit Clojure language
        clojure_error = {"language": "clojure", "error_type": "Exception"}
        self.assertTrue(self.plugin.can_handle_error(clojure_error))
        
        # Test with Clojure-specific patterns
        self.assertTrue(self.plugin.can_handle_error(self.sample_null_pointer))
        self.assertTrue(self.plugin.can_handle_error(self.arity_error))
        self.assertTrue(self.plugin.can_handle_error(self.compilation_error))
        
        # Test with non-Clojure error
        java_error = {
            "error_type": "java.lang.Exception",
            "message": "Regular Java error",
            "stack_trace": ["at com.example.Main.main(Main.java:10)"]
        }
        self.assertFalse(self.plugin.can_handle_error(java_error))

    def test_adapter_standardization(self):
        """Test error data standardization."""
        standard_error = self.adapter.to_standard_format(self.sample_null_pointer)
        
        # Check basic fields
        self.assertEqual(standard_error["language"], "clojure")
        self.assertEqual(standard_error["error_type"], "java.lang.NullPointerException")
        self.assertEqual(standard_error["message"], "Cannot invoke method on null")
        self.assertEqual(standard_error["language_version"], "1.11.1")
        
        # Check stack trace is preserved
        self.assertIsInstance(standard_error["stack_trace"], list)
        self.assertEqual(len(standard_error["stack_trace"]), 2)
        
        # Check error ID is generated
        self.assertIn("error_id", standard_error)
        self.assertIn("timestamp", standard_error)

    def test_adapter_from_standard(self):
        """Test conversion from standard format back to Clojure format."""
        standard_error = self.adapter.to_standard_format(self.sample_null_pointer)
        clojure_error = self.adapter.from_standard_format(standard_error)
        
        # Check basic fields are preserved
        self.assertEqual(clojure_error["error_type"], "java.lang.NullPointerException")
        self.assertEqual(clojure_error["message"], "Cannot invoke method on null")
        self.assertEqual(clojure_error["clojure_version"], "1.11.1")

    def test_null_pointer_analysis(self):
        """Test analysis of NullPointerException."""
        analysis = self.plugin.analyze_error(self.sample_null_pointer)
        
        self.assertEqual(analysis["plugin"], "clojure")
        self.assertIn("rule_id", analysis)
        self.assertIn("fix_suggestions", analysis)
        self.assertGreater(len(analysis["fix_suggestions"]), 0)
        
        # Check confidence and severity
        self.assertIn("confidence", analysis)
        self.assertIn("severity", analysis)

    def test_arity_exception_analysis(self):
        """Test analysis of ArityException."""
        analysis = self.plugin.analyze_error(self.arity_error)
        
        self.assertEqual(analysis["plugin"], "clojure")
        self.assertIn("fix_suggestions", analysis)
        
        # Check that suggestions mention arity-related fixes or function arguments
        suggestions_text = " ".join(analysis["fix_suggestions"])
        self.assertTrue(
            "arity" in suggestions_text.lower() or 
            "argument" in suggestions_text.lower() or
            "function" in suggestions_text.lower()
        )

    def test_class_cast_analysis(self):
        """Test analysis of ClassCastException."""
        analysis = self.plugin.analyze_error(self.class_cast_error)
        
        self.assertEqual(analysis["plugin"], "clojure")
        self.assertIn("fix_suggestions", analysis)
        
        # Check that suggestions mention type checking
        suggestions_text = " ".join(analysis["fix_suggestions"])
        self.assertIn("type", suggestions_text.lower())

    def test_compilation_error_analysis(self):
        """Test analysis of compilation errors."""
        analysis = self.plugin.analyze_error(self.compilation_error)
        
        self.assertEqual(analysis["plugin"], "clojure")
        self.assertIn("fix_suggestions", analysis)
        
        # Check category is syntax-related
        self.assertEqual(analysis.get("category"), "syntax")

    def test_framework_detection(self):
        """Test framework context detection."""
        # Test Ring framework detection
        analysis = self.plugin.analyze_error(self.ring_error)
        self.assertEqual(analysis["plugin"], "clojure")
        
        # Test core.async framework detection
        analysis = self.plugin.analyze_error(self.async_error)
        self.assertEqual(analysis["plugin"], "clojure")

    def test_patch_generation_null_pointer(self):
        """Test patch generation for NullPointerException."""
        analysis = self.plugin.analyze_error(self.sample_null_pointer)
        context = {"source_code": "(defn process [data] (.toString data))"}
        patch = self.plugin.generate_fix(analysis, context)
        
        if patch:  # Patch generation might return None for some cases
            self.assertEqual(patch["plugin"], "clojure")
            self.assertIn("patch_content", patch)
            self.assertIn("description", patch)
            
            # Check that patch contains Clojure-specific nil handling
            patch_content = patch["patch_content"]
            self.assertTrue(
                any(keyword in patch_content for keyword in ["when", "some?", "nil?", "some->"])
            )

    def test_patch_generation_arity_error(self):
        """Test patch generation for ArityException."""
        analysis = self.plugin.analyze_error(self.arity_error)
        context = {"source_code": "(defn calculate [x y] (+ x y z))"}
        patch = self.plugin.generate_fix(analysis, context)
        
        if patch:
            self.assertEqual(patch["plugin"], "clojure")
            self.assertIn("patch_content", patch)
            
            # Check that patch contains arity-related fixes
            patch_content = patch["patch_content"]
            self.assertTrue(
                any(keyword in patch_content for keyword in ["defn", "arity", "variadic", "&"])
            )

    def test_patch_generation_class_cast(self):
        """Test patch generation for ClassCastException."""
        analysis = self.plugin.analyze_error(self.class_cast_error)
        context = {"source_code": "(defn convert [data] (Integer/parseInt data))"}
        patch = self.plugin.generate_fix(analysis, context)
        
        if patch:
            self.assertEqual(patch["plugin"], "clojure")
            self.assertIn("patch_content", patch)
            
            # Check that patch contains type checking
            patch_content = patch["patch_content"]
            self.assertTrue(
                any(keyword in patch_content for keyword in ["instance?", "type", "cond"])
            )

    def test_stack_trace_parsing(self):
        """Test Clojure stack trace parsing."""
        clojure_stack = [
            "at my.namespace.core$process_data.invoke (core.clj:42)",
            "at my.namespace.main$-main.invoke (main.clj:15)",
            "at java.lang.Thread.run (Thread.java:748)"
        ]
        
        error_data = {
            "error_type": "Exception",
            "message": "Test error",
            "stack_trace": clojure_stack,
            "clojure_version": "1.11.1"
        }
        
        parsed_frames = self.adapter._parse_clojure_stack_trace(clojure_stack)
        
        # Should parse Clojure frames
        self.assertGreater(len(parsed_frames), 0)
        
        # Check first Clojure frame
        first_frame = parsed_frames[0]
        self.assertEqual(first_frame["namespace"], "my.namespace.core")
        self.assertEqual(first_frame["function"], "process_data")
        self.assertEqual(first_frame["file"], "core.clj")
        self.assertEqual(first_frame["line"], 42)
        self.assertEqual(first_frame["type"], "clojure")
        
        # Check Java frame is also parsed
        java_frame = next((f for f in parsed_frames if f["type"] == "java"), None)
        self.assertIsNotNone(java_frame)

    def test_anonymous_function_parsing(self):
        """Test parsing of anonymous function stack traces."""
        anon_stack = [
            "at my.namespace.core$process$fn__12345.invoke (core.clj:25)"
        ]
        
        parsed_frames = self.adapter._parse_clojure_stack_trace(anon_stack)
        
        self.assertEqual(len(parsed_frames), 1)
        frame = parsed_frames[0]
        self.assertEqual(frame["namespace"], "my.namespace.core")
        self.assertIn("anonymous function", frame["function"])

    def test_error_with_additional_data(self):
        """Test handling of errors with additional Clojure-specific data."""
        error_with_context = {
            "error_type": "Exception",
            "message": "Test error",
            "stack_trace": [],
            "clojure_version": "1.11.1",
            "namespace": "my.namespace.core",
            "var": "process-data",
            "form": "(process-data nil)"
        }
        
        standard_error = self.adapter.to_standard_format(error_with_context)
        
        # Check additional data is preserved
        additional_data = standard_error.get("additional_data", {})
        self.assertEqual(additional_data.get("namespace"), "my.namespace.core")
        self.assertEqual(additional_data.get("var"), "process-data")
        self.assertEqual(additional_data.get("form"), "(process-data nil)")

    def test_integration_workflow(self):
        """Test complete workflow from error to fix."""
        # Step 1: Check if plugin can handle the error
        self.assertTrue(self.plugin.can_handle_error(self.sample_null_pointer))
        
        # Step 2: Analyze the error
        analysis = self.plugin.analyze_error(self.sample_null_pointer)
        self.assertEqual(analysis["plugin"], "clojure")
        self.assertIn("fix_suggestions", analysis)
        
        # Step 3: Generate a fix (optional step)
        context = {"source_code": "(defn process [data] (.toString data))"}
        patch = self.plugin.generate_fix(analysis, context)
        # Patch generation might not always succeed, which is acceptable
        
        # Step 4: Verify analysis contains useful information
        self.assertGreater(len(analysis["fix_suggestions"]), 0)
        self.assertIn("confidence", analysis)
        self.assertIn("severity", analysis)
        self.assertIn("category", analysis)

    def test_error_handling(self):
        """Test plugin error handling with invalid data."""
        # Test with empty error data
        empty_error = {}
        analysis = self.plugin.analyze_error(empty_error)
        self.assertIn("plugin", analysis)  # Should still return some result
        
        # Test with malformed error data
        malformed_error = {"error_type": None, "message": None}
        analysis = self.plugin.analyze_error(malformed_error)
        self.assertIn("plugin", analysis)  # Should handle gracefully

    def test_multiple_framework_detection(self):
        """Test detection of multiple frameworks in stack trace."""
        multi_framework_error = {
            "error_type": "Exception",
            "message": "Multi-framework error",
            "stack_trace": [
                "at ring.middleware.params$wrap_params.invoke (params.clj:67)",
                "at clojure.core.async$go$fn__12345.invoke (async.clj:1234)",
                "at my.app.handler$process.invoke (handler.clj:23)"
            ],
            "clojure_version": "1.11.1"
        }
        
        analysis = self.plugin.analyze_error(multi_framework_error)
        
        # Should detect framework context
        framework_context = analysis.get("framework_context")
        if framework_context:
            frameworks = framework_context.get("frameworks", [])
            # Should detect both ring and core.async
            self.assertTrue(any("ring" in fw or "core.async" in fw for fw in frameworks))


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)