"""
Test cases for Svelte Framework Plugin

Tests comprehensive error detection and fixing capabilities for Svelte applications,
including component reactivity, SvelteKit routing, store management, and transitions.
"""

import sys
import unittest
from pathlib import Path

# Add the modules directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from analysis.plugins.svelte_plugin import (
    SvelteExceptionHandler,
    SvelteLanguagePlugin,
    SveltePatchGenerator,
)


class TestSvelteLanguagePlugin(unittest.TestCase):
    """Test cases for the main Svelte plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = SvelteLanguagePlugin()

    def test_plugin_initialization(self):
        """Test plugin initialization."""
        self.assertEqual(self.plugin.get_language_id(), "svelte")
        self.assertEqual(self.plugin.get_language_name(), "Svelte")
        self.assertEqual(self.plugin.get_language_version(), "3.x/4.x")
        self.assertIn("svelte", self.plugin.get_supported_frameworks())
        self.assertIn("sveltekit", self.plugin.get_supported_frameworks())

    def test_can_handle_svelte_errors(self):
        """Test detection of Svelte-specific errors."""
        # Test explicit framework detection
        svelte_error = {
            "framework": "svelte",
            "message": "Component error",
            "stack_trace": [],
        }
        self.assertTrue(self.plugin.can_handle(svelte_error))

        # Test SvelteKit detection
        sveltekit_error = {
            "message": "SvelteKit routing error",
            "stack_trace": "at +page.svelte:10",
        }
        self.assertTrue(self.plugin.can_handle(sveltekit_error))

        # Test reactive statement detection
        reactive_error = {
            "message": "reactive statement ran more than 10 times",
            "stack_trace": [],
        }
        self.assertTrue(self.plugin.can_handle(reactive_error))

        # Test store detection
        store_error = {
            "message": "writable is not defined",
            "stack_trace": "at stores.js:5",
        }
        self.assertTrue(self.plugin.can_handle(store_error))

        # Test .svelte file detection
        component_error = {
            "message": "Syntax error",
            "stack_trace": "at Component.svelte:15",
        }
        self.assertTrue(self.plugin.can_handle(component_error))

    def test_cannot_handle_non_svelte_errors(self):
        """Test rejection of non-Svelte errors."""
        non_svelte_error = {
            "message": "Generic JavaScript error",
            "stack_trace": "at main.js:10",
            "framework": "vanilla",
        }
        self.assertFalse(self.plugin.can_handle(non_svelte_error))

    def test_get_language_info(self):
        """Test language information retrieval."""
        info = self.plugin.get_language_info()
        self.assertEqual(info["language"], "svelte")
        self.assertIn(".svelte", info["supported_extensions"])
        self.assertIn("Svelte component reactivity error detection", info["features"])
        self.assertIn("browser", info["environments"])


class TestSvelteExceptionHandler(unittest.TestCase):
    """Test cases for Svelte exception analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = SvelteExceptionHandler()

    def test_analyze_reactivity_error(self):
        """Test reactivity error analysis."""
        error_data = {
            "error_type": "Error",
            "message": "reactive statement ran more than 10 times",
            "stack_trace": ["at Component.svelte:8"],
        }

        analysis = self.handler.analyze_reactivity_error(error_data)

        self.assertEqual(analysis["category"], "svelte")
        self.assertEqual(analysis["subcategory"], "reactivity")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["root_cause"], "svelte_reactive_infinite_loop")
        self.assertIn("circular dependencies", analysis["suggested_fix"])

    def test_analyze_reactive_undefined_variable(self):
        """Test reactive undefined variable error."""
        error_data = {
            "error_type": "ReferenceError",
            "message": "$: result = undefinedVariable * 2",
            "stack_trace": [],
        }

        analysis = self.handler.analyze_reactivity_error(error_data)

        self.assertEqual(analysis["subcategory"], "reactivity")
        self.assertEqual(analysis["root_cause"], "svelte_reactive_undefined_variable")
        self.assertIn("declare variables", analysis["suggested_fix"].lower())

    def test_analyze_store_error(self):
        """Test store error analysis."""
        error_data = {
            "error_type": "ReferenceError",
            "message": "writable is not defined",
            "stack_trace": ["at stores.js:5"],
        }

        analysis = self.handler.analyze_store_error(error_data)

        self.assertEqual(analysis["category"], "svelte")
        self.assertEqual(analysis["subcategory"], "stores")
        self.assertEqual(analysis["root_cause"], "svelte_store_not_imported")
        self.assertIn("import writable", analysis["suggested_fix"].lower())

    def test_analyze_store_subscription_leak(self):
        """Test store subscription leak detection."""
        error_data = {
            "error_type": "Warning",
            "message": "store subscription leak detected",
            "stack_trace": [],
        }

        analysis = self.handler.analyze_store_error(error_data)

        self.assertEqual(analysis["root_cause"], "svelte_store_subscription_leak")
        self.assertEqual(analysis["severity"], "warning")
        self.assertIn("unsubscribe", analysis["suggested_fix"].lower())

    def test_analyze_sveltekit_error(self):
        """Test SvelteKit error analysis."""
        error_data = {
            "error_type": "Error",
            "message": "load function must return an object",
            "stack_trace": ["+page.js:10"],
        }

        analysis = self.handler.analyze_sveltekit_error(error_data)

        self.assertEqual(analysis["category"], "svelte")
        self.assertEqual(analysis["subcategory"], "sveltekit")
        self.assertEqual(analysis["root_cause"], "sveltekit_load_return_type")
        self.assertIn("return an object", analysis["suggested_fix"])

    def test_analyze_sveltekit_goto_ssr_error(self):
        """Test SvelteKit goto SSR error."""
        error_data = {
            "error_type": "Error",
            "message": "cannot use goto during ssr",
            "stack_trace": [],
        }

        analysis = self.handler.analyze_sveltekit_error(error_data)

        self.assertEqual(analysis["root_cause"], "sveltekit_goto_ssr_error")
        self.assertIn("browser context", analysis["suggested_fix"])

    def test_analyze_hydration_mismatch(self):
        """Test hydration mismatch error."""
        error_data = {
            "error_type": "Error",
            "message": "hydration mismatch: server and client content differ",
            "stack_trace": [],
        }

        analysis = self.handler.analyze_sveltekit_error(error_data)

        self.assertEqual(analysis["root_cause"], "sveltekit_hydration_mismatch")
        self.assertIn("server and client", analysis["suggested_fix"])

    def test_generic_analysis(self):
        """Test generic error analysis fallback."""
        error_data = {
            "error_type": "Error",
            "message": "unknown svelte error",
            "stack_trace": [],
        }

        analysis = self.handler.analyze_exception(error_data)

        self.assertEqual(analysis["category"], "svelte")
        self.assertEqual(analysis["confidence"], "low")
        self.assertEqual(analysis["rule_id"], "svelte_generic_handler")


class TestSveltePatchGenerator(unittest.TestCase):
    """Test cases for Svelte patch generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = SveltePatchGenerator()

    def test_fix_reactive_infinite_loop(self):
        """Test fix for reactive infinite loop."""
        error_data = {"message": "reactive statement ran more than 10 times"}
        analysis = {"root_cause": "svelte_reactive_infinite_loop"}
        source_code = "$: result = input; $: input = result + 1;"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertEqual(patch["type"], "suggestion")
        self.assertIn("circular dependencies", patch["description"])
        self.assertIn("intermediate variables", patch["fix_commands"][1])

    def test_fix_reactive_undefined_variable(self):
        """Test fix for undefined variable in reactive statement."""
        error_data = {"message": "variable is not defined"}
        analysis = {"root_cause": "svelte_reactive_undefined_variable"}
        source_code = "$: result = undefinedVar * 2;"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertEqual(patch["type"], "suggestion")
        self.assertIn("declare variables", patch["description"].lower())
        self.assertIn("let count = 0;", patch["fix_code"])

    def test_fix_store_not_imported(self):
        """Test fix for missing store imports."""
        error_data = {"message": "writable is not defined"}
        analysis = {"root_cause": "svelte_store_not_imported"}
        source_code = "export const count = writable(0);"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertEqual(patch["type"], "line_addition")
        self.assertIn("import { writable }", patch["line_to_add"])
        self.assertEqual(patch["position"], "top")

    def test_fix_store_subscription_leak(self):
        """Test fix for store subscription leaks."""
        error_data = {"message": "store subscription leak"}
        analysis = {"root_cause": "svelte_store_subscription_leak"}
        source_code = "myStore.subscribe(value => { storeValue = value; });"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertEqual(patch["type"], "suggestion")
        self.assertIn("subscription", patch["description"].lower())
        self.assertIn("$: storeValue = $myStore;", patch["fix_code"])
        self.assertIn("onDestroy", patch["fix_code"])

    def test_fix_sveltekit_load_return_type(self):
        """Test fix for SvelteKit load function return type."""
        error_data = {"message": "load function must return an object"}
        analysis = {"root_cause": "sveltekit_load_return_type"}
        source_code = "export async function load() { return data; }"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertEqual(patch["type"], "suggestion")
        self.assertIn("return object", patch["description"])
        self.assertIn("props:", patch["fix_code"])
        self.assertIn("status:", patch["fix_code"])

    def test_fix_sveltekit_goto_ssr_error(self):
        """Test fix for SvelteKit goto SSR error."""
        error_data = {"message": "cannot use goto during ssr"}
        analysis = {"root_cause": "sveltekit_goto_ssr_error"}
        source_code = "goto('/target-page');"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertEqual(patch["type"], "suggestion")
        self.assertIn("browser context", patch["description"])
        self.assertIn("if (browser)", patch["fix_code"])
        self.assertIn("onMount", patch["fix_code"])

    def test_fix_sveltekit_hydration_mismatch(self):
        """Test fix for hydration mismatch."""
        error_data = {"message": "hydration mismatch"}
        analysis = {"root_cause": "sveltekit_hydration_mismatch"}
        source_code = "<div>{Math.random()}</div>"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertEqual(patch["type"], "suggestion")
        self.assertIn("server and client", patch["description"])
        self.assertIn("{#if browser}", patch["fix_code"])
        self.assertIn("mounted", patch["fix_code"])

    def test_fix_transition_error(self):
        """Test fix for transition errors."""
        error_data = {"message": "fade is not defined"}
        analysis = {"root_cause": "svelte_transition_error"}
        source_code = "<div transition:fade>Content</div>"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertEqual(patch["type"], "suggestion")
        self.assertIn("transition", patch["description"].lower())
        self.assertIn("svelte/transition", patch["fix_code"])
        self.assertIn("transition:fade", patch["fix_code"])

    def test_fix_binding_error(self):
        """Test fix for binding errors."""
        error_data = {"message": "cannot bind to value"}
        analysis = {"root_cause": "svelte_binding_error"}
        source_code = "<input bind:value={readonly} />"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertEqual(patch["type"], "suggestion")
        self.assertIn("binding", patch["description"].lower())
        self.assertIn("bind:value", patch["fix_code"])
        self.assertIn("let inputValue", patch["fix_code"])


class TestSvelteIntegration(unittest.TestCase):
    """Integration tests for the complete Svelte plugin workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = SvelteLanguagePlugin()

    def test_full_reactivity_error_workflow(self):
        """Test complete workflow for reactivity errors."""
        error_data = {
            "error_type": "Error",
            "message": "reactive statement ran more than 10 times",
            "stack_trace": ["at Component.svelte:8"],
            "framework": "svelte",
        }

        # Test error detection
        self.assertTrue(self.plugin.can_handle(error_data))

        # Test error analysis
        analysis = self.plugin.analyze_error(error_data)
        self.assertEqual(analysis["category"], "svelte")
        self.assertEqual(analysis["subcategory"], "reactivity")
        self.assertEqual(analysis["plugin"], "svelte")

        # Test fix generation
        source_code = "$: result = input; $: input = result + 1;"
        fix = self.plugin.generate_fix(error_data, analysis, source_code)
        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")

    def test_full_store_error_workflow(self):
        """Test complete workflow for store errors."""
        error_data = {
            "error_type": "ReferenceError",
            "message": "writable is not defined",
            "stack_trace": ["at stores.js:5"],
        }

        # Test error detection
        self.assertTrue(self.plugin.can_handle(error_data))

        # Test error analysis
        analysis = self.plugin.analyze_error(error_data)
        self.assertEqual(analysis["subcategory"], "stores")

        # Test fix generation
        source_code = "export const count = writable(0);"
        fix = self.plugin.generate_fix(error_data, analysis, source_code)
        self.assertIsNotNone(fix)
        self.assertIn("import", fix["line_to_add"])

    def test_full_sveltekit_error_workflow(self):
        """Test complete workflow for SvelteKit errors."""
        error_data = {
            "error_type": "Error",
            "message": "load function must return an object",
            "stack_trace": ["+page.js:10"],
        }

        # Test error detection
        self.assertTrue(self.plugin.can_handle(error_data))

        # Test error analysis
        analysis = self.plugin.analyze_error(error_data)
        self.assertEqual(analysis["subcategory"], "sveltekit")

        # Test fix generation
        source_code = "export async function load() { return data; }"
        fix = self.plugin.generate_fix(error_data, analysis, source_code)
        self.assertIsNotNone(fix)
        self.assertIn("props", fix["fix_code"])

    def test_error_analysis_with_missing_language(self):
        """Test error analysis when language is not set."""
        error_data = {
            "error_type": "Error",
            "message": "svelte component error",
            "stack_trace": [],
        }

        analysis = self.plugin.analyze_error(error_data)
        self.assertEqual(analysis["plugin"], "svelte")
        self.assertIn("language", analysis)

    def test_unsupported_error_handling(self):
        """Test handling of unsupported error types."""
        error_data = {
            "error_type": "UnknownError",
            "message": "completely unknown svelte error",
            "stack_trace": [],
        }

        analysis = self.plugin.analyze_error(error_data)
        self.assertEqual(analysis["category"], "svelte")
        self.assertEqual(analysis["confidence"], "low")


if __name__ == "__main__":
    # Create a test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestSvelteLanguagePlugin))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestSvelteExceptionHandler))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestSveltePatchGenerator))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestSvelteIntegration))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'='*50}")
    print("Svelte Plugin Test Summary")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    if result.wasSuccessful():
        print("\nAll tests passed! ✅")
    else:
        print("\nSome tests failed! ❌")
