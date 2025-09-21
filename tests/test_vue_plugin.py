"""
Test cases for Vue.js framework plugin functionality.

This module tests the Vue plugin's ability to analyze Vue-specific errors,
generate appropriate fixes, and handle various Vue frameworks and patterns.
"""

import os
import sys
import unittest

from modules.analysis.plugins.vue_plugin import (
    VueExceptionHandler,
    VueLanguagePlugin,
    VuePatchGenerator,
)

# Add the modules directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules"))


class TestVuePlugin(unittest.TestCase):
    """Test cases for Vue.js language plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = VueLanguagePlugin()
        self.exception_handler = VueExceptionHandler()
        self.patch_generator = VuePatchGenerator()

    def test_plugin_initialization(self):
        """Test that the Vue plugin initializes correctly."""
        self.assertEqual(self.plugin.get_language_id(), "vue")
        self.assertEqual(self.plugin.get_language_name(), "Vue.js")
        self.assertEqual(self.plugin.get_language_version(), "2.x/3.x")
        self.assertIn("vue", self.plugin.get_supported_frameworks())
        self.assertIn("nuxt", self.plugin.get_supported_frameworks())

    def test_can_handle_vue_errors(self):
        """Test that the plugin can identify Vue errors."""
        # Vue-specific error message
        vue_error = {
            "error_type": "Error",
            "message": "Vue warn: Failed to resolve component: MyButton",
            "stack_trace": "at VueComponent.render (app.vue:10:5)",
        }
        self.assertTrue(self.plugin.can_handle(vue_error))

        # Composition API error
        composition_error = {
            "error_type": "ReferenceError",
            "message": "ref is not defined",
            "stack_trace": "at setup (Component.vue:5:12)",
        }
        self.assertTrue(self.plugin.can_handle(composition_error))

        # Vuex error
        vuex_error = {
            "error_type": "Error",
            "message": "unknown mutation type: INCREMENT_COUNTER",
            "stack_trace": "at store.js:15:3",
        }
        self.assertTrue(self.plugin.can_handle(vuex_error))

        # Vue Router error
        router_error = {
            "error_type": "Error",
            "message": "No match found for location with path '/unknown'",
            "stack_trace": "at router.js:25:8",
        }
        self.assertTrue(self.plugin.can_handle(router_error))

        # Non-Vue error
        python_error = {
            "error_type": "NameError",
            "message": "name 'undefined_variable' is not defined",
            "stack_trace": "File test.py, line 10",
        }
        self.assertFalse(self.plugin.can_handle(python_error))

    def test_can_handle_framework_detection(self):
        """Test framework detection from error context."""
        # Explicit framework
        error_with_framework = {
            "error_type": "Error",
            "message": "Some error",
            "framework": "vue",
        }
        self.assertTrue(self.plugin.can_handle(error_with_framework))

        # Vue file extension
        error_with_vue_file = {
            "error_type": "Error",
            "message": "Some error",
            "stack_trace": "at Component.vue:10:5",
        }
        self.assertTrue(self.plugin.can_handle(error_with_vue_file))

        # Dependencies
        error_with_dependencies = {
            "error_type": "Error",
            "message": "Some error",
            "context": {"dependencies": ["vue", "vuex", "vue-router"]},
        }
        self.assertTrue(self.plugin.can_handle(error_with_dependencies))


class TestVueExceptionHandler(unittest.TestCase):
    """Test cases for Vue exception handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = VueExceptionHandler()

    def test_analyze_composition_api_error(self):
        """Test analysis of Composition API errors."""
        # Ref not defined error
        ref_error = {
            "error_type": "ReferenceError",
            "message": "ref is not defined",
            "stack_trace": "at setup (Component.vue:5:12)",
        }
        analysis = self.handler.analyze_composition_api_error(ref_error)

        self.assertEqual(analysis["category"], "vue")
        self.assertEqual(analysis["subcategory"], "composition")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["root_cause"], "vue_composition_ref_not_defined")
        self.assertIn("import", analysis["suggested_fix"].lower())
        self.assertIn("ref", analysis["suggested_fix"])

        # Computed not defined error
        computed_error = {
            "error_type": "ReferenceError",
            "message": "computed is not defined",
            "stack_trace": "at setup (Component.vue:8:15)",
        }
        analysis = self.handler.analyze_composition_api_error(computed_error)

        self.assertEqual(analysis["root_cause"], "vue_composition_computed_not_defined")
        self.assertIn("computed", analysis["suggested_fix"])

    def test_analyze_vuex_error(self):
        """Test analysis of Vuex errors."""
        # Store not defined error
        store_error = {
            "error_type": "ReferenceError",
            "message": "store is not defined",
            "stack_trace": "at Component.vue:15:8",
        }
        analysis = self.handler.analyze_vuex_error(store_error)

        self.assertEqual(analysis["category"], "vue")
        self.assertEqual(analysis["subcategory"], "vuex")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["root_cause"], "vuex_store_not_imported")
        self.assertIn("store", analysis["suggested_fix"].lower())

        # Mutation not found error
        mutation_error = {
            "error_type": "Error",
            "message": "unknown mutation type: INCREMENT_COUNTER",
            "stack_trace": "at store/index.js:20:5",
        }
        analysis = self.handler.analyze_vuex_error(mutation_error)

        self.assertEqual(analysis["root_cause"], "vuex_undefined_mutation")
        self.assertIn("mutation", analysis["suggested_fix"].lower())

        # Direct state mutation
        direct_mutation_error = {
            "error_type": "Error",
            "message": "Do not mutate vuex store state outside mutation handlers",
            "stack_trace": "at Component.vue:25:12",
        }
        analysis = self.handler.analyze_vuex_error(direct_mutation_error)

        self.assertEqual(analysis["root_cause"], "vuex_direct_state_mutation")
        self.assertIn("commit", analysis["suggested_fix"].lower())

    def test_analyze_router_error(self):
        """Test analysis of Vue Router errors."""
        # Router not defined error
        router_error = {
            "error_type": "ReferenceError",
            "message": "router is not defined",
            "stack_trace": "at Component.vue:18:10",
        }
        analysis = self.handler.analyze_router_error(router_error)

        self.assertEqual(analysis["category"], "vue")
        self.assertEqual(analysis["subcategory"], "router")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["root_cause"], "vue_router_not_imported")
        self.assertIn("router", analysis["suggested_fix"].lower())

        # Route not found error
        route_error = {
            "error_type": "Error",
            "message": "No match found for location with path '/unknown'",
            "stack_trace": "at router.js:30:15",
        }
        analysis = self.handler.analyze_router_error(route_error)

        self.assertEqual(analysis["root_cause"], "vue_router_route_not_found")
        self.assertIn("route", analysis["suggested_fix"].lower())

        # Navigation cancelled error
        navigation_error = {
            "error_type": "Error",
            "message": "Navigation cancelled from /home to /profile",
            "stack_trace": "at router.js:45:8",
        }
        analysis = self.handler.analyze_router_error(navigation_error)

        self.assertEqual(analysis["root_cause"], "vue_router_navigation_cancelled")
        self.assertIn("guard", analysis["suggested_fix"].lower())

    def test_generic_analysis(self):
        """Test generic analysis for unmatched Vue errors."""
        # Unknown Vue error
        unknown_error = {
            "error_type": "Error",
            "message": "Some unknown Vue error",
            "stack_trace": "at Component.vue:20:5",
        }
        analysis = self.handler.analyze_exception(unknown_error)

        self.assertEqual(analysis["category"], "vue")
        self.assertEqual(analysis["subcategory"], "unknown")
        self.assertEqual(analysis["confidence"], "low")
        self.assertEqual(analysis["rule_id"], "vue_generic_handler")

        # Component error
        component_error = {
            "error_type": "Error",
            "message": "Component rendering error",
            "stack_trace": "at MyComponent.vue:15:3",
        }
        analysis = self.handler.analyze_exception(component_error)

        self.assertEqual(analysis["subcategory"], "components")
        self.assertIn("component", analysis["suggested_fix"].lower())


class TestVuePatchGenerator(unittest.TestCase):
    """Test cases for Vue patch generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = VuePatchGenerator()

    def test_fix_ref_not_defined(self):
        """Test fix generation for ref not defined error."""
        error_data = {"error_type": "ReferenceError", "message": "ref is not defined"}
        analysis = {"root_cause": "vue_composition_ref_not_defined"}
        source_code = "export default { setup() { const count = ref(0) } }"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "line_addition")
        self.assertIn("import { ref }", fix["line_to_add"])
        self.assertIn("vue", fix["line_to_add"])

    def test_fix_computed_not_defined(self):
        """Test fix generation for computed not defined error."""
        error_data = {
            "error_type": "ReferenceError",
            "message": "computed is not defined",
        }
        analysis = {"root_cause": "vue_composition_computed_not_defined"}
        source_code = "export default { setup() { const doubled = computed(() => count.value * 2) } }"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "line_addition")
        self.assertIn("import { computed }", fix["line_to_add"])
        self.assertIn("vue", fix["line_to_add"])

    def test_fix_vuex_store_not_imported(self):
        """Test fix generation for Vuex store not imported error."""
        error_data = {"error_type": "ReferenceError", "message": "store is not defined"}
        analysis = {"root_cause": "vuex_store_not_imported"}
        source_code = "export default { mounted() { console.log(this.$store.state) } }"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("createStore", fix["fix_code"])
        self.assertIn("vuex", fix["fix_code"])

    def test_fix_vuex_undefined_mutation(self):
        """Test fix generation for undefined Vuex mutation."""
        error_data = {
            "error_type": "Error",
            "message": "unknown mutation type: 'INCREMENT_COUNTER'",
        }
        analysis = {"root_cause": "vuex_undefined_mutation"}
        source_code = "this.$store.commit('INCREMENT_COUNTER')"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("INCREMENT_COUNTER", fix["fix_code"])
        self.assertIn("mutations", fix["fix_code"])

    def test_fix_router_not_imported(self):
        """Test fix generation for Vue Router not imported error."""
        error_data = {
            "error_type": "ReferenceError",
            "message": "router is not defined",
        }
        analysis = {"root_cause": "vue_router_not_imported"}
        source_code = "export default { mounted() { this.$router.push('/home') } }"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("createRouter", fix["fix_code"])
        self.assertIn("vue-router", fix["fix_code"])

    def test_fix_setup_return_type(self):
        """Test fix generation for setup return type error."""
        error_data = {
            "error_type": "Error",
            "message": "setup() function must return an object",
        }
        analysis = {"root_cause": "vue_composition_setup_return_type"}
        source_code = "export default { setup() { const count = ref(0) } }"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("return", fix["fix_code"])
        self.assertIn("count", fix["fix_code"])


class TestVuePluginIntegration(unittest.TestCase):
    """Test cases for Vue plugin integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = VueLanguagePlugin()

    def test_analyze_error_composition_api(self):
        """Test full error analysis for Composition API errors."""
        error_data = {
            "error_type": "ReferenceError",
            "message": "ref is not defined",
            "stack_trace": "at setup (Component.vue:5:12)",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["plugin"], "vue")
        self.assertEqual(analysis["language"], "vue")
        self.assertEqual(analysis["category"], "vue")
        self.assertEqual(analysis["subcategory"], "composition")
        self.assertIn("plugin_version", analysis)

    def test_analyze_error_vuex(self):
        """Test full error analysis for Vuex errors."""
        error_data = {
            "error_type": "Error",
            "message": "unknown mutation type: INCREMENT",
            "stack_trace": "at store.js:20:5",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["plugin"], "vue")
        self.assertEqual(analysis["subcategory"], "vuex")

    def test_analyze_error_router(self):
        """Test full error analysis for Router errors."""
        error_data = {
            "error_type": "Error",
            "message": "route not found",
            "stack_trace": "at router.js:15:8",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["plugin"], "vue")
        self.assertEqual(analysis["subcategory"], "router")

    def test_generate_fix_integration(self):
        """Test full fix generation integration."""
        error_data = {"error_type": "ReferenceError", "message": "ref is not defined"}

        analysis = self.plugin.analyze_error(error_data)
        error_data["source_code"] = (
            "export default { setup() { const count = ref(0) } }"
        )

        fix = self.plugin.generate_fix(error_data, analysis)

        self.assertIsNotNone(fix)
        if fix:  # Only test if fix was generated
            self.assertIn("type", fix)
            self.assertIn("description", fix)

    def test_get_language_info(self):
        """Test language information retrieval."""
        info = self.plugin.get_language_info()

        self.assertEqual(info["language"], "vue")
        self.assertIn("supported_extensions", info)
        self.assertIn("supported_frameworks", info)
        self.assertIn("features", info)
        self.assertIn("environments", info)

        # Check specific features
        features = info["features"]
        self.assertTrue(any("Composition API" in feature for feature in features))
        self.assertTrue(any("Vuex" in feature for feature in features))
        self.assertTrue(any("Router" in feature for feature in features))


def run_vue_plugin_tests():
    """Run all Vue plugin tests."""
    test_classes = [
        TestVuePlugin,
        TestVueExceptionHandler,
        TestVuePatchGenerator,
        TestVuePluginIntegration,
    ]

    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_vue_plugin_tests()
    sys.exit(0 if success else 1)
