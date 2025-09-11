"""
Test cases for Ember.js framework plugin functionality.

This module tests the Ember plugin's ability to analyze Ember-specific errors,
generate appropriate fixes, and handle various Ember frameworks and patterns.
"""

import os
import sys
import unittest

from modules.analysis.plugins.ember_plugin import (
    EmberExceptionHandler,
    EmberLanguagePlugin,
    EmberPatchGenerator,
)

# Add the modules directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules"))


class TestEmberPlugin(unittest.TestCase):
    """Test cases for Ember.js language plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = EmberLanguagePlugin()
        self.exception_handler = EmberExceptionHandler()
        self.patch_generator = EmberPatchGenerator()

    def test_plugin_initialization(self):
        """Test that the Ember plugin initializes correctly."""
        self.assertEqual(self.plugin.get_language_id(), "ember")
        self.assertEqual(self.plugin.get_language_name(), "Ember.js")
        self.assertEqual(self.plugin.get_language_version(), "3.x/4.x")
        self.assertIn("ember", self.plugin.get_supported_frameworks())
        self.assertIn("ember-data", self.plugin.get_supported_frameworks())
        self.assertIn("ember-octane", self.plugin.get_supported_frameworks())

    def test_can_handle_ember_errors(self):
        """Test that the plugin can identify Ember errors."""
        # Ember-specific error message
        ember_error = {
            "error_type": "Error",
            "message": "Ember Component: component not found: my-button",
            "stack_trace": "at EmberComponent.render (app.js:10:5)",
        }
        self.assertTrue(self.plugin.can_handle(ember_error))

        # Ember Template error
        template_error = {
            "error_type": "Error",
            "message": "Unclosed element 'div' at line 5",
            "stack_trace": "at template.hbs:5:12",
        }
        self.assertTrue(self.plugin.can_handle(template_error))

        # Ember Data error
        data_error = {
            "error_type": "Error",
            "message": "No record was found at {id: 123}",
            "stack_trace": "at store.js:15:3",
        }
        self.assertTrue(self.plugin.can_handle(data_error))

        # Ember Router error
        router_error = {
            "error_type": "Error",
            "message": "No route matched 'user.profile'",
            "stack_trace": "at router.js:25:8",
        }
        self.assertTrue(self.plugin.can_handle(router_error))

        # Non-Ember error
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
            "framework": "ember",
        }
        self.assertTrue(self.plugin.can_handle(error_with_framework))

        # Ember file extension
        error_with_ember_file = {
            "error_type": "Error",
            "message": "Some error",
            "stack_trace": "at template.hbs:10:5",
        }
        self.assertTrue(self.plugin.can_handle(error_with_ember_file))

        # Dependencies
        error_with_dependencies = {
            "error_type": "Error",
            "message": "Some error",
            "context": {"dependencies": ["ember", "ember-data", "ember-cli"]},
        }
        self.assertTrue(self.plugin.can_handle(error_with_dependencies))


class TestEmberExceptionHandler(unittest.TestCase):
    """Test cases for Ember exception handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = EmberExceptionHandler()

    def test_analyze_template_error(self):
        """Test analysis of template errors."""
        # Syntax error
        syntax_error = {
            "error_type": "Error",
            "message": "Syntax error at line 5: Unclosed element 'div'",
            "stack_trace": "at template.hbs:5:12",
        }
        analysis = self.handler.analyze_template_error(syntax_error)

        self.assertEqual(analysis["category"], "ember")
        self.assertEqual(analysis["subcategory"], "templates")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["root_cause"], "ember_template_syntax_error")
        self.assertIn("syntax", analysis["suggested_fix"].lower())

        # Helper not found error
        helper_error = {
            "error_type": "Error",
            "message": "Helper 'format-date' not found",
            "stack_trace": "at template.hbs:8:15",
        }
        analysis = self.handler.analyze_template_error(helper_error)

        self.assertEqual(analysis["root_cause"], "ember_template_helper_not_found")
        self.assertIn("helper", analysis["suggested_fix"])

    def test_analyze_data_error(self):
        """Test analysis of Ember Data errors."""
        # Record not found error
        record_error = {
            "error_type": "Error",
            "message": "Record not found: {id: 123}",
            "stack_trace": "at store.js:15:8",
        }
        analysis = self.handler.analyze_data_error(record_error)

        self.assertEqual(analysis["category"], "ember")
        self.assertEqual(analysis["subcategory"], "data")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["root_cause"], "ember_data_record_not_found")
        self.assertIn("record", analysis["suggested_fix"].lower())

        # Relationship not loaded error
        relationship_error = {
            "error_type": "Error",
            "message": "Relationship 'comments' is not loaded",
            "stack_trace": "at model.js:20:5",
        }
        analysis = self.handler.analyze_data_error(relationship_error)

        self.assertEqual(analysis["root_cause"], "ember_data_relationship_not_loaded")
        self.assertIn("relationship", analysis["suggested_fix"].lower())

        # Store not injected
        store_error = {
            "error_type": "Error",
            "message": "Store is not injected",
            "stack_trace": "at component.js:25:12",
        }
        analysis = self.handler.analyze_data_error(store_error)

        self.assertEqual(analysis["root_cause"], "ember_data_store_not_injected")
        self.assertIn("store", analysis["suggested_fix"].lower())

    def test_analyze_router_error(self):
        """Test analysis of Router errors."""
        # Route not found error
        route_error = {
            "error_type": "Error",
            "message": "Route not found: 'user.profile'",
            "stack_trace": "at router.js:18:10",
        }
        analysis = self.handler.analyze_router_error(route_error)

        self.assertEqual(analysis["category"], "ember")
        self.assertEqual(analysis["subcategory"], "router")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(analysis["root_cause"], "ember_router_route_not_found")
        self.assertIn("route", analysis["suggested_fix"].lower())

        # Transition aborted error
        transition_error = {
            "error_type": "Error",
            "message": "Transition was aborted",
            "stack_trace": "at router.js:30:15",
        }
        analysis = self.handler.analyze_router_error(transition_error)

        self.assertEqual(analysis["root_cause"], "ember_router_transition_aborted")
        self.assertIn("transition", analysis["suggested_fix"].lower())

        # Dynamic segment error
        segment_error = {
            "error_type": "Error",
            "message": "Dynamic segment 'userId' not provided",
            "stack_trace": "at router.js:45:8",
        }
        analysis = self.handler.analyze_router_error(segment_error)

        self.assertEqual(analysis["root_cause"], "ember_router_dynamic_segment_error")
        self.assertIn("segment", analysis["suggested_fix"].lower())

    def test_analyze_octane_error(self):
        """Test analysis of Ember Octane errors."""
        # Tracked property error
        tracked_error = {
            "error_type": "Error",
            "message": "Property is not reactive - add @tracked",
            "stack_trace": "at component.js:20:5",
        }
        analysis = self.handler.analyze_octane_error(tracked_error)

        self.assertEqual(analysis["category"], "ember")
        self.assertEqual(analysis["subcategory"], "octane")
        self.assertEqual(analysis["confidence"], "high")
        self.assertEqual(
            analysis["root_cause"], "ember_octane_tracked_properties_error"
        )
        self.assertIn("@tracked", analysis["suggested_fix"])

        # Args access error
        args_error = {
            "error_type": "Error",
            "message": "Cannot set property of args - args is read only",
            "stack_trace": "at component.js:15:3",
        }
        analysis = self.handler.analyze_octane_error(args_error)

        self.assertEqual(analysis["root_cause"], "ember_octane_args_error")
        self.assertIn("this.args", analysis["suggested_fix"])

    def test_generic_analysis(self):
        """Test generic analysis for unmatched Ember errors."""
        # Unknown Ember error
        unknown_error = {
            "error_type": "Error",
            "message": "Some unknown Ember error",
            "stack_trace": "at component.js:20:5",
        }
        analysis = self.handler.analyze_exception(unknown_error)

        self.assertEqual(analysis["category"], "ember")
        self.assertEqual(analysis["subcategory"], "unknown")
        self.assertEqual(analysis["confidence"], "low")
        self.assertEqual(analysis["rule_id"], "ember_generic_handler")

        # Component error
        component_error = {
            "error_type": "Error",
            "message": "Component rendering error",
            "stack_trace": "at my-component.js:15:3",
        }
        analysis = self.handler.analyze_exception(component_error)

        self.assertEqual(analysis["subcategory"], "components")
        self.assertIn("component", analysis["suggested_fix"].lower())


class TestEmberPatchGenerator(unittest.TestCase):
    """Test cases for Ember patch generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = EmberPatchGenerator()

    def test_fix_template_syntax(self):
        """Test fix generation for template syntax errors."""
        error_data = {
            "error_type": "Error",
            "message": "Syntax error: Unclosed element 'div'",
        }
        analysis = {"root_cause": "ember_template_syntax_error"}
        source_code = "<div>\n  <p>Content</p>\n"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("curly braces", fix["fix_commands"][0])
        self.assertIn("closing tags", fix["fix_commands"][1])

    def test_fix_template_helper_not_found(self):
        """Test fix generation for template helper not found errors."""
        error_data = {
            "error_type": "Error",
            "message": "Helper 'format-date' not found",
        }
        analysis = {"root_cause": "ember_template_helper_not_found"}
        source_code = "{{format-date date}}"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("format-date", fix["description"])
        self.assertIn("helper", fix["fix_code"])

    def test_fix_data_record_not_found(self):
        """Test fix generation for Ember Data record not found errors."""
        error_data = {"error_type": "Error", "message": "Record not found: {id: 123}"}
        analysis = {"root_cause": "ember_data_record_not_found"}
        source_code = "this.store.findRecord('user', 123)"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("catch", fix["fix_code"])
        self.assertIn("not found", fix["description"])

    def test_fix_data_store_not_injected(self):
        """Test fix generation for store not injected errors."""
        error_data = {"error_type": "Error", "message": "Store is not injected"}
        analysis = {"root_cause": "ember_data_store_not_injected"}
        source_code = "this.store.findAll('user')"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("@service", fix["fix_code"])
        self.assertIn("store", fix["fix_code"])

    def test_fix_router_route_not_found(self):
        """Test fix generation for route not found errors."""
        error_data = {
            "error_type": "Error",
            "message": "Route not found: 'user.profile'",
        }
        analysis = {"root_cause": "ember_router_route_not_found"}
        source_code = "this.router.transitionTo('user.profile', 123)"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("'user.profile'", fix["description"])
        self.assertIn("router.js", fix["fix_code"])
        self.assertIn("this.route", fix["fix_code"])

    def test_fix_octane_tracked_properties(self):
        """Test fix generation for tracked properties errors."""
        error_data = {
            "error_type": "Error",
            "message": "Property is not reactive - add @tracked",
        }
        analysis = {"root_cause": "ember_octane_tracked_properties_error"}
        source_code = (
            "export default class MyComponent extends Component { count = 0; }"
        )

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("@tracked", fix["fix_code"])
        self.assertIn("glimmer/tracking", fix["fix_code"])

    def test_fix_octane_args_access(self):
        """Test fix generation for component args access errors."""
        error_data = {
            "error_type": "Error",
            "message": "Cannot set property of args - args is read only",
        }
        analysis = {"root_cause": "ember_octane_args_error"}
        source_code = "export default class MyComponent extends Component { setupArgs() { this.args.param = 'value'; } }"

        fix = self.generator.generate_patch(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        self.assertEqual(fix["type"], "suggestion")
        self.assertIn("this.args", fix["fix_code"])
        self.assertIn("read-only", fix["fix_commands"][1])


class TestEmberPluginIntegration(unittest.TestCase):
    """Test cases for Ember plugin integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = EmberLanguagePlugin()

    def test_analyze_error_template(self):
        """Test full error analysis for template errors."""
        error_data = {
            "error_type": "Error",
            "message": "Syntax error: Unclosed element 'div'",
            "stack_trace": "at template.hbs:5:12",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["plugin"], "ember")
        self.assertEqual(analysis["language"], "ember")
        self.assertEqual(analysis["category"], "ember")
        self.assertEqual(analysis["subcategory"], "templates")
        self.assertIn("plugin_version", analysis)

    def test_analyze_error_data(self):
        """Test full error analysis for Ember Data errors."""
        error_data = {
            "error_type": "Error",
            "message": "Record not found: {id: 123}",
            "stack_trace": "at store.js:20:5",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["plugin"], "ember")
        self.assertEqual(analysis["subcategory"], "data")

    def test_analyze_error_router(self):
        """Test full error analysis for Router errors."""
        error_data = {
            "error_type": "Error",
            "message": "Route not found: 'user.profile'",
            "stack_trace": "at router.js:15:8",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["plugin"], "ember")
        self.assertEqual(analysis["subcategory"], "router")

    def test_generate_fix_integration(self):
        """Test full fix generation integration."""
        error_data = {
            "error_type": "Error",
            "message": "Store is not injected",
            "stack_trace": "at component.js:10:5",
        }

        analysis = self.plugin.analyze_error(error_data)
        source_code = "export default class MyComponent extends Component { fetchData() { this.store.findAll('user'); } }"

        fix = self.plugin.generate_fix(error_data, analysis, source_code)

        self.assertIsNotNone(fix)
        if fix:  # Only test if fix was generated
            self.assertIn("type", fix)
            self.assertIn("description", fix)

    def test_get_language_info(self):
        """Test language information retrieval."""
        info = self.plugin.get_language_info()

        self.assertEqual(info["language"], "ember")
        self.assertIn("supported_extensions", info)
        self.assertIn("supported_frameworks", info)
        self.assertIn("features", info)
        self.assertIn("environments", info)

        # Check specific features
        features = info["features"]
        self.assertTrue(any("component" in feature.lower() for feature in features))
        self.assertTrue(any("data" in feature.lower() for feature in features))
        self.assertTrue(any("router" in feature.lower() for feature in features))
        self.assertTrue(any("octane" in feature.lower() for feature in features))


def run_ember_plugin_tests():
    """Run all Ember plugin tests."""
    test_classes = [
        TestEmberPlugin,
        TestEmberExceptionHandler,
        TestEmberPatchGenerator,
        TestEmberPluginIntegration,
    ]

    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_ember_plugin_tests()
    sys.exit(0 if success else 1)
