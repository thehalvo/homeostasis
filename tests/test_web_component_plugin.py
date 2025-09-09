"""
Tests for Web Component Plugin

This module contains tests for the Web Components plugin, which handles error detection and
healing for Custom Elements, Shadow DOM, HTML Templates, and framework interoperability.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

from modules.analysis.plugins.web_component_plugin import (
    WebComponentExceptionHandler, WebComponentLanguagePlugin,
    WebComponentPatchGenerator)

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestWebComponentExceptionHandler(unittest.TestCase):
    """Tests for the WebComponentExceptionHandler class."""

    def setUp(self):
        """Set up the test environment."""
        self.handler = WebComponentExceptionHandler()

    def test_lifecycle_error_detection(self):
        """Test detection of Custom Elements lifecycle errors."""
        # Test error about missing super() call in constructor
        error_data = {
            "error_type": "TypeError",
            "message": "Failed to construct 'MyElement': 1st argument is not an object, or super() not called",
            "stack_trace": [
                "TypeError: Failed to construct 'MyElement': 1st argument is not an object, or super() not called",
                "at new MyElement (http://example.com/components.js:23:5)",
            ],
        }

        analysis = self.handler.analyze_exception(error_data)

        self.assertEqual(analysis["category"], "lifecycle")
        self.assertEqual(analysis["root_cause"], "missing_super_call_in_constructor")
        self.assertEqual(analysis["severity"], "high")

    def test_shadow_dom_error_detection(self):
        """Test detection of Shadow DOM encapsulation errors."""
        # Test error about closed shadow root access
        error_data = {
            "error_type": "TypeError",
            "message": "Cannot read properties of null (reading 'querySelector')",
            "stack_trace": [
                "TypeError: Cannot read properties of null (reading 'querySelector')",
                "at MyElement._updateContent (http://example.com/components.js:45:28)",
                "at MyElement.connectedCallback (http://example.com/components.js:32:12)",
            ],
        }

        analysis = self.handler.analyze_exception(error_data)

        self.assertEqual(analysis["category"], "shadow_dom")
        self.assertEqual(analysis["root_cause"], "closed_shadow_root_access_attempt")

    def test_template_error_detection(self):
        """Test detection of HTML Template errors."""
        # Simulate an error with template content not being cloned
        error_data = {
            "error_type": "Error",
            "message": "Template content never cloned properly",
            "stack_trace": [
                "Error: Template content never cloned properly",
                "at MyElement._initTemplate (http://example.com/components.js:55:15)",
            ],
            "tags": ["webcomponents", "templates"],
        }

        analysis = self.handler.analyze_exception(error_data)

        self.assertEqual(analysis["category"], "templates")
        self.assertIn("template", analysis["tags"])

    def test_interoperability_error_detection(self):
        """Test detection of framework interoperability errors."""
        # Test error about React event binding
        error_data = {
            "error_type": "Error",
            "message": "React event handler not firing on custom element",
            "stack_trace": [
                "Error: React event handler not firing on custom element",
                "at ReactComponent (http://example.com/app.js:78:22)",
            ],
        }

        analysis = self.handler.analyze_exception(error_data)

        self.assertEqual(analysis["category"], "interop")
        self.assertIn("react", analysis["tags"])


class TestWebComponentPatchGenerator(unittest.TestCase):
    """Tests for the WebComponentPatchGenerator class."""

    def setUp(self):
        """Set up the test environment."""
        self.patch_generator = WebComponentPatchGenerator()

        # Mock the templates
        self.patch_generator.templates = {
            "constructor_super": "class {{className}} extends HTMLElement {\n  constructor() {\n    super();\n    // Initialize\n  }\n}",
            "shadow_dom_access": "class {{className}} extends HTMLElement {\n  constructor() {\n    super();\n    this.attachShadow({mode: 'open'});\n  }\n}",
            "template_optimization": "class {{className}} extends HTMLElement {\n  constructor() {\n    super();\n    if (!{{className}}.template) {\n      {{className}}.template = document.createElement('template');\n    }\n  }\n}",
        }

    def test_lifecycle_error_patch(self):
        """Test generating patches for lifecycle errors."""
        error_data = {
            "error_type": "TypeError",
            "message": "Failed to construct 'MyElement': 1st argument is not an object, or super() not called",
            "stack_trace": [
                "TypeError: Failed to construct 'MyElement': 1st argument is not an object, or super() not called",
                "at new MyElement (http://example.com/components.js:23:5)",
            ],
        }

        analysis = {
            "category": "lifecycle",
            "root_cause": "missing_super_call_in_constructor",
            "suggested_fix": "Ensure the constructor calls super() before any other statements",
        }

        source_code = "class MyElement extends HTMLElement {\n  constructor() {\n    this.foo = 'bar';\n  }\n}"

        lifecycle_patch = self.patch_generator.generate_patch(
            error_data, analysis, source_code
        )

        self.assertIsNotNone(lifecycle_patch)
        self.assertIn("suggestion", lifecycle_patch)

    def test_shadow_dom_error_patch(self):
        """Test generating patches for Shadow DOM errors."""
        error_data = {
            "error_type": "TypeError",
            "message": "Cannot read properties of null (reading 'querySelector')",
            "stack_trace": [
                "TypeError: Cannot read properties of null (reading 'querySelector')",
                "at MyElement._updateContent (http://example.com/components.js:45:28)",
                "at MyElement.connectedCallback (http://example.com/components.js:32:12)",
            ],
        }

        analysis = {
            "category": "shadow_dom",
            "root_cause": "closed_shadow_root_access_attempt",
            "suggested_fix": "Use 'open' mode for shadow root or store references to shadow DOM elements",
        }

        source_code = "class MyElement extends HTMLElement {\n  constructor() {\n    super();\n    this.attachShadow({mode: 'closed'});\n  }\n}"

        shadow_patch = self.patch_generator.generate_patch(
            error_data, analysis, source_code
        )

        self.assertIsNotNone(shadow_patch)
        self.assertIn("description", shadow_patch)

    def test_template_based_patch(self):
        """Test template-based patch generation."""
        error_data = {
            "error_type": "Error",
            "message": "Template content never cloned properly",
            "stack_trace": [
                "Error: Template content never cloned properly",
                "at MyElement._initTemplate (http://example.com/components.js:55:15)",
            ],
        }

        analysis = {
            "category": "templates",
            "root_cause": "template_content_not_cloned",
            "suggested_fix": "Use document.importNode(template.content, true) to clone template content",
        }

        source_code = "class MyElement extends HTMLElement {\n  constructor() {\n    super();\n    const template = document.querySelector('#my-template');\n    this.appendChild(template);\n  }\n}"

        template_patch = self.patch_generator._template_based_patch(
            error_data, analysis, source_code
        )

        self.assertIsNotNone(template_patch)


class TestWebComponentLanguagePlugin(unittest.TestCase):
    """Tests for the WebComponentLanguagePlugin class."""

    def setUp(self):
        """Set up the test environment."""
        self.plugin = WebComponentLanguagePlugin()

        # Mock the exception handler and patch generator
        self.plugin.exception_handler = MagicMock()
        self.plugin.patch_generator = MagicMock()

    def test_can_handle(self):
        """Test can_handle method."""
        # Should handle web component errors
        wc_error = {
            "error_type": "TypeError",
            "message": "Failed to execute 'define' on 'CustomElementRegistry': the name 'my-element' has already been used with this registry",
            "stack_trace": "at CustomElementRegistry.define ()",
        }
        self.assertTrue(self.plugin.can_handle(wc_error))

        # Should handle when language is explicitly set
        explicit_error = {
            "language": "webcomponents",
            "error_type": "Error",
            "message": "Generic error",
        }
        self.assertTrue(self.plugin.can_handle(explicit_error))

        # Should not handle non-web component errors
        non_wc_error = {
            "error_type": "SyntaxError",
            "message": "Unexpected token",
            "stack_trace": "at eval ()",
        }
        self.assertFalse(self.plugin.can_handle(non_wc_error))

    def test_analyze_error(self):
        """Test analyze_error method."""
        # Set up mock return value
        self.plugin.exception_handler.analyze_exception.return_value = {
            "category": "lifecycle",
            "root_cause": "missing_super_call_in_constructor",
            "confidence": "high",
            "suggested_fix": "Ensure the constructor calls super() before any other statements",
        }

        error_data = {
            "error_type": "TypeError",
            "message": "Failed to construct 'MyElement': 1st argument is not an object, or super() not called",
            "stack_trace": [
                "TypeError: Failed to construct 'MyElement': 1st argument is not an object, or super() not called",
                "at new MyElement (http://example.com/components.js:23:5)",
            ],
        }

        analysis = self.plugin.analyze_error(error_data)

        # Verify the plugin adds its metadata
        self.assertEqual(analysis["plugin"], "webcomponents")
        self.assertEqual(analysis["language"], "webcomponents")
        self.assertEqual(analysis["plugin_version"], "1.0.0")

        # Verify the handler was called
        self.plugin.exception_handler.analyze_exception.assert_called_once()

    def test_generate_fix(self):
        """Test generate_fix method."""
        # Set up mock return value
        self.plugin.patch_generator.generate_patch.return_value = {
            "type": "suggestion",
            "description": "Ensure the constructor calls super() before any other statements",
        }

        analysis = {
            "category": "lifecycle",
            "root_cause": "missing_super_call_in_constructor",
        }

        context = {
            "error_data": {
                "error_type": "TypeError",
                "message": "Failed to construct 'MyElement'",
            },
            "source_code": "class MyElement extends HTMLElement { constructor() { this.foo = 'bar'; } }",
        }

        fix = self.plugin.generate_fix(analysis, context)

        self.assertEqual(fix["type"], "suggestion")
        self.plugin.patch_generator.generate_patch.assert_called_once()

    def test_is_web_component_error(self):
        """Test _is_web_component_error method."""
        # Test with Custom Elements error
        ce_error = {
            "message": "Failed to execute 'define' on 'CustomElementRegistry'",
            "stack_trace": "",
        }
        self.assertTrue(self.plugin._is_web_component_error(ce_error))

        # Test with Shadow DOM error
        shadow_error = {
            "message": "Cannot read property 'querySelector' of null",
            "stack_trace": "this.shadowRoot.querySelector",
        }
        self.assertTrue(self.plugin._is_web_component_error(shadow_error))

        # Test with lifecycle method error
        lifecycle_error = {"message": "Error in connectedCallback", "stack_trace": ""}
        self.assertTrue(self.plugin._is_web_component_error(lifecycle_error))

        # Test with framework
        framework_error = {"message": "Error rendering component", "framework": "lit"}
        self.assertTrue(self.plugin._is_web_component_error(framework_error))


if __name__ == "__main__":
    unittest.main()
