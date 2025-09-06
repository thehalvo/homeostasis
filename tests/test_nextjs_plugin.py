"""
Tests for the Next.js plugin functionality in Homeostasis.

This module contains tests for the Next.js framework plugin, including error detection,
analysis, and patch generation for common Next.js errors.
"""

import os
import sys
import unittest

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.analysis.plugins.nextjs_plugin import (
    NextjsExceptionHandler,
    NextjsLanguagePlugin,
    NextjsPatchGenerator,
)


class TestNextjsPlugin(unittest.TestCase):
    """Test suite for the Next.js plugin functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = NextjsLanguagePlugin()
        self.exception_handler = NextjsExceptionHandler()
        self.patch_generator = NextjsPatchGenerator()

        # Sample error data
        self.getserversideprops_error = {
            "error_type": "Error",
            "message": "getServerSideProps must return an object with props",
            "stack_trace": "at Object.getServerSideProps (/project/pages/posts/[id].js:15:10)",
            "file": "/project/pages/posts/[id].js",
            "line": 15,
            "framework": "next.js",
        }

        self.api_route_error = {
            "error_type": "Error",
            "message": "API resolved without sending a response",
            "stack_trace": "at handler (/project/pages/api/users.js:10:5)",
            "file": "/project/pages/api/users.js",
            "line": 10,
            "framework": "next.js",
        }

        self.app_router_error = {
            "error_type": "Error",
            "message": "Client component cannot import Server Component",
            "stack_trace": "at ClientComponent (/project/app/dashboard/ClientComponent.js:5:10)",
            "file": "/project/app/dashboard/ClientComponent.js",
            "line": 5,
            "framework": "next.js",
        }

    def test_can_handle(self):
        """Test if the plugin can handle Next.js errors."""
        # Test with explicit framework
        self.assertTrue(self.plugin.can_handle({"framework": "next.js"}))
        self.assertTrue(self.plugin.can_handle({"framework": "nextjs"}))

        # Test with Next.js specific messages
        self.assertTrue(
            self.plugin.can_handle(
                {
                    "message": "Error in getServerSideProps",
                    "stack_trace": "at /pages/index.js",
                }
            )
        )

        # Test with file path patterns
        self.assertTrue(
            self.plugin.can_handle(
                {"message": "Some error", "file": "/project/pages/api/users.js"}
            )
        )

        # Test with non-Next.js error
        self.assertFalse(
            self.plugin.can_handle(
                {
                    "message": "Generic JavaScript error",
                    "stack_trace": "at /src/utils.js",
                    "framework": "vanilla-js",
                }
            )
        )

    def test_analyze_data_fetching_error(self):
        """Test analysis of data fetching errors."""
        result = self.exception_handler.analyze_data_fetching_error(
            self.getserversideprops_error
        )

        self.assertEqual(result["category"], "nextjs")
        self.assertEqual(result["subcategory"], "data_fetching")
        self.assertIn("getServerSideProps", result["suggested_fix"])

    def test_analyze_api_route_error(self):
        """Test analysis of API route errors."""
        result = self.exception_handler.analyze_api_route_error(self.api_route_error)

        self.assertEqual(result["category"], "nextjs")
        self.assertEqual(result["subcategory"], "api_routes")
        self.assertIn("response", result["suggested_fix"])

    def test_analyze_app_router_error(self):
        """Test analysis of App Router errors."""
        result = self.exception_handler.analyze_app_router_error(self.app_router_error)

        self.assertEqual(result["category"], "nextjs")
        self.assertEqual(result["subcategory"], "app_dir")
        self.assertIn("Client", result["suggested_fix"])
        self.assertIn("Server", result["suggested_fix"])

    def test_analyze_error(self):
        """Test the main error analysis method."""
        # Data fetching error
        result = self.plugin.analyze_error(self.getserversideprops_error)
        self.assertEqual(result["plugin"], "nextjs")
        self.assertEqual(result["subcategory"], "data_fetching")

        # API route error
        result = self.plugin.analyze_error(self.api_route_error)
        self.assertEqual(result["plugin"], "nextjs")
        self.assertEqual(result["subcategory"], "api_routes")

        # App router error
        result = self.plugin.analyze_error(self.app_router_error)
        self.assertEqual(result["plugin"], "nextjs")
        self.assertEqual(result["subcategory"], "app_dir")

    def test_generate_fix(self):
        """Test fix generation for Next.js errors."""
        # Data fetching error
        analysis = self.plugin.analyze_error(self.getserversideprops_error)
        fix = self.plugin.generate_fix(
            self.getserversideprops_error,
            analysis,
            "export function getServerSideProps() { return data; }",
        )

        self.assertIsNotNone(fix)
        self.assertIn("props", str(fix))

        # API route error
        analysis = self.plugin.analyze_error(self.api_route_error)
        fix = self.plugin.generate_fix(
            self.api_route_error,
            analysis,
            "export default function handler(req, res) { const data = process(req.body); }",
        )

        self.assertIsNotNone(fix)
        self.assertIn("res.status", str(fix))

        # App router error
        analysis = self.plugin.analyze_error(self.app_router_error)
        fix = self.plugin.generate_fix(
            self.app_router_error,
            analysis,
            "'use client'\nimport ServerComponent from '../ServerComponent';",
        )

        self.assertIsNotNone(fix)
        self.assertIn("children", str(fix))

    def test_rule_loading(self):
        """Test rule loading functionality."""
        # Verify rules were loaded
        self.assertGreater(len(self.exception_handler.rules.get("common", [])), 0)
        self.assertGreater(
            len(self.exception_handler.rules.get("data_fetching", [])), 0
        )
        self.assertGreater(len(self.exception_handler.rules.get("api_routes", [])), 0)
        self.assertGreater(len(self.exception_handler.rules.get("app_router", [])), 0)

    def test_template_loading(self):
        """Test template loading functionality."""
        # Verify templates were loaded
        self.assertIn("getserversideprops_fix", self.patch_generator.templates)
        self.assertIn("api_route_fix", self.patch_generator.templates)
        self.assertIn("middleware_fix", self.patch_generator.templates)


if __name__ == "__main__":
    unittest.main()
