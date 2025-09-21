"""
Test cases for cross-platform framework plugins (React Native, Flutter, Xamarin, Unity, Capacitor/Cordova)

This module contains comprehensive tests for all cross-platform mobile development framework plugins.
"""

import sys
import unittest
from pathlib import Path

from modules.analysis.plugins.capacitor_cordova_plugin import (
    CapacitorCordovaLanguagePlugin,
)
from modules.analysis.plugins.flutter_plugin import FlutterLanguagePlugin
from modules.analysis.plugins.react_native_plugin import ReactNativeLanguagePlugin
from modules.analysis.plugins.unity_plugin import UnityLanguagePlugin
from modules.analysis.plugins.xamarin_plugin import XamarinLanguagePlugin

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestReactNativePlugin(unittest.TestCase):
    """Test cases for React Native plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = ReactNativeLanguagePlugin()

    def test_plugin_initialization(self):
        """Test that React Native plugin initializes correctly."""
        self.assertEqual(self.plugin.get_language_id(), "react_native")
        self.assertEqual(self.plugin.get_language_name(), "React Native")
        self.assertIn("react-native", self.plugin.get_supported_frameworks())

    def test_can_handle_react_native_errors(self):
        """Test React Native error detection."""
        # Explicit React Native error
        rn_error = {
            "framework": "react-native",
            "message": "Metro bundler failed to start",
            "error_type": "Error",
        }
        self.assertTrue(self.plugin.can_handle(rn_error))

        # React Native specific patterns
        metro_error = {
            "message": "Unable to resolve module react-native",
            "stack_trace": "Metro bundler error",
        }
        self.assertTrue(self.plugin.can_handle(metro_error))

        # Native module error
        native_error = {
            "message": "Native module RCTCameraManager cannot be null",
            "runtime": "react-native",
        }
        self.assertTrue(self.plugin.can_handle(native_error))

        # Should not handle non-React Native errors
        regular_js_error = {
            "message": "Cannot read property of undefined",
            "framework": "vanilla-js",
        }
        self.assertFalse(self.plugin.can_handle(regular_js_error))

    def test_analyze_native_module_error(self):
        """Test analysis of React Native native module errors."""
        error_data = {
            "message": "Native module cannot be null",
            "error_type": "Error",
            "runtime": "react-native",
            "framework": "react-native",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["category"], "react_native")
        self.assertEqual(analysis["plugin"], "react_native")
        self.assertIn("native", analysis["subcategory"].lower())
        self.assertIn("link", analysis["suggested_fix"].lower())

    def test_analyze_metro_bundler_error(self):
        """Test analysis of Metro bundler errors."""
        error_data = {
            "message": "Unable to resolve module 'react-native-vector-icons'",
            "error_type": "Error",
            "runtime": "react-native",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["category"], "react_native")
        self.assertIn("bundler", analysis["subcategory"].lower())
        self.assertIn("metro", analysis["suggested_fix"].lower())

    def test_generate_fix_native_module(self):
        """Test fix generation for native module errors."""
        error_data = {
            "message": "Native module RCTCamera cannot be null",
            "error_type": "Error",
        }

        analysis = {
            "root_cause": "react_native_native_module_missing",
            "category": "react_native",
        }

        fix = self.plugin.generate_fix(analysis, {"source_code": ""})

        self.assertIsNotNone(fix)
        self.assertIn("link", str(fix).lower())
        self.assertIn("install", str(fix).lower())


class TestFlutterPlugin(unittest.TestCase):
    """Test cases for Flutter plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = FlutterLanguagePlugin()

    def test_plugin_initialization(self):
        """Test that Flutter plugin initializes correctly."""
        self.assertEqual(self.plugin.get_language_id(), "flutter")
        self.assertEqual(self.plugin.get_language_name(), "Flutter")
        self.assertIn("flutter", self.plugin.get_supported_frameworks())

    def test_can_handle_flutter_errors(self):
        """Test Flutter error detection."""
        # Explicit Flutter error
        flutter_error = {
            "framework": "flutter",
            "message": "RenderFlex overflowed by 123 pixels",
            "language": "dart",
        }
        self.assertTrue(self.plugin.can_handle(flutter_error))

        # Dart language error
        dart_error = {
            "message": "Null check operator used on a null value",
            "stack_trace": "main.dart:42",
        }
        self.assertTrue(self.plugin.can_handle(dart_error))

        # Widget error
        widget_error = {"message": "Widget build context error", "runtime": "flutter"}
        self.assertTrue(self.plugin.can_handle(widget_error))

        # Should not handle non-Flutter errors
        regular_error = {
            "message": "Regular JavaScript error",
            "language": "javascript",
        }
        self.assertFalse(self.plugin.can_handle(regular_error))

    def test_analyze_widget_overflow_error(self):
        """Test analysis of Flutter widget overflow errors."""
        error_data = {
            "message": "RenderFlex overflowed by 123 pixels on the right",
            "error_type": "FlutterError",
            "language": "dart",
            "framework": "flutter",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["category"], "flutter")
        self.assertEqual(analysis["plugin"], "flutter")
        self.assertIn("layout", analysis["subcategory"].lower())
        self.assertIn("expanded", analysis["suggested_fix"].lower())

    def test_analyze_null_safety_error(self):
        """Test analysis of Dart null safety errors."""
        error_data = {
            "message": "Null check operator used on a null value",
            "error_type": "_TypeError",
            "language": "dart",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["category"], "dart")
        self.assertIn("null", analysis["subcategory"].lower())
        self.assertIn("null", analysis["suggested_fix"].lower())

    def test_generate_fix_widget_overflow(self):
        """Test fix generation for widget overflow errors."""
        error_data = {"message": "RenderFlex overflowed", "error_type": "FlutterError"}

        analysis = {"root_cause": "flutter_widget_overflow", "category": "flutter"}

        fix = self.plugin.generate_fix(analysis, {"source_code": ""})

        self.assertIsNotNone(fix)
        self.assertIn("expanded", str(fix).lower())


class TestXamarinPlugin(unittest.TestCase):
    """Test cases for Xamarin plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = XamarinLanguagePlugin()

    def test_plugin_initialization(self):
        """Test that Xamarin plugin initializes correctly."""
        self.assertEqual(self.plugin.get_language_id(), "xamarin")
        self.assertEqual(self.plugin.get_language_name(), "Xamarin")
        self.assertIn("xamarin", self.plugin.get_supported_frameworks())

    def test_can_handle_xamarin_errors(self):
        """Test Xamarin error detection."""
        # Explicit Xamarin error
        xamarin_error = {
            "framework": "xamarin",
            "message": "DependencyService could not resolve interface",
            "runtime": "xamarin",
        }
        self.assertTrue(self.plugin.can_handle(xamarin_error))

        # Xamarin.Forms error
        forms_error = {
            "message": "Binding error in Xamarin.Forms",
            "stack_trace": "Xamarin.Forms.dll",
        }
        self.assertTrue(self.plugin.can_handle(forms_error))

        # Should not handle non-Xamarin errors
        regular_csharp_error = {
            "message": "NullReferenceException",
            "language": "csharp",
            "framework": "aspnet",
        }
        self.assertFalse(self.plugin.can_handle(regular_csharp_error))

    def test_analyze_dependency_service_error(self):
        """Test analysis of DependencyService errors."""
        error_data = {
            "message": "DependencyService could not resolve IMyService",
            "error_type": "InvalidOperationException",
            "framework": "xamarin",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["category"], "xamarin")
        self.assertEqual(analysis["plugin"], "xamarin")
        self.assertIn("dependency", analysis["subcategory"].lower())
        self.assertIn("register", analysis["suggested_fix"].lower())

    def test_analyze_forms_binding_error(self):
        """Test analysis of Xamarin.Forms binding errors."""
        error_data = {
            "message": "Binding path error in XAML",
            "error_type": "BindingException",
            "framework": "xamarin.forms",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["category"], "xamarin")
        self.assertIn("binding", analysis["subcategory"].lower())
        self.assertIn("bindingcontext", analysis["suggested_fix"].lower())

    def test_generate_fix_dependency_service(self):
        """Test fix generation for DependencyService errors."""
        error_data = {
            "message": "DependencyService could not resolve interface",
            "error_type": "InvalidOperationException",
        }

        analysis = {
            "root_cause": "xamarin_dependency_service_missing",
            "category": "xamarin",
        }

        fix = self.plugin.generate_fix(analysis, {"source_code": ""})

        self.assertIsNotNone(fix)
        self.assertIn("dependency", str(fix).lower())
        self.assertIn("register", str(fix).lower())


class TestUnityPlugin(unittest.TestCase):
    """Test cases for Unity plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = UnityLanguagePlugin()

    def test_plugin_initialization(self):
        """Test that Unity plugin initializes correctly."""
        self.assertEqual(self.plugin.get_language_id(), "unity")
        self.assertEqual(self.plugin.get_language_name(), "Unity")
        self.assertIn("unity", self.plugin.get_supported_frameworks())

    def test_can_handle_unity_errors(self):
        """Test Unity error detection."""
        # Explicit Unity error
        unity_error = {
            "framework": "unity",
            "message": "NullReferenceException in GameObject",
            "runtime": "unity",
        }
        self.assertTrue(self.plugin.can_handle(unity_error))

        # UnityEngine error
        unityengine_error = {
            "message": "UnityEngine.GameObject.GetComponent failed",
            "stack_trace": "UnityEngine.dll",
        }
        self.assertTrue(self.plugin.can_handle(unityengine_error))

        # Unity-specific patterns
        unity_pattern_error = {
            "message": "Coroutine couldn't be started",
            "stack_trace": "MonoBehaviour.cs:123",
        }
        self.assertTrue(self.plugin.can_handle(unity_pattern_error))

        # Should not handle non-Unity errors
        regular_csharp_error = {
            "message": "Regular C# exception",
            "language": "csharp",
            "framework": "webapi",
        }
        self.assertFalse(self.plugin.can_handle(regular_csharp_error))

    def test_analyze_null_reference_error(self):
        """Test analysis of Unity null reference errors."""
        error_data = {
            "message": "NullReferenceException: Object reference not set to an instance of an object",
            "error_type": "NullReferenceException",
            "runtime": "unity",
            "stack_trace": ["PlayerController.cs:45"],
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["category"], "unity")
        self.assertEqual(analysis["plugin"], "unity")
        self.assertIn("null", analysis["subcategory"].lower())
        self.assertIn("gameobject", analysis["suggested_fix"].lower())

    def test_analyze_missing_component_error(self):
        """Test analysis of Unity missing component errors."""
        error_data = {
            "message": "MissingComponentException: GetComponent returned null",
            "error_type": "MissingComponentException",
            "runtime": "unity",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["category"], "unity")
        self.assertIn("script", analysis["subcategory"].lower())
        self.assertIn("component", analysis["suggested_fix"].lower())

    def test_analyze_mobile_build_error(self):
        """Test analysis of Unity mobile build errors."""
        error_data = {
            "message": "Build failed for Android platform",
            "error_type": "BuildFailedException",
            "context": {"platform": "android", "build_target": "android"},
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["category"], "unity")
        self.assertIn("build", analysis["subcategory"].lower())
        self.assertIn("android", analysis["suggested_fix"].lower())

    def test_generate_fix_null_reference(self):
        """Test fix generation for Unity null reference errors."""
        error_data = {
            "message": "NullReferenceException in GameObject access",
            "error_type": "NullReferenceException",
        }

        analysis = {"root_cause": "unity_null_reference_error", "category": "unity"}

        fix = self.plugin.generate_fix(analysis, {"source_code": ""})

        self.assertIsNotNone(fix)
        self.assertIn("null", str(fix).lower())
        self.assertIn("gameobject", str(fix).lower())


class TestCapacitorCordovaPlugin(unittest.TestCase):
    """Test cases for Capacitor/Cordova plugin."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin = CapacitorCordovaLanguagePlugin()

    def test_plugin_initialization(self):
        """Test that Capacitor/Cordova plugin initializes correctly."""
        self.assertEqual(self.plugin.get_language_id(), "capacitor_cordova")
        self.assertEqual(self.plugin.get_language_name(), "Capacitor/Cordova")
        self.assertIn("capacitor", self.plugin.get_supported_frameworks())
        self.assertIn("cordova", self.plugin.get_supported_frameworks())

    def test_can_handle_capacitor_errors(self):
        """Test Capacitor error detection."""
        # Explicit Capacitor error
        capacitor_error = {
            "framework": "capacitor",
            "message": "Plugin not found: Camera",
            "runtime": "capacitor",
        }
        self.assertTrue(self.plugin.can_handle(capacitor_error))

        # Capacitor plugin error
        plugin_error = {
            "message": "@capacitor/camera plugin not installed",
            "stack_trace": "Capacitor.js",
        }
        self.assertTrue(self.plugin.can_handle(plugin_error))

        # Should not handle non-hybrid errors
        regular_js_error = {
            "message": "Regular JavaScript error",
            "framework": "vanilla",
        }
        self.assertFalse(self.plugin.can_handle(regular_js_error))

    def test_can_handle_cordova_errors(self):
        """Test Cordova error detection."""
        # Explicit Cordova error
        cordova_error = {
            "framework": "cordova",
            "message": "Plugin failed to install",
            "runtime": "cordova",
        }
        self.assertTrue(self.plugin.can_handle(cordova_error))

        # Cordova deviceready error
        deviceready_error = {
            "message": "deviceready event not fired",
            "stack_trace": "cordova.js",
        }
        self.assertTrue(self.plugin.can_handle(deviceready_error))

    def test_analyze_plugin_error(self):
        """Test analysis of plugin errors."""
        error_data = {
            "message": "Plugin Camera not found",
            "error_type": "Error",
            "framework": "capacitor",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["category"], "capacitor_cordova")
        self.assertEqual(analysis["plugin"], "capacitor_cordova")
        self.assertIn("plugin", analysis["subcategory"].lower())
        self.assertIn("install", analysis["suggested_fix"].lower())

    def test_analyze_permission_error(self):
        """Test analysis of permission errors."""
        error_data = {
            "message": "Permission denied for camera access",
            "error_type": "PermissionError",
            "framework": "capacitor",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["category"], "capacitor_cordova")
        self.assertIn("permission", analysis["subcategory"].lower())
        self.assertIn("permission", analysis["suggested_fix"].lower())

    def test_analyze_csp_violation(self):
        """Test analysis of CSP violations."""
        error_data = {
            "message": "Content Security Policy violation: unsafe-inline",
            "error_type": "CSPViolation",
            "framework": "cordova",
        }

        analysis = self.plugin.analyze_error(error_data)

        self.assertEqual(analysis["category"], "capacitor_cordova")
        self.assertIn("security", analysis["subcategory"].lower())
        self.assertIn("csp", analysis["suggested_fix"].lower())

    def test_generate_fix_plugin_installation(self):
        """Test fix generation for plugin installation errors."""
        error_data = {"message": "Plugin Camera not found", "error_type": "Error"}

        analysis = {
            "root_cause": "capacitor_plugin_not_found",
            "category": "capacitor_cordova",
        }

        fix = self.plugin.generate_fix(analysis, {"source_code": ""})

        self.assertIsNotNone(fix)
        self.assertIn("install", str(fix).lower())
        self.assertIn("sync", str(fix).lower())


class TestCrossPlatformIntegration(unittest.TestCase):
    """Test cross-platform framework integration and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugins = [
            ReactNativeLanguagePlugin(),
            FlutterLanguagePlugin(),
            XamarinLanguagePlugin(),
            UnityLanguagePlugin(),
            CapacitorCordovaLanguagePlugin(),
        ]

    def test_all_plugins_have_unique_ids(self):
        """Test that all cross-platform plugins have unique language IDs."""
        language_ids = [plugin.get_language_id() for plugin in self.plugins]
        self.assertEqual(len(language_ids), len(set(language_ids)))

    def test_plugin_mutual_exclusivity(self):
        """Test that plugins don't handle each other's errors."""
        # React Native specific error
        rn_error = {
            "framework": "react-native",
            "message": "Metro bundler error",
            "runtime": "react-native",
        }

        # Only React Native plugin should handle this
        handlers = [plugin for plugin in self.plugins if plugin.can_handle(rn_error)]
        self.assertEqual(len(handlers), 1)
        self.assertEqual(handlers[0].get_language_id(), "react_native")

        # Flutter specific error
        flutter_error = {
            "framework": "flutter",
            "message": "Widget build error",
            "language": "dart",
        }

        # Only Flutter plugin should handle this
        handlers = [
            plugin for plugin in self.plugins if plugin.can_handle(flutter_error)
        ]
        self.assertEqual(len(handlers), 1)
        self.assertEqual(handlers[0].get_language_id(), "flutter")

    def test_ambiguous_error_handling(self):
        """Test handling of potentially ambiguous errors."""
        # Generic mobile error - should be handled by specific plugins based on context
        mobile_error = {
            "message": "Permission denied",
            "context": {"platform": "android"},
        }

        # This should not be handled by any plugin without more specific context
        handlers = [
            plugin for plugin in self.plugins if plugin.can_handle(mobile_error)
        ]
        self.assertEqual(len(handlers), 0)

        # Add framework context
        mobile_error["framework"] = "react-native"
        handlers = [
            plugin for plugin in self.plugins if plugin.can_handle(mobile_error)
        ]
        self.assertEqual(len(handlers), 1)
        self.assertEqual(handlers[0].get_language_id(), "react_native")

    def test_error_analysis_consistency(self):
        """Test that all plugins return consistent analysis structure."""
        test_error = {"message": "Test error message", "error_type": "TestError"}

        required_fields = ["category", "plugin", "suggested_fix", "root_cause"]

        for plugin in self.plugins:
            # Create framework-specific error
            framework_error = test_error.copy()
            framework_error["framework"] = plugin.get_language_id()

            if plugin.can_handle(framework_error):
                analysis = plugin.analyze_error(framework_error)

                # Check required fields
                for field in required_fields:
                    self.assertIn(
                        field,
                        analysis,
                        f"Plugin {plugin.get_language_id()} missing {field}",
                    )

                # Check plugin metadata
                self.assertEqual(analysis["plugin"], plugin.get_language_id())

    def test_fix_generation_robustness(self):
        """Test that fix generation handles edge cases gracefully."""
        test_error = {"message": "Test error message", "error_type": "TestError"}

        test_analysis = {"root_cause": "unknown_error", "category": "test"}

        for plugin in self.plugins:
            # Test with empty source code
            fix = plugin.generate_fix(test_analysis, {"source_code": ""})
            # Should either return a fix or None, not raise an exception
            self.assertTrue(fix is None or isinstance(fix, dict))

            # Test with malformed analysis
            malformed_analysis = {"invalid": "data"}
            fix = plugin.generate_fix(malformed_analysis, {"source_code": "test code"})
            self.assertTrue(fix is None or isinstance(fix, dict))

    def test_plugin_language_info(self):
        """Test that all plugins provide comprehensive language info."""
        required_info_fields = [
            "language",
            "version",
            "supported_extensions",
            "supported_frameworks",
            "features",
        ]

        for plugin in self.plugins:
            info = plugin.get_language_info()

            # Check required fields
            for field in required_info_fields:
                self.assertIn(
                    field, info, f"Plugin {plugin.get_language_id()} missing {field}"
                )

            # Check data types
            self.assertIsInstance(info["supported_extensions"], list)
            self.assertIsInstance(info["supported_frameworks"], list)
            self.assertIsInstance(info["features"], list)

            # Check non-empty
            self.assertGreater(len(info["features"]), 0)
            self.assertGreater(len(info["supported_frameworks"]), 0)


class TestCrossPlatformErrorScenarios(unittest.TestCase):
    """Test realistic cross-platform error scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.rn_plugin = ReactNativeLanguagePlugin()
        self.flutter_plugin = FlutterLanguagePlugin()
        self.xamarin_plugin = XamarinLanguagePlugin()
        self.unity_plugin = UnityLanguagePlugin()
        self.capacitor_plugin = CapacitorCordovaLanguagePlugin()

    def test_react_native_red_screen_error(self):
        """Test React Native red screen error scenario."""
        error_data = {
            "message": "RedBox: Unable to resolve module 'react-native-vector-icons'",
            "error_type": "Error",
            "runtime": "react-native",
            "stack_trace": ["Metro bundler"],
        }

        analysis = self.rn_plugin.analyze_error(error_data)
        fix = self.rn_plugin.generate_fix(analysis, {"source_code": ""})

        self.assertEqual(analysis["category"], "react_native")
        self.assertIsNotNone(fix)
        self.assertIn("install", str(fix).lower())

    def test_flutter_widget_overflow_scenario(self):
        """Test Flutter widget overflow scenario."""
        error_data = {
            "message": "RenderFlex overflowed by 87 pixels on the right",
            "error_type": "FlutterError",
            "language": "dart",
            "file": "main.dart",
            "line": 123,
        }

        analysis = self.flutter_plugin.analyze_error(error_data)
        fix = self.flutter_plugin.generate_fix(analysis, {"source_code": ""})

        self.assertEqual(analysis["category"], "flutter")
        self.assertIsNotNone(fix)
        self.assertIn("expanded", str(fix).lower())

    def test_xamarin_forms_binding_scenario(self):
        """Test Xamarin.Forms binding error scenario."""
        error_data = {
            "message": "Binding: 'Name' property not found on 'object'",
            "error_type": "BindingException",
            "framework": "xamarin.forms",
            "file": "MainPage.xaml",
        }

        analysis = self.xamarin_plugin.analyze_error(error_data)
        fix = self.xamarin_plugin.generate_fix(
            analysis, {"source_code": "", "error_data": error_data}
        )

        self.assertEqual(analysis["category"], "xamarin")
        self.assertIsNotNone(fix)
        self.assertIn("binding", str(fix).lower())

    def test_unity_mobile_build_scenario(self):
        """Test Unity mobile build error scenario."""
        error_data = {
            "message": "Build failed: Android SDK not found",
            "error_type": "BuildFailedException",
            "context": {"platform": "android", "unity_version": "2022.3.0f1"},
        }

        analysis = self.unity_plugin.analyze_error(error_data)
        fix = self.unity_plugin.generate_fix(analysis, {"source_code": ""})

        self.assertEqual(analysis["category"], "unity")
        self.assertIsNotNone(fix)
        self.assertIn("sdk", str(fix).lower())

    def test_capacitor_plugin_missing_scenario(self):
        """Test Capacitor plugin missing scenario."""
        error_data = {
            "message": "Plugin 'Camera' not found",
            "error_type": "PluginNotFoundError",
            "framework": "capacitor",
            "runtime": "capacitor",
        }

        analysis = self.capacitor_plugin.analyze_error(error_data)
        fix = self.capacitor_plugin.generate_fix(analysis, {"source_code": ""})

        self.assertEqual(analysis["category"], "capacitor_cordova")
        self.assertIsNotNone(fix)
        self.assertIn("install", str(fix).lower())
        self.assertIn("sync", str(fix).lower())


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
