"""
Test cases for Java Android language plugin functionality.

This module tests the Java Android integration including:
- Activity lifecycle monitoring
- Fragment transaction error handling
- API compatibility checks
- Java-Kotlin interoperability
- Service and background task healing
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))

from analysis.plugins.java_android_plugin import AndroidJavaLanguagePlugin, AndroidJavaExceptionHandler


class TestJavaAndroidPlugin(unittest.TestCase):
    """Test Java Android plugin functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plugin = AndroidJavaLanguagePlugin()
        self.exception_handler = AndroidJavaExceptionHandler()
    
    def test_android_activity_not_found_detection(self):
        """Test detection of ActivityNotFoundException."""
        error_data = {
            "error_type": "android.content.ActivityNotFoundException",
            "message": "No Activity found to handle Intent",
            "stack_trace": [
                "at android.app.Instrumentation.checkStartActivityResult(Instrumentation.java:2019)",
                "at android.app.Instrumentation.execStartActivity(Instrumentation.java:1616)",
                "at android.app.Activity.startActivityForResult(Activity.java:4490)"
            ],
            "language": "java",
            "framework": "android"
        }
        
        analysis = self.exception_handler.analyze_exception(error_data)
        
        self.assertEqual(analysis["root_cause"], "java_android_activity_not_found")
        self.assertEqual(analysis["category"], "android")
        self.assertEqual(analysis["severity"], "high")
        self.assertIn("intent", analysis["suggestion"].lower())
    
    def test_android_fragment_not_attached_detection(self):
        """Test detection of fragment not attached errors."""
        error_data = {
            "error_type": "java.lang.IllegalStateException",
            "message": "Fragment not attached to a context",
            "stack_trace": [
                "at androidx.fragment.app.Fragment.requireContext(Fragment.java:860)",
                "at com.example.MyFragment.onViewCreated(MyFragment.java:45)"
            ],
            "language": "java",
            "framework": "android"
        }
        
        analysis = self.exception_handler.analyze_exception(error_data)
        
        self.assertEqual(analysis["root_cause"], "java_android_fragment_not_attached")
        self.assertEqual(analysis["category"], "android")
        self.assertIn("isAdded", analysis["suggestion"])
    
    def test_android_main_thread_network_detection(self):
        """Test detection of network on main thread violations."""
        error_data = {
            "error_type": "android.os.NetworkOnMainThreadException",
            "message": "Network operation attempted on main UI thread",
            "stack_trace": [
                "at android.os.StrictMode$AndroidBlockGuardPolicy.onNetwork(StrictMode.java:1513)",
                "at java.net.InetAddress.lookupHostByName(InetAddress.java:418)",
                "at java.net.InetAddress.getAllByNameImpl(InetAddress.java:236)"
            ],
            "language": "java",
            "framework": "android"
        }
        
        analysis = self.exception_handler.analyze_exception(error_data)
        
        self.assertEqual(analysis["root_cause"], "java_android_main_thread_violation")
        self.assertEqual(analysis["severity"], "high")
        self.assertIn("background", analysis["suggestion"].lower())
    
    def test_android_view_not_found_detection(self):
        """Test detection of view not found errors."""
        error_data = {
            "error_type": "java.lang.NullPointerException",
            "message": "findViewById returned null",
            "stack_trace": [
                "at com.example.MainActivity.onCreate(MainActivity.java:34)",
                "at android.app.Activity.performCreate(Activity.java:7802)"
            ],
            "language": "java",
            "framework": "android"
        }
        
        analysis = self.exception_handler.analyze_exception(error_data)
        
        self.assertEqual(analysis["root_cause"], "java_android_view_not_found")
        self.assertIn("layout", analysis["suggestion"].lower())
    
    def test_android_permission_denied_detection(self):
        """Test detection of permission denied errors."""
        error_data = {
            "error_type": "java.lang.SecurityException",
            "message": "Permission denied: reading com.android.providers.telephony.MmsSmsProvider",
            "stack_trace": [
                "at android.os.Parcel.readException(Parcel.java:2056)",
                "at android.content.ContentProviderProxy.query(ContentProviderNative.java:491)"
            ],
            "language": "java",
            "framework": "android"
        }
        
        analysis = self.exception_handler.analyze_exception(error_data)
        
        self.assertEqual(analysis["root_cause"], "java_android_permission_denied")
        self.assertIn("permission", analysis["suggestion"].lower())
    
    def test_android_ui_thread_violation_detection(self):
        """Test detection of UI operations from wrong thread."""
        error_data = {
            "error_type": "android.view.ViewRootImpl$CalledFromWrongThreadException",
            "message": "Only the original thread that created a view hierarchy can touch its views",
            "stack_trace": [
                "at android.view.ViewRootImpl.checkThread(ViewRootImpl.java:7581)",
                "at android.view.ViewRootImpl.invalidateChildInParent(ViewRootImpl.java:1392)"
            ],
            "language": "java",
            "framework": "android"
        }
        
        analysis = self.exception_handler.analyze_exception(error_data)
        
        self.assertEqual(analysis["root_cause"], "java_android_ui_thread_violation")
        self.assertIn("runOnUiThread", analysis["suggestion"])
    
    def test_kotlin_interop_null_safety_detection(self):
        """Test detection of Java-Kotlin null safety violations."""
        error_data = {
            "error_type": "kotlin.KotlinNullPointerException",
            "message": "null cannot be cast to non-null type",
            "stack_trace": [
                "at com.example.KotlinClass.processData(KotlinClass.kt:25)",
                "at com.example.JavaActivity.callKotlinMethod(JavaActivity.java:67)"
            ],
            "language": "java",
            "framework": "kotlin-interop"
        }
        
        analysis = self.exception_handler.analyze_exception(error_data)
        
        self.assertEqual(analysis["root_cause"], "java_kotlin_null_safety_violation")
        self.assertIn("@Nullable", analysis["suggestion"])
    
    def test_service_binding_failed_detection(self):
        """Test detection of service binding failures."""
        error_data = {
            "error_type": "ServiceBindingException",
            "message": "bindService returned false",
            "stack_trace": [
                "at com.example.MainActivity.bindToService(MainActivity.java:89)",
                "at com.example.MainActivity.onCreate(MainActivity.java:45)"
            ],
            "language": "java",
            "framework": "android"
        }
        
        analysis = self.exception_handler.analyze_exception(error_data)
        
        self.assertEqual(analysis["root_cause"], "java_android_service_binding_failed")
        self.assertIn("explicit", analysis["suggestion"].lower())
    
    def test_background_execution_limit_detection(self):
        """Test detection of background execution limit violations."""
        error_data = {
            "error_type": "java.lang.IllegalStateException",
            "message": "Not allowed to start service Intent: app is in background",
            "stack_trace": [
                "at android.app.ContextImpl.startServiceCommon(ContextImpl.java:1726)",
                "at android.app.ContextImpl.startService(ContextImpl.java:1681)"
            ],
            "language": "java",
            "framework": "android"
        }
        
        analysis = self.exception_handler.analyze_exception(error_data)
        
        self.assertEqual(analysis["root_cause"], "java_android_background_limit_violation")
        self.assertIn("foreground", analysis["suggestion"].lower())
    
    def test_api_compatibility_detection(self):
        """Test detection of API compatibility issues."""
        error_data = {
            "error_type": "java.lang.NoSuchMethodError",
            "message": "No virtual method setTextAppearance in class Landroid/widget/TextView",
            "stack_trace": [
                "at com.example.MainActivity.setupTextView(MainActivity.java:67)",
                "at com.example.MainActivity.onCreate(MainActivity.java:34)"
            ],
            "language": "java",
            "framework": "android"
        }
        
        analysis = self.exception_handler.analyze_exception(error_data)
        
        self.assertEqual(analysis["root_cause"], "java_android_api_compatibility")
        self.assertIn("SDK_INT", analysis["suggestion"])
    
    def test_patch_generation_activity_lifecycle(self):
        """Test patch generation for activity lifecycle issues."""
        analysis = {
            "root_cause": "java_android_lifecycle_violation",
            "error_data": {
                "error_type": "IllegalStateException",
                "message": "Activity destroyed",
                "stack_trace": []
            },
            "rule_id": "java_android_activity_lifecycle_state",
            "confidence": "high",
            "severity": "high"
        }
        
        context = {
            "code_snippet": "doSomething();",
            "framework": "android"
        }
        
        patch = self.plugin.generate_fix(analysis, context)
        
        self.assertEqual(patch["language"], "java")
        self.assertEqual(patch["framework"], "android")
        self.assertIn("isFinishing", patch["suggestion"])
    
    def test_patch_generation_main_thread_violation(self):
        """Test patch generation for main thread violations."""
        analysis = {
            "root_cause": "java_android_main_thread_violation",
            "error_data": {
                "error_type": "NetworkOnMainThreadException",
                "message": "Network on main thread",
                "stack_trace": []
            },
            "rule_id": "java_android_main_thread_network",
            "confidence": "high",
            "severity": "high"
        }
        
        context = {
            "code_snippet": "httpClient.execute(request);",
            "framework": "android"
        }
        
        patch = self.plugin.generate_fix(analysis, context)
        
        self.assertEqual(patch["language"], "java")
        self.assertIn("executor", patch["suggestion"].lower())
    
    def test_supported_frameworks(self):
        """Test that Android is listed as a supported framework."""
        frameworks = self.plugin.get_supported_frameworks()
        self.assertIn("android", frameworks)
    
    def test_android_rules_loading(self):
        """Test that Android-specific rules are loaded."""
        # Check that Android rules are included in the loaded rules
        android_rules = []
        # Flatten rules from all categories
        for category, rule_list in self.exception_handler.rules.items():
            if isinstance(rule_list, list):
                for rule in rule_list:
                    if isinstance(rule, dict) and rule.get("category") == "android":
                        android_rules.append(rule)
        
        self.assertGreater(len(android_rules), 0, "No Android rules loaded")
        
        # Check for specific Android rule IDs
        android_rule_ids = [rule.get("id") for rule in android_rules]
        expected_rules = [
            "java_android_activity_not_found",
            "java_android_fragment_not_attached",
            "java_android_main_thread_network",
            "java_android_permission_denied"
        ]
        
        for expected_rule in expected_rules:
            self.assertIn(expected_rule, android_rule_ids, 
                         f"Expected Android rule {expected_rule} not found")


class TestJavaAndroidIntegration(unittest.TestCase):
    """Test Java Android plugin integration scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plugin = AndroidJavaLanguagePlugin()
    
    def test_error_normalization_android(self):
        """Test error normalization for Android errors."""
        raw_error = {
            "exception_class": "android.content.ActivityNotFoundException",
            "message": "No Activity found to handle Intent",
            "stacktrace": "at android.app.Activity.startActivity(...)",
            "platform": "android"
        }
        
        normalized = self.plugin.normalize_error(raw_error)
        
        self.assertEqual(normalized["language"], "java")
        self.assertEqual(normalized["error_type"], "android.content.ActivityNotFoundException")
        self.assertIsInstance(normalized["stack_trace"], list)
    
    def test_end_to_end_android_error_analysis(self):
        """Test complete error analysis flow for Android errors."""
        error_data = {
            "error_type": "android.view.WindowManager$BadTokenException",
            "message": "Unable to add window -- token null is not valid",
            "stack_trace": [
                "at android.view.ViewRootImpl.setView(ViewRootImpl.java:958)",
                "at android.view.WindowManagerGlobal.addView(WindowManagerGlobal.java:387)"
            ],
            "language": "java",
            "framework": "android"
        }
        
        # Analyze the error
        analysis = self.plugin.analyze_error(error_data)
        
        # Generate a fix
        context = {"framework": "android", "code_snippet": "dialog.show();"}
        patch = self.plugin.generate_fix(analysis, context)
        
        # Verify the complete flow
        self.assertEqual(analysis["category"], "android")
        self.assertEqual(patch["language"], "java")
        self.assertIn("token", patch["suggestion"].lower())


if __name__ == '__main__':
    unittest.main()