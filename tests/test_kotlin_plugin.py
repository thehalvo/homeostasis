"""
Test cases for the Kotlin Language Plugin

This module contains comprehensive tests for Kotlin error detection, analysis,
and patch generation including Android, coroutines, Compose, Room, and multiplatform scenarios.
"""
import pytest
import json
import sys
import os
from pathlib import Path

# Add the modules directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.analysis.plugins.kotlin_plugin import KotlinLanguagePlugin, KotlinExceptionHandler, KotlinPatchGenerator
from modules.analysis.language_adapters import KotlinErrorAdapter


class TestKotlinErrorAdapter:
    """Test cases for Kotlin error adapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = KotlinErrorAdapter()
    
    def test_to_standard_format_basic_error(self):
        """Test basic Kotlin error conversion to standard format."""
        kotlin_error = {
            "type": "KotlinNullPointerException",
            "message": "Attempt to invoke virtual method on a null object reference",
            "location": "MainActivity.kt:42",
            "stackTrace": [
                "at com.example.MainActivity.onCreate(MainActivity.kt:42)",
                "at android.app.Activity.performCreate(Activity.java:7136)"
            ]
        }
        
        standard_error = self.adapter.to_standard_format(kotlin_error)
        
        assert standard_error["language"] == "kotlin"
        assert standard_error["error_type"] == "KotlinNullPointerException"
        assert standard_error["message"] == "Attempt to invoke virtual method on a null object reference"
        assert standard_error["file"] == "MainActivity.kt"
        assert standard_error["line"] == 42
        assert len(standard_error["stack_trace"]) == 2
    
    def test_to_standard_format_android_error(self):
        """Test Android-specific Kotlin error conversion."""
        kotlin_error = {
            "type": "ActivityNotFoundException",
            "message": "No Activity found to handle Intent",
            "android": {
                "api_level": 33,
                "device": "Pixel 6"
            },
            "stackTrace": "at android.app.Instrumentation.checkStartActivityResult(Instrumentation.java:2006)"
        }
        
        standard_error = self.adapter.to_standard_format(kotlin_error)
        
        assert standard_error["language"] == "kotlin"
        assert standard_error["android_api_level"] == 33
        assert standard_error["device"] == "Pixel 6"
    
    def test_from_standard_format(self):
        """Test conversion from standard format to Kotlin format."""
        standard_error = {
            "id": "test-123",
            "language": "kotlin",
            "error_type": "IllegalStateException",
            "message": "Fragment not attached",
            "file": "UserFragment.kt",
            "line": 67,
            "stack_trace": [
                {"function": "onCreate", "class": "UserFragment", "file": "UserFragment.kt", "line": 67}
            ]
        }
        
        kotlin_error = self.adapter.from_standard_format(standard_error)
        
        assert kotlin_error["type"] == "IllegalStateException"
        assert kotlin_error["location"] == "UserFragment.kt:67"
        assert len(kotlin_error["stackTrace"]) == 1
    
    def test_parse_kotlin_stack_trace(self):
        """Test parsing of Kotlin stack trace strings."""
        stack_trace_str = """
        at com.example.MainActivity.onCreate(MainActivity.kt:42)
        at kotlinx.coroutines.JobSupport.handleException(JobSupport.kt:123)
        at android.app.Activity.performCreate(Activity.java:7136)
        """
        
        frames = self.adapter._parse_kotlin_stack_trace(stack_trace_str.strip())
        
        assert len(frames) == 3
        assert frames[0]["function"] == "onCreate"
        assert frames[0]["class"] == "MainActivity"
        assert frames[0]["package"] == "com.example"
        assert frames[0]["file"] == "MainActivity.kt"
        assert frames[0]["line"] == 42


class TestKotlinExceptionHandler:
    """Test cases for Kotlin exception handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = KotlinExceptionHandler()
    
    def test_analyze_null_pointer_exception(self):
        """Test analysis of Kotlin null pointer exception."""
        error_data = {
            "error_type": "KotlinNullPointerException",
            "message": "Attempt to invoke virtual method on a null object reference",
            "stack_trace": ["at com.example.MainActivity.onCreate(MainActivity.kt:42)"]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["rule_id"] == "kotlin_null_pointer_exception"
        assert analysis["root_cause"] == "kotlin_null_pointer"
        assert analysis["severity"] == "high"
        assert analysis["category"] == "null_safety"
        assert "safe call operator" in analysis["suggestion"]
    
    def test_analyze_coroutine_cancellation(self):
        """Test analysis of coroutine cancellation exception."""
        error_data = {
            "error_type": "CancellationException",
            "message": "Job was cancelled",
            "stack_trace": ["at kotlinx.coroutines.JobSupport.cancel(JobSupport.kt:123)"]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "kotlin_coroutine_cancelled"
        assert analysis["category"] == "coroutines"
        assert analysis["framework"] == "coroutines"
        assert "CancellationException" in analysis["suggestion"]
    
    def test_analyze_room_main_thread_error(self):
        """Test analysis of Room main thread access error."""
        error_data = {
            "error_type": "IllegalStateException",
            "message": "Cannot access database on the main thread since it may potentially lock the UI",
            "stack_trace": ["at androidx.room.RoomDatabase.assertNotMainThread(RoomDatabase.kt:123)"]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "kotlin_room_main_thread_access"
        assert analysis["category"] == "room"
        assert analysis["framework"] == "room"
        assert "background thread" in analysis["suggestion"]
    
    def test_analyze_android_fragment_error(self):
        """Test analysis of Android fragment not attached error."""
        error_data = {
            "error_type": "IllegalStateException",
            "message": "Fragment UserFragment not attached to a context",
            "stack_trace": ["at androidx.fragment.app.Fragment.requireContext(Fragment.kt:123)"]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "kotlin_fragment_not_attached"
        assert analysis["category"] == "android"
        assert analysis["framework"] == "android"
        assert "isAdded" in analysis["suggestion"]
    
    def test_analyze_compose_recomposition_loop(self):
        """Test analysis of Compose infinite recomposition."""
        error_data = {
            "error_type": "InfiniteRecomposition",
            "message": "Infinite recomposition loop detected",
            "stack_trace": ["at androidx.compose.runtime.Recomposer.recompose(Recomposer.kt:123)"]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "kotlin_compose_infinite_recomposition"
        assert analysis["category"] == "compose"
        assert analysis["framework"] == "compose"
        assert "remember{}" in analysis["suggestion"]
    
    def test_fallback_analysis(self):
        """Test fallback analysis for unknown errors."""
        error_data = {
            "error_type": "CustomException",
            "message": "Unknown custom error",
            "stack_trace": []
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["rule_id"] == "kotlin_generic_fallback"
        assert analysis["root_cause"] == "kotlin_unknown_error"
        assert analysis["confidence"] == "low"


class TestKotlinPatchGenerator:
    """Test cases for Kotlin patch generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = KotlinPatchGenerator()
    
    def test_generate_null_safety_patch(self):
        """Test patch generation for null safety issues."""
        analysis = {
            "rule_id": "kotlin_null_pointer_exception",
            "root_cause": "kotlin_null_pointer",
            "error_data": {
                "error_type": "KotlinNullPointerException",
                "message": "Null pointer access"
            },
            "suggestion": "Use safe call operator",
            "confidence": "high",
            "severity": "high"
        }
        
        context = {
            "code_snippet": "user.name.length",
            "framework": ""
        }
        
        patch = self.generator.generate_patch(analysis, context)
        
        assert patch["language"] == "kotlin"
        assert patch["root_cause"] == "kotlin_null_pointer"
        assert "suggestion_code" in patch
        assert "safe call operator" in patch["suggestion_code"]
    
    def test_generate_coroutine_patch(self):
        """Test patch generation for coroutine cancellation."""
        analysis = {
            "rule_id": "kotlin_cancellation_exception",
            "root_cause": "kotlin_coroutine_cancelled",
            "error_data": {
                "error_type": "CancellationException"
            },
            "suggestion": "Handle cancellation",
            "confidence": "high",
            "severity": "medium"
        }
        
        context = {}
        
        patch = self.generator.generate_patch(analysis, context)
        
        assert patch["root_cause"] == "kotlin_coroutine_cancelled"
        assert "suggestion_code" in patch
        assert "CancellationException" in patch["suggestion_code"]
    
    def test_generate_room_patch(self):
        """Test patch generation for Room database issues."""
        analysis = {
            "rule_id": "kotlin_room_main_thread_query",
            "root_cause": "kotlin_room_main_thread_access",
            "error_data": {
                "error_type": "IllegalStateException"
            },
            "suggestion": "Use background thread",
            "confidence": "high",
            "severity": "high"
        }
        
        context = {}
        
        patch = self.generator.generate_patch(analysis, context)
        
        assert patch["root_cause"] == "kotlin_room_main_thread_access"
        assert "suggestion_code" in patch
        assert "Dispatchers.IO" in patch["suggestion_code"]
    
    def test_generate_compose_patch(self):
        """Test patch generation for Compose issues."""
        analysis = {
            "rule_id": "kotlin_compose_state_not_remembered",
            "root_cause": "kotlin_compose_state_not_remembered",
            "error_data": {
                "error_type": "StateNotRemembered"
            },
            "suggestion": "Use remember",
            "confidence": "high",
            "severity": "medium"
        }
        
        context = {}
        
        patch = self.generator.generate_patch(analysis, context)
        
        assert patch["root_cause"] == "kotlin_compose_state_not_remembered"
        assert "suggestion_code" in patch
        assert "remember {" in patch["suggestion_code"]


class TestKotlinLanguagePlugin:
    """Test cases for the main Kotlin language plugin."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = KotlinLanguagePlugin()
    
    def test_plugin_metadata(self):
        """Test plugin metadata and identification."""
        assert self.plugin.get_language_id() == "kotlin"
        assert self.plugin.get_language_name() == "Kotlin"
        assert self.plugin.get_language_version() == "1.8+"
        
        frameworks = self.plugin.get_supported_frameworks()
        assert "android" in frameworks
        assert "coroutines" in frameworks
        assert "compose" in frameworks
        assert "room" in frameworks
        assert "multiplatform" in frameworks
    
    def test_analyze_error(self):
        """Test error analysis through the plugin."""
        error_data = {
            "error_type": "KotlinNullPointerException",
            "message": "Attempt to invoke virtual method on a null object reference",
            "stack_trace": ["at com.example.MainActivity.onCreate(MainActivity.kt:42)"]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["root_cause"] == "kotlin_null_pointer"
        assert analysis["language"] == "kotlin"
        assert analysis["severity"] == "high"
    
    def test_normalize_error(self):
        """Test error normalization."""
        raw_error = {
            "type": "NullPointerException",
            "message": "null pointer",
            "location": "Test.kt:10"
        }
        
        normalized = self.plugin.normalize_error(raw_error)
        
        assert normalized["language"] == "kotlin"
        assert normalized["error_type"] == "NullPointerException"
        assert normalized["file"] == "Test.kt"
        assert normalized["line"] == 10
    
    def test_generate_fix(self):
        """Test fix generation through the plugin."""
        analysis = {
            "rule_id": "kotlin_null_pointer_exception",
            "root_cause": "kotlin_null_pointer",
            "error_data": {
                "error_type": "KotlinNullPointerException"
            },
            "suggestion": "Use null safety",
            "confidence": "high",
            "severity": "high"
        }
        
        context = {"code_snippet": "user.name"}
        
        fix = self.plugin.generate_fix(analysis, context)
        
        assert fix["language"] == "kotlin"
        assert fix["root_cause"] == "kotlin_null_pointer"
        assert "suggestion" in fix


class TestKotlinIntegrationScenarios:
    """Integration test scenarios for real-world Kotlin issues."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = KotlinLanguagePlugin()
    
    def test_android_lifecycle_scenario(self):
        """Test Android activity lifecycle error scenario."""
        error_data = {
            "error_type": "ActivityNotFoundException",
            "message": "No Activity found to handle Intent { act=android.intent.action.VIEW }",
            "stack_trace": [
                "at android.app.Instrumentation.checkStartActivityResult(Instrumentation.java:2006)",
                "at com.example.MainActivity.startActivity(MainActivity.kt:123)"
            ],
            "android": {
                "api_level": 33,
                "device": "Pixel 6"
            }
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["root_cause"] == "kotlin_activity_not_found"
        assert analysis["framework"] == "android"
        
        fix = self.plugin.generate_fix(analysis, {})
        assert "AndroidManifest.xml" in fix["suggestion"]
    
    def test_coroutine_timeout_scenario(self):
        """Test coroutine timeout error scenario."""
        error_data = {
            "error_type": "TimeoutCancellationException",
            "message": "Timed out waiting for 5000 ms",
            "stack_trace": [
                "at kotlinx.coroutines.TimeoutKt.TimeoutCancellationException(Timeout.kt:186)",
                "at com.example.NetworkService.fetchData(NetworkService.kt:45)"
            ],
            "coroutine": {
                "name": "NetworkCall",
                "dispatcher": "IO"
            }
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["root_cause"] == "kotlin_coroutine_timeout"
        assert analysis["framework"] == "coroutines"
        
        fix = self.plugin.generate_fix(analysis, {})
        assert "withTimeout" in fix["suggestion"]
    
    def test_room_migration_scenario(self):
        """Test Room database migration error scenario."""
        error_data = {
            "error_type": "IllegalStateException",
            "message": "A migration from 1 to 2 was required but not found",
            "stack_trace": [
                "at androidx.room.RoomOpenHelper.onUpgrade(RoomOpenHelper.java:94)",
                "at com.example.AppDatabase.migration(AppDatabase.kt:67)"
            ]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["root_cause"] == "kotlin_room_migration_missing"
        assert analysis["framework"] == "room"
        
        fix = self.plugin.generate_fix(analysis, {})
        assert "Migration" in fix["suggestion"]
    
    def test_compose_performance_scenario(self):
        """Test Compose performance issue scenario."""
        error_data = {
            "error_type": "InfiniteRecomposition",
            "message": "Too many recompositions detected",
            "stack_trace": [
                "at androidx.compose.runtime.Recomposer.performRecompose(Recomposer.kt:456)",
                "at com.example.ui.UserScreen.Content(UserScreen.kt:89)"
            ]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["root_cause"] == "kotlin_compose_infinite_recomposition"
        assert analysis["framework"] == "compose"
        
        fix = self.plugin.generate_fix(analysis, {})
        assert "remember" in fix["suggestion"]


if __name__ == "__main__":
    # Run tests with pytest
    import subprocess
    import sys
    
    # Run the tests
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    sys.exit(result.returncode)