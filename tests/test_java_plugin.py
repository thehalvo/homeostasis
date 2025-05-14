import json
import os
import pytest
from pathlib import Path

from modules.analysis.plugins.java_plugin import JavaLanguagePlugin, JavaExceptionHandler
from modules.analysis.language_adapters import JavaErrorAdapter

class TestJavaLanguagePlugin:
    """Tests for Java language plugin integration"""
    
    @pytest.fixture
    def java_plugin(self):
        """Create a Java language plugin instance for testing"""
        return JavaLanguagePlugin()
    
    @pytest.fixture
    def error_data_npe(self):
        """Sample NPE error data for testing"""
        return {
            "error_type": "java.lang.NullPointerException",
            "message": "Cannot invoke \"String.length()\" because \"str\" is null",
            "stack_trace": [
                {
                    "file": "StringProcessor.java",
                    "line": 42,
                    "class": "StringProcessor",
                    "function": "processString",
                    "package": "com.example.util"
                },
                {
                    "file": "Main.java",
                    "line": 25,
                    "class": "Main",
                    "function": "main",
                    "package": "com.example"
                }
            ],
            "timestamp": "2023-07-15T14:32:10.123Z",
            "application": "sample-java-app"
        }
    
    @pytest.fixture
    def error_data_concurrent(self):
        """Sample ConcurrentModificationException error data for testing"""
        return {
            "error_type": "java.util.ConcurrentModificationException",
            "message": "",
            "stack_trace": [
                {
                    "file": "CollectionProcessor.java",
                    "line": 78,
                    "class": "CollectionProcessor",
                    "function": "processItems",
                    "package": "com.example.collection"
                },
                {
                    "file": "Main.java",
                    "line": 30,
                    "class": "Main",
                    "function": "main",
                    "package": "com.example"
                }
            ],
            "timestamp": "2023-07-15T16:45:20.789Z",
            "application": "sample-java-app"
        }
    
    @pytest.fixture
    def error_data_spring(self):
        """Sample Spring bean definition error data for testing"""
        return {
            "error_type": "org.springframework.beans.factory.UnsatisfiedDependencyException",
            "message": "Error creating bean with name 'userService': Unsatisfied dependency expressed through constructor parameter 0; nested exception is org.springframework.beans.factory.NoSuchBeanDefinitionException: No qualifying bean of type 'com.example.repository.UserRepository' available",
            "stack_trace": [
                {
                    "file": "AbstractAutowireCapableBeanFactory.java",
                    "line": 1225,
                    "class": "AbstractAutowireCapableBeanFactory",
                    "function": "createBean",
                    "package": "org.springframework.beans.factory.support"
                },
                {
                    "file": "UserServiceImpl.java",
                    "line": 25,
                    "class": "UserServiceImpl",
                    "function": "<init>",
                    "package": "com.example.service"
                }
            ],
            "timestamp": "2023-07-15T17:30:45.123Z",
            "application": "spring-boot-app"
        }
    
    def test_plugin_initialization(self, java_plugin):
        """Test that the Java plugin initializes correctly"""
        assert java_plugin.get_language_id() == "java"
        assert java_plugin.get_language_name() == "Java"
        assert java_plugin.get_language_version() == "8+"
        assert "spring" in java_plugin.get_supported_frameworks()
        assert "hibernate" in java_plugin.get_supported_frameworks()
    
    def test_analyze_npe(self, java_plugin, error_data_npe):
        """Test analysis of NullPointerException"""
        analysis = java_plugin.analyze_error(error_data_npe)
        
        assert analysis is not None
        assert analysis["error_type"] == "NullPointerException"
        assert analysis["root_cause"] == "java_null_pointer"
        assert analysis["confidence"] == "high"
        assert analysis["severity"] == "high"
    
    def test_analyze_concurrent_modification(self, java_plugin, error_data_concurrent):
        """Test analysis of ConcurrentModificationException"""
        analysis = java_plugin.analyze_error(error_data_concurrent)
        
        assert analysis is not None
        assert analysis["error_type"] == "ConcurrentModificationException"
        assert analysis["root_cause"] == "java_concurrent_modification"
        assert analysis["confidence"] == "high"
        assert analysis["severity"] == "high"
    
    def test_analyze_spring_error(self, java_plugin, error_data_spring):
        """Test analysis of Spring Framework error"""
        analysis = java_plugin.analyze_error(error_data_spring)
        
        assert analysis is not None
        assert "spring" in analysis["root_cause"]
        assert "bean" in analysis["description"].lower()
        assert "framework" in analysis
        assert analysis["framework"] == "spring"
    
    def test_generate_fix_for_npe(self, java_plugin, error_data_npe):
        """Test fix generation for NullPointerException"""
        analysis = java_plugin.analyze_error(error_data_npe)
        context = {
            "code_snippet": "String result = str.length();",
            "method_params": "String str"
        }
        
        fix = java_plugin.generate_fix(analysis, context)
        
        assert fix is not None
        assert fix["patch_type"] in ["code", "suggestion"]
        assert fix["language"] == "java"
        assert "null" in fix.get("suggestion", "").lower()
        
        # When a template is available, it should generate actual code
        if "patch_code" in fix:
            assert "if (str == null)" in fix["patch_code"]
            assert "application_point" in fix
    
    def test_generate_fix_for_concurrent_modification(self, java_plugin, error_data_concurrent):
        """Test fix generation for ConcurrentModificationException"""
        analysis = java_plugin.analyze_error(error_data_concurrent)
        context = {
            "code_snippet": "for (Item item : items) {\n    if (item.isExpired()) {\n        items.remove(item);\n    }\n}",
            "method_params": "List<Item> items"
        }
        
        fix = java_plugin.generate_fix(analysis, context)
        
        assert fix is not None
        assert fix["patch_type"] in ["code", "suggestion"]
        assert fix["language"] == "java"
        assert "concurrent" in fix.get("suggestion", "").lower()
        
        # If we have actual code or a suggestion code
        suggestion_text = fix.get("patch_code", "") or fix.get("suggestion_code", "")
        assert "Iterator" in suggestion_text or "CopyOnWrite" in suggestion_text
    
    def test_rule_loading(self):
        """Test that rule files are properly loaded"""
        handler = JavaExceptionHandler()
        
        # Check core Java rules
        assert any(rule["id"] == "java_null_pointer" for rule in handler.rules)
        assert any(rule["id"] == "java_class_cast" for rule in handler.rules)
        
        # Check concurrency rules
        assert any(rule["id"] == "java_concurrent_modification" for rule in handler.rules)
        
        # Check IO rules
        assert any(rule["id"] == "java_file_not_found" for rule in handler.rules)
        
        # Check JDBC rules
        assert any(rule["id"] == "java_sql_exception" for rule in handler.rules)
    
    def test_error_normalization(self, java_plugin):
        """Test normalization of Java errors to standard format"""
        # Test with a raw Java stack trace
        raw_error = {
            "message": "Exception in thread \"main\" java.lang.NullPointerException\n\tat com.example.Main.process(Main.java:25)\n\tat com.example.Main.main(Main.java:10)"
        }
        
        standardized = java_plugin.normalize_error(raw_error)
        
        assert standardized["error_type"] == "java.lang.NullPointerException"
        assert isinstance(standardized["stack_trace"], list)
        assert len(standardized["stack_trace"]) == 2
    
    def test_adapter_standardization(self):
        """Test the JavaErrorAdapter for standardizing error formats"""
        adapter = JavaErrorAdapter()
        
        # Test with a standard Java exception format
        java_error = {
            "exception": "java.lang.ArrayIndexOutOfBoundsException: Index 5 out of bounds for length 5",
            "stacktrace": [
                "at com.example.ArrayProcessor.processArray(ArrayProcessor.java:42)",
                "at com.example.Main.main(Main.java:15)"
            ]
        }
        
        standard = adapter.to_standard_format(java_error)
        
        assert standard["error_type"] == "java.lang.ArrayIndexOutOfBoundsException"
        assert "Index 5 out of bounds" in standard["message"]
        assert isinstance(standard["stack_trace"], list)
        assert len(standard["stack_trace"]) == 2
        assert standard["language"] == "java"