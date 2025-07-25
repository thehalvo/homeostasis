"""
Test cases for Java language plugin.

This module contains comprehensive test cases for the Java plugin,
including error analysis, concurrency issues, Spring/Hibernate framework errors,
dependency handling, compilation errors, and fix generation.
"""
import json
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from modules.analysis.plugins.java_plugin import JavaLanguagePlugin, JavaExceptionHandler, JavaPatchGenerator
from modules.analysis.language_adapters import JavaErrorAdapter


class TestJavaErrorAdapter:
    """Test cases for Java error adapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = JavaErrorAdapter()
    
    def test_to_standard_format_basic_exception(self):
        """Test basic Java exception conversion to standard format."""
        java_error = {
            "exception": "java.lang.NullPointerException: Cannot invoke \"String.length()\" because \"str\" is null",
            "stacktrace": [
                "at com.example.StringProcessor.processString(StringProcessor.java:42)",
                "at com.example.Main.main(Main.java:25)"
            ]
        }
        
        standard_error = self.adapter.to_standard_format(java_error)
        
        assert standard_error["language"] == "java"
        assert standard_error["error_type"] == "java.lang.NullPointerException"
        assert standard_error["message"] == "Cannot invoke \"String.length()\" because \"str\" is null"
        assert len(standard_error["stack_trace"]) == 2
        assert standard_error["stack_trace"][0]["file"] == "StringProcessor.java"
        assert standard_error["stack_trace"][0]["line"] == 42
        assert standard_error["stack_trace"][0]["class"] == "StringProcessor"
        assert standard_error["stack_trace"][0]["function"] == "processString"
    
    def test_to_standard_format_compilation_error(self):
        """Test Java compilation error conversion."""
        java_error = {
            "type": "CompilationError",
            "file": "Calculator.java",
            "line": 15,
            "column": 25,
            "message": "cannot find symbol",
            "symbol": "variable total",
            "location": "class com.example.Calculator"
        }
        
        standard_error = self.adapter.to_standard_format(java_error)
        
        assert standard_error["language"] == "java"
        assert standard_error["error_type"] == "CompilationError"
        assert standard_error["file"] == "Calculator.java"
        assert standard_error["line"] == 15
        assert standard_error["column"] == 25
        assert "cannot find symbol" in standard_error["message"]
    
    def test_to_standard_format_spring_error(self):
        """Test Spring Framework error conversion."""
        java_error = {
            "exception": "org.springframework.beans.factory.BeanCreationException",
            "message": "Error creating bean with name 'userService'",
            "stacktrace": [
                "at org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.createBean(AbstractAutowireCapableBeanFactory.java:553)",
                "at com.example.config.ApplicationConfig.userService(ApplicationConfig.java:45)"
            ],
            "caused_by": {
                "exception": "java.lang.IllegalStateException",
                "message": "Repository not found"
            }
        }
        
        standard_error = self.adapter.to_standard_format(java_error)
        
        assert standard_error["language"] == "java"
        assert "BeanCreationException" in standard_error["error_type"]
        assert standard_error["framework"] == "spring"
        assert "caused_by" in standard_error
    
    def test_from_standard_format(self):
        """Test conversion from standard format to Java format."""
        standard_error = {
            "id": "test-123",
            "language": "java",
            "error_type": "java.lang.ClassCastException",
            "message": "Cannot cast String to Integer",
            "file": "TypeConverter.java",
            "line": 30,
            "stack_trace": [
                {"function": "convert", "class": "TypeConverter", "file": "TypeConverter.java", "line": 30}
            ]
        }
        
        java_error = self.adapter.from_standard_format(standard_error)
        
        assert java_error["exception"] == "java.lang.ClassCastException"
        assert java_error["message"] == "Cannot cast String to Integer"
        assert len(java_error["stacktrace"]) == 1
        assert "TypeConverter.java:30" in java_error["stacktrace"][0]
    
    def test_parse_java_stack_trace(self):
        """Test parsing of Java stack trace strings."""
        stack_trace_str = """
Exception in thread "main" java.lang.ArrayIndexOutOfBoundsException: Index 10 out of bounds for length 5
\tat com.example.ArrayProcessor.processArray(ArrayProcessor.java:42)
\tat com.example.Main.main(Main.java:15)
\tat sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
        """
        
        frames = self.adapter._parse_java_stack_trace(stack_trace_str.strip())
        
        assert len(frames) >= 2
        assert frames[0]["function"] == "processArray"
        assert frames[0]["class"] == "ArrayProcessor"
        assert frames[0]["package"] == "com.example"
        assert frames[0]["file"] == "ArrayProcessor.java"
        assert frames[0]["line"] == 42


class TestJavaExceptionHandler:
    """Test cases for Java exception handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = JavaExceptionHandler()
    
    def test_analyze_null_pointer_exception(self):
        """Test analysis of NullPointerException."""
        error_data = {
            "error_type": "java.lang.NullPointerException",
            "message": "Cannot invoke \"String.length()\" because \"str\" is null",
            "stack_trace": [{"class": "StringProcessor", "function": "processString"}]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["rule_id"] == "java_null_pointer"
        assert analysis["root_cause"] == "java_null_pointer"
        assert analysis["severity"] == "high"
        assert analysis["category"] == "runtime"
        assert "null check" in analysis["suggestion"].lower()
    
    def test_analyze_class_cast_exception(self):
        """Test analysis of ClassCastException."""
        error_data = {
            "error_type": "java.lang.ClassCastException",
            "message": "class java.lang.String cannot be cast to class java.lang.Integer",
            "stack_trace": []
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "java_class_cast"
        assert analysis["category"] == "runtime"
        assert "instanceof" in analysis["suggestion"]
    
    def test_analyze_concurrent_modification(self):
        """Test analysis of ConcurrentModificationException."""
        error_data = {
            "error_type": "java.util.ConcurrentModificationException",
            "message": "",
            "stack_trace": [{"class": "HashMap$HashIterator", "function": "nextNode"}]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "java_concurrent_modification"
        assert analysis["category"] == "concurrency"
        assert analysis["tags"] == ["concurrency", "collections"]
        assert "Iterator" in analysis["suggestion"] or "ConcurrentHashMap" in analysis["suggestion"]
    
    def test_analyze_deadlock_error(self):
        """Test analysis of deadlock detection."""
        error_data = {
            "error_type": "java.lang.IllegalMonitorStateException",
            "message": "Deadlock detected",
            "stack_trace": [{"class": "Thread", "function": "wait"}]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "deadlock" in analysis["root_cause"] or "monitor" in analysis["root_cause"]
        assert analysis["category"] == "concurrency"
        assert analysis["severity"] == "critical"
    
    def test_analyze_spring_bean_error(self):
        """Test analysis of Spring bean creation error."""
        error_data = {
            "error_type": "org.springframework.beans.factory.BeanCreationException",
            "message": "Error creating bean with name 'dataSource'",
            "stack_trace": [{"class": "AbstractAutowireCapableBeanFactory", "function": "createBean"}]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "spring" in analysis["root_cause"]
        assert analysis["framework"] == "spring"
        assert "bean" in analysis["tags"]
        assert "configuration" in analysis["suggestion"].lower()
    
    def test_analyze_hibernate_lazy_loading(self):
        """Test analysis of Hibernate lazy loading error."""
        error_data = {
            "error_type": "org.hibernate.LazyInitializationException",
            "message": "could not initialize proxy - no Session",
            "stack_trace": [{"class": "AbstractPersistentCollection", "function": "initialize"}]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "java_hibernate_lazy_loading"
        assert analysis["framework"] == "hibernate"
        assert analysis["category"] == "framework"
        assert "eager" in analysis["suggestion"].lower() or "session" in analysis["suggestion"].lower()
    
    def test_analyze_memory_error(self):
        """Test analysis of OutOfMemoryError."""
        error_data = {
            "error_type": "java.lang.OutOfMemoryError",
            "message": "Java heap space",
            "stack_trace": []
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "java_out_of_memory"
        assert analysis["category"] == "resources"
        assert analysis["severity"] == "critical"
        assert "-Xmx" in analysis["suggestion"]
    
    def test_analyze_compilation_error(self):
        """Test analysis of compilation error."""
        error_data = {
            "error_type": "CompilationError",
            "message": "cannot find symbol",
            "file": "Calculator.java",
            "line": 25
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "compilation"
        assert "import" in analysis["suggestion"] or "declare" in analysis["suggestion"]
    
    def test_fallback_analysis(self):
        """Test fallback analysis for unknown errors."""
        error_data = {
            "error_type": "com.custom.UnknownException",
            "message": "Something went wrong",
            "stack_trace": []
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["rule_id"] == "java_generic_fallback"
        assert analysis["root_cause"] == "java_unknown_error"
        assert analysis["confidence"] == "low"


class TestJavaPatchGenerator:
    """Test cases for Java patch generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = JavaPatchGenerator()
    
    def test_generate_null_check_patch(self):
        """Test patch generation for null pointer issues."""
        analysis = {
            "rule_id": "java_null_pointer",
            "root_cause": "java_null_pointer",
            "error_data": {
                "error_type": "java.lang.NullPointerException",
                "message": "Cannot invoke \"String.length()\" because \"str\" is null"
            },
            "suggestion": "Add null check before accessing object",
            "confidence": "high"
        }
        
        context = {
            "code_snippet": "int length = str.length();",
            "variable_name": "str"
        }
        
        patch = self.generator.generate_patch(analysis, context)
        
        assert patch["language"] == "java"
        assert patch["root_cause"] == "java_null_pointer"
        assert "suggestion_code" in patch
        assert "if (str != null)" in patch["suggestion_code"] or "Optional" in patch["suggestion_code"]
    
    def test_generate_concurrent_collection_patch(self):
        """Test patch generation for concurrent modification."""
        analysis = {
            "rule_id": "java_concurrent_modification",
            "root_cause": "java_concurrent_modification",
            "error_data": {
                "error_type": "java.util.ConcurrentModificationException"
            },
            "suggestion": "Use Iterator.remove() or concurrent collection",
            "confidence": "high"
        }
        
        context = {
            "code_snippet": "for (String item : list) { list.remove(item); }",
            "collection_type": "ArrayList"
        }
        
        patch = self.generator.generate_patch(analysis, context)
        
        assert patch["language"] == "java"
        assert "Iterator" in patch["suggestion_code"] or "ConcurrentHashMap" in patch["suggestion_code"]
    
    def test_generate_resource_leak_patch(self):
        """Test patch generation for resource leak issues."""
        analysis = {
            "rule_id": "java_resource_leak",
            "root_cause": "java_resource_leak",
            "error_data": {
                "error_type": "ResourceLeak",
                "message": "Resource 'reader' is not closed"
            },
            "suggestion": "Use try-with-resources",
            "confidence": "high"
        }
        
        context = {
            "code_snippet": "BufferedReader reader = new BufferedReader(new FileReader(file));",
            "resource_type": "BufferedReader"
        }
        
        patch = self.generator.generate_patch(analysis, context)
        
        assert patch["language"] == "java"
        assert "try (" in patch["suggestion_code"]
        assert "AutoCloseable" in patch["description"] or "resource" in patch["description"]


class TestJavaLanguagePlugin:
    """Test cases for the main Java language plugin."""
    
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
    
    def test_can_handle_java_errors(self, java_plugin):
        """Test plugin can handle Java errors."""
        # Test with explicit language
        error_data = {"language": "java", "error_type": "java.lang.Exception"}
        assert java_plugin.can_handle(error_data) is True
        
        # Test with Java error types
        java_errors = [
            {"error_type": "java.lang.NullPointerException", "message": "null"},
            {"error_type": "java.util.ConcurrentModificationException"},
            {"error_type": "org.springframework.beans.factory.BeanCreationException"},
            {"file": "Main.java", "error_type": "CompilationError"}
        ]
        
        for error in java_errors:
            assert java_plugin.can_handle(error) is True
    
    def test_language_info(self, java_plugin):
        """Test language info methods."""
        assert java_plugin.get_language_id() == "java"
        assert java_plugin.get_language_name() == "Java"
        assert java_plugin.get_language_version() == "8+"
        
        frameworks = java_plugin.get_supported_frameworks()
        assert "spring" in frameworks
        assert "spring-boot" in frameworks
        assert "hibernate" in frameworks
        assert "struts" in frameworks
        assert "play" in frameworks


class TestJavaStreamAndLambdaErrors:
    """Test cases for Java 8+ Stream and Lambda specific errors."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = JavaExceptionHandler()
        self.plugin = JavaLanguagePlugin()
    
    def test_analyze_stream_null_pointer(self):
        """Test analysis of null pointer in stream operations."""
        error_data = {
            "error_type": "java.lang.NullPointerException",
            "message": "Cannot read field \"name\" because \"<parameter1>\" is null",
            "stack_trace": [
                {"class": "Main", "function": "lambda$main$0", "line": 45},
                {"class": "Stream", "function": "map"}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "java_null_pointer"
        assert "stream" in analysis["tags"] or "lambda" in analysis["tags"]
    
    def test_analyze_illegal_state_in_stream(self):
        """Test analysis of illegal state in stream terminal operations."""
        error_data = {
            "error_type": "java.lang.IllegalStateException",
            "message": "stream has already been operated upon or closed",
            "stack_trace": [{"class": "AbstractPipeline", "function": "evaluate"}]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "stream" in analysis["root_cause"] or analysis["category"] == "runtime"
        assert "reuse" in analysis["suggestion"].lower()


class TestJavaGenericsAndReflectionErrors:
    """Test cases for Java generics and reflection errors."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = JavaExceptionHandler()
        self.generator = JavaPatchGenerator()
    
    def test_analyze_generic_array_creation(self):
        """Test analysis of generic array creation error."""
        error_data = {
            "error_type": "java.lang.ClassCastException",
            "message": "[Ljava.lang.Object; cannot be cast to [Ljava.lang.String;",
            "stack_trace": []
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "java_class_cast"
        assert "array" in analysis["suggestion"].lower() or "generic" in analysis["suggestion"].lower()
    
    def test_analyze_reflection_access_error(self):
        """Test analysis of reflection access errors."""
        error_data = {
            "error_type": "java.lang.IllegalAccessException",
            "message": "class Main cannot access a member of class SecureClass with modifiers \"private\"",
            "stack_trace": [{"class": "Method", "function": "invoke"}]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "security" or analysis["category"] == "runtime"
        assert "access" in analysis["suggestion"].lower()
    
    def test_analyze_instantiation_error(self):
        """Test analysis of instantiation errors."""
        error_data = {
            "error_type": "java.lang.InstantiationException",
            "message": "com.example.AbstractClass",
            "stack_trace": [{"class": "Class", "function": "newInstance"}]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert "abstract" in analysis["suggestion"].lower() or "instantiate" in analysis["suggestion"].lower()


class TestJavaPerformanceAndSecurityErrors:
    """Test cases for Java performance and security related errors."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = JavaExceptionHandler()
        self.plugin = JavaLanguagePlugin()
    
    def test_analyze_stack_overflow(self):
        """Test analysis of stack overflow errors."""
        error_data = {
            "error_type": "java.lang.StackOverflowError",
            "message": "",
            "stack_trace": [
                {"class": "RecursiveClass", "function": "recurse", "line": 10},
                {"class": "RecursiveClass", "function": "recurse", "line": 10},
                {"class": "RecursiveClass", "function": "recurse", "line": 10}
            ]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["root_cause"] == "java_stack_overflow"
        assert analysis["category"] == "resources"
        assert analysis["severity"] == "critical"
        assert "recursion" in analysis["suggestion"].lower()
    
    def test_analyze_security_exception(self):
        """Test analysis of security exceptions."""
        error_data = {
            "error_type": "java.lang.SecurityException",
            "message": "access denied (\"java.io.FilePermission\" \"/etc/passwd\" \"read\")",
            "stack_trace": [{"class": "SecurityManager", "function": "checkPermission"}]
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "security"
        assert analysis["severity"] == "high"
        assert "permission" in analysis["suggestion"].lower()


class TestJavaEdgeCases:
    """Test cases for Java edge cases and corner cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = JavaErrorAdapter()
        self.handler = JavaExceptionHandler()
    
    def test_parse_nested_exception(self):
        """Test parsing of nested exceptions."""
        java_error = {
            "exception": "java.lang.RuntimeException: Processing failed",
            "stacktrace": ["at com.example.Processor.process(Processor.java:50)"],
            "caused_by": {
                "exception": "java.sql.SQLException: Connection timeout",
                "stacktrace": ["at com.example.Database.connect(Database.java:30)"],
                "caused_by": {
                    "exception": "java.net.SocketTimeoutException: Read timed out",
                    "stacktrace": ["at java.net.SocketInputStream.read(SocketInputStream.java:150)"]
                }
            }
        }
        
        standard_error = self.adapter.to_standard_format(java_error)
        
        assert standard_error["error_type"] == "java.lang.RuntimeException"
        assert "caused_by" in standard_error
        assert standard_error["caused_by"]["error_type"] == "java.sql.SQLException"
        assert "caused_by" in standard_error["caused_by"]
    
    def test_parse_suppressed_exceptions(self):
        """Test parsing of suppressed exceptions (try-with-resources)."""
        java_error = {
            "exception": "java.lang.Exception: Primary exception",
            "stacktrace": ["at com.example.Main.main(Main.java:20)"],
            "suppressed": [
                {
                    "exception": "java.io.IOException: Failed to close resource",
                    "stacktrace": ["at java.io.FileInputStream.close(FileInputStream.java:300)"]
                }
            ]
        }
        
        standard_error = self.adapter.to_standard_format(java_error)
        
        assert "suppressed" in standard_error
        assert len(standard_error["suppressed"]) == 1
        assert standard_error["suppressed"][0]["error_type"] == "java.io.IOException"
    
    def test_parse_error_with_no_stack_trace(self):
        """Test parsing of errors without stack trace."""
        java_error = {
            "exception": "java.lang.OutOfMemoryError: Metaspace",
            "message": "Metaspace"
        }
        
        standard_error = self.adapter.to_standard_format(java_error)
        
        assert standard_error["error_type"] == "java.lang.OutOfMemoryError"
        assert standard_error["message"] == "Metaspace"
        assert standard_error["stack_trace"] == []
    
    def test_parse_compilation_error_with_multiple_issues(self):
        """Test parsing of compilation errors with multiple issues."""
        java_error = {
            "type": "CompilationError",
            "errors": [
                {
                    "file": "Main.java",
                    "line": 10,
                    "message": "';' expected"
                },
                {
                    "file": "Main.java",
                    "line": 15,
                    "message": "cannot find symbol"
                }
            ]
        }
        
        standard_error = self.adapter.to_standard_format(java_error)
        
        assert standard_error["error_type"] == "CompilationError"
        assert "errors" in standard_error


class TestJavaIntegration:
    """Integration tests for Java plugin with real-world scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = JavaLanguagePlugin()
    
    def test_end_to_end_null_pointer_fix(self):
        """Test end-to-end flow for fixing a null pointer exception."""
        # Simulate a real error
        error_data = {
            "language": "java",
            "error_type": "java.lang.NullPointerException",
            "message": "Cannot invoke \"User.getName()\" because \"user\" is null",
            "file": "UserService.java",
            "line": 25,
            "stack_trace": [
                {
                    "file": "UserService.java",
                    "line": 25,
                    "class": "UserService",
                    "function": "processUser",
                    "package": "com.example.service"
                }
            ]
        }
        
        # Analyze the error
        analysis = self.plugin.analyze_error(error_data)
        assert analysis is not None
        assert analysis["root_cause"] == "java_null_pointer"
        
        # Generate a fix
        context = {
            "code_snippet": "String userName = user.getName();",
            "method_signature": "public void processUser(User user)"
        }
        
        fix = self.plugin.generate_fix(analysis, context)
        assert fix is not None
        assert fix["language"] == "java"
        assert "suggestion" in fix or "patch_code" in fix
    
    def test_end_to_end_spring_configuration_error(self):
        """Test end-to-end flow for Spring configuration errors."""
        error_data = {
            "language": "java",
            "error_type": "org.springframework.beans.factory.NoSuchBeanDefinitionException",
            "message": "No qualifying bean of type 'com.example.repository.UserRepository' available",
            "stack_trace": [
                {
                    "class": "DefaultListableBeanFactory",
                    "function": "getBean",
                    "package": "org.springframework.beans.factory.support"
                }
            ]
        }
        
        analysis = self.plugin.analyze_error(error_data)
        assert analysis is not None
        assert "spring" in analysis["root_cause"]
        assert analysis["framework"] == "spring"
        
        # Check that suggestions are Spring-specific
        assert "@Component" in analysis["suggestion"] or "@Repository" in analysis["suggestion"]
    
    def test_framework_detection_in_error_analysis(self):
        """Test that framework-specific errors are properly detected and tagged."""
        test_cases = [
            {
                "error_type": "org.hibernate.HibernateException",
                "expected_framework": "hibernate"
            },
            {
                "error_type": "org.springframework.web.servlet.NoHandlerFoundException",
                "expected_framework": "spring"
            },
            {
                "error_type": "play.api.PlayException",
                "expected_framework": "play"
            },
            {
                "error_type": "org.apache.struts2.StrutsException",
                "expected_framework": "struts"
            }
        ]
        
        for test_case in test_cases:
            error_data = {
                "error_type": test_case["error_type"],
                "message": "Test error",
                "stack_trace": []
            }
            
            analysis = self.plugin.analyze_error(error_data)
            assert analysis["framework"] == test_case["expected_framework"]


class TestJavaPatchValidation:
    """Test cases for validating generated Java patches."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = JavaPatchGenerator()
    
    def test_patch_preserves_code_style(self):
        """Test that generated patches preserve code style."""
        analysis = {
            "rule_id": "java_null_pointer",
            "root_cause": "java_null_pointer",
            "error_data": {"error_type": "java.lang.NullPointerException"},
            "suggestion": "Add null check",
            "confidence": "high"
        }
        
        # Test with different code styles
        contexts = [
            {
                "code_snippet": "String s=obj.toString();",  # No spaces
                "style": "compact"
            },
            {
                "code_snippet": "String s = obj.toString();",  # With spaces
                "style": "standard"
            }
        ]
        
        for context in contexts:
            patch = self.generator.generate_patch(analysis, context)
            assert patch is not None
            # Patches should adapt to the code style
    
    def test_patch_handles_complex_expressions(self):
        """Test patch generation for complex expressions."""
        analysis = {
            "rule_id": "java_null_pointer",
            "root_cause": "java_null_pointer",
            "error_data": {"error_type": "java.lang.NullPointerException"},
            "suggestion": "Add null check",
            "confidence": "high"
        }
        
        context = {
            "code_snippet": "int count = user.getOrders().stream().filter(o -> o.isActive()).count();",
            "variable_name": "user"
        }
        
        patch = self.generator.generate_patch(analysis, context)
        assert patch is not None
        assert "suggestion_code" in patch
        # Should handle the complex chained expression


if __name__ == "__main__":
    pytest.main([__file__, "-v"])