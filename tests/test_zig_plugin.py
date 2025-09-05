"""
Tests for the Zig language plugin.
"""
import pytest

from modules.analysis.plugins.zig_plugin import (
    ZigLanguagePlugin, 
    ZigExceptionHandler, 
    ZigPatchGenerator
)


class TestZigExceptionHandler:
    """Test the Zig exception handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = ZigExceptionHandler()
    
    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "ZigError",
            "message": "error: expected ';', found 'const'",
            "file_path": "test.zig",
            "line_number": 10,
            "column_number": 5
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "zig"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]
    
    def test_analyze_type_error(self):
        """Test analysis of type errors."""
        error_data = {
            "error_type": "ZigError",
            "message": "error: expected type 'u32', found 'i32'",
            "file_path": "test.zig",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "zig"
        assert analysis["subcategory"] == "type"
        assert analysis["confidence"] == "high"
        assert "type" in analysis["tags"]
    
    def test_analyze_memory_error(self):
        """Test analysis of memory errors."""
        error_data = {
            "error_type": "ZigError",
            "message": "error: null pointer dereference",
            "file_path": "test.zig",
            "line_number": 20,
            "column_number": 12
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "zig"
        assert analysis["subcategory"] == "memory"
        assert analysis["confidence"] == "high"
        assert "memory" in analysis["tags"]
    
    def test_analyze_undefined_error(self):
        """Test analysis of undefined identifier errors."""
        error_data = {
            "error_type": "ZigError",
            "message": "error: use of undeclared identifier 'myFunc'",
            "file_path": "test.zig",
            "line_number": 25,
            "column_number": 15
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "zig"
        assert analysis["subcategory"] == "undefined"
        assert analysis["confidence"] == "high"
        assert "undefined" in analysis["tags"]
    
    def test_analyze_comptime_error(self):
        """Test analysis of compile-time errors."""
        error_data = {
            "error_type": "ZigError",
            "message": "error: unable to evaluate constant expression",
            "file_path": "test.zig",
            "line_number": 30,
            "column_number": 10
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "zig"
        assert analysis["subcategory"] == "comptime"
        assert analysis["confidence"] == "high"
        assert "comptime" in analysis["tags"]
    
    def test_analyze_import_error(self):
        """Test analysis of import errors."""
        error_data = {
            "error_type": "ZigError",
            "message": "error: unable to find 'std'",
            "file_path": "test.zig",
            "line_number": 1,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "zig"
        assert analysis["subcategory"] == "import"
        assert analysis["confidence"] == "high"
        assert "import" in analysis["tags"]
    
    def test_analyze_async_error(self):
        """Test analysis of async errors."""
        error_data = {
            "error_type": "ZigError",
            "message": "error: async function called without await",
            "file_path": "test.zig",
            "line_number": 40,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "zig"
        assert analysis["subcategory"] == "async"
        assert analysis["confidence"] == "high"
        assert "async" in analysis["tags"]
    
    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "ZigError",
            "message": "Some unknown error message",
            "file_path": "test.zig",
            "line_number": 45,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "zig"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestZigPatchGenerator:
    """Test the Zig patch generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ZigPatchGenerator()
    
    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {
            "message": "error: expected ';', found 'const'",
            "file_path": "test.zig"
        }
        
        analysis = {
            "root_cause": "zig_syntax_error",
            "subcategory": "syntax",
            "confidence": "high"
        }
        
        syntax_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert syntax_patch is not None
        assert syntax_patch["type"] == "suggestion"
        assert "semicolon" in syntax_patch["description"].lower()
    
    def test_generate_type_fix(self):
        """Test generation of type fixes."""
        error_data = {
            "message": "error: expected type 'u32', found 'i32'",
            "file_path": "test.zig"
        }
        
        analysis = {
            "root_cause": "zig_type_error",
            "subcategory": "type",
            "confidence": "high"
        }
        
        type_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert type_patch is not None
        assert type_patch["type"] == "suggestion"
        assert "type" in type_patch["description"].lower()
    
    def test_generate_memory_fix(self):
        """Test generation of memory fixes."""
        error_data = {
            "message": "error: null pointer dereference",
            "file_path": "test.zig"
        }
        
        analysis = {
            "root_cause": "zig_memory_error",
            "subcategory": "memory",
            "confidence": "high"
        }
        
        memory_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert memory_patch is not None
        assert memory_patch["type"] == "suggestion"
        assert "null" in memory_patch["description"].lower() or "pointer" in memory_patch["description"].lower()
    
    def test_generate_undefined_fix(self):
        """Test generation of undefined identifier fixes."""
        error_data = {
            "message": "error: use of undeclared identifier 'myFunc'",
            "file_path": "test.zig"
        }
        
        analysis = {
            "root_cause": "zig_undefined_error",
            "subcategory": "undefined",
            "confidence": "high"
        }
        
        undefined_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert undefined_patch is not None
        assert undefined_patch["type"] == "suggestion"
        assert "identifier" in undefined_patch["description"].lower() or "undefined" in undefined_patch["description"].lower()
    
    def test_generate_comptime_fix(self):
        """Test generation of compile-time fixes."""
        error_data = {
            "message": "error: unable to evaluate constant expression",
            "file_path": "test.zig"
        }
        
        analysis = {
            "root_cause": "zig_comptime_error",
            "subcategory": "comptime",
            "confidence": "high"
        }
        
        comptime_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert comptime_patch is not None
        assert comptime_patch["type"] == "suggestion"
        assert "comptime" in comptime_patch["description"].lower() or "constant" in comptime_patch["description"].lower()
    
    def test_generate_import_fix(self):
        """Test generation of import fixes."""
        error_data = {
            "message": "error: unable to find 'std'",
            "file_path": "test.zig"
        }
        
        analysis = {
            "root_cause": "zig_import_error",
            "subcategory": "import",
            "confidence": "high"
        }
        
        import_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert import_patch is not None
        assert import_patch["type"] == "suggestion"
        assert "import" in import_patch["description"].lower()
    
    def test_generate_async_fix(self):
        """Test generation of async fixes."""
        error_data = {
            "message": "error: async function called without await",
            "file_path": "test.zig"
        }
        
        analysis = {
            "root_cause": "zig_async_error",
            "subcategory": "async",
            "confidence": "high"
        }
        
        async_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert async_patch is not None
        assert async_patch["type"] == "suggestion"
        assert "async" in async_patch["description"].lower() or "await" in async_patch["description"].lower()


class TestZigLanguagePlugin:
    """Test the Zig language plugin."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = ZigLanguagePlugin()
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "zig"
        assert self.plugin.get_language_name() == "Zig"
        assert self.plugin.get_language_version() == "0.11+"
        
        frameworks = self.plugin.get_supported_frameworks()
        assert "zig" in frameworks
        assert "build.zig" in frameworks
    
    def test_normalize_error(self):
        """Test error normalization."""
        zig_error = {
            "error_type": "ZigError",
            "message": "Test error",
            "file": "test.zig",
            "line": 10,
            "column": 5,
            "description": "Test error description"
        }
        
        normalized = self.plugin.normalize_error(zig_error)
        
        assert normalized["language"] == "zig"
        assert normalized["error_type"] == "ZigError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "test.zig"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5
    
    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "zig",
            "error_type": "ZigError",
            "message": "Test error",
            "file_path": "test.zig",
            "line_number": 10,
            "column_number": 5,
            "severity": "high"
        }
        
        zig_error = self.plugin.denormalize_error(standard_error)
        
        assert zig_error["error_type"] == "ZigError"
        assert zig_error["message"] == "Test error"
        assert zig_error["file_path"] == "test.zig"
        assert zig_error["line_number"] == 10
        assert zig_error["column_number"] == 5
        assert zig_error["file"] == "test.zig"  # Alternative format
        assert zig_error["line"] == 10  # Alternative format
    
    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "ZigError",
            "message": "error: expected type 'u32', found 'i32'",
            "file_path": "test.zig",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "zig"
        assert analysis["language"] == "zig"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "zig"
        assert analysis["subcategory"] == "type"
    
    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "zig_type_error",
            "subcategory": "type",
            "confidence": "high",
            "suggested_fix": "Fix type mismatch"
        }
        
        context = {
            "error_data": {
                "message": "error: expected type 'u32', found 'i32'",
                "file_path": "test.zig"
            },
            "source_code": "const x: u32 = -42;"
        }
        
        fix = self.plugin.generate_fix(analysis, context)
        
        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".zig" in self.plugin.supported_extensions
    
    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "zig"
        assert analysis["language"] == "zig"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])