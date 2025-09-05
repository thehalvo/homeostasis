"""
Tests for the Lua language plugin.
"""
import pytest

from modules.analysis.plugins.lua_plugin import (
    LuaLanguagePlugin, 
    LuaExceptionHandler, 
    LuaPatchGenerator
)


class TestLuaExceptionHandler:
    """Test the Lua exception handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = LuaExceptionHandler()
    
    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "LuaError",
            "message": "unexpected symbol near ')'",
            "file_path": "test.lua",
            "line_number": 10,
            "column_number": 5
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "lua"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]
    
    def test_analyze_nil_error(self):
        """Test analysis of nil errors."""
        error_data = {
            "error_type": "LuaError",
            "message": "attempt to index a nil value",
            "file_path": "test.lua",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "lua"
        assert analysis["subcategory"] == "nil"
        assert analysis["confidence"] == "high"
        assert "nil" in analysis["tags"]
    
    def test_analyze_type_error(self):
        """Test analysis of type errors."""
        error_data = {
            "error_type": "LuaError",
            "message": "attempt to concatenate a nil value",
            "file_path": "test.lua",
            "line_number": 20,
            "column_number": 12
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "lua"
        assert analysis["subcategory"] == "type"
        assert analysis["confidence"] == "high"
        assert "type" in analysis["tags"]
    
    def test_analyze_function_error(self):
        """Test analysis of function errors."""
        error_data = {
            "error_type": "LuaError",
            "message": "attempt to call global 'myFunc' (a nil value)",
            "file_path": "test.lua",
            "line_number": 25,
            "column_number": 15
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "lua"
        assert analysis["subcategory"] == "function"
        assert analysis["confidence"] == "high"
        assert "function" in analysis["tags"]
    
    def test_analyze_table_error(self):
        """Test analysis of table errors."""
        error_data = {
            "error_type": "LuaError",
            "message": "table index is nil",
            "file_path": "test.lua",
            "line_number": 30,
            "column_number": 10
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "lua"
        assert analysis["subcategory"] == "table"
        assert analysis["confidence"] == "high"
        assert "table" in analysis["tags"]
    
    def test_analyze_module_error(self):
        """Test analysis of module errors."""
        error_data = {
            "error_type": "LuaError",
            "message": "module 'mymodule' not found",
            "file_path": "test.lua",
            "line_number": 1,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "lua"
        assert analysis["subcategory"] == "module"
        assert analysis["confidence"] == "high"
        assert "module" in analysis["tags"]
    
    def test_analyze_arithmetic_error(self):
        """Test analysis of arithmetic errors."""
        error_data = {
            "error_type": "LuaError",
            "message": "attempt to perform arithmetic on a string value",
            "file_path": "test.lua",
            "line_number": 35,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "lua"
        assert analysis["subcategory"] == "arithmetic"
        assert analysis["confidence"] == "high"
        assert "arithmetic" in analysis["tags"]
    
    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "LuaError",
            "message": "Some unknown error message",
            "file_path": "test.lua",
            "line_number": 45,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "lua"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestLuaPatchGenerator:
    """Test the Lua patch generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = LuaPatchGenerator()
    
    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {
            "message": "unexpected symbol near ')'",
            "file_path": "test.lua"
        }
        
        analysis = {
            "root_cause": "lua_syntax_error",
            "subcategory": "syntax",
            "confidence": "high"
        }
        
        syntax_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert syntax_patch is not None
        assert syntax_patch["type"] == "suggestion"
        assert "syntax" in syntax_patch["description"].lower()
    
    def test_generate_nil_fix(self):
        """Test generation of nil fixes."""
        error_data = {
            "message": "attempt to index a nil value",
            "file_path": "test.lua"
        }
        
        analysis = {
            "root_cause": "lua_nil_error",
            "subcategory": "nil",
            "confidence": "high"
        }
        
        nil_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert nil_patch is not None
        assert nil_patch["type"] == "suggestion"
        assert "nil" in nil_patch["description"].lower()
    
    def test_generate_type_fix(self):
        """Test generation of type fixes."""
        error_data = {
            "message": "attempt to concatenate a nil value",
            "file_path": "test.lua"
        }
        
        analysis = {
            "root_cause": "lua_type_error",
            "subcategory": "type",
            "confidence": "high"
        }
        
        type_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert type_patch is not None
        assert type_patch["type"] == "suggestion"
        assert "type" in type_patch["description"].lower() or "concatenate" in type_patch["description"].lower()
    
    def test_generate_function_fix(self):
        """Test generation of function fixes."""
        error_data = {
            "message": "attempt to call global 'myFunc' (a nil value)",
            "file_path": "test.lua"
        }
        
        analysis = {
            "root_cause": "lua_function_error",
            "subcategory": "function",
            "confidence": "high"
        }
        
        function_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert function_patch is not None
        assert function_patch["type"] == "suggestion"
        assert "function" in function_patch["description"].lower()
    
    def test_generate_table_fix(self):
        """Test generation of table fixes."""
        error_data = {
            "message": "table index is nil",
            "file_path": "test.lua"
        }
        
        analysis = {
            "root_cause": "lua_table_error",
            "subcategory": "table",
            "confidence": "high"
        }
        
        table_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert table_patch is not None
        assert table_patch["type"] == "suggestion"
        assert "table" in table_patch["description"].lower() or "index" in table_patch["description"].lower()
    
    def test_generate_module_fix(self):
        """Test generation of module fixes."""
        error_data = {
            "message": "module 'mymodule' not found",
            "file_path": "test.lua"
        }
        
        analysis = {
            "root_cause": "lua_module_error",
            "subcategory": "module",
            "confidence": "high"
        }
        
        module_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert module_patch is not None
        assert module_patch["type"] == "suggestion"
        assert "module" in module_patch["description"].lower() or "require" in module_patch["description"].lower()
    
    def test_generate_arithmetic_fix(self):
        """Test generation of arithmetic fixes."""
        error_data = {
            "message": "attempt to perform arithmetic on a string value",
            "file_path": "test.lua"
        }
        
        analysis = {
            "root_cause": "lua_arithmetic_error",
            "subcategory": "arithmetic",
            "confidence": "high"
        }
        
        arithmetic_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert arithmetic_patch is not None
        assert arithmetic_patch["type"] == "suggestion"
        assert "arithmetic" in arithmetic_patch["description"].lower() or "tonumber" in arithmetic_patch["description"].lower()


class TestLuaLanguagePlugin:
    """Test the Lua language plugin."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = LuaLanguagePlugin()
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "lua"
        assert self.plugin.get_language_name() == "Lua"
        assert self.plugin.get_language_version() == "5.4+"
        
        frameworks = self.plugin.get_supported_frameworks()
        assert "lua" in frameworks
        assert "luarocks" in frameworks
    
    def test_normalize_error(self):
        """Test error normalization."""
        lua_error = {
            "error_type": "LuaError",
            "message": "Test error",
            "file": "test.lua",
            "line": 10,
            "column": 5,
            "description": "Test error description"
        }
        
        normalized = self.plugin.normalize_error(lua_error)
        
        assert normalized["language"] == "lua"
        assert normalized["error_type"] == "LuaError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "test.lua"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5
    
    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "lua",
            "error_type": "LuaError",
            "message": "Test error",
            "file_path": "test.lua",
            "line_number": 10,
            "column_number": 5,
            "severity": "high"
        }
        
        lua_error = self.plugin.denormalize_error(standard_error)
        
        assert lua_error["error_type"] == "LuaError"
        assert lua_error["message"] == "Test error"
        assert lua_error["file_path"] == "test.lua"
        assert lua_error["line_number"] == 10
        assert lua_error["column_number"] == 5
        assert lua_error["file"] == "test.lua"  # Alternative format
        assert lua_error["line"] == 10  # Alternative format
    
    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "LuaError",
            "message": "attempt to index a nil value",
            "file_path": "test.lua",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "lua"
        assert analysis["language"] == "lua"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "lua"
        assert analysis["subcategory"] == "nil"
    
    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "lua_nil_error",
            "subcategory": "nil",
            "confidence": "high",
            "suggested_fix": "Fix nil value access"
        }
        
        context = {
            "error_data": {
                "message": "attempt to index a nil value",
                "file_path": "test.lua"
            },
            "source_code": "local x = nil\nprint(x.value)"
        }
        
        fix = self.plugin.generate_fix(analysis, context)
        
        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".lua" in self.plugin.supported_extensions
    
    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "lua"
        assert analysis["language"] == "lua"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])