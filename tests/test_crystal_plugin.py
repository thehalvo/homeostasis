"""
Tests for the Crystal language plugin.
"""
import pytest

from modules.analysis.plugins.crystal_plugin import (
    CrystalLanguagePlugin, 
    CrystalExceptionHandler, 
    CrystalPatchGenerator
)


class TestCrystalExceptionHandler:
    """Test the Crystal exception handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = CrystalExceptionHandler()
    
    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "CrystalError",
            "message": "syntax error in line 10: unexpected token 'end', expecting ';'",
            "file_path": "test.cr",
            "line_number": 10,
            "column_number": 5
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "crystal"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]
    
    def test_analyze_type_error(self):
        """Test analysis of type errors."""
        error_data = {
            "error_type": "CrystalError",
            "message": "no overload matches 'String#+' with type Int32",
            "file_path": "test.cr",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "crystal"
        assert analysis["subcategory"] == "type"
        assert analysis["confidence"] == "high"
        assert "type" in analysis["tags"]
    
    def test_analyze_nil_error(self):
        """Test analysis of nil errors."""
        error_data = {
            "error_type": "CrystalError",
            "message": "null reference: tried to call method on nil",
            "file_path": "test.cr",
            "line_number": 20,
            "column_number": 12
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "crystal"
        assert analysis["subcategory"] == "nil"
        assert analysis["confidence"] == "high"
        assert "nil" in analysis["tags"]
    
    def test_analyze_compilation_error(self):
        """Test analysis of compilation errors."""
        error_data = {
            "error_type": "CrystalError",
            "message": "undefined constant MyConstant",
            "file_path": "test.cr",
            "line_number": 25,
            "column_number": 15
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "crystal"
        assert analysis["subcategory"] == "compilation"
        assert analysis["confidence"] == "high"
        assert "compilation" in analysis["tags"]
    
    def test_analyze_fiber_error(self):
        """Test analysis of fiber errors."""
        error_data = {
            "error_type": "CrystalError",
            "message": "fiber error: deadlock detected",
            "file_path": "test.cr",
            "line_number": 30,
            "column_number": 10
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "crystal"
        assert analysis["subcategory"] == "fiber"
        assert analysis["confidence"] == "high"
        assert "fiber" in analysis["tags"]
    
    def test_analyze_method_error(self):
        """Test analysis of method errors."""
        error_data = {
            "error_type": "CrystalError",
            "message": "undefined method 'unknown_method' for String",
            "file_path": "test.cr",
            "line_number": 35,
            "column_number": 8
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "crystal"
        assert analysis["subcategory"] == "method"
        assert analysis["confidence"] == "high"
        assert "method" in analysis["tags"]
    
    def test_analyze_union_error(self):
        """Test analysis of union type errors."""
        error_data = {
            "error_type": "CrystalError",
            "message": "undefined method for union type String | Int32",
            "file_path": "test.cr",
            "line_number": 40,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "crystal"
        assert analysis["subcategory"] == "union"
        assert analysis["confidence"] == "high"
        assert "union" in analysis["tags"]
    
    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "CrystalError",
            "message": "Some unknown error message",
            "file_path": "test.cr",
            "line_number": 45,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "crystal"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestCrystalPatchGenerator:
    """Test the Crystal patch generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = CrystalPatchGenerator()
    
    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {
            "message": "syntax error: unexpected token 'end', expecting ';'",
            "file_path": "test.cr"
        }
        
        analysis = {
            "root_cause": "crystal_syntax_error",
            "subcategory": "syntax",
            "confidence": "high"
        }
        
        syntax_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert syntax_patch is not None
        assert syntax_patch["type"] in ["suggestion", "multiple_suggestions"]
        assert "syntax" in syntax_patch["description"].lower()
    
    def test_generate_type_fix(self):
        """Test generation of type fixes."""
        error_data = {
            "message": "no overload matches 'String#+' with type Int32",
            "file_path": "test.cr"
        }
        
        analysis = {
            "root_cause": "crystal_type_error",
            "subcategory": "type",
            "confidence": "high"
        }
        
        type_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert type_patch is not None
        assert type_patch["type"] == "suggestion"
        assert "type" in type_patch["description"].lower() or "overload" in type_patch["description"].lower()
    
    def test_generate_nil_fix(self):
        """Test generation of nil fixes."""
        error_data = {
            "message": "null reference: tried to call method on nil",
            "file_path": "test.cr"
        }
        
        analysis = {
            "root_cause": "crystal_nil_error",
            "subcategory": "nil",
            "confidence": "high"
        }
        
        nil_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert nil_patch is not None
        assert nil_patch["type"] == "suggestion"
        assert "nil" in nil_patch["description"].lower()
    
    def test_generate_compilation_fix(self):
        """Test generation of compilation fixes."""
        error_data = {
            "message": "undefined constant MyConstant",
            "file_path": "test.cr"
        }
        
        analysis = {
            "root_cause": "crystal_compilation_error",
            "subcategory": "compilation",
            "confidence": "high"
        }
        
        compilation_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert compilation_patch is not None
        assert compilation_patch["type"] == "suggestion"
        assert "constant" in compilation_patch["description"].lower()
    
    def test_generate_fiber_fix(self):
        """Test generation of fiber fixes."""
        error_data = {
            "message": "fiber error: deadlock detected",
            "file_path": "test.cr"
        }
        
        analysis = {
            "root_cause": "crystal_fiber_error",
            "subcategory": "fiber",
            "confidence": "high"
        }
        
        fiber_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert fiber_patch is not None
        assert fiber_patch["type"] == "suggestion"
        assert "fiber" in fiber_patch["description"].lower()
    
    def test_generate_method_fix(self):
        """Test generation of method fixes."""
        error_data = {
            "message": "wrong number of arguments (given 1, expected 2)",
            "file_path": "test.cr"
        }
        
        analysis = {
            "root_cause": "crystal_method_error",
            "subcategory": "method",
            "confidence": "high"
        }
        
        method_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert method_patch is not None
        assert method_patch["type"] == "suggestion"
        assert "arguments" in method_patch["description"].lower()
    
    def test_generate_union_fix(self):
        """Test generation of union type fixes."""
        error_data = {
            "message": "undefined method for union type String | Int32",
            "file_path": "test.cr"
        }
        
        analysis = {
            "root_cause": "crystal_union_error",
            "subcategory": "union",
            "confidence": "high"
        }
        
        union_patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert union_patch is not None
        assert union_patch["type"] == "suggestion"
        assert "union" in union_patch["description"].lower()


class TestCrystalLanguagePlugin:
    """Test the Crystal language plugin."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = CrystalLanguagePlugin()
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "crystal"
        assert self.plugin.get_language_name() == "Crystal"
        assert self.plugin.get_language_version() == "1.0+"
        
        frameworks = self.plugin.get_supported_frameworks()
        assert "crystal" in frameworks
        assert "shards" in frameworks
        assert "kemal" in frameworks
    
    def test_normalize_error(self):
        """Test error normalization."""
        crystal_error = {
            "error_type": "CrystalError",
            "message": "Test error",
            "file": "test.cr",
            "line": 10,
            "column": 5,
            "description": "Test error description"
        }
        
        normalized = self.plugin.normalize_error(crystal_error)
        
        assert normalized["language"] == "crystal"
        assert normalized["error_type"] == "CrystalError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "test.cr"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5
    
    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "crystal",
            "error_type": "CrystalError",
            "message": "Test error",
            "file_path": "test.cr",
            "line_number": 10,
            "column_number": 5,
            "severity": "high"
        }
        
        crystal_error = self.plugin.denormalize_error(standard_error)
        
        assert crystal_error["error_type"] == "CrystalError"
        assert crystal_error["message"] == "Test error"
        assert crystal_error["file_path"] == "test.cr"
        assert crystal_error["line_number"] == 10
        assert crystal_error["column_number"] == 5
        assert crystal_error["file"] == "test.cr"  # Alternative format
        assert crystal_error["line"] == 10  # Alternative format
    
    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "CrystalError",
            "message": "no overload matches 'String#+' with type Int32",
            "file_path": "test.cr",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "crystal"
        assert analysis["language"] == "crystal"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "crystal"
        assert analysis["subcategory"] == "type"
    
    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "crystal_type_error",
            "subcategory": "type",
            "confidence": "high",
            "suggested_fix": "Fix type mismatch"
        }
        
        context = {
            "error_data": {
                "message": "no overload matches 'String#+' with type Int32",
                "file_path": "test.cr"
            },
            "source_code": "\"hello\" + 42"
        }
        
        fix = self.plugin.generate_fix(analysis, context)
        
        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".cr" in self.plugin.supported_extensions
    
    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "crystal"
        assert analysis["language"] == "crystal"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])