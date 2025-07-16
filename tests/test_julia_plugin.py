"""
Tests for the Julia language plugin.
"""
import pytest
import json
from unittest.mock import Mock, patch

from modules.analysis.plugins.julia_plugin import (
    JuliaLanguagePlugin, 
    JuliaExceptionHandler, 
    JuliaPatchGenerator
)


class TestJuliaExceptionHandler:
    """Test the Julia exception handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = JuliaExceptionHandler()
    
    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "JuliaError",
            "message": "syntax: unexpected end of input",
            "file_path": "test.jl",
            "line_number": 10,
            "column_number": 5
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "julia"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]
    
    def test_analyze_type_error(self):
        """Test analysis of type errors."""
        error_data = {
            "error_type": "JuliaError",
            "message": "MethodError: no method matching +(::String, ::Int64)",
            "file_path": "test.jl",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "julia"
        assert analysis["subcategory"] == "type"
        assert analysis["confidence"] == "high"
        assert "type" in analysis["tags"]
    
    def test_analyze_bounds_error(self):
        """Test analysis of bounds errors."""
        error_data = {
            "error_type": "JuliaError",
            "message": "BoundsError: attempt to access 5-element Array{Int64,1} at index [10]",
            "file_path": "test.jl",
            "line_number": 20,
            "column_number": 12
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "julia"
        assert analysis["subcategory"] == "bounds"
        assert analysis["confidence"] == "high"
        assert "bounds" in analysis["tags"]
    
    def test_analyze_undefined_error(self):
        """Test analysis of undefined variable errors."""
        error_data = {
            "error_type": "JuliaError",
            "message": "UndefVarError: myVar not defined",
            "file_path": "test.jl",
            "line_number": 25,
            "column_number": 15
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "julia"
        assert analysis["subcategory"] == "undefined"
        assert analysis["confidence"] == "high"
        assert "undefined" in analysis["tags"]
    
    def test_analyze_import_error(self):
        """Test analysis of import errors."""
        error_data = {
            "error_type": "JuliaError",
            "message": "ArgumentError: Package MyPackage not found",
            "file_path": "test.jl",
            "line_number": 1,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "julia"
        assert analysis["subcategory"] == "import"
        assert analysis["confidence"] == "high"
        assert "import" in analysis["tags"]
    
    def test_analyze_dispatch_error(self):
        """Test analysis of dispatch errors."""
        error_data = {
            "error_type": "JuliaError",
            "message": "MethodError: no method matching foo(::Float64)",
            "file_path": "test.jl",
            "line_number": 30,
            "column_number": 10
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "julia"
        assert analysis["subcategory"] == "dispatch"
        assert analysis["confidence"] == "high"
        assert "dispatch" in analysis["tags"]
    
    def test_analyze_macro_error(self):
        """Test analysis of macro errors."""
        error_data = {
            "error_type": "JuliaError",
            "message": "LoadError: UndefVarError: @mymacro not defined",
            "file_path": "test.jl",
            "line_number": 35,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "julia"
        assert analysis["subcategory"] == "macro"
        assert analysis["confidence"] == "high"
        assert "macro" in analysis["tags"]
    
    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "JuliaError",
            "message": "Some unknown error message",
            "file_path": "test.jl",
            "line_number": 45,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "julia"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestJuliaPatchGenerator:
    """Test the Julia patch generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = JuliaPatchGenerator()
    
    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {
            "message": "syntax: unexpected end of input",
            "file_path": "test.jl"
        }
        
        analysis = {
            "root_cause": "julia_syntax_error",
            "subcategory": "syntax",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "syntax" in patch["description"].lower()
    
    def test_generate_type_fix(self):
        """Test generation of type fixes."""
        error_data = {
            "message": "MethodError: no method matching +(::String, ::Int64)",
            "file_path": "test.jl"
        }
        
        analysis = {
            "root_cause": "julia_type_error",
            "subcategory": "type",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "type" in patch["description"].lower() or "method" in patch["description"].lower()
    
    def test_generate_bounds_fix(self):
        """Test generation of bounds fixes."""
        error_data = {
            "message": "BoundsError: attempt to access 5-element Array{Int64,1} at index [10]",
            "file_path": "test.jl"
        }
        
        analysis = {
            "root_cause": "julia_bounds_error",
            "subcategory": "bounds",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "bounds" in patch["description"].lower() or "index" in patch["description"].lower()
    
    def test_generate_undefined_fix(self):
        """Test generation of undefined variable fixes."""
        error_data = {
            "message": "UndefVarError: myVar not defined",
            "file_path": "test.jl"
        }
        
        analysis = {
            "root_cause": "julia_undefined_error",
            "subcategory": "undefined",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "undefined" in patch["description"].lower() or "variable" in patch["description"].lower()
    
    def test_generate_import_fix(self):
        """Test generation of import fixes."""
        error_data = {
            "message": "ArgumentError: Package MyPackage not found",
            "file_path": "test.jl"
        }
        
        analysis = {
            "root_cause": "julia_import_error",
            "subcategory": "import",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "package" in patch["description"].lower() or "import" in patch["description"].lower()
    
    def test_generate_dispatch_fix(self):
        """Test generation of dispatch fixes."""
        error_data = {
            "message": "MethodError: no method matching foo(::Float64)",
            "file_path": "test.jl"
        }
        
        analysis = {
            "root_cause": "julia_dispatch_error",
            "subcategory": "dispatch",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "method" in patch["description"].lower() or "dispatch" in patch["description"].lower()
    
    def test_generate_macro_fix(self):
        """Test generation of macro fixes."""
        error_data = {
            "message": "LoadError: UndefVarError: @mymacro not defined",
            "file_path": "test.jl"
        }
        
        analysis = {
            "root_cause": "julia_macro_error",
            "subcategory": "macro",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "macro" in patch["description"].lower()


class TestJuliaLanguagePlugin:
    """Test the Julia language plugin."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = JuliaLanguagePlugin()
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "julia"
        assert self.plugin.get_language_name() == "Julia"
        assert self.plugin.get_language_version() == "1.8+"
        
        frameworks = self.plugin.get_supported_frameworks()
        assert "julia" in frameworks
        assert "pkg" in frameworks
    
    def test_normalize_error(self):
        """Test error normalization."""
        julia_error = {
            "error_type": "JuliaError",
            "message": "Test error",
            "file": "test.jl",
            "line": 10,
            "column": 5,
            "description": "Test error description"
        }
        
        normalized = self.plugin.normalize_error(julia_error)
        
        assert normalized["language"] == "julia"
        assert normalized["error_type"] == "JuliaError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "test.jl"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5
    
    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "julia",
            "error_type": "JuliaError",
            "message": "Test error",
            "file_path": "test.jl",
            "line_number": 10,
            "column_number": 5,
            "severity": "high"
        }
        
        julia_error = self.plugin.denormalize_error(standard_error)
        
        assert julia_error["error_type"] == "JuliaError"
        assert julia_error["message"] == "Test error"
        assert julia_error["file_path"] == "test.jl"
        assert julia_error["line_number"] == 10
        assert julia_error["column_number"] == 5
        assert julia_error["file"] == "test.jl"  # Alternative format
        assert julia_error["line"] == 10  # Alternative format
    
    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "JuliaError",
            "message": "MethodError: no method matching +(::String, ::Int64)",
            "file_path": "test.jl",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "julia"
        assert analysis["language"] == "julia"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "julia"
        assert analysis["subcategory"] == "type"
    
    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "julia_type_error",
            "subcategory": "type",
            "confidence": "high",
            "suggested_fix": "Fix type mismatch"
        }
        
        context = {
            "error_data": {
                "message": "MethodError: no method matching +(::String, ::Int64)",
                "file_path": "test.jl"
            },
            "source_code": "result = \"hello\" + 42"
        }
        
        fix = self.plugin.generate_fix(analysis, context)
        
        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".jl" in self.plugin.supported_extensions
    
    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "julia"
        assert analysis["language"] == "julia"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])