"""
Tests for the F# language plugin.
"""
import pytest
import json
from unittest.mock import Mock, patch

from modules.analysis.plugins.fsharp_plugin import (
    FSharpLanguagePlugin, 
    FSharpExceptionHandler, 
    FSharpPatchGenerator
)


class TestFSharpExceptionHandler:
    """Test the F# exception handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = FSharpExceptionHandler()
    
    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "FSharpError",
            "message": "Unexpected token 'let' in expression",
            "file_path": "test.fs",
            "line_number": 10,
            "column_number": 5
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "fsharp"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]
    
    def test_analyze_type_error(self):
        """Test analysis of type errors."""
        error_data = {
            "error_type": "FSharpError",
            "message": "Type mismatch. Expecting a 'int' but given a 'string'",
            "file_path": "test.fs",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "fsharp"
        assert analysis["subcategory"] == "type"
        assert analysis["confidence"] == "high"
        assert "type" in analysis["tags"]
    
    def test_analyze_pattern_error(self):
        """Test analysis of pattern match errors."""
        error_data = {
            "error_type": "FSharpError",
            "message": "Incomplete pattern matches on this expression",
            "file_path": "test.fs",
            "line_number": 20,
            "column_number": 12
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "fsharp"
        assert analysis["subcategory"] == "pattern"
        assert analysis["confidence"] == "high"
        assert "pattern" in analysis["tags"]
    
    def test_analyze_computation_expression_error(self):
        """Test analysis of computation expression errors."""
        error_data = {
            "error_type": "FSharpError",
            "message": "This computation expression requires a 'Bind' method",
            "file_path": "test.fs",
            "line_number": 25,
            "column_number": 15
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "fsharp"
        assert analysis["subcategory"] == "computation"
        assert analysis["confidence"] == "high"
        assert "computation" in analysis["tags"]
    
    def test_analyze_discriminated_union_error(self):
        """Test analysis of discriminated union errors."""
        error_data = {
            "error_type": "FSharpError",
            "message": "The union case 'Some' expects 1 argument",
            "file_path": "test.fs",
            "line_number": 30,
            "column_number": 10
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "fsharp"
        assert analysis["subcategory"] == "union"
        assert analysis["confidence"] == "high"
        assert "union" in analysis["tags"]
    
    def test_analyze_import_error(self):
        """Test analysis of import errors."""
        error_data = {
            "error_type": "FSharpError",
            "message": "The namespace or module 'MyModule' is not defined",
            "file_path": "test.fs",
            "line_number": 1,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "fsharp"
        assert analysis["subcategory"] == "import"
        assert analysis["confidence"] == "high"
        assert "import" in analysis["tags"]
    
    def test_analyze_null_reference_error(self):
        """Test analysis of null reference errors."""
        error_data = {
            "error_type": "FSharpError",
            "message": "Object reference not set to an instance of an object",
            "file_path": "test.fs",
            "line_number": 35,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "fsharp"
        assert analysis["subcategory"] == "null"
        assert analysis["confidence"] == "high"
        assert "null" in analysis["tags"]
    
    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "FSharpError",
            "message": "Some unknown error message",
            "file_path": "test.fs",
            "line_number": 45,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "fsharp"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestFSharpPatchGenerator:
    """Test the F# patch generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = FSharpPatchGenerator()
    
    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {
            "message": "Unexpected token 'let' in expression",
            "file_path": "test.fs"
        }
        
        analysis = {
            "root_cause": "fsharp_syntax_error",
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
            "message": "Type mismatch. Expecting a 'int' but given a 'string'",
            "file_path": "test.fs"
        }
        
        analysis = {
            "root_cause": "fsharp_type_error",
            "subcategory": "type",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "type" in patch["description"].lower()
    
    def test_generate_pattern_fix(self):
        """Test generation of pattern match fixes."""
        error_data = {
            "message": "Incomplete pattern matches on this expression",
            "file_path": "test.fs"
        }
        
        analysis = {
            "root_cause": "fsharp_pattern_error",
            "subcategory": "pattern",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "pattern" in patch["description"].lower()
    
    def test_generate_computation_expression_fix(self):
        """Test generation of computation expression fixes."""
        error_data = {
            "message": "This computation expression requires a 'Bind' method",
            "file_path": "test.fs"
        }
        
        analysis = {
            "root_cause": "fsharp_computation_error",
            "subcategory": "computation",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "computation" in patch["description"].lower() or "bind" in patch["description"].lower()
    
    def test_generate_union_fix(self):
        """Test generation of discriminated union fixes."""
        error_data = {
            "message": "The union case 'Some' expects 1 argument",
            "file_path": "test.fs"
        }
        
        analysis = {
            "root_cause": "fsharp_union_error",
            "subcategory": "union",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "union" in patch["description"].lower() or "argument" in patch["description"].lower()
    
    def test_generate_import_fix(self):
        """Test generation of import fixes."""
        error_data = {
            "message": "The namespace or module 'MyModule' is not defined",
            "file_path": "test.fs"
        }
        
        analysis = {
            "root_cause": "fsharp_import_error",
            "subcategory": "import",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "namespace" in patch["description"].lower() or "module" in patch["description"].lower()
    
    def test_generate_null_reference_fix(self):
        """Test generation of null reference fixes."""
        error_data = {
            "message": "Object reference not set to an instance of an object",
            "file_path": "test.fs"
        }
        
        analysis = {
            "root_cause": "fsharp_null_error",
            "subcategory": "null",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "null" in patch["description"].lower() or "reference" in patch["description"].lower()


class TestFSharpLanguagePlugin:
    """Test the F# language plugin."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = FSharpLanguagePlugin()
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "fsharp"
        assert self.plugin.get_language_name() == "F#"
        assert self.plugin.get_language_version() == "7.0+"
        
        frameworks = self.plugin.get_supported_frameworks()
        assert "fsharp" in frameworks
        assert "dotnet" in frameworks
    
    def test_normalize_error(self):
        """Test error normalization."""
        fsharp_error = {
            "error_type": "FSharpError",
            "message": "Test error",
            "file": "test.fs",
            "line": 10,
            "column": 5,
            "description": "Test error description"
        }
        
        normalized = self.plugin.normalize_error(fsharp_error)
        
        assert normalized["language"] == "fsharp"
        assert normalized["error_type"] == "FSharpError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "test.fs"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5
    
    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "fsharp",
            "error_type": "FSharpError",
            "message": "Test error",
            "file_path": "test.fs",
            "line_number": 10,
            "column_number": 5,
            "severity": "high"
        }
        
        fsharp_error = self.plugin.denormalize_error(standard_error)
        
        assert fsharp_error["error_type"] == "FSharpError"
        assert fsharp_error["message"] == "Test error"
        assert fsharp_error["file_path"] == "test.fs"
        assert fsharp_error["line_number"] == 10
        assert fsharp_error["column_number"] == 5
        assert fsharp_error["file"] == "test.fs"  # Alternative format
        assert fsharp_error["line"] == 10  # Alternative format
    
    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "FSharpError",
            "message": "Type mismatch. Expecting a 'int' but given a 'string'",
            "file_path": "test.fs",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "fsharp"
        assert analysis["language"] == "fsharp"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "fsharp"
        assert analysis["subcategory"] == "type"
    
    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "fsharp_type_error",
            "subcategory": "type",
            "confidence": "high",
            "suggested_fix": "Fix type mismatch"
        }
        
        context = {
            "error_data": {
                "message": "Type mismatch. Expecting a 'int' but given a 'string'",
                "file_path": "test.fs"
            },
            "source_code": 'let x: int = "hello"'
        }
        
        fix = self.plugin.generate_fix(analysis, context)
        
        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".fs" in self.plugin.supported_extensions
        assert ".fsx" in self.plugin.supported_extensions
        assert ".fsi" in self.plugin.supported_extensions
        assert ".fsproj" in self.plugin.supported_extensions
    
    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "fsharp"
        assert analysis["language"] == "fsharp"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])