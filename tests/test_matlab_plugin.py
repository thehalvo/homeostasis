"""
Tests for the MATLAB language plugin.
"""
import pytest
import json
from unittest.mock import Mock, patch

from modules.analysis.plugins.matlab_plugin import (
    MatlabLanguagePlugin, 
    MatlabExceptionHandler, 
    MatlabPatchGenerator
)


class TestMatlabExceptionHandler:
    """Test the MATLAB exception handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = MatlabExceptionHandler()
    
    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "MatlabError",
            "message": "Error: Expression or statement is incorrect--possibly unbalanced",
            "file_path": "test.m",
            "line_number": 10,
            "column_number": 5
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "matlab"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]
    
    def test_analyze_undefined_error(self):
        """Test analysis of undefined variable/function errors."""
        error_data = {
            "error_type": "MatlabError",
            "message": "Undefined function or variable 'myVar'",
            "file_path": "test.m",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "matlab"
        assert analysis["subcategory"] == "undefined"
        assert analysis["confidence"] == "high"
        assert "undefined" in analysis["tags"]
    
    def test_analyze_dimension_error(self):
        """Test analysis of dimension mismatch errors."""
        error_data = {
            "error_type": "MatlabError",
            "message": "Matrix dimensions must agree",
            "file_path": "test.m",
            "line_number": 20,
            "column_number": 12
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "matlab"
        assert analysis["subcategory"] == "dimension"
        assert analysis["confidence"] == "high"
        assert "dimension" in analysis["tags"]
    
    def test_analyze_index_error(self):
        """Test analysis of index errors."""
        error_data = {
            "error_type": "MatlabError",
            "message": "Index exceeds matrix dimensions",
            "file_path": "test.m",
            "line_number": 25,
            "column_number": 15
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "matlab"
        assert analysis["subcategory"] == "index"
        assert analysis["confidence"] == "high"
        assert "index" in analysis["tags"]
    
    def test_analyze_file_error(self):
        """Test analysis of file I/O errors."""
        error_data = {
            "error_type": "MatlabError",
            "message": "Unable to read file 'data.mat'. No such file or directory",
            "file_path": "test.m",
            "line_number": 30,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "matlab"
        assert analysis["subcategory"] == "file"
        assert analysis["confidence"] == "high"
        assert "file" in analysis["tags"]
    
    def test_analyze_type_error(self):
        """Test analysis of type errors."""
        error_data = {
            "error_type": "MatlabError",
            "message": "Conversion to double from cell is not possible",
            "file_path": "test.m",
            "line_number": 35,
            "column_number": 10
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "matlab"
        assert analysis["subcategory"] == "type"
        assert analysis["confidence"] == "high"
        assert "type" in analysis["tags"]
    
    def test_analyze_toolbox_error(self):
        """Test analysis of toolbox/license errors."""
        error_data = {
            "error_type": "MatlabError",
            "message": "License checkout failed. No such feature exists",
            "file_path": "test.m",
            "line_number": 1,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "matlab"
        assert analysis["subcategory"] == "toolbox"
        assert analysis["confidence"] == "high"
        assert "toolbox" in analysis["tags"]
    
    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "MatlabError",
            "message": "Some unknown error message",
            "file_path": "test.m",
            "line_number": 45,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "matlab"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestMatlabPatchGenerator:
    """Test the MATLAB patch generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = MatlabPatchGenerator()
    
    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {
            "message": "Error: Expression or statement is incorrect--possibly unbalanced",
            "file_path": "test.m"
        }
        
        analysis = {
            "root_cause": "matlab_syntax_error",
            "subcategory": "syntax",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "syntax" in patch["description"].lower()
    
    def test_generate_undefined_fix(self):
        """Test generation of undefined variable/function fixes."""
        error_data = {
            "message": "Undefined function or variable 'myVar'",
            "file_path": "test.m"
        }
        
        analysis = {
            "root_cause": "matlab_undefined_error",
            "subcategory": "undefined",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "undefined" in patch["description"].lower() or "variable" in patch["description"].lower()
    
    def test_generate_dimension_fix(self):
        """Test generation of dimension mismatch fixes."""
        error_data = {
            "message": "Matrix dimensions must agree",
            "file_path": "test.m"
        }
        
        analysis = {
            "root_cause": "matlab_dimension_error",
            "subcategory": "dimension",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "dimension" in patch["description"].lower() or "matrix" in patch["description"].lower()
    
    def test_generate_index_fix(self):
        """Test generation of index fixes."""
        error_data = {
            "message": "Index exceeds matrix dimensions",
            "file_path": "test.m"
        }
        
        analysis = {
            "root_cause": "matlab_index_error",
            "subcategory": "index",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "index" in patch["description"].lower()
    
    def test_generate_file_fix(self):
        """Test generation of file I/O fixes."""
        error_data = {
            "message": "Unable to read file 'data.mat'. No such file or directory",
            "file_path": "test.m"
        }
        
        analysis = {
            "root_cause": "matlab_file_error",
            "subcategory": "file",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "file" in patch["description"].lower()
    
    def test_generate_type_fix(self):
        """Test generation of type fixes."""
        error_data = {
            "message": "Conversion to double from cell is not possible",
            "file_path": "test.m"
        }
        
        analysis = {
            "root_cause": "matlab_type_error",
            "subcategory": "type",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "type" in patch["description"].lower() or "conversion" in patch["description"].lower()
    
    def test_generate_toolbox_fix(self):
        """Test generation of toolbox/license fixes."""
        error_data = {
            "message": "License checkout failed. No such feature exists",
            "file_path": "test.m"
        }
        
        analysis = {
            "root_cause": "matlab_toolbox_error",
            "subcategory": "toolbox",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "toolbox" in patch["description"].lower() or "license" in patch["description"].lower()


class TestMatlabLanguagePlugin:
    """Test the MATLAB language plugin."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = MatlabLanguagePlugin()
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "matlab"
        assert self.plugin.get_language_name() == "MATLAB"
        assert self.plugin.get_language_version() == "R2021a+"
        
        frameworks = self.plugin.get_supported_frameworks()
        assert "matlab" in frameworks
        assert "simulink" in frameworks
    
    def test_normalize_error(self):
        """Test error normalization."""
        matlab_error = {
            "error_type": "MatlabError",
            "message": "Test error",
            "file": "test.m",
            "line": 10,
            "column": 5,
            "description": "Test error description"
        }
        
        normalized = self.plugin.normalize_error(matlab_error)
        
        assert normalized["language"] == "matlab"
        assert normalized["error_type"] == "MatlabError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "test.m"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5
    
    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "matlab",
            "error_type": "MatlabError",
            "message": "Test error",
            "file_path": "test.m",
            "line_number": 10,
            "column_number": 5,
            "severity": "high"
        }
        
        matlab_error = self.plugin.denormalize_error(standard_error)
        
        assert matlab_error["error_type"] == "MatlabError"
        assert matlab_error["message"] == "Test error"
        assert matlab_error["file_path"] == "test.m"
        assert matlab_error["line_number"] == 10
        assert matlab_error["column_number"] == 5
        assert matlab_error["file"] == "test.m"  # Alternative format
        assert matlab_error["line"] == 10  # Alternative format
    
    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "MatlabError",
            "message": "Matrix dimensions must agree",
            "file_path": "test.m",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "matlab"
        assert analysis["language"] == "matlab"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "matlab"
        assert analysis["subcategory"] == "dimension"
    
    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "matlab_dimension_error",
            "subcategory": "dimension",
            "confidence": "high",
            "suggested_fix": "Fix matrix dimension mismatch"
        }
        
        context = {
            "error_data": {
                "message": "Matrix dimensions must agree",
                "file_path": "test.m"
            },
            "source_code": "A = [1 2 3]; B = [4; 5]; C = A * B;"
        }
        
        fix = self.plugin.generate_fix(analysis, context)
        
        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".m" in self.plugin.supported_extensions
        assert ".mlx" in self.plugin.supported_extensions
        assert ".mat" in self.plugin.supported_extensions
    
    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "matlab"
        assert analysis["language"] == "matlab"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])