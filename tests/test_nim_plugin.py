"""
Tests for the Nim language plugin.
"""
import pytest

from modules.analysis.plugins.nim_plugin import (
    NimLanguagePlugin, 
    NimExceptionHandler, 
    NimPatchGenerator
)


class TestNimExceptionHandler:
    """Test the Nim exception handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = NimExceptionHandler()
    
    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "NimError",
            "message": "Error: invalid indentation",
            "file_path": "test.nim",
            "line_number": 10,
            "column_number": 5
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "nim"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]
    
    def test_analyze_type_error(self):
        """Test analysis of type errors."""
        error_data = {
            "error_type": "NimError",
            "message": "Error: type mismatch: got <int> but expected <string>",
            "file_path": "test.nim",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "nim"
        assert analysis["subcategory"] == "type"
        assert analysis["confidence"] == "high"
        assert "type" in analysis["tags"]
    
    def test_analyze_nil_error(self):
        """Test analysis of nil errors."""
        error_data = {
            "error_type": "NimError",
            "message": "Error: cannot access field of nil object",
            "file_path": "test.nim",
            "line_number": 20,
            "column_number": 12
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "nim"
        assert analysis["subcategory"] == "nil_access"
        assert analysis["confidence"] == "high"
        assert "nil" in analysis["tags"]
    
    def test_analyze_undefined_error(self):
        """Test analysis of undefined identifier errors."""
        error_data = {
            "error_type": "NimError",
            "message": "Error: undeclared identifier: 'myVar'",
            "file_path": "test.nim",
            "line_number": 25,
            "column_number": 15
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "nim"
        assert analysis["subcategory"] == "undefined"
        assert analysis["confidence"] == "high"
        assert "undefined" in analysis["tags"]
    
    def test_analyze_pragma_error(self):
        """Test analysis of pragma errors."""
        error_data = {
            "error_type": "NimError",
            "message": "Error: invalid pragma: nosuchpragma",
            "file_path": "test.nim",
            "line_number": 30,
            "column_number": 10
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "nim"
        assert analysis["subcategory"] == "pragma"
        assert analysis["confidence"] == "high"
        assert "pragma" in analysis["tags"]
    
    def test_analyze_import_error(self):
        """Test analysis of import errors."""
        error_data = {
            "error_type": "NimError",
            "message": "Error: cannot open file: somemodule",
            "file_path": "test.nim",
            "line_number": 1,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "nim"
        assert analysis["subcategory"] == "import"
        assert analysis["confidence"] == "high"
        assert "import" in analysis["tags"]
    
    def test_analyze_compilation_error(self):
        """Test analysis of compilation errors."""
        error_data = {
            "error_type": "NimError",
            "message": "Error: internal error: getTypeDescAux",
            "file_path": "test.nim",
            "line_number": 40,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "nim"
        assert analysis["subcategory"] == "compilation"
        assert analysis["confidence"] == "high"
        assert "compilation" in analysis["tags"]
    
    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "NimError",
            "message": "Some unknown error message",
            "file_path": "test.nim",
            "line_number": 45,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "nim"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestNimPatchGenerator:
    """Test the Nim patch generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = NimPatchGenerator()
    
    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {
            "message": "Error: invalid indentation",
            "file_path": "test.nim"
        }
        
        analysis = {
            "root_cause": "nim_syntax_error",
            "subcategory": "syntax",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "indentation" in patch["description"].lower()
    
    def test_generate_type_fix(self):
        """Test generation of type fixes."""
        error_data = {
            "message": "Error: type mismatch: got <int> but expected <string>",
            "file_path": "test.nim"
        }
        
        analysis = {
            "root_cause": "nim_type_error",
            "subcategory": "type",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "type" in patch["description"].lower()
    
    def test_generate_nil_fix(self):
        """Test generation of nil fixes."""
        error_data = {
            "message": "Error: cannot access field of nil object",
            "file_path": "test.nim"
        }
        
        analysis = {
            "root_cause": "nim_nil_access",
            "subcategory": "nil_access",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "nil" in patch["description"].lower()
    
    def test_generate_undefined_fix(self):
        """Test generation of undefined identifier fixes."""
        error_data = {
            "message": "Error: undeclared identifier: 'myVar'",
            "file_path": "test.nim"
        }
        
        analysis = {
            "root_cause": "nim_undefined_identifier",
            "subcategory": "undefined",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "identifier" in patch["description"].lower() or "undefined" in patch["description"].lower()
    
    def test_generate_pragma_fix(self):
        """Test generation of pragma fixes."""
        error_data = {
            "message": "Error: invalid pragma: nosuchpragma",
            "file_path": "test.nim"
        }
        
        analysis = {
            "root_cause": "nim_pragma_error",
            "subcategory": "pragma",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "pragma" in patch["description"].lower()
    
    def test_generate_import_fix(self):
        """Test generation of import fixes."""
        error_data = {
            "message": "Error: cannot open file: somemodule",
            "file_path": "test.nim"
        }
        
        analysis = {
            "root_cause": "nim_import_error",
            "subcategory": "import",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "import" in patch["description"].lower() or "module" in patch["description"].lower()
    
    def test_generate_compilation_fix(self):
        """Test generation of compilation fixes."""
        error_data = {
            "message": "Error: internal error: getTypeDescAux",
            "file_path": "test.nim"
        }
        
        analysis = {
            "root_cause": "nim_compilation_error",
            "subcategory": "compilation",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "compilation" in patch["description"].lower() or "internal" in patch["description"].lower()


class TestNimLanguagePlugin:
    """Test the Nim language plugin."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = NimLanguagePlugin()
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "nim"
        assert self.plugin.get_language_name() == "Nim"
        assert self.plugin.get_language_version() == "2.0+"
        
        frameworks = self.plugin.get_supported_frameworks()
        assert "nim" in frameworks
        assert "nimble" in frameworks
    
    def test_normalize_error(self):
        """Test error normalization."""
        nim_error = {
            "error_type": "NimError",
            "message": "Test error",
            "file": "test.nim",
            "line": 10,
            "column": 5,
            "description": "Test error description"
        }
        
        normalized = self.plugin.normalize_error(nim_error)
        
        assert normalized["language"] == "nim"
        assert normalized["error_type"] == "NimError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "test.nim"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5
    
    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "nim",
            "error_type": "NimError",
            "message": "Test error",
            "file_path": "test.nim",
            "line_number": 10,
            "column_number": 5,
            "severity": "high"
        }
        
        nim_error = self.plugin.denormalize_error(standard_error)
        
        assert nim_error["error_type"] == "NimError"
        assert nim_error["message"] == "Test error"
        assert nim_error["file_path"] == "test.nim"
        assert nim_error["line_number"] == 10
        assert nim_error["column_number"] == 5
        assert nim_error["file"] == "test.nim"  # Alternative format
        assert nim_error["line"] == 10  # Alternative format
    
    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "NimError",
            "message": "Error: type mismatch: got <int> but expected <string>",
            "file_path": "test.nim",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "nim"
        assert analysis["language"] == "nim"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "nim"
        assert analysis["subcategory"] == "type"
    
    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "nim_type_error",
            "subcategory": "type",
            "confidence": "high",
            "suggested_fix": "Fix type mismatch"
        }
        
        context = {
            "error_data": {
                "message": "Error: type mismatch: got <int> but expected <string>",
                "file_path": "test.nim"
            },
            "source_code": 'let x: string = 42'
        }
        
        fix = self.plugin.generate_fix(analysis, context)
        
        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".nim" in self.plugin.supported_extensions
        assert ".nims" in self.plugin.supported_extensions
        assert ".nimble" in self.plugin.supported_extensions
    
    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "nim"
        assert analysis["language"] == "nim"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])