"""
Tests for the PowerShell language plugin.
"""
import pytest

from modules.analysis.plugins.powershell_plugin import (
    PowerShellLanguagePlugin, 
    PowerShellExceptionHandler, 
    PowerShellPatchGenerator
)


class TestPowerShellExceptionHandler:
    """Test the PowerShell exception handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = PowerShellExceptionHandler()
    
    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "PowerShellError",
            "message": "Unexpected token 'in' in expression or statement",
            "file_path": "test.ps1",
            "line_number": 10,
            "column_number": 5
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "powershell"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]
    
    def test_analyze_cmdlet_error(self):
        """Test analysis of cmdlet errors."""
        error_data = {
            "error_type": "PowerShellError",
            "message": "The term 'Get-Something' is not recognized as a cmdlet",
            "file_path": "test.ps1",
            "line_number": 15,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "powershell"
        assert analysis["subcategory"] == "cmdlet"
        assert analysis["confidence"] == "high"
        assert "cmdlet" in analysis["tags"]
    
    def test_analyze_variable_error(self):
        """Test analysis of variable errors."""
        error_data = {
            "error_type": "PowerShellError",
            "message": "Cannot find a variable with the name 'myVar'",
            "file_path": "test.ps1",
            "line_number": 20,
            "column_number": 12
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "powershell"
        assert analysis["subcategory"] == "variable"
        assert analysis["confidence"] == "high"
        assert "variable" in analysis["tags"]
    
    def test_analyze_permission_error(self):
        """Test analysis of permission errors."""
        error_data = {
            "error_type": "PowerShellError",
            "message": "Access to the path is denied",
            "file_path": "test.ps1",
            "line_number": 25,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "powershell"
        assert analysis["subcategory"] == "permission"
        assert analysis["confidence"] == "high"
        assert "permission" in analysis["tags"]
    
    def test_analyze_module_error(self):
        """Test analysis of module errors."""
        error_data = {
            "error_type": "PowerShellError",
            "message": "The specified module 'MyModule' was not loaded",
            "file_path": "test.ps1",
            "line_number": 1,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "powershell"
        assert analysis["subcategory"] == "module"
        assert analysis["confidence"] == "high"
        assert "module" in analysis["tags"]
    
    def test_analyze_type_error(self):
        """Test analysis of type errors."""
        error_data = {
            "error_type": "PowerShellError",
            "message": "Cannot convert value to type System.Int32",
            "file_path": "test.ps1",
            "line_number": 30,
            "column_number": 10
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "powershell"
        assert analysis["subcategory"] == "type"
        assert analysis["confidence"] == "high"
        assert "type" in analysis["tags"]
    
    def test_analyze_pipeline_error(self):
        """Test analysis of pipeline errors."""
        error_data = {
            "error_type": "PowerShellError",
            "message": "The input object cannot be bound to any parameters",
            "file_path": "test.ps1",
            "line_number": 35,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "powershell"
        assert analysis["subcategory"] == "pipeline"
        assert analysis["confidence"] == "high"
        assert "pipeline" in analysis["tags"]
    
    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "PowerShellError",
            "message": "Some unknown error message",
            "file_path": "test.ps1",
            "line_number": 45,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "powershell"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestPowerShellPatchGenerator:
    """Test the PowerShell patch generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PowerShellPatchGenerator()
    
    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {
            "message": "Unexpected token 'in' in expression or statement",
            "file_path": "test.ps1"
        }
        
        analysis = {
            "root_cause": "powershell_syntax_error",
            "subcategory": "syntax",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "syntax" in patch["description"].lower()
    
    def test_generate_cmdlet_fix(self):
        """Test generation of cmdlet fixes."""
        error_data = {
            "message": "The term 'Get-Something' is not recognized as a cmdlet",
            "file_path": "test.ps1"
        }
        
        analysis = {
            "root_cause": "powershell_cmdlet_error",
            "subcategory": "cmdlet",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "cmdlet" in patch["description"].lower()
    
    def test_generate_variable_fix(self):
        """Test generation of variable fixes."""
        error_data = {
            "message": "Cannot find a variable with the name 'myVar'",
            "file_path": "test.ps1"
        }
        
        analysis = {
            "root_cause": "powershell_variable_error",
            "subcategory": "variable",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "variable" in patch["description"].lower()
    
    def test_generate_permission_fix(self):
        """Test generation of permission fixes."""
        error_data = {
            "message": "Access to the path is denied",
            "file_path": "test.ps1"
        }
        
        analysis = {
            "root_cause": "powershell_permission_error",
            "subcategory": "permission",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "permission" in patch["description"].lower() or "access" in patch["description"].lower()
    
    def test_generate_module_fix(self):
        """Test generation of module fixes."""
        error_data = {
            "message": "The specified module 'MyModule' was not loaded",
            "file_path": "test.ps1"
        }
        
        analysis = {
            "root_cause": "powershell_module_error",
            "subcategory": "module",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "module" in patch["description"].lower() or "import" in patch["description"].lower()
    
    def test_generate_type_fix(self):
        """Test generation of type fixes."""
        error_data = {
            "message": "Cannot convert value to type System.Int32",
            "file_path": "test.ps1"
        }
        
        analysis = {
            "root_cause": "powershell_type_error",
            "subcategory": "type",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "type" in patch["description"].lower() or "convert" in patch["description"].lower()
    
    def test_generate_pipeline_fix(self):
        """Test generation of pipeline fixes."""
        error_data = {
            "message": "The input object cannot be bound to any parameters",
            "file_path": "test.ps1"
        }
        
        analysis = {
            "root_cause": "powershell_pipeline_error",
            "subcategory": "pipeline",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "pipeline" in patch["description"].lower() or "parameter" in patch["description"].lower()


class TestPowerShellLanguagePlugin:
    """Test the PowerShell language plugin."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = PowerShellLanguagePlugin()
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "powershell"
        assert self.plugin.get_language_name() == "PowerShell"
        assert self.plugin.get_language_version() == "7.0+"
        
        frameworks = self.plugin.get_supported_frameworks()
        assert "powershell" in frameworks
        assert "pwsh" in frameworks
    
    def test_normalize_error(self):
        """Test error normalization."""
        powershell_error = {
            "error_type": "PowerShellError",
            "message": "Test error",
            "file": "test.ps1",
            "line": 10,
            "column": 5,
            "description": "Test error description"
        }
        
        normalized = self.plugin.normalize_error(powershell_error)
        
        assert normalized["language"] == "powershell"
        assert normalized["error_type"] == "PowerShellError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "test.ps1"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5
    
    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "powershell",
            "error_type": "PowerShellError",
            "message": "Test error",
            "file_path": "test.ps1",
            "line_number": 10,
            "column_number": 5,
            "severity": "high"
        }
        
        powershell_error = self.plugin.denormalize_error(standard_error)
        
        assert powershell_error["error_type"] == "PowerShellError"
        assert powershell_error["message"] == "Test error"
        assert powershell_error["file_path"] == "test.ps1"
        assert powershell_error["line_number"] == 10
        assert powershell_error["column_number"] == 5
        assert powershell_error["file"] == "test.ps1"  # Alternative format
        assert powershell_error["line"] == 10  # Alternative format
    
    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "PowerShellError",
            "message": "The term 'Get-Something' is not recognized as a cmdlet",
            "file_path": "test.ps1",
            "line_number": 15,
            "column_number": 1
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "powershell"
        assert analysis["language"] == "powershell"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "powershell"
        assert analysis["subcategory"] == "cmdlet"
    
    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "powershell_cmdlet_error",
            "subcategory": "cmdlet",
            "confidence": "high",
            "suggested_fix": "Fix cmdlet not found"
        }
        
        context = {
            "error_data": {
                "message": "The term 'Get-Something' is not recognized as a cmdlet",
                "file_path": "test.ps1"
            },
            "source_code": "Get-Something -Name test"
        }
        
        fix = self.plugin.generate_fix(analysis, context)
        
        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".ps1" in self.plugin.supported_extensions
        assert ".psm1" in self.plugin.supported_extensions
        assert ".psd1" in self.plugin.supported_extensions
    
    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "powershell"
        assert analysis["language"] == "powershell"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])