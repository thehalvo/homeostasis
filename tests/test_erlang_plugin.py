"""
Tests for the Erlang language plugin.
"""
import pytest
import json
from unittest.mock import Mock, patch

from modules.analysis.plugins.erlang_plugin import (
    ErlangLanguagePlugin, 
    ErlangExceptionHandler, 
    ErlangPatchGenerator
)


class TestErlangExceptionHandler:
    """Test the Erlang exception handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = ErlangExceptionHandler()
    
    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "ErlangError",
            "message": "syntax error before: ')'",
            "file_path": "test.erl",
            "line_number": 10,
            "column_number": 5
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "erlang"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]
    
    def test_analyze_function_error(self):
        """Test analysis of function errors."""
        error_data = {
            "error_type": "ErlangError",
            "message": "function clause head cannot match",
            "file_path": "test.erl",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "erlang"
        assert analysis["subcategory"] == "function"
        assert analysis["confidence"] == "high"
        assert "function" in analysis["tags"]
    
    def test_analyze_pattern_error(self):
        """Test analysis of pattern match errors."""
        error_data = {
            "error_type": "ErlangError",
            "message": "no case clause matching",
            "file_path": "test.erl",
            "line_number": 20,
            "column_number": 12
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "erlang"
        assert analysis["subcategory"] == "pattern"
        assert analysis["confidence"] == "high"
        assert "pattern" in analysis["tags"]
    
    def test_analyze_otp_error(self):
        """Test analysis of OTP errors."""
        error_data = {
            "error_type": "ErlangError",
            "message": "gen_server terminated with reason",
            "file_path": "test.erl",
            "line_number": 25,
            "column_number": 15
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "erlang"
        assert analysis["subcategory"] == "otp"
        assert analysis["confidence"] == "high"
        assert "otp" in analysis["tags"]
    
    def test_analyze_process_error(self):
        """Test analysis of process errors."""
        error_data = {
            "error_type": "ErlangError",
            "message": "** exception error: bad argument",
            "file_path": "test.erl",
            "line_number": 30,
            "column_number": 10
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "erlang"
        assert analysis["subcategory"] == "process"
        assert analysis["confidence"] == "high"
        assert "process" in analysis["tags"]
    
    def test_analyze_module_error(self):
        """Test analysis of module errors."""
        error_data = {
            "error_type": "ErlangError",
            "message": "undefined function module:function/0",
            "file_path": "test.erl",
            "line_number": 1,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "erlang"
        assert analysis["subcategory"] == "module"
        assert analysis["confidence"] == "high"
        assert "module" in analysis["tags"]
    
    def test_analyze_compilation_error(self):
        """Test analysis of compilation errors."""
        error_data = {
            "error_type": "ErlangError",
            "message": "head mismatch",
            "file_path": "test.erl",
            "line_number": 35,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "erlang"
        assert analysis["subcategory"] == "compilation"
        assert analysis["confidence"] == "high"
        assert "compilation" in analysis["tags"]
    
    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "ErlangError",
            "message": "Some unknown error message",
            "file_path": "test.erl",
            "line_number": 45,
            "column_number": 1
        }
        
        analysis = self.handler.analyze_exception(error_data)
        
        assert analysis["category"] == "erlang"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestErlangPatchGenerator:
    """Test the Erlang patch generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ErlangPatchGenerator()
    
    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {
            "message": "syntax error before: ')'",
            "file_path": "test.erl"
        }
        
        analysis = {
            "root_cause": "erlang_syntax_error",
            "subcategory": "syntax",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "syntax" in patch["description"].lower()
    
    def test_generate_function_fix(self):
        """Test generation of function fixes."""
        error_data = {
            "message": "function clause head cannot match",
            "file_path": "test.erl"
        }
        
        analysis = {
            "root_cause": "erlang_function_error",
            "subcategory": "function",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "function" in patch["description"].lower() or "clause" in patch["description"].lower()
    
    def test_generate_pattern_fix(self):
        """Test generation of pattern match fixes."""
        error_data = {
            "message": "no case clause matching",
            "file_path": "test.erl"
        }
        
        analysis = {
            "root_cause": "erlang_pattern_error",
            "subcategory": "pattern",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "pattern" in patch["description"].lower() or "case" in patch["description"].lower()
    
    def test_generate_otp_fix(self):
        """Test generation of OTP fixes."""
        error_data = {
            "message": "gen_server terminated with reason",
            "file_path": "test.erl"
        }
        
        analysis = {
            "root_cause": "erlang_otp_error",
            "subcategory": "otp",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "gen_server" in patch["description"].lower() or "otp" in patch["description"].lower()
    
    def test_generate_process_fix(self):
        """Test generation of process fixes."""
        error_data = {
            "message": "** exception error: bad argument",
            "file_path": "test.erl"
        }
        
        analysis = {
            "root_cause": "erlang_process_error",
            "subcategory": "process",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "process" in patch["description"].lower() or "exception" in patch["description"].lower()
    
    def test_generate_module_fix(self):
        """Test generation of module fixes."""
        error_data = {
            "message": "undefined function module:function/0",
            "file_path": "test.erl"
        }
        
        analysis = {
            "root_cause": "erlang_module_error",
            "subcategory": "module",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "module" in patch["description"].lower() or "function" in patch["description"].lower()
    
    def test_generate_compilation_fix(self):
        """Test generation of compilation fixes."""
        error_data = {
            "message": "head mismatch",
            "file_path": "test.erl"
        }
        
        analysis = {
            "root_cause": "erlang_compilation_error",
            "subcategory": "compilation",
            "confidence": "high"
        }
        
        patch = self.generator.generate_patch(error_data, analysis, "")
        
        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "compilation" in patch["description"].lower() or "mismatch" in patch["description"].lower()


class TestErlangLanguagePlugin:
    """Test the Erlang language plugin."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = ErlangLanguagePlugin()
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "erlang"
        assert self.plugin.get_language_name() == "Erlang"
        assert self.plugin.get_language_version() == "OTP 24+"
        
        frameworks = self.plugin.get_supported_frameworks()
        assert "erlang" in frameworks
        assert "otp" in frameworks
        assert "rebar3" in frameworks
    
    def test_normalize_error(self):
        """Test error normalization."""
        erlang_error = {
            "error_type": "ErlangError",
            "message": "Test error",
            "file": "test.erl",
            "line": 10,
            "column": 5,
            "description": "Test error description"
        }
        
        normalized = self.plugin.normalize_error(erlang_error)
        
        assert normalized["language"] == "erlang"
        assert normalized["error_type"] == "ErlangError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "test.erl"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5
    
    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "erlang",
            "error_type": "ErlangError",
            "message": "Test error",
            "file_path": "test.erl",
            "line_number": 10,
            "column_number": 5,
            "severity": "high"
        }
        
        erlang_error = self.plugin.denormalize_error(standard_error)
        
        assert erlang_error["error_type"] == "ErlangError"
        assert erlang_error["message"] == "Test error"
        assert erlang_error["file_path"] == "test.erl"
        assert erlang_error["line_number"] == 10
        assert erlang_error["column_number"] == 5
        assert erlang_error["file"] == "test.erl"  # Alternative format
        assert erlang_error["line"] == 10  # Alternative format
    
    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "ErlangError",
            "message": "gen_server terminated with reason",
            "file_path": "test.erl",
            "line_number": 15,
            "column_number": 8
        }
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "erlang"
        assert analysis["language"] == "erlang"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "erlang"
        assert analysis["subcategory"] == "otp"
    
    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "erlang_otp_error",
            "subcategory": "otp",
            "confidence": "high",
            "suggested_fix": "Fix OTP behavior"
        }
        
        context = {
            "error_data": {
                "message": "gen_server terminated with reason",
                "file_path": "test.erl"
            },
            "source_code": '-module(test).\n-behaviour(gen_server).'
        }
        
        fix = self.plugin.generate_fix(analysis, context)
        
        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".erl" in self.plugin.supported_extensions
        assert ".hrl" in self.plugin.supported_extensions
        assert ".app" in self.plugin.supported_extensions
        assert ".app.src" in self.plugin.supported_extensions
    
    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}
        
        analysis = self.plugin.analyze_error(error_data)
        
        assert analysis["plugin"] == "erlang"
        assert analysis["language"] == "erlang"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])