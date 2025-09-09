"""
Tests for the R language plugin.
"""

import pytest

from modules.analysis.plugins.r_plugin import (RExceptionHandler,
                                               RLanguagePlugin,
                                               RPatchGenerator)


class TestRExceptionHandler:
    """Test the R exception handler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = RExceptionHandler()

    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "RError",
            "message": "Error: unexpected '}' in \"}\"",
            "file_path": "test.R",
            "line_number": 10,
            "column_number": 5,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "r"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]

    def test_analyze_object_error(self):
        """Test analysis of object not found errors."""
        error_data = {
            "error_type": "RError",
            "message": "Error: object 'myVar' not found",
            "file_path": "test.R",
            "line_number": 15,
            "column_number": 8,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "r"
        assert analysis["subcategory"] == "object"
        assert analysis["confidence"] == "high"
        assert "object" in analysis["tags"]

    def test_analyze_function_error(self):
        """Test analysis of function errors."""
        error_data = {
            "error_type": "RError",
            "message": 'Error: could not find function "myFunc"',
            "file_path": "test.R",
            "line_number": 20,
            "column_number": 12,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "r"
        assert analysis["subcategory"] == "function"
        assert analysis["confidence"] == "high"
        assert "function" in analysis["tags"]

    def test_analyze_type_error(self):
        """Test analysis of type errors."""
        error_data = {
            "error_type": "RError",
            "message": "Error in x + y : non-numeric argument to binary operator",
            "file_path": "test.R",
            "line_number": 25,
            "column_number": 15,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "r"
        assert analysis["subcategory"] == "type"
        assert analysis["confidence"] == "high"
        assert "type" in analysis["tags"]

    def test_analyze_package_error(self):
        """Test analysis of package errors."""
        error_data = {
            "error_type": "RError",
            "message": "Error in library(mypackage) : there is no package called 'mypackage'",
            "file_path": "test.R",
            "line_number": 1,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "r"
        assert analysis["subcategory"] == "package"
        assert analysis["confidence"] == "high"
        assert "package" in analysis["tags"]

    def test_analyze_dimension_error(self):
        """Test analysis of dimension errors."""
        error_data = {
            "error_type": "RError",
            "message": "Error: incorrect number of dimensions",
            "file_path": "test.R",
            "line_number": 30,
            "column_number": 10,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "r"
        assert analysis["subcategory"] == "dimension"
        assert analysis["confidence"] == "high"
        assert "dimension" in analysis["tags"]

    def test_analyze_na_error(self):
        """Test analysis of NA/missing value errors."""
        error_data = {
            "error_type": "RError",
            "message": "Error: missing values are not allowed",
            "file_path": "test.R",
            "line_number": 35,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "r"
        assert analysis["subcategory"] == "na"
        assert analysis["confidence"] == "high"
        assert "na" in analysis["tags"]

    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "RError",
            "message": "Some unknown error message",
            "file_path": "test.R",
            "line_number": 45,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "r"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestRPatchGenerator:
    """Test the R patch generator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = RPatchGenerator()

    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {
            "message": "Error: unexpected '}' in \"}\"",
            "file_path": "test.R",
        }

        analysis = {
            "root_cause": "r_syntax_error",
            "subcategory": "syntax",
            "confidence": "high",
        }

        syntax_patch = self.generator.generate_patch(error_data, analysis, "")

        assert syntax_patch is not None
        assert syntax_patch["type"] in ["suggestion", "multiple_suggestions"]
        assert "syntax" in syntax_patch["description"].lower()

    def test_generate_object_fix(self):
        """Test generation of object not found fixes."""
        error_data = {
            "message": "Error: object 'myVar' not found",
            "file_path": "test.R",
        }

        analysis = {
            "root_cause": "r_object_not_found",
            "subcategory": "object",
            "confidence": "high",
        }

        object_patch = self.generator.generate_patch(error_data, analysis, "")

        assert object_patch is not None
        assert object_patch["type"] == "suggestion"
        assert (
            "object" in object_patch["description"].lower()
            or "variable" in object_patch["description"].lower()
        )

    def test_generate_function_fix(self):
        """Test generation of function fixes."""
        error_data = {
            "message": 'Error: could not find function "myFunc"',
            "file_path": "test.R",
        }

        analysis = {
            "root_cause": "r_function_error",
            "subcategory": "function",
            "confidence": "high",
        }

        function_patch = self.generator.generate_patch(error_data, analysis, "")

        assert function_patch is not None
        assert function_patch["type"] == "suggestion"
        assert "function" in function_patch["description"].lower()

    def test_generate_type_fix(self):
        """Test generation of type fixes."""
        error_data = {
            "message": "Error in x + y : non-numeric argument to binary operator",
            "file_path": "test.R",
        }

        analysis = {
            "root_cause": "r_type_error",
            "subcategory": "type",
            "confidence": "high",
        }

        type_patch = self.generator.generate_patch(error_data, analysis, "")

        assert type_patch is not None
        assert type_patch["type"] == "suggestion"
        assert (
            "type" in type_patch["description"].lower()
            or "numeric" in type_patch["description"].lower()
        )

    def test_generate_package_fix(self):
        """Test generation of package fixes."""
        error_data = {
            "message": "Error in library(mypackage) : there is no package called 'mypackage'",
            "file_path": "test.R",
        }

        analysis = {
            "root_cause": "r_package_error",
            "subcategory": "package",
            "confidence": "high",
        }

        package_patch = self.generator.generate_patch(error_data, analysis, "")

        assert package_patch is not None
        assert package_patch["type"] == "suggestion"
        assert (
            "package" in package_patch["description"].lower()
            or "install" in package_patch["description"].lower()
        )

    def test_generate_dimension_fix(self):
        """Test generation of dimension fixes."""
        error_data = {
            "message": "Error: incorrect number of dimensions",
            "file_path": "test.R",
        }

        analysis = {
            "root_cause": "r_dimension_error",
            "subcategory": "dimension",
            "confidence": "high",
        }

        dimension_patch = self.generator.generate_patch(error_data, analysis, "")

        assert dimension_patch is not None
        assert dimension_patch["type"] == "suggestion"
        assert "dimension" in dimension_patch["description"].lower()

    def test_generate_na_fix(self):
        """Test generation of NA/missing value fixes."""
        error_data = {
            "message": "Error: missing values are not allowed",
            "file_path": "test.R",
        }

        analysis = {
            "root_cause": "r_na_error",
            "subcategory": "na",
            "confidence": "high",
        }

        na_patch = self.generator.generate_patch(error_data, analysis, "")

        assert na_patch is not None
        assert na_patch["type"] == "suggestion"
        assert (
            "na" in na_patch["description"].lower()
            or "missing" in na_patch["description"].lower()
        )


class TestRLanguagePlugin:
    """Test the R language plugin."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = RLanguagePlugin()

    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "r"
        assert self.plugin.get_language_name() == "R"
        assert self.plugin.get_language_version() == "4.0+"

        frameworks = self.plugin.get_supported_frameworks()
        assert "r" in frameworks
        assert "tidyverse" in frameworks
        assert "shiny" in frameworks

    def test_normalize_error(self):
        """Test error normalization."""
        r_error = {
            "error_type": "RError",
            "message": "Test error",
            "file": "test.R",
            "line": 10,
            "column": 5,
            "description": "Test error description",
        }

        normalized = self.plugin.normalize_error(r_error)

        assert normalized["language"] == "r"
        assert normalized["error_type"] == "RError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "test.R"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5

    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "r",
            "error_type": "RError",
            "message": "Test error",
            "file_path": "test.R",
            "line_number": 10,
            "column_number": 5,
            "severity": "high",
        }

        r_error = self.plugin.denormalize_error(standard_error)

        assert r_error["error_type"] == "RError"
        assert r_error["message"] == "Test error"
        assert r_error["file_path"] == "test.R"
        assert r_error["line_number"] == 10
        assert r_error["column_number"] == 5
        assert r_error["file"] == "test.R"  # Alternative format
        assert r_error["line"] == 10  # Alternative format

    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "RError",
            "message": "Error: object 'myVar' not found",
            "file_path": "test.R",
            "line_number": 15,
            "column_number": 8,
        }

        analysis = self.plugin.analyze_error(error_data)

        assert analysis["plugin"] == "r"
        assert analysis["language"] == "r"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "r"
        assert analysis["subcategory"] == "object"

    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "r_object_error",
            "subcategory": "object",
            "confidence": "high",
            "suggested_fix": "Fix object not found",
        }

        context = {
            "error_data": {
                "message": "Error: object 'myVar' not found",
                "file_path": "test.R",
            },
            "source_code": "print(myVar)",
        }

        fix = self.plugin.generate_fix(analysis, context)

        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix

    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".R" in self.plugin.supported_extensions
        assert ".r" in self.plugin.supported_extensions
        assert ".Rmd" in self.plugin.supported_extensions
        assert ".Rnw" in self.plugin.supported_extensions

    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}

        analysis = self.plugin.analyze_error(error_data)

        assert analysis["plugin"] == "r"
        assert analysis["language"] == "r"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])
