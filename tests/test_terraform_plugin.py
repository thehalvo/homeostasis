"""
Tests for the Terraform language plugin.
"""

import pytest

from modules.analysis.plugins.terraform_plugin import (
    TerraformExceptionHandler, TerraformLanguagePlugin,
    TerraformPatchGenerator)


class TestTerraformExceptionHandler:
    """Test the Terraform exception handler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = TerraformExceptionHandler()

    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "TerraformError",
            "message": "Invalid expression: Expected the start of an expression",
            "file_path": "main.tf",
            "line_number": 10,
            "column_number": 5,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "terraform"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]

    def test_analyze_resource_error(self):
        """Test analysis of resource errors."""
        error_data = {
            "error_type": "TerraformError",
            "message": "Error: Invalid resource type",
            "file_path": "main.tf",
            "line_number": 15,
            "column_number": 8,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "terraform"
        assert analysis["subcategory"] == "resource"
        assert analysis["confidence"] == "high"
        assert "resource" in analysis["tags"]

    def test_analyze_provider_error(self):
        """Test analysis of provider errors."""
        error_data = {
            "error_type": "TerraformError",
            "message": "Error: Failed to query available provider packages",
            "file_path": "providers.tf",
            "line_number": 1,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "terraform"
        assert analysis["subcategory"] == "provider"
        assert analysis["confidence"] == "high"
        assert "provider" in analysis["tags"]

    def test_analyze_variable_error(self):
        """Test analysis of variable errors."""
        error_data = {
            "error_type": "TerraformError",
            "message": "Error: Reference to undeclared variable",
            "file_path": "variables.tf",
            "line_number": 20,
            "column_number": 12,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "terraform"
        assert analysis["subcategory"] == "variable"
        assert analysis["confidence"] == "high"
        assert "variable" in analysis["tags"]

    def test_analyze_state_error(self):
        """Test analysis of state errors."""
        error_data = {
            "error_type": "TerraformError",
            "message": "Error acquiring the state lock",
            "file_path": "main.tf",
            "line_number": 1,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "terraform"
        assert analysis["subcategory"] == "state"
        assert analysis["confidence"] == "high"
        assert "state" in analysis["tags"]

    def test_analyze_validation_error(self):
        """Test analysis of validation errors."""
        error_data = {
            "error_type": "TerraformError",
            "message": "Error: Invalid value for variable",
            "file_path": "terraform.tfvars",
            "line_number": 5,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "terraform"
        assert analysis["subcategory"] == "validation"
        assert analysis["confidence"] == "high"
        assert "validation" in analysis["tags"]

    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "TerraformError",
            "message": "Some unknown error message",
            "file_path": "main.tf",
            "line_number": 45,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "terraform"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestTerraformPatchGenerator:
    """Test the Terraform patch generator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = TerraformPatchGenerator()

    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {
            "message": "Invalid expression: Expected the start of an expression",
            "file_path": "main.tf",
        }

        analysis = {
            "root_cause": "terraform_syntax_error",
            "subcategory": "syntax",
            "confidence": "high",
        }

        syntax_patch = self.generator.generate_patch(error_data, analysis, "")

        assert syntax_patch is not None
        assert syntax_patch["type"] == "suggestion"
        assert "syntax" in syntax_patch["description"].lower()

    def test_generate_resource_fix(self):
        """Test generation of resource fixes."""
        error_data = {"message": "Error: Invalid resource type", "file_path": "main.tf"}

        analysis = {
            "root_cause": "terraform_resource_error",
            "subcategory": "resource",
            "confidence": "high",
        }

        resource_patch = self.generator.generate_patch(error_data, analysis, "")

        assert resource_patch is not None
        assert resource_patch["type"] == "suggestion"
        assert "resource" in resource_patch["description"].lower()

    def test_generate_provider_fix(self):
        """Test generation of provider fixes."""
        error_data = {
            "message": "Error: Failed to query available provider packages",
            "file_path": "providers.tf",
        }

        analysis = {
            "root_cause": "terraform_provider_error",
            "subcategory": "provider",
            "confidence": "high",
        }

        provider_patch = self.generator.generate_patch(error_data, analysis, "")

        assert provider_patch is not None
        assert provider_patch["type"] == "suggestion"
        assert "provider" in provider_patch["description"].lower()

    def test_generate_variable_fix(self):
        """Test generation of variable fixes."""
        error_data = {
            "message": "Error: Reference to undeclared variable",
            "file_path": "variables.tf",
        }

        analysis = {
            "root_cause": "terraform_variable_error",
            "subcategory": "variable",
            "confidence": "high",
        }

        variable_patch = self.generator.generate_patch(error_data, analysis, "")

        assert variable_patch is not None
        assert variable_patch["type"] == "suggestion"
        assert "variable" in variable_patch["description"].lower()

    def test_generate_state_fix(self):
        """Test generation of state fixes."""
        error_data = {
            "message": "Error acquiring the state lock",
            "file_path": "main.tf",
        }

        analysis = {
            "root_cause": "terraform_state_error",
            "subcategory": "state",
            "confidence": "high",
        }

        state_patch = self.generator.generate_patch(error_data, analysis, "")

        assert state_patch is not None
        assert state_patch["type"] == "suggestion"
        assert (
            "state" in state_patch["description"].lower()
            or "lock" in state_patch["description"].lower()
        )


class TestTerraformLanguagePlugin:
    """Test the Terraform language plugin."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = TerraformLanguagePlugin()

    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "terraform"
        assert self.plugin.get_language_name() == "Terraform"
        assert self.plugin.get_language_version() == "1.0+"

        frameworks = self.plugin.get_supported_frameworks()
        assert "terraform" in frameworks
        assert "aws" in frameworks
        assert "azure" in frameworks
        assert "gcp" in frameworks

    def test_normalize_error(self):
        """Test error normalization."""
        terraform_error = {
            "error_type": "TerraformError",
            "message": "Test error",
            "file": "main.tf",
            "line": 10,
            "column": 5,
            "description": "Test error description",
        }

        normalized = self.plugin.normalize_error(terraform_error)

        assert normalized["language"] == "terraform"
        assert normalized["error_type"] == "TerraformError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "main.tf"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5

    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "terraform",
            "error_type": "TerraformError",
            "message": "Test error",
            "file_path": "main.tf",
            "line_number": 10,
            "column_number": 5,
            "severity": "high",
        }

        terraform_error = self.plugin.denormalize_error(standard_error)

        assert terraform_error["error_type"] == "TerraformError"
        assert terraform_error["message"] == "Test error"
        assert terraform_error["file_path"] == "main.tf"
        assert terraform_error["line_number"] == 10
        assert terraform_error["column_number"] == 5
        assert terraform_error["file"] == "main.tf"  # Alternative format
        assert terraform_error["line"] == 10  # Alternative format

    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "TerraformError",
            "message": "Error: Invalid resource type",
            "file_path": "main.tf",
            "line_number": 15,
            "column_number": 8,
        }

        analysis = self.plugin.analyze_error(error_data)

        assert analysis["plugin"] == "terraform"
        assert analysis["language"] == "terraform"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "terraform"
        assert analysis["subcategory"] == "resource"

    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "terraform_resource_error",
            "subcategory": "resource",
            "confidence": "high",
            "suggested_fix": "Fix resource type",
        }

        context = {
            "error_data": {
                "message": "Error: Invalid resource type",
                "file_path": "main.tf",
            },
            "source_code": 'resource "aws_instanc" "example" {}',
        }

        fix = self.plugin.generate_fix(analysis, context)

        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix

    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".tf" in self.plugin.supported_extensions
        assert ".tfvars" in self.plugin.supported_extensions

    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}

        analysis = self.plugin.analyze_error(error_data)

        assert analysis["plugin"] == "terraform"
        assert analysis["language"] == "terraform"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])
