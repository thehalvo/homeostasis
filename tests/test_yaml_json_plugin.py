"""
Tests for the YAML/JSON language plugin.
"""

import pytest

from modules.analysis.plugins.yaml_json_plugin import \
    YAMLJSONExceptionHandler as YamlJsonExceptionHandler
from modules.analysis.plugins.yaml_json_plugin import \
    YAMLJSONLanguagePlugin as YamlJsonLanguagePlugin
from modules.analysis.plugins.yaml_json_plugin import \
    YAMLJSONPatchGenerator as YamlJsonPatchGenerator


class TestYamlJsonExceptionHandler:
    """Test the YAML/JSON exception handler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = YamlJsonExceptionHandler()

    def test_analyze_yaml_syntax_error(self):
        """Test analysis of YAML syntax errors."""
        error_data = {
            "error_type": "YAMLError",
            "message": "while parsing a block mapping",
            "file_path": "config.yaml",
            "line_number": 10,
            "column_number": 5,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "yaml_json"
        assert analysis["subcategory"] == "yaml_syntax"
        assert analysis["confidence"] == "high"
        assert "yaml" in analysis["tags"]

    def test_analyze_json_syntax_error(self):
        """Test analysis of JSON syntax errors."""
        error_data = {
            "error_type": "JSONError",
            "message": "Expecting property name enclosed in double quotes",
            "file_path": "config.json",
            "line_number": 15,
            "column_number": 8,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "yaml_json"
        assert analysis["subcategory"] == "json_syntax"
        assert analysis["confidence"] == "high"
        assert "json" in analysis["tags"]

    def test_analyze_schema_error(self):
        """Test analysis of schema validation errors."""
        error_data = {
            "error_type": "SchemaError",
            "message": "'required' is a required property",
            "file_path": "schema.yaml",
            "line_number": 20,
            "column_number": 12,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "yaml_json"
        assert analysis["subcategory"] == "schema"
        assert analysis["confidence"] == "high"
        assert "schema" in analysis["tags"]

    def test_analyze_indentation_error(self):
        """Test analysis of indentation errors."""
        error_data = {
            "error_type": "YAMLError",
            "message": "inconsistent indentation",
            "file_path": "config.yml",
            "line_number": 25,
            "column_number": 3,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "yaml_json"
        assert analysis["subcategory"] == "indentation"
        assert analysis["confidence"] == "high"
        assert "indentation" in analysis["tags"]

    def test_analyze_type_error(self):
        """Test analysis of type errors."""
        error_data = {
            "error_type": "TypeError",
            "message": "expected string, got integer",
            "file_path": "data.json",
            "line_number": 30,
            "column_number": 10,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "yaml_json"
        assert analysis["subcategory"] == "type"
        assert analysis["confidence"] == "high"
        assert "type" in analysis["tags"]

    def test_analyze_duplicate_key_error(self):
        """Test analysis of duplicate key errors."""
        error_data = {
            "error_type": "YAMLError",
            "message": "found duplicate key",
            "file_path": "config.yaml",
            "line_number": 35,
            "column_number": 5,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "yaml_json"
        assert analysis["subcategory"] == "duplicate_key"
        assert analysis["confidence"] == "high"
        assert "duplicate" in analysis["tags"]

    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "YAMLError",
            "message": "Some unknown error message",
            "file_path": "test.yaml",
            "line_number": 45,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "yaml_json"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestYamlJsonPatchGenerator:
    """Test the YAML/JSON patch generator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = YamlJsonPatchGenerator()

    def test_generate_yaml_syntax_fix(self):
        """Test generation of YAML syntax fixes."""
        error_data = {
            "message": "while parsing a block mapping",
            "file_path": "config.yaml",
        }

        analysis = {
            "root_cause": "yaml_syntax_error",
            "subcategory": "yaml_syntax",
            "confidence": "high",
        }

        yaml_patch = self.generator.generate_patch(error_data, analysis, "")

        assert yaml_patch is not None
        assert yaml_patch["type"] == "suggestion"
        assert (
            "yaml" in yaml_patch["description"].lower()
            or "syntax" in yaml_patch["description"].lower()
        )

    def test_generate_json_syntax_fix(self):
        """Test generation of JSON syntax fixes."""
        error_data = {
            "message": "Expecting property name enclosed in double quotes",
            "file_path": "config.json",
        }

        analysis = {
            "root_cause": "json_syntax_error",
            "subcategory": "json_syntax",
            "confidence": "high",
        }

        json_patch = self.generator.generate_patch(error_data, analysis, "")

        assert json_patch is not None
        assert json_patch["type"] == "suggestion"
        assert (
            "json" in json_patch["description"].lower()
            or "quote" in json_patch["description"].lower()
        )

    def test_generate_schema_fix(self):
        """Test generation of schema validation fixes."""
        error_data = {
            "message": "'required' is a required property",
            "file_path": "schema.yaml",
        }

        analysis = {
            "root_cause": "schema_error",
            "subcategory": "schema",
            "confidence": "high",
        }

        schema_patch = self.generator.generate_patch(error_data, analysis, "")

        assert schema_patch is not None
        assert schema_patch["type"] == "suggestion"
        assert (
            "schema" in schema_patch["description"].lower()
            or "required" in schema_patch["description"].lower()
        )

    def test_generate_indentation_fix(self):
        """Test generation of indentation fixes."""
        error_data = {"message": "inconsistent indentation", "file_path": "config.yml"}

        analysis = {
            "root_cause": "indentation_error",
            "subcategory": "indentation",
            "confidence": "high",
        }

        indent_patch = self.generator.generate_patch(error_data, analysis, "")

        assert indent_patch is not None
        assert indent_patch["type"] == "suggestion"
        assert "indent" in indent_patch["description"].lower()

    def test_generate_type_fix(self):
        """Test generation of type fixes."""
        error_data = {
            "message": "expected string, got integer",
            "file_path": "data.json",
        }

        analysis = {
            "root_cause": "type_error",
            "subcategory": "type",
            "confidence": "high",
        }

        type_patch = self.generator.generate_patch(error_data, analysis, "")

        assert type_patch is not None
        assert type_patch["type"] == "suggestion"
        assert "type" in type_patch["description"].lower()


class TestYamlJsonLanguagePlugin:
    """Test the YAML/JSON language plugin."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = YamlJsonLanguagePlugin()

    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "yaml_json"
        assert self.plugin.get_language_name() == "YAML/JSON"
        assert self.plugin.get_language_version() == "1.2+"

        frameworks = self.plugin.get_supported_frameworks()
        assert "yaml" in frameworks
        assert "json" in frameworks
        assert "jsonschema" in frameworks

    def test_normalize_error(self):
        """Test error normalization."""
        yaml_error = {
            "error_type": "YAMLError",
            "message": "Test error",
            "file": "test.yaml",
            "line": 10,
            "column": 5,
            "description": "Test error description",
        }

        normalized = self.plugin.normalize_error(yaml_error)

        assert normalized["language"] == "yaml_json"
        assert normalized["error_type"] == "YAMLError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "test.yaml"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5

    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "yaml_json",
            "error_type": "YAMLError",
            "message": "Test error",
            "file_path": "test.yaml",
            "line_number": 10,
            "column_number": 5,
            "severity": "high",
        }

        yaml_error = self.plugin.denormalize_error(standard_error)

        assert yaml_error["error_type"] == "YAMLError"
        assert yaml_error["message"] == "Test error"
        assert yaml_error["file_path"] == "test.yaml"
        assert yaml_error["line_number"] == 10
        assert yaml_error["column_number"] == 5
        assert yaml_error["file"] == "test.yaml"  # Alternative format
        assert yaml_error["line"] == 10  # Alternative format

    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "YAMLError",
            "message": "while parsing a block mapping",
            "file_path": "test.yaml",
            "line_number": 15,
            "column_number": 8,
        }

        analysis = self.plugin.analyze_error(error_data)

        assert analysis["plugin"] == "yaml_json"
        assert analysis["language"] == "yaml_json"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "yaml_json"
        assert analysis["subcategory"] == "yaml_syntax"

    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "yaml_syntax_error",
            "subcategory": "yaml_syntax",
            "confidence": "high",
            "suggested_fix": "Fix YAML syntax",
        }

        context = {
            "error_data": {
                "message": "while parsing a block mapping",
                "file_path": "test.yaml",
            },
            "source_code": "key: value\n  - item",
        }

        fix = self.plugin.generate_fix(analysis, context)

        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix

    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".yaml" in self.plugin.supported_extensions
        assert ".yml" in self.plugin.supported_extensions
        assert ".json" in self.plugin.supported_extensions

    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}

        analysis = self.plugin.analyze_error(error_data)

        assert analysis["plugin"] == "yaml_json"
        assert analysis["language"] == "yaml_json"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])
