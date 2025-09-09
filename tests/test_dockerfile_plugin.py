"""
Tests for the Dockerfile language plugin.
"""

import pytest

from modules.analysis.plugins.dockerfile_plugin import (
    DockerfileExceptionHandler, DockerfileLanguagePlugin,
    DockerfilePatchGenerator)


class TestDockerfileExceptionHandler:
    """Test the Dockerfile exception handler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = DockerfileExceptionHandler()

    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "DockerfileError",
            "message": "Unknown instruction: INVALIDCMD",
            "file_path": "Dockerfile",
            "line_number": 10,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "dockerfile"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]

    def test_analyze_build_error(self):
        """Test analysis of build errors."""
        error_data = {
            "error_type": "DockerBuildError",
            "message": "failed to solve with frontend dockerfile.v0",
            "file_path": "Dockerfile",
            "line_number": 15,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "dockerfile"
        assert analysis["subcategory"] == "build"
        assert analysis["confidence"] == "high"
        assert "build" in analysis["tags"]

    def test_analyze_layer_error(self):
        """Test analysis of layer optimization errors."""
        error_data = {
            "error_type": "DockerfileWarning",
            "message": "Multiple consecutive RUN instructions",
            "file_path": "Dockerfile",
            "line_number": 20,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "dockerfile"
        assert analysis["subcategory"] == "layer"
        assert analysis["confidence"] == "high"
        assert "layer" in analysis["tags"]

    def test_analyze_security_error(self):
        """Test analysis of security errors."""
        error_data = {
            "error_type": "DockerSecurityError",
            "message": "Running as root user",
            "file_path": "Dockerfile",
            "line_number": 25,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "dockerfile"
        assert analysis["subcategory"] == "security"
        assert analysis["confidence"] == "high"
        assert "security" in analysis["tags"]

    def test_analyze_path_error(self):
        """Test analysis of path errors."""
        error_data = {
            "error_type": "DockerfileError",
            "message": "COPY failed: stat /var/lib/docker/tmp/docker-builder",
            "file_path": "Dockerfile",
            "line_number": 30,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        print(f"Analysis result: {analysis}")
        assert analysis["category"] == "dockerfile"
        assert analysis["subcategory"] == "path"
        assert analysis["confidence"] == "high"
        assert "path" in analysis["tags"]

    def test_analyze_arg_error(self):
        """Test analysis of ARG/ENV errors."""
        error_data = {
            "error_type": "DockerfileError",
            "message": "ARG requires exactly one argument",
            "file_path": "Dockerfile",
            "line_number": 5,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "dockerfile"
        assert analysis["subcategory"] == "arg"
        assert analysis["confidence"] == "high"
        assert "arg" in analysis["tags"]

    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "DockerfileError",
            "message": "Some unknown error message",
            "file_path": "Dockerfile",
            "line_number": 45,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "dockerfile"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestDockerfilePatchGenerator:
    """Test the Dockerfile patch generator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = DockerfilePatchGenerator()

    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {
            "message": "Unknown instruction: INVALIDCMD",
            "file_path": "Dockerfile",
        }

        analysis = {
            "root_cause": "dockerfile_syntax_error",
            "subcategory": "syntax",
            "confidence": "high",
        }

        syntax_patch = self.generator.generate_patch(error_data, analysis, "")

        assert syntax_patch is not None
        assert syntax_patch["type"] == "suggestion"
        assert (
            "instruction" in syntax_patch["description"].lower()
            or "syntax" in syntax_patch["description"].lower()
        )

    def test_generate_build_fix(self):
        """Test generation of build fixes."""
        error_data = {
            "message": "failed to solve with frontend dockerfile.v0",
            "file_path": "Dockerfile",
        }

        analysis = {
            "root_cause": "dockerfile_build_error",
            "subcategory": "build",
            "confidence": "high",
        }

        build_patch = self.generator.generate_patch(error_data, analysis, "")

        assert build_patch is not None
        assert build_patch["type"] == "suggestion"
        assert "build" in build_patch["description"].lower()

    def test_generate_layer_fix(self):
        """Test generation of layer optimization fixes."""
        error_data = {
            "message": "Multiple consecutive RUN instructions",
            "file_path": "Dockerfile",
        }

        analysis = {
            "root_cause": "dockerfile_layer_error",
            "subcategory": "layer",
            "confidence": "high",
        }

        layer_patch = self.generator.generate_patch(error_data, analysis, "")

        assert layer_patch is not None
        assert layer_patch["type"] == "suggestion"
        assert (
            "layer" in layer_patch["description"].lower()
            or "run" in layer_patch["description"].lower()
        )

    def test_generate_security_fix(self):
        """Test generation of security fixes."""
        error_data = {"message": "Running as root user", "file_path": "Dockerfile"}

        analysis = {
            "root_cause": "dockerfile_security_issue",
            "subcategory": "security",
            "confidence": "high",
        }

        security_patch = self.generator.generate_patch(error_data, analysis, "")

        assert security_patch is not None
        assert security_patch["type"] == "suggestion"
        assert (
            "security" in security_patch["description"].lower()
            or "user" in security_patch["description"].lower()
        )

    def test_generate_path_fix(self):
        """Test generation of path fixes."""
        error_data = {
            "message": "COPY failed: stat /var/lib/docker/tmp/docker-builder",
            "file_path": "Dockerfile",
        }

        analysis = {
            "root_cause": "dockerfile_path_error",
            "subcategory": "path",
            "confidence": "high",
        }

        path_patch = self.generator.generate_patch(error_data, analysis, "")

        assert path_patch is not None
        assert path_patch["type"] == "suggestion"
        assert (
            "path" in path_patch["description"].lower()
            or "copy" in path_patch["description"].lower()
        )


class TestDockerfileLanguagePlugin:
    """Test the Dockerfile language plugin."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = DockerfileLanguagePlugin()

    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "dockerfile"
        assert self.plugin.get_language_name() == "Dockerfile"
        assert self.plugin.get_language_version() == "1.0+"

        frameworks = self.plugin.get_supported_frameworks()
        assert "docker" in frameworks
        assert "docker-compose" in frameworks

    def test_normalize_error(self):
        """Test error normalization."""
        docker_error = {
            "error_type": "DockerfileError",
            "message": "Test error",
            "file": "Dockerfile",
            "line": 10,
            "column": 1,
            "description": "Test error description",
        }

        normalized = self.plugin.normalize_error(docker_error)

        assert normalized["language"] == "dockerfile"
        assert normalized["error_type"] == "DockerfileError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "Dockerfile"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 1

    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "dockerfile",
            "error_type": "DockerfileError",
            "message": "Test error",
            "file_path": "Dockerfile",
            "line_number": 10,
            "column_number": 1,
            "severity": "high",
        }

        docker_error = self.plugin.denormalize_error(standard_error)

        assert docker_error["error_type"] == "DockerfileError"
        assert docker_error["message"] == "Test error"
        assert docker_error["file_path"] == "Dockerfile"
        assert docker_error["line_number"] == 10
        assert docker_error["column_number"] == 1
        assert docker_error["file"] == "Dockerfile"  # Alternative format
        assert docker_error["line"] == 10  # Alternative format

    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "DockerfileError",
            "message": "Unknown instruction: INVALIDCMD",
            "file_path": "Dockerfile",
            "line_number": 15,
            "column_number": 1,
        }

        analysis = self.plugin.analyze_error(error_data)

        assert analysis["plugin"] == "dockerfile"
        assert analysis["language"] == "dockerfile"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "dockerfile"
        assert analysis["subcategory"] == "syntax"

    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "dockerfile_syntax_error",
            "subcategory": "syntax",
            "confidence": "high",
            "suggested_fix": "Fix invalid instruction",
        }

        context = {
            "error_data": {
                "message": "Unknown instruction: INVALIDCMD",
                "file_path": "Dockerfile",
            },
            "source_code": "FROM ubuntu\nINVALIDCMD test\nRUN apt-get update",
        }

        fix = self.plugin.generate_fix(analysis, context)

        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix

    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert "Dockerfile" in self.plugin.supported_extensions
        assert ".dockerfile" in self.plugin.supported_extensions

    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}

        analysis = self.plugin.analyze_error(error_data)

        assert analysis["plugin"] == "dockerfile"
        assert analysis["language"] == "dockerfile"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])
