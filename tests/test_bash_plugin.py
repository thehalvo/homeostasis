"""
Tests for the Bash/Shell language plugin.
"""

import pytest

from modules.analysis.plugins.bash_plugin import (BashExceptionHandler,
                                                  BashLanguagePlugin,
                                                  BashPatchGenerator)


class TestBashExceptionHandler:
    """Test the Bash exception handler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = BashExceptionHandler()

    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "BashError",
            "message": "syntax error near unexpected token ')'",
            "file_path": "test.sh",
            "line_number": 10,
            "column_number": 5,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "bash"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]

    def test_analyze_command_error(self):
        """Test analysis of command errors."""
        error_data = {
            "error_type": "BashError",
            "message": "command not found: gitx",
            "file_path": "test.sh",
            "line_number": 15,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "bash"
        assert analysis["subcategory"] == "command"
        assert analysis["confidence"] == "high"
        assert "command" in analysis["tags"]

    def test_analyze_variable_error(self):
        """Test analysis of variable errors."""
        error_data = {
            "error_type": "BashError",
            "message": "unbound variable: MY_VAR",
            "file_path": "test.sh",
            "line_number": 20,
            "column_number": 12,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "bash"
        assert analysis["subcategory"] == "variable"
        assert analysis["confidence"] == "high"
        assert "variable" in analysis["tags"]

    def test_analyze_permission_error(self):
        """Test analysis of permission errors."""
        error_data = {
            "error_type": "BashError",
            "message": "Permission denied",
            "file_path": "test.sh",
            "line_number": 25,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "bash"
        assert analysis["subcategory"] == "permission"
        assert analysis["confidence"] == "high"
        assert "permission" in analysis["tags"]

    def test_analyze_file_error(self):
        """Test analysis of file errors."""
        error_data = {
            "error_type": "BashError",
            "message": "No such file or directory",
            "file_path": "test.sh",
            "line_number": 30,
            "column_number": 10,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "bash"
        assert analysis["subcategory"] == "file"
        assert analysis["confidence"] == "high"
        assert "file" in analysis["tags"]

    def test_analyze_expansion_error(self):
        """Test analysis of expansion errors."""
        error_data = {
            "error_type": "BashError",
            "message": "bad substitution",
            "file_path": "test.sh",
            "line_number": 35,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "bash"
        assert analysis["subcategory"] == "expansion"
        assert analysis["confidence"] == "high"
        assert "expansion" in analysis["tags"]

    def test_analyze_pipe_error(self):
        """Test analysis of pipe errors."""
        error_data = {
            "error_type": "BashError",
            "message": "Broken pipe",
            "file_path": "test.sh",
            "line_number": 40,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "bash"
        assert analysis["subcategory"] == "pipe"
        assert analysis["confidence"] == "high"
        assert "pipe" in analysis["tags"]

    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "BashError",
            "message": "Some unknown error message",
            "file_path": "test.sh",
            "line_number": 45,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "bash"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestBashPatchGenerator:
    """Test the Bash patch generator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = BashPatchGenerator()

    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {
            "message": "syntax error near unexpected token ')'",
            "file_path": "test.sh",
        }

        analysis = {
            "root_cause": "bash_syntax_error",
            "subcategory": "syntax",
            "confidence": "high",
        }

        syntax_patch = self.generator.generate_patch(error_data, analysis, "")

        assert syntax_patch is not None
        assert syntax_patch["type"] == "suggestion"
        assert "syntax" in syntax_patch["description"].lower()

    def test_generate_command_fix(self):
        """Test generation of command fixes."""
        error_data = {"message": "command not found: gitx", "file_path": "test.sh"}

        analysis = {
            "root_cause": "bash_command_error",
            "subcategory": "command",
            "confidence": "high",
        }

        command_patch = self.generator.generate_patch(error_data, analysis, "")

        assert command_patch is not None
        assert command_patch["type"] == "suggestion"
        assert "command" in command_patch["description"].lower()

    def test_generate_variable_fix(self):
        """Test generation of variable fixes."""
        error_data = {"message": "unbound variable: MY_VAR", "file_path": "test.sh"}

        analysis = {
            "root_cause": "bash_variable_error",
            "subcategory": "variable",
            "confidence": "high",
        }

        variable_patch = self.generator.generate_patch(error_data, analysis, "")

        assert variable_patch is not None
        assert variable_patch["type"] == "suggestion"
        assert "variable" in variable_patch["description"].lower()

    def test_generate_permission_fix(self):
        """Test generation of permission fixes."""
        error_data = {"message": "Permission denied", "file_path": "test.sh"}

        analysis = {
            "root_cause": "bash_permission_error",
            "subcategory": "permission",
            "confidence": "high",
        }

        permission_patch = self.generator.generate_patch(error_data, analysis, "")

        assert permission_patch is not None
        assert permission_patch["type"] == "suggestion"
        assert (
            "permission" in permission_patch["description"].lower()
            or "chmod" in permission_patch["description"].lower()
        )

    def test_generate_file_fix(self):
        """Test generation of file fixes."""
        error_data = {"message": "No such file or directory", "file_path": "test.sh"}

        analysis = {
            "root_cause": "bash_file_error",
            "subcategory": "file",
            "confidence": "high",
        }

        file_patch = self.generator.generate_patch(error_data, analysis, "")

        assert file_patch is not None
        assert file_patch["type"] == "suggestion"
        assert (
            "file" in file_patch["description"].lower()
            or "directory" in file_patch["description"].lower()
        )

    def test_generate_expansion_fix(self):
        """Test generation of expansion fixes."""
        error_data = {"message": "bad substitution", "file_path": "test.sh"}

        analysis = {
            "root_cause": "bash_expansion_error",
            "subcategory": "expansion",
            "confidence": "high",
        }

        expansion_patch = self.generator.generate_patch(error_data, analysis, "")

        assert expansion_patch is not None
        assert expansion_patch["type"] == "suggestion"
        assert (
            "substitution" in expansion_patch["description"].lower()
            or "expansion" in expansion_patch["description"].lower()
        )

    def test_generate_pipe_fix(self):
        """Test generation of pipe fixes."""
        error_data = {"message": "Broken pipe", "file_path": "test.sh"}

        analysis = {
            "root_cause": "bash_pipe_error",
            "subcategory": "pipe",
            "confidence": "high",
        }

        pipe_patch = self.generator.generate_patch(error_data, analysis, "")

        assert pipe_patch is not None
        assert pipe_patch["type"] == "suggestion"
        assert "pipe" in pipe_patch["description"].lower()


class TestBashLanguagePlugin:
    """Test the Bash language plugin."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = BashLanguagePlugin()

    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "bash"
        assert self.plugin.get_language_name() == "Bash/Shell"
        assert self.plugin.get_language_version() == "5.0+"

        frameworks = self.plugin.get_supported_frameworks()
        assert "bash" in frameworks
        assert "sh" in frameworks
        assert "zsh" in frameworks

    def test_normalize_error(self):
        """Test error normalization."""
        bash_error = {
            "error_type": "BashError",
            "message": "Test error",
            "file": "test.sh",
            "line": 10,
            "column": 5,
            "description": "Test error description",
        }

        normalized = self.plugin.normalize_error(bash_error)

        assert normalized["language"] == "bash"
        assert normalized["error_type"] == "BashError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "test.sh"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5

    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "bash",
            "error_type": "BashError",
            "message": "Test error",
            "file_path": "test.sh",
            "line_number": 10,
            "column_number": 5,
            "severity": "high",
        }

        bash_error = self.plugin.denormalize_error(standard_error)

        assert bash_error["error_type"] == "BashError"
        assert bash_error["message"] == "Test error"
        assert bash_error["file_path"] == "test.sh"
        assert bash_error["line_number"] == 10
        assert bash_error["column_number"] == 5
        assert bash_error["file"] == "test.sh"  # Alternative format
        assert bash_error["line"] == 10  # Alternative format

    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "BashError",
            "message": "command not found: gitx",
            "file_path": "test.sh",
            "line_number": 15,
            "column_number": 1,
        }

        analysis = self.plugin.analyze_error(error_data)

        assert analysis["plugin"] == "bash"
        assert analysis["language"] == "bash"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "bash"
        assert analysis["subcategory"] == "command"

    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "bash_command_error",
            "subcategory": "command",
            "confidence": "high",
            "suggested_fix": "Fix command not found",
        }

        context = {
            "error_data": {
                "message": "command not found: gitx",
                "file_path": "test.sh",
            },
            "source_code": "#!/bin/bash\ngitx status",
        }

        fix = self.plugin.generate_fix(analysis, context)

        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix

    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".sh" in self.plugin.supported_extensions
        assert ".bash" in self.plugin.supported_extensions
        assert ".zsh" in self.plugin.supported_extensions
        assert ".ksh" in self.plugin.supported_extensions

    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}

        analysis = self.plugin.analyze_error(error_data)

        assert analysis["plugin"] == "bash"
        assert analysis["language"] == "bash"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])
