"""
Tests for the Haskell language plugin.
"""

import pytest

from modules.analysis.plugins.haskell_plugin import (
    HaskellExceptionHandler,
    HaskellLanguagePlugin,
    HaskellPatchGenerator,
)


class TestHaskellExceptionHandler:
    """Test the Haskell exception handler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = HaskellExceptionHandler()

    def test_analyze_syntax_error(self):
        """Test analysis of syntax errors."""
        error_data = {
            "error_type": "HaskellError",
            "message": "parse error on input 'where'",
            "file_path": "test.hs",
            "line_number": 10,
            "column_number": 5,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "haskell"
        assert analysis["subcategory"] == "syntax"
        assert analysis["confidence"] == "high"
        assert "syntax" in analysis["tags"]

    def test_analyze_type_error(self):
        """Test analysis of type errors."""
        error_data = {
            "error_type": "HaskellError",
            "message": "Couldn't match expected type 'Int' with actual type 'String'",
            "file_path": "test.hs",
            "line_number": 15,
            "column_number": 8,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "haskell"
        assert analysis["subcategory"] == "type"
        assert analysis["confidence"] == "high"
        assert "type" in analysis["tags"]

    def test_analyze_pattern_error(self):
        """Test analysis of pattern match errors."""
        error_data = {
            "error_type": "HaskellError",
            "message": "Non-exhaustive patterns in function myFunc",
            "file_path": "test.hs",
            "line_number": 20,
            "column_number": 12,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "haskell"
        assert analysis["subcategory"] == "pattern"
        assert analysis["confidence"] == "high"
        assert "pattern" in analysis["tags"]

    def test_analyze_monad_error(self):
        """Test analysis of monad errors."""
        error_data = {
            "error_type": "HaskellError",
            "message": "No instance for (Monad m) arising from a use of",
            "file_path": "test.hs",
            "line_number": 25,
            "column_number": 15,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "haskell"
        assert analysis["subcategory"] == "monad"
        assert analysis["confidence"] == "high"
        assert "monad" in analysis["tags"]

    def test_analyze_import_error(self):
        """Test analysis of import errors."""
        error_data = {
            "error_type": "HaskellError",
            "message": "Could not find module 'Data.MyModule'",
            "file_path": "test.hs",
            "line_number": 1,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "haskell"
        assert analysis["subcategory"] == "import"
        assert analysis["confidence"] == "high"
        assert "import" in analysis["tags"]

    def test_analyze_laziness_error(self):
        """Test analysis of laziness errors."""
        error_data = {
            "error_type": "HaskellError",
            "message": "*** Exception: Prelude.undefined",
            "file_path": "test.hs",
            "line_number": 30,
            "column_number": 10,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "haskell"
        assert analysis["subcategory"] == "laziness"
        assert analysis["confidence"] == "high"
        assert "laziness" in analysis["tags"]

    def test_analyze_typeclass_error(self):
        """Test analysis of type class errors."""
        error_data = {
            "error_type": "HaskellError",
            "message": "No instance for (Show a) arising from",
            "file_path": "test.hs",
            "line_number": 35,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "haskell"
        assert analysis["subcategory"] == "typeclass"
        assert analysis["confidence"] == "high"
        assert "typeclass" in analysis["tags"]

    def test_analyze_unknown_error(self):
        """Test analysis of unknown errors."""
        error_data = {
            "error_type": "HaskellError",
            "message": "Some unknown error message",
            "file_path": "test.hs",
            "line_number": 45,
            "column_number": 1,
        }

        analysis = self.handler.analyze_exception(error_data)

        assert analysis["category"] == "haskell"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"
        assert "generic" in analysis["tags"]


class TestHaskellPatchGenerator:
    """Test the Haskell patch generator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = HaskellPatchGenerator()

    def test_generate_syntax_fix(self):
        """Test generation of syntax fixes."""
        error_data = {"message": "parse error on input 'where'", "file_path": "test.hs"}

        analysis = {
            "root_cause": "haskell_syntax_error",
            "subcategory": "syntax",
            "confidence": "high",
        }

        syntax_patch = self.generator.generate_patch(error_data, analysis, "")

        assert syntax_patch is not None
        assert syntax_patch["type"] == "suggestion"
        assert "syntax" in syntax_patch["description"].lower()

    def test_generate_type_fix(self):
        """Test generation of type fixes."""
        error_data = {
            "message": "Couldn't match expected type 'Int' with actual type 'String'",
            "file_path": "test.hs",
        }

        analysis = {
            "root_cause": "haskell_type_error",
            "subcategory": "type",
            "confidence": "high",
        }

        type_patch = self.generator.generate_patch(error_data, analysis, "")

        assert type_patch is not None
        assert type_patch["type"] == "suggestion"
        assert "type" in type_patch["description"].lower()

    def test_generate_pattern_fix(self):
        """Test generation of pattern match fixes."""
        error_data = {
            "message": "Non-exhaustive patterns in function myFunc",
            "file_path": "test.hs",
        }

        analysis = {
            "root_cause": "haskell_pattern_error",
            "subcategory": "pattern",
            "confidence": "high",
        }

        pattern_patch = self.generator.generate_patch(error_data, analysis, "")

        assert pattern_patch is not None
        assert pattern_patch["type"] == "suggestion"
        assert "pattern" in pattern_patch["description"].lower()

    def test_generate_monad_fix(self):
        """Test generation of monad fixes."""
        error_data = {
            "message": "No instance for (Monad m) arising from a use of",
            "file_path": "test.hs",
        }

        analysis = {
            "root_cause": "haskell_monad_error",
            "subcategory": "monad",
            "confidence": "high",
        }

        monad_patch = self.generator.generate_patch(error_data, analysis, "")

        assert monad_patch is not None
        assert monad_patch["type"] == "suggestion"
        assert "monad" in monad_patch["description"].lower()

    def test_generate_import_fix(self):
        """Test generation of import fixes."""
        error_data = {
            "message": "Could not find module 'Data.MyModule'",
            "file_path": "test.hs",
        }

        analysis = {
            "root_cause": "haskell_import_error",
            "subcategory": "import",
            "confidence": "high",
        }

        import_patch = self.generator.generate_patch(error_data, analysis, "")

        assert import_patch is not None
        assert import_patch["type"] == "suggestion"
        assert (
            "import" in import_patch["description"].lower()
            or "module" in import_patch["description"].lower()
        )

    def test_generate_laziness_fix(self):
        """Test generation of laziness fixes."""
        error_data = {
            "message": "*** Exception: Prelude.undefined",
            "file_path": "test.hs",
        }

        analysis = {
            "root_cause": "haskell_lazy_error",
            "subcategory": "laziness",
            "confidence": "high",
        }

        laziness_patch = self.generator.generate_patch(error_data, analysis, "")

        assert laziness_patch is not None
        assert laziness_patch["type"] == "suggestion"
        assert (
            "undefined" in laziness_patch["description"].lower()
            or "laziness" in laziness_patch["description"].lower()
        )

    def test_generate_typeclass_fix(self):
        """Test generation of type class fixes."""
        error_data = {
            "message": "No instance for (Show a) arising from",
            "file_path": "test.hs",
        }

        analysis = {
            "root_cause": "haskell_type_class_error",
            "subcategory": "typeclass",
            "confidence": "high",
        }

        typeclass_patch = self.generator.generate_patch(error_data, analysis, "")

        assert typeclass_patch is not None
        assert typeclass_patch["type"] == "suggestion"
        assert (
            "instance" in typeclass_patch["description"].lower()
            or "typeclass" in typeclass_patch["description"].lower()
        )


class TestHaskellLanguagePlugin:
    """Test the Haskell language plugin."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = HaskellLanguagePlugin()

    def test_plugin_metadata(self):
        """Test plugin metadata."""
        assert self.plugin.get_language_id() == "haskell"
        assert self.plugin.get_language_name() == "Haskell"
        assert self.plugin.get_language_version() == "GHC 9.0+"

        frameworks = self.plugin.get_supported_frameworks()
        assert "haskell" in frameworks
        assert "stack" in frameworks
        assert "cabal" in frameworks

    def test_normalize_error(self):
        """Test error normalization."""
        haskell_error = {
            "error_type": "HaskellError",
            "message": "Test error",
            "file": "test.hs",
            "line": 10,
            "column": 5,
            "description": "Test error description",
        }

        normalized = self.plugin.normalize_error(haskell_error)

        assert normalized["language"] == "haskell"
        assert normalized["error_type"] == "HaskellError"
        assert normalized["message"] == "Test error"
        assert normalized["file_path"] == "test.hs"
        assert normalized["line_number"] == 10
        assert normalized["column_number"] == 5

    def test_denormalize_error(self):
        """Test error denormalization."""
        standard_error = {
            "language": "haskell",
            "error_type": "HaskellError",
            "message": "Test error",
            "file_path": "test.hs",
            "line_number": 10,
            "column_number": 5,
            "severity": "high",
        }

        haskell_error = self.plugin.denormalize_error(standard_error)

        assert haskell_error["error_type"] == "HaskellError"
        assert haskell_error["message"] == "Test error"
        assert haskell_error["file_path"] == "test.hs"
        assert haskell_error["line_number"] == 10
        assert haskell_error["column_number"] == 5
        assert haskell_error["file"] == "test.hs"  # Alternative format
        assert haskell_error["line"] == 10  # Alternative format

    def test_analyze_error(self):
        """Test error analysis."""
        error_data = {
            "error_type": "HaskellError",
            "message": "Couldn't match expected type 'Int' with actual type 'String'",
            "file_path": "test.hs",
            "line_number": 15,
            "column_number": 8,
        }

        analysis = self.plugin.analyze_error(error_data)

        assert analysis["plugin"] == "haskell"
        assert analysis["language"] == "haskell"
        assert analysis["plugin_version"] == "1.0.0"
        assert analysis["category"] == "haskell"
        assert analysis["subcategory"] == "type"

    def test_generate_fix(self):
        """Test fix generation."""
        analysis = {
            "root_cause": "haskell_type_error",
            "subcategory": "type",
            "confidence": "high",
            "suggested_fix": "Fix type mismatch",
        }

        context = {
            "error_data": {
                "message": "Couldn't match expected type 'Int' with actual type 'String'",
                "file_path": "test.hs",
            },
            "source_code": 'myFunc :: Int -> Int\nmyFunc x = "hello"',
        }

        fix = self.plugin.generate_fix(analysis, context)

        assert fix is not None
        assert fix["type"] == "suggestion"
        assert "description" in fix

    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert ".hs" in self.plugin.supported_extensions
        assert ".lhs" in self.plugin.supported_extensions
        assert ".cabal" in self.plugin.supported_extensions

    def test_error_analysis_with_invalid_data(self):
        """Test error analysis with invalid data."""
        error_data = {}

        analysis = self.plugin.analyze_error(error_data)

        assert analysis["plugin"] == "haskell"
        assert analysis["language"] == "haskell"
        # Should handle invalid data gracefully
        assert "category" in analysis
        assert "confidence" in analysis


if __name__ == "__main__":
    pytest.main([__file__])
