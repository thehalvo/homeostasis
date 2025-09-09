"""
Tests for CSS Framework Plugin

This module contains comprehensive tests for the CSS framework plugin,
covering Tailwind CSS, CSS-in-JS libraries, CSS Modules, layout debugging,
and animation error handling.
"""

from unittest.mock import patch

import pytest

# Import the CSS plugin components
from modules.analysis.plugins.css_plugin import (CSSExceptionHandler,
                                                 CSSLanguagePlugin,
                                                 CSSPatchGenerator)


class TestCSSLanguagePlugin:
    """Test the main CSS language plugin functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = CSSLanguagePlugin()

    def test_plugin_initialization(self):
        """Test that the CSS plugin initializes correctly."""
        assert self.plugin.get_language_id() == "css"
        assert self.plugin.get_language_name() == "CSS Frameworks"
        assert self.plugin.get_language_version() == "CSS3+"
        assert len(self.plugin.get_supported_frameworks()) > 0

    def test_can_handle_tailwind_errors(self):
        """Test that plugin can identify Tailwind CSS errors."""
        tailwind_error = {
            "message": "Unknown utility class: bg-blue-1000",
            "error_type": "CompilationError",
            "framework": "tailwindcss",
        }
        assert self.plugin.can_handle(tailwind_error) is True

    def test_can_handle_styled_components_errors(self):
        """Test that plugin can identify Styled Components errors."""
        styled_error = {
            "message": "styled-components: babel transform error",
            "error_type": "TransformError",
            "stack_trace": "at styled-components/babel",
        }
        assert self.plugin.can_handle(styled_error) is True

    def test_can_handle_emotion_errors(self):
        """Test that plugin can identify Emotion errors."""
        emotion_error = {
            "message": "emotion jsx pragma error",
            "error_type": "JSXError",
            "stack_trace": "@emotion/react",
        }
        assert self.plugin.can_handle(emotion_error) is True

    def test_can_handle_css_file_errors(self):
        """Test that plugin can identify CSS file errors."""
        css_error = {
            "message": "CSS compilation error",
            "error_type": "SyntaxError",
            "stack_trace": "at styles.scss:10:5",
        }
        assert self.plugin.can_handle(css_error) is True

    def test_cannot_handle_non_css_errors(self):
        """Test that plugin correctly rejects non-CSS errors."""
        non_css_error = {
            "message": "Python syntax error",
            "error_type": "SyntaxError",
            "stack_trace": "at main.py:15",
        }
        assert self.plugin.can_handle(non_css_error) is False

    def test_analyze_tailwind_error(self):
        """Test analysis of Tailwind CSS errors."""
        tailwind_error = {
            "message": "Unknown utility class: bg-invalid-color",
            "error_type": "CompilationError",
        }

        analysis = self.plugin.analyze_error(tailwind_error)

        assert analysis["category"] == "css"
        assert analysis["subcategory"] == "tailwind"
        assert analysis["plugin"] == "css"
        assert "tailwind" in analysis["tags"]

    def test_analyze_css_in_js_error(self):
        """Test analysis of CSS-in-JS errors."""
        css_in_js_error = {
            "message": "styled-components theme error",
            "error_type": "ThemeError",
        }

        analysis = self.plugin.analyze_error(css_in_js_error)

        assert analysis["category"] == "css"
        assert analysis["subcategory"] == "css_in_js"
        assert (
            "css-in-js" in analysis["tags"] or "styled-components" in analysis["tags"]
        )

    def test_analyze_layout_error(self):
        """Test analysis of CSS layout errors."""
        layout_error = {
            "message": "CSS Grid layout error: invalid grid-template",
            "error_type": "LayoutError",
        }

        analysis = self.plugin.analyze_error(layout_error)

        assert analysis["category"] == "css"
        assert analysis["subcategory"] == "layout"
        assert "grid" in analysis["tags"] or "layout" in analysis["tags"]


class TestCSSExceptionHandler:
    """Test the CSS exception handler functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = CSSExceptionHandler()

    def test_analyze_tailwind_unknown_class(self):
        """Test analysis of Tailwind unknown class errors."""
        error_data = {
            "message": "Unknown utility class: bg-purple-1000",
            "error_type": "CompilationError",
        }

        analysis = self.handler.analyze_tailwind_error(error_data)

        assert analysis["root_cause"] == "tailwind_unknown_class"
        assert analysis["confidence"] == "high"
        assert "Check Tailwind CSS class name" in analysis["suggested_fix"]

    def test_analyze_tailwind_purge_error(self):
        """Test analysis of Tailwind purge errors."""
        error_data = {
            "message": "Class bg-blue-500 was purged from build",
            "error_type": "PurgeError",
        }

        analysis = self.handler.analyze_tailwind_error(error_data)

        assert analysis["root_cause"] == "tailwind_purge_error"
        assert "purge configuration" in analysis["suggested_fix"]

    def test_analyze_styled_components_babel_error(self):
        """Test analysis of Styled Components Babel errors."""
        error_data = {
            "message": "babel-plugin-styled-components transform error",
            "error_type": "TransformError",
        }

        analysis = self.handler.analyze_css_in_js_error(error_data)

        assert analysis["root_cause"] == "styled_components_babel_error"
        assert "Babel plugin" in analysis["suggested_fix"]

    def test_analyze_emotion_jsx_error(self):
        """Test analysis of Emotion JSX errors."""
        error_data = {"message": "emotion jsx pragma missing", "error_type": "JSXError"}

        analysis = self.handler.analyze_css_in_js_error(error_data)

        assert analysis["root_cause"] == "emotion_jsx_error"
        assert "JSX pragma" in analysis["suggested_fix"]

    def test_analyze_css_grid_error(self):
        """Test analysis of CSS Grid errors."""
        error_data = {
            "message": "CSS Grid template error: invalid grid-template-columns",
            "error_type": "LayoutError",
        }

        analysis = self.handler.analyze_layout_error(error_data)

        assert analysis["root_cause"] == "css_grid_layout_error"
        assert "Grid" in analysis["suggested_fix"]

    def test_analyze_css_flexbox_error(self):
        """Test analysis of CSS Flexbox errors."""
        error_data = {
            "message": "CSS Flexbox error: invalid justify-content value",
            "error_type": "LayoutError",
        }

        analysis = self.handler.analyze_layout_error(error_data)

        assert analysis["root_cause"] == "css_flexbox_layout_error"
        assert "Flexbox" in analysis["suggested_fix"]

    def test_confidence_scoring(self):
        """Test confidence scoring for CSS error patterns."""
        high_confidence_error = {
            "message": "tailwind unknown class bg-invalid-color",
            "error_type": "CompilationError",
            "framework": "tailwindcss",
        }

        analysis = self.handler.analyze_exception(high_confidence_error)

        # Should have high confidence due to specific pattern match
        assert analysis["confidence"] in ["high", "medium"]

    def test_generic_analysis_fallback(self):
        """Test generic analysis for unmatched CSS errors."""
        generic_error = {
            "message": "CSS syntax error: unexpected token",
            "error_type": "SyntaxError",
        }

        analysis = self.handler.analyze_exception(generic_error)

        assert analysis["category"] == "css"
        assert analysis["subcategory"] == "unknown"
        assert analysis["confidence"] == "low"


class TestCSSPatchGenerator:
    """Test the CSS patch generator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = CSSPatchGenerator()

    def test_generate_tailwind_unknown_class_patch(self):
        """Test patch generation for Tailwind unknown class errors."""
        error_data = {"message": "Unknown utility class: bg-purple-1000"}
        analysis = {"root_cause": "tailwind_unknown_class"}
        source_code = ".container { @apply bg-purple-1000; }"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        assert patch is not None
        assert patch["type"] == "suggestion"
        assert "Tailwind CSS class name" in patch["description"]
        assert len(patch["fix_commands"]) > 0

    def test_generate_tailwind_purge_patch(self):
        """Test patch generation for Tailwind purge errors."""
        error_data = {"message": "Class was purged from build"}
        analysis = {"root_cause": "tailwind_purge_error"}
        source_code = "module.exports = { purge: [] }"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        assert patch is not None
        assert "purge configuration" in patch["description"]
        assert patch.get("template") == "tailwind_purge_fix"

    def test_generate_styled_components_babel_patch(self):
        """Test patch generation for Styled Components Babel errors."""
        error_data = {"message": "babel transform error"}
        analysis = {"root_cause": "styled_components_babel_error"}
        source_code = "const Button = styled.button``"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        assert patch is not None
        assert patch["type"] == "configuration"
        assert "Babel plugin" in patch["description"]
        assert "babel-plugin-styled-components" in patch["fix_code"]

    def test_generate_emotion_jsx_patch(self):
        """Test patch generation for Emotion JSX errors."""
        error_data = {"message": "jsx pragma error"}
        analysis = {"root_cause": "emotion_jsx_error"}
        source_code = "const Button = () => <div css={styles} />"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        assert patch is not None
        assert "JSX pragma" in patch["description"]
        assert "@jsx jsx" in patch["fix_code"]

    def test_generate_css_grid_patch(self):
        """Test patch generation for CSS Grid errors."""
        error_data = {"message": "grid layout error"}
        analysis = {"root_cause": "css_grid_layout_error"}
        source_code = ".container { display: grid; }"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        assert patch is not None
        assert "Grid layout" in patch["description"]
        assert patch.get("template") == "css_grid_fix"

    def test_generate_css_flexbox_patch(self):
        """Test patch generation for CSS Flexbox errors."""
        error_data = {"message": "flexbox error"}
        analysis = {"root_cause": "css_flexbox_layout_error"}
        source_code = ".container { display: flex; }"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        assert patch is not None
        assert "Flexbox layout" in patch["description"]
        assert patch.get("template") == "css_flexbox_fix"

    def test_template_based_patch_fallback(self):
        """Test template-based patch generation fallback."""
        error_data = {"message": "CSS error"}
        analysis = {"root_cause": "css_grid_layout_error"}
        source_code = ".test { display: grid; }"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        # Should fall back to template-based patch
        assert patch is not None
        if patch["type"] == "template":
            assert "template" in patch

    def test_no_patch_for_unknown_error(self):
        """Test that no patch is generated for unknown errors."""
        error_data = {"message": "Unknown error"}
        analysis = {"root_cause": "unknown_css_error"}
        source_code = ".test {}"

        patch = self.generator.generate_patch(error_data, analysis, source_code)

        # Should return None for unknown errors
        assert patch is None


class TestCSSRulesIntegration:
    """Test integration with CSS rules files."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = CSSExceptionHandler()

    def test_rules_loading(self):
        """Test that CSS rules are loaded correctly."""
        # Check that rules are loaded
        assert len(self.handler.rules) > 0

        # Check specific rule categories
        if "tailwind" in self.handler.rules:
            assert len(self.handler.rules["tailwind"]) > 0

        if "css_in_js" in self.handler.rules:
            assert len(self.handler.rules["css_in_js"]) > 0

    def test_pattern_compilation(self):
        """Test that regex patterns are compiled correctly."""
        assert len(self.handler.compiled_patterns) > 0

        # Check that patterns are compiled
        for category, patterns in self.handler.compiled_patterns.items():
            assert isinstance(patterns, list)
            for compiled_pattern, rule in patterns:
                assert hasattr(
                    compiled_pattern, "search"
                )  # Compiled regex has search method

    @patch("builtins.open")
    @patch("json.load")
    def test_rules_loading_error_handling(self, mock_json_load, mock_open):
        """Test error handling when rules files cannot be loaded."""
        mock_json_load.side_effect = Exception("File not found")

        # Should not raise exception, should handle gracefully
        handler = CSSExceptionHandler()
        assert handler.rules == {"tailwind": [], "css_in_js": [], "layout": []}


class TestCSSWorkflow:
    """Test end-to-end CSS error handling workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = CSSLanguagePlugin()

    def test_complete_tailwind_workflow(self):
        """Test complete workflow for Tailwind CSS error."""
        error_data = {
            "message": "Unknown utility class: bg-purple-1000",
            "error_type": "CompilationError",
            "framework": "tailwindcss",
        }
        source_code = ".button { @apply bg-purple-1000; }"

        # 1. Check if plugin can handle the error
        assert self.plugin.can_handle(error_data) is True

        # 2. Analyze the error
        analysis = self.plugin.analyze_error(error_data)
        assert analysis["category"] == "css"

        # 3. Generate fix
        fix = self.plugin.generate_fix(error_data, analysis, source_code)
        assert fix is not None
        assert "fix_commands" in fix or "fix_code" in fix

    def test_complete_css_in_js_workflow(self):
        """Test complete workflow for CSS-in-JS error."""
        error_data = {
            "message": "styled-components babel transform error",
            "error_type": "TransformError",
            "stack_trace": "babel-plugin-styled-components",
        }
        source_code = "const Button = styled.button`color: blue;`"

        # Complete workflow test
        assert self.plugin.can_handle(error_data) is True

        analysis = self.plugin.analyze_error(error_data)
        assert analysis["category"] == "css"

        fix = self.plugin.generate_fix(error_data, analysis, source_code)
        assert fix is not None

    def test_complete_layout_workflow(self):
        """Test complete workflow for CSS layout error."""
        error_data = {
            "message": "CSS Grid error: invalid grid-template-columns",
            "error_type": "LayoutError",
        }
        source_code = ".grid { display: grid; grid-template-columns: invalid; }"

        # Complete workflow test
        assert self.plugin.can_handle(error_data) is True

        analysis = self.plugin.analyze_error(error_data)
        assert analysis["category"] == "css"

        fix = self.plugin.generate_fix(error_data, analysis, source_code)
        assert fix is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
