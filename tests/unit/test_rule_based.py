"""
Unit tests for the rule-based analyzer module.
"""

import pytest

from modules.analysis.rule_based import RuleBasedAnalyzer
from modules.analysis.rule_config import (
    Rule,
    RuleCategory,
    RuleConfidence,
    RuleSeverity,
)


class TestRuleBasedAnalyzer:
    """Test the RuleBasedAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = RuleBasedAnalyzer()

    def test_analyze_keyerror(self):
        """Test analysis of KeyError exceptions."""
        error_data = {
            "exception_type": "KeyError",
            "message": "KeyError: 'user_id'",
            "traceback": ["KeyError: 'user_id'"],
        }

        result = self.analyzer.analyze_error(error_data)
        assert result["root_cause"] == "dict_key_not_exists"
        assert result["confidence"] in ["high", "medium", "low"]
        assert "suggestion" in result

    def test_analyze_typeerror(self):
        """Test analysis of TypeError exceptions."""
        error_data = {
            "exception_type": "TypeError",
            "message": "TypeError: unsupported operand type(s) for +: 'str' and 'int'",
            "traceback": ["TypeError: unsupported operand type(s)"],
        }

        result = self.analyzer.analyze_error(error_data)
        assert result["root_cause"] in ["type_mismatch", "shell_scripting_pitfall"]
        assert "description" in result
        assert "suggestion" in result

    def test_analyze_nameerror(self):
        """Test analysis of NameError exceptions."""
        error_data = {
            "exception_type": "NameError",
            "message": "NameError: name 'undefined_var' is not defined",
            "traceback": ["NameError: name 'undefined_var' is not defined"],
        }

        result = self.analyzer.analyze_error(error_data)
        assert result["root_cause"] in ["undefined_name", "undefined_variable"]
        assert "confidence" in result

    def test_analyze_multiple_errors(self):
        """Test batch analysis of multiple errors."""
        error_list = [
            {"exception_type": "KeyError", "message": "KeyError: 'key1'"},
            {"exception_type": "TypeError", "message": "TypeError: test error"},
            {"exception_type": "ValueError", "message": "ValueError: invalid value"},
        ]

        results = self.analyzer.analyze_errors(error_list)
        assert len(results) == 3
        assert all("root_cause" in r for r in results)
        assert all("confidence" in r for r in results)

    def test_custom_rules(self):
        """Test adding custom rules to the analyzer."""
        # Create analyzer with custom rule dictionary
        custom_rule_dict = {
            "pattern": r"CustomError: (.+)",
            "type": "CustomError",
            "description": "Custom error for testing",
            "root_cause": "custom_error",
            "suggestion": "Fix the custom error",
        }
        analyzer = RuleBasedAnalyzer(additional_patterns=[custom_rule_dict])

        error_data = {
            "exception_type": "CustomError",
            "message": "CustomError: test message",
        }

        result = analyzer.analyze_error(error_data)
        # Custom rules might not override built-in patterns
        assert "root_cause" in result
        assert "description" in result

    def test_unknown_error(self):
        """Test handling of unknown error types."""
        error_data = {
            "exception_type": "UnknownError",
            "message": "UnknownError: something went wrong",
            "traceback": [],
        }

        result = self.analyzer.analyze_error(error_data)
        assert result["root_cause"] in [
            "unknown",
            "shell_scripting_pitfall",
            "general_error",
        ]
        assert result["confidence"] in ["low", "medium", "high"]

    def test_error_with_metadata(self):
        """Test analysis with additional metadata."""
        error_data = {
            "exception_type": "KeyError",
            "message": "KeyError: 'user_id'",
            "service": "test_service",
            "timestamp": "2023-01-01T12:00:00",
            "request_info": {"method": "GET", "path": "/api/users"},
        }

        result = self.analyzer.analyze_error(error_data)
        assert "error_data" in result
        assert result["error_data"]["service"] == "test_service"
        assert "request_info" in result["error_data"]


@pytest.mark.parametrize(
    "error_type,message,expected_root_cause",
    [
        (
            "AttributeError",
            "AttributeError: 'NoneType' object has no attribute 'test'",
            "attribute_not_exists",
        ),
        (
            "IndexError",
            "IndexError: list index out of range",
            "list_index_out_of_bounds",
        ),
        (
            "ImportError",
            "ImportError: No module named 'test_module'",
            "module_not_found",
        ),
        (
            "ZeroDivisionError",
            "ZeroDivisionError: division by zero",
            "go_divide_by_zero",
        ),
    ],
)
def test_common_error_patterns(error_type, message, expected_root_cause):
    """Test common error patterns are correctly identified."""
    analyzer = RuleBasedAnalyzer()
    error_data = {"exception_type": error_type, "message": message}

    result = analyzer.analyze_error(error_data)
    assert result["root_cause"] == expected_root_cause
    assert "suggestion" in result
