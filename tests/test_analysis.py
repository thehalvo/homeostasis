"""
Tests for the analysis module.
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.analysis.analyzer import Analyzer
from modules.analysis.rule_based import RuleBasedAnalyzer
from modules.analysis.rule_config import get_all_rule_sets, load_rule_configs


def test_analyzer_initialization():
    """Test initializing the Analyzer with default settings."""
    analyzer = Analyzer()
    assert analyzer.use_ai is False  # Default is rule-based only
    assert isinstance(analyzer.rule_based_analyzer, RuleBasedAnalyzer)


def test_analyzer_initialization_with_ai():
    """Test initializing the Analyzer with AI enabled."""
    analyzer = Analyzer(use_ai=True)
    assert analyzer.use_ai is True


def test_analyze_errors_rule_based():
    """Test analyzing errors with rule-based engine."""
    # Create a mock error
    mock_error = {
        "level": "ERROR",
        "message": "KeyError: 'id'",
        "exception": "KeyError",
        "traceback": "File 'app.py', line 100\n    return data['id']",
        "service": "example_service"
    }
    
    # Create a mock RuleBasedAnalyzer that returns a specific result
    mock_result = {
        "root_cause": "key_error",
        "confidence": 0.9,
        "rule_id": "key_error_rule",
        "error_message": "KeyError: 'id'",
        "suggested_fix": "Check if 'id' key exists before accessing"
    }
    
    with patch("modules.analysis.rule_based.RuleBasedAnalyzer.analyze_error", return_value=mock_result):
        analyzer = Analyzer()
        results = analyzer.analyze_errors([mock_error])
        
        assert len(results) == 1
        assert results[0] == mock_result


def test_analyze_errors_ai_based():
    """Test analyzing errors with AI engine."""
    # Create a mock error
    mock_error = {
        "level": "ERROR",
        "message": "TypeError: cannot convert 'str' to 'int'",
        "exception": "TypeError",
        "traceback": "File 'app.py', line 120\n    value = int(input_value)",
        "service": "example_service"
    }
    
    # Create mock results for both analyzers
    mock_rule_result = {
        "root_cause": "type_error",
        "confidence": 0.6,
        "rule_id": "type_error_rule",
        "error_message": "TypeError: cannot convert 'str' to 'int'",
        "suggested_fix": "Try converting with error handling"
    }
    
    mock_ai_result = {
        "root_cause": "string_to_int_conversion_error",
        "confidence": 0.9,
        "analysis": "AI analysis shows integer conversion error",
        "suggested_fix": "Add try-except block around the conversion"
    }
    
    with patch("modules.analysis.rule_based.RuleBasedAnalyzer.analyze_error", return_value=mock_rule_result):
        with patch("modules.analysis.ai_stub.analyze_error", return_value=mock_ai_result):
            analyzer = Analyzer(use_ai=True)
            results = analyzer.analyze_errors([mock_error])
            
            assert len(results) == 1
            # AI result should be chosen due to higher confidence
            assert results[0] == mock_ai_result


def test_analyze_errors_mixed_results():
    """Test analyzing errors with mixed rule-based and AI results."""
    # Create two mock errors
    mock_errors = [
        {
            "level": "ERROR",
            "message": "KeyError: 'id'",
            "exception": "KeyError",
            "traceback": "File 'app.py', line 100\n    return data['id']",
            "service": "example_service"
        },
        {
            "level": "ERROR",
            "message": "TypeError: cannot convert 'str' to 'int'",
            "exception": "TypeError",
            "traceback": "File 'app.py', line 120\n    value = int(input_value)",
            "service": "example_service"
        }
    ]
    
    # Create mock results
    mock_rule_results = [
        {
            "root_cause": "key_error",
            "confidence": 0.9,
            "rule_id": "key_error_rule",
            "error_message": "KeyError: 'id'",
            "suggested_fix": "Check if 'id' key exists before accessing"
        },
        {
            "root_cause": "type_error",
            "confidence": 0.6,
            "rule_id": "type_error_rule",
            "error_message": "TypeError: cannot convert 'str' to 'int'",
            "suggested_fix": "Try converting with error handling"
        }
    ]
    
    mock_ai_results = [
        None,  # AI couldn't analyze the first error
        {
            "root_cause": "string_to_int_conversion_error",
            "confidence": 0.9,
            "analysis": "AI analysis shows integer conversion error",
            "suggested_fix": "Add try-except block around the conversion"
        }
    ]
    
    # Mock the analyze_error method for both analyzers to return the corresponding results
    with patch("modules.analysis.rule_based.RuleBasedAnalyzer.analyze_error", side_effect=mock_rule_results):
        with patch("modules.analysis.ai_stub.analyze_error", side_effect=mock_ai_results):
            analyzer = Analyzer(use_ai=True)
            results = analyzer.analyze_errors(mock_errors)
            
            assert len(results) == 2
            assert results[0] == mock_rule_results[0]  # Rule-based result for first error
            assert results[1] == mock_ai_results[1]  # AI result for second error


def test_rule_based_analyzer_initialization():
    """Test initializing the RuleBasedAnalyzer."""
    analyzer = RuleBasedAnalyzer()
    assert hasattr(analyzer, "rules")


def test_rule_based_analyzer_analyze_error():
    """Test analyzing an error with RuleBasedAnalyzer."""
    # Create a mock error
    mock_error = {
        "level": "ERROR",
        "message": "KeyError: 'id'",
        "exception": "KeyError",
        "traceback": "File 'app.py', line 100\n    return data['id']",
        "service": "example_service"
    }
    
    # Create a mock rule
    mock_rule = {
        "id": "key_error_rule",
        "pattern": "KeyError: '(.*?)'",
        "root_cause": "key_error",
        "confidence": 0.9,
        "suggested_fix": "Check if '{1}' key exists before accessing"
    }
    
    # Mock the rules to include only our test rule
    with patch.object(RuleBasedAnalyzer, "rules", [mock_rule]):
        analyzer = RuleBasedAnalyzer()
        result = analyzer.analyze_error(mock_error)
        
        assert result is not None
        assert result["root_cause"] == "key_error"
        assert result["rule_id"] == "key_error_rule"
        assert result["confidence"] == 0.9
        assert "Check if 'id' key exists before accessing" in result["suggested_fix"]


def test_rule_based_analyzer_analyze_error_no_match():
    """Test analyzing an error with no matching rule."""
    # Create a mock error with no matching rule
    mock_error = {
        "level": "ERROR",
        "message": "CustomError: something unusual happened",
        "exception": "CustomError",
        "traceback": "File 'app.py', line 200\n    raise CustomError('something unusual happened')",
        "service": "example_service"
    }
    
    # Create a mock rule that won't match
    mock_rule = {
        "id": "key_error_rule",
        "pattern": "KeyError: '(.*?)'",
        "root_cause": "key_error",
        "confidence": 0.9,
        "suggested_fix": "Check if '{1}' key exists before accessing"
    }
    
    # Mock the rules to include only our test rule
    with patch.object(RuleBasedAnalyzer, "rules", [mock_rule]):
        analyzer = RuleBasedAnalyzer()
        result = analyzer.analyze_error(mock_error)
        
        assert result is None  # No matching rule


def test_load_rule_configs():
    """Test loading rule configurations from files."""
    # Create a temporary rule file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump([
            {
                "id": "test_rule",
                "pattern": "TestError: (.*)",
                "root_cause": "test_error",
                "confidence": 0.8,
                "suggested_fix": "Fix the test error: {1}"
            }
        ], temp_file)
        rule_path = Path(temp_file.name)
    
    try:
        # Mock the glob to return our temporary file
        with patch("glob.glob", return_value=[str(rule_path)]):
            rules = load_rule_configs()
            
            assert len(rules) == 1
            assert rules[0]["id"] == "test_rule"
            assert rules[0]["pattern"] == "TestError: (.*)"
            assert rules[0]["root_cause"] == "test_error"
            assert rules[0]["confidence"] == 0.8
            assert rules[0]["suggested_fix"] == "Fix the test error: {1}"
    
    finally:
        # Clean up
        if rule_path.exists():
            rule_path.unlink()