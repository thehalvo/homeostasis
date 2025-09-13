"""
Unit tests for the analyzer module.
"""

import pytest

from modules.analysis.analyzer import (
    AnalysisStrategy,
    Analyzer,
    analyze_error_from_log,
    get_available_ai_models,
    get_available_ml_modes,
    get_available_strategies,
)


class TestAnalyzer:
    """Test the Analyzer class."""

    def test_analyzer_initialization(self):
        """Test analyzer initialization with different strategies."""
        # Test default initialization
        analyzer = Analyzer()
        assert analyzer.strategy == AnalysisStrategy.RULE_BASED_ONLY
        assert not analyzer.use_ai

        # Test with AI fallback
        analyzer = Analyzer(strategy=AnalysisStrategy.AI_FALLBACK)
        assert analyzer.strategy == AnalysisStrategy.AI_FALLBACK
        assert analyzer.use_ai

        # Test with ML-based strategy
        analyzer = Analyzer(strategy=AnalysisStrategy.ML_BASED)
        assert analyzer.strategy == AnalysisStrategy.ML_BASED

    def test_analyze_error_rule_based(self):
        """Test rule-based error analysis."""
        analyzer = Analyzer(strategy=AnalysisStrategy.RULE_BASED_ONLY)

        # Test KeyError analysis
        error_data = {
            "timestamp": "2023-01-01T12:00:00",
            "service": "test_service",
            "level": "ERROR",
            "message": "KeyError: 'user_id'",
            "exception_type": "KeyError",
            "traceback": ["KeyError: 'user_id'"],
        }

        result = analyzer.analyze_error(error_data)
        assert result["analysis_method"] == "rule_based"
        assert "root_cause" in result
        assert result["confidence_score"] > 0

    def test_analyze_errors_batch(self):
        """Test batch error analysis."""
        analyzer = Analyzer()

        error_list = [
            {
                "timestamp": "2023-01-01T12:00:00",
                "message": "KeyError: 'test'",
                "exception_type": "KeyError",
            },
            {
                "timestamp": "2023-01-01T12:01:00",
                "message": "TypeError: unsupported operand",
                "exception_type": "TypeError",
            },
        ]

        results = analyzer.analyze_errors(error_list)
        assert len(results) == 2
        assert all("analysis_method" in r for r in results)


class TestAnalyzerUtilities:
    """Test analyzer utility functions."""

    def test_get_available_strategies(self):
        """Test getting available strategies."""
        strategies = get_available_strategies()
        assert isinstance(strategies, list)
        assert AnalysisStrategy.RULE_BASED_ONLY in strategies
        assert AnalysisStrategy.AI_FALLBACK in strategies
        assert AnalysisStrategy.ML_BASED in strategies
        assert AnalysisStrategy.HYBRID in strategies

    def test_get_available_ai_models(self):
        """Test getting available AI models."""
        models = get_available_ai_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_get_available_ml_modes(self):
        """Test getting available ML modes."""
        modes = get_available_ml_modes()
        assert isinstance(modes, list)
        assert len(modes) > 0

    def test_analyze_error_from_log(self):
        """Test analyzing error from log utility function."""
        error_log = {
            "timestamp": "2023-01-01T12:00:00",
            "message": "NameError: name 'undefined_var' is not defined",
            "exception_type": "NameError",
        }

        result = analyze_error_from_log(error_log)
        assert isinstance(result, dict)
        assert "analysis_method" in result
        assert result["analysis_method"] == "rule_based"


@pytest.mark.parametrize(
    "strategy",
    [
        AnalysisStrategy.RULE_BASED_ONLY,
        AnalysisStrategy.AI_FALLBACK,
        AnalysisStrategy.AI_ENHANCED,
        AnalysisStrategy.AI_PRIMARY,
        AnalysisStrategy.ML_BASED,
        AnalysisStrategy.HYBRID,
    ],
)
def test_all_strategies_initialization(strategy):
    """Test that all strategies can be initialized."""
    analyzer = Analyzer(strategy=strategy)
    assert analyzer.strategy == strategy
    assert isinstance(analyzer.analyze_error({"message": "test"}), dict)
