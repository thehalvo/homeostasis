"""
Integration tests for the analyzer module.
"""

import pytest

from modules.analysis.analyzer import AnalysisStrategy, Analyzer


class TestAnalyzerIntegration:
    """Integration tests for the analyzer with other modules."""

    def test_rule_based_analysis_flow(self):
        """Test complete rule-based analysis flow."""
        analyzer = Analyzer(strategy=AnalysisStrategy.RULE_BASED_ONLY)

        # Simulate a complex error scenario
        error_data = {
            "timestamp": "2023-01-01T12:00:00",
            "service": "api_service",
            "level": "ERROR",
            "message": "KeyError: 'user_id'",
            "exception_type": "KeyError",
            "traceback": [
                "Traceback (most recent call last):",
                '  File "/app/api/handlers.py", line 42, in get_user',
                "    user_id = request_data['user_id']",
                "KeyError: 'user_id'",
            ],
            "error_details": {
                "exception_type": "KeyError",
                "message": "'user_id'",
                "detailed_frames": [
                    {
                        "file": "/app/api/handlers.py",
                        "line": 42,
                        "function": "get_user",
                        "locals": {"request_data": {"username": "test"}},
                    }
                ],
            },
        }

        result = analyzer.analyze_error(error_data)

        # Verify the analysis result
        assert result["analysis_method"] == "rule_based"
        assert result["root_cause"] == "dict_key_not_exists"
        assert result["confidence_score"] > 0.5
        assert "suggestion" in result
        assert "dict" in result.get("description", "") or "key" in result.get(
            "description", ""
        )

    def test_ml_based_analysis_flow(self):
        """Test ML-based analysis flow."""
        analyzer = Analyzer(strategy=AnalysisStrategy.ML_BASED, ml_mode="parallel")

        # Test with a structured error
        error_data = {
            "timestamp": "2023-01-01T12:00:00",
            "message": "TypeError: can't multiply sequence by non-int of type 'str'",
            "exception_type": "TypeError",
            "traceback": ["TypeError: can't multiply sequence by non-int"],
        }

        result = analyzer.analyze_error(error_data)

        # ML analyzer should provide structured output
        assert "error_type" in result or "root_cause" in result
        assert "confidence" in result or "confidence_score" in result

    def test_hybrid_analysis_flow(self):
        """Test hybrid analysis flow."""
        analyzer = Analyzer(strategy=AnalysisStrategy.HYBRID, use_llm=False)

        # Test with an ambiguous error
        error_data = {
            "timestamp": "2023-01-01T12:00:00",
            "message": "Error: Operation failed",
            "exception_type": "RuntimeError",
            "traceback": ["RuntimeError: Operation failed"],
        }

        result = analyzer.analyze_error(error_data)

        # Hybrid analyzer should combine multiple approaches
        assert isinstance(result, dict)
        assert any(
            key in result for key in ["error_type", "root_cause", "primary_method"]
        )

    def test_error_analysis_with_context(self):
        """Test error analysis with rich context."""
        analyzer = Analyzer(strategy=AnalysisStrategy.AI_FALLBACK)

        # Error with full context
        error_data = {
            "timestamp": "2023-01-01T12:00:00",
            "service": "payment_service",
            "level": "ERROR",
            "message": "AttributeError: 'NoneType' object has no attribute 'process'",
            "exception_type": "AttributeError",
            "traceback": [
                "Traceback (most recent call last):",
                '  File "/app/payment/processor.py", line 156, in process_payment',
                "    result = payment_gateway.process(amount)",
                "AttributeError: 'NoneType' object has no attribute 'process'",
            ],
            "request_info": {
                "method": "POST",
                "path": "/api/payments",
                "headers": {"Content-Type": "application/json"},
            },
            "error_details": {
                "exception_type": "AttributeError",
                "message": "'NoneType' object has no attribute 'process'",
                "detailed_frames": [
                    {
                        "file": "/app/payment/processor.py",
                        "line": 156,
                        "function": "process_payment",
                        "locals": {
                            "amount": 100.0,
                            "payment_gateway": None,
                            "user_id": "usr_123",
                        },
                    }
                ],
            },
        }

        result = analyzer.analyze_error(error_data)

        # Should provide detailed analysis
        assert result["root_cause"] in [
            "none_type_error",
            "uninitialized_variable",
            "attribute_not_exists",
        ]
        assert "suggestion" in result
        assert "attribute" in result.get("description", "").lower()


class TestErrorExtractorIntegration:
    """Test integration with error extractor module."""

    def test_analyze_extracted_errors(self):
        """Test analyzing errors with analyzer."""
        analyzer = Analyzer()

        # Create mock errors
        errors = [
            {
                "timestamp": "2023-01-01T12:00:00",
                "service": "test_service",
                "level": "ERROR",
                "message": "KeyError: 'config'",
                "exception_type": "KeyError",
            },
            {
                "timestamp": "2023-01-01T12:01:00",
                "service": "test_service",
                "level": "ERROR",
                "message": "ValueError: invalid literal for int()",
                "exception_type": "ValueError",
            },
        ]

        # Analyze each error
        results = analyzer.analyze_errors(errors)

        # Verify results structure
        assert len(results) == 2
        assert all("root_cause" in r for r in results)
        assert all("confidence" in r for r in results)

        # Check specific error types were analyzed
        error_types = [r.get("error_data", {}).get("exception_type") for r in results]
        assert "KeyError" in error_types
        assert "ValueError" in error_types


@pytest.mark.parametrize(
    "strategy,expected_keys",
    [
        (
            AnalysisStrategy.RULE_BASED_ONLY,
            ["root_cause", "confidence", "analysis_method"],
        ),
        (
            AnalysisStrategy.ML_BASED,
            ["error_type", "confidence", "analysis_method"],  # ML analyzer specific
        ),
        (
            AnalysisStrategy.HYBRID,
            ["primary_method", "error_type"],  # Hybrid specific
        ),
    ],
)
def test_strategy_specific_outputs(strategy, expected_keys):
    """Test that different strategies produce expected output formats."""
    analyzer = Analyzer(strategy=strategy)

    error_data = {
        "timestamp": "2023-01-01T12:00:00",
        "message": "Test error message",
        "exception_type": "RuntimeError",
    }

    result = analyzer.analyze_error(error_data)

    # Check for strategy-specific keys
    for key in expected_keys:
        assert key in result or any(
            key in str(v) for v in result.values() if isinstance(v, (str, dict))
        )
