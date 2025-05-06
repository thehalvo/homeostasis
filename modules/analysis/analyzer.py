"""
Analyzer interface for combining different analysis approaches.
"""
from typing import Dict, List, Optional, Any, Tuple

from .rule_based import RuleBasedAnalyzer, FASTAPI_ERROR_PATTERNS
from .ai_stub import AIAnalyzer


class Analyzer:
    """
    Unified analyzer that combines different analysis approaches.
    """

    def __init__(self, use_ai: bool = False):
        """
        Initialize the analyzer.

        Args:
            use_ai: Whether to use AI-based analysis
        """
        self.rule_analyzer = RuleBasedAnalyzer(additional_patterns=FASTAPI_ERROR_PATTERNS)
        self.ai_analyzer = AIAnalyzer() if use_ai else None
        self.use_ai = use_ai
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an error using available analyzers.

        Args:
            error_data: Error data to analyze

        Returns:
            Analysis results
        """
        # Always run rule-based analysis
        rule_analysis = self.rule_analyzer.analyze_error(error_data)
        
        # If using AI and rule-based confidence is low, try AI analysis
        if self.use_ai and rule_analysis["confidence"] == "low":
            ai_analysis = self.ai_analyzer.analyze_error(error_data)
            
            # Combine results, favoring the higher confidence result
            if ai_analysis["confidence"] != "none" and ai_analysis["confidence"] != "low":
                return {
                    **ai_analysis,
                    "rule_analysis": rule_analysis,
                    "analysis_method": "ai"
                }
        
        # Return rule-based analysis results
        return {
            **rule_analysis,
            "analysis_method": "rule_based"
        }
    
    def analyze_errors(self, error_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple errors.

        Args:
            error_data_list: List of error data to analyze

        Returns:
            List of analysis results
        """
        return [self.analyze_error(error_data) for error_data in error_data_list]


def analyze_error_from_log(error_log: Dict[str, Any], use_ai: bool = False) -> Dict[str, Any]:
    """
    Utility function to analyze a single error log entry.

    Args:
        error_log: Error log data
        use_ai: Whether to use AI-based analysis

    Returns:
        Analysis results
    """
    analyzer = Analyzer(use_ai=use_ai)
    return analyzer.analyze_error(error_log)


if __name__ == "__main__":
    # Example usage
    analyzer = Analyzer(use_ai=False)
    
    # Test with a sample error
    error_data = {
        "timestamp": "2023-01-01T12:00:00",
        "service": "example_service",
        "level": "ERROR",
        "message": "KeyError: 'todo_id'",
        "exception_type": "KeyError",
        "traceback": ["Traceback (most recent call last):", "  ...", "KeyError: 'todo_id'"]
    }
    
    analysis = analyzer.analyze_error(error_data)
    print(f"Analysis Method: {analysis['analysis_method']}")
    print(f"Root Cause: {analysis['root_cause']}")
    print(f"Description: {analysis['description']}")
    print(f"Suggestion: {analysis['suggestion']}")
    print(f"Confidence: {analysis['confidence']}")