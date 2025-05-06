"""
Rule-based error analysis module.
"""
import re
from typing import Dict, List, Optional, Any, Tuple

# Error patterns for common Python exceptions
ERROR_PATTERNS = [
    {
        "pattern": r"KeyError: '?([^']*)'?",
        "type": "KeyError",
        "description": "Accessing a dictionary key that doesn't exist",
        "root_cause": "dict_key_not_exists",
        "suggestion": "Check if the key exists before accessing it",
    },
    {
        "pattern": r"IndexError: list index out of range",
        "type": "IndexError",
        "description": "Accessing a list index that is out of bounds",
        "root_cause": "list_index_out_of_bounds",
        "suggestion": "Check the list length before accessing an index",
    },
    {
        "pattern": r"AttributeError: '([^']*)' object has no attribute '([^']*)'",
        "type": "AttributeError",
        "description": "Accessing an attribute that doesn't exist on an object",
        "root_cause": "attribute_not_exists",
        "suggestion": "Check if the attribute exists before accessing it",
    },
    {
        "pattern": r"TypeError: '([^']*)' object is not (subscriptable|iterable|callable)",
        "type": "TypeError",
        "description": "Using an object in a way that is not supported",
        "root_cause": "type_not_supported",
        "suggestion": "Ensure the object is of the expected type before using it",
    },
    {
        "pattern": r"ValueError: invalid literal for int\(\) with base (\d+): '([^']*)'",
        "type": "ValueError",
        "description": "Converting a string to an integer that is not a valid integer",
        "root_cause": "invalid_int_conversion",
        "suggestion": "Add error handling when converting strings to integers",
    },
    {
        "pattern": r"ZeroDivisionError: division by zero",
        "type": "ZeroDivisionError",
        "description": "Dividing by zero",
        "root_cause": "division_by_zero",
        "suggestion": "Check if the denominator is zero before dividing",
    },
    {
        "pattern": r"FileNotFoundError: \[Errno 2\] No such file or directory: '([^']*)'",
        "type": "FileNotFoundError",
        "description": "Trying to open a file that doesn't exist",
        "root_cause": "file_not_found",
        "suggestion": "Check if the file exists before trying to open it",
    }
]


class RuleBasedAnalyzer:
    """
    Analyzer that uses predefined rules to identify error patterns.
    """

    def __init__(self, additional_patterns: Optional[List[Dict[str, str]]] = None):
        """
        Initialize the analyzer with error patterns.

        Args:
            additional_patterns: Additional error patterns to use
        """
        self.patterns = ERROR_PATTERNS.copy()
        
        if additional_patterns:
            self.patterns.extend(additional_patterns)
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an error log entry and identify the root cause.

        Args:
            error_data: Error log data

        Returns:
            Analysis results including root cause and suggestions
        """
        # Extract error message
        error_message = error_data.get("message", "")
        
        # Extract traceback if available
        traceback = error_data.get("traceback", [])
        if isinstance(traceback, list) and traceback:
            traceback_str = "".join(traceback)
        else:
            traceback_str = str(traceback)
        
        # Extract exception type
        exception_type = error_data.get("exception_type", "")
        
        # Analyze error message and traceback
        for pattern in self.patterns:
            regex = pattern["pattern"]
            match = re.search(regex, error_message) or re.search(regex, traceback_str)
            
            if match:
                return {
                    "error_data": error_data,
                    "matched_pattern": pattern["pattern"],
                    "root_cause": pattern["root_cause"],
                    "description": pattern["description"],
                    "suggestion": pattern["suggestion"],
                    "match_groups": match.groups() if match.groups() else None,
                    "confidence": "high"
                }
        
        # If no pattern matches, try to make a best guess based on exception type
        if exception_type:
            for pattern in self.patterns:
                if pattern["type"] == exception_type:
                    return {
                        "error_data": error_data,
                        "matched_pattern": None,
                        "root_cause": pattern["root_cause"],
                        "description": pattern["description"],
                        "suggestion": pattern["suggestion"],
                        "match_groups": None,
                        "confidence": "medium"
                    }
        
        # If no match, return a generic analysis
        return {
            "error_data": error_data,
            "matched_pattern": None,
            "root_cause": "unknown",
            "description": "Unknown error type",
            "suggestion": "Manual investigation required",
            "match_groups": None,
            "confidence": "low"
        }
    
    def analyze_errors(self, error_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple error log entries.

        Args:
            error_data_list: List of error log data

        Returns:
            List of analysis results
        """
        return [self.analyze_error(error_data) for error_data in error_data_list]


# FastAPI-specific error patterns
FASTAPI_ERROR_PATTERNS = [
    {
        "pattern": r"KeyError: '([^']*)'",
        "type": "KeyError",
        "description": "Accessing a dictionary key that doesn't exist in a FastAPI endpoint",
        "root_cause": "dict_key_not_exists",
        "suggestion": "Add error handling to check if the key exists before accessing it",
    },
    {
        "pattern": r"pydantic.error_wrappers.ValidationError",
        "type": "ValidationError",
        "description": "Request data failed Pydantic validation",
        "root_cause": "request_validation_error",
        "suggestion": "Ensure the request data matches the expected schema",
    }
]


def create_fastapi_analyzer() -> RuleBasedAnalyzer:
    """
    Create an analyzer with FastAPI-specific patterns.

    Returns:
        RuleBasedAnalyzer configured for FastAPI
    """
    return RuleBasedAnalyzer(additional_patterns=FASTAPI_ERROR_PATTERNS)


if __name__ == "__main__":
    # Example usage
    analyzer = RuleBasedAnalyzer()
    
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
    print(f"Root Cause: {analysis['root_cause']}")
    print(f"Description: {analysis['description']}")
    print(f"Suggestion: {analysis['suggestion']}")
    print(f"Confidence: {analysis['confidence']}")