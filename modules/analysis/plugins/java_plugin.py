"""
Java Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Java applications.
"""
import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union

from ..language_plugin_system import LanguagePlugin, register_plugin
from ..language_adapters import JavaErrorAdapter

logger = logging.getLogger(__name__)


class JavaLanguagePlugin(LanguagePlugin):
    """
    Java language plugin for Homeostasis.
    
    Provides error analysis and fix generation for Java applications.
    """
    
    VERSION = "0.1.0"
    AUTHOR = "Homeostasis Contributors"
    
    def __init__(self):
        """Initialize the Java language plugin."""
        self.adapter = JavaErrorAdapter()
        self.rules = self._load_rules()
    
    def get_language_id(self) -> str:
        """Get the language identifier."""
        return "java"
    
    def get_language_name(self) -> str:
        """Get the language name."""
        return "Java"
    
    def get_language_version(self) -> str:
        """Get the language version."""
        return "8+"
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Java error.
        
        Args:
            error_data: Java error data
            
        Returns:
            Analysis results
        """
        # First, normalize the error
        if "language" not in error_data or error_data["language"] != "java":
            standard_error = self.normalize_error(error_data)
        else:
            standard_error = error_data
        
        # Extract essential information
        error_type = standard_error.get("error_type", "")
        message = standard_error.get("message", "")
        
        # Combine for rule matching
        match_text = f"{error_type}: {message}"
        
        if "stack_trace" in standard_error and standard_error["stack_trace"]:
            if isinstance(standard_error["stack_trace"], list):
                match_text += "\n" + "\n".join(
                    standard_error["stack_trace"] if isinstance(standard_error["stack_trace"][0], str)
                    else [f"at {frame.get('package', '')}.{frame.get('class', '')}."
                          f"{frame.get('function', '')}({frame.get('file', '')}:{frame.get('line', '?')})"
                          for frame in standard_error["stack_trace"]]
                )
        
        # Apply rules
        for rule in self.rules:
            pattern = rule.get("pattern", "")
            if not pattern:
                continue
                
            try:
                match = re.search(pattern, match_text, re.IGNORECASE)
                if match:
                    return {
                        "error_data": standard_error,
                        "rule_id": rule.get("id", "unknown"),
                        "error_type": rule.get("type", error_type),
                        "root_cause": rule.get("root_cause", "java_unknown_error"),
                        "description": rule.get("description", "Unknown Java error"),
                        "suggestion": rule.get("suggestion", "No suggestion available"),
                        "confidence": rule.get("confidence", "medium"),
                        "severity": rule.get("severity", "medium"),
                        "category": "java",
                        "match_groups": match.groups()
                    }
            except Exception as e:
                logger.warning(f"Error applying rule {rule.get('id', 'unknown')}: {e}")
        
        # Fallback for unmatched errors
        # Simple categorization based on common Java exception types
        if "NullPointerException" in error_type:
            return {
                "error_data": standard_error,
                "rule_id": "java_null_pointer",
                "error_type": error_type,
                "root_cause": "java_null_pointer",
                "description": "Attempted to access or use a null object reference",
                "suggestion": "Add null checks before accessing objects or methods",
                "confidence": "high",
                "severity": "high",
                "category": "java"
            }
        elif "ClassCastException" in error_type:
            return {
                "error_data": standard_error,
                "rule_id": "java_class_cast",
                "error_type": error_type,
                "root_cause": "java_invalid_cast",
                "description": "Attempted to cast an object to an incompatible type",
                "suggestion": "Verify object types before casting, use instanceof checks",
                "confidence": "high",
                "severity": "medium",
                "category": "java"
            }
        elif "ArrayIndexOutOfBoundsException" in error_type:
            return {
                "error_data": standard_error,
                "rule_id": "java_array_index",
                "error_type": error_type,
                "root_cause": "java_index_out_of_bounds",
                "description": "Attempted to access an array element with an invalid index",
                "suggestion": "Check array bounds before accessing elements",
                "confidence": "high",
                "severity": "medium",
                "category": "java"
            }
        elif "IllegalArgumentException" in error_type:
            return {
                "error_data": standard_error,
                "rule_id": "java_illegal_argument",
                "error_type": error_type,
                "root_cause": "java_invalid_argument",
                "description": "A method received an argument that was not valid",
                "suggestion": "Validate input arguments before passing them to methods",
                "confidence": "medium",
                "severity": "medium",
                "category": "java"
            }
        
        # Generic fallback
        return {
            "error_data": standard_error,
            "rule_id": "java_generic_error",
            "error_type": error_type or "Unknown",
            "root_cause": "java_unknown_error",
            "description": f"Unrecognized Java error: {error_type}",
            "suggestion": "Review the error message and stack trace for more details",
            "confidence": "low",
            "severity": "medium",
            "category": "java"
        }
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a Java error to the standard format.
        
        Args:
            error_data: Java error data
            
        Returns:
            Error data in the standard format
        """
        return self.adapter.to_standard_format(error_data)
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Java format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the Java format
        """
        return self.adapter.from_standard_format(standard_error)
    
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a fix for a Java error.
        
        Args:
            analysis: Error analysis
            context: Additional context for fix generation
            
        Returns:
            Generated fix data
        """
        # Since we don't have a Java-specific patch generator yet,
        # return a suggestion-based patch
        return {
            "patch_id": f"java_{analysis.get('rule_id', 'unknown')}",
            "patch_type": "suggestion",
            "suggestion": analysis.get("suggestion", "No suggestion available"),
            "root_cause": analysis.get("root_cause", "unknown"),
            "language": "java",
            "framework": context.get("framework", "")
        }
    
    def get_supported_frameworks(self) -> List[str]:
        """
        Get the list of frameworks supported by this language plugin.
        
        Returns:
            List of supported framework identifiers
        """
        return ["spring", "jakarta", "android", "base"]
    
    def _load_rules(self) -> List[Dict[str, Any]]:
        """
        Load Java error rules.
        
        Returns:
            List of rule definitions
        """
        # Sample rules for common Java exceptions
        return [
            {
                "id": "java_null_pointer",
                "pattern": "java\\.lang\\.NullPointerException(?:: (.*))?",
                "type": "NullPointerException",
                "description": "Attempted to access or use a null object reference",
                "root_cause": "java_null_pointer",
                "suggestion": "Add null checks before accessing objects or methods",
                "confidence": "high",
                "severity": "high"
            },
            {
                "id": "java_class_cast",
                "pattern": "java\\.lang\\.ClassCastException: ([^\\s]+) cannot be cast to ([^\\s]+)",
                "type": "ClassCastException",
                "description": "Attempted to cast an object to an incompatible type",
                "root_cause": "java_invalid_cast",
                "suggestion": "Verify object types before casting, use instanceof checks",
                "confidence": "high",
                "severity": "medium"
            },
            {
                "id": "java_array_index",
                "pattern": "java\\.lang\\.ArrayIndexOutOfBoundsException: (\\d+)",
                "type": "ArrayIndexOutOfBoundsException",
                "description": "Attempted to access an array element with an invalid index",
                "root_cause": "java_index_out_of_bounds",
                "suggestion": "Check array bounds before accessing elements",
                "confidence": "high",
                "severity": "medium"
            },
            {
                "id": "java_illegal_argument",
                "pattern": "java\\.lang\\.IllegalArgumentException: (.*)",
                "type": "IllegalArgumentException",
                "description": "A method received an argument that was not valid",
                "root_cause": "java_invalid_argument",
                "suggestion": "Validate input arguments before passing them to methods",
                "confidence": "medium",
                "severity": "medium"
            },
            {
                "id": "java_number_format",
                "pattern": "java\\.lang\\.NumberFormatException: (.*)",
                "type": "NumberFormatException",
                "description": "Failed to parse a string as a number",
                "root_cause": "java_invalid_number_format",
                "suggestion": "Ensure the string represents a valid number before parsing",
                "confidence": "high",
                "severity": "medium"
            },
            {
                "id": "java_io_exception",
                "pattern": "java\\.io\\.IOException: (.*)",
                "type": "IOException",
                "description": "An I/O operation failed or was interrupted",
                "root_cause": "java_io_error",
                "suggestion": "Implement appropriate error handling for I/O operations",
                "confidence": "medium",
                "severity": "medium"
            },
            {
                "id": "java_concurrent_modification",
                "pattern": "java\\.util\\.ConcurrentModificationException(?:: (.*))?",
                "type": "ConcurrentModificationException",
                "description": "Collection was modified during iteration",
                "root_cause": "java_concurrent_modification",
                "suggestion": "Use thread-safe collections or synchronize access to collections during iteration",
                "confidence": "high",
                "severity": "high"
            },
            {
                "id": "java_sql_exception",
                "pattern": "java\\.sql\\.SQLException: (.*)",
                "type": "SQLException",
                "description": "Database access error or other SQL-related issues",
                "root_cause": "java_sql_error",
                "suggestion": "Check database connection, SQL syntax, and implement proper exception handling",
                "confidence": "medium",
                "severity": "high"
            },
            {
                "id": "java_out_of_memory",
                "pattern": "java\\.lang\\.OutOfMemoryError: (.*)",
                "type": "OutOfMemoryError",
                "description": "Java Virtual Machine ran out of memory",
                "root_cause": "java_out_of_memory",
                "suggestion": "Increase JVM heap size, optimize memory usage, or fix memory leaks",
                "confidence": "high",
                "severity": "critical"
            },
            {
                "id": "java_stack_overflow",
                "pattern": "java\\.lang\\.StackOverflowError(?:: (.*))?",
                "type": "StackOverflowError",
                "description": "Recursive method calls exceeded the stack size",
                "root_cause": "java_stack_overflow",
                "suggestion": "Check for infinite recursion and ensure proper termination conditions",
                "confidence": "high",
                "severity": "high"
            }
        ]


# Register this plugin
register_plugin(JavaLanguagePlugin())