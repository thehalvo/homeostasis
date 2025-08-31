"""
JavaScript and Node.js Error Analyzer

This module provides error analysis for JavaScript and Node.js applications,
leveraging the language-agnostic error schema for cross-language compatibility.
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .rule_config import RuleLoader
from .language_adapters import JavaScriptErrorAdapter, ErrorAdapterFactory

logger = logging.getLogger(__name__)

# Load JavaScript-specific rules
RULES_DIR = Path(__file__).parent / "rules" / "javascript"
JS_COMMON_RULES_PATH = RULES_DIR / "js_common_errors.json"
NODEJS_RULES_PATH = RULES_DIR / "nodejs_errors.json"


class JavaScriptAnalyzer:
    """
    Analyzer for JavaScript and Node.js errors.
    
    This analyzer works with both browser JavaScript and Node.js errors,
    converting them to the standardized error schema for analysis.
    """
    
    def __init__(self):
        """Initialize the JavaScript analyzer."""
        self.adapter = JavaScriptErrorAdapter()
        
        # Load JavaScript rules
        self.rules = []
        
        try:
            if JS_COMMON_RULES_PATH.exists():
                js_ruleset = RuleLoader.load_from_file(JS_COMMON_RULES_PATH)
                self.rules.extend(js_ruleset.rules)
                logger.info(f"Loaded {len(js_ruleset.rules)} JavaScript common rules")
            
            if NODEJS_RULES_PATH.exists():
                nodejs_ruleset = RuleLoader.load_from_file(NODEJS_RULES_PATH)
                self.rules.extend(nodejs_ruleset.rules)
                logger.info(f"Loaded {len(nodejs_ruleset.rules)} Node.js rules")
                
        except Exception as e:
            logger.error(f"Error loading JavaScript rules: {e}")
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a JavaScript error.
        
        Args:
            error_data: JavaScript error data
            
        Returns:
            Analysis results
        """
        # First, check if the error data is already in the standard format
        if "language" in error_data and error_data["language"] in ["javascript", "typescript"]:
            standard_error = error_data
        else:
            # Convert to standard format
            standard_error = self.adapter.to_standard_format(error_data)
        
        # Analyze using rules
        analysis_result = self._apply_rules(standard_error)
        
        # If no rules matched, provide a generic analysis
        if not analysis_result:
            analysis_result = self._generic_analysis(standard_error)
        
        return analysis_result
    
    def _apply_rules(self, error_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply rules to the error data.
        
        Args:
            error_data: Error data in standard format
            
        Returns:
            Analysis results or None if no rules matched
        """
        # Extract error message and stack trace for pattern matching
        error_message = error_data.get("message", "")
        error_type = error_data.get("error_type", "")
        
        stack_trace = ""
        if "stack_trace" in error_data:
            if isinstance(error_data["stack_trace"], list):
                if all(isinstance(frame, str) for frame in error_data["stack_trace"]):
                    stack_trace = "\n".join(error_data["stack_trace"])
                elif all(isinstance(frame, dict) for frame in error_data["stack_trace"]):
                    # Convert structured frames to a string for pattern matching
                    stack_parts = []
                    for frame in error_data["stack_trace"]:
                        file = frame.get("file", "<unknown>")
                        line = frame.get("line", "?")
                        col = frame.get("column", "?")
                        func = frame.get("function", "<anonymous>")
                        stack_parts.append(f"at {func} ({file}:{line}:{col})")
                    stack_trace = "\n".join(stack_parts)
                    
        # Combined text for pattern matching
        if stack_trace:
            match_text = f"{error_type}: {error_message}\n{stack_trace}"
        else:
            match_text = f"{error_type}: {error_message}"
        
        # Apply each rule
        for rule in self.rules:
            match = rule.matches(match_text)
            if match:
                # Rule matched, create analysis result
                return {
                    "rule_id": rule.id,
                    "error_type": rule.type,
                    "root_cause": rule.root_cause,
                    "description": rule.description,
                    "suggestion": rule.suggestion,
                    "confidence": rule.confidence.value,
                    "severity": rule.severity.value,
                    "category": "javascript",
                    "language": error_data.get("language", "javascript"),
                    "framework": error_data.get("framework", ""),
                    "matched_pattern": match.group(0),
                    "match_groups": match.groups(),
                    "original_error": error_data
                }
        
        return None
    
    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide a generic analysis for errors that don't match any rules.
        
        Args:
            error_data: Error data in standard format
            
        Returns:
            Generic analysis results
        """
        error_type = error_data.get("error_type", "")
        error_message = error_data.get("message", "")
        
        # Basic categorization based on error type
        if "SyntaxError" in error_type:
            return {
                "rule_id": "generic_js_syntax_error",
                "error_type": "SyntaxError",
                "root_cause": "js_syntax_error",
                "description": "JavaScript syntax error",
                "suggestion": "Check for syntax errors such as missing brackets, parentheses, or commas",
                "confidence": "medium",
                "severity": "high",
                "category": "javascript",
                "language": error_data.get("language", "javascript"),
                "framework": error_data.get("framework", ""),
                "original_error": error_data
            }
        elif "TypeError" in error_type:
            return {
                "rule_id": "generic_js_type_error",
                "error_type": "TypeError",
                "root_cause": "js_type_error",
                "description": "JavaScript type error",
                "suggestion": "Ensure variables are of the expected type before operations",
                "confidence": "medium",
                "severity": "high",
                "category": "javascript",
                "language": error_data.get("language", "javascript"),
                "framework": error_data.get("framework", ""),
                "original_error": error_data
            }
        elif "ReferenceError" in error_type:
            return {
                "rule_id": "generic_js_reference_error",
                "error_type": "ReferenceError",
                "root_cause": "js_reference_error",
                "description": "JavaScript reference error (undefined variable)",
                "suggestion": "Check that all variables are defined before use",
                "confidence": "medium",
                "severity": "high",
                "category": "javascript",
                "language": error_data.get("language", "javascript"),
                "framework": error_data.get("framework", ""),
                "original_error": error_data
            }
        else:
            # Generic fallback
            return {
                "rule_id": "generic_js_error",
                "error_type": error_type or "UnknownError",
                "root_cause": "js_unknown_error",
                "description": f"Unrecognized JavaScript error: {error_type}",
                "suggestion": "Check the error message and stack trace for more details",
                "confidence": "low",
                "severity": "medium",
                "category": "javascript",
                "language": error_data.get("language", "javascript"),
                "framework": error_data.get("framework", ""),
                "original_error": error_data
            }
    
    def convert_to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert JavaScript error data to the standard format.
        
        Args:
            error_data: JavaScript error data
            
        Returns:
            Error data in the standard format
        """
        return self.adapter.to_standard_format(error_data)
    
    def convert_from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to JavaScript format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the JavaScript format
        """
        return self.adapter.from_standard_format(standard_error)


def analyze_javascript_error(error_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Utility function to analyze a JavaScript error.
    
    Args:
        error_data: JavaScript error data
        
    Returns:
        Analysis results
    """
    analyzer = JavaScriptAnalyzer()
    return analyzer.analyze_error(error_data)


def convert_error_format(error_data: Dict[str, Any], 
                        source_lang: str, 
                        target_lang: str) -> Dict[str, Any]:
    """
    Convert error data between different language formats.
    
    Args:
        error_data: Error data in the source language format
        source_lang: Source language (python, javascript, etc.)
        target_lang: Target language (python, javascript, etc.)
        
    Returns:
        Error data in the target language format
        
    Raises:
        ValueError: If source or target language is not supported
    """
    # Get the appropriate adapters
    source_adapter = ErrorAdapterFactory.get_adapter(source_lang)
    target_adapter = ErrorAdapterFactory.get_adapter(target_lang)
    
    # Convert to standard format
    standard_error = source_adapter.to_standard_format(error_data)
    
    # Convert to target format
    target_error = target_adapter.from_standard_format(standard_error)
    
    return target_error


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example JavaScript error
    js_error = {
        "name": "TypeError",
        "message": "Cannot read property 'id' of undefined",
        "stack": "TypeError: Cannot read property 'id' of undefined\n    at getUserId (/app/src/utils.js:45:20)\n    at processRequest (/app/src/controllers/user.js:23:15)\n    at /app/src/routes/index.js:10:12",
        "timestamp": "2023-08-20T14:30:45Z"
    }
    
    # Analyze the error
    analyzer = JavaScriptAnalyzer()
    analysis = analyzer.analyze_error(js_error)
    
    logger.info(f"JavaScript Error Analysis:")
    logger.info(f"  Rule: {analysis.get('rule_id')}")
    logger.info(f"  Root Cause: {analysis.get('root_cause')}")
    logger.info(f"  Suggestion: {analysis.get('suggestion')}")
    
    # Example Node.js error
    nodejs_error = {
        "name": "Error",
        "message": "Cannot find module 'express'",
        "stack": "Error: Cannot find module 'express'\n    at Function.Module._resolveFilename (node:internal/modules/cjs/loader:985:15)\n    at Function.Module._load (node:internal/modules/cjs/loader:833:27)\n    at Module.require (node:internal/modules/cjs/loader:1057:19)\n    at require (node:internal/modules/cjs/helpers:103:18)\n    at Object.<anonymous> (/app/server.js:1:17)",
        "timestamp": "2023-08-20T14:35:12Z"
    }
    
    # Analyze the Node.js error
    analysis = analyzer.analyze_error(nodejs_error)
    
    logger.info(f"Node.js Error Analysis:")
    logger.info(f"  Rule: {analysis.get('rule_id')}")
    logger.info(f"  Root Cause: {analysis.get('root_cause')}")
    logger.info(f"  Suggestion: {analysis.get('suggestion')}")
    
    # Convert between formats
    logger.info(f"Converting JavaScript error to Python format:")
    python_format = convert_error_format(js_error, "javascript", "python")
    logger.info(f"  Python exception_type: {python_format.get('exception_type')}")
    logger.info(f"  Python message: {python_format.get('message')}")
    
    # Convert back
    js_format = convert_error_format(python_format, "python", "javascript")
    logger.info(f"Converting back to JavaScript format:")
    logger.info(f"  JavaScript name: {js_format.get('name')}")
    logger.info(f"  JavaScript message: {js_format.get('message')}")