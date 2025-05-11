"""
Cross-Language Orchestrator

This module provides an orchestrator for handling errors across different programming languages.
It serves as a bridge between language-specific analyzers and enables cross-language error analysis.
"""
import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

from .analyzer import Analyzer, AnalysisStrategy
from .javascript_analyzer import JavaScriptAnalyzer
from .language_adapters import (
    ErrorAdapterFactory, 
    convert_to_standard_format, 
    convert_from_standard_format,
    ErrorSchemaValidator
)
from .rule_based import RuleBasedAnalyzer

logger = logging.getLogger(__name__)


class LanguageRegistry:
    """Registry of supported languages and their analyzers."""
    
    def __init__(self):
        """Initialize the language registry."""
        self.analyzers = {}
        self.register_default_analyzers()
    
    def register_default_analyzers(self):
        """Register the default language analyzers."""
        # Register Python analyzer
        self.register_analyzer("python", Analyzer(strategy=AnalysisStrategy.HYBRID))
        
        # Register JavaScript analyzer
        self.register_analyzer("javascript", JavaScriptAnalyzer())
        self.register_analyzer("typescript", JavaScriptAnalyzer())  # TypeScript uses the same analyzer
    
    def register_analyzer(self, language: str, analyzer: Any):
        """
        Register a language analyzer.
        
        Args:
            language: Language identifier
            analyzer: Language-specific analyzer
        """
        self.analyzers[language.lower()] = analyzer
        logger.info(f"Registered analyzer for language: {language}")
    
    def get_analyzer(self, language: str) -> Optional[Any]:
        """
        Get the analyzer for a language.
        
        Args:
            language: Language identifier
            
        Returns:
            Language-specific analyzer or None if not registered
        """
        return self.analyzers.get(language.lower())
    
    def is_language_supported(self, language: str) -> bool:
        """
        Check if a language is supported.
        
        Args:
            language: Language identifier
            
        Returns:
            True if the language is supported, False otherwise
        """
        return language.lower() in self.analyzers
    
    def get_supported_languages(self) -> List[str]:
        """
        Get the list of supported languages.
        
        Returns:
            List of supported language identifiers
        """
        return list(self.analyzers.keys())


class CrossLanguageOrchestrator:
    """
    Orchestrator for cross-language error analysis and handling.
    
    This class provides a unified interface for analyzing errors from different programming languages,
    converting between language-specific formats, and enabling cross-language learning.
    """
    
    def __init__(self):
        """Initialize the cross-language orchestrator."""
        self.registry = LanguageRegistry()
        self.validator = ErrorSchemaValidator()
        self.error_history = {}  # Store previous errors for learning
    
    def analyze_error(self, error_data: Dict[str, Any], 
                     language: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze an error from any supported language.
        
        Args:
            error_data: Error data
            language: Optional language identifier (auto-detected if not specified)
            
        Returns:
            Analysis results
            
        Raises:
            ValueError: If language is not supported or cannot be determined
        """
        # Detect language if not provided
        if language is None:
            language = self._detect_language(error_data)
            
            if language == "unknown":
                raise ValueError("Could not detect language from error data, please specify explicitly")
        
        language = language.lower()
        
        # Check if the language is supported
        if not self.registry.is_language_supported(language):
            raise ValueError(f"Unsupported language: {language}")
        
        # Get the appropriate analyzer
        analyzer = self.registry.get_analyzer(language)
        
        # Analyze the error
        analysis_result = analyzer.analyze_error(error_data)
        
        # Store in history for learning
        self._store_error_analysis(error_data, analysis_result, language)
        
        return analysis_result
    
    def convert_error(self, error_data: Dict[str, Any], 
                     source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Convert an error from one language format to another.
        
        Args:
            error_data: Error data in the source language format
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Error data in the target language format
            
        Raises:
            ValueError: If source or target language is not supported
        """
        source_lang = source_lang.lower()
        target_lang = target_lang.lower()
        
        # Check if languages are supported
        if not self.registry.is_language_supported(source_lang):
            raise ValueError(f"Unsupported source language: {source_lang}")
        
        if not self.registry.is_language_supported(target_lang):
            raise ValueError(f"Unsupported target language: {target_lang}")
        
        # If same language, no conversion needed
        if source_lang == target_lang:
            return error_data
        
        # Get adapters
        source_adapter = ErrorAdapterFactory.get_adapter(source_lang)
        target_adapter = ErrorAdapterFactory.get_adapter(target_lang)
        
        # Convert to standard format
        standard_error = source_adapter.to_standard_format(error_data)
        
        # Convert to target format
        target_error = target_adapter.from_standard_format(standard_error)
        
        return target_error
    
    def find_similar_errors(self, error_data: Dict[str, Any], 
                          language: Optional[str] = None,
                          max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find errors similar to the provided error across all languages.
        
        Args:
            error_data: Error data
            language: Optional language identifier (auto-detected if not specified)
            max_results: Maximum number of similar errors to return
            
        Returns:
            List of similar errors with similarity scores
        """
        # Detect language if not provided
        if language is None:
            language = self._detect_language(error_data)
            
            if language == "unknown":
                raise ValueError("Could not detect language from error data, please specify explicitly")
        
        language = language.lower()
        
        # Convert to standard format for comparison
        source_adapter = ErrorAdapterFactory.get_adapter(language)
        standard_error = source_adapter.to_standard_format(error_data)
        
        # Find similar errors
        similar_errors = []
        
        for error_id, entry in self.error_history.items():
            # Skip comparing with itself
            if error_id == standard_error.get("error_id"):
                continue
                
            # Compare the errors
            similarity = self._calculate_similarity(standard_error, entry["error"])
            
            if similarity > 0.3:  # Minimum similarity threshold
                similar_errors.append({
                    "error_id": error_id,
                    "language": entry["language"],
                    "error": entry["error"],
                    "analysis": entry["analysis"],
                    "similarity": similarity
                })
        
        # Sort by similarity (descending) and limit results
        similar_errors.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_errors[:max_results]
    
    def suggest_cross_language_fixes(self, error_data: Dict[str, Any],
                                   language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Suggest fixes for an error by looking at similar errors in other languages.
        
        Args:
            error_data: Error data
            language: Optional language identifier (auto-detected if not specified)
            
        Returns:
            List of suggested fixes from different languages
        """
        # Find similar errors
        similar_errors = self.find_similar_errors(error_data, language)
        
        # Extract suggested fixes
        suggested_fixes = []
        
        for error in similar_errors:
            # Skip errors in the same language
            if error["language"] == language:
                continue
                
            # Extract suggestion from analysis
            if "suggestion" in error["analysis"]:
                suggested_fixes.append({
                    "language": error["language"],
                    "suggestion": error["analysis"]["suggestion"],
                    "similarity": error["similarity"],
                    "source_error_id": error["error_id"]
                })
        
        return suggested_fixes
    
    def register_language_analyzer(self, language: str, analyzer: Any):
        """
        Register a new language analyzer.
        
        Args:
            language: Language identifier
            analyzer: Language-specific analyzer
        """
        self.registry.register_analyzer(language, analyzer)
    
    def get_supported_languages(self) -> List[str]:
        """
        Get the list of supported languages.
        
        Returns:
            List of supported language identifiers
        """
        return self.registry.get_supported_languages()
    
    def validate_error_schema(self, error_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate error data against the standard schema.
        
        Args:
            error_data: Error data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.validator.validate(error_data)
    
    def _detect_language(self, error_data: Dict[str, Any]) -> str:
        """
        Detect the language from error data.
        
        Args:
            error_data: The error data
            
        Returns:
            Detected language or "unknown"
        """
        # If language is explicitly specified
        if "language" in error_data:
            return error_data["language"].lower()
        
        # Otherwise, try to detect using the adapter factory
        return ErrorAdapterFactory.detect_language(error_data)
    
    def _store_error_analysis(self, error_data: Dict[str, Any], 
                             analysis: Dict[str, Any], 
                             language: str):
        """
        Store error analysis in history for learning.
        
        Args:
            error_data: Error data
            analysis: Analysis results
            language: Language identifier
        """
        # Convert to standard format for storage
        source_adapter = ErrorAdapterFactory.get_adapter(language)
        standard_error = source_adapter.to_standard_format(error_data)
        
        # Generate ID if not present
        error_id = standard_error.get("error_id")
        if not error_id:
            error_id = str(uuid.uuid4())
            standard_error["error_id"] = error_id
        
        # Store in history
        self.error_history[error_id] = {
            "error": standard_error,
            "analysis": analysis,
            "language": language,
            "timestamp": datetime.now().isoformat()
        }
        
        # Limit history size (keep last 1000 errors)
        if len(self.error_history) > 1000:
            oldest_key = min(self.error_history.keys(), 
                            key=lambda k: self.error_history[k].get("timestamp", ""))
            del self.error_history[oldest_key]
    
    def _calculate_similarity(self, error1: Dict[str, Any], 
                             error2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two errors in standard format.
        
        Args:
            error1: First error in standard format
            error2: Second error in standard format
            
        Returns:
            Similarity score (0-1)
        """
        # Simple similarity calculation based on error type and message
        score = 0.0
        
        # Compare error types
        if error1.get("error_type") == error2.get("error_type"):
            score += 0.5
        
        # Compare messages using simple string similarity
        message1 = error1.get("message", "")
        message2 = error2.get("message", "")
        
        if message1 and message2:
            # Simple word overlap similarity
            words1 = set(message1.lower().split())
            words2 = set(message2.lower().split())
            
            if words1 and words2:
                overlap = len(words1.intersection(words2))
                similarity = overlap / max(len(words1), len(words2))
                score += 0.3 * similarity
        
        # Compare stack traces if available
        if "stack_trace" in error1 and "stack_trace" in error2:
            # For simplicity, just check if they have the same structure
            if (isinstance(error1["stack_trace"], list) and 
                isinstance(error2["stack_trace"], list)):
                score += 0.2
        
        return score


def analyze_multi_language_error(error_data: Dict[str, Any], language: Optional[str] = None) -> Dict[str, Any]:
    """
    Utility function to analyze an error from any supported language.
    
    Args:
        error_data: Error data
        language: Optional language identifier (auto-detected if not specified)
        
    Returns:
        Analysis results
    """
    orchestrator = CrossLanguageOrchestrator()
    return orchestrator.analyze_error(error_data, language)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create orchestrator
    orchestrator = CrossLanguageOrchestrator()
    
    # Display supported languages
    languages = orchestrator.get_supported_languages()
    logger.info(f"Supported languages: {', '.join(languages)}")
    
    # Example Python error
    python_error = {
        "timestamp": "2023-08-15T12:34:56",
        "exception_type": "KeyError",
        "message": "'user_id'",
        "traceback": [
            "Traceback (most recent call last):",
            "  File \"app.py\", line 42, in get_user",
            "    user_id = data['user_id']",
            "KeyError: 'user_id'"
        ],
        "level": "ERROR",
        "python_version": "3.9.7",
        "framework": "FastAPI",
        "framework_version": "0.68.0"
    }
    
    # Example JavaScript error
    js_error = {
        "name": "TypeError",
        "message": "Cannot read property 'id' of undefined",
        "stack": "TypeError: Cannot read property 'id' of undefined\n    at getUserId (/app/src/utils.js:45:20)\n    at processRequest (/app/src/controllers/user.js:23:15)\n    at /app/src/routes/index.js:10:12",
        "timestamp": "2023-08-20T14:30:45Z"
    }
    
    # Analyze the Python error
    logger.info("Analyzing Python error...")
    python_analysis = orchestrator.analyze_error(python_error, "python")
    logger.info(f"Python analysis root cause: {python_analysis.get('root_cause')}")
    logger.info(f"Python analysis suggestion: {python_analysis.get('suggestion')}")
    
    # Analyze the JavaScript error
    logger.info("Analyzing JavaScript error...")
    js_analysis = orchestrator.analyze_error(js_error, "javascript")
    logger.info(f"JavaScript analysis root cause: {js_analysis.get('root_cause')}")
    logger.info(f"JavaScript analysis suggestion: {js_analysis.get('suggestion')}")
    
    # Convert JavaScript error to Python format
    logger.info("Converting JavaScript error to Python format...")
    python_format = orchestrator.convert_error(js_error, "javascript", "python")
    logger.info(f"Python format exception_type: {python_format.get('exception_type')}")
    logger.info(f"Python format message: {python_format.get('message')}")
    
    # Find similar errors
    logger.info("Finding similar errors...")
    similar = orchestrator.find_similar_errors(python_error, "python")
    logger.info(f"Found {len(similar)} similar errors")
    
    if similar:
        logger.info(f"Most similar error (similarity: {similar[0].get('similarity'):.2f}):")
        logger.info(f"  Language: {similar[0].get('language')}")
        logger.info(f"  Error type: {similar[0].get('error').get('error_type')}")
        logger.info(f"  Message: {similar[0].get('error').get('message')}")
    
    # Suggest cross-language fixes
    logger.info("Suggesting cross-language fixes...")
    fixes = orchestrator.suggest_cross_language_fixes(python_error, "python")
    logger.info(f"Found {len(fixes)} suggested fixes from other languages")
    
    if fixes:
        logger.info(f"Top suggestion from {fixes[0].get('language')}:")
        logger.info(f"  {fixes[0].get('suggestion')}")