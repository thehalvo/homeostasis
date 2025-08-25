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

from modules.analysis.analyzer import Analyzer, AnalysisStrategy
from modules.analysis.javascript_analyzer import JavaScriptAnalyzer
from modules.analysis.language_adapters import (
    ErrorAdapterFactory, 
    convert_to_standard_format, 
    convert_from_standard_format,
    ErrorSchemaValidator
)
from modules.analysis.rule_based import RuleBasedAnalyzer
from modules.analysis.language_plugin_system import get_plugin, get_supported_languages

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
        
        # Register Java plugin if available
        try:
            # Get Java plugin through the plugin system
            java_plugin = get_plugin("java")
            if java_plugin:
                self.register_analyzer("java", java_plugin)
        except Exception as e:
            logger.warning(f"Failed to load Java plugin: {e}")
            
        # Register Go plugin if available
        try:
            # Get Go plugin through the plugin system
            go_plugin = get_plugin("go")
            if go_plugin:
                self.register_analyzer("go", go_plugin)
        except Exception as e:
            logger.warning(f"Failed to load Go plugin: {e}")
            
        # Register Rust plugin if available
        try:
            # Get Rust plugin through the plugin system
            rust_plugin = get_plugin("rust")
            if rust_plugin:
                self.register_analyzer("rust", rust_plugin)
        except Exception as e:
            logger.warning(f"Failed to load Rust plugin: {e}")
            
        # Register PHP plugin if available
        try:
            # Get PHP plugin through the plugin system
            php_plugin = get_plugin("php")
            if php_plugin:
                self.register_analyzer("php", php_plugin)
        except Exception as e:
            logger.warning(f"Failed to load PHP plugin: {e}")
            
        # Register Ruby plugin if available
        try:
            # Get Ruby plugin through the plugin system
            ruby_plugin = get_plugin("ruby")
            if ruby_plugin:
                self.register_analyzer("ruby", ruby_plugin)
        except Exception as e:
            logger.warning(f"Failed to load Ruby plugin: {e}")
            
        # Register C/C++ plugin if available
        try:
            # Get C/C++ plugin through the plugin system
            cpp_plugin = get_plugin("cpp")
            if cpp_plugin:
                self.register_analyzer("cpp", cpp_plugin)
                self.register_analyzer("c", cpp_plugin)  # C uses the same analyzer
        except Exception as e:
            logger.warning(f"Failed to load C/C++ plugin: {e}")
    
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
    
    def analyze_cross_language_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an error that occurs at the boundary between two languages.
        
        This is specifically designed for errors that occur when one language calls into
        another (e.g., Python calling C extensions, JavaScript calling Rust WASM, etc.)
        
        Args:
            error_data: Error data containing information about both languages involved
            
        Returns:
            Analysis results with cross-language specific insights
        """
        # Extract the languages involved
        primary_language = error_data.get("primary_language", 
                                         error_data.get("language", "unknown"))
        secondary_language = error_data.get("secondary_language", "unknown")
        
        # If secondary language not specified, try to detect from context
        if secondary_language == "unknown":
            # Check error chain for secondary language
            error_chain = error_data.get("error_chain", [])
            for error in error_chain:
                lang = error.get("language", "unknown")
                if lang != "unknown" and lang != primary_language:
                    secondary_language = lang
                    break
            
            # If still unknown, check for specific patterns
            if secondary_language == "unknown":
                # Check for C extension patterns
                if primary_language == "python":
                    if any(key in error_data for key in ["c_function", "native_function", "segfault"]):
                        secondary_language = "c"
                # Check for WASM patterns
                elif primary_language == "javascript":
                    if any(key in error_data for key in ["wasm_module", "rust_panic", "memory_access"]):
                        secondary_language = "rust"
                # Check for JNI patterns
                elif primary_language == "java":
                    if any(key in error_data for key in ["jni_method", "native_method", "cpp_exception"]):
                        secondary_language = "cpp"
        
        # Analyze the error in the primary language context
        primary_analysis = {}
        if primary_language != "unknown" and self.registry.is_language_supported(primary_language):
            try:
                primary_analysis = self.analyze_error(error_data, primary_language)
            except Exception as e:
                logger.warning(f"Failed to analyze in primary language {primary_language}: {e}")
        
        # If we have a secondary language, analyze from that perspective too
        secondary_analysis = {}
        if secondary_language != "unknown" and self.registry.is_language_supported(secondary_language):
            try:
                # Convert error data to secondary language format if needed
                converted_data = error_data
                if primary_language != "unknown":
                    try:
                        converted_data = self.convert_error(error_data, primary_language, secondary_language)
                    except:
                        # If conversion fails, use original data
                        pass
                
                secondary_analysis = self.analyze_error(converted_data, secondary_language)
            except Exception as e:
                logger.warning(f"Failed to analyze in secondary language {secondary_language}: {e}")
        
        # Combine analyses and add cross-language specific insights
        result = {
            "primary_language": primary_language,
            "secondary_language": secondary_language,
            "primary_analysis": primary_analysis,
            "secondary_analysis": secondary_analysis,
            "cross_language_boundary": f"{primary_language}-{secondary_language}",
            "is_cross_language_error": True
        }
        
        # Analyze error chain to determine root cause language and propagation path
        error_chain = error_data.get("error_chain", [])
        propagation_path = []
        root_cause_language = secondary_language  # Default to secondary language
        
        if error_chain:
            # Build propagation path from error chain
            for error in error_chain:
                lang = error.get("language", "unknown")
                if lang != "unknown" and lang not in propagation_path:
                    propagation_path.append(lang)
            
            # Root cause is typically in the last/deepest error in chain
            if propagation_path:
                root_cause_language = propagation_path[-1]  # Last in chain is root
                # Reverse path so root cause is first
                propagation_path = list(reversed(propagation_path))
                
            # Check for specific error indicators
            for error in error_chain:
                if error.get("error_type") == "SegmentationFault":
                    root_cause_language = error.get("language", root_cause_language)
                    if error.get("address") == "0x0":
                        result["root_cause"] = "null pointer dereference"
                    break
                elif error.get("signal") == "SIGSEGV":
                    root_cause_language = error.get("language", root_cause_language)
                    result["root_cause"] = "segmentation fault - likely null pointer access"
                    break
                elif error.get("error_type") == "std::bad_alloc":
                    root_cause_language = error.get("language", root_cause_language)
                    result["root_cause"] = "memory allocation failure"
                    break
                elif "memory allocation failed" in error.get("message", "").lower():
                    root_cause_language = error.get("language", root_cause_language)
                    result["root_cause"] = "memory allocation error"
                    break
                elif error.get("error_type") == "BufferOverflow":
                    root_cause_language = error.get("language", root_cause_language)
                    result["root_cause"] = "buffer overflow"
                    break
                elif "buffer overflow" in error.get("message", "").lower():
                    root_cause_language = error.get("language", root_cause_language)
                    result["root_cause"] = "buffer overflow detected"
                    break
        
        result["root_cause_language"] = root_cause_language
        result["propagation_path"] = propagation_path if propagation_path else [secondary_language, primary_language]
        
        # Add specific suggestions for common cross-language scenarios
        suggestions = []
        
        if primary_language == "python" and secondary_language == "c":
            suggestions.extend([
                "Check for proper reference counting in C extension",
                "Verify argument types match between Python and C",
                "Ensure GIL is properly acquired/released",
                "Check for memory leaks in C code"
            ])
            result["common_causes"] = [
                "Incorrect reference counting",
                "Type mismatches between Python and C",
                "Memory management issues",
                "GIL (Global Interpreter Lock) violations"
            ]
        
        elif primary_language == "javascript" and secondary_language == "rust":
            suggestions.extend([
                "Check WASM memory boundaries",
                "Verify proper data serialization between JS and Rust",
                "Ensure proper error handling in Rust code",
                "Check for panic conditions in Rust"
            ])
            result["common_causes"] = [
                "Memory access violations",
                "Data serialization errors",
                "Unhandled Rust panics",
                "Type mismatches"
            ]
        
        elif primary_language == "java" and secondary_language == "cpp":
            suggestions.extend([
                "Verify JNI method signatures match",
                "Check for proper exception handling in JNI code",
                "Ensure proper memory management in C++",
                "Verify thread safety in native code"
            ])
            result["common_causes"] = [
                "JNI signature mismatches",
                "Unhandled C++ exceptions",
                "Memory management issues",
                "Thread safety violations"
            ]
        
        # Combine suggestions
        all_suggestions = []
        if "suggestion" in primary_analysis:
            all_suggestions.append(primary_analysis["suggestion"])
        if "suggestion" in secondary_analysis:
            all_suggestions.append(secondary_analysis["suggestion"])
        all_suggestions.extend(suggestions)
        
        result["suggestions"] = all_suggestions
        result["suggestion"] = " ".join(all_suggestions[:2])  # Primary suggestion
        
        # Generate fixes for each language involved as a dictionary
        fixes = {}
        
        # Add fix for primary language
        if primary_language != "unknown":
            primary_suggestions = []
            if "suggestion" in primary_analysis:
                primary_suggestions.append(primary_analysis["suggestion"])
            
            # Add language-specific suggestions
            if primary_language == "python" and secondary_language == "c":
                primary_suggestions.extend([
                    "Add error handling for native function calls",
                    "Validate arguments before passing to C extension"
                ])
            elif primary_language == "javascript" and secondary_language == "rust":
                primary_suggestions.extend([
                    "Add proper error handling for WASM calls",
                    "Validate data before passing to WASM module"
                ])
            elif primary_language == "java" and secondary_language == "cpp":
                primary_suggestions.extend([
                    "Wrap JNI calls in try-catch blocks",
                    "Check JNI return values for errors",
                    "Handle native exceptions properly"
                ])
            elif primary_language == "go" and secondary_language in ["c", "cpp"]:
                primary_suggestions.extend([
                    "Check CGO return values for errors",
                    "Use defer for cleanup in Go code",
                    "Handle C errors properly in Go"
                ])
            
            fixes[primary_language] = {
                "language": primary_language,
                "description": f"Fix in {primary_language} code",
                "suggestion": " ".join(primary_suggestions[:2]) if primary_suggestions else "Check error handling in calling code",
                "suggestions": primary_suggestions
            }
        
        # Add fix for secondary language
        if secondary_language != "unknown":
            secondary_suggestions = []
            if "suggestion" in secondary_analysis:
                secondary_suggestions.append(secondary_analysis["suggestion"])
            
            # Add language-specific suggestions
            if secondary_language == "c" and "null pointer" in result.get("root_cause", "").lower():
                secondary_suggestions.extend([
                    "Add null pointer checks before dereferencing",
                    "Initialize all pointers properly",
                    "Validate input parameters"
                ])
            elif secondary_language == "rust":
                # Check for specific Rust error patterns
                for error in error_chain:
                    if error.get("language") == "rust":
                        if "index out of bounds" in error.get("message", "").lower():
                            secondary_suggestions.extend([
                                "Add bounds checking before array access",
                                "Use safe indexing methods like get() instead of direct indexing"
                            ])
                        elif "panic" in error.get("message", "").lower():
                            secondary_suggestions.extend([
                                "Handle potential panic conditions explicitly",
                                "Use Result<T, E> for fallible operations"
                            ])
            elif secondary_language == "cpp" or secondary_language == "c":
                # Check for memory-related errors
                for error in error_chain:
                    if error.get("language") in ["cpp", "c"]:
                        if error.get("error_type") == "std::bad_alloc":
                            secondary_suggestions.extend([
                                "Handle memory allocation failures with proper exception handling",
                                "Consider using nothrow allocations and checking for null",
                                "Reduce memory usage or increase available memory"
                            ])
                            break
                        elif error.get("error_type") == "BufferOverflow" or "buffer overflow" in error.get("message", "").lower():
                            secondary_suggestions.extend([
                                "Add bounds checking before array/buffer access",
                                "Use safe string functions (strncpy instead of strcpy)",
                                "Validate input sizes before processing"
                            ])
                            break
                
                # Add general C/C++ cross-language suggestions
                if primary_language == "java":
                    secondary_suggestions.extend([
                        "Ensure proper exception handling in JNI code",
                        "Check for null pointers before use"
                    ])
                elif primary_language == "go":
                    secondary_suggestions.extend([
                        "Use proper CGO memory management",
                        "Add bounds checks in C code"
                    ])
            
            fixes[secondary_language] = {
                "language": secondary_language,
                "description": f"Fix in {secondary_language} code",
                "suggestion": " ".join(secondary_suggestions[:2]) if secondary_suggestions else "Validate inputs and handle errors properly",
                "suggestions": secondary_suggestions
            }
        
        result["fixes"] = fixes
        
        # Try to determine root cause if not already set
        if "root_cause" not in result:
            if "root_cause" in primary_analysis:
                result["root_cause"] = primary_analysis["root_cause"]
            elif "root_cause" in secondary_analysis:
                result["root_cause"] = secondary_analysis["root_cause"]
            else:
                result["root_cause"] = f"cross_language_{primary_language}_{secondary_language}_error"
        
        # Category
        result["category"] = "cross_language"
        
        # Severity - cross-language errors are often critical
        result["severity"] = primary_analysis.get("severity", 
                                                 secondary_analysis.get("severity", "high"))
        
        return result
    
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
        
        # Check for Java-style exceptions
        if "error_type" in error_data:
            error_type = error_data["error_type"]
            if error_type and isinstance(error_type, str):
                if ('java.lang' in error_type or 
                    'org.springframework' in error_type or
                    'java.util' in error_type or
                    'javax.' in error_type):
                    return 'java'
        
        # Check for Go-style errors
        if "goroutine_id" in error_data or "go_version" in error_data:
            return "go"
        
        # Check for Rust-style errors
        if "message" in error_data and isinstance(error_data["message"], str):
            message = error_data["message"]
            if ("panicked at" in message and ("unwrap()" in message or ".rs:" in message)):
                return "rust"
        
        # Check stack trace for language-specific patterns
        if "stack_trace" in error_data and error_data["stack_trace"]:
            stack_trace = error_data["stack_trace"]
            if isinstance(stack_trace, list):
                for frame in stack_trace:
                    if isinstance(frame, dict):
                        if "file" in frame:
                            if frame["file"].endswith(".java"):
                                return "java"
                            elif frame["file"].endswith(".go"):
                                return "go"
                            elif frame["file"].endswith(".rs"):
                                return "rust"
                    elif isinstance(frame, str):
                        if ".java:" in frame:
                            return "java"
                        elif ".go:" in frame or "goroutine " in frame:
                            return "go"
                        elif ".rs:" in frame or "thread '" in frame or "rustc[" in frame:
                            return "rust"
        
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