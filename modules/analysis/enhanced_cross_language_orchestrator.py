"""
Enhanced Cross-Language Orchestrator

This module extends the core cross-language orchestrator with advanced capabilities
for handling errors across different backend languages. It improves language detection,
adds similarity scoring, and enables more sophisticated cross-language learning.
"""
import logging
import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .cross_language_orchestrator import CrossLanguageOrchestrator
from .language_plugin_system import get_plugin, load_all_plugins
from .shared_error_schema import (
    SharedErrorSchema,
    normalize_error,
    denormalize_error,
    detect_language
)

logger = logging.getLogger(__name__)


class EnhancedCrossLanguageOrchestrator(CrossLanguageOrchestrator):
    """
    Enhanced orchestrator for cross-language error analysis and handling.
    
    This class extends the base CrossLanguageOrchestrator with:
    1. Improved language detection using shared schema
    2. Enhanced similarity scoring across languages
    3. More sophisticated cross-language learning
    4. Performance metrics and caching
    5. Cross-language rule application
    """
    
    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600):
        """
        Initialize the enhanced cross-language orchestrator.
        
        Args:
            cache_size: Maximum number of items to keep in the cache
            cache_ttl: Time-to-live for cache entries in seconds
        """
        super().__init__()
        
        # Initialize shared schema
        self.shared_schema = SharedErrorSchema()
        
        # Additional performance metrics
        self.metrics = {
            "language_detections": 0,
            "language_detection_times": [],
            "analysis_times": {},
            "fix_generation_times": {},
            "cross_language_matches": 0,
            "successful_applications": 0
        }
        
        # Cache settings
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        
        # Cache for analysis results and fix suggestions
        self.analysis_cache = {}
        self.fix_cache = {}
        self.cache_timestamps = {}
        
        # Load word embeddings for better similarity scoring
        self._load_word_embeddings()
    
    def analyze_error(self, error_data: Dict[str, Any], 
                     language: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze an error from any supported language with enhanced capabilities.
        
        Args:
            error_data: Error data
            language: Optional language identifier (auto-detected if not specified)
            
        Returns:
            Analysis results
            
        Raises:
            ValueError: If language is not supported or cannot be determined
        """
        # Check cache first
        cache_key = self._get_cache_key(error_data)
        if cache_key in self.analysis_cache:
            cache_entry = self.analysis_cache[cache_key]
            timestamp = self.cache_timestamps.get(cache_key, 0)
            
            # If cache entry is still valid
            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"Using cached analysis for {cache_key}")
                return cache_entry
        
        # Detect language if not provided, using enhanced detection
        start_time = time.time()
        if language is None:
            language = self._enhanced_language_detection(error_data)
            
            if language == "unknown":
                raise ValueError("Could not detect language from error data, please specify explicitly")
        
        language = language.lower()
        self.metrics["language_detections"] += 1
        self.metrics["language_detection_times"].append(time.time() - start_time)
        
        # Check if the language is supported
        if not self.registry.is_language_supported(language):
            raise ValueError(f"Unsupported language: {language}")
        
        # Get the appropriate analyzer
        analyzer = self.registry.get_analyzer(language)
        
        # Track analysis time
        start_time = time.time()
        
        # Analyze the error
        analysis_result = analyzer.analyze_error(error_data)
        
        # Update metrics
        analysis_time = time.time() - start_time
        if language not in self.metrics["analysis_times"]:
            self.metrics["analysis_times"][language] = []
        self.metrics["analysis_times"][language].append(analysis_time)
        
        # Store in history for learning and in cache
        self._store_enhanced_error_analysis(error_data, analysis_result, language)
        self._cache_analysis(cache_key, analysis_result)
        
        return analysis_result
    
    def generate_cross_language_fix(self, error_data: Dict[str, Any],
                                   language: Optional[str] = None,
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a fix for an error using cross-language knowledge.
        
        This method leverages fixes and patterns from other languages to improve
        fix generation for the target language.
        
        Args:
            error_data: Error data
            language: Optional language identifier (auto-detected if not specified)
            context: Context information for fix generation
            
        Returns:
            Fix data
        """
        # Detect language if not provided
        if language is None:
            language = self._enhanced_language_detection(error_data)
            
            if language == "unknown":
                raise ValueError("Could not detect language from error data, please specify explicitly")
        
        language = language.lower()
        context = context or {}
        
        # Get the plugin for this language
        plugin = get_plugin(language)
        if not plugin:
            raise ValueError(f"No plugin available for language: {language}")
        
        # First, analyze the error
        analysis = self.analyze_error(error_data, language)
        
        # Check cache for fix
        cache_key = self._get_cache_key(error_data, context=context)
        if cache_key in self.fix_cache:
            cache_entry = self.fix_cache[cache_key]
            timestamp = self.cache_timestamps.get(cache_key, 0)
            
            # If cache entry is still valid
            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"Using cached fix for {cache_key}")
                return cache_entry
        
        # Track fix generation time
        start_time = time.time()
        
        # Find relevant fixes from other languages
        cross_language_suggestions = self.find_cross_language_fixes(error_data, language)
        
        # Enhance context with cross-language information
        enhanced_context = self._enhance_context_with_cross_language(context, cross_language_suggestions)
        
        # Generate fix using the plugin
        fix = plugin.generate_fix(analysis, enhanced_context)
        
        # Update metrics
        fix_time = time.time() - start_time
        if language not in self.metrics["fix_generation_times"]:
            self.metrics["fix_generation_times"][language] = []
        self.metrics["fix_generation_times"][language].append(fix_time)
        
        # Check if we used cross-language knowledge
        if "used_cross_language" in enhanced_context and enhanced_context["used_cross_language"]:
            fix["used_cross_language"] = True
            fix["source_languages"] = enhanced_context.get("source_languages", [])
            self.metrics["cross_language_matches"] += 1
        
        # Cache the fix
        self._cache_fix(cache_key, fix)
        
        return fix
    
    def find_cross_language_fixes(self, error_data: Dict[str, Any],
                               language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find fixes for similar errors across different languages.
        
        Args:
            error_data: Error data
            language: Optional language identifier (auto-detected if not specified)
            
        Returns:
            List of cross-language fix suggestions
        """
        # Detect language if not provided
        if language is None:
            language = self._enhanced_language_detection(error_data)
            
            if language == "unknown":
                raise ValueError("Could not detect language from error data, please specify explicitly")
        
        language = language.lower()
        
        # Convert to standard format for better comparison
        # TODO: Use normalized_error for better cross-language comparison
        # normalized_error = normalize_error(error_data, language)
        
        # Find similar errors across languages
        similar_errors = self.find_similar_errors(error_data, language)
        
        # Get fixes for similar errors
        fixes = []
        for error in similar_errors:
            # Skip errors from the same language
            if error["language"] == language:
                continue
            
            # Extract information for fix generation
            error_lang = error["language"]
            similar_error = error["error"]
            similarity = error["similarity"]
            
            # Get the plugin for this language
            plugin = get_plugin(error_lang)
            if not plugin:
                continue
            
            # Try to get existing analysis or analyze
            error_id = similar_error.get("error_id", "")
            if error_id in self.error_history:
                analysis = self.error_history[error_id].get("analysis", {})
            else:
                try:
                    # We need to denormalize the error for the plugin
                    lang_error = denormalize_error(similar_error, error_lang)
                    analysis = plugin.analyze_error(lang_error)
                except Exception as e:
                    logger.warning(f"Error analyzing similar error in {error_lang}: {e}")
                    continue
            
            # Get suggestions from the analysis
            suggestion = analysis.get("suggestion", "")
            
            if suggestion:
                # Create a fix suggestion
                fix = {
                    "language": error_lang,
                    "target_language": language,
                    "suggestion": suggestion,
                    "similarity": similarity,
                    "root_cause": analysis.get("root_cause", "unknown"),
                    "confidence": analysis.get("confidence", "low") 
                }
                
                # Adjust confidence based on similarity
                if similarity < 0.5:
                    fix["confidence"] = "low"
                elif similarity < 0.7:
                    fix["confidence"] = "medium"
                
                fixes.append(fix)
        
        return fixes
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the orchestrator.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()
        
        # Calculate averages for timing metrics
        if metrics["language_detection_times"]:
            metrics["avg_language_detection_time"] = sum(metrics["language_detection_times"]) / len(metrics["language_detection_times"])
        
        # Calculate averages for analysis times by language
        metrics["avg_analysis_times"] = {}
        for lang, times in metrics["analysis_times"].items():
            if times:
                metrics["avg_analysis_times"][lang] = sum(times) / len(times)
        
        # Calculate averages for fix generation times by language
        metrics["avg_fix_generation_times"] = {}
        for lang, times in metrics["fix_generation_times"].items():
            if times:
                metrics["avg_fix_generation_times"][lang] = sum(times) / len(times)
        
        # Get cache statistics
        metrics["analysis_cache_size"] = len(self.analysis_cache)
        metrics["fix_cache_size"] = len(self.fix_cache)
        
        # Get language statistics
        metrics["supported_languages"] = self.registry.get_supported_languages()
        metrics["error_history_size"] = len(self.error_history)
        
        return metrics
    
    def clear_caches(self):
        """Clear all caches."""
        self.analysis_cache.clear()
        self.fix_cache.clear()
        self.cache_timestamps.clear()
        logger.info("Cleared all caches")
    
    def register_fix_success(self, fix_id: str):
        """
        Register that a fix was successfully applied.
        
        Args:
            fix_id: ID of the fix that was successful
        """
        self.metrics["successful_applications"] += 1
    
    def _enhanced_language_detection(self, error_data: Dict[str, Any]) -> str:
        """
        Enhanced language detection using multiple strategies.
        
        Args:
            error_data: Error data
            
        Returns:
            Detected language or "unknown"
        """
        # Try direct language field
        if "language" in error_data:
            return error_data["language"].lower()
        
        # Try using the shared schema detector
        language = detect_language(error_data)
        if language != "unknown":
            return language
        
        # Check for Rust-specific patterns (more extensive than base class)
        if "message" in error_data and isinstance(error_data["message"], str):
            message = error_data["message"]
            if ("panicked at" in message or "thread 'main'" in message or 
                    "unwrap()" in message or "Option::unwrap" in message or 
                    ".rs:" in message):
                return "rust"
                
        # Check for PHP-specific patterns
        if "message" in error_data and isinstance(error_data["message"], str):
            message = error_data["message"]
            if ("PHP Notice" in message or "PHP Warning" in message or 
                    "PHP Error" in message or "PHP Fatal error" in message or
                    "Call to undefined method" in message or 
                    "Call to undefined function" in message or
                    "Undefined variable" in message or
                    "Call to a member function" in message or
                    "SQLSTATE[" in message):
                return "php"
            
        # Check for Scala-specific patterns
        if "message" in error_data and isinstance(error_data["message"], str):
            message = error_data["message"]
            if ("scala.MatchError" in message or 
                    "scala.None$.get" in message or
                    "scala.Option" in message or
                    "akka." in message or
                    "play.api." in message or
                    "scala.concurrent.Future" in message):
                return "scala"
        
        # Check for error_type indicating Scala
        if "error_type" in error_data and isinstance(error_data["error_type"], str):
            error_type = error_data["error_type"]
            if (error_type.startswith("scala.") or 
                    error_type.startswith("akka.") or 
                    error_type.startswith("play.api.") or
                    "MatchError" in error_type):
                return "scala"
        
        # Check for stack trace patterns
        stack_trace_keys = ["stack_trace", "stacktrace", "trace", "backtrace"]
        for key in stack_trace_keys:
            if key in error_data:
                trace = error_data[key]
                
                # Check if it's a string or list
                if isinstance(trace, str):
                    if ".scala:" in trace:
                        return "scala"
                elif isinstance(trace, list) and len(trace) > 0:
                    # Check list items
                    if isinstance(trace[0], str) and ".scala:" in trace[0]:
                        return "scala"
                    elif isinstance(trace[0], dict):
                        # Check for Scala files in stack trace
                        for frame in trace:
                            file = frame.get("file", "")
                            if isinstance(file, str) and file.endswith(".scala"):
                                return "scala"
            
        # Check for PHP stack trace format
        if "trace" in error_data or "backtrace" in error_data:
            trace_key = "trace" if "trace" in error_data else "backtrace"
            trace = error_data[trace_key]
            
            if isinstance(trace, list) and len(trace) > 0:
                # Check if it looks like a PHP stack trace
                if isinstance(trace[0], dict) and "file" in trace[0] and ".php" in trace[0]["file"]:
                    return "php"
                elif isinstance(trace[0], str) and ".php" in trace[0]:
                    return "php"
            
        # Fall back to original detection logic
        return super()._detect_language(error_data)
    
    def _store_enhanced_error_analysis(self, error_data: Dict[str, Any], 
                                     analysis: Dict[str, Any], 
                                     language: str):
        """
        Store enhanced error analysis in history for learning.
        
        Args:
            error_data: Error data
            analysis: Analysis results
            language: Language identifier
        """
        # First normalize the error for consistent storage
        try:
            standard_error = normalize_error(error_data, language)
        except Exception as e:
            logger.warning(f"Error normalizing error data: {e}")
            # Fall back to default conversion
            super()._store_error_analysis(error_data, analysis, language)
            return
        
        # Generate ID if not present
        error_id = standard_error.get("error_id")
        if not error_id:
            error_id = str(uuid.uuid4())
            standard_error["error_id"] = error_id
        
        # Store in history with extra metadata
        self.error_history[error_id] = {
            "error": standard_error,
            "analysis": analysis,
            "language": language,
            "timestamp": datetime.now().isoformat(),
            "normalized": True  # Flag that this is a normalized error
        }
        
        # Limit history size
        if len(self.error_history) > 1000:
            oldest_key = min(self.error_history.keys(), 
                           key=lambda k: self.error_history[k].get("timestamp", ""))
            del self.error_history[oldest_key]
    
    def _get_cache_key(self, error_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a cache key for error data and optional context.
        
        Args:
            error_data: Error data
            context: Optional context data
            
        Returns:
            Cache key string
        """
        # Extract key components from error data
        key_parts = []
        
        # Get error type and message
        if "error_type" in error_data:
            key_parts.append(f"type:{error_data['error_type']}")
        elif "exception_type" in error_data:
            key_parts.append(f"type:{error_data['exception_type']}")
        elif "name" in error_data:
            key_parts.append(f"type:{error_data['name']}")
        
        if "message" in error_data:
            # Truncate message to reduce key size
            msg = error_data["message"]
            if len(msg) > 100:
                msg = msg[:100]
            key_parts.append(f"msg:{msg}")
        
        # Add language if available
        if "language" in error_data:
            key_parts.append(f"lang:{error_data['language']}")
        
        # Add context fingerprint if provided
        if context:
            context_str = json.dumps(context, sort_keys=True)
            import hashlib
            context_hash = hashlib.md5(context_str.encode()).hexdigest()[:10]
            key_parts.append(f"ctx:{context_hash}")
        
        return "|".join(key_parts)
    
    def _cache_analysis(self, key: str, analysis: Dict[str, Any]):
        """
        Cache analysis results.
        
        Args:
            key: Cache key
            analysis: Analysis results
        """
        # Check if cache is full
        if len(self.analysis_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(self.cache_timestamps.keys(), 
                           key=lambda k: self.cache_timestamps.get(k, 0))
            if oldest_key in self.analysis_cache:
                del self.analysis_cache[oldest_key]
            if oldest_key in self.cache_timestamps:
                del self.cache_timestamps[oldest_key]
        
        # Add to cache
        self.analysis_cache[key] = analysis
        self.cache_timestamps[key] = time.time()
    
    def _cache_fix(self, key: str, fix: Dict[str, Any]):
        """
        Cache fix results.
        
        Args:
            key: Cache key
            fix: Fix results
        """
        # Check if cache is full
        if len(self.fix_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(self.cache_timestamps.keys(), 
                           key=lambda k: self.cache_timestamps.get(k, 0))
            if oldest_key in self.fix_cache:
                del self.fix_cache[oldest_key]
            if oldest_key in self.cache_timestamps:
                del self.cache_timestamps[oldest_key]
        
        # Add to cache
        self.fix_cache[key] = fix
        self.cache_timestamps[key] = time.time()
    
    def _load_word_embeddings(self):
        """Load word embeddings for improved similarity scoring."""
        # This is a placeholder for loading word embeddings
        # In a real implementation, you would load pre-trained embeddings
        # For now, we'll just create a small vocabulary of common error terms
        self.word_vectors = {}
        self.word_vectors_loaded = False
        
        # Try to load minimal embeddings from a file if available
        embeddings_file = Path(__file__).parent / "data" / "error_term_embeddings.json"
        if embeddings_file.exists():
            try:
                with open(embeddings_file, 'r') as f:
                    self.word_vectors = json.load(f)
                self.word_vectors_loaded = True
                logger.info(f"Loaded {len(self.word_vectors)} word embeddings")
            except Exception as e:
                logger.warning(f"Error loading word embeddings: {e}")
    
    def _calculate_similarity(self, error1: Dict[str, Any], 
                             error2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two errors with enhanced scoring.
        
        Args:
            error1: First error in standard format
            error2: Second error in standard format
            
        Returns:
            Similarity score (0-1)
        """
        # Start with basic similarity from parent class
        score = super()._calculate_similarity(error1, error2)
        
        # If word vectors are loaded, use them for better scoring
        if self.word_vectors_loaded:
            # Compare error messages using word embeddings
            message1 = error1.get("message", "")
            message2 = error2.get("message", "")
            
            if message1 and message2:
                semantic_similarity = self._calculate_semantic_similarity(message1, message2)
                score += 0.2 * semantic_similarity
        
        # Compare root causes if available in both errors
        if "root_cause" in error1 and "root_cause" in error2:
            if error1["root_cause"] == error2["root_cause"]:
                score += 0.2
            elif error1["root_cause"] in error2["root_cause"] or error2["root_cause"] in error1["root_cause"]:
                score += 0.1
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # This is a placeholder for calculating semantic similarity
        # In a real implementation, you would use the word embeddings
        
        # Simple word overlap as fallback
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            return overlap / max(len(words1), len(words2))
        
        return 0.0
    
    def _enhance_context_with_cross_language(self, context: Dict[str, Any],
                                          suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhance context for fix generation with cross-language suggestions.
        
        Args:
            context: Original context
            suggestions: Cross-language fix suggestions
            
        Returns:
            Enhanced context
        """
        enhanced_context = context.copy()
        
        if suggestions:
            # Filter suggestions by confidence
            good_suggestions = [s for s in suggestions if s.get("confidence") != "low"]
            
            if good_suggestions:
                # Add cross-language suggestions to context
                enhanced_context["cross_language_suggestions"] = good_suggestions
                enhanced_context["used_cross_language"] = True
                enhanced_context["source_languages"] = [s["language"] for s in good_suggestions]
        
        return enhanced_context


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create an enhanced orchestrator
    orchestrator = EnhancedCrossLanguageOrchestrator()
    
    # Load all plugins
    load_all_plugins()
    
    # Display supported languages
    languages = orchestrator.get_supported_languages()
    logger.info(f"Supported languages: {', '.join(languages)}")
    
    # Test the orchestrator with examples from each supported language
    examples = {
        "python": {
            "exception_type": "KeyError",
            "message": "'user_id'",
            "traceback": [
                "Traceback (most recent call last):",
                "  File \"app.py\", line 42, in get_user",
                "    user_id = data['user_id']",
                "KeyError: 'user_id'"
            ],
            "level": "ERROR",
            "python_version": "3.9.7"
        },
        "javascript": {
            "name": "TypeError",
            "message": "Cannot read property 'id' of undefined",
            "stack": "TypeError: Cannot read property 'id' of undefined\n    at getUserId (/app/src/utils.js:45:20)\n    at processRequest (/app/src/controllers/user.js:23:15)",
            "level": "error"
        },
        "java": {
            "exception_class": "java.lang.NullPointerException",
            "message": "Cannot invoke \"String.length()\" because \"str\" is null",
            "stack_trace": "java.lang.NullPointerException: Cannot invoke \"String.length()\" because \"str\" is null\n    at com.example.StringProcessor.processString(StringProcessor.java:42)\n    at com.example.Main.main(Main.java:25)",
            "level": "SEVERE"
        },
        "go": {
            "error_type": "runtime error",
            "message": "nil pointer dereference",
            "stack_trace": "goroutine 1 [running]:\nmain.processValue()\n\t/app/main.go:25\nmain.main()\n\t/app/main.go:12",
            "level": "error"
        },
        "rust": {
            "error_type": "Panic",
            "message": "thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', src/main.rs:42:14",
            "stack_trace": "thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', src/main.rs:42:14\nstack backtrace:\n   0: std::panicking::begin_panic\n   1: core::option::Option<T>::unwrap\n   2: rust_example::process_data\n   3: rust_example::main",
            "level": "error"
        },
        "php": {
            "type": "ErrorException",
            "message": "Undefined variable: user",
            "file": "/var/www/html/app/Controllers/UserController.php",
            "line": 25,
            "trace": [
                {
                    "file": "/var/www/html/app/Controllers/UserController.php",
                    "line": 25,
                    "function": "getUserProfile",
                    "class": "App\\Controllers\\UserController"
                },
                {
                    "file": "/var/www/html/routes/web.php",
                    "line": 16,
                    "function": "handle",
                    "class": "App\\Http\\Kernel"
                }
            ],
            "level": "E_NOTICE",
            "php_version": "8.1.0"
        }
    }
    
    # Run analysis with each example
    for lang, error_data in examples.items():
        if lang in languages:
            logger.info(f"\nAnalyzing {lang} error...")
            
            try:
                # Test analysis
                analysis = orchestrator.analyze_error(error_data, lang)
                logger.info(f"{lang} analysis root cause: {analysis.get('root_cause')}")
                logger.info(f"{lang} analysis suggestion: {analysis.get('suggestion')}")
                
                # Test cross-language fix generation
                context = {"code_snippet": "Example code snippet"}
                fix = orchestrator.generate_cross_language_fix(error_data, lang, context)
                logger.info(f"{lang} fix: {fix.get('suggestion') or fix.get('patch_code', '(no suggestion)')}")
                
                # Find similar errors
                similar = orchestrator.find_similar_errors(error_data, lang)
                logger.info(f"Found {len(similar)} similar errors")
                
                # Find cross-language fixes
                cross_fixes = orchestrator.find_cross_language_fixes(error_data, lang)
                logger.info(f"Found {len(cross_fixes)} cross-language fix suggestions")
                
                for i, cross_fix in enumerate(cross_fixes):
                    if i < 2:  # Show only the first 2
                        logger.info(f"  Suggestion from {cross_fix['language']} (similarity: {cross_fix['similarity']:.2f}):")
                        logger.info(f"    {cross_fix['suggestion']}")
            
            except Exception as e:
                logger.error(f"Error processing {lang} example: {e}")
    
    # Show metrics
    metrics = orchestrator.get_metrics()
    logger.info("\nMetrics:")
    logger.info(f"  Language detections: {metrics['language_detections']}")
    logger.info(f"  Cross-language matches: {metrics['cross_language_matches']}")
    
    if "avg_language_detection_time" in metrics:
        logger.info(f"  Avg language detection time: {metrics['avg_language_detection_time']:.3f}s")
    
    if "avg_analysis_times" in metrics:
        logger.info("  Avg analysis times by language:")
        for lang, avg_time in metrics['avg_analysis_times'].items():
            logger.info(f"    {lang}: {avg_time:.3f}s")