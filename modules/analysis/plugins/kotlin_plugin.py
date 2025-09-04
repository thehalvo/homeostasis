"""
Kotlin Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Kotlin applications.
It provides comprehensive error handling for Android development, Kotlin Multiplatform projects,
including support for coroutines, Jetpack Compose, Room database, and modern Kotlin features.
"""
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..language_plugin_system import LanguagePlugin, register_plugin
from ..language_adapters import KotlinErrorAdapter

logger = logging.getLogger(__name__)


class KotlinExceptionHandler:
    """
    Handles Kotlin exceptions with a robust error detection and classification system.
    
    This class provides logic for categorizing Kotlin exceptions based on their type,
    message, and stack trace patterns. It supports Android, JVM, and multiplatform projects.
    """
    
    def __init__(self):
        """Initialize the Kotlin exception handler."""
        self.rule_categories = {
            "core": "Core Kotlin language exceptions",
            "coroutines": "Kotlin Coroutines and async exceptions",
            "android": "Android platform and lifecycle exceptions",
            "compose": "Jetpack Compose UI exceptions",
            "room": "Room database and persistence exceptions",
            "multiplatform": "Kotlin Multiplatform project exceptions",
            "collections": "Kotlin collections and data structure exceptions",
            "null_safety": "Null safety and optionals exceptions",
            "serialization": "Kotlin serialization exceptions",
            "interop": "Java/Kotlin interoperability exceptions",
            "gradle": "Gradle build and dependency exceptions",
            "ktor": "Ktor networking framework exceptions"
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load Kotlin error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "kotlin"
        
        try:
            # Create rules directory if it doesn't exist
            rules_dir.mkdir(parents=True, exist_ok=True)
            
            # Load common Kotlin rules
            common_rules_path = rules_dir / "kotlin_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, 'r') as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common Kotlin rules")
            else:
                rules["common"] = self._get_default_common_rules()
            
            # Load Android specific rules
            android_rules_path = rules_dir / "kotlin_android_errors.json"
            if android_rules_path.exists():
                with open(android_rules_path, 'r') as f:
                    android_data = json.load(f)
                    rules["android"] = android_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['android'])} Android rules")
            else:
                rules["android"] = self._get_default_android_rules()

            # Load coroutines specific rules
            coroutines_rules_path = rules_dir / "kotlin_coroutines_errors.json"
            if coroutines_rules_path.exists():
                with open(coroutines_rules_path, 'r') as f:
                    coroutines_data = json.load(f)
                    rules["coroutines"] = coroutines_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['coroutines'])} coroutines rules")
            else:
                rules["coroutines"] = self._get_default_coroutines_rules()

            # Load Compose specific rules
            compose_rules_path = rules_dir / "kotlin_compose_errors.json"
            if compose_rules_path.exists():
                with open(compose_rules_path, 'r') as f:
                    compose_data = json.load(f)
                    rules["compose"] = compose_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['compose'])} Compose rules")
            else:
                rules["compose"] = self._get_default_compose_rules()

            # Load Room specific rules
            room_rules_path = rules_dir / "kotlin_room_errors.json"
            if room_rules_path.exists():
                with open(room_rules_path, 'r') as f:
                    room_data = json.load(f)
                    rules["room"] = room_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['room'])} Room rules")
            else:
                rules["room"] = self._get_default_room_rules()
                    
        except Exception as e:
            logger.error(f"Error loading Kotlin rules: {e}")
            rules = {
                "common": self._get_default_common_rules(),
                "android": self._get_default_android_rules(),
                "coroutines": self._get_default_coroutines_rules(),
                "compose": self._get_default_compose_rules(),
                "room": self._get_default_room_rules()
            }
        
        return rules
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        self.compiled_patterns = {}
        
        for category, rule_list in self.rules.items():
            self.compiled_patterns[category] = []
            for rule in rule_list:
                try:
                    pattern = rule.get("pattern", "")
                    if pattern:
                        compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                        self.compiled_patterns[category].append((compiled, rule))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern in rule {rule.get('id', 'unknown')}: {e}")
    
    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Kotlin exception and determine its type and potential fixes.
        
        Args:
            error_data: Kotlin error data in standard format
            
        Returns:
            Analysis results with categorization and fix suggestions
        """
        error_type = error_data.get("error_type", "Error")
        message = error_data.get("message", "")
        stack_trace = error_data.get("stack_trace", [])
        
        # Convert stack trace to string for pattern matching
        stack_str = ""
        if isinstance(stack_trace, list):
            stack_str = "\n".join([str(frame) for frame in stack_trace])
        elif isinstance(stack_trace, str):
            stack_str = stack_trace
        
        # Combine error info for analysis
        full_error_text = f"{error_type}: {message}\n{stack_str}"
        
        # Debug logging
        logger.debug(f"Analyzing error_type: {error_type}")
        logger.debug(f"Full error text for pattern matching: {full_error_text[:200]}...")
        
        # Find matching rules
        matches = self._find_matching_rules(full_error_text, error_data)
        
        if matches:
            # Use the best match (highest confidence)
            best_match = max(matches, key=lambda x: x.get("confidence_score", 0))
            return self._create_analysis_result(best_match, error_data)
        
        # If no specific rule matched, use fallback analysis
        return self._create_fallback_analysis(error_data)
    
    def _find_matching_rules(self, error_text: str, error_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find rules that match the given error."""
        matches = []
        
        for category, pattern_list in self.compiled_patterns.items():
            for compiled_pattern, rule in pattern_list:
                try:
                    match = compiled_pattern.search(error_text)
                    if match:
                        confidence_score = self._calculate_confidence(rule, error_data, match)
                        rule_copy = rule.copy()
                        rule_copy["confidence_score"] = confidence_score
                        rule_copy["match_groups"] = match.groups()
                        # Keep the rule's category if it has one, otherwise use the rule set category
                        if "category" not in rule_copy:
                            rule_copy["category"] = category
                        matches.append(rule_copy)
                except Exception as e:
                    logger.warning(f"Error applying rule {rule.get('id', 'unknown')}: {e}")
        
        return matches
    
    def _calculate_confidence(self, rule: Dict[str, Any], error_data: Dict[str, Any], match: Any) -> float:
        """Calculate confidence score for a rule match."""
        confidence_str = rule.get("confidence", "medium")
        
        # Map confidence strings to numeric values
        confidence_map = {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7
        }
        base_confidence = confidence_map.get(confidence_str, 0.5)
        
        # Boost confidence if error type matches exactly
        error_type = error_data.get("error_type", "")
        if rule.get("error_type") and rule["error_type"] in error_type:
            base_confidence += 0.2
        
        # Boost confidence for specific patterns with capture groups
        if match.groups():
            base_confidence += 0.1
        
        # Boost confidence for framework-specific rules when framework is detected
        framework = rule.get("framework", "")
        if framework and framework in error_data.get("context", {}).get("frameworks", []):
            base_confidence += 0.15
        
        return min(base_confidence, 1.0)
    
    def _create_analysis_result(self, rule: Dict[str, Any], error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create analysis result from a matched rule."""
        return {
            "error_data": error_data,
            "rule_id": rule.get("id", "unknown"),
            "error_type": rule.get("error_type", error_data.get("error_type", "Unknown")),
            "root_cause": rule.get("root_cause", "kotlin_unknown_error"),
            "description": rule.get("description", "Unknown Kotlin error"),
            "suggestion": rule.get("suggestion", "No suggestion available"),
            "confidence": rule.get("confidence", "medium"),
            "severity": rule.get("severity", "medium"),
            "category": rule.get("category", "kotlin"),
            "framework": rule.get("framework", ""),
            "match_groups": rule.get("match_groups", tuple()),
            "confidence_score": rule.get("confidence_score", 0.5)
        }
    
    def _create_fallback_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback analysis for unmatched errors."""
        error_type = error_data.get("error_type", "Unknown")
        
        # Basic classification based on common Kotlin exceptions
        if "KotlinNullPointerException" in error_type or "NullPointerException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "kotlin_null_pointer_fallback",
                "error_type": error_type,
                "root_cause": "kotlin_null_pointer",
                "description": "Attempted to access a null reference",
                "suggestion": "Add null safety checks using safe call operator (?.) or null checks",
                "confidence": "medium",
                "severity": "high",
                "category": "null_safety",
                "framework": "",
                "match_groups": tuple(),
                "confidence_score": 0.6
            }
        elif "IndexOutOfBoundsException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "kotlin_index_out_of_bounds_fallback",
                "error_type": error_type,
                "root_cause": "kotlin_index_out_of_bounds",
                "description": "Attempted to access a list/array element with invalid index",
                "suggestion": "Check collection bounds using size property before accessing elements",
                "confidence": "medium",
                "severity": "medium",
                "category": "collections",
                "framework": "",
                "match_groups": tuple(),
                "confidence_score": 0.6
            }
        elif "CancellationException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "kotlin_cancellation_fallback",
                "error_type": error_type,
                "root_cause": "kotlin_coroutine_cancelled",
                "description": "Coroutine was cancelled during execution",
                "suggestion": "Handle CancellationException appropriately or check if job is active",
                "confidence": "medium",
                "severity": "medium",
                "category": "coroutines",
                "framework": "coroutines",
                "match_groups": tuple(),
                "confidence_score": 0.6
            }
        
        # Generic fallback
        return {
            "error_data": error_data,
            "rule_id": "kotlin_generic_fallback",
            "error_type": error_type,
            "root_cause": "kotlin_unknown_error",
            "description": f"Unrecognized Kotlin error: {error_type}",
            "suggestion": "Review the error message and stack trace for more details",
            "confidence": "low",
            "severity": "medium",
            "category": "core",
            "framework": "",
            "match_groups": tuple(),
            "confidence_score": 0.3
        }
    
    def _get_default_common_rules(self) -> List[Dict[str, Any]]:
        """Get default common Kotlin rules."""
        return [
            {
                "id": "kotlin_null_pointer_exception",
                "pattern": r"(kotlin\.)?KotlinNullPointerException(?::\s*(.*))?",
                "error_type": "KotlinNullPointerException",
                "description": "Attempted to access a null reference",
                "root_cause": "kotlin_null_pointer",
                "suggestion": "Use safe call operator (?.) or add null checks before accessing properties/methods",
                "confidence": "high",
                "severity": "high",
                "category": "null_safety"
            },
            {
                "id": "kotlin_null_pointer_exception",
                "pattern": r"Attempt to invoke virtual method.*on a null object reference",
                "error_type": "NullPointerException",
                "description": "Attempted to access a null reference",
                "root_cause": "kotlin_null_pointer",
                "suggestion": "Use safe call operator (?.) or add null checks before accessing properties/methods",
                "confidence": "high",
                "severity": "high",
                "category": "null_safety"
            },
            {
                "id": "kotlin_class_cast_exception",
                "pattern": r"java\.lang\.ClassCastException.*cannot be cast to.*",
                "error_type": "ClassCastException",
                "description": "Failed to cast object to expected type",
                "root_cause": "kotlin_invalid_cast",
                "suggestion": "Use safe cast operator (as?) instead of unsafe cast (as) or add type checks",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "kotlin_illegal_argument_exception",
                "pattern": r"java\.lang\.IllegalArgumentException(?::\s*(.*))?",
                "error_type": "IllegalArgumentException",
                "description": "Invalid argument passed to function",
                "root_cause": "kotlin_invalid_argument",
                "suggestion": "Validate function arguments using require() or check preconditions",
                "confidence": "medium",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "kotlin_illegal_state_exception",
                "pattern": r"java\.lang\.IllegalStateException(?::\s*(.*))?",
                "error_type": "IllegalStateException",
                "description": "Object is in invalid state for requested operation",
                "root_cause": "kotlin_invalid_state",
                "suggestion": "Use check() to validate object state or ensure proper initialization",
                "confidence": "medium",
                "severity": "medium",
                "category": "core"
            }
        ]
    
    def _get_default_android_rules(self) -> List[Dict[str, Any]]:
        """Get default Android-specific Kotlin rules."""
        return [
            {
                "id": "kotlin_activity_not_found",
                "pattern": r"ActivityNotFoundException",
                "error_type": "ActivityNotFoundException",
                "description": "No activity found to handle the given intent",
                "root_cause": "kotlin_activity_not_found",
                "suggestion": "Check intent action and verify activity is declared in AndroidManifest.xml",
                "confidence": "high",
                "severity": "high",
                "category": "android",
                "framework": "android"
            },
            {
                "id": "kotlin_view_not_found",
                "pattern": r"findViewById.*returned null",
                "error_type": "NullPointerException",
                "description": "View with specified ID not found in layout",
                "root_cause": "kotlin_view_not_found",
                "suggestion": "Verify view ID exists in layout file and is accessible from current activity/fragment",
                "confidence": "high",
                "severity": "medium",
                "category": "android",
                "framework": "android"
            },
            {
                "id": "kotlin_fragment_not_attached",
                "pattern": r"Fragment.*not attached to.*context",
                "error_type": "IllegalStateException",
                "description": "Fragment is not attached to an activity",
                "root_cause": "kotlin_fragment_not_attached",
                "suggestion": "Check if fragment is attached using isAdded before accessing context or activity",
                "confidence": "high",
                "severity": "medium",
                "category": "android",
                "framework": "android"
            },
            {
                "id": "kotlin_memory_leak",
                "pattern": r".*memory leak.*Activity.*has leaked.*",
                "error_type": "MemoryLeak",
                "description": "Activity context is being held after destruction",
                "root_cause": "kotlin_memory_leak",
                "suggestion": "Use weak references for long-lived objects holding activity context or use application context",
                "confidence": "medium",
                "severity": "high",
                "category": "android",
                "framework": "android"
            }
        ]
    
    def _get_default_coroutines_rules(self) -> List[Dict[str, Any]]:
        """Get default coroutines-specific Kotlin rules."""
        return [
            {
                "id": "kotlin_cancellation_exception",
                "pattern": r"kotlinx\.coroutines\.CancellationException",
                "error_type": "CancellationException",
                "description": "Coroutine was cancelled during execution",
                "root_cause": "kotlin_coroutine_cancelled",
                "suggestion": "Handle CancellationException properly or check Job.isActive before performing operations",
                "confidence": "high",
                "severity": "medium",
                "category": "coroutines",
                "framework": "coroutines"
            },
            {
                "id": "kotlin_timeout_cancellation",
                "pattern": r"kotlinx\.coroutines\.TimeoutCancellationException",
                "error_type": "TimeoutCancellationException",
                "description": "Coroutine operation timed out",
                "root_cause": "kotlin_coroutine_timeout",
                "suggestion": "Increase timeout duration or optimize the coroutine operation for better performance",
                "confidence": "high",
                "severity": "medium",
                "category": "coroutines",
                "framework": "coroutines"
            },
            {
                "id": "kotlin_job_cancellation",
                "pattern": r".*Job was cancelled.*",
                "error_type": "CancellationException",
                "description": "Parent job was cancelled, cancelling child coroutines",
                "root_cause": "kotlin_coroutine_cancelled",
                "suggestion": "Handle CancellationException gracefully and avoid starting new coroutines from cancelled job",
                "confidence": "medium",
                "severity": "medium",
                "category": "coroutines",
                "framework": "coroutines"
            },
            {
                "id": "kotlin_concurrent_modification",
                "pattern": r".*ConcurrentModificationException.*suspend.*",
                "error_type": "ConcurrentModificationException",
                "description": "Collection modified while being accessed in coroutine",
                "root_cause": "kotlin_concurrent_modification",
                "suggestion": "Use thread-safe collections or synchronize access using Mutex",
                "confidence": "medium",
                "severity": "high",
                "category": "coroutines",
                "framework": "coroutines"
            }
        ]
    
    def _get_default_compose_rules(self) -> List[Dict[str, Any]]:
        """Get default Jetpack Compose-specific Kotlin rules."""
        return [
            {
                "id": "kotlin_compose_recomposition_loop",
                "pattern": r".*recomposition.*infinite.*loop.*",
                "error_type": "InfiniteRecomposition",
                "description": "Infinite recomposition loop in Compose UI",
                "root_cause": "kotlin_compose_infinite_recomposition",
                "suggestion": "Use remember{} for expensive calculations and avoid creating new objects in composition",
                "confidence": "medium",
                "severity": "high",
                "category": "compose",
                "framework": "compose"
            },
            {
                "id": "kotlin_compose_state_not_remembered",
                "pattern": r".*State.*not.*remembered.*",
                "error_type": "StateNotRemembered",
                "description": "State object not properly remembered across recompositions",
                "root_cause": "kotlin_compose_state_not_remembered",
                "suggestion": "Wrap state creation with remember{} to persist across recompositions",
                "confidence": "high",
                "severity": "medium",
                "category": "compose",
                "framework": "compose"
            },
            {
                "id": "kotlin_compose_invalid_composition",
                "pattern": r".*invalid.*composition.*context.*",
                "error_type": "InvalidCompositionContext",
                "description": "Composable called outside valid composition context",
                "root_cause": "kotlin_compose_invalid_context",
                "suggestion": "Ensure Composables are only called from within other Composables or composition functions",
                "confidence": "high",
                "severity": "high",
                "category": "compose",
                "framework": "compose"
            }
        ]
    
    def _get_default_room_rules(self) -> List[Dict[str, Any]]:
        """Get default Room database-specific Kotlin rules."""
        return [
            {
                "id": "kotlin_room_database_not_initialized",
                "pattern": r".*Room.*database.*not.*initialized.*",
                "error_type": "IllegalStateException",
                "description": "Room database accessed before initialization",
                "root_cause": "kotlin_room_not_initialized",
                "suggestion": "Initialize Room database instance before accessing DAOs",
                "confidence": "high",
                "severity": "high",
                "category": "room",
                "framework": "room"
            },
            {
                "id": "kotlin_room_main_thread_query",
                "pattern": r".*Cannot access database on the main thread.*",
                "error_type": "IllegalStateException",
                "description": "Database query executed on main thread",
                "root_cause": "kotlin_room_main_thread_access",
                "suggestion": "Execute database operations in background thread or use suspend functions",
                "confidence": "high",
                "severity": "high",
                "category": "room",
                "framework": "room"
            },
            {
                "id": "kotlin_room_migration_missing",
                "pattern": r".*Migration.*not found.*",
                "error_type": "IllegalStateException",
                "description": "Database migration path not found for schema version",
                "root_cause": "kotlin_room_migration_missing",
                "suggestion": "Implement migration strategy or allow destructive migrations for development",
                "confidence": "high",
                "severity": "high",
                "category": "room",
                "framework": "room"
            }
        ]


class KotlinPatchGenerator:
    """
    Generates patch solutions for Kotlin exceptions.
    
    This class provides capabilities to generate code fixes for common Kotlin errors,
    using templates and contextual information about the exception.
    """
    
    def __init__(self):
        """Initialize the Kotlin patch generator."""
        self.templates_dir = Path(__file__).parent.parent / "patch_generation" / "templates" / "kotlin"
        self.templates_dir.mkdir(exist_ok=True, parents=True)
        
        # Cache for loaded templates
        self.template_cache = {}
    
    def generate_patch(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a patch for a Kotlin error based on analysis.
        
        Args:
            analysis: Error analysis containing root cause and other details
            context: Additional context about the error, including code snippets
            
        Returns:
            Patch data including patch type, code, and application instructions
        """
        root_cause = analysis.get("root_cause", "unknown")
        rule_id = analysis.get("rule_id", "unknown")
        framework = analysis.get("framework", "")
        
        # Basic patch result structure
        patch_result = {
            "patch_id": f"kotlin_{rule_id}",
            "patch_type": "suggestion",
            "language": "kotlin",
            "framework": framework,
            "suggestion": analysis.get("suggestion", "No suggestion available"),
            "confidence": analysis.get("confidence", "low"),
            "severity": analysis.get("severity", "medium"),
            "root_cause": root_cause
        }
        
        # Try to find a specific template for this root cause
        template_name = f"{root_cause}.kt.template"
        template_path = self.templates_dir / template_name
        
        code_snippet = context.get("code_snippet", "")
        stack_frames = analysis.get("error_data", {}).get("stack_trace", [])
        
        # If we have a template and enough context, generate actual code
        if template_path.exists() and (code_snippet or stack_frames):
            try:
                template_content = self._load_template(template_path)
                
                # Extract variable names and contextual information
                variables = self._extract_variables(analysis, context)
                
                # Apply template with variables
                patch_code = self._apply_template(template_content, variables)
                
                # Update patch result with actual code
                patch_result.update({
                    "patch_type": "code",
                    "patch_code": patch_code,
                    "application_point": self._determine_application_point(analysis, context),
                    "instructions": self._generate_instructions(analysis, patch_code)
                })
                
                # Increase confidence for code patches
                if patch_result["confidence"] == "low":
                    patch_result["confidence"] = "medium"
            except Exception as e:
                logger.warning(f"Error generating patch for {root_cause}: {e}")
        
        # If we don't have a specific template, return a suggestion-based patch
        if "patch_code" not in patch_result:
            # Enhance the suggestion based on the rule
            suggestion_code = self._generate_suggestion_code(analysis, context)
            if suggestion_code:
                patch_result["suggestion_code"] = suggestion_code
        
        return patch_result
    
    def _load_template(self, template_path: Path) -> str:
        """Load a template from the filesystem or cache."""
        path_str = str(template_path)
        if path_str not in self.template_cache:
            if template_path.exists():
                with open(template_path, 'r') as f:
                    self.template_cache[path_str] = f.read()
            else:
                raise FileNotFoundError(f"Template not found: {template_path}")
        
        return self.template_cache[path_str]
    
    def _extract_variables(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Extract variables from analysis and context for template substitution."""
        variables = {}
        
        # Extract basic information
        error_data = analysis.get("error_data", {})
        variables["ERROR_TYPE"] = error_data.get("error_type", "Exception")
        variables["ERROR_MESSAGE"] = error_data.get("message", "Unknown error")
        
        # Extract information from stack trace
        stack_trace = error_data.get("stack_trace", [])
        if stack_trace and isinstance(stack_trace, list):
            if isinstance(stack_trace[0], dict):
                # Structured stack trace
                if stack_trace:
                    top_frame = stack_trace[0]
                    variables["CLASS_NAME"] = top_frame.get("class", "")
                    variables["METHOD_NAME"] = top_frame.get("function", "")
                    variables["FILE_NAME"] = top_frame.get("file", "")
                    variables["LINE_NUMBER"] = str(top_frame.get("line", ""))
                    variables["PACKAGE_NAME"] = top_frame.get("package", "")
        
        # Extract variables from context
        variables["CODE_SNIPPET"] = context.get("code_snippet", "")
        variables["METHOD_PARAMS"] = context.get("method_params", "")
        variables["CLASS_IMPORTS"] = context.get("imports", "")
        variables["EXCEPTION_VAR"] = "e"  # Default exception variable name
        
        return variables
    
    def _apply_template(self, template: str, variables: Dict[str, str]) -> str:
        """Apply variables to a template."""
        result = template
        for key, value in variables.items():
            placeholder = f"${{{key}}}"
            result = result.replace(placeholder, value)
        return result
    
    def _determine_application_point(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Determine where to apply the patch."""
        error_data = analysis.get("error_data", {})
        stack_trace = error_data.get("stack_trace", [])
        
        application_point = {
            "type": "suggestion",
            "description": "Review the code based on the suggestion"
        }
        
        if stack_trace and isinstance(stack_trace, list):
            if isinstance(stack_trace[0], dict):
                # We have structured stack trace, extract file and line
                top_frame = stack_trace[0]
                application_point.update({
                    "type": "line",
                    "file": top_frame.get("file", ""),
                    "line": top_frame.get("line", 0),
                    "class": top_frame.get("class", ""),
                    "method": top_frame.get("function", "")
                })
        
        return application_point
    
    def _generate_instructions(self, analysis: Dict[str, Any], patch_code: str) -> str:
        """Generate human-readable instructions for applying the patch."""
        root_cause = analysis.get("root_cause", "unknown")
        
        if "null_pointer" in root_cause:
            return ("Add null safety checks using Kotlin's null safety features. " 
                   f"Consider implementing this fix: {patch_code}")
        elif "index_out_of_bounds" in root_cause:
            return ("Validate indices before accessing collections or arrays. " 
                   f"Implement bounds checking as shown: {patch_code}")
        elif "coroutine" in root_cause:
            return ("Handle coroutine lifecycle and cancellation properly. " 
                   f"Use the provided solution: {patch_code}")
        else:
            return f"Apply the following fix to address the issue: {patch_code}"
    
    def _generate_suggestion_code(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Optional[str]:
        """Generate suggestion code based on the error type."""
        root_cause = analysis.get("root_cause", "")
        
        if "kotlin_null_pointer" in root_cause:
            return self._generate_null_safety_suggestion()
        elif "kotlin_index_out_of_bounds" in root_cause:
            return self._generate_bounds_check_suggestion()
        elif "kotlin_coroutine_cancelled" in root_cause:
            return self._generate_coroutine_cancellation_suggestion()
        elif "kotlin_compose" in root_cause:
            return self._generate_compose_suggestion(root_cause)
        elif "kotlin_room" in root_cause:
            return self._generate_room_suggestion(root_cause)
        
        return None
    
    def _generate_null_safety_suggestion(self) -> str:
        """Generate suggestion for null safety issues."""
        return """// Use Kotlin's null safety features
// Option 1: Safe call operator
val result = obj?.method()

// Option 2: Null check with early return
if (obj == null) return

// Option 3: Elvis operator with default
val value = obj?.property ?: "default"

// Option 4: Let function for non-null execution
obj?.let { 
    // Execute only if obj is not null
    it.doSomething()
}"""
    
    def _generate_bounds_check_suggestion(self) -> str:
        """Generate suggestion for bounds checking."""
        return """// Check collection bounds before accessing
if (index >= 0 && index < list.size) {
    val item = list[index]
} else {
    // Handle invalid index
    println("Index $index is out of bounds")
}

// Or use getOrNull for safe access
val item = list.getOrNull(index)
item?.let { 
    // Use item safely
}

// Or use getOrElse with default
val item = list.getOrElse(index) { defaultValue }"""
    
    def _generate_coroutine_cancellation_suggestion(self) -> str:
        """Generate suggestion for coroutine cancellation handling."""
        return """// Handle coroutine cancellation properly
try {
    // Coroutine work
    delay(1000)
    // More work
} catch (e: CancellationException) {
    // Cleanup if needed
    throw e // Re-throw cancellation
}

// Or check if job is still active
if (coroutineContext.isActive) {
    // Continue work only if not cancelled
}

// Use withContext for cancellable operations
withContext(Dispatchers.IO) {
    // IO work that respects cancellation
}"""
    
    def _generate_compose_suggestion(self, root_cause: str) -> str:
        """Generate suggestion for Compose-related issues."""
        if "infinite_recomposition" in root_cause:
            return """// Avoid infinite recomposition
@Composable
fun MyComposable() {
    // Use remember for expensive calculations
    val expensiveValue = remember {
        calculateExpensiveValue()
    }
    
    // Use derivedStateOf for derived state
    val derivedState by remember {
        derivedStateOf { someState.value.transform() }
    }
}"""
        elif "state_not_remembered" in root_cause:
            return """// Remember state across recompositions
@Composable
fun MyComposable() {
    // Wrong: State will be recreated on every recomposition
    // val state = mutableStateOf("")
    
    // Correct: State is remembered
    val state = remember { mutableStateOf("") }
    
    // Or use rememberSaveable for process death
    val savedState = rememberSaveable { mutableStateOf("") }
}"""
        else:
            return """// Ensure Composables are called in proper context
@Composable
fun MyComposable() {
    // Composables can only be called from other Composables
    // or from composition functions
}"""
    
    def _generate_room_suggestion(self, root_cause: str) -> str:
        """Generate suggestion for Room database issues."""
        if "main_thread_access" in root_cause:
            return """// Execute database operations off main thread
class MyRepository(private val dao: MyDao) {
    // Use suspend functions
    suspend fun getData(): List<MyEntity> {
        return withContext(Dispatchers.IO) {
            dao.getAllData()
        }
    }
    
    // Or use callbacks/LiveData/Flow
    fun getDataAsync(): LiveData<List<MyEntity>> {
        return dao.getAllDataLiveData()
    }
}"""
        elif "migration_missing" in root_cause:
            return """// Handle database migrations
@Database(
    entities = [MyEntity::class],
    version = 2,
    exportSchema = false
)
@TypeConverters(Converters::class)
abstract class AppDatabase : RoomDatabase() {
    
    companion object {
        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL("ALTER TABLE MyEntity ADD COLUMN newColumn TEXT")
            }
        }
        
        // For development only
        fun buildDatabase(context: Context): AppDatabase {
            return Room.databaseBuilder(context, AppDatabase::class.java, "app.db")
                .addMigrations(MIGRATION_1_2)
                // .fallbackToDestructiveMigration() // Only for development
                .build()
        }
    }
}"""
        else:
            return """// Initialize Room database properly
class DatabaseModule {
    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): AppDatabase {
        return Room.databaseBuilder(
            context,
            AppDatabase::class.java,
            "app_database"
        ).build()
    }
    
    @Provides
    fun provideDao(database: AppDatabase): MyDao {
        return database.myDao()
    }
}"""


class KotlinLanguagePlugin(LanguagePlugin):
    """
    Kotlin language plugin for Homeostasis.
    
    Provides comprehensive error analysis and fix generation for Kotlin applications,
    including support for Android, coroutines, Jetpack Compose, Room, and multiplatform projects.
    """
    
    VERSION = "0.1.0"
    AUTHOR = "Homeostasis Contributors"
    
    def __init__(self):
        """Initialize the Kotlin language plugin."""
        # We'll create the adapter when needed
        self.adapter = None
        self.exception_handler = KotlinExceptionHandler()
        self.patch_generator = KotlinPatchGenerator()
    
    def get_language_id(self) -> str:
        """Get the language identifier."""
        return "kotlin"
    
    def get_language_name(self) -> str:
        """Get the language name."""
        return "Kotlin"
    
    def get_language_version(self) -> str:
        """Get the language version."""
        return "1.8+"
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Kotlin error.
        
        Args:
            error_data: Kotlin error data
            
        Returns:
            Analysis results
        """
        # First, normalize the error if needed
        if "language" not in error_data or error_data["language"] != "kotlin":
            standard_error = self.normalize_error(error_data)
        else:
            standard_error = error_data
        
        # Use the exception handler to analyze the error
        analysis = self.exception_handler.analyze_exception(standard_error)
        
        # Ensure language is included in the analysis result
        analysis["language"] = "kotlin"
        
        return analysis
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a Kotlin error to the standard format.
        
        Args:
            error_data: Kotlin error data
            
        Returns:
            Error data in the standard format
        """
        # Create adapter if needed
        if self.adapter is None:
            self.adapter = KotlinErrorAdapter()
        
        return self.adapter.to_standard_format(error_data)
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Kotlin format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the Kotlin format
        """
        # Create adapter if needed
        if self.adapter is None:
            self.adapter = KotlinErrorAdapter()
        
        return self.adapter.from_standard_format(standard_error)
    
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a fix for a Kotlin error.
        
        Args:
            analysis: Error analysis
            context: Additional context for fix generation
            
        Returns:
            Generated fix data
        """
        return self.patch_generator.generate_patch(analysis, context)
    
    def get_supported_frameworks(self) -> List[str]:
        """
        Get the list of frameworks supported by this language plugin.
        
        Returns:
            List of supported framework identifiers
        """
        return ["android", "coroutines", "compose", "room", "multiplatform", "ktor", "base"]


# Register this plugin
register_plugin(KotlinLanguagePlugin())