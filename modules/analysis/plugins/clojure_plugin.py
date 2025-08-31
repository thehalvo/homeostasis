"""
Clojure Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Clojure applications.
It provides comprehensive exception handling for standard Clojure errors and
framework-specific issues including Ring, Compojure, core.async, and JVM interop.
"""
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from ..language_plugin_system import LanguagePlugin, register_plugin
from ..language_adapters import ClojureErrorAdapter

logger = logging.getLogger(__name__)


class ClojureExceptionHandler:
    """
    Handles Clojure exceptions with a robust error detection and classification system.
    
    This class provides logic for categorizing Clojure exceptions based on their type,
    message, and stack trace patterns. It supports both standard Clojure exceptions and
    framework-specific exceptions.
    """
    
    def __init__(self):
        """Initialize the Clojure exception handler."""
        self.rule_categories = {
            "core": "Core Clojure exceptions",
            "syntax": "Syntax and compilation errors",
            "runtime": "Runtime and evaluation errors",
            "collections": "Collection and sequence exceptions",
            "concurrency": "Threading and concurrency exceptions",
            "ring": "Ring web framework exceptions",
            "compojure": "Compojure routing exceptions",
            "async": "core.async channel and blocking exceptions",
            "functional": "Functional programming related exceptions",
            "macros": "Macro expansion and compilation exceptions",
            "deps": "Dependency and namespace loading exceptions",
            "jvm": "JVM interop exceptions",
            "repl": "REPL evaluation exceptions",
            "spec": "clojure.spec validation exceptions"
        }
        
        # Load rules from different categories
        self.rules = self._load_all_rules()
        
        # Initialize caches for performance
        self.pattern_cache = {}  # Compiled regex patterns
        self.rule_match_cache = {}  # Previous rule matches
    
    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Clojure exception to determine its root cause and suggest potential fixes.
        
        Args:
            error_data: Clojure error data in standard format
            
        Returns:
            Analysis result with root cause, description, and fix suggestions
        """
        error_type = error_data.get("error_type", "")
        message = error_data.get("message", "")
        stack_trace = error_data.get("stack_trace", [])
        
        # Create a consolidated text for pattern matching
        match_text = self._create_match_text(error_type, message, stack_trace)
        
        # Try to match against known rules
        for rule in self.rules:
            pattern = rule.get("pattern", "")
            if not pattern:
                continue
            
            # Skip rules that don't apply to this category of exception
            if rule.get("applies_to") and error_type:
                applies_to_patterns = rule.get("applies_to")
                if not any(re.search(pattern, error_type) for pattern in applies_to_patterns):
                    continue
            
            # Get or compile the regex pattern
            if pattern not in self.pattern_cache:
                try:
                    self.pattern_cache[pattern] = re.compile(pattern, re.IGNORECASE | re.DOTALL)
                except Exception as e:
                    logger.warning(f"Invalid pattern in rule {rule.get('id', 'unknown')}: {e}")
                    continue
            
            # Try to match the pattern
            try:
                match = self.pattern_cache[pattern].search(match_text)
                if match:
                    # Create analysis result based on the matched rule
                    result = {
                        "error_data": error_data,
                        "rule_id": rule.get("id", "unknown"),
                        "error_type": rule.get("type", error_type),
                        "category": rule.get("category", "unknown"),
                        "description": rule.get("description", ""),
                        "root_cause": rule.get("root_cause", ""),
                        "fix_suggestions": rule.get("fix_suggestions", []),
                        "confidence": rule.get("confidence", 0.5),
                        "severity": rule.get("severity", "medium"),
                        "tags": rule.get("tags", []),
                        "frameworks": rule.get("frameworks", []),
                        "clojure_versions": rule.get("clojure_versions", []),
                        "match_groups": match.groups() if match.groups() else []
                    }
                    
                    # Enhanced analysis based on context
                    result = self._enhance_analysis(result, error_data)
                    
                    return result
            except Exception as e:
                logger.warning(f"Error matching pattern in rule {rule.get('id', 'unknown')}: {e}")
                continue
        
        # No specific rule matched, return a generic analysis
        return self._create_generic_analysis(error_data)
    
    def _load_all_rules(self) -> List[Dict[str, Any]]:
        """
        Load all Clojure error detection rules from JSON files.
        
        Returns:
            List of all loaded rules
        """
        rules = []
        rules_dir = Path(__file__).parent.parent / "rules" / "clojure"
        
        if not rules_dir.exists():
            logger.warning(f"Clojure rules directory not found: {rules_dir}")
            return rules
        
        # Load rules from different category files
        rule_files = [
            "common_errors.json",
            "syntax_errors.json", 
            "runtime_errors.json",
            "ring_errors.json",
            "compojure_errors.json",
            "async_errors.json",
            "deps_errors.json",
            "jvm_interop_errors.json",
            "spec_errors.json"
        ]
        
        for rule_file in rule_files:
            rule_path = rules_dir / rule_file
            if rule_path.exists():
                try:
                    with open(rule_path, 'r') as f:
                        file_rules = json.load(f)
                        if isinstance(file_rules, list):
                            rules.extend(file_rules)
                        elif isinstance(file_rules, dict) and "rules" in file_rules:
                            rules.extend(file_rules["rules"])
                        else:
                            logger.warning(f"Unexpected format in rule file: {rule_file}")
                except Exception as e:
                    logger.error(f"Error loading Clojure rules from {rule_file}: {e}")
        
        logger.info(f"Loaded {len(rules)} Clojure error detection rules")
        return rules
    
    def _create_match_text(self, error_type: str, message: str, stack_trace: Union[List, str]) -> str:
        """
        Create consolidated text for pattern matching.
        
        Args:
            error_type: Exception type/class
            message: Exception message
            stack_trace: Stack trace (list or string)
            
        Returns:
            Consolidated text for matching
        """
        parts = [error_type, message]
        
        if isinstance(stack_trace, list):
            if all(isinstance(frame, str) for frame in stack_trace):
                parts.extend(stack_trace[:5])  # First 5 stack frames
            elif all(isinstance(frame, dict) for frame in stack_trace):
                # Extract string representations from structured frames
                for frame in stack_trace[:5]:
                    frame_text = f"{frame.get('namespace', '')}.{frame.get('function', '')} ({frame.get('file', '')}:{frame.get('line', '')})"
                    parts.append(frame_text)
        elif isinstance(stack_trace, str):
            # Take first 5 lines of string stack trace
            stack_lines = stack_trace.split('\n')[:5]
            parts.extend(stack_lines)
        
        return '\n'.join(filter(None, parts))
    
    def _enhance_analysis(self, base_result: Dict[str, Any], error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance analysis result with additional context and recommendations.
        
        Args:
            base_result: Basic analysis result from rule matching
            error_data: Original error data
            
        Returns:
            Enhanced analysis result
        """
        # Detect framework context
        framework_context = self._detect_framework_context(error_data)
        if framework_context:
            base_result["framework_context"] = framework_context
        
        # Detect Clojure version specific issues
        version_context = self._detect_version_context(error_data)
        if version_context:
            base_result["version_context"] = version_context
        
        # Add namespace context
        namespace_context = self._extract_namespace_context(error_data)
        if namespace_context:
            base_result["namespace_context"] = namespace_context
        
        # Enhance fix suggestions based on context
        base_result["fix_suggestions"] = self._enhance_fix_suggestions(
            base_result.get("fix_suggestions", []),
            error_data,
            framework_context
        )
        
        return base_result
    
    def _detect_framework_context(self, error_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect web framework or library context from error data."""
        stack_trace = error_data.get("stack_trace", [])
        
        framework_patterns = {
            "ring": ["ring.", "compojure.", "bidi.", "reitit."],
            "pedestal": ["io.pedestal"],
            "luminus": ["luminus.", "mount."],
            "datomic": ["datomic."],
            "core.async": ["clojure.core.async"],
            "spec": ["clojure.spec"],
            "test": ["clojure.test"]
        }
        
        detected_frameworks = []
        
        if isinstance(stack_trace, list):
            stack_text = ' '.join(str(frame) for frame in stack_trace)
        else:
            stack_text = str(stack_trace)
        
        for framework, patterns in framework_patterns.items():
            for pattern in patterns:
                if pattern in stack_text:
                    detected_frameworks.append(framework)
                    break
        
        if detected_frameworks:
            return {"frameworks": detected_frameworks}
        
        return None
    
    def _detect_version_context(self, error_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect Clojure version specific context."""
        # This would need to be implemented based on actual version detection
        # For now, return None
        return None
    
    def _extract_namespace_context(self, error_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract namespace information from error data."""
        namespace_info = {}
        
        # Check if namespace is in additional_data
        additional_data = error_data.get("additional_data", {})
        if "namespace" in additional_data:
            namespace_info["current_namespace"] = additional_data["namespace"]
        
        # Extract namespace from stack trace
        stack_trace = error_data.get("stack_trace", [])
        if isinstance(stack_trace, list) and stack_trace:
            for frame in stack_trace[:3]:  # Check first 3 frames
                if isinstance(frame, dict) and "namespace" in frame:
                    namespace_info["error_namespace"] = frame["namespace"]
                    break
        
        return namespace_info if namespace_info else None
    
    def _enhance_fix_suggestions(self, base_suggestions: List[str], error_data: Dict[str, Any], framework_context: Optional[Dict[str, Any]]) -> List[str]:
        """Enhance fix suggestions based on context."""
        enhanced = base_suggestions.copy()
        
        # Add framework-specific suggestions
        if framework_context:
            frameworks = framework_context.get("frameworks", [])
            
            if "ring" in frameworks:
                enhanced.append("Check Ring middleware configuration and handler return values")
            
            if "core.async" in frameworks:
                enhanced.append("Verify channel operations and ensure proper blocking/non-blocking usage")
            
            if "spec" in frameworks:
                enhanced.append("Validate data against spec definitions and check instrumented functions")
        
        return enhanced
    
    def _create_generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a generic analysis when no specific rule matches."""
        error_type = error_data.get("error_type", "")
        
        # Determine category based on error type
        category = "unknown"
        if "Exception" in error_type:
            if "NullPointer" in error_type:
                category = "runtime"
            elif "IllegalArgument" in error_type:
                category = "runtime"
            elif "ClassCast" in error_type:
                category = "runtime"
            elif "Compiler" in error_type:
                category = "syntax"
            else:
                category = "core"
        
        return {
            "error_data": error_data,
            "rule_id": "generic_clojure_analysis",
            "error_type": error_type,
            "category": category,
            "description": f"Generic analysis for {error_type}",
            "root_cause": "Unable to determine specific root cause",
            "fix_suggestions": [
                "Check the stack trace for the immediate cause",
                "Verify function arguments and data types",
                "Ensure all required namespaces are loaded",
                "Check for nil values and proper error handling"
            ],
            "confidence": 0.3,
            "severity": "medium",
            "tags": ["generic", "clojure"],
            "frameworks": [],
            "clojure_versions": []
        }


class ClojurePatchGenerator:
    """
    Generates code patches for Clojure errors based on analysis results.
    
    This class takes error analysis results and generates appropriate fixes
    using templates and code generation strategies.
    """
    
    def __init__(self):
        """Initialize the Clojure patch generator."""
        self.templates_dir = Path(__file__).parent.parent / "patch_generation" / "templates" / "clojure"
        
    def generate_patch(self, analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the analyzed error.
        
        Args:
            analysis_result: Result from ClojureExceptionHandler analysis
            
        Returns:
            Patch information or None if no patch can be generated
        """
        rule_id = analysis_result.get("rule_id", "")
        category = analysis_result.get("category", "")
        error_data = analysis_result.get("error_data", {})
        
        # Try to find a specific template for this rule
        template_path = self._find_template(rule_id, category)
        
        if template_path and template_path.exists():
            return self._generate_from_template(template_path, analysis_result)
        
        # Fall back to generic patch generation
        return self._generate_generic_patch(analysis_result)
    
    def _find_template(self, rule_id: str, category: str) -> Optional[Path]:
        """Find appropriate template file for the error."""
        # Try rule-specific template first
        rule_template = self.templates_dir / f"{rule_id}.clj.template"
        if rule_template.exists():
            return rule_template
        
        # Try category-specific template
        category_template = self.templates_dir / f"{category}_error.clj.template"
        if category_template.exists():
            return category_template
        
        return None
    
    def _generate_from_template(self, template_path: Path, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate patch from template file."""
        try:
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            # Extract context from analysis result
            context = self._extract_template_context(analysis_result)
            
            # Apply template substitutions
            patch_content = self._apply_template_substitutions(template_content, context)
            
            return {
                "patch_type": "template",
                "template_path": str(template_path),
                "patch_content": patch_content,
                "context": context,
                "confidence": analysis_result.get("confidence", 0.5),
                "description": f"Generated from template {template_path.name}"
            }
        
        except Exception as e:
            logger.error(f"Error generating patch from template {template_path}: {e}")
            return None
    
    def _generate_generic_patch(self, analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a generic patch when no template is available."""
        category = analysis_result.get("category", "")
        error_type = analysis_result.get("error_type", "")
        
        # Generate basic patches for common patterns
        if "NullPointerException" in error_type:
            return {
                "patch_type": "generic",
                "patch_content": ";; Add nil check\n(when (some? value)\n  ;; your code here\n  )",
                "description": "Add nil check to prevent NullPointerException",
                "confidence": 0.4
            }
        
        elif "ArityException" in error_type:
            return {
                "patch_type": "generic", 
                "patch_content": ";; Check function arity\n;; Ensure the function is called with the correct number of arguments",
                "description": "Fix function arity mismatch",
                "confidence": 0.4
            }
        
        elif "ClassCastException" in error_type:
            return {
                "patch_type": "generic",
                "patch_content": ";; Add type check\n(when (instance? ExpectedType value)\n  ;; your code here\n  )",
                "description": "Add type checking to prevent ClassCastException", 
                "confidence": 0.4
            }
        
        return None
    
    def _extract_template_context(self, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """Extract context variables for template substitution."""
        error_data = analysis_result.get("error_data", {})
        
        context = {
            "error_type": analysis_result.get("error_type", ""),
            "message": error_data.get("message", ""),
            "namespace": "",
            "function": "",
            "variable": "",
            "value": ""
        }
        
        # Extract namespace context
        namespace_context = analysis_result.get("namespace_context", {})
        if namespace_context:
            context["namespace"] = namespace_context.get("current_namespace", "")
        
        # Extract additional context from error data
        additional_data = error_data.get("additional_data", {})
        if "var" in additional_data:
            context["variable"] = additional_data["var"]
        
        return context
    
    def _apply_template_substitutions(self, template_content: str, context: Dict[str, str]) -> str:
        """Apply variable substitutions to template content."""
        result = template_content
        
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, str(value))
        
        return result


class ClojureLanguagePlugin(LanguagePlugin):
    """
    Main Clojure language plugin for Homeostasis.
    
    This plugin provides comprehensive Clojure error analysis and patch generation
    capabilities, integrating with the Homeostasis framework.
    """
    
    def __init__(self):
        """Initialize the Clojure language plugin."""
        super().__init__()
        self.exception_handler = ClojureExceptionHandler()
        self.patch_generator = ClojurePatchGenerator()
        self.adapter = ClojureErrorAdapter()
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "clojure"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the language."""
        return "Clojure"
    
    def get_language_version(self) -> str:
        """Get the version of the language supported by this plugin."""
        return "1.8+"
    
    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return [
            "ring",
            "compojure",
            "luminus", 
            "pedestal",
            "core.async",
            "spec",
            "datomic"
        ]
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize error data to the standard Homeostasis format."""
        return self.adapter.to_standard_format(error_data)
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard format error data back to the language-specific format."""
        return self.adapter.from_standard_format(standard_error)
    
    def get_language_info(self) -> Dict[str, Any]:
        """
        Get information about this language plugin.
        
        Returns:
            Dictionary containing plugin metadata
        """
        return {
            "language": "clojure",
            "version": "1.0.0",
            "description": "Clojure language support for Homeostasis",
            "supported_versions": ["1.8+", "1.9+", "1.10+", "1.11+"],
            "frameworks": [
                "ring",
                "compojure",
                "luminus", 
                "pedestal",
                "core.async",
                "spec",
                "datomic"
            ],
            "capabilities": [
                "error_analysis",
                "patch_generation",
                "stack_trace_parsing",
                "framework_detection",
                "jvm_interop_support"
            ]
        }
    
    def can_handle_error(self, error_data: Dict[str, Any]) -> bool:
        """
        Determine if this plugin can handle the given error.
        
        Args:
            error_data: Error data to check
            
        Returns:
            True if this plugin can handle the error
        """
        # Check language field
        if error_data.get("language") == "clojure":
            return True
        
        # Check for Clojure-specific patterns
        error_type = error_data.get("error_type", "")
        message = error_data.get("message", "")
        stack_trace = str(error_data.get("stack_trace", ""))
        
        clojure_indicators = [
            "clojure.lang.",
            ".clj:",
            "CompilerException",
            "ArityException",
            "$fn__",
            "clojure.core",
            "REPL:"
        ]
        
        text_to_check = f"{error_type} {message} {stack_trace}"
        
        return any(indicator in text_to_check for indicator in clojure_indicators)
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Clojure error and provide detailed information.
        
        Args:
            error_data: Error data in standard format
            
        Returns:
            Analysis result with recommendations
        """
        try:
            # Ensure error data is in standard format
            if not error_data.get("language"):
                error_data = self.adapter.to_standard_format(error_data)
            
            # Perform analysis
            analysis_result = self.exception_handler.analyze_exception(error_data)
            
            # Add plugin metadata
            analysis_result["plugin"] = "clojure"
            analysis_result["plugin_version"] = "1.0.0"
            analysis_result["analysis_timestamp"] = self._get_timestamp()
            
            return analysis_result
        
        except Exception as e:
            logger.error(f"Error analyzing Clojure exception: {e}")
            return {
                "error": f"Analysis failed: {e}",
                "plugin": "clojure",
                "success": False
            }
    
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a fix for an error based on the analysis.
        
        Args:
            analysis: Error analysis
            context: Additional context for fix generation
            
        Returns:
            Generated fix data
        """
        try:
            # Combine analysis with context for patch generation
            analysis_result = {**analysis}
            if context:
                analysis_result["context"] = context
                
            patch_result = self.patch_generator.generate_patch(analysis_result)
            
            if patch_result:
                patch_result["plugin"] = "clojure"
                patch_result["generation_timestamp"] = self._get_timestamp()
                return patch_result
            
            # Return empty dict if no patch generated (as per abstract method)
            return {}
        
        except Exception as e:
            logger.error(f"Error generating Clojure fix: {e}")
            return {}
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


# Register the plugin
register_plugin(ClojureLanguagePlugin())