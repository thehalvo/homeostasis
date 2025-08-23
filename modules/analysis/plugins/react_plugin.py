"""
React Framework Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in React applications.
It provides comprehensive error handling for React components, hooks, lifecycle issues,
state management (including Redux and Context), JSX problems, and performance issues.
"""
import logging
import re
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from ..language_plugin_system import LanguagePlugin, register_plugin
from ..language_adapters import JavaScriptErrorAdapter

logger = logging.getLogger(__name__)


class ReactExceptionHandler:
    """
    Handles React-specific exceptions with comprehensive error detection and classification.
    
    This class provides logic for categorizing React component errors, hooks violations,
    lifecycle issues, state management problems, and JSX-related errors.
    """
    
    def __init__(self):
        """Initialize the React exception handler."""
        self.rule_categories = {
            "hooks": "React Hooks related errors",
            "lifecycle": "Component lifecycle errors",
            "state": "State management errors",
            "redux": "Redux state management errors",
            "context": "React Context errors",
            "jsx": "JSX syntax and usage errors",
            "props": "Props validation and usage errors",
            "rendering": "Rendering and performance errors",
            "events": "Event handling errors",
            "components": "Component definition and export errors"
        }
        
        # Load rules from different categories
        self.rules = self._load_rules()
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
    
    def _load_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load React error rules from rule files."""
        rules = {}
        rules_dir = Path(__file__).parent.parent / "rules" / "react"
        
        try:
            # Load common React rules
            common_rules_path = rules_dir / "react_common_errors.json"
            if common_rules_path.exists():
                with open(common_rules_path, 'r') as f:
                    common_data = json.load(f)
                    rules["common"] = common_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['common'])} common React rules")
            
            # Load React hooks rules
            hooks_rules_path = rules_dir / "react_hooks_errors.json"
            if hooks_rules_path.exists():
                with open(hooks_rules_path, 'r') as f:
                    hooks_data = json.load(f)
                    rules["hooks"] = hooks_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['hooks'])} React hooks rules")
            
            # Load state management rules
            state_rules_path = rules_dir / "react_state_management_errors.json"
            if state_rules_path.exists():
                with open(state_rules_path, 'r') as f:
                    state_data = json.load(f)
                    rules["state"] = state_data.get("rules", [])
                    logger.info(f"Loaded {len(rules['state'])} React state management rules")
                    
        except Exception as e:
            logger.error(f"Error loading React rules: {e}")
            rules = {"common": [], "hooks": [], "state": []}
        
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
                    logger.warning(f"Invalid regex pattern in React rule {rule.get('id', 'unknown')}: {e}")
    
    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a React exception and determine its type and potential fixes.
        
        Args:
            error_data: React error data in standard format
            
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
        
        # Find matching rules
        matches = self._find_matching_rules(full_error_text, error_data)
        
        if matches:
            # Use the best match (highest confidence)
            best_match = max(matches, key=lambda x: x.get("confidence_score", 0))
            return {
                "category": best_match.get("category", "react"),
                "subcategory": best_match.get("subcategory", "unknown"),
                "confidence": best_match.get("confidence", "medium"),
                "suggested_fix": best_match.get("suggestion", ""),
                "root_cause": best_match.get("root_cause", ""),
                "severity": best_match.get("severity", "medium"),
                "rule_id": best_match.get("id", ""),
                "tags": best_match.get("tags", []),
                "fix_commands": best_match.get("fix_commands", []),
                "all_matches": matches
            }
        
        # If no rules matched, provide generic analysis
        return self._generic_analysis(error_data)
    
    def _find_matching_rules(self, error_text: str, error_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find all rules that match the given error."""
        matches = []
        
        for category, patterns in self.compiled_patterns.items():
            for compiled_pattern, rule in patterns:
                match = compiled_pattern.search(error_text)
                if match:
                    # Calculate confidence score based on match quality
                    confidence_score = self._calculate_confidence(match, rule, error_data)
                    
                    match_info = rule.copy()
                    match_info["confidence_score"] = confidence_score
                    match_info["match_groups"] = match.groups() if match.groups() else []
                    matches.append(match_info)
        
        return matches
    
    def _calculate_confidence(self, match: re.Match, rule: Dict[str, Any], 
                             error_data: Dict[str, Any]) -> float:
        """Calculate confidence score for a rule match."""
        base_confidence = 0.5
        
        # Boost confidence for React-specific patterns
        message = error_data.get("message", "").lower()
        if "react" in message or "hook" in message or "jsx" in message:
            base_confidence += 0.3
        
        # Boost confidence based on rule reliability
        reliability = rule.get("reliability", "medium")
        reliability_boost = {"high": 0.2, "medium": 0.1, "low": 0.0}
        base_confidence += reliability_boost.get(reliability, 0.0)
        
        # Boost confidence for rules with specific tags that match context
        rule_tags = set(rule.get("tags", []))
        context_tags = set()
        
        # Infer context from error data
        if "react" in error_data.get("framework", "").lower():
            context_tags.add("react")
        if "redux" in message:
            context_tags.add("redux")
        if "hook" in message:
            context_tags.add("hooks")
        if "jsx" in message or "tsx" in str(error_data.get("stack_trace", "")):
            context_tags.add("jsx")
        
        if context_tags & rule_tags:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _generic_analysis(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide generic analysis for unmatched errors."""
        error_type = error_data.get("error_type", "Error")
        message = error_data.get("message", "").lower()
        
        # Basic categorization based on error patterns
        if "hook" in message:
            category = "hooks"
            suggestion = "Check React Hooks usage - ensure hooks are called at top level and follow Rules of Hooks"
        elif "jsx" in message or "element" in message:
            category = "jsx"
            suggestion = "Check JSX syntax and element usage"
        elif "prop" in message:
            category = "props"
            suggestion = "Check prop types and prop usage"
        elif "state" in message:
            category = "state"
            suggestion = "Check state management and updates"
        elif "render" in message:
            category = "rendering"
            suggestion = "Check component rendering logic"
        else:
            category = "unknown"
            suggestion = "Review React component implementation"
        
        return {
            "category": "react",
            "subcategory": category,
            "confidence": "low",
            "suggested_fix": suggestion,
            "root_cause": f"react_{category}_error",
            "severity": "medium",
            "rule_id": "react_generic_handler",
            "tags": ["react", "generic", category]
        }
    
    def analyze_hooks_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze React Hooks specific errors.
        
        Args:
            error_data: Error data with hooks-related issues
            
        Returns:
            Analysis results with hooks-specific fixes
        """
        message = error_data.get("message", "")
        
        # Common hooks error patterns
        hooks_patterns = {
            "invalid hook call": {
                "cause": "react_invalid_hook_call",
                "fix": "Move hook calls to the top level of React function components",
                "severity": "error"
            },
            "hooks can only be called inside": {
                "cause": "react_hooks_outside_component",
                "fix": "Call hooks only inside React function components or custom hooks",
                "severity": "error"
            },
            "called conditionally": {
                "cause": "react_conditional_hook_call",
                "fix": "Remove conditional logic around hook calls - hooks must be called in the same order every time",
                "severity": "error"
            },
            "missing dependency": {
                "cause": "react_missing_dependency",
                "fix": "Add missing dependencies to useEffect, useCallback, or useMemo dependency arrays",
                "severity": "warning"
            },
            "exhaustive-deps": {
                "cause": "react_exhaustive_deps",
                "fix": "Include all variables used inside hooks in their dependency arrays",
                "severity": "warning"
            }
        }
        
        for pattern, info in hooks_patterns.items():
            if pattern in message.lower():
                return {
                    "category": "react",
                    "subcategory": "hooks",
                    "confidence": "high",
                    "suggested_fix": info["fix"],
                    "root_cause": info["cause"],
                    "severity": info["severity"],
                    "tags": ["react", "hooks", "rules-of-hooks"]
                }
        
        # Generic hooks error
        return {
            "category": "react",
            "subcategory": "hooks",
            "confidence": "medium",
            "suggested_fix": "Check React Hooks usage and Rules of Hooks",
            "root_cause": "react_hooks_error",
            "severity": "warning",
            "tags": ["react", "hooks"]
        }
    
    def analyze_state_management_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze React state management errors (including Redux and Context).
        
        Args:
            error_data: Error data with state management issues
            
        Returns:
            Analysis results with state management fixes
        """
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        # Redux specific errors
        if "redux" in message or "store" in message or "dispatch" in message:
            return self._analyze_redux_error(message, error_data)
        
        # Context specific errors
        if "context" in message or "provider" in message:
            return self._analyze_context_error(message, error_data)
        
        # General state errors
        if "state" in message:
            return {
                "category": "react",
                "subcategory": "state",
                "confidence": "medium",
                "suggested_fix": "Check state management - avoid direct mutations and use proper setState patterns",
                "root_cause": "react_state_management_error",
                "severity": "warning",
                "tags": ["react", "state"]
            }
        
        return {
            "category": "react",
            "subcategory": "state_management",
            "confidence": "low",
            "suggested_fix": "Review state management implementation",
            "root_cause": "react_state_unknown_error",
            "severity": "medium",
            "tags": ["react", "state"]
        }
    
    def _analyze_redux_error(self, message: str, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Redux-specific errors."""
        if "store" in message and ("undefined" in message or "not found" in message):
            return {
                "category": "react",
                "subcategory": "redux",
                "confidence": "high",
                "suggested_fix": "Wrap your app with Redux Provider and ensure store is properly configured",
                "root_cause": "redux_store_not_connected",
                "severity": "error",
                "tags": ["react", "redux", "store", "provider"]
            }
        
        if "dispatch" in message:
            return {
                "category": "react",
                "subcategory": "redux",
                "confidence": "high",
                "suggested_fix": "Use useDispatch hook to get dispatch function",
                "root_cause": "redux_dispatch_error",
                "severity": "error",
                "tags": ["react", "redux", "dispatch"]
            }
        
        if "non-serializable" in message or "mutation" in message:
            return {
                "category": "react",
                "subcategory": "redux",
                "confidence": "high",
                "suggested_fix": "Avoid mutating state directly - use immutable updates in reducers",
                "root_cause": "redux_state_mutation",
                "severity": "error",
                "tags": ["react", "redux", "immutability"]
            }
        
        return {
            "category": "react",
            "subcategory": "redux",
            "confidence": "medium",
            "suggested_fix": "Check Redux configuration and usage",
            "root_cause": "redux_general_error",
            "severity": "medium",
            "tags": ["react", "redux"]
        }
    
    def _analyze_context_error(self, message: str, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze React Context-specific errors."""
        if "provider" in message and ("outside" in message or "must be used within" in message):
            return {
                "category": "react",
                "subcategory": "context",
                "confidence": "high",
                "suggested_fix": "Wrap component with the appropriate Context Provider",
                "root_cause": "context_provider_missing",
                "severity": "error",
                "tags": ["react", "context", "provider"]
            }
        
        if "provider" in message and "value" in message:
            return {
                "category": "react",
                "subcategory": "context",
                "confidence": "medium",
                "suggested_fix": "Provide a value prop to Context Provider",
                "root_cause": "context_provider_no_value",
                "severity": "warning",
                "tags": ["react", "context", "provider", "value"]
            }
        
        return {
            "category": "react",
            "subcategory": "context",
            "confidence": "medium",
            "suggested_fix": "Check React Context usage",
            "root_cause": "context_general_error",
            "severity": "medium",
            "tags": ["react", "context"]
        }


class ReactPatchGenerator:
    """
    Generates patches for React errors based on analysis results.
    
    This class creates code fixes for common React errors using templates
    and heuristics specific to React patterns and best practices.
    """
    
    def __init__(self):
        """Initialize the React patch generator."""
        self.template_dir = Path(__file__).parent.parent / "patch_generation" / "templates"
        self.react_template_dir = self.template_dir / "react"
        
        # Ensure template directory exists
        self.react_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load patch templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load React patch templates."""
        templates = {}
        
        if not self.react_template_dir.exists():
            logger.warning(f"React templates directory not found: {self.react_template_dir}")
            return templates
        
        for template_file in self.react_template_dir.glob("*.jsx.template"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem.replace('.jsx', '')
                    templates[template_name] = f.read()
                    logger.debug(f"Loaded React template: {template_name}")
            except Exception as e:
                logger.error(f"Error loading React template {template_file}: {e}")
        
        return templates
    
    def generate_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                      source_code: str) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for the React error.
        
        Args:
            error_data: The React error data
            analysis: Analysis results from ReactExceptionHandler
            source_code: The source code where the error occurred
            
        Returns:
            Patch information or None if no patch can be generated
        """
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to patch strategies
        patch_strategies = {
            "react_invalid_hook_call": self._fix_invalid_hook_call,
            "react_missing_dependency": self._fix_missing_dependency,
            "react_missing_key_prop": self._fix_missing_key_prop,
            "react_state_update_unmounted": self._fix_state_update_unmounted,
            "react_jsx_scope_error": self._fix_jsx_scope_error,
            "react_conditional_hook_call": self._fix_conditional_hook_call,
            "redux_store_not_connected": self._fix_redux_store_connection,
            "context_provider_missing": self._fix_context_provider_missing
        }
        
        strategy = patch_strategies.get(root_cause)
        if strategy:
            try:
                return strategy(error_data, analysis, source_code)
            except Exception as e:
                logger.error(f"Error generating React patch for {root_cause}: {e}")
        
        # Try to use templates if no specific strategy matches
        return self._template_based_patch(error_data, analysis, source_code)
    
    def _fix_invalid_hook_call(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                              source_code: str) -> Optional[Dict[str, Any]]:
        """Fix invalid hook call errors."""
        return {
            "type": "suggestion",
            "description": "Move hook calls to the top level of React function components. Hooks cannot be called inside loops, conditions, or nested functions.",
            "fix_commands": [
                "Move useState, useEffect, and other hooks to the top of the function component",
                "Ensure hooks are called before any early returns",
                "Use conditional logic inside hooks, not around them"
            ]
        }
    
    def _fix_missing_dependency(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                               source_code: str) -> Optional[Dict[str, Any]]:
        """Fix missing dependency errors in useEffect, useCallback, useMemo."""
        message = error_data.get("message", "")
        
        # Extract hook name and missing dependency
        hook_match = re.search(r"React Hook (\w+) has a missing dependency.*'([^']+)'", message)
        if hook_match:
            hook_name = hook_match.group(1)
            missing_dep = hook_match.group(2)
            
            return {
                "type": "suggestion",
                "description": f"Add '{missing_dep}' to the {hook_name} dependency array",
                "fix_code": f"Add '{missing_dep}' to the dependency array: [{missing_dep}]",
                "template": "hook_dependency_fix"
            }
        
        return {
            "type": "suggestion",
            "description": "Add missing dependencies to hook dependency arrays",
            "template": "hook_dependency_fix"
        }
    
    def _fix_missing_key_prop(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                             source_code: str) -> Optional[Dict[str, Any]]:
        """Fix missing key prop in list rendering."""
        return {
            "type": "suggestion",
            "description": "Add unique key prop to each element in the list",
            "fix_commands": [
                "Add key={item.id} or similar unique identifier",
                "Use array index as key only if list items never reorder",
                "Ensure keys are stable, predictable, and unique"
            ],
            "template": "key_prop_fix"
        }
    
    def _fix_state_update_unmounted(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                   source_code: str) -> Optional[Dict[str, Any]]:
        """Fix state updates on unmounted components."""
        return {
            "type": "suggestion",
            "description": "Add cleanup to prevent state updates after component unmounts",
            "fix_commands": [
                "Return cleanup function from useEffect",
                "Use AbortController for fetch requests",
                "Check component mount status before setState"
            ],
            "template": "cleanup_effect"
        }
    
    def _fix_jsx_scope_error(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Fix JSX scope errors."""
        return {
            "type": "line_addition",
            "description": "Add React import for JSX usage",
            "line_to_add": "import React from 'react';",
            "position": "top",
            "alternative": "Configure JSX transform in build configuration"
        }
    
    def _fix_conditional_hook_call(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                  source_code: str) -> Optional[Dict[str, Any]]:
        """Fix conditional hook calls."""
        return {
            "type": "suggestion",
            "description": "Move hook calls outside of conditional statements",
            "fix_commands": [
                "Move all hook calls to the top level of the component",
                "Use conditional logic inside hooks, not around them",
                "Place hook calls before any early returns"
            ],
            "template": "conditional_hook_fix"
        }
    
    def _fix_redux_store_connection(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                   source_code: str) -> Optional[Dict[str, Any]]:
        """Fix Redux store connection issues."""
        return {
            "type": "suggestion",
            "description": "Wrap your app with Redux Provider",
            "fix_commands": [
                "Import { Provider } from 'react-redux'",
                "Wrap App component with <Provider store={store}>",
                "Ensure Redux store is properly configured"
            ],
            "fix_code": """
import { Provider } from 'react-redux';
import store from './store';

<Provider store={store}>
  <App />
</Provider>
"""
        }
    
    def _fix_context_provider_missing(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                                     source_code: str) -> Optional[Dict[str, Any]]:
        """Fix missing Context Provider."""
        message = error_data.get("message", "")
        
        # Extract context name if possible
        context_match = re.search(r"useContext.*must be used within.*(\w+)Provider", message)
        context_name = context_match.group(1) if context_match else "YourContext"
        
        return {
            "type": "suggestion",
            "description": f"Wrap component with {context_name}Provider",
            "fix_code": f"""
<{context_name}.Provider value={{contextValue}}>
  <YourComponent />
</{context_name}.Provider>
""",
            "fix_commands": [
                f"Import {context_name} from the appropriate module",
                f"Wrap the component tree with <{context_name}.Provider>",
                "Provide a value prop to the Provider"
            ]
        }
    
    def _template_based_patch(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                            source_code: str) -> Optional[Dict[str, Any]]:
        """Generate patch using templates."""
        root_cause = analysis.get("root_cause", "")
        
        # Map root causes to template names
        template_map = {
            "react_missing_dependency": "hook_dependency_fix",
            "react_missing_key_prop": "key_prop_fix",
            "react_state_update_unmounted": "cleanup_effect",
            "react_conditional_hook_call": "conditional_hook_fix"
        }
        
        template_name = template_map.get(root_cause)
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            
            return {
                "type": "template",
                "template": template,
                "description": f"Applied React template fix for {root_cause}"
            }
        
        return None


class ReactLanguagePlugin(LanguagePlugin):
    """
    Main React framework plugin for Homeostasis.
    
    This plugin orchestrates React error analysis and patch generation,
    supporting React components, hooks, state management, and JSX.
    """
    
    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Team"
    
    def __init__(self):
        """Initialize the React language plugin."""
        self.language = "react"
        self.supported_extensions = {".jsx", ".tsx", ".js", ".ts"}
        self.supported_frameworks = [
            "react", "create-react-app", "next", "gatsby", "remix",
            "react-native", "expo", "vite-react", "parcel-react"
        ]
        
        # Initialize components
        self.adapter = JavaScriptErrorAdapter()  # Reuse JavaScript adapter
        self.exception_handler = ReactExceptionHandler()
        self.patch_generator = ReactPatchGenerator()
        
        logger.info("React framework plugin initialized")
    
    def get_language_id(self) -> str:
        """Get the unique identifier for this language."""
        return "react"
    
    def get_language_name(self) -> str:
        """Get the human-readable name of the framework."""
        return "React"
    
    def get_language_version(self) -> str:
        """Get the version of the framework supported by this plugin."""
        return "16.8+"
    
    def get_supported_frameworks(self) -> List[str]:
        """Get the list of frameworks supported by this language plugin."""
        return self.supported_frameworks
    
    def can_handle(self, error_data: Dict[str, Any]) -> bool:
        """
        Check if this plugin can handle the given error.
        
        Args:
            error_data: Error data to check
            
        Returns:
            True if this plugin can handle the error, False otherwise
        """
        # Check if framework is explicitly set
        framework = error_data.get("framework", "").lower()
        if "react" in framework:
            return True
        
        # Check error message for React-specific patterns
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        react_patterns = [
            r"react",
            r"jsx",
            r"hook",
            r"usestate",
            r"useeffect",
            r"usecallback",
            r"usememo",
            r"usecontext",
            r"useref",
            r"usereducer",
            r"react.*component",
            r"invalid hook call",
            r"hooks can only be called",
            r"exhaustive-deps",
            r"missing dependency",
            r"jsx element",
            r"react-dom",
            r"createelement"
        ]
        
        for pattern in react_patterns:
            if re.search(pattern, message + stack_trace):
                return True
        
        # Check file extensions for React files
        if re.search(r'\.(jsx|tsx):', stack_trace):
            return True
        
        # Check for React in package dependencies (if available)
        context = error_data.get("context", {})
        dependencies = context.get("dependencies", [])
        if any("react" in dep.lower() for dep in dependencies):
            return True
        
        return False
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a React error.
        
        Args:
            error_data: React error data
            
        Returns:
            Analysis results
        """
        try:
            # Ensure error data is in standard format
            if not error_data.get("language"):
                standard_error = self.adapter.to_standard_format(error_data)
            else:
                standard_error = error_data
            
            message = standard_error.get("message", "").lower()
            
            # Check if it's a hooks-related error
            if self._is_hooks_error(standard_error):
                analysis = self.exception_handler.analyze_hooks_error(standard_error)
            
            # Check if it's a state management error
            elif self._is_state_management_error(standard_error):
                analysis = self.exception_handler.analyze_state_management_error(standard_error)
            
            # Default React error analysis
            else:
                analysis = self.exception_handler.analyze_exception(standard_error)
            
            # Add plugin metadata
            analysis["plugin"] = "react"
            analysis["language"] = "react"
            analysis["plugin_version"] = self.VERSION
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing React error: {e}")
            return {
                "category": "react",
                "subcategory": "unknown",
                "confidence": "low",
                "suggested_fix": "Unable to analyze React error",
                "error": str(e),
                "plugin": "react"
            }
    
    def _is_hooks_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a React Hooks related error."""
        message = error_data.get("message", "").lower()
        
        hooks_patterns = [
            "hook",
            "usestate",
            "useeffect",
            "usecallback",
            "usememo",
            "usecontext",
            "useref",
            "usereducer",
            "invalid hook call",
            "hooks can only be called",
            "exhaustive-deps",
            "missing dependency"
        ]
        
        return any(pattern in message for pattern in hooks_patterns)
    
    def _is_state_management_error(self, error_data: Dict[str, Any]) -> bool:
        """Check if this is a state management related error."""
        message = error_data.get("message", "").lower()
        stack_trace = str(error_data.get("stack_trace", "")).lower()
        
        state_patterns = [
            "redux",
            "store",
            "dispatch",
            "context",
            "provider",
            "state",
            "setstate"
        ]
        
        return any(pattern in message or pattern in stack_trace for pattern in state_patterns)
    
    def generate_fix(self, error_data: Dict[str, Any], analysis: Dict[str, Any], 
                    source_code: str) -> Optional[Dict[str, Any]]:
        """
        Generate a fix for the React error.
        
        Args:
            error_data: The React error data
            analysis: Analysis results
            source_code: Source code where the error occurred
            
        Returns:
            Fix information or None if no fix can be generated
        """
        try:
            return self.patch_generator.generate_patch(error_data, analysis, source_code)
        except Exception as e:
            logger.error(f"Error generating React fix: {e}")
            return None
    
    def get_language_info(self) -> Dict[str, Any]:
        """
        Get information about this language plugin.
        
        Returns:
            Language plugin information
        """
        return {
            "language": self.language,
            "version": self.VERSION,
            "supported_extensions": list(self.supported_extensions),
            "supported_frameworks": list(self.supported_frameworks),
            "features": [
                "React component lifecycle error handling",
                "React Hooks rule validation and fixes",
                "JSX syntax error detection",
                "State management error handling (React state, Redux, Context)",
                "Performance optimization suggestions",
                "Props validation error fixes",
                "Event handling error detection",
                "Server components support",
                "Memory leak prevention"
            ],
            "environments": ["browser", "node", "react-native", "electron"]
        }

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize error data to the standard Homeostasis format.
        
        Args:
            error_data: Language-specific error data
            
        Returns:
            Standardized error format
        """
        return {
            "language": self.get_language_id(),
            "type": error_data.get("type", "unknown"),
            "message": error_data.get("message", ""),
            "file": error_data.get("file", ""),
            "line": error_data.get("line", 0),
            "column": error_data.get("column", 0),
            "severity": error_data.get("severity", "error"),
            "context": error_data.get("context", {}),
            "raw_data": error_data
        }

    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data back to the language-specific format.
        
        Args:
            standard_error: Standardized error data
            
        Returns:
            Language-specific error format
        """
        return {
            "type": standard_error.get("type", "unknown"),
            "message": standard_error.get("message", ""),
            "file": standard_error.get("file", ""),
            "line": standard_error.get("line", 0),
            "column": standard_error.get("column", 0),
            "severity": standard_error.get("severity", "error"),
            "context": standard_error.get("context", {}),
            "language_specific": standard_error.get("raw_data", {})
        }


# Register the plugin
register_plugin(ReactLanguagePlugin())