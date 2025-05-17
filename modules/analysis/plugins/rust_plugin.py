"""
Rust Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Rust applications.
It provides error handling for Rust's common error patterns and supports Rust's
memory safety features, async/await, and popular Rust frameworks like Actix and Rocket.
"""
import logging
import re
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from ..language_plugin_system import LanguagePlugin, register_plugin
from ..language_adapters import RustErrorAdapter

logger = logging.getLogger(__name__)


class RustErrorHandler:
    """
    Handles Rust errors with pattern-based error detection and classification.
    
    This class provides logic for categorizing Rust errors based on their type,
    message, and stack trace patterns. It supports both common Rust errors and
    framework-specific errors.
    """
    
    def __init__(self):
        """Initialize the Rust error handler."""
        self.rule_categories = {
            "runtime": "Rust runtime errors",
            "compile_time": "Rust compile-time errors",
            "concurrency": "Thread and concurrency errors",
            "io": "I/O and file errors",
            "network": "Network-related errors",
            "serialization": "Serialization/deserialization errors",
            "framework": "Framework-specific errors (Actix, Rocket, etc.)",
            "memory": "Memory safety errors",
            "tokio": "Tokio async runtime errors",
            "cargo": "Cargo and dependency errors"
        }
        
        # Load rules from different categories
        self.rules = self._load_all_rules()
        
        # Initialize caches for performance
        self.pattern_cache = {}  # Compiled regex patterns
        self.rule_match_cache = {}  # Previous rule matches
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Rust error to determine its root cause and suggest potential fixes.
        
        Args:
            error_data: Rust error data in standard format
            
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
            
            # Skip rules that don't apply to this category of error
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
                        "root_cause": rule.get("root_cause", "rust_unknown_error"),
                        "description": rule.get("description", "Unknown Rust error"),
                        "suggestion": rule.get("suggestion", "No suggestion available"),
                        "confidence": rule.get("confidence", "medium"),
                        "severity": rule.get("severity", "medium"),
                        "category": rule.get("category", "runtime"),
                        "match_groups": match.groups() if match.groups() else tuple(),
                        "framework": rule.get("framework", "")
                    }
                    
                    # Cache the result for this error signature
                    error_signature = f"{error_type}:{message[:100]}"
                    self.rule_match_cache[error_signature] = result
                    
                    return result
            except Exception as e:
                logger.warning(f"Error applying rule {rule.get('id', 'unknown')}: {e}")
        
        # If no rule matched, try the fallback handlers
        return self._handle_fallback(error_data)
    
    def _create_match_text(self, error_type: str, message: str, stack_trace: List) -> str:
        """
        Create a consolidated text for pattern matching from error components.
        
        Args:
            error_type: Error type
            message: Error message
            stack_trace: Stack trace frames
            
        Returns:
            Consolidated text for pattern matching
        """
        match_text = f"{error_type}: {message}"
        
        # Add stack trace information if available
        if stack_trace:
            if isinstance(stack_trace, list):
                if stack_trace and isinstance(stack_trace[0], str):
                    match_text += "\n" + "\n".join(stack_trace)
                else:
                    # Convert structured frames to text
                    trace_lines = []
                    for frame in stack_trace:
                        if isinstance(frame, dict):
                            module = frame.get("module", "")
                            func_name = frame.get("function", "")
                            file_path = frame.get("file", "")
                            line_num = frame.get("line", "")
                            
                            # Format the frame info
                            frame_text = f"{module}::{func_name}" if module else func_name
                            trace_lines.append(frame_text)
                            if file_path:
                                trace_lines.append(f"   at {file_path}:{line_num}")
                    
                    if trace_lines:
                        match_text += "\n" + "\n".join(trace_lines)
        
        return match_text
    
    def _handle_fallback(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors that didn't match any specific rule.
        
        Args:
            error_data: Rust error data in standard format
            
        Returns:
            Fallback analysis result
        """
        error_type = error_data.get("error_type", "")
        message = error_data.get("message", "")
        
        # Check for panic messages
        if "panicked at" in message:
            return {
                "error_data": error_data,
                "rule_id": "rust_panic",
                "error_type": error_type or "Panic",
                "root_cause": "rust_panic",
                "description": "Runtime panic in Rust program",
                "suggestion": "Add proper error handling with Result<T, E> instead of panicking. Check for None values with Option's methods like unwrap_or, unwrap_or_else, or match expressions.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime",
                "match_groups": tuple(),
                "framework": ""
            }
        
        # Check for common error patterns
        if "unwrap" in message and "None" in message:
            return {
                "error_data": error_data,
                "rule_id": "rust_unwrap_none",
                "error_type": error_type or "Panic",
                "root_cause": "rust_unwrap_none",
                "description": "Called unwrap() on a None value",
                "suggestion": "Instead of unwrap(), use unwrap_or(), unwrap_or_else(), or match expressions to handle the None case gracefully. Consider using the ? operator with Option<T>::ok_or() for early returns.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "index out of bounds" in message:
            return {
                "error_data": error_data,
                "rule_id": "rust_index_out_of_bounds",
                "error_type": error_type or "Panic",
                "root_cause": "rust_index_out_of_bounds",
                "description": "Attempted to access an index beyond the bounds of a collection",
                "suggestion": "Check that the index is within bounds before accessing it. Use methods like .get() that return an Option instead of direct indexing, or check indices against collection length.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "divide by zero" in message or "division by zero" in message:
            return {
                "error_data": error_data,
                "rule_id": "rust_division_by_zero",
                "error_type": error_type or "Panic",
                "root_cause": "rust_division_by_zero",
                "description": "Attempted to divide by zero",
                "suggestion": "Check divisors before performing division. Use if statements or match expressions to handle zero divisors as special cases.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "deadlock" in message:
            return {
                "error_data": error_data,
                "rule_id": "rust_deadlock",
                "error_type": error_type or "DeadlockError",
                "root_cause": "rust_deadlock",
                "description": "Deadlock detected in thread synchronization",
                "suggestion": "Ensure consistent lock ordering across threads, limit lock scope, use a timeout with try_lock methods, implement deadlock detection, or restructure your code to avoid multiple locks.",
                "confidence": "high",
                "severity": "critical",
                "category": "concurrency",
                "match_groups": tuple(),
                "framework": ""
            }
        
        # Generic fallback for unknown errors
        return {
            "error_data": error_data,
            "rule_id": "rust_generic_error",
            "error_type": error_type or "Error",
            "root_cause": "rust_unknown_error",
            "description": f"Unrecognized Rust error",
            "suggestion": "Review error message and stack trace for more details. Consider implementing proper error handling with Result<T, E>.",
            "confidence": "low",
            "severity": "medium",
            "category": "runtime",
            "match_groups": tuple(),
            "framework": ""
        }
    
    def _load_all_rules(self) -> List[Dict[str, Any]]:
        """
        Load Rust error rules from all categories.
        
        Returns:
            Combined list of rule definitions
        """
        all_rules = []
        
        # Core Rust errors (always included)
        all_rules.extend(self._load_core_rust_rules())
        
        # Load additional rules from files if available
        rules_dir = Path(__file__).parent.parent / "rules" / "rust"
        if rules_dir.exists():
            for rule_file in rules_dir.glob("*.json"):
                try:
                    with open(rule_file, 'r') as f:
                        data = json.load(f)
                        all_rules.extend(data.get("rules", []))
                except Exception as e:
                    logger.warning(f"Error loading rules from {rule_file}: {e}")
        
        return all_rules
    
    def _load_core_rust_rules(self) -> List[Dict[str, Any]]:
        """Load rules for core Rust errors."""
        return [
            {
                "id": "rust_panic",
                "pattern": "thread '.*' panicked at '(.*?)'",
                "type": "Panic",
                "description": "Runtime panic in Rust program",
                "root_cause": "rust_panic",
                "suggestion": "Add proper error handling with Result<T, E> instead of panicking. Check for None values with Option's methods like unwrap_or, unwrap_or_else, or match expressions.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime"
            },
            {
                "id": "rust_unwrap_on_none",
                "pattern": "panicked at '.*unwrap\\(\\).*: None'",
                "type": "Panic",
                "description": "Called unwrap() on a None value",
                "root_cause": "rust_unwrap_none",
                "suggestion": "Instead of unwrap(), use unwrap_or(), unwrap_or_else(), or match expressions to handle the None case gracefully. Consider using the ? operator with Option<T>::ok_or() for early returns.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime"
            },
            {
                "id": "rust_unwrap_on_err",
                "pattern": "panicked at '.*unwrap\\(\\).*: (.*?)'",
                "type": "Panic",
                "description": "Called unwrap() on an Err value",
                "root_cause": "rust_unwrap_err",
                "suggestion": "Instead of unwrap(), use unwrap_or(), unwrap_or_else(), or match expressions to handle the Err case gracefully. Consider using the ? operator for early returns or using proper error handling.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime"
            },
            {
                "id": "rust_index_out_of_bounds",
                "pattern": "panicked at '.*index out of bounds: the len is (\\d+) but the index is (\\d+)'",
                "type": "Panic",
                "description": "Attempted to access an index beyond the bounds of a collection",
                "root_cause": "rust_index_out_of_bounds",
                "suggestion": "Check that the index is within bounds before accessing it. Use methods like .get() that return an Option instead of direct indexing, or check indices against collection length.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime"
            },
            {
                "id": "rust_division_by_zero",
                "pattern": "panicked at 'attempt to divide by zero'",
                "type": "Panic",
                "description": "Attempted to divide by zero",
                "root_cause": "rust_division_by_zero",
                "suggestion": "Check divisors before performing division. Use if statements or match expressions to handle zero divisors as special cases.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime"
            }
        ]


class RustPatchGenerator:
    """
    Generates patch solutions for Rust errors.
    
    This class provides capabilities to generate code fixes for common Rust errors,
    using templates and contextual information about the error.
    """
    
    def __init__(self):
        """Initialize the Rust patch generator."""
        self.templates_dir = Path(__file__).parent.parent / "patch_generation" / "templates" / "rust"
        self.templates_dir.mkdir(exist_ok=True, parents=True)
        
        # Cache for loaded templates
        self.template_cache = {}
    
    def generate_patch(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a patch for a Rust error based on analysis.
        
        Args:
            analysis: Error analysis containing root cause and other details
            context: Additional context about the error, including code snippets
            
        Returns:
            Patch data including patch type, code, and application instructions
        """
        root_cause = analysis.get("root_cause", "unknown")
        rule_id = analysis.get("rule_id", "unknown")
        
        # Basic patch result structure
        patch_result = {
            "patch_id": f"rust_{rule_id}",
            "patch_type": "suggestion",
            "language": "rust",
            "framework": context.get("framework", ""),
            "suggestion": analysis.get("suggestion", "No suggestion available"),
            "confidence": analysis.get("confidence", "low"),
            "severity": analysis.get("severity", "medium"),
            "root_cause": root_cause
        }
        
        # Try to find a specific template for this root cause
        template_name = f"{root_cause}.rs.template"
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
            # Generate code suggestions based on the root cause
            if root_cause == "rust_unwrap_none":
                patch_result["suggestion_code"] = self._generate_unwrap_none_suggestion(analysis, context)
            elif root_cause == "rust_unwrap_err":
                patch_result["suggestion_code"] = self._generate_unwrap_err_suggestion(analysis, context)
            elif root_cause == "rust_index_out_of_bounds":
                patch_result["suggestion_code"] = self._generate_bounds_check_suggestion(analysis, context)
            elif root_cause == "rust_division_by_zero":
                patch_result["suggestion_code"] = self._generate_division_check_suggestion(analysis, context)
            elif root_cause == "rust_deadlock":
                patch_result["suggestion_code"] = self._generate_deadlock_suggestion(analysis, context)
        
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
        variables["ERROR_TYPE"] = error_data.get("error_type", "Error")
        variables["ERROR_MESSAGE"] = error_data.get("message", "Unknown error")
        
        # Extract information from stack trace
        stack_trace = error_data.get("stack_trace", [])
        if stack_trace and isinstance(stack_trace, list):
            if isinstance(stack_trace[0], dict):
                # Structured stack trace
                if stack_trace:
                    top_frame = stack_trace[0]
                    variables["MODULE"] = top_frame.get("module", "")
                    variables["FUNCTION"] = top_frame.get("function", "")
                    variables["FILE"] = top_frame.get("file", "")
                    variables["LINE"] = str(top_frame.get("line", ""))
        
        # Extract variables from context
        variables["CODE_SNIPPET"] = context.get("code_snippet", "")
        variables["IMPORTS"] = context.get("imports", "")
        
        # Extract match groups from the rule match
        match_groups = analysis.get("match_groups", ())
        for i, group in enumerate(match_groups):
            variables[f"MATCH_{i+1}"] = str(group)
        
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
                    "module": top_frame.get("module", ""),
                    "function": top_frame.get("function", "")
                })
        
        return application_point
    
    def _generate_instructions(self, analysis: Dict[str, Any], patch_code: str) -> str:
        """Generate human-readable instructions for applying the patch."""
        root_cause = analysis.get("root_cause", "unknown")
        
        if "unwrap_none" in root_cause:
            return "Replace unwrap() on Options with safe alternatives that handle the None case."
        elif "unwrap_err" in root_cause:
            return "Replace unwrap() on Results with safe alternatives that handle the Err case."
        elif "index_out_of_bounds" in root_cause:
            return "Use bounds-checking methods like .get() or check indices before accessing collections."
        elif "division_by_zero" in root_cause:
            return "Check for zero divisors before performing division operations."
        elif "deadlock" in root_cause:
            return "Revise mutex acquisition order or use alternate synchronization patterns."
        else:
            return f"Apply the suggested fix to resolve the error."
    
    def _generate_unwrap_none_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for Option unwrapping in Rust."""
        return """// Replace:
let value = optional.unwrap();

// With one of these alternatives:
// 1. Provide a default value
let value = optional.unwrap_or(default_value);

// 2. Compute a default value
let value = optional.unwrap_or_else(|| compute_default());

// 3. Use pattern matching for more control
let value = match optional {
    Some(v) => v,
    None => {
        // Handle None case
        // Log error, provide default, or bail out
        default_value
    }
};

// 4. Use the ? operator (for functions that return Result or Option)
// This will return early if None is encountered
let value = optional.ok_or(SomeError)?;"""
    
    def _generate_unwrap_err_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for Result unwrapping in Rust."""
        return """// Replace:
let value = result.unwrap();

// With one of these alternatives:
// 1. Provide a default value
let value = result.unwrap_or(default_value);

// 2. Compute a default with the error available
let value = result.unwrap_or_else(|err| {
    // Log the error
    eprintln!("Error occurred: {:?}", err);
    // Return the default
    default_value
});

// 3. Use pattern matching for more control
let value = match result {
    Ok(v) => v,
    Err(e) => {
        // Handle error case
        // Log error, provide default, or bail out
        eprintln!("Error: {:?}", e);
        default_value
    }
};

// 4. Use the ? operator to return errors early
// This will return early if Err is encountered
let value = result?;"""
    
    def _generate_bounds_check_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for bounds checking in Rust."""
        return """// Replace direct indexing:
let value = collection[index];

// With one of these alternatives:
// 1. Check bounds before indexing
if index < collection.len() {
    let value = collection[index];
    // Use value
} else {
    // Handle out-of-bounds case
    // Log error or use default
}

// 2. Use get() which returns Option<&T>
match collection.get(index) {
    Some(value) => {
        // Use value
    },
    None => {
        // Handle out-of-bounds case
    }
}

// 3. Use get() with unwrap_or
let value = collection.get(index).unwrap_or(&default_value);

// 4. For mutable access, use get_mut()
if let Some(value) = collection.get_mut(index) {
    // Modify value
    *value = new_value;
}"""
    
    def _generate_division_check_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for safe division in Rust."""
        return """// Replace:
let result = dividend / divisor;

// With one of these alternatives:
// 1. Check for zero before dividing
if divisor != 0 {
    let result = dividend / divisor;
    // Use result
} else {
    // Handle division by zero
    // Log error or use default
}

// 2. Use checked_div which returns None for division by zero
match dividend.checked_div(divisor) {
    Some(result) => {
        // Use result
    },
    None => {
        // Handle division by zero
    }
}

// 3. Return Result type from a function
fn safe_division(dividend: i32, divisor: i32) -> Result<i32, &'static str> {
    if divisor == 0 {
        Err("division by zero")
    } else {
        Ok(dividend / divisor)
    }
}"""
    
    def _generate_deadlock_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for avoiding deadlocks in Rust."""
        return """// Problem: Potential deadlock when acquiring locks

// Solution 1: Ensure consistent lock ordering
// Always acquire locks in the same order in all code paths
let _lock_a = mutex_a.lock().unwrap();
let _lock_b = mutex_b.lock().unwrap();

// Solution 2: Reduce lock scope
{
    let _lock_a = mutex_a.lock().unwrap();
    // Do minimal work with lock A
}
// Lock A is released here
{
    let _lock_b = mutex_b.lock().unwrap();
    // Do work with lock B
}

// Solution 3: Use try_lock with timeout (requires parking_lot)
use std::time::Duration;
use parking_lot::Mutex;

match mutex_a.try_lock_for(Duration::from_millis(100)) {
    Some(lock_a) => {
        // Got lock A, try for lock B
    },
    None => {
        // Could not get lock A, handle or retry
    }
};

// Solution 4: Use channels instead of mutexes
use std::sync::mpsc::channel;
let (sender, receiver) = channel();
// Send data through the channel
// Receive in another thread"""


class RustLanguagePlugin(LanguagePlugin):
    """
    Rust language plugin for Homeostasis.
    
    Provides comprehensive error analysis and fix generation for Rust applications,
    including support for memory safety, concurrency, and popular Rust frameworks
    like Actix Web and Rocket.
    """
    
    VERSION = "0.1.0"
    AUTHOR = "Homeostasis Contributors"
    
    def __init__(self):
        """Initialize the Rust language plugin."""
        self.adapter = RustErrorAdapter()
        self.error_handler = RustErrorHandler()
        self.patch_generator = RustPatchGenerator()
    
    def get_language_id(self) -> str:
        """Get the language identifier."""
        return "rust"
    
    def get_language_name(self) -> str:
        """Get the language name."""
        return "Rust"
    
    def get_language_version(self) -> str:
        """Get the language version."""
        return "1.0+"
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Rust error.
        
        Args:
            error_data: Rust error data
            
        Returns:
            Analysis results
        """
        # First, normalize the error
        if "language" not in error_data or error_data["language"] != "rust":
            standard_error = self.normalize_error(error_data)
        else:
            standard_error = error_data
        
        # Use the error handler to analyze the error
        return self.error_handler.analyze_error(standard_error)
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a Rust error to the standard format.
        
        Args:
            error_data: Rust error data
            
        Returns:
            Error data in the standard format
        """
        return self.adapter.to_standard_format(error_data)
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Rust format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the Rust format
        """
        return self.adapter.from_standard_format(standard_error)
    
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a fix for a Rust error.
        
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
        return ["actix", "rocket", "tokio", "diesel", "base"]


# Register this plugin
register_plugin(RustLanguagePlugin())