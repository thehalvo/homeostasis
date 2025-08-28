"""
Scala Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Scala applications.
It provides comprehensive exception handling for standard Scala errors and
framework-specific issues including Akka, Play Framework, and other JVM-related errors.
"""
import logging
import re
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from ..language_plugin_system import LanguagePlugin, register_plugin
from ..language_adapters import ScalaErrorAdapter

logger = logging.getLogger(__name__)


class ScalaExceptionHandler:
    """
    Handles Scala exceptions with a robust error detection and classification system.
    
    This class provides logic for categorizing Scala exceptions based on their type,
    message, and stack trace patterns. It supports both standard Scala exceptions and
    framework-specific exceptions.
    """
    
    def __init__(self):
        """Initialize the Scala exception handler."""
        self.rule_categories = {
            "core": "Core Scala exceptions",
            "io": "IO and file-related exceptions",
            "collections": "Collection and data structure exceptions",
            "concurrency": "Threading and concurrency exceptions",
            "akka": "Akka framework exceptions",
            "play": "Play Framework exceptions",
            "functional": "Functional programming related exceptions",
            "typesystem": "Scala type system related exceptions",
            "jdbc": "Database and JDBC exceptions",
            "sbt": "SBT build tool exceptions",
            "jvm": "JVM-related exceptions",
            "security": "Security-related exceptions",
            "network": "Network and remote communication exceptions"
        }
        
        # Load rules from different categories
        self.rules = self._load_all_rules()
        
        # Initialize caches for performance
        self.pattern_cache = {}  # Compiled regex patterns
        self.rule_match_cache = {}  # Previous rule matches
    
    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Scala exception to determine its root cause and suggest potential fixes.
        
        Args:
            error_data: Scala error data in standard format
            
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
                        "error_type": error_type,  # Preserve original error type
                        "root_cause": rule.get("root_cause", "scala_unknown_error"),
                        "description": rule.get("description", "Unknown Scala error"),
                        "suggestion": rule.get("suggestion", "No suggestion available"),
                        "confidence": rule.get("confidence", "medium"),
                        "severity": rule.get("severity", "medium"),
                        "category": rule.get("category", "scala"),
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
            error_type: Exception type
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
                            line = f"at {frame.get('package', '')}.{frame.get('class', '')}."
                            line += f"{frame.get('function', '')}({frame.get('file', '')}:{frame.get('line', '?')})"
                            trace_lines.append(line)
                    
                    match_text += "\n" + "\n".join(trace_lines)
        
        return match_text
    
    def _handle_fallback(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle exceptions that didn't match any specific rule.
        
        Args:
            error_data: Scala error data in standard format
            
        Returns:
            Fallback analysis result
        """
        error_type = error_data.get("error_type", "")
        
        # Check for common Scala exception types and apply basic categorization
        if "NullPointerException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "scala_null_pointer",
                "error_type": error_type,
                "root_cause": "scala_null_pointer",
                "description": "Attempted to access or use a null object reference",
                "suggestion": "Add null checks before accessing objects or methods. Consider using Option[T] for values that might be null.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "NoSuchElementException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "scala_no_such_element",
                "error_type": error_type,
                "root_cause": "scala_no_such_element",
                "description": "Attempted to access a non-existent element, often with .head or .get on an empty collection",
                "suggestion": "Use .headOption instead of .head, or pattern matching with isEmpty/nonEmpty checks before accessing elements. Consider using getOrElse or fold for Option types.",
                "confidence": "high",
                "severity": "medium",
                "category": "collections",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "IndexOutOfBoundsException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "scala_index_out_of_bounds",
                "error_type": error_type,
                "root_cause": "scala_index_out_of_bounds",
                "description": "Attempted to access an array or collection element with an invalid index",
                "suggestion": "Check collection bounds before accessing elements with indices. Use collection.indices.contains(index) or lift/get to safely access elements.",
                "confidence": "high",
                "severity": "medium",
                "category": "collections",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "MatchError" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "scala_match_error",
                "error_type": error_type,
                "root_cause": "scala_incomplete_match",
                "description": "Pattern match is not exhaustive, missing cases for some possible values",
                "suggestion": "Ensure pattern matches cover all possible cases. Add a wildcard case (_) as fallback or use Option pattern matching for safer handling.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "ClassCastException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "scala_class_cast",
                "error_type": error_type,
                "root_cause": "scala_invalid_cast",
                "description": "Attempted to cast an object to an incompatible type",
                "suggestion": "Verify object types before casting using match, isInstanceOf, or use pattern matching with type ascription. Consider better type design with sealed traits.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "IllegalArgumentException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "scala_illegal_argument",
                "error_type": error_type,
                "root_cause": "scala_invalid_argument",
                "description": "A method received an argument that was not valid",
                "suggestion": "Add parameter validation checks. Consider using require() for preconditions, or design with Option/Either to handle invalid inputs functionally.",
                "confidence": "medium",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        
        # Generic fallback for unknown exceptions
        return {
            "error_data": error_data,
            "rule_id": "scala_generic_error",
            "error_type": error_type or "Unknown",
            "root_cause": "scala_unknown_error",
            "description": f"Unrecognized Scala error: {error_type}",
            "suggestion": "Review the error message and stack trace for more details. Check the documentation for this exception type.",
            "confidence": "low",
            "severity": "medium",
            "category": "scala",
            "match_groups": tuple(),
            "framework": ""
        }
    
    def _load_all_rules(self) -> List[Dict[str, Any]]:
        """
        Load Scala error rules from all categories.
        
        Returns:
            Combined list of rule definitions
        """
        all_rules = []
        
        # Core Scala exceptions (always included)
        all_rules.extend(self._load_core_scala_rules())
        
        # Load additional rules from files if they exist
        rules_dir = Path(__file__).parent.parent / "rules" / "scala"
        if rules_dir.exists():
            for rule_file in rules_dir.glob("*.json"):
                try:
                    with open(rule_file, 'r') as f:
                        data = json.load(f)
                        if "rules" in data and isinstance(data["rules"], list):
                            all_rules.extend(data["rules"])
                except Exception as e:
                    logger.error(f"Error loading rules from {rule_file}: {e}")
        
        return all_rules
    
    def _load_core_scala_rules(self) -> List[Dict[str, Any]]:
        """Load rules for core Scala exceptions."""
        return [
            {
                "id": "scala_null_pointer",
                "pattern": "java\\.lang\\.NullPointerException(?:: (.*))?",
                "type": "NullPointerException",
                "description": "Attempted to access or use a null object reference",
                "root_cause": "scala_null_pointer",
                "suggestion": "Add null checks before accessing objects or methods. Consider using Option[T] for values that might be null.",
                "confidence": "high",
                "severity": "high",
                "category": "core"
            },
            {
                "id": "scala_no_such_element",
                "pattern": "java\\.util\\.NoSuchElementException(?:: (.*))?",
                "type": "NoSuchElementException",
                "description": "Attempted to access a non-existent element, often with .head or .get on an empty collection",
                "root_cause": "scala_no_such_element",
                "suggestion": "Use .headOption instead of .head, or pattern matching with isEmpty/nonEmpty checks before accessing elements. Consider using getOrElse or fold for Option types.",
                "confidence": "high",
                "severity": "medium",
                "category": "collections"
            },
            {
                "id": "scala_match_error", 
                "pattern": "scala\\.MatchError: (.*)",
                "type": "MatchError",
                "description": "Pattern match is not exhaustive, missing cases for some possible values",
                "root_cause": "scala_incomplete_match",
                "suggestion": "Ensure pattern matches are exhaustive and cover all possible cases. Add a wildcard case (_) as fallback or use Option pattern matching for safer handling.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "scala_index_out_of_bounds",
                "pattern": "java\\.lang\\.(ArrayIndexOutOfBoundsException|IndexOutOfBoundsException|StringIndexOutOfBoundsException): (\\d+)",
                "type": "IndexOutOfBoundsException",
                "description": "Attempted to access an array or collection element with an invalid index",
                "root_cause": "scala_index_out_of_bounds",
                "suggestion": "Check collection bounds before accessing elements with indices. Use collection.indices.contains(index) or lift/get to safely access elements.",
                "confidence": "high",
                "severity": "medium",
                "category": "collections"
            },
            {
                "id": "scala_class_cast",
                "pattern": "java\\.lang\\.ClassCastException: ([^\\s]+) cannot be cast to ([^\\s]+)",
                "type": "ClassCastException",
                "description": "Attempted to cast an object to an incompatible type",
                "root_cause": "scala_invalid_cast",
                "suggestion": "Verify object types before casting using match, isInstanceOf, or use pattern matching with type ascription. Consider better type design with sealed traits.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "scala_option_empty",
                "pattern": "java\\.util\\.NoSuchElementException: None\\.get",
                "type": "NoSuchElementException",
                "description": "Called .get on an Option that is None",
                "root_cause": "scala_option_get_on_none",
                "suggestion": "Never call .get on an Option without first checking if it's defined. Use getOrElse, fold, or pattern matching instead.",
                "confidence": "high",
                "severity": "high",
                "category": "typesystem"
            },
            {
                "id": "scala_division_by_zero",
                "pattern": "java\\.lang\\.ArithmeticException: (/ by zero|Division by zero)",
                "type": "ArithmeticException",
                "description": "Attempted division by zero",
                "root_cause": "scala_division_by_zero",
                "suggestion": "Check for zero before division. Consider using Option to handle division operations safely.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "scala_illegal_argument",
                "pattern": "java\\.lang\\.IllegalArgumentException: (.*)",
                "type": "IllegalArgumentException",
                "description": "A method received an argument that was not valid",
                "root_cause": "scala_invalid_argument",
                "suggestion": "Add parameter validation checks. Consider using require() for preconditions, or design with Option/Either to handle invalid inputs functionally.",
                "confidence": "medium",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "scala_illegal_state",
                "pattern": "java\\.lang\\.IllegalStateException: (.*)",
                "type": "IllegalStateException",
                "description": "Object is in an invalid state for the requested operation",
                "root_cause": "scala_invalid_state",
                "suggestion": "Check object state before performing operations. Consider using the State monad or immutable objects to prevent invalid state transitions.",
                "confidence": "medium",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "scala_future_timeout",
                "pattern": "java\\.util\\.concurrent\\.TimeoutException(?:: (.*))?",
                "type": "TimeoutException",
                "description": "Future operation timed out",
                "root_cause": "scala_future_timeout",
                "suggestion": "Increase the timeout or optimize the operation. Consider non-blocking alternatives or circuit breakers for slow operations.",
                "confidence": "high",
                "severity": "high",
                "category": "concurrency"
            },
            {
                "id": "scala_interrupted_exception",
                "pattern": "java\\.lang\\.InterruptedException(?:: (.*))?",
                "type": "InterruptedException",
                "description": "Thread was interrupted while waiting, sleeping, or otherwise occupied",
                "root_cause": "scala_thread_interrupted",
                "suggestion": "Ensure InterruptedException is properly handled, and the interrupt state is restored if necessary.",
                "confidence": "high",
                "severity": "medium",
                "category": "concurrency"
            },
            {
                "id": "scala_parsing_error",
                "pattern": "scala\\.util\\.parsing\\.combinator\\.Parsers\\$Error: (.*)",
                "type": "Parser.Error",
                "description": "Error during parsing with the Scala parser combinators",
                "root_cause": "scala_parser_error",
                "suggestion": "Check the input being parsed and ensure it matches the grammar. Improve error reporting with custom parsers.",
                "confidence": "medium",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "scala_stack_overflow",
                "pattern": "java\\.lang\\.StackOverflowError(?:: (.*))?",
                "type": "StackOverflowError",
                "description": "Recursive function calls exceeded the stack size, typically due to infinite recursion",
                "root_cause": "scala_stack_overflow",
                "suggestion": "Check for proper recursion termination conditions. Consider using tail recursion with the @tailrec annotation or convert to an iterative approach.",
                "confidence": "high",
                "severity": "high",
                "category": "core"
            },
            {
                "id": "scala_out_of_memory",
                "pattern": "java\\.lang\\.OutOfMemoryError: (.*)",
                "type": "OutOfMemoryError",
                "description": "JVM ran out of memory",
                "root_cause": "scala_out_of_memory",
                "suggestion": "Increase JVM heap size (-Xmx), check for memory leaks, or optimize memory-intensive operations. Consider using streams for large data processing.",
                "confidence": "high",
                "severity": "critical",
                "category": "jvm"
            },
            {
                "id": "scala_type_mismatch",
                "pattern": "scala\\.reflect\\.internal\\.FatalError: type mismatch",
                "type": "FatalError",
                "description": "Type mismatch error at runtime, likely due to macro expansion or reflection",
                "root_cause": "scala_type_mismatch",
                "suggestion": "Check type parameters and ensure types are compatible. Review implicit conversions and type classes.",
                "confidence": "medium",
                "severity": "high",
                "category": "typesystem"
            }
        ]


class ScalaPatchGenerator:
    """
    Generates patch solutions for Scala exceptions.
    
    This class provides capabilities to generate code fixes for common Scala errors,
    using templates and contextual information about the exception.
    """
    
    def __init__(self):
        """Initialize the Scala patch generator."""
        self.templates_dir = Path(__file__).parent.parent / "patch_generation" / "templates" / "scala"
        self.templates_dir.mkdir(exist_ok=True, parents=True)
        
        # Cache for loaded templates
        self.template_cache = {}
    
    def generate_patch(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a patch for a Scala error based on analysis.
        
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
            "patch_id": f"scala_{rule_id}",
            "patch_type": "suggestion",
            "language": "scala",
            "framework": context.get("framework", ""),
            "suggestion": analysis.get("suggestion", "No suggestion available"),
            "confidence": analysis.get("confidence", "low"),
            "severity": analysis.get("severity", "medium"),
            "root_cause": root_cause
        }
        
        # Try to find a specific template for this root cause
        template_name = f"{root_cause}.scala.template"
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
        
        # If we don't have a specific template, return a suggestion-based patch with code examples
        if "patch_code" not in patch_result:
            # Generate specialized suggestions based on the error type
            if root_cause == "scala_null_pointer":
                patch_result["suggestion_code"] = self._generate_null_check_suggestion(analysis, context)
            elif root_cause == "scala_no_such_element":
                patch_result["suggestion_code"] = self._generate_option_handling_suggestion(analysis, context)
            elif root_cause == "scala_index_out_of_bounds":
                patch_result["suggestion_code"] = self._generate_bounds_check_suggestion(analysis, context)
            elif root_cause == "scala_incomplete_match":
                patch_result["suggestion_code"] = self._generate_match_handling_suggestion(analysis, context)
            elif root_cause == "scala_future_timeout":
                patch_result["suggestion_code"] = self._generate_future_handling_suggestion(analysis, context)
        
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
        variables["EXCEPTION_VAR"] = "ex"  # Default exception variable name
        
        # Additional variables based on error type
        if "scala_null_pointer" in analysis.get("root_cause", ""):
            variables["NULL_CHECK_VAR"] = self._extract_null_variable(analysis, context)
        
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
            return ("Add null checks or convert to Option type. " 
                   f"Consider implementing this fix: {patch_code}")
        elif "no_such_element" in root_cause:
            return ("Use safer methods for accessing optional values. " 
                   f"Implement option handling as shown: {patch_code}")
        elif "index_out_of_bounds" in root_cause:
            return ("Validate indices before accessing collections. " 
                   f"Implement bounds checking as shown: {patch_code}")
        elif "incomplete_match" in root_cause:
            return ("Ensure pattern matches cover all possible cases. " 
                   f"Use exhaustive pattern matching as shown: {patch_code}")
        else:
            return f"Apply the following fix to address the issue: {patch_code}"
    
    def _extract_null_variable(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Extract the likely null variable from an NPE."""
        message = analysis.get("error_data", {}).get("message", "")
        
        # Java 14+ helpful NPE message: "Cannot invoke "String.length()" because "str" is null"
        if "because" in message and "is null" in message:
            parts = message.split("because")
            if len(parts) > 1:
                # Extract the variable name in quotes
                import re
                match = re.search(r'"([^"]+)"', parts[1])
                if match:
                    return match.group(1)
        
        # Default fallback
        return "value"
    
    def _generate_null_check_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for null checking in Scala."""
        var_name = self._extract_null_variable(analysis, context)
        
        return f"""// Option 1: Add null check before accessing {var_name}
if ({var_name} == null) {{
  // Handle null case
  // For example, provide a default value or early return
  return defaultValue // or throw a more descriptive exception
}} else {{
  // Safe to use {var_name}
  {var_name}.someMethod()
}}

// Option 2: Convert to Option (preferred Scala approach)
Option({var_name}) match {{
  case Some(value) => value.someMethod() // Safe to use value
  case None => // Handle null case
    defaultValue // or throw a more descriptive exception
}}

// Option 3: Use Option with map or fold for more concise code
Option({var_name})
  .map(value => value.someMethod())
  .getOrElse(defaultValue)
"""
    
    def _generate_option_handling_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for safe Option handling in Scala."""
        return """// Option 1: Use getOrElse instead of get
val safeValue = optionValue.getOrElse(defaultValue)

// Option 2: Use pattern matching
optionValue match {
  case Some(value) => // Do something with value
  case None => // Handle empty case
}

// Option 3: Use map and getOrElse for transformation
val transformed = optionValue
  .map(value => transformValue(value))
  .getOrElse(defaultTransformed)

// Option 4: Use fold for different operations on Some/None
val result = optionValue.fold(
  ifEmpty = defaultValue,
  f = value => transformValue(value)
)

// Option 5: For collections, use headOption instead of head
val firstElement = collection.headOption
  .getOrElse(defaultElement)

// Option 6: For map lookups, use get (which returns Option)
val lookupResult = map.get(key)
  .getOrElse(defaultValue)
"""
    
    def _generate_bounds_check_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for bounds checking in Scala."""
        return """// Option 1: Check indices before accessing elements
if (index >= 0 && index < array.length) {
  val value = array(index) // Safe access
} else {
  // Handle invalid index
  handleInvalidIndex(index)
}

// Option 2: Use lift for collections (returns Option)
val elementOption = collection.lift(index)
elementOption match {
  case Some(value) => // Use value safely
  case None => // Handle out of bounds
}

// Option 3: More concise with getOrElse
val safeValue = collection.lift(index).getOrElse(defaultValue)

// Option 4: Check with contains for Sets
if (collection.indices.contains(index)) {
  val value = collection(index) // Safe access
} else {
  // Handle invalid index
}

// Option 5: Use applyOrElse for partial functions
val safeGet = Function.unlift(collection.lift _)
val value = safeGet.applyOrElse(index, (_: Int) => defaultValue)
"""
    
    def _generate_match_handling_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for exhaustive pattern matching in Scala."""
        return """// Option 1: Add wildcard case as fallback
value match {
  case SomeType1(x) => handleType1(x)
  case SomeType2(y) => handleType2(y)
  case _ => handleDefault() // Catch all other cases
}

// Option 2: For sealed traits/case classes, make pattern matching exhaustive
// If using a sealed hierarchy, the compiler will warn you about missing cases
sealed trait MyType
case class Type1(value: String) extends MyType
case class Type2(value: Int) extends MyType
case object Type3 extends MyType

def process(myType: MyType): String = myType match {
  case Type1(str) => s"String: $str"
  case Type2(num) => s"Number: $num"
  case Type3 => "Type3"
  // No wildcard needed - compiler will verify all cases are covered
}

// Option 3: Use Option with pattern matching
Option(value) match {
  case Some(x: Type1) => handleType1(x)
  case Some(x: Type2) => handleType2(x)
  case Some(_) => handleUnknownType()
  case None => handleNull()
}

// Option 4: Use @unchecked to silence compiler warnings (use with caution)
(value: @unchecked) match {
  case SomeType1(x) => handleType1(x)
  case SomeType2(y) => handleType2(y)
  // No default case, but compiler won't warn
}
"""
    
    def _generate_future_handling_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for handling Future timeouts in Scala."""
        return """import scala.concurrent.{Future, Await}
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Failure, Success, Try}

// Option 1: Increase timeout duration
val result = Await.result(future, 30.seconds) // Increase from default 10s

// Option 2: Handle timeout explicitly with Try
val result = Try(Await.result(future, 10.seconds)) match {
  case Success(value) => value
  case Failure(ex: TimeoutException) => 
    // Handle timeout specifically
    fallbackValue
  case Failure(ex) => 
    // Handle other exceptions
    throw ex
}

// Option 3: Use non-blocking approach (preferred)
future
  .map(result => handleSuccess(result))
  .recover {
    case ex: TimeoutException => handleTimeout()
    case ex => handleOtherErrors(ex)
  }

// Option 4: Use fallback future on timeout
import scala.concurrent.TimeoutException

val futureWithFallback = future
  .recover {
    case _: TimeoutException => fallbackValue
  }

// Option 5: Use Future.firstCompletedOf for race with timeout
val timeout = Future.failed(
  new TimeoutException("Operation timed out")
).delayed(10.seconds)

Future.firstCompletedOf(Seq(future, timeout))
  .recover {
    case _: TimeoutException => fallbackValue
  }
"""


class ScalaLanguagePlugin(LanguagePlugin):
    """
    Scala language plugin for Homeostasis.
    
    Provides comprehensive error analysis and fix generation for Scala applications,
    including support for functional programming patterns, Akka concurrency issues,
    and Play Framework.
    """
    
    VERSION = "0.1.0"
    AUTHOR = "Homeostasis Contributors"
    
    def __init__(self):
        """Initialize the Scala language plugin."""
        self.adapter = ScalaErrorAdapter()
        self.exception_handler = ScalaExceptionHandler()
        self.patch_generator = ScalaPatchGenerator()
    
    def get_language_id(self) -> str:
        """Get the language identifier."""
        return "scala"
    
    def get_language_name(self) -> str:
        """Get the language name."""
        return "Scala"
    
    def get_language_version(self) -> str:
        """Get the language version."""
        return "2.12+"
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Scala error.
        
        Args:
            error_data: Scala error data
            
        Returns:
            Analysis results
        """
        # First, normalize the error
        if "language" not in error_data or error_data["language"] != "scala":
            standard_error = self.normalize_error(error_data)
        else:
            standard_error = error_data
        
        # Use the exception handler to analyze the error
        return self.exception_handler.analyze_exception(standard_error)
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a Scala error to the standard format.
        
        Args:
            error_data: Scala error data
            
        Returns:
            Error data in the standard format
        """
        return self.adapter.to_standard_format(error_data)
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Scala format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the Scala format
        """
        return self.adapter.from_standard_format(standard_error)
    
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a fix for a Scala error.
        
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
        return ["akka", "play", "sbt", "cats", "zio", "base"]


# Register this plugin
register_plugin(ScalaLanguagePlugin())