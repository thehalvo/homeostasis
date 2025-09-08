"""
PHP Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in PHP applications.
It provides error handling for PHP's exception patterns and supports PHP frameworks
including Laravel, Symfony, and WordPress.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from ..language_adapters import PHPErrorAdapter
from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class PHPExceptionHandler:
    """
    Handles PHP exceptions with pattern-based error detection and classification.

    This class provides logic for categorizing PHP exceptions based on their type,
    message, and stack trace patterns. It supports both standard PHP exceptions and
    framework-specific exceptions.
    """

    def __init__(self):
        """Initialize the PHP exception handler."""
        self.rule_categories = {
            "core": "Core PHP exceptions",
            "laravel": "Laravel framework exceptions",
            "symfony": "Symfony framework exceptions",
            "wordpress": "WordPress CMS exceptions",
            "composer": "Composer dependency exceptions",
            "database": "Database-related exceptions",
            "validation": "Input validation exceptions",
            "security": "Security-related exceptions",
            "syntax": "PHP syntax errors",
            "runtime": "Runtime exceptions",
            "extension": "PHP extension-related exceptions",
        }

        # Load rules from different categories
        self.rules = self._load_all_rules()

        # Initialize caches for performance
        self.pattern_cache = {}  # Compiled regex patterns
        self.rule_match_cache = {}  # Previous rule matches

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a PHP exception to determine its root cause and suggest potential fixes.

        Args:
            error_data: PHP error data in standard format

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
                if not any(
                    re.search(pattern, error_type) for pattern in applies_to_patterns
                ):
                    continue

            # Get or compile the regex pattern
            if pattern not in self.pattern_cache:
                try:
                    self.pattern_cache[pattern] = re.compile(
                        pattern, re.IGNORECASE | re.DOTALL
                    )
                except Exception as e:
                    logger.warning(
                        f"Invalid pattern in rule {rule.get('id', 'unknown')}: {e}"
                    )
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
                        "root_cause": rule.get("root_cause", "php_unknown_error"),
                        "description": rule.get("description", "Unknown PHP error"),
                        "suggestion": rule.get("suggestion", "No suggestion available"),
                        "confidence": rule.get("confidence", "medium"),
                        "severity": rule.get("severity", "medium"),
                        "category": rule.get("category", "php"),
                        "match_groups": match.groups() if match.groups() else tuple(),
                        "framework": rule.get("framework", ""),
                    }

                    # Cache the result for this error signature
                    error_signature = f"{error_type}:{message[:100]}"
                    self.rule_match_cache[error_signature] = result

                    return result
            except Exception as e:
                logger.warning(f"Error applying rule {rule.get('id', 'unknown')}: {e}")

        # If no rule matched, try the fallback handlers
        return self._handle_fallback(error_data)

    def _create_match_text(
        self, error_type: str, message: str, stack_trace: List
    ) -> str:
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
                            file_path = frame.get("file", "<unknown>")
                            line_num = frame.get("line", "?")
                            method = frame.get("function", "<unknown>")

                            if method:
                                trace_lines.append(
                                    f"{file_path}:{line_num}:in `{method}'"
                                )
                            else:
                                trace_lines.append(f"{file_path}:{line_num}")

                    match_text += "\n" + "\n".join(trace_lines)

        return match_text

    def _handle_fallback(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle exceptions that didn't match any specific rule.

        Args:
            error_data: PHP error data in standard format

        Returns:
            Fallback analysis result
        """
        error_type = error_data.get("error_type", "")
        message = error_data.get("message", "")

        # Try to categorize by common PHP exception types
        if "Undefined variable" in error_type or "Undefined variable" in message:
            return {
                "error_data": error_data,
                "rule_id": "php_undefined_variable",
                "error_type": error_type,
                "root_cause": "php_undefined_variable",
                "description": "Undefined variable accessed",
                "suggestion": "Initialize the variable before using it or check with isset() if it exists. Consider using null coalescing operator (??) for fallback values.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": "",
            }
        elif ("Call to undefined method" in error_type or
                "Call to undefined method" in message):
            return {
                "error_data": error_data,
                "rule_id": "php_undefined_method",
                "error_type": error_type,
                "root_cause": "php_undefined_method",
                "description": "Call to a method that doesn't exist",
                "suggestion": "Check method name for typos. Verify the class implements the method or use method_exists() to check before calling.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
                "match_groups": tuple(),
                "framework": "",
            }
        elif ("Call to undefined function" in error_type or
                "Call to undefined function" in message):
            return {
                "error_data": error_data,
                "rule_id": "php_undefined_function",
                "error_type": error_type,
                "root_cause": "php_undefined_function",
                "description": "Call to a function that doesn't exist",
                "suggestion": "Check function name for typos. Ensure the required extension is installed or the function is properly imported.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
                "match_groups": tuple(),
                "framework": "",
            }
        elif ("Call to a member function" in error_type or
                "Call to a member function" in message):
            return {
                "error_data": error_data,
                "rule_id": "php_null_reference",
                "error_type": error_type,
                "root_cause": "php_null_reference",
                "description": "Attempted to call a method on a null object",
                "suggestion": "Add null check before calling methods on potentially null variables. Use null safe operator (?->) in PHP 8+ or use optional chaining pattern.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
                "match_groups": tuple(),
                "framework": "",
            }
        elif "Parse error" in error_type or "syntax error" in message.lower():
            return {
                "error_data": error_data,
                "rule_id": "php_syntax_error",
                "error_type": error_type,
                "root_cause": "php_syntax_error",
                "description": "PHP code contains syntax errors",
                "suggestion": "Fix the syntax issue. Common issues include missing semicolons, unbalanced brackets, or incorrect PHP tags.",
                "confidence": "high",
                "severity": "high",
                "category": "syntax",
                "match_groups": tuple(),
                "framework": "",
            }
        elif ("require" in error_type.lower() or
                "include" in error_type.lower() or
                "Failed opening required" in message):
            return {
                "error_data": error_data,
                "rule_id": "php_require_error",
                "error_type": error_type,
                "root_cause": "php_missing_file",
                "description": "Failed to require or include a file",
                "suggestion": "Check the file path is correct. Ensure the file exists and is readable by the web server.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
                "match_groups": tuple(),
                "framework": "",
            }
        elif "ModelNotFoundException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "laravel_model_not_found",
                "error_type": error_type,
                "root_cause": "laravel_model_not_found",
                "description": "Attempted to find a model that doesn't exist",
                "suggestion": "Use findOrFail() with try/catch or find() with a null check. Consider using firstOrNew(), firstOrCreate(), or updateOrCreate() for operations that might need to create records.",
                "confidence": "high",
                "severity": "medium",
                "category": "database",
                "match_groups": tuple(),
                "framework": "laravel",
            }
        elif "ValidationException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "laravel_validation_failed",
                "error_type": error_type,
                "root_cause": "laravel_validation_error",
                "description": "Form validation failed for user input",
                "suggestion": "Ensure proper validation rules are defined. Add clear validation error messages. Use old() helper to preserve input on validation failure.",
                "confidence": "high",
                "severity": "medium",
                "category": "validation",
                "match_groups": tuple(),
                "framework": "laravel",
            }

        # Generic fallback for unknown exceptions
        return {
            "error_data": error_data,
            "rule_id": "php_generic_error",
            "error_type": error_type or "Error",
            "root_cause": "php_unknown_error",
            "description": "Unrecognized PHP error",
            "suggestion": "Review the exception details and stack trace for more information.",
            "confidence": "low",
            "severity": "medium",
            "category": "core",
            "match_groups": tuple(),
            "framework": "",
        }

    def _load_all_rules(self) -> List[Dict[str, Any]]:
        """
        Load PHP exception rules from all categories.

        Returns:
            Combined list of rule definitions
        """
        all_rules = []

        # Core PHP exceptions (always included)
        all_rules.extend(self._load_core_php_rules())

        # Load additional rules from files if available
        rules_dir = Path(__file__).parent.parent / "rules" / "php"
        if rules_dir.exists():
            for rule_file in rules_dir.glob("*.json"):
                try:
                    with open(rule_file, "r") as f:
                        data = json.load(f)
                        all_rules.extend(data.get("rules", []))
                except Exception as e:
                    logger.warning(f"Error loading rules from {rule_file}: {e}")

        return all_rules

    def _load_core_php_rules(self) -> List[Dict[str, Any]]:
        """Load rules for core PHP exceptions."""
        return [
            {
                "id": "php_undefined_variable",
                "pattern": "Undefined variable\\s*:\\s*\\$(\\w+)",
                "type": "E_NOTICE",
                "description": "Undefined variable accessed",
                "root_cause": "php_undefined_variable",
                "suggestion": "Initialize the variable before using it or check with isset() if it exists. Consider using null coalescing operator (??) for fallback values.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
            },
            {
                "id": "php_undefined_method",
                "pattern": "Call to undefined method ([\\w\\\\]+)::([\\w]+)\\(\\)",
                "type": "E_ERROR",
                "description": "Call to a method that doesn't exist",
                "root_cause": "php_undefined_method",
                "suggestion": "Check method name for typos. Verify the class implements the method or use method_exists() to check before calling.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
            },
            {
                "id": "php_undefined_function",
                "pattern": "Call to undefined function ([\\w\\\\]+)\\(\\)",
                "type": "E_ERROR",
                "description": "Call to a function that doesn't exist",
                "root_cause": "php_undefined_function",
                "suggestion": "Check function name for typos. Ensure the required extension is installed or the function is properly imported.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
            },
            {
                "id": "php_null_reference",
                "pattern": "Call to a member function ([\\w]+)\\(\\) on null",
                "type": "E_ERROR",
                "description": "Attempted to call a method on a null object",
                "root_cause": "php_null_reference",
                "suggestion": "Add null check before calling methods on potentially null variables. Use null safe operator (?->) in PHP 8+ or use optional chaining pattern.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
            },
            {
                "id": "php_type_error",
                "pattern": "TypeError: Argument (\\d+) passed to ([\\w\\\\]+)::([\\w]+)\\(\\) must be of (the type|type) ([\\w]+), ([\\w]+) given",
                "type": "TypeError",
                "description": "Function called with incorrect argument type",
                "root_cause": "php_type_error",
                "suggestion": "Ensure correct type is passed to the function. Use type casting or validation before passing the argument. In PHP 7+, consider adding type declarations to functions.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
            },
            {
                "id": "php_syntax_error",
                "pattern": "Parse error: syntax error, unexpected ([^,]+)( in |,)",
                "type": "E_PARSE",
                "description": "PHP code contains syntax errors",
                "root_cause": "php_syntax_error",
                "suggestion": "Fix the syntax issue. Common issues include missing semicolons, unbalanced brackets, or incorrect PHP tags.",
                "confidence": "high",
                "severity": "high",
                "category": "syntax",
            },
            {
                "id": "php_index_error",
                "pattern": "Undefined (index|offset|array key):\\s*([^\\s]+)",
                "type": "E_NOTICE",
                "description": "Attempted to access an array index that doesn't exist",
                "root_cause": "php_undefined_index",
                "suggestion": "Check if the index exists with isset() or array_key_exists() before accessing it. Use null coalescing operator (??) for fallback values.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
            },
            {
                "id": "php_require_error",
                "pattern": "Failed opening required '([^']+)'",
                "type": "E_COMPILE_ERROR",
                "description": "Failed to require a file",
                "root_cause": "php_missing_file",
                "suggestion": "Check the file path is correct. Ensure the file exists and is readable by the web server.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
            },
            {
                "id": "php_class_not_found",
                "pattern": "Class '([^']+)' not found",
                "type": "E_ERROR",
                "description": "Referenced a class that doesn't exist",
                "root_cause": "php_missing_class",
                "suggestion": "Check class name for typos. Ensure the class file is included/required or properly autoloaded. Verify namespace declarations.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
            },
            {
                "id": "php_autoload_error",
                "pattern": "spl_autoload_call\\(\\):\\s*failed to open stream",
                "type": "E_ERROR",
                "description": "Class autoloader failed to load a class",
                "root_cause": "php_autoload_error",
                "suggestion": "Check autoloader configuration. Ensure namespace and directory structure match PSR-4 standards. Verify composer dump-autoload has been run.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
            },
            {
                "id": "php_division_by_zero",
                "pattern": "Division by zero",
                "type": "E_WARNING",
                "description": "Attempted to divide by zero",
                "root_cause": "php_division_by_zero",
                "suggestion": "Add a check to ensure the divisor is not zero before performing division.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
            },
            {
                "id": "php_memory_limit",
                "pattern": "Allowed memory size of (\\d+) bytes exhausted",
                "type": "E_ERROR",
                "description": "Script exceeded memory limit",
                "root_cause": "php_memory_limit",
                "suggestion": "Optimize memory usage in the script. Consider increasing memory_limit in php.ini for memory-intensive operations.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime",
            },
            {
                "id": "php_max_execution_time",
                "pattern": "Maximum execution time of (\\d+) seconds exceeded",
                "type": "E_ERROR",
                "description": "Script execution timed out",
                "root_cause": "php_timeout",
                "suggestion": "Optimize script performance. For long-running tasks, consider breaking them into smaller chunks or using background processing.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime",
            },
            {
                "id": "php_pdo_error",
                "pattern": "SQLSTATE\\[([^\\]]+)\\]",
                "type": "PDOException",
                "description": "Database operation failed",
                "root_cause": "php_database_error",
                "suggestion": "Check SQL syntax. Verify database credentials and connection parameters. Consider using prepared statements to prevent SQL injection.",
                "confidence": "high",
                "severity": "high",
                "category": "database",
            },
        ]


class PHPPatchGenerator:
    """
    Generates patch solutions for PHP exceptions.

    This class provides capabilities to generate code fixes for common PHP errors,
    using templates and contextual information about the exception.
    """

    def __init__(self):
        """Initialize the PHP patch generator."""
        self.templates_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates" / "php"
        )
        self.templates_dir.mkdir(exist_ok=True, parents=True)

        # Cache for loaded templates
        self.template_cache = {}

    def generate_patch(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a patch for a PHP error based on analysis.

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
            "patch_id": f"php_{rule_id}",
            "patch_type": "suggestion",
            "language": "php",
            "framework": context.get("framework", ""),
            "suggestion": analysis.get("suggestion", "No suggestion available"),
            "confidence": analysis.get("confidence", "low"),
            "severity": analysis.get("severity", "medium"),
            "root_cause": root_cause,
        }

        # Try to find a specific template for this root cause
        template_name = f"{root_cause}.php.template"
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
                patch_result.update(
                    {
                        "patch_type": "code",
                        "patch_code": patch_code,
                        "application_point": self._determine_application_point(
                            analysis, context
                        ),
                        "instructions": self._generate_instructions(
                            analysis, patch_code
                        ),
                    }
                )

                # Increase confidence for code patches
                if patch_result["confidence"] == "low":
                    patch_result["confidence"] = "medium"
            except Exception as e:
                logger.warning(f"Error generating patch for {root_cause}: {e}")

        # If we don't have a specific template, return a suggestion-based patch
        if "patch_code" not in patch_result:
            # Generate code suggestions based on the root cause
            if root_cause == "php_undefined_variable":
                patch_result["suggestion_code"] = (
                    self._generate_undefined_variable_suggestion(analysis, context)
                )
            elif root_cause == "php_null_reference":
                patch_result["suggestion_code"] = self._generate_null_check_suggestion(
                    analysis, context
                )
            elif root_cause == "php_undefined_index":
                patch_result["suggestion_code"] = (
                    self._generate_undefined_index_suggestion(analysis, context)
                )
            elif root_cause == "laravel_model_not_found":
                patch_result["suggestion_code"] = (
                    self._generate_model_not_found_suggestion(analysis, context)
                )

        return patch_result

    def _load_template(self, template_path: Path) -> str:
        """Load a template from the filesystem or cache."""
        path_str = str(template_path)
        if path_str not in self.template_cache:
            if template_path.exists():
                with open(template_path, "r") as f:
                    self.template_cache[path_str] = f.read()
            else:
                raise FileNotFoundError(f"Template not found: {template_path}")

        return self.template_cache[path_str]

    def _extract_variables(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Extract variables from analysis and context for template substitution."""
        variables = {}

        # Extract basic information
        error_data = analysis.get("error_data", {})
        variables["ERROR_TYPE"] = error_data.get("error_type", "")
        variables["ERROR_MESSAGE"] = error_data.get("message", "")

        # Extract information from stack trace
        stack_trace = error_data.get("stack_trace", [])
        if stack_trace and isinstance(stack_trace, list):
            if isinstance(stack_trace[0], dict):
                # Structured stack trace
                if stack_trace:
                    top_frame = stack_trace[0]
                    variables["FILE"] = top_frame.get("file", "")
                    variables["LINE"] = str(top_frame.get("line", ""))
                    variables["METHOD"] = top_frame.get("function", "")
                    variables["CLASS"] = top_frame.get("class", "")

        # Extract variables from context
        variables["CODE_SNIPPET"] = context.get("code_snippet", "")

        # Extract match groups from the rule match
        match_groups = analysis.get("match_groups", ())
        for i, group in enumerate(match_groups):
            variables[f"MATCH_{i + 1}"] = str(group)

        # Extract common variables based on error type
        if analysis.get("root_cause") == "php_undefined_variable" and match_groups:
            variables["VARIABLE"] = match_groups[0]
        elif analysis.get("root_cause") == "php_null_reference" and match_groups:
            variables["METHOD"] = match_groups[0]
        elif analysis.get("root_cause") == "laravel_model_not_found" and match_groups:
            variables["MODEL_CLASS"] = match_groups[0]
            variables["MODEL_VAR"] = (
                match_groups[0].lower() if match_groups[0] else "model"
            )

        return variables

    def _apply_template(self, template: str, variables: Dict[str, str]) -> str:
        """Apply variables to a template."""
        result = template
        for key, value in variables.items():
            placeholder = f"${{{key}}}"
            result = result.replace(placeholder, value)
        return result

    def _determine_application_point(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine where to apply the patch."""
        error_data = analysis.get("error_data", {})
        stack_trace = error_data.get("stack_trace", [])

        application_point = {
            "type": "suggestion",
            "description": "Review the code based on the suggestion",
        }

        if stack_trace and isinstance(stack_trace, list):
            if isinstance(stack_trace[0], dict):
                # We have structured stack trace, extract file and line
                top_frame = stack_trace[0]
                application_point.update(
                    {
                        "type": "line",
                        "file": top_frame.get("file", ""),
                        "line": top_frame.get("line", 0),
                        "method": top_frame.get("function", ""),
                    }
                )

        return application_point

    def _generate_instructions(self, analysis: Dict[str, Any], patch_code: str) -> str:
        """Generate human-readable instructions for applying the patch."""
        root_cause = analysis.get("root_cause", "unknown")

        if "undefined_variable" in root_cause:
            return "Initialize the variable before use or add checks with isset()."
        elif "null_reference" in root_cause:
            return "Add null checks before accessing object methods."
        elif "undefined_index" in root_cause:
            return "Check array keys with isset() or array_key_exists() before access."
        elif "model_not_found" in root_cause:
            return "Use find() with a null check or try/catch with findOrFail()."
        else:
            return "Apply the suggested fix to resolve the error."

    def _generate_undefined_variable_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for undefined variable handling in PHP."""
        variable = ""
        if analysis.get("match_groups"):
            variable = analysis.get("match_groups")[0]
        else:
            # Try to extract from error message
            message = analysis.get("error_data", {}).get("message", "")
            match = re.search(r"Undefined variable:\s*\$?(\w+)", message)
            if match:
                variable = match.group(1)

        if not variable:
            variable = "variable"

        return f"""// Option 1: Initialize the variable with a default value
${variable} = null; // or appropriate default

// Option 2: Check if variable exists before use
if (isset(${variable})) {{
    // Use ${variable} here
}} else {{
    // Handle the case where ${variable} is not set
    ${variable} = null; // or default value
}}

// Option 3: Use null coalescing operator (PHP 7+)
${variable} = ${variable} ?? null; // or default value

// Option 4: For arrays, initialize as empty array
${variable} = [];

// Option 5: Use variable variables cautiously if that's what you intended
$var_name = '{variable}';
$$var_name = 'some value';
"""

    def _generate_null_check_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for null checking in PHP."""
        method = ""
        if analysis.get("match_groups"):
            method = analysis.get("match_groups")[0]

        variable = "object"

        return f"""// Option 1: Use simple null check
if (${variable} !== null) {{
    ${variable}->{method}();
}} else {{
    // Handle null case
    // e.g., log error, return early, use default
}}

// Option 2: Use null coalescing operator (PHP 7+)
$result = ${variable}?->{method}() ?? null;

// Option 3: Use class_exists and method_exists for dynamic calls
if (${variable} !== null && method_exists(${variable}, '{method}')) {{
    ${variable}->{method}();
}}

// Option 4: For optional operations, use a try-catch
try {{
    if (${variable} !== null) {{
        ${variable}->{method}();
    }}
}} catch (Error $e) {{
    // Log or handle the error
    error_log("Error calling {method}: " . $e->getMessage());
}}
"""

    def _generate_undefined_index_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for undefined array index in PHP."""
        index = ""
        if analysis.get("match_groups"):
            index = analysis.get("match_groups")[1]  # Typically the second match group

        if not index:
            index = "key"

        return f"""// Option 1: Check with isset() before accessing
if (isset($array['{index}'])) {{
    $value = $array['{index}'];
}} else {{
    $value = null; // or default value
}}

// Option 2: Use array_key_exists for checking
if (array_key_exists('{index}', $array)) {{
    $value = $array['{index}'];
}}

// Option 3: Use null coalescing operator (PHP 7+)
$value = $array['{index}'] ?? null; // or default value

// Option 4: For nested arrays, use the null coalescing operator chain
$value = $array['{index}']['nested_key'] ?? null;

// Option 5: For numeric indices in sequential arrays, check array bounds
if (is_array($array) && count($array) > {index}) {{
    $value = $array[{index}];
}}
"""

    def _generate_model_not_found_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for Laravel ModelNotFoundException."""
        model = ""
        if analysis.get("match_groups"):
            model = analysis.get("match_groups")[0]

        if not model:
            model = "Model"

        var_name = model.split("\\")[-1].lower()

        return f"""// Option 1: Use find() with null check (doesn't throw exception)
${var_name} = {model}::find($id);
if (${var_name} === null) {{
    // Handle not found case
    return redirect()->back()->with('error', '{model} not found');
}}

// Option 2: Use findOrFail() with try-catch
try {{
    ${var_name} = {model}::findOrFail($id);
    // Continue with the model
}} catch (\\Illuminate\\Database\\Eloquent\\ModelNotFoundException $e) {{
    // Handle the exception
    return response()->json(['error' => '{model} not found'], 404);
}}

// Option 3: Use firstWhere (returns null if not found)
${var_name} = {model}::where('id', $id)->first();
if (${var_name} === null) {{
    // Handle not found
}}

// Option 4: Use firstOrNew/firstOrCreate (creates if not found)
${var_name} = {model}::firstOrCreate(
    ['id' => $id],  // attributes to search by
    ['name' => 'Default Name']  // default attributes if creating
);
"""


class PHPLanguagePlugin(LanguagePlugin):
    """
    PHP language plugin for Homeostasis.

    Provides comprehensive error analysis and fix generation for PHP applications,
    including support for Laravel, Symfony, and other PHP frameworks.
    """

    VERSION = "0.1.0"
    AUTHOR = "Homeostasis Contributors"

    def __init__(self):
        """Initialize the PHP language plugin."""
        self.adapter = PHPErrorAdapter()
        self.exception_handler = PHPExceptionHandler()
        self.patch_generator = PHPPatchGenerator()

    def get_language_id(self) -> str:
        """Get the language identifier."""
        return "php"

    def get_language_name(self) -> str:
        """Get the language name."""
        return "PHP"

    def get_language_version(self) -> str:
        """Get the language version."""
        return "7.0+"

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a PHP error.

        Args:
            error_data: PHP error data

        Returns:
            Analysis results
        """
        # First, normalize the error
        if "language" not in error_data or error_data["language"] != "php":
            standard_error = self.normalize_error(error_data)
        else:
            standard_error = error_data

        # Use the exception handler to analyze the error
        return self.exception_handler.analyze_error(standard_error)

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a PHP error to the standard format.

        Args:
            error_data: PHP error data

        Returns:
            Error data in the standard format
        """
        return self.adapter.to_standard_format(error_data)

    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to PHP format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the PHP format
        """
        return self.adapter.from_standard_format(standard_error)

    def generate_fix(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a fix for a PHP error.

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
        return ["laravel", "symfony", "wordpress", "codeigniter", "base"]


# Register this plugin
register_plugin(PHPLanguagePlugin())
