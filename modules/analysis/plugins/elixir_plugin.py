"""
Elixir Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Elixir/Erlang applications.
It provides comprehensive exception handling for Elixir's standard errors, Phoenix Framework,
Ecto database issues, and OTP/BEAM VM-related errors.
"""
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any, List

from ..language_plugin_system import LanguagePlugin, register_plugin
from ..language_adapters import ElixirErrorAdapter

logger = logging.getLogger(__name__)


class ElixirExceptionHandler:
    """
    Handles Elixir exceptions with a robust error detection and classification system.
    
    This class provides logic for categorizing Elixir exceptions based on their type,
    message, and stack trace patterns. It supports both standard Elixir exceptions and
    framework-specific exceptions like Phoenix, Ecto, and OTP-related errors.
    """
    
    def __init__(self):
        """Initialize the Elixir exception handler."""
        self.rule_categories = {
            "core": "Core Elixir exceptions",
            "phoenix": "Phoenix Framework exceptions",
            "ecto": "Ecto database exceptions",
            "otp": "OTP and supervision tree exceptions",
            "beam": "BEAM VM exceptions",
            "plug": "Plug and web request exceptions",
            "runtime": "Runtime and protocol exceptions",
            "guard": "Guard clause and pattern matching exceptions",
            "io": "IO and file-related exceptions",
            "network": "Network and distribution exceptions",
            "security": "Security and access exceptions"
        }
        
        # Load rules from different categories
        self.rules = self._load_all_rules()
        
        # Initialize caches for performance
        self.pattern_cache = {}  # Compiled regex patterns
        self.rule_match_cache = {}  # Previous rule matches
    
    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an Elixir exception to determine its root cause and suggest potential fixes.
        
        Args:
            error_data: Elixir error data in standard format
            
        Returns:
            Analysis result with root cause, description, and fix suggestions
        """
        exception_type = error_data.get("error_type", "")
        message = error_data.get("message", "")
        stack_trace = error_data.get("stack_trace", [])
        
        # Create a consolidated text for pattern matching
        match_text = self._create_match_text(exception_type, message, stack_trace)
        
        # Check if we have previously analyzed this error
        error_signature = f"{exception_type}:{message[:100]}"
        if error_signature in self.rule_match_cache:
            return self.rule_match_cache[error_signature]
        
        # Try to match against known rules
        for rule in self.rules:
            pattern = rule.get("pattern", "")
            if not pattern:
                continue
            
            # Skip rules that don't apply to this category of exception
            if rule.get("applies_to") and exception_type:
                applies_to_patterns = rule.get("applies_to")
                if not any(re.search(pattern, exception_type) for pattern in applies_to_patterns):
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
                        "error_type": rule.get("type", exception_type),
                        "root_cause": rule.get("root_cause", "elixir_unknown_error"),
                        "description": rule.get("description", "Unknown Elixir error"),
                        "suggestion": rule.get("suggestion", "No suggestion available"),
                        "confidence": rule.get("confidence", "medium"),
                        "severity": rule.get("severity", "medium"),
                        "category": rule.get("category", "elixir"),
                        "match_groups": match.groups() if match.groups() else tuple(),
                        "framework": rule.get("framework", "")
                    }
                    
                    # Cache the result for this error signature
                    self.rule_match_cache[error_signature] = result
                    
                    return result
            except Exception as e:
                logger.warning(f"Error applying rule {rule.get('id', 'unknown')}: {e}")
        
        # If no rule matched, try the fallback handlers
        return self._handle_fallback(error_data)
    
    def _create_match_text(self, exception_type: str, message: str, stack_trace: List) -> str:
        """
        Create a consolidated text for pattern matching from error components.
        
        Args:
            exception_type: Exception type
            message: Error message
            stack_trace: Stack trace frames
            
        Returns:
            Consolidated text for pattern matching
        """
        match_text = f"{exception_type}: {message}"
        
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
                            function = frame.get("function", "")
                            arity = frame.get("arity", "")
                            file = frame.get("file", "")
                            line = frame.get("line", "?")
                            
                            arity_str = f"/{arity}" if arity else ""
                            line_str = f":{line}" if line else ""
                            
                            trace_lines.append(f"    ({module}) {file}{line_str}: {function}{arity_str}")
                    
                    match_text += "\n" + "\n".join(trace_lines)
        
        return match_text
    
    def _handle_fallback(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle exceptions that didn't match any specific rule.
        
        Args:
            error_data: Elixir error data in standard format
            
        Returns:
            Fallback analysis result
        """
        exception_type = error_data.get("error_type", "")
        
        # Check for common Elixir exception types and apply basic categorization
        if "ArgumentError" in exception_type:
            return {
                "error_data": error_data,
                "rule_id": "elixir_argument_error",
                "error_type": exception_type,
                "root_cause": "elixir_argument_error",
                "description": "Invalid argument provided to a function",
                "suggestion": "Check the function documentation to ensure you're passing the correct arguments. Validate inputs before passing them to functions.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "FunctionClauseError" in exception_type:
            return {
                "error_data": error_data,
                "rule_id": "elixir_function_clause_error",
                "error_type": exception_type,
                "root_cause": "elixir_function_clause_error",
                "description": "No function clause matched the provided arguments",
                "suggestion": "Check the function's pattern matching clauses. Add additional clauses to handle all possible input patterns, or validate inputs before calling the function.",
                "confidence": "high",
                "severity": "medium",
                "category": "guard",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "ArithmeticError" in exception_type:
            return {
                "error_data": error_data,
                "rule_id": "elixir_arithmetic_error",
                "error_type": exception_type,
                "root_cause": "elixir_arithmetic_error",
                "description": "Error during an arithmetic operation",
                "suggestion": "Check for division by zero or other invalid arithmetic operations. Validate inputs to arithmetic functions.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "KeyError" in exception_type:
            return {
                "error_data": error_data,
                "rule_id": "elixir_key_error",
                "error_type": exception_type,
                "root_cause": "elixir_key_error",
                "description": "Tried to access a key that doesn't exist in a map or keyword list",
                "suggestion": "Use Map.get/3 with a default value, or pattern match with the `%{key: value} = map` syntax with a guard clause. For keyword lists, use Keyword.get/3.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "UndefinedFunctionError" in exception_type:
            return {
                "error_data": error_data,
                "rule_id": "elixir_undefined_function_error",
                "error_type": exception_type,
                "root_cause": "elixir_undefined_function_error",
                "description": "Called a function that doesn't exist",
                "suggestion": "Check function name and arity. Ensure the module is correctly imported or required. Check for typos in function names.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "MatchError" in exception_type:
            return {
                "error_data": error_data,
                "rule_id": "elixir_match_error",
                "error_type": exception_type,
                "root_cause": "elixir_match_error",
                "description": "Pattern match failed",
                "suggestion": "Check the pattern match expression and ensure the data structure matches the expected pattern. Consider using case statements with multiple pattern clauses or adding guard clauses.",
                "confidence": "high",
                "severity": "medium",
                "category": "guard",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "ErlangError" in exception_type:
            return {
                "error_data": error_data,
                "rule_id": "elixir_erlang_error",
                "error_type": exception_type,
                "root_cause": "elixir_erlang_error",
                "description": "Error from the underlying Erlang runtime",
                "suggestion": "Check the specific Erlang error. Common issues include trying to call functions on atoms, using improper lists, or interacting with ports or processes that have terminated.",
                "confidence": "medium",
                "severity": "medium",
                "category": "beam",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "BadArityError" in exception_type:
            return {
                "error_data": error_data,
                "rule_id": "elixir_bad_arity_error",
                "error_type": exception_type,
                "root_cause": "elixir_bad_arity_error",
                "description": "Called a function with the wrong number of arguments",
                "suggestion": "Check the function's documentation for the correct number of arguments. Ensure you're passing the expected number of arguments.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "BadMapError" in exception_type:
            return {
                "error_data": error_data,
                "rule_id": "elixir_bad_map_error",
                "error_type": exception_type,
                "root_cause": "elixir_bad_map_error",
                "description": "Tried to use a map operation on a value that isn't a map",
                "suggestion": "Check the value type before using map operations. Use pattern matching or the `is_map/1` guard function to verify the value is a map.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "Phoenix" in exception_type:
            # Handle all Phoenix-specific errors
            if "NoRouteError" in exception_type:
                return {
                    "error_data": error_data,
                    "rule_id": "phoenix_no_route_error",
                    "error_type": exception_type,
                    "root_cause": "phoenix_no_route_error",
                    "description": "No route found for the given request",
                    "suggestion": "Check your router configuration. Ensure the route is defined for the given method and path. Use `mix phx.routes` to list all available routes.",
                    "confidence": "high",
                    "severity": "medium",
                    "category": "phoenix",
                    "match_groups": tuple(),
                    "framework": "phoenix"
                }
            elif "ForbiddenError" in exception_type:
                return {
                    "error_data": error_data,
                    "rule_id": "phoenix_forbidden_error",
                    "error_type": exception_type,
                    "root_cause": "phoenix_forbidden_error",
                    "description": "User tried to access a forbidden resource in Phoenix",
                    "suggestion": "Check your authorization logic. Ensure users have the necessary permissions before allowing access to resources. Consider implementing role-based access control.",
                    "confidence": "high",
                    "severity": "medium",
                    "category": "phoenix",
                    "match_groups": tuple(),
                    "framework": "phoenix"
                }
            else:
                # Generic Phoenix error
                return {
                    "error_data": error_data,
                    "rule_id": "phoenix_generic_error",
                    "error_type": exception_type,
                    "root_cause": "phoenix_generic_error",
                    "description": f"Phoenix framework error: {exception_type}",
                    "suggestion": "Check Phoenix documentation for this specific error type. Review your Phoenix application configuration and ensure all dependencies are properly configured.",
                    "confidence": "medium",
                    "severity": "medium",
                    "category": "phoenix",
                    "match_groups": tuple(),
                    "framework": "phoenix"
                }
        elif "ChangesetError" in exception_type and "Ecto" in exception_type:
            return {
                "error_data": error_data,
                "rule_id": "ecto_changeset_error",
                "error_type": exception_type,
                "root_cause": "ecto_changeset_error",
                "description": "Error with Ecto changeset operation",
                "suggestion": "Check the changeset for validation errors using `changeset.errors` before performing the operation. Ensure all required fields have values and all validations pass.",
                "confidence": "high",
                "severity": "medium",
                "category": "ecto",
                "match_groups": tuple(),
                "framework": "ecto"
            }
        elif "InvalidChangesetError" in exception_type and "Ecto" in exception_type:
            return {
                "error_data": error_data,
                "rule_id": "ecto_invalid_changeset_error",
                "error_type": exception_type,
                "root_cause": "ecto_invalid_changeset_error",
                "description": "Attempted to perform a database operation with an invalid Ecto changeset",
                "suggestion": "Check the changeset for validation errors using `changeset.errors` before performing the operation. Ensure all required fields have values and all validations pass.",
                "confidence": "high",
                "severity": "medium",
                "category": "ecto",
                "match_groups": tuple(),
                "framework": "ecto"
            }
        
        # Generic fallback for unknown exceptions
        return {
            "error_data": error_data,
            "rule_id": "elixir_generic_error",
            "error_type": exception_type or "Unknown",
            "root_cause": "elixir_unknown_error",
            "description": f"Unrecognized Elixir error: {exception_type}",
            "suggestion": "Review the error message and stack trace for more details. Check the documentation for this exception type.",
            "confidence": "low",
            "severity": "medium",
            "category": "elixir",
            "match_groups": tuple(),
            "framework": ""
        }
    
    def _load_all_rules(self) -> List[Dict[str, Any]]:
        """
        Load Elixir error rules from all categories.
        
        Returns:
            Combined list of rule definitions
        """
        all_rules = []
        
        # Core Elixir exceptions (always included)
        all_rules.extend(self._load_core_elixir_rules())
        
        # Load additional rules from files if they exist
        rules_dir = Path(__file__).parent.parent / "rules" / "elixir"
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
    
    def _load_core_elixir_rules(self) -> List[Dict[str, Any]]:
        """Load rules for core Elixir exceptions."""
        return [
            {
                "id": "elixir_argument_error",
                "pattern": "\\*\\* \\(ArgumentError\\) (.*)",
                "type": "ArgumentError",
                "description": "Invalid argument provided to a function",
                "root_cause": "elixir_argument_error",
                "suggestion": "Check the function documentation to ensure you're passing the correct arguments. Validate inputs before passing them to functions.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "elixir_function_clause_error",
                "pattern": "\\*\\* \\(FunctionClauseError\\) (.*?)\\n.*?([^:]+\\.ex:\\d+)(.*)",
                "type": "FunctionClauseError",
                "description": "No function clause matched the provided arguments",
                "root_cause": "elixir_function_clause_error",
                "suggestion": "Check the function's pattern matching clauses. Add additional clauses to handle all possible input patterns, or validate inputs before calling the function.",
                "confidence": "high",
                "severity": "medium",
                "category": "guard"
            },
            {
                "id": "elixir_undefined_function_error",
                "pattern": "\\*\\* \\(UndefinedFunctionError\\) function ([^.]+)\\.([^/]+)/(\\d+) is undefined.*",
                "type": "UndefinedFunctionError",
                "description": "Called a function that doesn't exist",
                "root_cause": "elixir_undefined_function_error",
                "suggestion": "Check function name and arity. Ensure the module is correctly imported or required. Check for typos in function names.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "elixir_key_error",
                "pattern": "\\*\\* \\(KeyError\\) key (.*?) not found in: (.*)",
                "type": "KeyError",
                "description": "Tried to access a key that doesn't exist in a map or keyword list",
                "root_cause": "elixir_key_error",
                "suggestion": "Use Map.get/3 with a default value, or pattern match with the `%{key: value} = map` syntax with a guard clause. For keyword lists, use Keyword.get/3.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "elixir_case_clause_error",
                "pattern": "\\*\\* \\(CaseClauseError\\) no case clause matching: (.*)",
                "type": "CaseClauseError",
                "description": "None of the patterns in a case expression matched the value",
                "root_cause": "elixir_case_clause_error",
                "suggestion": "Add a catch-all clause to your case statement using the underscore pattern (_). Alternatively, ensure the value you're matching has the expected structure.",
                "confidence": "high",
                "severity": "medium",
                "category": "guard"
            },
            {
                "id": "elixir_match_error",
                "pattern": "\\*\\* \\(MatchError\\) no match of right hand side value: (.*)",
                "type": "MatchError",
                "description": "Pattern match failed",
                "root_cause": "elixir_match_error",
                "suggestion": "Check the pattern match expression and ensure the data structure matches the expected pattern. Consider using case statements with multiple pattern clauses or adding guard clauses.",
                "confidence": "high",
                "severity": "medium",
                "category": "guard"
            },
            {
                "id": "elixir_bad_map_error",
                "pattern": "\\*\\* \\(BadMapError\\) (.*)",
                "type": "BadMapError",
                "description": "Tried to use a map operation on a value that isn't a map",
                "root_cause": "elixir_bad_map_error",
                "suggestion": "Check the value type before using map operations. Use pattern matching or the `is_map/1` guard function to verify the value is a map.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "elixir_bad_arity_error",
                "pattern": "\\*\\* \\(BadArityError\\) (.*)",
                "type": "BadArityError",
                "description": "Called a function with the wrong number of arguments",
                "root_cause": "elixir_bad_arity_error",
                "suggestion": "Check the function's documentation for the correct number of arguments. Ensure you're passing the expected number of arguments.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "elixir_arithmetic_error",
                "pattern": "\\*\\* \\(ArithmeticError\\) (.*)",
                "type": "ArithmeticError",
                "description": "Error during an arithmetic operation",
                "root_cause": "elixir_arithmetic_error",
                "suggestion": "Check for division by zero or other invalid arithmetic operations. Validate inputs to arithmetic functions.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "elixir_file_error",
                "pattern": "\\*\\* \\(File\\.Error\\) (.*?): (.*)",
                "type": "File.Error",
                "description": "File system operation failed",
                "root_cause": "elixir_file_error",
                "suggestion": "Check file paths and permissions. Ensure the file exists and the application has the necessary permissions. Use File.exists?/1 before trying to open files.",
                "confidence": "high",
                "severity": "medium",
                "category": "io"
            },
            {
                "id": "elixir_protocol_undefined_error",
                "pattern": "\\*\\* \\(Protocol\\.UndefinedError\\) protocol (.*?) not implemented for (.*)",
                "type": "Protocol.UndefinedError",
                "description": "Tried to use a protocol with a data type that doesn't implement it",
                "root_cause": "elixir_protocol_undefined_error",
                "suggestion": "Implement the protocol for the given data type, or ensure you're only using the protocol with supported types. Check the value's type before calling protocol functions.",
                "confidence": "high",
                "severity": "medium",
                "category": "runtime"
            },
            {
                "id": "elixir_enum_empty_error",
                "pattern": "\\*\\* \\(Enum\\.EmptyError\\) (.*)",
                "type": "Enum.EmptyError",
                "description": "Called an Enum function that doesn't work on empty collections",
                "root_cause": "elixir_enum_empty_error",
                "suggestion": "Check if the collection is empty before using functions like Enum.max/1 or Enum.min/1. Use Enum.count/1 > 0 or Enum.empty?/1 to check for emptiness.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            }
        ]


class ElixirPatchGenerator:
    """
    Generates patch solutions for Elixir exceptions.
    
    This class provides capabilities to generate code fixes for common Elixir errors,
    using templates and contextual information about the exception.
    """
    
    def __init__(self):
        """Initialize the Elixir patch generator."""
        self.templates_dir = Path(__file__).parent.parent / "patch_generation" / "templates" / "elixir"
        self.templates_dir.mkdir(exist_ok=True, parents=True)
        
        # Cache for loaded templates
        self.template_cache = {}
    
    def generate_patch(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a patch for an Elixir error based on analysis.
        
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
            "patch_id": f"elixir_{rule_id}",
            "patch_type": "suggestion",
            "language": "elixir",
            "framework": context.get("framework", ""),
            "suggestion": analysis.get("suggestion", "No suggestion available"),
            "confidence": analysis.get("confidence", "low"),
            "severity": analysis.get("severity", "medium"),
            "root_cause": root_cause
        }
        
        # Try to find a specific template for this root cause
        template_name = f"{root_cause}.ex.template"
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
            if root_cause == "elixir_argument_error":
                patch_result["suggestion_code"] = self._generate_argument_error_suggestion(analysis, context)
            elif root_cause == "elixir_function_clause_error":
                patch_result["suggestion_code"] = self._generate_function_clause_suggestion(analysis, context)
            elif root_cause == "elixir_key_error":
                patch_result["suggestion_code"] = self._generate_key_error_suggestion(analysis, context)
            elif root_cause == "elixir_match_error":
                patch_result["suggestion_code"] = self._generate_match_error_suggestion(analysis, context)
            elif root_cause == "elixir_case_clause_error":
                patch_result["suggestion_code"] = self._generate_case_clause_suggestion(analysis, context)
            elif root_cause == "phoenix_forbidden_error":
                patch_result["suggestion_code"] = self._generate_phoenix_suggestion(analysis, context)
            elif root_cause == "ecto_invalid_changeset_error":
                patch_result["suggestion_code"] = self._generate_ecto_suggestion(analysis, context)
        
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
                    variables["MODULE"] = top_frame.get("module", "")
                    variables["FUNCTION"] = top_frame.get("function", "")
                    variables["ARITY"] = top_frame.get("arity", "")
                    variables["FILE"] = top_frame.get("file", "")
                    variables["LINE"] = str(top_frame.get("line", ""))
                    variables["NAMESPACE"] = top_frame.get("namespace", "")
        
        # Extract variables from context
        variables["CODE_SNIPPET"] = context.get("code_snippet", "")
        variables["FUNCTION_ARGS"] = context.get("function_args", "")
        variables["MODULE_IMPORTS"] = context.get("imports", "")
        
        # Extract variables from match groups
        match_groups = analysis.get("match_groups", tuple())
        if match_groups:
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
        
        if "argument_error" in root_cause:
            return ("Validate function arguments before using them. " 
                   f"Consider implementing this fix: {patch_code}")
        elif "function_clause_error" in root_cause:
            return ("Add additional pattern matching clauses to handle all possible input cases. " 
                   f"Implement the suggested fix: {patch_code}")
        elif "key_error" in root_cause:
            return ("Use safe access methods for map and keyword list keys. " 
                   f"Implement the suggested fix: {patch_code}")
        elif "match_error" in root_cause:
            return ("Ensure pattern matching expressions handle all possible value structures. " 
                   f"Consider this implementation: {patch_code}")
        elif "phoenix" in root_cause:
            return ("Update Phoenix controller authentication and authorization checks. " 
                   f"Implement the suggested fix: {patch_code}")
        elif "ecto" in root_cause:
            return ("Validate Ecto changesets before performing database operations. " 
                   f"Use the suggested implementation: {patch_code}")
        else:
            return f"Apply the following fix to address the issue: {patch_code}"
    
    def _generate_argument_error_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for handling argument errors in Elixir."""
        return """# Option 1: Add argument validation using guards
def my_function(arg) when is_binary(arg) do
  # Function body for string arguments
end

def my_function(arg) when is_integer(arg) do
  # Function body for integer arguments
end

def my_function(_) do
  raise ArgumentError, "Expected argument to be a string or integer"
end

# Option 2: Validate arguments at the beginning of the function
def my_function(arg) do
  case arg do
    arg when is_binary(arg) ->
      # Handle string case
    arg when is_integer(arg) ->
      # Handle integer case
    _ ->
      raise ArgumentError, "Expected argument to be a string or integer"
  end
end

# Option 3: Use with to validate arguments
def my_function(arg) do
  with {:ok, arg} <- validate_argument(arg) do
    # Function body with validated argument
  else
    {:error, reason} -> raise ArgumentError, reason
  end
end

defp validate_argument(arg) when is_binary(arg), do: {:ok, arg}
defp validate_argument(arg) when is_integer(arg), do: {:ok, arg}
defp validate_argument(_), do: {:error, "Expected argument to be a string or integer"}
"""
    
    def _generate_function_clause_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for handling function clause errors in Elixir."""
        return """# Option 1: Add a catch-all clause with explicit error
def process_data(data) when is_map(data) do
  # Process map data
end

def process_data(data) when is_list(data) do
  # Process list data
end

def process_data(_data) do
  # Either raise a descriptive error:
  raise ArgumentError, "Expected data to be a map or list"
  
  # Or handle the case gracefully:
  {:error, :invalid_data_type}
end

# Option 2: Pattern match with case
def process_data(data) do
  case data do
    data when is_map(data) ->
      # Process map data
    data when is_list(data) ->
      # Process list data
    _ ->
      {:error, :invalid_data_type}
  end
end

# Option 3: Use typespecs to document expected argument types
@spec process_data(map() | list()) :: {:ok, term()} | {:error, atom()}
def process_data(data)
"""
    
    def _generate_key_error_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for handling key errors in Elixir."""
        return """# Option 1: Use Map.get/3 with a default value
def get_user_name(user) do
  Map.get(user, :name, "Default Name")
end

# Option 2: Pattern match with default using Map.fetch
def get_user_name(user) do
  case Map.fetch(user, :name) do
    {:ok, name} -> name
    :error -> "Default Name"
  end
end

# Option 3: Safe pattern matching with guard
def process_user(%{name: name} = user) when is_binary(name) do
  # Process user with name
end
def process_user(user) when is_map(user) do
  # Process user without name or with invalid name
  user = Map.put_new(user, :name, "Default Name")
  process_user(user)
end

# Option 4: For keyword lists, use Keyword functions
def get_option(options) do
  Keyword.get(options, :name, "Default Name")
end

# Option 5: Use Access syntax with default values
def get_deep_field(data) do
  get_in(data, [:user, :profile, :settings, :theme]) || "default_theme"
end
"""
    
    def _generate_match_error_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for handling match errors in Elixir."""
        return """# Option 1: Use case for more flexible pattern matching
def process_result(result) do
  case result do
    {:ok, value} ->
      # Handle success case
      IO.puts("Success: #{value}")
    {:error, reason} ->
      # Handle error case
      IO.puts("Error: #{reason}")
    unexpected ->
      # Handle unexpected values
      IO.puts("Unexpected result: #{inspect(unexpected)}")
  end
end

# Option 2: Validate structures before matching
def extract_data(data) do
  if is_map(data) and Map.has_key?(data, :user) do
    %{user: user} = data
    user
  else
    # Handle invalid data structure
    {:error, :invalid_data}
  end
end

# Option 3: Use with for multiple matches
def process_user_input(input) do
  with {:ok, data} <- Jason.decode(input),
       %{"user" => user} <- data,
       %{"name" => name, "email" => email} <- user do
    # All patterns matched successfully
    create_user(name, email)
  else
    {:error, %Jason.DecodeError{}} ->
      {:error, :invalid_json}
    %{"user" => _} ->
      {:error, :invalid_user_data}
    _ ->
      {:error, :invalid_input}
  end
end
"""
    
    def _generate_case_clause_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for handling case clause errors in Elixir."""
        return """# Option 1: Add a catch-all clause to case statements
def process_status(status) do
  case status do
    :ok -> "Everything is fine"
    :error -> "An error occurred"
    :pending -> "Operation in progress"
    _ -> "Unknown status: #{inspect(status)}"  # Catch-all clause
  end
end

# Option 2: Validate input before case statement
def process_status(status) when status in [:ok, :error, :pending] do
  case status do
    :ok -> "Everything is fine"
    :error -> "An error occurred"
    :pending -> "Operation in progress"
  end
end
def process_status(status) do
  "Unknown status: #{inspect(status)}"
end

# Option 3: Use cond for more flexible conditions
def check_value(value) do
  cond do
    is_integer(value) and value > 0 -> "Positive integer"
    is_integer(value) and value < 0 -> "Negative integer"
    is_integer(value) and value == 0 -> "Zero"
    is_float(value) -> "Float: #{value}"
    is_binary(value) -> "String: #{value}"
    true -> "Other type: #{inspect(value)}"  # Always matches
  end
end
"""
    
    def _generate_phoenix_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for handling Phoenix authorization errors."""
        return """# Option 1: Add authorization checks in controllers
defmodule MyApp.PostController do
  use MyApp.Web, :controller
  
  plug :authorize_user when action in [:edit, :update, :delete]
  
  # ...controller actions...
  
  defp authorize_user(conn, _) do
    post_id = conn.params["id"]
    post = Repo.get!(Post, post_id)
    
    if post.user_id == conn.assigns.current_user.id do
      conn
    else
      conn
      |> put_flash(:error, "You are not authorized to modify this post")
      |> redirect(to: post_path(conn, :index))
      |> halt()
    end
  end
end

# Option 2: Use a policy module for authorization
defmodule MyApp.Policy do
  def authorize(:show_post, user, post), do: true
  def authorize(:edit_post, user, post), do: user.id == post.user_id
  def authorize(:delete_post, user, post), do: user.id == post.user_id or user.admin?
  def authorize(_, _, _), do: false
end

# In your controller:
def edit(conn, %{"id" => id}) do
  post = Repo.get!(Post, id)
  
  if MyApp.Policy.authorize(:edit_post, conn.assigns.current_user, post) do
    changeset = Post.changeset(post)
    render(conn, "edit.html", post: post, changeset: changeset)
  else
    conn
    |> put_status(:forbidden)
    |> put_view(MyApp.ErrorView)
    |> render("403.html")
  end
end

# Option 3: Use a library like Canada for authorization
defimpl Canada.Can, for: User do
  def can?(%User{id: user_id}, action, %Post{user_id: user_id})
    when action in [:edit, :update, :delete], do: true
  def can?(%User{admin: true}, _, _), do: true
  def can?(_, :show, %Post{}), do: true
  def can?(_, _, _), do: false
end

# Then in your controller:
def edit(conn, %{"id" => id}) do
  post = Repo.get!(Post, id)
  
  if Canada.Can.can?(conn.assigns.current_user, :edit, post) do
    changeset = Post.changeset(post)
    render(conn, "edit.html", post: post, changeset: changeset)
  else
    conn
    |> put_status(:forbidden)
    |> put_view(MyApp.ErrorView)
    |> render("403.html")
  end
end
"""
    
    def _generate_ecto_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for handling Ecto changeset errors."""
        return """# Option 1: Check changeset validity before database operations
def create_user(attrs) do
  %User{}
  |> User.changeset(attrs)
  |> case do
    %{valid?: true} = changeset ->
      Repo.insert(changeset)
    changeset ->
      {:error, changeset}
  end
end

# Option 2: Use with for changeset operations
def update_user(user, attrs) do
  with {:ok, changeset} <- validate_user_attrs(attrs),
       {:ok, updated_user} <- Repo.update(User.changeset(user, changeset)) do
    {:ok, updated_user}
  end
end

defp validate_user_attrs(attrs) do
  types = %{name: :string, email: :string, age: :integer}
  
  {%{}, types}
  |> Ecto.Changeset.cast(attrs, Map.keys(types))
  |> Ecto.Changeset.validate_required([:name, :email])
  |> Ecto.Changeset.validate_format(:email, ~r/@/)
  |> case do
    %{valid?: true} = changeset ->
      {:ok, Ecto.Changeset.apply_changes(changeset)}
    changeset ->
      {:error, changeset}
  end
end

# Option 3: Handle changeset errors in the controller
def create(conn, %{"user" => user_params}) do
  case Accounts.create_user(user_params) do
    {:ok, user} ->
      conn
      |> put_flash(:info, "User created successfully.")
      |> redirect(to: Routes.user_path(conn, :show, user))
    
    {:error, %Ecto.Changeset{} = changeset} ->
      # Render the form again with error messages
      render(conn, "new.html", changeset: changeset)
  end
end

# Option 4: Use insert_or_update with on_conflict option
def upsert_user(attrs) do
  %User{}
  |> User.changeset(attrs)
  |> Repo.insert_or_update(
    on_conflict: {:replace_all_except, [:id, :inserted_at]},
    conflict_target: :email
  )
end
"""


class ElixirLanguagePlugin(LanguagePlugin):
    """
    Elixir language plugin for Homeostasis.
    
    Provides comprehensive error analysis and fix generation for Elixir/Erlang applications,
    including support for Phoenix Framework, Ecto database issues, and OTP supervision
    tree integration.
    """
    
    VERSION = "0.1.0"
    AUTHOR = "Homeostasis Contributors"
    
    def __init__(self):
        """Initialize the Elixir language plugin."""
        self.adapter = ElixirErrorAdapter()
        self.exception_handler = ElixirExceptionHandler()
        self.patch_generator = ElixirPatchGenerator()
    
    def get_language_id(self) -> str:
        """Get the language identifier."""
        return "elixir"
    
    def get_language_name(self) -> str:
        """Get the language name."""
        return "Elixir"
    
    def get_language_version(self) -> str:
        """Get the language version."""
        return "1.10+"
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an Elixir error.
        
        Args:
            error_data: Elixir error data
            
        Returns:
            Analysis results
        """
        # First, normalize the error
        if "language" not in error_data or error_data["language"] != "elixir":
            standard_error = self.normalize_error(error_data)
        else:
            standard_error = error_data
        
        # Use the exception handler to analyze the error
        return self.exception_handler.analyze_exception(standard_error)
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize an Elixir error to the standard format.
        
        Args:
            error_data: Elixir error data
            
        Returns:
            Error data in the standard format
        """
        return self.adapter.to_standard_format(error_data)
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Elixir format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the Elixir format
        """
        return self.adapter.from_standard_format(standard_error)
    
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a fix for an Elixir error.
        
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
        return ["phoenix", "ecto", "plug", "otp", "base"]


# Register this plugin
register_plugin(ElixirLanguagePlugin())