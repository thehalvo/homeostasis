"""
Ruby Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Ruby applications.
It provides error handling for Ruby's exception patterns and supports Ruby frameworks
including Rails, Sinatra, and Rack applications.
"""
import logging
import re
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from ..language_plugin_system import LanguagePlugin, register_plugin
from ..language_adapters import RubyErrorAdapter

logger = logging.getLogger(__name__)


class RubyExceptionHandler:
    """
    Handles Ruby exceptions with pattern-based error detection and classification.
    
    This class provides logic for categorizing Ruby exceptions based on their type,
    message, and backtrace patterns. It supports both standard Ruby exceptions and
    framework-specific exceptions.
    """
    
    def __init__(self):
        """Initialize the Ruby exception handler."""
        self.rule_categories = {
            "core": "Core Ruby exceptions",
            "rails": "Ruby on Rails framework exceptions",
            "activerecord": "ActiveRecord ORM exceptions",
            "sinatra": "Sinatra framework exceptions",
            "rack": "Rack middleware exceptions",
            "gems": "Ruby gem exceptions",
            "concurrency": "Concurrency and threading exceptions",
            "io": "IO and file-related exceptions",
            "network": "Network and service exceptions",
            "syntax": "Ruby syntax errors",
            "metaprogramming": "Metaprogramming-related exceptions"
        }
        
        # Load rules from different categories
        self.rules = self._load_all_rules()
        
        # Initialize caches for performance
        self.pattern_cache = {}  # Compiled regex patterns
        self.rule_match_cache = {}  # Previous rule matches
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Ruby exception to determine its root cause and suggest potential fixes.
        
        Args:
            error_data: Ruby error data in standard format
            
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
                        "root_cause": rule.get("root_cause", "ruby_unknown_error"),
                        "description": rule.get("description", "Unknown Ruby error"),
                        "suggestion": rule.get("suggestion", "No suggestion available"),
                        "confidence": rule.get("confidence", "medium"),
                        "severity": rule.get("severity", "medium"),
                        "category": rule.get("category", "ruby"),
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
                            file_path = frame.get("file", "<unknown>")
                            line_num = frame.get("line", "?")
                            method = frame.get("function", "<unknown>")
                            
                            if method:
                                trace_lines.append(f"{file_path}:{line_num}:in `{method}'")
                            else:
                                trace_lines.append(f"{file_path}:{line_num}")
                    
                    match_text += "\n" + "\n".join(trace_lines)
        
        return match_text
    
    def _handle_fallback(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle exceptions that didn't match any specific rule.
        
        Args:
            error_data: Ruby error data in standard format
            
        Returns:
            Fallback analysis result
        """
        error_type = error_data.get("error_type", "")
        message = error_data.get("message", "")
        
        # Try to categorize by common Ruby exception types
        if "NoMethodError" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "ruby_no_method_error",
                "error_type": error_type,
                "root_cause": "ruby_no_method",
                "description": "Attempted to call a method that doesn't exist, possibly on nil",
                "suggestion": "Check for nil values before method calls. Use try/&./dig or handle nil with a conditional.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "NameError" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "ruby_name_error",
                "error_type": error_type,
                "root_cause": "ruby_undefined_name",
                "description": "Referenced a variable or constant that is not defined",
                "suggestion": "Ensure the variable is defined before use. Check for typos in variable names.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "ArgumentError" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "ruby_argument_error",
                "error_type": error_type,
                "root_cause": "ruby_invalid_argument",
                "description": "Method called with wrong number or type of arguments",
                "suggestion": "Check the method signature and ensure you're providing the correct arguments. Verify argument types.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "TypeError" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "ruby_type_error",
                "error_type": error_type,
                "root_cause": "ruby_type_mismatch",
                "description": "Operation performed on an object of incorrect type",
                "suggestion": "Check the types of objects before performing operations. Use type conversion methods if needed.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "SyntaxError" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "ruby_syntax_error",
                "error_type": error_type,
                "root_cause": "ruby_syntax_error",
                "description": "Ruby code contains syntax errors",
                "suggestion": "Fix the syntax issue. Common issues include missing end keywords, unbalanced brackets, or incorrect indentation.",
                "confidence": "high",
                "severity": "high",
                "category": "syntax",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "LoadError" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "ruby_load_error",
                "error_type": error_type,
                "root_cause": "ruby_missing_file",
                "description": "Failed to load a required file or gem",
                "suggestion": "Ensure the gem is in your Gemfile and has been installed. Check require statements and file paths.",
                "confidence": "high",
                "severity": "high",
                "category": "gems",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "ActiveRecord::RecordNotFound" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "rails_record_not_found",
                "error_type": error_type,
                "root_cause": "rails_record_not_found",
                "description": "Attempted to find a record that doesn't exist",
                "suggestion": "Use find_by or where methods with appropriate error handling. Consider using find_by_id which returns nil instead of raising an exception.",
                "confidence": "high",
                "severity": "medium",
                "category": "activerecord",
                "match_groups": tuple(),
                "framework": "rails"
            }
        
        # Generic fallback for unknown exceptions
        return {
            "error_data": error_data,
            "rule_id": "ruby_generic_error",
            "error_type": error_type or "RuntimeError",
            "root_cause": "ruby_unknown_error",
            "description": f"Unrecognized Ruby error",
            "suggestion": "Review the exception details and backtrace for more information.",
            "confidence": "low",
            "severity": "medium",
            "category": "core",
            "match_groups": tuple(),
            "framework": ""
        }
    
    def _load_all_rules(self) -> List[Dict[str, Any]]:
        """
        Load Ruby exception rules from all categories.
        
        Returns:
            Combined list of rule definitions
        """
        all_rules = []
        
        # Core Ruby exceptions (always included)
        all_rules.extend(self._load_core_ruby_rules())
        
        # Load additional rules from files if available
        rules_dir = Path(__file__).parent.parent / "rules" / "ruby"
        if rules_dir.exists():
            for rule_file in rules_dir.glob("*.json"):
                try:
                    with open(rule_file, 'r') as f:
                        data = json.load(f)
                        all_rules.extend(data.get("rules", []))
                except Exception as e:
                    logger.warning(f"Error loading rules from {rule_file}: {e}")
        
        return all_rules
    
    def _load_core_ruby_rules(self) -> List[Dict[str, Any]]:
        """Load rules for core Ruby exceptions."""
        return [
            {
                "id": "ruby_no_method_error",
                "pattern": "NoMethodError:.*undefined method `([^']+)' for (.*)",
                "type": "NoMethodError",
                "description": "Attempted to call a method that doesn't exist",
                "root_cause": "ruby_no_method",
                "suggestion": "Check for nil values before method calls. Use the safe navigation operator (&.) or respond_to? to prevent this error.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "ruby_nil_reference",
                "pattern": "NoMethodError:.*undefined method `([^']+)' for nil:NilClass",
                "type": "NoMethodError",
                "description": "Attempted to call a method on nil",
                "root_cause": "ruby_nil_reference",
                "suggestion": "Add nil check before calling the method. Consider using the safe navigation operator (&.) or try() in Rails.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "ruby_name_error",
                "pattern": "NameError: undefined local variable or method `([^']+)'",
                "type": "NameError",
                "description": "Referenced a variable or method that is not defined",
                "root_cause": "ruby_undefined_name",
                "suggestion": "Ensure the variable is defined before use. Check for typos in variable names.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "ruby_uninitialized_constant",
                "pattern": "NameError: uninitialized constant ([A-Z][A-Za-z0-9:]*)",
                "type": "NameError",
                "description": "Referenced a constant that is not defined",
                "root_cause": "ruby_undefined_constant",
                "suggestion": "Ensure the class or module is properly required/imported. Check namespace and spelling.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "ruby_argument_error",
                "pattern": "ArgumentError: (wrong number of arguments|given \\d+, expected \\d+)",
                "type": "ArgumentError",
                "description": "Method called with wrong number of arguments",
                "root_cause": "ruby_wrong_arguments",
                "suggestion": "Check the method signature and ensure you're providing the correct number of arguments.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "ruby_type_error",
                "pattern": "TypeError: (no implicit conversion of|can't convert) ([A-Za-z0-9:]+) (into|to) ([A-Za-z0-9:]+)",
                "type": "TypeError",
                "description": "Type conversion error",
                "root_cause": "ruby_type_conversion",
                "suggestion": "Ensure objects are of the correct type before operations. Use explicit type conversion methods like to_s, to_i, etc.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "ruby_load_error",
                "pattern": "LoadError: cannot load such file -- ([^\\s]+)",
                "type": "LoadError",
                "description": "Failed to load a required file",
                "root_cause": "ruby_missing_file",
                "suggestion": "Ensure the file path is correct and the file exists. Check your load path settings.",
                "confidence": "high",
                "severity": "high",
                "category": "core"
            },
            {
                "id": "ruby_gem_load_error",
                "pattern": "LoadError: cannot load such file -- ([^\\s/]+)(?:/|$)",
                "type": "LoadError",
                "description": "Failed to load a gem",
                "root_cause": "ruby_missing_gem",
                "suggestion": "Ensure the gem is in your Gemfile and has been installed. Run 'bundle install' to update dependencies.",
                "confidence": "high",
                "severity": "high",
                "category": "gems"
            },
            {
                "id": "ruby_syntax_error",
                "pattern": "SyntaxError: (.*)",
                "type": "SyntaxError",
                "description": "Ruby code contains syntax errors",
                "root_cause": "ruby_syntax_error",
                "suggestion": "Fix the syntax issue. Common issues include missing end keywords, unbalanced brackets, or incorrect indentation.",
                "confidence": "high",
                "severity": "high",
                "category": "syntax"
            },
            {
                "id": "ruby_stack_level_too_deep",
                "pattern": "SystemStackError: stack level too deep",
                "type": "SystemStackError",
                "description": "Stack overflow due to infinite recursion",
                "root_cause": "ruby_stack_overflow",
                "suggestion": "Check for infinite recursion in your code. Ensure recursive methods have proper termination conditions.",
                "confidence": "high",
                "severity": "high",
                "category": "core"
            },
            {
                "id": "ruby_timeout_error",
                "pattern": "Timeout::Error",
                "type": "Timeout::Error",
                "description": "Operation timed out",
                "root_cause": "ruby_timeout",
                "suggestion": "Increase timeout value or optimize the operation. Consider using background processing for long-running tasks.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "ruby_file_not_found",
                "pattern": "Errno::ENOENT: No such file or directory(?: @ \\w+ - |: )(.+)",
                "type": "Errno::ENOENT",
                "description": "Attempted to access a file that doesn't exist",
                "root_cause": "ruby_file_not_found",
                "suggestion": "Check file paths and ensure the file exists before accessing. Use File.exist? to verify file existence.",
                "confidence": "high",
                "severity": "medium",
                "category": "io"
            },
            {
                "id": "ruby_permission_denied",
                "pattern": "Errno::EACCES: Permission denied(?: @ \\w+ - |: )(.+)",
                "type": "Errno::EACCES",
                "description": "Insufficient permissions to access file or directory",
                "root_cause": "ruby_permission_denied",
                "suggestion": "Check file permissions and ensure your application has appropriate access rights.",
                "confidence": "high",
                "severity": "medium",
                "category": "io"
            },
            {
                "id": "ruby_thread_error",
                "pattern": "ThreadError: (.+)",
                "type": "ThreadError",
                "description": "Error in thread operation",
                "root_cause": "ruby_thread_error",
                "suggestion": "Review your thread management code. Ensure proper synchronization between threads.",
                "confidence": "high",
                "severity": "medium",
                "category": "concurrency"
            },
            {
                "id": "ruby_deadlock_detected",
                "pattern": "ThreadError: deadlock detected(?:.+)?",
                "type": "ThreadError",
                "description": "Deadlock between threads detected",
                "root_cause": "ruby_deadlock",
                "suggestion": "Review your synchronization strategy. Ensure resources are locked and unlocked in the correct order to prevent circular dependencies.",
                "confidence": "high",
                "severity": "high",
                "category": "concurrency"
            },
            {
                "id": "ruby_connection_refused",
                "pattern": "Errno::ECONNREFUSED: Connection refused",
                "type": "Errno::ECONNREFUSED",
                "description": "Failed to establish a connection",
                "root_cause": "ruby_connection_refused",
                "suggestion": "Ensure the service you're connecting to is running and accessible. Check hostnames, ports, and network configuration.",
                "confidence": "high",
                "severity": "high",
                "category": "network"
            },
            # Rails-specific rules
            {
                "id": "rails_record_not_found",
                "pattern": "ActiveRecord::RecordNotFound: Couldn't find ([A-Za-z:]+) with( '?\\w+'? = |ID(\\s+or\\s+\\w+ID)?\\s+).+",
                "type": "ActiveRecord::RecordNotFound",
                "description": "Attempted to find a record that doesn't exist",
                "root_cause": "rails_record_not_found",
                "suggestion": "Use find_by or where methods with appropriate error handling. Consider using find_by_id which returns nil instead of raising an exception.",
                "confidence": "high",
                "severity": "medium",
                "category": "activerecord",
                "framework": "rails"
            },
            {
                "id": "rails_record_invalid",
                "pattern": "ActiveRecord::RecordInvalid: Validation failed: (.+)",
                "type": "ActiveRecord::RecordInvalid",
                "description": "Record validation failed during save/create!",
                "root_cause": "rails_record_invalid",
                "suggestion": "Check your model validations. Validate user input before attempting to save. Use save instead of save! or handle the exception.",
                "confidence": "high",
                "severity": "medium",
                "category": "activerecord",
                "framework": "rails"
            },
            {
                "id": "rails_routing_error",
                "pattern": "ActionController::RoutingError: No route matches \\[([A-Z]+)\\] \"(.+)\"",
                "type": "ActionController::RoutingError",
                "description": "No route found for the requested URL",
                "root_cause": "rails_routing_error",
                "suggestion": "Check your routes configuration. Ensure the path and HTTP method are correctly defined in config/routes.rb.",
                "confidence": "high",
                "severity": "medium",
                "category": "rails",
                "framework": "rails"
            },
            {
                "id": "rails_unknown_format",
                "pattern": "ActionController::UnknownFormat",
                "type": "ActionController::UnknownFormat",
                "description": "Controller action doesn't support the requested format",
                "root_cause": "rails_unknown_format",
                "suggestion": "Add respond_to block for the requested format. Check content negotiation in your controller actions.",
                "confidence": "high",
                "severity": "medium",
                "category": "rails",
                "framework": "rails"
            },
            {
                "id": "rails_param_missing",
                "pattern": "ActionController::ParameterMissing: param(eter)? (required|is missing|not found|missing): (.+)",
                "type": "ActionController::ParameterMissing",
                "description": "Required parameter is missing from the request",
                "root_cause": "rails_parameter_missing",
                "suggestion": "Ensure required parameters are provided in the request. Check your strong parameters configuration.",
                "confidence": "high",
                "severity": "medium",
                "category": "rails",
                "framework": "rails"
            },
            {
                "id": "rails_template_error",
                "pattern": "ActionView::Template(Error|SyntaxError): (.+)",
                "type": "ActionView::TemplateError",
                "description": "Error in view template",
                "root_cause": "rails_template_error",
                "suggestion": "Check the view template for syntax errors. Ensure variables used in the template are properly initialized.",
                "confidence": "high",
                "severity": "medium",
                "category": "rails",
                "framework": "rails"
            },
            {
                "id": "rails_missing_template",
                "pattern": "ActionView::MissingTemplate: Missing template (.+)",
                "type": "ActionView::MissingTemplate",
                "description": "View template not found",
                "root_cause": "rails_missing_template",
                "suggestion": "Ensure the template file exists in the correct path. Check controller render calls for typos in template names.",
                "confidence": "high",
                "severity": "medium",
                "category": "rails",
                "framework": "rails"
            },
            {
                "id": "rails_connection_pool_timeout",
                "pattern": "ActiveRecord::ConnectionTimeoutError: could not obtain a (?:database )?connection within (\\d+\\.?\\d*) seconds",
                "type": "ActiveRecord::ConnectionTimeoutError",
                "description": "Database connection pool exhausted",
                "root_cause": "rails_db_connection_pool_timeout",
                "suggestion": "Increase the connection pool size in database.yml. Ensure connections are properly released after use.",
                "confidence": "high",
                "severity": "high",
                "category": "activerecord",
                "framework": "rails"
            },
            {
                "id": "sinatra_not_found",
                "pattern": "Sinatra::NotFound",
                "type": "Sinatra::NotFound",
                "description": "No route matched the request",
                "root_cause": "sinatra_not_found",
                "suggestion": "Ensure the route is correctly defined in your Sinatra application. Check HTTP methods and path patterns.",
                "confidence": "high",
                "severity": "medium",
                "category": "sinatra",
                "framework": "sinatra"
            },
            {
                "id": "metaprogramming_method_missing",
                "pattern": "NoMethodError:.*undefined method `([^']+)'.*\\(method_missing\\)",
                "type": "NoMethodError",
                "description": "method_missing handler failed or threw an error",
                "root_cause": "ruby_method_missing",
                "suggestion": "Check your method_missing implementation. Ensure it correctly handles or delegates the method call.",
                "confidence": "medium",
                "severity": "medium",
                "category": "metaprogramming"
            },
            {
                "id": "const_missing_error",
                "pattern": "NameError:.*\\(const_missing\\)",
                "type": "NameError",
                "description": "const_missing handler failed or threw an error",
                "root_cause": "ruby_const_missing",
                "suggestion": "Check your const_missing implementation. Ensure it correctly loads or resolves the constant.",
                "confidence": "medium",
                "severity": "medium",
                "category": "metaprogramming"
            }
        ]


class RubyPatchGenerator:
    """
    Generates patch solutions for Ruby exceptions.
    
    This class provides capabilities to generate code fixes for common Ruby errors,
    using templates and contextual information about the exception.
    """
    
    def __init__(self):
        """Initialize the Ruby patch generator."""
        self.templates_dir = Path(__file__).parent.parent / "patch_generation" / "templates" / "ruby"
        self.templates_dir.mkdir(exist_ok=True, parents=True)
        
        # Cache for loaded templates
        self.template_cache = {}
    
    def generate_patch(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a patch for a Ruby error based on analysis.
        
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
            "patch_id": f"ruby_{rule_id}",
            "patch_type": "suggestion",
            "language": "ruby",
            "framework": context.get("framework", ""),
            "suggestion": analysis.get("suggestion", "No suggestion available"),
            "confidence": analysis.get("confidence", "low"),
            "severity": analysis.get("severity", "medium"),
            "root_cause": root_cause
        }
        
        # Try to find a specific template for this root cause
        template_name = f"{root_cause}.rb.template"
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
            if root_cause == "ruby_nil_reference":
                patch_result["suggestion_code"] = self._generate_nil_check_suggestion(analysis, context)
            elif root_cause == "ruby_record_not_found":
                patch_result["suggestion_code"] = self._generate_record_not_found_suggestion(analysis, context)
            elif root_cause == "ruby_load_error" or root_cause == "ruby_missing_gem":
                patch_result["suggestion_code"] = self._generate_load_error_suggestion(analysis, context)
            elif root_cause == "ruby_wrong_arguments":
                patch_result["suggestion_code"] = self._generate_argument_error_suggestion(analysis, context)
        
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
                    variables["MODULE"] = top_frame.get("module", "")
        
        # Extract variables from context
        variables["CODE_SNIPPET"] = context.get("code_snippet", "")
        
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
                    "method": top_frame.get("function", "")
                })
        
        return application_point
    
    def _generate_instructions(self, analysis: Dict[str, Any], patch_code: str) -> str:
        """Generate human-readable instructions for applying the patch."""
        root_cause = analysis.get("root_cause", "unknown")
        
        if "nil_reference" in root_cause:
            return "Add nil checks before accessing objects."
        elif "record_not_found" in root_cause:
            return "Use find_by instead of find to avoid exceptions for missing records."
        elif "missing_gem" in root_cause:
            return "Add the missing gem to your Gemfile and run bundle install."
        elif "wrong_arguments" in root_cause:
            return "Check the method signature and provide the correct arguments."
        else:
            return f"Apply the suggested fix to resolve the error."
    
    def _generate_nil_check_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for nil checking in Ruby."""
        method = ""
        if analysis.get("match_groups"):
            method = analysis.get("match_groups")[0]
        
        return f"""# Option 1: Use safe navigation operator (Ruby 2.3+)
object&.{method}

# Option 2: Use try method (Rails)
object.try(:{method})

# Option 3: Use conditional
if object
  object.{method}
else
  # Handle nil case
  default_value
end

# Option 4: Use Ruby 2.3+ dig method for nested hash/array access
params.dig(:user, :name) # Instead of params[:user][:name]
"""
    
    def _generate_record_not_found_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for handling ActiveRecord::RecordNotFound."""
        model = ""
        if analysis.get("match_groups"):
            model = analysis.get("match_groups")[0]
        
        return f"""# Option 1: Use find_by instead of find (returns nil instead of raising exception)
{model}.find_by(id: params[:id])

# Option 2: Add error handling
begin
  @{model.lower()} = {model}.find(params[:id])
rescue ActiveRecord::RecordNotFound
  # Handle the error, e.g. redirect or render a 404 page
  flash[:error] = "{model} not found"
  redirect_to {model.lower()}s_path
end

# Option 3: Use where + first (returns nil for no results)
@{model.lower()} = {model}.where(id: params[:id]).first
"""
    
    def _generate_load_error_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for handling LoadError."""
        gem_name = ""
        if analysis.get("match_groups"):
            gem_name = analysis.get("match_groups")[0]
        
        return f"""# Add the gem to your Gemfile
gem '{gem_name}'

# Then run:
# $ bundle install

# For a specific version:
gem '{gem_name}', '~> x.y.z'

# Alternative: Install the gem directly
# $ gem install {gem_name}

# If it's a local file, check the require path
require_relative 'path/to/{gem_name}'
"""
    
    def _generate_argument_error_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for handling ArgumentError."""
        return """# Check the method signature and ensure correct arguments
# Example method definition:
def method_name(required_arg, optional_arg = default_value, keyword_arg: nil)
  # method body
end

# Call with correct arguments:
method_name(required_arg, optional_arg, keyword_arg: value)

# For unknown method signatures, use keyword arguments with defaults:
def method_name(**options)
  defaults = { key1: 'default1', key2: 'default2' }
  options = defaults.merge(options)
  # method body
end
"""


class RubyLanguagePlugin(LanguagePlugin):
    """
    Ruby language plugin for Homeostasis.
    
    Provides comprehensive error analysis and fix generation for Ruby applications,
    including support for Rails, Sinatra, and other Ruby frameworks.
    """
    
    VERSION = "0.1.0"
    AUTHOR = "Homeostasis Contributors"
    
    def __init__(self):
        """Initialize the Ruby language plugin."""
        self.adapter = RubyErrorAdapter()
        self.exception_handler = RubyExceptionHandler()
        self.patch_generator = RubyPatchGenerator()
    
    def get_language_id(self) -> str:
        """Get the language identifier."""
        return "ruby"
    
    def get_language_name(self) -> str:
        """Get the language name."""
        return "Ruby"
    
    def get_language_version(self) -> str:
        """Get the language version."""
        return "2.5+"
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Ruby error.
        
        Args:
            error_data: Ruby error data
            
        Returns:
            Analysis results
        """
        # First, normalize the error
        if "language" not in error_data or error_data["language"] != "ruby":
            standard_error = self.normalize_error(error_data)
        else:
            standard_error = error_data
        
        # Use the exception handler to analyze the error
        return self.exception_handler.analyze_error(standard_error)
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a Ruby error to the standard format.
        
        Args:
            error_data: Ruby error data
            
        Returns:
            Error data in the standard format
        """
        return self.adapter.to_standard_format(error_data)
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Ruby format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the Ruby format
        """
        return self.adapter.from_standard_format(standard_error)
    
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a fix for a Ruby error.
        
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
        return ["rails", "sinatra", "rack", "base"]


# Register this plugin
register_plugin(RubyLanguagePlugin())