"""
C# Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in C# applications.
It provides error handling for C# exception patterns and supports ASP.NET Core,
Entity Framework, and other .NET frameworks.
"""
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any, List

from ..language_plugin_system import LanguagePlugin, register_plugin
from ..language_adapters import CSharpErrorAdapter

logger = logging.getLogger(__name__)


class CSharpExceptionHandler:
    """
    Handles C# exceptions with pattern-based error detection and classification.
    
    This class provides logic for categorizing C# exceptions based on their type,
    message, and stack trace patterns. It supports both standard .NET exceptions and
    framework-specific exceptions.
    """
    
    def __init__(self):
        """Initialize the C# exception handler."""
        self.rule_categories = {
            "core": "Core C# and .NET exceptions",
            "aspnetcore": "ASP.NET Core framework exceptions",
            "entityframework": "Entity Framework database exceptions",
            "async": "Async/await related exceptions",
            "dependency": "Dependency injection and service resolution exceptions",
            "azure": "Azure services and cloud exceptions",
            "io": "IO and file-related exceptions",
            "network": "Network and service exceptions",
            "concurrency": "Concurrency and threading exceptions",
            "serialization": "Serialization and data conversion exceptions",
            "security": "Security and authentication exceptions"
        }
        
        # Load rules from different categories
        self.rules = self._load_all_rules()
        
        # Initialize caches for performance
        self.pattern_cache = {}  # Compiled regex patterns
        self.rule_match_cache = {}  # Previous rule matches
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a C# exception to determine its root cause and suggest potential fixes.
        
        Args:
            error_data: C# error data in standard format
            
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
                        "root_cause": rule.get("root_cause", "csharp_unknown_error"),
                        "description": rule.get("description", "Unknown C# error"),
                        "suggestion": rule.get("suggestion", "No suggestion available"),
                        "confidence": rule.get("confidence", "medium"),
                        "severity": rule.get("severity", "medium"),
                        "category": rule.get("category", "csharp"),
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
                            
                            trace_lines.append(f"   at {method} in {file_path}:line {line_num}")
                    
                    match_text += "\n" + "\n".join(trace_lines)
        
        return match_text
    
    def _handle_fallback(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle exceptions that didn't match any specific rule.
        
        Args:
            error_data: C# error data in standard format
            
        Returns:
            Fallback analysis result
        """
        error_type = error_data.get("error_type", "")
        message = error_data.get("message", "")
        
        # Try to categorize by common C# exception types
        if "NullReferenceException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "csharp_null_reference",
                "error_type": error_type,
                "root_cause": "csharp_null_reference",
                "description": "Attempted to access a member on a null object reference",
                "suggestion": "Use null-conditional operators (?.), null-coalescing operators (??), or add explicit null checks before accessing properties or methods.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "ArgumentNullException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "csharp_argument_null",
                "error_type": error_type,
                "root_cause": "csharp_argument_null",
                "description": "Null argument provided to a method that doesn't accept it",
                "suggestion": "Check for null before passing arguments to methods, or use defensive programming with guard clauses.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "ArgumentException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "csharp_invalid_argument",
                "error_type": error_type,
                "root_cause": "csharp_invalid_argument",
                "description": "Invalid argument provided to a method",
                "suggestion": "Verify that the arguments you're passing match the expected types and constraints of the method.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "InvalidOperationException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "csharp_invalid_operation",
                "error_type": error_type,
                "root_cause": "csharp_invalid_operation",
                "description": "Method call is invalid for the object's current state",
                "suggestion": "Check the object's state before performing operations. Ensure prerequisites are met before calling methods.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "IndexOutOfRangeException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "csharp_index_out_of_range",
                "error_type": error_type,
                "root_cause": "csharp_index_out_of_range",
                "description": "Attempted to access an element at an index outside the bounds of an array",
                "suggestion": "Check array bounds before accessing elements. Use array.Length to validate indexes.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "KeyNotFoundException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "csharp_key_not_found",
                "error_type": error_type,
                "root_cause": "csharp_key_not_found",
                "description": "Attempted to access a key that doesn't exist in a dictionary",
                "suggestion": "Use TryGetValue or ContainsKey to check if a key exists before accessing it, or use dictionary indexers with null-coalescing operators.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "FormatException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "csharp_format_exception",
                "error_type": error_type,
                "root_cause": "csharp_format_exception",
                "description": "Input string was in an incorrect format for the attempted conversion",
                "suggestion": "Use TryParse methods instead of direct conversion. Validate string format before parsing.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": ""
            }
        elif "DbUpdateException" in error_type or "DbUpdateConcurrencyException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "ef_db_update_exception",
                "error_type": error_type,
                "root_cause": "ef_db_update_failed",
                "description": "Entity Framework failed to execute a database update",
                "suggestion": "Check for constraint violations, concurrency conflicts, or database connectivity issues. Wrap database operations in try-catch blocks.",
                "confidence": "high",
                "severity": "high",
                "category": "entityframework",
                "match_groups": tuple(),
                "framework": "entityframework"
            }
        elif "TaskCanceledException" in error_type or "OperationCanceledException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "csharp_task_canceled",
                "error_type": error_type,
                "root_cause": "csharp_task_canceled",
                "description": "Async operation was canceled via a cancellation token",
                "suggestion": "Check if cancellation is expected or handle the exception appropriately. Ensure proper use of CancellationTokens.",
                "confidence": "high",
                "severity": "medium",
                "category": "async",
                "match_groups": tuple(),
                "framework": ""
            }
        
        # Generic fallback for unknown exceptions
        return {
            "error_data": error_data,
            "rule_id": "csharp_generic_error",
            "error_type": error_type or "Exception",
            "root_cause": "csharp_unknown_error",
            "description": "Unrecognized C# error",
            "suggestion": "Review the exception details and stack trace for more information.",
            "confidence": "low",
            "severity": "medium",
            "category": "core",
            "match_groups": tuple(),
            "framework": ""
        }
    
    def _load_all_rules(self) -> List[Dict[str, Any]]:
        """
        Load C# exception rules from all categories.
        
        Returns:
            Combined list of rule definitions
        """
        all_rules = []
        
        # Core C# exceptions (always included)
        all_rules.extend(self._load_core_csharp_rules())
        
        # Load additional rules from files if available
        rules_dir = Path(__file__).parent.parent / "rules" / "csharp"
        if rules_dir.exists():
            for rule_file in rules_dir.glob("*.json"):
                try:
                    with open(rule_file, 'r') as f:
                        data = json.load(f)
                        all_rules.extend(data.get("rules", []))
                except Exception as e:
                    logger.warning(f"Error loading rules from {rule_file}: {e}")
        
        return all_rules
    
    def _load_core_csharp_rules(self) -> List[Dict[str, Any]]:
        """Load rules for core C# exceptions."""
        return [
            {
                "id": "csharp_null_reference",
                "pattern": "System\\.NullReferenceException: Object reference not set to an instance of an object",
                "type": "System.NullReferenceException",
                "description": "Attempted to access a member on a null object reference",
                "root_cause": "csharp_null_reference",
                "suggestion": "Use null-conditional operators (?.), null-coalescing operators (??), or add explicit null checks before accessing properties or methods.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "csharp_argument_null",
                "pattern": "System\\.ArgumentNullException: Value cannot be null\\. (?:Parameter name: |Arg_ParamName_Name)?([\\w]+)",
                "type": "System.ArgumentNullException",
                "description": "Method received a null argument that doesn't accept null",
                "root_cause": "csharp_argument_null",
                "suggestion": "Ensure that the specified parameter is not null before passing it to methods. Consider using the null-coalescing operator (??) or providing default values.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "csharp_argument_out_of_range",
                "pattern": "System\\.ArgumentOutOfRangeException: (?:Specified argument was out of the range of valid values\\.|Index was out of range\\.) (?:Parameter name: |Arg_ParamName_Name)?([\\w]+)",
                "type": "System.ArgumentOutOfRangeException",
                "description": "Argument provided is outside the range of acceptable values",
                "root_cause": "csharp_argument_out_of_range",
                "suggestion": "Verify that the argument value falls within the expected range. Add validation before calling methods with this parameter.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "csharp_format_exception",
                "pattern": "System\\.FormatException: (?:Input string was not in a correct format|Format of the initialization string does not conform to specification)",
                "type": "System.FormatException",
                "description": "String format is not valid for the attempted conversion",
                "root_cause": "csharp_format_exception",
                "suggestion": "Use TryParse methods instead of direct parsing. Validate string format before conversion.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "csharp_invalid_cast",
                "pattern": "System\\.InvalidCastException: (?:Unable to cast object of type|Specified cast is not valid)",
                "type": "System.InvalidCastException",
                "description": "Invalid type conversion or explicit cast",
                "root_cause": "csharp_invalid_cast",
                "suggestion": "Use 'is' or 'as' operators to safely check and convert types. Verify object types before casting.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "csharp_io_file_not_found",
                "pattern": "System\\.IO\\.FileNotFoundException: Could not find file '([^']+)'",
                "type": "System.IO.FileNotFoundException",
                "description": "Referenced file does not exist at the specified path",
                "root_cause": "csharp_file_not_found",
                "suggestion": "Verify file paths before attempting operations. Use File.Exists() to check if files exist before accessing them.",
                "confidence": "high",
                "severity": "medium",
                "category": "io"
            },
            {
                "id": "csharp_directory_not_found",
                "pattern": "System\\.IO\\.DirectoryNotFoundException: Could not find directory '([^']+)'",
                "type": "System.IO.DirectoryNotFoundException",
                "description": "Referenced directory does not exist at the specified path",
                "root_cause": "csharp_directory_not_found",
                "suggestion": "Verify directory paths before attempting operations. Use Directory.Exists() to check if directories exist before accessing them.",
                "confidence": "high",
                "severity": "medium",
                "category": "io"
            },
            {
                "id": "csharp_unauthorized_access",
                "pattern": "System\\.UnauthorizedAccessException: Access to the path '([^']+)' is denied",
                "type": "System.UnauthorizedAccessException",
                "description": "Application does not have required permissions to access resource",
                "root_cause": "csharp_unauthorized_access",
                "suggestion": "Check file/directory permissions. Run the application with appropriate privileges or request only necessary access rights.",
                "confidence": "high",
                "severity": "high",
                "category": "security"
            },
            {
                "id": "csharp_timeout_exception",
                "pattern": "System\\.TimeoutException: The operation has timed out",
                "type": "System.TimeoutException",
                "description": "Operation did not complete within the allotted time",
                "root_cause": "csharp_operation_timeout",
                "suggestion": "Increase timeout values, optimize the operation, or implement asynchronous processing with longer timeouts.",
                "confidence": "high",
                "severity": "medium",
                "category": "network"
            },
            {
                "id": "csharp_index_out_of_range",
                "pattern": "System\\.IndexOutOfRangeException: Index was outside the bounds of the array",
                "type": "System.IndexOutOfRangeException",
                "description": "Attempted to access an array element with an index outside its bounds",
                "root_cause": "csharp_index_out_of_range",
                "suggestion": "Check array bounds before accessing elements. Use array.Length to validate indexes.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "csharp_key_not_found",
                "pattern": "System\\.Collections\\.Generic\\.KeyNotFoundException: The given key was not present in the dictionary",
                "type": "System.Collections.Generic.KeyNotFoundException",
                "description": "Attempted to retrieve a key that doesn't exist in a dictionary",
                "root_cause": "csharp_key_not_found",
                "suggestion": "Use TryGetValue or ContainsKey to check if a key exists before accessing it.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "csharp_object_disposed",
                "pattern": "System\\.ObjectDisposedException: Cannot access a disposed object",
                "type": "System.ObjectDisposedException",
                "description": "Attempted to access an object that has been disposed",
                "root_cause": "csharp_object_disposed",
                "suggestion": "Check if objects are disposed before using them. Consider restructuring code to ensure proper object lifecycle management.",
                "confidence": "high",
                "severity": "medium",
                "category": "core"
            },
            {
                "id": "csharp_task_canceled",
                "pattern": "System\\.Threading\\.Tasks\\.TaskCanceledException: A task was canceled",
                "type": "System.Threading.Tasks.TaskCanceledException",
                "description": "Async operation was canceled via a cancellation token",
                "root_cause": "csharp_task_canceled",
                "suggestion": "Handle cancellation appropriately. Check if cancellation is expected or provide fallback behavior.",
                "confidence": "high",
                "severity": "medium",
                "category": "async"
            },
            {
                "id": "csharp_aggregate_exception",
                "pattern": "System\\.AggregateException: (?:One or more errors occurred|A Task's exception\\(s\\) were not observed)",
                "type": "System.AggregateException",
                "description": "Multiple exceptions occurred during parallel or async operations",
                "root_cause": "csharp_multiple_errors",
                "suggestion": "Examine InnerExceptions property to identify and handle specific exceptions. Use Task.Wait and ContinueWith with proper exception handling.",
                "confidence": "high",
                "severity": "high",
                "category": "async"
            },
            {
                "id": "aspnet_route_not_found",
                "pattern": "Microsoft\\.AspNetCore\\.Routing\\.EndpointNotFoundException: No route matches the supplied values",
                "type": "Microsoft.AspNetCore.Routing.EndpointNotFoundException",
                "description": "No route found that matches the requested URL pattern",
                "root_cause": "aspnetcore_route_not_found",
                "suggestion": "Check route configuration and URL generation. Ensure routes are correctly defined in Startup.Configure().",
                "confidence": "high",
                "severity": "medium",
                "category": "aspnetcore",
                "framework": "aspnetcore"
            },
            {
                "id": "aspnet_model_validation",
                "pattern": "Microsoft\\.AspNetCore\\.Mvc\\.BadRequestObjectResult: One or more validation errors occurred",
                "type": "Microsoft.AspNetCore.Mvc.BadRequestObjectResult",
                "description": "Model validation failed for an API request",
                "root_cause": "aspnetcore_model_validation",
                "suggestion": "Validate model state in controllers. Add proper validation attributes to model properties.",
                "confidence": "high",
                "severity": "medium",
                "category": "aspnetcore",
                "framework": "aspnetcore"
            },
            {
                "id": "ef_db_update_exception",
                "pattern": "Microsoft\\.EntityFrameworkCore\\.DbUpdateException: (?:An error occurred while updating|Failed executing DbCommand)",
                "type": "Microsoft.EntityFrameworkCore.DbUpdateException",
                "description": "Entity Framework failed to execute a database update",
                "root_cause": "ef_db_update_failed",
                "suggestion": "Look for constraint violations or database connectivity issues. Wrap database operations in try-catch blocks.",
                "confidence": "high",
                "severity": "high",
                "category": "entityframework",
                "framework": "entityframework"
            },
            {
                "id": "ef_db_concurrency_exception",
                "pattern": "Microsoft\\.EntityFrameworkCore\\.DbUpdateConcurrencyException: (?:The database operation was expected to affect|Attempted to update or delete an entity)",
                "type": "Microsoft.EntityFrameworkCore.DbUpdateConcurrencyException",
                "description": "Concurrency conflict during database update",
                "root_cause": "ef_concurrency_conflict",
                "suggestion": "Implement proper concurrency handling with RowVersion/Timestamp properties. Use optimistic concurrency patterns.",
                "confidence": "high",
                "severity": "high",
                "category": "entityframework",
                "framework": "entityframework"
            },
            {
                "id": "di_service_not_registered",
                "pattern": "System\\.InvalidOperationException: No service for type '([^']+)' has been registered",
                "type": "System.InvalidOperationException",
                "description": "Attempted to resolve service not registered in dependency injection container",
                "root_cause": "di_service_not_registered",
                "suggestion": "Register the service in Startup.ConfigureServices() or use appropriate service lifetime.",
                "confidence": "high",
                "severity": "medium",
                "category": "dependency",
                "framework": "aspnetcore"
            },
            {
                "id": "azure_storage_exception",
                "pattern": "Microsoft\\.Azure\\.Storage\\.StorageException: (?:The remote server returned an error|The specified resource does not exist)",
                "type": "Microsoft.Azure.Storage.StorageException",
                "description": "Error accessing Azure Storage services",
                "root_cause": "azure_storage_error",
                "suggestion": "Check Azure Storage connection strings and container/blob names. Verify account access keys and permissions.",
                "confidence": "high",
                "severity": "high",
                "category": "azure",
                "framework": "azure"
            }
        ]


class CSharpPatchGenerator:
    """
    Generates patch solutions for C# exceptions.
    
    This class provides capabilities to generate code fixes for common C# errors,
    using templates and contextual information about the exception.
    """
    
    def __init__(self):
        """Initialize the C# patch generator."""
        self.templates_dir = Path(__file__).parent.parent / "patch_generation" / "templates" / "csharp"
        self.templates_dir.mkdir(exist_ok=True, parents=True)
        
        # Cache for loaded templates
        self.template_cache = {}
    
    def generate_patch(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a patch for a C# error based on analysis.
        
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
            "patch_id": f"csharp_{rule_id}",
            "patch_type": "suggestion",
            "language": "csharp",
            "framework": context.get("framework", ""),
            "suggestion": analysis.get("suggestion", "No suggestion available"),
            "confidence": analysis.get("confidence", "low"),
            "severity": analysis.get("severity", "medium"),
            "root_cause": root_cause
        }
        
        # Try to find a specific template for this root cause
        template_name = f"{root_cause}.cs.template"
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
            if root_cause == "csharp_null_reference":
                patch_result["suggestion_code"] = self._generate_null_check_suggestion(analysis, context)
            elif root_cause == "csharp_argument_null":
                patch_result["suggestion_code"] = self._generate_argument_null_suggestion(analysis, context)
            elif root_cause == "ef_db_update_failed" or root_cause == "ef_concurrency_conflict":
                patch_result["suggestion_code"] = self._generate_ef_exception_suggestion(analysis, context)
            elif root_cause == "csharp_index_out_of_range":
                patch_result["suggestion_code"] = self._generate_index_check_suggestion(analysis, context)
            elif root_cause == "csharp_task_canceled":
                patch_result["suggestion_code"] = self._generate_task_cancellation_suggestion(analysis, context)
        
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
                    variables["NAMESPACE"] = top_frame.get("namespace", "")
                    variables["CLASS"] = top_frame.get("class", "")
        
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
                    "method": top_frame.get("function", ""),
                    "class": top_frame.get("class", "")
                })
        
        return application_point
    
    def _generate_instructions(self, analysis: Dict[str, Any], patch_code: str) -> str:
        """Generate human-readable instructions for applying the patch."""
        root_cause = analysis.get("root_cause", "unknown")
        
        if "null_reference" in root_cause or "argument_null" in root_cause:
            return "Add null checks before accessing objects or method parameters."
        elif "index_out_of_range" in root_cause:
            return "Validate array indexes before accessing elements."
        elif "db_update" in root_cause or "concurrency" in root_cause:
            return "Implement proper error handling for database operations."
        elif "task_canceled" in root_cause:
            return "Handle task cancellation properly in async operations."
        else:
            return "Apply the suggested fix to resolve the error."
    
    def _generate_null_check_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for null checking in C#."""
        return """// Option 1: Use null-conditional operator (?.)
var result = obj?.Property?.Method();

// Option 2: Use null-coalescing operator (??)
var safeObj = obj ?? defaultObj;
var result = safeObj.Property;

// Option 3: Use null check with conditional
if (obj != null && obj.Property != null)
{
    var result = obj.Property.Method();
}

// Option 4: Use C# 8.0+ null forgiving operator (!) if you're sure it's not null
var result = obj!.Property;

// Option 5: Use guard clauses at the beginning of methods
public void ProcessObject(MyClass obj)
{
    if (obj == null)
    {
        throw new ArgumentNullException(nameof(obj));
    }
    
    // Rest of the method can assume obj is not null
    obj.Process();
}
"""
    
    def _generate_argument_null_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for handling ArgumentNullException."""
        param = ""
        if analysis.get("match_groups") and len(analysis.get("match_groups")) > 0:
            param = analysis.get("match_groups")[0]
        
        return f"""// Option 1: Add guard clause at the beginning of the method
public void MyMethod({param} value)
{{
    if (value == null)
    {{
        throw new ArgumentNullException(nameof(value));
    }}
    
    // Process with value safely
}}

// Option 2: Use default value if null is acceptable
public void MyMethod({param} value = null)
{{
    value = value ?? DefaultValue;
    // Process with value safely
}}

// Option 3: Use C# 8.0+ nullable reference types
public void MyMethod({param}? value)
{{
    if (value == null)
    {{
        // Handle null case or return early
        return;
    }}
    
    // Process with value safely
}}

// Option 4: C# 7.0+ with out parameter pattern
public bool TryGetValue(out {param} value)
{{
    value = null;
    // Try to get value
    if (success)
    {{
        value = retrievedValue;
        return true;
    }}
    return false;
}}
"""
    
    def _generate_ef_exception_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for handling Entity Framework exceptions."""
        return """// Option 1: Basic try-catch with logging
try
{
    await _context.SaveChangesAsync();
}
catch (DbUpdateException ex)
{
    // Log the error
    _logger.LogError(ex, "Error saving changes to database");
    
    // Handle specific known issues
    if (ex.InnerException is SqlException sqlEx)
    {
        // Check for constraint violations, deadlocks, etc.
        switch (sqlEx.Number)
        {
            case 2601: // Unique index violation
            case 2627: // Unique constraint violation
                return Problem("A record with the same key already exists.");
            case 547:  // Constraint check violation
                return Problem("The change you requested violates a database constraint.");
            default:
                return Problem("A database error occurred.");
        }
    }
    
    // Generic error handling
    return Problem("Could not save changes to the database.");
}

// Option 2: Concurrency conflict handling
try
{
    await _context.SaveChangesAsync();
}
catch (DbUpdateConcurrencyException ex)
{
    // Get entries with concurrency conflicts
    var entry = ex.Entries.Single();
    
    // Reload current values from database
    await entry.ReloadAsync();
    
    // Option 2A: Keep original values and try again (client wins)
    entry.OriginalValues.SetValues(entry.GetDatabaseValues());
    await _context.SaveChangesAsync();
    
    // Option 2B: Use database values (database wins)
    entry.CurrentValues.SetValues(entry.GetDatabaseValues());
    
    // Option 2C: Show conflict to user
    var databaseValues = entry.GetDatabaseValues();
    var clientValues = entry.CurrentValues;
    // Compare values and show conflict resolution UI
}

// Option 3: Validation before save
foreach (var entry in _context.ChangeTracker.Entries())
{
    if (entry.State == EntityState.Added || entry.State == EntityState.Modified)
    {
        // Perform custom validation
        var entity = entry.Entity;
        if (!IsValid(entity))
        {
            // Handle validation error
            return BadRequest("Validation failed");
        }
    }
}

// Then try to save
try
{
    await _context.SaveChangesAsync();
}
catch (Exception ex)
{
    // Handle exception
}
"""

    def _generate_index_check_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for handling index range issues."""
        return """// Option 1: Check index before accessing
if (index >= 0 && index < array.Length)
{
    var item = array[index];
    // Process item
}
else
{
    // Handle invalid index (return default, throw, etc.)
}

// Option 2: Use LINQ's ElementAtOrDefault (returns default if out of range)
var item = array.ElementAtOrDefault(index);
if (item != null) // or != default if value type
{
    // Process item
}

// Option 3: Use C# 8.0+ index/range syntax with bounds checking
var count = array.Length;
if (count > 0)
{
    var item = array[Math.Min(index, count - 1)];
    // Process the item
}

// Option 4: Try/catch approach (less efficient but sometimes necessary)
try
{
    var item = array[index];
    // Process item
}
catch (IndexOutOfRangeException)
{
    // Handle exception
}
"""

    def _generate_task_cancellation_suggestion(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a code snippet for handling task cancellation."""
        return """// Option 1: Handle cancellation in async method
public async Task<Result> ProcessAsync(CancellationToken cancellationToken = default)
{
    try
    {
        // Use cancellationToken in async calls
        await SomeAsyncOperation(cancellationToken);
        return new SuccessResult();
    }
    catch (OperationCanceledException)
    {
        // Log cancellation
        _logger.LogInformation("Operation was canceled by user");
        return new CanceledResult();
    }
}

// Option 2: Check cancellation before expensive operations
public async Task ProcessLargeDataAsync(IEnumerable<Data> items, CancellationToken cancellationToken)
{
    foreach (var item in items)
    {
        // Check cancellation before each item
        cancellationToken.ThrowIfCancellationRequested();
        
        await ProcessItemAsync(item, cancellationToken);
    }
}

// Option 3: Combine multiple operations with WhenAll and handle cancellation
public async Task<IEnumerable<Result>> ProcessManyAsync(IEnumerable<Item> items, CancellationToken cancellationToken)
{
    try
    {
        var tasks = items.Select(item => ProcessItemAsync(item, cancellationToken));
        return await Task.WhenAll(tasks);
    }
    catch (OperationCanceledException)
    {
        // Handle cancellation of any task
        return Enumerable.Empty<Result>();
    }
}

// Option 4: Set up a timeout with CancellationTokenSource
public async Task<Result> ProcessWithTimeoutAsync()
{
    using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30)))
    {
        try
        {
            return await ProcessAsync(cts.Token);
        }
        catch (OperationCanceledException)
        {
            // Handle timeout
            return new TimeoutResult();
        }
    }
}
"""


class CSharpLanguagePlugin(LanguagePlugin):
    """
    C# language plugin for Homeostasis.
    
    Provides comprehensive error analysis and fix generation for C# applications,
    including support for ASP.NET Core, Entity Framework, and Azure services.
    """
    
    VERSION = "0.1.0"
    AUTHOR = "Homeostasis Contributors"
    
    def __init__(self):
        """Initialize the C# language plugin."""
        self.adapter = CSharpErrorAdapter()
        self.exception_handler = CSharpExceptionHandler()
        self.patch_generator = CSharpPatchGenerator()
    
    def get_language_id(self) -> str:
        """Get the language identifier."""
        return "csharp"
    
    def get_language_name(self) -> str:
        """Get the language name."""
        return "C#"
    
    def get_language_version(self) -> str:
        """Get the language version."""
        return "7.0+"
    
    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a C# error.
        
        Args:
            error_data: C# error data
            
        Returns:
            Analysis results
        """
        # First, normalize the error
        if "language" not in error_data or error_data["language"] != "csharp":
            standard_error = self.normalize_error(error_data)
        else:
            standard_error = error_data
        
        # Use the exception handler to analyze the error
        return self.exception_handler.analyze_error(standard_error)
    
    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a C# error to the standard format.
        
        Args:
            error_data: C# error data
            
        Returns:
            Error data in the standard format
        """
        return self.adapter.to_standard_format(error_data)
    
    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to C# format.
        
        Args:
            standard_error: Error data in the standard format
            
        Returns:
            Error data in the C# format
        """
        return self.adapter.from_standard_format(standard_error)
    
    def generate_fix(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a fix for a C# error.
        
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
        return ["aspnetcore", "entityframework", "azure", "netstandard", "netframework", "netcore"]


# Register this plugin
register_plugin(CSharpLanguagePlugin())