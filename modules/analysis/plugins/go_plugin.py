"""
Go Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Go applications.
It provides error handling for Go's common error patterns and supports Go modules,
goroutine management, and popular Go frameworks like Gin and Echo.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..language_adapters import GoErrorAdapter
from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class GoErrorHandler:
    """
    Handles Go errors with pattern-based error detection and classification.

    This class provides logic for categorizing Go errors based on their type,
    message, and stack trace patterns. It supports both standard Go errors and
    framework-specific errors.
    """

    def __init__(self):
        """Initialize the Go error handler."""
        self.rule_categories = {
            "core": "Core Go errors",
            "runtime": "Go runtime errors",
            "goroutine": "Goroutine and concurrency errors",
            "modules": "Go modules and dependency errors",
            "network": "Network and I/O errors",
            "database": "Database and SQL errors",
            "gin": "Gin framework errors",
            "echo": "Echo framework errors",
            "json": "JSON processing errors",
            "xml": "XML processing errors",
            "http": "HTTP client/server errors",
        }

        # Load rules from different categories
        self.rules = self._load_all_rules()

        # Initialize caches for performance
        self.pattern_cache = {}  # Compiled regex patterns
        self.rule_match_cache = {}  # Previous rule matches

    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Go exception to determine its root cause and potential fixes.

        This is an alias for analyze_error to match the expected interface.

        Args:
            error_data: Go error data in standard format

        Returns:
            Analysis results with categorization and fix suggestions
        """
        return self.analyze_error(error_data)

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Go error to determine its root cause and suggest potential fixes.

        Args:
            error_data: Go error data in standard format

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
                        "root_cause": rule.get("root_cause", "go_unknown_error"),
                        "category": rule.get("category", "go"),
                        "subcategory": rule.get(
                            "subcategory",
                            error_type.lower() if error_type else "unknown",
                        ),
                        "error_type": rule.get("type", error_type),
                        "description": rule.get("description", "Unknown Go error"),
                        "suggestion": rule.get("suggestion", "No suggestion available"),
                        "confidence": rule.get("confidence", "medium"),
                        "severity": rule.get("severity", "medium"),
                        "rule_id": rule.get("id", "unknown"),
                        "match_groups": match.groups() if match.groups() else tuple(),
                        "framework": error_data.get(
                            "framework", rule.get("framework", "")
                        ),
                        "tags": rule.get("tags", []),
                        "file_path": error_data.get("file", ""),
                        "line_number": error_data.get("line", 0),
                        "column_number": error_data.get("column", 0),
                    }

                    # Cache the result for this error signature
                    error_signature = f"{error_type}:{message[:100]}"
                    self.rule_match_cache[error_signature] = result

                    return result
            except Exception as e:
                logger.warning(f"Error applying rule {rule.get('id', 'unknown')}: {e}")

        # Check for framework-specific errors before general fallback
        if error_type == "BindingError" or "validation" in message.lower():
            return {
                "error_data": error_data,
                "rule_id": "go_validation_error",
                "error_type": error_type or "validation error",
                "root_cause": "go_validation_error",
                "description": "Field validation failed",
                "suggestion": "Check the validation rules for the failing field. Ensure the input data meets all validation constraints.",
                "confidence": "high",
                "severity": "medium",
                "category": "validation",
                "subcategory": "binding",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
                "tags": ["validation", "binding"],
            }

        # If no rule matched, try the fallback handlers
        return self._handle_fallback(error_data)

    def _create_match_text(
        self, error_type: str, message: str, stack_trace: List
    ) -> str:
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
                            package = frame.get("package", "")
                            func_name = frame.get("function", "")
                            file_path = frame.get("file", "")
                            line_num = frame.get("line", "")

                            func_full = (
                                f"{package}.{func_name}" if package else func_name
                            )
                            trace_lines.append(f"{func_full}()")
                            trace_lines.append(f"\t{file_path}:{line_num}")

                    if trace_lines:
                        match_text += "\n" + "\n".join(trace_lines)

        return match_text

    def _handle_fallback(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors that didn't match any specific rule.

        Args:
            error_data: Go error data in standard format

        Returns:
            Fallback analysis result
        """
        error_type = error_data.get("error_type", "")
        message = error_data.get("message", "")

        # Handle special error types first
        if error_type == "MemoryLeak":
            return {
                "error_data": error_data,
                "rule_id": "go_memory_leak",
                "error_type": error_type,
                "root_cause": "go_memory_leak",
                "description": "Memory leak detected in Go application",
                "suggestion": "Use Go's built-in profiling tools (pprof) to analyze memory allocation patterns. Check for unbounded growth of slices, maps, or goroutines.",
                "confidence": "high",
                "severity": "high",
                "category": "performance",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
            }
        elif error_type == "SecurityVulnerability" and (
            "SQL" in message or "injection" in message
        ):
            return {
                "error_data": error_data,
                "rule_id": "go_sql_injection",
                "error_type": error_type,
                "root_cause": "go_sql_injection_vulnerability",
                "description": "Potential SQL injection vulnerability",
                "suggestion": "Use prepared statements or parameterized queries to prevent SQL injection. Never concatenate user input directly into SQL queries.",
                "confidence": "high",
                "severity": "critical",
                "category": "security",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
            }
        elif error_type == "InfiniteLoop":
            return {
                "error_data": error_data,
                "rule_id": "go_infinite_loop",
                "error_type": error_type,
                "root_cause": "go_infinite_loop",
                "description": "Potential infinite loop detected",
                "suggestion": "Add proper loop termination conditions. Consider using context for cancellation. Add timeout mechanisms to prevent infinite execution.",
                "confidence": "high",
                "severity": "high",
                "category": "performance",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
            }

        # Handle common Go error scenarios based on message patterns
        if "nil pointer dereference" in message or "invalid memory address" in message:
            return {
                "root_cause": "go_nil_pointer_dereference",
                "category": "runtime",
                "subcategory": "panic",
                "error_type": error_type or "runtime error",
                "description": "Attempted to dereference a nil pointer",
                "suggestion": "Add nil checks before accessing pointers. Use safe accessor patterns.",
                "confidence": "high",
                "severity": "high",
                "rule_id": "go_nil_pointer",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
                "tags": ["go", "runtime", "panic", "nil"],
                "file_path": error_data.get("file", ""),
                "line_number": error_data.get("line", 0),
                "column_number": error_data.get("column", 0),
            }
        elif "index out of range" in message:
            return {
                "error_data": error_data,
                "rule_id": "go_index_out_of_range",
                "error_type": error_type or "runtime error",
                "root_cause": "go_index_out_of_range",
                "description": "Array or slice index out of bounds",
                "suggestion": "Add bounds checking before accessing arrays or slices. Use len() to verify index is valid.",
                "confidence": "high",
                "severity": "medium",
                "category": "runtime",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
            }
        elif "assignment to entry in nil map" in message:
            return {
                "error_data": error_data,
                "rule_id": "go_nil_map",
                "error_type": error_type or "runtime error",
                "root_cause": "go_nil_map",
                "description": "Attempted to write to a nil map",
                "suggestion": "Initialize maps with make(map[KeyType]ValueType) before use.",
                "confidence": "high",
                "severity": "medium",
                "category": "runtime",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
            }
        elif "concurrent map" in message and ("read" in message or "write" in message):
            return {
                "error_data": error_data,
                "rule_id": "go_concurrent_map",
                "error_type": error_type or "runtime error",
                "root_cause": "go_concurrent_map_access",
                "description": "Concurrent map access detected",
                "suggestion": "Use sync.RWMutex to protect map access in concurrent code or use sync.Map for concurrent access.",
                "confidence": "high",
                "severity": "high",
                "category": "concurrency",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
            }
        elif "deadlock" in message.lower():
            return {
                "error_data": error_data,
                "rule_id": "go_deadlock",
                "error_type": error_type or "fatal error",
                "root_cause": "go_deadlock",
                "description": "Goroutine deadlock detected",
                "suggestion": "Review mutex acquisition order, check for missing unlock calls, or ensure channels have enough buffer space.",
                "confidence": "high",
                "severity": "critical",
                "category": "concurrency",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
            }
        elif "json" in error_type.lower() or (
            "unmarshal" in message.lower() and "json" in message.lower()
        ):
            return {
                "error_data": error_data,
                "rule_id": "go_json_error",
                "error_type": error_type or "json.SyntaxError",
                "root_cause": "go_json_error",
                "description": "JSON parsing or encoding error",
                "suggestion": "Validate JSON structure and types. Check for missing struct tags or incompatible types.",
                "confidence": "medium",
                "severity": "medium",
                "category": "json",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
            }

        # Generic fallback for unknown errors
        return {
            "error_data": error_data,
            "rule_id": "go_generic_error",
            "error_type": error_type or "error",
            "root_cause": "go_unknown_error",
            "description": "Unrecognized Go error",
            "suggestion": "Review error message and stack trace for more details.",
            "confidence": "low",
            "severity": "medium",
            "category": "go",
            "match_groups": tuple(),
            "framework": error_data.get("framework", ""),
        }

    def _load_all_rules(self) -> List[Dict[str, Any]]:
        """
        Load Go error rules from all categories.

        Returns:
            Combined list of rule definitions
        """
        all_rules = []

        # Core Go errors (always included)
        all_rules.extend(self._load_core_go_rules())

        # Load additional rules from files if available
        rules_dir = Path(__file__).parent.parent / "rules" / "go"
        if rules_dir.exists():
            for rule_file in rules_dir.glob("*.json"):
                try:
                    with open(rule_file, "r") as f:
                        data = json.load(f)
                        all_rules.extend(data.get("rules", []))
                except Exception as e:
                    logger.warning(f"Error loading rules from {rule_file}: {e}")

        return all_rules

    def _load_core_go_rules(self) -> List[Dict[str, Any]]:
        """Load rules for core Go errors."""
        return [
            {
                "id": "go_nil_pointer",
                "pattern": "(?:nil pointer dereference|invalid memory address)",
                "type": "runtime error",
                "description": "Attempted to dereference a nil pointer",
                "root_cause": "go_nil_pointer_dereference",
                "suggestion": "Add nil checks before accessing pointers. Use safe accessor patterns.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime",
            },
            {
                "id": "go_index_out_of_range",
                "pattern": "index out of range \\[(\\d+)\\] with length (\\d+)",
                "type": "runtime error",
                "description": "Array or slice index out of bounds",
                "root_cause": "go_index_out_of_range",
                "suggestion": "Add bounds checking before accessing arrays or slices. Use len() to verify index is valid.",
                "confidence": "high",
                "severity": "medium",
                "category": "runtime",
            },
            {
                "id": "go_nil_map",
                "pattern": "assignment to entry in nil map",
                "type": "runtime error",
                "description": "Attempted to write to a nil map",
                "root_cause": "go_nil_map",
                "suggestion": "Initialize maps with make(map[KeyType]ValueType) before use.",
                "confidence": "high",
                "severity": "medium",
                "category": "runtime",
            },
            {
                "id": "go_concurrent_map_write",
                "pattern": "concurrent map writes",
                "type": "runtime error",
                "description": "Multiple goroutines writing to a map concurrently",
                "root_cause": "go_concurrent_map_write",
                "suggestion": "Use sync.Mutex to protect map access in concurrent code or use sync.Map for concurrent access.",
                "confidence": "high",
                "severity": "high",
                "category": "concurrency",
            },
            {
                "id": "go_concurrent_map_read_write",
                "pattern": "concurrent map read and map write",
                "type": "runtime error",
                "description": "Goroutines reading and writing a map concurrently",
                "root_cause": "go_concurrent_map_access",
                "suggestion": "Use sync.RWMutex to protect map access in concurrent code or use sync.Map for concurrent access.",
                "confidence": "high",
                "severity": "high",
                "category": "concurrency",
            },
            {
                "id": "go_all_goroutines_asleep",
                "pattern": "all goroutines are asleep - deadlock",
                "type": "fatal error",
                "description": "All goroutines are blocked waiting - deadlock detected",
                "root_cause": "go_channel_deadlock",
                "suggestion": "Check for channel operations that are blocking without a sender/receiver, or missing unlock operations.",
                "confidence": "high",
                "severity": "critical",
                "category": "concurrency",
            },
            {
                "id": "go_slice_bounds",
                "pattern": "slice bounds out of range",
                "type": "runtime error",
                "description": "Attempted to create a slice with invalid bounds",
                "root_cause": "go_slice_bounds",
                "suggestion": "Ensure slice bounds are within the valid range (0 <= low <= high <= cap).",
                "confidence": "high",
                "severity": "medium",
                "category": "runtime",
            },
            {
                "id": "go_divide_by_zero",
                "pattern": "(?:divide by zero|division by zero)",
                "type": "runtime error",
                "description": "Division by zero error",
                "root_cause": "go_divide_by_zero",
                "suggestion": "Add checks to prevent division by zero. Validate denominators before division operations.",
                "confidence": "high",
                "severity": "medium",
                "category": "runtime",
            },
            {
                "id": "go_type_assertion",
                "pattern": "interface conversion: interface \\{.*\\} is (\\w+), not (\\w+)",
                "type": "runtime error",
                "description": "Type assertion failed",
                "root_cause": "go_type_assertion_failure",
                "suggestion": "Use type assertion with ok idiom (v, ok := x.(Type)) or type switch to safely handle interface conversions.",
                "confidence": "high",
                "severity": "medium",
                "category": "runtime",
            },
            {
                "id": "go_json_syntax",
                "pattern": "(?:invalid character|unexpected end of JSON input)",
                "type": "json.SyntaxError",
                "description": "Invalid JSON syntax",
                "root_cause": "go_json_syntax",
                "suggestion": "Validate JSON input format. Check quotes, braces, and commas.",
                "confidence": "high",
                "severity": "medium",
                "category": "json",
            },
            {
                "id": "go_json_unmarshal_type",
                "pattern": "cannot unmarshal (\\w+) into Go value of type ([\\w\\.]+)",
                "type": "json.UnmarshalTypeError",
                "description": "Type mismatch during JSON unmarshaling",
                "root_cause": "go_json_type_mismatch",
                "suggestion": "Ensure JSON structure matches Go struct. Verify struct field types match JSON types.",
                "confidence": "high",
                "severity": "medium",
                "category": "json",
            },
            {
                "id": "go_json_field",
                "pattern": 'json: unknown field "([^"]+)"',
                "type": "json.UnmarshalTypeError",
                "description": "Unknown field during JSON unmarshaling",
                "root_cause": "go_json_unknown_field",
                "suggestion": "Add the field to your struct or use a map[string]interface{} to capture all fields.",
                "confidence": "high",
                "severity": "medium",
                "category": "json",
            },
            {
                "id": "go_sql_no_rows",
                "pattern": "sql: no rows in result set",
                "type": "sql.ErrNoRows",
                "description": "SQL query returned no rows when one was expected",
                "root_cause": "go_sql_no_rows",
                "suggestion": "Check if the record exists before querying. Handle the 'no rows' case explicitly.",
                "confidence": "high",
                "severity": "medium",
                "category": "database",
            },
            {
                "id": "go_connection_refused",
                "pattern": "(?:connection refused|cannot connect to server)",
                "type": "net.OpError",
                "description": "Network connection refused",
                "root_cause": "go_connection_refused",
                "suggestion": "Ensure the server is running and accessible. Add connection retry logic with backoff.",
                "confidence": "medium",
                "severity": "high",
                "category": "network",
            },
            {
                "id": "go_data_race",
                "pattern": "WARNING: DATA RACE",
                "type": "DataRace",
                "description": "Data race detected",
                "root_cause": "go_data_race",
                "suggestion": "Use sync.Mutex or atomic operations to protect shared data. Run with -race flag to detect races.",
                "confidence": "high",
                "severity": "high",
                "category": "concurrency",
                "tags": ["go", "concurrency", "race"],
            },
            {
                "id": "go_concurrent_map_access",
                "pattern": "fatal error: concurrent map read and map write",
                "type": "runtime error",
                "description": "Concurrent map access detected",
                "root_cause": "go_concurrent_map_access",
                "suggestion": "Use sync.Map for concurrent access or protect map with sync.RWMutex.",
                "confidence": "high",
                "severity": "high",
                "category": "concurrency",
                "tags": ["go", "concurrency", "map"],
            },
            {
                "id": "go_type_mismatch",
                "pattern": "cannot use .* \\(type .*\\) as type .* in",
                "type": "CompilationError",
                "description": "Type mismatch in compilation",
                "root_cause": "go_type_mismatch",
                "suggestion": "Check type compatibility. Use type conversion if necessary.",
                "confidence": "high",
                "severity": "high",
                "category": "compilation",
                "tags": ["go", "compilation", "type"],
            },
            {
                "id": "go_import_cycle",
                "pattern": "import cycle not allowed",
                "type": "CompilationError",
                "description": "Import cycle detected",
                "root_cause": "go_import_cycle",
                "suggestion": "Refactor package dependencies to remove circular imports. Consider using interfaces to break dependencies.",
                "confidence": "high",
                "severity": "high",
                "category": "compilation",
                "tags": ["go", "compilation", "import"],
            },
            {
                "id": "go_context_canceled",
                "pattern": "context canceled",
                "type": "context.Canceled",
                "description": "Operation canceled due to context cancellation",
                "root_cause": "go_context_canceled",
                "suggestion": "Handle context cancellation explicitly in your code. This is often part of normal operation.",
                "confidence": "high",
                "severity": "low",
                "category": "core",
            },
            {
                "id": "go_context_deadline_exceeded",
                "pattern": "context deadline exceeded",
                "type": "context.DeadlineExceeded",
                "description": "Operation took too long and exceeded context deadline",
                "root_cause": "go_context_deadline",
                "suggestion": "Increase timeout duration or optimize the operation. Add appropriate error handling for timeouts.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
            },
            {
                "id": "go_goroutine_leak",
                "pattern": "goroutine.*leak|possible.*goroutine.*leak",
                "type": "runtime warning",
                "description": "Potential goroutine leak detected",
                "root_cause": "go_goroutine_leak",
                "suggestion": "Review goroutine lifecycle management. Ensure all goroutines have proper termination conditions. Use context for cancellation.",
                "confidence": "medium",
                "severity": "high",
                "category": "performance",
            },
            {
                "id": "go_memory_leak",
                "pattern": "MemoryLeak|memory leak detected|excessive memory usage|possible memory leak",
                "type": "performance issue",
                "description": "Memory leak detected in Go application",
                "root_cause": "go_memory_leak",
                "suggestion": "Use Go's built-in profiling tools (pprof) to analyze memory allocation patterns. Check for unbounded growth of slices, maps, or goroutines.",
                "confidence": "high",
                "severity": "high",
                "category": "performance",
            },
            {
                "id": "go_sql_injection",
                "pattern": "SQL injection|SecurityVulnerability.*SQL|potential SQL injection",
                "type": "security vulnerability",
                "description": "Potential SQL injection vulnerability",
                "root_cause": "go_sql_injection_vulnerability",
                "suggestion": "Use prepared statements or parameterized queries to prevent SQL injection. Never concatenate user input directly into SQL queries.",
                "confidence": "high",
                "severity": "critical",
                "category": "security",
            },
            {
                "id": "go_infinite_loop",
                "pattern": "InfiniteLoop|infinite loop detected|CPU.*100%.*loop|possible infinite loop",
                "type": "performance issue",
                "description": "Potential infinite loop detected",
                "root_cause": "go_infinite_loop",
                "suggestion": "Add proper loop termination conditions. Consider using context for cancellation. Add timeout mechanisms to prevent infinite execution.",
                "confidence": "high",
                "severity": "high",
                "category": "performance",
            },
        ]


class GoPatchGenerator:
    """
    Generates patch solutions for Go errors.

    This class provides capabilities to generate code fixes for common Go errors,
    using templates and contextual information about the error.
    """

    def __init__(self):
        """Initialize the Go patch generator."""
        self.templates_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates" / "go"
        )
        self.templates_dir.mkdir(exist_ok=True, parents=True)

        # Cache for loaded templates
        self.template_cache = {}

    def generate_patch(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a patch for a Go error based on analysis.

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
            "patch_id": f"go_{rule_id}",
            "patch_type": "suggestion",
            "language": "go",
            "framework": context.get("framework", ""),
            "suggestion": analysis.get("suggestion", "No suggestion available"),
            "confidence": analysis.get("confidence", "low"),
            "severity": analysis.get("severity", "medium"),
            "root_cause": root_cause,
        }

        # Try to find a specific template for this root cause
        template_name = f"{root_cause}.go.template"
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

                # Return the template-based patch with the expected format
                return {
                    "type": "code_modification",
                    "content": patch_code,
                    "description": patch_result.get(
                        "instructions", f"Apply patch for {root_cause}"
                    ),
                }
            except Exception as e:
                logger.warning(f"Error generating patch for {root_cause}: {e}")

        # If we don't have a specific template, generate code based on root cause
        if "patch_code" not in patch_result:
            # Generate code suggestions based on the root cause
            if root_cause == "go_nil_pointer_dereference":
                content = self._generate_nil_check_suggestion(analysis, context)
                if content:
                    return {
                        "type": "code_modification",
                        "content": content,
                        "description": "Add nil check before dereferencing pointer",
                    }
            elif root_cause == "go_index_out_of_range":
                content = self._generate_bounds_check_suggestion(analysis, context)
                if content:
                    return {
                        "type": "code_modification",
                        "content": content,
                        "description": "Add bounds check before array access",
                    }
            elif root_cause == "go_nil_map":
                content = self._generate_nil_map_suggestion(analysis, context)
                if content:
                    return {
                        "type": "code_modification",
                        "content": content,
                        "description": "Initialize map before use",
                    }
            elif root_cause in [
                "go_concurrent_map_write",
                "go_concurrent_map_read_write",
                "go_concurrent_map_access",
            ]:
                content = self._generate_concurrent_map_suggestion(analysis, context)
                if content:
                    return {
                        "type": "code_modification",
                        "content": content,
                        "description": "Add synchronization for concurrent map access",
                        "imports": ["sync"],
                    }
            elif root_cause in ["go_deadlock", "go_channel_deadlock"]:
                content = self._generate_deadlock_suggestion(analysis, context)
                if content:
                    return {
                        "type": "suggestion",
                        "content": content,
                        "description": "Fix deadlock in goroutine communication",
                    }
            elif root_cause == "go_json_type_mismatch":
                content = self._generate_json_type_suggestion(analysis, context)
                if content:
                    return {
                        "type": "code_modification",
                        "content": content,
                        "description": "Fix JSON type mismatch",
                    }
            elif root_cause == "go_type_assertion_failure":
                content = self._generate_type_assertion_suggestion(analysis, context)
                if content:
                    return {
                        "type": "code_modification",
                        "content": content,
                        "description": "Use safe type assertion with ok check",
                    }
            elif root_cause in ["go_race_condition", "go_data_race"]:
                content = self._generate_race_condition_suggestion(analysis, context)
                if content:
                    return {
                        "type": "code_modification",
                        "content": content,
                        "description": "Add mutex protection for shared variable",
                        "imports": ["sync.Mutex", "sync.RWMutex"],
                    }

        # Return suggestion if no specific patch can be generated
        result = {
            "type": "suggestion",
            "content": analysis.get("suggestion", "No specific fix available"),
            "description": f"Fix for {root_cause}",
        }
        return result

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
        variables["ERROR_TYPE"] = error_data.get("error_type", "error")
        variables["ERROR_MESSAGE"] = error_data.get("message", "Unknown error")

        # Extract information from stack trace
        stack_trace = error_data.get("stack_trace", [])
        if stack_trace and isinstance(stack_trace, list):
            if isinstance(stack_trace[0], dict):
                # Structured stack trace
                if stack_trace:
                    top_frame = stack_trace[0]
                    variables["PACKAGE"] = top_frame.get("package", "")
                    variables["FUNCTION"] = top_frame.get("function", "")
                    variables["FILE"] = top_frame.get("file", "")
                    variables["LINE"] = str(top_frame.get("line", ""))

        # Extract variables from context
        variables["CODE_SNIPPET"] = context.get("code_snippet", "")
        variables["IMPORTS"] = context.get("imports", "")

        # Add all context variables directly (for custom fields like array, index, etc.)
        for key, value in context.items():
            if key not in variables:  # Don't override existing variables
                variables[key] = str(value)

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
                        "package": top_frame.get("package", ""),
                        "function": top_frame.get("function", ""),
                    }
                )

        return application_point

    def _generate_instructions(self, analysis: Dict[str, Any], patch_code: str) -> str:
        """Generate human-readable instructions for applying the patch."""
        root_cause = analysis.get("root_cause", "unknown")

        if "nil_pointer" in root_cause:
            return "Add nil checks before accessing pointers."
        elif "index_out_of_range" in root_cause:
            return "Validate array or slice indices before accessing elements."
        elif "nil_map" in root_cause:
            return "Initialize the map with make() before use."
        elif "concurrent_map" in root_cause:
            return "Protect concurrent map access with a mutex or use sync.Map."
        elif "deadlock" in root_cause:
            return "Review goroutine synchronization patterns to prevent deadlocks."
        else:
            return "Apply the suggested fix to resolve the error."

    def _generate_nil_check_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for nil checking in Go."""
        variable = context.get("variable", "pointer")
        code_snippet = context.get("code_snippet", "")

        # Generate specific suggestion based on context
        if variable:
            return f"""// Add nil check before accessing the pointer
if {variable} != nil {{
    // Safe to access {variable}
    {code_snippet}
}} else {{
    // Handle nil case - either return early with an error, or provide a default
    return fmt.Errorf("{variable} is nil")
}}"""

        return """// Add nil check before accessing the pointer
if pointer == nil {
    // Handle nil case - either return early with an error, or provide a default
    return fmt.Errorf("pointer is nil")
    // Alternatively: pointer = getDefaultValue()
}
// Then proceed with original code that uses the pointer
"""

    def _generate_bounds_check_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for bounds checking in Go."""
        array = context.get("array", "slice")
        index = context.get("index", "index")
        code_snippet = context.get("code_snippet", f"value := {array}[{index}]")

        return f"""// Add bounds check before accessing slice/array elements
if {index} >= 0 && {index} < len({array}) {{
    // Safe to access {array}[{index}]
    {code_snippet}
}} else {{
    // Handle invalid index - either skip, log, or return error
    return fmt.Errorf("index %d out of bounds (length %d)", {index}, len({array}))
}}"""

    def _generate_nil_map_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for nil map checking in Go."""
        return """// Initialize map before use
if myMap == nil {
    myMap = make(map[KeyType]ValueType)
}
// Now safe to use
myMap[key] = value
"""

    def _generate_concurrent_map_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for handling concurrent map access in Go."""
        return """// Option 1: Use a mutex to protect map access
var mu sync.RWMutex
mu.Lock()
myMap[key] = value
mu.Unlock()

// For read operations, use RLock() instead
mu.RLock()
value := myMap[key]
mu.RUnlock()

// Option 2: Use sync.Map for concurrent access
var concurrentMap sync.Map
concurrentMap.Store(key, value)    // Write operation
value, ok := concurrentMap.Load(key)  // Read operation
"""

    def _generate_deadlock_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for avoiding deadlocks in Go."""
        return """// Common deadlock fixes:

// 1. Use buffered channels when appropriate
ch := make(chan Type, bufferSize)  // Prevents deadlock when producers > consumers

// 2. Ensure consistent lock ordering
// Always acquire locks in the same order across goroutines
mu1.Lock()
mu2.Lock()
// Critical section
mu2.Unlock()
mu1.Unlock()

// 3. Use select with default case or timeout to prevent blocking indefinitely
select {
case msg := <-ch:
    // Process message
case <-time.After(5 * time.Second):
    // Handle timeout
case <-ctx.Done():
    // Handle cancellation
default:
    // Non-blocking alternative
}
"""

    def _generate_type_assertion_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for safe type assertion in Go."""
        # Check both interface and interface_var for backwards compatibility
        interface_var = context.get("interface_var", context.get("interface", "v"))
        target_type = context.get("target_type", "string")
        code_snippet = context.get("code_snippet", "")

        return f"""// Use safe type assertion with ok idiom
val, ok := {interface_var}.({target_type})
if ok {{
    // Type assertion succeeded
    {code_snippet if code_snippet else f"// Use val as {target_type}"}
}} else {{
    // Type assertion failed - handle gracefully
    return fmt.Errorf("type assertion failed: expected {target_type}")
}}"""

    def _generate_race_condition_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for fixing race conditions in Go."""
        shared_var = context.get("shared_var", "sharedData")
        code_snippet = context.get("code_snippet", f"{shared_var}++")

        return f"""// Add mutex protection for shared variable
var mu sync.Mutex

// Protect access to shared variable
mu.Lock()
{code_snippet}
mu.Unlock()

// Alternative: Use defer to ensure unlock
mu.Lock()
defer mu.Unlock()
{code_snippet}

// For read-heavy workloads, consider sync.RWMutex
var rwmu sync.RWMutex

// For reads:
rwmu.RLock()
value := {shared_var}
rwmu.RUnlock()

// For writes:
rwmu.Lock()
{code_snippet}
rwmu.Unlock()"""

    def _generate_json_type_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for handling JSON type mismatches in Go."""
        return """// Ensure struct field types match expected JSON types
type MyStruct struct {
    // Use proper types and tags
    ID       int     `json:"id"`              // For JSON numbers
    Name     string  `json:"name"`            // For JSON strings
    Active   bool    `json:"active"`          // For JSON booleans
    Tags     []string `json:"tags,omitempty"` // For JSON arrays
    Metadata map[string]interface{} `json:"metadata,omitempty"` // For nested objects
}

// For handling optional/nullable fields, use pointers
type UserProfile struct {
    Name  string  `json:"name"`
    Email *string `json:"email,omitempty"` // Can be null in JSON
    Age   *int    `json:"age,omitempty"`   // Can be null in JSON
}
"""


class GoLanguagePlugin(LanguagePlugin):
    """
    Go language plugin for Homeostasis.

    Provides comprehensive error analysis and fix generation for Go applications,
    including support for goroutine management, Go modules, and popular Go frameworks.
    """

    VERSION = "1.0.0"
    AUTHOR = "Homeostasis Contributors"

    def __init__(self):
        """Initialize the Go language plugin."""
        self.adapter = GoErrorAdapter()
        self.error_handler = GoErrorHandler()
        self.patch_generator = GoPatchGenerator()
        self.supported_frameworks = ["gin", "echo", "beego", "fiber", "gorm", "buffalo"]

    def get_language_id(self) -> str:
        """Get the language identifier."""
        return "go"

    def get_language_name(self) -> str:
        """Get the language name."""
        return "Go"

    def get_language_version(self) -> str:
        """Get the language version."""
        return "1.13+"

    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        return {
            "name": "Go Language Support",
            "language": "go",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "supported_frameworks": self.supported_frameworks,
            "description": "Comprehensive error analysis and fix generation for Go applications",
        }

    def can_handle(self, error_data: Dict[str, Any]) -> bool:
        """Check if this plugin can handle the given error."""
        language = error_data.get("language", "").lower()
        return language == "go" or language == "golang"

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Go error.

        Args:
            error_data: Go error data

        Returns:
            Analysis results
        """
        # First, normalize the error
        if "language" not in error_data or error_data["language"] != "go":
            standard_error = self.normalize_error(error_data)
        else:
            standard_error = error_data

        # Use the error handler to analyze the error
        analysis = self.error_handler.analyze_error(standard_error)

        # Detect framework from stack trace
        framework = self._detect_framework(standard_error)
        if framework:
            analysis["framework"] = framework

        return analysis

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a Go error to the standard format.

        Args:
            error_data: Go error data

        Returns:
            Error data in the standard format
        """
        return self.adapter.to_standard_format(error_data)

    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Go format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Go format
        """
        return self.adapter.from_standard_format(standard_error)

    def _detect_framework(self, error_data: Dict[str, Any]) -> Optional[str]:
        """
        Detect the Go framework from error data.

        Args:
            error_data: Error data containing stack trace

        Returns:
            Detected framework name or None
        """
        # Check if framework is already specified
        if "framework" in error_data and error_data["framework"]:
            return error_data["framework"]

        # Framework detection patterns
        framework_patterns = {
            "gin": ["github.com/gin-gonic/gin"],
            "echo": ["github.com/labstack/echo"],
            "fiber": ["github.com/gofiber/fiber"],
            "beego": ["github.com/astaxie/beego", "github.com/beego/beego"],
            "gorm": ["gorm.io/gorm", "github.com/jinzhu/gorm"],
            "buffalo": ["github.com/gobuffalo/buffalo"],
        }

        # Check stack trace for framework signatures
        stack_trace = error_data.get("stack_trace", [])
        if isinstance(stack_trace, list):
            for frame in stack_trace:
                if isinstance(frame, dict):
                    function = frame.get("function", "")
                    file = frame.get("file", "")

                    # Check against framework patterns
                    for framework, patterns in framework_patterns.items():
                        for pattern in patterns:
                            if pattern in function or pattern in file:
                                return framework

        # Check error message for framework mentions
        message = error_data.get("message", "")
        for framework, patterns in framework_patterns.items():
            for pattern in patterns:
                if pattern in message:
                    return framework

        return None

    def generate_fix(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a fix for a Go error.

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
        return ["gin", "echo", "standard", "base"]


# Register this plugin
register_plugin(GoLanguagePlugin())
