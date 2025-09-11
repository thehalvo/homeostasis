"""
Java Language Plugin for Homeostasis

This plugin enables Homeostasis to analyze and fix errors in Java applications.
It provides comprehensive exception handling for standard Java errors and
framework-specific issues including Spring, Jakarta EE, Hibernate, and Android.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from ..language_adapters import JavaErrorAdapter
from ..language_plugin_system import LanguagePlugin, register_plugin

logger = logging.getLogger(__name__)


class JavaExceptionHandler:
    """
    Handles Java exceptions with a robust error detection and classification system.

    This class provides logic for categorizing Java exceptions based on their type,
    message, and stack trace patterns. It supports both standard Java exceptions and
    framework-specific exceptions.
    """

    def __init__(self):
        """Initialize the Java exception handler."""
        self.rule_categories = {
            "core": "Core Java exceptions",
            "io": "IO and file-related exceptions",
            "collections": "Collection and data structure exceptions",
            "concurrency": "Threading and concurrency exceptions",
            "reflection": "Reflection and class loading exceptions",
            "jdbc": "Database and JDBC exceptions",
            "spring": "Spring Framework exceptions",
            "hibernate": "Hibernate and JPA exceptions",
            "jakarta": "Jakarta EE exceptions",
            "servlet": "Servlet and JSP exceptions",
            "android": "Android platform exceptions",
            "security": "Security-related exceptions",
            "xml": "XML processing exceptions",
            "json": "JSON processing exceptions",
            "network": "Network and remote communication exceptions",
        }

        # Load rules from different categories
        self.rules = self._load_all_rules()

        # Initialize caches for performance
        self.pattern_cache = {}  # Compiled regex patterns
        self.rule_match_cache = {}  # Previous rule matches

    def analyze_exception(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Java exception to determine its root cause and suggest potential fixes.

        Args:
            error_data: Java error data in standard format

        Returns:
            Analysis result with root cause, description, and fix suggestions
        """
        error_type = error_data.get("error_type", "")
        message = error_data.get("message", "")
        stack_trace = error_data.get("stack_trace", [])

        # Create a consolidated text for pattern matching
        match_text = self._create_match_text(error_type, message, stack_trace)

        # Get framework from error data (if provided)
        framework = error_data.get("framework", "").lower()

        # Sort rules to prioritize framework-specific rules
        sorted_rules = self.rules
        if framework:
            # Put framework-specific rules first
            framework_rules = [
                r for r in self.rules if r.get("framework", "").lower() == framework
            ]
            other_rules = [
                r for r in self.rules if r.get("framework", "").lower() != framework
            ]
            sorted_rules = framework_rules + other_rules

        # Try to match against known rules
        for rule in sorted_rules:
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
                        "root_cause": rule.get("root_cause", "java_unknown_error"),
                        "description": rule.get("description", "Unknown Java error"),
                        "suggestion": rule.get("suggestion", "No suggestion available"),
                        "confidence": rule.get("confidence", "medium"),
                        "severity": rule.get("severity", "medium"),
                        "category": rule.get("category", "java"),
                        "match_groups": match.groups() if match.groups() else tuple(),
                        "framework": rule.get("framework", ""),
                    }

                    # Add tags if present in rule
                    if "tags" in rule:
                        result["tags"] = rule["tags"]

                    # Check for stream/lambda context in stack trace
                    tags = result.get("tags", [])
                    for frame in stack_trace:
                        if isinstance(frame, dict):
                            func_name = frame.get("function", "").lower()
                            class_name = frame.get("class", "").lower()
                            # Check for lambda functions
                            if "lambda$" in func_name or "lambda" in func_name:
                                if "lambda" not in tags:
                                    tags.append("lambda")
                            # Check for stream operations
                            if "stream" in class_name or any(
                                stream_method in func_name
                                for stream_method in [
                                    "map",
                                    "filter",
                                    "flatmap",
                                    "reduce",
                                    "collect",
                                    "foreach",
                                ]
                            ):
                                if "stream" not in tags:
                                    tags.append("stream")

                    if tags:
                        result["tags"] = tags

                    # Special case: IllegalMonitorStateException with deadlock message
                    if (
                        rule.get("id") == "java_illegal_monitor_state" and
                        "deadlock" in message.lower()
                    ):
                        result["severity"] = "critical"
                        result["root_cause"] = "java_deadlock"

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
                            line = f"at {frame.get('package', '')}.{frame.get('class', '')}."
                            line += f"{frame.get('function', '')}({frame.get('file', '')}:{frame.get('line', '?')})"
                            trace_lines.append(line)

                    match_text += "\n" + "\n".join(trace_lines)

        return match_text

    def _handle_fallback(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle exceptions that didn't match any specific rule.

        Args:
            error_data: Java error data in standard format

        Returns:
            Fallback analysis result
        """
        error_type = error_data.get("error_type", "")

        # Check for common Java exception types and apply basic categorization
        if "NullPointerException" in error_type:
            result = {
                "error_data": error_data,
                "rule_id": "java_null_pointer",
                "error_type": error_type,
                "root_cause": "java_null_pointer",
                "description": "Attempted to access or use a null object reference",
                "suggestion": "Add null checks before accessing objects or methods. Consider using Optional<T> for values that might be null.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
            }

            # Check if this is a stream/lambda related NPE
            stack_trace = error_data.get("stack_trace", [])
            tags = []
            for frame in stack_trace:
                if isinstance(frame, dict):
                    func_name = frame.get("function", "").lower()
                    class_name = frame.get("class", "").lower()
                    # Check for lambda functions
                    if "lambda$" in func_name or "lambda" in func_name:
                        tags.append("lambda")
                    # Check for stream operations
                    if "stream" in class_name or any(
                        stream_method in func_name
                        for stream_method in [
                            "map",
                            "filter",
                            "flatmap",
                            "reduce",
                            "collect",
                            "foreach",
                        ]
                    ):
                        tags.append("stream")

            if tags:
                result["tags"] = list(set(tags))  # Remove duplicates
                # Enhance suggestion for stream/lambda context
                if "stream" in tags:
                    result[
                        "suggestion"
                    ] += " In stream operations, use filter(Objects::nonNull) to remove null values before processing."

            return result
        elif "ClassCastException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "java_class_cast",
                "error_type": error_type,
                "root_cause": "java_class_cast",
                "description": "Attempted to cast an object to an incompatible type",
                "suggestion": "Verify object types before casting using instanceof checks. Consider using generics to ensure type safety at compile time.",
                "confidence": "high",
                "severity": "medium",
                "category": "runtime",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
            }
        elif (
            "ArrayIndexOutOfBoundsException" in error_type or
            "IndexOutOfBoundsException" in error_type
        ):
            return {
                "error_data": error_data,
                "rule_id": "java_index_out_of_bounds",
                "error_type": error_type,
                "root_cause": "java_index_out_of_bounds",
                "description": "Attempted to access an array or collection element with an invalid index",
                "suggestion": "Check array/collection bounds before accessing elements. Use collection.size() or array.length to ensure the index is valid.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
            }
        elif "IllegalArgumentException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "java_illegal_argument",
                "error_type": error_type,
                "root_cause": "java_invalid_argument",
                "description": "A method received an argument that was not valid",
                "suggestion": "Validate input arguments before passing them to methods. Add precondition checks with descriptive error messages.",
                "confidence": "medium",
                "severity": "medium",
                "category": "core",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
            }
        elif "UnsupportedOperationException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "java_unsupported_operation",
                "error_type": error_type,
                "root_cause": "java_unsupported_operation",
                "description": "An operation is not supported on this object",
                "suggestion": "Check if the object supports the operation before calling it. Common with immutable collections or views.",
                "confidence": "medium",
                "severity": "medium",
                "category": "collections",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
            }
        elif "InstantiationException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "java_instantiation_exception",
                "error_type": error_type,
                "root_cause": "java_instantiation_error",
                "description": "Cannot instantiate the specified class",
                "suggestion": "Cannot instantiate abstract classes or interfaces. Ensure the class has a public no-argument constructor and is not abstract.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
            }
        elif "SecurityException" in error_type:
            return {
                "error_data": error_data,
                "rule_id": "java_security_exception",
                "error_type": error_type,
                "root_cause": "java_security_violation",
                "description": "Security manager denied access to a protected resource",
                "suggestion": "Check security policy and permissions. Grant necessary permissions in security policy file or disable security manager if appropriate.",
                "confidence": "high",
                "severity": "high",
                "category": "security",
                "match_groups": tuple(),
                "framework": error_data.get("framework", ""),
            }

        # Generic fallback for unknown exceptions
        return {
            "error_data": error_data,
            "rule_id": "java_generic_fallback",
            "error_type": error_type or "Unknown",
            "root_cause": "java_unknown_error",
            "description": f"Unrecognized Java error: {error_type}",
            "suggestion": "Review the error message and stack trace for more details. Check the documentation for this exception type.",
            "confidence": "low",
            "severity": "medium",
            "category": "java",
            "match_groups": tuple(),
            "framework": error_data.get("framework", ""),
        }

    def _load_all_rules(self) -> List[Dict[str, Any]]:
        """
        Load Java error rules from all categories.

        Returns:
            Combined list of rule definitions
        """
        all_rules = []

        # Core Java exceptions (always included)
        all_rules.extend(self._load_core_java_rules())

        # Load additional rules for other categories
        all_rules.extend(self._load_collection_rules())
        all_rules.extend(self._load_stream_rules())
        all_rules.extend(self._load_generics_rules())
        all_rules.extend(self._load_reflection_rules())
        all_rules.extend(self._load_io_rules())
        all_rules.extend(self._load_concurrency_rules())
        all_rules.extend(self._load_jdbc_rules())

        # Framework-specific rules
        all_rules.extend(self._load_spring_rules())
        all_rules.extend(self._load_hibernate_rules())
        all_rules.extend(self._load_android_rules())
        # all_rules.extend(self._load_jakarta_rules())

        return all_rules

    def _load_core_java_rules(self) -> List[Dict[str, Any]]:
        """Load rules for core Java exceptions."""
        return [
            {
                "id": "java_null_pointer",
                "pattern": "java\\.lang\\.NullPointerException(?:: (.*))?",
                "type": "NullPointerException",
                "description": "Attempted to access or use a null object reference",
                "root_cause": "java_null_pointer",
                "suggestion": "Add null checks before accessing objects or methods. Consider using Optional<T> for values that might be null.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime",
            },
            {
                "id": "java_class_cast",
                "pattern": "java\\.lang\\.ClassCastException: ([^\\s]+) cannot be cast to ([^\\s]+)",
                "type": "ClassCastException",
                "description": "Attempted to cast an object to an incompatible type",
                "root_cause": "java_class_cast",
                "suggestion": "Verify object types before casting using instanceof checks. Consider using generics to ensure type safety at compile time.",
                "confidence": "high",
                "severity": "medium",
                "category": "runtime",
            },
            {
                "id": "java_array_index",
                "pattern": "java\\.lang\\.ArrayIndexOutOfBoundsException: (\\d+)",
                "type": "ArrayIndexOutOfBoundsException",
                "description": "Attempted to access an array element with an invalid index",
                "root_cause": "java_index_out_of_bounds",
                "suggestion": "Check array bounds before accessing elements. Use array.length to ensure the index is valid.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
            },
            {
                "id": "java_string_index",
                "pattern": "java\\.lang\\.StringIndexOutOfBoundsException: (.*)",
                "type": "StringIndexOutOfBoundsException",
                "description": "Attempted to access a character in a string with an invalid index",
                "root_cause": "java_string_index_out_of_bounds",
                "suggestion": "Check string length before accessing characters. Use string.length() to ensure the index is valid.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
            },
            {
                "id": "java_illegal_argument",
                "pattern": "java\\.lang\\.IllegalArgumentException: (.*)",
                "type": "IllegalArgumentException",
                "description": "A method received an argument that was not valid",
                "root_cause": "java_invalid_argument",
                "suggestion": "Validate input arguments before passing them to methods. Add precondition checks with descriptive error messages.",
                "confidence": "medium",
                "severity": "medium",
                "category": "core",
            },
            {
                "id": "java_illegal_state",
                "pattern": "java\\.lang\\.IllegalStateException: (?!.*stream has already been operated upon or closed)(.*)",
                "type": "IllegalStateException",
                "description": "Object is in an invalid state for the requested operation",
                "root_cause": "java_invalid_state",
                "suggestion": "Check object state before performing operations. Ensure operations are called in the correct sequence.",
                "confidence": "medium",
                "severity": "medium",
                "category": "core",
            },
            {
                "id": "java_unsupported_operation",
                "pattern": "java\\.lang\\.UnsupportedOperationException(?:: (.*))?",
                "type": "UnsupportedOperationException",
                "description": "The requested operation is not supported",
                "root_cause": "java_unsupported_operation",
                "suggestion": "Check if the object supports the operation before calling it. Common with immutable collections or views.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
            },
            {
                "id": "java_number_format",
                "pattern": "java\\.lang\\.NumberFormatException: (.*)",
                "type": "NumberFormatException",
                "description": "Failed to parse a string as a number",
                "root_cause": "java_invalid_number_format",
                "suggestion": "Ensure the string represents a valid number before parsing. Use try-catch blocks for parse operations and provide default values.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
            },
            {
                "id": "java_arithmetic",
                "pattern": "java\\.lang\\.ArithmeticException: (.*)",
                "type": "ArithmeticException",
                "description": "Arithmetic operation error, such as division by zero",
                "root_cause": "java_arithmetic_error",
                "suggestion": "Check for division by zero and other invalid arithmetic operations. Consider using BigDecimal for precise decimal calculations.",
                "confidence": "high",
                "severity": "medium",
                "category": "core",
            },
            {
                "id": "java_class_not_found",
                "pattern": "java\\.lang\\.ClassNotFoundException: (.*)",
                "type": "ClassNotFoundException",
                "description": "The specified class could not be found",
                "root_cause": "java_class_not_found",
                "suggestion": "Check classpath configuration and ensure the class is available. Verify package names and class names for typos.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
            },
            {
                "id": "java_no_such_method",
                "pattern": "java\\.lang\\.NoSuchMethodException: (.*)",
                "type": "NoSuchMethodException",
                "description": "The specified method could not be found",
                "root_cause": "java_no_such_method",
                "suggestion": "Check method name, parameters, and accessibility. Ensure the method exists in the class you're calling.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
            },
            {
                "id": "java_no_such_field",
                "pattern": "java\\.lang\\.NoSuchFieldException: (.*)",
                "type": "NoSuchFieldException",
                "description": "The specified field could not be found",
                "root_cause": "java_no_such_field",
                "suggestion": "Check field name and accessibility. Ensure the field exists in the class you're accessing.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
            },
            {
                "id": "java_out_of_memory",
                "pattern": "java\\.lang\\.OutOfMemoryError: (.*)",
                "type": "OutOfMemoryError",
                "description": "Java Virtual Machine ran out of memory",
                "root_cause": "java_out_of_memory",
                "suggestion": "Increase JVM heap size (-Xmx), optimize memory usage, check for memory leaks using a profiler, or implement object pooling for large objects.",
                "confidence": "high",
                "severity": "critical",
                "category": "resources",
            },
            {
                "id": "java_stack_overflow",
                "pattern": "java\\.lang\\.StackOverflowError(?:: (.*))?",
                "type": "StackOverflowError",
                "description": "Recursive method calls exceeded the stack size",
                "root_cause": "java_stack_overflow",
                "suggestion": "Check for infinite recursion and ensure proper termination conditions. Consider rewriting recursive algorithms iteratively.",
                "confidence": "high",
                "severity": "critical",
                "category": "resources",
            },
            {
                "id": "java_security_exception",
                "pattern": "java\\.lang\\.SecurityException:.*access denied.*",
                "type": "SecurityException",
                "description": "Security manager denied access to a protected resource",
                "root_cause": "java_security_violation",
                "suggestion": "Check security policy and permissions. Grant necessary permissions in security policy file or disable security manager if appropriate.",
                "confidence": "high",
                "severity": "high",
                "category": "security",
            },
            {
                "id": "java_compilation_error",
                "pattern": "CompilationError(?:.*?cannot find symbol.*?)?",
                "type": "CompilationError",
                "description": "Java source code failed to compile",
                "root_cause": "java_compilation_error",
                "suggestion": "Check that all required imports are present and class/method names are spelled correctly. Ensure all dependencies are properly declared in your build configuration.",
                "confidence": "high",
                "severity": "high",
                "category": "compilation",
            },
            {
                "id": "java_class_format",
                "pattern": "java\\.lang\\.ClassFormatError: (.*)",
                "type": "ClassFormatError",
                "description": "Class file is malformed or contains incompatible version information",
                "root_cause": "java_class_format_error",
                "suggestion": "Ensure compatible Java versions between compilation and runtime. Rebuild the application with the correct Java version.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
            },
            {
                "id": "java_assertion",
                "pattern": "java\\.lang\\.AssertionError(?:: (.*))?",
                "type": "AssertionError",
                "description": "An assertion has failed",
                "root_cause": "java_assertion_failed",
                "suggestion": "Fix the condition that caused the assertion to fail. Assertions typically indicate programming errors or invalid assumptions.",
                "confidence": "high",
                "severity": "high",
                "category": "core",
            },
        ]

    def _load_collection_rules(self) -> List[Dict[str, Any]]:
        """Load rules for Java collection framework exceptions."""
        return [
            {
                "id": "java_concurrent_modification",
                "pattern": "java\\.util\\.ConcurrentModificationException(?:: (.*))?",
                "type": "ConcurrentModificationException",
                "description": "Collection was modified during iteration",
                "root_cause": "java_concurrent_modification",
                "suggestion": "Use thread-safe collections (ConcurrentHashMap, CopyOnWriteArrayList) or synchronize access to collections during iteration. Consider using the Iterator's remove() method instead of directly modifying the collection.",
                "confidence": "high",
                "severity": "high",
                "category": "concurrency",
                "tags": ["concurrency", "collections"],
            },
            {
                "id": "java_no_such_element",
                "pattern": "java\\.util\\.NoSuchElementException(?:: (.*))?",
                "type": "NoSuchElementException",
                "description": "Attempted to access an element that doesn't exist",
                "root_cause": "java_no_such_element",
                "suggestion": "Check if an element exists before accessing it. Use methods like containsKey() for maps or isEmpty() for collections.",
                "confidence": "high",
                "severity": "medium",
                "category": "collections",
            },
            {
                "id": "java_empty_stack",
                "pattern": "java\\.util\\.EmptyStackException",
                "type": "EmptyStackException",
                "description": "Attempted to pop an element from an empty stack",
                "root_cause": "java_empty_stack",
                "suggestion": "Check if the stack is empty with isEmpty() before calling pop() or peek()",
                "confidence": "high",
                "severity": "medium",
                "category": "collections",
            },
            {
                "id": "java_map_missing_key",
                "pattern": "java\\.lang\\.NullPointerException.*\\.get\\(([^\\)]+)\\)",
                "type": "NullPointerException",
                "description": "Attempted to access a map value with a key that doesn't exist",
                "root_cause": "java_map_missing_key",
                "suggestion": "Use containsKey() to check if a key exists before calling get(), or use getOrDefault() to provide a fallback value",
                "confidence": "medium",
                "severity": "medium",
                "category": "collections",
            },
        ]

    def _load_stream_rules(self) -> List[Dict[str, Any]]:
        """Load rules for Java Stream API and lambda expressions."""
        rules_file = (
            Path(__file__).parent.parent / "rules" / "java" / "java_streams_errors.json"
        )

        if rules_file.exists():
            try:
                with open(rules_file, "r") as f:
                    rules_data = json.load(f)
                    return rules_data.get("rules", [])
            except Exception as e:
                logger.warning(f"Error loading Java stream rules: {e}")

        # Fallback rules if file not found
        return [
            {
                "id": "java_stream_already_operated",
                "pattern": "stream has already been operated upon or closed|IllegalStateException.*stream",
                "type": "IllegalStateException",
                "description": "Stream has already been consumed or closed",
                "root_cause": "java_stream_reuse",
                "suggestion": "Streams can only be used once and cannot be reused. Create a new stream for each terminal operation. Use Supplier<Stream> for reusable stream creation.",
                "confidence": "high",
                "severity": "high",
                "category": "runtime",
                "tags": ["stream"],
            }
        ]

    def _load_generics_rules(self) -> List[Dict[str, Any]]:
        """Load rules for Java generics and type system errors."""
        rules_file = (
            Path(__file__).parent.parent /
            "rules" /
            "java" /
            "java_generics_errors.json"
        )

        if rules_file.exists():
            try:
                with open(rules_file, "r") as f:
                    rules_data = json.load(f)
                    return rules_data.get("rules", [])
            except Exception as e:
                logger.warning(f"Error loading Java generics rules: {e}")

        return []

    def _load_reflection_rules(self) -> List[Dict[str, Any]]:
        """Load rules for Java reflection errors."""
        rules_file = (
            Path(__file__).parent.parent /
            "rules" /
            "java" /
            "java_reflection_errors.json"
        )

        if rules_file.exists():
            try:
                with open(rules_file, "r") as f:
                    rules_data = json.load(f)
                    rules = rules_data.get("rules", [])
                    # Update category for reflection rules to be security when appropriate
                    for rule in rules:
                        if rule.get("id") == "java_illegal_access_reflection":
                            rule["category"] = "security"
                        elif rule.get("id") == "java_security_manager_reflection":
                            rule["category"] = "security"
                    return rules
            except Exception as e:
                logger.warning(f"Error loading Java reflection rules: {e}")

        return []

    def _load_io_rules(self) -> List[Dict[str, Any]]:
        """Load rules for Java I/O exceptions."""
        return [
            {
                "id": "java_io_exception",
                "pattern": "java\\.io\\.IOException: (.*)",
                "type": "IOException",
                "description": "An I/O operation failed or was interrupted",
                "root_cause": "java_io_error",
                "suggestion": "Implement appropriate error handling for I/O operations, including closing resources in finally blocks or using try-with-resources",
                "confidence": "medium",
                "severity": "medium",
                "category": "io",
            },
            {
                "id": "java_file_not_found",
                "pattern": "java\\.io\\.FileNotFoundException: (.*)",
                "type": "FileNotFoundException",
                "description": "The specified file could not be found",
                "root_cause": "java_file_not_found",
                "suggestion": "Check file paths and permissions. Use File.exists() to verify file existence before opening.",
                "confidence": "high",
                "severity": "medium",
                "category": "io",
            },
            {
                "id": "java_eof",
                "pattern": "java\\.io\\.EOFException(?:: (.*))?",
                "type": "EOFException",
                "description": "Reached end of file unexpectedly",
                "root_cause": "java_unexpected_eof",
                "suggestion": "Check if the input stream has sufficient data before reading. Handle unexpected end-of-file conditions gracefully.",
                "confidence": "high",
                "severity": "medium",
                "category": "io",
            },
            {
                "id": "java_socket_timeout",
                "pattern": "java\\.net\\.SocketTimeoutException: (.*)",
                "type": "SocketTimeoutException",
                "description": "Socket operation timed out",
                "root_cause": "java_socket_timeout",
                "suggestion": "Increase socket timeout, implement retry mechanisms, or add fallback strategies for network operations",
                "confidence": "high",
                "severity": "medium",
                "category": "io",
            },
            {
                "id": "java_socket_connection",
                "pattern": "java\\.net\\.ConnectException: (.*)",
                "type": "ConnectException",
                "description": "Failed to establish a connection",
                "root_cause": "java_connection_failed",
                "suggestion": "Check network connectivity, server availability, and firewall settings. Implement proper connection error handling and retry logic.",
                "confidence": "high",
                "severity": "high",
                "category": "io",
            },
            {
                "id": "java_malformed_url",
                "pattern": "java\\.net\\.MalformedURLException: (.*)",
                "type": "MalformedURLException",
                "description": "URL is not formatted correctly",
                "root_cause": "java_invalid_url",
                "suggestion": "Validate URL format before creating URL objects. Ensure all URL components are properly encoded.",
                "confidence": "high",
                "severity": "medium",
                "category": "io",
            },
        ]

    def _load_concurrency_rules(self) -> List[Dict[str, Any]]:
        """Load rules for Java concurrency exceptions."""
        rules_file = (
            Path(__file__).parent.parent / "rules" / "java" / "concurrency_errors.json"
        )

        if rules_file.exists():
            try:
                with open(rules_file, "r") as f:
                    data = json.load(f)
                    return data.get("rules", [])
            except Exception as e:
                logger.error(f"Error loading concurrency rules from {rules_file}: {e}")

        # Fallback to built-in rules if file doesn't exist or can't be loaded
        return [
            {
                "id": "java_interrupted_exception",
                "pattern": "java\\.lang\\.InterruptedException(?:: (.*))?",
                "type": "InterruptedException",
                "description": "Thread was interrupted while waiting, sleeping, or otherwise occupied",
                "root_cause": "java_thread_interrupted",
                "suggestion": "Handle InterruptedException properly by either restoring the interrupt status or propagating the exception",
                "confidence": "high",
                "severity": "medium",
                "category": "concurrency",
            },
            {
                "id": "java_deadlock",
                "pattern": ".*?deadlock.*?detected.*?",
                "type": "DeadlockDetected",
                "description": "A deadlock was detected in concurrent threads",
                "root_cause": "java_deadlock",
                "suggestion": "Avoid nested synchronized blocks, always acquire locks in the same order across threads, use tryLock with timeout, or use higher-level concurrency utilities",
                "confidence": "medium",
                "severity": "critical",
                "category": "concurrency",
            },
            {
                "id": "java_illegal_monitor_state",
                "pattern": "java\\.lang\\.IllegalMonitorStateException(?:: (.*))?",
                "type": "IllegalMonitorStateException",
                "description": "Thread attempted to wait or notify on an object's monitor without owning it",
                "root_cause": "java_monitor_state_error",
                "suggestion": "Ensure wait(), notify(), and notifyAll() are called within synchronized blocks on the same object",
                "confidence": "high",
                "severity": "critical",
                "category": "concurrency",
            },
            {
                "id": "java_thread_death",
                "pattern": "java\\.lang\\.ThreadDeath",
                "type": "ThreadDeath",
                "description": "Thread was terminated by the deprecated Thread.stop() method",
                "root_cause": "java_thread_stop",
                "suggestion": "Avoid using Thread.stop() as it is unsafe and deprecated. Use interruption or other cooperative cancellation mechanisms.",
                "confidence": "high",
                "severity": "high",
                "category": "concurrency",
            },
            {
                "id": "java_execution_exception",
                "pattern": "java\\.util\\.concurrent\\.ExecutionException: (.*)",
                "type": "ExecutionException",
                "description": "Exception thrown during the execution of a task in a Future",
                "root_cause": "java_task_execution_error",
                "suggestion": "Examine the cause of the ExecutionException to find the actual error that occurred during task execution",
                "confidence": "medium",
                "severity": "high",
                "category": "concurrency",
            },
            {
                "id": "java_timeout_exception",
                "pattern": "java\\.util\\.concurrent\\.TimeoutException(?:: (.*))?",
                "type": "TimeoutException",
                "description": "Operation timed out before completion",
                "root_cause": "java_operation_timeout",
                "suggestion": "Increase timeout duration, optimize the operation for better performance, or handle timeouts gracefully",
                "confidence": "high",
                "severity": "medium",
                "category": "concurrency",
            },
            {
                "id": "java_rejected_execution",
                "pattern": "java\\.util\\.concurrent\\.RejectedExecutionException(?:: (.*))?",
                "type": "RejectedExecutionException",
                "description": "Task was rejected by a thread pool executor",
                "root_cause": "java_task_rejected",
                "suggestion": "Configure an appropriate RejectedExecutionHandler, increase executor queue size, or implement back pressure mechanisms",
                "confidence": "high",
                "severity": "high",
                "category": "concurrency",
            },
            {
                "id": "java_concurrent_modification",
                "pattern": "java\\.util\\.ConcurrentModificationException(?:.*?)",
                "type": "ConcurrentModificationException",
                "description": "Collection was modified while being iterated",
                "root_cause": "java_concurrent_modification",
                "suggestion": "Use thread-safe collections (ConcurrentHashMap, CopyOnWriteArrayList) or synchronize access to collections. For modifications during iteration, use Iterator.remove() instead of Collection.remove(), or use a snapshot copy for iteration.",
                "confidence": "high",
                "severity": "high",
                "category": "concurrency",
                "tags": ["concurrency", "collections"],
            },
            {
                "id": "java_race_condition",
                "pattern": "(?:race condition|atomicity violation|inconsistent state|thread safety violation)",
                "type": "RaceCondition",
                "description": "Race condition detected due to unprotected shared state access",
                "root_cause": "java_race_condition",
                "suggestion": "Protect shared state access with synchronized blocks, java.util.concurrent.locks, or use thread-safe data structures. Consider using atomic variables (AtomicInteger, AtomicReference) for simple cases, or volatile for visibility.",
                "confidence": "medium",
                "severity": "high",
                "category": "concurrency",
            },
        ]

    def _load_jdbc_rules(self) -> List[Dict[str, Any]]:
        """Load rules for Java JDBC and database exceptions."""
        return [
            {
                "id": "java_sql_exception",
                "pattern": "java\\.sql\\.SQLException: (.*)",
                "type": "SQLException",
                "description": "Database access error or other SQL-related issues",
                "root_cause": "java_sql_error",
                "suggestion": "Check database connection, SQL syntax, and implement proper exception handling. Consider using a connection pool for better resource management.",
                "confidence": "medium",
                "severity": "high",
                "category": "jdbc",
            },
            {
                "id": "java_sql_syntax",
                "pattern": "java\\.sql\\.SQLException: (.*syntax.*|.*valid.*)",
                "type": "SQLException",
                "description": "SQL syntax error",
                "root_cause": "java_sql_syntax_error",
                "suggestion": "Check SQL query syntax. Use prepared statements to avoid syntax and injection issues.",
                "confidence": "high",
                "severity": "high",
                "category": "jdbc",
            },
            {
                "id": "java_sql_duplicate_key",
                "pattern": "java\\.sql\\.SQLException: (.*duplicate.*key.*|.*constraint.*)",
                "type": "SQLException",
                "description": "Duplicate key or constraint violation",
                "root_cause": "java_sql_constraint_violation",
                "suggestion": "Check for existing records before inserting or use INSERT ... ON DUPLICATE KEY UPDATE syntax",
                "confidence": "high",
                "severity": "medium",
                "category": "jdbc",
            },
            {
                "id": "java_sql_column_not_found",
                "pattern": "java\\.sql\\.SQLException: (.*column.*not.*found.*|.*unknown.*column.*)",
                "type": "SQLException",
                "description": "Referenced a column that doesn't exist",
                "root_cause": "java_sql_invalid_column",
                "suggestion": "Verify column names in queries against the actual database schema",
                "confidence": "high",
                "severity": "medium",
                "category": "jdbc",
            },
            {
                "id": "java_sql_data_truncation",
                "pattern": "java\\.sql\\.DataTruncation(?:: (.*))?",
                "type": "DataTruncation",
                "description": "Data was truncated when reading or writing to the database",
                "root_cause": "java_sql_data_truncation",
                "suggestion": "Increase column size to accommodate data or validate data length before inserting/updating",
                "confidence": "high",
                "severity": "medium",
                "category": "jdbc",
            },
            {
                "id": "java_sql_connection_timeout",
                "pattern": "java\\.sql\\.SQLException: (.*timeout.*|.*connection.*fail.*)",
                "type": "SQLException",
                "description": "Database connection timeout or failure",
                "root_cause": "java_sql_connection_error",
                "suggestion": "Check database availability, connection settings, and implement connection retry logic",
                "confidence": "medium",
                "severity": "high",
                "category": "jdbc",
            },
        ]

    def _load_spring_rules(self) -> List[Dict[str, Any]]:
        """Load rules for Spring Framework and Spring Boot exceptions."""
        rules_file = (
            Path(__file__).parent.parent / "rules" / "java" / "spring_boot_errors.json"
        )

        if rules_file.exists():
            try:
                with open(rules_file, "r") as f:
                    data = json.load(f)
                    return data.get("rules", [])
            except Exception as e:
                logger.error(f"Error loading Spring rules from {rules_file}: {e}")

        # Fallback to built-in rules if file doesn't exist or can't be loaded
        return [
            {
                "id": "spring_bean_creation_error",
                "pattern": "org\\.springframework\\.beans\\.factory\\.BeanCreationException(?:.*?Error creating bean with name '([^']+)'.*?)?",
                "type": "BeanCreationException",
                "description": "Spring could not create a bean instance",
                "root_cause": "spring_bean_creation_error",
                "suggestion": "Check the bean configuration and initialization. Ensure all dependencies are available, constructor arguments are correct, and initialization methods are properly implemented. Check for circular dependencies or missing required beans.",
                "confidence": "high",
                "severity": "high",
                "category": "spring",
                "framework": "spring",
                "tags": ["bean", "configuration", "initialization"],
            },
            {
                "id": "spring_autowired_failure",
                "pattern": "(?:org\\.springframework\\.beans\\.factory\\.UnsatisfiedDependencyException|org\\.springframework\\.beans\\.factory\\.NoSuchBeanDefinitionException)(?:.*?Consider defining a bean of type '([^']+)'.*?|.*?No qualifying bean of type '([^']+)' available.*?)",
                "type": "UnsatisfiedDependencyException",
                "description": "Spring could not autowire a dependency because the required bean was not found",
                "root_cause": "spring_missing_bean",
                "suggestion": "Make sure the dependency is properly declared as a bean. Check that component scanning is properly configured to include the appropriate package. You may need to add @Component, @Service, @Repository, or @Bean annotation.",
                "confidence": "high",
                "severity": "high",
                "category": "spring",
                "framework": "spring",
            },
            {
                "id": "spring_circular_dependency",
                "pattern": "(?:org\\.springframework\\.beans\\.factory\\.BeanCurrentlyInCreationException|The dependencies of some of the beans in the application context form a cycle)",
                "type": "BeanCurrentlyInCreationException",
                "description": "Circular dependency detected in Spring application context",
                "root_cause": "spring_circular_dependency",
                "suggestion": "Break the circular dependency by: 1) Redesign the components to avoid circular references, 2) Use @Lazy to defer one of the dependencies, 3) Use setter injection instead of constructor injection, or 4) Introduce an interface to decouple the components.",
                "confidence": "high",
                "severity": "high",
                "category": "spring",
                "framework": "spring",
            },
            {
                "id": "spring_security_access_denied",
                "pattern": "org\\.springframework\\.security\\.access\\.AccessDeniedException(?:.*?Access is denied.*?)?",
                "type": "AccessDeniedException",
                "description": "Security access denied due to insufficient permissions",
                "root_cause": "spring_access_denied",
                "suggestion": "Verify the user has the required roles/authorities for the operation. Check your security configuration, especially method-level security expressions. Implement proper error handling for access denied situations.",
                "confidence": "high",
                "severity": "high",
                "category": "spring",
                "framework": "spring",
            },
            {
                "id": "spring_request_method_not_supported",
                "pattern": "org\\.springframework\\.web\\.HttpRequestMethodNotSupportedException(?:.*?method '([^']+)'.*?)?",
                "type": "HttpRequestMethodNotSupportedException",
                "description": "HTTP request method not supported by the endpoint",
                "root_cause": "spring_method_not_supported",
                "suggestion": "Ensure the client is using the correct HTTP method (GET, POST, PUT, DELETE, etc.) for the endpoint. Update the @RequestMapping to include the expected method or add an additional handler for the used method.",
                "confidence": "high",
                "severity": "medium",
                "category": "spring",
                "framework": "spring",
            },
        ]

    def _load_hibernate_rules(self) -> List[Dict[str, Any]]:
        """Load rules for Hibernate and JPA exceptions."""
        rules_file = (
            Path(__file__).parent.parent /
            "rules" /
            "hibernate" /
            "hibernate_errors.json"
        )

        if rules_file.exists():
            try:
                with open(rules_file, "r") as f:
                    data = json.load(f)
                    return data.get("rules", [])
            except Exception as e:
                logger.error(f"Error loading Hibernate rules from {rules_file}: {e}")

        # Fallback to built-in rules if file doesn't exist or can't be loaded
        return [
            {
                "id": "java_hibernate_lazy_loading",
                "pattern": "org\\.hibernate\\.LazyInitializationException(?:.*?could not initialize proxy.*?)?",
                "type": "LazyInitializationException",
                "description": "Attempted to access a lazy-loaded entity or collection outside of an active session",
                "root_cause": "java_hibernate_lazy_loading",
                "suggestion": "Use eager fetching (@ManyToOne(fetch=FetchType.EAGER)), initialize the collection within the session, or use Open Session in View pattern. Consider using DTOs to avoid lazy loading issues.",
                "confidence": "high",
                "severity": "high",
                "category": "framework",
                "framework": "hibernate",
            }
        ]

    def _load_android_rules(self) -> List[Dict[str, Any]]:
        """Load rules for Android platform exceptions."""
        rules_file = (
            Path(__file__).parent.parent / "rules" / "java" / "android_errors.json"
        )

        if rules_file.exists():
            try:
                with open(rules_file, "r") as f:
                    data = json.load(f)
                    return data.get("rules", [])
            except Exception as e:
                logger.error(f"Error loading Android rules from {rules_file}: {e}")

        # Fallback to empty list if file doesn't exist or can't be loaded
        logger.warning(f"Android rules file not found at {rules_file}")
        return []


class JavaPatchGenerator:
    """
    Generates patch solutions for Java exceptions.

    This class provides capabilities to generate code fixes for common Java errors,
    using templates and contextual information about the exception.
    """

    def __init__(self):
        """Initialize the Java patch generator."""
        self.templates_dir = (
            Path(__file__).parent.parent / "patch_generation" / "templates" / "java"
        )
        self.templates_dir.mkdir(exist_ok=True, parents=True)

        # Cache for loaded templates
        self.template_cache = {}

    def generate_patch(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a patch for a Java error based on analysis.

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
            "patch_id": f"java_{rule_id}",
            "patch_type": "suggestion",
            "language": "java",
            "framework": context.get("framework", ""),
            "description": analysis.get(
                "description",
                analysis.get("error_data", {}).get("message", "Java error fix"),
            ),
            "suggestion": analysis.get("suggestion", "No suggestion available"),
            "confidence": analysis.get("confidence", "low"),
            "severity": analysis.get("severity", "medium"),
            "root_cause": root_cause,
        }

        # Try to find a specific template for this root cause
        template_name = f"{root_cause}.java.template"
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
                        "suggestion_code": patch_code,  # Add suggestion_code for compatibility
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
            # Enhance the suggestion based on the rule
            if root_cause == "java_null_pointer":
                patch_result["suggestion_code"] = self._generate_null_check_suggestion(
                    analysis, context
                )
            elif root_cause == "java_index_out_of_bounds":
                patch_result["suggestion_code"] = (
                    self._generate_bounds_check_suggestion(analysis, context)
                )
            elif root_cause == "java_concurrent_modification":
                patch_result["suggestion_code"] = (
                    self._generate_concurrent_modification_suggestion(analysis, context)
                )
            elif root_cause == "java_resource_leak":
                patch_result["suggestion_code"] = (
                    self._generate_resource_leak_suggestion(analysis, context)
                )
                # Always update description for resource leaks to ensure it contains expected keywords
                patch_result["description"] = (
                    "Resource leak detected - use try-with-resources or ensure proper cleanup of AutoCloseable resources"
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

        # Additional variables based on error type
        if "java_null_pointer" in analysis.get("root_cause", ""):
            variables["NULL_CHECK_VAR"] = self._extract_null_variable(analysis, context)

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
                        "class": top_frame.get("class", ""),
                        "method": top_frame.get("function", ""),
                    }
                )

        return application_point

    def _generate_instructions(self, analysis: Dict[str, Any], patch_code: str) -> str:
        """Generate human-readable instructions for applying the patch."""
        root_cause = analysis.get("root_cause", "unknown")

        if "null_pointer" in root_cause:
            return (
                "Add null checks before accessing the variable. "
                f"Consider implementing this fix: {patch_code}"
            )
        elif "index_out_of_bounds" in root_cause:
            return (
                "Validate indices before accessing arrays or collections. "
                f"Implement bounds checking as shown: {patch_code}"
            )
        elif "concurrent_modification" in root_cause:
            return (
                "Modify your collection iteration approach to avoid concurrent modification. "
                f"Use the provided solution: {patch_code}"
            )
        else:
            return f"Apply the following fix to address the issue: {patch_code}"

    def _extract_null_variable(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Extract the likely null variable from an NPE."""
        # Check if variable_name is provided in context
        if "variable_name" in context:
            return context["variable_name"]

        message = analysis.get("error_data", {}).get("message", "")

        # Common NPE message patterns
        if "trying to access" in message and "of a null object" in message:
            # Android-style NPE
            parts = message.split("trying to access")
            if len(parts) > 1:
                parts = parts[1].split("of a null")
                if len(parts) > 0:
                    return parts[0].strip()

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
        return "object"

    def _generate_null_check_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for null checking."""
        var_name = self._extract_null_variable(analysis, context)

        return f"""// Add null check before accessing {var_name}
if ({var_name} == null) {{
    // Handle null case - either return early, throw custom exception, or provide default
    // return; // Early return
    // throw new IllegalArgumentException("{var_name} must not be null");
    // {var_name} = getDefaultValue(); // Provide default
}}
// Then proceed with original code that uses {var_name}
"""

    def _generate_bounds_check_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for bounds checking."""
        return """// Add bounds check before accessing array/collection elements
if (index >= 0 && index < array.length) {
    // Safe to access array[index]
    value = array[index];
} else {
    // Handle invalid index - either skip, log, throw custom exception, or use default
    // throw new IllegalArgumentException("Index out of bounds: " + index);
    // log.warn("Skipping invalid index: {}", index);
    // Use a default value instead
}
"""

    def _generate_concurrent_modification_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for avoiding concurrent modification."""
        return """// Option 1: Use CopyOnWriteArrayList for thread safety
List<Type> threadSafeList = new CopyOnWriteArrayList<>(originalList);
for (Type item : threadSafeList) {
    // Safe to modify originalList here
}

// Option 2: Use a copy of the collection for iteration
List<Type> copy = new ArrayList<>(originalList);
for (Type item : copy) {
    originalList.remove(item); // Safe because we're iterating over a copy
}

// Option 3: Use Iterator's remove method
Iterator<Type> iterator = list.iterator();
while (iterator.hasNext()) {
    Type item = iterator.next();
    if (shouldRemove(item)) {
        iterator.remove(); // Safe way to remove during iteration
    }
}
"""

    def _generate_resource_leak_suggestion(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Generate a code snippet for proper resource handling."""
        resource_type = context.get("resource_type", "AutoCloseable")
        return f"""// Use try-with-resources to ensure proper resource cleanup
try ({resource_type} resource = new {resource_type}(...)) {{
    // Use the resource
    // Resource will be automatically closed when leaving the try block
}} catch (IOException e) {{
    // Handle exception
    e.printStackTrace();
}}

// Or if you need to keep the resource open longer:
{resource_type} resource = null;
try {{
    resource = new {resource_type}(...);
    // Use the resource
}} finally {{
    if (resource != null) {{
        try {{
            resource.close();
        }} catch (IOException e) {{
            // Log but don't rethrow from finally
            e.printStackTrace();
        }}
    }}
}}"""


class JavaLanguagePlugin(LanguagePlugin):
    """
    Java language plugin for Homeostasis.

    Provides comprehensive error analysis and fix generation for Java applications,
    including support for Spring, Jakarta EE, Hibernate, and Android frameworks.
    """

    VERSION = "0.2.0"
    AUTHOR = "Homeostasis Contributors"

    def __init__(self):
        """Initialize the Java language plugin."""
        self.adapter = JavaErrorAdapter()
        self.exception_handler = JavaExceptionHandler()
        self.patch_generator = JavaPatchGenerator()

    def get_language_id(self) -> str:
        """Get the language identifier."""
        return "java"

    def get_language_name(self) -> str:
        """Get the language name."""
        return "Java"

    def get_language_version(self) -> str:
        """Get the language version."""
        return "8+"

    def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a Java error.

        Args:
            error_data: Java error data

        Returns:
            Analysis results
        """
        # First, normalize the error
        if "language" not in error_data or error_data["language"] != "java":
            standard_error = self.normalize_error(error_data)
        else:
            standard_error = error_data

        # Use the exception handler to analyze the error
        return self.exception_handler.analyze_exception(standard_error)

    def normalize_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a Java error to the standard format.

        Args:
            error_data: Java error data

        Returns:
            Error data in the standard format
        """
        return self.adapter.to_standard_format(error_data)

    def denormalize_error(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Java format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Java format
        """
        return self.adapter.from_standard_format(standard_error)

    def generate_fix(
        self, analysis: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a fix for a Java error.

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
        return [
            "spring",
            "spring-boot",
            "struts",
            "play",
            "jakarta",
            "hibernate",
            "android",
            "base",
        ]

    def can_handle(self, error_data: Dict[str, Any]) -> bool:
        """
        Check if this plugin can handle the given error.

        Args:
            error_data: Error data to check

        Returns:
            True if the plugin can handle the error, False otherwise
        """
        # Check if language is explicitly set to Java
        if error_data.get("language", "").lower() == "java":
            return True

        # Check if error_type contains Java patterns
        error_type = error_data.get("error_type", "")
        if error_type:
            # Check for Java package patterns
            if error_type.startswith("java.") or error_type.startswith("javax."):
                return True
            # Check for common Java framework patterns
            if error_type.startswith("org.springframework."):
                return True
            if error_type.startswith("org.hibernate."):
                return True
            if error_type.startswith("jakarta."):
                return True
            if error_type.startswith("android."):
                return True
            if error_type.startswith("com.android."):
                return True

        # Check if file extension is Java
        file_path = error_data.get("file", "")
        if file_path and file_path.endswith(".java"):
            return True

        # Check stack trace for Java patterns
        stack_trace = error_data.get("stack_trace", error_data.get("stacktrace", []))
        if stack_trace:
            # If it's a string, check for Java patterns
            if isinstance(stack_trace, str):
                if "at " in stack_trace and ".java:" in stack_trace:
                    return True
            # If it's a list, check the frames
            elif isinstance(stack_trace, list) and stack_trace:
                for frame in stack_trace:
                    if isinstance(frame, dict):
                        if frame.get("file", "").endswith(".java"):
                            return True
                    elif isinstance(frame, str):
                        if ".java:" in frame or "at " in frame:
                            return True

        # Check for Java-specific error patterns in message
        message = error_data.get("message", "")
        if message:
            java_patterns = [
                "java.lang.",
                "java.util.",
                "java.io.",
                "Exception in thread",
                "at com.",
                "at org.",
                ".java:",
                "ClassNotFoundException",
                "NoClassDefFoundError",
            ]
            for pattern in java_patterns:
                if pattern in message:
                    return True

        return False


# Register this plugin
register_plugin(JavaLanguagePlugin())
