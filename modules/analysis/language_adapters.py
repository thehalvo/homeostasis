"""
Multi-Language Error Adapters

This module provides adapters for converting language-specific error formats
to the standardized Homeostasis error schema and back. It enables cross-language
error analysis and pattern matching.
"""

import json
import logging
import os
import re
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set up logging
logger = logging.getLogger(__name__)

# Path to the schema directory
SCHEMA_DIR = Path(__file__).parent / "schemas"


class ErrorSchemaValidator:
    """Validator for the Homeostasis error schema."""

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the validator with the error schema.

        Args:
            schema_path: Path to the schema file, default to the standard error_schema.json
        """
        if schema_path is None:
            schema_path = SCHEMA_DIR / "error_schema.json"

        with open(schema_path, "r") as f:
            self.schema = json.load(f)

        try:
            # Try to import jsonschema for validation
            import jsonschema

            self.jsonschema = jsonschema
            self.can_validate = True
        except ImportError:
            logger.warning(
                "jsonschema package not available, schema validation disabled"
            )
            self.jsonschema = None
            self.can_validate = False

    def validate(self, error_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate error data against the schema.

        Args:
            error_data: Error data to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.can_validate:
            logger.warning("Schema validation skipped (jsonschema not available)")
            return True, None

        try:
            self.jsonschema.validate(error_data, self.schema)
            return True, None
        except self.jsonschema.exceptions.ValidationError as e:
            return False, str(e)


class LanguageAdapter:
    """Base class for language-specific error adapters."""

    def __init__(self):
        """Initialize the adapter."""
        self.validator = ErrorSchemaValidator()

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert language-specific error data to the standard format.

        Args:
            error_data: Language-specific error data

        Returns:
            Error data in the standard format
        """
        raise NotImplementedError("Subclasses must implement to_standard_format")

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to language-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the language-specific format
        """
        raise NotImplementedError("Subclasses must implement from_standard_format")

    def validate_error(self, error_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate error data against the schema.

        Args:
            error_data: Error data to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.validator.validate(error_data)


class PythonErrorAdapter(LanguageAdapter):
    """Adapter for Python error formats."""

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Python error data to the standard format.

        Args:
            error_data: Python error data

        Returns:
            Error data in the standard format
        """
        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "python",
            "language_version": error_data.get("python_version", ""),
            "error_type": error_data.get("exception_type", ""),
            "message": error_data.get("message", ""),
        }

        # Add Python version if available
        if "python_version" in error_data:
            standard_error["language_version"] = error_data["python_version"]

        # Handle stack trace
        if "traceback" in error_data:
            standard_error["stack_trace"] = error_data["traceback"]

        # Handle exception details for more structured data
        if "error_details" in error_data:
            details = error_data["error_details"]

            if "exception_type" in details and not standard_error["error_type"]:
                standard_error["error_type"] = details["exception_type"]

            if "message" in details and not standard_error["message"]:
                standard_error["message"] = details["message"]

            # Add detailed frames if available
            if "detailed_frames" in details and isinstance(
                details["detailed_frames"], list
            ):
                standard_error["stack_trace"] = details["detailed_frames"]

        # Add framework information if available
        if "framework" in error_data:
            standard_error["framework"] = error_data["framework"]

            if "framework_version" in error_data:
                standard_error["framework_version"] = error_data["framework_version"]

        # Add request information if available
        if "request" in error_data:
            standard_error["request"] = error_data["request"]

        # Add any additional context
        if "context" in error_data:
            standard_error["context"] = error_data["context"]

        # Add severity if available
        if "level" in error_data:
            # Convert Python logging levels to the standard format
            level_map = {
                "DEBUG": "debug",
                "INFO": "info",
                "WARNING": "warning",
                "ERROR": "error",
                "CRITICAL": "critical",
                "FATAL": "fatal",
            }
            standard_error["severity"] = level_map.get(
                error_data["level"].upper(), "error"
            )

        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]

        # Add additional Python-specific data
        python_specific = {}
        for key, value in error_data.items():
            if key not in standard_error and key not in [
                "traceback",
                "error_details",
                "request",
                "context",
            ]:
                python_specific[key] = value

        if python_specific:
            standard_error["additional_data"] = python_specific

        return standard_error

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Python-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Python-specific format
        """
        # Create a Python error object
        python_error = {
            "timestamp": standard_error.get("timestamp", datetime.now().isoformat()),
            "exception_type": standard_error.get("error_type", ""),
            "message": standard_error.get("message", ""),
        }

        # Convert severity to Python logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "DEBUG",
                "info": "INFO",
                "warning": "WARNING",
                "error": "ERROR",
                "critical": "CRITICAL",
                "fatal": "FATAL",
            }
            python_error["level"] = level_map.get(
                standard_error["severity"].lower(), "ERROR"
            )

        # Convert stack trace to Python format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]

            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in Python traceback format
                    python_error["traceback"] = stack_trace
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to Python traceback format
                    python_error["traceback"] = self._convert_frames_to_traceback(
                        stack_trace
                    )

                    # Also add detailed frames
                    python_error["error_details"] = {
                        "exception_type": standard_error.get("error_type", ""),
                        "message": standard_error.get("message", ""),
                        "detailed_frames": stack_trace,
                    }

        # Add request information if available
        if "request" in standard_error:
            python_error["request"] = standard_error["request"]

        # Add context information if available
        if "context" in standard_error:
            python_error["context"] = standard_error["context"]

        # Add Python version if available
        if "language_version" in standard_error:
            python_error["python_version"] = standard_error["language_version"]

        # Add framework information if available
        if "framework" in standard_error:
            python_error["framework"] = standard_error["framework"]

            if "framework_version" in standard_error:
                python_error["framework_version"] = standard_error["framework_version"]

        # Add handled flag if available
        if "handled" in standard_error:
            python_error["handled"] = standard_error["handled"]

        # Add additional data if available
        if "additional_data" in standard_error:
            for key, value in standard_error["additional_data"].items():
                python_error[key] = value

        return python_error

    def _convert_frames_to_traceback(self, frames: List[Dict[str, Any]]) -> List[str]:
        """
        Convert structured stack frames to Python traceback format.

        Args:
            frames: Structured stack frames

        Returns:
            Python traceback strings
        """
        traceback_lines = ["Traceback (most recent call last):"]

        for frame in frames:
            file_path = frame.get("file", "<unknown>")
            line_num = frame.get("line", "?")
            function = frame.get("function", "<unknown>")

            # Format like Python traceback
            traceback_lines.append(
                f'  File "{file_path}", line {line_num}, in {function}'
            )

            # Add context if available
            if "context" in frame:
                traceback_lines.append(f"    {frame['context']}")

        # Add error type and message
        error_type = frames[0].get("error_type", "")
        message = frames[0].get("message", "")

        if error_type and message:
            traceback_lines.append(f"{error_type}: {message}")

        return traceback_lines


class JavaScriptErrorAdapter(LanguageAdapter):
    """Adapter for JavaScript error formats."""

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert JavaScript error data to the standard format.

        Args:
            error_data: JavaScript error data

        Returns:
            Error data in the standard format
        """
        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "javascript",
            "error_type": error_data.get("name", "Error"),
            "message": error_data.get("message", ""),
        }

        # Add JavaScript version/runtime if available
        if "runtime" in error_data:
            standard_error["runtime"] = error_data["runtime"]

            if "runtime_version" in error_data:
                standard_error["runtime_version"] = error_data["runtime_version"]

        # Handle stack trace
        if "stack" in error_data:
            # JavaScript stack traces are usually a single string
            if isinstance(error_data["stack"], str):
                # Split into lines
                stack_lines = error_data["stack"].split("\n")

                # Try to parse structured data from the stack trace
                parsed_frames = self._parse_js_stack_trace(stack_lines)

                if parsed_frames:
                    standard_error["stack_trace"] = parsed_frames
                else:
                    standard_error["stack_trace"] = stack_lines
            elif isinstance(error_data["stack"], list):
                # Already in a list format
                standard_error["stack_trace"] = error_data["stack"]

        # Add framework information if available
        if "framework" in error_data:
            standard_error["framework"] = error_data["framework"]

            if "framework_version" in error_data:
                standard_error["framework_version"] = error_data["framework_version"]

        # Add request information if available
        if "request" in error_data:
            standard_error["request"] = error_data["request"]

        # Add any additional context
        if "context" in error_data:
            standard_error["context"] = error_data["context"]

        # Add severity if available
        if "level" in error_data:
            # Map JS log levels to standard format
            level_map = {
                "debug": "debug",
                "info": "info",
                "warn": "warning",
                "warning": "warning",
                "error": "error",
                "fatal": "fatal",
                "critical": "critical",
            }
            standard_error["severity"] = level_map.get(
                error_data["level"].lower(), "error"
            )

        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]

        # Add additional JavaScript-specific data
        js_specific = {}
        for key, value in error_data.items():
            if key not in standard_error and key not in ["stack", "request", "context"]:
                js_specific[key] = value

        if js_specific:
            standard_error["additional_data"] = js_specific

        return standard_error

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to JavaScript-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the JavaScript-specific format
        """
        # Create a JavaScript error object
        js_error = {
            "timestamp": standard_error.get("timestamp", datetime.now().isoformat()),
            "name": standard_error.get("error_type", "Error"),
            "message": standard_error.get("message", ""),
        }

        # Convert severity to JavaScript logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "debug",
                "info": "info",
                "warning": "warn",
                "error": "error",
                "critical": "error",
                "fatal": "fatal",
            }
            js_error["level"] = level_map.get(
                standard_error["severity"].lower(), "error"
            )

        # Convert stack trace to JavaScript format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]

            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in JavaScript stack trace string format
                    js_error["stack"] = "\n".join(stack_trace)
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to JavaScript stack trace format
                    js_error["stack"] = self._convert_frames_to_js_stack(
                        standard_error["error_type"],
                        standard_error["message"],
                        stack_trace,
                    )

        # Add request information if available
        if "request" in standard_error:
            js_error["request"] = standard_error["request"]

        # Add context information if available
        if "context" in standard_error:
            js_error["context"] = standard_error["context"]

        # Add runtime information if available
        if "runtime" in standard_error:
            js_error["runtime"] = standard_error["runtime"]

            if "runtime_version" in standard_error:
                js_error["runtime_version"] = standard_error["runtime_version"]

        # Add framework information if available
        if "framework" in standard_error:
            js_error["framework"] = standard_error["framework"]

            if "framework_version" in standard_error:
                js_error["framework_version"] = standard_error["framework_version"]

        # Add handled flag if available
        if "handled" in standard_error:
            js_error["handled"] = standard_error["handled"]

        # Add additional data if available
        if "additional_data" in standard_error:
            for key, value in standard_error["additional_data"].items():
                js_error[key] = value

        return js_error

    def _parse_js_stack_trace(self, stack_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parse a JavaScript stack trace into structured frames.

        Args:
            stack_lines: JavaScript stack trace lines

        Returns:
            Structured frames or None if parsing fails
        """
        frames = []

        # Common JS stack trace patterns
        # Chrome: at functionName (file:line:column)
        # Node.js: at functionName (file:line:column)
        # Firefox: functionName@file:line:column
        frame_patterns = [
            r"\s*at\s+([^(]+)\s+\(([^:]+):(\d+):(\d+)\)",  # Chrome/Node with function
            r"\s*at\s+([^:]+):(\d+):(\d+)",  # Chrome/Node without function
            r"\s*([^@]+)@([^:]+):(\d+):(\d+)",  # Firefox
        ]

        try:
            for line in stack_lines:
                for pattern in frame_patterns:
                    match = re.search(pattern, line)
                    if match:
                        if len(match.groups()) == 4:  # Full match with function
                            frames.append(
                                {
                                    "function": match.group(1).strip(),
                                    "file": match.group(2),
                                    "line": int(match.group(3)),
                                    "column": int(match.group(4)),
                                }
                            )
                        elif len(match.groups()) == 3:  # Missing function name
                            frames.append(
                                {
                                    "file": match.group(1),
                                    "line": int(match.group(2)),
                                    "column": int(match.group(3)),
                                }
                            )
                        break

            return frames if frames else None
        except Exception as e:
            logger.debug(f"Failed to parse JS stack trace: {e}")
            return None

    def _convert_frames_to_js_stack(
        self, error_type: str, message: str, frames: List[Dict[str, Any]]
    ) -> str:
        """
        Convert structured frames to a JavaScript stack trace string.

        Args:
            error_type: Error type/name
            message: Error message
            frames: Structured frames

        Returns:
            JavaScript stack trace string
        """
        stack_lines = [f"{error_type}: {message}"]

        for frame in frames:
            file_path = frame.get("file", "<unknown>")
            line_num = frame.get("line", "?")
            column = frame.get("column", "?")
            function = frame.get("function", "<anonymous>")

            # Format like a Node.js/Chrome stack trace
            stack_lines.append(f"    at {function} ({file_path}:{line_num}:{column})")

        return "\n".join(stack_lines)


class JavaErrorAdapter(LanguageAdapter):
    """Adapter for Java error formats."""

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Java error data to the standard format.

        Args:
            error_data: Java error data

        Returns:
            Error data in the standard format
        """
        # Parse exception field if present
        error_type = ""
        message = ""

        if "exception" in error_data:
            # Parse combined exception string (e.g., "java.lang.NullPointerException: message")
            exception_str = error_data["exception"]
            if ": " in exception_str:
                error_type, message = exception_str.split(": ", 1)
            else:
                error_type = exception_str
        else:
            # Use separate fields
            error_type = error_data.get(
                "error_type",
                error_data.get("exception_class", error_data.get("type", "")),
            )
            message = error_data.get("message", "")

            # Check if message contains a raw Java stack trace
            if not error_type and message and "Exception in thread" in message:
                # Parse error type from raw stack trace
                # Format: Exception in thread "main" java.lang.NullPointerException
                match = re.search(r'Exception in thread[^"]*"[^"]*"\s+(\S+)', message)
                if match:
                    error_type = match.group(1)
                    # Extract the actual message (if any) after the exception type
                    remaining = message[match.end() :]
                    if remaining.startswith(": "):
                        actual_message = remaining[2:].split("\n")[0].strip()
                        if actual_message:
                            message = actual_message

        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "java",
            "error_type": error_type,
            "message": message,
        }

        # Add Java version if available
        if "java_version" in error_data:
            standard_error["language_version"] = error_data["java_version"]

        # Handle stack trace
        stack_data = error_data.get("stack_trace") or error_data.get("stacktrace")

        # If no stack_data but message contains stack trace, extract it
        if not stack_data and message and "\tat " in error_data.get("message", ""):
            # Extract stack trace from message
            full_message = error_data.get("message", "")
            stack_lines = []
            for line in full_message.split("\n"):
                if "\tat " in line:
                    stack_lines.append(line.strip())
            if stack_lines:
                stack_data = stack_lines

        if stack_data:
            # Java stack traces can be a string or a list
            if isinstance(stack_data, str):
                # Split into lines
                stack_lines = stack_data.split("\n")

                # Try to parse structured data from the stack trace
                parsed_frames = self._parse_java_stack_trace(stack_lines)

                if parsed_frames:
                    standard_error["stack_trace"] = parsed_frames
                else:
                    standard_error["stack_trace"] = stack_lines
            elif isinstance(stack_data, list):
                if all(isinstance(frame, dict) for frame in stack_data):
                    # Already in structured format
                    standard_error["stack_trace"] = stack_data
                else:
                    # List of strings - parse them
                    parsed_frames = self._parse_java_stack_trace(stack_data)
                    if parsed_frames:
                        standard_error["stack_trace"] = parsed_frames
                    else:
                        standard_error["stack_trace"] = stack_data
        else:
            # No stack trace provided - set empty list
            standard_error["stack_trace"] = []

        # Add file, line, column information if available
        if "file" in error_data:
            standard_error["file"] = error_data["file"]
        if "line" in error_data:
            standard_error["line"] = error_data["line"]
        if "column" in error_data:
            standard_error["column"] = error_data["column"]

        # Add framework information if available
        if "framework" in error_data:
            standard_error["framework"] = error_data["framework"]

            if "framework_version" in error_data:
                standard_error["framework_version"] = error_data["framework_version"]
        else:
            # Try to detect framework from error type and stack trace
            if error_type:
                if "springframework" in error_type:
                    standard_error["framework"] = "spring"
                elif "hibernate" in error_type.lower():
                    standard_error["framework"] = "hibernate"
                elif "play" in error_type.lower():
                    standard_error["framework"] = "play"
                elif "struts" in error_type.lower():
                    standard_error["framework"] = "struts"

            if "framework" not in standard_error and stack_data:
                # Check stack trace for framework indicators
                stack_str = str(stack_data)
                if "org.springframework" in stack_str:
                    standard_error["framework"] = "spring"
                elif "javax.servlet" in stack_str or "jakarta.servlet" in stack_str:
                    standard_error["framework"] = "servlet"
                elif "org.hibernate" in stack_str:
                    standard_error["framework"] = "hibernate"
                elif "play.api" in stack_str or "play.mvc" in stack_str:
                    standard_error["framework"] = "play"
                elif "org.apache.struts" in stack_str:
                    standard_error["framework"] = "struts"

        # Add request information if available
        if "request" in error_data:
            standard_error["request"] = error_data["request"]

        # Add any additional context
        if "context" in error_data:
            standard_error["context"] = error_data["context"]

        # Add severity if available
        if "level" in error_data:
            # Map Java log levels to standard format
            level_map = {
                "finest": "debug",
                "finer": "debug",
                "fine": "debug",
                "config": "info",
                "info": "info",
                "warning": "warning",
                "severe": "error",
            }
            standard_error["severity"] = level_map.get(
                error_data["level"].lower(), "error"
            )

        # Add runtime if available
        if "jvm" in error_data:
            standard_error["runtime"] = "JVM"
            standard_error["runtime_version"] = error_data["jvm"]

        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]

        # Add caused_by information if available
        if "caused_by" in error_data:
            # Recursively convert caused_by to standard format
            standard_error["caused_by"] = self.to_standard_format(
                error_data["caused_by"]
            )

        # Add suppressed exceptions if available (from try-with-resources)
        if "suppressed" in error_data:
            standard_error["suppressed"] = []
            for suppressed in error_data["suppressed"]:
                # Recursively convert each suppressed exception to standard format
                standard_error["suppressed"].append(self.to_standard_format(suppressed))

        # Add compilation errors if available
        if "errors" in error_data and error_type == "CompilationError":
            standard_error["errors"] = error_data["errors"]

        # Add additional Java-specific data
        java_specific = {}
        for key, value in error_data.items():
            if key not in standard_error and key not in [
                "stack_trace",
                "request",
                "context",
            ]:
                java_specific[key] = value

        if java_specific:
            standard_error["additional_data"] = java_specific

        return standard_error

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Java-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Java-specific format
        """
        # Create a Java error object
        java_error = {
            "timestamp": standard_error.get("timestamp", datetime.now().isoformat()),
            "exception": standard_error.get("error_type", "java.lang.Exception"),
            "message": standard_error.get("message", ""),
        }

        # Convert severity to Java logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "fine",
                "info": "info",
                "warning": "warning",
                "error": "severe",
                "critical": "severe",
                "fatal": "severe",
            }
            java_error["level"] = level_map.get(
                standard_error["severity"].lower(), "severe"
            )

        # Convert stack trace to Java format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]

            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in Java stack trace string format
                    java_error["stacktrace"] = stack_trace
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to Java stack trace format
                    stacktrace_lines = []
                    for frame in stack_trace:
                        class_name = frame.get("class", "Unknown")
                        method = frame.get("function", "unknown")
                        file = frame.get("file", "Unknown.java")
                        line_num = frame.get("line", "?")

                        # Format like a Java stack trace
                        stacktrace_lines.append(
                            f"at {class_name}.{method}({file}:{line_num})"
                        )

                    java_error["stacktrace"] = stacktrace_lines

        # Add request information if available
        if "request" in standard_error:
            java_error["request"] = standard_error["request"]

        # Add context information if available
        if "context" in standard_error:
            java_error["context"] = standard_error["context"]

        # Add Java version if available
        if "language_version" in standard_error:
            java_error["java_version"] = standard_error["language_version"]

        # Add runtime information if available
        if "runtime" in standard_error and standard_error["runtime"] == "JVM":
            java_error["jvm"] = standard_error.get("runtime_version", "")

        # Add framework information if available
        if "framework" in standard_error:
            java_error["framework"] = standard_error["framework"]

            if "framework_version" in standard_error:
                java_error["framework_version"] = standard_error["framework_version"]

        # Add handled flag if available
        if "handled" in standard_error:
            java_error["handled"] = standard_error["handled"]

        # Add additional data if available
        if "additional_data" in standard_error:
            for key, value in standard_error["additional_data"].items():
                java_error[key] = value

        return java_error

    def _parse_java_stack_trace(
        self, stack_data: Union[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Parse a Java stack trace into structured frames.

        Args:
            stack_data: Java stack trace as string or list of lines

        Returns:
            Structured frames or None if parsing fails
        """
        # Convert to list if string
        if isinstance(stack_data, str):
            stack_lines = stack_data.split("\n")
        else:
            stack_lines = stack_data
        frames = []

        # Java stack trace pattern: at package.Class.method(File.java:line)
        # Extended to handle com.example.StringProcessor.processString(StringProcessor.java:42)
        frame_pattern = r"\s*at\s+((?:[a-zA-Z0-9_]+\.)*?)([a-zA-Z0-9_$]+)\.([a-zA-Z0-9_$]+)\(([^:]+):(\d+)\)"

        try:
            for line in stack_lines:
                match = re.search(frame_pattern, line)
                if match:
                    package_prefix = (
                        match.group(1).rstrip(".") if match.group(1) else ""
                    )
                    class_name = match.group(2)
                    method = match.group(3)
                    file = match.group(4)
                    line_num = int(match.group(5))

                    frames.append(
                        {
                            "package": package_prefix,
                            "class": class_name,
                            "function": method,
                            "file": file,
                            "line": line_num,
                        }
                    )

            return frames if frames else None
        except Exception as e:
            logger.debug(f"Failed to parse Java stack trace: {e}")
            return None

    def _convert_frames_to_java_stack(
        self, error_type: str, message: str, frames: List[Dict[str, Any]]
    ) -> str:
        """
        Convert structured frames to a Java stack trace string.

        Args:
            error_type: Error type/name
            message: Error message
            frames: Structured frames

        Returns:
            Java stack trace string
        """
        stack_lines = [f"{error_type}: {message}"]

        for frame in frames:
            package = frame.get("package", "")
            class_name = frame.get("class", "Unknown")
            method = frame.get("function", "unknown")
            file = frame.get("file", "Unknown.java")
            line_num = frame.get("line", "?")

            # Construct the full class name with package
            full_class = f"{package}.{class_name}" if package else class_name

            # Format like a Java stack trace
            stack_lines.append(f"    at {full_class}.{method}({file}:{line_num})")

        return "\n".join(stack_lines)


class GoErrorAdapter(LanguageAdapter):
    """Adapter for Go error formats."""

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Go error data to the standard format.

        Args:
            error_data: Go error data

        Returns:
            Error data in the standard format
        """
        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "go",
            "error_type": error_data.get("type", error_data.get("error_type", "")),
            "message": error_data.get("message", ""),
        }

        # Add Go version if available
        if "go_version" in error_data:
            standard_error["language_version"] = error_data["go_version"]

        # Add file location information
        if "file" in error_data:
            standard_error["file"] = error_data["file"]
        if "line" in error_data:
            standard_error["line"] = error_data["line"]
        if "column" in error_data:
            standard_error["column"] = error_data["column"]

        # Handle stack trace (check both stack_trace and stacktrace)
        if "stack_trace" in error_data or "stacktrace" in error_data:
            stack_trace_data = error_data.get(
                "stack_trace", error_data.get("stacktrace")
            )
            # Go stack traces can be a string or a list
            if isinstance(stack_trace_data, str):
                # Split into lines
                stack_lines = stack_trace_data.split("\n")

                # Try to parse structured data from the stack trace
                parsed_frames = self._parse_go_stack_trace(stack_lines)

                if parsed_frames:
                    standard_error["stack_trace"] = parsed_frames
                else:
                    standard_error["stack_trace"] = stack_lines
            elif isinstance(stack_trace_data, list):
                if all(isinstance(frame, dict) for frame in stack_trace_data):
                    # Already in structured format
                    standard_error["stack_trace"] = stack_trace_data
                else:
                    # List of strings - try to parse them
                    parsed_frames = self._parse_go_stack_trace(stack_trace_data)
                    if parsed_frames:
                        standard_error["stack_trace"] = parsed_frames
                    else:
                        standard_error["stack_trace"] = stack_trace_data

        # Add framework information if available
        if "framework" in error_data:
            standard_error["framework"] = error_data["framework"]

            if "framework_version" in error_data:
                standard_error["framework_version"] = error_data["framework_version"]

        # Add request information if available
        if "request" in error_data:
            standard_error["request"] = error_data["request"]

        # Add any additional context
        if "context" in error_data:
            standard_error["context"] = error_data["context"]

        # Add severity if available
        if "level" in error_data:
            # Map Go log levels to standard format
            level_map = {
                "debug": "debug",
                "info": "info",
                "warning": "warning",
                "warn": "warning",
                "error": "error",
                "fatal": "fatal",
                "panic": "critical",
            }
            standard_error["severity"] = level_map.get(
                error_data["level"].lower(), "error"
            )

        # Add runtime if available
        if "runtime" in error_data:
            standard_error["runtime"] = error_data["runtime"]

            if "runtime_version" in error_data:
                standard_error["runtime_version"] = error_data["runtime_version"]

        # Add goroutine information if available
        if "goroutine_id" in error_data:
            if "additional_data" not in standard_error:
                standard_error["additional_data"] = {}
            standard_error["additional_data"]["goroutine_id"] = error_data[
                "goroutine_id"
            ]

        # Add goroutines list for deadlock errors
        if "goroutines" in error_data:
            standard_error["goroutines"] = error_data["goroutines"]

        # Add signal information if available
        if "signal" in error_data:
            standard_error["signal"] = error_data["signal"]

        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]

        # Add additional Go-specific data
        go_specific = {}
        for key, value in error_data.items():
            if (
                key not in standard_error
                and key not in ["stack_trace", "request", "context"]
                and not key.startswith("_")
            ):
                go_specific[key] = value

        if go_specific:
            if "additional_data" in standard_error:
                standard_error["additional_data"].update(go_specific)
            else:
                standard_error["additional_data"] = go_specific

        return standard_error

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Go-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Go-specific format
        """
        # Create a Go error object
        go_error = {
            "timestamp": standard_error.get("timestamp", datetime.now().isoformat()),
            "type": standard_error.get("error_type", "error"),
            "message": standard_error.get("message", ""),
        }

        # Add file location information
        if "file" in standard_error:
            go_error["file"] = standard_error["file"]
        if "line" in standard_error:
            go_error["line"] = standard_error["line"]
        if "column" in standard_error:
            go_error["column"] = standard_error["column"]

        # Convert severity to Go logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "debug",
                "info": "info",
                "warning": "warn",
                "error": "error",
                "critical": "panic",
                "fatal": "fatal",
            }
            go_error["level"] = level_map.get(
                standard_error["severity"].lower(), "error"
            )

        # Convert stack trace to Go format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]

            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in Go stack trace string format
                    go_error["stack_trace"] = "\n".join(stack_trace)
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to Go stack trace format
                    go_error["stack_trace"] = self._convert_frames_to_go_stack(
                        standard_error.get("error_type", "error"),
                        standard_error.get("message", ""),
                        stack_trace,
                    )
                    # Also keep the structured version
                    go_error["stack_frames"] = stack_trace

        # Add request information if available
        if "request" in standard_error:
            go_error["request"] = standard_error["request"]

        # Add context information if available
        if "context" in standard_error:
            go_error["context"] = standard_error["context"]

        # Add Go version if available
        if "language_version" in standard_error:
            go_error["go_version"] = standard_error["language_version"]

        # Add framework information if available
        if "framework" in standard_error:
            go_error["framework"] = standard_error["framework"]

            if "framework_version" in standard_error:
                go_error["framework_version"] = standard_error["framework_version"]

        # Add runtime information if available
        if "runtime" in standard_error:
            go_error["runtime"] = standard_error["runtime"]

            if "runtime_version" in standard_error:
                go_error["runtime_version"] = standard_error["runtime_version"]

        # Add handled flag if available
        if "handled" in standard_error:
            go_error["handled"] = standard_error["handled"]

        # Add additional data if available
        if "additional_data" in standard_error:
            for key, value in standard_error["additional_data"].items():
                # Extract goroutine_id as a top-level field
                if key == "goroutine_id":
                    go_error["goroutine_id"] = value
                else:
                    go_error[key] = value

        return go_error

    def _parse_go_stack_trace(self, stack_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parse a Go stack trace into structured frames.

        Args:
            stack_lines: Go stack trace lines

        Returns:
            Structured frames or None if parsing fails
        """
        frames = []
        goroutine_id = None
        current_function = None

        # Common Go stack trace patterns
        # Goroutine line: goroutine N [status]:
        # Function line: function_name(args)
        #    path/to/file.go:line
        goroutine_pattern = r"goroutine (\d+) \[([^\]]+)\]:"
        function_pattern = r"([^\(]+)\(.*\)"
        file_pattern = r"\s*([^:]+):(\d+)(?:\s+(\+0x[0-9a-f]+))?"

        try:
            i = 0
            while i < len(stack_lines):
                line = stack_lines[i]

                # Check for goroutine header
                goroutine_match = re.search(goroutine_pattern, line)
                if goroutine_match:
                    goroutine_id = int(goroutine_match.group(1))
                    i += 1
                    continue

                # Check for function name
                func_match = re.search(function_pattern, line)
                if func_match:
                    current_function = func_match.group(1).strip()

                    # Check if next line is the file location
                    if i + 1 < len(stack_lines):
                        file_line = stack_lines[i + 1]
                        file_match = re.search(file_pattern, file_line)

                        if file_match:
                            file_path = file_match.group(1)
                            line_num = int(file_match.group(2))

                            # Extract package and function names but keep full name for function field
                            func_parts = current_function.split(".")
                            package = (
                                ".".join(func_parts[:-1]) if len(func_parts) > 1 else ""
                            )

                            frames.append(
                                {
                                    "package": package,
                                    "function": current_function,  # Keep full function name including package
                                    "file": file_path,
                                    "line": line_num,
                                    "goroutine_id": goroutine_id,
                                }
                            )

                            i += 2  # Skip the file line
                            continue

                i += 1

            return frames if frames else None
        except Exception as e:
            logger.debug(f"Failed to parse Go stack trace: {e}")
            return None

    def _convert_frames_to_go_stack(
        self, error_type: str, message: str, frames: List[Dict[str, Any]]
    ) -> str:
        """
        Convert structured frames to a Go stack trace string.

        Args:
            error_type: Error type/name
            message: Error message
            frames: Structured frames

        Returns:
            Go stack trace string
        """
        # Extract goroutine ID (use the first one available)
        goroutine_id = None
        for frame in frames:
            if "goroutine_id" in frame:
                goroutine_id = frame["goroutine_id"]
                break

        # Start with the error message
        stack_lines = [f"{error_type}: {message}"]

        # Add goroutine header if available
        if goroutine_id is not None:
            stack_lines.insert(0, f"goroutine {goroutine_id} [running]:")

        # Add frames
        for frame in frames:
            package = frame.get("package", "")
            func_name = frame.get("function", "unknown")
            file_path = frame.get("file", "unknown")
            line_num = frame.get("line", 0)

            # Format like a Go stack trace
            full_func = f"{package}.{func_name}" if package else func_name
            stack_lines.append(f"{full_func}()")
            stack_lines.append(f"\t{file_path}:{line_num}")

        return "\n".join(stack_lines)


class RubyErrorAdapter(LanguageAdapter):
    """Adapter for Ruby error formats."""

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Ruby error data to the standard format.

        Args:
            error_data: Ruby error data

        Returns:
            Error data in the standard format
        """
        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "ruby",
            "error_type": error_data.get("exception_class", ""),
            "message": error_data.get("message", ""),
        }

        # Add Ruby version if available
        if "ruby_version" in error_data:
            standard_error["language_version"] = error_data["ruby_version"]

        # Handle stack trace
        if "backtrace" in error_data:
            # Ruby backtraces can be a string or a list
            if isinstance(error_data["backtrace"], str):
                # Split into lines
                stack_lines = error_data["backtrace"].split("\n")

                # Try to parse structured data from the stack trace
                parsed_frames = self._parse_ruby_stack_trace(stack_lines)

                if parsed_frames:
                    standard_error["stack_trace"] = parsed_frames
                else:
                    standard_error["stack_trace"] = stack_lines
            elif isinstance(error_data["backtrace"], list):
                if all(isinstance(frame, dict) for frame in error_data["backtrace"]):
                    # Already in structured format
                    standard_error["stack_trace"] = error_data["backtrace"]
                else:
                    # List of strings
                    standard_error["stack_trace"] = error_data["backtrace"]

        # Add framework information if available
        if "framework" in error_data:
            standard_error["framework"] = error_data["framework"]

            if "framework_version" in error_data:
                standard_error["framework_version"] = error_data["framework_version"]

        # Add request information if available
        if "request" in error_data:
            standard_error["request"] = error_data["request"]

        # Add any additional context
        if "context" in error_data:
            standard_error["context"] = error_data["context"]

        # Add severity if available
        if "level" in error_data:
            # Map Ruby log levels to standard format
            level_map = {
                "debug": "debug",
                "info": "info",
                "warn": "warning",
                "warning": "warning",
                "error": "error",
                "fatal": "fatal",
            }
            standard_error["severity"] = level_map.get(
                error_data["level"].lower(), "error"
            )

        # Add runtime if available
        if "runtime" in error_data:
            standard_error["runtime"] = error_data["runtime"]

            if "runtime_version" in error_data:
                standard_error["runtime_version"] = error_data["runtime_version"]

        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]

        # Add additional Ruby-specific data
        ruby_specific = {}
        for key, value in error_data.items():
            if key not in standard_error and key not in [
                "backtrace",
                "request",
                "context",
            ]:
                ruby_specific[key] = value

        if ruby_specific:
            standard_error["additional_data"] = ruby_specific

        return standard_error

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Ruby-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Ruby-specific format
        """
        # Create a Ruby error object
        ruby_error = {
            "timestamp": standard_error.get("timestamp", datetime.now().isoformat()),
            "exception_class": standard_error.get("error_type", "RuntimeError"),
            "message": standard_error.get("message", ""),
        }

        # Convert severity to Ruby logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "debug",
                "info": "info",
                "warning": "warn",
                "error": "error",
                "critical": "fatal",
                "fatal": "fatal",
            }
            ruby_error["level"] = level_map.get(
                standard_error["severity"].lower(), "error"
            )

        # Convert stack trace to Ruby format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]

            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in Ruby stack trace string format
                    ruby_error["backtrace"] = stack_trace
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to Ruby stack trace format
                    ruby_error["backtrace"] = self._convert_frames_to_ruby_backtrace(
                        stack_trace
                    )
                    # Also keep the structured version
                    ruby_error["structured_backtrace"] = stack_trace

        # Add request information if available
        if "request" in standard_error:
            ruby_error["request"] = standard_error["request"]

        # Add context information if available
        if "context" in standard_error:
            ruby_error["context"] = standard_error["context"]

        # Add Ruby version if available
        if "language_version" in standard_error:
            ruby_error["ruby_version"] = standard_error["language_version"]

        # Add framework information if available
        if "framework" in standard_error:
            ruby_error["framework"] = standard_error["framework"]

            if "framework_version" in standard_error:
                ruby_error["framework_version"] = standard_error["framework_version"]

        # Add runtime information if available
        if "runtime" in standard_error:
            ruby_error["runtime"] = standard_error["runtime"]

            if "runtime_version" in standard_error:
                ruby_error["runtime_version"] = standard_error["runtime_version"]

        # Add handled flag if available
        if "handled" in standard_error:
            ruby_error["handled"] = standard_error["handled"]

        # Add additional data if available
        if "additional_data" in standard_error:
            for key, value in standard_error["additional_data"].items():
                ruby_error[key] = value

        return ruby_error

    def _parse_ruby_stack_trace(self, stack_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parse a Ruby stack trace into structured frames.

        Args:
            stack_lines: Ruby stack trace lines

        Returns:
            Structured frames or None if parsing fails
        """
        frames = []

        # Ruby stack trace patterns:
        # app/models/user.rb:32:in `find_by_id'
        # /path/to/file.rb:123:in `method_name'
        # /gems/activerecord-6.1.0/lib/active_record/relation.rb:17:in `block in find'
        frame_pattern = r"([^:]+):(\d+)(?::in\s+`([^\']+)\')?"

        try:
            for line in stack_lines:
                match = re.search(frame_pattern, line)
                if match:
                    file_path = match.group(1)
                    line_num = int(match.group(2))
                    method_name = match.group(3) if match.groups()[2] else ""

                    # Attempt to extract class/module information from the file path
                    module_name = ""
                    file_name = os.path.basename(file_path)
                    if file_name.endswith(".rb"):
                        module_parts = file_name[:-3].split("_")
                        module_name = "".join(part.title() for part in module_parts)

                    frames.append(
                        {
                            "file": file_path,
                            "line": line_num,
                            "function": method_name,
                            "module": module_name,
                        }
                    )

            return frames if frames else None
        except Exception as e:
            logger.debug(f"Failed to parse Ruby stack trace: {e}")
            return None

    def _convert_frames_to_ruby_backtrace(
        self, frames: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Convert structured frames to a Ruby backtrace.

        Args:
            frames: Structured frames

        Returns:
            Ruby backtrace lines
        """
        backtrace_lines = []

        for frame in frames:
            file_path = frame.get("file", "<unknown>")
            line_num = frame.get("line", "?")
            method_name = frame.get("function", "<unknown>")

            # Format like a Ruby backtrace
            if method_name:
                backtrace_lines.append(f"{file_path}:{line_num}:in `{method_name}'")
            else:
                backtrace_lines.append(f"{file_path}:{line_num}")

        return backtrace_lines


class RustErrorAdapter(LanguageAdapter):
    """Adapter for Rust error formats."""

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Rust error data to the standard format.

        Args:
            error_data: Rust error data

        Returns:
            Error data in the standard format
        """
        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "rust",
            "error_type": error_data.get("error_type", ""),
            "message": error_data.get("message", ""),
        }

        # Add Rust version if available
        if "rust_version" in error_data:
            standard_error["language_version"] = error_data["rust_version"]

        # Handle stack trace
        if "backtrace" in error_data:
            # Rust backtraces can be a string or a list
            if isinstance(error_data["backtrace"], str):
                # Split into lines
                stack_lines = error_data["backtrace"].split("\n")

                # Try to parse structured data from the stack trace
                parsed_frames = self._parse_rust_stack_trace(stack_lines)

                if parsed_frames:
                    standard_error["stack_trace"] = parsed_frames
                else:
                    standard_error["stack_trace"] = stack_lines
            elif isinstance(error_data["backtrace"], list):
                if all(isinstance(frame, dict) for frame in error_data["backtrace"]):
                    # Already in structured format
                    standard_error["stack_trace"] = error_data["backtrace"]
                else:
                    # List of strings
                    standard_error["stack_trace"] = error_data["backtrace"]

        # Add framework information if available
        if "framework" in error_data:
            standard_error["framework"] = error_data["framework"]

            if "framework_version" in error_data:
                standard_error["framework_version"] = error_data["framework_version"]

        # Add request information if available
        if "request" in error_data:
            standard_error["request"] = error_data["request"]

        # Add any additional context
        if "context" in error_data:
            standard_error["context"] = error_data["context"]

        # Add severity if available
        if "level" in error_data:
            # Map Rust log levels to standard format
            level_map = {
                "trace": "debug",
                "debug": "debug",
                "info": "info",
                "warn": "warning",
                "warning": "warning",
                "error": "error",
                "fatal": "fatal",
            }
            standard_error["severity"] = level_map.get(
                error_data["level"].lower(), "error"
            )

        # Add panic information if available
        if "is_panic" in error_data:
            if "additional_data" not in standard_error:
                standard_error["additional_data"] = {}
            standard_error["additional_data"]["is_panic"] = error_data["is_panic"]

        # Add thread information if available
        if "thread" in error_data:
            if "additional_data" not in standard_error:
                standard_error["additional_data"] = {}
            standard_error["additional_data"]["thread"] = error_data["thread"]

        # Add runtime if available
        if "runtime" in error_data:
            standard_error["runtime"] = error_data["runtime"]

            if "runtime_version" in error_data:
                standard_error["runtime_version"] = error_data["runtime_version"]

        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]

        # Add additional Rust-specific data
        rust_specific = {}
        for key, value in error_data.items():
            if key not in standard_error and key not in [
                "backtrace",
                "request",
                "context",
                "thread",
                "is_panic",
            ]:
                rust_specific[key] = value

        if rust_specific:
            if "additional_data" in standard_error:
                standard_error["additional_data"].update(rust_specific)
            else:
                standard_error["additional_data"] = rust_specific

        return standard_error

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Rust-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Rust-specific format
        """
        # Create a Rust error object
        rust_error = {
            "timestamp": standard_error.get("timestamp", datetime.now().isoformat()),
            "error_type": standard_error.get("error_type", "Error"),
            "message": standard_error.get("message", ""),
        }

        # Convert severity to Rust logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "debug",
                "info": "info",
                "warning": "warn",
                "error": "error",
                "critical": "error",
                "fatal": "error",
            }
            rust_error["level"] = level_map.get(
                standard_error["severity"].lower(), "error"
            )

        # Convert stack trace to Rust format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]

            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in Rust stack trace string format
                    rust_error["backtrace"] = stack_trace
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to Rust backtrace format
                    rust_error["backtrace"] = self._convert_frames_to_rust_backtrace(
                        stack_trace
                    )
                    # Also keep the structured version
                    rust_error["structured_backtrace"] = stack_trace

        # Add request information if available
        if "request" in standard_error:
            rust_error["request"] = standard_error["request"]

        # Add context information if available
        if "context" in standard_error:
            rust_error["context"] = standard_error["context"]

        # Add Rust version if available
        if "language_version" in standard_error:
            rust_error["rust_version"] = standard_error["language_version"]

        # Add framework information if available
        if "framework" in standard_error:
            rust_error["framework"] = standard_error["framework"]

            if "framework_version" in standard_error:
                rust_error["framework_version"] = standard_error["framework_version"]

        # Add runtime information if available
        if "runtime" in standard_error:
            rust_error["runtime"] = standard_error["runtime"]

            if "runtime_version" in standard_error:
                rust_error["runtime_version"] = standard_error["runtime_version"]

        # Add handled flag if available
        if "handled" in standard_error:
            rust_error["handled"] = standard_error["handled"]

        # Extract Rust-specific data from additional_data
        if "additional_data" in standard_error:
            for key, value in standard_error["additional_data"].items():
                if key == "thread":
                    rust_error["thread"] = value
                elif key == "is_panic":
                    rust_error["is_panic"] = value
                else:
                    rust_error[key] = value

        return rust_error

    def _parse_rust_stack_trace(self, stack_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parse a Rust stack trace into structured frames.

        Args:
            stack_lines: Rust stack trace lines

        Returns:
            Structured frames or None if parsing fails
        """
        frames = []

        # Rust stack trace patterns:
        # Standard backtrace format:
        # 0: module::function
        #    at /path/to/file.rs:42
        #
        # Alternate format:
        # 0: 0x7f92f72eef64 - std::panicking::begin_panic_handler::{{closure}}

        # Regular backtrace line pattern
        line_pattern = r"^\s*(\d+):\s+(.*?)$"

        # File location pattern
        file_pattern = r"^\s*at\s+([^:]+):(\d+)(?::(\d+))?"

        try:
            current_frame = None

            for line in stack_lines:
                # Try to match a new frame
                line_match = re.match(line_pattern, line)
                if line_match:
                    # If we have a current frame, add it to frames
                    if current_frame is not None:
                        frames.append(current_frame)

                    # Start a new frame
                    frame_index = int(line_match.group(1))
                    frame_info = line_match.group(2).strip()

                    # Parse function name from frame info
                    function_name = frame_info

                    # Check for address format: 0x7f92f72eef64 - std::panicking::begin_panic_handler
                    addr_match = re.match(r"(0x[0-9a-f]+)\s+-\s+(.*)", frame_info)
                    if addr_match:
                        function_name = addr_match.group(2).strip()

                    # Extract module and function
                    module_parts = function_name.split("::")
                    if len(module_parts) > 1:
                        module = "::".join(module_parts[:-1])
                        func = module_parts[-1]
                    else:
                        module = ""
                        func = function_name

                    # Create the new frame
                    current_frame = {
                        "index": frame_index,
                        "function": func,
                        "module": module,
                    }
                else:
                    # Try to match a file location for the current frame
                    file_match = re.match(file_pattern, line)
                    if file_match and current_frame is not None:
                        file_path = file_match.group(1)
                        line_num = int(file_match.group(2))
                        column = (
                            int(file_match.group(3)) if file_match.group(3) else None
                        )

                        current_frame["file"] = file_path
                        current_frame["line"] = line_num
                        if column:
                            current_frame["column"] = column

            # Add the last frame if there is one
            if current_frame is not None:
                frames.append(current_frame)

            return frames if frames else None
        except Exception as e:
            logger.debug(f"Failed to parse Rust stack trace: {e}")
            return None

    def _convert_frames_to_rust_backtrace(
        self, frames: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Convert structured frames to a Rust backtrace.

        Args:
            frames: Structured frames

        Returns:
            Rust backtrace lines
        """
        backtrace_lines = []

        for i, frame in enumerate(frames):
            module = frame.get("module", "")
            func = frame.get("function", "<unknown>")

            # Format function with module
            if module:
                func_name = f"{module}::{func}"
            else:
                func_name = func

            # Format the frame line
            backtrace_lines.append(f"{i}: {func_name}")

            # Add file location if available
            file_path = frame.get("file")
            line_num = frame.get("line")
            if file_path and line_num:
                column = frame.get("column")
                if column:
                    backtrace_lines.append(f"   at {file_path}:{line_num}:{column}")
                else:
                    backtrace_lines.append(f"   at {file_path}:{line_num}")

        return backtrace_lines


class CSharpErrorAdapter(LanguageAdapter):
    """Adapter for C# error formats."""

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert C# error data to the standard format.

        Args:
            error_data: C# error data

        Returns:
            Error data in the standard format
        """
        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "csharp",
            "error_type": error_data.get("exception_type", ""),
            "message": error_data.get("message", ""),
        }

        # Add .NET version if available
        if "dotnet_version" in error_data:
            standard_error["language_version"] = error_data["dotnet_version"]

        # Handle stack trace
        if "stack_trace" in error_data:
            # C# stack traces can be a string or a list
            if isinstance(error_data["stack_trace"], str):
                # Split into lines
                stack_lines = error_data["stack_trace"].split("\n")

                # Try to parse structured data from the stack trace
                parsed_frames = self._parse_csharp_stack_trace(stack_lines)

                if parsed_frames:
                    standard_error["stack_trace"] = parsed_frames
                else:
                    standard_error["stack_trace"] = stack_lines
            elif isinstance(error_data["stack_trace"], list):
                if all(isinstance(frame, dict) for frame in error_data["stack_trace"]):
                    # Already in structured format
                    standard_error["stack_trace"] = error_data["stack_trace"]
                else:
                    # List of strings
                    standard_error["stack_trace"] = error_data["stack_trace"]

        # Add framework information if available
        if "framework" in error_data:
            standard_error["framework"] = error_data["framework"]

            if "framework_version" in error_data:
                standard_error["framework_version"] = error_data["framework_version"]

        # Add request information if available
        if "request" in error_data:
            standard_error["request"] = error_data["request"]

        # Add any additional context
        if "context" in error_data:
            standard_error["context"] = error_data["context"]

        # Add inner exception if available
        if "inner_exception" in error_data:
            if "additional_data" not in standard_error:
                standard_error["additional_data"] = {}
            standard_error["additional_data"]["inner_exception"] = error_data[
                "inner_exception"
            ]

        # Add severity if available
        if "level" in error_data:
            # Map C# log levels to standard format
            level_map = {
                "trace": "debug",
                "debug": "debug",
                "information": "info",
                "warning": "warning",
                "error": "error",
                "critical": "critical",
                "fatal": "fatal",
            }
            standard_error["severity"] = level_map.get(
                error_data["level"].lower(), "error"
            )

        # Add runtime if available
        if "runtime" in error_data:
            standard_error["runtime"] = error_data["runtime"]

            if "runtime_version" in error_data:
                standard_error["runtime_version"] = error_data["runtime_version"]

        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]

        # Add additional C#-specific data
        csharp_specific = {}
        for key, value in error_data.items():
            if key not in standard_error and key not in [
                "stack_trace",
                "request",
                "context",
                "inner_exception",
            ]:
                csharp_specific[key] = value

        if csharp_specific:
            if "additional_data" in standard_error:
                standard_error["additional_data"].update(csharp_specific)
            else:
                standard_error["additional_data"] = csharp_specific

        return standard_error

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to C#-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the C#-specific format
        """
        # Create a C# error object
        csharp_error = {
            "timestamp": standard_error.get("timestamp", datetime.now().isoformat()),
            "exception_type": standard_error.get("error_type", "System.Exception"),
            "message": standard_error.get("message", ""),
        }

        # Convert severity to C# logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "Debug",
                "info": "Information",
                "warning": "Warning",
                "error": "Error",
                "critical": "Critical",
                "fatal": "Fatal",
            }
            csharp_error["level"] = level_map.get(
                standard_error["severity"].lower(), "Error"
            )

        # Convert stack trace to C# format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]

            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in C# stack trace string format
                    csharp_error["stack_trace"] = "\n".join(stack_trace)
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to C# stack trace format
                    csharp_error["stack_trace"] = self._convert_frames_to_csharp_stack(
                        standard_error.get("error_type", "System.Exception"),
                        standard_error.get("message", ""),
                        stack_trace,
                    )
                    # Also keep the structured version
                    csharp_error["structured_stack_trace"] = stack_trace

        # Add request information if available
        if "request" in standard_error:
            csharp_error["request"] = standard_error["request"]

        # Add context information if available
        if "context" in standard_error:
            csharp_error["context"] = standard_error["context"]

        # Add .NET version if available
        if "language_version" in standard_error:
            csharp_error["dotnet_version"] = standard_error["language_version"]

        # Add framework information if available
        if "framework" in standard_error:
            csharp_error["framework"] = standard_error["framework"]

            if "framework_version" in standard_error:
                csharp_error["framework_version"] = standard_error["framework_version"]

        # Add runtime information if available
        if "runtime" in standard_error:
            csharp_error["runtime"] = standard_error["runtime"]

            if "runtime_version" in standard_error:
                csharp_error["runtime_version"] = standard_error["runtime_version"]

        # Add handled flag if available
        if "handled" in standard_error:
            csharp_error["handled"] = standard_error["handled"]

        # Extract C#-specific data from additional_data
        if "additional_data" in standard_error:
            for key, value in standard_error["additional_data"].items():
                if key == "inner_exception":
                    csharp_error["inner_exception"] = value
                elif key == "level":
                    # Skip level from additional_data - we already set it from severity
                    pass
                else:
                    csharp_error[key] = value

        return csharp_error

    def _parse_csharp_stack_trace(self, stack_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parse a C# stack trace into structured frames.

        Args:
            stack_lines: C# stack trace lines

        Returns:
            Structured frames or None if parsing fails
        """
        frames = []

        # C# stack trace patterns:
        # Standard format: at Namespace.Class.Method(parameters) in File:line number
        # Alternative: at Namespace.Class.Method(parameters) in c:\path\to\file.cs:line 42
        # Or simpler: at Namespace.Class.Method(parameters)

        # Regular patterns - simpler approach focused on the 'in file:line' pattern
        # First extract method signature
        method_pattern = r"\s*at\s+([^(]+)\(([^)]*)\)"
        # Then separately extract file location
        file_pattern = r"\s+in\s+(.+):line\s+(\d+)"

        try:
            for line in stack_lines:
                # Skip empty lines
                if not line.strip():
                    continue

                # Extract method first
                method_match = re.search(method_pattern, line)
                if not method_match:
                    continue

                full_method = method_match.group(1)
                parameters = method_match.group(2)

                # Extract file and line if present
                file_match = re.search(file_pattern, line)
                file_path = file_match.group(1) if file_match else ""
                line_num = int(file_match.group(2)) if file_match else 0

                # Parse the full method name into namespace, class, and method
                parts = full_method.split(".")

                # Handle lambda/anonymous method cases (compiler-generated classes)
                # e.g., MyCompany.MyApp.Services.DataService.<>c__DisplayClass5_0.<ProcessItems>b__0
                if "<>" in full_method:
                    # Find the real class name (before the <>c__ part)
                    lambda_parts = []
                    real_class_idx = -1
                    for i, part in enumerate(parts):
                        if part.startswith("<>"):
                            # This is a compiler-generated class
                            real_class_idx = i - 1
                            break
                        lambda_parts.append(part)

                    if real_class_idx >= 0:
                        # Extract namespace, real class, and full method including lambda parts
                        if real_class_idx > 0:
                            namespace = ".".join(lambda_parts[:-1])
                            class_name = lambda_parts[-1]
                        else:
                            namespace = ""
                            class_name = lambda_parts[0]
                        # Include everything after the class as the method
                        method = ".".join(parts[real_class_idx + 1 :])
                    else:
                        # Fall back to standard parsing
                        if len(parts) >= 3:
                            namespace = ".".join(parts[:-2])
                            class_name = parts[-2]
                            method = parts[-1]
                        elif len(parts) == 2:
                            namespace = ""
                            class_name = parts[0]
                            method = parts[1]
                        else:
                            namespace = ""
                            class_name = ""
                            method = full_method
                else:
                    # Standard case (no lambda)
                    if len(parts) >= 3:
                        # Standard case: Namespace.Class.Method
                        namespace = ".".join(parts[:-2])
                        class_name = parts[-2]
                        method = parts[-1]
                    elif len(parts) == 2:
                        # Case: Class.Method
                        namespace = ""
                        class_name = parts[0]
                        method = parts[1]
                    else:
                        # Case: Method (rare)
                        namespace = ""
                        class_name = ""
                        method = full_method

                frames.append(
                    {
                        "namespace": namespace,
                        "class": class_name,
                        "function": method,
                        "parameters": parameters,
                        "file": file_path,
                        "line": line_num,
                    }
                )

            return frames if frames else None
        except Exception as e:
            logger.debug(f"Failed to parse C# stack trace: {e}")
            return None

    def _convert_frames_to_csharp_stack(
        self, error_type: str, message: str, frames: List[Dict[str, Any]]
    ) -> str:
        """
        Convert structured frames to a C# stack trace string.

        Args:
            error_type: Error type/name
            message: Error message
            frames: Structured frames

        Returns:
            C# stack trace string
        """
        stack_lines = [f"{error_type}: {message}"]

        for frame in frames:
            namespace = frame.get("namespace", "")
            class_name = frame.get("class", "")
            method = frame.get("function", "")
            parameters = frame.get("parameters", "")
            file = frame.get("file", "")
            line_num = frame.get("line", 0)

            # Build the full method name
            if namespace and class_name:
                full_method = f"{namespace}.{class_name}.{method}"
            elif class_name:
                full_method = f"{class_name}.{method}"
            else:
                full_method = method

            # Format the stack frame
            if file and line_num:
                stack_lines.append(
                    f"   at {full_method}({parameters}) in {file}:line {line_num}"
                )
            else:
                stack_lines.append(f"   at {full_method}({parameters})")

        return "\n".join(stack_lines)


class ErrorAdapterFactory:
    """Factory for creating language-specific error adapters."""

    @staticmethod
    def get_adapter(language: str) -> LanguageAdapter:
        """
        Get the appropriate adapter for a language.

        Args:
            language: The programming language

        Returns:
            Language-specific adapter

        Raises:
            ValueError: If an adapter for the language is not available
        """
        language = language.lower()

        if language == "python":
            return PythonErrorAdapter()
        elif language in ["javascript", "typescript"]:
            return JavaScriptErrorAdapter()
        elif language == "java":
            return JavaErrorAdapter()
        elif language == "go":
            return GoErrorAdapter()
        elif language == "ruby":
            return RubyErrorAdapter()
        elif language == "rust":
            return RustErrorAdapter()
        elif language == "csharp" or language == "c#":
            return CSharpErrorAdapter()
        elif language == "php":
            return PHPErrorAdapter()
        elif language in ["c", "cpp", "c++"]:
            from modules.analysis.cpp_adapter import CPPErrorAdapter

            return CPPErrorAdapter()
        elif language == "kotlin":
            return KotlinErrorAdapter()
        else:
            raise ValueError(f"No adapter available for language: {language}")

    @staticmethod
    def detect_language(error_data: Dict[str, Any]) -> str:
        """
        Detect the language from error data.

        Args:
            error_data: The error data

        Returns:
            Detected language or "unknown"
        """
        # Check if language is explicitly specified
        if "language" in error_data:
            return error_data["language"].lower()

        # Look for language-specific indicators
        if "exception_type" in error_data and "traceback" in error_data:
            return "python"
        elif "name" in error_data and "stack" in error_data:
            return "javascript"
        elif "exception_class" in error_data and any(
            key in error_data for key in ["stack_trace", "java_version", "jvm"]
        ):
            return "java"
        elif "goroutine_id" in error_data or "go_version" in error_data:
            return "go"
        elif "exception_class" in error_data and "backtrace" in error_data:
            return "ruby"
        elif "rust_version" in error_data or "is_panic" in error_data:
            return "rust"
        elif "dotnet_version" in error_data or (
            "exception_type" in error_data
            and error_data["exception_type"].startswith("System.")
        ):
            return "csharp"
        elif "php_version" in error_data or (
            "backtrace" in error_data
            and any(
                isinstance(frame, dict)
                and "file" in frame
                and ".php" in frame.get("file", "")
                for frame in error_data["backtrace"][:10]
                if isinstance(frame, dict)
            )
        ):
            return "php"
        elif (
            "kotlin_version" in error_data
            or "android" in error_data
            or (
                "error_type" in error_data
                and "KotlinNullPointerException" in error_data["error_type"]
            )
            or (
                "stack_trace" in error_data
                and isinstance(error_data["stack_trace"], (str, list))
                and any(
                    (
                        ".kt:" in str(frame)
                        or "kotlinx.coroutines" in str(frame)
                        or "kotlin." in str(frame)
                    )
                    for frame in (
                        error_data["stack_trace"]
                        if isinstance(error_data["stack_trace"], list)
                        else [error_data["stack_trace"]]
                    )
                )
            )
        ):
            return "kotlin"

        # Try to detect from stack trace format
        if "stack_trace" in error_data and isinstance(error_data["stack_trace"], str):
            stack = error_data["stack_trace"]
            if "goroutine " in stack and ".go:" in stack:
                return "go"
            elif "at " in stack and ".java:" in stack:
                return "java"
            elif "at " in stack and ".kt:" in stack:
                return "kotlin"
            elif "at " in stack and (
                re.search(r"\.cs:line \d+", stack) or "System." in stack
            ):
                return "csharp"

        if "stack" in error_data and isinstance(error_data["stack"], str):
            stack = error_data["stack"]
            if "at " in stack and ").js" in stack:
                return "javascript"
            elif "at " in stack and ".java:" in stack:
                return "java"
            elif "at " in stack and ".kt:" in stack:
                return "kotlin"
            elif "goroutine " in stack and ".go:" in stack:
                return "go"
            elif "at " in stack and (
                re.search(r"\.cs:line \d+", stack) or "System." in stack
            ):
                return "csharp"

        if "backtrace" in error_data:
            # Check for Ruby backtrace format
            if isinstance(error_data["backtrace"], list) and any(
                isinstance(line, str) and re.search(r"\.rb:\d+:in", line)
                for line in error_data["backtrace"][:10]
                if isinstance(line, str)
            ):
                return "ruby"
            elif (
                isinstance(error_data["backtrace"], str)
                and ".rb:" in error_data["backtrace"]
            ):
                return "ruby"
            # Check for Rust backtrace format
            elif isinstance(error_data["backtrace"], str) and (
                ".rs:" in error_data["backtrace"]
                or any(
                    pattern in error_data["backtrace"]
                    for pattern in ["panicked at", "thread", "rust_panic"]
                )
            ):
                return "rust"
            elif isinstance(error_data["backtrace"], list) and any(
                isinstance(line, str)
                and (re.search(r"\.rs:\d+", line) or "::{{closure}}" in line)
                for line in error_data["backtrace"][:10]
                if isinstance(line, str)
            ):
                return "rust"
            elif isinstance(error_data["backtrace"], list) and any(
                isinstance(line, str) and re.search(r"\.php(?::\d+)?", line)
                for line in error_data["backtrace"][:10]
                if isinstance(line, str)
            ):
                return "php"

        return "unknown"


def convert_to_standard_format(
    error_data: Dict[str, Any], language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert language-specific error data to the standard format.

    Args:
        error_data: Language-specific error data
        language: Optional language identifier (auto-detected if not specified)

    Returns:
        Error data in the standard format

    Raises:
        ValueError: If language cannot be determined or is not supported
    """
    if language is None:
        language = ErrorAdapterFactory.detect_language(error_data)

        if language == "unknown":
            raise ValueError("Could not detect language from error data")

    adapter = ErrorAdapterFactory.get_adapter(language)
    return adapter.to_standard_format(error_data)


def convert_from_standard_format(
    standard_error: Dict[str, Any], target_language: str
) -> Dict[str, Any]:
    """
    Convert standard format error data to a language-specific format.

    Args:
        standard_error: Error data in the standard format
        target_language: Target language for conversion

    Returns:
        Error data in the language-specific format

    Raises:
        ValueError: If target language is not supported
    """
    adapter = ErrorAdapterFactory.get_adapter(target_language)
    return adapter.from_standard_format(standard_error)


def validate_error_schema(error_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate error data against the standard schema.

    Args:
        error_data: Error data to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    validator = ErrorSchemaValidator()
    return validator.validate(error_data)


def extract_python_exception(exception: Exception) -> Dict[str, Any]:
    """
    Extract error information from a Python exception.

    Args:
        exception: Python exception object

    Returns:
        Error data dictionary
    """
    error_data = {
        "timestamp": datetime.now().isoformat(),
        "exception_type": exception.__class__.__name__,
        "message": str(exception),
        "language": "python",
        "language_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "traceback": traceback.format_exception(
            type(exception), exception, exception.__traceback__
        ),
    }

    return convert_to_standard_format(error_data, "python")


class PHPErrorAdapter(LanguageAdapter):
    """Adapter for PHP error formats."""

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert PHP error data to the standard format.

        Args:
            error_data: PHP error data

        Returns:
            Error data in the standard format
        """
        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "php",
            "error_type": error_data.get("type", error_data.get("error_type", "")),
            "message": error_data.get("message", ""),
        }

        # Add PHP version if available
        if "php_version" in error_data:
            standard_error["language_version"] = error_data["php_version"]

        # Handle stack trace (PHP uses "trace", "backtrace", or "stack_trace")
        for trace_key in ["trace", "backtrace", "stack_trace"]:
            if trace_key in error_data:
                trace_data = error_data[trace_key]
                standard_error["stack_trace"] = self._parse_php_stack_trace(trace_data)
                break

        # Add file and line information (common in PHP errors)
        if "file" in error_data and "line" in error_data:
            file_path = error_data["file"]
            line_number = error_data["line"]

            # If no stack trace was found, create a basic one from file/line
            if "stack_trace" not in standard_error:
                standard_error["stack_trace"] = [
                    {
                        "file": file_path,
                        "line": line_number,
                        "function": error_data.get("function", ""),
                    }
                ]

        # Add framework information if available
        if "framework" in error_data:
            standard_error["framework"] = error_data["framework"]

            if "framework_version" in error_data:
                standard_error["framework_version"] = error_data["framework_version"]

        # Add request information if available
        if "request" in error_data:
            standard_error["request"] = error_data["request"]

        # Add any additional context
        if "context" in error_data:
            standard_error["context"] = error_data["context"]

        # Add severity if available
        if "level" in error_data:
            # Map PHP error levels to standard format
            level_map = {
                "E_NOTICE": "info",
                "E_WARNING": "warning",
                "E_ERROR": "error",
                "E_PARSE": "error",
                "E_CORE_ERROR": "critical",
                "E_COMPILE_ERROR": "critical",
                "E_USER_ERROR": "error",
                "E_USER_WARNING": "warning",
                "E_USER_NOTICE": "info",
                "E_STRICT": "info",
                "E_RECOVERABLE_ERROR": "error",
                "E_DEPRECATED": "info",
                "E_USER_DEPRECATED": "info",
                "DEBUG": "debug",
                "INFO": "info",
                "WARNING": "warning",
                "ERROR": "error",
                "CRITICAL": "critical",
                "ALERT": "critical",
                "EMERGENCY": "fatal",
            }
            standard_error["severity"] = level_map.get(
                error_data["level"].upper(), "error"
            )

        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]

        # Add additional PHP-specific data
        php_specific = {}
        for key, value in error_data.items():
            if key not in standard_error and key not in [
                "trace",
                "backtrace",
                "request",
                "context",
            ]:
                php_specific[key] = value

        if php_specific:
            standard_error["additional_data"] = php_specific

        return standard_error

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to PHP-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the PHP-specific format
        """
        # Create a PHP error object
        php_error = {
            "timestamp": standard_error.get("timestamp", datetime.now().isoformat()),
            "type": standard_error.get("error_type", "Error"),
            "message": standard_error.get("message", ""),
        }

        # Convert severity to PHP error level
        if "severity" in standard_error:
            level_map = {
                "debug": "E_NOTICE",
                "info": "E_NOTICE",
                "warning": "E_WARNING",
                "error": "E_ERROR",
                "critical": "E_CORE_ERROR",
                "fatal": "E_CORE_ERROR",
            }
            php_error["level"] = level_map.get(
                standard_error["severity"].lower(), "E_ERROR"
            )

        # Convert stack trace to PHP format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]

            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Plain text stack trace
                    php_error["trace"] = stack_trace
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to PHP stack trace format
                    php_error["trace"] = self._convert_frames_to_php_trace(stack_trace)

                    # Extract file and line from the first frame
                    if (
                        stack_trace
                        and "file" in stack_trace[0]
                        and "line" in stack_trace[0]
                    ):
                        php_error["file"] = stack_trace[0]["file"]
                        php_error["line"] = stack_trace[0]["line"]

        # Add request information if available
        if "request" in standard_error:
            php_error["request"] = standard_error["request"]

        # Add context information if available
        if "context" in standard_error:
            php_error["context"] = standard_error["context"]

        # Add PHP version if available
        if "language_version" in standard_error:
            php_error["php_version"] = standard_error["language_version"]

        # Add framework information if available
        if "framework" in standard_error:
            php_error["framework"] = standard_error["framework"]

            if "framework_version" in standard_error:
                php_error["framework_version"] = standard_error["framework_version"]

        # Add handled flag if available
        if "handled" in standard_error:
            php_error["handled"] = standard_error["handled"]

        # Add additional data if available
        if "additional_data" in standard_error:
            for key, value in standard_error["additional_data"].items():
                php_error[key] = value

        return php_error

    def _parse_php_stack_trace(self, stack_trace) -> List[Dict[str, Any]]:
        """
        Parse a PHP stack trace to a structured format.

        Args:
            stack_trace: PHP stack trace data

        Returns:
            Structured stack frames
        """
        frames = []

        if isinstance(stack_trace, list):
            for item in stack_trace:
                if isinstance(item, dict):
                    # Already structured frame
                    frame = {
                        "file": item.get("file", ""),
                        "line": item.get("line", 0),
                        "function": item.get("function", ""),
                    }

                    # Add class if available
                    if "class" in item:
                        frame["class"] = item["class"]

                    # Add type if available (-> or ::)
                    if "type" in item:
                        frame["type"] = item["type"]

                    # Add args if available
                    if "args" in item:
                        frame["args"] = item["args"]

                    frames.append(frame)
                elif isinstance(item, str):
                    # Parse string format like "#0 /path/to/file.php(123): Class->method()"
                    match = re.match(
                        r"#\d+\s+([^(]+)\((\d+)\):\s+(?:([^->\(:]+)(?:->|::))?([^(]+)",
                        item,
                    )
                    if match:
                        file_path, line_num, class_name, method = match.groups()

                        frame = {
                            "file": file_path.strip(),
                            "line": int(line_num) if line_num else 0,
                            "function": method.strip() if method else "",
                        }

                        if class_name:
                            frame["class"] = class_name.strip()

                        frames.append(frame)
        elif isinstance(stack_trace, str):
            # Split by lines and parse
            lines = stack_trace.split("\n")
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue

                # Parse "#0 /path/to/file.php(123): Class->method()" format
                match = re.match(
                    r"#\d+\s+([^(]+)\((\d+)\):\s+(?:([^->\(:]+)(?:->|::))?([^(]+)", line
                )
                if match:
                    file_path, line_num, class_name, method = match.groups()

                    frame = {
                        "file": file_path.strip(),
                        "line": int(line_num) if line_num else 0,
                        "function": method.strip() if method else "",
                    }

                    if class_name:
                        frame["class"] = class_name.strip()

                    frames.append(frame)

        return frames

    def _convert_frames_to_php_trace(
        self, frames: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert structured stack frames to PHP trace format.

        Args:
            frames: Structured stack frames

        Returns:
            PHP trace format
        """
        php_frames = []

        for i, frame in enumerate(frames):
            php_frame = {
                "file": frame.get("file", ""),
                "line": frame.get("line", 0),
                "function": frame.get("function", ""),
            }

            # Add class if available
            if "class" in frame:
                php_frame["class"] = frame["class"]
            elif "module" in frame:
                php_frame["class"] = frame["module"]

            # Add type (-> or ::)
            if "type" in frame:
                php_frame["type"] = frame["type"]
            else:
                php_frame["type"] = "->"  # Default to instance method

            # Add args placeholder
            php_frame["args"] = []

            php_frames.append(php_frame)

        return php_frames


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Example Python error data
    python_error = {
        "timestamp": "2023-08-15T12:34:56",
        "exception_type": "KeyError",
        "message": "'user_id'",
        "traceback": [
            "Traceback (most recent call last):",
            '  File "app.py", line 42, in get_user',
            "    user_id = data['user_id']",
            "KeyError: 'user_id'",
        ],
        "level": "ERROR",
        "python_version": "3.9.7",
        "framework": "FastAPI",
        "framework_version": "0.68.0",
    }

    # Convert to standard format
    python_adapter = PythonErrorAdapter()
    standard_python = python_adapter.to_standard_format(python_error)
    logger.info(f"Standard Python format: {json.dumps(standard_python, indent=2)}")

    # Validate against schema
    is_valid, error = validate_error_schema(standard_python)
    logger.info(f"Schema validation: {'Valid' if is_valid else 'Invalid'}")
    if error:
        logger.error(f"Validation error: {error}")

    # Example JavaScript error data
    js_error = {
        "timestamp": "2023-08-15T12:34:56",
        "name": "TypeError",
        "message": "Cannot read property 'id' of undefined",
        "stack": "TypeError: Cannot read property 'id' of undefined\n    at getUser (app.js:25:20)\n    at processRequest (server.js:42:10)\n    at Server.<anonymous> (index.js:10:5)",
        "level": "error",
        "runtime": "Node.js",
        "runtime_version": "14.17.5",
    }

    # Convert to standard format
    js_adapter = JavaScriptErrorAdapter()
    standard_js = js_adapter.to_standard_format(js_error)
    logger.info(f"Standard JavaScript format: {json.dumps(standard_js, indent=2)}")

    # Convert JavaScript to Python format
    js_to_python = convert_from_standard_format(standard_js, "python")
    logger.info(
        f"JavaScript converted to Python format: {json.dumps(js_to_python, indent=2)}"
    )

    # Example Ruby error data
    ruby_error = {
        "timestamp": "2023-08-15T12:34:56",
        "exception_class": "NoMethodError",
        "message": "undefined method `[]' for nil:NilClass",
        "backtrace": [
            "app/controllers/users_controller.rb:25:in `show'",
            "app/controllers/application_controller.rb:10:in `authorize_user!'",
            "/gems/actionpack-6.1.0/lib/action_controller/metal/basic_implicit_render.rb:6:in `send_action'",
        ],
        "level": "error",
        "ruby_version": "3.0.2",
        "framework": "Rails",
        "framework_version": "6.1.0",
    }

    # Convert to standard format
    ruby_adapter = RubyErrorAdapter()
    standard_ruby = ruby_adapter.to_standard_format(ruby_error)
    logger.info(f"Standard Ruby format: {json.dumps(standard_ruby, indent=2)}")

    # Convert Ruby to Python format
    ruby_to_python = convert_from_standard_format(standard_ruby, "python")
    logger.info(
        f"Ruby converted to Python format: {json.dumps(ruby_to_python, indent=2)}"
    )


class ScalaErrorAdapter(LanguageAdapter):
    """Adapter for Scala error formats."""

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Scala error data to the standard format.

        Args:
            error_data: Scala error data

        Returns:
            Error data in the standard format
        """
        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "scala",
            "error_type": error_data.get("error_type", ""),
            "message": error_data.get("message", ""),
        }

        # Add Scala version if available
        if "scala_version" in error_data:
            standard_error["language_version"] = error_data["scala_version"]

        # Handle stack trace
        if "stack_trace" in error_data:
            # Scala stack traces can be a string or a list
            if isinstance(error_data["stack_trace"], str):
                # Split into lines
                stack_lines = error_data["stack_trace"].split("\n")

                # Try to parse structured data from the stack trace
                parsed_frames = self._parse_scala_stack_trace(stack_lines)

                if parsed_frames:
                    standard_error["stack_trace"] = parsed_frames
                else:
                    standard_error["stack_trace"] = stack_lines
            elif isinstance(error_data["stack_trace"], list):
                if all(isinstance(frame, dict) for frame in error_data["stack_trace"]):
                    # Already in structured format
                    standard_error["stack_trace"] = error_data["stack_trace"]
                else:
                    # List of strings
                    standard_error["stack_trace"] = error_data["stack_trace"]

        # Add framework information if available
        if "framework" in error_data:
            standard_error["framework"] = error_data["framework"]

            if "framework_version" in error_data:
                standard_error["framework_version"] = error_data["framework_version"]

        # Add request information if available
        if "request" in error_data:
            standard_error["request"] = error_data["request"]

        # Add any additional context
        if "context" in error_data:
            standard_error["context"] = error_data["context"]

        # Add severity if available
        if "level" in error_data:
            # Map Scala log levels to standard format
            level_map = {
                "debug": "debug",
                "info": "info",
                "warning": "warning",
                "warn": "warning",
                "error": "error",
                "fatal": "fatal",
            }
            standard_error["severity"] = level_map.get(
                error_data["level"].lower(), "error"
            )

        # Add runtime if available
        if "runtime" in error_data:
            standard_error["runtime"] = error_data["runtime"]

            if "runtime_version" in error_data:
                standard_error["runtime_version"] = error_data["runtime_version"]

        # Add JVM information if available
        if "jvm_info" in error_data:
            if "additional_data" not in standard_error:
                standard_error["additional_data"] = {}
            standard_error["additional_data"]["jvm_info"] = error_data["jvm_info"]

        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]

        # Add additional Scala-specific data
        scala_specific = {}
        for key, value in error_data.items():
            if (
                key not in standard_error
                and key not in ["stack_trace", "request", "context"]
                and not key.startswith("_")
            ):
                scala_specific[key] = value

        if scala_specific:
            if "additional_data" in standard_error:
                standard_error["additional_data"].update(scala_specific)
            else:
                standard_error["additional_data"] = scala_specific

        return standard_error

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Scala-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Scala-specific format
        """
        # Create a Scala error object
        scala_error = {
            "timestamp": standard_error.get("timestamp", datetime.now().isoformat()),
            "error_type": standard_error.get("error_type", "ScalaError"),
            "message": standard_error.get("message", ""),
        }

        # Convert severity to Scala logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "debug",
                "info": "info",
                "warning": "warn",
                "error": "error",
                "critical": "error",
                "fatal": "fatal",
            }
            scala_error["level"] = level_map.get(
                standard_error["severity"].lower(), "error"
            )

        # Convert stack trace to Scala format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]

            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in Scala stack trace string format
                    scala_error["stack_trace"] = "\n".join(stack_trace)
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to Scala stack trace format
                    scala_error["stack_trace"] = self._convert_frames_to_scala_stack(
                        standard_error.get("error_type", "ScalaError"),
                        standard_error.get("message", ""),
                        stack_trace,
                    )
                    # Also keep the structured version
                    scala_error["stack_frames"] = stack_trace

        # Add request information if available
        if "request" in standard_error:
            scala_error["request"] = standard_error["request"]

        # Add context information if available
        if "context" in standard_error:
            scala_error["context"] = standard_error["context"]

        # Add Scala version if available
        if "language_version" in standard_error:
            scala_error["scala_version"] = standard_error["language_version"]

        # Add framework information if available
        if "framework" in standard_error:
            scala_error["framework"] = standard_error["framework"]

            if "framework_version" in standard_error:
                scala_error["framework_version"] = standard_error["framework_version"]

        # Add runtime information if available
        if "runtime" in standard_error:
            scala_error["runtime"] = standard_error["runtime"]

            if "runtime_version" in standard_error:
                scala_error["runtime_version"] = standard_error["runtime_version"]

        # Add handled flag if available
        if "handled" in standard_error:
            scala_error["handled"] = standard_error["handled"]

        # Add additional data if available
        if "additional_data" in standard_error:
            # Extract JVM info as a top-level field
            if "jvm_info" in standard_error["additional_data"]:
                scala_error["jvm_info"] = standard_error["additional_data"]["jvm_info"]

            # Add remaining fields
            for key, value in standard_error["additional_data"].items():
                if key != "jvm_info":
                    scala_error[key] = value

        return scala_error

    def _parse_scala_stack_trace(self, stack_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parse a Scala stack trace into structured frames.

        Args:
            stack_lines: Stack trace lines

        Returns:
            List of structured stack frames
        """
        frames = []

        # Regex pattern for Scala stack trace lines
        # Example: at com.example.MyClass.myMethod(MyClass.scala:42)
        pattern = r"\s*at\s+([^(]+)\.([^.]+)\(([^:]+):(\d+)\)"

        for line in stack_lines:
            match = re.search(pattern, line)
            if match:
                package_class = match.group(1)
                method = match.group(2)
                file = match.group(3)
                line_num = int(match.group(4))

                # Split package and class
                parts = package_class.split(".")
                if len(parts) > 1:
                    package = ".".join(parts[:-1])
                    class_name = parts[-1]
                else:
                    package = ""
                    class_name = package_class

                frames.append(
                    {
                        "package": package,
                        "class": class_name,
                        "function": method,
                        "file": file,
                        "line": line_num,
                    }
                )

        return frames if frames else []

    def _convert_frames_to_scala_stack(
        self, error_type: str, message: str, frames: List[Dict[str, Any]]
    ) -> str:
        """
        Convert structured stack frames to a Scala stack trace string.

        Args:
            error_type: Error type
            message: Error message
            frames: Structured stack frames

        Returns:
            Scala stack trace string
        """
        stack_lines = [f"{error_type}: {message}"]

        for frame in frames:
            package = frame.get("package", "")
            class_name = frame.get("class", "")
            method = frame.get("function", "")
            file = frame.get("file", "")
            line = frame.get("line", "?")

            if package:
                full_class = f"{package}.{class_name}"
            else:
                full_class = class_name

            stack_lines.append(f"\tat {full_class}.{method}({file}:{line})")

        return "\n".join(stack_lines)


class ElixirErrorAdapter(LanguageAdapter):
    """Adapter for Elixir/Erlang error formats."""

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Elixir/Erlang error data to the standard format.

        Args:
            error_data: Elixir/Erlang error data

        Returns:
            Error data in the standard format
        """
        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "elixir",
            "error_type": error_data.get("exception", error_data.get("error_type", "")),
            "message": error_data.get("message", ""),
        }

        # Add Elixir version if available
        if "elixir_version" in error_data:
            standard_error["language_version"] = error_data["elixir_version"]

        # Add Erlang version if available
        if "erlang_version" in error_data:
            if "additional_data" not in standard_error:
                standard_error["additional_data"] = {}
            standard_error["additional_data"]["erlang_version"] = error_data[
                "erlang_version"
            ]

        # Handle stack trace
        if "stacktrace" in error_data:
            stacktrace = error_data["stacktrace"]

            # Elixir stacktraces are usually already in list format
            if isinstance(stacktrace, list):
                if stacktrace and isinstance(stacktrace[0], dict):
                    # Already in structured format
                    standard_error["stack_trace"] = stacktrace
                elif stacktrace and isinstance(stacktrace[0], str):
                    # List of strings
                    parsed_frames = self._parse_elixir_stack_trace(stacktrace)
                    if parsed_frames:
                        standard_error["stack_trace"] = parsed_frames
                    else:
                        standard_error["stack_trace"] = stacktrace
            elif isinstance(stacktrace, str):
                # Convert string to list by splitting on newlines
                stack_lines = stacktrace.split("\n")
                parsed_frames = self._parse_elixir_stack_trace(stack_lines)
                if parsed_frames:
                    standard_error["stack_trace"] = parsed_frames
                else:
                    standard_error["stack_trace"] = stack_lines

        # Add framework information if available
        if "framework" in error_data:
            standard_error["framework"] = error_data["framework"]

            if "framework_version" in error_data:
                standard_error["framework_version"] = error_data["framework_version"]

        # Add request information if available
        if "request" in error_data:
            standard_error["request"] = error_data["request"]

        # Add any additional context
        if "context" in error_data:
            standard_error["context"] = error_data["context"]

        # Add severity if available
        if "level" in error_data:
            # Map Elixir log levels to standard format
            level_map = {
                "debug": "debug",
                "info": "info",
                "warn": "warning",
                "warning": "warning",
                "error": "error",
                "critical": "critical",
                "fatal": "fatal",
            }
            standard_error["severity"] = level_map.get(
                error_data["level"].lower(), "error"
            )

        # Add OTP application/supervisor information
        if "otp_app" in error_data:
            if "additional_data" not in standard_error:
                standard_error["additional_data"] = {}
            standard_error["additional_data"]["otp_app"] = error_data["otp_app"]

        if "supervisor" in error_data:
            if "additional_data" not in standard_error:
                standard_error["additional_data"] = {}
            standard_error["additional_data"]["supervisor"] = error_data["supervisor"]

        # Add runtime if available (typically BEAM VM)
        if "runtime" in error_data:
            standard_error["runtime"] = error_data["runtime"]

            if "runtime_version" in error_data:
                standard_error["runtime_version"] = error_data["runtime_version"]

        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]

        # Add additional Elixir-specific data
        elixir_specific = {}
        for key, value in error_data.items():
            if (
                key not in standard_error
                and key not in ["stacktrace", "request", "context"]
                and not key.startswith("_")
            ):
                elixir_specific[key] = value

        if elixir_specific:
            if "additional_data" in standard_error:
                standard_error["additional_data"].update(elixir_specific)
            else:
                standard_error["additional_data"] = elixir_specific

        return standard_error

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Elixir-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Elixir-specific format
        """
        # Create an Elixir error object
        elixir_error = {
            "timestamp": standard_error.get("timestamp", datetime.now().isoformat()),
            "exception": standard_error.get("error_type", "RuntimeError"),
            "message": standard_error.get("message", ""),
        }

        # Convert severity to Elixir logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "debug",
                "info": "info",
                "warning": "warn",
                "error": "error",
                "critical": "error",
                "fatal": "fatal",
            }
            elixir_error["level"] = level_map.get(
                standard_error["severity"].lower(), "error"
            )

        # Convert stack trace to Elixir format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]

            # Convert structured frames to Elixir stacktrace format
            if isinstance(stack_trace, list):
                if all(isinstance(frame, dict) for frame in stack_trace):
                    formatted_frames = self._convert_frames_to_elixir_stacktrace(
                        stack_trace
                    )
                    elixir_error["stacktrace"] = formatted_frames
                else:
                    # Already list of strings
                    elixir_error["stacktrace"] = stack_trace

        # Add language version if available
        if "language_version" in standard_error:
            elixir_error["elixir_version"] = standard_error["language_version"]

        # Add framework information if available
        if "framework" in standard_error:
            elixir_error["framework"] = standard_error["framework"]

            if "framework_version" in standard_error:
                elixir_error["framework_version"] = standard_error["framework_version"]

        # Add request information if available
        if "request" in standard_error:
            elixir_error["request"] = standard_error["request"]

        # Add context information if available
        if "context" in standard_error:
            elixir_error["context"] = standard_error["context"]

        # Add OTP application information if available
        if "additional_data" in standard_error:
            if "otp_app" in standard_error["additional_data"]:
                elixir_error["otp_app"] = standard_error["additional_data"]["otp_app"]

            if "supervisor" in standard_error["additional_data"]:
                elixir_error["supervisor"] = standard_error["additional_data"][
                    "supervisor"
                ]

            if "erlang_version" in standard_error["additional_data"]:
                elixir_error["erlang_version"] = standard_error["additional_data"][
                    "erlang_version"
                ]

            # Add any other additional data
            for key, value in standard_error["additional_data"].items():
                if key not in ["otp_app", "supervisor", "erlang_version"]:
                    elixir_error[key] = value

        # Add runtime information
        if "runtime" in standard_error:
            elixir_error["runtime"] = standard_error["runtime"]

            if "runtime_version" in standard_error:
                elixir_error["runtime_version"] = standard_error["runtime_version"]

        # Add handled flag if available
        if "handled" in standard_error:
            elixir_error["handled"] = standard_error["handled"]

        return elixir_error

    def _parse_elixir_stack_trace(self, stack_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parse an Elixir stack trace into structured frames.

        Args:
            stack_lines: Stack trace lines

        Returns:
            List of structured stack frames
        """
        frames = []

        # Regex patterns for Elixir stack trace lines
        # Example: (MyApp.SomeModule) lib/my_app/some_module.ex:42: MyApp.SomeModule.some_function/2
        # Example: (MyApp.SomeModule) lib/my_app/some_module.ex:42: MyApp.SomeModule.some_function(arg1, arg2)
        # Example: (MyApp.SomeModule) lib/my_app/some_module.ex:42: anonymous fn/3 in MyApp.SomeModule.some_function/1

        module_pattern = r"\s*\(([^)]+)\)\s+([^:]+):(\d+):\s+(.+)"
        func_pattern = r"([^/]+)/(\d+)"
        anonymous_pattern = r"anonymous fn/\d+ in ([^/]+)/\d+"

        for line in stack_lines:
            module_match = re.search(module_pattern, line)

            if module_match:
                module = module_match.group(1)
                file = module_match.group(2)
                line_num = int(module_match.group(3))
                func_info = module_match.group(4)

                # Extract function name and arity
                func_match = re.search(func_pattern, func_info)
                anon_match = re.search(anonymous_pattern, func_info)

                if func_match:
                    function = func_match.group(1)
                    arity = func_match.group(2)
                elif anon_match:
                    function = f"anonymous fn in {anon_match.group(1)}"
                    arity = "0"
                else:
                    # Try to extract function name from the pattern "Module.function(args)"
                    func_call_match = re.search(r"([^(]+)\(", func_info)
                    if func_call_match:
                        function = func_call_match.group(1)
                        arity = "?"
                    else:
                        function = func_info
                        arity = "?"

                # Split module into namespace parts
                parts = module.split(".")
                namespace = ".".join(parts[:-1]) if len(parts) > 1 else ""

                frames.append(
                    {
                        "module": module,
                        "namespace": namespace,
                        "function": function,
                        "arity": arity,
                        "file": file,
                        "line": line_num,
                    }
                )

        return frames if frames else []

    def _convert_frames_to_elixir_stacktrace(
        self, frames: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Convert structured stack frames to Elixir stacktrace format.

        Args:
            frames: Structured stack frames

        Returns:
            List of stacktrace lines in Elixir format
        """
        stacktrace = []

        for frame in frames:
            module = frame.get("module", "")
            function = frame.get("function", "")
            arity = frame.get("arity", "?")
            file = frame.get("file", "")
            line = frame.get("line", "?")

            # If module is missing but namespace exists, try to reconstruct it
            if not module and "namespace" in frame:
                if frame.get("class", ""):
                    module = (
                        f"{frame['namespace']}.{frame['class']}"
                        if frame["namespace"]
                        else frame["class"]
                    )

            # Format function info with arity
            func_info = f"{function}/{arity}" if arity != "?" else function

            # Format the stacktrace line
            stack_line = f"    ({module}) {file}:{line}: {func_info}"
            stacktrace.append(stack_line)

        return stacktrace


class ClojureErrorAdapter(LanguageAdapter):
    """Adapter for Clojure error formats."""

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Clojure error data to the standard format.

        Args:
            error_data: Clojure error data

        Returns:
            Error data in the standard format
        """
        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "clojure",
            "error_type": error_data.get("error_type", ""),
            "message": error_data.get("message", ""),
        }

        # Add Clojure version if available
        if "clojure_version" in error_data:
            standard_error["language_version"] = error_data["clojure_version"]

        # Handle stack trace
        if "stack_trace" in error_data:
            # Clojure stack traces can be a string or a list
            if isinstance(error_data["stack_trace"], str):
                # Split into lines
                stack_lines = error_data["stack_trace"].split("\n")

                # Try to parse structured data from the stack trace
                parsed_frames = self._parse_clojure_stack_trace(stack_lines)

                if parsed_frames:
                    standard_error["stack_trace"] = parsed_frames
                else:
                    standard_error["stack_trace"] = stack_lines
            elif isinstance(error_data["stack_trace"], list):
                if all(isinstance(frame, dict) for frame in error_data["stack_trace"]):
                    # Already in structured format
                    standard_error["stack_trace"] = error_data["stack_trace"]
                else:
                    # List of strings
                    standard_error["stack_trace"] = error_data["stack_trace"]

        # Add framework information if available
        if "framework" in error_data:
            standard_error["framework"] = error_data["framework"]

            if "framework_version" in error_data:
                standard_error["framework_version"] = error_data["framework_version"]

        # Add request information if available
        if "request" in error_data:
            standard_error["request"] = error_data["request"]

        # Add any additional context
        if "context" in error_data:
            standard_error["context"] = error_data["context"]

        # Add severity if available
        if "level" in error_data:
            # Map Clojure log levels to standard format
            level_map = {
                "trace": "debug",
                "debug": "debug",
                "info": "info",
                "warning": "warning",
                "warn": "warning",
                "error": "error",
                "fatal": "fatal",
            }
            standard_error["severity"] = level_map.get(
                error_data["level"].lower(), "error"
            )

        # Add runtime if available (JVM)
        if "runtime" in error_data:
            standard_error["runtime"] = error_data["runtime"]

            if "runtime_version" in error_data:
                standard_error["runtime_version"] = error_data["runtime_version"]

        # Add JVM information if available
        if "jvm_info" in error_data:
            if "additional_data" not in standard_error:
                standard_error["additional_data"] = {}
            standard_error["additional_data"]["jvm_info"] = error_data["jvm_info"]

        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]

        # Add namespace information specific to Clojure
        if "namespace" in error_data:
            if "additional_data" not in standard_error:
                standard_error["additional_data"] = {}
            standard_error["additional_data"]["namespace"] = error_data["namespace"]

        # Add var and form information specific to Clojure
        if "var" in error_data:
            if "additional_data" not in standard_error:
                standard_error["additional_data"] = {}
            standard_error["additional_data"]["var"] = error_data["var"]

        if "form" in error_data:
            if "additional_data" not in standard_error:
                standard_error["additional_data"] = {}
            standard_error["additional_data"]["form"] = error_data["form"]

        # Add additional Clojure-specific data
        clojure_specific = {}
        for key, value in error_data.items():
            if (
                key not in standard_error
                and key not in ["stack_trace", "request", "context"]
                and not key.startswith("_")
            ):
                clojure_specific[key] = value

        if clojure_specific:
            if "additional_data" in standard_error:
                standard_error["additional_data"].update(clojure_specific)
            else:
                standard_error["additional_data"] = clojure_specific

        return standard_error

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Clojure-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Clojure-specific format
        """
        # Create a Clojure error object
        clojure_error = {
            "timestamp": standard_error.get("timestamp", datetime.now().isoformat()),
            "error_type": standard_error.get("error_type", "clojure.lang.Exception"),
            "message": standard_error.get("message", ""),
        }

        # Convert severity to Clojure logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "debug",
                "info": "info",
                "warning": "warn",
                "error": "error",
                "critical": "error",
                "fatal": "fatal",
            }
            clojure_error["level"] = level_map.get(
                standard_error["severity"].lower(), "error"
            )

        # Convert stack trace to Clojure format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]

            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in Clojure stack trace string format
                    clojure_error["stack_trace"] = "\n".join(stack_trace)
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to Clojure stack trace format
                    clojure_error["stack_trace"] = (
                        self._convert_frames_to_clojure_stack(
                            standard_error.get("error_type", "clojure.lang.Exception"),
                            standard_error.get("message", ""),
                            stack_trace,
                        )
                    )
                    # Also keep the structured version
                    clojure_error["stack_frames"] = stack_trace

        # Add request information if available
        if "request" in standard_error:
            clojure_error["request"] = standard_error["request"]

        # Add context information if available
        if "context" in standard_error:
            clojure_error["context"] = standard_error["context"]

        # Add Clojure version if available
        if "language_version" in standard_error:
            clojure_error["clojure_version"] = standard_error["language_version"]

        # Add framework information if available
        if "framework" in standard_error:
            clojure_error["framework"] = standard_error["framework"]

            if "framework_version" in standard_error:
                clojure_error["framework_version"] = standard_error["framework_version"]

        # Add runtime information if available
        if "runtime" in standard_error:
            clojure_error["runtime"] = standard_error["runtime"]

            if "runtime_version" in standard_error:
                clojure_error["runtime_version"] = standard_error["runtime_version"]

        # Add handled flag if available
        if "handled" in standard_error:
            clojure_error["handled"] = standard_error["handled"]

        # Add additional data if available
        if "additional_data" in standard_error:
            # Extract JVM info as a top-level field
            if "jvm_info" in standard_error["additional_data"]:
                clojure_error["jvm_info"] = standard_error["additional_data"][
                    "jvm_info"
                ]

            # Extract Clojure-specific fields
            if "namespace" in standard_error["additional_data"]:
                clojure_error["namespace"] = standard_error["additional_data"][
                    "namespace"
                ]

            if "var" in standard_error["additional_data"]:
                clojure_error["var"] = standard_error["additional_data"]["var"]

            if "form" in standard_error["additional_data"]:
                clojure_error["form"] = standard_error["additional_data"]["form"]

            # Add remaining fields
            for key, value in standard_error["additional_data"].items():
                if key not in ["jvm_info", "namespace", "var", "form"]:
                    clojure_error[key] = value

        return clojure_error

    def _parse_clojure_stack_trace(
        self, stack_lines: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Parse a Clojure stack trace into structured frames.

        Args:
            stack_lines: Clojure stack trace lines

        Returns:
            Structured frames or None if parsing fails
        """
        frames = []

        # Clojure stack trace patterns:
        # 1. Standard Java interop: at namespace.function (file.clj:line)
        # 2. REPL: at namespace.function (REPL:line)
        # 3. Anonymous function: at namespace$function$fn__123.invoke (file.clj:line)
        # 4. Java method: at java.lang.Class.method (Source.java:line)

        # Pattern for anonymous functions (namespace$function$fn__XXX.invoke)
        anon_frame_pattern = r"\s*at\s+([a-zA-Z0-9_.-]+)\$([a-zA-Z0-9_$-]+)\$fn__(\d+)\.([a-zA-Z0-9_$-]+)\s*\(([^:)]+):(\d+)\)"
        # Pattern for regular Clojure functions (namespace$function.invoke)
        clojure_fn_pattern = r"\s*at\s+([a-zA-Z0-9_.-]+)\$([a-zA-Z0-9_$-]+)\.([a-zA-Z0-9_$-]+)\s*\(([^:)]+):(\d+)\)"
        clojure_frame_pattern = (
            r"\s*at\s+([a-zA-Z0-9_.$-]+)\.([a-zA-Z0-9_$-]+)\s*\(([^:)]+):(\d+)\)"
        )
        java_frame_pattern = r"\s*at\s+([a-zA-Z0-9_.]+)\.([a-zA-Z0-9_$]+)\.([a-zA-Z0-9_$]+)\(([^:)]+):(\d+)\)"

        for line in stack_lines:
            line = line.strip()
            if not line or line.startswith("Caused by"):
                continue

            # Try anonymous function pattern first
            match = re.match(anon_frame_pattern, line)
            if match:
                namespace = match.group(1)
                function = match.group(2)
                # fn_num = match.group(3)  # TODO: Use to distinguish anonymous functions
                method = match.group(4)
                file = match.group(5)
                line_num = int(match.group(6))

                function_desc = f"{function} (anonymous function)"

                frames.append(
                    {
                        "namespace": namespace,
                        "function": function_desc,
                        "file": file,
                        "line": line_num,
                        "type": "clojure",
                    }
                )
                continue

            # Try regular Clojure function pattern (namespace$function.invoke)
            match = re.match(clojure_fn_pattern, line)
            if match:
                namespace = match.group(1)
                function = match.group(2)
                method = match.group(3)
                file = match.group(4)
                line_num = int(match.group(5))

                # Check if this is actually a Java frame (has .java file)
                frame_type = "java" if file.endswith(".java") else "clojure"

                frame_dict = {
                    "namespace": namespace,
                    "function": function,
                    "file": file,
                    "line": line_num,
                    "type": frame_type,
                }

                # For Java frames, extract the class name
                if frame_type == "java" and "." in namespace:
                    parts = namespace.rsplit(".", 1)
                    frame_dict["namespace"] = parts[0]
                    frame_dict["class"] = parts[1]

                frames.append(frame_dict)
                continue

            # Try regular Clojure frame pattern
            match = re.match(clojure_frame_pattern, line)
            if match:
                namespace = match.group(1)
                function = match.group(2)
                file = match.group(3)
                line_num = int(match.group(4))

                # Check if this looks like an anonymous function
                if "$fn__" in function:
                    # Extract the original function name
                    base_function = function.split("$fn__")[0]
                    function = f"{base_function} (anonymous function)"

                # Check if this is actually a Java frame (has .java file)
                frame_type = "java" if file.endswith(".java") else "clojure"

                frame_dict = {
                    "namespace": namespace,
                    "function": function,
                    "file": file,
                    "line": line_num,
                    "type": frame_type,
                }

                # For Java frames, extract the class name
                if frame_type == "java" and "." in namespace:
                    parts = namespace.rsplit(".", 1)
                    frame_dict["namespace"] = parts[0]
                    frame_dict["class"] = parts[1]

                frames.append(frame_dict)
                continue

            # Try Java frame pattern
            match = re.match(java_frame_pattern, line)
            if match:
                package = match.group(1)
                class_name = match.group(2)
                method = match.group(3)
                file = match.group(4)
                line_num = int(match.group(5))

                frames.append(
                    {
                        "namespace": package,
                        "class": class_name,
                        "function": method,
                        "file": file,
                        "line": line_num,
                        "type": "java",
                    }
                )

        return frames if frames else []

    def _convert_frames_to_clojure_stack(
        self, error_type: str, message: str, frames: List[Dict[str, Any]]
    ) -> str:
        """
        Convert structured stack frames to Clojure stack trace format.

        Args:
            error_type: The exception type
            message: The error message
            frames: Structured stack frames

        Returns:
            Full stack trace string in Clojure format
        """
        lines = [f"{error_type}: {message}"]

        for frame in frames:
            frame_type = frame.get("type", "unknown")

            if frame_type == "clojure":
                namespace = frame.get("namespace", "")
                function = frame.get("function", "")
                file = frame.get("file", "")
                line = frame.get("line", "?")

                line_str = f"    at {namespace}.{function} ({file}:{line})"
                lines.append(line_str)

            elif frame_type == "java":
                namespace = frame.get("namespace", "")
                class_name = frame.get("class", "")
                method = frame.get("function", "")
                file = frame.get("file", "")
                line = frame.get("line", "?")

                line_str = f"    at {namespace}.{class_name}.{method} ({file}:{line})"
                lines.append(line_str)

            else:
                # Generic format
                namespace = frame.get("namespace", "")
                function = frame.get("function", "")
                file = frame.get("file", "")
                line = frame.get("line", "?")

                if namespace and function:
                    line_str = f"    at {namespace}.{function} ({file}:{line})"
                    lines.append(line_str)

        return "\n".join(lines)


class TypeScriptErrorAdapter(LanguageAdapter):
    """Adapter for TypeScript error formats."""

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert TypeScript error data to the standard format.

        Args:
            error_data: TypeScript error data

        Returns:
            Error data in the standard format
        """
        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "typescript",
            "error_type": error_data.get("name", error_data.get("code", "TSError")),
            "message": error_data.get("message", ""),
        }

        # Add TypeScript version if available
        if "typescript_version" in error_data:
            standard_error["language_version"] = error_data["typescript_version"]
        elif "ts_version" in error_data:
            standard_error["language_version"] = error_data["ts_version"]

        # Handle TypeScript error codes (TS2304, etc.)
        if "code" in error_data:
            if isinstance(error_data["code"], str) and error_data["code"].startswith(
                "TS"
            ):
                standard_error["error_code"] = error_data["code"]
                standard_error["error_type"] = error_data["code"]
            elif isinstance(error_data["code"], int):
                standard_error["error_code"] = f"TS{error_data['code']}"
                standard_error["error_type"] = f"TS{error_data['code']}"

        # Handle TypeScript compiler output format
        if "file" in error_data and "line" in error_data and "column" in error_data:
            file_info = {
                "file": error_data["file"],
                "line": error_data["line"],
                "column": error_data.get("column", 0),
            }
            standard_error["file_info"] = file_info

            # Create a simplified stack trace for TypeScript compilation errors
            stack_frame = (
                f"at {file_info['file']}:{file_info['line']}:{file_info['column']}"
            )
            standard_error["stack_trace"] = [stack_frame]

        # Handle stack trace from runtime TypeScript errors
        if "stack" in error_data:
            if isinstance(error_data["stack"], str):
                # Split into lines
                stack_lines = error_data["stack"].split("\n")

                # Try to parse structured data from the stack trace
                parsed_frames = self._parse_ts_stack_trace(stack_lines)

                if parsed_frames:
                    standard_error["stack_trace"] = parsed_frames
                else:
                    standard_error["stack_trace"] = stack_lines
            elif isinstance(error_data["stack"], list):
                # Already in a list format
                standard_error["stack_trace"] = error_data["stack"]

        # Add severity based on TypeScript error type
        if "severity" in error_data:
            standard_error["severity"] = error_data["severity"]
        else:
            # Map TypeScript error types to severity
            error_type = standard_error.get("error_type", "")
            if error_type.startswith("TS"):
                # Compilation errors are typically high severity
                if error_type in ["TS2304", "TS2305", "TS2307", "TS2322", "TS2339"]:
                    standard_error["severity"] = "error"
                elif error_type in ["TS6133", "TS7006"]:
                    standard_error["severity"] = "warning"
                else:
                    standard_error["severity"] = "error"
            else:
                standard_error["severity"] = "error"

        # Add framework information if available
        if "framework" in error_data:
            standard_error["framework"] = error_data["framework"]

            if "framework_version" in error_data:
                standard_error["framework_version"] = error_data["framework_version"]

        # Add TypeScript compiler options if available
        if "compiler_options" in error_data:
            standard_error["compiler_options"] = error_data["compiler_options"]

        # Add tsconfig.json path if available
        if "config_file" in error_data:
            standard_error["config_file"] = error_data["config_file"]

        # Add any additional context
        if "context" in error_data:
            standard_error["context"] = error_data["context"]

        # Add request information if available
        if "request" in error_data:
            standard_error["request"] = error_data["request"]

        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]

        # Add additional TypeScript-specific data
        ts_specific = {}
        for key, value in error_data.items():
            if key not in standard_error and key not in [
                "stack",
                "request",
                "context",
                "file",
                "line",
                "column",
            ]:
                ts_specific[key] = value

        if ts_specific:
            standard_error["additional_data"] = ts_specific

        return standard_error

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to TypeScript-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the TypeScript-specific format
        """
        # Create a TypeScript error object
        ts_error = {
            "timestamp": standard_error.get("timestamp", datetime.now().isoformat()),
            "name": standard_error.get("error_type", "TSError"),
            "message": standard_error.get("message", ""),
        }

        # Add TypeScript error code if available
        if "error_code" in standard_error:
            ts_error["code"] = standard_error["error_code"]
        elif standard_error.get("error_type", "").startswith("TS"):
            ts_error["code"] = standard_error["error_type"]

        # Add file information if available
        if "file_info" in standard_error:
            file_info = standard_error["file_info"]
            ts_error["file"] = file_info.get("file", "")
            ts_error["line"] = file_info.get("line", 0)
            ts_error["column"] = file_info.get("column", 0)

        # Convert severity to TypeScript diagnostic level
        if "severity" in standard_error:
            severity_map = {
                "debug": "suggestion",
                "info": "info",
                "warning": "warning",
                "error": "error",
                "critical": "error",
                "fatal": "error",
            }
            ts_error["severity"] = severity_map.get(
                standard_error["severity"].lower(), "error"
            )

        # Convert stack trace to TypeScript format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]

            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in TypeScript stack trace string format
                    ts_error["stack"] = "\n".join(stack_trace)
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to TypeScript stack trace format
                    ts_error["stack"] = self._convert_frames_to_ts_stack(
                        standard_error["error_type"],
                        standard_error["message"],
                        stack_trace,
                    )

        # Add TypeScript version if available
        if "language_version" in standard_error:
            ts_error["typescript_version"] = standard_error["language_version"]

        # Add compiler options if available
        if "compiler_options" in standard_error:
            ts_error["compiler_options"] = standard_error["compiler_options"]

        # Add config file if available
        if "config_file" in standard_error:
            ts_error["config_file"] = standard_error["config_file"]

        # Add framework information if available
        if "framework" in standard_error:
            ts_error["framework"] = standard_error["framework"]

            if "framework_version" in standard_error:
                ts_error["framework_version"] = standard_error["framework_version"]

        # Add request information if available
        if "request" in standard_error:
            ts_error["request"] = standard_error["request"]

        # Add context information if available
        if "context" in standard_error:
            ts_error["context"] = standard_error["context"]

        # Add handled flag if available
        if "handled" in standard_error:
            ts_error["handled"] = standard_error["handled"]

        # Add additional data if available
        if "additional_data" in standard_error:
            for key, value in standard_error["additional_data"].items():
                ts_error[key] = value

        return ts_error

    def _parse_ts_stack_trace(self, stack_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parse a TypeScript stack trace into structured frames.

        Args:
            stack_lines: TypeScript stack trace lines

        Returns:
            Structured frames or None if parsing fails
        """
        frames = []

        # TypeScript stack traces are similar to JavaScript but may include .ts files
        # Chrome: at functionName (file.ts:line:column)
        # Node.js: at functionName (file.ts:line:column)
        # TypeScript compilation: file.ts(line,column): error TSxxxx
        frame_patterns = [
            r"\s*at\s+([^(]+)\s+\(([^:]+\.tsx?):(\d+):(\d+)\)",  # Chrome/Node with function and TS file
            r"\s*at\s+([^:]+\.tsx?):(\d+):(\d+)",  # Chrome/Node without function, TS file
            r"\s*([^@]+)@([^:]+\.tsx?):(\d+):(\d+)",  # Firefox with TS file
            r"([^(]+)\((\d+),(\d+)\):\s*error\s+TS\d+",  # TypeScript compiler format
            r"\s*at\s+([^(]+)\s+\(([^:]+):(\d+):(\d+)\)",  # Fallback to regular JS patterns
            r"\s*at\s+([^:]+):(\d+):(\d+)",  # Fallback without function
            r"\s*([^@]+)@([^:]+):(\d+):(\d+)",  # Fallback Firefox
        ]

        try:
            for line in stack_lines:
                for pattern in frame_patterns:
                    match = re.search(pattern, line)
                    if match:
                        groups = match.groups()

                        if len(groups) == 4:  # Full match with function
                            frames.append(
                                {
                                    "function": groups[0].strip(),
                                    "file": groups[1],
                                    "line": int(groups[2]),
                                    "column": int(groups[3]),
                                }
                            )
                        elif (
                            len(groups) == 3
                        ):  # File, line, column (compilation error format)
                            frames.append(
                                {
                                    "function": "<compilation>",
                                    "file": groups[0],
                                    "line": int(groups[1]),
                                    "column": int(groups[2]),
                                }
                            )
                        break

            return frames if frames else None
        except Exception as e:
            logger.debug(f"Failed to parse TypeScript stack trace: {e}")
            return None

    def _convert_frames_to_ts_stack(
        self, error_type: str, message: str, frames: List[Dict[str, Any]]
    ) -> str:
        """
        Convert structured frames to a TypeScript stack trace string.

        Args:
            error_type: Error type/name
            message: Error message
            frames: Structured frames

        Returns:
            TypeScript stack trace string
        """
        # Start with the error message
        stack_lines = [f"{error_type}: {message}"]

        # Add frames in TypeScript format
        for frame in frames:
            function = frame.get("function", "unknown")
            file_path = frame.get("file", "unknown")
            line_num = frame.get("line", 0)
            col_num = frame.get("column", 0)

            # Format like a TypeScript/JavaScript stack trace
            if function == "<compilation>":
                # Compilation error format
                stack_lines.append(f"    at {file_path}({line_num},{col_num})")
            else:
                # Runtime error format
                stack_lines.append(
                    f"    at {function} ({file_path}:{line_num}:{col_num})"
                )

        return "\n".join(stack_lines)


class ReactErrorAdapter(LanguageAdapter):
    """Adapter for React framework error formats."""

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert React error data to the standard format.

        Args:
            error_data: React error data

        Returns:
            Error data in the standard format
        """
        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "react",
            "framework": "react",
            "error_type": error_data.get("name", error_data.get("type", "")),
            "message": error_data.get("message", ""),
        }

        # Add React version if available
        if "react_version" in error_data:
            standard_error["framework_version"] = error_data["react_version"]
        elif "framework_version" in error_data:
            standard_error["framework_version"] = error_data["framework_version"]

        # Handle stack trace - React errors may come from JavaScript runtime
        if "stack" in error_data:
            standard_error["stack_trace"] = error_data["stack"]
        elif "stacktrace" in error_data:
            standard_error["stack_trace"] = error_data["stacktrace"]
        elif "stack_trace" in error_data:
            standard_error["stack_trace"] = error_data["stack_trace"]

        # Handle React-specific component stack
        if "componentStack" in error_data:
            standard_error["component_stack"] = error_data["componentStack"]
        elif "component_stack" in error_data:
            standard_error["component_stack"] = error_data["component_stack"]

        # Add React-specific error info
        react_specific = {}

        # Error boundary information
        if "errorInfo" in error_data:
            react_specific["error_info"] = error_data["errorInfo"]

        # Component that caused the error
        if "source" in error_data:
            react_specific["error_source"] = error_data["source"]

        # React development mode information
        if "development" in error_data:
            react_specific["development_mode"] = error_data["development"]

        # React DevTools information
        if "devtools" in error_data:
            react_specific["devtools_info"] = error_data["devtools"]

        # Hook information for hook-related errors
        if "hook" in error_data:
            react_specific["hook_info"] = error_data["hook"]

        # JSX transform information
        if "jsx_runtime" in error_data:
            react_specific["jsx_runtime"] = error_data["jsx_runtime"]

        # Server component information
        if "server_component" in error_data:
            react_specific["server_component"] = error_data["server_component"]

        # Add framework information
        if "bundler" in error_data:
            react_specific["bundler"] = error_data["bundler"]

        # Add severity if available
        if "level" in error_data:
            # React errors typically use console levels
            level_map = {
                "log": "info",
                "info": "info",
                "warn": "warning",
                "warning": "warning",
                "error": "error",
                "assert": "error",
            }
            standard_error["severity"] = level_map.get(
                error_data["level"].lower(), "error"
            )
        elif "severity" in error_data:
            standard_error["severity"] = error_data["severity"]

        # Handle React runtime environment
        if "runtime" in error_data:
            standard_error["runtime"] = error_data["runtime"]
        elif "environment" in error_data:
            standard_error["runtime"] = error_data["environment"]

        # Add request information if available (for SSR)
        if "request" in error_data:
            standard_error["request"] = error_data["request"]

        # Add any additional context
        if "context" in error_data:
            standard_error["context"] = error_data["context"]

        # Add React-specific data
        if react_specific:
            standard_error["additional_data"] = react_specific

        # Add any other fields not explicitly handled
        for key, value in error_data.items():
            if key not in [
                "name",
                "type",
                "message",
                "stack",
                "stacktrace",
                "stack_trace",
                "componentStack",
                "component_stack",
                "errorInfo",
                "source",
                "development",
                "devtools",
                "hook",
                "jsx_runtime",
                "server_component",
                "bundler",
                "level",
                "severity",
                "runtime",
                "environment",
                "request",
                "context",
                "timestamp",
                "react_version",
                "framework_version",
            ]:
                if "additional_data" not in standard_error:
                    standard_error["additional_data"] = {}
                standard_error["additional_data"][key] = value

        return standard_error

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to React-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the React-specific format
        """
        # Create a React error object
        react_error = {
            "timestamp": standard_error.get("timestamp", datetime.now().isoformat()),
            "name": standard_error.get("error_type", "Error"),
            "message": standard_error.get("message", ""),
        }

        # Add React version if available
        if "framework_version" in standard_error:
            react_error["react_version"] = standard_error["framework_version"]

        # Convert stack trace to React/JavaScript format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]

            if isinstance(stack_trace, str):
                react_error["stack"] = stack_trace
            elif isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Join string frames
                    react_error["stack"] = "\n".join(stack_trace)
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to React/JS stack format
                    react_error["stack"] = self._convert_frames_to_react_stack(
                        react_error["name"], react_error["message"], stack_trace
                    )

        # Add React component stack if available
        if "component_stack" in standard_error:
            react_error["componentStack"] = standard_error["component_stack"]

        # Convert severity to React/console level
        if "severity" in standard_error:
            level_map = {
                "debug": "log",
                "info": "info",
                "warning": "warn",
                "error": "error",
                "critical": "error",
                "fatal": "error",
            }
            react_error["level"] = level_map.get(
                standard_error["severity"].lower(), "error"
            )

        # Add runtime information if available
        if "runtime" in standard_error:
            react_error["runtime"] = standard_error["runtime"]

        # Add request information if available (for SSR)
        if "request" in standard_error:
            react_error["request"] = standard_error["request"]

        # Add context information if available
        if "context" in standard_error:
            react_error["context"] = standard_error["context"]

        # Add React-specific data from additional_data
        if "additional_data" in standard_error:
            additional = standard_error["additional_data"]

            # Extract React-specific fields
            react_fields = [
                "error_info",
                "error_source",
                "development_mode",
                "devtools_info",
                "hook_info",
                "jsx_runtime",
                "server_component",
                "bundler",
            ]

            for field in react_fields:
                if field in additional:
                    # Convert back to original field names
                    if field == "error_info":
                        react_error["errorInfo"] = additional[field]
                    elif field == "error_source":
                        react_error["source"] = additional[field]
                    elif field == "development_mode":
                        react_error["development"] = additional[field]
                    elif field == "devtools_info":
                        react_error["devtools"] = additional[field]
                    elif field == "hook_info":
                        react_error["hook"] = additional[field]
                    else:
                        react_error[field] = additional[field]

            # Add any other additional data
            for key, value in additional.items():
                if key not in react_fields:
                    react_error[key] = value

        return react_error

    def _convert_frames_to_react_stack(
        self, error_type: str, message: str, frames: List[Dict[str, Any]]
    ) -> str:
        """
        Convert structured frames to a React/JavaScript stack trace string.

        Args:
            error_type: Error type/name
            message: Error message
            frames: Structured frames

        Returns:
            React/JavaScript stack trace string
        """
        # Start with the error message
        stack_lines = [f"{error_type}: {message}"]

        # Add frames in JavaScript/React format
        for frame in frames:
            function = frame.get("function", "unknown")
            file_path = frame.get("file", "unknown")
            line_num = frame.get("line", 0)
            col_num = frame.get("column", 0)

            # Format like a JavaScript stack trace
            if function == "unknown" or function == "<anonymous>":
                stack_lines.append(f"    at {file_path}:{line_num}:{col_num}")
            else:
                stack_lines.append(
                    f"    at {function} ({file_path}:{line_num}:{col_num})"
                )

        return "\n".join(stack_lines)

    def extract_component_stack(
        self, error_data: Dict[str, Any]
    ) -> Optional[List[str]]:
        """
        Extract React component stack from error data.

        Args:
            error_data: React error data

        Returns:
            List of component names in the stack, or None if not available
        """
        component_stack = error_data.get("componentStack") or error_data.get(
            "component_stack"
        )

        if not component_stack:
            return None

        if isinstance(component_stack, str):
            # Parse component stack string
            # React component stacks look like:
            # "    in ComponentName (at Component.jsx:10:5)\n    in App (at App.jsx:15:3)"
            components = []
            lines = component_stack.strip().split("\n")

            for line in lines:
                line = line.strip()
                if line.startswith("in "):
                    # Extract component name
                    match = re.match(r"in (\w+)", line)
                    if match:
                        components.append(match.group(1))

            return components if components else None
        elif isinstance(component_stack, list):
            return component_stack

        return None


class SwiftErrorAdapter(LanguageAdapter):
    """Adapter for Swift error formats."""

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Swift error data to the standard format.

        Args:
            error_data: Swift error data

        Returns:
            Error data in the standard format
        """
        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "swift",
            "language_version": error_data.get("swift_version", ""),
            "error_type": error_data.get("error_type", ""),
            "message": error_data.get("message", ""),
        }

        # Add Swift version if available
        if "swift_version" in error_data:
            standard_error["language_version"] = error_data["swift_version"]

        # Handle stack trace
        if "stack_trace" in error_data:
            standard_error["stack_trace"] = error_data["stack_trace"]
        elif "backtrace" in error_data:
            standard_error["stack_trace"] = error_data["backtrace"]

        # Handle iOS/macOS specific crash info
        if "crash_info" in error_data:
            crash_info = error_data["crash_info"]

            # Extract signal information
            if "signal" in crash_info:
                if not standard_error["error_type"]:
                    standard_error["error_type"] = crash_info["signal"]

            # Extract exception information
            if "exception" in crash_info:
                exception = crash_info["exception"]
                if "type" in exception and not standard_error["error_type"]:
                    standard_error["error_type"] = exception["type"]
                if "message" in exception and not standard_error["message"]:
                    standard_error["message"] = exception["message"]

            # Extract thread information
            if "threads" in crash_info:
                standard_error["stack_trace"] = crash_info["threads"]

        # Add platform information
        if "platform" in error_data:
            standard_error["runtime"] = error_data["platform"]
        elif "os" in error_data:
            standard_error["runtime"] = error_data["os"]

        # Add framework information if available
        if "framework" in error_data:
            standard_error["framework"] = error_data["framework"]

            if "framework_version" in error_data:
                standard_error["framework_version"] = error_data["framework_version"]

        # Add Xcode/build information
        if "xcode_version" in error_data:
            standard_error["build_info"] = {
                "xcode_version": error_data["xcode_version"]
            }
            if "build_number" in error_data:
                standard_error["build_info"]["build_number"] = error_data[
                    "build_number"
                ]

        # Add device information if available
        if "device_info" in error_data:
            standard_error["context"] = {"device": error_data["device_info"]}

        # Add severity if available
        if "level" in error_data:
            # Convert iOS logging levels to the standard format
            level_map = {
                "DEFAULT": "info",
                "INFO": "info",
                "DEBUG": "debug",
                "ERROR": "error",
                "FAULT": "critical",
                "FATAL": "fatal",
            }
            standard_error["severity"] = level_map.get(
                error_data["level"].upper(), "error"
            )
        elif "fatal error" in standard_error["message"].lower():
            standard_error["severity"] = "fatal"
        elif "warning" in standard_error["message"].lower():
            standard_error["severity"] = "warning"
        else:
            standard_error["severity"] = "error"

        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]
        elif "fatal error" in standard_error["message"].lower():
            standard_error["handled"] = False

        # Add additional Swift-specific data
        swift_specific = {}
        for key, value in error_data.items():
            if key not in standard_error and key not in [
                "stack_trace",
                "backtrace",
                "crash_info",
                "device_info",
                "xcode_version",
                "build_number",
                "platform",
                "os",
            ]:
                swift_specific[key] = value

        if swift_specific:
            standard_error["additional_data"] = swift_specific

        return standard_error

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Swift-specific format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Swift-specific format
        """
        # Create a Swift error object
        swift_error = {
            "timestamp": standard_error.get("timestamp", datetime.now().isoformat()),
            "error_type": standard_error.get("error_type", "Error"),
            "message": standard_error.get("message", ""),
        }

        # Add Swift version
        if "language_version" in standard_error:
            swift_error["swift_version"] = standard_error["language_version"]

        # Convert severity to iOS logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "DEBUG",
                "info": "INFO",
                "warning": "DEFAULT",
                "error": "ERROR",
                "critical": "FAULT",
                "fatal": "FAULT",
            }
            swift_error["level"] = level_map.get(
                standard_error["severity"].lower(), "ERROR"
            )

        # Convert stack trace to Swift format
        if "stack_trace" in standard_error:
            swift_error["stack_trace"] = self._format_swift_stack_trace(
                standard_error["stack_trace"]
            )

        # Add runtime platform
        if "runtime" in standard_error:
            swift_error["platform"] = standard_error["runtime"]

        # Add framework information
        if "framework" in standard_error:
            swift_error["framework"] = standard_error["framework"]
            if "framework_version" in standard_error:
                swift_error["framework_version"] = standard_error["framework_version"]

        # Add build information
        if "build_info" in standard_error:
            build_info = standard_error["build_info"]
            if "xcode_version" in build_info:
                swift_error["xcode_version"] = build_info["xcode_version"]
            if "build_number" in build_info:
                swift_error["build_number"] = build_info["build_number"]

        # Add device context
        if "context" in standard_error and "device" in standard_error["context"]:
            swift_error["device_info"] = standard_error["context"]["device"]

        # Add handled flag
        if "handled" in standard_error:
            swift_error["handled"] = standard_error["handled"]

        # Add any additional data
        if "additional_data" in standard_error:
            swift_error.update(standard_error["additional_data"])

        return swift_error

    def _format_swift_stack_trace(self, stack_trace: Union[List, str]) -> List[str]:
        """
        Format stack trace for Swift.

        Args:
            stack_trace: Stack trace data

        Returns:
            List of formatted stack trace strings
        """
        if isinstance(stack_trace, str):
            return stack_trace.split("\n")

        if not isinstance(stack_trace, list):
            return []

        # If it's already a list of strings, return as-is
        if all(isinstance(frame, str) for frame in stack_trace):
            return stack_trace

        # Convert structured frames to Swift format
        if all(isinstance(frame, dict) for frame in stack_trace):
            frames = []
            for i, frame in enumerate(stack_trace):
                function = frame.get("function", "unknown")
                file_path = frame.get("file", "unknown")
                line_num = frame.get("line", 0)

                # Extract just the filename for cleaner display
                filename = file_path.split("/")[-1] if "/" in file_path else file_path

                # Format like Swift stack trace
                frames.append(f"{i}\t{filename}\t{function} + {line_num}")

            return frames

        # Fallback: convert to strings
        return [str(frame) for frame in stack_trace]


class KotlinErrorAdapter(LanguageAdapter):
    """Adapter for Kotlin errors."""

    def to_standard_format(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Kotlin error data to the standard format.

        Args:
            error_data: Kotlin error data

        Returns:
            Error data in the standard format
        """
        # Base error structure
        standard_error = {
            "id": error_data.get("id", str(uuid.uuid4())),
            "timestamp": error_data.get("timestamp", datetime.utcnow().isoformat()),
            "language": "kotlin",
            "error_type": error_data.get(
                "error_type", error_data.get("type", "Exception")
            ),
            "message": error_data.get("message", ""),
            "severity": error_data.get("severity", "error"),
            "handled": error_data.get("handled", False),
        }

        # Extract file and line information
        if "file" in error_data and "line" in error_data:
            standard_error["file"] = error_data["file"]
            standard_error["line"] = error_data["line"]
        elif "location" in error_data:
            # Parse location like "MainActivity.kt:42"
            location = error_data["location"]
            if ":" in location:
                file_part, line_part = location.rsplit(":", 1)
                try:
                    standard_error["file"] = file_part
                    standard_error["line"] = int(line_part)
                except ValueError:
                    standard_error["file"] = location
                    standard_error["line"] = 0
            else:
                standard_error["file"] = location
                standard_error["line"] = 0

        # Extract function/method information
        if "function" in error_data:
            standard_error["function"] = error_data["function"]
        elif "method" in error_data:
            standard_error["function"] = error_data["method"]

        # Extract stack trace
        if "stack_trace" in error_data:
            if isinstance(error_data["stack_trace"], str):
                # Parse string stack trace
                standard_error["stack_trace"] = self._parse_kotlin_stack_trace(
                    error_data["stack_trace"]
                )
            elif isinstance(error_data["stack_trace"], list):
                if error_data["stack_trace"] and isinstance(
                    error_data["stack_trace"][0], dict
                ):
                    # Structured stack trace
                    standard_error["stack_trace"] = error_data["stack_trace"]
                else:
                    # List of strings
                    standard_error["stack_trace"] = error_data["stack_trace"]
        elif "stackTrace" in error_data:
            # Handle camelCase variant
            if isinstance(error_data["stackTrace"], str):
                standard_error["stack_trace"] = self._parse_kotlin_stack_trace(
                    error_data["stackTrace"]
                )
            else:
                standard_error["stack_trace"] = error_data["stackTrace"]

        # Add framework information if available
        if "framework" in error_data:
            standard_error["framework"] = error_data["framework"]

            if "framework_version" in error_data:
                standard_error["framework_version"] = error_data["framework_version"]

        # Add Android-specific information
        if "android" in error_data:
            android_info = error_data["android"]
            standard_error["android"] = android_info

            # Extract API level
            if "api_level" in android_info:
                standard_error["android_api_level"] = android_info["api_level"]

            # Extract device info
            if "device" in android_info:
                standard_error["device"] = android_info["device"]

        # Add request information if available (for server-side Kotlin)
        if "request" in error_data:
            standard_error["request"] = error_data["request"]

        # Add any additional context
        if "context" in error_data:
            standard_error["context"] = error_data["context"]

        # Add severity mapping from Android log levels
        if "level" in error_data:
            level_map = {
                "verbose": "debug",
                "debug": "debug",
                "info": "info",
                "warn": "warning",
                "error": "error",
                "assert": "critical",
            }
            standard_error["severity"] = level_map.get(
                error_data["level"].lower(), "error"
            )

        # Add runtime information
        if "kotlin_version" in error_data:
            standard_error["runtime"] = "Kotlin"
            standard_error["runtime_version"] = error_data["kotlin_version"]
        elif "jvm" in error_data:
            standard_error["runtime"] = "JVM"
            standard_error["runtime_version"] = error_data["jvm"]

        # Add coroutine information if available
        if "coroutine" in error_data:
            standard_error["coroutine"] = error_data["coroutine"]

        # Add thread information
        if "thread" in error_data:
            standard_error["thread"] = error_data["thread"]
        elif "thread_name" in error_data:
            standard_error["thread"] = error_data["thread_name"]

        # Add additional Kotlin-specific data
        kotlin_specific = {}
        for key, value in error_data.items():
            if key not in standard_error and key not in [
                "stack_trace",
                "stackTrace",
                "request",
                "context",
                "android",
            ]:
                kotlin_specific[key] = value

        if kotlin_specific:
            standard_error["kotlin_specific"] = kotlin_specific

        return standard_error

    def from_standard_format(self, standard_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard format error data to Kotlin format.

        Args:
            standard_error: Error data in the standard format

        Returns:
            Error data in the Kotlin format
        """
        # Base Kotlin error structure
        kotlin_error = {
            "id": standard_error.get("id"),
            "timestamp": standard_error.get("timestamp"),
            "type": standard_error.get("error_type", "Exception"),
            "message": standard_error.get("message", ""),
            "severity": standard_error.get("severity", "error"),
            "handled": standard_error.get("handled", False),
        }

        # Add location information
        if "file" in standard_error and "line" in standard_error:
            kotlin_error["location"] = (
                f"{standard_error['file']}:{standard_error['line']}"
            )
        elif "file" in standard_error:
            kotlin_error["location"] = standard_error["file"]

        # Add function information
        if "function" in standard_error:
            kotlin_error["function"] = standard_error["function"]

        # Convert stack trace to Kotlin format
        if "stack_trace" in standard_error:
            kotlin_error["stackTrace"] = self._format_kotlin_stack_trace(
                standard_error["stack_trace"]
            )

        # Add framework information
        if "framework" in standard_error:
            kotlin_error["framework"] = standard_error["framework"]

            if "framework_version" in standard_error:
                kotlin_error["framework_version"] = standard_error["framework_version"]

        # Add Android-specific information
        if "android" in standard_error:
            kotlin_error["android"] = standard_error["android"]
        elif "android_api_level" in standard_error or "device" in standard_error:
            android_info = {}
            if "android_api_level" in standard_error:
                android_info["api_level"] = standard_error["android_api_level"]
            if "device" in standard_error:
                android_info["device"] = standard_error["device"]
            kotlin_error["android"] = android_info

        # Add request information
        if "request" in standard_error:
            kotlin_error["request"] = standard_error["request"]

        # Add context
        if "context" in standard_error:
            kotlin_error["context"] = standard_error["context"]

        # Add runtime information
        if "runtime" in standard_error and standard_error["runtime"] == "Kotlin":
            if "runtime_version" in standard_error:
                kotlin_error["kotlin_version"] = standard_error["runtime_version"]
        elif "runtime" in standard_error and standard_error["runtime"] == "JVM":
            if "runtime_version" in standard_error:
                kotlin_error["jvm"] = standard_error["runtime_version"]

        # Add coroutine information
        if "coroutine" in standard_error:
            kotlin_error["coroutine"] = standard_error["coroutine"]

        # Add thread information
        if "thread" in standard_error:
            kotlin_error["thread"] = standard_error["thread"]

        # Add Kotlin-specific data
        if "kotlin_specific" in standard_error:
            kotlin_error.update(standard_error["kotlin_specific"])

        return kotlin_error

    def _parse_kotlin_stack_trace(self, stack_trace_str: str) -> List[Dict[str, Any]]:
        """
        Parse Kotlin stack trace string into structured format.

        Args:
            stack_trace_str: Stack trace as string

        Returns:
            List of structured stack frames
        """
        if not stack_trace_str:
            return []

        frames = []
        lines = stack_trace_str.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("Caused by:"):
                continue

            # Match Kotlin/Java stack trace format:
            # at com.example.MainActivity.onCreate(MainActivity.kt:42)
            # at kotlinx.coroutines.CoroutineExceptionHandler.handleException(...)
            match = re.match(r"\s*at\s+([^(]+)\(([^:)]+)(?::(\d+))?\)", line)

            if match:
                full_method = match.group(1)
                file_name = match.group(2)
                line_number = int(match.group(3)) if match.group(3) else 0

                # Split package.class.method
                if "." in full_method:
                    parts = full_method.rsplit(".", 2)
                    if len(parts) >= 2:
                        package_class = ".".join(parts[:-1])
                        method = parts[-1]

                        # Try to separate package and class
                        if "." in package_class:
                            package_parts = package_class.split(".")
                            # Assume class name starts with uppercase
                            for i in range(len(package_parts) - 1, -1, -1):
                                if package_parts[i] and package_parts[i][0].isupper():
                                    package = ".".join(package_parts[:i])
                                    class_name = ".".join(package_parts[i:])
                                    break
                            else:
                                package = ".".join(package_parts[:-1])
                                class_name = package_parts[-1]
                        else:
                            package = ""
                            class_name = package_class
                    else:
                        package = ""
                        class_name = ""
                        method = full_method
                else:
                    package = ""
                    class_name = ""
                    method = full_method

                frame = {
                    "function": method,
                    "class": class_name,
                    "package": package,
                    "file": file_name,
                    "line": line_number,
                    "raw": line,
                }
                frames.append(frame)
            else:
                # If we can't parse it, keep the raw line
                frames.append({"raw": line})

        return frames

    def _format_kotlin_stack_trace(
        self, stack_trace: List[Union[str, Dict[str, Any]]]
    ) -> List[str]:
        """
        Format stack trace for Kotlin format output.

        Args:
            stack_trace: List of stack frames (strings or dicts)

        Returns:
            List of formatted stack trace strings
        """
        if not stack_trace:
            return []

        if not isinstance(stack_trace, list):
            return []

        # If it's already a list of strings, return as-is
        if all(isinstance(frame, str) for frame in stack_trace):
            return stack_trace

        # Convert structured frames to Kotlin format
        if all(isinstance(frame, dict) for frame in stack_trace):
            frames = []
            for frame in stack_trace:
                if "raw" in frame:
                    frames.append(frame["raw"])
                else:
                    # Reconstruct stack trace line
                    package = frame.get("package", "")
                    class_name = frame.get("class", "")
                    function = frame.get("function", "unknown")
                    file_name = frame.get("file", "unknown")
                    line_num = frame.get("line", 0)

                    # Build full method name
                    method_parts = []
                    if package:
                        method_parts.append(package)
                    if class_name:
                        method_parts.append(class_name)
                    method_parts.append(function)

                    full_method = ".".join(method_parts)

                    if line_num > 0:
                        frames.append(f"at {full_method}({file_name}:{line_num})")
                    else:
                        frames.append(f"at {full_method}({file_name})")

            return frames

        # Fallback: convert to strings
        return [str(frame) for frame in stack_trace]
