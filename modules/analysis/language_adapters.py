"""
Multi-Language Error Adapters

This module provides adapters for converting language-specific error formats
to the standardized Homeostasis error schema and back. It enables cross-language
error analysis and pattern matching.
"""
import json
import re
import uuid
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import os

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
        
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
        
        try:
            # Try to import jsonschema for validation
            import jsonschema
            self.jsonschema = jsonschema
            self.can_validate = True
        except ImportError:
            logger.warning("jsonschema package not available, schema validation disabled")
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
            "message": error_data.get("message", "")
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
            if "detailed_frames" in details and isinstance(details["detailed_frames"], list):
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
                "FATAL": "fatal"
            }
            standard_error["severity"] = level_map.get(error_data["level"].upper(), "error")
        
        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]
        
        # Add additional Python-specific data
        python_specific = {}
        for key, value in error_data.items():
            if key not in standard_error and key not in ["traceback", "error_details", "request", "context"]:
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
            "message": standard_error.get("message", "")
        }
        
        # Convert severity to Python logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "DEBUG",
                "info": "INFO",
                "warning": "WARNING",
                "error": "ERROR",
                "critical": "CRITICAL",
                "fatal": "FATAL"
            }
            python_error["level"] = level_map.get(standard_error["severity"].lower(), "ERROR")
        
        # Convert stack trace to Python format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]
            
            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in Python traceback format
                    python_error["traceback"] = stack_trace
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to Python traceback format
                    python_error["traceback"] = self._convert_frames_to_traceback(stack_trace)
                    
                    # Also add detailed frames
                    python_error["error_details"] = {
                        "exception_type": standard_error.get("error_type", ""),
                        "message": standard_error.get("message", ""),
                        "detailed_frames": stack_trace
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
            traceback_lines.append(f'  File "{file_path}", line {line_num}, in {function}')
            
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
            "message": error_data.get("message", "")
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
                "critical": "critical"
            }
            standard_error["severity"] = level_map.get(error_data["level"].lower(), "error")
        
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
            "message": standard_error.get("message", "")
        }
        
        # Convert severity to JavaScript logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "debug",
                "info": "info",
                "warning": "warn",
                "error": "error",
                "critical": "error",
                "fatal": "fatal"
            }
            js_error["level"] = level_map.get(standard_error["severity"].lower(), "error")
        
        # Convert stack trace to JavaScript format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]
            
            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in JavaScript stack trace string format
                    js_error["stack"] = "\n".join(stack_trace)
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to JavaScript stack trace format
                    js_error["stack"] = self._convert_frames_to_js_stack(standard_error["error_type"], 
                                                                         standard_error["message"], 
                                                                         stack_trace)
        
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
            r'\s*at\s+([^(]+)\s+\(([^:]+):(\d+):(\d+)\)',  # Chrome/Node with function
            r'\s*at\s+([^:]+):(\d+):(\d+)',  # Chrome/Node without function
            r'\s*([^@]+)@([^:]+):(\d+):(\d+)'  # Firefox
        ]
        
        try:
            for line in stack_lines:
                for pattern in frame_patterns:
                    match = re.search(pattern, line)
                    if match:
                        if len(match.groups()) == 4:  # Full match with function
                            frames.append({
                                "function": match.group(1).strip(),
                                "file": match.group(2),
                                "line": int(match.group(3)),
                                "column": int(match.group(4))
                            })
                        elif len(match.groups()) == 3:  # Missing function name
                            frames.append({
                                "file": match.group(1),
                                "line": int(match.group(2)),
                                "column": int(match.group(3))
                            })
                        break
                        
            return frames if frames else None
        except Exception as e:
            logger.debug(f"Failed to parse JS stack trace: {e}")
            return None
    
    def _convert_frames_to_js_stack(self, error_type: str, message: str, 
                                   frames: List[Dict[str, Any]]) -> str:
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
        # Create a standard error object
        standard_error = {
            "error_id": str(uuid.uuid4()),
            "timestamp": error_data.get("timestamp", datetime.now().isoformat()),
            "language": "java",
            "error_type": error_data.get("exception_class", ""),
            "message": error_data.get("message", "")
        }
        
        # Add Java version if available
        if "java_version" in error_data:
            standard_error["language_version"] = error_data["java_version"]
        
        # Handle stack trace
        if "stack_trace" in error_data:
            # Java stack traces can be a string or a list
            if isinstance(error_data["stack_trace"], str):
                # Split into lines
                stack_lines = error_data["stack_trace"].split("\n")
                
                # Try to parse structured data from the stack trace
                parsed_frames = self._parse_java_stack_trace(stack_lines)
                
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
            # Map Java log levels to standard format
            level_map = {
                "finest": "debug",
                "finer": "debug",
                "fine": "debug",
                "config": "info",
                "info": "info",
                "warning": "warning",
                "severe": "error"
            }
            standard_error["severity"] = level_map.get(error_data["level"].lower(), "error")
        
        # Add runtime if available
        if "jvm" in error_data:
            standard_error["runtime"] = "JVM"
            standard_error["runtime_version"] = error_data["jvm"]
        
        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]
        
        # Add additional Java-specific data
        java_specific = {}
        for key, value in error_data.items():
            if key not in standard_error and key not in ["stack_trace", "request", "context"]:
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
            "exception_class": standard_error.get("error_type", "java.lang.Exception"),
            "message": standard_error.get("message", "")
        }
        
        # Convert severity to Java logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "fine",
                "info": "info",
                "warning": "warning",
                "error": "severe",
                "critical": "severe",
                "fatal": "severe"
            }
            java_error["level"] = level_map.get(standard_error["severity"].lower(), "severe")
        
        # Convert stack trace to Java format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]
            
            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in Java stack trace string format
                    java_error["stack_trace"] = "\n".join(stack_trace)
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to Java stack trace format
                    java_error["stack_trace"] = self._convert_frames_to_java_stack(
                        standard_error["error_type"], 
                        standard_error["message"], 
                        stack_trace
                    )
                    # Also keep the structured version
                    java_error["stack_frames"] = stack_trace
        
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
    
    def _parse_java_stack_trace(self, stack_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parse a Java stack trace into structured frames.
        
        Args:
            stack_lines: Java stack trace lines
            
        Returns:
            Structured frames or None if parsing fails
        """
        frames = []
        
        # Java stack trace pattern: at package.Class.method(File.java:line)
        frame_pattern = r'\s*at\s+([a-zA-Z0-9_.]+)\.([a-zA-Z0-9_$]+)\.([a-zA-Z0-9_$]+)\(([^:]+):(\d+)\)'
        
        try:
            for line in stack_lines:
                match = re.search(frame_pattern, line)
                if match:
                    package = match.group(1)
                    class_name = match.group(2)
                    method = match.group(3)
                    file = match.group(4)
                    line_num = int(match.group(5))
                    
                    frames.append({
                        "package": package,
                        "class": class_name,
                        "function": method,
                        "file": file,
                        "line": line_num
                    })
                    
            return frames if frames else None
        except Exception as e:
            logger.debug(f"Failed to parse Java stack trace: {e}")
            return None
    
    def _convert_frames_to_java_stack(self, error_type: str, message: str, 
                                     frames: List[Dict[str, Any]]) -> str:
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
            "error_type": error_data.get("error_type", ""),
            "message": error_data.get("message", "")
        }
        
        # Add Go version if available
        if "go_version" in error_data:
            standard_error["language_version"] = error_data["go_version"]
        
        # Handle stack trace
        if "stack_trace" in error_data:
            # Go stack traces can be a string or a list
            if isinstance(error_data["stack_trace"], str):
                # Split into lines
                stack_lines = error_data["stack_trace"].split("\n")
                
                # Try to parse structured data from the stack trace
                parsed_frames = self._parse_go_stack_trace(stack_lines)
                
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
            # Map Go log levels to standard format
            level_map = {
                "debug": "debug",
                "info": "info",
                "warning": "warning",
                "warn": "warning",
                "error": "error",
                "fatal": "fatal",
                "panic": "critical"
            }
            standard_error["severity"] = level_map.get(error_data["level"].lower(), "error")
        
        # Add runtime if available
        if "runtime" in error_data:
            standard_error["runtime"] = error_data["runtime"]
            
            if "runtime_version" in error_data:
                standard_error["runtime_version"] = error_data["runtime_version"]
        
        # Add goroutine information if available
        if "goroutine_id" in error_data:
            if "additional_data" not in standard_error:
                standard_error["additional_data"] = {}
            standard_error["additional_data"]["goroutine_id"] = error_data["goroutine_id"]
        
        # Add handled flag if available
        if "handled" in error_data:
            standard_error["handled"] = error_data["handled"]
        
        # Add additional Go-specific data
        go_specific = {}
        for key, value in error_data.items():
            if key not in standard_error and key not in ["stack_trace", "request", "context"] and not key.startswith("_"):
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
            "error_type": standard_error.get("error_type", "error"),
            "message": standard_error.get("message", "")
        }
        
        # Convert severity to Go logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "debug",
                "info": "info",
                "warning": "warn",
                "error": "error",
                "critical": "panic",
                "fatal": "fatal"
            }
            go_error["level"] = level_map.get(standard_error["severity"].lower(), "error")
        
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
                        stack_trace
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
        goroutine_pattern = r'goroutine (\d+) \[([^\]]+)\]:'
        function_pattern = r'([^\(]+)\(.*\)'
        file_pattern = r'\s*([^:]+):(\d+)(?:\s+(\+0x[0-9a-f]+))?'
        
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
                            
                            # Extract package and function names
                            func_parts = current_function.split('.')
                            package = ".".join(func_parts[:-1]) if len(func_parts) > 1 else ""
                            func_name = func_parts[-1] if func_parts else current_function
                            
                            frames.append({
                                "package": package,
                                "function": func_name,
                                "file": file_path,
                                "line": line_num,
                                "goroutine_id": goroutine_id
                            })
                            
                            i += 2  # Skip the file line
                            continue
                
                i += 1
                
            return frames if frames else None
        except Exception as e:
            logger.debug(f"Failed to parse Go stack trace: {e}")
            return None
    
    def _convert_frames_to_go_stack(self, error_type: str, message: str, 
                                   frames: List[Dict[str, Any]]) -> str:
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
            "message": error_data.get("message", "")
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
                "fatal": "fatal"
            }
            standard_error["severity"] = level_map.get(error_data["level"].lower(), "error")
        
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
            if key not in standard_error and key not in ["backtrace", "request", "context"]:
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
            "message": standard_error.get("message", "")
        }
        
        # Convert severity to Ruby logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "debug",
                "info": "info",
                "warning": "warn",
                "error": "error",
                "critical": "fatal",
                "fatal": "fatal"
            }
            ruby_error["level"] = level_map.get(standard_error["severity"].lower(), "error")
        
        # Convert stack trace to Ruby format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]
            
            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in Ruby stack trace string format
                    ruby_error["backtrace"] = stack_trace
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to Ruby stack trace format
                    ruby_error["backtrace"] = self._convert_frames_to_ruby_backtrace(stack_trace)
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
        frame_pattern = r'([^:]+):(\d+)(?::in\s+`([^\']+)\')?'
        
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
                    if file_name.endswith('.rb'):
                        module_parts = file_name[:-3].split('_')
                        module_name = ''.join(part.title() for part in module_parts)
                    
                    frames.append({
                        "file": file_path,
                        "line": line_num,
                        "function": method_name,
                        "module": module_name
                    })
            
            return frames if frames else None
        except Exception as e:
            logger.debug(f"Failed to parse Ruby stack trace: {e}")
            return None
    
    def _convert_frames_to_ruby_backtrace(self, frames: List[Dict[str, Any]]) -> List[str]:
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
            "message": error_data.get("message", "")
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
                "fatal": "fatal"
            }
            standard_error["severity"] = level_map.get(error_data["level"].lower(), "error")
        
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
            if key not in standard_error and key not in ["backtrace", "request", "context", "thread", "is_panic"]:
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
            "message": standard_error.get("message", "")
        }
        
        # Convert severity to Rust logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "debug",
                "info": "info",
                "warning": "warn",
                "error": "error",
                "critical": "error",
                "fatal": "error"
            }
            rust_error["level"] = level_map.get(standard_error["severity"].lower(), "error")
        
        # Convert stack trace to Rust format
        if "stack_trace" in standard_error:
            stack_trace = standard_error["stack_trace"]
            
            if isinstance(stack_trace, list):
                if all(isinstance(frame, str) for frame in stack_trace):
                    # Already in Rust stack trace string format
                    rust_error["backtrace"] = stack_trace
                elif all(isinstance(frame, dict) for frame in stack_trace):
                    # Convert structured frames to Rust backtrace format
                    rust_error["backtrace"] = self._convert_frames_to_rust_backtrace(stack_trace)
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
        line_pattern = r'^\s*(\d+):\s+(.*?)$'
        
        # File location pattern
        file_pattern = r'^\s*at\s+([^:]+):(\d+)(?::(\d+))?'
        
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
                    addr_match = re.match(r'(0x[0-9a-f]+)\s+-\s+(.*)', frame_info)
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
                        "module": module
                    }
                else:
                    # Try to match a file location for the current frame
                    file_match = re.match(file_pattern, line)
                    if file_match and current_frame is not None:
                        file_path = file_match.group(1)
                        line_num = int(file_match.group(2))
                        column = int(file_match.group(3)) if file_match.group(3) else None
                        
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
    
    def _convert_frames_to_rust_backtrace(self, frames: List[Dict[str, Any]]) -> List[str]:
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
            "message": error_data.get("message", "")
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
            standard_error["additional_data"]["inner_exception"] = error_data["inner_exception"]
        
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
                "fatal": "fatal"
            }
            standard_error["severity"] = level_map.get(error_data["level"].lower(), "error")
        
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
            if key not in standard_error and key not in ["stack_trace", "request", "context", "inner_exception"]:
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
            "message": standard_error.get("message", "")
        }
        
        # Convert severity to C# logging level
        if "severity" in standard_error:
            level_map = {
                "debug": "Debug",
                "info": "Information",
                "warning": "Warning",
                "error": "Error",
                "critical": "Critical",
                "fatal": "Fatal"
            }
            csharp_error["level"] = level_map.get(standard_error["severity"].lower(), "Error")
        
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
                        stack_trace
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
        method_pattern = r'\s*at\s+([^(]+)\(([^)]*)\)'
        # Then separately extract file location
        file_pattern = r'\s+in\s+(.+):line\s+(\d+)'
        
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
                parts = full_method.split('.')
                
                if len(parts) >= 3:
                    # Standard case: Namespace.Class.Method
                    namespace = '.'.join(parts[:-2])
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
                
                # Handle special cases like lambda expressions
                if '<' in method and '>' in method:
                    # This is likely a lambda or anonymous method
                    pass  # We'll keep the method name with the angle brackets
                
                frames.append({
                    "namespace": namespace,
                    "class": class_name,
                    "function": method,
                    "parameters": parameters,
                    "file": file_path,
                    "line": line_num
                })
            
            return frames if frames else None
        except Exception as e:
            logger.debug(f"Failed to parse C# stack trace: {e}")
            return None
    
    def _convert_frames_to_csharp_stack(self, error_type: str, message: str, frames: List[Dict[str, Any]]) -> str:
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
                stack_lines.append(f"   at {full_method}({parameters}) in {file}:line {line_num}")
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
        elif "dotnet_version" in error_data or "exception_type" in error_data and error_data["exception_type"].startswith("System."):
            return "csharp"
            
        # Try to detect from stack trace format
        if "stack_trace" in error_data and isinstance(error_data["stack_trace"], str):
            stack = error_data["stack_trace"]
            if "goroutine " in stack and ".go:" in stack:
                return "go"
            elif "at " in stack and ".java:" in stack:
                return "java"
            elif "at " in stack and (re.search(r'\.cs:line \d+', stack) or "System." in stack):
                return "csharp"
                
        if "stack" in error_data and isinstance(error_data["stack"], str):
            stack = error_data["stack"]
            if "at " in stack and ").js" in stack:
                return "javascript"
            elif "at " in stack and ".java:" in stack:
                return "java"
            elif "goroutine " in stack and ".go:" in stack:
                return "go"
            elif "at " in stack and (re.search(r'\.cs:line \d+', stack) or "System." in stack):
                return "csharp"
        
        if "backtrace" in error_data:
            # Check for Ruby backtrace format
            if isinstance(error_data["backtrace"], list) and any(
                isinstance(line, str) and re.search(r'\.rb:\d+:in', line) 
                for line in error_data["backtrace"][:10] if isinstance(line, str)
            ):
                return "ruby"
            elif isinstance(error_data["backtrace"], str) and ".rb:" in error_data["backtrace"]:
                return "ruby"
            # Check for Rust backtrace format
            elif isinstance(error_data["backtrace"], str) and (".rs:" in error_data["backtrace"] or 
                                                              any(pattern in error_data["backtrace"] for pattern in 
                                                                  ["panicked at", "thread", "rust_panic"])):
                return "rust"
            elif isinstance(error_data["backtrace"], list) and any(
                isinstance(line, str) and (re.search(r'\.rs:\d+', line) or "::{{closure}}" in line)
                for line in error_data["backtrace"][:10] if isinstance(line, str)
            ):
                return "rust"
                
        return "unknown"


def convert_to_standard_format(error_data: Dict[str, Any], language: Optional[str] = None) -> Dict[str, Any]:
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


def convert_from_standard_format(standard_error: Dict[str, Any], target_language: str) -> Dict[str, Any]:
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
        "traceback": traceback.format_exception(type(exception), exception, exception.__traceback__)
    }
    
    return convert_to_standard_format(error_data, "python")


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
            "  File \"app.py\", line 42, in get_user",
            "    user_id = data['user_id']",
            "KeyError: 'user_id'"
        ],
        "level": "ERROR",
        "python_version": "3.9.7",
        "framework": "FastAPI",
        "framework_version": "0.68.0"
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
        "runtime_version": "14.17.5"
    }
    
    # Convert to standard format
    js_adapter = JavaScriptErrorAdapter()
    standard_js = js_adapter.to_standard_format(js_error)
    logger.info(f"Standard JavaScript format: {json.dumps(standard_js, indent=2)}")
    
    # Convert JavaScript to Python format
    js_to_python = convert_from_standard_format(standard_js, "python")
    logger.info(f"JavaScript converted to Python format: {json.dumps(js_to_python, indent=2)}")
    
    # Example Ruby error data
    ruby_error = {
        "timestamp": "2023-08-15T12:34:56",
        "exception_class": "NoMethodError",
        "message": "undefined method `[]' for nil:NilClass",
        "backtrace": [
            "app/controllers/users_controller.rb:25:in `show'",
            "app/controllers/application_controller.rb:10:in `authorize_user!'",
            "/gems/actionpack-6.1.0/lib/action_controller/metal/basic_implicit_render.rb:6:in `send_action'"
        ],
        "level": "error",
        "ruby_version": "3.0.2",
        "framework": "Rails",
        "framework_version": "6.1.0"
    }
    
    # Convert to standard format
    ruby_adapter = RubyErrorAdapter()
    standard_ruby = ruby_adapter.to_standard_format(ruby_error)
    logger.info(f"Standard Ruby format: {json.dumps(standard_ruby, indent=2)}")
    
    # Convert Ruby to Python format
    ruby_to_python = convert_from_standard_format(standard_ruby, "python")
    logger.info(f"Ruby converted to Python format: {json.dumps(ruby_to_python, indent=2)}")