"""
Monitoring module for capturing and formatting logs.
Enhanced to include detailed stack traces and system environment metadata.
"""

import json
import logging
import os
import platform
import socket
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure log file
LOG_FILE = LOGS_DIR / "homeostasis.log"

# Get basic system information
SYSTEM_INFO: Dict[str, Any] = {
    "hostname": socket.gethostname(),
    "os_name": os.name,
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "processor": platform.processor() or "unknown",
    "architecture": platform.architecture()[0],
}


class MonitoringLogger:
    """
    Logger class for monitoring and capturing errors/exceptions.
    """

    def __init__(
        self,
        service_name: str,
        log_level: str = "INFO",
        log_file: Optional[Path] = None,
        include_system_info: bool = True,
        enable_console_output: bool = True,
    ):
        """
        Initialize the logger with enhanced features.

        Args:
            service_name: Name of the service being monitored
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Custom log file path. If None, defaults to LOG_FILE
            include_system_info: Whether to include system information in logs
            enable_console_output: Whether to enable console output
        """
        self.service_name = service_name
        self.log_level = getattr(logging, log_level)
        self.include_system_info = include_system_info
        self.log_file = log_file or LOG_FILE

        # Store system info globally for this logger instance
        self.system_info: Dict[str, Any] = SYSTEM_INFO.copy()
        self.system_info["service_name"] = service_name

        # Add environment variables (excluding sensitive ones)
        env_vars: Dict[str, str] = {}
        # Extended list of sensitive environment variable patterns
        excluded_env_vars = [
            "API_KEY",
            "SECRET",
            "PASSWORD",
            "TOKEN",
            "CREDENTIAL",
            "APIKEY",
            "AUTH",
            "PRIVATE",
            "KEY",
            "CERT",
            "SALT",
            "PASSPHRASE",
            "ACCESS_KEY",
            "PWD",
            "LOGIN",
            "STRIPE",
        ]
        for key, value in os.environ.items():
            # Skip sensitive environment variables
            if any(excluded in key.upper() for excluded in excluded_env_vars):
                # Mask the value completely
                continue
            env_vars[key] = value

        # Remove sensitive values even if not caught by the key filter
        self.system_info["environment_variables"] = env_vars

        # Configure logging with homeostasis namespace
        self.logger = logging.getLogger(f"homeostasis.{service_name}")
        self.logger.setLevel(self.log_level)

        # Remove existing handlers if any
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Console handler (optional)
        if enable_console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def log(
        self, level: str, message: str, include_call_location: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """
        Log a message with additional context as JSON.

        Args:
            level: Log level
            message: Log message
            include_call_location: Whether to include the file, function, and line number where the log was called
            **kwargs: Additional context data

        Returns:
            The log record as a dictionary
        """
        # Create base log record
        log_record: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "service": self.service_name,
            "level": level,
            "message": message,
        }

        # Include system info if enabled
        if self.include_system_info:
            log_record["system_info"] = self.system_info

        # Include call location if requested
        if include_call_location:
            frame = sys._getframe(2)  # Get the frame of the caller of this method
            log_record["call_location"] = {
                "file": frame.f_code.co_filename,
                "function": frame.f_code.co_name,
                "line_number": frame.f_lineno,
            }

        # Add the rest of the kwargs
        log_record.update(kwargs)

        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(log_record))

        return log_record

    def debug(self, message: str, **kwargs) -> Dict[str, Any]:
        """Log a debug message."""
        return self.log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs) -> Dict[str, Any]:
        """Log an info message."""
        return self.log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs) -> Dict[str, Any]:
        """Log a warning message."""
        return self.log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs) -> Dict[str, Any]:
        """Log an error message."""
        return self.log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs) -> Dict[str, Any]:
        """Log a critical message."""
        return self.log("CRITICAL", message, **kwargs)

    def exception(
        self,
        e: Exception,
        include_locals: bool = True,
        include_globals: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Log an exception with enhanced traceback and context information.

        Args:
            e: The exception to log
            include_locals: Whether to include local variables from the exception frame
            include_globals: Whether to include global variables from the exception frame
            **kwargs: Additional context data

        Returns:
            The log record as a dictionary
        """
        exc_type, exc_value, exc_traceback = sys.exc_info()

        # Format the basic traceback
        stack_trace = traceback.format_exception(exc_type, exc_value, exc_traceback)

        # Capture detailed traceback information
        tb_frames = []
        tb = exc_traceback
        while tb:
            frame = tb.tb_frame
            filename = frame.f_code.co_filename
            line_number = tb.tb_lineno
            function_name = frame.f_code.co_name

            frame_info = {
                "file": filename,
                "line": line_number,
                "function": function_name,
            }

            # Add local variables if requested (excluding large objects and excluding potentially sensitive data)
            if include_locals:
                safe_locals = {}
                for key, value in frame.f_locals.items():
                    # Skip large objects and potentially sensitive data
                    if key.startswith("__") or any(
                        sensitive in key.lower()
                        for sensitive in [
                            "password",
                            "secret",
                            "token",
                            "key",
                            "credential",
                        ]
                    ):
                        continue

                    try:
                        # Convert to string and limit size
                        str_value = str(value)
                        if len(str_value) > 1000:  # Limit very large values
                            str_value = str_value[:1000] + "... [truncated]"
                        safe_locals[key] = str_value
                    except Exception:
                        safe_locals[key] = "<unable to convert to string>"

                frame_info["locals"] = safe_locals

            # Add global variables if requested (with similar restrictions)
            if include_globals:
                safe_globals = {}
                for key, value in frame.f_globals.items():
                    # Skip large objects, modules, functions, and potentially sensitive data
                    if (
                        key.startswith("__")
                        or any(
                            sensitive in key.lower()
                            for sensitive in [
                                "password",
                                "secret",
                                "token",
                                "key",
                                "credential",
                            ]
                        )
                        or callable(value)  # Skip functions
                        or "module" in str(type(value)).lower()
                    ):  # Skip modules
                        continue

                    try:
                        # Convert to string and limit size
                        str_value = str(value)
                        if len(str_value) > 1000:  # Limit very large values
                            str_value = str_value[:1000] + "... [truncated]"
                        safe_globals[key] = str_value
                    except Exception:
                        safe_globals[key] = "<unable to convert to string>"

                frame_info["globals"] = safe_globals

            tb_frames.append(frame_info)
            tb = tb.tb_next

        # Create enhanced error data
        error_data = {
            "exception_type": e.__class__.__name__,
            "message": str(e),
            "traceback": stack_trace,
            "detailed_frames": tb_frames,
            "error_id": f"{e.__class__.__name__}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{id(e)}",
            "occured_at": datetime.now().isoformat(),
        }

        # Log the exception
        # Get message from kwargs or use the exception message
        message = kwargs.get("message", str(e))
        # Remove 'message' from kwargs if it exists to avoid conflict
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "message"}

        log_record = self.log(
            "ERROR",
            message,
            include_call_location=True,
            error_details=error_data,
            **filtered_kwargs,
        )

        return log_record


# JSON schema for enhanced error logs
ERROR_SCHEMA = {
    "type": "object",
    "required": ["timestamp", "service", "level", "message"],
    "properties": {
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 timestamp",
        },
        "service": {
            "type": "string",
            "description": "Name of the service that generated the error",
        },
        "level": {
            "type": "string",
            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "description": "Log level",
        },
        "message": {"type": "string", "description": "Error message"},
        "system_info": {
            "type": "object",
            "description": "System information",
            "properties": {
                "hostname": {"type": "string", "description": "Host name"},
                "os_name": {"type": "string", "description": "Operating system name"},
                "platform": {"type": "string", "description": "Platform information"},
                "python_version": {"type": "string", "description": "Python version"},
                "processor": {"type": "string", "description": "Processor information"},
                "architecture": {
                    "type": "string",
                    "description": "System architecture",
                },
                "service_name": {
                    "type": "string",
                    "description": "Name of the service",
                },
                "environment_variables": {
                    "type": "object",
                    "description": "Environment variables (excluding sensitive ones)",
                },
            },
        },
        "call_location": {
            "type": "object",
            "description": "Location where the log was called",
            "properties": {
                "file": {"type": "string", "description": "File path"},
                "function": {"type": "string", "description": "Function name"},
                "line_number": {"type": "integer", "description": "Line number"},
            },
        },
        "error_details": {
            "type": "object",
            "description": "Detailed information about the error",
            "properties": {
                "exception_type": {
                    "type": "string",
                    "description": "Type of exception",
                },
                "message": {"type": "string", "description": "Error message"},
                "traceback": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Stack trace for the error",
                },
                "detailed_frames": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "description": "File path"},
                            "line": {"type": "integer", "description": "Line number"},
                            "function": {
                                "type": "string",
                                "description": "Function name",
                            },
                            "locals": {
                                "type": "object",
                                "description": "Local variables in the frame",
                            },
                            "globals": {
                                "type": "object",
                                "description": "Global variables in the frame",
                            },
                        },
                    },
                    "description": "Detailed information about each frame in the traceback",
                },
                "error_id": {
                    "type": "string",
                    "description": "Unique identifier for the error",
                },
                "occured_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Timestamp when the error occurred",
                },
            },
        },
        "context": {
            "type": "object",
            "description": "Additional context about the error",
        },
    },
}
