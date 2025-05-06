"""
Monitoring module for capturing and formatting logs.
"""
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure log file
LOG_FILE = LOGS_DIR / "homeostasis.log"


class MonitoringLogger:
    """
    Logger class for monitoring and capturing errors/exceptions.
    """

    def __init__(self, service_name: str, log_level: str = "INFO"):
        """
        Initialize the logger.

        Args:
            service_name: Name of the service being monitored
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.service_name = service_name
        self.log_level = getattr(logging, log_level)
        
        # Configure logging
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(self.log_level)
        
        # Remove existing handlers if any
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(self.log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def log(self, level: str, message: str, **kwargs) -> Dict[str, Any]:
        """
        Log a message with additional context as JSON.

        Args:
            level: Log level
            message: Log message
            **kwargs: Additional context data
        
        Returns:
            The log record as a dictionary
        """
        log_record = {
            "timestamp": datetime.now().isoformat(),
            "service": self.service_name,
            "level": level,
            "message": message,
            **kwargs
        }
        
        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(log_record))
        
        return log_record

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

    def exception(self, e: Exception, **kwargs) -> Dict[str, Any]:
        """
        Log an exception with full traceback.

        Args:
            e: The exception to log
            **kwargs: Additional context data
        
        Returns:
            The log record as a dictionary
        """
        exc_type, exc_value, exc_traceback = sys.exc_info()
        stack_trace = traceback.format_exception(exc_type, exc_value, exc_traceback)
        
        log_record = self.log(
            "ERROR",
            str(e),
            exception_type=e.__class__.__name__,
            traceback=stack_trace,
            **kwargs
        )
        
        return log_record


# JSON schema for error logs
ERROR_SCHEMA = {
    "type": "object",
    "required": [
        "timestamp",
        "service",
        "level",
        "message"
    ],
    "properties": {
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 timestamp"
        },
        "service": {
            "type": "string",
            "description": "Name of the service that generated the error"
        },
        "level": {
            "type": "string",
            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "description": "Log level"
        },
        "message": {
            "type": "string",
            "description": "Error message"
        },
        "exception_type": {
            "type": "string",
            "description": "Type of exception"
        },
        "traceback": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Stack trace for the error"
        },
        "context": {
            "type": "object",
            "description": "Additional context about the error"
        }
    }
}