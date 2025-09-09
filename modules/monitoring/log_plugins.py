"""
Logging service plugins for Homeostasis monitoring.

This module provides integration with various logging frameworks and services,
allowing Homeostasis to capture logs from diverse sources and formats.
"""

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import the base logger
from .logger import LOG_FILE, MonitoringLogger

try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    from loguru import logger as loguru_logger

    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False

try:
    import opentelemetry.trace as trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


class LoggingPlugin(ABC):
    """Base class for logging service plugins."""

    def __init__(self, service_name: str, **kwargs):
        """
        Initialize the logging plugin.

        Args:
            service_name: Name of the service being monitored
            **kwargs: Additional configuration options
        """
        self.service_name = service_name
        self.config = kwargs
        self.enabled = True

        # Initialize the underlying logger
        self._initialize_logger()

    @abstractmethod
    def _initialize_logger(self) -> None:
        """Initialize the logger implementation."""
        pass

    @abstractmethod
    def log(self, level: str, message: str, **kwargs) -> Dict[str, Any]:
        """
        Log a message with the plugin.

        Args:
            level: Log level
            message: Log message
            **kwargs: Additional context data

        Returns:
            The log record as a dictionary
        """
        pass

    @abstractmethod
    def exception(self, e: Exception, **kwargs) -> Dict[str, Any]:
        """
        Log an exception with the plugin.

        Args:
            e: The exception to log
            **kwargs: Additional context data

        Returns:
            The log record as a dictionary
        """
        pass

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

    def enable(self) -> None:
        """Enable the plugin."""
        self.enabled = True

    def disable(self) -> None:
        """Disable the plugin."""
        self.enabled = False


class StandardLoggerPlugin(LoggingPlugin):
    """Plugin for the built-in MonitoringLogger."""

    def _initialize_logger(self) -> None:
        """Initialize the standard logger."""
        self.logger = MonitoringLogger(
            service_name=self.service_name,
            log_level=self.config.get("log_level", "INFO"),
            log_file=self.config.get("log_file"),
            include_system_info=self.config.get("include_system_info", True),
            enable_console_output=self.config.get("enable_console_output", True),
        )

    def log(self, level: str, message: str, **kwargs) -> Dict[str, Any]:
        """
        Log a message with the standard logger.

        Args:
            level: Log level
            message: Log message
            **kwargs: Additional context data

        Returns:
            The log record as a dictionary
        """
        if not self.enabled:
            return {}

        return self.logger.log(level, message, **kwargs)

    def exception(self, e: Exception, **kwargs) -> Dict[str, Any]:
        """
        Log an exception with the standard logger.

        Args:
            e: The exception to log
            **kwargs: Additional context data

        Returns:
            The log record as a dictionary
        """
        if not self.enabled:
            return {}

        return self.logger.exception(
            e,
            include_locals=self.config.get("include_locals", True),
            include_globals=self.config.get("include_globals", False),
            **kwargs,
        )


class StructlogPlugin(LoggingPlugin):
    """Plugin for the structlog library."""

    def _initialize_logger(self) -> None:
        """Initialize the structlog logger."""
        if not STRUCTLOG_AVAILABLE:
            raise ImportError(
                "structlog is not installed. Install it with 'pip install structlog'."
            )

        # Configure structlog to work with standard logging
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Create the structlog logger
        self.logger = structlog.get_logger(self.service_name)

        # Also initialize a standard logger for compatibility
        self.std_logger = MonitoringLogger(
            service_name=self.service_name,
            log_level=self.config.get("log_level", "INFO"),
            log_file=self.config.get("log_file"),
            include_system_info=self.config.get("include_system_info", True),
            enable_console_output=False,  # We'll use structlog for console output
        )

    def log(self, level: str, message: str, **kwargs) -> Dict[str, Any]:
        """
        Log a message with structlog.

        Args:
            level: Log level
            message: Log message
            **kwargs: Additional context data

        Returns:
            The log record as a dictionary
        """
        if not self.enabled:
            return {}

        # Log with standard logger first for compatibility
        std_record = self.std_logger.log(level, message, **kwargs)

        # Get the structlog method for this level
        log_method = getattr(self.logger, level.lower())

        # Create a structured event with the additional context
        log_method(message, **kwargs)

        return std_record

    def exception(self, e: Exception, **kwargs) -> Dict[str, Any]:
        """
        Log an exception with structlog.

        Args:
            e: The exception to log
            **kwargs: Additional context data

        Returns:
            The log record as a dictionary
        """
        if not self.enabled:
            return {}

        # Log with standard logger for compatibility and detailed error capture
        std_record = self.std_logger.exception(
            e,
            include_locals=self.config.get("include_locals", True),
            include_globals=self.config.get("include_globals", False),
            **kwargs,
        )

        # Get exception info
        exc_info = sys.exc_info()

        # Log with structlog including the exception info
        self.logger.exception(str(e), exc_info=exc_info, **kwargs)

        return std_record


class LoguruPlugin(LoggingPlugin):
    """Plugin for the loguru library."""

    def _initialize_logger(self) -> None:
        """Initialize the loguru logger."""
        if not LOGURU_AVAILABLE:
            raise ImportError(
                "loguru is not installed. Install it with 'pip install loguru'."
            )

        # Configure loguru
        log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        log_file = self.config.get("log_file") or LOG_FILE

        # Make sure the log file is a Path object
        if isinstance(log_file, str):
            log_file = Path(log_file)

        # Clear existing handlers
        loguru_logger.remove()

        # Add handlers
        if self.config.get("enable_console_output", True):
            loguru_logger.add(
                sys.stdout,
                format=log_format,
                level=self.config.get("log_level", "INFO"),
            )

        # Add file handler with rotation
        if self.config.get("rotation_size", None):
            loguru_logger.add(
                str(log_file),
                format=log_format,
                level=self.config.get("log_level", "INFO"),
                rotation=self.config.get("rotation_size", "20 MB"),
                retention=self.config.get("retention", "1 week"),
                compression=self.config.get("compression", "zip"),
            )
        else:
            loguru_logger.add(
                str(log_file),
                format=log_format,
                level=self.config.get("log_level", "INFO"),
            )

        # Store the logger
        self.logger = loguru_logger.bind(service=self.service_name)

        # Also initialize a standard logger for compatibility
        self.std_logger = MonitoringLogger(
            service_name=self.service_name,
            log_level=self.config.get("log_level", "INFO"),
            log_file=self.config.get("log_file"),
            include_system_info=self.config.get("include_system_info", True),
            enable_console_output=False,  # We'll use loguru for console output
        )

    def log(self, level: str, message: str, **kwargs) -> Dict[str, Any]:
        """
        Log a message with loguru.

        Args:
            level: Log level
            message: Log message
            **kwargs: Additional context data

        Returns:
            The log record as a dictionary
        """
        if not self.enabled:
            return {}

        # Log with standard logger first for compatibility
        std_record = self.std_logger.log(level, message, **kwargs)

        # Get the loguru method for this level
        log_method = getattr(self.logger, level.lower())

        # Create context string if there are additional kwargs
        if kwargs:
            # Format context as JSON
            context_str = json.dumps(kwargs, default=str)
            log_method(f"{message} | Context: {context_str}")
        else:
            log_method(message)

        return std_record

    def exception(self, e: Exception, **kwargs) -> Dict[str, Any]:
        """
        Log an exception with loguru.

        Args:
            e: The exception to log
            **kwargs: Additional context data

        Returns:
            The log record as a dictionary
        """
        if not self.enabled:
            return {}

        # Log with standard logger for compatibility and detailed error capture
        std_record = self.std_logger.exception(
            e,
            include_locals=self.config.get("include_locals", True),
            include_globals=self.config.get("include_globals", False),
            **kwargs,
        )

        # Format context as JSON if there are additional kwargs
        if kwargs:
            context_str = json.dumps(kwargs, default=str)
            self.logger.exception(f"{str(e)} | Context: {context_str}")
        else:
            self.logger.exception(str(e))

        return std_record


class OpenTelemetryPlugin(LoggingPlugin):
    """Plugin for OpenTelemetry integration."""

    def _initialize_logger(self) -> None:
        """Initialize the OpenTelemetry integration."""
        if not OPENTELEMETRY_AVAILABLE:
            raise ImportError(
                "OpenTelemetry is not installed. "
                "Install it with 'pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp'"
            )

        # Configure resource
        resource = Resource(
            attributes={
                "service.name": self.service_name,
                "service.version": self.config.get("service_version", "0.1.0"),
            }
        )

        # Set up tracer provider
        trace.set_tracer_provider(TracerProvider(resource=resource))
        self.tracer = trace.get_tracer(self.service_name)

        # Set up exporter if configured
        exporter_endpoint = self.config.get("exporter_endpoint")
        if exporter_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import \
                    OTLPSpanExporter

                otlp_exporter = OTLPSpanExporter(endpoint=exporter_endpoint)
                span_processor = BatchSpanProcessor(otlp_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)
            except ImportError:
                print(
                    "OTLP exporter not available. Install it with 'pip install opentelemetry-exporter-otlp'"
                )

        # Also initialize a standard logger for compatibility
        self.std_logger = MonitoringLogger(
            service_name=self.service_name,
            log_level=self.config.get("log_level", "INFO"),
            log_file=self.config.get("log_file"),
            include_system_info=self.config.get("include_system_info", True),
            enable_console_output=self.config.get("enable_console_output", True),
        )

    def log(self, level: str, message: str, **kwargs) -> Dict[str, Any]:
        """
        Log a message with OpenTelemetry.

        Args:
            level: Log level
            message: Log message
            **kwargs: Additional context data

        Returns:
            The log record as a dictionary
        """
        if not self.enabled:
            return {}

        # Log with standard logger first for compatibility
        std_record = self.std_logger.log(level, message, **kwargs)

        # Create a new span for the log entry if high severity
        if level in ["ERROR", "CRITICAL"]:
            with self.tracer.start_as_current_span(f"log_{level.lower()}") as span:
                span.set_attribute("log.level", level)
                span.set_attribute("log.message", message)

                # Add additional attributes
                for key, value in kwargs.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"log.context.{key}", str(value))

        return std_record

    def exception(self, e: Exception, **kwargs) -> Dict[str, Any]:
        """
        Log an exception with OpenTelemetry.

        Args:
            e: The exception to log
            **kwargs: Additional context data

        Returns:
            The log record as a dictionary
        """
        if not self.enabled:
            return {}

        # Log with standard logger for compatibility and detailed error capture
        std_record = self.std_logger.exception(
            e,
            include_locals=self.config.get("include_locals", True),
            include_globals=self.config.get("include_globals", False),
            **kwargs,
        )

        # Record the exception in the current span or create a new one
        with self.tracer.start_as_current_span(
            f"exception_{e.__class__.__name__}"
        ) as span:
            # Record the exception details
            span.record_exception(e)
            span.set_attribute("exception.type", e.__class__.__name__)
            span.set_attribute("exception.message", str(e))

            # Add additional attributes
            for key, value in kwargs.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"exception.context.{key}", str(value))

            # Set status to error
            span.set_status(trace.StatusCode.ERROR)

        return std_record


class LoggingPluginManager:
    """Manager for logging service plugins."""

    def __init__(self, default_service_name: str = "homeostasis"):
        """
        Initialize the logging plugin manager.

        Args:
            default_service_name: Default service name to use
        """
        self.default_service_name = default_service_name
        self.plugins: Dict[str, LoggingPlugin] = {}

        # Initialize with the standard logger by default
        self.add_plugin("standard", StandardLoggerPlugin(default_service_name))

    def add_plugin(self, name: str, plugin: LoggingPlugin) -> None:
        """
        Add a logging plugin.

        Args:
            name: Name of the plugin
            plugin: Plugin instance
        """
        self.plugins[name] = plugin

    def remove_plugin(self, name: str) -> None:
        """
        Remove a logging plugin.

        Args:
            name: Name of the plugin
        """
        if name in self.plugins:
            del self.plugins[name]

    def enable_plugin(self, name: str) -> None:
        """
        Enable a logging plugin.

        Args:
            name: Name of the plugin
        """
        if name in self.plugins:
            self.plugins[name].enable()

    def disable_plugin(self, name: str) -> None:
        """
        Disable a logging plugin.

        Args:
            name: Name of the plugin
        """
        if name in self.plugins:
            self.plugins[name].disable()

    def log(self, level: str, message: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Log a message with all enabled plugins.

        Args:
            level: Log level
            message: Log message
            **kwargs: Additional context data

        Returns:
            List of log records from each plugin
        """
        return [
            plugin.log(level, message, **kwargs)
            for plugin in self.plugins.values()
            if plugin.enabled
        ]

    def exception(self, e: Exception, **kwargs) -> List[Dict[str, Any]]:
        """
        Log an exception with all enabled plugins.

        Args:
            e: The exception to log
            **kwargs: Additional context data

        Returns:
            List of log records from each plugin
        """
        return [
            plugin.exception(e, **kwargs)
            for plugin in self.plugins.values()
            if plugin.enabled
        ]

    def info(self, message: str, **kwargs) -> List[Dict[str, Any]]:
        """Log an info message with all enabled plugins."""
        return self.log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs) -> List[Dict[str, Any]]:
        """Log a warning message with all enabled plugins."""
        return self.log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs) -> List[Dict[str, Any]]:
        """Log an error message with all enabled plugins."""
        return self.log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs) -> List[Dict[str, Any]]:
        """Log a critical message with all enabled plugins."""
        return self.log("CRITICAL", message, **kwargs)

    def get_plugin(self, name: str) -> Optional[LoggingPlugin]:
        """
        Get a plugin by name.

        Args:
            name: Name of the plugin

        Returns:
            The plugin instance, or None if not found
        """
        return self.plugins.get(name)

    def get_enabled_plugins(self) -> Dict[str, LoggingPlugin]:
        """
        Get all enabled plugins.

        Returns:
            Dictionary of enabled plugins
        """
        return {name: plugin for name, plugin in self.plugins.items() if plugin.enabled}


def create_default_plugin_manager(
    service_name: str = "homeostasis",
) -> LoggingPluginManager:
    """
    Create a default plugin manager with all available plugins.

    Args:
        service_name: Service name to use

    Returns:
        Configured plugin manager
    """
    manager = LoggingPluginManager(service_name)

    # Add Structlog plugin if available
    if STRUCTLOG_AVAILABLE:
        try:
            manager.add_plugin(
                "structlog",
                StructlogPlugin(
                    service_name, log_level="INFO", include_system_info=True
                ),
            )
        except Exception as e:
            print(f"Failed to initialize Structlog plugin: {str(e)}")

    # Add Loguru plugin if available
    if LOGURU_AVAILABLE:
        try:
            manager.add_plugin(
                "loguru",
                LoguruPlugin(
                    service_name,
                    log_level="INFO",
                    rotation_size="20 MB",
                    retention="1 week",
                    compression="zip",
                ),
            )
        except Exception as e:
            print(f"Failed to initialize Loguru plugin: {str(e)}")

    # Add OpenTelemetry plugin if available
    if OPENTELEMETRY_AVAILABLE:
        try:
            manager.add_plugin(
                "opentelemetry",
                OpenTelemetryPlugin(service_name, service_version="0.1.0"),
            )
        except Exception as e:
            print(f"Failed to initialize OpenTelemetry plugin: {str(e)}")

    return manager


# Global plugin manager instance
_plugin_manager = None


def get_plugin_manager(service_name: str = "homeostasis") -> LoggingPluginManager:
    """
    Get the global plugin manager instance, creating it if necessary.

    Args:
        service_name: Service name to use

    Returns:
        Global plugin manager instance
    """
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = create_default_plugin_manager(service_name)
    return _plugin_manager


# Convenience logging functions
def log(level: str, message: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Log a message with all enabled plugins.

    Args:
        level: Log level
        message: Log message
        **kwargs: Additional context data

    Returns:
        List of log records from each plugin
    """
    return get_plugin_manager().log(level, message, **kwargs)


def info(message: str, **kwargs) -> List[Dict[str, Any]]:
    """Log an info message with all enabled plugins."""
    return get_plugin_manager().info(message, **kwargs)


def warning(message: str, **kwargs) -> List[Dict[str, Any]]:
    """Log a warning message with all enabled plugins."""
    return get_plugin_manager().warning(message, **kwargs)


def error(message: str, **kwargs) -> List[Dict[str, Any]]:
    """Log an error message with all enabled plugins."""
    return get_plugin_manager().error(message, **kwargs)


def critical(message: str, **kwargs) -> List[Dict[str, Any]]:
    """Log a critical message with all enabled plugins."""
    return get_plugin_manager().critical(message, **kwargs)


def exception(e: Exception, **kwargs) -> List[Dict[str, Any]]:
    """Log an exception with all enabled plugins."""
    return get_plugin_manager().exception(e, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("Logging Plugin Demo")
    print("==================")

    # Create a plugin manager
    manager = create_default_plugin_manager("demo_service")

    # List available plugins
    plugins = manager.plugins
    print(f"Available plugins: {', '.join(plugins.keys())}")

    # Log a test message
    print("\nLogging test message...")
    manager.info("This is a test message", context="example")

    # Log an exception
    print("\nLogging test exception...")
    try:
        # Generate an exception
        result = 1 / 0
    except Exception as e:
        manager.exception(e, context="exception_example")

    print("\nCheck the log file for details.")
    print(f"Log file: {LOG_FILE}")
