# Homeostasis Monitoring Module

This module is responsible for capturing, formatting, and extracting logs, exceptions, and performance metrics for the Homeostasis framework's self-healing capabilities.

## Features

- **Enhanced Error Logging**: Detailed error information including stack traces, local variables, and system metadata
- **Rich Context**: Capture and preserve call location, environment variables, and request information
- **Advanced Filtering**: Filter logs by level, timestamp, service name, exception type, and custom tags
- **Comprehensive Summaries**: Generate statistics and insights about errors for easier troubleshooting
- **Security-Aware**: Automatically redacts sensitive information like passwords, secrets, and tokens
- **Performance metric tracking** (future enhancement)

## Usage

```python
from modules.monitoring.logger import MonitoringLogger

# Initialize logger
logger = MonitoringLogger(
    service_name="my_service",
    log_level="INFO",
    include_system_info=True,
    enable_console_output=True
)

# Log different levels
logger.info("Information message", include_call_location=True)
logger.warning("Warning message", some_context="value")
logger.error("Error occurred", tags=["database", "connection"])

# Log exceptions with rich context
try:
    # Some code that might raise an exception
    raise ValueError("Invalid input")
except Exception as e:
    logger.exception(e, include_locals=True, operation="data_processing")
```

## FastAPI Integration

```python
from fastapi import FastAPI
from modules.monitoring.middleware import add_logging_middleware

app = FastAPI()

# Add logging middleware
add_logging_middleware(
    app, 
    service_name="api_service",
    log_level="INFO", 
    exclude_paths=["/health", "/metrics"],
    sensitive_headers=["x-custom-auth"]
)
```

## Working with Logs

```python
from modules.monitoring.extractor import get_latest_errors, get_error_summary

# Get recent errors with filtering
errors = get_latest_errors(
    limit=10,
    levels=["ERROR", "CRITICAL"],
    service_name="api_service",
    exception_types=["KeyError", "ValueError"],
    tags=["database"],
    hours_back=24
)

# Generate comprehensive error summary
summary = get_error_summary(days_back=7, service_name="api_service")
```