"""
Tests for the monitoring module.
"""
import os
import sys
import json
import logging
import pytest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.monitoring.logger import MonitoringLogger
from modules.monitoring.extractor import get_latest_errors, get_error_summary
from modules.monitoring.middleware import add_logging_middleware


def test_monitoring_logger_initialization():
    """Test initializing the MonitoringLogger."""
    logger = MonitoringLogger("test_module")
    assert logger.logger.name == "homeostasis.test_module"
    assert logger.logger.level == logging.INFO  # Default level


def test_monitoring_logger_custom_level():
    """Test initializing the MonitoringLogger with custom level."""
    logger = MonitoringLogger("test_module", log_level="DEBUG")
    assert logger.logger.level == logging.DEBUG


@pytest.fixture
def temp_log_file():
    """Create a temporary log file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as temp_file:
        log_path = Path(temp_file.name)
    
    # Yield the temporary file path
    yield log_path
    
    # Clean up
    if log_path.exists():
        log_path.unlink()


def test_logger_debug(temp_log_file):
    """Test logging debug messages."""
    with patch("logging.FileHandler", return_value=logging.FileHandler(temp_log_file)):
        logger = MonitoringLogger("test_module", log_level="DEBUG")
        logger.debug("Debug message", extra_field="test")
        
        # Check the log file content
        with open(temp_log_file, "r") as f:
            log_content = f.read()
            assert "DEBUG" in log_content
            assert "Debug message" in log_content
            assert "extra_field" in log_content
            assert "test" in log_content


def test_logger_info(temp_log_file):
    """Test logging info messages."""
    with patch("logging.FileHandler", return_value=logging.FileHandler(temp_log_file)):
        logger = MonitoringLogger("test_module")
        logger.info("Info message", status="success")
        
        # Check the log file content
        with open(temp_log_file, "r") as f:
            log_content = f.read()
            assert "INFO" in log_content
            assert "Info message" in log_content
            assert "status" in log_content
            assert "success" in log_content


def test_logger_warning(temp_log_file):
    """Test logging warning messages."""
    with patch("logging.FileHandler", return_value=logging.FileHandler(temp_log_file)):
        logger = MonitoringLogger("test_module")
        logger.warning("Warning message", code=404)
        
        # Check the log file content
        with open(temp_log_file, "r") as f:
            log_content = f.read()
            assert "WARNING" in log_content
            assert "Warning message" in log_content
            assert "code" in log_content
            assert "404" in log_content


def test_logger_error(temp_log_file):
    """Test logging error messages."""
    with patch("logging.FileHandler", return_value=logging.FileHandler(temp_log_file)):
        logger = MonitoringLogger("test_module")
        logger.error("Error message", error_type="ValueError")
        
        # Check the log file content
        with open(temp_log_file, "r") as f:
            log_content = f.read()
            assert "ERROR" in log_content
            assert "Error message" in log_content
            assert "error_type" in log_content
            assert "ValueError" in log_content


def test_logger_exception(temp_log_file):
    """Test logging exceptions."""
    with patch("logging.FileHandler", return_value=logging.FileHandler(temp_log_file)):
        logger = MonitoringLogger("test_module")
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.exception(e, context="test_context")
        
        # Check the log file content
        with open(temp_log_file, "r") as f:
            log_content = f.read()
            assert "ERROR" in log_content
            assert "Test exception" in log_content
            assert "Traceback" in log_content
            assert "context" in log_content
            assert "test_context" in log_content


def test_logger_critical(temp_log_file):
    """Test logging critical messages."""
    with patch("logging.FileHandler", return_value=logging.FileHandler(temp_log_file)):
        logger = MonitoringLogger("test_module")
        logger.critical("Critical message", impact="high")
        
        # Check the log file content
        with open(temp_log_file, "r") as f:
            log_content = f.read()
            assert "CRITICAL" in log_content
            assert "Critical message" in log_content
            assert "impact" in log_content
            assert "high" in log_content


def test_get_latest_errors():
    """Test getting latest errors from log file."""
    mock_log_lines = [
        '{"timestamp": "2023-01-01T12:00:00", "level": "INFO", "message": "Test info"}',
        '{"timestamp": "2023-01-01T12:01:00", "level": "ERROR", "message": "Test error 1", "exception": "ValueError"}',
        '{"timestamp": "2023-01-01T12:02:00", "level": "ERROR", "message": "Test error 2", "exception": "KeyError"}'
    ]
    
    # Create a mock for open and readlines
    with patch("builtins.open", mock_open(read_data="\n".join(mock_log_lines))):
        with patch("os.path.exists", return_value=True):
            errors = get_latest_errors(limit=10)
            
            assert len(errors) == 2  # Only ERROR messages
            assert errors[0]["message"] == "Test error 1"
            assert errors[1]["message"] == "Test error 2"


def test_get_latest_errors_with_limit():
    """Test getting limited number of errors."""
    mock_log_lines = [
        '{"timestamp": "2023-01-01T12:00:00", "level": "ERROR", "message": "Test error 1"}',
        '{"timestamp": "2023-01-01T12:01:00", "level": "ERROR", "message": "Test error 2"}',
        '{"timestamp": "2023-01-01T12:02:00", "level": "ERROR", "message": "Test error 3"}'
    ]
    
    # Create a mock for open and readlines
    with patch("builtins.open", mock_open(read_data="\n".join(mock_log_lines))):
        with patch("os.path.exists", return_value=True):
            errors = get_latest_errors(limit=2)
            
            assert len(errors) == 2  # Only the latest 2 errors
            assert errors[0]["message"] == "Test error 2"
            assert errors[1]["message"] == "Test error 3"


def test_get_error_summary():
    """Test getting error summary from log file."""
    mock_errors = [
        {"level": "ERROR", "message": "KeyError: 'test'", "exception": "KeyError"},
        {"level": "ERROR", "message": "KeyError: 'test'", "exception": "KeyError"},
        {"level": "ERROR", "message": "ValueError: invalid value", "exception": "ValueError"}
    ]
    
    summary = get_error_summary(mock_errors)
    
    assert len(summary) == 2  # Two types of errors
    assert summary[0]["count"] == 2  # Two KeyErrors
    assert summary[0]["exception"] == "KeyError"
    assert summary[1]["count"] == 1  # One ValueError
    assert summary[1]["exception"] == "ValueError"


def test_add_logging_middleware():
    """Test adding logging middleware to FastAPI app."""
    # Create a mock FastAPI app
    mock_app = MagicMock()
    
    add_logging_middleware(mock_app, "test_service")
    
    # The middleware should have been added to the app
    mock_app.middleware.assert_called_once()