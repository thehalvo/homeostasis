"""
Tests for the orchestrator module.
"""
import os
import sys
import pytest
import subprocess
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestrator.orchestrator import Orchestrator

# Sample configuration for testing
SAMPLE_CONFIG = {
    "service": {
        "path": "services/example_service",
        "start_command": "python app.py",
        "stop_command": "kill -9 $(lsof -t -i:8000)",
        "health_check_url": "http://localhost:8000/health",
        "health_check_timeout": 3
    },
    "monitoring": {
        "check_interval": 5,
        "log_level": "INFO"
    },
    "analysis": {
        "rule_based": {
            "enabled": True
        },
        "ai_based": {
            "enabled": False
        }
    },
    "patch_generation": {
        "generated_patches_dir": "patches",
        "backup_original_files": True
    },
    "testing": {
        "enabled": True,
        "test_command": "pytest tests/test_app.py -v",
        "test_timeout": 30
    },
    "deployment": {
        "enabled": True,
        "restart_service": True,
        "backup_dir": "backups"
    }
}


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
        yaml.dump(SAMPLE_CONFIG, temp_file)
        temp_path = Path(temp_file.name)
    
    yield temp_path
    
    # Clean up
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def orchestrator(temp_config_file):
    """Create an Orchestrator instance for testing."""
    with patch("orchestrator.orchestrator.MonitoringLogger"):
        with patch("orchestrator.orchestrator.Analyzer"):
            with patch("orchestrator.orchestrator.PatchGenerator"):
                orchestrator = Orchestrator(temp_config_file, log_level="DEBUG")
                # Mock the directories creation
                with patch.object(orchestrator, "_create_directories"):
                    yield orchestrator


def test_orchestrator_initialization(orchestrator):
    """Test that the orchestrator initializes correctly."""
    assert orchestrator.config == SAMPLE_CONFIG
    assert orchestrator.service_process is None


def test_load_config(temp_config_file):
    """Test loading configuration from a file."""
    with patch("orchestrator.orchestrator.MonitoringLogger"):
        with patch("orchestrator.orchestrator.Analyzer"):
            with patch("orchestrator.orchestrator.PatchGenerator"):
                with patch.object(Orchestrator, "_create_directories"):
                    orchestrator = Orchestrator(temp_config_file)
                    assert orchestrator.config == SAMPLE_CONFIG


def test_start_service(orchestrator):
    """Test starting the service."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None  # Process is running
    
    with patch("subprocess.Popen", return_value=mock_process):
        with patch("time.sleep"):
            orchestrator.start_service()
            assert orchestrator.service_process is not None


def test_start_service_failure(orchestrator):
    """Test handling of service start failure."""
    mock_process = MagicMock()
    mock_process.poll.return_value = 1  # Process failed to start
    mock_process.stderr.read.return_value = "Error starting service"
    
    with patch("subprocess.Popen", return_value=mock_process):
        with patch("time.sleep"):
            orchestrator.start_service()
            assert orchestrator.service_process is not None  # Process reference is still stored
            orchestrator.logger.error.assert_called_once()  # Error should be logged


def test_stop_service(orchestrator):
    """Test stopping the service."""
    mock_process = MagicMock()
    orchestrator.service_process = mock_process
    
    with patch("subprocess.run"):
        orchestrator.stop_service()
        mock_process.terminate.assert_called_once()
        assert orchestrator.service_process is None


def test_stop_service_with_timeout(orchestrator):
    """Test stopping service that does not terminate within timeout."""
    mock_process = MagicMock()
    mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
    orchestrator.service_process = mock_process
    
    with patch("subprocess.run"):
        orchestrator.stop_service()
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()  # Should force kill
        assert orchestrator.service_process is None


def test_check_service_health_process_dead(orchestrator):
    """Test health check when process is not running."""
    mock_process = MagicMock()
    mock_process.poll.return_value = 1  # Process has terminated
    orchestrator.service_process = mock_process
    
    assert orchestrator.check_service_health() is False


def test_check_service_health_endpoint_success(orchestrator):
    """Test health check with successful response from endpoint."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None  # Process is running
    orchestrator.service_process = mock_process
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    
    with patch("requests.get", return_value=mock_response):
        assert orchestrator.check_service_health() is True


def test_check_service_health_endpoint_failure(orchestrator):
    """Test health check with failed response from endpoint."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None  # Process is running
    orchestrator.service_process = mock_process
    
    mock_response = MagicMock()
    mock_response.status_code = 500
    
    with patch("requests.get", return_value=mock_response):
        assert orchestrator.check_service_health() is False


def test_check_service_health_request_exception(orchestrator):
    """Test health check with request exception."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None  # Process is running
    orchestrator.service_process = mock_process
    
    with patch("requests.get", side_effect=Exception("Connection error")):
        assert orchestrator.check_service_health() is False


def test_monitor_for_errors(orchestrator):
    """Test monitoring for errors."""
    mock_errors = [{"error": "test error"}]
    
    with patch("orchestrator.orchestrator.get_latest_errors", return_value=mock_errors):
        errors = orchestrator.monitor_for_errors()
        assert errors == mock_errors


def test_analyze_errors(orchestrator):
    """Test analyzing errors."""
    mock_errors = [{"error": "test error"}]
    mock_results = [{"root_cause": "test cause", "confidence": 0.8}]
    orchestrator.analyzer.analyze_errors.return_value = mock_results
    
    results = orchestrator.analyze_errors(mock_errors)
    assert results == mock_results
    orchestrator.analyzer.analyze_errors.assert_called_once_with(mock_errors)


def test_generate_patches(orchestrator):
    """Test generating patches based on analysis results."""
    mock_results = [
        {"root_cause": "keyerror", "confidence": 0.9},
        {"root_cause": "unknown_error", "confidence": 0.5}
    ]
    
    # Mock that only one patch can be generated
    orchestrator.patch_generator.generate_patch_from_analysis.side_effect = [
        {"patch_id": "patch1", "file_path": "test.py"},
        None
    ]
    
    patches = orchestrator.generate_patches(mock_results)
    assert len(patches) == 1
    assert patches[0]["patch_id"] == "patch1"


def test_apply_patches(orchestrator):
    """Test applying patches."""
    mock_patches = [
        {
            "patch_id": "patch1",
            "patch_type": "specific",
            "file_path": "test.py"
        },
        {
            "patch_id": "patch2",
            "patch_type": "general",  # This should be skipped
            "file_path": "test2.py"
        }
    ]
    
    # Mock successful patch application
    orchestrator.patch_generator.apply_patch.return_value = True
    
    with patch("pathlib.Path.exists", return_value=True):
        with patch("shutil.copy2"):
            applied = orchestrator.apply_patches(mock_patches)
            assert len(applied) == 1
            assert applied[0] == "patch1"
            # Should only try to apply the specific patch
            orchestrator.patch_generator.apply_patch.assert_called_once()


def test_run_tests_success(orchestrator):
    """Test running tests with success."""
    mock_process = MagicMock()
    mock_process.returncode = 0  # Tests passed
    
    with patch("subprocess.run", return_value=mock_process):
        assert orchestrator.run_tests() is True


def test_run_tests_failure(orchestrator):
    """Test running tests with failure."""
    mock_process = MagicMock()
    mock_process.returncode = 1  # Tests failed
    mock_process.stderr = "Test failures"
    
    with patch("subprocess.run", return_value=mock_process):
        assert orchestrator.run_tests() is False


def test_run_tests_disabled(orchestrator):
    """Test skipping tests when disabled in config."""
    orchestrator.config["testing"]["enabled"] = False
    assert orchestrator.run_tests() is True  # Should pass without running tests


def test_deploy_changes_success(orchestrator):
    """Test successful deployment of changes."""
    with patch.object(orchestrator, "stop_service"):
        with patch.object(orchestrator, "start_service"):
            with patch.object(orchestrator, "check_service_health", return_value=True):
                assert orchestrator.deploy_changes() is True


def test_deploy_changes_health_check_failure(orchestrator):
    """Test deployment with health check failure."""
    with patch.object(orchestrator, "stop_service"):
        with patch.object(orchestrator, "start_service"):
            with patch.object(orchestrator, "check_service_health", return_value=False):
                assert orchestrator.deploy_changes() is False


def test_deploy_changes_disabled(orchestrator):
    """Test skipping deployment when disabled in config."""
    orchestrator.config["deployment"]["enabled"] = False
    assert orchestrator.deploy_changes() is True  # Should pass without deploying


def test_run_self_healing_cycle_no_errors(orchestrator):
    """Test self-healing cycle with no errors found."""
    with patch.object(orchestrator, "monitor_for_errors", return_value=[]):
        assert orchestrator.run_self_healing_cycle() is True


def test_run_self_healing_cycle_no_analysis(orchestrator):
    """Test self-healing cycle with errors but no analysis results."""
    with patch.object(orchestrator, "monitor_for_errors", return_value=["error"]):
        with patch.object(orchestrator, "analyze_errors", return_value=[]):
            assert orchestrator.run_self_healing_cycle() is True


def test_run_self_healing_cycle_no_patches(orchestrator):
    """Test self-healing cycle with analysis but no patches."""
    with patch.object(orchestrator, "monitor_for_errors", return_value=["error"]):
        with patch.object(orchestrator, "analyze_errors", return_value=["analysis"]):
            with patch.object(orchestrator, "generate_patches", return_value=[]):
                assert orchestrator.run_self_healing_cycle() is True


def test_run_self_healing_cycle_patch_failure(orchestrator):
    """Test self-healing cycle with patches but application failure."""
    with patch.object(orchestrator, "monitor_for_errors", return_value=["error"]):
        with patch.object(orchestrator, "analyze_errors", return_value=["analysis"]):
            with patch.object(orchestrator, "generate_patches", return_value=["patch"]):
                with patch.object(orchestrator, "apply_patches", return_value=[]):
                    assert orchestrator.run_self_healing_cycle() is False


def test_run_self_healing_cycle_test_failure(orchestrator):
    """Test self-healing cycle with patches applied but test failure."""
    with patch.object(orchestrator, "monitor_for_errors", return_value=["error"]):
        with patch.object(orchestrator, "analyze_errors", return_value=["analysis"]):
            with patch.object(orchestrator, "generate_patches", return_value=["patch"]):
                with patch.object(orchestrator, "apply_patches", return_value=["patch1"]):
                    with patch.object(orchestrator, "run_tests", return_value=False):
                        assert orchestrator.run_self_healing_cycle() is False


def test_run_self_healing_cycle_deployment_failure(orchestrator):
    """Test self-healing cycle with successful patches and tests but deployment failure."""
    with patch.object(orchestrator, "monitor_for_errors", return_value=["error"]):
        with patch.object(orchestrator, "analyze_errors", return_value=["analysis"]):
            with patch.object(orchestrator, "generate_patches", return_value=["patch"]):
                with patch.object(orchestrator, "apply_patches", return_value=["patch1"]):
                    with patch.object(orchestrator, "run_tests", return_value=True):
                        with patch.object(orchestrator, "deploy_changes", return_value=False):
                            assert orchestrator.run_self_healing_cycle() is False


def test_run_self_healing_cycle_success(orchestrator):
    """Test successful self-healing cycle."""
    with patch.object(orchestrator, "monitor_for_errors", return_value=["error"]):
        with patch.object(orchestrator, "analyze_errors", return_value=["analysis"]):
            with patch.object(orchestrator, "generate_patches", return_value=["patch"]):
                with patch.object(orchestrator, "apply_patches", return_value=["patch1"]):
                    with patch.object(orchestrator, "run_tests", return_value=True):
                        with patch.object(orchestrator, "deploy_changes", return_value=True):
                            assert orchestrator.run_self_healing_cycle() is True