#!/usr/bin/env python3
"""
Homeostasis Orchestrator

Coordinates the self-healing process from error detection to patch deployment.
"""
import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Generator

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules
from modules.monitoring.logger import MonitoringLogger
from modules.monitoring.extractor import get_latest_errors, get_error_summary
from modules.monitoring.post_deployment import PostDeploymentMonitor, SuccessRateTracker
from modules.monitoring.metrics_collector import MetricsCollector
from modules.monitoring.feedback_loop import FeedbackLoop
from modules.monitoring.alert_system import AlertManager, AnomalyDetector
from modules.analysis.analyzer import Analyzer, AnalysisStrategy
from modules.patch_generation.patcher import PatchGenerator
from modules.testing.runner import TestRunner
from modules.testing.container_manager import ContainerManager
from modules.testing.parallel_runner import ParallelTestRunner
from modules.testing.regression_generator import RegressionTestGenerator


class Orchestrator:
    """
    Main orchestrator class that manages the self-healing workflow.
    """

    def __init__(self, config_path: Path, log_level: str = "INFO"):
        """
        Initialize the orchestrator with the specified configuration.

        Args:
            config_path: Path to the configuration file
            log_level: Logging level
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Set up logging
        self.logger = MonitoringLogger("orchestrator", log_level=log_level)
        
        # Initialize components
        # Convert boolean to strategy string
        strategy = AnalysisStrategy.AI_FALLBACK if self.config["analysis"]["ai_based"]["enabled"] else AnalysisStrategy.RULE_BASED_ONLY
        self.analyzer = Analyzer(strategy=strategy)
        self.patch_generator = PatchGenerator()
        
        # Initialize testing components
        if self.config["testing"]["containers"]["enabled"]:
            self.container_manager = ContainerManager(log_level=log_level)
            self.test_runner = ParallelTestRunner(
                log_level=log_level,
                max_workers=self.config["testing"]["parallel"]["max_workers"],
                use_containers=True
            )
        else:
            self.test_runner = ParallelTestRunner(
                log_level=log_level,
                max_workers=self.config["testing"]["parallel"]["max_workers"],
                use_containers=False
            )
            
        # Initialize regression test generator
        if self.config["testing"]["regression"]["enabled"]:
            self.regression_generator = RegressionTestGenerator(
                output_dir=project_root / self.config["testing"]["regression"]["save_path"],
                log_level=log_level
            )
        
        # Initialize monitoring components
        self.metrics_collector = MetricsCollector(log_level=log_level)
        
        # Initialize alert manager
        self.alert_manager = AlertManager(log_level=log_level)
        
        # Initialize post-deployment monitor
        self.post_deployment_monitor = PostDeploymentMonitor(
            config=self.config["monitoring"]["post_deployment"],
            log_level=log_level
        )
        
        # Initialize success rate tracker
        self.success_tracker = SuccessRateTracker(log_level=log_level)
        
        # Initialize feedback loop
        self.feedback_loop = FeedbackLoop(
            metrics_collector=self.metrics_collector,
            log_level=log_level
        )
        
        # Initialize anomaly detector
        self.anomaly_detector = AnomalyDetector(
            alert_manager=self.alert_manager,
            log_level=log_level
        )
        
        # Create necessary directories
        self._create_directories()
        
        # Service process
        self.service_process = None
        
        self.logger.info("Orchestrator initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from the config file.

        Returns:
            Configuration dictionary
        """
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        # Create logs directory
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create patches directory
        patches_dir = project_root / self.config["patch_generation"]["generated_patches_dir"]
        patches_dir.parent.mkdir(exist_ok=True)
        patches_dir.mkdir(exist_ok=True)
        
        # Create backups directory
        backups_dir = project_root / self.config["deployment"]["backup_dir"]
        backups_dir.parent.mkdir(exist_ok=True)
        backups_dir.mkdir(exist_ok=True)
        
        # Create sessions directory for storing applied patches metadata
        sessions_dir = project_root / "sessions"
        sessions_dir.mkdir(exist_ok=True)
    
    def start_service(self) -> None:
        """Start the monitored service."""
        service_path = project_root / self.config["service"]["path"]
        command = self.config["service"]["start_command"]
        
        self.logger.info(f"Starting service: {command}", service_path=str(service_path))
        
        # Start the service
        try:
            self.service_process = subprocess.Popen(
                command,
                shell=True,
                cwd=service_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for the service to start
            time.sleep(2)
            
            # Check if the service started successfully
            if self.service_process.poll() is not None:
                stderr = self.service_process.stderr.read()
                self.logger.error(f"Service failed to start: {stderr}")
                return
            
            self.logger.info("Service started successfully")
        except Exception as e:
            self.logger.exception(e, message="Failed to start service")
    
    def stop_service(self) -> None:
        """Stop the monitored service."""
        if not self.service_process:
            return
        
        self.logger.info("Stopping service")
        
        # Try to stop gracefully
        self.service_process.terminate()
        
        # Wait for process to terminate
        try:
            self.service_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't terminate
            self.service_process.kill()
        
        self.service_process = None
        
        # Also use the configured stop command
        try:
            stop_command = self.config["service"]["stop_command"]
            subprocess.run(stop_command, shell=True, check=False)
        except Exception as e:
            self.logger.exception(e, message="Error running stop command")
    
    def check_service_health(self) -> bool:
        """
        Check if the service is healthy.

        Returns:
            True if the service is healthy, False otherwise
        """
        # First check if the process is still running
        if self.service_process and self.service_process.poll() is not None:
            self.logger.warning("Service process is not running")
            return False
        
        # Then check the health endpoint
        import requests
        
        health_url = self.config["service"]["health_check_url"]
        timeout = self.config["service"]["health_check_timeout"]
        
        try:
            response = requests.get(health_url, timeout=timeout)
            return response.status_code == 200
        except requests.RequestException as e:
            self.logger.warning(f"Health check failed: {str(e)}")
            return False
    
    def monitor_for_errors(self) -> List[Dict[str, Any]]:
        """
        Monitor logs for errors.

        Returns:
            List of error log entries
        """
        self.logger.info("Checking for errors in logs")
        
        # Get the latest errors from the log file
        errors = get_latest_errors(limit=10)
        
        if errors:
            self.logger.info(f"Found {len(errors)} errors")
        else:
            self.logger.info("No errors found")
        
        return errors
    
    def analyze_errors(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze errors to identify root causes.

        Args:
            errors: List of error log entries

        Returns:
            List of analysis results
        """
        self.logger.info(f"Analyzing {len(errors)} errors")
        
        # Analyze each error
        analysis_results = self.analyzer.analyze_errors(errors)
        
        # Log analysis results
        for i, result in enumerate(analysis_results):
            self.logger.info(
                f"Analysis result {i+1}/{len(analysis_results)}: {result['root_cause']}",
                confidence=result["confidence"]
            )
        
        return analysis_results
    
    def generate_patches(self, analysis_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate patches based on analysis results.

        Args:
            analysis_results: List of analysis results

        Returns:
            List of generated patches
        """
        self.logger.info("Generating patches")
        
        patches = []
        
        # Check if we can generate a patch for each analysis result
        for result in analysis_results:
            patch = self.patch_generator.generate_patch_from_analysis(result)
            if patch:
                patches.append(patch)
                self.logger.info(f"Generated patch for {result['root_cause']}")
            else:
                self.logger.info(f"No patch template available for {result['root_cause']}")
        
        return patches
    
    def generate_patches_for_known_bugs(self) -> List[Dict[str, Any]]:
        """
        Generate patches for known bugs (for demonstration).

        Returns:
            List of generated patches
        """
        self.logger.info("Generating patches for known bugs")
        
        patches = self.patch_generator.generate_patches_for_all_known_bugs()
        
        self.logger.info(f"Generated {len(patches)} patches for known bugs")
        
        return patches
    
    def apply_patches(self, patches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply patches to the codebase.

        Args:
            patches: List of patches to apply

        Returns:
            List of successfully applied patches with metadata
        """
        self.logger.info(f"Applying {len(patches)} patches")
        
        applied_patches = []
        
        for patch in patches:
            if patch["patch_type"] != "specific":
                self.logger.warning(f"Cannot automatically apply patch type: {patch['patch_type']}")
                continue
            
            self.logger.info(f"Applying patch to {patch['file_path']}")
            
            # Back up the file before modifying it
            if self.config["patch_generation"]["backup_original_files"]:
                file_path = project_root / patch["file_path"]
                backup_dir = project_root / self.config["deployment"]["backup_dir"]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"{file_path.name}.{timestamp}.bak"
                
                try:
                    shutil.copy2(file_path, backup_path)
                    self.logger.info(f"Backed up {file_path} to {backup_path}")
                except Exception as e:
                    self.logger.exception(e, message=f"Failed to backup {file_path}")
                    continue
            
            # Apply the patch
            success = self.patch_generator.apply_patch(patch, project_root)
            
            if success:
                # Store patch information with backup path for potential rollback
                patch_info = {
                    "patch_id": patch["patch_id"],
                    "file_path": patch["file_path"],
                    "backup_path": str(backup_path) if self.config["patch_generation"]["backup_original_files"] else None,
                    "timestamp": timestamp
                }
                applied_patches.append(patch_info)
                self.logger.info(f"Successfully applied patch {patch['patch_id']}")
            else:
                self.logger.error(f"Failed to apply patch {patch['patch_id']}")
        
        return applied_patches
    
    def run_tests(self, patches: List[Dict[str, Any]] = None) -> bool:
        """
        Run tests to validate the applied patches.

        Args:
            patches: Optional list of patches to validate

        Returns:
            True if tests pass, False otherwise
        """
        if not self.config["testing"]["enabled"]:
            self.logger.info("Testing is disabled, skipping tests")
            return True
        
        self.logger.info("Running tests")
        
        # Get test configuration
        test_command = self.config["testing"]["test_command"]
        test_timeout = self.config["testing"]["test_timeout"]
        
        # Check if we should use advanced testing features
        if self.config["testing"]["graduated_testing"]["enabled"] and patches:
            self.logger.info("Using graduated testing strategy")
            
            # Get test levels
            test_levels = self.config["testing"]["graduated_testing"]["levels"]
            
            # Get test commands
            test_commands = self.config["testing"]["graduated_testing"]["commands"]
            
            # Get timeouts
            timeouts = self.config["testing"]["graduated_testing"]["timeouts"]
            
            # Get resource limits
            resource_limits = self.config["testing"]["graduated_testing"]["resource_limits"]
            
            # Run graduated tests for each patch
            results = []
            for patch in patches:
                patch_id = patch.get("patch_id", "unknown")
                
                self.logger.info(f"Running graduated tests for patch {patch_id}")
                
                result = self.test_runner.run_graduated_tests(
                    patch,
                    test_types=test_levels,
                    test_commands=test_commands,
                    timeouts=timeouts,
                    resource_limits=resource_limits
                )
                
                results.append(result)
                
                # Record metrics
                self.metrics_collector.record_test_metric(
                    patch_id,
                    result["valid"],
                    result["test_levels"].get(test_levels[-1], {}).get("test_results", {}).get("duration", 0),
                    other_data={"test_type": "graduated", "levels": test_levels}
                )
                
                # Generate regression tests if enabled
                if self.config["testing"]["regression"]["enabled"] and result["valid"]:
                    if hasattr(self, "regression_generator"):
                        self.logger.info(f"Generating regression test for patch {patch_id}")
                        
                        error_info = None
                        if "analysis_result" in patch:
                            error_info = patch["analysis_result"]
                            
                        test_path = self.regression_generator.generate_regression_test(patch, error_info)
                        
                        if test_path:
                            self.logger.info(f"Generated regression test at {test_path}")
            
            # Check if all patches passed tests
            all_passed = all(result["valid"] for result in results)
            
            if all_passed:
                self.logger.info("All graduated tests passed")
            else:
                failed_patches = [result["patch_id"] for result in results if not result["valid"]]
                self.logger.error(f"Graduated tests failed for patches: {', '.join(failed_patches)}")
                
            return all_passed
        
        # Use simple test runner if not using advanced features
        try:
            # Run tests
            result = subprocess.run(
                test_command,
                shell=True,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=test_timeout
            )
            
            # Check if tests passed
            success = result.returncode == 0
            
            # Record metrics if patches are provided
            if patches:
                for patch in patches:
                    patch_id = patch.get("patch_id", "unknown")
                    
                    self.metrics_collector.record_test_metric(
                        patch_id,
                        success,
                        time.time(),  # We don't have accurate duration for subprocess
                        other_data={"test_type": "simple", "command": test_command}
                    )
            
            if success:
                self.logger.info("Tests passed")
                return True
            else:
                self.logger.error(f"Tests failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.logger.error(f"Tests timed out after {test_timeout} seconds")
            return False
        except Exception as e:
            self.logger.exception(e, message="Error running tests")
            return False
    
    def deploy_changes(self, patches: List[Dict[str, Any]] = None) -> bool:
        """
        Deploy the changes by restarting the service.

        Args:
            patches: Optional list of patches being deployed

        Returns:
            True if deployment succeeded, False otherwise
        """
        if not self.config["deployment"]["enabled"]:
            self.logger.info("Deployment is disabled, skipping service restart")
            return True
        
        self.logger.info("Deploying changes")
        
        start_time = time.time()
        
        # Restart the service
        if self.config["deployment"]["restart_service"]:
            self.logger.info("Restarting service")
            
            # Stop the service
            self.stop_service()
            
            # Start the service
            self.start_service()
            
            # Check if the service is healthy
            success = False
            for _ in range(3):  # Try a few times
                if self.check_service_health():
                    self.logger.info("Service is healthy after restart")
                    success = True
                    break
                time.sleep(1)
            
            if not success:
                self.logger.error("Service is not healthy after restart")
                return False
            
            # Record deployment metrics
            if patches:
                duration = time.time() - start_time
                for patch in patches:
                    patch_id = patch.get("patch_id", "unknown")
                    
                    # Record deployment metrics
                    self.metrics_collector.record_deployment_metric(
                        patch_id,
                        success,
                        duration,
                        other_data={"restart": True}
                    )
                    
            # Start post-deployment monitoring if enabled
            if self.config["monitoring"]["post_deployment"]["enabled"] and patches:
                health_url = self.config["service"]["health_check_url"]
                service_url = health_url.rsplit("/health", 1)[0]  # Extract base URL
                
                for patch in patches:
                    patch_id = patch.get("patch_id", "unknown")
                    
                    # Start monitoring for this patch
                    self.logger.info(f"Starting post-deployment monitoring for patch {patch_id}")
                    self.post_deployment_monitor.start_monitoring(service_url, patch_id)
                    
                    # Set up anomaly detection
                    if hasattr(self, "anomaly_detector"):
                        self.logger.info(f"Starting anomaly detection for patch {patch_id}")
                        
                        # Start anomaly detection in a separate thread
                        threading.Thread(
                            target=self.anomaly_detector.monitor_for_anomalies,
                            args=(service_url, patch_id),
                            daemon=True
                        ).start()
            
            return success
        
        return True
    
    def run_self_healing_cycle(self) -> bool:
        """
        Run a complete self-healing cycle.
        
        1. Monitor for errors
        2. Analyze errors
        3. Generate patches
        4. Apply patches
        5. Run tests
        6. Deploy changes
        
        Returns:
            True if the cycle completed successfully, False otherwise
        """
        self.logger.info("Starting self-healing cycle")
        
        # Monitor for errors
        errors = self.monitor_for_errors()
        
        if not errors:
            self.logger.info("No errors found, nothing to fix")
            return True
        
        # Analyze errors
        analysis_results = self.analyze_errors(errors)
        
        if not analysis_results:
            self.logger.info("No analysis results, nothing to fix")
            return True
        
        # Generate patches
        patches = self.generate_patches(analysis_results)
        
        if not patches:
            self.logger.info("No patches generated, nothing to fix")
            return True
        
        # Apply patches
        applied_patches = self.apply_patches(patches)
        
        if not applied_patches:
            self.logger.warning("No patches applied, self-healing failed")
            return False
            
        # Store the applied patches in a session file for potential rollback
        self._store_session_patches(applied_patches)
        
        # Run tests
        tests_passed = self.run_tests(patches)
        
        if not tests_passed:
            self.logger.warning("Tests failed, rolling back changes")
            
            # Check if auto-rollback is enabled in the config
            if self.config.get("rollback", {}).get("enabled", True) and \
               self.config.get("rollback", {}).get("auto_rollback_on_failure", True):
                rollback_success = self._rollback_changes(applied_patches)
                if not rollback_success:
                    self.logger.error("Rollback failed, system may be in an inconsistent state")
            else:
                self.logger.info("Auto-rollback disabled in config, skipping rollback")
            
            # Record fix failure in the feedback loop
            for patch in patches:
                patch_id = patch.get("patch_id", "unknown")
                bug_id = patch.get("bug_id", "")
                template_name = patch.get("template_name", "unknown")
                
                # Record in success tracker
                self.success_tracker.record_fix(patch_id, bug_id, False)
                
                # Record in feedback loop
                self.feedback_loop.record_fix_result(patch_id, template_name, bug_id, False)
                
            return False
        
        # Deploy changes
        deployment_succeeded = self.deploy_changes(patches)
        
        if not deployment_succeeded:
            self.logger.warning("Deployment failed")
            
            # Record fix failure in the feedback loop
            for patch in patches:
                patch_id = patch.get("patch_id", "unknown")
                bug_id = patch.get("bug_id", "")
                template_name = patch.get("template_name", "unknown")
                
                # Record in success tracker
                self.success_tracker.record_fix(patch_id, bug_id, False)
                
                # Record in feedback loop
                self.feedback_loop.record_fix_result(patch_id, template_name, bug_id, False)
                
            return False
        
        # Record successful fixes
        for patch in patches:
            patch_id = patch.get("patch_id", "unknown")
            bug_id = patch.get("bug_id", "")
            template_name = patch.get("template_name", "unknown")
            
            # Record in success tracker
            self.success_tracker.record_fix(patch_id, bug_id, True)
            
            # Record in feedback loop
            self.feedback_loop.record_fix_result(patch_id, template_name, bug_id, True)
            
            # Record fix metrics
            service_url = self.config["service"]["health_check_url"].rsplit("/health", 1)[0]
            health_data = self.check_health(service_url)
            
            self.metrics_collector.record_fix_metric(
                patch_id,
                bug_id,
                True,
                response_time=health_data.get("response_time"),
                error_rate=health_data.get("error_rate"),
                other_data={
                    "template_name": template_name,
                    "deployment_success": deployment_succeeded
                }
            )
        
        # Generate insights from feedback loop
        recommendations = self.feedback_loop.generate_recommendations()
        if recommendations:
            self.logger.info(f"Generated {len(recommendations)} recommendations for improvement")
            for rec in recommendations:
                self.logger.info(f"Recommendation: {rec['message']}", suggestion=rec.get("suggestion", ""))
        
        self.logger.info("Self-healing cycle completed successfully")
        return True
        
    def check_health(self, service_url: str) -> Dict[str, Any]:
        """
        Check service health and collect basic metrics.
        
        Args:
            service_url: Base URL of the service
            
        Returns:
            Health data with metrics
        """
        metrics = {
            "timestamp": time.time()
        }
        
        try:
            # Response time (health check)
            start_time = time.time()
            response = requests.get(f"{service_url}/health", timeout=5)
            elapsed = time.time() - start_time
            
            metrics["response_time"] = elapsed * 1000  # Convert to ms
            metrics["status_code"] = response.status_code
            metrics["healthy"] = response.status_code == 200
            
            # Try to parse response body if it's JSON
            try:
                response_json = response.json()
                if "memory_usage" in response_json:
                    metrics["memory_usage"] = response_json["memory_usage"]
            except Exception:
                pass
                
        except requests.RequestException as e:
            metrics["error"] = str(e)
            metrics["healthy"] = False
        
        # Error rate (sample requests to endpoints)
        error_count = 0
        request_count = 0
        
        # Try common endpoints
        for endpoint in ["/", "/api", "/status"]:
            try:
                response = requests.get(f"{service_url}{endpoint}", timeout=5)
                request_count += 1
                if response.status_code >= 400:
                    error_count += 1
            except requests.RequestException:
                request_count += 1
                error_count += 1
        
        if request_count > 0:
            metrics["error_rate"] = error_count / request_count
            
        return metrics
    
    def _store_session_patches(self, applied_patches: List[Dict[str, Any]]) -> None:
        """
        Store information about applied patches in a session file for potential rollback.
        Also cleans up old sessions based on config settings.

        Args:
            applied_patches: List of successfully applied patches with metadata
        """
        if not applied_patches:
            return

        # Create a session ID based on timestamp
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        sessions_dir = project_root / "sessions"
        session_file = sessions_dir / f"session_{session_id}.json"

        # Store session information
        session_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "patches": applied_patches
        }

        try:
            with open(session_file, "w") as f:
                json.dump(session_data, f, indent=2)
            self.logger.info(f"Stored session information in {session_file}")
            
            # Clean up old sessions if needed
            self._cleanup_old_sessions()
        except Exception as e:
            self.logger.exception(e, message=f"Failed to store session information")

    def _cleanup_old_sessions(self) -> None:
        """
        Clean up old sessions based on the max_sessions_to_keep config setting.
        """
        # Get the max sessions to keep from config
        max_sessions = self.config.get("rollback", {}).get("max_sessions_to_keep", 10)
        
        sessions_dir = project_root / "sessions"
        if not sessions_dir.exists():
            return
            
        # Get all session files sorted by modification time (oldest first)
        session_files = sorted(sessions_dir.glob("session_*.json"), key=lambda p: p.stat().st_mtime)
        
        # If we have more sessions than the max, delete the oldest ones
        if len(session_files) > max_sessions:
            sessions_to_delete = session_files[:-max_sessions]  # Keep only the newest max_sessions
            for session_file in sessions_to_delete:
                try:
                    session_file.unlink()  # Delete the file
                    self.logger.debug(f"Deleted old session file: {session_file.name}")
                except Exception as e:
                    self.logger.exception(e, message=f"Failed to delete old session file: {session_file.name}")
    
    def _get_latest_session(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest session information for rollback.

        Returns:
            Dictionary with session information or None if no session is found
        """
        sessions_dir = project_root / "sessions"
        if not sessions_dir.exists():
            return None

        # Find the latest session file
        session_files = sorted(sessions_dir.glob("session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not session_files:
            return None

        # Read the latest session file
        try:
            with open(session_files[0], "r") as f:
                session_data = json.load(f)
            return session_data
        except Exception as e:
            self.logger.exception(e, message=f"Failed to read session information from {session_files[0]}")
            return None

    def _rollback_changes(self, applied_patches: List[Dict[str, Any]]) -> bool:
        """
        Roll back changes by restoring backup files.

        Args:
            applied_patches: List of patches to roll back

        Returns:
            True if rollback was successful, False otherwise
        """
        if not applied_patches:
            self.logger.warning("No patches to roll back")
            return True

        success = True
        self.logger.info(f"Rolling back {len(applied_patches)} patches")

        for patch_info in applied_patches:
            patch_id = patch_info.get("patch_id")
            file_path = patch_info.get("file_path")
            backup_path = patch_info.get("backup_path")

            if not backup_path or not Path(backup_path).exists():
                self.logger.warning(f"No backup found for patch {patch_id}, cannot roll back")
                success = False
                continue

            try:
                # Restore the original file from backup
                target_path = project_root / file_path
                shutil.copy2(backup_path, target_path)
                self.logger.info(f"Restored {file_path} from backup {backup_path}")
            except Exception as e:
                self.logger.exception(e, message=f"Failed to roll back patch {patch_id}")
                success = False

        return success

    def rollback_latest_session(self) -> bool:
        """
        Roll back the latest session.

        Returns:
            True if rollback was successful, False otherwise
        """
        # Get the latest session information
        session_data = self._get_latest_session()
        if not session_data:
            self.logger.warning("No session found to roll back")
            return False

        # Extract patches from the session
        patches = session_data.get("patches", [])
        if not patches:
            self.logger.warning("No patches found in session to roll back")
            return False

        # Roll back the patches
        self.logger.info(f"Rolling back session {session_data['session_id']} with {len(patches)} patches")
        rollback_success = self._rollback_changes(patches)

        # Restart the service after rollback if needed
        if rollback_success and self.config["deployment"]["restart_service"]:
            self.logger.info("Restarting service after rollback")
            self.stop_service()
            self.start_service()

            # Check if the service is healthy after restart
            for _ in range(3):  # Try a few times
                if self.check_service_health():
                    self.logger.info("Service is healthy after rollback and restart")
                    return True
                time.sleep(1)

            self.logger.error("Service is not healthy after rollback and restart")
            return False

        return rollback_success

    def demo_known_bugs(self) -> None:
        """Run a demonstration of fixing known bugs."""
        self.logger.info("Running demonstration of fixing known bugs")
        
        # Generate patches for known bugs
        patches = self.generate_patches_for_known_bugs()
        
        # Apply patches
        applied_patches = self.apply_patches(patches)
        
        # Run tests with our advanced testing features
        tests_passed = self.run_tests(patches)
        
        # Deploy changes with monitoring
        if tests_passed:
            deployment_succeeded = self.deploy_changes(patches)
            
            if deployment_succeeded:
                self.logger.info("Demonstration completed successfully")
                
                # Record successful fixes in feedback loop
                for patch in patches:
                    patch_id = patch.get("patch_id", "unknown")
                    bug_id = patch.get("bug_id", "")
                    template_name = patch.get("template_name", "unknown")
                    
                    # Record in success tracker
                    self.success_tracker.record_fix(patch_id, bug_id, True)
                    
                    # Record in feedback loop
                    self.feedback_loop.record_fix_result(patch_id, template_name, bug_id, True)
                
                # Generate recommendations
                recommendations = self.feedback_loop.generate_recommendations()
                if recommendations:
                    self.logger.info(f"Generated {len(recommendations)} recommendations for improvement")
            else:
                self.logger.warning("Demonstration deployment failed")
        else:
            self.logger.warning("Demonstration tests failed")
    
    def run(self, demo_mode: bool = False) -> None:
        """
        Run the orchestrator.

        Args:
            demo_mode: Whether to run in demonstration mode
        """
        self.logger.info(f"Starting orchestrator in {'demo' if demo_mode else 'normal'} mode")
        
        try:
            # Start the service
            self.start_service()
            
            if not self.check_service_health():
                self.logger.error("Service is not healthy, cannot proceed")
                return
            
            if demo_mode:
                # Run demonstration
                self.demo_known_bugs()
            else:
                # Run continuous monitoring and self-healing
                check_interval = self.config["monitoring"]["check_interval"]
                
                while True:
                    self.run_self_healing_cycle()
                    time.sleep(check_interval)
        
        except KeyboardInterrupt:
            self.logger.info("Orchestrator interrupted by user")
        finally:
            # Clean up
            self.stop_service()
            self.logger.info("Orchestrator stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Homeostasis Orchestrator")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run in demonstration mode"
    )
    parser.add_argument(
        "--rollback", "-r",
        action="store_true",
        help="Roll back the latest session"
    )
    
    args = parser.parse_args()
    
    # Determine the config path
    if os.path.isabs(args.config):
        config_path = Path(args.config)
    else:
        # Relative to the orchestrator directory
        config_path = Path(__file__).parent / args.config
    
    # Initialize and run the orchestrator
    orchestrator = Orchestrator(config_path, log_level=args.log_level)
    
    if args.rollback:
        # Roll back the latest session
        success = orchestrator.rollback_latest_session()
        exit(0 if success else 1)
    else:
        # Run the orchestrator normally
        orchestrator.run(demo_mode=args.demo)


if __name__ == "__main__":
    main()