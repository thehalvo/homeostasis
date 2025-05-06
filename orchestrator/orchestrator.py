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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Generator

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules
from modules.monitoring.logger import MonitoringLogger
from modules.monitoring.extractor import get_latest_errors, get_error_summary
from modules.analysis.analyzer import Analyzer
from modules.patch_generation.patcher import PatchGenerator


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
        self.analyzer = Analyzer(use_ai=self.config["analysis"]["ai_based"]["enabled"])
        self.patch_generator = PatchGenerator()
        
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
    
    def apply_patches(self, patches: List[Dict[str, Any]]) -> List[str]:
        """
        Apply patches to the codebase.

        Args:
            patches: List of patches to apply

        Returns:
            List of successfully applied patch IDs
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
                applied_patches.append(patch["patch_id"])
                self.logger.info(f"Successfully applied patch {patch['patch_id']}")
            else:
                self.logger.error(f"Failed to apply patch {patch['patch_id']}")
        
        return applied_patches
    
    def run_tests(self) -> bool:
        """
        Run tests to validate the applied patches.

        Returns:
            True if tests pass, False otherwise
        """
        if not self.config["testing"]["enabled"]:
            self.logger.info("Testing is disabled, skipping tests")
            return True
        
        self.logger.info("Running tests")
        
        test_command = self.config["testing"]["test_command"]
        test_timeout = self.config["testing"]["test_timeout"]
        
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
            if result.returncode == 0:
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
    
    def deploy_changes(self) -> bool:
        """
        Deploy the changes by restarting the service.

        Returns:
            True if deployment succeeded, False otherwise
        """
        if not self.config["deployment"]["enabled"]:
            self.logger.info("Deployment is disabled, skipping service restart")
            return True
        
        self.logger.info("Deploying changes")
        
        # Restart the service
        if self.config["deployment"]["restart_service"]:
            self.logger.info("Restarting service")
            
            # Stop the service
            self.stop_service()
            
            # Start the service
            self.start_service()
            
            # Check if the service is healthy
            for _ in range(3):  # Try a few times
                if self.check_service_health():
                    self.logger.info("Service is healthy after restart")
                    return True
                time.sleep(1)
            
            self.logger.error("Service is not healthy after restart")
            return False
        
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
        
        # Run tests
        tests_passed = self.run_tests()
        
        if not tests_passed:
            self.logger.warning("Tests failed, rolling back changes")
            # TODO: Implement rollback
            return False
        
        # Deploy changes
        deployment_succeeded = self.deploy_changes()
        
        if not deployment_succeeded:
            self.logger.warning("Deployment failed")
            return False
        
        self.logger.info("Self-healing cycle completed successfully")
        return True
    
    def demo_known_bugs(self) -> None:
        """Run a demonstration of fixing known bugs."""
        self.logger.info("Running demonstration of fixing known bugs")
        
        # Generate patches for known bugs
        patches = self.generate_patches_for_known_bugs()
        
        # Apply patches
        applied_patches = self.apply_patches(patches)
        
        # Run tests
        tests_passed = self.run_tests()
        
        # Deploy changes
        if tests_passed:
            deployment_succeeded = self.deploy_changes()
            
            if deployment_succeeded:
                self.logger.info("Demonstration completed successfully")
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
    
    args = parser.parse_args()
    
    # Determine the config path
    if os.path.isabs(args.config):
        config_path = Path(args.config)
    else:
        # Relative to the orchestrator directory
        config_path = Path(__file__).parent / args.config
    
    # Initialize and run the orchestrator
    orchestrator = Orchestrator(config_path, log_level=args.log_level)
    orchestrator.run(demo_mode=args.demo)


if __name__ == "__main__":
    main()