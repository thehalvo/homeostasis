#!/usr/bin/env python3
"""
Homeostasis Orchestrator

Coordinates the self-healing process from error detection to patch deployment.
"""
import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

from modules.analysis.analyzer import AnalysisStrategy, Analyzer
from modules.deployment.blue_green import get_blue_green_deployment
from modules.deployment.canary import CanaryStatus, get_canary_deployment
from modules.deployment.cloud.provider_factory import get_cloud_provider
from modules.deployment.kubernetes.kubernetes_deployment import (
    get_kubernetes_deployment,
)
from modules.deployment.traffic_manager import (
    get_cloud_manager,
    get_kubernetes_manager,
    get_nginx_manager,
    get_traffic_splitter,
)
from modules.monitoring.alert_system import AlertManager, AnomalyDetector
from modules.monitoring.audit_monitor import get_audit_monitor
from modules.monitoring.extractor import get_latest_errors
from modules.monitoring.feedback_loop import FeedbackLoop
from modules.monitoring.healing_audit import (
    end_healing_session,
    get_healing_auditor,
    start_healing_session,
)

# Import modules
from modules.monitoring.logger import MonitoringLogger
from modules.monitoring.metrics_collector import MetricsCollector
from modules.monitoring.post_deployment import PostDeploymentMonitor, SuccessRateTracker
from modules.patch_generation.patcher import PatchGenerator
from modules.security.approval import (
    ApprovalError,
    ApprovalStatus,
    ApprovalType,
    create_approval_request,
    get_approval_manager,
    needs_approval,
)
from modules.security.audit import get_audit_logger, log_fix
from modules.security.healing_rate_limiter import get_healing_rate_limiter
from modules.suggestion.suggestion_manager import (
    SuggestionStatus,
    get_suggestion_manager,
)
from modules.testing.container_manager import ContainerManager
from modules.testing.parallel_runner import ParallelTestRunner
from modules.testing.regression_generator import RegressionTestGenerator

# Define project root
project_root = Path(__file__).parent.parent


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

        # Type declarations for optional components
        self.rate_limiter: Optional[Any] = None

        # Initialize components
        # Convert boolean to strategy string
        strategy = (
            AnalysisStrategy.AI_FALLBACK
            if self.config.get("analysis", {}).get("ai_based", {}).get("enabled", False)
            else AnalysisStrategy.RULE_BASED_ONLY
        )
        self.analyzer = Analyzer(strategy=strategy)
        self.patch_generator = PatchGenerator()

        # Initialize testing components
        containers_enabled = (
            self.config.get("testing", {}).get("containers", {}).get("enabled", False)
        )
        if containers_enabled:
            self.container_manager = ContainerManager(log_level=log_level)
            self.test_runner = ParallelTestRunner(
                log_level=log_level,
                max_workers=self.config.get("testing", {})
                .get("parallel", {})
                .get("max_workers", 4),
                use_containers=True,
            )
        else:
            self.test_runner = ParallelTestRunner(
                log_level=log_level,
                max_workers=self.config.get("testing", {})
                .get("parallel", {})
                .get("max_workers", 4),
                use_containers=False,
            )

        # Initialize regression test generator
        regression_enabled = (
            self.config.get("testing", {}).get("regression", {}).get("enabled", False)
        )
        if regression_enabled:
            self.regression_generator = RegressionTestGenerator(
                output_dir=project_root
                / self.config.get("testing", {})
                .get("regression", {})
                .get("save_path", "tests/regression"),
                log_level=log_level,
            )

        # Initialize monitoring components
        self.metrics_collector = MetricsCollector(log_level=log_level)

        # Initialize alert manager
        self.alert_manager = AlertManager(log_level=log_level)

        # Initialize post-deployment monitor
        post_deployment_config = self.config.get("monitoring", {}).get(
            "post_deployment", {}
        )
        self.post_deployment_monitor = PostDeploymentMonitor(
            config=post_deployment_config, log_level=log_level
        )

        # Initialize security components (approval manager and audit logger)
        if "security" in self.config and self.config["security"].get("enabled", False):
            self.security_enabled = True

            # Initialize approval manager if approval workflow is enabled
            if self.config.get("security", {}).get("approval", {}).get(
                "enabled", False
            ) or self.config["deployment"].get("production", {}).get(
                "require_approval", False
            ):
                self.approval_required = True
                self.approval_manager = get_approval_manager(
                    self.config.get("security", {}).get("approval", {})
                )
                self.logger.info("Approval workflow initialized for critical changes")
            else:
                self.approval_required = False

            # Initialize audit logger
            self.audit_logger = get_audit_logger(
                self.config.get("security", {}).get("audit", {})
            )
            self.logger.info("Audit logging initialized")

            # Initialize healing activity auditor
            healing_audit_config = self.config.get("healing_audit", {})
            self.healing_auditor = get_healing_auditor(healing_audit_config)
            self.logger.info("Healing activity auditor initialized")

            # Initialize audit monitor
            audit_monitor_config = self.config.get("audit_monitor", {})
            self.audit_monitor = get_audit_monitor(audit_monitor_config)
            self.logger.info("Audit monitor initialized")

            # Initialize healing rate limiter if enabled
            healing_rate_limits_config = self.config.get("security", {}).get(
                "healing_rate_limits", {}
            )
            if healing_rate_limits_config.get("enabled", False):
                # Add current environment to the config for environment-specific limits
                healing_rate_limits_config["environment"] = self.config["general"][
                    "environment"
                ]
                self.rate_limiter = get_healing_rate_limiter(healing_rate_limits_config)
                self.logger.info(
                    "Healing rate limiter initialized for throttling healing actions"
                )
            else:
                self.rate_limiter = None
        else:
            self.security_enabled = False
            self.approval_required = False
            self.rate_limiter = None

        # Initialize cloud provider if enabled
        cloud_config = self.config["deployment"].get("cloud", {})
        cloud_provider = cloud_config.get("provider", "none").lower()

        if cloud_provider != "none":
            if cloud_config.get(cloud_provider, {}).get("enabled", False):
                self.cloud_enabled = True
                self.cloud_provider = get_cloud_provider(cloud_config)
                self.logger.info(
                    f"Cloud provider {cloud_provider} initialized for deployment"
                )
            else:
                self.cloud_enabled = False
        else:
            self.cloud_enabled = False

        # Initialize Kubernetes deployment if enabled
        k8s_config = self.config["deployment"].get("kubernetes", {})
        if k8s_config.get("enabled", False):
            self.kubernetes_enabled = True
            self.kubernetes_deployment = get_kubernetes_deployment(k8s_config)
            self.logger.info("Kubernetes deployment initialized")
        else:
            self.kubernetes_enabled = False

        # Initialize traffic managers
        self.traffic_splitter = get_traffic_splitter()
        self.nginx_manager = get_nginx_manager(self.config["deployment"])
        self.kubernetes_manager = get_kubernetes_manager(k8s_config)
        self.cloud_manager = get_cloud_manager(cloud_config)
        self.logger.info("Traffic managers initialized")

        # Initialize canary deployment if enabled
        print(f"DEBUG: Config keys: {list(self.config.keys())}")
        print(f"DEBUG: Security config: {self.config.get('security', 'NOT FOUND')}")

        if self.config.get("security", {}).get("canary", {}).get(
            "enabled", False
        ) or self.config["deployment"].get("production", {}).get(
            "canary_deployment", False
        ):
            self.canary_enabled = True

            # Combine canary configs from security and deployment sections
            canary_config = {}
            if "canary" in self.config.get("security", {}):
                canary_config.update(self.config.get("security", {})["canary"])
            if "canary" in self.config.get("deployment", {}):
                canary_config.update(self.config["deployment"]["canary"])

            # Initialize canary deployment manager
            self.canary_deployment = get_canary_deployment(canary_config)
            self.logger.info("Canary deployment initialized for gradual rollout")
        else:
            self.canary_enabled = False

        # Initialize blue-green deployment if enabled
        if self.config["deployment"].get("production", {}).get("blue_green", False):
            self.blue_green_enabled = True

            # Get blue-green config
            blue_green_config = self.config["deployment"].get("blue_green", {})

            # Initialize blue-green deployment manager
            self.blue_green_deployment = get_blue_green_deployment(blue_green_config)
            self.logger.info(
                "Blue-green deployment initialized for zero-downtime deployments"
            )
        else:
            self.blue_green_enabled = False

        # Initialize success rate tracker
        self.success_tracker = SuccessRateTracker(log_level=log_level)

        # Initialize feedback loop
        self.feedback_loop = FeedbackLoop(
            metrics_collector=self.metrics_collector, log_level=log_level
        )

        # Initialize anomaly detector
        self.anomaly_detector = AnomalyDetector(
            alert_manager=self.alert_manager, log_level=log_level
        )

        # Initialize suggestion manager for human review
        suggestion_config = self.config.get("suggestion", {})
        self.suggestion_manager = get_suggestion_manager(
            storage_dir=str(
                project_root / suggestion_config.get("storage_dir", "logs/suggestions")
            ),
            config=suggestion_config,
        )
        self.logger.info("Suggestion manager initialized for human review")

        # Create necessary directories
        self._create_directories()

        # Service process
        self.service_process: Optional[subprocess.Popen] = None

        self.logger.info("Orchestrator initialized")

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from the config file.

        Returns:
            Configuration dictionary
        """
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        return dict(config)

    def _create_directories(self) -> None:
        """Create necessary directories."""
        # Create logs directory
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Create patches directory
        patches_dir = (
            project_root / self.config["patch_generation"]["generated_patches_dir"]
        )
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
                text=True,
            )

            # Wait for the service to start
            time.sleep(2)

            # Check if the service started successfully
            if self.service_process.poll() is not None:
                stderr = ""
                if self.service_process.stderr:
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

        # Get log file path from configuration
        log_file_path = self.config.get("monitoring", {}).get("log_file")
        if log_file_path:
            from pathlib import Path

            log_file = Path(log_file_path)
            # Import extract_errors directly to pass log_file parameter
            from modules.monitoring.extractor import extract_errors

            errors = extract_errors(
                log_file=log_file, levels=["ERROR", "CRITICAL"], limit=10
            )
        else:
            # Fall back to default
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
                f"Analysis result {i + 1}/{len(analysis_results)}: {result['root_cause']}",
                confidence=result.get("confidence", "unknown"),
            )

        return analysis_results

    def generate_patches(
        self, analysis_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate patches based on analysis results.

        Args:
            analysis_results: List of analysis results

        Returns:
            List of generated patches
        """
        self.logger.info("Generating patches")

        patches = []

        # Check if we should use the suggestion interface for human review
        use_suggestion_interface = self.config.get("suggestion", {}).get(
            "enabled", False
        )
        require_human_review = self.config.get("suggestion", {}).get(
            "require_human_review", False
        )

        # Check if we can generate a patch for each analysis result
        for result in analysis_results:
            error_id = result.get("error_id", str(uuid.uuid4()))

            # Generate the patch using the patch generator
            patch = self.patch_generator.generate_patch_from_analysis(result)

            if patch:
                patches.append(patch)
                self.logger.info(f"Generated patch for {result['root_cause']}")

                # If suggestion interface is enabled, add the suggestion to the suggestion manager
                if use_suggestion_interface:
                    self.logger.info("Creating fix suggestion for human review")

                    # Create error info dictionary for suggestion manager
                    error_info = {
                        "error_id": error_id,
                        "file_path": patch.get("file_path"),
                        "error_type": result.get("error_type"),
                        "error_message": result.get("error_message"),
                        "line_number": result.get("line_number"),
                        "stack_trace": result.get("stack_trace"),
                    }

                    # Generate suggestions (this will create variations of the patch)
                    suggestions = self.suggestion_manager.generate_suggestions(
                        error_id, error_info
                    )

                    # If human review is required, mark the patch for review
                    if require_human_review:
                        # Set a flag in the patch to indicate it needs review
                        patch["requires_review"] = True
                        patch["error_id"] = error_id
                        self.logger.info("Patch requires human review")

                    self.logger.info(
                        f"Generated {len(suggestions)} suggestions for human review"
                    )
            else:
                self.logger.info(
                    f"No patch template available for {result['root_cause']}"
                )

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

        # Check patch application rate limit if rate limiter is enabled
        if hasattr(self, "rate_limiter") and self.rate_limiter:
            if not self.rate_limiter.check_patch_application_limit(len(patches)):
                self.logger.warning(
                    f"Patch application rate limit exceeded, cannot apply {len(patches)} patches"
                )
                return []

        applied_patches = []

        for patch in patches:
            if patch.get("patch_type", "specific") != "specific":
                self.logger.warning(
                    f"Cannot automatically apply patch type: {patch['patch_type']}"
                )
                continue

            file_path = patch.get("file_path")
            if not file_path:
                self.logger.warning("Patch missing file_path, skipping")
                continue
            self.logger.info(f"Applying patch to {file_path}")

            # Check file-specific rate limit if rate limiter is enabled
            if hasattr(self, "rate_limiter") and self.rate_limiter:
                if not self.rate_limiter.check_file_limit(file_path):
                    self.logger.warning(
                        f"File rate limit exceeded for {file_path}, skipping patch"
                    )
                    continue

            # Back up the file before modifying it
            if self.config["patch_generation"]["backup_original_files"]:
                file_path_obj = project_root / file_path
                backup_dir = project_root / self.config["deployment"]["backup_dir"]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"{file_path_obj.name}.{timestamp}.bak"

                try:
                    shutil.copy2(file_path_obj, backup_path)
                    self.logger.info(f"Backed up {file_path_obj} to {backup_path}")
                except Exception as e:
                    self.logger.exception(
                        e, message=f"Failed to backup {file_path_obj}"
                    )
                    continue

            # Apply the patch
            success = self.patch_generator.apply_patch(patch, project_root)

            if success:
                # Store patch information with backup path for potential rollback
                patch_info = {
                    "patch_id": patch["patch_id"],
                    "file_path": file_path,
                    "backup_path": (
                        str(backup_path)
                        if self.config["patch_generation"]["backup_original_files"]
                        else None
                    ),
                    "timestamp": timestamp,
                }
                applied_patches.append(patch_info)
                self.logger.info(f"Successfully applied patch {patch['patch_id']}")
            else:
                self.logger.error(f"Failed to apply patch {patch['patch_id']}")

                # If rate limiter is enabled, place the file in cooldown if patch application failed
                if hasattr(self, "rate_limiter") and self.rate_limiter:
                    self.rate_limiter.place_file_in_cooldown(file_path)

        return applied_patches

    def run_tests(self, patches: Optional[List[Dict[str, Any]]] = None) -> bool:
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
            resource_limits = self.config["testing"]["graduated_testing"][
                "resource_limits"
            ]

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
                    resource_limits=resource_limits,
                )

                results.append(result)

                # Record metrics
                self.metrics_collector.record_test_metric(
                    patch_id,
                    result["valid"],
                    result["test_levels"]
                    .get(test_levels[-1], {})
                    .get("test_results", {})
                    .get("duration", 0),
                    other_data={"test_type": "graduated", "levels": test_levels},
                )

                # Generate regression tests if enabled
                if self.config["testing"]["regression"]["enabled"] and result["valid"]:
                    if hasattr(self, "regression_generator"):
                        self.logger.info(
                            f"Generating regression test for patch {patch_id}"
                        )

                        error_info = None
                        if "analysis_result" in patch:
                            error_info = patch["analysis_result"]

                        test_path = self.regression_generator.generate_regression_test(
                            patch, error_info
                        )

                        if test_path:
                            self.logger.info(
                                f"Generated regression test at {test_path}"
                            )

            # Check if all patches passed tests
            all_passed = all(result["valid"] for result in results)

            if all_passed:
                self.logger.info("All graduated tests passed")
            else:
                failed_patches = [
                    result["patch_id"] for result in results if not result["valid"]
                ]
                self.logger.error(
                    f"Graduated tests failed for patches: {', '.join(failed_patches)}"
                )

            return all_passed

        # Use simple test runner if not using advanced features
        try:
            # Run tests
            test_result = subprocess.run(
                test_command,
                shell=True,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=test_timeout,
            )

            # Check if tests passed
            success = test_result.returncode == 0

            # Record metrics if patches are provided
            if patches:
                for patch in patches:
                    patch_id = patch.get("patch_id", "unknown")

                    self.metrics_collector.record_test_metric(
                        patch_id,
                        success,
                        time.time(),  # We don't have accurate duration for subprocess
                        other_data={"test_type": "simple", "command": test_command},
                    )

            if success:
                self.logger.info("Tests passed")
                return True
            else:
                self.logger.error(f"Tests failed: {test_result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.logger.error(f"Tests timed out after {test_timeout} seconds")
            return False
        except Exception as e:
            self.logger.exception(e, message="Error running tests")
            return False

    def requires_approval(self, patches: List[Dict[str, Any]]) -> bool:
        """
        Determine if the patches require approval before deployment.

        Args:
            patches: List of patches to check

        Returns:
            True if approval is required, False otherwise
        """
        # If approval workflow is not enabled, no approval needed
        if not hasattr(self, "approval_required") or not self.approval_required:
            return False

        # Check if we're in production environment
        is_production = self.config["general"]["environment"] == "production"

        # If in production and approval required in production, require approval
        if is_production and self.config["deployment"].get("production", {}).get(
            "require_approval", False
        ):
            return True

        # Check if any patch contains critical fix types that require approval
        for patch in patches:
            fix_type = patch.get("fix_type", "unknown")
            if needs_approval(fix_type):
                return True

        return False

    def create_approval_request(
        self, patches: List[Dict[str, Any]], requester: str = "system"
    ) -> str:
        """
        Create an approval request for the patches.

        Args:
            patches: List of patches to deploy
            requester: Username of the requester

        Returns:
            str: Request ID
        """
        # Create a descriptive title
        if len(patches) == 1:
            patch = patches[0]
            title = f"Fix for {patch.get('error_type', 'unknown error')} in {patch.get('file_path', 'unknown file')}"
        else:
            title = f"Deployment of {len(patches)} fixes"

        # Create detailed description
        description = "The following fixes require approval before deployment:\n\n"
        for i, patch in enumerate(patches, 1):
            description += f"{i}. Fix for {patch.get('error_type', 'unknown error')} "
            description += f"in {patch.get('file_path', 'unknown file')}\n"
            description += f"   - Fix type: {patch.get('fix_type', 'unknown')}\n"
            description += f"   - Confidence: {patch.get('confidence', 'unknown')}\n"
            description += f"   - Description: {patch.get('description', 'No description provided')}\n\n"

        # Create request
        request = create_approval_request(
            request_type=ApprovalType.FIX_DEPLOYMENT,
            requester=requester,
            title=title,
            description=description,
            data={"patches": patches},
            required_approvers=self.config.get("security", {})
            .get("approval", {})
            .get("required_approvers", 1),
        )

        self.logger.info(
            f"Created approval request {request.request_id} for {len(patches)} patch(es)"
        )
        return request.request_id

    def check_approval_status(self, request_id: str) -> Tuple[bool, str]:
        """
        Check if an approval request has been approved.

        Args:
            request_id: Request ID to check

        Returns:
            Tuple[bool, str]: (is_approved, status_description)
        """
        # Get the request
        request = get_approval_manager().get_request(request_id)
        if not request:
            return False, "Approval request not found"

        # Check if expired
        if request.is_expired():
            return False, "Approval request has expired"

        # Check status
        if request.status == ApprovalStatus.APPROVED:
            return True, "Approved"
        elif request.status == ApprovalStatus.REJECTED:
            return False, "Rejected"
        elif request.status == ApprovalStatus.CANCELLED:
            return False, "Cancelled"
        else:
            return False, "Pending approval"

    def deploy_changes(
        self, patches: Optional[List[Dict[str, Any]]] = None, requester: str = "system"
    ) -> bool:
        """
        Deploy the changes by restarting the service.

        Args:
            patches: Optional list of patches being deployed
            requester: Username of the requester

        Returns:
            True if deployment succeeded, False otherwise
        """
        if not self.config["deployment"]["enabled"]:
            self.logger.info("Deployment is disabled, skipping service restart")
            return True

        # No patches to deploy
        if not patches:
            self.logger.info("No patches to deploy")
            return True

        # Check deployment rate limit if rate limiter is enabled
        if hasattr(self, "rate_limiter") and self.rate_limiter:
            if not self.rate_limiter.check_deployment_limit():
                self.logger.warning(
                    "Deployment rate limit exceeded, cannot deploy changes"
                )
                return False

        # Check if any patch requires human review via suggestion interface
        requires_human_review = any(
            patch.get("requires_review", False) for patch in patches
        )
        if requires_human_review:
            self.logger.info(
                "Deployment requires human review through the suggestion interface"
            )

            # Log that human review is required
            for patch in patches:
                if patch.get("requires_review", False):
                    error_id = patch.get("error_id")
                    self.logger.info(
                        f"Patch for error {error_id} requires human review"
                    )

                    # Log the review request if security is enabled
                    if self.security_enabled:
                        log_fix(
                            fix_id=patch.get("patch_id", "unknown"),
                            event_type="fix_human_review_requested",
                            user=requester,
                            details={"error_id": error_id},
                        )

            # Return false for now, since deployment is pending human review
            return False

        # Check if approval is required from the approval workflow
        if self.requires_approval(patches):
            self.logger.info("Deployment requires approval")

            # Create approval request
            request_id = self.create_approval_request(patches, requester)

            # Log the request
            if self.security_enabled:
                for patch in patches:
                    log_fix(
                        fix_id=patch.get("patch_id", "unknown"),
                        event_type="fix_approval_requested",
                        user=requester,
                        details={"request_id": request_id},
                    )

            # Return false for now, since deployment is pending approval
            return False

        # Begin deployment
        self.logger.info("Deploying changes")

        # Audit log the deployment if security is enabled
        if self.security_enabled:
            for patch in patches:
                log_fix(
                    fix_id=patch.get("patch_id", "unknown"),
                    event_type="fix_deployed",
                    user=requester,
                )

        start_time = time.time()

        # Restart the service
        if self.config["deployment"]["restart_service"]:
            self.logger.info("Restarting service")

            # Check if we should use canary deployment
            if (
                self.canary_enabled
                and self.config["general"]["environment"] == "production"
            ):
                self.logger.info("Using canary deployment for gradual rollout")

                # Get the service name from config
                service_name = self.config["service"]["name"]

                # Generate a deployment ID if multiple patches
                if len(patches) == 1:
                    deployment_id = patches[0].get("patch_id", "unknown")
                else:
                    deployment_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                # Start canary deployment
                if self.canary_deployment.start(service_name, deployment_id):
                    self.logger.info(
                        f"Started canary deployment with ID {deployment_id} "
                        f"at {self.canary_deployment.current_percentage}%"
                    )
                else:
                    self.logger.warning("Failed to start canary deployment")

                # Note: The actual rollout will be managed by the canary deployment manager
                # We'll continue with the service restart, but traffic redirection will be
                # handled separately based on the canary percentage

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

                # Log failure if security is enabled
                if self.security_enabled:
                    for patch in patches:
                        log_fix(
                            fix_id=patch.get("patch_id", "unknown"),
                            event_type="fix_deployment_failed",
                            user=requester,
                            status="failure",
                            details={"reason": "Service not healthy after restart"},
                        )

                return False

            # Record deployment metrics
            duration = time.time() - start_time
            for patch in patches:
                patch_id = patch.get("patch_id", "unknown")

                # Record deployment metrics
                self.metrics_collector.record_deployment_metric(
                    patch_id, success, duration, other_data={"restart": True}
                )

            # Start post-deployment monitoring if enabled
            if self.config["monitoring"]["post_deployment"]["enabled"]:
                health_url = self.config["service"]["health_check_url"]
                service_url = health_url.rsplit("/health", 1)[0]  # Extract base URL

                for patch in patches:
                    patch_id = patch.get("patch_id", "unknown")

                    # Start monitoring for this patch
                    self.logger.info(
                        f"Starting post-deployment monitoring for patch {patch_id}"
                    )
                    self.post_deployment_monitor.start_monitoring(service_url, patch_id)

                    # Set up anomaly detection
                    if hasattr(self, "anomaly_detector"):
                        self.logger.info(
                            f"Starting anomaly detection for patch {patch_id}"
                        )

                        # Start anomaly detection in a separate thread
                        threading.Thread(
                            target=self.anomaly_detector.monitor_for_anomalies,
                            args=(service_url, patch_id),
                            daemon=True,
                        ).start()

            # Log successful deployment if security is enabled
            if self.security_enabled:
                for patch in patches:
                    log_fix(
                        fix_id=patch.get("patch_id", "unknown"),
                        event_type="fix_deployment_succeeded",
                        user=requester,
                        details={"duration": duration},
                    )

            return success

        return True

    def deploy_approved_fix(self, request_id: str, approver: str) -> bool:
        """
        Deploy a fix that has been approved.

        Args:
            request_id: ID of the approved request
            approver: Username of the approver

        Returns:
            bool: True if deployment succeeded, False otherwise
        """
        # Get the approval request
        request = get_approval_manager().get_request(request_id)
        if not request:
            self.logger.error(f"Approval request {request_id} not found")
            return False

        # Check if request is approved
        if request.status != ApprovalStatus.APPROVED:
            self.logger.error(
                f"Approval request {request_id} is not approved (status: {request.status.value})"
            )
            return False

        # Extract patches from request data
        patches = request.data.get("patches", [])
        if not patches:
            self.logger.error(f"No patches found in approval request {request_id}")
            return False

        # Log the approval
        if self.security_enabled:
            for patch in patches:
                log_fix(
                    fix_id=patch.get("patch_id", "unknown"),
                    event_type="fix_approved",
                    user=approver,
                    details={"request_id": request_id},
                )

        # Deploy the changes
        self.logger.info(f"Deploying {len(patches)} approved patch(es)")
        return self.deploy_changes(patches, approver)

    def check_pending_approvals(self) -> List[Dict[str, Any]]:
        """
        Check for any pending approval requests.

        Returns:
            List[Dict[str, Any]]: List of pending approval requests
        """
        if not hasattr(self, "approval_required") or not self.approval_required:
            return []

        # Get pending requests of type FIX_DEPLOYMENT
        pending_requests = get_approval_manager().list_requests(
            status=ApprovalStatus.PENDING, request_type=ApprovalType.FIX_DEPLOYMENT
        )

        # Format for easier consumption
        result = []
        for request in pending_requests:
            result.append(
                {
                    "request_id": request.request_id,
                    "title": request.title,
                    "requester": request.requester,
                    "created_at": request.created_at.isoformat(),
                    "expires_at": (
                        request.expiry.isoformat() if request.expiry else None
                    ),
                    "required_approvers": request.required_approvers,
                    "current_approvals": len(request.approvals),
                    "patches": request.data.get("patches", []),
                }
            )

        return result

    def check_pending_suggestions(self) -> List[Dict[str, Any]]:
        """
        Check for any pending suggestions that need human review.

        Returns:
            List[Dict[str, Any]]: List of pending suggestions
        """
        # Check if suggestion interface is enabled
        if not hasattr(self, "suggestion_manager"):
            return []

        # Get all suggestions (grouped by error_id)
        all_suggestions = {}
        for error_id in self.suggestion_manager.suggestions.keys():
            suggestions = self.suggestion_manager.get_suggestions(error_id)

            # Filter only generated suggestions
            pending_suggestions = [
                s for s in suggestions if s.status == SuggestionStatus.GENERATED
            ]
            if pending_suggestions:
                all_suggestions[error_id] = pending_suggestions

        # Format for easier consumption
        result = []
        for error_id, suggestions in all_suggestions.items():
            # Use the first suggestion to get common information
            first = suggestions[0]

            result.append(
                {
                    "error_id": error_id,
                    "file_path": first.file_path,
                    "suggestion_count": len(suggestions),
                    "best_confidence": max(s.confidence for s in suggestions),
                    "created_at": first.created_at,
                    "suggestions": suggestions,  # Include the actual suggestion objects
                }
            )

        return result

    def deploy_approved_suggestion(
        self, error_id: str, suggestion_id: str, reviewer: str
    ) -> bool:
        """
        Deploy a suggestion that has been approved through the suggestion interface.

        Args:
            error_id: ID of the error
            suggestion_id: ID of the approved suggestion
            reviewer: Username of the reviewer

        Returns:
            bool: True if deployment succeeded, False otherwise
        """
        # Get the suggestion
        suggestion = self.suggestion_manager.get_suggestion(suggestion_id)
        if not suggestion:
            self.logger.error(
                f"Suggestion {suggestion_id} not found for error {error_id}"
            )
            return False

        # Check if the suggestion is already approved
        if suggestion.status != SuggestionStatus.APPROVED:
            self.logger.error(
                f"Suggestion {suggestion_id} is not approved (status: {suggestion.status.value})"
            )
            return False

        # Create a patch from the suggestion
        patch = {
            "patch_id": suggestion.suggestion_id,
            "file_path": suggestion.file_path,
            "fix_type": suggestion.fix_type,
            "confidence": suggestion.confidence,
            "description": suggestion.description,
            "original_code": suggestion.original_code,
            "new_code": suggestion.suggested_code,
        }

        # Apply the patch
        self.logger.info(f"Applying approved suggestion {suggestion_id}")
        applied_patches = self.apply_patches([patch])

        if not applied_patches:
            self.logger.warning("Failed to apply approved suggestion")

            # Update suggestion status to failed
            self.suggestion_manager.mark_failed(suggestion_id)

            # Log the failure if security is enabled
            if self.security_enabled:
                log_fix(
                    fix_id=suggestion_id,
                    event_type="suggestion_deployment_failed",
                    user=reviewer,
                    status="failure",
                    details={"error_id": error_id},
                )

            return False

        # Run tests
        tests_passed = self.run_tests([patch])

        if not tests_passed:
            self.logger.warning(
                "Tests failed for approved suggestion, rolling back changes"
            )

            # Update suggestion status to failed
            self.suggestion_manager.mark_failed(suggestion_id)

            # Roll back the changes
            rollback_success = self._rollback_changes(applied_patches)
            if not rollback_success:
                self.logger.error(
                    "Rollback failed, system may be in an inconsistent state"
                )

            # Log the failure if security is enabled
            if self.security_enabled:
                log_fix(
                    fix_id=suggestion_id,
                    event_type="suggestion_test_failed",
                    user=reviewer,
                    status="failure",
                    details={"error_id": error_id},
                )

            return False

        # Deploy the changes
        deployment_succeeded = self.deploy_changes([patch], reviewer)

        if not deployment_succeeded:
            self.logger.warning("Deployment failed for approved suggestion")

            # Update suggestion status to failed
            self.suggestion_manager.mark_failed(suggestion_id)

            # Log the failure if security is enabled
            if self.security_enabled:
                log_fix(
                    fix_id=suggestion_id,
                    event_type="suggestion_deployment_failed",
                    user=reviewer,
                    status="failure",
                    details={"error_id": error_id},
                )

            return False

        # Mark the suggestion as deployed
        self.suggestion_manager.mark_deployed(suggestion_id)

        # Log the success if security is enabled
        if self.security_enabled:
            log_fix(
                fix_id=suggestion_id,
                event_type="suggestion_deployed",
                user=reviewer,
                status="success",
                details={"error_id": error_id},
            )

        self.logger.info(f"Successfully deployed approved suggestion {suggestion_id}")
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

        # Check healing cycle rate limit if rate limiter is enabled
        if hasattr(self, "rate_limiter") and self.rate_limiter:
            if not self.rate_limiter.check_healing_cycle_limit():
                self.logger.warning("Healing cycle rate limit exceeded, skipping cycle")
                return False

        # Start healing session and get session ID for audit tracking
        session_id = start_healing_session(
            trigger="scheduled",
            details={"environment": self.config["general"]["environment"]},
        )
        self.logger.info(f"Started healing session {session_id}")

        try:
            # Check for and clean up expired approval requests
            if hasattr(self, "approval_required") and self.approval_required:
                expired_count = get_approval_manager().cleanup_expired_requests()
                if expired_count > 0:
                    self.logger.info(
                        f"Cleaned up {expired_count} expired approval requests"
                    )

            # Monitor for errors
            errors = self.monitor_for_errors()

            if not errors:
                self.logger.info("No errors found, nothing to fix")
                end_healing_session(session_id, "completed", {"reason": "no_errors"})
                return True

            # Log error detection in audit trail
            for i, error in enumerate(errors):
                error_id = error.get("error_id", f"err_{session_id}_{i}")
                error_type = error.get("error_type", "unknown")
                source = error.get("source", "log_monitoring")

                # Add error_id to the error if not present
                if "error_id" not in error:
                    error["error_id"] = error_id

                if hasattr(self, "healing_auditor"):
                    self.healing_auditor.log_error_detection(
                        session_id=session_id,
                        error_id=error_id,
                        error_type=error_type,
                        source=source,
                        details=error,
                    )

            # Analyze errors
            start_time = time.time()
            analysis_results = self.analyze_errors(errors)
            analysis_duration_ms = (time.time() - start_time) * 1000

            if not analysis_results:
                self.logger.info("No analysis results, nothing to fix")
                end_healing_session(
                    session_id, "completed", {"reason": "no_analysis_results"}
                )
                return True

            # Log error analysis in audit trail
            for result in analysis_results:
                error_id = result.get("error_id", "unknown")
                analysis_type = "rule_based"
                if self.config["analysis"]["ai_based"]["enabled"]:
                    analysis_type = result.get("analysis_type", "hybrid")

                root_cause = result.get("root_cause", "unknown")
                confidence = result.get("confidence", 0.0)

                if hasattr(self, "healing_auditor"):
                    self.healing_auditor.log_error_analysis(
                        session_id=session_id,
                        error_id=error_id,
                        analysis_type=analysis_type,
                        root_cause=root_cause,
                        confidence=confidence,
                        duration_ms=analysis_duration_ms / len(analysis_results),
                        details=result,
                    )

            # Generate patches
            patches = self.generate_patches(analysis_results)

            if not patches:
                self.logger.info("No patches generated, nothing to fix")
                end_healing_session(session_id, "completed", {"reason": "no_patches"})
                return True

            # Log patch generation in audit trail
            for patch in patches:
                patch_id = patch.get("patch_id", "unknown")
                error_id = patch.get("error_id", "unknown")
                fix_type = patch.get("fix_type", "unknown")
                file_path = patch.get("file_path", "unknown")
                template_name = patch.get("template_name")
                confidence = patch.get("confidence", 0.0)

                if hasattr(self, "healing_auditor"):
                    self.healing_auditor.log_patch_generation(
                        session_id=session_id,
                        error_id=error_id,
                        fix_id=patch_id,
                        fix_type=fix_type,
                        file_path=file_path,
                        template_name=template_name,
                        confidence=confidence,
                        details=patch,
                    )

            # Apply patches
            applied_patches = self.apply_patches(patches)

            if not applied_patches:
                self.logger.warning("No patches applied, self-healing failed")
                end_healing_session(
                    session_id, "failed", {"reason": "patch_application_failed"}
                )
                return False

            # Log patch application in audit trail
            for patch_info in applied_patches:
                patch_id = patch_info.get("patch_id", "unknown")
                file_path = patch_info.get("file_path", "unknown")

                if hasattr(self, "healing_auditor"):
                    self.healing_auditor.log_patch_application(
                        session_id=session_id,
                        fix_id=patch_id,
                        file_path=file_path,
                        status="success",
                        details=patch_info,
                    )

            # Store the applied patches in a session file for potential rollback
            self._store_session_patches(applied_patches)

            # Run tests
            start_time = time.time()
            tests_passed = self.run_tests(patches)
            test_duration_ms = (time.time() - start_time) * 1000

            # Log test execution in audit trail
            for patch in patches:
                patch_id = patch.get("patch_id", "unknown")

                if hasattr(self, "healing_auditor"):
                    self.healing_auditor.log_test_execution(
                        session_id=session_id,
                        fix_id=patch_id,
                        test_type="system",
                        status="success" if tests_passed else "failure",
                        duration_ms=test_duration_ms,
                        details={"patches_tested": len(patches)},
                    )

            if not tests_passed:
                self.logger.warning("Tests failed, rolling back changes")

                # Check if auto-rollback is enabled in the config
                if self.config.get("rollback", {}).get(
                    "enabled", True
                ) and self.config.get("rollback", {}).get(
                    "auto_rollback_on_failure", True
                ):
                    rollback_success = self._rollback_changes(applied_patches)

                    # Log rollback in audit trail
                    for patch_info in applied_patches:
                        patch_id = patch_info.get("patch_id", "unknown")

                        if hasattr(self, "healing_auditor"):
                            self.healing_auditor.log_rollback(
                                session_id=session_id,
                                fix_id=patch_id,
                                status="success" if rollback_success else "failure",
                                reason="test_failure",
                                details={"auto_rollback": True},
                            )

                    if not rollback_success:
                        self.logger.error(
                            "Rollback failed, system may be in an inconsistent state"
                        )
                else:
                    self.logger.info(
                        "Auto-rollback disabled in config, skipping rollback"
                    )

                # Record fix failure in the feedback loop
                for patch in patches:
                    patch_id = patch.get("patch_id", "unknown")
                    bug_id = patch.get("bug_id", "")
                    template_name = patch.get("template_name", "unknown")

                    # Record in success tracker
                    self.success_tracker.record_fix(patch_id, bug_id, False)

                    # Record in feedback loop
                    self.feedback_loop.record_fix_result(
                        patch_id, template_name, bug_id, False
                    )

                end_healing_session(session_id, "failed", {"reason": "tests_failed"})
                return False

            # Deploy changes
            environment = self.config["general"]["environment"]
            deployment_succeeded = self.deploy_changes(patches)

            # Log deployment in audit trail
            for patch in patches:
                patch_id = patch.get("patch_id", "unknown")

                if hasattr(self, "healing_auditor"):
                    deployment_type = "standard"
                    if self.canary_enabled and environment == "production":
                        deployment_type = "canary"

                    self.healing_auditor.log_deployment(
                        session_id=session_id,
                        fix_id=patch_id,
                        environment=environment,
                        status="success" if deployment_succeeded else "failure",
                        deployment_type=deployment_type,
                        details={"approval_required": self.requires_approval(patches)},
                    )

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
                    self.feedback_loop.record_fix_result(
                        patch_id, template_name, bug_id, False
                    )

                end_healing_session(
                    session_id, "failed", {"reason": "deployment_failed"}
                )
                return False

            # Record successful fixes
            for patch in patches:
                patch_id = patch.get("patch_id", "unknown")
                bug_id = patch.get("bug_id", "")
                template_name = patch.get("template_name", "unknown")

                # Record in success tracker
                self.success_tracker.record_fix(patch_id, bug_id, True)

                # Record in feedback loop
                self.feedback_loop.record_fix_result(
                    patch_id, template_name, bug_id, True
                )

                # Record fix metrics
                service_url = self.config["service"]["health_check_url"].rsplit(
                    "/health", 1
                )[0]
                health_data = self.check_health(service_url)

                self.metrics_collector.record_fix_metric(
                    patch_id,
                    bug_id,
                    True,
                    response_time=health_data.get("response_time"),
                    error_rate=health_data.get("error_rate"),
                    other_data={
                        "template_name": template_name,
                        "deployment_success": deployment_succeeded,
                        "session_id": session_id,
                    },
                )

            # Generate insights from feedback loop
            recommendations = self.feedback_loop.generate_recommendations()
            if recommendations:
                self.logger.info(
                    f"Generated {len(recommendations)} recommendations for improvement"
                )
                for rec in recommendations:
                    self.logger.info(
                        f"Recommendation: {rec['message']}",
                        suggestion=rec.get("suggestion", ""),
                    )

            self.logger.info("Self-healing cycle completed successfully")
            end_healing_session(
                session_id,
                "completed",
                {
                    "errors_processed": len(errors),
                    "fixes_applied": len(applied_patches),
                    "success": True,
                },
            )

            return True

        except Exception as e:
            # Log exception and end session
            self.logger.exception(e, message="Error in self-healing cycle")
            end_healing_session(
                session_id, "failed", {"reason": "exception", "error": str(e)}
            )
            return False

    def check_health(self, service_url: str) -> Dict[str, Any]:
        """
        Check service health and collect basic metrics.

        Args:
            service_url: Base URL of the service

        Returns:
            Health data with metrics
        """
        metrics: Dict[str, Any] = {"timestamp": time.time()}

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
            "patches": applied_patches,
        }

        try:
            with open(session_file, "w") as f:
                json.dump(session_data, f, indent=2)
            self.logger.info(f"Stored session information in {session_file}")

            # Clean up old sessions if needed
            self._cleanup_old_sessions()
        except Exception as e:
            self.logger.exception(e, message="Failed to store session information")

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
        session_files = sorted(
            sessions_dir.glob("session_*.json"), key=lambda p: p.stat().st_mtime
        )

        # If we have more sessions than the max, delete the oldest ones
        if len(session_files) > max_sessions:
            sessions_to_delete = session_files[
                :-max_sessions
            ]  # Keep only the newest max_sessions
            for session_file in sessions_to_delete:
                try:
                    session_file.unlink()  # Delete the file
                    self.logger.debug(f"Deleted old session file: {session_file.name}")
                except Exception as e:
                    self.logger.exception(
                        e,
                        message=f"Failed to delete old session file: {session_file.name}",
                    )

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
        session_files = sorted(
            sessions_dir.glob("session_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not session_files:
            return None

        # Read the latest session file
        try:
            with open(session_files[0], "r") as f:
                session_data = json.load(f)
            return dict(session_data)
        except Exception as e:
            self.logger.exception(
                e, message=f"Failed to read session information from {session_files[0]}"
            )
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
                self.logger.warning(
                    f"No backup found for patch {patch_id}, cannot roll back"
                )
                success = False
                continue

            if not file_path:
                self.logger.warning(
                    f"No file path found for patch {patch_id}, cannot roll back"
                )
                success = False
                continue

            try:
                # Restore the original file from backup
                target_path = project_root / file_path
                shutil.copy2(backup_path, target_path)
                self.logger.info(f"Restored {file_path} from backup {backup_path}")
            except Exception as e:
                self.logger.exception(
                    e, message=f"Failed to roll back patch {patch_id}"
                )
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
        self.logger.info(
            f"Rolling back session {session_data['session_id']} with {len(patches)} patches"
        )
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
        self.apply_patches(patches)

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
                    self.feedback_loop.record_fix_result(
                        patch_id, template_name, bug_id, True
                    )

                # Generate recommendations
                recommendations = self.feedback_loop.generate_recommendations()
                if recommendations:
                    self.logger.info(
                        f"Generated {len(recommendations)} recommendations for improvement"
                    )
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
        self.logger.info(
            f"Starting orchestrator in {'demo' if demo_mode else 'normal'} mode"
        )

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
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--demo", "-d", action="store_true", help="Run in demonstration mode"
    )
    parser.add_argument(
        "--rollback", "-r", action="store_true", help="Roll back the latest session"
    )

    # Approval-related arguments
    approval_group = parser.add_argument_group("Approval Workflow")
    approval_group.add_argument(
        "--list-approvals",
        action="store_true",
        help="List all pending approval requests",
    )
    approval_group.add_argument(
        "--approve",
        type=str,
        metavar="REQUEST_ID",
        help="Approve a pending fix with the given request ID",
    )
    approval_group.add_argument(
        "--reject",
        type=str,
        metavar="REQUEST_ID",
        help="Reject a pending fix with the given request ID",
    )
    approval_group.add_argument(
        "--comment",
        type=str,
        metavar="COMMENT",
        help="Comment to include with approve/reject action",
    )
    approval_group.add_argument(
        "--user",
        type=str,
        default="admin",
        help="Username to use for approval/rejection actions",
    )

    # Deployment-related arguments
    deployment_group = parser.add_argument_group("Deployment")

    # Canary deployment related arguments
    canary_group = parser.add_argument_group("Canary Deployment")
    canary_group.add_argument(
        "--canary-status",
        action="store_true",
        help="Show status of active canary deployments",
    )
    canary_group.add_argument(
        "--canary-promote",
        type=str,
        metavar="DEPLOYMENT_ID",
        help="Promote a canary deployment to the next percentage",
    )
    canary_group.add_argument(
        "--canary-complete",
        type=str,
        metavar="DEPLOYMENT_ID",
        help="Complete a canary deployment (set to 100%%)",
    )
    canary_group.add_argument(
        "--canary-rollback",
        type=str,
        metavar="DEPLOYMENT_ID",
        help="Roll back a canary deployment",
    )

    # Blue-Green deployment related arguments
    blue_green_group = parser.add_argument_group("Blue-Green Deployment")
    blue_green_group.add_argument(
        "--blue-green-status",
        action="store_true",
        help="Show status of active blue-green deployments",
    )
    blue_green_group.add_argument(
        "--blue-green-deploy",
        type=str,
        metavar="DEPLOYMENT_ID",
        help="Deploy a service to the inactive environment",
    )
    blue_green_group.add_argument(
        "--blue-green-test",
        type=str,
        metavar="DEPLOYMENT_ID",
        help="Test the inactive environment",
    )
    blue_green_group.add_argument(
        "--blue-green-switch",
        type=str,
        metavar="DEPLOYMENT_ID",
        help="Switch traffic to the inactive environment",
    )
    blue_green_group.add_argument(
        "--blue-green-rollback",
        type=str,
        metavar="DEPLOYMENT_ID",
        help="Roll back to the previous active environment",
    )

    # Kubernetes deployment related arguments
    k8s_group = parser.add_argument_group("Kubernetes Deployment")
    k8s_group.add_argument(
        "--k8s-deploy", action="store_true", help="Deploy to Kubernetes"
    )
    k8s_group.add_argument(
        "--k8s-undeploy", action="store_true", help="Undeploy from Kubernetes"
    )
    k8s_group.add_argument(
        "--k8s-status", action="store_true", help="Get Kubernetes deployment status"
    )

    # Cloud provider related arguments
    cloud_group = parser.add_argument_group("Cloud Provider")
    cloud_group.add_argument(
        "--cloud-deploy", action="store_true", help="Deploy to cloud provider"
    )
    cloud_group.add_argument(
        "--cloud-undeploy", action="store_true", help="Undeploy from cloud provider"
    )
    cloud_group.add_argument(
        "--cloud-status", action="store_true", help="Get cloud deployment status"
    )
    cloud_group.add_argument(
        "--cloud-logs", action="store_true", help="Get cloud service logs"
    )

    # Common deployment arguments
    deployment_group.add_argument(
        "--service", type=str, help="Service name for deployment commands"
    )
    deployment_group.add_argument(
        "--deployment-id", type=str, help="Deployment ID for deployment commands"
    )
    deployment_group.add_argument(
        "--image", type=str, help="Docker image for deployment commands"
    )
    deployment_group.add_argument(
        "--namespace", type=str, help="Kubernetes namespace for deployment commands"
    )

    # Suggestion interface related arguments
    suggestion_group = parser.add_argument_group("Fix Suggestion Interface")
    suggestion_group.add_argument(
        "--list-suggestions",
        action="store_true",
        help="List all pending fix suggestions",
    )
    suggestion_group.add_argument(
        "--deploy-suggestion",
        type=str,
        metavar="SUGGESTION_ID",
        help="Deploy an approved suggestion with the given ID",
    )
    suggestion_group.add_argument(
        "--error-id", type=str, help="Error ID for suggestion commands"
    )

    # Audit related arguments
    audit_group = parser.add_argument_group("Audit and Reporting")
    audit_group.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate an audit report of healing activities",
    )
    audit_group.add_argument(
        "--report-period",
        type=str,
        choices=["hour", "day", "week", "month"],
        default="day",
        help="Time period for the report",
    )
    audit_group.add_argument(
        "--report-format",
        type=str,
        choices=["text", "json", "csv"],
        default="text",
        help="Format for the report",
    )
    audit_group.add_argument(
        "--report-type",
        type=str,
        choices=["summary", "healing", "security", "full"],
        default="summary",
        help="Type of report to generate",
    )
    audit_group.add_argument(
        "--report-output",
        type=str,
        help="Output file for the report (if not specified, prints to stdout)",
    )

    # Rate limiting related arguments
    rate_limit_group = parser.add_argument_group("Rate Limiting and Throttling")
    rate_limit_group.add_argument(
        "--rate-limit-stats",
        action="store_true",
        help="Show current rate limit usage statistics",
    )
    rate_limit_group.add_argument(
        "--reset-rate-limits", action="store_true", help="Reset all rate limit counters"
    )
    rate_limit_group.add_argument(
        "--cooldown-file",
        type=str,
        metavar="FILE_PATH",
        help="Place a file in cooldown to prevent modifications",
    )
    rate_limit_group.add_argument(
        "--cooldown-duration",
        type=int,
        default=3600,
        help="Duration in seconds for file cooldown (default: 3600)",
    )

    args = parser.parse_args()

    # Determine the config path
    if os.path.isabs(args.config):
        config_path = Path(args.config)
    else:
        # Relative to the orchestrator directory
        config_path = Path(__file__).parent / args.config

    # Initialize the orchestrator
    orchestrator = Orchestrator(config_path, log_level=args.log_level)

    # Handle approval-related commands
    if args.list_approvals:
        # List pending approval requests
        pending_approvals = orchestrator.check_pending_approvals()
        if not pending_approvals:
            print("No pending approval requests.")
        else:
            print(f"Found {len(pending_approvals)} pending approval requests:")
            for i, req in enumerate(pending_approvals, 1):
                print(f"\n{i}. Request: {req['request_id']}")
                print(f"   Title: {req['title']}")
                print(f"   Requester: {req['requester']}")
                print(f"   Created: {req['created_at']}")
                print(f"   Expires: {req['expires_at'] or 'Never'}")
                print(
                    f"   Approvals: {req['current_approvals']}/{req['required_approvers']}"
                )
                print(f"   Patches: {len(req['patches'])}")
        exit(0)

    elif args.approve:
        # Approve a fix
        if not orchestrator.security_enabled or not orchestrator.approval_required:
            print("Error: Approval workflow is not enabled in the configuration.")
            exit(1)

        try:
            # First check if the request exists and is pending
            is_approved, status = orchestrator.check_approval_status(args.approve)
            if is_approved:
                print(f"Request {args.approve} is already approved.")
                # Deploy the approved fix
                if orchestrator.deploy_approved_fix(args.approve, args.user):
                    print("Successfully deployed the approved fix.")
                    exit(0)
                else:
                    print("Failed to deploy the approved fix.")
                    exit(1)
            elif status != "Pending approval":
                print(f"Cannot approve request {args.approve}: {status}")
                exit(1)

            # Approve the request
            approval_mgr = get_approval_manager()
            approval_mgr.approve_request(args.approve, args.user, args.comment)
            print(f"Successfully approved request {args.approve}.")

            # Deploy the approved fix
            if orchestrator.deploy_approved_fix(args.approve, args.user):
                print("Successfully deployed the approved fix.")
                exit(0)
            else:
                print("Failed to deploy the approved fix.")
                exit(1)

        except ApprovalError as e:
            print(f"Error: {str(e)}")
            exit(1)

    elif args.reject:
        # Reject a fix
        if not orchestrator.security_enabled or not orchestrator.approval_required:
            print("Error: Approval workflow is not enabled in the configuration.")
            exit(1)

        try:
            # Check if the request exists and is pending
            is_approved, status = orchestrator.check_approval_status(args.reject)
            if status != "Pending approval":
                print(f"Cannot reject request {args.reject}: {status}")
                exit(1)

            # Reject the request
            approval_mgr = get_approval_manager()
            approval_mgr.reject_request(args.reject, args.user, args.comment)
            print(f"Successfully rejected request {args.reject}.")

            # Log the rejection
            if orchestrator.security_enabled:
                # Get the patches from the request
                request = approval_mgr.get_request(args.reject)
                if request and request.data.get("patches"):
                    for patch in request.data["patches"]:
                        log_fix(
                            fix_id=patch.get("patch_id", "unknown"),
                            event_type="fix_rejected",
                            user=args.user,
                            status="failure",
                            details={"request_id": args.reject, "reason": args.comment},
                        )
            exit(0)

        except ApprovalError as e:
            print(f"Error: {str(e)}")
            exit(1)

    # Handle canary deployment commands
    elif args.list_suggestions:
        # List pending fix suggestions
        suggestion_config = orchestrator.config.get("suggestion", {})

        if not suggestion_config.get("enabled", False):
            print(
                "Error: Fix suggestion interface is not enabled in the configuration."
            )
            exit(1)

        pending_suggestions = orchestrator.check_pending_suggestions()
        if not pending_suggestions:
            print("No pending fix suggestions found.")
        else:
            print(
                f"Found {len(pending_suggestions)} errors with pending fix suggestions:"
            )
            for i, suggestion in enumerate(pending_suggestions, 1):
                print(f"\n{i}. Error ID: {suggestion['error_id']}")
                print(f"   File: {suggestion['file_path']}")
                print(f"   Suggestions: {suggestion['suggestion_count']}")
                print(f"   Best confidence: {suggestion['best_confidence'] * 100:.1f}%")
                print(f"   Created: {suggestion['created_at']}")
        exit(0)

    elif args.deploy_suggestion:
        # Deploy an approved suggestion
        suggestion_config = orchestrator.config.get("suggestion", {})

        if not suggestion_config.get("enabled", False):
            print(
                "Error: Fix suggestion interface is not enabled in the configuration."
            )
            exit(1)

        if not args.error_id:
            print("Error: --error-id is required when deploying a suggestion.")
            exit(1)

        # Deploy the suggestion
        if orchestrator.deploy_approved_suggestion(
            args.error_id, args.deploy_suggestion, args.user
        ):
            print(f"Successfully deployed suggestion {args.deploy_suggestion}.")
            exit(0)
        else:
            print(f"Failed to deploy suggestion {args.deploy_suggestion}.")
            exit(1)

    elif (
        args.canary_status
        or args.canary_promote
        or args.canary_complete
        or args.canary_rollback
    ):
        # Check if canary deployment is enabled
        if not orchestrator.canary_enabled:
            print("Error: Canary deployment is not enabled in the configuration.")
            exit(1)

        canary_deployment = orchestrator.canary_deployment
        service_name = args.service or orchestrator.config["service"]["name"]

        if args.canary_status:
            # Show status of active canary deployments
            canary_status: Dict[str, Any] = canary_deployment.get_status()

            if canary_status["status"] is None:
                print("No active canary deployments.")
            else:
                print(
                    f"Canary Deployment Status for {canary_status['service_name']} ({canary_status['fix_id']}):"
                )
                print(f"  Status: {canary_status['status']}")
                print(f"  Current percentage: {canary_status['current_percentage']}%")
                print(f"  Start time: {canary_status['start_time']}")
                if canary_status["completion_time"]:
                    print(f"  Completion time: {canary_status['completion_time']}")
                else:
                    print(
                        f"  Elapsed time: {canary_status['elapsed_time']:.2f} seconds"
                    )
                print("\nMetrics:")
                print(f"  Error rate: {canary_status['metrics']['error_rate']:.3f}")
                print(f"  Success rate: {canary_status['metrics']['success_rate']:.3f}")
                print(
                    f"  Response time: {canary_status['metrics']['response_time']:.3f} ms"
                )
            exit(0)

        elif args.canary_promote:
            # Promote a canary deployment
            if (
                canary_deployment.service_name != service_name
                or canary_deployment.fix_id != args.canary_promote
            ):
                print(
                    f"Loading canary deployment {args.canary_promote} for service {service_name}..."
                )
                canary_deployment._load_state(service_name, args.canary_promote)

            if canary_deployment.status != CanaryStatus.IN_PROGRESS:
                print(
                    f"Cannot promote canary: status is {canary_deployment.status.value}"
                )
                exit(1)

            if canary_deployment.promote():
                print(
                    f"Successfully promoted canary deployment to {canary_deployment.current_percentage}%"
                )
                exit(0)
            else:
                print("Failed to promote canary deployment")
                exit(1)

        elif args.canary_complete:
            # Complete a canary deployment
            if (
                canary_deployment.service_name != service_name
                or canary_deployment.fix_id != args.canary_complete
            ):
                print(
                    f"Loading canary deployment {args.canary_complete} for service {service_name}..."
                )
                canary_deployment._load_state(service_name, args.canary_complete)

            if canary_deployment.status not in [
                CanaryStatus.IN_PROGRESS,
                CanaryStatus.PAUSED,
            ]:
                print(
                    f"Cannot complete canary: status is {canary_deployment.status.value}"
                )
                exit(1)

            if canary_deployment.complete():
                print("Successfully completed canary deployment")
                if canary_deployment.completion_time and canary_deployment.start_time:
                    print(
                        f"Time taken: {(canary_deployment.completion_time - canary_deployment.start_time).total_seconds():.2f} seconds"
                    )
                exit(0)
            else:
                print("Failed to complete canary deployment")
                exit(1)

        elif args.canary_rollback:
            # Roll back a canary deployment
            if (
                canary_deployment.service_name != service_name
                or canary_deployment.fix_id != args.canary_rollback
            ):
                print(
                    f"Loading canary deployment {args.canary_rollback} for service {service_name}..."
                )
                canary_deployment._load_state(service_name, args.canary_rollback)

            if canary_deployment.status not in [
                CanaryStatus.IN_PROGRESS,
                CanaryStatus.PAUSED,
            ]:
                print(
                    f"Cannot roll back canary: status is {canary_deployment.status.value}"
                )
                exit(1)

            if canary_deployment.rollback():
                print(
                    f"Successfully rolled back canary deployment from {canary_deployment.current_percentage}%"
                )
                exit(0)
            else:
                print("Failed to roll back canary deployment")
                exit(1)

    elif args.rollback:
        # Roll back the latest session
        success = orchestrator.rollback_latest_session()
        exit(0 if success else 1)
    elif args.generate_report:
        # Import the audit report generator
        from modules.monitoring.audit_monitor import generate_activity_report

        # Generate the report
        try:
            # Generate report
            report = generate_activity_report(args.report_period)

            # Format the report based on report type and format
            if args.report_format == "json":
                import json

                if args.report_type == "summary":
                    output = json.dumps(
                        {
                            "summary": {
                                "time_period": report["time_period"],
                                "start_time": report["start_time"],
                                "end_time": report["end_time"],
                                "total_events": report["total_events"],
                                "event_counts": report["event_counts"],
                                "healing_activities": report["healing_activities"],
                            }
                        },
                        indent=2,
                    )
                elif args.report_type == "healing":
                    output = json.dumps(
                        {"healing_activities": report["healing_activities"]}, indent=2
                    )
                elif args.report_type == "security":
                    output = json.dumps(
                        {
                            "security_events": [
                                e
                                for e in report.get("error_events", [])
                                if "security" in e.get("event_type", "")
                            ]
                        },
                        indent=2,
                    )
                else:  # full
                    output = json.dumps(report, indent=2)
            elif args.report_format == "csv":
                if not args.report_output:
                    print("Error: --report-output is required for CSV format")
                    exit(1)

                import csv

                if args.report_type == "summary":
                    with open(args.report_output, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Event Type", "Count"])
                        for event_type, count in report["event_counts"].items():
                            writer.writerow([event_type, count])
                elif args.report_type == "healing":
                    with open(args.report_output, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Activity", "Count"])
                        for activity, count in report["healing_activities"].items():
                            writer.writerow([activity, count])
                elif args.report_type == "security":
                    with open(args.report_output, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Timestamp", "Event Type", "User", "Details"])
                        for event in report.get("error_events", []):
                            if "security" in event.get("event_type", ""):
                                writer.writerow(
                                    [
                                        event.get("timestamp", ""),
                                        event.get("event_type", ""),
                                        event.get("user", ""),
                                        str(event.get("details", {})),
                                    ]
                                )
                else:  # full
                    with open(args.report_output, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Event Type", "Count"])
                        for event_type, count in report["event_counts"].items():
                            writer.writerow([event_type, count])

                        writer.writerow([])
                        writer.writerow(["Activity", "Count"])
                        for activity, count in report["healing_activities"].items():
                            writer.writerow([activity, count])

                        writer.writerow([])
                        writer.writerow(["Timestamp", "Event Type", "User", "Details"])
                        for event in report.get("error_events", []):
                            writer.writerow(
                                [
                                    event.get("timestamp", ""),
                                    event.get("event_type", ""),
                                    event.get("user", ""),
                                    str(event.get("details", {})),
                                ]
                            )

                print(f"Report saved to {args.report_output}")
                exit(0)
            else:  # text
                from modules.monitoring.audit_report import (
                    format_table,
                    get_summary_report,
                )

                if args.report_type == "summary":
                    output = get_summary_report(report)
                elif args.report_type == "healing":
                    healing = report["healing_activities"]
                    healing_table = [
                        ["Errors Detected", healing.get("errors_detected", 0)],
                        ["Fixes Generated", healing.get("fixes_generated", 0)],
                        ["Fixes Deployed", healing.get("fixes_deployed", 0)],
                        ["Fixes Approved", healing.get("fixes_approved", 0)],
                        ["Fixes Rejected", healing.get("fixes_rejected", 0)],
                        [
                            "Success Rate",
                            f"{healing.get('success_rate', 0) * 100:.1f}%",
                        ],
                    ]

                    output = f"Healing Activities Report ({report['time_period']})\n"
                    output += (
                        f"From: {report['start_time']} to {report['end_time']}\n\n"
                    )
                    output += format_table(healing_table, ["Activity", "Count"])
                elif args.report_type == "security":
                    security_events = [
                        e
                        for e in report.get("error_events", [])
                        if "security" in e.get("event_type", "")
                    ]

                    output = f"Security Events Report ({report['time_period']})\n"
                    output += (
                        f"From: {report['start_time']} to {report['end_time']}\n\n"
                    )

                    if security_events:
                        security_table = []
                        for event in security_events:
                            timestamp = event.get("timestamp", "")
                            event_type = event.get("event_type", "")
                            user = event.get("user", "")
                            details_str = str(event.get("details", {}))
                            if len(details_str) > 50:
                                details_str = details_str[:47] + "..."
                            security_table.append(
                                [timestamp, event_type, user, details_str]
                            )

                        output += format_table(
                            security_table,
                            ["Timestamp", "Event Type", "User", "Details"],
                        )
                    else:
                        output += "No security events recorded."
                else:  # full
                    from modules.monitoring.audit_report import get_user_activity_report

                    summary = get_summary_report(report)
                    user_activity = get_user_activity_report(report)

                    output = summary + "\n\n" + user_activity

            # Output the report
            if args.report_output:
                with open(args.report_output, "w") as f:
                    f.write(output)
                print(f"Report saved to {args.report_output}")
            else:
                print(output)

            exit(0)
        except Exception as e:
            print(f"Error generating report: {e}")
            exit(1)
    elif args.rate_limit_stats:
        # Show rate limit usage statistics
        if not hasattr(orchestrator, "rate_limiter") or not orchestrator.rate_limiter:
            print("Rate limiting is not enabled in the configuration.")
            exit(1)

        # Get rate limit stats
        stats = orchestrator.rate_limiter.get_usage_stats()

        # Format and display stats
        print("Rate Limit Usage Statistics:")
        print("============================")

        # Global actions
        print("\nGlobal Actions:")
        for action_type in ["healing_cycle", "patch_application", "deployment"]:
            action_stats = stats[action_type]
            print(f"  {action_type.replace('_', ' ').title()}:")
            print(
                f"    Count: {action_stats['count']} of {action_stats['limit']} "
                f"({action_stats['percent_used']:.1f}% used)"
            )
            print(
                f"    Window: {action_stats['window']} seconds "
                f"({int(action_stats['window_remaining'])} seconds remaining)"
            )

        # Files in cooldown
        if stats["cooldowns"]:
            print("\nFiles in Cooldown:")
            for file_path, cooldown in stats["cooldowns"].items():
                print(
                    f"  {file_path}: {int(cooldown['remaining'])} seconds remaining "
                    f"(until {cooldown['until']})"
                )
        else:
            print("\nNo files in cooldown.")

        # Top files
        if stats["top_files"]:
            print("\nTop Files by Modification Count:")
            for file in stats["top_files"]:
                critical = " (CRITICAL)" if file["is_critical"] else ""
                print(f"  {file['file_path']}: {file['count']} modifications{critical}")
        else:
            print("\nNo file modifications tracked.")

        exit(0)
    elif args.reset_rate_limits:
        # Reset rate limit counters
        if not hasattr(orchestrator, "rate_limiter") or not orchestrator.rate_limiter:
            print("Rate limiting is not enabled in the configuration.")
            exit(1)

        orchestrator.rate_limiter.reset_counters()
        print("Rate limit counters have been reset.")
        exit(0)
    elif args.cooldown_file:
        # Place a file in cooldown
        if not hasattr(orchestrator, "rate_limiter") or not orchestrator.rate_limiter:
            print("Rate limiting is not enabled in the configuration.")
            exit(1)

        file_path = args.cooldown_file
        duration = args.cooldown_duration

        orchestrator.rate_limiter.place_file_in_cooldown(file_path, duration)
        print(f"File {file_path} placed in cooldown for {duration} seconds.")
        exit(0)
    else:
        # Run the orchestrator normally
        orchestrator.run(demo_mode=args.demo)


if __name__ == "__main__":
    main()
