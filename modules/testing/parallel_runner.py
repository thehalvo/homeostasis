"""
Parallel test runner for validating multiple patches concurrently.
"""

import concurrent.futures
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.monitoring.logger import MonitoringLogger
from modules.testing.container_manager import ContainerManager

# Define project root
project_root = Path(__file__).parent.parent.parent


class ParallelTestRunner:
    """
    Class for running tests in parallel to validate multiple patches.
    """

    def __init__(
        self,
        log_level: str = "INFO",
        max_workers: Optional[int] = None,
        use_containers: bool = True,
    ):
        """
        Initialize the parallel test runner.

        Args:
            log_level: Logging level
            max_workers: Maximum number of concurrent worker threads/processes
            use_containers: Whether to use Docker containers for isolation
        """
        self.logger = MonitoringLogger("parallel_test_runner", log_level=log_level)
        self.log_level = log_level
        cpu_count = os.cpu_count() or 1
        self.max_workers = max_workers or min(
            cpu_count * 2, 8
        )  # Default to CPU count * 2 or 8, whichever is smaller
        self.use_containers = use_containers

        # Initialize the container manager if using containers
        if self.use_containers:
            self.container_manager = ContainerManager(log_level=log_level)

        self.logger.info(
            f"Initialized parallel test runner with {self.max_workers} workers"
        )

    def _validate_patch_container(
        self,
        patch_id: str,
        test_type: str,
        test_command: str,
        timeout: int,
        resource_limits: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate a patch using a container.

        Args:
            patch_id: ID of the patch to validate
            test_type: Type of test (unit, integration, system)
            test_command: Command to run tests
            timeout: Timeout in seconds
            resource_limits: Optional resource limits for the container

        Returns:
            Validation results
        """
        if not self.use_containers:
            self.logger.warning(
                "Container validation requested but containers are disabled"
            )
            from modules.testing.runner import TestRunner

            runner = TestRunner(log_level=self.log_level)
            return runner.validate_patch(patch_id, test_command, project_root, timeout)

        return self.container_manager.validate_patch(
            patch_id, test_type, test_command, timeout, resource_limits
        )

    def _validate_patch_local(
        self,
        patch_id: str,
        test_command: str,
        working_dir: Optional[Path] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Validate a patch locally without containers.

        Args:
            patch_id: ID of the patch to validate
            test_command: Command to run tests
            working_dir: Working directory for the command
            timeout: Timeout in seconds

        Returns:
            Validation results
        """
        from modules.testing.runner import TestRunner

        runner = TestRunner(log_level=self.log_level)
        return runner.validate_patch(patch_id, test_command, working_dir, timeout)

    def validate_patches(
        self,
        patches: List[Dict[str, Any]],
        test_command: str = "pytest tests/",
        test_type: str = "unit",
        timeout: int = 60,
        resource_limits: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate multiple patches in parallel.

        Args:
            patches: List of patches to validate
            test_command: Command to run tests
            test_type: Type of test (unit, integration, system)
            timeout: Timeout in seconds
            resource_limits: Optional resource limits for containers

        Returns:
            List of validation results
        """
        self.logger.info(
            f"Validating {len(patches)} patches in parallel with {self.max_workers} workers"
        )

        validation_results = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Create a future for each patch
            futures = {}
            for patch in patches:
                patch_id = patch.get("patch_id", "unknown")

                if self.use_containers:
                    future = executor.submit(
                        self._validate_patch_container,
                        patch_id,
                        test_type,
                        test_command,
                        timeout,
                        resource_limits,
                    )
                else:
                    future = executor.submit(
                        self._validate_patch_local,
                        patch_id,
                        test_command,
                        project_root,
                        timeout,
                    )

                futures[future] = patch_id

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                patch_id = futures[future]
                self.logger.info(f"Completed validation for patch {patch_id}")

                try:
                    result = future.result()
                    validation_results.append(result)

                    # Log the result
                    if result.get("valid", False):
                        self.logger.info(f"Patch {patch_id} is valid")
                    else:
                        self.logger.warning(f"Patch {patch_id} is invalid")

                except Exception as e:
                    self.logger.exception(
                        e, message=f"Error validating patch {patch_id}"
                    )

                    # Add a failed result
                    validation_results.append(
                        {
                            "patch_id": patch_id,
                            "valid": False,
                            "error": str(e),
                            "timestamp": time.time(),
                        }
                    )

        # Clean up if using containers
        if self.use_containers:
            self.container_manager.cleanup()

        return validation_results

    def run_graduated_tests(
        self,
        patch: Dict[str, Any],
        test_types: Optional[List[str]] = None,
        test_commands: Optional[Dict[str, str]] = None,
        timeouts: Optional[Dict[str, int]] = None,
        resource_limits: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Run graduated testing (unit -> integration -> system).

        Tests are run sequentially, and if any test level fails, the process stops.

        Args:
            patch: Patch to validate
            test_types: Ordered list of test types to run
            test_commands: Dictionary mapping test types to commands
            timeouts: Dictionary mapping test types to timeouts
            resource_limits: Dictionary mapping test types to resource limits

        Returns:
            Validation results with details for each test level
        """
        patch_id = patch.get("patch_id", "unknown")
        self.logger.info(f"Running graduated tests for patch {patch_id}")

        # Set up default test types if not provided
        if test_types is None:
            test_types = ["unit", "integration", "system"]

        # Set up default test commands if not provided
        if test_commands is None:
            test_commands = {
                "unit": "pytest tests/ -m unit -v",
                "integration": "pytest tests/ -m integration -v",
                "system": "pytest tests/ -m system -v",
            }

        # Set up default timeouts if not provided
        if timeouts is None:
            timeouts = {"unit": 30, "integration": 60, "system": 120}

        # Set up default resource limits if not provided
        if resource_limits is None:
            resource_limits = {
                "unit": {"cpu": "0.5", "memory": "512m"},
                "integration": {"cpu": "1.0", "memory": "1g"},
                "system": {"cpu": "2.0", "memory": "2g"},
            }

        # Initialize results
        results = {
            "patch_id": patch_id,
            "valid": True,
            "test_levels": {},
            "timestamp": time.time(),
        }

        # Run tests at each level
        for test_type in test_types:
            self.logger.info(f"Running {test_type} tests for patch {patch_id}")

            # Get test parameters
            test_command = test_commands.get(
                test_type, f"pytest tests/ -m {test_type} -v"
            )
            timeout = timeouts.get(test_type, 60)
            limits = resource_limits.get(test_type, {})

            # Run the test
            if self.use_containers:
                validation = self._validate_patch_container(
                    patch_id, test_type, test_command, timeout, limits
                )
            else:
                validation = self._validate_patch_local(
                    patch_id, test_command, project_root, timeout
                )

            # Store results for this level
            results["test_levels"][test_type] = validation

            # If this level failed, stop testing
            if not validation.get("valid", False):
                self.logger.warning(f"{test_type} tests failed for patch {patch_id}")
                results["valid"] = False
                results["failed_level"] = test_type
                break

            self.logger.info(f"{test_type} tests passed for patch {patch_id}")

        # Clean up if using containers
        if self.use_containers:
            self.container_manager.cleanup()

        return results

    def validate_patches_graduated(
        self,
        patches: List[Dict[str, Any]],
        test_types: Optional[List[str]] = None,
        test_commands: Optional[Dict[str, str]] = None,
        timeouts: Optional[Dict[str, int]] = None,
        resource_limits: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate multiple patches using graduated testing (unit -> integration -> system).

        Each patch is validated sequentially through the test levels, but multiple patches
        are validated in parallel.

        Args:
            patches: List of patches to validate
            test_types: Ordered list of test types to run
            test_commands: Dictionary mapping test types to commands
            timeouts: Dictionary mapping test types to timeouts
            resource_limits: Dictionary mapping test types to resource limits

        Returns:
            List of validation results
        """
        self.logger.info(f"Validating {len(patches)} patches with graduated testing")

        # Set up default test types if not provided
        if test_types is None:
            test_types = ["unit", "integration", "system"]

        validation_results = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Create a future for each patch
            futures = {}
            for patch in patches:
                patch_id = patch.get("patch_id", "unknown")

                future = executor.submit(
                    self.run_graduated_tests,
                    patch,
                    test_types,
                    test_commands,
                    timeouts,
                    resource_limits,
                )

                futures[future] = patch_id

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                patch_id = futures[future]
                self.logger.info(f"Completed graduated validation for patch {patch_id}")

                try:
                    result = future.result()
                    validation_results.append(result)

                    # Log the result
                    if result.get("valid", False):
                        self.logger.info(f"Patch {patch_id} passed all test levels")
                    else:
                        failed_level = result.get("failed_level", "unknown")
                        self.logger.warning(
                            f"Patch {patch_id} failed at test level: {failed_level}"
                        )

                except Exception as e:
                    self.logger.exception(
                        e, message=f"Error validating patch {patch_id}"
                    )

                    # Add a failed result
                    validation_results.append(
                        {
                            "patch_id": patch_id,
                            "valid": False,
                            "error": str(e),
                            "timestamp": time.time(),
                        }
                    )

        return validation_results


if __name__ == "__main__":
    # Example usage
    runner = ParallelTestRunner()

    # Create some example patches
    patches = [
        {"patch_id": "patch1", "file_path": "services/example_service/app.py"},
        {"patch_id": "patch2", "file_path": "services/example_service/app.py"},
    ]

    # Validate patches in parallel
    results = runner.validate_patches(
        patches, test_command="pytest tests/test_patches.py -v"
    )

    print(f"Validated {len(results)} patches")
    for result in results:
        print(
            f"Patch {result['patch_id']}: {'valid' if result['valid'] else 'invalid'}"
        )

    # Run graduated testing for a patch
    graduated_result = runner.run_graduated_tests(patches[0])

    print(
        f"Graduated testing for patch {graduated_result['patch_id']}: {'passed all levels' if graduated_result['valid'] else 'failed'}"
    )
    if not graduated_result["valid"]:
        print(f"Failed at level: {graduated_result.get('failed_level', 'unknown')}")
