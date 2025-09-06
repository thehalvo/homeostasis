"""
Docker container management for testing.

This module provides a wrapper around Docker functionality to:
1. Create test containers
2. Run tests in isolated environments
3. Manage container lifecycle
4. Reuse containers for faster testing
"""

import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

from modules.monitoring.logger import MonitoringLogger

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent


class ContainerManager:
    """
    Manages Docker containers for testing.
    """

    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the container manager.

        Args:
            log_level: Logging level
        """
        self.logger = MonitoringLogger("container_manager", log_level=log_level)
        self.containers = {}  # Map of test_id -> container_id
        self.container_cache = {}  # Map of cache_key -> container_id

        # Check if Docker is available
        try:
            result = subprocess.run(
                ["docker", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            self.logger.info(f"Docker detected: {result.stdout.strip()}")
            self.docker_available = True
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.warning("Docker not available, using local testing environment")
            self.docker_available = False

    def _generate_test_id(self) -> str:
        """
        Generate a unique test ID.

        Returns:
            Unique test ID
        """
        return str(uuid.uuid4())

    def _generate_cache_key(self, test_type: str, requirements_hash: str) -> str:
        """
        Generate a cache key for container reuse.

        Args:
            test_type: Type of test (unit, integration, system)
            requirements_hash: Hash of requirements to ensure environment consistency

        Returns:
            Cache key for container reuse
        """
        return f"{test_type}_{requirements_hash}"

    def _get_requirements_hash(self) -> str:
        """
        Get a hash of the requirements file for cache key generation.

        Returns:
            Hash of requirements.txt
        """
        import hashlib

        req_path = project_root / "requirements.txt"
        if req_path.exists():
            with open(req_path, "rb") as f:
                return hashlib.md5(f.read(), usedforsecurity=False).hexdigest()
        return "no_requirements"

    def create_container(
        self,
        test_type: str = "unit",
        use_cache: bool = True,
        resource_limits: Dict[str, str] = None,
    ) -> str:
        """
        Create a new Docker container for testing.

        Args:
            test_type: Type of test (unit, integration, system)
            use_cache: Whether to reuse cached containers
            resource_limits: Optional resource limits for the container

        Returns:
            Test ID for container management
        """
        if not self.docker_available:
            self.logger.info("Docker not available, returning stub test ID")
            test_id = self._generate_test_id()
            self.containers[test_id] = "local"
            return test_id

        test_id = self._generate_test_id()

        # Create a cache key if using cache
        if use_cache:
            requirements_hash = self._get_requirements_hash()
            cache_key = self._generate_cache_key(test_type, requirements_hash)

            # Check if we have a cached container
            if cache_key in self.container_cache:
                cached_container_id = self.container_cache[cache_key]

                # Check if the cached container is still running
                try:
                    result = subprocess.run(
                        [
                            "docker",
                            "inspect",
                            "--format",
                            "{{.State.Running}}",
                            cached_container_id,
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True,
                    )

                    if result.stdout.strip() == "true":
                        self.logger.info(
                            f"Reusing cached container {cached_container_id}"
                        )
                        self.containers[test_id] = cached_container_id
                        return test_id
                except subprocess.SubprocessError:
                    self.logger.info(
                        f"Cached container {cached_container_id} no longer exists"
                    )
                    del self.container_cache[cache_key]

        # Build the command
        cmd = ["docker-compose", "run", "-d"]

        # Add resource limits if specified
        if resource_limits:
            if "cpu" in resource_limits:
                cmd.extend(["--cpu-quota", resource_limits["cpu"]])
            if "memory" in resource_limits:
                cmd.extend(["--memory", resource_limits["memory"]])

        # Add the service name based on test type
        service_name = f"test-{test_type}"
        cmd.append(service_name)

        # Start the container
        try:
            self.logger.info(f"Creating {test_type} test container")
            result = subprocess.run(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            container_id = result.stdout.strip()
            self.logger.info(f"Created container: {container_id}")

            # Store the container ID
            self.containers[test_id] = container_id

            # If using cache, store in the cache
            if use_cache:
                self.container_cache[cache_key] = container_id

            return test_id

        except subprocess.SubprocessError as e:
            self.logger.error(
                f"Failed to create container: {str(e)}",
                stderr=e.stderr if hasattr(e, "stderr") else None,
            )
            raise RuntimeError(f"Failed to create test container: {str(e)}")

    def run_tests(
        self, test_id: str, test_command: str = "pytest tests/", timeout: int = 60
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run tests in the container.

        Args:
            test_id: Test ID returned by create_container
            test_command: Command to run tests
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, results)
        """
        # Check if the test ID exists
        if test_id not in self.containers:
            raise ValueError(f"Invalid test ID: {test_id}")

        # If Docker isn't available, run locally
        if not self.docker_available or self.containers[test_id] == "local":
            self.logger.info("Running tests locally")
            from modules.testing.runner import TestRunner

            runner = TestRunner(log_level=self.logger.level)
            return runner.run_tests(test_command, project_root, timeout)

        container_id = self.containers[test_id]

        self.logger.info(f"Running tests in container {container_id}: {test_command}")

        start_time = time.time()

        try:
            # Execute the command in the container
            result = subprocess.run(
                ["docker", "exec", container_id, "bash", "-c", test_command],
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )

            duration = time.time() - start_time
            success = result.returncode == 0

            # Prepare results
            test_results = {
                "success": success,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
                "command": test_command,
                "container_id": container_id,
            }

            # Log results
            if success:
                self.logger.info(
                    f"Tests passed in {duration:.2f} seconds",
                    container_id=container_id,
                    return_code=result.returncode,
                )
            else:
                self.logger.error(
                    f"Tests failed in {duration:.2f} seconds",
                    container_id=container_id,
                    return_code=result.returncode,
                    stderr=result.stderr,
                )

            return success, test_results

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time

            self.logger.error(
                f"Tests timed out after {timeout} seconds",
                container_id=container_id,
                command=test_command,
            )

            return False, {
                "success": False,
                "return_code": None,
                "stdout": "",
                "stderr": f"Timed out after {timeout} seconds",
                "duration": duration,
                "command": test_command,
                "container_id": container_id,
            }

        except Exception as e:
            duration = time.time() - start_time

            self.logger.exception(
                e,
                message=f"Error running tests: {str(e)}",
                container_id=container_id,
                command=test_command,
            )

            return False, {
                "success": False,
                "return_code": None,
                "stdout": "",
                "stderr": str(e),
                "duration": duration,
                "command": test_command,
                "container_id": container_id,
            }

    def copy_files(
        self, test_id: str, source_path: Path, container_path: str = "/app"
    ) -> bool:
        """
        Copy files to a container.

        Args:
            test_id: Test ID returned by create_container
            source_path: Source path on the host
            container_path: Destination path in the container

        Returns:
            True if successful, False otherwise
        """
        # Check if the test ID exists
        if test_id not in self.containers:
            raise ValueError(f"Invalid test ID: {test_id}")

        # If Docker isn't available, don't copy
        if not self.docker_available or self.containers[test_id] == "local":
            self.logger.info("Docker not available, skipping file copy")
            return True

        container_id = self.containers[test_id]

        try:
            # Copy the files to the container
            subprocess.run(
                ["docker", "cp", str(source_path), f"{container_id}:{container_path}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            self.logger.info(
                f"Copied {source_path} to container {container_id}:{container_path}"
            )
            return True

        except subprocess.SubprocessError as e:
            self.logger.error(
                f"Failed to copy files: {str(e)}",
                stderr=e.stderr if hasattr(e, "stderr") else None,
            )
            return False

    def stop_container(self, test_id: str, remove: bool = True) -> bool:
        """
        Stop and optionally remove a container.

        Args:
            test_id: Test ID returned by create_container
            remove: Whether to remove the container after stopping

        Returns:
            True if successful, False otherwise
        """
        # Check if the test ID exists
        if test_id not in self.containers:
            self.logger.warning(f"Invalid test ID: {test_id}")
            return False

        # If Docker isn't available, don't stop
        if not self.docker_available or self.containers[test_id] == "local":
            self.logger.info("Docker not available, skipping container stop")
            del self.containers[test_id]
            return True

        container_id = self.containers[test_id]

        try:
            # First try to stop the container gracefully
            subprocess.run(
                ["docker", "stop", container_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            self.logger.info(f"Stopped container {container_id}")

            # Remove the container if requested
            if remove:
                subprocess.run(
                    ["docker", "rm", container_id],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )

                self.logger.info(f"Removed container {container_id}")

                # Remove the container from the cache
                for key, value in list(self.container_cache.items()):
                    if value == container_id:
                        del self.container_cache[key]

            # Remove the test ID
            del self.containers[test_id]

            return True

        except subprocess.SubprocessError as e:
            self.logger.error(
                f"Failed to stop container: {str(e)}",
                stderr=e.stderr if hasattr(e, "stderr") else None,
            )
            return False

    def validate_patch(
        self,
        patch_id: str,
        test_type: str = "unit",
        test_command: str = "pytest tests/",
        timeout: int = 60,
        resource_limits: Dict[str, str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Validate a patch by running tests in a container.

        Args:
            patch_id: ID of the patch to validate
            test_type: Type of test (unit, integration, system)
            test_command: Command to run tests
            timeout: Timeout in seconds
            resource_limits: Optional resource limits for the container
            use_cache: Whether to reuse cached containers

        Returns:
            Validation results
        """
        self.logger.info(f"Validating patch {patch_id} with {test_type} tests")

        # Create a container for testing
        test_id = self.create_container(test_type, use_cache, resource_limits)

        try:
            # Run the tests
            success, test_results = self.run_tests(test_id, test_command, timeout)

            # Prepare validation results
            validation_results = {
                "patch_id": patch_id,
                "valid": success,
                "test_results": test_results,
                "test_type": test_type,
                "timestamp": time.time(),
            }

            # Log validation results
            if success:
                self.logger.info(
                    f"Patch {patch_id} is valid with {test_type} tests",
                    test_command=test_command,
                )
            else:
                self.logger.warning(
                    f"Patch {patch_id} is invalid with {test_type} tests",
                    test_command=test_command,
                    stderr=test_results.get("stderr", ""),
                )

            return validation_results

        finally:
            # If we're not using caching, stop the container
            if not use_cache:
                self.stop_container(test_id)

    def cleanup(self) -> None:
        """
        Clean up all containers.
        """
        self.logger.info("Cleaning up containers")

        # Stop all containers
        for test_id in list(self.containers.keys()):
            self.stop_container(test_id)

        # Clear the cache
        self.container_cache.clear()


if __name__ == "__main__":
    # Example usage
    manager = ContainerManager()

    # Create a container for unit tests
    test_id = manager.create_container("unit")

    # Run tests
    success, results = manager.run_tests(test_id, "pytest tests/test_patches.py -v")

    print(f"Tests {'passed' if success else 'failed'}")
    if not success:
        print(f"Error: {results['stderr']}")

    # Stop the container
    manager.stop_container(test_id)

    # Validate a patch
    validation = manager.validate_patch(
        "example-patch-id",
        test_type="unit",
        test_command="pytest tests/test_patches.py -v",
    )

    print(f"Patch is {'valid' if validation['valid'] else 'invalid'}")
