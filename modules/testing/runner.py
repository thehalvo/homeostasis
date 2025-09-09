"""
Test runner module for validating patches.
"""

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from modules.monitoring.logger import MonitoringLogger

# Define project root
project_root = Path(__file__).parent.parent.parent


class TestRunner:
    """
    Class for running tests to validate patches.
    """

    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the test runner.

        Args:
            log_level: Logging level
        """
        self.logger = MonitoringLogger("test_runner", log_level=log_level)

    def run_tests(
        self, test_command: str, working_dir: Optional[Path] = None, timeout: int = 30
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run tests with the specified command.

        Args:
            test_command: Command to run tests
            working_dir: Working directory for the command
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, results)
        """
        if working_dir is None:
            working_dir = project_root

        self.logger.info(f"Running tests: {test_command}", working_dir=str(working_dir))

        start_time = time.time()

        try:
            # Run the tests
            import shlex

            # Parse command into list for safer execution
            cmd_list = shlex.split(test_command)

            result = subprocess.run(
                cmd_list,
                shell=False,
                cwd=working_dir,
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
            }

            # Log results
            if success:
                self.logger.info(
                    f"Tests passed in {duration:.2f} seconds",
                    return_code=result.returncode,
                )
            else:
                self.logger.error(
                    f"Tests failed in {duration:.2f} seconds",
                    return_code=result.returncode,
                    stderr=result.stderr,
                )

            return success, test_results

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time

            self.logger.error(
                f"Tests timed out after {timeout} seconds", command=test_command
            )

            return False, {
                "success": False,
                "return_code": None,
                "stdout": "",
                "stderr": f"Timed out after {timeout} seconds",
                "duration": duration,
                "command": test_command,
            }

        except Exception as e:
            duration = time.time() - start_time

            self.logger.exception(
                e, message=f"Error running tests: {str(e)}", command=test_command
            )

            return False, {
                "success": False,
                "return_code": None,
                "stdout": "",
                "stderr": str(e),
                "duration": duration,
                "command": test_command,
            }

    def validate_patch(
        self,
        patch_id: str,
        test_command: str,
        working_dir: Optional[Path] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Validate a patch by running tests.

        Args:
            patch_id: ID of the patch to validate
            test_command: Command to run tests
            working_dir: Working directory for the command
            timeout: Timeout in seconds

        Returns:
            Validation results
        """
        self.logger.info(f"Validating patch {patch_id}")

        # Run the tests
        success, test_results = self.run_tests(test_command, working_dir, timeout)

        # Prepare validation results
        validation_results = {
            "patch_id": patch_id,
            "valid": success,
            "test_results": test_results,
            "timestamp": time.time(),
        }

        # Log validation results
        if success:
            self.logger.info(f"Patch {patch_id} is valid", test_command=test_command)
        else:
            self.logger.warning(
                f"Patch {patch_id} is invalid",
                test_command=test_command,
                stderr=test_results.get("stderr", ""),
            )

        return validation_results


if __name__ == "__main__":
    # Example usage
    runner = TestRunner()

    # Run tests for a patch
    success, results = runner.run_tests("pytest -xvs tests/")

    print(f"Tests {'passed' if success else 'failed'}")
    if not success:
        print(f"Error: {results['stderr']}")

    # Validate a patch
    validation = runner.validate_patch("example-patch-id", "pytest -xvs tests/")

    print(f"Patch is {'valid' if validation['valid'] else 'invalid'}")
